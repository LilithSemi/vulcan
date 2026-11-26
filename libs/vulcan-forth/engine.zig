//! Runs a compiled program on the host: JITs every word, owns the data memory and stands
//! in for the I/O window.

const std = @import("std");
const vt = @import("vulcan-target");
const ast = @import("ast.zig");
const lower = @import("lower.zig");

const native = vt.native;

pub const Error = native.Error || error{ NoMain, MissingWord, WrongArity, OutOfMemory };

pub const IoWrite = struct { addr: u16, value: u16 };

pub const Machine = struct {
    allocator: std.mem.Allocator,
    program: ast.Program,
    jitted: native.JittedModule,
    /// The data area. The whole 16 bit address space is backed, and only the part below the
    /// I/O window is reachable from Forth.
    memory: []u8,
    /// What generated code reads, see the offsets in `lower.zig`.
    context: [lower.ctx_size / 8]u64 align(8),
    /// Every store to the I/O window, in order.
    io_writes: std.ArrayList(IoWrite) = .empty,
    /// What a load from the I/O window returns.
    io_read_value: u16 = 0,

    /// JIT `program` and set up its memory. The machine owns the program from here on.
    pub fn init(allocator: std.mem.Allocator, program: ast.Program) Error!Machine {
        var owned = program;
        errdefer owned.deinit();
        var funcs = try allocator.alloc(native.ModuleFunction, owned.functions.len);
        defer allocator.free(funcs);
        for (owned.functions, 0..) |*lowered, i| funcs[i] = .{ .name = lowered.name, .func = &lowered.func };
        var jitted = try native.jitModule(allocator, funcs);
        errdefer jitted.deinit();

        const memory = try allocator.alloc(u8, 1 << 16);
        errdefer allocator.free(memory);
        @memset(memory, 0);
        @memcpy(memory[0..owned.data.len], owned.data);

        var machine: Machine = .{
            .allocator = allocator,
            .program = owned,
            .jitted = jitted,
            .memory = memory,
            .context = @splat(0),
        };
        machine.context[lower.ctx_mem / 8] = @intFromPtr(memory.ptr);
        machine.context[lower.ctx_io_read / 8] = @intFromPtr(&ioRead);
        machine.context[lower.ctx_io_write / 8] = @intFromPtr(&ioWrite);
        return machine;
    }

    pub fn deinit(self: *Machine) void {
        self.io_writes.deinit(self.allocator);
        self.allocator.free(self.memory);
        self.jitted.deinit();
        self.program.deinit();
    }

    fn fromContext(ctx: *anyopaque) *Machine {
        const slots: *[lower.ctx_size / 8]u64 = @ptrCast(@alignCast(ctx));
        return @alignCast(@fieldParentPtr("context", slots));
    }

    fn ioRead(ctx: *anyopaque, addr: u32) callconv(.c) u32 {
        _ = addr;
        return fromContext(ctx).io_read_value;
    }

    fn ioWrite(ctx: *anyopaque, addr: u32, value: u32) callconv(.c) void {
        const self = fromContext(ctx);
        self.io_writes.append(self.allocator, .{ .addr = @intCast(addr), .value = @intCast(value) }) catch {
            // A test machine that cannot record a write has nothing else to report it with.
            std.debug.panic("out of memory while recording an I/O write", .{});
        };
    }

    /// Call word `index` with `args`, deepest first. The result is the data stack the word
    /// leaves, deepest first. The caller frees it.
    pub fn call(self: *Machine, index: usize, args: []const u16) Error![]u16 {
        const word = self.program.words[index];
        if (args.len != word.effect.in) return error.WrongArity;
        var name_buf: [16]u8 = undefined;
        const name = std.fmt.bufPrint(&name_buf, "w{d}", .{index}) catch unreachable;
        const addr = @intFromPtr(self.jitted.entry(*const fn () callconv(.c) void, name) orelse return error.MissingWord);
        const ctx: *anyopaque = @ptrCast(&self.context);

        var a: [4]u32 = @splat(0);
        if (args.len > a.len) return error.WrongArity;
        for (args, 0..) |v, k| a[k] = v;
        const F = *const fn (*anyopaque, u32, u32, u32, u32) callconv(.c) u32;
        const ret: u32 = @as(F, @ptrFromInt(addr))(ctx, a[0], a[1], a[2], a[3]);

        const out = try self.allocator.alloc(u16, word.effect.out);
        errdefer self.allocator.free(out);
        if (out.len > 0) {
            const results: [*]const u32 = @ptrCast(@alignCast(@as([*]const u8, @ptrCast(&self.context)) + lower.ctx_results));
            for (out[0 .. out.len - 1], 0..) |*cell, k| cell.* = @intCast(results[k]);
            out[out.len - 1] = @intCast(ret);
        }
        return out;
    }

    /// Run the last word named `main`, which takes no cells. Returns the data stack.
    pub fn runMain(self: *Machine) Error![]u16 {
        const index = self.program.main orelse return error.NoMain;
        return self.call(index, &.{});
    }
};
