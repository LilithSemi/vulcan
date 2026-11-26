//! Runs a Forth program as Geyser code: select instructions, lift the image to IR and run it on
//! the host.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const forth = @import("vulcan-forth");

const lift = target.geyser.lift;
const Run = *const fn (state: *lift.Core) callconv(.c) u32;

pub const memory_words = 4096;
pub const stack_depth = target.geyser.encode.default_stack_depth;

pub const Result = struct {
    stack: []u16,
    io_writes: []forth.IoWrite,
    exit: lift.Exit,
    image_words: usize,
    slots: u32,

    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        allocator.free(self.stack);
        allocator.free(self.io_writes);
    }
};

const Io = struct {
    allocator: std.mem.Allocator,
    writes: std.ArrayList(forth.IoWrite) = .empty,
    read_value: u16 = 0,

    fn callback(ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u32 {
        const self: *Io = @ptrCast(@alignCast(ctx.?));
        if (kind == lift.io.read) return self.read_value;
        self.writes.append(self.allocator, .{ .addr = @intCast(at), .value = @intCast(value) }) catch {
            std.debug.panic("out of memory while recording an I/O write", .{});
        };
        return 0;
    }
};

/// Compile `source` for Geyser and run it until the entry code halts.
pub fn run(allocator: std.mem.Allocator, source: []const u8) !Result {
    var diag: forth.Diagnostic = .{};
    var program = forth.compile(allocator, source, &diag) catch |err| {
        std.debug.print("compile error: {s}\n", .{diag.message()});
        return err;
    };
    defer program.deinit();
    const image = try forth.compileGeyser(allocator, &program, .{});
    defer allocator.free(image.words);
    return execute(allocator, image.words, image.slots);
}

pub fn execute(allocator: std.mem.Allocator, words: []const u16, slots: u32) !Result {
    var func = ir.function.Function.init(allocator);
    defer func.deinit();
    try lift.region(allocator, &func, words, 0, stack_depth);
    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();

    const memory = try allocator.alloc(u16, memory_words);
    defer allocator.free(memory);
    @memset(memory, 0);
    @memcpy(memory[0..words.len], words);
    var io: Io = .{ .allocator = allocator };
    errdefer io.writes.deinit(allocator);
    var core: lift.Core = .{ .mem = memory.ptr, .mem_mask = memory_words - 1, .io = Io.callback, .io_ctx = &io };
    const exit: lift.Exit = @fromBackingInt(@intCast(code.entry(Run, 0)(&core)));

    var stack: std.ArrayList(u16) = .empty;
    errdefer stack.deinit(allocator);
    if (core.dsp > 0) {
        for (core.d[2 .. core.dsp + 1]) |cell| try stack.append(allocator, cell);
        try stack.append(allocator, @intCast(core.t));
    }
    return .{
        .stack = try stack.toOwnedSlice(allocator),
        .io_writes = try io.writes.toOwnedSlice(allocator),
        .exit = exit,
        .image_words = words.len,
        .slots = slots,
    };
}
