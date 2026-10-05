//! Live regression for scalar-expanded fp32 matmul over arbitrarily aligned global memory.
//!
//! The NVIDIA backend does not lower `matmul` itself. This test owns the policy step: it clones
//! the canonical function, expands only that clone, then sends the scalar IR through the ordinary
//! NVIDIA compile and dispatch path. Both odd and aligned buffers run on the real GPU.

const std = @import("std");
const host = @import("builtin");
const ir = @import("vulcan-ir");
const gpu_abi = @import("vulcan-gpu");
const target = @import("vulcan-target");
const nvidia = @import("nvidia");

const compute = nvidia.compute;
const isel = target.nvidia.isel;
const Function = ir.function.Function;
const MatMul = ir.function.MatMul;
const testing = std.testing;

const runner_abi: gpu_abi.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

fn noGpu(err: anyerror) bool {
    return switch (err) {
        error.SkipZigTest,
        error.FileNotFound,
        error.AccessDenied,
        error.PermissionDenied,
        error.DeviceBusy,
        error.NoDevice,
        => true,
        else => false,
    };
}

const Harness = struct {
    runner: compute.Runner,
    params: compute.Buffer,

    fn open() !Harness {
        if (host.os.tag != .linux) return error.SkipZigTest;
        var runner = compute.Runner.init() catch |err| {
            if (noGpu(err)) return error.SkipZigTest;
            return err;
        };
        errdefer runner.deinit();
        return .{
            .params = try runner.alloc(.system, 0x1000),
            .runner = runner,
        };
    }

    fn deinit(self: *Harness) void {
        self.runner.deinit();
    }

    fn alloc(self: *Harness, size: usize) !compute.Buffer {
        return self.runner.alloc(.system, @intCast(size));
    }

    fn compile(self: *Harness, func: *Function) !Launch {
        var kernel = try isel.compileKernel(testing.allocator, func, runner_abi);
        errdefer kernel.deinit(testing.allocator);
        const code_bytes = kernel.code.len * @sizeOf(u32);
        if (self.runner.code.bytes.len < code_bytes) {
            self.runner.code = try self.runner.alloc(.system_wc, @intCast(code_bytes));
        }
        return .{ .harness = self, .kernel = kernel };
    }
};

const Launch = struct {
    harness: *Harness,
    kernel: isel.Kernel,

    fn deinit(self: *Launch) void {
        self.kernel.deinit(testing.allocator);
    }

    fn setPtr(self: *Launch, index: usize, address: u64) void {
        const offset = self.kernel.launch.params[index].offset;
        std.mem.writeInt(u64, self.harness.params.bytes[offset..][0..8], address, .little);
    }

    fn run(self: *Launch) !void {
        try self.harness.runner.run(self.kernel.code, .{
            .grid = .{ 1, 1, 1 },
            .block = self.kernel.launch.block,
            .register_count = self.kernel.launch.reg_count,
            .cbuf0_va = self.harness.params.va,
            .cbuf0_size = @max(self.kernel.launch.param_bytes, 16),
            .shared_mem_bytes = self.kernel.launch.shared_bytes,
        });
    }
};

const Snapshot = struct {
    text: []u8,
    bitcode: []u8,
    blocks: usize,
    insts: usize,
    values: usize,
    types: usize,
    symbols: usize,
    attrs: usize,

    fn take(allocator: std.mem.Allocator, func: *const Function) !Snapshot {
        const text = try std.fmt.allocPrint(allocator, "{f}", .{func});
        errdefer allocator.free(text);
        return .{
            .text = text,
            .bitcode = try ir.bitcode.encode(allocator, func),
            .blocks = func.blockCount(),
            .insts = func.instCount(),
            .values = func.valueCount(),
            .types = func.types.count(),
            .symbols = func.symbolCount(),
            .attrs = func.attributeEntries().len,
        };
    }

    fn deinit(self: Snapshot, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.bitcode);
    }

    fn expectUnchanged(self: Snapshot, allocator: std.mem.Allocator, func: *const Function) !void {
        const text = try std.fmt.allocPrint(allocator, "{f}", .{func});
        defer allocator.free(text);
        const bitcode = try ir.bitcode.encode(allocator, func);
        defer allocator.free(bitcode);

        try testing.expectEqualStrings(self.text, text);
        try testing.expectEqualSlices(u8, self.bitcode, bitcode);
        try testing.expectEqual(self.blocks, func.blockCount());
        try testing.expectEqual(self.insts, func.instCount());
        try testing.expectEqual(self.values, func.valueCount());
        try testing.expectEqual(self.types, func.types.count());
        try testing.expectEqual(self.symbols, func.symbolCount());
        try testing.expectEqual(self.attrs, func.attributeEntries().len);
    }
};

fn matmulFunction(allocator: std.mem.Allocator, accumulate: bool) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);

    try func.addAttr(.func, .{ .custom = .{
        .namespace = "test",
        .key = "canonical-matmul",
        .value = .flag,
    } });
    try func.addAttr(.{ .value = a }, .{ .custom = .{
        .namespace = "test",
        .key = "read-only-input",
        .value = .flag,
    } });

    const mm: MatMul = .{
        .a = a,
        .b = b,
        .c = c,
        .m = 2,
        .n = 4,
        .k = 3,
        .dtype = .fp32,
        .accumulate = accumulate,
    };
    _ = try func.appendStmtRaw(entry, .{ .matmul = mm });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return func;
}

const a_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
const b_values = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
const c_seed = [_]f32{ 100, 200, 300, 400, 500, 600, 700, 800 };
const product = [_]f32{ 38, 44, 50, 56, 83, 98, 113, 128 };
const accumulated = [_]f32{ 138, 244, 350, 456, 583, 698, 813, 928 };

const Guarded = struct {
    buffer: compute.Buffer,
    offset: usize,
    active_bytes: usize,
    prefix: u8,
    suffix: u8,

    fn address(self: Guarded) u64 {
        return self.buffer.va + self.offset;
    }

    fn expectGuards(self: Guarded) !void {
        for (self.buffer.bytes[0..self.offset]) |byte| try testing.expectEqual(self.prefix, byte);
        for (self.buffer.bytes[self.offset + self.active_bytes ..]) |byte| try testing.expectEqual(self.suffix, byte);
    }
};

fn guarded(
    harness: *Harness,
    offset: usize,
    active_bytes: usize,
    prefix: u8,
    suffix: u8,
) !Guarded {
    const buffer = try harness.alloc(offset + active_bytes + 19);
    @memset(buffer.bytes[0..offset], prefix);
    @memset(buffer.bytes[offset .. offset + active_bytes], 0xcc);
    @memset(buffer.bytes[offset + active_bytes ..], suffix);
    return .{
        .buffer = buffer,
        .offset = offset,
        .active_bytes = active_bytes,
        .prefix = prefix,
        .suffix = suffix,
    };
}

fn writeF32s(region: Guarded, values: []const f32) void {
    for (values, 0..) |value, index| {
        const start = region.offset + index * @sizeOf(f32);
        std.mem.writeInt(u32, region.buffer.bytes[start..][0..4], @bitCast(value), .little);
    }
}

fn expectF32Bits(region: Guarded, expected: []const f32) !void {
    for (expected, 0..) |value, index| {
        const start = region.offset + index * @sizeOf(f32);
        const got = std.mem.readInt(u32, region.buffer.bytes[start..][0..4], .little);
        try testing.expectEqual(@as(u32, @bitCast(value)), got);
    }
}

fn runCase(harness: *Harness, launch: *Launch, accumulate: bool, offsets: [3]usize) !void {
    const a = try guarded(harness, offsets[0], a_values.len * @sizeOf(f32), 0xa1, 0xa2);
    const b = try guarded(harness, offsets[1], b_values.len * @sizeOf(f32), 0xb1, 0xb2);
    const c = try guarded(harness, offsets[2], c_seed.len * @sizeOf(f32), 0xc1, 0xc2);
    writeF32s(a, &a_values);
    writeF32s(b, &b_values);
    writeF32s(c, &c_seed);

    const a_before = try testing.allocator.dupe(u8, a.buffer.bytes);
    defer testing.allocator.free(a_before);
    const b_before = try testing.allocator.dupe(u8, b.buffer.bytes);
    defer testing.allocator.free(b_before);

    launch.setPtr(0, a.address());
    launch.setPtr(1, b.address());
    launch.setPtr(2, c.address());
    try launch.run();

    try testing.expectEqualSlices(u8, a_before, a.buffer.bytes);
    try testing.expectEqualSlices(u8, b_before, b.buffer.bytes);
    try a.expectGuards();
    try b.expectGuards();
    try c.expectGuards();
    try expectF32Bits(c, if (accumulate) &accumulated else &product);
}

test "live: scalar-expanded fp32 matmul is exact over odd and aligned global memory" {
    const allocator = testing.allocator;
    var harness = try Harness.open();
    defer harness.deinit();

    for ([_]bool{ false, true }) |accumulate| {
        var canonical = try matmulFunction(allocator, accumulate);
        defer canonical.deinit();
        const snapshot = try Snapshot.take(allocator, &canonical);
        defer snapshot.deinit(allocator);

        var expanded = try canonical.clone(allocator);
        defer expanded.deinit();
        try testing.expect(try ir.expand.expandMatmul(allocator, &expanded));
        for (0..expanded.blockCount()) |block_index| {
            for (expanded.blockInsts(@fromBackingInt(@intCast(block_index)))) |inst| {
                try testing.expect(expanded.opcode(inst) != .matmul);
            }
        }
        var diagnostics = try ir.verify.verify(allocator, &expanded, .low);
        defer diagnostics.deinit();
        try testing.expect(diagnostics.ok());
        try snapshot.expectUnchanged(allocator, &canonical);

        var launch = try harness.compile(&expanded);
        defer launch.deinit();
        try runCase(&harness, &launch, accumulate, .{ 3, 5, 7 });
        try runCase(&harness, &launch, accumulate, .{ 16, 16, 16 });
        try snapshot.expectUnchanged(allocator, &canonical);
    }
}
