const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const target = @import("vulcan-target");
const nvidia = @import("nvidia");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const compute = nvidia.compute;
const samples = 7;
const elements = 256 * 1024;

const runner_abi: gpu.Abi = .{
    .param_base = 0,
    .pointer_bytes = 8,
    .param_align = 4,
    .max_shared_bytes = 48 * 1024,
    .linear_thread_id = false,
};

const Timing = struct {
    best_ns: u64,
    median_ns: u64,
};

const PtxasKernel = struct {
    code: []u32,
    registers: u32,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const ptx_source = args.next() orelse return usage();
    const cuda_source = args.next() orelse return usage();
    const output_dir = args.next() orelse return usage();
    if (args.next() != null) return usage();

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buffer);
    const writer = &stdout.interface;
    if (!try programAvailable(allocator, io, "ptxas") or !try programAvailable(allocator, io, "nvcc")) {
        try writer.writeAll("\nCUDA benchmark skipped: nvcc or ptxas is unavailable\n");
        return writer.flush();
    }

    var runner = compute.Runner.init() catch |err| {
        if (noGpu(err)) {
            try writer.writeAll("\nCUDA benchmark skipped: no usable NVIDIA GPU\n");
            return writer.flush();
        }
        return err;
    };
    defer runner.deinitChecked(1_000_000) catch {};

    try std.Io.Dir.cwd().createDirPath(io, output_dir);
    const ptxas_cubin = try std.fs.path.join(allocator, &.{ output_dir, "vector_mix.ptxas.cubin" });
    const nvcc_cubin = try std.fs.path.join(allocator, &.{ output_dir, "vector_mix.nvcc.cubin" });

    var func = try buildVectorMix(allocator);
    defer func.deinit();
    const vulcan_compile = try benchmarkVulcanCompile(allocator, io, &func);
    const ptxas_compile = try benchmarkPtxasCompile(allocator, io, ptx_source, ptxas_cubin);
    const nvcc_compile = try benchmarkNvccCompile(allocator, io, cuda_source, nvcc_cubin);

    var vulcan_kernel = try target.nvidia.isel.compileKernel(allocator, &func, runner_abi);
    defer vulcan_kernel.deinit(allocator);
    const ptxas_kernel = try loadPtxasKernel(allocator, io, ptxas_cubin, ptxas_compile.registers);
    const nvcc_kernel = try loadPtxasKernel(allocator, io, nvcc_cubin, nvcc_compile.registers);

    const params = try runner.alloc(.system, 0x1000);
    const out = try runner.alloc(.system_wc, elements * @sizeOf(u32));
    const a = try runner.alloc(.system, elements * @sizeOf(u32));
    const b = try runner.alloc(.system, elements * @sizeOf(u32));
    for (a.slice(u32)[0..elements], b.slice(u32)[0..elements], 0..) |*av, *bv, i| {
        av.* = @truncate(i *% 747796405 +% 2891336453);
        bv.* = @truncate(i *% 277803737 +% 1013904223);
    }

    setPtr(params.bytes, vulcan_kernel.launch.params[0].offset, out.va);
    setPtr(params.bytes, vulcan_kernel.launch.params[1].offset, a.va);
    setPtr(params.bytes, vulcan_kernel.launch.params[2].offset, b.va);
    const vulcan_grid: compute.Grid = .{
        .grid = .{ elements / 256, 1, 1 },
        .block = vulcan_kernel.launch.block,
        .register_count = vulcan_kernel.launch.reg_count,
        .cbuf0_va = params.va,
        .cbuf0_size = @max(vulcan_kernel.launch.param_bytes, 16),
        .shared_mem_bytes = vulcan_kernel.launch.shared_bytes,
    };
    const vulcan_run = try benchmarkKernel(io, &runner, vulcan_kernel.code, vulcan_grid);
    const vulcan_checksum = checksum(out.slice(u32)[0..elements]);

    @memset(out.slice(u32)[0..elements], 0);
    // CUDA 12.9 places the launch dimensions and parameters at these sm_120
    // constant-bank offsets. The checksum validates this ABI against Vulcan.
    setU32(params.bytes, 0x360, 256);
    setPtr(params.bytes, 0x380, out.va);
    setPtr(params.bytes, 0x388, a.va);
    setPtr(params.bytes, 0x390, b.va);
    const ptxas_grid: compute.Grid = .{
        .grid = .{ elements / 256, 1, 1 },
        .block = .{ 256, 1, 1 },
        .register_count = registerAllocation(ptxas_kernel.registers),
        .cbuf0_va = params.va,
        .cbuf0_size = 0x398,
        .shared_mem_bytes = 0,
    };
    const ptxas_run = try benchmarkKernel(io, &runner, ptxas_kernel.code, ptxas_grid);
    const ptxas_checksum = checksum(out.slice(u32)[0..elements]);
    if (vulcan_checksum != ptxas_checksum) return error.ChecksumMismatch;

    @memset(out.slice(u32)[0..elements], 0);
    var nvcc_grid = ptxas_grid;
    nvcc_grid.register_count = registerAllocation(nvcc_kernel.registers);
    const nvcc_run = try benchmarkKernel(io, &runner, nvcc_kernel.code, nvcc_grid);
    const nvcc_checksum = checksum(out.slice(u32)[0..elements]);
    if (vulcan_checksum != nvcc_checksum) return error.ChecksumMismatch;

    try writer.print(
        "\nCUDA benchmark (best/median of {d}; lower is better)\n" ++
            "vector_mix processes {d} u32 elements on the GPU.\n" ++
            "External compiler process startup is included.\n\n" ++
            "compiler       best       median   time/vulcan  code B  launch GPRs\n" ++
            "--------  ---------  -----------  ------------  ------  -----------\n",
        .{ samples, elements },
    );
    try printRow(writer, "vulcan", vulcan_compile, vulcan_compile.best_ns, vulcan_kernel.code.len * @sizeOf(u32), vulcan_kernel.launch.reg_count);
    try printRow(writer, "ptxas", ptxas_compile.timing, vulcan_compile.best_ns, ptxas_kernel.code.len * @sizeOf(u32), registerAllocation(ptxas_kernel.registers));
    try printRow(writer, "nvcc", nvcc_compile.timing, vulcan_compile.best_ns, nvcc_kernel.code.len * @sizeOf(u32), registerAllocation(nvcc_kernel.registers));
    try writer.writeAll("\nwarm dispatch and synchronization\n");
    try writer.writeAll("compiler       best       median   time/vulcan  checksum\n");
    try writer.writeAll("--------  ---------  -----------  ------------  --------\n");
    try printRunRow(writer, "vulcan", vulcan_run, vulcan_run.best_ns, vulcan_checksum);
    try printRunRow(writer, "ptxas", ptxas_run, vulcan_run.best_ns, ptxas_checksum);
    try printRunRow(writer, "nvcc", nvcc_run, vulcan_run.best_ns, nvcc_checksum);
    try writer.flush();
}

fn usage() error{InvalidArguments} {
    std.debug.print("usage: vulcan-cuda-bench <ptx-source> <cuda-source> <output-dir>\n", .{});
    return error.InvalidArguments;
}

fn noGpu(err: anyerror) bool {
    return switch (err) {
        error.SkipZigTest,
        error.NoDevice,
        error.FileNotFound,
        error.AccessDenied,
        error.PermissionDenied,
        => true,
        else => false,
    };
}

fn buildVectorMix(allocator: std.mem.Allocator) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const block = try func.appendBlock();
    const out = try func.appendBlockParam(block, ptr_t);
    const a = try func.appendBlockParam(block, ptr_t);
    const b = try func.appendBlockParam(block, ptr_t);
    // These parameters are pointers to u32, so their language-level contract guarantees the
    // alignment required by one B32 transaction. Without the IR fact, the NVIDIA backend must
    // conservatively preserve an arbitrary byte address with four U8 accesses.
    try func.addAttr(.{ .value = out }, .{ .@"align" = @alignOf(u32) });
    try func.addAttr(.{ .value = a }, .{ .@"align" = @alignOf(u32) });
    try func.addAttr(.{ .value = b }, .{ .@"align" = @alignOf(u32) });
    const gid = try func.appendBlockParam(block, u32_t);
    try gpu.attrs.setBuiltin(&func, gid, .global_id_x);
    try gpu.attrs.setLocalSize(&func, .{ 256, 1, 1 });

    const byte = try imm(&func, block, u32_t, .shl, gid, 2);
    const av = try loadAt(&func, block, u32_t, ptr_t, a, byte);
    const bv = try loadAt(&func, block, u32_t, ptr_t, b, byte);
    const scaled = try imm(&func, block, u32_t, .mul, av, 1664525);
    var x = try bin(&func, block, u32_t, .add, scaled, bv);
    x = try bin(&func, block, u32_t, .bit_xor, x, try imm(&func, block, u32_t, .shr, x, 16));
    x = try imm(&func, block, u32_t, .mul, x, 2246822519);
    x = try bin(&func, block, u32_t, .bit_xor, x, try imm(&func, block, u32_t, .shr, x, 13));
    const out_ptr = try bin(&func, block, ptr_t, .add, out, byte);
    try func.appendStore(block, x, out_ptr);
    func.setTerminator(block, .{ .ret = ir.function.Ret.none() });
    return func;
}

fn loadAt(func: *Function, block: Block, ty: ir.types.Type, ptr_t: ir.types.Type, base: Value, byte: Value) !Value {
    const ptr = try bin(func, block, ptr_t, .add, base, byte);
    return func.appendInst(block, ty, .{ .load = .{ .ptr = ptr } });
}

fn bin(func: *Function, block: Block, ty: ir.types.Type, op: ir.function.BinOp, lhs: Value, rhs: Value) !Value {
    return func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
}

fn imm(func: *Function, block: Block, ty: ir.types.Type, op: ir.function.BinOp, lhs: Value, value: i64) !Value {
    return func.appendArithImm(block, ty, op, lhs, value);
}

fn benchmarkVulcanCompile(allocator: std.mem.Allocator, io: std.Io, func: *Function) !Timing {
    var warm = try target.nvidia.isel.compileKernel(allocator, func, runner_abi);
    warm.deinit(allocator);
    var times: [samples]u64 = undefined;
    for (&times) |*elapsed| {
        const start = std.Io.Timestamp.now(io, .awake);
        var kernel = try target.nvidia.isel.compileKernel(allocator, func, runner_abi);
        const finish = std.Io.Timestamp.now(io, .awake);
        kernel.deinit(allocator);
        elapsed.* = @intCast(start.durationTo(finish).nanoseconds);
    }
    return timing(&times);
}

const PtxasTiming = struct {
    timing: Timing,
    registers: u32,
};

fn benchmarkPtxasCompile(allocator: std.mem.Allocator, io: std.Io, source: []const u8, output: []const u8) !PtxasTiming {
    _ = try runPtxas(allocator, io, source, output);
    var times: [samples]u64 = undefined;
    var registers: u32 = 0;
    for (&times) |*elapsed| {
        const start = std.Io.Timestamp.now(io, .awake);
        registers = try runPtxas(allocator, io, source, output);
        const finish = std.Io.Timestamp.now(io, .awake);
        elapsed.* = @intCast(start.durationTo(finish).nanoseconds);
    }
    return .{ .timing = timing(&times), .registers = registers };
}

fn benchmarkNvccCompile(allocator: std.mem.Allocator, io: std.Io, source: []const u8, output: []const u8) !PtxasTiming {
    _ = try runNvcc(allocator, io, source, output);
    var times: [samples]u64 = undefined;
    var registers: u32 = 0;
    for (&times) |*elapsed| {
        const start = std.Io.Timestamp.now(io, .awake);
        registers = try runNvcc(allocator, io, source, output);
        const finish = std.Io.Timestamp.now(io, .awake);
        elapsed.* = @intCast(start.durationTo(finish).nanoseconds);
    }
    return .{ .timing = timing(&times), .registers = registers };
}

fn runPtxas(allocator: std.mem.Allocator, io: std.Io, source: []const u8, output: []const u8) !u32 {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ "ptxas", "-O3", "-arch=sm_120", "-v", source, "-o", output },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("ptxas failed (exit {d}):\n{s}\n", .{ code, result.stderr });
            return error.CompileFailed;
        },
        else => return error.CompileFailed,
    }
    return parseRegisterCount(result.stderr);
}

fn runNvcc(allocator: std.mem.Allocator, io: std.Io, source: []const u8, output: []const u8) !u32 {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ "nvcc", "-O3", "-arch=sm_120", "--cubin", "--ptxas-options=-v", source, "-o", output },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("nvcc failed (exit {d}):\n{s}\n", .{ code, result.stderr });
            return error.CompileFailed;
        },
        else => return error.CompileFailed,
    }
    return parseRegisterCount(result.stderr);
}

fn parseRegisterCount(log: []const u8) !u32 {
    const marker = "Used ";
    const start = (std.mem.indexOf(u8, log, marker) orelse return error.MissingRegisterCount) + marker.len;
    const end = std.mem.indexOfScalarPos(u8, log, start, ' ') orelse return error.MissingRegisterCount;
    return std.fmt.parseInt(u32, log[start..end], 10);
}

fn loadPtxasKernel(allocator: std.mem.Allocator, io: std.Io, path: []const u8, registers: u32) !PtxasKernel {
    const image = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(16 * 1024 * 1024));
    const bytes = try elfSection(image, ".text.vector_mix");
    if (bytes.len % @sizeOf(u32) != 0) return error.MalformedCubin;
    const code = try allocator.alloc(u32, bytes.len / @sizeOf(u32));
    @memcpy(std.mem.sliceAsBytes(code), bytes);
    return .{ .code = code, .registers = registers };
}

fn elfSection(image: []const u8, wanted: []const u8) ![]const u8 {
    if (image.len < 64 or !std.mem.eql(u8, image[0..4], "\x7fELF") or image[4] != 2 or image[5] != 1) return error.MalformedCubin;
    const shoff = readInt(u64, image, 40) orelse return error.MalformedCubin;
    const shentsize = readInt(u16, image, 58) orelse return error.MalformedCubin;
    const shnum = readInt(u16, image, 60) orelse return error.MalformedCubin;
    const shstrndx = readInt(u16, image, 62) orelse return error.MalformedCubin;
    if (shentsize < 64 or shstrndx >= shnum) return error.MalformedCubin;
    const names_header = sectionHeader(image, shoff, shentsize, shstrndx) orelse return error.MalformedCubin;
    const names = sliceRange(image, names_header.offset, names_header.size) orelse return error.MalformedCubin;
    for (0..shnum) |index| {
        const header = sectionHeader(image, shoff, shentsize, @intCast(index)) orelse return error.MalformedCubin;
        const name = stringAt(names, header.name) orelse return error.MalformedCubin;
        if (std.mem.eql(u8, name, wanted)) return sliceRange(image, header.offset, header.size) orelse error.MalformedCubin;
    }
    return error.MissingKernel;
}

const SectionHeader = struct {
    name: u32,
    offset: u64,
    size: u64,
};

fn sectionHeader(image: []const u8, shoff: u64, size: u16, index: u16) ?SectionHeader {
    const offset = std.math.add(u64, shoff, std.math.mul(u64, size, index) catch return null) catch return null;
    return .{
        .name = readInt(u32, image, offset) orelse return null,
        .offset = readInt(u64, image, offset + 24) orelse return null,
        .size = readInt(u64, image, offset + 32) orelse return null,
    };
}

fn readInt(comptime T: type, bytes: []const u8, offset: u64) ?T {
    const start: usize = std.math.cast(usize, offset) orelse return null;
    const end = std.math.add(usize, start, @sizeOf(T)) catch return null;
    if (end > bytes.len) return null;
    return std.mem.readInt(T, bytes[start..][0..@sizeOf(T)], .little);
}

fn sliceRange(bytes: []const u8, offset: u64, size: u64) ?[]const u8 {
    const start: usize = std.math.cast(usize, offset) orelse return null;
    const len: usize = std.math.cast(usize, size) orelse return null;
    const end = std.math.add(usize, start, len) catch return null;
    if (end > bytes.len) return null;
    return bytes[start..end];
}

fn stringAt(table: []const u8, offset: u32) ?[]const u8 {
    if (offset >= table.len) return null;
    const start: usize = offset;
    const end = std.mem.indexOfScalarPos(u8, table, start, 0) orelse return null;
    return table[start..end];
}

fn benchmarkKernel(io: std.Io, runner: *compute.Runner, code: []const u32, grid: compute.Grid) !Timing {
    try runner.run(code, grid);
    var times: [samples]u64 = undefined;
    for (&times) |*elapsed| {
        const start = std.Io.Timestamp.now(io, .awake);
        try runner.run(code, grid);
        const finish = std.Io.Timestamp.now(io, .awake);
        elapsed.* = @intCast(start.durationTo(finish).nanoseconds);
    }
    return timing(&times);
}

fn timing(times: *[samples]u64) Timing {
    std.mem.sort(u64, times, {}, std.sort.asc(u64));
    return .{ .best_ns = times[0], .median_ns = times[samples / 2] };
}

fn setPtr(bytes: []u8, offset: u32, address: u64) void {
    std.mem.writeInt(u64, bytes[offset..][0..8], address, .little);
}

fn setU32(bytes: []u8, offset: u32, value: u32) void {
    std.mem.writeInt(u32, bytes[offset..][0..4], value, .little);
}

fn registerAllocation(used: u32) u32 {
    return @max(16, (used + 2 + 7) & ~@as(u32, 7));
}

fn checksum(values: []const u32) u64 {
    var result: u64 = 0xcbf29ce484222325;
    for (values) |value| result = (result ^ value) *% 0x100000001b3;
    return result;
}

fn programAvailable(allocator: std.mem.Allocator, io: std.Io, program: []const u8) !bool {
    const result = std.process.run(allocator, io, .{ .argv = &.{ program, "--version" } }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return result.term == .exited and result.term.exited == 0;
}

fn printRow(writer: *std.Io.Writer, name: []const u8, result: Timing, baseline: u64, code_bytes: usize, registers: u32) !void {
    try writer.print("{s:<8}  ", .{name});
    try printDuration(writer, result.best_ns);
    try writer.writeAll("  ");
    try printDuration(writer, result.median_ns);
    try writer.print("  {d:>10}%  {d:>6}  {d:>9}\n", .{ @divTrunc(result.best_ns * 100, baseline), code_bytes, registers });
}

fn printRunRow(writer: *std.Io.Writer, name: []const u8, result: Timing, baseline: u64, sum: u64) !void {
    try writer.print("{s:<8}  ", .{name});
    try printDuration(writer, result.best_ns);
    try writer.writeAll("  ");
    try printDuration(writer, result.median_ns);
    try writer.print("  {d:>10}%  {x:0>16}\n", .{ @divTrunc(result.best_ns * 100, baseline), sum });
}

fn printDuration(writer: *std.Io.Writer, nanoseconds: u64) !void {
    const microseconds = nanoseconds / std.time.ns_per_us;
    try writer.print("{d:>5}.{d:0>3} ms", .{ microseconds / 1000, microseconds % 1000 });
}
