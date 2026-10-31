//! End-to-end C frontend compile-time benchmark driver.
//! VCC is supplied by the build graph; Clang and GCC are included when present on PATH.

const std = @import("std");

const samples = 7;

const Compiler = struct {
    name: []const u8,
    command: []const u8,
    required: bool,
};

const Result = struct {
    best_ns: u64,
    median_ns: u64,
    object_bytes: u64,
};

const ExecutionResult = struct {
    best_ns: u64,
    median_ns: u64,
    checksum: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const io = init.io;

    var args = try init.minimal.args.iterateAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const vcc = args.next() orelse return usage();
    const source_dir = args.next() orelse return usage();
    const output_dir = args.next() orelse return usage();
    if (args.next() != null) return usage();

    try std.Io.Dir.cwd().createDirPath(io, output_dir);
    var workloads = try discoverWorkloads(allocator, io, source_dir);
    defer workloads.deinit(allocator);
    if (workloads.items.len == 0) return error.NoWorkloads;

    const candidates = [_]Compiler{
        .{ .name = "vcc", .command = vcc, .required = true },
        .{ .name = "clang", .command = "clang", .required = false },
        .{ .name = "gcc", .command = "gcc", .required = false },
    };
    var compilers: std.ArrayList(Compiler) = .empty;
    defer compilers.deinit(allocator);
    var skipped: std.ArrayList([]const u8) = .empty;
    defer skipped.deinit(allocator);
    for (candidates) |compiler| {
        if (try compilerAvailable(allocator, io, compiler.command)) {
            try compilers.append(allocator, compiler);
        } else if (compiler.required) {
            return error.VccUnavailable;
        } else {
            try skipped.append(allocator, compiler.name);
        }
    }

    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &buffer);
    const writer = &stdout.interface;
    try writer.print(
        "C frontend benchmark (-O2 -c, best/median of {d}; lower is better)\n" ++
            "VCC is built ReleaseFast; compiler process startup is included.\n",
        .{samples},
    );
    if (skipped.items.len != 0) {
        try writer.writeAll("optional compilers not found:");
        for (skipped.items) |name| try writer.print(" {s}", .{name});
        try writer.writeByte('\n');
    }
    try writer.writeAll("\nworkload             compiler       best       median   time/vcc  object B\n");
    try writer.writeAll("-------------------  --------  ---------  -----------  ---------  --------\n");

    var totals = try allocator.alloc(u64, compilers.items.len);
    @memset(totals, 0);
    for (workloads.items) |workload| {
        const source_path = try std.fs.path.join(allocator, &.{ source_dir, workload });
        var vcc_best: u64 = 0;
        for (compilers.items, 0..) |compiler, compiler_index| {
            const stem = workload[0 .. workload.len - 2];
            const output_name = try std.fmt.allocPrint(allocator, "{s}.{s}.o", .{ stem, compiler.name });
            const output_path = try std.fs.path.join(allocator, &.{ output_dir, output_name });
            const result = try benchmarkCompiler(allocator, io, compiler, source_path, output_path);
            if (compiler_index == 0) vcc_best = result.best_ns;
            totals[compiler_index] += result.best_ns;
            try writer.print("{s:<19}  {s:<8}  ", .{ workload, compiler.name });
            try printDuration(writer, result.best_ns);
            try writer.writeAll("  ");
            try printDuration(writer, result.median_ns);
            try writer.print("  {d:>7}%  {d:>8}\n", .{
                @divTrunc(result.best_ns * 100, vcc_best),
                result.object_bytes,
            });
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("aggregate best time\n");
    const vcc_total = totals[0];
    for (compilers.items, 0..) |compiler, index| {
        try writer.print("  {s:<8}  ", .{compiler.name});
        try printDuration(writer, totals[index]);
        try writer.print("  {d:>7}% of vcc\n", .{@divTrunc(totals[index] * 100, vcc_total)});
    }

    @memset(totals, 0);
    try writer.writeAll("\ngenerated-code execution (best/median of 7; lower is better)\n");
    try writer.writeAll("Checksums must match across compilers; process startup is included.\n\n");
    try writer.writeAll("workload             compiler       best       median   time/vcc  checksum\n");
    try writer.writeAll("-------------------  --------  ---------  -----------  ---------  --------\n");
    for (workloads.items) |workload| {
        const source_path = try std.fs.path.join(allocator, &.{ source_dir, workload });
        var vcc_best: u64 = 0;
        var expected_checksum: ?[]const u8 = null;
        for (compilers.items, 0..) |compiler, compiler_index| {
            const stem = workload[0 .. workload.len - 2];
            const output_name = try std.fmt.allocPrint(allocator, "{s}.{s}.bench", .{ stem, compiler.name });
            const output_path = try std.fs.path.join(allocator, &.{ output_dir, output_name });
            try buildExecutable(allocator, io, compiler, source_path, output_path);
            const result = try benchmarkExecutable(allocator, io, compiler, output_path);
            if (expected_checksum) |expected| {
                if (!std.mem.eql(u8, expected, result.checksum)) {
                    std.debug.print("checksum mismatch for {s}: vcc={s}, {s}={s}\n", .{ workload, expected, compiler.name, result.checksum });
                    return error.ChecksumMismatch;
                }
            } else {
                expected_checksum = result.checksum;
            }
            if (compiler_index == 0) vcc_best = result.best_ns;
            totals[compiler_index] += result.best_ns;
            try writer.print("{s:<19}  {s:<8}  ", .{ workload, compiler.name });
            try printDuration(writer, result.best_ns);
            try writer.writeAll("  ");
            try printDuration(writer, result.median_ns);
            try writer.print("  {d:>7}%  {s}\n", .{
                @divTrunc(result.best_ns * 100, vcc_best),
                result.checksum,
            });
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("aggregate best execution time\n");
    const vcc_execution_total = totals[0];
    for (compilers.items, 0..) |compiler, index| {
        try writer.print("  {s:<8}  ", .{compiler.name});
        try printDuration(writer, totals[index]);
        try writer.print("  {d:>7}% of vcc\n", .{@divTrunc(totals[index] * 100, vcc_execution_total)});
    }
    try writer.flush();
}

fn usage() error{InvalidArguments} {
    std.debug.print("usage: vulcan-frontend-bench <vcc> <source-dir> <output-dir>\n", .{});
    return error.InvalidArguments;
}

fn discoverWorkloads(allocator: std.mem.Allocator, io: std.Io, source_dir: []const u8) !std.ArrayList([]const u8) {
    var result: std.ArrayList([]const u8) = .empty;
    errdefer result.deinit(allocator);
    var dir = try std.Io.Dir.cwd().openDir(io, source_dir, .{ .iterate = true });
    defer dir.close(io);
    var iterator = dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, ".c")) continue;
        try result.append(allocator, try allocator.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, result.items, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);
    return result;
}

fn compilerAvailable(allocator: std.mem.Allocator, io: std.Io, command: []const u8) !bool {
    const result = std.process.run(allocator, io, .{ .argv = &.{ command, "--version" } }) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    return switch (result.term) {
        .exited => |code| code == 0,
        else => false,
    };
}

fn benchmarkCompiler(allocator: std.mem.Allocator, io: std.Io, compiler: Compiler, source: []const u8, output: []const u8) !Result {
    _ = try compileOnce(allocator, io, compiler, source, output, false);
    var times: [samples]u64 = undefined;
    for (&times) |*elapsed| elapsed.* = try compileOnce(allocator, io, compiler, source, output, true);
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    const object_bytes = (try std.Io.Dir.cwd().statFile(io, output, .{})).size;
    return .{ .best_ns = times[0], .median_ns = times[samples / 2], .object_bytes = object_bytes };
}

fn compileOnce(allocator: std.mem.Allocator, io: std.Io, compiler: Compiler, source: []const u8, output: []const u8, timed: bool) !u64 {
    const start = if (timed) std.Io.Timestamp.now(io, .awake) else undefined;
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ compiler.command, "-O2", "-c", source, "-o", output },
    });
    const finish = if (timed) std.Io.Timestamp.now(io, .awake) else undefined;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("{s} failed compiling {s} (exit {d}):\n{s}\n", .{ compiler.name, source, code, result.stderr });
            return error.CompileFailed;
        },
        else => {
            std.debug.print("{s} terminated abnormally while compiling {s}\n", .{ compiler.name, source });
            return error.CompileFailed;
        },
    }
    return if (timed) @intCast(start.durationTo(finish).nanoseconds) else 0;
}

fn buildExecutable(allocator: std.mem.Allocator, io: std.Io, compiler: Compiler, source: []const u8, output: []const u8) !void {
    const result = try std.process.run(allocator, io, .{
        .argv = &.{ compiler.command, "-O2", "-DBENCH_EXECUTE", source, "-o", output },
    });
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    switch (result.term) {
        .exited => |code| if (code != 0) {
            std.debug.print("{s} failed building executable for {s} (exit {d}):\n{s}\n", .{ compiler.name, source, code, result.stderr });
            return error.CompileFailed;
        },
        else => return error.CompileFailed,
    }
}

fn benchmarkExecutable(allocator: std.mem.Allocator, io: std.Io, compiler: Compiler, executable: []const u8) !ExecutionResult {
    var checksum: ?[]const u8 = null;
    var times: [samples]u64 = undefined;
    for (&times) |*elapsed| {
        const start = std.Io.Timestamp.now(io, .awake);
        const result = try std.process.run(allocator, io, .{ .argv = &.{executable} });
        const finish = std.Io.Timestamp.now(io, .awake);
        defer allocator.free(result.stdout);
        defer allocator.free(result.stderr);
        switch (result.term) {
            .exited => |code| if (code != 0) {
                std.debug.print("{s} executable {s} failed (exit {d}):\n{s}\n", .{ compiler.name, executable, code, result.stderr });
                return error.ExecutionFailed;
            },
            else => return error.ExecutionFailed,
        }
        const current = std.mem.trim(u8, result.stdout, " \t\r\n");
        if (checksum) |expected| {
            if (!std.mem.eql(u8, expected, current)) {
                std.debug.print("unstable checksum from {s} executable {s}: first={s}, current={s}\n", .{ compiler.name, executable, expected, current });
                return error.UnstableChecksum;
            }
        } else {
            checksum = try allocator.dupe(u8, current);
        }
        elapsed.* = @intCast(start.durationTo(finish).nanoseconds);
    }
    std.mem.sort(u64, &times, {}, std.sort.asc(u64));
    return .{ .best_ns = times[0], .median_ns = times[samples / 2], .checksum = checksum.? };
}

fn printDuration(writer: *std.Io.Writer, nanoseconds: u64) !void {
    const microseconds = nanoseconds / std.time.ns_per_us;
    try writer.print("{d:>5}.{d:0>3} ms", .{ microseconds / 1000, microseconds % 1000 });
}
