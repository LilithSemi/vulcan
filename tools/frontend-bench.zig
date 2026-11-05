//! End-to-end C frontend compile-time benchmark driver.
//! VCC is supplied by the build graph. Clang, GCC, and TCC are included when present on PATH.

const std = @import("std");
const builtin = @import("builtin");

const samples = 7;

/// The system this runs on, in the names Nix uses, such as `aarch64-darwin`.
const system = @tagName(builtin.cpu.arch) ++ "-" ++ (if (builtin.os.tag == .macos) "darwin" else @tagName(builtin.os.tag));

/// VCC links ELF on Linux and Mach-O through the system linker on Darwin.
const executes = builtin.os.tag == .linux or builtin.os.tag.isDarwin();

const Compiler = struct {
    name: []const u8,
    command: []const u8,
    required: bool,
    environ_map: ?*const std.process.Environ.Map = null,
    /// Its code is the same as another row, so it is only in the compile table.
    compile_only: bool = false,
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

const Format = enum { text, github };

const Report = struct {
    /// The microarch model vcc tunes for here, from `vcc -print-cpu-model`.
    cpu_model: []const u8,
    compilers: []const Compiler,
    skipped: []const []const u8,
    workloads: []const []const u8,
    /// Indexed by workload, then compiler.
    compile: []const []const Result,
    /// Indexed by workload, then compiler. Null for a compile-only row.
    execute: []const []const ?ExecutionResult,

    fn compileTotal(report: Report, compiler: usize) u64 {
        var total: u64 = 0;
        for (report.compile) |row| total += row[compiler].best_ns;
        return total;
    }

    fn executeTotal(report: Report, compiler: usize) u64 {
        var total: u64 = 0;
        for (report.execute) |row| total += if (row[compiler]) |result| result.best_ns else 0;
        return total;
    }
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
    var format: Format = .text;
    while (args.next()) |arg| {
        if (!std.mem.eql(u8, arg, "--format")) return usage();
        const name = args.next() orelse return usage();
        format = std.meta.stringToEnum(Format, name) orelse return usage();
    }

    try std.Io.Dir.cwd().createDirPath(io, output_dir);
    var workloads = try discoverWorkloads(allocator, io, source_dir);
    defer workloads.deinit(allocator);
    if (workloads.items.len == 0) return error.NoWorkloads;

    // A cold VCC runs without its cache, so the cost of the cache misses is visible.
    var cold_env = try init.environ_map.clone(allocator);
    try cold_env.put("VCC_NO_CACHE", "1");
    const candidates = [_]Compiler{
        .{ .name = "vcc", .command = vcc, .required = true },
        .{ .name = "vcc-cold", .command = vcc, .required = true, .environ_map = &cold_env, .compile_only = true },
        .{ .name = "clang", .command = "clang", .required = false },
        .{ .name = "gcc", .command = "gcc", .required = false },
        .{ .name = "tcc", .command = "tcc", .required = false },
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

    const compile = try allocator.alloc([]Result, workloads.items.len);
    const execute = try allocator.alloc([]?ExecutionResult, workloads.items.len);
    for (workloads.items, compile, execute) |workload, *compile_row, *execute_row| {
        const source_path = try std.fs.path.join(allocator, &.{ source_dir, workload });
        const stem = workload[0 .. workload.len - 2];
        compile_row.* = try allocator.alloc(Result, compilers.items.len);
        execute_row.* = try allocator.alloc(?ExecutionResult, compilers.items.len);
        var expected_checksum: ?[]const u8 = null;
        for (compilers.items, compile_row.*, execute_row.*) |compiler, *compiled, *executed| {
            const object_name = try std.fmt.allocPrint(allocator, "{s}.{s}.o", .{ stem, compiler.name });
            const object_path = try std.fs.path.join(allocator, &.{ output_dir, object_name });
            compiled.* = try benchmarkCompiler(allocator, io, compiler, source_path, object_path);
            executed.* = null;
            if (compiler.compile_only or !executes) continue;

            const exe_name = try std.fmt.allocPrint(allocator, "{s}.{s}.bench", .{ stem, compiler.name });
            const exe_path = try std.fs.path.join(allocator, &.{ output_dir, exe_name });
            try buildExecutable(allocator, io, compiler, source_path, exe_path);
            const result = try benchmarkExecutable(allocator, io, compiler, exe_path);
            if (expected_checksum) |expected| {
                if (!sameChecksum(expected, result.checksum)) {
                    std.debug.print("checksum mismatch for {s}: vcc={s}, {s}={s}\n", .{ workload, expected, compiler.name, result.checksum });
                    return error.ChecksumMismatch;
                }
            } else {
                expected_checksum = result.checksum;
            }
            executed.* = result;
        }
    }

    const cpu_model = try cpuModel(allocator, io, vcc);
    const report: Report = .{
        .cpu_model = cpu_model,
        .compilers = compilers.items,
        .skipped = skipped.items,
        .workloads = workloads.items,
        .compile = compile,
        .execute = execute,
    };
    var buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &buffer);
    const writer = &stdout.interface;
    switch (format) {
        .text => try writeText(writer, report),
        .github => try writeGithub(writer, report),
    }
    try writer.flush();
}

/// Whether two checksums agree. Integer output must match exactly. A floating point result may
/// differ in its last bits, since C lets a compiler fuse a multiply and an add, so decimal output
/// matches within a relative 1e-9.
fn sameChecksum(expected: []const u8, actual: []const u8) bool {
    if (std.mem.eql(u8, expected, actual)) return true;
    var left = std.mem.tokenizeAny(u8, expected, " \t\r\n");
    var right = std.mem.tokenizeAny(u8, actual, " \t\r\n");
    while (true) {
        const a = left.next();
        const b = right.next();
        if (a == null or b == null) return a == null and b == null;
        if (std.mem.eql(u8, a.?, b.?)) continue;
        if (std.mem.indexOfScalar(u8, a.?, '.') == null or std.mem.indexOfScalar(u8, b.?, '.') == null) return false;
        const x = std.fmt.parseFloat(f64, a.?) catch return false;
        const y = std.fmt.parseFloat(f64, b.?) catch return false;
        if (@abs(x - y) > 1e-9 * @max(@abs(x), @abs(y))) return false;
    }
}

test "sameChecksum allows a last-bits float difference and nothing else" {
    try std.testing.expect(sameChecksum("12496323.004", "12496323.002"));
    try std.testing.expect(!sameChecksum("12496323.004", "12496323.104"));
    try std.testing.expect(!sameChecksum("918405997", "918405996"));
    try std.testing.expect(sameChecksum("5 6 7", "5 6 7"));
    try std.testing.expect(!sameChecksum("5 6", "5 6 7"));
}

/// The model vcc picks for this CPU, or `unsupported (<cpu>)` when it has none.
fn cpuModel(allocator: std.mem.Allocator, io: std.Io, vcc: []const u8) ![]const u8 {
    const result = std.process.run(allocator, io, .{ .argv = &.{ vcc, "-print-cpu-model" } }) catch return "unknown";
    const trimmed = std.mem.trim(u8, result.stdout, " \t\r\n");
    return if (trimmed.len == 0) "unknown" else trimmed;
}

fn percent(value: u64, base: u64) u64 {
    return @divTrunc(value * 100, @max(base, 1));
}

fn writeText(writer: *std.Io.Writer, report: Report) !void {
    try writer.print(
        "C frontend benchmark (-O2 -c, best/median of {d}; lower is better)\n" ++
            "VCC is built ReleaseFast; compiler process startup is included.\n",
        .{samples},
    );
    try writer.print("CPU model: {s}\n", .{report.cpu_model});
    if (report.skipped.len != 0) {
        try writer.writeAll("optional compilers not found:");
        for (report.skipped) |name| try writer.print(" {s}", .{name});
        try writer.writeByte('\n');
    }
    try writer.writeAll("\nworkload             compiler       best       median   time/vcc  object B\n");
    try writer.writeAll("-------------------  --------  ---------  -----------  ---------  --------\n");
    for (report.workloads, report.compile) |workload, row| {
        for (report.compilers, row) |compiler, result| {
            try writer.print("{s:<19}  {s:<8}  ", .{ workload, compiler.name });
            try printDuration(writer, result.best_ns);
            try writer.writeAll("  ");
            try printDuration(writer, result.median_ns);
            try writer.print("  {d:>7}%  {d:>8}\n", .{ percent(result.best_ns, row[0].best_ns), result.object_bytes });
        }
        try writer.writeByte('\n');
    }
    try writer.writeAll("aggregate best time\n");
    for (report.compilers, 0..) |compiler, index| {
        try writer.print("  {s:<8}  ", .{compiler.name});
        try printDuration(writer, report.compileTotal(index));
        try writer.print("  {d:>7}% of vcc\n", .{percent(report.compileTotal(index), report.compileTotal(0))});
    }

    if (!executes) {
        try writer.writeAll("\ngenerated code not timed: vcc cannot link executables on " ++ system ++ " yet\n");
        return;
    }
    try writer.print("\ngenerated-code execution (best/median of {d}; lower is better)\n", .{samples});
    try writer.writeAll("Checksums must match across compilers; process startup is included.\n\n");
    try writer.writeAll("workload             compiler       best       median   time/vcc  checksum\n");
    try writer.writeAll("-------------------  --------  ---------  -----------  ---------  --------\n");
    for (report.workloads, report.execute) |workload, row| {
        for (report.compilers, row) |compiler, maybe| {
            const result = maybe orelse continue;
            try writer.print("{s:<19}  {s:<8}  ", .{ workload, compiler.name });
            try printDuration(writer, result.best_ns);
            try writer.writeAll("  ");
            try printDuration(writer, result.median_ns);
            try writer.print("  {d:>7}%  {s}\n", .{ percent(result.best_ns, row[0].?.best_ns), result.checksum });
        }
        try writer.writeByte('\n');
    }
    try writer.writeAll("aggregate best execution time\n");
    for (report.compilers, 0..) |compiler, index| {
        if (compiler.compile_only) continue;
        try writer.print("  {s:<8}  ", .{compiler.name});
        try printDuration(writer, report.executeTotal(index));
        try writer.print("  {d:>7}% of vcc\n", .{percent(report.executeTotal(index), report.executeTotal(0))});
    }
}

/// Markdown for a GitHub comment. The totals are visible, and the per-workload tables are
/// in collapsed `<details>` blocks.
fn writeGithub(writer: *std.Io.Writer, report: Report) !void {
    try writer.writeAll("## Vulcan benchmark on " ++ system ++ "\n\n");
    try writer.print("CPU model: `{s}`\n\n", .{report.cpu_model});
    try writer.print("Best of {d} runs, lower is better. Process startup is included.", .{samples});
    if (report.skipped.len != 0) {
        try writer.writeAll(" Not found:");
        for (report.skipped) |name| try writer.print(" {s}", .{name});
        try writer.writeByte('.');
    }
    if (!executes) {
        try writer.writeAll(" Generated code is not timed, since vcc cannot link executables on " ++ system ++ " yet.");
        try writer.writeAll("\n\n| compiler | compile (`-O2 -c`) | vs vcc |\n|---|---:|---:|\n");
        for (report.compilers, 0..) |compiler, index| {
            try writer.print("| {s} | ", .{compiler.name});
            try writeMs(writer, report.compileTotal(index));
            try writer.print(" | {d}% |\n", .{percent(report.compileTotal(index), report.compileTotal(0))});
        }
        try writer.writeAll("\n<details>\n<summary>Compile time per workload</summary>\n\n");
        try writeCompileDetails(writer, report);
        return;
    }
    try writer.writeAll("\n\n| compiler | compile (`-O2 -c`) | vs vcc | generated code | vs vcc |\n");
    try writer.writeAll("|---|---:|---:|---:|---:|\n");
    for (report.compilers, 0..) |compiler, index| {
        try writer.print("| {s} | ", .{compiler.name});
        try writeMs(writer, report.compileTotal(index));
        try writer.print(" | {d}% | ", .{percent(report.compileTotal(index), report.compileTotal(0))});
        if (compiler.compile_only) {
            try writer.writeAll(" | |\n");
            continue;
        }
        try writeMs(writer, report.executeTotal(index));
        try writer.print(" | {d}% |\n", .{percent(report.executeTotal(index), report.executeTotal(0))});
    }

    try writer.writeAll("\n<details>\n<summary>Generated code per workload</summary>\n\n");
    try writeGithubHeader(writer, report, true);
    for (report.workloads, report.execute) |workload, row| {
        try writer.print("| {s} |", .{workload});
        for (report.compilers, row) |compiler, maybe| {
            if (compiler.compile_only) continue;
            try writer.writeByte(' ');
            try writeMs(writer, maybe.?.best_ns);
            try writer.print(" ({d}%) |", .{percent(maybe.?.best_ns, row[0].?.best_ns)});
        }
        try writer.print(" `{s}` |\n", .{row[0].?.checksum});
    }

    try writer.writeAll("\n</details>\n\n<details>\n<summary>Compile time per workload</summary>\n\n");
    try writeCompileDetails(writer, report);
}

/// The per-workload compile time and object size tables, the first already opened.
fn writeCompileDetails(writer: *std.Io.Writer, report: Report) !void {
    try writeGithubHeader(writer, report, false);
    for (report.workloads, report.compile) |workload, row| {
        try writer.print("| {s} |", .{workload});
        for (row) |result| {
            try writer.writeByte(' ');
            try writeMs(writer, result.best_ns);
            try writer.print(" ({d}%) |", .{percent(result.best_ns, row[0].best_ns)});
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("\n</details>\n\n<details>\n<summary>Object size per workload (bytes)</summary>\n\n");
    try writeGithubHeader(writer, report, false);
    for (report.workloads, report.compile) |workload, row| {
        try writer.print("| {s} |", .{workload});
        for (row) |result| try writer.print(" {d} |", .{result.object_bytes});
        try writer.writeByte('\n');
    }
    try writer.writeAll("\n</details>\n");
}

fn writeGithubHeader(writer: *std.Io.Writer, report: Report, execute: bool) !void {
    try writer.writeAll("| workload |");
    var columns: usize = 0;
    for (report.compilers) |compiler| {
        if (execute and compiler.compile_only) continue;
        try writer.print(" {s} |", .{compiler.name});
        columns += 1;
    }
    if (execute) try writer.writeAll(" checksum |");
    try writer.writeAll("\n|---|");
    for (0..columns) |_| try writer.writeAll("---:|");
    if (execute) try writer.writeAll("---|");
    try writer.writeByte('\n');
}

fn usage() error{InvalidArguments} {
    std.debug.print("usage: vulcan-frontend-bench <vcc> <source-dir> <output-dir> [--format text|github]\n", .{});
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
        .environ_map = compiler.environ_map,
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

fn writeMs(writer: *std.Io.Writer, nanoseconds: u64) !void {
    const microseconds = nanoseconds / std.time.ns_per_us;
    try writer.print("{d}.{d:0>3} ms", .{ microseconds / 1000, microseconds % 1000 });
}
