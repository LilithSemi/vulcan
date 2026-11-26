//! vulcan-forth: compile Forth source with the Vulcan engine.
//!   vulcan-forth --target host <file.fs>
//!   vulcan-forth --target geyser <file.fs> -o <out.bin>
//! JITs the program for the host, runs `main` and prints the I/O window writes, then the
//! data stack it left, deepest cell first as signed decimals.

const std = @import("std");
const forth = @import("vulcan-forth");

const usage = "usage: vulcan-forth --target host <file.fs>\n       vulcan-forth --target geyser <file.fs> -o <out.bin>\n";

/// Print what is queued on stderr and leave without a stack trace.
fn die(errw: *std.Io.Writer, code: u8) noreturn {
    errw.flush() catch {};
    std.process.exit(code);
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.arena.allocator();

    var errbuf: [512]u8 = undefined;
    var errfile = std.Io.File.stderr().writer(io, &errbuf);
    const errw = &errfile.interface;
    defer errw.flush() catch {};

    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip();

    var target: ?[]const u8 = null;
    var path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--target")) {
            target = it.next() orelse {
                errw.writeAll(usage) catch {};
                die(errw, 2);
            };
        } else if (std.mem.eql(u8, arg, "-o")) {
            out_path = it.next() orelse {
                errw.writeAll(usage) catch {};
                die(errw, 2);
            };
        } else if (path == null) {
            path = arg;
        } else {
            errw.writeAll(usage) catch {};
            die(errw, 2);
        }
    }
    const source_path = path orelse {
        errw.writeAll(usage) catch {};
        die(errw, 2);
    };
    const target_name = target orelse "";
    const is_geyser = std.mem.eql(u8, target_name, "geyser");
    if (!is_geyser and !std.mem.eql(u8, target_name, "host")) {
        errw.writeAll(usage) catch {};
        die(errw, 2);
    }
    if (is_geyser and out_path == null) {
        errw.writeAll(usage) catch {};
        die(errw, 2);
    }

    const source = try std.Io.Dir.cwd().readFileAlloc(io, source_path, allocator, .limited(16 * 1024 * 1024));

    var diag: forth.Diagnostic = .{};
    var program = forth.compile(allocator, source, &diag) catch |err| switch (err) {
        error.Compile => {
            errw.print("vulcan-forth: {s}\n", .{diag.message()}) catch {};
            die(errw, 1);
        },
        else => return err,
    };
    if (is_geyser) {
        var geyser_diag: forth.geyser.Diagnostic = .{};
        const image = forth.compileGeyser(allocator, &program, .{ .diag = &geyser_diag }) catch |err| {
            errw.print("vulcan-forth: {s}\n", .{@errorName(err)}) catch {};
            if (geyser_diag.len != 0) errw.print("vulcan-forth: {s}\n", .{geyser_diag.message()}) catch {};
            die(errw, 1);
        };
        var file = try std.Io.Dir.cwd().createFile(io, out_path.?, .{});
        defer file.close(io);
        var buf: [512]u8 = undefined;
        var writer = file.writer(io, &buf);
        try forth.geyserWriteImage(image.words, &writer.interface);
        try writer.interface.flush();
        errw.print("vulcan-forth: {d} words, {d} slots\n", .{ image.words.len, image.slots }) catch {};
        return;
    }
    var machine = try forth.Machine.init(allocator, program);
    defer machine.deinit();
    const stack = machine.runMain() catch |err| {
        errw.print("vulcan-forth: {s}\n", .{@errorName(err)}) catch {};
        die(errw, 1);
    };

    var buf: [512]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buf);
    const w = &out.interface;
    for (machine.io_writes.items) |write| try w.print("io 0x{X:0>4} <- 0x{X:0>4}\n", .{ write.addr, write.value });
    for (stack, 0..) |cell, k| {
        if (k > 0) try w.writeByte(' ');
        try w.print("{d}", .{@as(i16, @bitCast(cell))});
    }
    try w.writeByte('\n');
    try w.flush();
}
