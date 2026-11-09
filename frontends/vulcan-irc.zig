//! vulcan-irc: compile Vulcan IR to a selected target representation.
//! The initial target is combinational Verilog netlist emission.
//! Usage: vulcan-irc --target netlist [--format verilog|json] input.vulcan [-o output]

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip();
    var target_name: ?[]const u8 = null;
    var format_name: ?[]const u8 = null;
    var input: ?[]const u8 = null;
    var module_name: []const u8 = "vulcan_module";
    var output: ?[]const u8 = null;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--target")) {
            target_name = it.next() orelse return error.Usage;
        } else if (std.mem.eql(u8, arg, "--format")) {
            format_name = it.next() orelse return error.Usage;
        } else if (std.mem.eql(u8, arg, "-o")) {
            output = it.next() orelse return error.Usage;
        } else if (input == null) {
            input = arg;
        } else {
            module_name = arg;
        }
    }
    const selected = target_name orelse return error.Usage;
    const is_netlist = std.mem.eql(u8, selected, "netlist");
    const is_aarch64 = std.mem.eql(u8, selected, "aarch64") or std.mem.eql(u8, selected, "arm64");
    const is_x86_64 = std.mem.eql(u8, selected, "x86_64") or std.mem.eql(u8, selected, "amd64");
    const is_x86 = std.mem.eql(u8, selected, "x86") or std.mem.eql(u8, selected, "i386");
    const is_riscv64 = std.mem.eql(u8, selected, "riscv64") or std.mem.eql(u8, selected, "rv64");
    if (!is_netlist and !is_aarch64 and !is_x86_64 and !is_x86 and !is_riscv64) return unsupportedTarget(selected);
    const input_path = input orelse return error.Usage;
    const text = try std.Io.Dir.cwd().readFileAlloc(init.io, input_path, allocator, .limited(64 * 1024 * 1024));
    if (is_netlist and startsStructuralCircuit(text)) {
        var parsed = ir.circuit_parser.parse(allocator, text) catch |err| {
            std.log.err("cannot parse structural circuit: {s}", .{@errorName(err)});
            return err;
        };
        defer parsed.circuit.deinit();
        const selected_format = format_name orelse "verilog";
        const circuit_name = if (std.mem.eql(u8, module_name, "vulcan_module")) parsed.name else module_name;
        const artifact = if (std.mem.eql(u8, selected_format, "verilog"))
            parsed.circuit.emitVerilog(allocator, circuit_name)
        else if (std.mem.eql(u8, selected_format, "json"))
            parsed.circuit.emitJson(allocator, circuit_name)
        else
            return unsupportedFormat(selected_format);
        const bytes = try artifact;
        if (output) |path| try writeFile(init.io, path, bytes) else try writeStdout(init.io, bytes);
        return;
    }
    var func = try ir.parser.parse(allocator, text);
    defer func.deinit();
    if (is_netlist) {
        const selected_format = format_name orelse "verilog";
        const artifact = if (std.mem.eql(u8, selected_format, "verilog"))
            ir.netlist.emitVerilog(allocator, &func, module_name)
        else if (std.mem.eql(u8, selected_format, "json"))
            ir.netlist.emitJson(allocator, &func, module_name)
        else
            return unsupportedFormat(selected_format);
        const bytes = artifact catch |err| {
            std.log.err("cannot compile IR for netlist: {s}", .{@errorName(err)});
            return err;
        };
        if (output) |path| {
            try writeFile(init.io, path, bytes);
        } else {
            try writeStdout(init.io, bytes);
        }
    } else {
        const selected_format = format_name orelse "elf";
        if (!std.mem.eql(u8, selected_format, "elf") and !std.mem.eql(u8, selected_format, "macho"))
            return unsupportedFormat(selected_format);
        if (std.mem.eql(u8, selected_format, "macho") and !is_aarch64)
            return unsupportedFormat("macho (only supported for aarch64)");
        const Arch = @import("vulcan-link").Arch;
        const arch: Arch = if (is_aarch64) .aarch64 else if (is_x86_64) .x86_64 else if (is_x86) .x86 else .riscv64;
        const funcs = [_]target.native.ModuleFunction{.{ .name = module_name, .func = &func }};
        const object = if (std.mem.eql(u8, selected_format, "elf"))
            target.native.writeObjectDataFor(allocator, arch, &funcs, &.{})
        else
            target.native.writeMachObjectDataForModel(allocator, &funcs, &.{}, null);
        const bytes = object catch |err| {
            std.log.err("cannot compile IR for {s}: {s}", .{ selected, @errorName(err) });
            return err;
        };
        const path = output orelse if (std.mem.eql(u8, selected_format, "elf")) "vulcan.o" else "vulcan.o";
        try writeFile(init.io, path, bytes);
    }
}

fn startsStructuralCircuit(text: []const u8) bool {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        return std.mem.startsWith(u8, line, "circuit ");
    }
    return false;
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    var file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    try file.writeStreamingAll(io, bytes);
}

fn writeStdout(io: std.Io, bytes: []const u8) !void {
    var buf: [4096]u8 = undefined;
    var w = std.Io.File.stdout().writer(io, &buf);
    try w.interface.writeAll(bytes);
    try w.interface.flush();
}

fn unsupportedTarget(name: []const u8) error{UnsupportedTarget} {
    std.log.err("target '{s}' is recognized but not implemented by vulcan-irc yet", .{name});
    return error.UnsupportedTarget;
}

fn unsupportedFormat(name: []const u8) error{UnsupportedFormat} {
    std.log.err("format '{s}' is not supported for this target", .{name});
    return error.UnsupportedFormat;
}
