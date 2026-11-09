//! Board-neutral blinky circuit built directly with Vulcan's circuit API.
//! The build graph can use this as an example artifact without making it a compiler frontend.

const std = @import("std");
const ir = @import("vulcan-ir");

pub fn build(allocator: std.mem.Allocator, width: u16) !ir.circuit.Circuit {
    if (width < 2 or width > 63) return error.InvalidWidth;
    var circuit = ir.circuit.Circuit.init(allocator);
    errdefer circuit.deinit();
    const clock = try circuit.appendInput("clk", 1);
    const reset = try circuit.appendInput("reset", 1);
    const counter = try circuit.appendRegister(clock, reset, width, 0);
    const one = try circuit.append(.{ .kind = .constant, .bits = width, .value = 1 });
    const zero = try circuit.append(.{ .kind = .constant, .bits = width, .value = 0 });
    const next = try circuit.append(.{ .kind = .add, .bits = width, .a = counter, .b = one });
    circuit.setRegisterNext(counter, next);
    const mask = try circuit.append(.{ .kind = .constant, .bits = width, .value = @as(i64, 1) << @intCast(width - 1) });
    const high = try circuit.append(.{ .kind = .bit_and, .bits = width, .a = counter, .b = mask });
    const led = try circuit.append(.{ .kind = .ne, .bits = 1, .a = high, .b = zero });
    try circuit.appendOutput("led", led, 1);
    return circuit;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    var it = try init.minimal.args.iterateAllocator(allocator);
    defer it.deinit();
    _ = it.skip();
    var width: u16 = 24;
    var format: []const u8 = "verilog";
    var output: ?[]const u8 = null;
    while (it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--")) continue else if (std.mem.eql(u8, arg, "--width")) width = try std.fmt.parseInt(u16, it.next() orelse return error.Usage, 10) else if (std.mem.eql(u8, arg, "--format")) format = it.next() orelse return error.Usage else if (std.mem.eql(u8, arg, "-o")) output = it.next() orelse return error.Usage else return error.Usage;
    }
    if (!std.mem.eql(u8, format, "verilog") and !std.mem.eql(u8, format, "json")) return error.UnsupportedFormat;
    var circuit = try build(allocator, width);
    defer circuit.deinit();
    const bytes = if (std.mem.eql(u8, format, "json")) try circuit.emitJson(allocator, "blinky") else try circuit.emitVerilog(allocator, "blinky");
    if (output) |path| try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = bytes }) else {
        var buffer: [4096]u8 = undefined;
        var writer = std.Io.File.stdout().writer(init.io, &buffer);
        try writer.interface.writeAll(bytes);
        try writer.interface.flush();
    }
}

test "blinky example emits a clocked counter" {
    var circuit = try build(std.testing.allocator, 4);
    defer circuit.deinit();
    const verilog = try circuit.emitVerilog(std.testing.allocator, "blinky");
    defer std.testing.allocator.free(verilog);
    try std.testing.expect(std.mem.indexOf(u8, verilog, "always @(posedge") != null);
    try std.testing.expect(std.mem.indexOf(u8, verilog, "led") != null);
}
