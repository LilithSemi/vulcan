//! Small textual structural-circuit format used by vulcan-irc for sequential designs.
//!
//! Example:
//!   circuit blinky
//!   input clk 1
//!   input reset 1
//!   reg q 4 clk reset 0
//!   const one 4 1
//!   add next q one
//!   next q next
//!   output led q
//!   end

const std = @import("std");
const circuit_mod = @import("circuit.zig");

pub const Error = circuit_mod.Error || error{ ParseError, UnknownWire, InvalidDirective };
pub const Parsed = struct { circuit: circuit_mod.Circuit, name: []const u8 };

pub fn parse(allocator: std.mem.Allocator, source: []const u8) (Error || std.mem.Allocator.Error)!Parsed {
    var c = circuit_mod.Circuit.init(allocator);
    errdefer c.deinit();
    var wires = std.StringHashMap(circuit_mod.Wire).init(allocator);
    defer wires.deinit();
    var name: ?[]const u8 = null;
    var saw_end = false;
    var lines = std.mem.splitScalar(u8, source, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        var words = std.mem.tokenizeAny(u8, line, " \t");
        const op = words.next() orelse continue;
        if (std.mem.eql(u8, op, "circuit")) {
            if (name != null) return error.InvalidDirective;
            name = words.next() orelse return error.ParseError;
        } else if (std.mem.eql(u8, op, "input")) {
            const n = words.next() orelse return error.ParseError;
            const bits = try parseBits(words.next() orelse return error.ParseError);
            try put(&wires, n, try c.appendInput(n, bits));
        } else if (std.mem.eql(u8, op, "const")) {
            const n = words.next() orelse return error.ParseError;
            const bits = try parseBits(words.next() orelse return error.ParseError);
            const value = try parseValue(words.next() orelse return error.ParseError);
            try put(&wires, n, try c.append(.{ .kind = .constant, .bits = bits, .value = value }));
        } else if (std.mem.eql(u8, op, "reg")) {
            const n = words.next() orelse return error.ParseError;
            const bits = try parseBits(words.next() orelse return error.ParseError);
            const clock = try get(&wires, words.next() orelse return error.ParseError);
            const reset = try get(&wires, words.next() orelse return error.ParseError);
            const value = try parseValue(words.next() orelse return error.ParseError);
            try put(&wires, n, try c.appendRegister(clock, reset, bits, value));
        } else if (std.mem.eql(u8, op, "next")) {
            const state = try get(&wires, words.next() orelse return error.ParseError);
            const next = try get(&wires, words.next() orelse return error.ParseError);
            c.setRegisterNext(state, next);
        } else if (std.mem.eql(u8, op, "output")) {
            const n = words.next() orelse return error.ParseError;
            const wire = try get(&wires, words.next() orelse return error.ParseError);
            try c.appendOutput(n, wire, c.node(wire).bits);
        } else if (std.mem.eql(u8, op, "end")) {
            saw_end = true;
            break;
        } else if (std.mem.eql(u8, op, "mux")) {
            try binaryOrTernary(&c, &wires, op, words.next(), words.next(), words.next(), words.next());
        } else if (isBinary(op)) {
            const n = words.next() orelse return error.ParseError;
            const a = try get(&wires, words.next() orelse return error.ParseError);
            const b = try get(&wires, words.next() orelse return error.ParseError);
            try put(&wires, n, try c.append(.{ .kind = kind(op), .bits = if (isCompare(op)) 1 else c.node(a).bits, .a = a, .b = b }));
        } else return error.InvalidDirective;
    }
    if (name == null or !saw_end) return error.ParseError;
    try c.validate();
    return .{ .circuit = c, .name = name.? };
}

fn parseBits(word: []const u8) !u16 {
    return std.fmt.parseInt(u16, word, 10) catch error.ParseError;
}
fn parseValue(word: []const u8) !i64 {
    return std.fmt.parseInt(i64, word, 0) catch error.ParseError;
}
fn put(map: *std.StringHashMap(circuit_mod.Wire), name: []const u8, wire: circuit_mod.Wire) !void {
    try map.put(name, wire);
}
fn get(map: *const std.StringHashMap(circuit_mod.Wire), name: []const u8) !circuit_mod.Wire {
    return map.get(name) orelse error.UnknownWire;
}
fn isCompare(op: []const u8) bool {
    return std.mem.eql(u8, op, "eq") or std.mem.eql(u8, op, "ne") or std.mem.eql(u8, op, "lt") or std.mem.eql(u8, op, "le") or std.mem.eql(u8, op, "gt") or std.mem.eql(u8, op, "ge");
}
fn isBinary(op: []const u8) bool {
    return std.mem.eql(u8, op, "add") or std.mem.eql(u8, op, "sub") or
        std.mem.eql(u8, op, "mul") or std.mem.eql(u8, op, "and") or
        std.mem.eql(u8, op, "or") or std.mem.eql(u8, op, "xor") or
        isCompare(op);
}
fn kind(op: []const u8) circuit_mod.NodeKind {
    const names = [_][]const u8{ "add", "sub", "mul", "and", "or", "xor", "eq", "ne", "lt", "le", "gt", "ge" };
    const kinds = [_]circuit_mod.NodeKind{ .add, .sub, .mul, .bit_and, .bit_or, .bit_xor, .eq, .ne, .lt, .le, .gt, .ge };
    for (names, 0..) |n, i| if (std.mem.eql(u8, op, n)) return kinds[i];
    unreachable;
}
fn binaryOrTernary(c: *circuit_mod.Circuit, wires: *std.StringHashMap(circuit_mod.Wire), _: []const u8, n_: ?[]const u8, a_: ?[]const u8, b_: ?[]const u8, c_: ?[]const u8) !void {
    const n = n_ orelse return error.ParseError;
    const a = try get(wires, a_ orelse return error.ParseError);
    const b = try get(wires, b_ orelse return error.ParseError);
    const d = try get(wires, c_ orelse return error.ParseError);
    try put(wires, n, try c.append(.{ .kind = .mux, .bits = c.node(b).bits, .a = a, .b = b, .c = d }));
}

test "parse sequential circuit text" {
    const text = "circuit demo\ninput clk 1\ninput reset 1\nreg q 2 clk reset 0\nconst one 2 1\nadd next q one\nnext q next\noutput q q\nend\n";
    var parsed = try parse(std.testing.allocator, text);
    defer parsed.circuit.deinit();
    try std.testing.expectEqualStrings("demo", parsed.name);
    var state = [_]u64{0};
    var output = [_]u64{0};
    try parsed.circuit.simulateStep(&.{ 1, 0 }, &state, &output);
    try std.testing.expectEqual(@as(u64, 1), output[0]);
}

test "sequential registers receive independent state slots" {
    const text = "circuit demo\ninput clk 1\ninput reset 1\nreg a 1 clk reset 0\nreg b 1 clk reset 1\nconst zero 1 0\nnext a zero\nnext b zero\noutput a a\noutput b b\nend\n";
    var parsed = try parse(std.testing.allocator, text);
    defer parsed.circuit.deinit();
    var state = [_]u64{ 0, 1 };
    var output = [_]u64{ 0, 0 };
    try parsed.circuit.simulateStep(&.{ 1, 0 }, &state, &output);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0 }, &output);
}
