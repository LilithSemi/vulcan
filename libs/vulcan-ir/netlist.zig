//! Format emitters for Vulcan's structural circuit category.
//!
//! Lowering is intentionally owned by `circuit.zig`; this module only selects
//! a serialization format. That keeps Verilog, Yosys JSON, simulation, and
//! future graph emitters as interpretations of one circuit representation.

const std = @import("std");
const function = @import("function.zig");
const circuit = @import("circuit.zig");

pub const Error = circuit.Error;

pub fn emitVerilog(allocator: std.mem.Allocator, func: *const function.Function, name: []const u8) ![]u8 {
    var c = try circuit.lower(allocator, func);
    defer c.deinit();
    return c.emitVerilog(allocator, name);
}

pub fn emitJson(allocator: std.mem.Allocator, func: *const function.Function, name: []const u8) ![]u8 {
    var c = try circuit.lower(allocator, func);
    defer c.deinit();
    return c.emitJson(allocator, name);
}

test "format emitters interpret the same circuit" {
    var f = function.Function.init(std.testing.allocator);
    defer f.deinit();
    const bool_ty = try f.types.intern(.bool);
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, bool_ty);
    f.setTerminator(b, .{ .ret = function.Ret.one(a) });
    const verilog = try emitVerilog(std.testing.allocator, &f, "top");
    defer std.testing.allocator.free(verilog);
    const json = try emitJson(std.testing.allocator, &f, "top");
    defer std.testing.allocator.free(json);
    try std.testing.expect(std.mem.indexOf(u8, verilog, "module top") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"modules\"") != null);
}
