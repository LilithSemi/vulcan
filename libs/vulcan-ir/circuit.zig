//! Structural circuit category for pure Vulcan functions.
//!
//! Wires are first-class values. Reusing a WireId is fanout; a node is a pure
//! morphism from its input wires to one output wire. `append` performs a small
//! structural normalization pass, so equal pure morphisms share one node.

const std = @import("std");
const function = @import("function.zig");

pub const Error = error{
    NotCombinational,
    UnsupportedType,
    UnsupportedOperation,
    MissingReturn,
    InvalidInput,
    InvalidCircuit,
};

pub const Wire = enum(u32) { _ };
pub const NodeKind = enum { input, constant, add, sub, mul, bit_and, bit_or, bit_xor, eq, ne, lt, le, gt, ge, mux, register };

pub const Node = struct {
    kind: NodeKind,
    bits: u16,
    a: ?Wire = null,
    b: ?Wire = null,
    c: ?Wire = null,
    value: i64 = 0,
    input: u32 = 0,
};

pub const Port = struct { name: []const u8, wire: Wire, bits: u16 };
pub const Product = struct { left: Wire, right: Wire };

pub const Circuit = struct {
    allocator: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    net_bases: std.ArrayList(u32) = .empty,
    inputs: std.ArrayList(Port) = .empty,
    outputs: std.ArrayList(Port) = .empty,
    owned_names: std.ArrayList([]u8) = .empty,
    next_net: u32 = 2,

    pub fn init(allocator: std.mem.Allocator) Circuit {
        return .{ .allocator = allocator };
    }
    pub fn deinit(self: *Circuit) void {
        self.nodes.deinit(self.allocator);
        self.net_bases.deinit(self.allocator);
        self.inputs.deinit(self.allocator);
        self.outputs.deinit(self.allocator);
        for (self.owned_names.items) |name| self.allocator.free(name);
        self.owned_names.deinit(self.allocator);
    }

    fn ownedName(self: *Circuit, comptime fmt: []const u8, args: anytype) std.mem.Allocator.Error![]const u8 {
        const name = try std.fmt.allocPrint(self.allocator, fmt, args);
        try self.owned_names.append(self.allocator, name);
        return name;
    }

    pub fn node(self: *const Circuit, wire: Wire) Node {
        return self.nodes.items[@backingInt(wire)];
    }
    fn netBase(self: *const Circuit, wire: Wire) u32 {
        return self.net_bases.items[@backingInt(wire)];
    }

    /// Identity is represented by retaining the same wire. This is the
    /// categorical identity morphism; no redundant buffer node is introduced.
    pub fn identity(_: *const Circuit, wire: Wire) Wire {
        return wire;
    }

    /// Fanout is structural sharing. Both projections intentionally carry the
    /// same wire identity, which is exactly how a circuit graph represents a
    /// wire feeding multiple consumers.
    pub fn fanout(_: *const Circuit, wire: Wire) [2]Wire {
        return .{ wire, wire };
    }

    /// Product pairing is represented by a pair of independently typed wires.
    /// Downstream morphisms consume the pair's components without introducing a
    /// runtime aggregate or memory object.
    pub fn product(_: *const Circuit, left: Wire, right: Wire) Product {
        return .{ .left = left, .right = right };
    }

    pub fn appendInput(self: *Circuit, name: []const u8, bits_: u16) std.mem.Allocator.Error!Wire {
        const wire = try self.appendRaw(.{ .kind = .input, .bits = bits_, .input = @intCast(self.inputs.items.len) });
        try self.inputs.append(self.allocator, .{ .name = name, .wire = wire, .bits = bits_ });
        return wire;
    }

    pub fn appendOutput(self: *Circuit, name: []const u8, wire: Wire, bits_: u16) std.mem.Allocator.Error!void {
        try self.outputs.append(self.allocator, .{ .name = name, .wire = wire, .bits = bits_ });
    }

    /// Add a clocked state element. Registers are intentionally not passed through
    /// pure-node CSE: each call denotes a distinct piece of state.
    pub fn appendRegister(self: *Circuit, clock: Wire, reset: Wire, bits_: u16, reset_value: i64) std.mem.Allocator.Error!Wire {
        const register_index = self.registerCount();
        return self.appendRaw(.{ .kind = .register, .bits = bits_, .b = clock, .c = reset, .value = reset_value, .input = @intCast(register_index) });
    }

    pub fn setRegisterNext(self: *Circuit, state: Wire, next: Wire) void {
        self.nodes.items[@backingInt(state)].a = next;
    }

    pub fn registerCount(self: *const Circuit) usize {
        var count: usize = 0;
        for (self.nodes.items) |n| {
            if (n.kind == .register) count += 1;
        }
        return count;
    }

    /// Check graph references and sequential port widths before interpretation or emission.
    pub fn validate(self: *const Circuit) Error!void {
        for (self.nodes.items, 0..) |n, i| {
            const check = struct {
                fn wire(c: *const Circuit, maybe: ?Wire) Error!Node {
                    const w = maybe orelse return error.InvalidCircuit;
                    const index = @backingInt(w);
                    if (index >= c.nodes.items.len) return error.InvalidCircuit;
                    return c.nodes.items[index];
                }
            }.wire;
            switch (n.kind) {
                .input, .constant => {},
                .register => {
                    const next = try check(self, n.a);
                    const clock = try check(self, n.b);
                    const reset = try check(self, n.c);
                    if (next.bits != n.bits or clock.bits != 1 or reset.bits != 1) return error.InvalidCircuit;
                    if (n.input >= self.registerCount()) return error.InvalidCircuit;
                },
                .mux => {
                    const cond = try check(self, n.a);
                    const then_value = try check(self, n.b);
                    const else_value = try check(self, n.c);
                    if (cond.bits != 1 or then_value.bits != n.bits or else_value.bits != n.bits) return error.InvalidCircuit;
                },
                else => {
                    const left = try check(self, n.a);
                    const right = try check(self, n.b);
                    if (left.bits != right.bits) return error.InvalidCircuit;
                },
            }
            _ = i;
        }
        for (self.outputs.items) |p| {
            if (@backingInt(p.wire) >= self.nodes.items.len or self.node(p.wire).bits != p.bits) return error.InvalidCircuit;
        }
    }

    pub fn append(self: *Circuit, node_: Node) std.mem.Allocator.Error!Wire {
        var normalized = node_;
        if (isCommutative(node_.kind) and @backingInt(normalized.a.?) > @backingInt(normalized.b.?)) {
            const tmp = normalized.a.?;
            normalized.a = normalized.b;
            normalized.b = tmp;
        }
        for (self.nodes.items, 0..) |old, i| if (std.meta.eql(old, normalized)) return @fromBackingInt(@intCast(i));
        return self.appendRaw(normalized);
    }

    fn appendRaw(self: *Circuit, node_: Node) std.mem.Allocator.Error!Wire {
        const wire: Wire = @fromBackingInt(@intCast(self.nodes.items.len));
        try self.nodes.append(self.allocator, node_);
        try self.net_bases.append(self.allocator, self.nextNet());
        return wire;
    }

    fn nextNet(self: *Circuit) u32 {
        const base = self.next_net;
        self.next_net += self.nodes.items[self.nodes.items.len - 1].bits;
        return base;
    }

    /// Interpret the circuit category directly. This is useful both as a
    /// reference semantics for emitters and as a cheap combinational test oracle.
    pub fn simulate(self: *const Circuit, inputs: []const u64, outputs: []u64) (Error || std.mem.Allocator.Error)!void {
        if (inputs.len != self.inputs.items.len or outputs.len != self.outputs.items.len) return error.InvalidInput;
        try self.validate();
        const values = try self.allocator.alloc(u64, self.nodes.items.len);
        defer self.allocator.free(values);
        try self.evaluate(inputs, null, values);
        for (self.nodes.items) |n| if (n.kind == .register) return error.UnsupportedOperation;
        for (self.outputs.items, 0..) |p, i| outputs[i] = values[@backingInt(p.wire)];
    }

    /// Advance every register by one rising clock edge. `state` is indexed by
    /// register insertion order and is updated in place; reset is synchronous
    /// to this abstract edge even though emission uses an asynchronous reset.
    pub fn simulateStep(self: *const Circuit, inputs: []const u64, state: []u64, outputs: []u64) (Error || std.mem.Allocator.Error)!void {
        if (inputs.len != self.inputs.items.len or outputs.len != self.outputs.items.len or state.len != self.registerCount()) return error.InvalidInput;
        try self.validate();
        const values = try self.allocator.alloc(u64, self.nodes.items.len);
        defer self.allocator.free(values);
        try self.evaluate(inputs, state, values);
        for (self.nodes.items) |n| if (n.kind == .register) {
            const mask: u64 = if (n.bits >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(n.bits)) - 1;
            state[n.input] = if (values[@backingInt(n.c.?)] != 0) @as(u64, @bitCast(n.value)) & mask else values[@backingInt(n.a.?)] & mask;
        };
        try self.evaluate(inputs, state, values);
        for (self.outputs.items, 0..) |p, i| outputs[i] = values[@backingInt(p.wire)];
    }

    fn evaluate(self: *const Circuit, inputs: []const u64, state: ?[]const u64, values: []u64) Error!void {
        for (self.nodes.items, 0..) |n, i| {
            const mask: u64 = if (n.bits >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(n.bits)) - 1;
            values[i] = switch (n.kind) {
                .input => inputs[n.input] & mask,
                .constant => @as(u64, @bitCast(n.value)) & mask,
                .add => (values[@backingInt(n.a.?)] +% values[@backingInt(n.b.?)]) & mask,
                .sub => (values[@backingInt(n.a.?)] -% values[@backingInt(n.b.?)]) & mask,
                .mul => (values[@backingInt(n.a.?)] *% values[@backingInt(n.b.?)]) & mask,
                .bit_and => values[@backingInt(n.a.?)] & values[@backingInt(n.b.?)],
                .bit_or => values[@backingInt(n.a.?)] | values[@backingInt(n.b.?)],
                .bit_xor => values[@backingInt(n.a.?)] ^ values[@backingInt(n.b.?)],
                .eq => @intFromBool(values[@backingInt(n.a.?)] == values[@backingInt(n.b.?)]),
                .ne => @intFromBool(values[@backingInt(n.a.?)] != values[@backingInt(n.b.?)]),
                .lt => @intFromBool(values[@backingInt(n.a.?)] < values[@backingInt(n.b.?)]),
                .le => @intFromBool(values[@backingInt(n.a.?)] <= values[@backingInt(n.b.?)]),
                .gt => @intFromBool(values[@backingInt(n.a.?)] > values[@backingInt(n.b.?)]),
                .ge => @intFromBool(values[@backingInt(n.a.?)] >= values[@backingInt(n.b.?)]),
                .mux => if (values[@backingInt(n.a.?)] != 0) values[@backingInt(n.b.?)] else values[@backingInt(n.c.?)],
                .register => if (state) |s| s[n.input] & mask else return error.UnsupportedOperation,
            };
        }
    }

    pub fn emitVerilog(self: *const Circuit, allocator: std.mem.Allocator, name: []const u8) (Error || std.mem.Allocator.Error || std.Io.Writer.Error)![]u8 {
        try self.validate();
        var w = std.Io.Writer.Allocating.init(allocator);
        defer w.deinit();
        try w.writer.print("// Generated from Vulcan's structural circuit category.\nmodule {s}(\n", .{name});
        var first = true;
        for (self.inputs.items) |p| {
            if (!first) try w.writer.writeAll(",\n");
            first = false;
            try w.writer.print("  input ", .{});
            try writeWidth(&w.writer, p.bits);
            try w.writer.print(" {s}", .{p.name});
        }
        for (self.outputs.items) |p| {
            if (!first) try w.writer.writeAll(",\n");
            first = false;
            try w.writer.print("  output ", .{});
            try writeWidth(&w.writer, p.bits);
            try w.writer.print(" {s}", .{p.name});
        }
        try w.writer.writeAll("\n);\n");
        for (self.inputs.items) |p| {
            try w.writer.print("wire ", .{}); // Keep internal wire names stable for expressions.
            try w.writer.print("w{d}", .{@backingInt(p.wire)});
            try w.writer.writeAll(";");
        }
        if (self.inputs.items.len != 0) try w.writer.writeByte('\n');
        for (self.inputs.items) |p| try w.writer.print("assign w{d} = {s};\n", .{ @backingInt(p.wire), p.name });
        for (self.nodes.items, 0..) |n, i| if (n.kind != .input) {
            if (n.kind == .register) {
                try w.writer.print("reg ", .{});
                try writeWidth(&w.writer, n.bits);
                try w.writer.print(" w{d};\n", .{i});
                try w.writer.print("always @(posedge w{d} or posedge w{d}) begin\n  if (w{d}) w{d} <= ", .{ @backingInt(n.b.?), @backingInt(n.c.?), @backingInt(n.c.?), i });
                try w.writer.print("{d};\n  else w{d} <= w{d};\nend\n", .{ n.value, i, @backingInt(n.a.?) });
            } else {
                try w.writer.print("wire ", .{});
                try writeWidth(&w.writer, n.bits);
                try w.writer.print(" w{d}; assign w{d} = ", .{ i, i });
                try writeExpr(&w.writer, n);
                try w.writer.writeAll(";\n");
            }
        };
        for (self.outputs.items) |p| try w.writer.print("assign {s} = w{d};\n", .{ p.name, @backingInt(p.wire) });
        try w.writer.writeAll("endmodule\n");
        return try w.toOwnedSlice();
    }

    pub fn emitJson(self: *const Circuit, allocator: std.mem.Allocator, name: []const u8) (Error || std.mem.Allocator.Error || std.Io.Writer.Error)![]u8 {
        try self.validate();
        var w = std.Io.Writer.Allocating.init(allocator);
        defer w.deinit();
        try w.writer.print("{{\"creator\":\"Vulcan circuit category\",\"modules\":{{\"{s}\":{{\"attributes\":{{}},\"ports\":{{", .{name});
        var first = true;
        for (self.inputs.items) |p| {
            if (!first) try w.writer.writeByte(',');
            first = false;
            try writePortJson(&w.writer, self, p, "input");
        }
        for (self.outputs.items) |p| {
            if (!first) try w.writer.writeByte(',');
            first = false;
            try writePortJson(&w.writer, self, p, "output");
        }
        try w.writer.writeAll("},\"cells\":{");
        var cell_first = true;
        for (self.nodes.items, 0..) |n, i| if (n.kind != .input and n.kind != .constant) {
            if (!cell_first) try w.writer.writeByte(',');
            cell_first = false;
            try w.writer.print("\"w{d}\":{{\"hide_name\":1,\"type\":\"{s}\",\"parameters\":", .{ i, cellType(n.kind) });
            if (n.kind == .register) {
                try w.writer.writeAll("{\"\\\\WIDTH\":\"");
                try writeUnsignedBits(&w.writer, n.bits, 32);
                try w.writer.writeAll("\",\"\\\\CLK_POLARITY\":\"1\",\"\\\\ARST_POLARITY\":\"1\",\"\\\\ARST_VALUE\":\"");
                try writeValueBits(&w.writer, n.value, n.bits);
                try w.writer.writeAll("\"}");
            } else if (n.kind == .mux) {
                try w.writer.writeAll("{\"\\\\WIDTH\":\"");
                try writeUnsignedBits(&w.writer, n.bits, 32);
                try w.writer.writeAll("\"}");
            } else {
                try writeBinaryParameters(&w.writer, self, n);
            }
            try w.writer.writeAll(",\"attributes\":{},\"port_directions\":{");
            try writePortDirectionsJson(&w.writer, n.kind);
            try w.writer.writeAll("},\"connections\":{");
            try writeConnectionsJson(&w.writer, self, @fromBackingInt(@intCast(i)), n);
            try w.writer.writeAll("}}");
        };
        try w.writer.writeAll("},\"netnames\":{");
        first = true;
        for (self.inputs.items) |p| {
            if (!first) try w.writer.writeByte(',');
            first = false;
            try writeNetnameJson(&w.writer, self, p);
        }
        for (self.outputs.items) |p| {
            if (!first) try w.writer.writeByte(',');
            first = false;
            try writeNetnameJson(&w.writer, self, p);
        }
        try w.writer.writeAll("}}}}\n");
        return try w.toOwnedSlice();
    }
};

pub fn lower(allocator: std.mem.Allocator, func: *const function.Function) (Error || std.mem.Allocator.Error)!Circuit {
    if (func.blockCount() != 1) return error.NotCombinational;
    const block: function.Block = @fromBackingInt(@intCast(0));
    const term = func.terminator(block) orelse return error.MissingReturn;
    const ret = switch (term) {
        .ret => |r| r,
        .jump => return error.NotCombinational,
    };
    var c = Circuit.init(allocator);
    errdefer c.deinit();
    var values = std.AutoHashMap(function.Value, Wire).init(allocator);
    defer values.deinit();
    for (func.blockParams(block), 0..) |v, i| {
        const wire = try c.appendInput(try c.ownedName("in{d}", .{i}), try bits(func, v));
        try values.put(v, wire);
    }
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse return error.UnsupportedOperation;
        const node = try lowerNode(func, &values, inst, result);
        try values.put(result, try c.append(node));
    }
    for (ret.slice(), 0..) |v, i| {
        const wire = values.get(v) orelse return error.UnsupportedOperation;
        try c.outputs.append(allocator, .{ .name = try c.ownedName("out{d}", .{i}), .wire = wire, .bits = try bits(func, v) });
    }
    return c;
}

fn lowerNode(func: *const function.Function, values: *const std.AutoHashMap(function.Value, Wire), inst: function.Inst, result: function.Value) (Error)!Node {
    const op = func.opcode(inst);
    const out_bits = try bits(func, result);
    const get = struct {
        fn f(m: *const std.AutoHashMap(function.Value, Wire), v: function.Value) Error!Wire {
            return m.get(v) orelse error.UnsupportedOperation;
        }
    }.f;
    return switch (op) {
        .iconst => |value| .{ .kind = .constant, .bits = out_bits, .value = value },
        .arith => |a| .{ .kind = switch (a.op) {
            .add => .add,
            .sub => .sub,
            .mul => .mul,
            .bit_and => .bit_and,
            .bit_or => .bit_or,
            .bit_xor => .bit_xor,
            else => return error.UnsupportedOperation,
        }, .bits = out_bits, .a = try get(values, a.lhs), .b = try get(values, a.rhs) },
        .icmp => |a| .{ .kind = switch (a.op) {
            .eq => .eq,
            .ne => .ne,
            .lt => .lt,
            .le => .le,
            .gt => .gt,
            .ge => .ge,
        }, .bits = 1, .a = try get(values, a.lhs), .b = try get(values, a.rhs) },
        .select => |s| .{ .kind = .mux, .bits = out_bits, .a = try get(values, s.cond), .b = try get(values, s.then), .c = try get(values, s.@"else") },
        else => return error.UnsupportedOperation,
    };
}

fn bits(func: *const function.Function, value: function.Value) Error!u16 {
    return switch (func.types.type_kind(func.valueType(value))) {
        .bool => 1,
        .int => |i| i.bits,
        else => error.UnsupportedType,
    };
}

fn isCommutative(kind: NodeKind) bool {
    return switch (kind) {
        .add, .mul, .bit_and, .bit_or, .bit_xor, .eq, .ne => true,
        else => false,
    };
}
fn cellType(kind: NodeKind) []const u8 {
    return switch (kind) {
        .constant => "$const",
        .add => "$add",
        .sub => "$sub",
        .mul => "$mul",
        .bit_and => "$and",
        .bit_or => "$or",
        .bit_xor => "$xor",
        .eq => "$eq",
        .ne => "$ne",
        .lt => "$lt",
        .le => "$le",
        .gt => "$gt",
        .ge => "$ge",
        .mux => "$mux",
        .register => "$adff",
        .input => unreachable,
    };
}
fn writeWidth(w: *std.Io.Writer, bits_: u16) std.Io.Writer.Error!void {
    if (bits_ != 1) try w.print("[{d}:0]", .{bits_ - 1});
}
fn writeExpr(w: *std.Io.Writer, n: Node) std.Io.Writer.Error!void {
    if (n.kind == .constant) return w.print("{d}", .{n.value});
    if (n.kind == .mux) return w.print("w{d} ? w{d} : w{d}", .{ @backingInt(n.a.?), @backingInt(n.b.?), @backingInt(n.c.?) });
    const symbol = switch (n.kind) {
        .add => "+",
        .sub => "-",
        .mul => "*",
        .bit_and => "&",
        .bit_or => "|",
        .bit_xor => "^",
        .eq => "==",
        .ne => "!=",
        .lt => "<",
        .le => "<=",
        .gt => ">",
        .ge => ">=",
        else => unreachable,
    };
    return w.print("w{d} {s} w{d}", .{ @backingInt(n.a.?), symbol, @backingInt(n.b.?) });
}
fn writePortJson(w: *std.Io.Writer, c: *const Circuit, p: Port, direction: []const u8) std.Io.Writer.Error!void {
    try w.print("\"{s}\":{{\"direction\":\"{s}\",\"bits\":[", .{ p.name, direction });
    try writeBitsJson(w, c.netBase(p.wire), p.bits);
    try w.writeAll("]}");
}
fn writeNetnameJson(w: *std.Io.Writer, c: *const Circuit, p: Port) std.Io.Writer.Error!void {
    try w.print("\"{s}\":{{\"hide_name\":0,\"bits\":[", .{p.name});
    try writeBitsJson(w, c.netBase(p.wire), p.bits);
    try w.writeAll("]}");
}
fn writeBitsJson(w: *std.Io.Writer, base: u32, count: u16) std.Io.Writer.Error!void {
    for (0..count) |i| {
        if (i != 0) try w.writeByte(',');
        try w.print("{d}", .{base + @as(u32, @intCast(i))});
    }
}

fn writeWireBitsJson(w: *std.Io.Writer, c: *const Circuit, wire: Wire) std.Io.Writer.Error!void {
    const n = c.node(wire);
    if (n.kind != .constant) return writeBitsJson(w, c.netBase(wire), n.bits);
    const raw: u64 = @bitCast(n.value);
    for (0..n.bits) |i| {
        if (i != 0) try w.writeByte(',');
        const set = i < 64 and ((raw >> @intCast(i)) & 1) != 0;
        try w.print("{d}", .{@intFromBool(set)});
    }
}

fn writeValueBits(w: *std.Io.Writer, value: i64, bits_: u16) std.Io.Writer.Error!void {
    const raw: u64 = @bitCast(value);
    var remaining = bits_;
    while (remaining != 0) : (remaining -= 1) {
        const bit = remaining - 1;
        const set = bit < 64 and ((raw >> @intCast(bit)) & 1) != 0;
        try w.writeByte(if (set) '1' else '0');
    }
}

fn writeUnsignedBits(w: *std.Io.Writer, value: u16, bits_: u16) std.Io.Writer.Error!void {
    var remaining = bits_;
    while (remaining != 0) : (remaining -= 1) {
        const bit = remaining - 1;
        try w.writeByte(if (bit < 16 and ((value >> @intCast(bit)) & 1) != 0) '1' else '0');
    }
}

fn writeBinaryParameters(w: *std.Io.Writer, c: *const Circuit, n: Node) std.Io.Writer.Error!void {
    const left_bits = c.node(n.a.?).bits;
    const right_bits = c.node(n.b.?).bits;
    const result_bits: u16 = if (n.kind == .eq or n.kind == .ne or n.kind == .lt or n.kind == .le or n.kind == .gt or n.kind == .ge) 1 else n.bits;
    try w.writeAll("{\"\\\\A_SIGNED\":\"");
    try writeUnsignedBits(w, 0, 32);
    try w.writeAll("\",\"\\\\A_WIDTH\":\"");
    try writeUnsignedBits(w, left_bits, 32);
    try w.writeAll("\",\"\\\\B_SIGNED\":\"");
    try writeUnsignedBits(w, 0, 32);
    try w.writeAll("\",\"\\\\B_WIDTH\":\"");
    try writeUnsignedBits(w, right_bits, 32);
    try w.writeAll("\",\"\\\\Y_WIDTH\":\"");
    try writeUnsignedBits(w, result_bits, 32);
    try w.writeAll("\"}");
}
fn writeConnectionsJson(w: *std.Io.Writer, c: *const Circuit, result: Wire, n: Node) std.Io.Writer.Error!void {
    if (n.kind == .constant) {
        try w.print("\"Y\":[", .{});
        try writeBitsJson(w, c.netBase(result), n.bits);
        try w.writeByte(']');
        return;
    }
    if (n.kind == .register) {
        try w.print("\"D\":[", .{});
        try writeWireBitsJson(w, c, n.a.?);
        try w.print("],\"Q\":[", .{});
        try writeBitsJson(w, c.netBase(result), n.bits);
        try w.print("],\"CLK\":[", .{});
        try writeWireBitsJson(w, c, n.b.?);
        try w.print("],\"ARST\":[", .{});
        try writeWireBitsJson(w, c, n.c.?);
        try w.writeByte(']');
        return;
    }
    if (n.kind == .mux) {
        try w.print("\"A\":[", .{});
        try writeWireBitsJson(w, c, n.c.?);
        try w.print("],\"B\":[", .{});
        try writeWireBitsJson(w, c, n.b.?);
        try w.print("],\"S\":[", .{});
        try writeWireBitsJson(w, c, n.a.?);
        try w.print("],\"Y\":[", .{});
        try writeBitsJson(w, c.netBase(result), n.bits);
        try w.writeByte(']');
        return;
    }
    try w.print("\"A\":[", .{});
    try writeWireBitsJson(w, c, n.a.?);
    try w.print("],\"B\":[", .{});
    try writeWireBitsJson(w, c, n.b.?);
    try w.print("],\"Y\":[", .{});
    try writeBitsJson(w, c.netBase(result), n.bits);
    try w.writeByte(']');
}

fn writePortDirectionsJson(w: *std.Io.Writer, kind: NodeKind) std.Io.Writer.Error!void {
    switch (kind) {
        .constant => try w.writeAll("\"Y\":\"output\""),
        .register => try w.writeAll("\"ARST\":\"input\",\"CLK\":\"input\",\"D\":\"input\",\"Q\":\"output\""),
        .mux => try w.writeAll("\"A\":\"input\",\"B\":\"input\",\"S\":\"input\",\"Y\":\"output\""),
        else => try w.writeAll("\"A\":\"input\",\"B\":\"input\",\"Y\":\"output\""),
    }
}

test "categorical circuit shares structurally equal pure nodes" {
    var c = Circuit.init(std.testing.allocator);
    defer c.deinit();
    const a = try c.appendInput("a", 1);
    const b = try c.appendInput("b", 1);
    try std.testing.expectEqual(a, c.identity(a));
    const copies = c.fanout(a);
    try std.testing.expectEqual(copies[0], copies[1]);
    const pair = c.product(a, b);
    try std.testing.expectEqual(a, pair.left);
    try std.testing.expectEqual(b, pair.right);
    const x = try c.append(.{ .kind = .add, .bits = 1, .a = a, .b = b });
    const y = try c.append(.{ .kind = .add, .bits = 1, .a = b, .b = a });
    try std.testing.expectEqual(x, y);
}

test "circuit interpretation evaluates a mux" {
    var c = Circuit.init(std.testing.allocator);
    defer c.deinit();
    const s = try c.appendInput("s", 1);
    const a = try c.appendInput("a", 8);
    const b = try c.appendInput("b", 8);
    const y = try c.append(.{ .kind = .mux, .bits = 8, .a = s, .b = a, .c = b });
    try c.outputs.append(std.testing.allocator, .{ .name = "y", .wire = y, .bits = 8 });
    var outputs = [_]u64{0};
    try c.simulate(&.{ 1, 0xaa, 0x55 }, &outputs);
    try std.testing.expectEqual(@as(u64, 0xaa), outputs[0]);
}

test "sequential interpretation advances and resets registers" {
    var c = Circuit.init(std.testing.allocator);
    defer c.deinit();
    const clock = try c.appendInput("clk", 1);
    const reset = try c.appendInput("reset", 1);
    const state_wire = try c.appendRegister(clock, reset, 2, 0);
    const one = try c.append(.{ .kind = .constant, .bits = 2, .value = 1 });
    const next = try c.append(.{ .kind = .add, .bits = 2, .a = state_wire, .b = one });
    c.setRegisterNext(state_wire, next);
    try c.appendOutput("q", state_wire, 2);
    var state = [_]u64{99};
    var outputs = [_]u64{99};
    try c.simulateStep(&.{ 1, 1 }, &state, &outputs);
    try std.testing.expectEqual(@as(u64, 0), outputs[0]);
    try c.simulateStep(&.{ 1, 0 }, &state, &outputs);
    try std.testing.expectEqual(@as(u64, 1), outputs[0]);
    try c.simulateStep(&.{ 1, 0 }, &state, &outputs);
    try std.testing.expectEqual(@as(u64, 2), outputs[0]);
}
