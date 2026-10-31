//! Bounded equality saturation for pure integer expressions. The pass imports
//! block-local integer trees, adds equivalent factored, cancelled, and absorbed
//! forms, and extracts the expression with the fewest arithmetic instructions.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const BinOp = ir.function.BinOp;
const Block = ir.function.Block;
const Function = ir.function.Function;
const Type = ir.types.Type;
const Value = ir.function.Value;

pub const pass_def = pass.Pass{ .name = "egraph", .run = run, .preserves_cfg = true };

pub const Limits = struct {
    max_nodes: usize = 512,
    max_iters: usize = 6,
};

const OpNode = struct {
    op: BinOp,
    lhs: u32,
    rhs: u32,
    ty: Type,
    existing: ?Value,
};

const Node = union(enum) {
    leaf: Value,
    op: OpNode,
};

const Entry = struct {
    class: u32,
    node: Node,
};

const OpKey = struct {
    op: BinOp,
    lhs: u32,
    rhs: u32,
    ty: Type,
};

const Graph = struct {
    allocator: std.mem.Allocator,
    limits: Limits,
    parents: std.ArrayList(u32) = .empty,
    nodes: std.ArrayList(Entry) = .empty,
    leaves: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    ops: std.AutoHashMapUnmanaged(OpKey, usize) = .empty,
    ops_valid: bool = true,
    hit_limit: bool = false,

    fn deinit(self: *Graph) void {
        self.parents.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.leaves.deinit(self.allocator);
        self.ops.deinit(self.allocator);
    }

    fn find(self: *Graph, class: u32) u32 {
        var root = class;
        while (self.parents.items[root] != root) root = self.parents.items[root];
        var current = class;
        while (self.parents.items[current] != current) {
            const next = self.parents.items[current];
            self.parents.items[current] = root;
            current = next;
        }
        return root;
    }

    fn merge(self: *Graph, a: u32, b: u32) bool {
        const ar = self.find(a);
        const br = self.find(b);
        if (ar == br) return false;
        self.parents.items[br] = ar;
        self.ops_valid = false;
        return true;
    }

    fn addClass(self: *Graph) !u32 {
        const class: u32 = @intCast(self.parents.items.len);
        try self.parents.append(self.allocator, class);
        return class;
    }

    fn addLeaf(self: *Graph, value: Value) !?u32 {
        if (self.leaves.get(value)) |class| return self.find(class);
        if (self.nodes.items.len >= self.limits.max_nodes) {
            self.hit_limit = true;
            return null;
        }
        const class = try self.addClass();
        try self.nodes.append(self.allocator, .{ .class = class, .node = .{ .leaf = value } });
        try self.leaves.put(self.allocator, value, class);
        return class;
    }

    fn addOp(self: *Graph, raw: OpNode) !?u32 {
        if (!self.ops_valid) _ = try self.rebuild();
        var node = raw;
        node.lhs = self.find(node.lhs);
        node.rhs = self.find(node.rhs);
        if (isCommutative(node.op) and node.lhs > node.rhs) std.mem.swap(u32, &node.lhs, &node.rhs);
        const key = opKey(node);
        if (self.ops.get(key)) |index| {
            const entry = &self.nodes.items[index];
            if (entry.node.op.existing == null) entry.node.op.existing = node.existing;
            return self.find(entry.class);
        }
        if (self.nodes.items.len >= self.limits.max_nodes) {
            self.hit_limit = true;
            return null;
        }
        const class = try self.addClass();
        try self.nodes.append(self.allocator, .{ .class = class, .node = .{ .op = node } });
        try self.ops.put(self.allocator, key, self.nodes.items.len - 1);
        return class;
    }

    fn rebuild(self: *Graph) !bool {
        var changed = false;
        while (true) {
            var merged = false;
            self.ops.clearRetainingCapacity();
            for (self.nodes.items, 0..) |entry, index| switch (entry.node) {
                .leaf => {},
                .op => |raw| {
                    const key = opKey(canonical(self, raw));
                    const result = try self.ops.getOrPut(self.allocator, key);
                    if (result.found_existing) {
                        const prior = self.nodes.items[result.value_ptr.*];
                        merged = self.merge(prior.class, entry.class) or merged;
                    } else {
                        result.value_ptr.* = index;
                    }
                },
            };
            changed = changed or merged;
            if (!merged) {
                self.ops_valid = true;
                return changed;
            }
        }
    }
};

fn opKey(op: OpNode) OpKey {
    return .{
        .op = op.op,
        .lhs = op.lhs,
        .rhs = op.rhs,
        .ty = op.ty,
    };
}

fn canonical(graph: *Graph, op: OpNode) OpNode {
    var result = op;
    result.lhs = graph.find(result.lhs);
    result.rhs = graph.find(result.rhs);
    if (isCommutative(result.op) and result.lhs > result.rhs) std.mem.swap(u32, &result.lhs, &result.rhs);
    return result;
}

fn isCommutative(op: BinOp) bool {
    return switch (op) {
        .add, .mul, .bit_and, .bit_or, .bit_xor => true,
        else => false,
    };
}

/// The operation that distributes over `outer`, if the factoring rule set has
/// an exact integer identity for the pair.
fn factorInner(outer: BinOp) ?BinOp {
    return switch (outer) {
        .add, .sub => .mul,
        .bit_or, .bit_xor => .bit_and,
        .bit_and => .bit_or,
        else => null,
    };
}

fn cancellationInner(outer: BinOp) ?BinOp {
    return switch (outer) {
        .sub => .add,
        .bit_xor => .bit_xor,
        else => null,
    };
}

fn absorptionInner(outer: BinOp) ?BinOp {
    return switch (outer) {
        .bit_and => .bit_or,
        .bit_or => .bit_and,
        else => null,
    };
}

fn isInteger(func: *const Function, value: Value) bool {
    return switch (func.types.type_kind(func.valueType(value))) {
        .int, .bool => true,
        else => false,
    };
}

/// Avoid graph construction for blocks that cannot match the current rule set.
fn hasFactorCandidate(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        const sum = switch (func.opcode(inst)) {
            .arith => |arith| if (factorInner(arith.op) != null or cancellationInner(arith.op) != null or absorptionInner(arith.op) != null) arith else continue,
            else => continue,
        };
        const lhs = if (func.definingInst(sum.lhs)) |lhs_inst| switch (func.opcode(lhs_inst)) {
            .arith => |arith| arith,
            else => null,
        } else null;
        const rhs = if (func.definingInst(sum.rhs)) |rhs_inst| switch (func.opcode(rhs_inst)) {
            .arith => |arith| arith,
            else => null,
        } else null;
        if (lhs != null and rhs != null) {
            if (factorInner(sum.op)) |inner| if (lhs.?.op == inner and rhs.?.op == inner) return true;
            if (cancellationInner(sum.op)) |inner| if (lhs.?.op == inner and rhs.?.op == inner) return true;
        }
        if (absorptionInner(sum.op)) |inner| {
            if (lhs) |op| if (op.op == inner and (op.lhs == sum.rhs or op.rhs == sum.rhs)) return true;
            if (rhs) |op| if (op.op == inner and (op.lhs == sum.lhs or op.rhs == sum.lhs)) return true;
        }
    }
    return false;
}

fn classForValue(graph: *Graph, classes: []?u32, costs: []usize, value: Value) !?u32 {
    const index = @backingInt(value);
    if (classes[index]) |class| return class;
    const class = try graph.addLeaf(value) orelse return null;
    classes[index] = class;
    costs[index] = 0;
    return class;
}

fn importBlock(graph: *Graph, func: *const Function, block: Block, classes: []?u32, costs: []usize) !void {
    for (func.blockParams(block)) |param| _ = try classForValue(graph, classes, costs, param);
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse continue;
        if (!isInteger(func, result)) {
            _ = try classForValue(graph, classes, costs, result);
            continue;
        }
        const arith = switch (func.opcode(inst)) {
            .arith => |arith| arith,
            else => {
                _ = try classForValue(graph, classes, costs, result);
                continue;
            },
        };
        if (arith.op != .add and arith.op != .sub and arith.op != .mul and arith.op != .bit_and and arith.op != .bit_or and arith.op != .bit_xor) {
            _ = try classForValue(graph, classes, costs, result);
            continue;
        }
        const lhs = try classForValue(graph, classes, costs, arith.lhs) orelse return;
        const rhs = try classForValue(graph, classes, costs, arith.rhs) orelse return;
        const class = try graph.addOp(.{
            .op = arith.op,
            .lhs = lhs,
            .rhs = rhs,
            .ty = func.valueType(result),
            .existing = result,
        }) orelse return;
        classes[@backingInt(result)] = class;
        costs[@backingInt(result)] = 1 + costs[@backingInt(arith.lhs)] + costs[@backingInt(arith.rhs)];
    }
}

fn matchingOps(graph: *Graph, class: u32, op: BinOp, ty: Type, output: *std.ArrayList(OpNode)) !void {
    const root = graph.find(class);
    for (graph.nodes.items) |entry| {
        if (graph.find(entry.class) != root) continue;
        switch (entry.node) {
            .leaf => {},
            .op => |candidate| if (candidate.op == op and candidate.ty == ty) {
                try output.append(graph.allocator, canonical(graph, candidate));
            },
        }
    }
}

const CommonOperands = struct { shared: u32, lhs_other: u32, rhs_other: u32 };

fn commonOperands(left: OpNode, right: OpNode) ?CommonOperands {
    var common: ?u32 = null;
    var lhs_other: u32 = undefined;
    var rhs_other: u32 = undefined;
    if (left.lhs == right.lhs) {
        common = left.lhs;
        lhs_other = left.rhs;
        rhs_other = right.rhs;
    } else if (left.lhs == right.rhs) {
        common = left.lhs;
        lhs_other = left.rhs;
        rhs_other = right.lhs;
    } else if (left.rhs == right.lhs) {
        common = left.rhs;
        lhs_other = left.lhs;
        rhs_other = right.rhs;
    } else if (left.rhs == right.rhs) {
        common = left.rhs;
        lhs_other = left.lhs;
        rhs_other = right.lhs;
    }
    return .{ .shared = common orelse return null, .lhs_other = lhs_other, .rhs_other = rhs_other };
}

fn factorPair(graph: *Graph, sum_class: u32, sum_op: BinOp, inner_op: BinOp, left: OpNode, right: OpNode) !bool {
    const operands = commonOperands(left, right) orelse return false;
    const inner = try graph.addOp(.{ .op = sum_op, .lhs = operands.lhs_other, .rhs = operands.rhs_other, .ty = left.ty, .existing = null }) orelse return false;
    const product = try graph.addOp(.{ .op = inner_op, .lhs = operands.shared, .rhs = inner, .ty = left.ty, .existing = null }) orelse return false;
    return graph.merge(sum_class, product);
}

fn cancelPair(graph: *Graph, sum_class: u32, sum_op: BinOp, left: OpNode, right: OpNode) !bool {
    const operands = commonOperands(left, right) orelse return false;
    const reduced = try graph.addOp(.{ .op = sum_op, .lhs = operands.lhs_other, .rhs = operands.rhs_other, .ty = left.ty, .existing = null }) orelse return false;
    return graph.merge(sum_class, reduced);
}

fn absorbPair(graph: *Graph, outer_class: u32, shared_class: u32, inner: OpNode) bool {
    const shared = graph.find(shared_class);
    if (graph.find(inner.lhs) != shared and graph.find(inner.rhs) != shared) return false;
    return graph.merge(outer_class, shared);
}

fn saturate(graph: *Graph) !void {
    var left: std.ArrayList(OpNode) = .empty;
    defer left.deinit(graph.allocator);
    var right: std.ArrayList(OpNode) = .empty;
    defer right.deinit(graph.allocator);

    var iter: usize = 0;
    while (iter < graph.limits.max_iters and !graph.hit_limit) : (iter += 1) {
        var changed = try graph.rebuild();
        const count = graph.nodes.items.len;
        for (graph.nodes.items[0..count]) |entry| {
            const sum = switch (entry.node) {
                .op => |op| if (factorInner(op.op) != null or cancellationInner(op.op) != null or absorptionInner(op.op) != null) canonical(graph, op) else continue,
                .leaf => continue,
            };
            if (factorInner(sum.op)) |inner_op| {
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, inner_op, sum.ty, &left);
                try matchingOps(graph, sum.rhs, inner_op, sum.ty, &right);
                for (left.items) |a| for (right.items) |b| {
                    changed = try factorPair(graph, entry.class, sum.op, inner_op, a, b) or changed;
                    if (graph.hit_limit) break;
                };
            }
            if (cancellationInner(sum.op)) |inner_op| {
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, inner_op, sum.ty, &left);
                try matchingOps(graph, sum.rhs, inner_op, sum.ty, &right);
                for (left.items) |a| for (right.items) |b| {
                    changed = try cancelPair(graph, entry.class, sum.op, a, b) or changed;
                    if (graph.hit_limit) break;
                };
            }
            if (absorptionInner(sum.op)) |inner_op| {
                left.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, inner_op, sum.ty, &left);
                for (left.items) |inner| changed = absorbPair(graph, entry.class, sum.rhs, inner) or changed;
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.rhs, inner_op, sum.ty, &right);
                for (right.items) |inner| changed = absorbPair(graph, entry.class, sum.lhs, inner) or changed;
            }
            if (graph.hit_limit) break;
        }
        if (!changed) break;
    }
    _ = try graph.rebuild();
}

const Extraction = struct {
    costs: []usize,
    best: []?usize,

    fn deinit(self: *Extraction, allocator: std.mem.Allocator) void {
        allocator.free(self.costs);
        allocator.free(self.best);
    }
};

fn extract(allocator: std.mem.Allocator, graph: *Graph) !Extraction {
    const costs = try allocator.alloc(usize, graph.parents.items.len);
    errdefer allocator.free(costs);
    @memset(costs, std.math.maxInt(usize));
    const best = try allocator.alloc(?usize, graph.parents.items.len);
    errdefer allocator.free(best);
    @memset(best, null);

    var changed = true;
    while (changed) {
        changed = false;
        for (graph.nodes.items, 0..) |entry, index| {
            const root = graph.find(entry.class);
            const cost = switch (entry.node) {
                .leaf => 0,
                .op => |raw| blk: {
                    const op = canonical(graph, raw);
                    const lhs = costs[graph.find(op.lhs)];
                    const rhs = costs[graph.find(op.rhs)];
                    if (lhs == std.math.maxInt(usize) or rhs == std.math.maxInt(usize)) continue;
                    break :blk 1 + lhs + rhs;
                },
            };
            if (cost < costs[root]) {
                costs[root] = cost;
                best[root] = index;
                changed = true;
            }
        }
    }
    return .{ .costs = costs, .best = best };
}

fn materialize(graph: *Graph, extraction: *const Extraction, func: *Function, block: Block, class: u32, memo: []?Value) !Value {
    const root = graph.find(class);
    if (memo[root]) |value| return value;
    const index = extraction.best[root] orelse unreachable;
    const value = switch (graph.nodes.items[index].node) {
        .leaf => |leaf| leaf,
        .op => |raw| blk: {
            const op = canonical(graph, raw);
            const lhs = try materialize(graph, extraction, func, block, op.lhs, memo);
            const rhs = try materialize(graph, extraction, func, block, op.rhs, memo);
            if (op.existing) |existing| {
                const existing_op = func.opcode(func.definingInst(existing).?).arith;
                if (existing_op.lhs == lhs and existing_op.rhs == rhs) break :blk existing;
                if (isCommutative(op.op) and existing_op.lhs == rhs and existing_op.rhs == lhs) break :blk existing;
            }
            break :blk try func.appendInst(block, op.ty, .{ .arith = .{ .op = op.op, .lhs = lhs, .rhs = rhs } });
        },
    };
    memo[root] = value;
    return value;
}

fn optimizeRoot(graph: *Graph, extraction: *const Extraction, func: *Function, block: Block, classes: []const ?u32, original_costs: []const usize, root_value: Value, memo: []?Value) !?Value {
    const class = classes[@backingInt(root_value)] orelse return null;
    const root = graph.find(class);
    if (extraction.costs[root] >= original_costs[@backingInt(root_value)]) return null;
    return try materialize(graph, extraction, func, block, root, memo);
}

pub fn runWithLimits(allocator: std.mem.Allocator, func: *Function, limits: Limits) pass.Error!bool {
    if (limits.max_nodes == 0 or limits.max_iters == 0) return false;
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const term = func.terminator(block) orelse continue;
        if (!hasFactorCandidate(func, block)) continue;
        var graph = Graph{ .allocator = allocator, .limits = limits };
        defer graph.deinit();
        const classes = try allocator.alloc(?u32, func.valueCount());
        defer allocator.free(classes);
        @memset(classes, null);
        const original_costs = try allocator.alloc(usize, func.valueCount());
        defer allocator.free(original_costs);
        @memset(original_costs, 0);
        try importBlock(&graph, func, block, classes, original_costs);
        try saturate(&graph);
        var extraction = try extract(allocator, &graph);
        defer extraction.deinit(allocator);
        const memo = try allocator.alloc(?Value, graph.parents.items.len);
        defer allocator.free(memo);
        @memset(memo, null);

        switch (term) {
            .ret => |ret| for (ret.slice(), 0..) |value, index| {
                if (try optimizeRoot(&graph, &extraction, func, block, classes, original_costs, value, memo)) |replacement| {
                    func.terminatorPtr(block).*.?.ret.values[index] = replacement;
                    changed = true;
                }
            },
            .jump => |jump| for (func.valueListMut(jump.args)) |*value| {
                if (try optimizeRoot(&graph, &extraction, func, block, classes, original_costs, value.*, memo)) |replacement| {
                    value.* = replacement;
                    changed = true;
                }
            },
        }
    }
    return changed;
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    return runWithLimits(allocator, func, .{});
}

test "egraph hash-consing follows merged child classes" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);

    var graph = Graph{ .allocator = allocator, .limits = .{} };
    defer graph.deinit();
    const ca = (try graph.addLeaf(a)).?;
    const cb = (try graph.addLeaf(b)).?;
    const cc = (try graph.addLeaf(c)).?;
    const ab = (try graph.addOp(.{ .op = .mul, .lhs = ca, .rhs = cb, .ty = ty, .existing = null })).?;
    const ac = (try graph.addOp(.{ .op = .mul, .lhs = ca, .rhs = cc, .ty = ty, .existing = null })).?;
    try std.testing.expect(graph.merge(cb, cc));
    try std.testing.expect(try graph.rebuild());
    try std.testing.expectEqual(graph.find(ab), graph.find(ac));

    const node_count = graph.nodes.items.len;
    _ = try graph.addOp(.{ .op = .mul, .lhs = ca, .rhs = cb, .ty = ty, .existing = null });
    try std.testing.expectEqual(node_count, graph.nodes.items.len);
}

test "egraph factors a shared multiplier from a returned expression" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(sum) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const product = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.mul, product.op);
    const inner_value = if (product.lhs == a) product.rhs else product.lhs;
    const inner = func.opcode(func.definingInst(inner_value).?).arith;
    try std.testing.expectEqual(BinOp.add, inner.op);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "egraph factors a shared multiplier from a subtraction" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const product = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.mul, product.op);
    const inner_value = if (product.lhs == a) product.rhs else product.lhs;
    const inner = func.opcode(func.definingInst(inner_value).?).arith;
    try std.testing.expectEqual(BinOp.sub, inner.op);
    try std.testing.expectEqual(b, inner.lhs);
    try std.testing.expectEqual(c, inner.rhs);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "egraph factors bitwise and across bitwise or" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = c } });
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const outer = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.bit_and, outer.op);
    const inner_value = if (outer.lhs == a) outer.rhs else outer.lhs;
    const inner = func.opcode(func.definingInst(inner_value).?).arith;
    try std.testing.expectEqual(BinOp.bit_or, inner.op);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "egraph cancels a shared additive term" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = c } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.sub, reduced.op);
    try std.testing.expectEqual(b, reduced.lhs);
    try std.testing.expectEqual(c, reduced.rhs);
}

test "egraph cancels a shared xor term" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = c } });
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.bit_xor, reduced.op);
    try std.testing.expect((reduced.lhs == b and reduced.rhs == c) or (reduced.lhs == c and reduced.rhs == b));
}

test "egraph absorbs an or containing the other and operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } });
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = either, .rhs = a } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(both) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(a, func.terminator(block).?.ret.values[0]);
}

test "egraph absorbs an and containing the other or operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = both } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(either) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(a, func.terminator(block).?.ret.values[0]);
}

test "egraph obeys a hard node budget" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = a } });
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(sum) });
    try std.testing.expect(!try runWithLimits(allocator, &func, .{ .max_nodes = 1 }));
}
