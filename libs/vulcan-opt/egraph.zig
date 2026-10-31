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

const reachable_marker = std.math.maxInt(u32);

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
    next_in_class: ?usize = null,
};

const OpKey = struct {
    op: BinOp,
    lhs: u32,
    rhs: u32,
    ty: Type,
};

const ClassData = struct {
    head: ?usize = null,
    tail: ?usize = null,
    ops: u16 = 0,
};

const Graph = struct {
    allocator: std.mem.Allocator,
    limits: Limits,
    parents: std.ArrayList(u32) = .empty,
    class_data: std.ArrayList(ClassData) = .empty,
    nodes: std.ArrayList(Entry) = .empty,
    leaves: std.AutoHashMapUnmanaged(Value, u32) = .empty,
    ops: std.AutoHashMapUnmanaged(OpKey, usize) = .empty,
    match_left: std.ArrayList(OpNode) = .empty,
    match_right: std.ArrayList(OpNode) = .empty,
    match_inner: std.ArrayList(OpNode) = .empty,
    op_mask: u16 = 0,
    ops_valid: bool = true,
    hit_limit: bool = false,

    fn deinit(self: *Graph) void {
        self.parents.deinit(self.allocator);
        self.class_data.deinit(self.allocator);
        self.nodes.deinit(self.allocator);
        self.leaves.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.match_left.deinit(self.allocator);
        self.match_right.deinit(self.allocator);
        self.match_inner.deinit(self.allocator);
    }

    fn reset(self: *Graph) void {
        self.parents.clearRetainingCapacity();
        self.class_data.clearRetainingCapacity();
        self.nodes.clearRetainingCapacity();
        self.leaves.clearRetainingCapacity();
        self.ops.clearRetainingCapacity();
        self.match_left.clearRetainingCapacity();
        self.match_right.clearRetainingCapacity();
        self.match_inner.clearRetainingCapacity();
        self.op_mask = 0;
        self.ops_valid = true;
        self.hit_limit = false;
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
        if (self.class_data.items[br].head) |br_head| {
            if (self.class_data.items[ar].head) |ar_head| {
                self.nodes.items[self.class_data.items[br].tail.?].next_in_class = ar_head;
                self.class_data.items[ar].head = br_head;
            } else {
                self.class_data.items[ar].head = br_head;
                self.class_data.items[ar].tail = self.class_data.items[br].tail;
            }
        }
        self.class_data.items[ar].ops |= self.class_data.items[br].ops;
        self.ops_valid = false;
        return true;
    }

    fn addClass(self: *Graph) !u32 {
        const class: u32 = @intCast(self.parents.items.len);
        try self.parents.append(self.allocator, class);
        try self.class_data.append(self.allocator, .{});
        return class;
    }

    fn attachNode(self: *Graph, class: u32, index: usize) void {
        std.debug.assert(self.class_data.items[class].head == null);
        self.class_data.items[class].head = index;
        self.class_data.items[class].tail = index;
        switch (self.nodes.items[index].node) {
            .leaf => {},
            .op => |op| self.class_data.items[class].ops |= opBit(op.op),
        }
    }

    fn addLeaf(self: *Graph, value: Value) !?u32 {
        if (self.leaves.get(value)) |class| return self.find(class);
        if (self.nodes.items.len >= self.limits.max_nodes) {
            self.hit_limit = true;
            return null;
        }
        const class = try self.addClass();
        try self.nodes.append(self.allocator, .{ .class = class, .node = .{ .leaf = value } });
        self.attachNode(class, self.nodes.items.len - 1);
        try self.leaves.put(self.allocator, value, class);
        return class;
    }

    fn addOp(self: *Graph, raw: OpNode) !?u32 {
        // Rebuild is deliberately deferred to the saturation iteration boundary. After a merge,
        // stale keys can only cause a missed lookup: canonicalization below never produces the
        // losing root again, so it cannot falsely match an entry whose children changed. A miss
        // may add a duplicate node, which the next rebuild congruence-merges. This is the bounded
        // "rebuilding" technique used by equality-saturation engines, avoiding a whole-graph
        // rebuild after every individual rule merge.
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
        const index = self.nodes.items.len - 1;
        self.attachNode(class, index);
        self.op_mask |= opBit(node.op);
        try self.ops.put(self.allocator, key, index);
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

    fn rebuildIfNeeded(self: *Graph) !bool {
        if (self.ops_valid) return false;
        return self.rebuild();
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

fn opBit(op: BinOp) u16 {
    return @as(u16, 1) << @intCast(@backingInt(op));
}

fn classHasOp(graph: *Graph, class: u32, op: BinOp) bool {
    return graph.class_data.items[graph.find(class)].ops & opBit(op) != 0;
}

fn isAssociative(op: BinOp) bool {
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

fn hasOptimizableRoots(term: ir.function.Terminator) bool {
    return switch (term) {
        .ret => |ret| ret.count != 0,
        .jump => |jump| jump.args.len != 0,
    };
}

/// Whether flattening an associative `outer` tree rooted at `value` finds at
/// least two terms headed by `inner`. Saturation can rotate those terms next to
/// one another and let the ordinary factoring rule inspect them. A fixed local
/// stack keeps this allocation-free; overflowing it conservatively enables the
/// bounded graph instead of missing an optimization in a very deep tree.
fn hasNestedFactorCandidate(func: *const Function, value: Value, outer: BinOp, inner: BinOp) bool {
    var pending: [64]Value = undefined;
    var len: usize = 1;
    var matches: u2 = 0;
    pending[0] = value;
    while (len != 0) {
        len -= 1;
        const current = pending[len];
        const inst = func.definingInst(current) orelse continue;
        const arith = switch (func.opcode(inst)) {
            .arith => |arith| arith,
            else => continue,
        };
        if (arith.op == outer) {
            if (len + 2 > pending.len) return true;
            pending[len] = arith.lhs;
            pending[len + 1] = arith.rhs;
            len += 2;
        } else if (arith.op == inner) {
            matches += 1;
            if (matches == 2) return true;
        }
    }
    return false;
}

fn sameValueOperands(left: ir.function.Arith, right: ir.function.Arith) bool {
    return (left.lhs == right.lhs and left.rhs == right.rhs) or
        (left.lhs == right.rhs and left.rhs == right.lhs);
}

fn isXorCarryCandidate(func: *const Function, exclusive: ir.function.Arith, carry: ir.function.Arith) bool {
    if (exclusive.op != .bit_xor or carry.op != .add or carry.lhs != carry.rhs) return false;
    const intersection_inst = func.definingInst(carry.lhs) orelse return false;
    const intersection = switch (func.opcode(intersection_inst)) {
        .arith => |arith| if (arith.op == .bit_and) arith else return false,
        else => return false,
    };
    return sameValueOperands(exclusive, intersection);
}

fn bitwisePairResult(outer: BinOp, left: BinOp, right: BinOp) ?BinOp {
    const and_or = (left == .bit_and and right == .bit_or) or (left == .bit_or and right == .bit_and);
    const and_xor = (left == .bit_and and right == .bit_xor) or (left == .bit_xor and right == .bit_and);
    const or_xor = (left == .bit_or and right == .bit_xor) or (left == .bit_xor and right == .bit_or);
    return switch (outer) {
        .bit_xor => if (and_or) .bit_xor else if (or_xor) .bit_and else if (and_xor) .bit_or else null,
        .bit_or => if (and_or or and_xor or or_xor) .bit_or else null,
        .bit_and => if (and_or) .bit_and else if (or_xor) .bit_xor else null,
        else => null,
    };
}

fn isDisjointBitwisePair(outer: BinOp, left: BinOp, right: BinOp) bool {
    return outer == .bit_and and
        ((left == .bit_and and right == .bit_xor) or (left == .bit_xor and right == .bit_and));
}

fn hasFactorCandidateAt(func: *const Function, value: Value, sum: ir.function.Arith) bool {
    const lhs = if (func.definingInst(sum.lhs)) |lhs_inst| switch (func.opcode(lhs_inst)) {
        .arith => |arith| arith,
        else => null,
    } else null;
    const rhs = if (func.definingInst(sum.rhs)) |rhs_inst| switch (func.opcode(rhs_inst)) {
        .arith => |arith| arith,
        else => null,
    } else null;
    if (factorInner(sum.op)) |inner| {
        if (lhs != null and rhs != null and lhs.?.op == inner and rhs.?.op == inner) return true;
        if (isAssociative(sum.op) and hasNestedFactorCandidate(func, value, sum.op, inner)) return true;
    }
    if (lhs != null and rhs != null) {
        if (cancellationInner(sum.op)) |inner| if (lhs.?.op == inner and rhs.?.op == inner) return true;
        if (sum.op == .sub and lhs.?.op == .sub and rhs.?.op == .sub) return true;
    }
    if (absorptionInner(sum.op)) |inner| {
        if (lhs) |op| if (op.op == inner and (op.lhs == sum.rhs or op.rhs == sum.rhs)) return true;
        if (rhs) |op| if (op.op == inner and (op.lhs == sum.lhs or op.rhs == sum.lhs)) return true;
    }
    if (sum.op == .add) {
        if (lhs) |op| if (op.op == .sub and op.rhs == sum.rhs) return true;
        if (rhs) |op| if (op.op == .sub and op.rhs == sum.lhs) return true;
        if (lhs != null and rhs != null) {
            if (isXorCarryCandidate(func, lhs.?, rhs.?) or isXorCarryCandidate(func, rhs.?, lhs.?)) return true;
            if (lhs.?.op == .sub and rhs.?.op == .add and (lhs.?.rhs == rhs.?.lhs or lhs.?.rhs == rhs.?.rhs)) return true;
            if (lhs.?.op == .add and rhs.?.op == .sub and (rhs.?.rhs == lhs.?.lhs or rhs.?.rhs == lhs.?.rhs)) return true;
            if (lhs.?.op == .sub and rhs.?.op == .sub and (lhs.?.rhs == rhs.?.lhs or lhs.?.lhs == rhs.?.rhs)) return true;
        }
        if (lhs != null and rhs != null and
            (((lhs.?.op == .add and rhs.?.op == .sub) or (lhs.?.op == .sub and rhs.?.op == .add)) and
                sameValueOperands(lhs.?, rhs.?))) return true;
        if (lhs != null and rhs != null and
            ((lhs.?.op == .bit_or and rhs.?.op == .bit_and) or
                (lhs.?.op == .bit_and and rhs.?.op == .bit_or)) and
            sameValueOperands(lhs.?, rhs.?)) return true;
    } else if (sum.op == .sub) {
        if (lhs) |op| if (op.op == .add and (op.lhs == sum.rhs or op.rhs == sum.rhs)) return true;
        if (rhs) |op| if (op.op == .sub and op.lhs == sum.lhs) return true;
        if (lhs != null and rhs != null and lhs.?.op == .add and rhs.?.op == .sub and (rhs.?.lhs == lhs.?.lhs or rhs.?.lhs == lhs.?.rhs)) return true;
        if (lhs != null and rhs != null and lhs.?.op == .add and rhs.?.op == .sub and sameValueOperands(lhs.?, rhs.?)) return true;
        if (lhs != null and rhs != null and lhs.?.op == .bit_or and (rhs.?.op == .bit_and or rhs.?.op == .bit_xor) and sameValueOperands(lhs.?, rhs.?)) return true;
        if (lhs != null and rhs != null and lhs.?.op == .add and
            (rhs.?.op == .bit_or or rhs.?.op == .bit_and or rhs.?.op == .bit_xor) and
            sameValueOperands(lhs.?, rhs.?)) return true;
    } else if (sum.op == .bit_xor) {
        if (lhs) |op| if (op.op == .bit_xor and (op.lhs == sum.rhs or op.rhs == sum.rhs)) return true;
        if (rhs) |op| if (op.op == .bit_xor and (op.lhs == sum.lhs or op.rhs == sum.lhs)) return true;
        if (lhs != null and rhs != null and bitwisePairResult(sum.op, lhs.?.op, rhs.?.op) != null and sameValueOperands(lhs.?, rhs.?)) return true;
    } else if (sum.op == .bit_and or sum.op == .bit_or) {
        if (lhs) |op| if (op.op == sum.op and (op.lhs == sum.rhs or op.rhs == sum.rhs)) return true;
        if (rhs) |op| if (op.op == sum.op and (op.lhs == sum.lhs or op.rhs == sum.lhs)) return true;
        if (lhs != null and rhs != null and
            (bitwisePairResult(sum.op, lhs.?.op, rhs.?.op) != null or isDisjointBitwisePair(sum.op, lhs.?.op, rhs.?.op)) and
            sameValueOperands(lhs.?, rhs.?)) return true;
    }
    return false;
}

fn isEgraphOp(op: BinOp) bool {
    return switch (op) {
        .add, .sub, .mul, .bit_and, .bit_or, .bit_xor => true,
        else => false,
    };
}

const ReachableExpressions = struct {
    values: [512]Value = undefined,
    len: usize = 0,
    candidate: bool = false,
    complete: bool = true,

    fn reset(self: *ReachableExpressions) void {
        self.len = 0;
        self.candidate = false;
        self.complete = true;
    }
};

fn collectReachableFromRoot(func: *const Function, root: Value, reachable: *ReachableExpressions) void {
    var pending: [64]Value = undefined;
    var len: usize = 1;
    pending[0] = root;
    while (len != 0) {
        len -= 1;
        const value = pending[len];
        if (reachable.len == reachable.values.len) {
            reachable.complete = false;
            reachable.candidate = true;
            return;
        }
        reachable.values[reachable.len] = value;
        reachable.len += 1;
        const inst = func.definingInst(value) orelse continue;
        const arith = switch (func.opcode(inst)) {
            .arith => |arith| if (isEgraphOp(arith.op)) arith else continue,
            else => continue,
        };
        if (!reachable.candidate and
            (factorInner(arith.op) != null or cancellationInner(arith.op) != null or absorptionInner(arith.op) != null) and
            hasFactorCandidateAt(func, value, arith)) reachable.candidate = true;
        if (len + 2 > pending.len) {
            reachable.complete = false;
            reachable.candidate = true;
            return;
        }
        pending[len] = arith.lhs;
        pending[len + 1] = arith.rhs;
        len += 2;
    }
}

/// Avoid graph construction unless an outgoing expression can match the current rule set.
fn collectReachableExpressions(func: *const Function, term: ir.function.Terminator, reachable: *ReachableExpressions) bool {
    reachable.reset();
    switch (term) {
        .ret => |ret| for (ret.slice()) |root| {
            collectReachableFromRoot(func, root, reachable);
            if (!reachable.complete) break;
        },
        .jump => |jump| for (func.valueList(jump.args)) |root| {
            collectReachableFromRoot(func, root, reachable);
            if (!reachable.complete) break;
        },
    }
    return reachable.candidate;
}

fn markReachableValue(allocator: std.mem.Allocator, classes: []?u32, touched: *std.ArrayList(usize), value: Value) !bool {
    const index = @backingInt(value);
    if (classes[index] != null) return false;
    classes[index] = reachable_marker;
    try touched.append(allocator, index);
    return true;
}

fn markWholeBlock(allocator: std.mem.Allocator, func: *const Function, block: Block, classes: []?u32, touched: *std.ArrayList(usize)) !void {
    for (func.blockParams(block)) |param| _ = try markReachableValue(allocator, classes, touched, param);
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse continue;
        _ = try markReachableValue(allocator, classes, touched, result);
    }
}

fn recordClass(allocator: std.mem.Allocator, classes: []?u32, costs: []usize, touched: *std.ArrayList(usize), value: Value, class: u32, cost: usize) !void {
    const index = @backingInt(value);
    if (classes[index] == null) try touched.append(allocator, index);
    classes[index] = class;
    costs[index] = cost;
}

fn classForValue(graph: *Graph, classes: []?u32, costs: []usize, touched: *std.ArrayList(usize), value: Value) !?u32 {
    const index = @backingInt(value);
    if (classes[index]) |class| if (class != reachable_marker) return class;
    const class = try graph.addLeaf(value) orelse return null;
    try recordClass(graph.allocator, classes, costs, touched, value, class, 0);
    return class;
}

fn importBlock(graph: *Graph, func: *const Function, block: Block, classes: []?u32, costs: []usize, touched: *std.ArrayList(usize)) !void {
    for (func.blockParams(block)) |param| if (classes[@backingInt(param)] != null) {
        _ = try classForValue(graph, classes, costs, touched, param);
    };
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse continue;
        if (classes[@backingInt(result)] == null) continue;
        if (!isInteger(func, result)) {
            _ = try classForValue(graph, classes, costs, touched, result);
            continue;
        }
        const arith = switch (func.opcode(inst)) {
            .arith => |arith| arith,
            else => {
                _ = try classForValue(graph, classes, costs, touched, result);
                continue;
            },
        };
        if (!isEgraphOp(arith.op)) {
            _ = try classForValue(graph, classes, costs, touched, result);
            continue;
        }
        const lhs = try classForValue(graph, classes, costs, touched, arith.lhs) orelse return;
        const rhs = try classForValue(graph, classes, costs, touched, arith.rhs) orelse return;
        const class = try graph.addOp(.{
            .op = arith.op,
            .lhs = lhs,
            .rhs = rhs,
            .ty = func.valueType(result),
            .existing = result,
        }) orelse return;
        const rhs_cost = if (arith.lhs == arith.rhs) 0 else costs[@backingInt(arith.rhs)];
        try recordClass(graph.allocator, classes, costs, touched, result, class, 1 + costs[@backingInt(arith.lhs)] + rhs_cost);
    }
}

fn importRoots(graph: *Graph, func: *const Function, term: ir.function.Terminator, classes: []?u32, costs: []usize, touched: *std.ArrayList(usize)) !void {
    switch (term) {
        .ret => |ret| for (ret.slice()) |root| {
            _ = try classForValue(graph, classes, costs, touched, root);
        },
        .jump => |jump| for (func.valueList(jump.args)) |root| {
            _ = try classForValue(graph, classes, costs, touched, root);
        },
    }
}

fn matchingOps(graph: *Graph, class: u32, op: BinOp, ty: Type, output: *std.ArrayList(OpNode)) !void {
    const root = graph.find(class);
    if (graph.class_data.items[root].ops & opBit(op) == 0) return;
    var current = graph.class_data.items[root].head;
    while (current) |index| {
        const entry = graph.nodes.items[index];
        switch (entry.node) {
            .leaf => {},
            .op => |candidate| if (candidate.op == op and candidate.ty == ty) {
                try output.append(graph.allocator, canonical(graph, candidate));
            },
        }
        current = entry.next_in_class;
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

fn factorPair(graph: *Graph, sum_class: u32, sum_op: BinOp, inner_op: BinOp, left: OpNode, right: OpNode) !?bool {
    const operands = commonOperands(left, right) orelse return null;
    const inner = try graph.addOp(.{ .op = sum_op, .lhs = operands.lhs_other, .rhs = operands.rhs_other, .ty = left.ty, .existing = null }) orelse return false;
    const product = try graph.addOp(.{ .op = inner_op, .lhs = operands.shared, .rhs = inner, .ty = left.ty, .existing = null }) orelse return false;
    return graph.merge(sum_class, product);
}

fn cancelPair(graph: *Graph, sum_class: u32, sum_op: BinOp, left: OpNode, right: OpNode) !?bool {
    const operands = commonOperands(left, right) orelse return null;
    const reduced = try graph.addOp(.{ .op = sum_op, .lhs = operands.lhs_other, .rhs = operands.rhs_other, .ty = left.ty, .existing = null }) orelse return false;
    return graph.merge(sum_class, reduced);
}

fn cancelSubtractions(graph: *Graph, outer_class: u32, left: OpNode, right: OpNode) !bool {
    const reduced = if (left.lhs == right.lhs)
        // (a - b) - (a - c) == c - b
        try graph.addOp(.{ .op = .sub, .lhs = right.rhs, .rhs = left.rhs, .ty = left.ty, .existing = null })
    else if (left.rhs == right.rhs)
        // (a - b) - (c - b) == a - c
        try graph.addOp(.{ .op = .sub, .lhs = left.lhs, .rhs = right.lhs, .ty = left.ty, .existing = null })
    else
        return false;
    return graph.merge(outer_class, reduced orelse return false);
}

fn cancelAddedSubtractions(graph: *Graph, outer_class: u32, left: OpNode, right: OpNode) !bool {
    const reduced = if (graph.find(left.rhs) == graph.find(right.lhs))
        // (a - b) + (b - c) == a - c
        try graph.addOp(.{ .op = .sub, .lhs = left.lhs, .rhs = right.rhs, .ty = left.ty, .existing = null })
    else if (graph.find(left.lhs) == graph.find(right.rhs))
        // (a - b) + (c - a) == c - b
        try graph.addOp(.{ .op = .sub, .lhs = right.lhs, .rhs = left.rhs, .ty = left.ty, .existing = null })
    else
        return false;
    return graph.merge(outer_class, reduced orelse return false);
}

fn absorbPair(graph: *Graph, outer_class: u32, shared_class: u32, inner: OpNode) bool {
    const shared = graph.find(shared_class);
    if (graph.find(inner.lhs) != shared and graph.find(inner.rhs) != shared) return false;
    return graph.merge(outer_class, shared);
}

fn associatePair(graph: *Graph, outer_class: u32, op: BinOp, nested: OpNode, other: u32) !bool {
    const inner = try graph.addOp(.{ .op = op, .lhs = nested.rhs, .rhs = other, .ty = nested.ty, .existing = null }) orelse return false;
    const rotated = try graph.addOp(.{ .op = op, .lhs = nested.lhs, .rhs = inner, .ty = nested.ty, .existing = null }) orelse return false;
    return graph.merge(outer_class, rotated);
}

fn sameClassOperands(graph: *Graph, left: OpNode, right: OpNode) bool {
    const ll = graph.find(left.lhs);
    const lr = graph.find(left.rhs);
    const rl = graph.find(right.lhs);
    const rr = graph.find(right.rhs);
    return (ll == rl and lr == rr) or (ll == rr and lr == rl);
}

/// Arithmetic combinations of matching bitwise terms that reduce to one operation for wrapping
/// integers: OR + AND == ADD, OR - AND == XOR, and OR - XOR == AND.
fn foldBitwiseArithmetic(graph: *Graph, outer_class: u32, outer_op: BinOp, disjunction: OpNode, right: OpNode) !bool {
    if (!sameClassOperands(graph, disjunction, right)) return false;
    const replacement_op: BinOp = if (outer_op == .add)
        .add
    else if (right.op == .bit_and)
        .bit_xor
    else
        .bit_and;
    const replacement = try graph.addOp(.{
        .op = replacement_op,
        .lhs = disjunction.lhs,
        .rhs = disjunction.rhs,
        .ty = disjunction.ty,
        .existing = null,
    }) orelse return false;
    return graph.merge(outer_class, replacement);
}

/// (a + b) - (a | b) == a & b, (a + b) - (a & b) == a | b, and
/// (a + b) - (a ^ b) == (a & b) + (a & b).
fn foldAddBitwiseDifference(graph: *Graph, outer_class: u32, addition: OpNode, bitwise: OpNode) !bool {
    if (!sameClassOperands(graph, addition, bitwise)) return false;
    if (bitwise.op == .bit_xor) {
        const intersection = try graph.addOp(.{
            .op = .bit_and,
            .lhs = addition.lhs,
            .rhs = addition.rhs,
            .ty = addition.ty,
            .existing = null,
        }) orelse return false;
        return mergeDouble(graph, outer_class, addition.ty, intersection);
    }
    const replacement = try graph.addOp(.{
        .op = if (bitwise.op == .bit_or) .bit_and else .bit_or,
        .lhs = addition.lhs,
        .rhs = addition.rhs,
        .ty = addition.ty,
        .existing = null,
    }) orelse return false;
    return graph.merge(outer_class, replacement);
}

fn foldXorCarryOrientation(graph: *Graph, outer_class: u32, exclusive_class: u32, carry_class: u32, ty: Type, exclusives: *std.ArrayList(OpNode), carries: *std.ArrayList(OpNode), intersections: *std.ArrayList(OpNode)) !bool {
    exclusives.clearRetainingCapacity();
    carries.clearRetainingCapacity();
    try matchingOps(graph, exclusive_class, .bit_xor, ty, exclusives);
    try matchingOps(graph, carry_class, .add, ty, carries);
    var changed = false;
    for (exclusives.items) |exclusive| for (carries.items) |carry| {
        if (graph.find(carry.lhs) != graph.find(carry.rhs)) continue;
        intersections.clearRetainingCapacity();
        try matchingOps(graph, carry.lhs, .bit_and, ty, intersections);
        for (intersections.items) |intersection| {
            if (!sameClassOperands(graph, exclusive, intersection)) continue;
            const replacement = try graph.addOp(.{
                .op = .add,
                .lhs = exclusive.lhs,
                .rhs = exclusive.rhs,
                .ty = ty,
                .existing = null,
            }) orelse return changed;
            changed = graph.merge(outer_class, replacement) or changed;
        }
        if (graph.hit_limit) break;
    };
    return changed;
}

fn foldXorCarry(graph: *Graph, outer_class: u32, outer: OpNode, exclusives: *std.ArrayList(OpNode), carries: *std.ArrayList(OpNode), intersections: *std.ArrayList(OpNode)) !bool {
    var changed = false;
    if (classHasOp(graph, outer.lhs, .bit_xor) and classHasOp(graph, outer.rhs, .add)) {
        changed = try foldXorCarryOrientation(graph, outer_class, outer.lhs, outer.rhs, outer.ty, exclusives, carries, intersections) or changed;
    }
    if (!graph.hit_limit and classHasOp(graph, outer.rhs, .bit_xor) and classHasOp(graph, outer.lhs, .add)) {
        changed = try foldXorCarryOrientation(graph, outer_class, outer.rhs, outer.lhs, outer.ty, exclusives, carries, intersections) or changed;
    }
    return changed;
}

fn foldNestedIdempotent(graph: *Graph, outer_class: u32, outer: OpNode, matches: *std.ArrayList(OpNode)) !?bool {
    var matched = false;
    var changed = false;
    matches.clearRetainingCapacity();
    try matchingOps(graph, outer.lhs, outer.op, outer.ty, matches);
    const rhs = graph.find(outer.rhs);
    for (matches.items) |nested| {
        if (graph.find(nested.lhs) == rhs or graph.find(nested.rhs) == rhs) {
            matched = true;
            changed = graph.merge(outer_class, outer.lhs) or changed;
        }
    }
    matches.clearRetainingCapacity();
    try matchingOps(graph, outer.rhs, outer.op, outer.ty, matches);
    const lhs = graph.find(outer.lhs);
    for (matches.items) |nested| {
        if (graph.find(nested.lhs) == lhs or graph.find(nested.rhs) == lhs) {
            matched = true;
            changed = graph.merge(outer_class, outer.rhs) or changed;
        }
    }
    return if (matched) changed else null;
}

fn foldBitwisePairs(graph: *Graph, outer_class: u32, outer: OpNode, left: *std.ArrayList(OpNode), right: *std.ArrayList(OpNode)) !?bool {
    const bit_ops = [_]BinOp{ .bit_and, .bit_or, .bit_xor };
    var matched = false;
    var changed = false;
    for (bit_ops) |lhs_op| for (bit_ops) |rhs_op| {
        const result_op = bitwisePairResult(outer.op, lhs_op, rhs_op);
        const produces_zero = isDisjointBitwisePair(outer.op, lhs_op, rhs_op);
        if (result_op == null and !produces_zero) continue;
        left.clearRetainingCapacity();
        right.clearRetainingCapacity();
        try matchingOps(graph, outer.lhs, lhs_op, outer.ty, left);
        try matchingOps(graph, outer.rhs, rhs_op, outer.ty, right);
        for (left.items) |a| for (right.items) |b| {
            if (!sameClassOperands(graph, a, b)) continue;
            matched = true;
            const replacement = try graph.addOp(.{
                .op = if (produces_zero) .bit_xor else result_op.?,
                .lhs = a.lhs,
                .rhs = if (produces_zero) a.lhs else a.rhs,
                .ty = a.ty,
                .existing = null,
            }) orelse return changed;
            changed = graph.merge(outer_class, replacement) or changed;
        };
        if (graph.hit_limit) break;
    };
    return if (matched) changed else null;
}

fn mergeDouble(graph: *Graph, outer_class: u32, ty: Type, value: u32) !bool {
    const doubled = try graph.addOp(.{ .op = .add, .lhs = value, .rhs = value, .ty = ty, .existing = null }) orelse return false;
    return graph.merge(outer_class, doubled);
}

fn foldMixedAddSub(graph: *Graph, outer_class: u32, outer: OpNode, left: *std.ArrayList(OpNode), right: *std.ArrayList(OpNode)) !bool {
    var changed = false;
    left.clearRetainingCapacity();
    right.clearRetainingCapacity();
    try matchingOps(graph, outer.lhs, .add, outer.ty, left);
    try matchingOps(graph, outer.rhs, .sub, outer.ty, right);
    for (left.items) |addition| for (right.items) |subtraction| {
        if (!sameClassOperands(graph, addition, subtraction)) continue;
        const repeated = if (outer.op == .add) subtraction.lhs else subtraction.rhs;
        changed = try mergeDouble(graph, outer_class, outer.ty, repeated) or changed;
    };
    if (outer.op != .add or graph.hit_limit) return changed;

    left.clearRetainingCapacity();
    right.clearRetainingCapacity();
    try matchingOps(graph, outer.lhs, .sub, outer.ty, left);
    try matchingOps(graph, outer.rhs, .add, outer.ty, right);
    for (left.items) |subtraction| for (right.items) |addition| {
        if (!sameClassOperands(graph, subtraction, addition)) continue;
        changed = try mergeDouble(graph, outer_class, outer.ty, subtraction.lhs) or changed;
    };
    return changed;
}

fn hasMixedAddSubChildren(graph: *Graph, outer: OpNode) bool {
    const add_sub = classHasOp(graph, outer.lhs, .add) and classHasOp(graph, outer.rhs, .sub);
    return add_sub or (outer.op == .add and classHasOp(graph, outer.lhs, .sub) and classHasOp(graph, outer.rhs, .add));
}

fn hasInverseChildren(graph: *Graph, outer: OpNode) bool {
    return switch (outer.op) {
        .add => classHasOp(graph, outer.lhs, .sub) or classHasOp(graph, outer.rhs, .sub),
        .sub => classHasOp(graph, outer.lhs, .add) or classHasOp(graph, outer.rhs, .sub),
        .bit_xor => classHasOp(graph, outer.lhs, .bit_xor) or classHasOp(graph, outer.rhs, .bit_xor),
        else => false,
    };
}

fn hasBitwiseArithmeticChildren(graph: *Graph, outer: OpNode) bool {
    const rhs_match = classHasOp(graph, outer.rhs, .bit_and) or (outer.op == .sub and classHasOp(graph, outer.rhs, .bit_xor));
    const forward = classHasOp(graph, outer.lhs, .bit_or) and rhs_match;
    return forward or (outer.op == .add and classHasOp(graph, outer.rhs, .bit_or) and classHasOp(graph, outer.lhs, .bit_and));
}

fn emitCrossCancellation(graph: *Graph, outer_class: u32, ty: Type, subtraction: OpNode, addition: OpNode, subtract_outer: bool) !bool {
    const cancelled = graph.find(if (subtract_outer) subtraction.lhs else subtraction.rhs);
    const other = if (graph.find(addition.lhs) == cancelled)
        addition.rhs
    else if (graph.find(addition.rhs) == cancelled)
        addition.lhs
    else
        return false;
    const kept = if (subtract_outer) subtraction.rhs else subtraction.lhs;
    const replacement = try graph.addOp(.{ .op = .add, .lhs = kept, .rhs = other, .ty = ty, .existing = null }) orelse return false;
    return graph.merge(outer_class, replacement);
}

fn cancelCrossTerms(graph: *Graph, outer_class: u32, outer: OpNode, left: *std.ArrayList(OpNode), right: *std.ArrayList(OpNode)) !bool {
    var changed = false;
    left.clearRetainingCapacity();
    right.clearRetainingCapacity();
    try matchingOps(graph, outer.lhs, if (outer.op == .add) .sub else .add, outer.ty, left);
    try matchingOps(graph, outer.rhs, if (outer.op == .add) .add else .sub, outer.ty, right);
    for (left.items) |a| for (right.items) |b| {
        changed = try emitCrossCancellation(
            graph,
            outer_class,
            outer.ty,
            if (outer.op == .add) a else b,
            if (outer.op == .add) b else a,
            outer.op == .sub,
        ) or changed;
    };
    if (outer.op != .add or graph.hit_limit) return changed;

    left.clearRetainingCapacity();
    right.clearRetainingCapacity();
    try matchingOps(graph, outer.lhs, .add, outer.ty, left);
    try matchingOps(graph, outer.rhs, .sub, outer.ty, right);
    for (left.items) |addition| for (right.items) |subtraction| {
        changed = try emitCrossCancellation(graph, outer_class, outer.ty, subtraction, addition, false) or changed;
    };
    return changed;
}

fn cancelInverse(graph: *Graph, outer_class: u32, outer: OpNode, left: *std.ArrayList(OpNode), right: *std.ArrayList(OpNode)) !?bool {
    var matched = false;
    var changed = false;
    if (outer.op == .add) {
        left.clearRetainingCapacity();
        try matchingOps(graph, outer.lhs, .sub, outer.ty, left);
        for (left.items) |nested| {
            if (graph.find(nested.rhs) == graph.find(outer.rhs)) {
                matched = true;
                changed = graph.merge(outer_class, nested.lhs) or changed;
            }
        }
        right.clearRetainingCapacity();
        try matchingOps(graph, outer.rhs, .sub, outer.ty, right);
        for (right.items) |nested| {
            if (graph.find(nested.rhs) == graph.find(outer.lhs)) {
                matched = true;
                changed = graph.merge(outer_class, nested.lhs) or changed;
            }
        }
    } else if (outer.op == .sub) {
        left.clearRetainingCapacity();
        try matchingOps(graph, outer.lhs, .add, outer.ty, left);
        for (left.items) |nested| {
            const rhs = graph.find(outer.rhs);
            if (graph.find(nested.lhs) == rhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.rhs) or changed;
            } else if (graph.find(nested.rhs) == rhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.lhs) or changed;
            }
        }
        right.clearRetainingCapacity();
        try matchingOps(graph, outer.rhs, .sub, outer.ty, right);
        for (right.items) |nested| {
            if (graph.find(nested.lhs) == graph.find(outer.lhs)) {
                matched = true;
                changed = graph.merge(outer_class, nested.rhs) or changed;
            }
        }
    } else if (outer.op == .bit_xor) {
        left.clearRetainingCapacity();
        try matchingOps(graph, outer.lhs, .bit_xor, outer.ty, left);
        for (left.items) |nested| {
            const rhs = graph.find(outer.rhs);
            if (graph.find(nested.lhs) == rhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.rhs) or changed;
            } else if (graph.find(nested.rhs) == rhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.lhs) or changed;
            }
        }
        right.clearRetainingCapacity();
        try matchingOps(graph, outer.rhs, .bit_xor, outer.ty, right);
        for (right.items) |nested| {
            const lhs = graph.find(outer.lhs);
            if (graph.find(nested.lhs) == lhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.rhs) or changed;
            } else if (graph.find(nested.rhs) == lhs) {
                matched = true;
                changed = graph.merge(outer_class, nested.lhs) or changed;
            }
        }
    }
    return if (matched) changed else null;
}

fn saturate(graph: *Graph) !void {
    const left = &graph.match_left;
    const right = &graph.match_right;
    const nested_matches = &graph.match_inner;

    var iter: usize = 0;
    while (iter < graph.limits.max_iters and !graph.hit_limit) : (iter += 1) {
        var changed = try graph.rebuildIfNeeded();
        var node_index: usize = 0;
        while (node_index < graph.nodes.items.len) : (node_index += 1) {
            // Rules append nodes and may reallocate the ArrayList. Fetch each entry by index
            // instead of retaining a slice, and process new nodes in the same saturation round.
            const entry = graph.nodes.items[node_index];
            const sum = switch (entry.node) {
                .op => |op| if (isAssociative(op.op) or factorInner(op.op) != null or cancellationInner(op.op) != null or absorptionInner(op.op) != null) canonical(graph, op) else continue,
                .leaf => continue,
            };
            var direct_match = false;
            if (factorInner(sum.op)) |inner_op| factor: {
                if (!classHasOp(graph, sum.lhs, inner_op) or !classHasOp(graph, sum.rhs, inner_op)) break :factor;
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, inner_op, sum.ty, left);
                try matchingOps(graph, sum.rhs, inner_op, sum.ty, right);
                for (left.items) |a| for (right.items) |b| {
                    if (try factorPair(graph, entry.class, sum.op, inner_op, a, b)) |pair_changed| {
                        direct_match = true;
                        changed = pair_changed or changed;
                    }
                    if (graph.hit_limit) break;
                };
            }
            if (cancellationInner(sum.op)) |inner_op| cancellation: {
                if (!classHasOp(graph, sum.lhs, inner_op) or !classHasOp(graph, sum.rhs, inner_op)) break :cancellation;
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, inner_op, sum.ty, left);
                try matchingOps(graph, sum.rhs, inner_op, sum.ty, right);
                for (left.items) |a| for (right.items) |b| {
                    if (try cancelPair(graph, entry.class, sum.op, a, b)) |pair_changed| {
                        direct_match = true;
                        changed = pair_changed or changed;
                    }
                    if (graph.hit_limit) break;
                };
            }
            if (sum.op == .bit_and or sum.op == .bit_or) {
                if (try foldNestedIdempotent(graph, entry.class, sum, left)) |rule_changed| {
                    direct_match = true;
                    changed = rule_changed or changed;
                }
            }
            if (sum.op == .bit_and or sum.op == .bit_or or sum.op == .bit_xor) {
                if (try foldBitwisePairs(graph, entry.class, sum, left, right)) |rule_changed| {
                    direct_match = true;
                    changed = rule_changed or changed;
                }
            }
            if (hasInverseChildren(graph, sum)) {
                if (try cancelInverse(graph, entry.class, sum, left, right)) |rule_changed| {
                    direct_match = true;
                    changed = rule_changed or changed;
                }
            }
            // A direct reduction already introduces the remaining associative expression. Let
            // the eager walk process that smaller expression instead of expanding this node first.
            if (!direct_match and isAssociative(sum.op) and classHasOp(graph, sum.lhs, sum.op)) {
                left.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, sum.op, sum.ty, left);
                for (left.items) |nested| {
                    changed = try associatePair(graph, entry.class, sum.op, nested, sum.rhs) or changed;
                    if (graph.hit_limit) break;
                }
            }
            if (!direct_match and isAssociative(sum.op) and classHasOp(graph, sum.rhs, sum.op)) {
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.rhs, sum.op, sum.ty, right);
                for (right.items) |nested| {
                    changed = try associatePair(graph, entry.class, sum.op, nested, sum.lhs) or changed;
                    if (graph.hit_limit) break;
                }
            }
            if (sum.op == .sub and classHasOp(graph, sum.lhs, .sub) and classHasOp(graph, sum.rhs, .sub)) {
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, .sub, sum.ty, left);
                try matchingOps(graph, sum.rhs, .sub, sum.ty, right);
                for (left.items) |a| for (right.items) |b| {
                    changed = try cancelSubtractions(graph, entry.class, a, b) or changed;
                    if (graph.hit_limit) break;
                };
            }
            if (sum.op == .add and classHasOp(graph, sum.lhs, .sub) and classHasOp(graph, sum.rhs, .sub)) {
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, .sub, sum.ty, left);
                try matchingOps(graph, sum.rhs, .sub, sum.ty, right);
                for (left.items) |a| for (right.items) |b| {
                    changed = try cancelAddedSubtractions(graph, entry.class, a, b) or changed;
                    if (graph.hit_limit) break;
                };
            }
            if ((sum.op == .add or sum.op == .sub) and hasMixedAddSubChildren(graph, sum)) {
                changed = try foldMixedAddSub(graph, entry.class, sum, left, right) or changed;
                changed = try cancelCrossTerms(graph, entry.class, sum, left, right) or changed;
            }
            if (sum.op == .add and graph.op_mask & opBit(.bit_xor) != 0) {
                changed = try foldXorCarry(graph, entry.class, sum, left, right, nested_matches) or changed;
            }
            if ((sum.op == .add or sum.op == .sub) and hasBitwiseArithmeticChildren(graph, sum)) {
                left.clearRetainingCapacity();
                right.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, .bit_or, sum.ty, left);
                try matchingOps(graph, sum.rhs, .bit_and, sum.ty, right);
                for (left.items) |disjunction| for (right.items) |conjunction| {
                    changed = try foldBitwiseArithmetic(graph, entry.class, sum.op, disjunction, conjunction) or changed;
                    if (graph.hit_limit) break;
                };
                if (sum.op == .sub) {
                    right.clearRetainingCapacity();
                    try matchingOps(graph, sum.rhs, .bit_xor, sum.ty, right);
                    for (left.items) |disjunction| for (right.items) |exclusive| {
                        changed = try foldBitwiseArithmetic(graph, entry.class, sum.op, disjunction, exclusive) or changed;
                        if (graph.hit_limit) break;
                    };
                } else {
                    left.clearRetainingCapacity();
                    right.clearRetainingCapacity();
                    try matchingOps(graph, sum.rhs, .bit_or, sum.ty, left);
                    try matchingOps(graph, sum.lhs, .bit_and, sum.ty, right);
                    for (left.items) |disjunction| for (right.items) |conjunction| {
                        changed = try foldBitwiseArithmetic(graph, entry.class, sum.op, disjunction, conjunction) or changed;
                        if (graph.hit_limit) break;
                    };
                }
            }
            if (sum.op == .sub and classHasOp(graph, sum.lhs, .add) and
                (classHasOp(graph, sum.rhs, .bit_or) or classHasOp(graph, sum.rhs, .bit_and) or classHasOp(graph, sum.rhs, .bit_xor)))
            {
                left.clearRetainingCapacity();
                try matchingOps(graph, sum.lhs, .add, sum.ty, left);
                for ([_]BinOp{ .bit_or, .bit_and, .bit_xor }) |rhs_op| {
                    right.clearRetainingCapacity();
                    try matchingOps(graph, sum.rhs, rhs_op, sum.ty, right);
                    for (left.items) |addition| for (right.items) |bitwise| {
                        changed = try foldAddBitwiseDifference(graph, entry.class, addition, bitwise) or changed;
                        if (graph.hit_limit) break;
                    };
                    if (graph.hit_limit) break;
                }
            }
            if (absorptionInner(sum.op)) |inner_op| {
                if (classHasOp(graph, sum.lhs, inner_op)) {
                    left.clearRetainingCapacity();
                    try matchingOps(graph, sum.lhs, inner_op, sum.ty, left);
                    for (left.items) |inner| changed = absorbPair(graph, entry.class, sum.rhs, inner) or changed;
                }
                if (classHasOp(graph, sum.rhs, inner_op)) {
                    right.clearRetainingCapacity();
                    try matchingOps(graph, sum.rhs, inner_op, sum.ty, right);
                    for (right.items) |inner| changed = absorbPair(graph, entry.class, sum.lhs, inner) or changed;
                }
            }
            if (graph.hit_limit) break;
        }
        if (!changed) break;
    }
    _ = try graph.rebuildIfNeeded();
}

const Extraction = struct {
    costs: []usize,
    best: []?usize,
};

fn extract(graph: *Graph, costs: []usize, best: []?usize) Extraction {
    std.debug.assert(costs.len == graph.parents.items.len);
    std.debug.assert(best.len == graph.parents.items.len);
    @memset(costs, std.math.maxInt(usize));
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
                    const lhs_root = graph.find(op.lhs);
                    const rhs_root = graph.find(op.rhs);
                    const lhs = costs[lhs_root];
                    const rhs = costs[rhs_root];
                    if (lhs == std.math.maxInt(usize) or rhs == std.math.maxInt(usize)) continue;
                    break :blk 1 + lhs + if (lhs_root == rhs_root) 0 else rhs;
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
    if (class == reachable_marker) return null;
    const root = graph.find(class);
    if (extraction.costs[root] >= original_costs[@backingInt(root_value)]) return null;
    return try materialize(graph, extraction, func, block, root, memo);
}

pub fn runWithLimits(allocator: std.mem.Allocator, func: *Function, limits: Limits) pass.Error!bool {
    if (limits.max_nodes == 0 or limits.max_iters == 0) return false;
    var changed = false;
    // These maps are indexed by function-global Value handles, but every graph below is
    // block-local. Allocate them lazily for the first candidate and reuse the storage rather than
    // allocating two valueCount-sized maps per candidate block. This matters especially for arena
    // allocators, where free is intentionally a no-op until the whole compilation is torn down.
    var classes_storage: ?[]?u32 = null;
    defer if (classes_storage) |classes| allocator.free(classes);
    var costs_storage: ?[]usize = null;
    defer if (costs_storage) |costs| allocator.free(costs);
    var touched: std.ArrayList(usize) = .empty;
    defer touched.deinit(allocator);
    var graph = Graph{ .allocator = allocator, .limits = limits };
    defer graph.deinit();
    var extraction_costs: std.ArrayList(usize) = .empty;
    defer extraction_costs.deinit(allocator);
    var extraction_best: std.ArrayList(?usize) = .empty;
    defer extraction_best.deinit(allocator);
    var memo_storage: std.ArrayList(?Value) = .empty;
    defer memo_storage.deinit(allocator);
    var reachable: ReachableExpressions = .{};
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const term = func.terminator(block) orelse continue;
        if (!hasOptimizableRoots(term)) continue;
        if (!collectReachableExpressions(func, term, &reachable)) continue;
        graph.reset();
        if (classes_storage == null) {
            const classes = try allocator.alloc(?u32, func.valueCount());
            errdefer allocator.free(classes);
            const costs = try allocator.alloc(usize, func.valueCount());
            @memset(classes, null);
            classes_storage = classes;
            costs_storage = costs;
        }
        const classes = classes_storage.?;
        const original_costs = costs_storage.?;
        std.debug.assert(touched.items.len == 0);
        if (reachable.complete) {
            for (reachable.values[0..reachable.len]) |value| {
                _ = try markReachableValue(allocator, classes, &touched, value);
            }
        } else {
            try markWholeBlock(allocator, func, block, classes, &touched);
        }
        try importBlock(&graph, func, block, classes, original_costs, &touched);
        try importRoots(&graph, func, term, classes, original_costs, &touched);
        try saturate(&graph);
        try extraction_costs.resize(allocator, graph.parents.items.len);
        try extraction_best.resize(allocator, graph.parents.items.len);
        const extraction = extract(&graph, extraction_costs.items, extraction_best.items);
        try memo_storage.resize(allocator, graph.parents.items.len);
        const memo = memo_storage.items;
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
        for (touched.items) |index| classes[index] = null;
        touched.clearRetainingCapacity();
    }
    return changed;
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    return runWithLimits(allocator, func, .{});
}

fn expectBitwisePair(outer_op: BinOp, left_op: BinOp, right_op: BinOp, expected: BinOp) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const left = try func.appendInst(block, ty, .{ .arith = .{ .op = left_op, .lhs = a, .rhs = b } });
    const right = try func.appendInst(block, ty, .{ .arith = .{ .op = right_op, .lhs = b, .rhs = a } });
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = outer_op, .lhs = left, .rhs = right } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(expected, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

fn expectAddBitwiseDifference(rhs_op: BinOp, expected: BinOp) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const bitwise = try func.appendInst(block, ty, .{ .arith = .{ .op = rhs_op, .lhs = b, .rhs = a } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = sum, .rhs = bitwise } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(expected, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

fn expectXorCarry(exclusive_on_left: bool) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const intersection = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const carry = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = intersection, .rhs = intersection } });
    const exclusive = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = b, .rhs = a } });
    const lhs = if (exclusive_on_left) exclusive else carry;
    const rhs = if (exclusive_on_left) carry else exclusive;
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = lhs, .rhs = rhs } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.add, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

fn expectMixedAddSub(outer_op: BinOp, addition_on_left: bool, repeat_a: bool) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const addition = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = b, .rhs = a } });
    const subtraction = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const lhs = if (addition_on_left) addition else subtraction;
    const rhs = if (addition_on_left) subtraction else addition;
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = outer_op, .lhs = lhs, .rhs = rhs } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const doubled = func.opcode(func.definingInst(replacement).?).arith;
    const expected = if (repeat_a) a else b;
    try std.testing.expectEqual(BinOp.add, doubled.op);
    try std.testing.expectEqual(expected, doubled.lhs);
    try std.testing.expectEqual(expected, doubled.rhs);
}

fn expectCrossCancellation(outer_op: BinOp, subtraction_on_left: bool) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const addition = if (outer_op == .add)
        try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = c, .rhs = b } })
    else
        try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = b, .rhs = a } });
    const subtraction = if (outer_op == .add)
        try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } })
    else
        try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = b, .rhs = c } });
    const lhs = if (subtraction_on_left) subtraction else addition;
    const rhs = if (subtraction_on_left) addition else subtraction;
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = outer_op, .lhs = lhs, .rhs = rhs } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.add, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == c) or (reduced.lhs == c and reduced.rhs == a));
}

fn expectAddedSubtractionCancellation(cancel_subtrahend: bool) !void {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const left = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const right = if (cancel_subtrahend)
        try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = b, .rhs = c } })
    else
        try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = c, .rhs = a } });
    const combined = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = left, .rhs = right } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(combined) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.sub, reduced.op);
    if (cancel_subtrahend) {
        try std.testing.expectEqual(a, reduced.lhs);
        try std.testing.expectEqual(c, reduced.rhs);
    } else {
        try std.testing.expectEqual(c, reduced.lhs);
        try std.testing.expectEqual(b, reduced.rhs);
    }
}

test "egraph recognizes terminators without optimizable roots" {
    try std.testing.expect(!hasOptimizableRoots(.{ .ret = ir.function.Ret.none() }));
    try std.testing.expect(hasOptimizableRoots(.{ .ret = ir.function.Ret.one(@fromBackingInt(0)) }));
    try std.testing.expect(!hasOptimizableRoots(.{ .jump = .{ .target = @fromBackingInt(0), .args = .{ .start = 0, .len = 0 } } }));
    try std.testing.expect(hasOptimizableRoots(.{ .jump = .{ .target = @fromBackingInt(0), .args = .{ .start = 0, .len = 1 } } }));
}

test "egraph candidate discovery follows outgoing expressions" {
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

    var reachable: ReachableExpressions = .{};
    try std.testing.expect(!collectReachableExpressions(&func, .{ .ret = ir.function.Ret.one(a) }, &reachable));
    try std.testing.expect(collectReachableExpressions(&func, .{ .ret = ir.function.Ret.one(sum) }, &reachable));
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

test "egraph chains direct factoring across an add tree" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const d = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const ad = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = d } });
    const partial = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
    const total = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = partial, .rhs = ad } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(total) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const product = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.mul, product.op);
    const sum_value = if (product.lhs == a) product.rhs else product.lhs;
    try std.testing.expect(product.lhs == a or product.rhs == a);
    try std.testing.expectEqual(BinOp.add, func.opcode(func.definingInst(sum_value).?).arith.op);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "egraph optimizes multiple candidate blocks while values grow" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const exit = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ty);
    const b = try func.appendBlockParam(entry, ty);
    const c = try func.appendBlockParam(entry, ty);
    _ = try func.appendBlockParam(exit, ty);

    const ab0 = try func.appendInst(entry, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac0 = try func.appendInst(entry, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const sum0 = try func.appendInst(entry, ty, .{ .arith = .{ .op = .add, .lhs = ab0, .rhs = ac0 } });
    try func.setJump(entry, exit, &.{sum0});

    const ab1 = try func.appendInst(exit, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac1 = try func.appendInst(exit, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const sum1 = try func.appendInst(exit, ty, .{ .arith = .{ .op = .add, .lhs = ab1, .rhs = ac1 } });
    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(sum1) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const jump_value = func.blockArgs(func.terminator(entry).?.jump)[0];
    const return_value = func.terminator(exit).?.ret.values[0];
    try std.testing.expectEqual(BinOp.mul, func.opcode(func.definingInst(jump_value).?).arith.op);
    try std.testing.expectEqual(BinOp.mul, func.opcode(func.definingInst(return_value).?).arith.op);
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

test "egraph cancels a shared subtraction minuend with reversed remainder" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = c } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = ab, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.sub, reduced.op);
    try std.testing.expectEqual(c, reduced.lhs);
    try std.testing.expectEqual(b, reduced.rhs);
}

test "egraph cancels a shared subtraction subtrahend" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const cb = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = c, .rhs = b } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = ab, .rhs = cb } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.sub, reduced.op);
    try std.testing.expectEqual(a, reduced.lhs);
    try std.testing.expectEqual(c, reduced.rhs);
}

test "egraph cancels add and subtract inverse pairs" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const restored = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = difference, .rhs = b } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(restored) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(a, func.terminator(block).?.ret.values[0]);
}

test "egraph cancels a nested add from either side of subtraction" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const restored = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = sum, .rhs = b } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(restored) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(a, func.terminator(block).?.ret.values[0]);
}

test "egraph cancels a nested subtraction from its minuend" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } });
    const restored = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = difference } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(restored) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(b, func.terminator(block).?.ret.values[0]);
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

test "egraph cancels an xor term nested on either side" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = b } });
    const restored = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = ab } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(restored) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(b, func.terminator(block).?.ret.values[0]);
}

test "egraph reduces the sum of bitwise union and intersection" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } });
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = b, .rhs = a } });
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = both, .rhs = either } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(sum) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.add, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

test "egraph reduces the difference of bitwise union and intersection" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } });
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = either, .rhs = both } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.bit_xor, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

test "egraph reduces the difference of bitwise union and xor" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } });
    const exclusive = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = b, .rhs = a } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = either, .rhs = exclusive } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const reduced = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.bit_and, reduced.op);
    try std.testing.expect((reduced.lhs == a and reduced.rhs == b) or (reduced.lhs == b and reduced.rhs == a));
}

test "egraph reduces addition minus matching bitwise terms" {
    try expectAddBitwiseDifference(.bit_or, .bit_and);
    try expectAddBitwiseDifference(.bit_and, .bit_or);
}

test "egraph reduces addition minus xor to doubled intersection" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    const exclusive = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = b, .rhs = a } });
    const difference = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = sum, .rhs = exclusive } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const doubled = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.add, doubled.op);
    try std.testing.expectEqual(doubled.lhs, doubled.rhs);
    const intersection = func.opcode(func.definingInst(doubled.lhs).?).arith;
    try std.testing.expectEqual(BinOp.bit_and, intersection.op);
    try std.testing.expect((intersection.lhs == a and intersection.rhs == b) or (intersection.lhs == b and intersection.rhs == a));
}

test "egraph recomposes xor and carry into addition" {
    try expectXorCarry(true);
    try expectXorCarry(false);
}

test "egraph folds xor combinations of matching bitwise terms" {
    try expectBitwisePair(.bit_xor, .bit_and, .bit_or, .bit_xor);
    try expectBitwisePair(.bit_xor, .bit_or, .bit_xor, .bit_and);
    try expectBitwisePair(.bit_xor, .bit_and, .bit_xor, .bit_or);
}

test "egraph folds and-or combinations of matching bitwise terms" {
    try expectBitwisePair(.bit_or, .bit_and, .bit_xor, .bit_or);
    try expectBitwisePair(.bit_or, .bit_and, .bit_or, .bit_or);
    try expectBitwisePair(.bit_or, .bit_or, .bit_xor, .bit_or);
    try expectBitwisePair(.bit_and, .bit_and, .bit_or, .bit_and);
    try expectBitwisePair(.bit_and, .bit_or, .bit_xor, .bit_xor);
}

test "egraph exposes zero from disjoint bitwise terms" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const exclusive = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = b, .rhs = a } });
    const disjoint = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = both, .rhs = exclusive } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(disjoint) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const zero = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.bit_xor, zero.op);
    try std.testing.expectEqual(zero.lhs, zero.rhs);
}

test "egraph folds mixed addition and subtraction pairs" {
    try expectMixedAddSub(.add, true, true);
    try expectMixedAddSub(.add, false, true);
    try expectMixedAddSub(.sub, true, false);
}

test "egraph cancels cross terms between addition and subtraction" {
    try expectCrossCancellation(.add, true);
    try expectCrossCancellation(.add, false);
    try expectCrossCancellation(.sub, false);
}

test "egraph cancels between added subtractions" {
    try expectAddedSubtractionCancellation(true);
    try expectAddedSubtractionCancellation(false);
}

test "egraph reassociates an add tree to expose factoring" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const d = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const partial = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = d } });
    const total = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = partial, .rhs = ac } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(total) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const outer = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.add, outer.op);
    const product_value = if (outer.lhs == d) outer.rhs else outer.lhs;
    try std.testing.expect(outer.lhs == d or outer.rhs == d);
    const product = func.opcode(func.definingInst(product_value).?).arith;
    try std.testing.expectEqual(BinOp.mul, product.op);
    try std.testing.expect(product.lhs == a or product.rhs == a);
}

test "egraph reassociates products to expose a shared multiplier" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const c = try func.appendBlockParam(block, ty);
    const x = try func.appendBlockParam(block, ty);
    const y = try func.appendBlockParam(block, ty);
    const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
    const abx = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = ab, .rhs = x } });
    const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
    const acy = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = ac, .rhs = y } });
    const total = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = abx, .rhs = acy } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(total) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    const replacement = func.terminator(block).?.ret.values[0];
    const product = func.opcode(func.definingInst(replacement).?).arith;
    try std.testing.expectEqual(BinOp.mul, product.op);
    try std.testing.expect(product.lhs == a or product.rhs == a);
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

test "egraph removes a repeated nested and operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const both = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } });
    const repeated = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = both } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(repeated) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(both, func.terminator(block).?.ret.values[0]);
}

test "egraph removes a repeated nested or operand" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const a = try func.appendBlockParam(block, ty);
    const b = try func.appendBlockParam(block, ty);
    const either = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } });
    const repeated = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = either, .rhs = b } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(repeated) });

    try std.testing.expect(try runWithLimits(allocator, &func, .{}));
    try std.testing.expectEqual(either, func.terminator(block).?.ret.values[0]);
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
