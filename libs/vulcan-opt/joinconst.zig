//! Known-value join parameters. A block J that uses a value `x` and has several predecessors gets
//! a new parameter for `x`. Each predecessor that is only reached through the true edge of
//! `x == k` passes the constant `k`. Every other predecessor passes `x`. Then J and the blocks it
//! dominates read the parameter instead of `x`.
//!
//! This is what lets jump threading remove a dispatch on a merged constant. After an inlined
//! `classify` returns a constant into a chain of `kind == k` tests, the code after the arms still
//! reads `kind`. That use is below the chain, so threading an edge past the chain would leave it
//! without a definition. With the parameter, the code below the arms no longer reads `kind`.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const cfg_mod = @import("cfg.zig");
const dom = @import("dominators.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "joinconst", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const graph = try analyses.cfg();
    const doms = try analyses.dominators();
    const def_block = try analyses.definingBlocks();
    var used: std.ArrayList(Value) = .empty;
    defer used.deinit(allocator);
    for (0..func.blockCount()) |ji| {
        if (!doms.isReachable(ji) or graph.predecessors(ji).len < 2) continue;
        const join: Block = @fromBackingInt(@intCast(ji));
        used.clearRetainingCapacity();
        try collectUsed(allocator, func, join, &used);
        for (used.items) |x| {
            if (@backingInt(x) >= def_block.len or def_block[@backingInt(x)] == ji) continue;
            if (func.types.type_kind(func.valueType(x)) != .int) continue;
            if (!anyKnown(func, graph, doms, def_block, x, ji)) continue;
            try rewrite(allocator, func, graph, doms, def_block, x, join);
            // One rewrite per run. The pass manager runs it again with fresh analyses.
            analyses.invalidate();
            return true;
        }
    }
    return false;
}

fn collectUsed(allocator: std.mem.Allocator, func: *const Function, block: Block, out: *std.ArrayList(Value)) pass.Error!void {
    const Ctx = struct {
        list: *std.ArrayList(Value),
        allocator: std.mem.Allocator,
        failed: bool = false,
        fn add(self: *@This(), v: Value) void {
            for (self.list.items) |seen| if (seen == v) return;
            self.list.append(self.allocator, v) catch {
                self.failed = true;
            };
        }
    };
    var ctx: Ctx = .{ .list = out, .allocator = allocator };
    for (func.blockInsts(block)) |inst| {
        var buffer: [16]Value = undefined;
        for (operandsOf(func, inst, &buffer)) |v| ctx.add(v);
    }
    if (func.terminator(block)) |term| switch (term) {
        .jump => |j| for (func.blockArgs(j)) |v| ctx.add(v),
        .ret => |r| for (r.slice()) |v| ctx.add(v),
    };
    if (ctx.failed) return error.OutOfMemory;
}

/// The plain value operands of `inst`, for the integer instructions this pass looks at. Other
/// instructions report none, so their uses are never a reason to rewrite.
fn operandsOf(func: *const Function, inst: Inst, buffer: *[16]Value) []const Value {
    return switch (func.opcode(inst)) {
        .arith => |a| blk: {
            buffer[0] = a.lhs;
            buffer[1] = a.rhs;
            break :blk buffer[0..2];
        },
        .arith_imm => |a| blk: {
            buffer[0] = a.lhs;
            break :blk buffer[0..1];
        },
        .convert => |c| blk: {
            buffer[0] = c.value;
            break :blk buffer[0..1];
        },
        .icmp => |c| blk: {
            buffer[0] = c.lhs;
            buffer[1] = c.rhs;
            break :blk buffer[0..2];
        },
        else => buffer[0..0],
    };
}

fn anyKnown(func: *const Function, graph: *const cfg_mod.Cfg, doms: *const dom.Dominators, def_block: []const u32, x: Value, ji: usize) bool {
    for (graph.predecessors(ji)) |p| {
        if (knownAt(func, graph, doms, def_block, x, p) != null) return true;
    }
    return false;
}

/// The constant `x` holds at the end of block `p`, when `p` is only reached through the true edge
/// of `x == k` (or the false edge of `x != k`).
fn knownAt(func: *const Function, graph: *const cfg_mod.Cfg, doms: *const dom.Dominators, def_block: []const u32, x: Value, p: u32) ?i64 {
    if (!doms.isReachable(p)) return null;
    const top = def_block[@backingInt(x)];
    var t: u32 = p;
    while (true) {
        if (guardOf(func, graph, doms, x, t)) |k| return k;
        if (t == top or t == 0) return null;
        const up = doms.immediateDominator(t);
        if (up == t) return null;
        t = up;
    }
}

/// The constant `x` equals on entry to `t`, when `t` has a single entry edge that tests it.
/// Other predecessors of `t` are back edges, which `t` dominates.
fn guardOf(func: *const Function, graph: *const cfg_mod.Cfg, doms: *const dom.Dominators, x: Value, t: u32) ?i64 {
    var entry: ?u32 = null;
    for (graph.predecessors(t)) |c| {
        if (doms.dominates(t, c)) continue;
        if (entry != null and entry.? != c) return null;
        entry = c;
    }
    const c = entry orelse return null;
    const insts = func.blockInsts(@fromBackingInt(@intCast(c)));
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |f| f,
        else => return null,
    };
    const then_hit = @backingInt(cf.then.target) == t;
    const else_hit = @backingInt(cf.@"else".target) == t;
    if (then_hit == else_hit) return null;
    const cmp_inst = func.definingInst(cf.cond) orelse return null;
    const cmp = switch (func.opcode(cmp_inst)) {
        .icmp => |m| m,
        else => return null,
    };
    const want_true = switch (cmp.op) {
        .eq => true,
        .ne => false,
        else => return null,
    };
    if (then_hit != want_true) return null;
    const other = if (cmp.lhs == x) cmp.rhs else if (cmp.rhs == x) cmp.lhs else return null;
    const k_inst = func.definingInst(other) orelse return null;
    return switch (func.opcode(k_inst)) {
        .iconst => |k| k,
        else => null,
    };
}

fn rewrite(
    allocator: std.mem.Allocator,
    func: *Function,
    graph: *const cfg_mod.Cfg,
    doms: *const dom.Dominators,
    def_block: []const u32,
    x: Value,
    join: Block,
) pass.Error!void {
    const ji = @backingInt(join);
    const ty = func.valueType(x);

    // Every block with an edge into the join passes an argument, a dead one included. The
    // constants go first in the entry block, which dominates every predecessor.
    var sources: std.ArrayList(u32) = .empty;
    defer sources.deinit(allocator);
    var args: std.ArrayList(Value) = .empty;
    defer args.deinit(allocator);
    var consts: std.ArrayList(Inst) = .empty;
    defer consts.deinit(allocator);
    for (0..func.blockCount()) |bi| {
        if (!hasEdgeTo(func, @fromBackingInt(@intCast(bi)), join)) continue;
        const known = if (bi < doms.n) knownAt(func, graph, doms, def_block, x, @intCast(bi)) else null;
        const arg = if (known) |k| blk: {
            const value = try func.createInst(ty, .{ .iconst = k });
            try consts.append(allocator, func.definingInst(value).?);
            break :blk value;
        } else x;
        try sources.append(allocator, @intCast(bi));
        try args.append(allocator, arg);
    }
    const entry: Block = @fromBackingInt(0);
    var entry_insts: std.ArrayList(Inst) = .empty;
    defer entry_insts.deinit(allocator);
    try entry_insts.appendSlice(allocator, consts.items);
    try entry_insts.appendSlice(allocator, func.blockInsts(entry));
    try func.setBlockInsts(entry, entry_insts.items);

    const param = try func.appendBlockParam(join, ty);
    for (sources.items, args.items) |source, arg| {
        try appendArgOnEdges(allocator, func, @fromBackingInt(@intCast(source)), join, arg);
    }

    for (0..func.blockCount()) |bi| {
        if (bi < doms.n and doms.isReachable(bi) and doms.dominates(ji, bi)) {
            func.replaceUsesIn(@fromBackingInt(@intCast(bi)), x, param);
        }
    }
}

fn hasEdgeTo(func: *const Function, from: Block, to: Block) bool {
    if (func.terminator(from)) |term| switch (term) {
        .jump => |j| if (j.target == to) return true,
        .ret => {},
    };
    for (func.blockInsts(from)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        const cf = func.opcode(inst).@"if";
        if (cf.then.target == to or cf.@"else".target == to) return true;
    }
    return false;
}

/// Append `arg` to every edge from `from` to `to`.
fn appendArgOnEdges(allocator: std.mem.Allocator, func: *Function, from: Block, to: Block, arg: Value) pass.Error!void {
    var list: std.ArrayList(Value) = .empty;
    defer list.deinit(allocator);
    if (func.terminator(from)) |term| switch (term) {
        .jump => |j| if (j.target == to) {
            list.clearRetainingCapacity();
            try list.appendSlice(allocator, func.blockArgs(j));
            try list.append(allocator, arg);
            const interned = try func.internValues(list.items);
            func.terminatorPtr(from).*.?.jump.args = interned;
        },
        .ret => {},
    };
    for (func.blockInsts(from)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        const cf = func.opcode(inst).@"if";
        if (cf.then.target == to) {
            list.clearRetainingCapacity();
            try list.appendSlice(allocator, func.blockArgs(cf.then));
            try list.append(allocator, arg);
            const interned = try func.internValues(list.items);
            func.opcodeMut(inst).@"if".then.args = interned;
        }
        if (cf.@"else".target == to) {
            list.clearRetainingCapacity();
            try list.appendSlice(allocator, func.blockArgs(cf.@"else"));
            try list.append(allocator, arg);
            const interned = try func.internValues(list.items);
            func.opcodeMut(inst).@"if".@"else".args = interned;
        }
    }
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

test "a join after an equality dispatch reads a parameter with the known constants" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const second = try func.appendBlock();
    const zero_arm = try func.appendBlock();
    const one_arm = try func.appendBlock();
    const other_arm = try func.appendBlock();
    const join = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const k0 = try func.appendInst(entry, t, .{ .iconst = 0 });
    const is0 = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = k0 } });
    try func.appendIf(entry, is0, .{ .target = zero_arm }, .{ .target = second });
    const k1 = try func.appendInst(second, t, .{ .iconst = 1 });
    const is1 = try func.appendInst(second, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = k1 } });
    try func.appendIf(second, is1, .{ .target = one_arm }, .{ .target = other_arm });
    try func.setJump(zero_arm, join, &.{});
    try func.setJump(one_arm, join, &.{});
    try func.setJump(other_arm, join, &.{});
    const y = try func.appendArithImm(join, t, .add, x, 5);
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(try runOnce(allocator, &func));
    const params = func.blockParams(join);
    try testing.expectEqual(@as(usize, 1), params.len);
    try testing.expectEqual(params[0], func.opcode(func.definingInst(y).?).arith_imm.lhs);
    const zero_arg = func.blockArgs(func.terminator(zero_arm).?.jump)[0];
    try testing.expectEqual(@as(i64, 0), func.opcode(func.definingInst(zero_arg).?).iconst);
    const one_arg = func.blockArgs(func.terminator(one_arm).?.jump)[0];
    try testing.expectEqual(@as(i64, 1), func.opcode(func.definingInst(one_arg).?).iconst);
    try testing.expectEqual(x, func.blockArgs(func.terminator(other_arm).?.jump)[0]);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a join with no guarded predecessor is left alone" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const join = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const k = try func.appendInst(entry, t, .{ .iconst = 4 });
    const less = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = k } });
    try func.appendIf(entry, less, .{ .target = left }, .{ .target = right });
    try func.setJump(left, join, &.{});
    try func.setJump(right, join, &.{});
    const y = try func.appendArithImm(join, t, .add, x, 5);
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(!try runOnce(allocator, &func));
}
