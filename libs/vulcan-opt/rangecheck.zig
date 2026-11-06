//! Range-check fusion: two chained compares of one integer against constants, which together
//! accept `lo <= x <= hi`, become one unsigned compare `(x - lo) <= (hi - lo)`.
//!
//! Block A ends in `if x op k1`. One edge goes to block B with no arguments. B has only A as its
//! predecessor, and holds only constants and `if x op k2`. The other edge of A and one edge of B
//! go to the same block F with the same arguments. The edge of B that does not go to F goes to T.
//! After the change, A branches to T or F with one compare, and B is dead. This is the shape a
//! `&&` or `||` of two bounds lowers to, for example `c >= 'a' && c <= 'z'`.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Jump = ir.function.Jump;

pub const pass_def = pass.Pass{ .name = "rangecheck", .run = run };

const Interval = struct { lo: i128, hi: i128 };

const Bound = struct {
    x: Value,
    signed: bool,
    bits: u16,
    then_set: Interval,
    else_set: Interval,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const graph = try analyses.cfg();
    var changed = false;
    for (0..func.blockCount()) |ai| {
        const a: Block = @fromBackingInt(@intCast(ai));
        if (try fuse(allocator, func, graph, a)) changed = true;
        if (!changed and try fuseBitRanges(allocator, func, graph, a)) {
            analyses.invalidate();
            return true;
        }
    }
    if (changed) analyses.invalidate();
    return changed;
}

const BitRange = struct { x: Value, offset: Inst, width_inst: Inst, cmp: Inst, lo: u64, width: u64, bits: u16 };

fn bitRange(func: *const Function, cond: Value) ?BitRange {
    const cmp_inst = func.definingInst(cond) orelse return null;
    const cmp = switch (func.opcode(cmp_inst)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .le) return null;
    const int = switch (func.types.type_kind(func.valueType(cmp.lhs))) {
        .int => |i| i,
        else => return null,
    };
    if (int.signedness != .unsigned or int.bits == 0 or int.bits > 64) return null;
    const offset = func.definingInst(cmp.lhs) orelse return null;
    const sub = switch (func.opcode(offset)) {
        .arith_imm => |a| a,
        else => return null,
    };
    if (sub.op != .sub or sub.imm < 0) return null;
    const width_inst = func.definingInst(cmp.rhs) orelse return null;
    const width = switch (func.opcode(width_inst)) {
        .iconst => |k| k,
        else => return null,
    };
    if (width < 0) return null;
    const lo: u64 = @intCast(sub.imm);
    const w: u64 = @intCast(width);
    const max = if (int.bits == 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(int.bits)) - 1;
    if (lo > max or w > max - lo) return null;
    return .{ .x = sub.lhs, .offset = offset, .width_inst = width_inst, .cmp = cmp_inst, .lo = lo, .width = w, .bits = int.bits };
}

/// Two equal unsigned ranges differing in one bit can clear that bit before
/// one range check. For example [A,Z] or [a,z] becomes (x & ~32)-'A' <= 25.
fn fuseBitRanges(allocator: std.mem.Allocator, func: *Function, graph: anytype, a: Block) pass.Error!bool {
    const ai = lastIf(func, a) orelse return false;
    const acf = func.opcode(ai).@"if";
    const ar = bitRange(func, acf.cond) orelse return false;
    const b = acf.@"else".target;
    if (b == a or b == acf.then.target or func.blockParams(b).len != 0 or func.blockArgs(acf.@"else").len != 0) return false;
    const preds = graph.predecessors(@backingInt(b));
    if (preds.len != 1 or preds[0] != @backingInt(a)) return false;
    const bi = lastIf(func, b) orelse return false;
    const bcf = func.opcode(bi).@"if";
    if (bcf.then.target != acf.then.target or !sameArgs(func, acf.then, bcf.then)) return false;
    const br = bitRange(func, bcf.cond) orelse return false;
    if (ar.x != br.x or ar.bits != br.bits or ar.width != br.width) return false;
    const bit = ar.lo ^ br.lo;
    if (bit == 0 or !std.math.isPowerOfTwo(bit)) return false;
    const lo = @min(ar.lo, br.lo);
    // Neither interval may cross the bit being cleared.
    if (ar.width >= bit or (lo & (bit - 1)) > bit - 1 - ar.width) return false;
    for (func.blockInsts(b)) |inst| {
        if (inst != bi and inst != br.offset and inst != br.width_inst and inst != br.cmp) return false;
    }
    const yes = try allocator.dupe(Value, func.blockArgs(acf.then));
    defer allocator.free(yes);
    const no = try allocator.dupe(Value, func.blockArgs(bcf.@"else"));
    defer allocator.free(no);
    var rebuilt: std.ArrayList(Inst) = .empty;
    defer rebuilt.deinit(allocator);
    for (func.blockInsts(a)) |inst| if (inst != ai) try rebuilt.append(allocator, inst);
    // Retain any independently used pure definitions from the bypassed block.
    for (func.blockInsts(b)) |inst| if (inst != bi) try rebuilt.append(allocator, inst);
    try func.setBlockInsts(a, rebuilt.items);
    try func.setBlockInsts(b, &.{});
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    const ty = func.valueType(ar.x);
    const mask = try func.appendInst(a, ty, .{ .iconst = @bitCast(~bit) });
    const folded = try func.appendInst(a, ty, .{ .arith = .{ .op = .bit_and, .lhs = ar.x, .rhs = mask } });
    const offset = try func.appendArithImm(a, ty, .sub, folded, @intCast(lo));
    const width = try func.appendInst(a, ty, .{ .iconst = @intCast(ar.width) });
    const cond = try func.appendInst(a, func.valueType(acf.cond), .{ .icmp = .{ .op = .le, .lhs = offset, .rhs = width } });
    try func.appendIf(a, cond, .{ .target = acf.then.target, .args = yes }, .{ .target = bcf.@"else".target, .args = no });
    return true;
}

fn fuse(allocator: std.mem.Allocator, func: *Function, graph: anytype, a: Block) pass.Error!bool {
    const a_if = lastIf(func, a) orelse return false;
    const a_cf = func.opcode(a_if).@"if";
    const a_bound = bound(func, a_cf.cond) orelse return false;

    for ([_]bool{ true, false }) |inner_is_then| {
        const to_b = if (inner_is_then) a_cf.then else a_cf.@"else";
        const to_f = if (inner_is_then) a_cf.@"else" else a_cf.then;
        const b = to_b.target;
        if (b == a or b == to_f.target) continue;
        if (func.blockArgs(to_b).len != 0 or func.blockParams(b).len != 0) continue;
        const preds = graph.predecessors(@backingInt(b));
        if (preds.len != 1 or preds[0] != @backingInt(a)) continue;

        const b_insts = func.blockInsts(b);
        if (b_insts.len < 2) continue;
        const b_if = b_insts[b_insts.len - 1];
        if (func.opcode(b_if) != .@"if") continue;
        const b_cf = func.opcode(b_if).@"if";
        const cmp_inst = func.definingInst(b_cf.cond) orelse continue;
        var only_consts = true;
        for (b_insts[0 .. b_insts.len - 1]) |inst| {
            if (inst == cmp_inst) continue;
            if (func.opcode(inst) != .iconst) only_consts = false;
        }
        if (!only_consts) continue;
        if (std.mem.indexOfScalar(Inst, b_insts, cmp_inst) == null) continue;
        const b_bound = bound(func, b_cf.cond) orelse continue;
        if (b_bound.x != a_bound.x) continue;

        const t_is_then = if (b_cf.@"else".target == to_f.target and sameArgs(func, b_cf.@"else", to_f))
            true
        else if (b_cf.then.target == to_f.target and sameArgs(func, b_cf.then, to_f))
            false
        else
            continue;
        const to_t = if (t_is_then) b_cf.then else b_cf.@"else";
        if (to_t.target == b) continue;
        if (std.mem.indexOfScalar(Value, func.blockArgs(to_t), b_cf.cond) != null) continue;

        const outer = if (inner_is_then) a_bound.then_set else a_bound.else_set;
        const inner = if (t_is_then) b_bound.then_set else b_bound.else_set;
        const lo = @max(outer.lo, inner.lo);
        const hi = @min(outer.hi, inner.hi);
        const full = typeRange(a_bound.signed, a_bound.bits);
        if (lo > hi or lo == full.lo or hi == full.hi) continue;

        try rewrite(allocator, func, a, a_if, b, a_bound, lo, hi, to_t, to_f);
        return true;
    }
    return false;
}

fn rewrite(
    allocator: std.mem.Allocator,
    func: *Function,
    a: Block,
    a_if: Inst,
    b: Block,
    x_bound: Bound,
    lo: i128,
    hi: i128,
    to_t: Jump,
    to_f: Jump,
) pass.Error!void {
    const t_args = try allocator.dupe(Value, func.blockArgs(to_t));
    defer allocator.free(t_args);
    const f_args = try allocator.dupe(Value, func.blockArgs(to_f));
    defer allocator.free(f_args);

    // B's constants move into A, so a use of one below T still has a dominating definition.
    var insts: std.ArrayList(Inst) = .empty;
    defer insts.deinit(allocator);
    for (func.blockInsts(a)) |inst| if (inst != a_if) try insts.append(allocator, inst);
    for (func.blockInsts(b)) |inst| if (func.opcode(inst) == .iconst) try insts.append(allocator, inst);
    try func.setBlockInsts(a, insts.items);
    try func.setBlockInsts(b, &.{});
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    const unsigned_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = x_bound.bits } });
    const bool_ty = try func.types.intern(.bool);
    const x = if (x_bound.signed)
        try func.appendInst(a, unsigned_ty, .{ .convert = .{ .value = x_bound.x } })
    else
        x_bound.x;
    const lo_bits: i64 = @truncate(lo);
    const offset = try func.appendArithImm(a, unsigned_ty, .sub, x, lo_bits);
    const width = try func.appendInst(a, unsigned_ty, .{ .iconst = @as(i64, @truncate(hi - lo)) });
    const inside = try func.appendInst(a, bool_ty, .{ .icmp = .{ .op = .le, .lhs = offset, .rhs = width } });
    try func.appendIf(a, inside, .{ .target = to_t.target, .args = t_args }, .{ .target = to_f.target, .args = f_args });
}

fn lastIf(func: *const Function, block: Block) ?Inst {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const last = insts[insts.len - 1];
    return if (func.opcode(last) == .@"if") last else null;
}

fn sameArgs(func: *const Function, x: Jump, y: Jump) bool {
    return std.mem.eql(Value, func.blockArgs(x), func.blockArgs(y));
}

fn typeRange(signed: bool, bits: u16) Interval {
    if (signed) {
        const half = @as(i128, 1) << @intCast(bits - 1);
        return .{ .lo = -half, .hi = half - 1 };
    }
    return .{ .lo = 0, .hi = (@as(i128, 1) << @intCast(bits)) - 1 };
}

/// The sets of `x` that take each edge of `if (x op k)`, for an ordering compare against a
/// constant. Returns null for any other condition.
fn bound(func: *const Function, cond: Value) ?Bound {
    const inst = func.definingInst(cond) orelse return null;
    const cmp = switch (func.opcode(inst)) {
        .icmp => |c| c,
        else => return null,
    };
    const int = switch (func.types.type_kind(func.valueType(cmp.lhs))) {
        .int => |i| i,
        else => return null,
    };
    if (int.bits == 0 or int.bits > 64) return null;
    const k_inst = func.definingInst(cmp.rhs) orelse return null;
    const raw = switch (func.opcode(k_inst)) {
        .iconst => |c| c,
        else => return null,
    };
    const signed = int.signedness == .signed;
    const shift: u7 = @intCast(64 - @as(u32, int.bits));
    const k: i128 = if (signed)
        @as(i64, @bitCast(@as(u64, @bitCast(raw)) << @intCast(shift))) >> @intCast(shift)
    else
        (@as(u64, @bitCast(raw)) << @intCast(shift)) >> @intCast(shift);
    const full = typeRange(signed, int.bits);
    const below: Interval = .{ .lo = full.lo, .hi = k - 1 };
    const at_most: Interval = .{ .lo = full.lo, .hi = k };
    const above: Interval = .{ .lo = k + 1, .hi = full.hi };
    const at_least: Interval = .{ .lo = k, .hi = full.hi };
    const sets: [2]Interval = switch (cmp.op) {
        .lt => .{ below, at_least },
        .le => .{ at_most, above },
        .gt => .{ above, at_most },
        .ge => .{ at_least, below },
        .eq, .ne => return null,
    };
    return .{ .x = cmp.lhs, .signed = signed, .bits = int.bits, .then_set = sets[0], .else_set = sets[1] };
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

/// Builds `f(x) = (x >= lo && x <= hi) ? 1 : 0` as two chained blocks, with the
/// `||` form when `either` is set: `(x < lo || x > hi) ? 0 : 1`.
fn buildChain(func: *Function, signed: bool, bits: u16, lo: i64, hi: i64, either: bool) !struct { entry: Block, inner: Block, yes: Block, no: Block } {
    const t = try func.types.intern(.{ .int = .{ .signedness = if (signed) .signed else .unsigned, .bits = bits } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const inner = try func.appendBlock();
    const yes = try func.appendBlock();
    const no = try func.appendBlock();
    const x = try func.appendBlockParam(entry, t);
    const k_lo = try func.appendInst(entry, t, .{ .iconst = lo });
    if (either) {
        const c1 = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = k_lo } });
        try func.appendIf(entry, c1, .{ .target = no }, .{ .target = inner });
        const k_hi = try func.appendInst(inner, t, .{ .iconst = hi });
        const c2 = try func.appendInst(inner, bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = k_hi } });
        try func.appendIf(inner, c2, .{ .target = no }, .{ .target = yes });
    } else {
        const c1 = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ge, .lhs = x, .rhs = k_lo } });
        try func.appendIf(entry, c1, .{ .target = inner }, .{ .target = no });
        const k_hi = try func.appendInst(inner, t, .{ .iconst = hi });
        const c2 = try func.appendInst(inner, bool_t, .{ .icmp = .{ .op = .le, .lhs = x, .rhs = k_hi } });
        try func.appendIf(inner, c2, .{ .target = yes }, .{ .target = no });
    }
    const one = try func.appendInst(yes, t, .{ .iconst = 1 });
    func.setTerminator(yes, .{ .ret = ir.function.Ret.one(one) });
    const zero = try func.appendInst(no, t, .{ .iconst = 0 });
    func.setTerminator(no, .{ .ret = ir.function.Ret.one(zero) });
    return .{ .entry = entry, .inner = inner, .yes = yes, .no = no };
}

fn expectFused(func: *const Function, entry: Block, yes: Block, no: Block, lo: i64, width: i64) !void {
    const cf = func.opcode(lastIf(func, entry).?).@"if";
    try testing.expectEqual(yes, cf.then.target);
    try testing.expectEqual(no, cf.@"else".target);
    const cmp = func.opcode(func.definingInst(cf.cond).?).icmp;
    try testing.expectEqual(ir.function.CmpOp.le, cmp.op);
    try testing.expectEqual(width, func.opcode(func.definingInst(cmp.rhs).?).iconst);
    const sub = func.opcode(func.definingInst(cmp.lhs).?).arith_imm;
    try testing.expectEqual(lo, sub.imm);
    const kind = func.types.type_kind(func.valueType(cmp.lhs)).int;
    try testing.expectEqual(std.builtin.Signedness.unsigned, kind.signedness);
}

test "a signed && range check becomes one unsigned compare" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const chain = try buildChain(&func, true, 32, 'a', 'z', false);
    try testing.expect(try runOnce(allocator, &func));
    try expectFused(&func, chain.entry, chain.yes, chain.no, 'a', 'z' - 'a');
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "an unsigned || range check becomes one unsigned compare" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const chain = try buildChain(&func, false, 8, '0', '9', true);
    try testing.expect(try runOnce(allocator, &func));
    try expectFused(&func, chain.entry, chain.yes, chain.no, '0', 9);
}

test "a negative signed lower bound wraps into the unsigned offset" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const chain = try buildChain(&func, true, 32, -5, 5, false);
    try testing.expect(try runOnce(allocator, &func));
    try expectFused(&func, chain.entry, chain.yes, chain.no, -5, 10);
}

test "an empty or one-sided range is left alone" {
    const allocator = testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        _ = try buildChain(&func, true, 32, 10, 5, false);
        try testing.expect(!try runOnce(allocator, &func));
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        _ = try buildChain(&func, false, 8, 0, 9, false);
        try testing.expect(!try runOnce(allocator, &func));
    }
}

test "an inner block with other work is left alone" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const chain = try buildChain(&func, true, 32, 'a', 'z', false);
    const t = func.valueType(func.blockParams(chain.entry)[0]);
    const insts = try allocator.dupe(Inst, func.blockInsts(chain.inner));
    defer allocator.free(insts);
    _ = try func.appendArithImm(chain.inner, t, .add, func.blockParams(chain.entry)[0], 1);
    const with_work = try allocator.alloc(Inst, insts.len + 1);
    defer allocator.free(with_work);
    with_work[0] = func.blockInsts(chain.inner)[insts.len];
    @memcpy(with_work[1..], insts);
    try func.setBlockInsts(chain.inner, with_work);
    try testing.expect(!try runOnce(allocator, &func));
}

test "one-bit range unions match every byte and reject crossing ranges" {
    const allocator = testing.allocator;
    for ([_]struct { lo: i64, width: i64, bit: i64, accepts: bool }{
        .{ .lo = 65, .width = 25, .bit = 32, .accepts = true },
        .{ .lo = 4, .width = 3, .bit = 8, .accepts = true },
        .{ .lo = 3, .width = 6, .bit = 8, .accepts = false },
        .{ .lo = 1, .width = 7, .bit = 8, .accepts = false },
    }) |case| {
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const bt = try f.types.intern(.bool);
        const a = try f.appendBlock();
        const b = try f.appendBlock();
        const yes = try f.appendBlock();
        const no = try f.appendBlock();
        const x = try f.appendBlockParam(a, t);
        for ([_]Block{ a, b }, 0..) |block, index| {
            const off = try f.appendArithImm(block, t, .sub, x, case.lo + if (index == 0) @as(i64, 0) else case.bit);
            const w = try f.appendInst(block, t, .{ .iconst = case.width });
            const cmp = try f.appendInst(block, bt, .{ .icmp = .{ .op = .le, .lhs = off, .rhs = w } });
            try f.appendIf(block, cmp, .{ .target = yes }, .{ .target = if (index == 0) b else no });
            f.setTerminator(block, .{ .ret = ir.function.Ret.none() });
        }
        const one = try f.appendInst(yes, t, .{ .iconst = 1 });
        const zero = try f.appendInst(no, t, .{ .iconst = 0 });
        f.setTerminator(yes, .{ .ret = ir.function.Ret.one(one) });
        f.setTerminator(no, .{ .ret = ir.function.Ret.one(zero) });
        try testing.expectEqual(case.accepts, try runOnce(allocator, &f));
        if (!case.accepts) continue;
        var diags = try ir.verify.verify(allocator, &f, .high);
        defer diags.deinit();
        try testing.expect(diags.ok());
        const values = try allocator.alloc(u8, f.valueCount());
        defer allocator.free(values);
        for (0..256) |input| {
            values[@backingInt(x)] = @intCast(input);
            for (f.blockInsts(a)) |inst| {
                const result = f.instResult(inst) orelse continue;
                values[@backingInt(result)] = switch (f.opcode(inst)) {
                    .iconst => |k| @truncate(@as(u64, @bitCast(k))),
                    .arith_imm => |ar| values[@backingInt(ar.lhs)] -% @as(u8, @truncate(@as(u64, @bitCast(ar.imm)))),
                    .arith => |ar| values[@backingInt(ar.lhs)] & values[@backingInt(ar.rhs)],
                    .icmp => |cmp| @intFromBool(values[@backingInt(cmp.lhs)] <= values[@backingInt(cmp.rhs)]),
                    else => unreachable,
                };
            }
            const iff = f.opcode(lastIf(&f, a).?).@"if";
            const n: i64 = @intCast(input);
            const expected = (n >= case.lo and n <= case.lo + case.width) or (n >= case.lo + case.bit and n <= case.lo + case.bit + case.width);
            try testing.expectEqual(expected, values[@backingInt(iff.cond)] != 0);
        }
    }
}
