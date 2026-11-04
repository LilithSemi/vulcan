//! A loop that steps a signed counter `j` by one and a pointer `q` by one element, with
//! `q == base + j*size`, tests and indexes through `q` alone. The loop test `j op k` becomes
//! `q op base + k*size`, an address after the loop built from `j` becomes `q` plus a constant, and
//! `q` starts from an outer loop's pointer when `j` starts from that loop's counter. Once only its
//! own step reads `j`, the counter goes away.
//!
//! Signed overflow is undefined in C, so `j` does not wrap between its start and the test that
//! ends the loop, and `q` only feeds loads and stores, so an address outside the object never
//! happens in a valid program. Both make the pointer forms exact.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");
const loops_mod = @import("loops.zig");
const dominators = @import("dominators.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "lftr", .run = run };

/// `q == base + j << shift` on every trip, with `j` stepping by `step`.
const Pair = struct {
    header: Block,
    preheader: Block,
    latch: Block,
    j: usize,
    q: usize,
    base: Value,
    shift: u6,
    step: i64,
    /// The start of `q` is `base + sext(j0) << shift` exactly.
    exact: bool,
    /// Only loads, stores and steps read `q`, so its start may be rebuilt.
    address_only: bool,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    for (func.attributeEntries()) |entry| if (entry.target == .block) return false;
    const loop_info = try analyses.loops();
    const doms = try analyses.dominators();
    const cfg = try analyses.cfg();
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    dce.countUses(func, uses);

    // A loop whose pointer starts from an outer pair needs that pair first.
    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(allocator);
    const found = try allocator.alloc(bool, loop_info.loops.len);
    defer allocator.free(found);
    @memset(found, false);
    var grew = true;
    while (grew) {
        grew = false;
        for (loop_info.loops, found) |*loop, *done| {
            if (done.*) continue;
            const pair = findPair(func, cfg, loop, uses, pairs.items) orelse continue;
            try pairs.append(allocator, pair);
            done.* = true;
            grew = true;
        }
    }
    var changed = false;
    for (pairs.items) |pair| {
        if (try rewrite(allocator, func, doms, pairs.items, pair)) {
            changed = true;
            break;
        }
    }
    if (!changed) {
        for (pairs.items) |pair| {
            if (try dropCounter(allocator, func, cfg, pair, uses)) {
                changed = true;
                break;
            }
        }
    }
    if (changed) analyses.invalidate();
    return changed;
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        else => null,
    };
}

/// `v` as `x + c` for a constant `c`.
fn plusConst(func: *const Function, v: Value) ?struct { x: Value, c: i64 } {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .arith_imm => |a| switch (a.op) {
            .add => .{ .x = a.lhs, .c = a.imm },
            .sub => .{ .x = a.lhs, .c = -%a.imm },
            else => null,
        },
        .arith => |a| switch (a.op) {
            .add => if (constOf(func, a.rhs)) |c| .{ .x = a.lhs, .c = c } else if (constOf(func, a.lhs)) |c| .{ .x = a.rhs, .c = c } else null,
            .sub => if (constOf(func, a.rhs)) |c| .{ .x = a.lhs, .c = -%c } else null,
            else => null,
        },
        else => null,
    };
}

/// `v` as `x << shift`, from a shift or a multiply by a power of two.
fn scaled(func: *const Function, v: Value) ?struct { x: Value, shift: u6 } {
    const inst = func.definingInst(v) orelse return null;
    const x: Value, const op: ir.function.BinOp, const c: i64 = switch (func.opcode(inst)) {
        .arith_imm => |a| .{ a.lhs, a.op, a.imm },
        .arith => |a| .{ a.lhs, a.op, constOf(func, a.rhs) orelse return null },
        else => return null,
    };
    const amount: i64 = switch (op) {
        .shl => c,
        .mul => if (c > 0 and std.math.isPowerOfTwo(c)) std.math.log2_int(u64, @intCast(c)) else return null,
        else => return null,
    };
    if (amount < 0 or amount > 4) return null;
    return .{ .x = x, .shift = @intCast(amount) };
}

/// The value under a widening convert, or `v` itself.
fn unwiden(func: *const Function, v: Value) Value {
    const inst = func.definingInst(v) orelse return v;
    return switch (func.opcode(inst)) {
        .convert => |c| c.value,
        else => v,
    };
}

fn signedInt(func: *const Function, v: Value) ?u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| if (i.signedness == .signed and i.bits <= 64) i.bits else null,
        else => null,
    };
}

fn isPtr(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .ptr;
}

fn findPair(func: *const Function, cfg: anytype, loop: *const loops_mod.Loop, uses: []const u32, known: []const Pair) ?Pair {
    const header: Block = @fromBackingInt(@intCast(loop.header));
    const pre_index = loop.preheader orelse return null;
    const preheader: Block = @fromBackingInt(@intCast(pre_index));
    var latch: ?Block = null;
    for (cfg.predecessors(loop.header)) |p| {
        if (!loop.contains(p)) continue;
        if (latch != null) return null;
        latch = @fromBackingInt(@intCast(p));
    }
    const back = latch orelse return null;
    const back_term = func.terminator(back) orelse return null;
    if (back_term != .jump) return null;
    const pre_term = func.terminator(preheader) orelse return null;
    if (pre_term != .jump) return null;
    const params = func.blockParams(header);
    const init = func.blockArgs(pre_term.jump);
    const next = func.blockArgs(back_term.jump);

    for (params, 0..) |j, ji| {
        if (signedInt(func, j) == null) continue;
        const jstep = plusConst(func, next[ji]) orelse continue;
        if (jstep.x != j or (jstep.c != 1 and jstep.c != -1)) continue;
        for (params, 0..) |q, qi| {
            if (!isPtr(func, q)) continue;
            const qstep = plusConst(func, next[qi]) orelse continue;
            if (qstep.x != q or qstep.c == 0) continue;
            const size = @abs(qstep.c);
            if (!std.math.isPowerOfTwo(size) or size > 16) continue;
            const shift: u6 = @intCast(std.math.log2_int(u64, size));
            if (qstep.c != jstep.c << shift) continue;
            const start = startOf(func, init[qi], init[ji], shift, known) orelse continue;
            return .{
                .header = header,
                .preheader = preheader,
                .latch = back,
                .j = ji,
                .q = qi,
                .base = start.base,
                .shift = shift,
                .step = jstep.c,
                .exact = start.exact,
                .address_only = addressOnly(func, q, uses, next[qi]),
            };
        }
    }
    return null;
}

/// `q0` as `base + j0 << shift`, and whether that sum is exact in pointer width.
fn startOf(func: *const Function, q0: Value, j0: Value, shift: u6, known: []const Pair) ?struct { base: Value, exact: bool } {
    // Started from an outer pair: `q0 = outer_q + c << shift` with `j0 = outer_j + c`.
    if (plusConst(func, q0)) |pq| {
        const jx: Value, const c: i64 = if (plusConst(func, j0)) |pj| .{ pj.x, pj.c } else .{ j0, 0 };
        for (known) |outer| {
            if (!outer.exact or outer.shift != shift) continue;
            const params = func.blockParams(outer.header);
            if (params[outer.q] == pq.x and params[outer.j] == jx and pq.c == c << shift) return .{ .base = outer.base, .exact = true };
        }
    }
    if (plusConst(func, q0)) |pc| if (constOf(func, j0)) |c| {
        if (isPtr(func, pc.x) and pc.c == c << shift) return .{ .base = pc.x, .exact = true };
    };
    const inst = func.definingInst(q0) orelse return null;
    const a = switch (func.opcode(inst)) {
        .arith => |a| a,
        else => return null,
    };
    if (a.op != .add or !isPtr(func, a.lhs)) return null;
    const s = scaled(func, a.rhs) orelse return null;
    if (s.shift != shift) return null;
    if (s.x == j0) return .{ .base = a.lhs, .exact = (signedInt(func, j0) orelse 0) == 64 };
    if (unwiden(func, s.x) == j0 and s.x != j0) return .{ .base = a.lhs, .exact = (signedInt(func, s.x) orelse 0) == 64 };
    return null;
}

/// Every use of `q` reads or writes through it, steps it, or is `latch_next`.
fn addressOnly(func: *const Function, q: Value, uses: []const u32, latch_next: Value) bool {
    var seen: u32 = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .load => |l| seen += @intFromBool(l.ptr == q),
            .store => |s| {
                if (s.value == q) return false;
                seen += @intFromBool(s.ptr == q);
            },
            // A test this pass made from the counter.
            .icmp => |c| seen += @intFromBool(c.lhs == q) + @intFromBool(c.rhs == q),
            .arith_imm => |a| if (a.lhs == q) {
                if (a.op != .add) return false;
                const r = func.instResult(inst).?;
                if (r != latch_next and !derefOnly(func, r)) return false;
                seen += 1;
            },
            else => {},
        };
    }
    // Edge arguments and anything else count too, so they must all be the latch step.
    return seen + 1 >= uses[@backingInt(q)];
}

/// Every use of `v` is the address of a load or a store.
fn derefOnly(func: *const Function, v: Value) bool {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .load => |l| if (l.ptr == v) continue,
                .store => |s| if (s.ptr == v and s.value != v) continue,
                else => {},
            }
            if (reads(func, inst, v)) return false;
        }
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| for (func.blockArgs(j)) |arg| if (arg == v) return false,
            .ret => |r| for (r.slice()) |arg| if (arg == v) return false,
        };
    }
    return true;
}

fn reads(func: *const Function, inst: Inst, v: Value) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.lhs == v or a.rhs == v,
        .arith_imm => |a| a.lhs == v,
        .icmp => |c| c.lhs == v or c.rhs == v,
        .convert => |c| c.value == v,
        .select => |s| s.cond == v or s.then == v or s.@"else" == v,
        .load => |l| l.ptr == v,
        .store => |s| s.ptr == v or s.value == v,
        .@"if" => |cf| blk: {
            if (cf.cond == v) break :blk true;
            for (func.blockArgs(cf.then)) |arg| if (arg == v) break :blk true;
            for (func.blockArgs(cf.@"else")) |arg| if (arg == v) break :blk true;
            break :blk false;
        },
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => false,
        // Anything else may read it in a way this pass does not follow.
        else => true,
    };
}

fn rewrite(allocator: std.mem.Allocator, func: *Function, doms: *const dominators.Dominators, pairs: []const Pair, pair: Pair) !bool {
    const params = func.blockParams(pair.header);
    const j = params[pair.j];
    const q = params[pair.q];
    var changed = false;

    // The start: from an outer pair, or rebuilt exactly.
    const pre_jump = func.terminator(pair.preheader).?.jump;
    const j0 = func.blockArgs(pre_jump)[pair.j];
    const from_outer = if (outerStart(func, pairs, pair, j0)) |o| !hasForm(func, func.blockArgs(pre_jump)[pair.q], o.q, o.c << pair.shift) else false;
    if (pair.address_only and ((!pair.exact and !compared(func, q)) or from_outer)) {
        const q0 = if (outerStart(func, pairs, pair, j0)) |o| blk: {
            if (o.c == 0) break :blk o.q;
            break :blk try func.appendArithImm(pair.preheader, func.valueType(q), .add, o.q, o.c << pair.shift);
        } else try exactAddress(func, pair.preheader, pair.base, j0, pair.shift, func.valueType(q));
        const args = try allocator.dupe(Value, func.blockArgs(pre_jump));
        defer allocator.free(args);
        if (args[pair.q] != q0) {
            args[pair.q] = q0;
            try func.setJump(pair.preheader, pair.header, args);
            changed = true;
        }
    }
    if (changed) return true;
    if (!pair.exact) return false;

    // The loop test.
    const insts = func.blockInsts(pair.header);
    for (insts) |inst| {
        const cmp = switch (func.opcode(inst)) {
            .icmp => |c| c,
            else => continue,
        };
        const bound, const flip = if (cmp.lhs == j) .{ cmp.rhs, false } else if (cmp.rhs == j) .{ cmp.lhs, true } else continue;
        if (!definedOutside(func, doms, bound, pair.header)) continue;
        const target = try exactAddress(func, pair.preheader, pair.base, bound, pair.shift, func.valueType(q));
        func.opcodeMut(inst).icmp = if (flip)
            .{ .op = cmp.op, .lhs = target, .rhs = q }
        else
            .{ .op = cmp.op, .lhs = q, .rhs = target };
        return true;
    }

    // Addresses after the loop built from `j`.
    for (0..func.blockCount()) |bi| {
        if (!doms.isReachable(bi) or !doms.dominates(@backingInt(pair.header), bi)) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const a = switch (func.opcode(inst)) {
                .arith => |a| a,
                else => continue,
            };
            if (a.op != .add or a.lhs != pair.base) continue;
            const s = scaled(func, a.rhs) orelse continue;
            if (s.shift != pair.shift) continue;
            const e = indexOffset(func, s.x, j) orelse continue;
            const result = func.instResult(inst).?;
            if (!derefOnly(func, result)) continue;
            func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = .add, .lhs = q, .imm = e << pair.shift } };
            return true;
        }
    }
    return false;
}

/// Whether `v` is `x + c`, or `x` itself for a zero `c`.
fn hasForm(func: *const Function, v: Value, x: Value, c: i64) bool {
    if (c == 0) return v == x;
    const pc = plusConst(func, v) orelse return false;
    return pc.x == x and pc.c == c;
}

/// Whether a compare reads `q`.
fn compared(func: *const Function, q: Value) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| switch (func.opcode(inst)) {
            .icmp => |c| if (c.lhs == q or c.rhs == q) return true,
            else => {},
        };
    }
    return false;
}

/// `x` as `j + e`, through a widening convert on either side of the add.
fn indexOffset(func: *const Function, x: Value, j: Value) ?i64 {
    if (x == j) return 0;
    if (unwiden(func, x) == j) return 0;
    if (plusConst(func, x)) |pc| {
        if (pc.x == j or unwiden(func, pc.x) == j) return pc.c;
    }
    if (plusConst(func, unwiden(func, x))) |pc| {
        if (pc.x == j and signedInt(func, j) != null) return pc.c;
    }
    return null;
}

fn definedOutside(func: *const Function, doms: *const dominators.Dominators, v: Value, header: Block) bool {
    const home = blockOf(func, v) orelse return true;
    return home != @backingInt(header) and doms.dominates(home, @backingInt(header));
}

fn blockOf(func: *const Function, v: Value) ?usize {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block)) |p| if (p == v) return bi;
        for (func.blockInsts(block)) |inst| if (func.instResult(inst) == v) return bi;
    }
    return null;
}

/// When `j0` is `outer_j + c` for an outer pair with the same base and size, its pointer and `c`.
fn outerStart(func: *const Function, pairs: []const Pair, pair: Pair, j0: Value) ?struct { q: Value, c: i64 } {
    const jx, const c = if (plusConst(func, j0)) |pc| .{ pc.x, pc.c } else .{ j0, 0 };
    for (pairs) |outer| {
        if (outer.header == pair.header or !outer.exact) continue;
        if (outer.base != pair.base or outer.shift != pair.shift) continue;
        const params = func.blockParams(outer.header);
        if (params[outer.j] == jx) return .{ .q = params[outer.q], .c = c };
    }
    return null;
}

/// `base + sext(index) << shift` in `block`.
fn exactAddress(func: *Function, block: Block, base: Value, index: Value, shift: u6, ty: ir.types.Type) !Value {
    if (constOf(func, index)) |c| return if (c == 0) base else func.appendArithImm(block, ty, .add, base, c << shift);
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const wide = if (func.valueType(index) == i64_t) index else try func.appendInst(block, i64_t, .{ .convert = .{ .value = index } });
    const offset = if (shift == 0) wide else try func.appendArithImm(block, i64_t, .shl, wide, shift);
    return func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
}

/// Remove `j` when only its own step reads it.
fn dropCounter(allocator: std.mem.Allocator, func: *Function, cfg: anytype, pair: Pair, uses: []const u32) !bool {
    const params = func.blockParams(pair.header);
    const j = params[pair.j];
    const latch_jump = func.terminator(pair.latch).?.jump;
    const next = func.blockArgs(latch_jump)[pair.j];
    if (uses[@backingInt(j)] != 1 or uses[@backingInt(next)] != 1) return false;
    if (plusConst(func, next).?.x != j) return false;

    var np: std.ArrayList(Value) = .empty;
    defer np.deinit(allocator);
    for (params, 0..) |p, i| if (i != pair.j) try np.append(allocator, p);
    try func.setBlockParams(pair.header, np.items);
    for (cfg.predecessors(@backingInt(pair.header))) |pi| {
        const pred: Block = @fromBackingInt(@intCast(pi));
        const jump = func.terminator(pred).?.jump;
        var args: std.ArrayList(Value) = .empty;
        defer args.deinit(allocator);
        for (func.blockArgs(jump), 0..) |arg, i| if (i != pair.j) try args.append(allocator, arg);
        try func.setJump(pred, pair.header, args.items);
    }
    return true;
}

const testing = std.testing;

test "a counter that steps with a pointer leaves, and its test and exit address use the pointer" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    // for (j = 7, q = base + 28; j >= 0; --j, q -= 4) *q = 0; base[j + 1] = 1;
    const base = try func.appendBlockParam(entry, ptr_t);
    const seven = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const start = try func.appendArithImm(entry, ptr_t, .add, base, 28);
    try func.setJump(entry, header, &.{ seven, start });
    const j = try func.appendBlockParam(header, i32_t);
    const q = try func.appendBlockParam(header, ptr_t);
    const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .ge, .lhs = j, .rhs = zero } });
    try func.appendIf(header, more, .{ .target = body }, .{ .target = done });
    try func.appendStore(body, zero, q);
    const j_next = try func.appendArithImm(body, i32_t, .add, j, -1);
    const q_next = try func.appendArithImm(body, ptr_t, .add, q, -4);
    try func.setJump(body, header, &.{ j_next, q_next });
    const after = try func.appendArithImm(done, i32_t, .add, j, 1);
    const wide = try func.appendInst(done, i64_t, .{ .convert = .{ .value = after } });
    const offset = try func.appendArithImm(done, i64_t, .shl, wide, 2);
    const slot = try func.appendInst(done, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const one = try func.appendInst(done, i32_t, .{ .iconst = 1 });
    try func.appendStore(done, one, slot);
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    try testing.expect(try pass.runToFixpoint(allocator, &func, &.{ pass_def, dce.pass_def }, 16));
    try testing.expectEqual(@as(usize, 1), func.blockParams(header).len);
    const pointer = func.blockParams(header)[0];
    const cmp = func.opcode(func.blockInsts(header)[0]).icmp;
    try testing.expectEqual(pointer, cmp.lhs);
    try testing.expectEqual(base, cmp.rhs);
    const store_addr = func.opcode(func.blockInsts(done)[func.blockInsts(done).len - 1]).store.ptr;
    const rebuilt = func.opcode(func.definingInst(store_addr).?).arith_imm;
    try testing.expectEqual(pointer, rebuilt.lhs);
    try testing.expectEqual(@as(i64, 4), rebuilt.imm);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}
