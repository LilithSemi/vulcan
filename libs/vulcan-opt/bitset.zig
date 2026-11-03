//! Set membership as one bit test. A chain of blocks that each test `x == k` and send every match
//! to the same target T becomes a range check of `x - lo` and one test of bit `x - lo` in a
//! constant mask. For example `c == ' ' || c == '\t' || c == '\n'`. One branch then decides the
//! set, where the chain took a branch per member that the data could make hard to predict.
//!
//! Each block after the first has the previous one as its only predecessor and holds only
//! constants and its compare. The members fit one 64-bit mask.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Jump = ir.function.Jump;

pub const pass_def = pass.Pass{ .name = "bitset", .run = run };

/// The fewest members worth a bit test: two compares cost no more than the range check and test.
const min_members = 3;

const Member = struct { x: Value, k: i128, to_t: Jump, other: Jump };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const graph = try analyses.cfg();
    for (0..func.blockCount()) |ai| {
        const a: Block = @fromBackingInt(@intCast(ai));
        if (try fuse(allocator, func, graph, a)) {
            analyses.invalidate();
            return true;
        }
    }
    return false;
}

fn fuse(allocator: std.mem.Allocator, func: *Function, graph: anytype, a: Block) pass.Error!bool {
    const a_if = lastIf(func, a) orelse return false;
    const first = member(func, func.opcode(a_if).@"if") orelse return false;
    var chain: std.ArrayList(Block) = .empty;
    defer chain.deinit(allocator);
    var lo = first.k;
    var hi = first.k;
    var mask_members: std.ArrayList(i128) = .empty;
    defer mask_members.deinit(allocator);
    try mask_members.append(allocator, first.k);
    var last = first;
    while (true) {
        const b = last.other.target;
        if (b == a or std.mem.indexOfScalar(Block, chain.items, b) != null) break;
        if (func.blockArgs(last.other).len != 0 or func.blockParams(b).len != 0) break;
        const preds = graph.predecessors(@backingInt(b));
        if (preds.len != 1) break;
        const b_if = lastIf(func, b) orelse break;
        const next = member(func, func.opcode(b_if).@"if") orelse break;
        if (next.x != first.x or next.to_t.target != first.to_t.target) break;
        if (!std.mem.eql(Value, func.blockArgs(next.to_t), func.blockArgs(first.to_t))) break;
        const cmp_inst = func.definingInst(func.opcode(b_if).@"if".cond).?;
        var plain = true;
        for (func.blockInsts(b)) |inst| {
            if (inst == b_if or inst == cmp_inst) continue;
            if (func.opcode(inst) != .iconst) plain = false;
        }
        if (!plain or std.mem.indexOfScalar(Inst, func.blockInsts(b), cmp_inst) == null) break;
        const new_lo = @min(lo, next.k);
        const new_hi = @max(hi, next.k);
        if (new_hi - new_lo >= 64) break;
        lo = new_lo;
        hi = new_hi;
        try mask_members.append(allocator, next.k);
        try chain.append(allocator, b);
        last = next;
    }
    if (mask_members.items.len < min_members) return false;
    for (func.blockArgs(last.other)) |arg| {
        for (chain.items) |b| for (func.blockInsts(b)) |inst| {
            if (func.instResult(inst) == arg and func.opcode(inst) != .iconst) return false;
        };
    }
    var mask: u64 = 0;
    for (mask_members.items) |k| mask |= @as(u64, 1) << @intCast(k - lo);
    try rewrite(allocator, func, a, a_if, chain.items, first, last.other, lo, hi, mask);
    return true;
}

fn rewrite(allocator: std.mem.Allocator, func: *Function, a: Block, a_if: Inst, chain: []const Block, first: Member, to_f: Jump, lo: i128, hi: i128, mask: u64) pass.Error!void {
    const t_args = try allocator.dupe(Value, func.blockArgs(first.to_t));
    defer allocator.free(t_args);
    const f_args = try allocator.dupe(Value, func.blockArgs(to_f));
    defer allocator.free(f_args);

    // The chain's constants move into A, so a use of one below still has a dominating definition.
    var insts: std.ArrayList(Inst) = .empty;
    defer insts.deinit(allocator);
    for (func.blockInsts(a)) |inst| if (inst != a_if) try insts.append(allocator, inst);
    for (chain) |b| {
        for (func.blockInsts(b)) |inst| if (func.opcode(inst) == .iconst) try insts.append(allocator, inst);
        try func.setBlockInsts(b, &.{});
        func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    }
    try func.setBlockInsts(a, insts.items);

    const int = func.types.type_kind(func.valueType(first.x)).int;
    const unsigned_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = int.bits } });
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const x = if (int.signedness == .signed)
        try func.appendInst(a, unsigned_ty, .{ .convert = .{ .value = first.x } })
    else
        first.x;
    const offset = try func.appendArithImm(a, unsigned_ty, .sub, x, @truncate(lo));
    const width = try func.appendInst(a, unsigned_ty, .{ .iconst = @truncate(hi - lo) });
    const inside = try func.appendInst(a, bool_t, .{ .icmp = .{ .op = .le, .lhs = offset, .rhs = width } });
    const test_block = try func.appendBlock();
    try func.appendIf(a, inside, .{ .target = test_block }, .{ .target = to_f.target, .args = f_args });
    func.setTerminator(a, .{ .ret = ir.function.Ret.none() });

    const wide = if (int.bits < 64) try func.appendInst(test_block, u64_t, .{ .convert = .{ .value = offset } }) else offset;
    const bits = try func.appendInst(test_block, u64_t, .{ .iconst = @bitCast(mask) });
    const shifted = try func.appendInst(test_block, u64_t, .{ .arith = .{ .op = .shr, .lhs = bits, .rhs = wide } });
    const bit = try func.appendArithImm(test_block, u64_t, .bit_and, shifted, 1);
    const zero = try func.appendInst(test_block, u64_t, .{ .iconst = 0 });
    const hit = try func.appendInst(test_block, bool_t, .{ .icmp = .{ .op = .ne, .lhs = bit, .rhs = zero } });
    try func.appendIf(test_block, hit, .{ .target = first.to_t.target, .args = t_args }, .{ .target = to_f.target, .args = f_args });
    func.setTerminator(test_block, .{ .ret = ir.function.Ret.none() });
}

/// The `x == k` test of an `if`, with the edge taken on a match and the other edge.
fn member(func: *const Function, cf: ir.function.If) ?Member {
    const inst = func.definingInst(cf.cond) orelse return null;
    const cmp = switch (func.opcode(inst)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .eq and cmp.op != .ne) return null;
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
    const shift: u6 = @intCast(64 - @as(u32, int.bits) & 63);
    const k: i128 = if (int.signedness == .signed)
        @as(i64, @bitCast(@as(u64, @bitCast(raw)) << shift)) >> shift
    else
        (@as(u64, @bitCast(raw)) << shift) >> shift;
    if (std.mem.indexOfScalar(Value, func.blockArgs(cf.then), cf.cond) != null) return null;
    if (std.mem.indexOfScalar(Value, func.blockArgs(cf.@"else"), cf.cond) != null) return null;
    return if (cmp.op == .eq)
        .{ .x = cmp.lhs, .k = k, .to_t = cf.then, .other = cf.@"else" }
    else
        .{ .x = cmp.lhs, .k = k, .to_t = cf.@"else", .other = cf.then };
}

fn lastIf(func: *const Function, block: Block) ?Inst {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const last = insts[insts.len - 1];
    return if (func.opcode(last) == .@"if") last else null;
}

const testing = std.testing;

test "three equality tests of one byte become a range check and a bit test" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const a = try func.appendBlock();
    const b = try func.appendBlock();
    const c = try func.appendBlock();
    const yes = try func.appendBlock();
    const no = try func.appendBlock();
    const x = try func.appendBlockParam(a, i32_t);
    for ([_]struct { Block, i64, Block }{ .{ a, ' ', b }, .{ b, '\t', c }, .{ c, '\n', no } }) |step| {
        const k = try func.appendInst(step[0], i32_t, .{ .iconst = step[1] });
        const same = try func.appendInst(step[0], bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = k } });
        try func.appendIf(step[0], same, .{ .target = yes }, .{ .target = step[2] });
        func.setTerminator(step[0], .{ .ret = ir.function.Ret.none() });
    }
    const one = try func.appendInst(yes, i32_t, .{ .iconst = 1 });
    func.setTerminator(yes, .{ .ret = ir.function.Ret.one(one) });
    const zero = try func.appendInst(no, i32_t, .{ .iconst = 0 });
    func.setTerminator(no, .{ .ret = ir.function.Ret.one(zero) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    try testing.expectEqual(@as(usize, 0), func.blockInsts(b).len);
    try testing.expectEqual(@as(usize, 0), func.blockInsts(c).len);
    // '\t' is 9, '\n' is 10, ' ' is 32: bits 0, 1 and 23 above the low end.
    var mask: ?i64 = null;
    for (func.blockInsts(@fromBackingInt(5))) |inst| switch (func.opcode(inst)) {
        .iconst => |k| if (k > 1) {
            mask = k;
        },
        else => {},
    };
    try testing.expectEqual(@as(?i64, (1 << 0) | (1 << 1) | (1 << 23)), mask);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "two equality tests stay as they are" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const a = try func.appendBlock();
    const b = try func.appendBlock();
    const yes = try func.appendBlock();
    const no = try func.appendBlock();
    const x = try func.appendBlockParam(a, i32_t);
    for ([_]struct { Block, i64, Block }{ .{ a, ' ', b }, .{ b, '\n', no } }) |step| {
        const k = try func.appendInst(step[0], i32_t, .{ .iconst = step[1] });
        const same = try func.appendInst(step[0], bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = k } });
        try func.appendIf(step[0], same, .{ .target = yes }, .{ .target = step[2] });
        func.setTerminator(step[0], .{ .ret = ir.function.Ret.none() });
    }
    func.setTerminator(yes, .{ .ret = ir.function.Ret.one(x) });
    func.setTerminator(no, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses));
}
