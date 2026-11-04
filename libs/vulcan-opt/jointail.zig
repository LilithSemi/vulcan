//! A small join that ends in a two-way branch gets a copy for each block that jumps to it.
//! Each path then runs the join inline and branches once, without a jump to the join first.
//! A loop whose body splits into several paths that meet at the latch gains the most.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");
const dominators = @import("dominators.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;

pub const pass_def = pass.Pass{ .name = "jointail", .run = run };

const max_insts = 16;
const max_preds = 4;

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    for (func.attributeEntries()) |entry| if (entry.target == .block) return false;
    var doms = try dominators.compute(allocator, func);
    defer doms.deinit(allocator);
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    dce.countUses(func, uses);
    const local = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(local);

    var preds: std.ArrayList(Block) = .empty;
    defer preds.deinit(allocator);
    var changed = false;
    const count = func.blockCount();
    for (1..count) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        if (!doms.isReachable(bi)) continue;
        preds.clearRetainingCapacity();
        if (!try jumpPreds(allocator, func, &doms, block, &preds)) continue;
        if (preds.items.len < 2 or preds.items.len > max_preds) continue;
        @memset(local, 0);
        dce.countBlockUses(func, block, local);
        if (!duplicable(func, &doms, block, uses, local)) continue;
        for (preds.items[1..]) |pred| {
            var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
            defer map.deinit(allocator);
            const copy = try func.cloneBlock(allocator, block, &map);
            const args = try allocator.dupe(Value, func.blockArgs(func.terminator(pred).?.jump));
            defer allocator.free(args);
            try func.setJump(pred, copy, args);
        }
        changed = true;
    }
    return changed;
}

/// A short block of plain instructions that ends in a branch, whose params and values are read
/// only inside the block. A latch that jumps back to a loop test counts, since the selector
/// repeats that test in the latch.
fn duplicable(func: *const Function, doms: *const dominators.Dominators, block: Block, uses: []const u32, local: []const u32) bool {
    const insts = func.blockInsts(block);
    const body = if (func.terminator(block)) |term| switch (term) {
        .jump => |j| if (doms.dominates(@backingInt(j.target), @backingInt(block)) and testOnly(func, j.target)) insts else return false,
        .ret => return false,
    } else body: {
        if (insts.len == 0) return false;
        const cf = switch (func.opcode(insts[insts.len - 1])) {
            .@"if" => |c| c,
            else => return false,
        };
        if (cf.then.target == block or cf.@"else".target == block) return false;
        break :body insts[0 .. insts.len - 1];
    };
    if (body.len > max_insts) return false;
    for (func.blockParams(block)) |param| {
        if (uses[@backingInt(param)] != local[@backingInt(param)]) return false;
    }
    for (body) |inst| {
        switch (func.opcode(inst)) {
            .iconst, .arith_imm, .icmp, .convert, .select, .unary => {},
            .arith => |a| if (a.op == .div or a.op == .rem) return false,
            .load => |l| if (l.mask != null) return false,
            .store => |s| if (s.mask != null) return false,
            else => return false,
        }
        const result = func.instResult(inst) orelse continue;
        if (uses[@backingInt(result)] != local[@backingInt(result)]) return false;
    }
    return true;
}

/// A loop test: a compare and an `if` on it.
fn testOnly(func: *const Function, block: Block) bool {
    const insts = func.blockInsts(block);
    if (insts.len != 2) return false;
    return func.opcode(insts[0]) == .icmp and func.opcode(insts[1]) == .@"if";
}

/// Collect the predecessors of `block` into `preds`. False when any edge is not a plain jump
/// or the block heads a loop.
fn jumpPreds(allocator: std.mem.Allocator, func: *const Function, doms: *const dominators.Dominators, block: Block, preds: *std.ArrayList(Block)) !bool {
    for (0..func.blockCount()) |pi| {
        if (!doms.isReachable(pi)) continue;
        const pred: Block = @fromBackingInt(@intCast(pi));
        for (func.blockInsts(pred)) |inst| if (func.opcode(inst) == .@"if") {
            const cf = func.opcode(inst).@"if";
            if (cf.then.target == block or cf.@"else".target == block) return false;
        };
        const term = func.terminator(pred) orelse continue;
        switch (term) {
            .jump => |j| if (j.target == block) {
                if (doms.dominates(@backingInt(block), pi)) return false;
                try preds.append(allocator, pred);
            },
            .ret => {},
        }
    }
    return true;
}

const testing = std.testing;

test "each block that jumps to a short branching join gets its own copy" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const join = try func.appendBlock();
    const done = try func.appendBlock();
    const flag = try func.appendBlockParam(entry, bool_t);
    const n = try func.appendBlockParam(entry, i64_t);
    try func.appendIf(entry, flag, .{ .target = left }, .{ .target = right });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    const one = try func.appendInst(left, i64_t, .{ .iconst = 1 });
    try func.setJump(left, join, &.{one});
    const two = try func.appendInst(right, i64_t, .{ .iconst = 2 });
    try func.setJump(right, join, &.{two});
    const k = try func.appendBlockParam(join, i64_t);
    const sum = try func.appendInst(join, i64_t, .{ .arith = .{ .op = .add, .lhs = k, .rhs = n } });
    const small = try func.appendInst(join, bool_t, .{ .icmp = .{ .op = .lt, .lhs = sum, .rhs = n } });
    try func.appendIf(join, small, .{ .target = done, .args = &.{sum} }, .{ .target = done, .args = &.{k} });
    const r = try func.appendBlockParam(done, i64_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(r) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    try testing.expectEqual(@as(usize, 6), func.blockCount());
    try testing.expectEqual(join, func.terminator(left).?.jump.target);
    const copy = func.terminator(right).?.jump.target;
    try testing.expect(copy != join);
    try testing.expectEqual(func.blockInsts(join).len, func.blockInsts(copy).len);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a loop header keeps its single copy" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i64_t);
    const zero = try func.appendInst(entry, i64_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{zero});
    const i = try func.appendBlockParam(header, i64_t);
    const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, more, .{ .target = body }, .{ .target = done });
    const next = try func.appendArithImm(body, i64_t, .add, i, 1);
    try func.setJump(body, header, &.{next});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses));
}
