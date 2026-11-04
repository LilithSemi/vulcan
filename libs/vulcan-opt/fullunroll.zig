//! A loop with a known trip count of at most `max_trips` and a short pure body becomes straight-line
//! code. The steps then share one block, so later passes see across them. A CRC bit loop is the
//! case this is for: once unrolled, each step's mask reads the starting value, not the last step.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "fullunroll", .run = run };

const max_trips = 8;
const max_insts = 64;

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    for (func.attributeEntries()) |entry| if (entry.target == .block) return false;
    const loop_info = try analyses.loops();
    const cfg = try analyses.cfg();
    for (loop_info.loops) |*loop| {
        var chain: std.ArrayList(Block) = .empty;
        defer chain.deinit(allocator);
        const plan = try planLoop(allocator, func, cfg, loop, &chain) orelse continue;
        try unroll(allocator, func, plan, chain.items);
        analyses.invalidate();
        return true;
    }
    return false;
}

const Plan = struct {
    header: Block,
    entry: Block,
    trips: u32,
    exit: ir.function.Jump,
};

/// The header holds only a compare of a counter with a constant and an `if` on it. The body is
/// a chain of jumps back to the header. The counter starts at a constant and steps by one.
fn planLoop(allocator: std.mem.Allocator, func: *const Function, cfg: anytype, loop: anytype, chain: *std.ArrayList(Block)) !?Plan {
    const header: Block = @fromBackingInt(@intCast(loop.header));
    const insts = func.blockInsts(header);
    if (insts.len != 2 or func.terminator(header) != null) return null;
    const cmp = switch (func.opcode(insts[0])) {
        .icmp => |c| c,
        else => return null,
    };
    const cf = switch (func.opcode(insts[1])) {
        .@"if" => |c| c,
        else => return null,
    };
    if (cf.cond != func.instResult(insts[0]).?) return null;
    if (!loop.contains(@backingInt(cf.then.target)) or loop.contains(@backingInt(cf.@"else".target))) return null;
    if (func.blockArgs(cf.then).len != 0) return null;

    const preds = cfg.predecessors(loop.header);
    if (preds.len != 2) return null;
    var entry: ?Block = null;
    for (preds) |p| {
        if (loop.contains(p)) continue;
        const term = func.terminator(@fromBackingInt(@intCast(p))) orelse return null;
        if (term != .jump) return null;
        entry = @fromBackingInt(@intCast(p));
    }
    const pred = entry orelse return null;

    // Walk the body chain back to the header.
    var block = cf.then.target;
    var count: usize = 0;
    var latch_args: []const Value = &.{};
    while (true) {
        if (func.blockParams(block).len != 0) return null;
        try chain.append(allocator, block);
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .iconst, .arith, .arith_imm, .icmp, .convert, .select, .unary => {},
                else => return null,
            }
            count += 1;
        }
        const term = func.terminator(block) orelse return null;
        const j = switch (term) {
            .jump => |j| j,
            .ret => return null,
        };
        if (j.target == header) {
            latch_args = func.blockArgs(j);
            break;
        }
        if (func.blockArgs(j).len != 0 or chain.items.len > 8) return null;
        block = j.target;
    }
    var in_loop: usize = 0;
    for (loop.body) |member| in_loop += @intFromBool(member);
    if (in_loop != chain.items.len + 1) return null;

    const trips = tripCount(func, header, cmp, func.blockArgs(func.terminator(pred).?.jump), latch_args) orelse return null;
    if (trips == 0 or trips * count > max_insts) return null;
    return .{ .header = header, .entry = pred, .trips = trips, .exit = cf.@"else" };
}

fn tripCount(func: *const Function, header: Block, cmp: ir.function.Compare, entry_args: []const Value, latch_args: []const Value) ?u32 {
    const params = func.blockParams(header);
    const k = std.mem.indexOfScalar(Value, params, cmp.lhs) orelse return null;
    const bound = constOf(func, cmp.rhs) orelse return null;
    var v = constOf(func, entry_args[k]) orelse return null;
    const step_inst = func.definingInst(latch_args[k]) orelse return null;
    const step = switch (func.opcode(step_inst)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == cmp.lhs) a.imm else return null,
        .arith => |a| if (a.op == .add and a.lhs == cmp.lhs) (constOf(func, a.rhs) orelse return null) else return null,
        else => return null,
    };
    const int = switch (func.types.type_kind(func.valueType(cmp.lhs))) {
        .int => |i| i,
        else => return null,
    };
    if (int.bits == 0 or int.bits > 64) return null;
    const signed = int.signedness == .signed;
    var trips: u32 = 0;
    while (holds(cmp.op, v, bound, int.bits, signed)) : (trips += 1) {
        if (trips == max_trips) return null;
        v +%= step;
    }
    return trips;
}

fn holds(op: ir.function.CmpOp, a: i64, b: i64, bits: u16, signed: bool) bool {
    const shift: u6 = @intCast(64 - @as(u32, bits));
    const ua = (@as(u64, @bitCast(a)) << shift) >> shift;
    const ub = (@as(u64, @bitCast(b)) << shift) >> shift;
    const sa = @as(i64, @bitCast(@as(u64, @bitCast(a)) << shift)) >> shift;
    const sb = @as(i64, @bitCast(@as(u64, @bitCast(b)) << shift)) >> shift;
    return switch (op) {
        .eq => ua == ub,
        .ne => ua != ub,
        .lt => if (signed) sa < sb else ua < ub,
        .le => if (signed) sa <= sb else ua <= ub,
        .gt => if (signed) sa > sb else ua > ub,
        .ge => if (signed) sa >= sb else ua >= ub,
    };
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        else => null,
    };
}

fn unroll(allocator: std.mem.Allocator, func: *Function, plan: Plan, chain: []const Block) !void {
    const params = try allocator.dupe(Value, func.blockParams(plan.header));
    defer allocator.free(params);
    const current = try allocator.dupe(Value, func.blockArgs(func.terminator(plan.entry).?.jump));
    defer allocator.free(current);
    const latch = chain[chain.len - 1];
    const latch_args = try allocator.dupe(Value, func.blockArgs(func.terminator(latch).?.jump));
    defer allocator.free(latch_args);

    const straight = try func.appendBlock();
    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(allocator);
    for (0..plan.trips) |_| {
        map.clearRetainingCapacity();
        for (params, current) |param, value| try map.put(allocator, param, value);
        for (chain) |block| {
            for (func.blockInsts(block)) |inst| {
                const result = func.instResult(inst).?;
                const op = try func.remapOpcode(allocator, func.opcode(inst), &map);
                try map.put(allocator, result, try func.appendInst(straight, func.valueType(result), op));
            }
        }
        for (latch_args, current) |arg, *value| value.* = map.get(arg) orelse arg;
    }

    map.clearRetainingCapacity();
    for (params, current) |param, value| try map.put(allocator, param, value);
    const exit_args = try allocator.dupe(Value, func.blockArgs(plan.exit));
    defer allocator.free(exit_args);
    for (exit_args) |*arg| arg.* = map.get(arg.*) orelse arg.*;
    try func.setJump(straight, plan.exit.target, exit_args);
    try func.setJump(plan.entry, straight, &.{});
    // The exit may read the header's values directly, since the header dominated it.
    for (params, current) |param, value| func.replaceAllUses(param, value);
}

const testing = std.testing;

test "a loop of eight steps with a constant start becomes one straight block" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, u32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const eight = try func.appendInst(entry, i32_t, .{ .iconst = 8 });
    try func.setJump(entry, header, &.{ x, zero });
    const acc = try func.appendBlockParam(header, u32_t);
    const bit = try func.appendBlockParam(header, i32_t);
    const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = bit, .rhs = eight } });
    try func.appendIf(header, more, .{ .target = body }, .{ .target = done });
    const half = try func.appendArithImm(body, u32_t, .shr, acc, 1);
    const next = try func.appendArithImm(body, i32_t, .add, bit, 1);
    try func.setJump(body, header, &.{ half, next });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(acc) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    const straight = func.terminator(entry).?.jump.target;
    var shifts: usize = 0;
    for (func.blockInsts(straight)) |inst| {
        if (func.opcode(inst) == .arith_imm and func.opcode(inst).arith_imm.op == .shr) shifts += 1;
    }
    try testing.expectEqual(@as(usize, 8), shifts);
    try testing.expectEqual(done, func.terminator(straight).?.jump.target);
    const result = func.terminator(done).?.ret.slice()[0];
    try testing.expect(std.mem.indexOfScalar(Inst, func.blockInsts(straight), func.definingInst(result).?) != null);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}
