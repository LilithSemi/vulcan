//! Convert small, pure control-flow diamonds into straight-line `select` operations.
//!
//! Both arms execute after conversion, so eligibility is deliberately narrow: each arm must
//! contain at most a small shared budget of integer operations that cannot trap or touch memory.
//! The two arms must be the merge block's only predecessors. This removes unpredictable branches
//! without duplicating work or changing any other incoming edge.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

const max_speculated_insts = 8;

pub const pass_def = pass.Pass{ .name = "ifconvert", .run = run };
pub const model_pass_def = pass.Pass{ .name = "ifconvert-model", .run = runModel };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    return runWithBudget(allocator, func, analyses, max_speculated_insts);
}

fn runModel(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    return runWithBudget(allocator, func, analyses, 16);
}

fn runWithBudget(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses, budget: usize) pass.Error!bool {
    const graph = try analyses.cfg();
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        if (insts.len == 0 or func.opcode(insts[insts.len - 1]) != .@"if") continue;
        if (func.terminator(block) != null) continue;
        const cf = func.opcode(insts[insts.len - 1]).@"if";
        if (cf.then.target == cf.@"else".target) continue;
        if (func.blockArgs(cf.then).len != 0 or func.blockArgs(cf.@"else").len != 0) continue;

        const then_block = cf.then.target;
        const else_block = cf.@"else".target;
        if (!singlePredecessor(graph, then_block, bi) or !singlePredecessor(graph, else_block, bi)) continue;
        if (func.blockParams(then_block).len != 0 or func.blockParams(else_block).len != 0) continue;
        if (hasBlockAttrs(func, then_block) or hasBlockAttrs(func, else_block)) continue;

        const then_jump = jumpTerminator(func, then_block) orelse continue;
        const else_jump = jumpTerminator(func, else_block) orelse continue;
        if (then_jump.target != else_jump.target) continue;
        const merge = then_jump.target;
        if (merge == block or merge == then_block or merge == else_block) continue;
        const predecessors = graph.predecessors(@backingInt(merge));
        if (predecessors.len != 2 or !contains(predecessors, @backingInt(then_block)) or !contains(predecessors, @backingInt(else_block))) continue;

        const params = func.blockParams(merge);
        const then_args = func.blockArgs(then_jump);
        const else_args = func.blockArgs(else_jump);
        if (params.len == 0 or then_args.len != params.len or else_args.len != params.len) continue;
        if (!cheapArm(func, then_block) or !cheapArm(func, else_block)) continue;
        if (func.blockInsts(then_block).len + func.blockInsts(else_block).len + params.len > budget) continue;

        try convertDiamond(allocator, func, block, then_block, else_block, merge, cf.cond, params, then_args, else_args);
        return true;
    }
    return false;
}

fn singlePredecessor(graph: anytype, block: Block, predecessor: usize) bool {
    const preds = graph.predecessors(@backingInt(block));
    return preds.len == 1 and preds[0] == predecessor;
}

fn contains(items: []const u32, needle: u32) bool {
    for (items) |item| if (item == needle) return true;
    return false;
}

fn hasBlockAttrs(func: *const Function, block: Block) bool {
    var attrs = func.attributesOf(.{ .block = block });
    return attrs.next() != null;
}

fn jumpTerminator(func: *const Function, block: Block) ?ir.function.Jump {
    const term = func.terminator(block) orelse return null;
    return switch (term) {
        .jump => |jump| jump,
        .ret => null,
    };
}

fn cheapArm(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        const result = func.instResult(inst) orelse return false;
        const kind = func.types.type_kind(func.valueType(result));
        if (kind != .int and kind != .bool and kind != .ptr) return false;
        switch (func.opcode(inst)) {
            .iconst, .icmp, .select, .global_addr => {},
            .arith => |arith| if (arith.op == .div or arith.op == .rem) return false,
            .arith_imm => |arith| if (arith.op == .div or arith.op == .rem) return false,
            .convert => |convert| {
                const source = func.types.type_kind(func.valueType(convert.value));
                if (source != .int and source != .bool and source != .ptr) return false;
            },
            else => return false,
        }
    }
    return true;
}

fn convertDiamond(
    allocator: std.mem.Allocator,
    func: *Function,
    block: Block,
    then_block: Block,
    else_block: Block,
    merge: Block,
    cond: ir.function.Value,
    params: []const ir.function.Value,
    then_args: []const ir.function.Value,
    else_args: []const ir.function.Value,
) !void {
    const owned_params = try allocator.dupe(ir.function.Value, params);
    defer allocator.free(owned_params);
    const owned_then_args = try allocator.dupe(ir.function.Value, then_args);
    defer allocator.free(owned_then_args);
    const owned_else_args = try allocator.dupe(ir.function.Value, else_args);
    defer allocator.free(owned_else_args);
    const then_insts = try allocator.dupe(Inst, func.blockInsts(then_block));
    defer allocator.free(then_insts);
    const else_insts = try allocator.dupe(Inst, func.blockInsts(else_block));
    defer allocator.free(else_insts);

    const parent = func.blockInstsMut(block);
    _ = parent.pop();
    try parent.appendSlice(allocator, then_insts);
    try parent.appendSlice(allocator, else_insts);
    func.blockInstsMut(then_block).clearRetainingCapacity();
    func.blockInstsMut(else_block).clearRetainingCapacity();
    func.terminatorPtr(then_block).* = null;
    func.terminatorPtr(else_block).* = null;

    for (owned_params, owned_then_args, owned_else_args) |param, then_value, else_value| {
        const selected = try func.appendInst(block, func.valueType(param), .{ .select = .{
            .cond = cond,
            .then = then_value,
            .@"else" = else_value,
        } });
        func.replaceAllUses(param, selected);
        func.retargetAttrs(.{ .value = param }, .{ .value = selected });
    }
    try func.setBlockParams(merge, &.{});
    try func.setJump(block, merge, &.{});
}

const testing = std.testing;

test "ifconvert turns a cheap integer diamond into a select" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_block = try func.appendBlock();
    const else_block = try func.appendBlock();
    const merge = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, i32_t);
    const result = try func.appendBlockParam(merge, i32_t);
    try func.appendIf(entry, cond, .{ .target = then_block }, .{ .target = else_block });
    const one = try func.appendInst(then_block, i32_t, .{ .iconst = 1 });
    const plus = try func.appendInst(then_block, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = one } });
    try func.setJump(then_block, merge, &.{plus});
    const minus = try func.appendInst(else_block, i32_t, .{ .arith_imm = .{ .op = .sub, .lhs = x, .imm = 1 } });
    try func.setJump(else_block, merge, &.{minus});
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(result) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    try testing.expectEqual(@as(usize, 0), func.blockParams(merge).len);
    try testing.expect(func.terminator(entry).? == .jump);
    const insts = func.blockInsts(entry);
    try testing.expect(func.opcode(insts[insts.len - 1]) == .select);
    try testing.expectEqual(func.instResult(insts[insts.len - 1]).?, func.terminator(merge).?.ret.values[0]);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "ifconvert refuses a potentially trapping arm" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_block = try func.appendBlock();
    const else_block = try func.appendBlock();
    const merge = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, i32_t);
    const y = try func.appendBlockParam(entry, i32_t);
    const result = try func.appendBlockParam(merge, i32_t);
    try func.appendIf(entry, cond, .{ .target = then_block }, .{ .target = else_block });
    const quotient = try func.appendInst(then_block, i32_t, .{ .arith = .{ .op = .div, .lhs = x, .rhs = y } });
    try func.setJump(then_block, merge, &.{quotient});
    try func.setJump(else_block, merge, &.{x});
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(result) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses));
}

test "model ifconvert admits a wider pure diamond" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const then_block = try func.appendBlock();
    const else_block = try func.appendBlock();
    const merge = try func.appendBlock();
    const cond = try func.appendBlockParam(entry, bool_t);
    const x = try func.appendBlockParam(entry, i32_t);
    const result = try func.appendBlockParam(merge, i32_t);
    try func.appendIf(entry, cond, .{ .target = then_block }, .{ .target = else_block });

    var then_value = x;
    var else_value = x;
    for (0..5) |_| {
        then_value = try func.appendInst(then_block, i32_t, .{ .arith_imm = .{ .op = .add, .lhs = then_value, .imm = 1 } });
        else_value = try func.appendInst(else_block, i32_t, .{ .arith_imm = .{ .op = .sub, .lhs = else_value, .imm = 1 } });
    }
    try func.setJump(then_block, merge, &.{then_value});
    try func.setJump(else_block, merge, &.{else_value});
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(result) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses));
    try testing.expect(try runModel(allocator, &func, &analyses));
    try testing.expect(func.terminator(entry).? == .jump);
}
