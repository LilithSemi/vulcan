//! Rotate a canonical top-tested loop so its recurring test runs in the latch.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;

pub const pass_def = pass.Pass{ .name = "looprotate", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const graph = try analyses.cfg();
    const doms = try analyses.dominators();
    const loop_info = try analyses.loops();
    const def_blocks = try analyses.definingBlocks();

    for (loop_info.loops) |loop| {
        const header: Block = @fromBackingInt(@intCast(loop.header));
        const header_insts = func.blockInsts(header);
        if (header_insts.len != 2 or func.terminator(header) != null) continue;

        const compare_result = func.instResult(header_insts[0]) orelse continue;
        const compare = switch (func.opcode(header_insts[0])) {
            .icmp => |icmp| icmp,
            else => continue,
        };
        const branch = switch (func.opcode(header_insts[1])) {
            .@"if" => |conditional| conditional,
            else => continue,
        };
        if (branch.cond != compare_result) continue;

        const body = branch.then.target;
        const exit = branch.@"else".target;
        if (!loop.contains(@backingInt(body)) or loop.contains(@backingInt(exit))) continue;
        if (graph.predecessors(@backingInt(body)).len != 1 or graph.predecessors(@backingInt(body))[0] != loop.header) continue;
        if (graph.predecessors(@backingInt(exit)).len != 1 or graph.predecessors(@backingInt(exit))[0] != loop.header) continue;
        if (graph.successors(@backingInt(exit)).len != 0) continue;

        var latch_index: ?u32 = null;
        for (graph.predecessors(loop.header)) |pred| {
            if (!loop.contains(pred)) continue;
            if (latch_index != null) {
                latch_index = null;
                break;
            }
            latch_index = pred;
        }
        const latch_raw = latch_index orelse continue;
        const latch: Block = @fromBackingInt(@intCast(latch_raw));
        const latch_jump = switch (func.terminator(latch) orelse continue) {
            .jump => |jump| jump,
            .ret => continue,
        };
        if (latch_jump.target != header) continue;

        var body_dominates_loop = true;
        for (loop.body, 0..) |inside, block_index| {
            if (!inside or block_index == loop.header) continue;
            if (!doms.dominates(@backingInt(body), block_index)) {
                body_dominates_loop = false;
                break;
            }
        }
        if (!body_dominates_loop) continue;

        const header_params = try allocator.dupe(Value, func.blockParams(header));
        defer allocator.free(header_params);
        if (header_params.len > 1) continue;
        if (func.blockArgs(latch_jump).len != header_params.len) continue;
        if (!operandAvailableAtLatch(header_params, def_blocks, loop.header, compare.lhs) or
            !operandAvailableAtLatch(header_params, def_blocks, loop.header, compare.rhs)) continue;

        const then_args = try allocator.dupe(Value, func.blockArgs(branch.then));
        defer allocator.free(then_args);
        const else_args = try allocator.dupe(Value, func.blockArgs(branch.@"else"));
        defer allocator.free(else_args);

        const body_params = try allocator.alloc(Value, header_params.len);
        defer allocator.free(body_params);
        const exit_params = try allocator.alloc(Value, header_params.len);
        defer allocator.free(exit_params);
        for (header_params, 0..) |param, index| {
            body_params[index] = try func.appendBlockParam(body, func.valueType(param));
            exit_params[index] = try func.appendBlockParam(exit, func.valueType(param));
        }

        for (loop.body, 0..) |inside, block_index| {
            if (!inside or block_index == loop.header) continue;
            const block: Block = @fromBackingInt(@intCast(block_index));
            for (header_params, body_params) |param, replacement| func.replaceUsesIn(block, param, replacement);
        }
        for (header_params, exit_params) |param, replacement| func.replaceUsesIn(exit, param, replacement);

        const next_jump = func.terminator(latch).?.jump;
        const next_args = try allocator.dupe(Value, func.blockArgs(next_jump));
        defer allocator.free(next_args);

        const latch_lhs = remap(header_params, next_args, compare.lhs);
        const latch_rhs = remap(header_params, next_args, compare.rhs);
        const latch_condition = try func.appendInst(latch, func.valueType(compare_result), .{
            .icmp = .{ .op = compare.op, .lhs = latch_lhs, .rhs = latch_rhs },
        });

        const header_body_args = try concat(allocator, then_args, header_params);
        defer allocator.free(header_body_args);
        const header_exit_args = try concat(allocator, else_args, header_params);
        defer allocator.free(header_exit_args);
        const latch_body_prefix = try remapAll(allocator, header_params, next_args, then_args);
        defer allocator.free(latch_body_prefix);
        const latch_exit_prefix = try remapAll(allocator, header_params, next_args, else_args);
        defer allocator.free(latch_exit_prefix);
        const latch_body_args = try concat(allocator, latch_body_prefix, next_args);
        defer allocator.free(latch_body_args);
        const latch_exit_args = try concat(allocator, latch_exit_prefix, next_args);
        defer allocator.free(latch_exit_args);

        const header_branch = &func.opcodeMut(header_insts[1]).@"if";
        header_branch.then.args = try func.internValues(header_body_args);
        header_branch.@"else".args = try func.internValues(header_exit_args);
        func.terminatorPtr(latch).* = null;
        try func.appendIf(latch, latch_condition, .{ .target = body, .args = latch_body_args }, .{
            .target = exit,
            .args = latch_exit_args,
        });
        return true;
    }
    return false;
}

fn operandAvailableAtLatch(header_params: []const Value, def_blocks: []const u32, header: u32, value: Value) bool {
    if (std.mem.indexOfScalar(Value, header_params, value) != null) return true;
    return def_blocks[@backingInt(value)] != header;
}

fn remap(from: []const Value, to: []const Value, value: Value) Value {
    const index = std.mem.indexOfScalar(Value, from, value) orelse return value;
    return to[index];
}

fn remapAll(allocator: std.mem.Allocator, from: []const Value, to: []const Value, values: []const Value) std.mem.Allocator.Error![]Value {
    const result = try allocator.alloc(Value, values.len);
    for (values, result) |value, *mapped| mapped.* = remap(from, to, value);
    return result;
}

fn concat(allocator: std.mem.Allocator, first: []const Value, second: []const Value) std.mem.Allocator.Error![]Value {
    const result = try allocator.alloc(Value, first.len + second.len);
    @memcpy(result[0..first.len], first);
    @memcpy(result[first.len..], second);
    return result;
}

test "rotates a top tested loop through its latch" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const limit = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{zero});
    const index = try func.appendBlockParam(header, i32_t);
    const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = index, .rhs = limit } });
    try func.appendIf(header, more, .{ .target = body }, .{ .target = done });
    const contribution = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = index, .rhs = limit } });
    try func.setJump(body, latch, &.{});
    const next = try func.appendInst(latch, i32_t, .{ .arith_imm = .{ .op = .add, .lhs = index, .imm = 1 } });
    try func.setJump(latch, header, &.{next});
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(index) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockParams(body).len);
    try std.testing.expectEqual(@as(usize, 1), func.blockParams(done).len);
    try std.testing.expect(func.terminator(latch) == null);
    const latch_insts = func.blockInsts(latch);
    try std.testing.expect(func.opcode(latch_insts[latch_insts.len - 1]) == .@"if");
    try std.testing.expectEqual(func.blockParams(body)[0], func.opcode(func.definingInst(contribution).?).arith.lhs);

    var diagnostics = try ir.verify.verify(allocator, &func, .high);
    defer diagnostics.deinit();
    try std.testing.expect(diagnostics.ok());
}
