//! Rotate a canonical top-tested loop so its recurring test runs in the latch.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");
const dce = @import("../dce.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;

pub const pass_def = pass.Pass{ .name = "looprotate", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const graph = try analyses.cfg();
    const doms = try analyses.dominators();
    const loop_info = try analyses.loops();
    const def_blocks = try analyses.definingBlocks();
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);

    for (loop_info.loops) |loop| {
        const header: Block = @fromBackingInt(@intCast(loop.header));
        const header_insts = func.blockInsts(header);
        if (header_insts.len != 2) continue;
        // C lowering leaves an unreachable `ret void` after an `if`. Its two
        // edges transfer control before that return, so it is still a canonical
        // test header. A real jump or value-return is outside this shape.
        if (func.terminator(header)) |term| switch (term) {
            .ret => |ret| if (ret.count != 0) continue,
            .jump => continue,
        };

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
        dce.countUses(func, uses);
        if (uses[@backingInt(compare_result)] != 1) continue;

        const body = branch.then.target;
        const exit = branch.@"else".target;
        if (!loop.contains(@backingInt(body)) or loop.contains(@backingInt(exit))) continue;
        if (graph.predecessors(@backingInt(body)).len != 1 or graph.predecessors(@backingInt(body))[0] != loop.header) continue;
        // A nested loop may continue after its exit, and a body guard may
        // reach the same exit. Every such edge must carry the current values.
        var exit_is_local = true;
        for (graph.predecessors(@backingInt(exit))) |pred| {
            if (!loop.contains(pred)) exit_is_local = false;
        }
        if (!exit_is_local) continue;

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
        // Uses reached through some other exit cannot be repaired by params
        // on this exit. Refuse those shapes rather than leave stale values.
        @memset(uses, 0);
        for (0..func.blockCount()) |bi| {
            if (loop.contains(bi) or doms.dominates(@backingInt(exit), bi)) continue;
            dce.countBlockUses(func, @fromBackingInt(@intCast(bi)), uses);
        }
        var escaping_elsewhere = false;
        for (header_params) |param| if (uses[@backingInt(param)] != 0) {
            escaping_elsewhere = true;
        };
        if (escaping_elsewhere) continue;
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
        for (0..func.blockCount()) |bi| {
            if (loop.contains(bi) or !doms.dominates(@backingInt(exit), bi)) continue;
            for (header_params, exit_params) |param, replacement|
                func.replaceUsesIn(@fromBackingInt(@intCast(bi)), param, replacement);
        }

        // A body-side early exit observes the current iteration, while the
        // latch exit below observes its updated back-edge values.
        for (loop.body, 0..) |inside, bi| {
            if (!inside or bi == loop.header) continue;
            const block: Block = @fromBackingInt(@intCast(bi));
            for (func.blockInsts(block)) |inst| {
                var cf = switch (func.opcode(inst)) {
                    .@"if" => |cf| cf,
                    else => continue,
                };
                if (cf.then.target == exit) cf.then.args = try appendArgs(allocator, func, cf.then, body_params);
                if (cf.@"else".target == exit) cf.@"else".args = try appendArgs(allocator, func, cf.@"else", body_params);
                func.opcodeMut(inst).* = .{ .@"if" = cf };
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |jump| if (jump.target == exit) {
                    var updated = jump;
                    updated.args = try appendArgs(allocator, func, jump, body_params);
                    func.setTerminator(block, .{ .jump = updated });
                },
                .ret => {},
            };
        }

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

fn appendArgs(allocator: std.mem.Allocator, func: *Function, jump: ir.function.Jump, extra: []const Value) !ir.function.ValueList {
    const args = try concat(allocator, func.blockArgs(jump), extra);
    defer allocator.free(args);
    return func.internValues(args);
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
    func.setTerminator(header, .{ .ret = ir.function.Ret.none() });
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

test "rotation preserves early exits and values used after a nested loop" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bt = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const body = try f.appendBlock();
    const latch = try f.appendBlock();
    const done = try f.appendBlock();
    const after = try f.appendBlock();
    const n = try f.appendBlockParam(entry, t);
    const stop = try f.appendBlockParam(entry, t);
    const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
    const i = try f.appendBlockParam(header, t);
    try f.setJump(entry, header, &.{zero});
    const more = try f.appendInst(header, bt, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try f.appendIf(header, more, .{ .target = body }, .{ .target = done });
    const early = try f.appendInst(body, bt, .{ .icmp = .{ .op = .eq, .lhs = i, .rhs = stop } });
    try f.appendIf(body, early, .{ .target = done }, .{ .target = latch });
    const next = try f.appendArithImm(latch, t, .add, i, 1);
    try f.setJump(latch, header, &.{next});
    try f.setJump(done, after, &.{});
    const result = try f.appendArithImm(after, t, .add, i, 10);
    f.setTerminator(after, .{ .ret = ir.function.Ret.one(result) });
    var original = try f.clone(allocator);
    defer original.deinit();
    try std.testing.expect(try pass.run(allocator, &f, &.{pass_def}));
    var diags = try ir.verify.verify(allocator, &f, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
    for ([_]i64{ -1, 0, 1, 2, 7, 20 }) |limit| {
        for ([_]i64{ -1, 0, 1, 3, 19, 30 }) |stop_at| {
            try std.testing.expectEqual(try evalRotation(&original, &.{ limit, stop_at }), try evalRotation(&f, &.{ limit, stop_at }));
        }
    }
}

fn evalRotation(f: *const Function, inputs: []const i64) !i64 {
    const allocator = std.testing.allocator;
    const values = try allocator.alloc(i64, f.valueCount());
    defer allocator.free(values);
    const pending = try allocator.alloc(i64, f.valueCount());
    defer allocator.free(pending);
    for (f.blockParams(@fromBackingInt(0)), inputs) |param, input| values[@backingInt(param)] = input;
    var block: Block = @fromBackingInt(0);
    for (0..1000) |_| {
        var edge: ?ir.function.Jump = null;
        for (f.blockInsts(block)) |inst| {
            const op = f.opcode(inst);
            if (op == .@"if") {
                edge = if (values[@backingInt(op.@"if".cond)] != 0) op.@"if".then else op.@"if".@"else";
                break;
            }
            const result = f.instResult(inst).?;
            values[@backingInt(result)] = switch (op) {
                .iconst => |c| c,
                .arith_imm => |a| values[@backingInt(a.lhs)] + a.imm,
                .icmp => |c| switch (c.op) {
                    .lt => @intFromBool(values[@backingInt(c.lhs)] < values[@backingInt(c.rhs)]),
                    .eq => @intFromBool(values[@backingInt(c.lhs)] == values[@backingInt(c.rhs)]),
                    else => unreachable,
                },
                else => unreachable,
            };
        }
        if (edge == null) switch (f.terminator(block).?) {
            .ret => |r| return values[@backingInt(r.values[0])],
            .jump => |jump| edge = jump,
        };
        const jump = edge.?;
        const args = f.blockArgs(jump);
        for (args, 0..) |arg, index| pending[index] = values[@backingInt(arg)];
        for (f.blockParams(jump.target), pending[0..args.len]) |param, arg| values[@backingInt(param)] = arg;
        block = jump.target;
    }
    return error.TestUnexpectedResult;
}
