//! Combine two single-use constant-offset values before instruction selection.
//!
//! `(x + A) + (y + B)` becomes `(x + y) + (A + B)`. Subtract-immediate source forms are treated as
//! negative offsets. The AArch64 IR score gate owns profitability; this pass only proves the integer
//! reassociation and exposes one fewer arithmetic instruction.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");
const dce = @import("../dce.zig");
const Dominators = @import("../dominators.zig").Dominators;

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "offsetcombine", .run = run, .preserves_cfg = true, .preserves_value_defs = true };

const Offset = struct {
    base: Value,
    amount: i64,
    inst: Inst,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    @memset(uses, 0);
    dce.countUses(func, uses);
    const def_blocks = try analyses.definingBlocks();
    const doms = try analyses.dominators();

    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        for (insts) |inst| {
            const result = func.instResult(inst) orelse continue;
            if (func.types.type_kind(func.valueType(result)) != .int) continue;
            const sum = switch (func.opcode(inst)) {
                .arith => |a| if (a.op == .add) a else continue,
                else => continue,
            };
            const lhs = offsetOf(func, sum.lhs) orelse continue;
            const rhs = offsetOf(func, sum.rhs) orelse continue;
            if (func.valueType(lhs.base) != func.valueType(result) or func.valueType(rhs.base) != func.valueType(result)) continue;
            if (uses[@backingInt(sum.lhs)] != 1 or uses[@backingInt(sum.rhs)] != 1) continue;
            const amount = std.math.add(i64, lhs.amount, rhs.amount) catch continue;
            const rewrite, const combined = if (canRewriteAt(func, def_blocks, doms, lhs, rhs.base))
                .{ lhs, sum.lhs }
            else if (canRewriteAt(func, def_blocks, doms, rhs, lhs.base))
                .{ rhs, sum.rhs }
            else
                continue;

            func.opcodeMut(rewrite.inst).* = .{ .arith = .{ .op = .add, .lhs = lhs.base, .rhs = rhs.base } };
            if (amount == 0) {
                func.replaceAllUses(result, combined);
            } else {
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = .add, .lhs = combined, .imm = amount } };
            }
            changed = true;
        }
    }
    return changed;
}

fn canRewriteAt(func: *const Function, def_blocks: []const u32, doms: *const Dominators, offset: Offset, other_base: Value) bool {
    const result = func.instResult(offset.inst) orelse return false;
    const block_index = def_blocks[@backingInt(result)];
    const block: Block = @fromBackingInt(block_index);
    const insts = func.blockInsts(block);
    const index = std.mem.indexOfScalar(Inst, insts, offset.inst) orelse return false;
    return availableBefore(func, def_blocks, doms, offset.base, block_index, insts, index) and
        availableBefore(func, def_blocks, doms, other_base, block_index, insts, index);
}

fn availableBefore(func: *const Function, def_blocks: []const u32, doms: *const Dominators, value: Value, block_index: u32, insts: []const Inst, index: usize) bool {
    const def_block = def_blocks[@backingInt(value)];
    if (def_block != block_index) return doms.dominates(def_block, block_index);
    const def_inst = func.definingInst(value) orelse return true;
    const def_index = std.mem.indexOfScalar(Inst, insts, def_inst) orelse return false;
    return def_index < index;
}

fn offsetOf(func: *const Function, value: Value) ?Offset {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .arith_imm => |a| switch (a.op) {
            .add => .{ .base = a.lhs, .amount = a.imm, .inst = inst },
            .sub => .{ .base = a.lhs, .amount = std.math.negate(a.imm) catch return null, .inst = inst },
            else => null,
        },
        .arith => |a| switch (a.op) {
            .add => if (constantOf(func, a.rhs)) |c|
                .{ .base = a.lhs, .amount = c, .inst = inst }
            else if (constantOf(func, a.lhs)) |c|
                .{ .base = a.rhs, .amount = c, .inst = inst }
            else
                null,
            .sub => if (constantOf(func, a.rhs)) |c|
                .{ .base = a.lhs, .amount = std.math.negate(c) catch return null, .inst = inst }
            else
                null,
            else => null,
        },
        else => null,
    };
}

fn constantOf(func: *const Function, value: Value) ?i64 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |constant| constant,
        else => null,
    };
}

test "combines two subtract offsets" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const block = try func.appendBlock();
    const x = try func.appendBlockParam(block, ty);
    const y = try func.appendBlockParam(block, ty);
    const ca = try func.appendInst(block, ty, .{ .iconst = 1024 });
    const cb = try func.appendInst(block, ty, .{ .iconst = 512 });
    const a = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = ca } });
    const b = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = y, .rhs = cb } });
    const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    const base_sum = func.opcode(func.definingInst(a).?).arith;
    try std.testing.expectEqual(x, base_sum.lhs);
    try std.testing.expectEqual(y, base_sum.rhs);
    const combined = func.opcode(func.definingInst(sum).?).arith_imm;
    try std.testing.expectEqual(a, combined.lhs);
    try std.testing.expectEqual(@as(i64, -1536), combined.imm);
}

test "combines offsets before a dominated merge" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const merge = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ty);
    const mask = try func.appendBlockParam(entry, ty);
    const ca = try func.appendInst(entry, ty, .{ .iconst = 1024 });
    const cb = try func.appendInst(entry, ty, .{ .iconst = 512 });
    const a = try func.appendInst(entry, ty, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = ca } });
    const y = try func.appendInst(entry, ty, .{ .arith = .{ .op = .bit_xor, .lhs = x, .rhs = mask } });
    const b = try func.appendInst(entry, ty, .{ .arith = .{ .op = .sub, .lhs = y, .rhs = cb } });
    try func.setJump(entry, merge, &.{});
    const sum = try func.appendInst(merge, ty, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } });
    func.setTerminator(merge, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    const base_sum = func.opcode(func.definingInst(b).?).arith;
    try std.testing.expectEqual(x, base_sum.lhs);
    try std.testing.expectEqual(y, base_sum.rhs);
    const combined = func.opcode(func.definingInst(sum).?).arith_imm;
    try std.testing.expectEqual(b, combined.lhs);
    try std.testing.expectEqual(@as(i64, -1536), combined.imm);
}
