//! Combine two single-use constant-offset values before instruction selection.
//!
//! `(x + A) + (y + B)` becomes `(x + y) + (A + B)`. Subtract-immediate source forms are treated as
//! negative offsets. The AArch64 IR score gate owns profitability; this pass only proves the integer
//! reassociation and exposes one fewer arithmetic instruction.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");
const dce = @import("../dce.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "offsetcombine", .run = run, .preserves_cfg = true };

const Offset = struct {
    base: Value,
    amount: i64,
    inst: Inst,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    @memset(uses, 0);
    dce.countUses(func, uses);

    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, parent_index| {
            const result = func.instResult(inst) orelse continue;
            if (func.types.type_kind(func.valueType(result)) != .int) continue;
            const sum = switch (func.opcode(inst)) {
                .arith => |a| if (a.op == .add) a else continue,
                else => continue,
            };
            const lhs = offsetOf(func, sum.lhs) orelse continue;
            const rhs = offsetOf(func, sum.rhs) orelse continue;
            const lhs_index = std.mem.indexOfScalar(Inst, insts, lhs.inst) orelse continue;
            const rhs_index = std.mem.indexOfScalar(Inst, insts, rhs.inst) orelse continue;
            if (lhs_index >= parent_index or rhs_index >= parent_index) continue;
            if (func.valueType(lhs.base) != func.valueType(result) or func.valueType(rhs.base) != func.valueType(result)) continue;
            if (uses[@backingInt(sum.lhs)] != 1 or uses[@backingInt(sum.rhs)] != 1) continue;
            const amount = std.math.add(i64, lhs.amount, rhs.amount) catch continue;

            func.opcodeMut(lhs.inst).* = .{ .arith = .{ .op = .add, .lhs = lhs.base, .rhs = rhs.base } };
            if (amount == 0) {
                func.replaceAllUses(result, sum.lhs);
            } else {
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = .add, .lhs = sum.lhs, .imm = amount } };
            }
            changed = true;
        }
    }
    return changed;
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
