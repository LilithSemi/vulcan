//! Immediate-form canonicalization: an `arith` with one integer-constant operand is rewritten into
//! the `arith_imm` form (the constant carried in the instruction, not in a separate `iconst`). This
//! lets the backend fold a constant address offset straight into a load or store (`ldr [base, #off]`
//! instead of materializing the offset and adding it), and lets it pick an immediate-form arithmetic
//! instruction. The rewrite is a pure representation change: `arith_imm{op, x, c}` computes exactly
//! what `arith{op, x, (iconst c)}` did. The now-orphaned `iconst` is left for DCE.
//!
//! Only add is canonicalized here. Mul/div/rem are left to strength reduction, while the other
//! operations retain their register form for folds that specifically inspect it. Add is commutative,
//! so the constant may sit on either side. An instruction with two constant operands is skipped, so
//! constant folding collapses it to a single `iconst` instead.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const BinOp = ir.function.BinOp;

pub const pass_def = pass.Pass{ .name = "immform", .run = run, .preserves_cfg = true, .preserves_value_defs = true };

/// Whether `op` has an `arith_imm` form this pass emits, and whether it is commutative (so the
/// constant may sit on the left). `add` is canonicalized because backends fold it into either a
/// load/store address or an immediate-form arithmetic instruction. The other ops (sub, bitwise, shift)
/// gain nothing from the `arith_imm` form here (the backend materializes their immediate either way)
/// and canonicalizing them only disturbs folds that read the `arith` form, so they are left alone.
fn immOp(op: BinOp) ?struct { commutative: bool } {
    return switch (op) {
        .add => .{ .commutative = true },
        else => null,
    };
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    if (!hasImmediateCandidate(func)) return false;

    // Map each value defined by an `iconst` to its constant. Iterate LIVE block instructions, not the
    // pool, so a dropped instruction is not re-processed (matching the other passes).
    var consts = try allocator.alloc(?i64, func.valueCount());
    defer allocator.free(consts);
    @memset(consts, null);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            if (func.opcode(inst) == .iconst) {
                if (func.instResult(inst)) |r| consts[@backingInt(r)] = func.opcode(inst).iconst;
            }
        }
    }

    // Only a SINGLE-use constant is folded (a constant shared by several `arith` would be
    // re-materialized inside each `arith_imm`, a loss over the one shared register the `iconst` gives).
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    @memset(uses, 0);
    dce.countUses(func, uses);

    var changed = false;
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const a = switch (func.opcode(inst)) {
                .arith => |a| a,
                else => continue,
            };
            const info = immOp(a.op) orelse continue;
            const lc = consts[@backingInt(a.lhs)];
            const rc = consts[@backingInt(a.rhs)];
            if (lc != null and rc != null) continue; // both constant: leave it for constant folding
            if (rc) |c| {
                if (uses[@backingInt(a.rhs)] != 1) continue;
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = a.op, .lhs = a.lhs, .imm = c } };
                changed = true;
            } else if (info.commutative) if (lc) |c| {
                if (uses[@backingInt(a.lhs)] != 1) continue;
                func.opcodeMut(inst).* = .{ .arith_imm = .{ .op = a.op, .lhs = a.rhs, .imm = c } };
                changed = true;
            };
        }
    }
    return changed;
}

fn hasImmediateCandidate(func: *const Function) bool {
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const a = switch (func.opcode(inst)) {
                .arith => |a| if (immOp(a.op) != null) a else continue,
                else => continue,
            };
            for ([_]Value{ a.lhs, a.rhs }) |operand| {
                const def = func.definingInst(operand) orelse continue;
                if (func.opcode(def) == .iconst) return true;
            }
        }
    }
    return false;
}

test "single-use constant adds become arith_imm" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    // An ADDRESS add: base + 40, its result used as a load pointer -> canonicalized.
    const off = try func.appendInst(b, i32_t, .{ .iconst = 40 });
    const addr = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off } });
    const loaded = try func.appendInst(b, i32_t, .{ .load = .{ .ptr = addr } });
    // A plain add also benefits: its backend immediate no longer consumes a register.
    const seven = try func.appendInst(b, i32_t, .{ .iconst = 7 });
    const plain = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = seven } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = loaded, .rhs = plain } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // The address add carries its offset inline now.
    const a2 = func.opcode(func.definingInst(addr).?).arith_imm;
    try std.testing.expectEqual(@as(i64, 40), a2.imm);
    try std.testing.expectEqual(base, a2.lhs);
    const p2 = func.opcode(func.definingInst(plain).?).arith_imm;
    try std.testing.expectEqual(@as(i64, 7), p2.imm);
    try std.testing.expectEqual(x, p2.lhs);
}

test "a plain constant add needs no memory use" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const c = try func.appendInst(b, i32_t, .{ .iconst = 8 });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = c } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(hasImmediateCandidate(&func));
    try std.testing.expect(try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(i64, 8), func.opcode(func.definingInst(sum).?).arith_imm.imm);
}

test "a shared constant remains register form" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const y = try func.appendBlockParam(b, i32_t);
    const c = try func.appendInst(b, i32_t, .{ .iconst = 8 });
    const xc = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = c } });
    const yc = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = c } });
    const sum = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = xc, .rhs = yc } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expect(func.opcode(func.definingInst(xc).?) == .arith);
    try std.testing.expect(func.opcode(func.definingInst(yc).?) == .arith);
}

test {
    std.testing.refAllDecls(@This());
}
