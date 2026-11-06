//! Recognize the unsigned Kernighan loop x &= x - 1; ++count.
//! Replace the loop with a fixed-cost parallel bit count and jump to its exit.
//! This portable form also works on backends without a native population count.
const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;
const BinOp = ir.function.BinOp;
pub const pass_def = pass.Pass{ .name = "popcount-loop", .run = run };

fn constant(f: *const Function, v: Value, n: i64) bool {
    const i = f.definingInst(v) orelse return false;
    return switch (f.opcode(i)) {
        .iconst => |c| c == n,
        else => false,
    };
}
fn increment(f: *const Function, v: Value, base: Value, op: BinOp) bool {
    const i = f.definingInst(v) orelse return false;
    return switch (f.opcode(i)) {
        .arith => |a| a.op == op and a.lhs == base and constant(f, a.rhs, 1),
        .arith_imm => |a| a.op == op and a.lhs == base and a.imm == 1,
        else => false,
    };
}
fn emit(f: *Function, b: Block, t: Type, op: BinOp, x: Value, k: i64) !Value {
    const c = try f.appendInst(b, t, .{ .iconst = k });
    return f.appendInst(b, t, .{ .arith = .{ .op = op, .lhs = x, .rhs = c } });
}
fn binary(f: *Function, b: Block, t: Type, op: BinOp, x: Value, y: Value) !Value {
    return f.appendInst(b, t, .{ .arith = .{ .op = op, .lhs = x, .rhs = y } });
}
pub fn run(allocator: std.mem.Allocator, f: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const cfg = try analyses.cfg();
    for (0..f.blockCount()) |bi| {
        const h: Block = @fromBackingInt(@intCast(bi));
        const params = f.blockParams(h);
        if (params.len != 2) continue;
        const insts = f.blockInsts(h);
        if (insts.len != 2) continue;
        const branch = switch (f.opcode(insts[1])) {
            .@"if" => |v| v,
            else => continue,
        };
        const cmp = switch (f.opcode(insts[0])) {
            .icmp => |v| v,
            else => continue,
        };
        if (f.instResult(insts[0]) != branch.cond or cmp.op != .ne or !constant(f, cmp.rhs, 0)) continue;
        const xi: usize = if (cmp.lhs == params[0]) 0 else if (cmp.lhs == params[1]) 1 else continue;
        const x = params[xi];
        const count = params[1 - xi];
        const t = f.valueType(x);
        if (f.valueType(count) != t) continue;
        const int = switch (f.types.type_kind(t)) {
            .int => |v| v,
            else => continue,
        };
        if (int.signedness != .unsigned or (int.bits != 32 and int.bits != 64)) continue;
        const b = branch.then.target;
        if (b == h or branch.@"else".target == b or f.blockParams(b).len != 0 or f.blockArgs(branch.then).len != 0) continue;
        const preds = cfg.predecessors(@backingInt(b));
        if (preds.len != 1 or preds[0] != bi) continue;
        const jump = switch (f.terminator(b) orelse continue) {
            .jump => |v| v,
            else => continue,
        };
        if (jump.target != h or f.blockInsts(b).len != 3) continue;
        const args = f.blockArgs(jump);
        if (args.len != 2 or !increment(f, args[1 - xi], count, .add)) continue;
        const and_inst = f.definingInst(args[xi]) orelse continue;
        const a = switch (f.opcode(and_inst)) {
            .arith => |v| v,
            else => continue,
        };
        if (a.op != .bit_and) continue;
        const dec = if (a.lhs == x) a.rhs else if (a.rhs == x) a.lhs else continue;
        if (!increment(f, dec, x, .sub)) continue;
        // All three body instructions must be precisely the matched recurrence;
        // in particular there must be no memory effects or unrelated live values.
        var matched = true;
        for (f.blockInsts(b)) |i| {
            const r = f.instResult(i) orelse {
                matched = false;
                break;
            };
            if (r != dec and r != args[xi] and r != args[1 - xi]) matched = false;
        }
        if (!matched) continue;
        // The header compare is discarded. A caller may also pass that bool
        // on an edge or return it, which this narrow idiom does not rewrite.
        const uses = try allocator.alloc(u32, f.valueCount());
        defer allocator.free(uses);
        @memset(uses, 0);
        @import("dce.zig").countUses(f, uses);
        if (uses[@backingInt(branch.cond)] != 1) continue;
        const exit_args = try allocator.dupe(Value, f.blockArgs(branch.@"else"));
        defer allocator.free(exit_args);
        try f.setBlockInsts(b, &.{});
        f.setTerminator(b, .{ .ret = ir.function.Ret.none() });
        try f.setBlockInsts(h, &.{});
        const dest = h;
        const wide = int.bits == 64;
        const shifted = try emit(f, dest, t, .shr, x, 1);
        const masked = try emit(f, dest, t, .bit_and, shifted, if (wide) 0x5555555555555555 else 0x55555555);
        var v = try binary(f, dest, t, .sub, x, masked);
        const lo = try emit(f, dest, t, .bit_and, v, if (wide) 0x3333333333333333 else 0x33333333);
        const hi = try emit(f, dest, t, .bit_and, try emit(f, dest, t, .shr, v, 2), if (wide) 0x3333333333333333 else 0x33333333);
        v = try binary(f, dest, t, .add, lo, hi);
        v = try binary(f, dest, t, .add, v, try emit(f, dest, t, .shr, v, 4));
        v = try emit(f, dest, t, .bit_and, v, if (wide) 0x0f0f0f0f0f0f0f0f else 0x0f0f0f0f);
        v = try emit(f, dest, t, .mul, v, if (wide) 0x0101010101010101 else 0x01010101);
        v = try emit(f, dest, t, .shr, v, if (wide) 56 else 24);
        const total = try binary(f, dest, t, .add, count, v);
        const zero = try f.appendInst(dest, t, .{ .iconst = 0 });
        for (0..f.blockCount()) |other| {
            const ob: Block = @fromBackingInt(@intCast(other));
            if (ob == h or ob == b) continue;
            f.replaceUsesIn(ob, count, total);
            f.replaceUsesIn(ob, x, zero);
        }
        for (exit_args) |*arg| {
            if (arg.* == count) arg.* = total else if (arg.* == x) arg.* = zero;
        }
        try f.setJump(h, branch.@"else".target, exit_args);
        analyses.invalidate();
        return true;
    }
    return false;
}

test "Kernighan recurrence agrees with population count for both unsigned widths" {
    const testing = std.testing;
    for ([_]u16{ 32, 64 }) |bits| {
        const allocator = testing.allocator;
        var f = Function.init(allocator);
        defer f.deinit();
        const t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = bits } });
        const bt = try f.types.intern(.bool);
        const entry = try f.appendBlock();
        const h = try f.appendBlock();
        const b = try f.appendBlock();
        const exit = try f.appendBlock();
        const input = try f.appendBlockParam(entry, t);
        const initial = try f.appendBlockParam(entry, t);
        const zero = try f.appendInst(entry, t, .{ .iconst = 0 });
        const one = try f.appendInst(entry, t, .{ .iconst = 1 });
        try f.setJump(entry, h, &.{ input, initial });
        const x = try f.appendBlockParam(h, t);
        const count = try f.appendBlockParam(h, t);
        const cond = try f.appendInst(h, bt, .{ .icmp = .{ .op = .ne, .lhs = x, .rhs = zero } });
        try f.appendIf(h, cond, .{ .target = b }, .{ .target = exit });
        f.setTerminator(h, .{ .ret = ir.function.Ret.none() });
        const dec = try binary(&f, b, t, .sub, x, one);
        const cleared = try binary(&f, b, t, .bit_and, x, dec);
        const next = try emit(&f, b, t, .add, count, 1);
        // The canonicalizer has already folded the increment's constant.
        const ni = f.definingInst(next).?;
        f.opcodeMut(ni).* = .{ .arith_imm = .{ .op = .add, .lhs = count, .imm = 1 } };
        const old = try allocator.dupe(ir.function.Inst, f.blockInsts(b));
        defer allocator.free(old);
        try f.setBlockInsts(b, &.{ old[0], old[1], old[3] });
        try f.setJump(b, h, &.{ cleared, next });
        f.setTerminator(exit, .{ .ret = ir.function.Ret.one(count) });
        var analyses = pass.Analyses{ .allocator = allocator, .func = &f };
        defer analyses.deinit();
        f.opcodeMut(ni).arith_imm.imm = 2;
        try testing.expect(!try run(allocator, &f, &analyses));
        f.opcodeMut(ni).arith_imm.imm = 1;
        try testing.expect(try run(allocator, &f, &analyses));
        try testing.expect(!try run(allocator, &f, &analyses));
        var diags = try ir.verify.verify(allocator, &f, .high);
        defer diags.deinit();
        try testing.expect(diags.ok());
        const values = try allocator.alloc(u64, f.valueCount());
        defer allocator.free(values);
        const mask: u64 = if (bits == 64) std.math.maxInt(u64) else std.math.maxInt(u32);
        var random: u64 = 1;
        for (0..10000) |sample| {
            random = random *% 6364136223846793005 +% 1;
            const n = (if (sample < 64) @as(u64, 1) << @intCast(sample) else if (sample == 64) 0 else if (sample == 65) mask else random) & mask;
            const start = random & mask;
            values[@backingInt(x)] = n;
            values[@backingInt(count)] = start;
            for (f.blockInsts(h)) |inst| {
                const result = f.instResult(inst).?;
                values[@backingInt(result)] = switch (f.opcode(inst)) {
                    .iconst => |k| @as(u64, @bitCast(k)),
                    .arith => |a| eval(a.op, values[@backingInt(a.lhs)], values[@backingInt(a.rhs)]),
                    else => unreachable,
                } & mask;
            }
            const result = f.terminator(exit).?.ret.values[0];
            try testing.expectEqual((start +% @as(u64, @popCount(n))) & mask, values[@backingInt(result)]);
        }
    }
}
fn eval(op: BinOp, x: u64, y: u64) u64 {
    return switch (op) {
        .add => x +% y,
        .sub => x -% y,
        .mul => x *% y,
        .bit_and => x & y,
        .shr => x >> @intCast(y),
        else => unreachable,
    };
}
