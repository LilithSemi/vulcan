//! Sink a constant to its use.
//!
//! A frontend with nowhere natural to put a constant puts it at the top of a block: a SPIR-V
//! module-scope `OpConstant` becomes an `iconst` in the entry block, so a kernel that writes a
//! table of 64 of them has 64 values live at once. A target with a small register file then spills
//! every one, which costs a store and a reload where re-emitting the constant costs one
//! instruction. On a target that cannot spill far it is a refusal rather than slow code.
//!
//! This moves a constant that is read ONCE to just before the instruction that reads it, when both
//! are in one block, which shortens its live range to a single instruction. A constant reads no
//! operand, so moving it later extends nothing: this cannot make a function worse.
//!
//! A constant read more than once stays where it is. Copying one to each reader is
//! rematerialisation, which is a bigger change and belongs in the allocator.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    if (func.blockCount() == 0) return false;

    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    dce.countUses(func, uses);

    // Per block: whether a value is a constant read once here, and which instruction reads it.
    const candidate = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(candidate);
    const reader = try allocator.alloc(?Inst, func.valueCount());
    defer allocator.free(reader);

    // A chain of sunk constants per reader, so the rebuild below stays one pass.
    const first = try allocator.alloc(?Inst, func.instCount());
    defer allocator.free(first);
    const next = try allocator.alloc(?Inst, func.instCount());
    defer allocator.free(next);

    var order: std.ArrayList(Inst) = .empty;
    defer order.deinit(allocator);

    var moved = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        @memset(candidate, false);
        @memset(reader, null);

        var any = false;
        for (func.blockInsts(block)) |inst| {
            if (!isConstant(func, inst)) continue;
            const result = func.instResult(inst) orelse continue;
            if (uses[@backingInt(result)] != 1) continue;
            candidate[@backingInt(result)] = true;
            any = true;
        }
        if (!any) continue;

        for (func.blockInsts(block)) |inst| noteReads(func, inst, candidate, reader);

        // Chain each sunk constant onto the instruction that reads it. A candidate whose one use is
        // a terminator or an edge argument has no reader here and stays where it is.
        @memset(first, null);
        var sinking = false;
        for (func.blockInsts(block)) |inst| {
            if (!isConstant(func, inst)) continue;
            const result = func.instResult(inst) orelse continue;
            if (!candidate[@backingInt(result)]) continue;
            const at = reader[@backingInt(result)] orelse continue;
            next[@backingInt(inst)] = first[@backingInt(at)];
            first[@backingInt(at)] = inst;
            sinking = true;
        }
        if (!sinking) continue;

        order.clearRetainingCapacity();
        for (func.blockInsts(block)) |inst| {
            var chained = first[@backingInt(inst)];
            while (chained) |c| {
                try order.append(allocator, c);
                chained = next[@backingInt(c)];
            }
            const result = func.instResult(inst);
            const sunk = isConstant(func, inst) and result != null and
                candidate[@backingInt(result.?)] and reader[@backingInt(result.?)] != null;
            if (!sunk) try order.append(allocator, inst);
        }

        const insts = func.blockInstsMut(block);
        std.debug.assert(order.items.len == insts.items.len);
        if (!std.mem.eql(Inst, insts.items, order.items)) moved = true;
        @memcpy(insts.items, order.items);
    }
    return moved;
}

/// Whether `inst` materializes a constant, which reads no value and so may move freely later.
fn isConstant(func: *const Function, inst: Inst) bool {
    return switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128 => true,
        else => false,
    };
}

/// Record, for each candidate `inst` reads, that `inst` is its reader.
///
/// This mirrors `dce.countUses`. An opcode left out would leave a candidate with no reader, so it
/// simply does not move; what must never happen is naming an instruction that does NOT read the
/// value, which would sink the constant past its real use.
fn noteReads(func: *const Function, inst: Inst, candidate: []const bool, reader: []?Inst) void {
    const note = struct {
        fn f(cand: []const bool, out: []?Inst, at: Inst, v: Value) void {
            if (cand[@backingInt(v)]) out[@backingInt(v)] = at;
        }
    }.f;
    switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
        .arith => |a| {
            note(candidate, reader, inst, a.lhs);
            note(candidate, reader, inst, a.rhs);
        },
        .arith_imm => |a| note(candidate, reader, inst, a.lhs),
        .icmp => |c| {
            note(candidate, reader, inst, c.lhs);
            note(candidate, reader, inst, c.rhs);
        },
        .select => |s| {
            note(candidate, reader, inst, s.cond);
            note(candidate, reader, inst, s.then);
            note(candidate, reader, inst, s.@"else");
        },
        .extract => |e| note(candidate, reader, inst, e.aggregate),
        .convert => |cv| note(candidate, reader, inst, cv.value),
        .decode_low_float, .encode_low_float => |cv| note(candidate, reader, inst, cv.value),
        .unary => |u| note(candidate, reader, inst, u.value),
        .load => |l| note(candidate, reader, inst, l.ptr),
        .store => |st| {
            note(candidate, reader, inst, st.value);
            note(candidate, reader, inst, st.ptr);
            if (st.mask) |m| note(candidate, reader, inst, m);
        },
        .prefetch => |pf| note(candidate, reader, inst, pf.ptr),
        .dot => |d| {
            note(candidate, reader, inst, d.acc);
            note(candidate, reader, inst, d.a);
            note(candidate, reader, inst, d.b);
        },
        .reduce => |red| note(candidate, reader, inst, red.vector),
        .splat => |sp| note(candidate, reader, inst, sp.scalar),
        .atomic_rmw => |a| {
            note(candidate, reader, inst, a.ptr);
            note(candidate, reader, inst, a.value);
            if (a.compare) |c| note(candidate, reader, inst, c);
        },
        // Everything else keeps its constants where they are, which is always correct.
        else => {},
    }
}

test "a constant read once moves to its reader" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const base = try func.appendGlobalAddr(entry, ptr_t, "data");
    // Three constants up front, then the three stores that read them.
    var consts: [3]Value = undefined;
    for (&consts, 0..) |*c, i| c.* = try func.appendInst(entry, i32_t, .{ .iconst = @intCast(10 + i) });
    for (consts, 0..) |c, i| {
        const at = try func.appendArithImm(entry, ptr_t, .add, base, @intCast(i * 4));
        try func.appendStore(entry, c, at);
    }
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    var analyses: pass.Analyses = .{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // Each constant now sits directly in front of the store that reads it, so only one is live at
    // a time. The order is: base, (addr, const, store) three times.
    const insts = func.blockInsts(entry);
    try std.testing.expectEqual(@as(usize, 10), insts.len);
    for (0..3) |i| {
        const c = insts[1 + i * 3 + 1];
        try std.testing.expect(func.opcode(c) == .iconst);
        try std.testing.expect(func.opcode(insts[1 + i * 3 + 2]) == .store);
    }

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expectEqualSlices(ir.verify.Diagnostic, &.{}, diags.items());
}

test "a constant read twice stays where it is" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();

    const entry = try func.appendBlock();
    const base = try func.appendGlobalAddr(entry, ptr_t, "data");
    const shared = try func.appendInst(entry, i32_t, .{ .iconst = 7 });
    // Something that is not a constant sits between the definition and the readers, so sinking
    // would visibly move it if the pass were willing to.
    const at = try func.appendArithImm(entry, ptr_t, .add, base, 4);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = shared, .rhs = shared } });
    try func.appendStore(entry, sum, at);
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    var analyses: pass.Analyses = .{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    // Copying it to each reader is rematerialisation, which this does not do.
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(shared, func.instResult(func.blockInsts(entry)[1]).?);
}
