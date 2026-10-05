//! Dead-code elimination: remove pure instructions whose result is never used,
//! iterating to a fixpoint (removing one dead value can make its operands dead).
//! Impure instructions (loads, stores, calls, `if`) are always kept.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;

pub const pass_def = pass.Pass{ .name = "dce", .run = run };

/// Whether an instruction has no side effects, so it may be dropped when unused.
fn isPure(op: ir.function.Opcode) bool {
    return switch (op) {
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .struct_new, .extract, .convert, .decode_low_float, .encode_low_float, .dequantize_nvfp4, .quantize_nvfp4, .unary, .alloca, .global_addr, .dot, .reduce, .splat => true,
        // A prefetch hint has no result but must be kept, like a store. A
        // matmul writes the `c` memory, likewise kept.
        .load, .store, .prefetch, .matmul, .@"if", .call, .call_indirect => false,
        // SM12 T3: mutate/read the `va_list` object at `list`, like `load`/`store` above.
        .va_start, .va_arg, .va_end => false,
        // An atomic is a STORE as well as a load. Deleting one whose old value nobody
        // reads loses the write, so the reduction form is kept as firmly as the reading
        // form. This is the guard the optional result makes necessary.
        .atomic_rmw => false,
        // A barrier synchronizes threads and fences memory. It produces no result, so a
        // purity rule keyed on an unused result would delete every one of them.
        .barrier => false,
    };
}

/// Count uses of each value across live instructions, `if` edges, and terminators.
pub fn countUses(func: *const Function, uses: []u32) void {
    @memset(uses, 0);
    for (0..func.blockCount()) |bi| {
        const block: ir.function.Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| {
            switch (func.opcode(inst)) {
                .atomic_rmw => |a| {
                    uses[@backingInt(a.ptr)] += 1;
                    uses[@backingInt(a.value)] += 1;
                    if (a.compare) |c| uses[@backingInt(c)] += 1;
                },
                // A barrier uses no Value, so it adds no use count.
                .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
                .arith => |a| {
                    uses[@backingInt(a.lhs)] += 1;
                    uses[@backingInt(a.rhs)] += 1;
                },
                .arith_imm => |a| uses[@backingInt(a.lhs)] += 1,
                .icmp => |c| {
                    uses[@backingInt(c.lhs)] += 1;
                    uses[@backingInt(c.rhs)] += 1;
                },
                .select => |s| {
                    uses[@backingInt(s.cond)] += 1;
                    uses[@backingInt(s.then)] += 1;
                    uses[@backingInt(s.@"else")] += 1;
                },
                .extract => |e| uses[@backingInt(e.aggregate)] += 1,
                .convert => |cv| uses[@backingInt(cv.value)] += 1,
                .decode_low_float, .encode_low_float => |cv| uses[@backingInt(cv.value)] += 1,
                .dequantize_nvfp4, .quantize_nvfp4 => |cv| {
                    uses[@backingInt(cv.value)] += 1;
                    uses[@backingInt(cv.block_scale)] += 1;
                    uses[@backingInt(cv.global_scale)] += 1;
                },
                .unary => |u| uses[@backingInt(u.value)] += 1,
                .load => |l| uses[@backingInt(l.ptr)] += 1,
                .store => |st| {
                    uses[@backingInt(st.value)] += 1;
                    uses[@backingInt(st.ptr)] += 1;
                },
                .prefetch => |pf| uses[@backingInt(pf.ptr)] += 1,
                .va_start => |vs| uses[@backingInt(vs.list)] += 1,
                .va_arg => |va| uses[@backingInt(va.list)] += 1,
                .va_end => |ve| uses[@backingInt(ve.list)] += 1,
                .dot => |d| {
                    uses[@backingInt(d.acc)] += 1;
                    uses[@backingInt(d.a)] += 1;
                    uses[@backingInt(d.b)] += 1;
                },
                .reduce => |red| uses[@backingInt(red.vector)] += 1,
                .splat => |sp| uses[@backingInt(sp.scalar)] += 1,
                .matmul => |mm| {
                    uses[@backingInt(mm.a)] += 1;
                    uses[@backingInt(mm.b)] += 1;
                    uses[@backingInt(mm.c)] += 1;
                },
                .struct_new => |sn| for (func.valueList(sn.fields)) |f| {
                    uses[@backingInt(f)] += 1;
                },
                .call => |c| {
                    for (func.valueList(c.args)) |arg| uses[@backingInt(arg)] += 1;
                    if (c.ret_dest) |rd| uses[@backingInt(rd)] += 1; // SM14 M4d-c T1: dest kept alive for the post-call store
                },
                .call_indirect => |c| {
                    uses[@backingInt(c.target)] += 1;
                    for (func.valueList(c.args)) |arg| uses[@backingInt(arg)] += 1;
                    if (c.ret_dest) |rd| uses[@backingInt(rd)] += 1; // SM14 M4d-c T1: dest kept alive for the post-call store
                },
                .@"if" => |cf| {
                    uses[@backingInt(cf.cond)] += 1;
                    for (func.blockArgs(cf.then)) |arg| uses[@backingInt(arg)] += 1;
                    for (func.blockArgs(cf.@"else")) |arg| uses[@backingInt(arg)] += 1;
                },
            }
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| for (r.slice()) |vv| {
                uses[@backingInt(vv)] += 1;
            },
            .jump => |j| for (func.blockArgs(j)) |arg| {
                uses[@backingInt(arg)] += 1;
            },
        };
    }
}

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);

    var changed = false;
    while (true) {
        countUses(func, uses);
        var removed = false;
        for (0..func.blockCount()) |bi| {
            const insts = func.blockInstsMut(@fromBackingInt(@intCast(bi)));
            var w: usize = 0;
            for (insts.items) |inst| {
                const dead = isPure(func.opcode(inst)) and
                    if (func.instResult(inst)) |r| uses[@backingInt(r)] == 0 else false;
                if (dead) {
                    removed = true;
                    continue;
                }
                insts.items[w] = inst;
                w += 1;
            }
            insts.shrinkRetainingCapacity(w);
        }
        if (!removed) break;
        changed = true;
    }
    return changed;
}

test "removes a chain of dead pure instructions" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    // dead1 = x + x, dead2 = dead1 * x, (neither used), ret x
    const dead1 = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });
    _ = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = dead1, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    try std.testing.expectEqual(@as(usize, 2), func.blockInsts(b).len);

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));

    // Both dead instructions are gone.
    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(b).len);
}

test "removes a dead reduce and a dead splat, both are pure" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const v4i32 = try func.types.intern(.{ .vector = .{ .len = 4, .elem = i32_t } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const vec = try func.appendSplat(b, v4i32, x); // dead: never read
    _ = try func.appendReduce(b, .add, vec); // dead: never read
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    try std.testing.expectEqual(@as(usize, 2), func.blockInsts(b).len);

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 0), func.blockInsts(b).len);
}

test "keeps an impure call even if its result is unused" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    _ = try func.appendCall(b, i32_t, "sink", &.{x}); // result unused, but a call has effects
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
}

test "keeps a matmul even though it has no result to be used" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const a = try func.appendBlockParam(b, ptr_t);
    const bp = try func.appendBlockParam(b, ptr_t);
    const c = try func.appendBlockParam(b, ptr_t);
    try func.appendMatmul(b, a, bp, c, 4, 4, 4, .fp32, false);
    func.setTerminator(b, .{ .ret = ir.function.Ret.none() });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
}

test "keeps a barrier even though it has no result to be used" {
    // The whole reason a barrier is an opcode and not a call: a purity rule keyed on an
    // unused result would delete it, and the deletion would be silent.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    try func.appendBarrier(b, .workgroup);
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(b).len);
    try std.testing.expect(func.opcode(func.blockInsts(b)[0]) == .barrier);
}

test "keeps both forms of an atomic, read result or not" {
    // An atomic is a STORE as well as a load. The reduction form has no result to be used,
    // and the reading form's result is deliberately left unread here: a purity rule keyed
    // on an unused result would delete BOTH and lose two writes.
    //
    // The pure `arith` beside them is the control: it is also unused, and it IS deleted, so
    // the pass really ran on this block.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const x = try func.appendBlockParam(b, i32_t);
    // `used` feeds the atomic and nothing else, so `countUses` must count an atomic's
    // operands. If it does not, this pure multiply looks dead and its deletion leaves the
    // atomic naming a value nothing defines.
    const used = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = x } });
    try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = used, .ordering = .relaxed, .scope = .device });
    _ = try func.appendAtomicRmw(b, .{ .op = .bit_or, .ptr = p, .value = x, .ordering = .relaxed, .scope = .device });
    _ = try func.appendInst(b, i32_t, .{ .arith = .{ .op = .sub, .lhs = x, .rhs = x } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(x) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses)); // the dead subtract went

    const insts = func.blockInsts(b);
    try std.testing.expectEqual(@as(usize, 3), insts.len);
    try std.testing.expect(func.opcode(insts[0]) == .arith); // the multiply the atomic uses
    try std.testing.expectEqual(used, func.instResult(insts[0]).?);
    try std.testing.expect(func.opcode(insts[1]) == .atomic_rmw);
    try std.testing.expect(func.opcode(insts[2]) == .atomic_rmw);
}

test "low float conversion liveness controls dead code elimination" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const payload = try func.appendBlockParam(block, u8_t);
    const live = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .value = payload, .format = .f8_e4m3 } });
    _ = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .value = payload, .format = .f8_e5m2 } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(live) });
    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    try std.testing.expectEqual(@as(usize, 1), func.blockInsts(block).len);
    try std.testing.expectEqual(ir.function.LowFloatFormat.f8_e4m3, func.opcode(func.blockInsts(block)[0]).decode_low_float.format);
}

test "nvfp4 conversion liveness counts every operand and removes an unused conversion" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const payload = try func.appendInst(block, u8_t, .{ .iconst = 1 });
    const block_scale = try func.appendInst(block, u8_t, .{ .iconst = 2 });
    const global_scale = try func.appendInst(block, f32_t, .{ .fconst = 3.0 });
    const conversion: ir.function.NvFp4Convert = .{
        .value = payload,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = .multiply,
        .global_application = .divide,
    };
    const live = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = conversion });
    _ = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = conversion });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(live) });
    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    // Each defining constant is live only through a different NVFP4 operand. Omitting any
    // operand from the use count lets DCE erase its definition and makes this assertion fail.
    try std.testing.expectEqual(@as(usize, 4), func.blockInsts(block).len);
    const kept = func.opcode(func.blockInsts(block)[3]).dequantize_nvfp4;
    try std.testing.expectEqual(payload, kept.value);
    try std.testing.expectEqual(block_scale, kept.block_scale);
    try std.testing.expectEqual(global_scale, kept.global_scale);
}
