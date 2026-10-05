//! Differential JIT oracle for induction-variable strength reduction over a pointer chain
//! (libs/vulcan-opt/ivsr.zig). The motivating shape: a loop computes a row address `base + i*stride`
//! and then derives two further addresses from it by adding small constants, loading through all
//! three. Before the transitive `chain_safe` fix, the pass refused the base (its uses are not ALL
//! dereferences: two are further address computations) while still reducing the two derived
//! addresses on their own, the worst combination (the multiply stays AND the loop gains two extra
//! carried pointers). We build the kernel twice, run one copy through the late pipeline (which is
//! where `ivsr` lives), JIT both on the host, and require bit-identical sums over real arrays across
//! several trip counts, including 0 and 1. A mistake here reads or writes the wrong memory, so the
//! JIT run is the actual proof; the structural checks alongside it confirm the fix earns its keep
//! (one variable, not three) rather than merely staying correct by accident.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;
const Block = ir.function.Block;

fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .riscv64, .x86 => true,
        else => false,
    };
}

/// The blocks a caller needs after building the kernel: the header, to read back how many values
/// the loop carries, and the body, to inspect what is left in it.
const Kernel = struct { head: Block, body: Block };

/// `for (i = 0; i < bound; i += 1) acc += base[3i] + base[3i+1] + base[3i+2]; return acc`, addressed
/// as `v28 = base + i*12; load v28; v38 = v28 + 4; load v38; v42 = v28 + 8; load v42`. `v28`'s only
/// uses are the direct load and the two chain steps that build `v38` and `v42`, which is exactly the
/// shape `pointerUsesAreAddresses` used to refuse.
fn buildBaseAndOffsets(func: *Function) anyerror!Kernel {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const base = try func.appendBlockParam(entry, ptr_t);
    const bound = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });

    const i = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = bound } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    // v28 = base + i*12: the row address, written the way a frontend writes it (scale, then add).
    const row_off = try func.appendArithImm(body, i32_t, .mul, bi, 12);
    const v28 = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = row_off } });
    const x0 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = v28 } });
    const v38 = try func.appendArithImm(body, ptr_t, .add, v28, 4);
    const x1 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = v38 } });
    const v42 = try func.appendArithImm(body, ptr_t, .add, v28, 8);
    const x2 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = v42 } });
    const s01 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = x0, .rhs = x1 } });
    const s012 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = s01, .rhs = x2 } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = s012 } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, head, &.{ ni, nacc });

    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });
    return .{ .head = head, .body = body };
}

/// The same shape, except `v38` is ALSO stored into a `sink` pointer, so its own value escapes the
/// loop rather than only ever being dereferenced. Neither `v38` nor `v28` (whose only non-dereference
/// use is building the now-unsafe `v38`) may be reduced: the escape must refuse the whole chain back
/// to its root, not just the link that escapes.
fn buildEscapingChainStep(func: *Function) anyerror!Kernel {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const base = try func.appendBlockParam(entry, ptr_t);
    const sink = try func.appendBlockParam(entry, ptr_t);
    const bound = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });

    const i = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = bound } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const row_off = try func.appendArithImm(body, i32_t, .mul, bi, 12);
    const v28 = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = row_off } });
    const x0 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = v28 } });
    const v38 = try func.appendArithImm(body, ptr_t, .add, v28, 4);
    const x1 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = v38 } });
    try func.appendStore(body, v38, sink); // v38 escapes: its numeric value is now observable
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = x0 } });
    const nacc2 = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = nacc, .rhs = x1 } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, head, &.{ ni, nacc2 });

    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });
    return .{ .head = head, .body = body };
}

/// `for (i = 0; i < bound; i += 1) acc += base[i] + base[i+1]`, addressed as two INDEPENDENT
/// chains rather than one derived from the other: `addr0 = base + i*4`, `idx1 = i + 1`,
/// `addr1 = base + idx1*4`. This is the shape `splitunroll`'s cloned bodies produce (see
/// `ivsr.zig`'s coalescing rule): same base, same scale, offsets a constant apart, with no
/// syntactic predecessor relation between the two chains for `chainPredecessor` to find. Before
/// coalescing, both addresses reduced independently and the loop carried two walking pointers for
/// one address stream; after, `addr1` folds to `addr0`'s walker plus a constant.
fn buildIndependentSiblings(func: *Function) anyerror!Kernel {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();

    const base = try func.appendBlockParam(entry, ptr_t);
    const bound = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{ zero, zero });

    const i = try func.appendBlockParam(head, i32_t);
    const acc = try func.appendBlockParam(head, i32_t);
    const lt = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = bound } });
    try func.appendIf(head, lt, .{ .target = body, .args = &.{ i, acc } }, .{ .target = done, .args = &.{acc} });

    const bi = try func.appendBlockParam(body, i32_t);
    const bacc = try func.appendBlockParam(body, i32_t);
    const off0 = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const addr0 = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off0 } });
    const x0 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr0 } });
    const idx1 = try func.appendArithImm(body, i32_t, .add, bi, 1);
    const off1 = try func.appendArithImm(body, i32_t, .mul, idx1, 4);
    const addr1 = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = off1 } });
    const x1 = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr1 } });
    const sum = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = x0, .rhs = x1 } });
    const nacc = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bacc, .rhs = sum } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, head, &.{ ni, nacc });

    const fin = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(fin) });
    return .{ .head = head, .body = body };
}

fn referenceSiblings(base: []const i32, n: i32) i32 {
    var acc: i32 = 0;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        const row: usize = @intCast(i);
        acc +%= base[row] +% base[row + 1];
    }
    return acc;
}

/// How many `mul` instructions (either form) remain in `block`.
fn countMul(func: *const Function, block: Block) usize {
    var n: usize = 0;
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .arith => |a| if (a.op == .mul) {
            n += 1;
        },
        .arith_imm => |a| if (a.op == .mul) {
            n += 1;
        },
        else => {},
    };
    return n;
}

/// The scalar reference: exactly what the kernel is supposed to compute, over a real array.
fn reference(base: []const i32, n: i32) i32 {
    var acc: i32 = 0;
    var i: i32 = 0;
    while (i < n) : (i += 1) {
        const row: usize = @intCast(i * 3);
        acc +%= base[row] +% base[row + 1] +% base[row + 2];
    }
    return acc;
}

const KernelFn = *const fn ([*]const i32, i32) callconv(.c) i32;

test "ivsr differential: a base address with two further offsets reduces to one walking pointer and matches the scalar baseline" {
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    _ = try buildBaseAndOffsets(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    const k = try buildBaseAndOffsets(&tuned);

    const before_carried = tuned.blockParams(k.head).len; // i, acc: 2
    try std.testing.expectEqual(@as(usize, 2), before_carried);

    _ = try opt.optimizeLate(allocator, &tuned);

    var diags = try ir.verify.verify(allocator, &tuned, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    // The payoff: the base earns its own induction variable (one new header parameter, not three),
    // and the late pipeline's `dce` clears the multiply that built it out of the body entirely.
    const after_carried = tuned.blockParams(k.head).len;
    try std.testing.expectEqual(@as(usize, 3), after_carried);
    try std.testing.expectEqual(@as(usize, 0), countMul(&tuned, k.body));
    // Before this fix, `pointerUsesAreAddresses` refused the base outright (two of its three uses
    // are not a dereference), while `v38` and `v42` still passed the gate on their own and each got a
    // walking pointer: 4 carried values (i, acc, and the two of them), with the multiply still in the
    // body because the base was never touched. That count is the regression this fix exists to
    // close, so the fixed count must not exceed it.
    const carried_before_fix = before_carried + 2;
    try std.testing.expect(after_carried <= carried_before_fix);

    if (comptime !hasJit()) return error.SkipZigTest;

    var buf_b = try target.native.jitFunction(allocator, &baseline);
    defer buf_b.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_b = buf_b.entry(KernelFn, 0);
    const f_t = buf_t.entry(KernelFn, 0);

    var arr: [32]i32 = undefined;
    for (&arr, 0..) |*e, idx| e.* = @intCast(idx);

    const trips = [_]i32{ 0, 1, 2, 3, 5, 8 };
    for (trips) |n| {
        const got_b = f_b(&arr, n);
        const got_t = f_t(&arr, n);
        try std.testing.expectEqual(got_b, got_t);
        try std.testing.expectEqual(reference(&arr, n), got_t);
    }
}

test "ivsr differential: a chain step that escapes leaves the whole address chain unreduced" {
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    _ = try buildEscapingChainStep(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    const k = try buildEscapingChainStep(&tuned);

    _ = try opt.optimizeLate(allocator, &tuned);

    var diags = try ir.verify.verify(allocator, &tuned, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    // Neither the escaping step nor the base it was built from earns a variable: the loop still
    // carries exactly what it started with.
    try std.testing.expectEqual(@as(usize, 2), tuned.blockParams(k.head).len);

    if (comptime !hasJit()) return error.SkipZigTest;

    const Fn = *const fn ([*]const i32, [*]i32, i32) callconv(.c) i32;
    var buf_b = try target.native.jitFunction(allocator, &baseline);
    defer buf_b.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_b = buf_b.entry(Fn, 0);
    const f_t = buf_t.entry(Fn, 0);

    var arr: [32]i32 = undefined;
    for (&arr, 0..) |*e, idx| e.* = @intCast(idx);
    var sink_b: [32]i32 = @splat(0);
    var sink_t: [32]i32 = @splat(0);

    const trips = [_]i32{ 0, 1, 2, 3, 5, 8 };
    for (trips) |n| {
        const got_b = f_b(&arr, &sink_b, n);
        const got_t = f_t(&arr, &sink_t, n);
        try std.testing.expectEqual(got_b, got_t);
    }
}

test "ivsr differential: two independent same-scale address chains coalesce to one walking pointer and match the scalar baseline" {
    const allocator = std.testing.allocator;

    var baseline = Function.init(allocator);
    defer baseline.deinit();
    _ = try buildIndependentSiblings(&baseline);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    const k = try buildIndependentSiblings(&tuned);

    const before_carried = tuned.blockParams(k.head).len; // i, acc: 2
    try std.testing.expectEqual(@as(usize, 2), before_carried);

    _ = try opt.optimizeLate(allocator, &tuned);

    var diags = try ir.verify.verify(allocator, &tuned, .high);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    // One new walking pointer for the whole family, not one per chain.
    const after_carried = tuned.blockParams(k.head).len;
    try std.testing.expectEqual(@as(usize, 3), after_carried);
    try std.testing.expectEqual(@as(usize, 0), countMul(&tuned, k.body));

    if (comptime !hasJit()) return error.SkipZigTest;

    var buf_b = try target.native.jitFunction(allocator, &baseline);
    defer buf_b.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_b = buf_b.entry(KernelFn, 0);
    const f_t = buf_t.entry(KernelFn, 0);

    var arr: [32]i32 = undefined;
    for (&arr, 0..) |*e, idx| e.* = @intCast(idx);

    const trips = [_]i32{ 0, 1, 2, 3, 5, 8 };
    for (trips) |n| {
        const got_b = f_b(&arr, n);
        const got_t = f_t(&arr, n);
        try std.testing.expectEqual(got_b, got_t);
        try std.testing.expectEqual(referenceSiblings(&arr, n), got_t);
    }
}
