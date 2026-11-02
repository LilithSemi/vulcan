//! Differential JIT oracle for the structural loop vectorizer. For each map-loop shape we build two
//! identical functions, run loopvec (and, in the second test group, the full microarch pipeline so
//! SLP widens it) on one, JIT both on the host, run them over real arrays, and require bit-identical
//! output across trip counts chosen to exercise the remainder: 0, below V, exactly V, V-multiples,
//! and non-multiples. Any divergence is a miscompile. Runs where the native JIT has a backend.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");
const target = @import("vulcan-target");

const Function = ir.function.Function;

fn ampere() *const opt.microarch.Model {
    return opt.microarch.modelFor(.@"ampere-altra");
}

fn zen4() *const opt.microarch.Model {
    return opt.microarch.modelFor(.zen4);
}

fn hasJit() bool {
    return switch (builtin.cpu.arch) {
        .aarch64, .x86_64, .riscv64, .x86 => true,
        else => false,
    };
}

/// `for (i = 0; i < n; i += 1) y[i] = a*x[i] + y[i];` over f32 arrays.
fn buildSaxpy(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    const yv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = yaddr } });
    const ax = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = xv } });
    const res = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = ax, .rhs = yv } });
    try func.appendStore(body, res, yaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

/// `for (i = 0; i < n; i += 1) y[i] = x[i] * x[i];` (a pure map, no read of the output).
fn buildSquare(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t); // unused, keeps the 4-arg signature uniform
    _ = a;
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr } });
    const sq = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = xv, .rhs = xv } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    try func.appendStore(body, sq, yaddr);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

/// `s = 0; for (i = 0; i < n; i += 1) s += a[i]; return s;` over f32. Marked fast_math so the FP
/// reduction may reassociate (vector partial sums + horizontal reduce).
fn buildSumReduction(func: *Function) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    try func.addAttr(.func, .{ .custom = .{ .namespace = "vulcan", .key = "fast_math", .value = .flag } });
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero_i = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero_f = try func.appendInst(entry, f32_t, .{ .fconst = 0 });
    try func.setJump(entry, loop, &.{ zero_i, zero_f });
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, f32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done, .args = &.{s} });
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, f32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = off } });
    const v = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = addr } });
    const ns = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = v } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    const rs = try func.appendBlockParam(done, f32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(rs) });
}

/// `s = 0; for (i = 0; i < n; i += 1) s += a[i]; return s;` over i32. Integer add is reorderable
/// without fast_math, so this exercises the same vector-accumulator path through the GPR lane
/// encoders instead of the NEON ones the f32 reduction test uses.
fn buildSumReductionI32(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero_i = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero_s = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero_i, zero_s });
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done, .args = &.{s} });
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = off } });
    const v = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr } });
    const ns = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = v } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    const rs = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(rs) });
}

/// `for (i = 0; i < n; i += 1) s += a[i] * a[i];` over i32, so the vector body has a lane multiply.
fn buildSquareSumI32(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero_i = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const zero_s = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero_i, zero_s });
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done, .args = &.{s} });
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = off } });
    const v = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr } });
    const sq = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = v, .rhs = v } });
    const ns = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = sq } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    const rs = try func.appendBlockParam(done, i32_t);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(rs) });
}

/// Frontend-shaped three-point stencil with a stored map result and XOR checksum reduction.
fn buildStencil(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const input = try func.appendBlockParam(entry, ptr_t);
    const output = try func.appendBlockParam(entry, ptr_t);
    const count = try func.appendBlockParam(entry, u32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const one = try func.appendInst(entry, u32_t, .{ .iconst = 1 });
    try func.setJump(entry, header, &.{ one, zero });
    const index = try func.appendBlockParam(header, u32_t);
    const checksum = try func.appendBlockParam(header, i32_t);
    const next = try func.appendArithImm(header, u32_t, .add, index, 1);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next, .rhs = count } });
    try func.appendIf(header, cond, .{ .target = body }, .{ .target = done });

    var values: [3]ir.function.Value = undefined;
    for ([_]i64{ -1, 0, 1 }, 0..) |bias, lane| {
        const neighbour = if (bias == 0) index else try func.appendArithImm(body, u32_t, .add, index, bias);
        const wide = try func.appendInst(body, i64_t, .{ .convert = .{ .value = neighbour } });
        const offset = try func.appendArithImm(body, i64_t, .shl, wide, 2);
        const address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = input, .rhs = offset } });
        values[lane] = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = address } });
    }
    const doubled = try func.appendArithImm(body, i32_t, .shl, values[1], 1);
    const partial = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = values[0], .rhs = doubled } });
    const value = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = partial, .rhs = values[2] } });
    const out_wide = try func.appendInst(body, i64_t, .{ .convert = .{ .value = index } });
    const out_offset = try func.appendArithImm(body, i64_t, .shl, out_wide, 2);
    const out_address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = output, .rhs = out_offset } });
    try func.appendStore(body, value, out_address);
    const updated = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_xor, .lhs = checksum, .rhs = value } });
    try func.setJump(body, latch, &.{});
    try func.setJump(latch, header, &.{ next, updated });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(checksum) });
}

const Builder = *const fn (*Function) anyerror!void;
const SaxpyFn = *const fn (x: [*]f32, y: [*]f32, a: f32, n: i32) callconv(.c) void;
const SumFn = *const fn (a: [*]f32, n: i32) callconv(.c) f32;
const SumFnI32 = *const fn (a: [*]i32, n: i32) callconv(.c) i32;
const StencilFn = *const fn (input: [*]const i32, output: [*]i32, n: u32) callconv(.c) i32;

/// Transform: apply loopvec only, or the full microarch pipeline (so SLP widens the body too).
const Mode = enum { loopvec_only, full };

fn expectMapMatches(build: Builder, mode: Mode, model: *const opt.microarch.Model) !void {
    const allocator = std.testing.allocator;
    const trip_counts = [_]i32{ 0, 1, 3, 4, 5, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try build(&orig);

    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try build(&tuned);
    switch (mode) {
        .loopvec_only => try std.testing.expect(try opt.microarch.loopvec.run(allocator, &tuned, model)),
        .full => _ = try opt.microarch.optimize(allocator, &tuned, model),
    }

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(SaxpyFn, 0);
    const f_t = buf_t.entry(SaxpyFn, 0);

    for (trip_counts) |n| {
        var x: [128]f32 = undefined;
        var y_o: [128]f32 = undefined;
        var y_t: [128]f32 = undefined;
        for (0..128) |k| {
            x[k] = @floatFromInt(k + 1);
            y_o[k] = @floatFromInt(200 - @as(i32, @intCast(k)));
            y_t[k] = y_o[k];
        }
        f_o(&x, &y_o, 2.5, n);
        f_t(&x, &y_t, 2.5, n);
        if (!std.mem.eql(f32, &y_o, &y_t)) {
            std.debug.print("\nn={d}: y_o[0..8]={any}\n       y_t[0..8]={any}\n", .{ n, y_o[0..8], y_t[0..8] });
        }
        try std.testing.expectEqualSlices(f32, &y_o, &y_t); // whole buffer: no over/under-run either
    }
}

test "loopvec differential: saxpy, scalar V-unroll, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSaxpy, .loopvec_only, ampere());
}

test "loopvec differential: pure square map, scalar V-unroll, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSquare, .loopvec_only, ampere());
}

test "loopvec differential: saxpy through the full pipeline (SLP-widened), all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSaxpy, .full, ampere());
}

test "loopvec differential: pure square map through the full pipeline, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectMapMatches(buildSquare, .full, ampere());
}

fn hasVectorValue(func: *const Function) bool {
    var i: usize = 0;
    while (i < func.valueCount()) : (i += 1) {
        const v: ir.function.Value = @fromBackingInt(@intCast(@as(u32, @intCast(i))));
        if (func.types.type_kind(func.valueType(v)) == .vector) return true;
    }
    return false;
}

test "loopvec output is SLP-fusable: force-fusing the main body produces vector ops" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func);
    // loopvec makes a 4-wide op-major main body; the SLP vectorizer's correctness (force) path then
    // fuses it into wide loads, a vector fmul, a vector fadd, and a wide store. This proves the
    // structure loopvec produces is exactly what SLP consumes.
    try std.testing.expect(try opt.microarch.loopvec.run(allocator, &func, ampere()));
    try std.testing.expect(try opt.vectorize.runLanes(allocator, &func, 4));
    try std.testing.expect(hasVectorValue(&func));
}

test "loopvec differential: f32 sum reduction (vector accumulator), all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const trip_counts = [_]i32{ 0, 1, 3, 4, 5, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try buildSumReduction(&orig);
    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildSumReduction(&tuned);
    try std.testing.expect(try opt.microarch.loopvec.run(allocator, &tuned, ampere()));
    try std.testing.expect(hasVectorValue(&tuned)); // a genuine vector accumulator was emitted

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(SumFn, 0);
    const f_t = buf_t.entry(SumFn, 0);

    for (trip_counts) |n| {
        // Small integers as f32: their partial sums stay exact (< 2^24), so serial and vector-partial
        // reductions agree bit-for-bit despite the reassociation.
        var arr: [128]f32 = undefined;
        for (0..128) |k| arr[k] = @floatFromInt((k % 7) + 1);
        try std.testing.expectEqual(f_o(&arr, n), f_t(&arr, n));
    }
}

fn expectI32ReductionMatches(build: Builder, mode: Mode, model: *const opt.microarch.Model) !void {
    const allocator = std.testing.allocator;
    const trip_counts = [_]i32{ 0, 1, 3, 4, 5, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try build(&orig);
    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try build(&tuned);
    switch (mode) {
        .loopvec_only => try std.testing.expect(try opt.microarch.loopvec.run(allocator, &tuned, model)),
        .full => _ = try opt.microarch.optimize(allocator, &tuned, model),
    }
    try std.testing.expect(hasVectorValue(&tuned)); // a genuine vector accumulator was emitted

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(SumFnI32, 0);
    const f_t = buf_t.entry(SumFnI32, 0);

    for (trip_counts) |n| {
        // Integer add is associative and exact, so serial and vector-partial reductions
        // must agree bit-for-bit at every trip count, not just in the common case.
        var arr: [128]i32 = undefined;
        for (0..128) |k| arr[k] = @as(i32, @intCast((k * 2654435761) % 100003)) - 50000;
        try std.testing.expectEqual(f_o(&arr, n), f_t(&arr, n));
    }
}

test "loopvec differential: i32 sum reduction (vector accumulator), all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectI32ReductionMatches(buildSumReductionI32, .loopvec_only, ampere());
}

fn expectStencilMatches(model: *const opt.microarch.Model) !void {
    const allocator = std.testing.allocator;
    const trip_counts = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 17, 100 };

    var orig = Function.init(allocator);
    defer orig.deinit();
    try buildStencil(&orig);
    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildStencil(&tuned);
    _ = try opt.microarch.optimize(allocator, &tuned, model);

    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const f_o = buf_o.entry(StencilFn, 0);
    const f_t = buf_t.entry(StencilFn, 0);

    var input: [128]i32 = undefined;
    for (0..128) |k| input[k] = @as(i32, @intCast((k * 17 + 3) % 101)) - 50;
    for (trip_counts) |n| {
        var out_o: [128]i32 = undefined;
        @memset(&out_o, 0x13579);
        var out_t = out_o;
        try std.testing.expectEqual(f_o(&input, &out_o, n), f_t(&input, &out_t, n));
        try std.testing.expectEqualSlices(i32, &out_o, &out_t);
    }
}

test "loopvec differential: stencil map plus xor reduction, all trip counts" {
    if (comptime !hasJit()) return error.SkipZigTest;
    try expectStencilMatches(ampere());
}

/// `for i < 4: for j < n: { s = 0; for k < 8: s += A[i][k] * B[k][j]; C[i][j] = s; }` with
/// 8-wide rows, shaped like the C frontend's output.
fn buildMatmulNest(func: *Function) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const ih = try func.appendBlock();
    const ib = try func.appendBlock();
    const jh = try func.appendBlock();
    const jb = try func.appendBlock();
    const kh = try func.appendBlock();
    const kb = try func.appendBlock();
    const kx = try func.appendBlock();
    const il = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const c = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const four = try func.appendInst(entry, i32_t, .{ .iconst = 4 });
    const eight = try func.appendInst(entry, i32_t, .{ .iconst = 8 });
    try func.setJump(entry, ih, &.{zero});

    const i = try func.appendBlockParam(ih, i32_t);
    const i_more = try func.appendInst(ih, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = four } });
    try func.appendIf(ih, i_more, .{ .target = ib }, .{ .target = done });
    const ix = try func.appendInst(ib, i64_t, .{ .convert = .{ .value = i } });
    const row = try func.appendArithImm(ib, i64_t, .shl, ix, 5);
    const arow = try func.appendInst(ib, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = row } });
    const crow = try func.appendInst(ib, ptr_t, .{ .arith = .{ .op = .add, .lhs = c, .rhs = row } });
    try func.setJump(ib, jh, &.{zero});

    const j = try func.appendBlockParam(jh, i32_t);
    const j_more = try func.appendInst(jh, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n } });
    try func.appendIf(jh, j_more, .{ .target = jb }, .{ .target = il });
    const jx = try func.appendInst(jb, i64_t, .{ .convert = .{ .value = j } });
    const jo = try func.appendArithImm(jb, i64_t, .shl, jx, 2);
    try func.setJump(jb, kh, &.{ zero, zero });

    const k = try func.appendBlockParam(kh, i32_t);
    const s = try func.appendBlockParam(kh, i32_t);
    const k_more = try func.appendInst(kh, bool_t, .{ .icmp = .{ .op = .lt, .lhs = k, .rhs = eight } });
    try func.appendIf(kh, k_more, .{ .target = kb }, .{ .target = kx });
    const kx64 = try func.appendInst(kb, i64_t, .{ .convert = .{ .value = k } });
    const ko = try func.appendArithImm(kb, i64_t, .shl, kx64, 2);
    const ap = try func.appendInst(kb, ptr_t, .{ .arith = .{ .op = .add, .lhs = arow, .rhs = ko } });
    const av = try func.appendInst(kb, i32_t, .{ .load = .{ .ptr = ap } });
    const bo = try func.appendArithImm(kb, i64_t, .shl, kx64, 5);
    const brow = try func.appendInst(kb, ptr_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = bo } });
    const bp = try func.appendInst(kb, ptr_t, .{ .arith = .{ .op = .add, .lhs = brow, .rhs = jo } });
    const bv = try func.appendInst(kb, i32_t, .{ .load = .{ .ptr = bp } });
    const prod = try func.appendInst(kb, i32_t, .{ .arith = .{ .op = .mul, .lhs = av, .rhs = bv } });
    const s2 = try func.appendInst(kb, i32_t, .{ .arith = .{ .op = .add, .lhs = s, .rhs = prod } });
    const k2 = try func.appendArithImm(kb, i32_t, .add, k, 1);
    try func.setJump(kb, kh, &.{ k2, s2 });

    const cp = try func.appendInst(kx, ptr_t, .{ .arith = .{ .op = .add, .lhs = crow, .rhs = jo } });
    try func.appendStore(kx, s, cp);
    const j2 = try func.appendArithImm(kx, i32_t, .add, j, 1);
    try func.setJump(kx, jh, &.{j2});
    const i_next = try func.appendArithImm(il, i32_t, .add, i, 1);
    try func.setJump(il, ih, &.{i_next});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

test "zen4 differential: an outer-vectorized matmul nest matches at every column count" {
    if (comptime !hasJit()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    var orig = Function.init(allocator);
    defer orig.deinit();
    try buildMatmulNest(&orig);
    var tuned = Function.init(allocator);
    defer tuned.deinit();
    try buildMatmulNest(&tuned);
    try std.testing.expect(try opt.microarch.outervec.run(allocator, &tuned, zen4()));
    try std.testing.expect(hasVectorValue(&tuned));
    var diags = try ir.verify.verify(allocator, &tuned, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var buf_o = try target.native.jitFunction(allocator, &orig);
    defer buf_o.deinit();
    var buf_t = try target.native.jitFunction(allocator, &tuned);
    defer buf_t.deinit();
    const Fn = *const fn ([*]const i32, [*]const i32, [*]i32, i32) callconv(.c) void;
    const f_o = buf_o.entry(Fn, 0);
    const f_t = buf_t.entry(Fn, 0);
    var a_mem: [32]i32 = undefined;
    var b_mem: [64]i32 = undefined;
    for (&a_mem, 0..) |*v, idx| v.* = @as(i32, @intCast((idx * 7 + 3) % 23)) - 11;
    for (&b_mem, 0..) |*v, idx| v.* = @as(i32, @intCast((idx * 13 + 5) % 31)) - 15;
    for (0..9) |cols| {
        var c_o: [32]i32 = @splat(0x5a5a);
        var c_t: [32]i32 = @splat(0x5a5a);
        f_o(&a_mem, &b_mem, &c_o, @intCast(cols));
        f_t(&a_mem, &b_mem, &c_t, @intCast(cols));
        try std.testing.expectEqualSlices(i32, &c_o, &c_t);
    }
}

test "x86 differential: i32 reductions, stencil, and maps through the full pipeline" {
    if (comptime !hasJit()) return error.SkipZigTest;
    // Cascade Lake declares 512-bit vectors, wider than the x86 backend lowers.
    for ([_]*const opt.microarch.Model{ zen4(), opt.microarch.modelFor(.@"cascadelake-sp") }) |model| {
        try expectI32ReductionMatches(buildSumReductionI32, .full, model);
        try expectI32ReductionMatches(buildSquareSumI32, .full, model);
        try expectStencilMatches(model);
        try expectMapMatches(buildSaxpy, .full, model);
        try expectMapMatches(buildSquare, .full, model);
    }
}

test "the full ampere pipeline vectorizes saxpy (splat-aware cost model)" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func);
    _ = try opt.microarch.optimize(allocator, &func, ampere());
    // The invariant multiplier `a` is a splat (one `dup`, not a 4-lane pack), so the SLP cost model
    // judges the saxpy group profitable on the wide OoO ampere and vectorizes it, like gcc -O3's fmla.
    try std.testing.expect(hasVectorValue(&func));
}
