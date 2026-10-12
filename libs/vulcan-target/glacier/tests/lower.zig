//! The Mandelbrot kernel again, this time from Vulcan IR through the Glacier instruction selector.
//!
//! The hand-assembled program in `mandelbrot.zig` is the oracle: it matches Glacier's own word
//! stream for a frame that renders on an ECP5 FPGA today. A compiler picks different registers and
//! a different instruction order, so the two word streams do not match and never will. What must
//! match is the IMAGE, so both streams run on the warp model in `warp.zig` and the frames are
//! diffed word for word.
//!
//! The kernel built here is the same algorithm, down to the sign-magnitude fixed-point multiply.
//! That shift must truncate toward zero: an arithmetic shift of a negative product rounds toward
//! minus infinity instead, and the two differ by one unit in the last place on exactly the pixels
//! at the boundary of the set, which is where the whole image is. The types carry that difference,
//! because the signedness of a `shr` comes from its operand type.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const glacier = @import("vulcan-target").glacier;

const hand = @import("mandelbrot.zig");
const warp = @import("warp.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;

const width: i32 = 64;
const height: i32 = 48;
const iterations: u16 = 16;
const fb_base: u32 = 0;
const fraction_bits: i64 = 12;

/// Where the host put the frame. The kernel names it, and the backend turns the name into the
/// absolute address, because a Glacier kernel reaches memory by address and nothing else.
const framebuffer = "framebuffer";

/// Build the renderer as Vulcan IR. The block order is dominance order, which the backends rely
/// on, and every loop carries its state in block parameters.
fn buildKernel(allocator: std.mem.Allocator, view: hand.View) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();

    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const row_stride: i64 = @as(i64, width) * 4;
    const escape: i64 = 4 << fraction_bits;
    const shade_shift: i64 = hand.colourShift(iterations);

    const entry = try f.appendBlock();
    // A warp is the workgroup: it owns a program counter and a register file, and its id is
    // unique across the SoC. The lanes inside it are the threads, and this kernel uses none.
    const warp_id = try f.appendBlockParam(entry, u32_t);
    const warp_count = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, warp_id, .block_id_x);
    try gpu.attrs.setBuiltin(&f, warp_count, .grid_dim_x);

    const row_loop = try f.appendBlock();
    const row = try f.appendBlockParam(row_loop, u32_t);
    // The imaginary coordinate is SIGNED, because the sign-magnitude multiply below restores the
    // sign of the product and then adds this to it.
    const cy = try f.appendBlockParam(row_loop, i32_t);
    const row_pixel = try f.appendBlockParam(row_loop, ptr_t);

    const col_loop = try f.appendBlock();
    const col = try f.appendBlockParam(col_loop, u32_t);
    const cx = try f.appendBlockParam(col_loop, u32_t);
    const pixel = try f.appendBlockParam(col_loop, ptr_t);

    const iter_loop = try f.appendBlock();
    const zx = try f.appendBlockParam(iter_loop, i32_t);
    const zy = try f.appendBlockParam(iter_loop, i32_t);
    const left = try f.appendBlockParam(iter_loop, u32_t);

    const advance = try f.appendBlock();
    const shade = try f.appendBlock();
    const shade_left = try f.appendBlockParam(shade, u32_t);
    const row_end = try f.appendBlock();
    const exit = try f.appendBlock();

    // The prologue. Each warp asks the hardware who it is, skips to its own first row, and works
    // out the strides that carry it past the rows the other warps own.
    const fb = try f.appendGlobalAddr(entry, ptr_t, framebuffer);
    const stride = try f.appendInst(entry, u32_t, .{ .iconst = row_stride });
    const dy = try f.appendInst(entry, i32_t, .{ .iconst = view.dy });
    const row_offset = try mul(&f, entry, u32_t, warp_id, stride);
    const first_pixel = try add(&f, entry, ptr_t, fb, row_offset);
    const y_min = try f.appendInst(entry, i32_t, .{ .iconst = view.y_min });
    const warp_index = try reinterpret(&f, entry, i32_t, warp_id);
    const warps = try reinterpret(&f, entry, i32_t, warp_count);
    const cy_offset = try mul(&f, entry, i32_t, warp_index, dy);
    const first_cy = try add(&f, entry, i32_t, y_min, cy_offset);
    const others = try f.appendArithImm(entry, u32_t, .sub, warp_count, 1);
    const row_skip = try mul(&f, entry, u32_t, others, stride);
    const cy_step = try mul(&f, entry, i32_t, warps, dy);
    try f.setJump(entry, row_loop, &.{ warp_id, first_cy, first_pixel });

    const x_min = try f.appendInst(row_loop, u32_t, .{ .iconst = @as(u32, @bitCast(view.x_min)) });
    const zero_col = try f.appendInst(row_loop, u32_t, .{ .iconst = 0 });
    try f.setJump(row_loop, col_loop, &.{ zero_col, x_min, row_pixel });

    const z_start = try f.appendInst(col_loop, i32_t, .{ .iconst = 0 });
    const left_start = try f.appendInst(col_loop, u32_t, .{ .iconst = iterations });
    try f.setJump(col_loop, iter_loop, &.{ z_start, z_start, left_start });

    // The escape test. A square is never negative, so the low word of each product is read as an
    // unsigned value and a logical shift scales it back down.
    const zx_sq = try mul(&f, iter_loop, i32_t, zx, zx);
    const zx_sq_u = try reinterpret(&f, iter_loop, u32_t, zx_sq);
    const zx2 = try f.appendArithImm(iter_loop, u32_t, .shr, zx_sq_u, fraction_bits);
    const zy_sq = try mul(&f, iter_loop, i32_t, zy, zy);
    const zy_sq_u = try reinterpret(&f, iter_loop, u32_t, zy_sq);
    const zy2 = try f.appendArithImm(iter_loop, u32_t, .shr, zy_sq_u, fraction_bits);
    const norm = try add(&f, iter_loop, u32_t, zx2, zy2);
    const limit = try f.appendInst(iter_loop, u32_t, .{ .iconst = escape });
    const escaped = try f.appendInst(iter_loop, bool_t, .{ .icmp = .{ .op = .ge, .lhs = norm, .rhs = limit } });
    try f.appendIf(iter_loop, escaped, .{ .target = shade, .args = &.{left} }, .{ .target = advance });

    // zy = 2*zx*zy + cy. The product holds 24 fraction bits and the shift takes it back to 12.
    // It has to truncate toward zero, so a negative product is shifted as a magnitude and the
    // sign is put back afterwards.
    const cross = try mul(&f, advance, i32_t, zx, zy);
    const sign = try f.appendArithImm(advance, i32_t, .shr, cross, 31);
    const flipped = try bits(&f, advance, i32_t, .bit_xor, cross, sign);
    const magnitude = try sub(&f, advance, i32_t, flipped, sign);
    const magnitude_u = try reinterpret(&f, advance, u32_t, magnitude);
    const scaled_u = try f.appendArithImm(advance, u32_t, .shr, magnitude_u, fraction_bits);
    const scaled = try reinterpret(&f, advance, i32_t, scaled_u);
    const back = try bits(&f, advance, i32_t, .bit_xor, scaled, sign);
    const restored = try sub(&f, advance, i32_t, back, sign);
    const doubled = try f.appendArithImm(advance, i32_t, .shl, restored, 1);
    const next_zy = try add(&f, advance, i32_t, doubled, cy);
    const difference = try sub(&f, advance, u32_t, zx2, zy2);
    const next_cx = try add(&f, advance, u32_t, difference, cx);
    const next_zx = try reinterpret(&f, advance, i32_t, next_cx);
    const next_left = try f.appendArithImm(advance, u32_t, .sub, left, 1);
    const spent = try f.appendInst(advance, u32_t, .{ .iconst = 0 });
    const more = try f.appendInst(advance, bool_t, .{ .icmp = .{ .op = .ne, .lhs = next_left, .rhs = spent } });
    try f.appendIf(
        advance,
        more,
        .{ .target = iter_loop, .args = &.{ next_zx, next_zy, next_left } },
        .{ .target = shade, .args = &.{next_left} },
    );

    // Colour from the iterations left. A point that survived every iteration is inside the set
    // and stays black, and the sooner one escaped the brighter it is.
    const level = try f.appendArithImm(shade, u32_t, .shl, shade_left, shade_shift);
    const green = try f.appendArithImm(shade, u32_t, .shl, level, 8);
    const colour = try bits(&f, shade, u32_t, .bit_or, level, green);
    try f.appendStore(shade, colour, pixel);
    const next_pixel = try f.appendArithImm(shade, ptr_t, .add, pixel, 4);
    const stepped_cx = try f.appendArithImm(shade, u32_t, .add, cx, view.dx);
    const next_col = try f.appendArithImm(shade, u32_t, .add, col, 1);
    const last_col = try f.appendInst(shade, u32_t, .{ .iconst = width });
    const more_cols = try f.appendInst(shade, bool_t, .{ .icmp = .{ .op = .ne, .lhs = next_col, .rhs = last_col } });
    try f.appendIf(
        shade,
        more_cols,
        .{ .target = col_loop, .args = &.{ next_col, stepped_cx, next_pixel } },
        .{ .target = row_end },
    );

    const skipped = try add(&f, row_end, ptr_t, next_pixel, row_skip);
    const stepped_cy = try add(&f, row_end, i32_t, cy, cy_step);
    const next_row = try add(&f, row_end, u32_t, row, warp_count);
    const last_row = try f.appendInst(row_end, u32_t, .{ .iconst = height });
    const more_rows = try f.appendInst(row_end, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next_row, .rhs = last_row } });
    try f.appendIf(
        row_end,
        more_rows,
        .{ .target = row_loop, .args = &.{ next_row, stepped_cy, skipped } },
        .{ .target = exit },
    );

    f.setTerminator(exit, .{ .ret = ir.function.Ret.none() });
    return f;
}

fn mul(f: *Function, block: ir.function.Block, ty: ir.types.Type, lhs: Value, rhs: Value) !Value {
    return f.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = lhs, .rhs = rhs } });
}

fn add(f: *Function, block: ir.function.Block, ty: ir.types.Type, lhs: Value, rhs: Value) !Value {
    return f.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = lhs, .rhs = rhs } });
}

fn sub(f: *Function, block: ir.function.Block, ty: ir.types.Type, lhs: Value, rhs: Value) !Value {
    return f.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = lhs, .rhs = rhs } });
}

fn bits(f: *Function, block: ir.function.Block, ty: ir.types.Type, op: ir.function.BinOp, lhs: Value, rhs: Value) !Value {
    return f.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
}

fn reinterpret(f: *Function, block: ir.function.Block, ty: ir.types.Type, value: Value) !Value {
    return f.appendInst(block, ty, .{ .unary = .{ .op = .reinterpret, .value = value } });
}

const pixels: usize = @intCast(width * height);
/// Instructions one warp may run before the model calls it a loop that does not end. The whole
/// frame is about 1.5 million, so this leaves room for a lowering that is several times longer
/// without letting a runaway kernel hang the test.
const budget: usize = 50_000_000;

fn handWords(out: []u32, view: hand.View) []const u32 {
    return hand.kernel(out, @bitCast(fb_base), width, height, iterations, view);
}

fn lower(allocator: std.mem.Allocator, view: hand.View) ![]u32 {
    var f = try buildKernel(allocator, view);
    defer f.deinit();
    return glacier.isel.compileKernel(allocator, &f, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = framebuffer, .address = fb_base }},
    });
}

test "the lowered kernel renders the frame the hand-assembled kernel renders" {
    const allocator = std.testing.allocator;
    const view = hand.View.fit(width, height);

    var buf: [128]u32 = undefined;
    const expected = try warp.render(allocator, handWords(&buf, view), fb_base, pixels, 4, budget);
    defer allocator.free(expected);

    const words = try lower(allocator, view);
    defer allocator.free(words);
    const got = try warp.render(allocator, words, fb_base, pixels, 4, budget);
    defer allocator.free(got);

    try std.testing.expectEqualSlices(u32, expected, got);

    // A frame of zeros would pass the diff and prove nothing, and the set is a shape, so the
    // frame must hold more than one colour.
    var lit: usize = 0;
    for (got) |px| {
        if (px != 0) lit += 1;
    }
    try std.testing.expect(lit > pixels / 8);
    try std.testing.expect(lit < pixels);
}

test "the lowered kernel fills the same frame whatever the warp count" {
    const allocator = std.testing.allocator;
    const view = hand.View.fit(width, height);
    const words = try lower(allocator, view);
    defer allocator.free(words);

    const one = try warp.render(allocator, words, fb_base, pixels, 1, budget);
    defer allocator.free(one);
    for ([_]u32{ 2, 4, 8, 16 }) |warps| {
        const many = try warp.render(allocator, words, fb_base, pixels, warps, budget);
        defer allocator.free(many);
        try std.testing.expectEqualSlices(u32, one, many);
    }
}

test "the lowered kernel retires the warp and fits the instruction aperture" {
    const allocator = std.testing.allocator;
    const words = try lower(allocator, hand.View.fit(width, height));
    defer allocator.free(words);
    // The kernel retires once. EBREAK is not the last word: splitting the critical edges appends
    // the forwarding blocks after the exit block, so the jumps that serve the loops come last.
    var retires: usize = 0;
    for (words) |w| {
        if (w == glacier.encode.ebreak) retires += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), retires);
    try std.testing.expect(words.len <= 8192);
    // Every jump was patched. A surviving zero decodes to nothing useful and would run on into
    // the next instruction instead of branching.
    for (words) |w| try std.testing.expect(w != 0);
}

test "the rendered frame mirrors about its middle row" {
    // `View.fit` puts the real axis exactly on the middle row, so the set is symmetric about it.
    // This checks the image itself rather than the two word streams agreeing with each other.
    const allocator = std.testing.allocator;
    const words = try lower(allocator, hand.View.fit(width, height));
    defer allocator.free(words);
    const frame = try warp.render(allocator, words, fb_base, pixels, 4, budget);
    defer allocator.free(frame);

    const w: usize = @intCast(width);
    const h: usize = @intCast(height);
    var y: usize = 1;
    while (y < h / 2) : (y += 1) {
        const above = frame[(h / 2 - y) * w ..][0..w];
        const below = frame[(h / 2 + y) * w ..][0..w];
        try std.testing.expectEqualSlices(u32, above, below);
    }
}

test "the backend refuses what a Glacier warp cannot do" {
    const allocator = std.testing.allocator;

    // A kernel that returns a value. There is no caller to return it to.
    {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const entry = try f.appendBlock();
        const n = try f.appendInst(entry, u32_t, .{ .iconst = 7 });
        f.setTerminator(entry, .{ .ret = ir.function.Ret.one(n) });
        try std.testing.expectError(
            error.Unsupported,
            glacier.isel.compileKernel(allocator, &f, .{ .profile = .@"gc1.n" }),
        );
    }

    // An entry parameter that is not a hardware builtin. A kernel has no parameter block to read
    // it from, so the value would have to come from nowhere.
    {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const entry = try f.appendBlock();
        _ = try f.appendBlockParam(entry, u32_t);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
        try std.testing.expectError(
            error.Unsupported,
            glacier.isel.compileKernel(allocator, &f, .{ .profile = .@"gc1.n" }),
        );
    }

    // A global the options do not place. Guessing an address writes somewhere real.
    {
        var f = Function.init(allocator);
        defer f.deinit();
        const ptr_t = try f.types.ptrGlobal();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const entry = try f.appendBlock();
        const p = try f.appendGlobalAddr(entry, ptr_t, "nowhere");
        const v = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
        try f.appendStore(entry, v, p);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
        try std.testing.expectError(
            error.Unsupported,
            glacier.isel.compileKernel(allocator, &f, .{ .profile = .@"gc1.n" }),
        );
    }
}

/// The same fixed-point algorithm written plainly, so neither word stream is checked only against
/// the other. A frame that matches this matches a third description of the arithmetic.
fn reference(allocator: std.mem.Allocator, view: hand.View) ![]u32 {
    const frame = try allocator.alloc(u32, pixels);
    const shift = hand.colourShift(iterations);
    const escape: u32 = 4 << 12;
    for (0..@intCast(height)) |y| {
        const cy: i32 = view.y_min + @as(i32, @intCast(y)) * view.dy;
        for (0..@intCast(width)) |x| {
            const cx: i32 = view.x_min + @as(i32, @intCast(x)) * view.dx;
            var zx: i32 = 0;
            var zy: i32 = 0;
            var left: u32 = iterations;
            while (true) {
                const zx2: u32 = @as(u32, @bitCast(zx *% zx)) >> 12;
                const zy2: u32 = @as(u32, @bitCast(zy *% zy)) >> 12;
                if (zx2 +% zy2 >= escape) break;
                const cross = zx *% zy;
                const sign = cross >> 31;
                const magnitude: u32 = @bitCast((cross ^ sign) -% sign);
                // Put the sign back on after the logical shift, so the scaling truncates toward
                // zero. An arithmetic shift would round toward minus infinity instead.
                const scaled: i32 = @bitCast(magnitude >> 12);
                zy = ((scaled ^ sign) -% sign) *% 2 +% cy;
                zx = @bitCast(zx2 -% zy2 +% @as(u32, @bitCast(cx)));
                left -= 1;
                if (left == 0) break;
            }
            const level = left << shift;
            frame[y * @as(usize, @intCast(width)) + x] = level | (level << 8);
        }
    }
    return frame;
}

test "both word streams render what the plain algorithm renders" {
    const allocator = std.testing.allocator;
    const view = hand.View.fit(width, height);
    const expected = try reference(allocator, view);
    defer allocator.free(expected);

    var buf: [128]u32 = undefined;
    const by_hand = try warp.render(allocator, handWords(&buf, view), fb_base, pixels, 4, budget);
    defer allocator.free(by_hand);
    try std.testing.expectEqualSlices(u32, expected, by_hand);

    const words = try lower(allocator, view);
    defer allocator.free(words);
    const lowered = try warp.render(allocator, words, fb_base, pixels, 4, budget);
    defer allocator.free(lowered);
    try std.testing.expectEqualSlices(u32, expected, lowered);
}

/// Where the small kernels below put their data. Not zero, so forming the address needs both
/// halves of a wide constant rather than a single immediate.
const data_base: u32 = 0x1000;

/// Run one warp over `data`, with the words after `scratch_at` standing in for the hart scratch
/// region the SoC build reserves. A hart reads one past the top of its own slice from `mscratch`.
fn runOnce(code: []const u32, data: []u32, hart_id: u32, hart_count: u32) !void {
    const stride = glacier.encode.Profile.@"gc1.n".scratchBytes();
    const scratch_words = (stride / 4) * hart_count;
    const scratch_at = if (data.len > scratch_words) data.len - scratch_words else data.len;
    const scratch_base = data_base + @as(u32, @intCast(scratch_at)) * 4;
    var mem: warp.Memory = .{
        .base = data_base,
        .words = data,
        .scratch = .{ .base = scratch_base, .stride = stride, .harts = hart_count },
    };
    var one: warp.Warp = .{
        .hart_id = hart_id,
        .hart_count = hart_count,
        // The warp runs the full width of the profile the kernels here are built for. Leaving
        // this at its default ran ONE lane, which made a per-lane test pass on lane zero alone.
        .lanes = glacier.encode.Profile.@"gc1.n".lanes(),
        .stack_top = scratch_base + (hart_id + 1) * stride,
    };
    _ = try warp.run(code, &mem, &one, budget);
}

fn compile(allocator: std.mem.Allocator, f: *const Function) ![]u32 {
    return glacier.isel.compileKernel(allocator, f, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "data", .address = data_base }},
    });
}

test "a loop that loads, selects and stores" {
    const allocator = std.testing.allocator;
    const count: i64 = 6;

    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const warp_id = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, warp_id, .block_id_x);
    const loop = try f.appendBlock();
    const index = try f.appendBlockParam(loop, u32_t);
    const slot = try f.appendBlockParam(loop, ptr_t);
    const exit = try f.appendBlock();

    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const start = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
    try f.setJump(entry, loop, &.{ start, base });

    const value = try f.appendInst(loop, i32_t, .{ .load = .{ .ptr = slot } });
    const floor = try f.appendInst(loop, i32_t, .{ .iconst = 7 });
    const above = try f.appendInst(loop, bool_t, .{ .icmp = .{ .op = .gt, .lhs = value, .rhs = floor } });
    const clamped = try f.appendInst(loop, i32_t, .{ .select = .{ .cond = above, .then = value, .@"else" = floor } });
    const out = try f.appendArithImm(loop, ptr_t, .add, slot, count * 4);
    try f.appendStore(loop, clamped, out);
    const next_slot = try f.appendArithImm(loop, ptr_t, .add, slot, 4);
    const next_index = try f.appendArithImm(loop, u32_t, .add, index, 1);
    const limit = try f.appendInst(loop, u32_t, .{ .iconst = count });
    const more = try f.appendInst(loop, bool_t, .{ .icmp = .{ .op = .ne, .lhs = next_index, .rhs = limit } });
    try f.appendIf(loop, more, .{ .target = loop, .args = &.{ next_index, next_slot } }, .{ .target = exit });
    f.setTerminator(exit, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);

    const inputs = [_]i32{ -3, 0, 7, 8, 100, -1 };
    var data: [12]u32 = @splat(0);
    for (inputs, 0..) |v, i| data[i] = @bitCast(v);
    try runOnce(words, &data, 0, 1);
    for (inputs, 0..) |v, i| {
        try std.testing.expectEqual(@as(i32, @max(v, 7)), @as(i32, @bitCast(data[6 + i])));
    }
}

test "a conditional reaches both arms whichever one is laid out next" {
    // The then arm's target is the very next block. A jump whose target follows it can be left
    // out, but not this one: the branch skips exactly one word, so dropping the then jump would
    // make a false condition skip the else jump instead and fall into the then arm.
    const allocator = std.testing.allocator;

    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const warp_id = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, warp_id, .block_id_x);
    const first = try f.appendBlock();
    const second = try f.appendBlock();
    const join = try f.appendBlock();

    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const zero = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
    const later = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = warp_id, .rhs = zero } });
    try f.appendIf(entry, later, .{ .target = first }, .{ .target = second });

    const eleven = try f.appendInst(first, u32_t, .{ .iconst = 11 });
    try f.appendStore(first, eleven, base);
    try f.setJump(first, join, &.{});
    const twenty_two = try f.appendInst(second, u32_t, .{ .iconst = 22 });
    try f.appendStore(second, twenty_two, base);
    try f.setJump(second, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);

    var data: [1]u32 = .{0};
    try runOnce(words, &data, 0, 2);
    try std.testing.expectEqual(@as(u32, 22), data[0]);
    try runOnce(words, &data, 1, 2);
    try std.testing.expectEqual(@as(u32, 11), data[0]);
}

/// A kernel that loads `count` words, holds them all live at once, and stores their sum. Every
/// load happens before the first add, so `count` values are live together and anything above the
/// register file has to go to the scratch region.
fn buildSumKernel(allocator: std.mem.Allocator, count: usize) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");

    const loaded = try allocator.alloc(Value, count);
    defer allocator.free(loaded);
    for (loaded, 0..) |*v, i| {
        const at = try f.appendArithImm(entry, ptr_t, .add, base, @intCast(i * 4));
        v.* = try f.appendInst(entry, u32_t, .{ .load = .{ .ptr = at } });
    }
    var total = loaded[0];
    for (loaded[1..]) |v| total = try add(&f, entry, u32_t, total, v);
    const out = try f.appendArithImm(entry, ptr_t, .add, base, @intCast(count * 4));
    try f.appendStore(entry, total, out);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    return f;
}

test "a kernel with more live values than registers spills into the hart's scratch" {
    const allocator = std.testing.allocator;
    const count = 40; // above the 29 allocatable registers

    var f = try buildSumKernel(allocator, count);
    defer f.deinit();
    const words = try compile(allocator, &f);
    defer allocator.free(words);

    // The spill path has to have actually run, or this test passes for the wrong reason. A kernel
    // that spills reads `sp` from mscratch in its prologue, and nothing else here reads a CSR.
    try std.testing.expectEqual(
        glacier.encode.rv.csrrs(.x2, glacier.encode.Csr.mscratch, .x0),
        words[0],
    );

    // The words past the inputs and the output stand in for the hart's scratch region, so a spill
    // that walked out of its slice would be caught rather than silently land in the inputs.
    const scratch_words = comptime glacier.encode.Profile.@"gc1.n".scratchBytes() / 4;
    var data: [count + 1 + scratch_words]u32 = @splat(0);
    var want: u32 = 0;
    for (0..count) |i| {
        data[i] = @intCast(i * 7 + 1);
        want +%= data[i];
    }
    try runOnce(words, &data, 0, 1);
    try std.testing.expectEqual(want, data[count]);
}

test "a kernel that needs more scratch than the core has is refused" {
    // gc1.n holds 256 bytes per hart, so 64 slots. Far more live values than that cannot spill
    // anywhere, and walking below the region would corrupt the hart underneath.
    const allocator = std.testing.allocator;
    var f = try buildSumKernel(allocator, 400);
    defer f.deinit();
    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a kernel that does not spill never touches the stack pointer" {
    // The ABI reserves x2, and a kernel that has nothing to spill leaves it alone rather than
    // paying a word for a pointer it will not use.
    const allocator = std.testing.allocator;
    const words = try lower(allocator, hand.View.fit(width, height));
    defer allocator.free(words);
    const load_sp = glacier.encode.rv.csrrs(.x2, glacier.encode.Csr.mscratch, .x0);
    for (words) |w| try std.testing.expect(w != load_sp);
}

test "a workgroup barrier emits nothing, because a workgroup is one warp" {
    // The lanes of a warp are elements of one instruction stream, so they cannot be at different
    // points in the program. Glacier's own barrier synchronizes WARPS, which is wider than this
    // opcode asks for, and emitting it would park a warp until every other warp arrived.
    const allocator = std.testing.allocator;

    var plain = Function.init(allocator);
    defer plain.deinit();
    var barriered = Function.init(allocator);
    defer barriered.deinit();

    for ([_]*Function{ &plain, &barriered }, 0..) |f, i| {
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();
        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const v = try f.appendInst(entry, u32_t, .{ .iconst = 5 });
        if (i == 1) try f.appendBarrier(entry, .workgroup);
        try f.appendStore(entry, v, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    }

    const without = try compile(allocator, &plain);
    defer allocator.free(without);
    const with = try compile(allocator, &barriered);
    defer allocator.free(with);
    try std.testing.expectEqualSlices(u32, without, with);

    // And the cross-warp barrier is recorded even though nothing emits it, so the encoding and
    // the caution about its reserved FENCE field live where a reader will look.
    try std.testing.expectEqual(@as(u32, 0x1000000F), glacier.encode.warp_barrier);
    for (with) |w| try std.testing.expect(w != glacier.encode.warp_barrier);
}

test "a constant nobody reads costs no words" {
    // A frontend leaves these behind: an eliminated bounds check removes the only consumer of the
    // array length but not the constant itself. The allocator still places a result nobody reads,
    // so the emitter has to drop it on the use count rather than on whether it has a home.
    const allocator = std.testing.allocator;

    var plain = Function.init(allocator);
    defer plain.deinit();
    var littered = Function.init(allocator);
    defer littered.deinit();

    for ([_]*Function{ &plain, &littered }, 0..) |f, i| {
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();
        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        if (i == 1) {
            _ = try f.appendInst(entry, u32_t, .{ .iconst = 4096 });
            _ = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
        }
        const v = try f.appendInst(entry, u32_t, .{ .iconst = 7 });
        try f.appendStore(entry, v, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    }

    const without = try compile(allocator, &plain);
    defer allocator.free(without);
    const with = try compile(allocator, &littered);
    defer allocator.free(with);
    try std.testing.expectEqualSlices(u32, without, with);
}

test "a per-lane operation against a constant has a vector form for every operator" {
    // The vector unit has a scalar-operand form for only a few operators, so everything else
    // broadcasts the constant and uses vector-vector. A bitwise AND against a mask and a modulo
    // by a constant are both ordinary per-lane shapes, and refusing them would send a whole
    // kernel back to scalar for the sake of one instruction.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    // (tid & 6) ^ 1, then % 3, then - 1: and, xor, rem and sub all lack a scalar-operand form.
    const masked = try f.appendArithImm(entry, u32_t, .bit_and, tid, 6);
    const flipped = try f.appendArithImm(entry, u32_t, .bit_xor, masked, 1);
    const wrapped = try f.appendArithImm(entry, u32_t, .rem, flipped, 3);
    const less = try f.appendArithImm(entry, u32_t, .sub, wrapped, 1);
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, less, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const tidv: u32 = @intCast(i);
        const want = (((tidv & 6) ^ 1) % 3) -% 1;
        try std.testing.expectEqual(want, data[i]);
    }
}

test "a same-width convert costs no instruction" {
    // The SPIR-V frontend emits `convert` for `OpUConvert` and `OpSConvert`, and a kernel that
    // only ever holds words gets one between two 32-bit integers, which changes the signedness
    // and nothing else. Refusing it sent a whole kernel back when a move, coalesced away, is the
    // entire lowering.
    const allocator = std.testing.allocator;

    var plain = Function.init(allocator);
    defer plain.deinit();
    var converted = Function.init(allocator);
    defer converted.deinit();

    for ([_]*Function{ &plain, &converted }, 0..) |f, i| {
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();
        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const v = try f.appendInst(entry, u32_t, .{ .iconst = 9 });
        const stored = if (i == 1) blk: {
            const as_signed = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = v } });
            break :blk try f.appendInst(entry, u32_t, .{ .convert = .{ .value = as_signed } });
        } else v;
        try f.appendStore(entry, stored, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    }

    const without = try compile(allocator, &plain);
    defer allocator.free(without);
    const with = try compile(allocator, &converted);
    defer allocator.free(with);
    try std.testing.expectEqualSlices(u32, without, with);

    var data: [1]u32 = .{0};
    try runOnce(with, &data, 0, 1);
    try std.testing.expectEqual(@as(u32, 9), data[0]);
}

test "the scalar divide matches the standard layout" {
    // `fdiv.s f8, f9, f10` from `zig cc -target riscv64-linux-musl -mcpu=generic_rv64+v+f`.
    // Glacier has no f register file, so the fields name X registers: the same word, read against
    // a different file.
    //
    // This sits here rather than beside the encoder because `glacier/encode.zig` is not a test
    // root of its own and `refAllDecls` does not reach into it, so a test written there runs
    // nowhere. Checked: it did not appear in any suite until it moved.
    try std.testing.expectEqual(@as(u32, 0x18a4f453), glacier.encode.fdiv(.x8, .x9, .x10));
}

test "a per-lane value stays per-lane through a signedness convert" {
    // `convert` is in the set `checkSupported` accepts, so the divergence analysis has to visit
    // its operand. Without that the result of a convert reads as warp-uniform while its source
    // sits in a vector register, and the two register files are indexed by the same number.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const signed = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const shifted = try f.appendArithImm(entry, i32_t, .add, signed, 100);
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, shifted, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(100 + i)), data[i]);
    }
}

test "a float constant is its bit pattern, with a subnormal flushed to zero" {
    // The lane flushes a subnormal on every operation, so a constant that stayed subnormal would
    // not equal the same number the hardware computes.
    const allocator = std.testing.allocator;
    const cases = [_]struct { value: f64, want: u32 }{
        .{ .value = 1.0, .want = 0x3F80_0000 },
        .{ .value = -2.5, .want = 0xC020_0000 },
        .{ .value = 0.0, .want = 0 },
        // Smaller than the smallest normal f32, so it flushes and keeps its sign.
        .{ .value = 1.0e-40, .want = 0 },
        .{ .value = -1.0e-40, .want = 0x8000_0000 },
    };

    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const k = try f.appendInst(entry, f32_t, .{ .fconst = c.value });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = k } });
        try f.appendStore(entry, raw, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var data: [1]u32 = .{0xDEAD_BEEF};
        try runOnce(words, &data, 0, 1);
        try std.testing.expectEqual(c.want, data[0]);
    }
}

test "the three converts the core decodes run per lane" {
    // `vfcvt.f.x.v` reads its source SIGNED, `vfcvt.rtz.xu.f.v` writes an UNSIGNED result, and
    // `vfncvt.f.f.w` rounds to nearest even, so an overflow gives infinity. Each lane gets its
    // own value, so a convert that collapsed to lane zero shows up here.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    // lane - 1, so lane zero is negative and proves the source is read signed.
    const signed = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const biased = try f.appendArithImm(entry, i32_t, .sub, signed, 1);
    const half = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = biased } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = half } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, raw, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    var data: [8]u32 = @splat(0);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const want: f16 = @floatFromInt(@as(i32, @intCast(i)) - 1);
        try std.testing.expectEqual(@as(u32, @as(u16, @bitCast(want))), data[i]);
    }
}

test "a narrowing convert rounds to nearest even and overflows to infinity" {
    // Two things are pinned here. The ROUNDING: 2051 sits exactly between the halves 2050 and
    // 2052, so nearest even gives 2052 while a toward-zero narrowing gives 2050. The OVERFLOW:
    // nearest even carries a value past the largest finite half up to infinity, where toward zero
    // would stop at 0x7BFF.
    //
    // 1.5 is here because it is what tells a narrowing convert apart from the integer one: read
    // as a signed integer, 1.5's bit pattern is about 1.07e9 and converts to infinity, which is
    // the same answer the overflow case wants.
    const allocator = std.testing.allocator;
    const cases = [_]struct { value: f64, want: u32, not: u32 }{
        .{ .value = 1.5, .want = 0x3E00, .not = 0x7C00 },
        .{ .value = 2051.0, .want = 0x6802, .not = 0x6801 },
        .{ .value = 100000.0, .want = 0x7C00, .not = 0x7BFF },
    };
    // The expectations above are what nearest even gives, and the rejected ones what toward zero
    // gives one step below.
    try std.testing.expectEqual(cases[1].want, @as(u32, @as(u16, @bitCast(@as(f16, 2052.0)))));
    try std.testing.expectEqual(cases[1].not, @as(u32, @as(u16, @bitCast(@as(f16, 2050.0)))));

    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const f16_t = try f.types.intern(.{ .float = .f16 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const wide = try f.appendInst(entry, f32_t, .{ .fconst = c.value });
        const half = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = wide } });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = half } });
        try f.appendStore(entry, raw, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var data: [1]u32 = .{0};
        try runOnce(words, &data, 0, 1);
        try std.testing.expectEqual(c.want, data[0]);
    }
}

test "a float to unsigned convert truncates toward zero" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const v = try f.appendInst(entry, f32_t, .{ .fconst = 7.9 });
    const n = try f.appendInst(entry, u32_t, .{ .convert = .{ .value = v } });
    try f.appendStore(entry, n, base);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    var data: [1]u32 = .{0};
    try runOnce(words, &data, 0, 1);
    try std.testing.expectEqual(@as(u32, 7), data[0]);
}

test "a warp-uniform float divide becomes the one scalar divide" {
    // The perspective divide is what this exists for: one divide per vertex or per scanline, not
    // per pixel. It truncates, which 1.0 / 3.0 shows.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const one = try f.appendInst(entry, f32_t, .{ .fconst = 1.0 });
    const three = try f.appendInst(entry, f32_t, .{ .fconst = 3.0 });
    const q = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .div, .lhs = one, .rhs = three } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = q } });
    try f.appendStore(entry, raw, base);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    var data: [1]u32 = .{0};
    try runOnce(words, &data, 0, 1);
    // 1.0 / 3.0 rounds to nearest UP, so truncating gives one unit in the last place less.
    try std.testing.expectEqual(@as(u32, @as(u32, @bitCast(@as(f32, 1.0) / @as(f32, 3.0))) - 1), data[0]);
}

test "the backend refuses the float work Glacier cannot do" {
    const allocator = std.testing.allocator;
    const Shape = enum { add, compare, half_store, per_lane_divide, widen };
    for ([_]Shape{ .add, .compare, .half_store, .per_lane_divide, .widen }) |shape| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const f16_t = try f.types.intern(.{ .float = .f16 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const a = try f.appendInst(entry, f32_t, .{ .fconst = 3.0 });
        const b = try f.appendInst(entry, f32_t, .{ .fconst = 5.0 });
        const stored = switch (shape) {
            // An addition of two f32 VALUES has no lowering. The MAC covers an addition only when one
            // side is an f16 factor or a constant an f16 holds, because the hardware adds nothing
            // that is not a widening product.
            .add => blk: {
                const x = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = base } });
                const y = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = base } });
                const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
                break :blk try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = sum } });
            },
            // There is no float compare, and an integer compare over the bits is a different
            // relation.
            .compare => blk: {
                const lt = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } });
                const one = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
                const zero = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
                break :blk try f.appendInst(entry, u32_t, .{ .select = .{ .cond = lt, .then = one, .@"else" = zero } });
            },
            // An f16 is the low half of a lane, so a word store would write bits that are not
            // part of the value.
            .half_store => blk: {
                const signed_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
                const s = try f.appendInst(entry, signed_t, .{ .convert = .{ .value = tid } });
                break :blk try f.appendInst(entry, f16_t, .{ .convert = .{ .value = s } });
            },
            // A per-lane divide would have to feed the single scalar divider a lane at a time.
            .per_lane_divide => blk: {
                const lane = try f.appendInst(entry, f32_t, .{ .unary = .{ .op = .reinterpret, .value = tid } });
                const q = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .div, .lhs = lane, .rhs = b } });
                break :blk try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = q } });
            },
            // Nothing WIDENS an f16: the core has no `vfwcvt`.
            .widen => blk: {
                const signed_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
                const s = try f.appendInst(entry, signed_t, .{ .convert = .{ .value = tid } });
                const h = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = s } });
                const wide = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = h } });
                break :blk try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = wide } });
            },
        };
        try f.appendStore(entry, stored, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        try std.testing.expectError(error.Unsupported, compile(allocator, &f));
    }
}

/// How many words in `code` splat x0 into a vector, which is how a destination is zeroed.
fn countZeroSplats(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b100) continue;
        if (w >> 26 != 0b010111) continue;
        if ((w >> 15) & 0x1F != 0) continue;
        n += 1;
    }
    return n;
}

/// How many words in `code` are `vmv.v.v`: OP-V, OPIVV, funct6 010111.
fn countVectorMoves(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b000) continue;
        if (w >> 26 != 0b010111) continue;
        n += 1;
    }
    return n;
}

/// How many words in `code` are `vfwmacc.vv`: OP-V, OPFVV, funct6 111100.
fn countMacs(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b001) continue;
        if (w >> 26 != 0b111100) continue;
        n += 1;
    }
    return n;
}

test "a per-lane multiply-accumulate over f16 becomes one widening MAC" {
    // `out[lane] += (lane + 1) * (lane + 2)`, with the two factors as f16 and the accumulator as
    // f32, which is the shape a GEMM inner loop has. The IR spells it as two widening converts, a
    // multiply and an add, and Glacier has an instruction for none of those on their own.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    // The accumulator is read per lane, which is the only way to get a per-lane f32: nothing
    // widens an f16 outside the MAC itself.
    const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const first = try f.appendArithImm(entry, i32_t, .add, lane, 1);
    const second = try f.appendArithImm(entry, i32_t, .add, lane, 2);
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = first } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = second } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wb } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, 100.0));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        const want: f32 = 100.0 + (n + 1.0) * (n + 2.0);
        try std.testing.expectEqual(@as(u32, @bitCast(want)), data[i]);
    }
}

test "a widening convert read twice cannot be absorbed, so the tree is refused" {
    // Absorbing an instruction is only safe when one reader exists. Here the convert feeds both
    // factors, so neither the full MAC nor the bare product matches, and a widening convert has no
    // instruction of its own.
    const allocator = std.testing.allocator;
    for ([_]bool{ false, true }) |second_reader| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const f16_t = try f.types.intern(.{ .float = .f16 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
        const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
        const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
        const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
        const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wa } });
        const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
        try f.appendStore(entry, sum, at);
        if (second_reader) {
            const other = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = product, .rhs = product } });
            try f.appendStore(entry, other, base);
        }
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        // `wa` is read twice even in the first case, so neither shape folds.
        try std.testing.expectError(error.Unsupported, compile(allocator, &f));
    }
}

test "a warp-uniform multiply-accumulate goes out to the vector unit and comes back" {
    // `vfwmacc.vv` is the only float multiplier the core has, so a MAC whose operands are all the
    // same in every lane still runs on the vector unit: the accumulator and both factors are
    // broadcast into scratches, and lane zero carries the answer back. That is three scratches at
    // once, which is the most any one MAC needs.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const acc = try f.appendInst(entry, f32_t, .{ .fconst = 1.0 });
    const three = try f.appendInst(entry, i32_t, .{ .iconst = 3 });
    const four = try f.appendInst(entry, i32_t, .{ .iconst = 4 });
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = three } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = four } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wb } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = sum } });
    try f.appendStore(entry, raw, base);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    var data: [1]u32 = .{0};
    try runOnce(words, &data, 0, 1);
    try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.0 + 3.0 * 4.0))), data[0]);
}

test "a MAC mixes a per-lane accumulator with warp-uniform factors" {
    // The accumulator varies per lane and both factors do not, so each factor is broadcast into a
    // scratch while the accumulator keeps its own vector register. Reading a warp-uniform factor
    // as a vector register would read whatever that scalar register number holds in the other
    // file, so this is the shape that used to be refused.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
    const seven = try f.appendInst(entry, i32_t, .{ .iconst = 7 });
    const eight = try f.appendInst(entry, i32_t, .{ .iconst = 8 });
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = seven } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = eight } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wb } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, @floatFromInt(i * 10)));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i * 10);
        try std.testing.expectEqual(@as(u32, @bitCast(n + 56.0)), data[i]);
    }
}

test "a bare widening product zeroes the destination and accumulates into it" {
    // `out[lane] = (lane + 1) * (lane + 2)` as f16 factors with an f32 result. There is nothing to
    // accumulate into, and no source rewrite can invent one: an added 0.0 folds away before the
    // module arrives. The sum of a zero and the exact product is the product.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const first = try f.appendArithImm(entry, i32_t, .add, lane, 1);
    const second = try f.appendArithImm(entry, i32_t, .add, lane, 2);
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = first } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = second } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wb } });
    try f.appendStore(entry, product, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));
    // The accumulator starts at zero, and whatever the destination register held before is not
    // part of the answer. The behaviour check below cannot see this: an f16 bit pattern read as an
    // f32 is subnormal, which the lane flushes to zero anyway.
    try std.testing.expectEqual(@as(usize, 1), countZeroSplats(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast((n + 1.0) * (n + 2.0))), data[i]);
    }
}

test "an addition of a widened f16 becomes a MAC against one" {
    // The core has no float add, so `acc + f32(a)` is a MAC whose second factor is 1.0. Writing
    // the multiply by hand does not work: the optimiser folds `a * 1.0` back to `a` before the
    // module arrives, which is how this shape reaches the selector in the first place.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const first = try f.appendArithImm(entry, i32_t, .add, lane, 1);
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = first } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = wa } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, 40.0));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(40.0 + n + 1.0)), data[i]);
    }
}

test "a chain of MACs keeps one accumulator register" {
    // `out[lane] += a * b + c` over three MACs: a full one, a bare product and an addition. The
    // accumulator hint is what keeps them on one register, so the chain costs no copies.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });

    // Each factor needs its own convert, because a value read twice cannot be absorbed.
    var halves: [5]Value = undefined;
    for (&halves, 1..) |*h, k| {
        const n = try f.appendArithImm(entry, i32_t, .add, lane, @intCast(k));
        h.* = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = n } });
    }
    const wides = blk: {
        var w: [5]Value = undefined;
        for (&w, halves) |*dst, h| dst.* = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = h } });
        break :blk w;
    };
    // acc + h1 * h2
    const p1 = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wides[0], .rhs = wides[1] } });
    const s1 = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p1 } });
    // + h3 * h4, which the IR spells as a bare product added on
    const p2 = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wides[2], .rhs = wides[3] } });
    const s2 = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = s1, .rhs = p2 } });
    // + h5
    const s3 = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = s2, .rhs = wides[4] } });
    try f.appendStore(entry, s3, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 3), countMacs(words));
    // One accumulator register means no `vmv.v.v` between the three.
    try std.testing.expectEqual(@as(usize, 0), countVectorMoves(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, 7.0));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        const want = 7.0 + (n + 1.0) * (n + 2.0) + (n + 3.0) * (n + 4.0) + (n + 5.0);
        try std.testing.expectEqual(@as(u32, @bitCast(want)), data[i]);
    }
}

test "a per-lane same-width convert costs no instruction either" {
    // The scalar path has always elided a move onto the same register. The vector path emitted
    // `vmv.v.v v1, v1`, one wasted word for every signedness convert of a per-lane value.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const signed = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const doubled = try f.appendArithImm(entry, i32_t, .mul, signed, 3);
    try f.appendStore(entry, doubled, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countVectorMoves(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i * 3)), data[i]);
    }
}

test "an addition of two widened f16 is refused rather than split into two MACs" {
    // `f32(x) + f32(y)` has no accumulator on either side. It would take a zeroed destination and
    // then two MACs against one, which means synthesizing an intermediate the IR does not hold.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const other = try f.appendArithImm(entry, i32_t, .add, lane, 1);
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = other } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = wa, .rhs = wb } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a constant factor reaches the MAC already widened by the frontend" {
    // `@as(f32, a) * @as(f32, b)` with a comptime `b` arrives as a multiply by an f32 CONSTANT:
    // widening a constant folds in the frontend, so only one side is a convert. Every rotation
    // constant in a renderer has this shape.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const three = try f.appendInst(entry, f32_t, .{ .fconst = 3.0 });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = three } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = product } });
    try f.appendStore(entry, raw, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n * 3.0)), data[i]);
    }
}

test "an addition of a constant is a MAC with both factors constant" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const acc = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = at } });
    const two = try f.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = two } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = sum } });
    try f.appendStore(entry, raw, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, @floatFromInt(i)));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n + 2.0)), data[i]);
    }
}

test "a constant an f16 cannot hold exactly is refused as a factor" {
    // Two ways a constant fails. 0.1 has no exact f16, so the core would multiply by a different
    // number than the IR named. 2^-15 has one, but only as a subnormal, and the lane flushes those
    // to zero, so the constant would behave as a zero.
    const allocator = std.testing.allocator;
    for ([_]f64{ 0.1, 3.0517578125e-05 }) |value| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const f16_t = try f.types.intern(.{ .float = .f16 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
        const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
        const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
        const k = try f.appendInst(entry, f32_t, .{ .fconst = value });
        const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = k } });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = product } });
        try f.appendStore(entry, raw, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        try std.testing.expectError(error.Unsupported, compile(allocator, &f));
    }
}

test "a MAC mixes a warp-uniform accumulator with a per-lane factor" {
    // The result varies per lane while the accumulator does not, so the two are in different
    // register FILES and must not be coalesced onto one register number. `copySource` reports the
    // accumulator to the allocator, and the files are indexed by the same numbers.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const f16_t = try f.types.intern(.{ .float = .f16 });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const acc = try f.appendInst(entry, f32_t, .{ .fconst = 1000.0 });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const two = try f.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = two } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(1000.0 + n * 2.0)), data[i]);
    }
}
