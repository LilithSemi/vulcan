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
    // A hart holds 1024 bytes, so 256 slots. Far more live values than that cannot spill anywhere,
    // and walking below the region would corrupt the hart underneath.
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
    const Shape = enum { half_arith, compare, half_store, per_lane_divide };
    for ([_]Shape{ .half_arith, .compare, .half_store, .per_lane_divide }) |shape| {
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
            // f32 arithmetic works; f16 arithmetic does not. The element width is fixed at 32 bits,
            // so a plain float op reads whole lanes, and an f16 is the low half of one. Arithmetic
            // on a pair of them is the MAC's shape and nothing else.
            .half_arith => blk: {
                const signed_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
                const s = try f.appendInst(entry, signed_t, .{ .convert = .{ .value = tid } });
                const h = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = s } });
                const sum = try f.appendInst(entry, f16_t, .{ .arith = .{ .op = .add, .lhs = h, .rhs = h } });
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

/// How many words in `code` are `vfwcvt.f.f.v`: OP-V, OPFVV, funct6 0x12, vs1 field 0x0C.
fn countWidens(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b001) continue;
        if (w >> 26 != 0x12 or (w >> 15) & 0x1F != 0x0C) continue;
        n += 1;
    }
    return n;
}

/// How many words in `code` are a plain float add, subtract or multiply.
fn countFloatArith(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b001) continue;
        switch (w >> 26) {
            0x00, 0x02, 0x24 => n += 1,
            else => {},
        }
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
    // The accumulator is read per lane, which is the GEMM shape: load the running sum, accumulate
    // into it, store it back.
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

test "one widened f16 feeds two products" {
    // `sphi` widened once and read by both `ux` and `uz` is what a sphere writes, and absorbing a
    // convert used to need exactly one reader. The MAC reads the f16 SOURCE, so the number of
    // readers of the f32 never mattered; what it needed was for the allocator to be told, which
    // `fusedOperands` now does.
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
    const plus = try f.appendArithImm(entry, i32_t, .add, lane, 2);

    const shared = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = plus } });
    const other = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
    // One widening convert, two readers.
    const wide = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = shared } });
    const wother = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = other } });
    const ux = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wide, .rhs = wother } });
    const uz = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wide, .rhs = wide } });
    const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = ux, .rhs = uz } });
    try f.appendStore(entry, sum, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // `uz` is a bare product; `ux` accumulates it, so two MACs cover all three operations.
    try std.testing.expectEqual(@as(usize, 2), countMacs(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        const shared_v = n + 2.0;
        try std.testing.expectEqual(@as(u32, @bitCast(shared_v * n + shared_v * shared_v)), data[i]);
    }
}

test "a partly absorbed widening convert still produces its f32" {
    // One reader is a MAC and the other is a store, so the convert cannot be suppressed: it is
    // widened on its own for the store with `vfwcvt.f.f.v`, and the MAC reads the f16 beside it.
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
    const far = try f.appendArithImm(entry, u32_t, .add, offset, 64);
    const at2 = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = far } });
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const plus = try f.appendArithImm(entry, i32_t, .add, lane, 3);
    const h = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = plus } });
    const wide = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = h } });
    const two = try f.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wide, .rhs = two } });
    try f.appendStore(entry, product, at);
    try f.appendStore(entry, wide, at2);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // One MAC for the product, and the convert the store needs is one widening instruction.
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));
    try std.testing.expectEqual(@as(usize, 1), countWidens(words));

    var data: [32]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast((n + 3.0) * 2.0)), data[i]);
        try std.testing.expectEqual(@as(u32, @bitCast(n + 3.0)), data[16 + i]);
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

test "an addition of a widened f16 is a widening convert and an add" {
    // Two instructions. A MAC against a constant 1.0 would be four once the constant is
    // materialized and broadcast, so the native pair wins now that the core has a float add.
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
    try std.testing.expectEqual(@as(usize, 0), countMacs(words));
    try std.testing.expectEqual(@as(usize, 1), countWidens(words));
    try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

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
    // `out[lane] += a * b + c * d + e`: a full MAC, a bare product the addition accumulates, and a
    // final widened term that is now a convert and an add rather than a third MAC. The accumulator
    // hint is what keeps the chain on one register, so it costs no copies.
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
    try std.testing.expectEqual(@as(usize, 2), countMacs(words));
    try std.testing.expectEqual(@as(usize, 1), countWidens(words));
    // One accumulator register means no `vmv.v.v` between them.
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

test "an addition of two widened f16 is two converts and an add" {
    // Neither side is an accumulator and neither needs to be: each f16 widens with one
    // instruction and the addition is one more. No MAC is involved at all.
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

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countMacs(words));
    try std.testing.expectEqual(@as(usize, 2), countWidens(words));
    try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n + (n + 1.0))), data[i]);
    }
}

test "a widening convert on its own is one instruction" {
    // `vfwcvt.f.f.v`. It used to be a MAC against a constant 1.0, which cost a zeroed destination,
    // a materialized constant and a broadcast: five words where this is one.
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
    const wide = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = wide } });
    try f.appendStore(entry, raw, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countMacs(words));
    try std.testing.expectEqual(@as(usize, 1), countWidens(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n)), data[i]);
    }
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

test "an addition of a float constant is one add" {
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
    try std.testing.expectEqual(@as(usize, 0), countMacs(words));
    try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var data: [8]u32 = @splat(0);
    for (0..lanes) |i| data[i] = @bitCast(@as(f32, @floatFromInt(i)));
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n + 2.0)), data[i]);
    }
}

test "a constant no f16 holds exactly keeps its full precision" {
    // Neither can be a MAC factor: 0.1 has no exact f16, and 2^-15 has one only as a SUBNORMAL,
    // which the lane flushes to zero. That used to be a refusal, because the MAC was the only
    // float multiply. With a native one the constant stays an f32 and the f16 widens, so the
    // answer is the one the source asked for rather than a narrowed approximation of it.
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
        const plus = try f.appendArithImm(entry, i32_t, .add, lane, 3);
        const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = plus } });
        const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
        const k = try f.appendInst(entry, f32_t, .{ .fconst = value });
        const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = k } });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = product } });
        try f.appendStore(entry, raw, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 0), countMacs(words));
        try std.testing.expectEqual(@as(usize, 1), countWidens(words));
        try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        // The exact f32 product is the discriminator on its own: the MAC path could not produce it
        // for either constant. 0.1 would have been rounded to an f16 first, and 2^-15 IS an exact
        // f16 but only a subnormal one, which the lane flushes, so that path would give zero.
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            const n: f32 = @floatFromInt(i + 3);
            const want = n * @as(f32, @floatCast(value));
            try std.testing.expectEqual(@as(u32, @bitCast(want)), data[i]);
            try std.testing.expect(want != 0.0);
        }
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

test "a positive zero accumulator becomes a zeroed destination, a negative one does not" {
    // `mac(0.0, a, b)` written by hand carries a real `0.0 +`. Reading that as a bare product
    // costs one word instead of materializing the zero into a register and broadcasting it.
    //
    // A MINUS zero is kept, because it is the one accumulator value the shortcut would change:
    // adding plus zero to a minus zero product gives plus zero.
    const allocator = std.testing.allocator;
    for ([_]f64{ 0.0, -0.0 }) |zero| {
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
        const acc = try f.appendInst(entry, f32_t, .{ .fconst = zero });
        const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
        const one_more = try f.appendArithImm(entry, i32_t, .add, lane, 1);
        const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = one_more } });
        const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
        const three = try f.appendInst(entry, f32_t, .{ .fconst = 3.0 });
        const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = three } });
        const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
        try f.appendStore(entry, sum, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 1), countMacs(words));
        // A plus zero zeroes the destination from x0; a minus zero is broadcast like any value.
        const want_splats: usize = if (zero == 0.0 and !std.math.signbit(zero)) 1 else 0;
        try std.testing.expectEqual(want_splats, countZeroSplats(words));

        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            const n: f32 = @floatFromInt(i);
            try std.testing.expectEqual(@as(u32, @bitCast((n + 1.0) * 3.0)), data[i]);
        }
    }
}

test "a MAC keeps its factors alive past the converts it absorbed" {
    // An absorbed convert emits nothing, but the IR still says the f16 dies there, so the
    // allocator is free to hand that register to a value defined between the convert and the MAC.
    // The MAC would then multiply whatever landed in it. `fusedOperands` is what tells the
    // allocator that the MAC, not the convert, is the real reader.
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
    const lane = try f.appendInst(entry, i32_t, .{ .convert = .{ .value = tid } });
    const other = try f.appendArithImm(entry, i32_t, .add, lane, 1);
    const ha = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = lane } });
    const hb = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = other } });
    const wa = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = ha } });
    const wb = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = hb } });
    // Per-lane values born after the absorbed converts, each live past the MAC, so the allocator
    // has every reason to reuse the registers the two f16 appear to have vacated.
    var junk: [6]Value = undefined;
    for (&junk, 1..) |*j, k| j.* = try f.appendArithImm(entry, u32_t, .mul, tid, @intCast(k + 1));
    const product = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .mul, .lhs = wa, .rhs = wb } });

    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, product, at);
    for (junk, 0..) |j, n| {
        const slot = try f.appendArithImm(entry, u32_t, .add, offset, @intCast((n + 1) * 32));
        const where = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = slot } });
        try f.appendStore(entry, j, where);
    }
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countMacs(words));

    var data: [64]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const n: f32 = @floatFromInt(i);
        try std.testing.expectEqual(@as(u32, @bitCast(n * (n + 1.0))), data[i]);
    }
}

test "a table of constants written once spills nothing" {
    // A SPIR-V module constant has nowhere natural to live, so the lowering puts every one at the
    // top of the entry block. A kernel that writes a table of them then holds the whole table at
    // once and the allocator spills it, which costs a store and a reload each where re-emitting
    // the constant costs one word. A warp has 256 bytes of scratch on the small profiles, so past
    // about 64 entries that stops being slow code and becomes a refusal.
    const allocator = std.testing.allocator;
    const entries = 48;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    var consts: [entries]Value = undefined;
    for (&consts, 0..) |*c, i| c.* = try f.appendInst(entry, u32_t, .{ .iconst = @intCast(0x1234_0000 + i) });
    for (consts, 0..) |c, i| {
        const at = try f.appendArithImm(entry, ptr_t, .add, base, @intCast(i * 4));
        try f.appendStore(entry, c, at);
    }
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // The prologue loads the stack pointer only when something spills.
    const load_sp = glacier.encode.rv.csrrs(.x2, glacier.encode.Csr.mscratch, .x0);
    for (words) |w| try std.testing.expect(w != load_sp);

    var data: [64]u32 = @splat(0);
    try runOnce(words, &data, 0, 1);
    for (0..entries) |i| try std.testing.expectEqual(@as(u32, @intCast(0x1234_0000 + i)), data[i]);
}

test "an atomic add is one instruction and gives back the old value" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const one = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
    const old = try f.appendAtomicRmw(entry, .{
        .op = .add,
        .ptr = base,
        .value = one,
        .ordering = .relaxed,
        .scope = .device,
    });
    const at = try f.appendArithImm(entry, ptr_t, .add, base, 4);
    try f.appendStore(entry, old, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    var amos: usize = 0;
    for (words) |w| {
        if (w & 0x7F == 0b0101111) amos += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), amos);

    var data: [4]u32 = @splat(0);
    data[0] = 40;
    try runOnce(words, &data, 0, 1);
    // Memory advanced, and what came back is what was there before.
    try std.testing.expectEqual(@as(u32, 41), data[0]);
    try std.testing.expectEqual(@as(u32, 40), data[1]);
}

test "the atomics Glacier cannot do are refused" {
    const allocator = std.testing.allocator;
    const Shape = enum { compare_exchange, per_lane };
    for ([_]Shape{ .compare_exchange, .per_lane }) |shape| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const one = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
        const old = switch (shape) {
            // Compare-and-swap needs the load-reserved pair, and Glacier decodes the AMOs without
            // it. See `encode.zig`.
            .compare_exchange => try f.appendAtomicRmw(entry, .{
                .op = .compare_exchange,
                .ptr = base,
                .value = one,
                .compare = one,
                .ordering = .relaxed,
                .scope = .device,
            }),
            // One bus, one read-modify-write. A per-lane atomic is a loop over the lanes.
            .per_lane => blk: {
                const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
                const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
                break :blk try f.appendAtomicRmw(entry, .{
                    .op = .add,
                    .ptr = at,
                    .value = one,
                    .ordering = .relaxed,
                    .scope = .device,
                });
            },
        };
        try f.appendStore(entry, old, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        try std.testing.expectError(error.Unsupported, compile(allocator, &f));
    }
}

test "an atomic minimum compares as the operand type, not always as signed" {
    // 0xFFFFFFFF is the largest unsigned value and minus one as a signed one, so the two forms of
    // the instruction disagree about it. The IR says the compare is against the OPERAND type.
    const allocator = std.testing.allocator;
    for ([_]std.builtin.Signedness{ .unsigned, .signed }) |signedness| {
        var f = Function.init(allocator);
        defer f.deinit();
        const ty = try f.types.intern(.{ .int = .{ .signedness = signedness, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const one = try f.appendInst(entry, ty, .{ .iconst = 1 });
        const old = try f.appendAtomicRmw(entry, .{
            .op = .min,
            .ptr = base,
            .value = one,
            .ordering = .relaxed,
            .scope = .device,
        });
        const at = try f.appendArithImm(entry, ptr_t, .add, base, 4);
        try f.appendStore(entry, old, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var data: [4]u32 = @splat(0);
        data[0] = 0xFFFF_FFFF;
        try runOnce(words, &data, 0, 1);
        // Unsigned keeps the 1; signed keeps the minus one that was already there.
        const want: u32 = if (signedness == .unsigned) 1 else 0xFFFF_FFFF;
        try std.testing.expectEqual(want, data[0]);
        try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), data[1]);
    }
}

test "the ordering an atomic asks for reaches its aq and rl bits" {
    // A warp model that runs one warp to completion cannot observe ordering at all, so this is
    // checked where it is decided: in the emitted word.
    const allocator = std.testing.allocator;
    const cases = [_]struct { ordering: ir.function.AtomicOrdering, aq: bool, rl: bool }{
        .{ .ordering = .relaxed, .aq = false, .rl = false },
        .{ .ordering = .acquire, .aq = true, .rl = false },
        .{ .ordering = .release, .aq = false, .rl = true },
        .{ .ordering = .acq_rel, .aq = true, .rl = true },
        .{ .ordering = .seq_cst, .aq = true, .rl = true },
    };
    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const one = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
        try f.appendAtomicRmwStmt(entry, .{
            .op = .add,
            .ptr = base,
            .value = one,
            .ordering = c.ordering,
            .scope = .device,
        });
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var found = false;
        for (words) |w| {
            if (w & 0x7F != 0b0101111) continue;
            try std.testing.expectEqual(c.aq, (w >> 26) & 1 == 1);
            try std.testing.expectEqual(c.rl, (w >> 25) & 1 == 1);
            found = true;
        }
        // The old value goes nowhere, and the read-modify-write still has to happen.
        try std.testing.expect(found);
    }
}

test "warp-uniform float arithmetic goes out to the vector unit and comes back" {
    // There is no scalar form of any of these, so both operands are broadcast, the vector unit
    // does the work and lane zero carries the answer back.
    //
    // 2^20 plus 1 is deliberate on both counts. It is EXACT in f32, needing 21 of the 24 bits, so
    // no result here depends on the rounding mode, which is not recorded anywhere for these three.
    // And no f16 holds it, so the MAC cannot claim the multiply: a product of two constants that
    // ARE exact halves is still lowered as a MAC, correctly and in fewer words, which means a test
    // using small integers would never reach `vfmul` at all.
    const allocator = std.testing.allocator;
    const cases = [_]struct { op: ir.function.BinOp, want: f32 }{
        .{ .op = .add, .want = 1048579.0 },
        .{ .op = .sub, .want = 1048575.0 },
        .{ .op = .mul, .want = 2097154.0 },
    };
    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const big = try f.appendInst(entry, f32_t, .{ .fconst = 1048577.0 });
        const two = try f.appendInst(entry, f32_t, .{ .fconst = 2.0 });
        const r = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = c.op, .lhs = big, .rhs = two } });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = r } });
        try f.appendStore(entry, raw, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

        var data: [4]u32 = @splat(0);
        try runOnce(words, &data, 0, 1);
        try std.testing.expectEqual(@as(u32, @bitCast(c.want)), data[0]);
    }
}

test "per-lane float arithmetic over widened f16" {
    // `out[lane] = f32(lane + 5) - f32(lane + 3)`, which is 2 in every lane. Written as a subtract
    // so an encoder that reached for the add would answer differently in every lane.
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
    const a5 = try f.appendArithImm(entry, i32_t, .add, lane, 5);
    const a3 = try f.appendArithImm(entry, i32_t, .add, lane, 3);
    const h5 = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = a5 } });
    const h3 = try f.appendInst(entry, f16_t, .{ .convert = .{ .value = a3 } });
    const w5 = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = h5 } });
    const w3 = try f.appendInst(entry, f32_t, .{ .convert = .{ .value = h3 } });
    const diff = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .sub, .lhs = w5, .rhs = w3 } });
    try f.appendStore(entry, diff, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countMacs(words));
    try std.testing.expectEqual(@as(usize, 2), countWidens(words));
    try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 2.0))), data[i]);
    }
}

test "a float add rounds to nearest even and the MAC truncates" {
    // The lane is not uniform about rounding, so the two modes are pinned against each other on
    // the SAME arithmetic value. 1 + 3*2^-25 is three quarters of an ulp above 1.0 in f32, so
    // nearest even carries it up one ulp and truncation leaves it at 1.0. A rounding mux that
    // flipped either way would move that bit with no error anywhere, which is how it would be
    // missed: every other float test here uses operands whose result is exact.
    //
    // This pins the MODEL and the selection, not the hardware. The hardware's modes are recorded
    // in `warp.zig` with where they were read from.
    const allocator = std.testing.allocator;
    const quarter_ulp: f32 = @floatCast(@as(f64, 3.0) / 33554432.0); // 3 * 2^-25, exact in f32
    const rounded_up: f32 = 1.0 + std.math.floatEps(f32);
    try std.testing.expect(rounded_up != 1.0);

    // The add: nearest even, so the answer is one ulp above 1.0.
    {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const one = try f.appendInst(entry, f32_t, .{ .fconst = 1.0 });
        const tiny = try f.appendInst(entry, f32_t, .{ .fconst = quarter_ulp });
        const sum = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = one, .rhs = tiny } });
        const raw = try f.appendInst(entry, u32_t, .{ .unary = .{ .op = .reinterpret, .value = sum } });
        try f.appendStore(entry, raw, base);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 1), countFloatArith(words));
        var data: [4]u32 = @splat(0);
        try runOnce(words, &data, 0, 1);
        try std.testing.expectEqual(@as(u32, @bitCast(rounded_up)), data[0]);
    }

    // The MAC: toward zero on the same quantity, so the answer stays 1.0. The factors are exact
    // f16 and their product is exactly the three quarters of an ulp the add rounded up.
    {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const f16_t = try f.types.intern(.{ .float = .f16 });
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const acc = try f.appendInst(entry, f32_t, .{ .fconst = 1.0 });
        // 3 * 2^-13 and 2^-12, both normal halves, whose product is 3 * 2^-25 exactly.
        const ha = try f.appendInst(entry, f16_t, .{ .fconst = @as(f64, 3.0) / 8192.0 });
        const hb = try f.appendInst(entry, f16_t, .{ .fconst = @as(f64, 1.0) / 4096.0 });
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
        var data: [4]u32 = @splat(0);
        try runOnce(words, &data, 0, 1);
        try std.testing.expectEqual(@as(u32, @bitCast(@as(f32, 1.0))), data[0]);
    }
}

/// How many words in `code` are an integer vector compare into a mask: OP-V, OPIVV or OPIVX, with
/// a funct6 in the compare block.
fn countCompares(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        const f3 = (w >> 12) & 0x7;
        if (f3 != 0b000 and f3 != 0b100) continue;
        if (w >> 26 < 0b011000 or w >> 26 > 0b011111) continue;
        n += 1;
    }
    return n;
}

/// How many words in `code` are `vmerge`, the one masked instruction emitted: funct6 010111 with
/// `vm` clear.
fn countMerges(code: []const u32) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if (w >> 26 != 0b010111 or (w >> 25) & 1 != 0) continue;
        n += 1;
    }
    return n;
}

test "a per-lane compare and select pick a different answer in each lane" {
    // `out[lane] = cmp(lane, 2) ? 100 : 200`, which is the shape a rasteriser's span test has and
    // which used to be refused outright: a per-lane compare writes a MASK, and reading lane zero's
    // answer for the whole warp would have been a wrong frame rather than a slow one.
    const allocator = std.testing.allocator;
    const cases = [_]struct { op: ir.function.CmpOp, taken: [4]bool }{
        .{ .op = .lt, .taken = .{ true, true, false, false } },
        .{ .op = .le, .taken = .{ true, true, true, false } },
        .{ .op = .gt, .taken = .{ false, false, false, true } },
        .{ .op = .ge, .taken = .{ false, false, true, true } },
        .{ .op = .eq, .taken = .{ false, false, true, false } },
        .{ .op = .ne, .taken = .{ true, true, false, true } },
    };
    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
        const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = c.op, .lhs = tid, .rhs = two } });
        const hit = try f.appendInst(entry, u32_t, .{ .iconst = 100 });
        const miss = try f.appendInst(entry, u32_t, .{ .iconst = 200 });
        const pick = try f.appendInst(entry, u32_t, .{ .select = .{ .cond = cond, .then = hit, .@"else" = miss } });
        try f.appendStore(entry, pick, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        // One compare into a mask and one merge under it. Greater-than has no vector-vector
        // encoding, so it is the less-than with the operands swapped and still one instruction.
        try std.testing.expectEqual(@as(usize, 1), countCompares(words));
        try std.testing.expectEqual(@as(usize, 1), countMerges(words));

        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            try std.testing.expectEqual(@as(u32, if (c.taken[i]) 100 else 200), data[i]);
        }
    }
}

test "a warp-uniform condition still merges when the sides are per-lane" {
    // The condition is the same in every lane but the two sides are not, so the answer is per-lane
    // and the merge is still how it is built. The mask comes from comparing the broadcast condition
    // against zero, which is all ones where it is FALSE, so the sides go in the other order.
    const allocator = std.testing.allocator;
    for ([_]i64{ 0, 1 }) |flag| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        const cond = try f.appendInst(entry, bool_t, .{ .iconst = flag });
        const ten = try f.appendArithImm(entry, u32_t, .mul, tid, 10);
        const hundred = try f.appendArithImm(entry, u32_t, .mul, tid, 100);
        const pick = try f.appendInst(entry, u32_t, .{ .select = .{ .cond = cond, .then = ten, .@"else" = hundred } });
        try f.appendStore(entry, pick, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 1), countMerges(words));

        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            const want: u32 = @intCast(i * (if (flag != 0) @as(usize, 10) else 100));
            try std.testing.expectEqual(want, data[i]);
        }
    }
}

test "a per-lane compare takes its signedness from the operand type" {
    // The lane index is small, so signed and unsigned compares agree on it and a test against a
    // small bound cannot tell them apart. These bounds are chosen so they disagree on every lane:
    // 0x80000000 is half the unsigned range and the most negative signed value.
    const allocator = std.testing.allocator;
    const cases = [_]struct { signedness: std.builtin.Signedness, bound: i64, taken: bool }{
        // Unsigned: every lane is below half the range.
        .{ .signedness = .unsigned, .bound = 0x8000_0000, .taken = true },
        // Signed: the same bits are the most negative value, so no lane is below it.
        .{ .signedness = .signed, .bound = -0x8000_0000, .taken = false },
    };
    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const ty = try f.types.intern(.{ .int = .{ .signedness = c.signedness, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        const lane = try f.appendInst(entry, ty, .{ .convert = .{ .value = tid } });
        const bound = try f.appendInst(entry, ty, .{ .iconst = c.bound });
        const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = lane, .rhs = bound } });
        const hit = try f.appendInst(entry, u32_t, .{ .iconst = 7 });
        const miss = try f.appendInst(entry, u32_t, .{ .iconst = 9 });
        const pick = try f.appendInst(entry, u32_t, .{ .select = .{ .cond = cond, .then = hit, .@"else" = miss } });
        try f.appendStore(entry, pick, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            try std.testing.expectEqual(@as(u32, if (c.taken) 7 else 9), data[i]);
        }
    }
}

test "a per-lane float compare answers false where it should" {
    // Checking only the TRUE cases cannot see a compare that always says true, which is exactly
    // how a mis-wired one fails: the peer's lane read the fp32 operands as fp16 in their low bits,
    // both zero there, so every compare returned an all-ones mask and looked plausible. So every
    // relation here is driven with operands that differ, and each one is checked in BOTH
    // directions. `5 < 3` being false is what pins the left operand as the left operand.
    const allocator = std.testing.allocator;
    const cases = [_]struct { op: ir.function.CmpOp, lhs: f64, rhs: f64, taken: bool }{
        .{ .op = .eq, .lhs = 3.0, .rhs = 5.0, .taken = false },
        .{ .op = .eq, .lhs = 3.0, .rhs = 3.0, .taken = true },
        .{ .op = .ne, .lhs = 3.0, .rhs = 5.0, .taken = true },
        .{ .op = .ne, .lhs = 3.0, .rhs = 3.0, .taken = false },
        .{ .op = .lt, .lhs = 3.0, .rhs = 5.0, .taken = true },
        .{ .op = .lt, .lhs = 5.0, .rhs = 3.0, .taken = false },
        .{ .op = .lt, .lhs = 3.0, .rhs = 3.0, .taken = false },
        .{ .op = .le, .lhs = 3.0, .rhs = 5.0, .taken = true },
        .{ .op = .le, .lhs = 5.0, .rhs = 3.0, .taken = false },
        .{ .op = .le, .lhs = 3.0, .rhs = 3.0, .taken = true },
        .{ .op = .gt, .lhs = 5.0, .rhs = 3.0, .taken = true },
        .{ .op = .gt, .lhs = 3.0, .rhs = 5.0, .taken = false },
        .{ .op = .ge, .lhs = 3.0, .rhs = 3.0, .taken = true },
        .{ .op = .ge, .lhs = 3.0, .rhs = 5.0, .taken = false },
    };
    for (cases) |c| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try f.types.intern(.{ .float = .f32 });
        const bool_t = try f.types.intern(.bool);
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
        const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        // Made per-lane by adding a zeroed lane index, so the compare is a vector one rather than
        // something a uniform path could answer.
        const zero = try f.appendArithImm(entry, u32_t, .mul, tid, 0);
        const bias = try f.appendInst(entry, f32_t, .{ .unary = .{ .op = .reinterpret, .value = zero } });
        const kl = try f.appendInst(entry, f32_t, .{ .fconst = c.lhs });
        const kr = try f.appendInst(entry, f32_t, .{ .fconst = c.rhs });
        const lhs = try f.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = kl, .rhs = bias } });
        const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = c.op, .lhs = lhs, .rhs = kr } });
        const hit = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
        const miss = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
        const pick = try f.appendInst(entry, u32_t, .{ .select = .{ .cond = cond, .then = hit, .@"else" = miss } });
        try f.appendStore(entry, pick, at);
        f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            try std.testing.expectEqual(@as(u32, if (c.taken) 1 else 0), data[i]);
        }
    }
}

test "a divergent branch becomes straight-line code with a select" {
    // `out[lane] = (lane < 2) ? lane * 10 : lane + 100`, written as a BRANCH rather than a select.
    // Lanes disagree about which side to take, which used to be refused outright because a scalar
    // branch would have sent the whole warp whichever way lane zero went.
    //
    // Note what the join parameter would look like to an analysis that only follows operands: both
    // incoming values are per-lane here, but they need not be, and that is the case this shape is
    // converted to avoid rather than to tolerate.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    const ten = try f.appendArithImm(then_b, u32_t, .mul, tid, 10);
    const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
    const result = try f.appendBlockParam(join, u32_t);

    try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
    try f.setJump(then_b, join, &.{ten});
    try f.setJump(else_b, join, &.{hundred});

    const offset = try f.appendArithImm(join, u32_t, .shl, tid, 2);
    const at = try f.appendInst(join, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(join, result, at);
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // One compare into a mask and one merge, and NO branch: the arms were spliced in and the join
    // parameter became a select, so nothing jumps anywhere.
    try std.testing.expectEqual(@as(usize, 1), countCompares(words));
    try std.testing.expectEqual(@as(usize, 1), countMerges(words));
    for (words) |w| {
        try std.testing.expect(w & 0x7F != 0b1100011); // no conditional branch
        try std.testing.expect(w & 0x7F != 0b1101111); // no jump
    }

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const want: u32 = if (i < 2) @intCast(i * 10) else @intCast(i + 100);
        try std.testing.expectEqual(want, data[i]);
    }
}

test "a store under a divergent branch becomes a masked scatter" {
    // The span fill: `if (lane < 2) data[lane] = 0xAA`. Both arms run for every lane after
    // if-conversion, so the store carries the condition as a MASK and the lanes that did not
    // choose it make no bus access at all. The words they would have written are the proof.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const then_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    const fill = try f.appendInst(then_b, u32_t, .{ .iconst = 0xAA });
    const offset = try f.appendArithImm(then_b, u32_t, .shl, tid, 2);
    const at = try f.appendInst(then_b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(then_b, fill, at);

    try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = join });
    try f.setJump(then_b, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // One predicated scatter and no branch. `vm` clear is what makes it predicated, so the test
    // reads that bit rather than trusting the opcode.
    var masked_stores: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b0100111) continue;
        try std.testing.expectEqual(@as(u32, 0), (w >> 25) & 1);
        masked_stores += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), masked_stores);
    // The run is contiguous as well as masked, so it is the unit-stride form.
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b00));
    for (words) |w| {
        try std.testing.expect(w & 0x7F != 0b1100011);
        try std.testing.expect(w & 0x7F != 0b1101111);
    }

    const sentinel: u32 = 0x1234_5678;
    var data: [8]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(if (i < 2) @as(u32, 0xAA) else sentinel, data[i]);
    }
}

test "a store in the else arm takes the complement of the condition" {
    // `if (lane < 2) {} else data[lane] = 0xBB`. The else arm runs where the condition is FALSE,
    // so its mask is the inverted compare, and a wrong polarity shows up as the other two words.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });

    const fill = try f.appendInst(else_b, u32_t, .{ .iconst = 0xBB });
    const offset = try f.appendArithImm(else_b, u32_t, .shl, tid, 2);
    const at = try f.appendInst(else_b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(else_b, fill, at);

    try f.appendIf(entry, cond, .{ .target = join }, .{ .target = else_b });
    try f.setJump(else_b, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    // ONE compare. The complement is a real instruction, because a mask cannot be negated where
    // it is used, and the original condition is then dead: the then arm is empty and no join
    // parameter reads it, so dead-code elimination takes it out.
    try std.testing.expectEqual(@as(usize, 1), countCompares(words));

    const sentinel: u32 = 0x1234_5678;
    var data: [8]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(if (i < 2) sentinel else @as(u32, 0xBB), data[i]);
    }
}

test "a store in the else arm of a float compare is refused" {
    // `!(a < b)` is not `a >= b` when either is a NaN, so the complement this needs does not
    // exist for a float compare and the diamond stays a branch.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try f.types.intern(.{ .float = .f32 });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const lane_off = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const lane_at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = lane_off } });
    const lane_f = try f.appendInst(entry, f32_t, .{ .load = .{ .ptr = lane_at } });
    const limit = try f.appendInst(entry, f32_t, .{ .fconst = 2.0 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = lane_f, .rhs = limit } });

    const fill = try f.appendInst(else_b, u32_t, .{ .iconst = 0xCC });
    const offset = try f.appendArithImm(else_b, u32_t, .shl, tid, 2);
    const at = try f.appendInst(else_b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(else_b, fill, at);

    try f.appendIf(entry, cond, .{ .target = join }, .{ .target = else_b });
    try f.setJump(else_b, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a warp-uniform branch keeps its branch instead of running both arms" {
    // Converting this would be correct and slower: every lane agrees, so one jump runs one arm,
    // where a merge would run both and then throw one away.
    const allocator = std.testing.allocator;
    for ([_]i64{ 0, 1 }) |flag| {
        var f = Function.init(allocator);
        defer f.deinit();
        const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const bool_t = try f.types.intern(.bool);
        const ptr_t = try f.types.ptrGlobal();

        const entry = try f.appendBlock();
        const then_b = try f.appendBlock();
        const else_b = try f.appendBlock();
        const join = try f.appendBlock();

        const tid = try f.appendBlockParam(entry, u32_t);
        try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
        const base = try f.appendGlobalAddr(entry, ptr_t, "data");
        const cond = try f.appendInst(entry, bool_t, .{ .iconst = flag });
        const ten = try f.appendArithImm(then_b, u32_t, .mul, tid, 10);
        const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
        const result = try f.appendBlockParam(join, u32_t);

        try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
        try f.setJump(then_b, join, &.{ten});
        try f.setJump(else_b, join, &.{hundred});

        const offset = try f.appendArithImm(join, u32_t, .shl, tid, 2);
        const at = try f.appendInst(join, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
        try f.appendStore(join, result, at);
        f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

        const words = try compile(allocator, &f);
        defer allocator.free(words);
        try std.testing.expectEqual(@as(usize, 0), countMerges(words));
        var branches: usize = 0;
        for (words) |w| {
            if (w & 0x7F == 0b1100011) branches += 1;
        }
        try std.testing.expectEqual(@as(usize, 1), branches);

        var data: [8]u32 = @splat(0xFFFF_FFFF);
        try runOnce(words, &data, 0, 1);
        for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
            const want: u32 = if (flag == 1) @intCast(i * 10) else @intCast(i + 100);
            try std.testing.expectEqual(want, data[i]);
        }
    }
}

test "a divergent branch sharing an arm with another path is refused" {
    // `then` is also the target of a jump the branch does not control, so moving its body into the
    // branch would leave that other path running an empty block. Nothing converts here and the
    // refusal answers instead.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const other = try f.appendBlock();
    const fork = try f.appendBlock();
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const flag = try f.appendInst(entry, bool_t, .{ .iconst = 1 });
    try f.appendIf(entry, flag, .{ .target = other }, .{ .target = fork });
    try f.setJump(other, then_b, &.{});

    const two = try f.appendInst(fork, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(fork, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    try f.appendIf(fork, cond, .{ .target = then_b }, .{ .target = else_b });

    const ten = try f.appendArithImm(then_b, u32_t, .mul, tid, 10);
    const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
    const result = try f.appendBlockParam(join, u32_t);
    try f.setJump(then_b, join, &.{ten});
    try f.setJump(else_b, join, &.{hundred});

    const offset = try f.appendArithImm(join, u32_t, .shl, tid, 2);
    const at = try f.appendInst(join, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(join, result, at);
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a join reached from outside the branch keeps its parameters" {
    // A third path reaches the join, so its parameter takes a value the two arms never pass and
    // cannot become a select between them.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const other = try f.appendBlock();
    const fork = try f.appendBlock();
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const flag = try f.appendInst(entry, bool_t, .{ .iconst = 1 });
    try f.appendIf(entry, flag, .{ .target = other }, .{ .target = fork });

    const seven = try f.appendInst(other, u32_t, .{ .iconst = 7 });
    try f.setJump(other, join, &.{seven});

    const two = try f.appendInst(fork, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(fork, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    try f.appendIf(fork, cond, .{ .target = then_b }, .{ .target = else_b });

    const ten = try f.appendArithImm(then_b, u32_t, .mul, tid, 10);
    const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
    const result = try f.appendBlockParam(join, u32_t);
    try f.setJump(then_b, join, &.{ten});
    try f.setJump(else_b, join, &.{hundred});

    const offset = try f.appendArithImm(join, u32_t, .shl, tid, 2);
    const at = try f.appendInst(join, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(join, result, at);
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a divergent branch that passes an argument to its arm is refused" {
    // The arm reads a parameter the branch edge supplies, which if-conversion has nowhere to put:
    // splicing the body into the branch would leave that parameter with no value. So the shape is
    // one of those the refusal has to catch, and this pins that it does.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    const ten = try f.appendArithImm(entry, u32_t, .mul, tid, 10);

    const carried = try f.appendBlockParam(then_b, u32_t);
    const bumped = try f.appendArithImm(then_b, u32_t, .add, carried, 1);
    const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
    const result = try f.appendBlockParam(join, u32_t);

    try f.appendIf(entry, cond, .{ .target = then_b, .args = &.{ten} }, .{ .target = else_b });
    try f.setJump(then_b, join, &.{bumped});
    try f.setJump(else_b, join, &.{hundred});

    const offset = try f.appendArithImm(join, u32_t, .shl, tid, 2);
    const at = try f.appendInst(join, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(join, result, at);
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a divergent branch whose arms go to different places is refused" {
    // There is no one join to turn into selects: each arm ends somewhere of its own. Nothing here
    // converts, so the refusal is what answers.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const then_end = try f.appendBlock();
    const else_end = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });

    const ten = try f.appendArithImm(then_b, u32_t, .mul, tid, 10);
    const hundred = try f.appendArithImm(else_b, u32_t, .add, tid, 100);
    try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = else_b });
    try f.setJump(then_b, then_end, &.{});
    try f.setJump(else_b, else_end, &.{});

    try f.appendStore(then_end, ten, at);
    f.setTerminator(then_end, .{ .ret = ir.function.Ret.none() });
    try f.appendStore(else_end, hundred, at);
    f.setTerminator(else_end, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a masked store whose address and value are the same in every lane writes nothing when no lane is active" {
    // `if (lane > 100) data[0] = 0xAA`. Nothing about the address or the value is per-lane, so
    // only the mask makes this a vector instruction, and no lane of this warp is in it. The
    // sentinel surviving is the whole assertion: an unmasked scatter would write the word.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const then_b = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const limit = try f.appendInst(entry, u32_t, .{ .iconst = 100 });
    const cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = tid, .rhs = limit } });
    const fill = try f.appendInst(then_b, u32_t, .{ .iconst = 0xAA });
    try f.appendStore(then_b, fill, base);

    try f.appendIf(entry, cond, .{ .target = then_b }, .{ .target = join });
    try f.setJump(then_b, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    const sentinel: u32 = 0x1234_5678;
    var data: [8]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    try std.testing.expectEqual(sentinel, data[0]);
}

test "a store under two conditions runs where both hold" {
    // `if (lane < 3) { if (lane > 0) data[lane] = 0xDD }`. The inner diamond converts first and
    // leaves the store carrying the inner condition, then the outer one ANDs the two masks. Lane
    // 0 fails the inner test and lane 3 the outer one, so both ends of the span are the
    // assertion: dropping either mask writes a word that has to stay as it was.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const outer = try f.appendBlock();
    const inner = try f.appendBlock();
    const inner_join = try f.appendBlock();
    const join = try f.appendBlock();

    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const three = try f.appendInst(entry, u32_t, .{ .iconst = 3 });
    const outer_cond = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = three } });
    try f.appendIf(entry, outer_cond, .{ .target = outer }, .{ .target = join });

    const zero = try f.appendInst(outer, u32_t, .{ .iconst = 0 });
    const inner_cond = try f.appendInst(outer, bool_t, .{ .icmp = .{ .op = .gt, .lhs = tid, .rhs = zero } });
    try f.appendIf(outer, inner_cond, .{ .target = inner }, .{ .target = inner_join });

    const fill = try f.appendInst(inner, u32_t, .{ .iconst = 0xDD });
    const offset = try f.appendArithImm(inner, u32_t, .shl, tid, 2);
    const at = try f.appendInst(inner, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(inner, fill, at);
    try f.setJump(inner, inner_join, &.{});
    try f.setJump(inner_join, join, &.{});
    f.setTerminator(join, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    var masked_stores: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b0100111) continue;
        try std.testing.expectEqual(@as(u32, 0), (w >> 25) & 1);
        masked_stores += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), masked_stores);
    for (words) |w| {
        try std.testing.expect(w & 0x7F != 0b1100011);
        try std.testing.expect(w & 0x7F != 0b1101111);
    }

    // The two masks combine with `vmand.mm`, funct6 011001 in the OPMVV form, and NOT with the
    // element-wise `vand.vv`. Both give the same answer at every width this core is built in,
    // because sixteen lanes of mask fit in the first element, so the structure is the only place
    // the difference shows. Nothing here can tell them apart by value.
    // The one into v0 is the move that applies the finished mask to the store, so it is not
    // counted: the combine is the one that writes an ordinary register.
    var mask_ands: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0b111 != 0b010) continue;
        if (w >> 26 != 0b011001) continue;
        if ((w >> 7) & 0x1F == 0) continue;
        mask_ands += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), mask_ands);

    const sentinel: u32 = 0x1234_5678;
    var data: [8]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        const want: u32 = if (i > 0 and i < 3) 0xDD else sentinel;
        try std.testing.expectEqual(want, data[i]);
    }
}

test "a masked store whose mask is the same in every lane is refused" {
    // Every lane agrees, so the answer is a branch over the store and not a predicate on it.
    // Nothing the compiler writes has this shape, and hand-written IR says it in one line.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const fill = try f.appendInst(entry, u32_t, .{ .iconst = 0xEE });
    const always = try f.appendInst(entry, bool_t, .{ .iconst = 1 });
    try f.appendStoreMasked(entry, fill, base, always);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

test "a mask combined with a warp-uniform bool is refused" {
    // A per-lane bool is mask BITS and a warp-uniform one is a 0 or a 1 in a scalar register.
    // Broadcasting the second gives one value per element, not one bit per lane, so reading it as
    // a mask would make the store write by the wrong rule. Only hand-written IR says this.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const two = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const lanes = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = tid, .rhs = two } });
    const always = try f.appendInst(entry, bool_t, .{ .iconst = 1 });
    const both = try f.appendInst(entry, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = lanes, .rhs = always } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStoreMasked(entry, two, at, both);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(error.Unsupported, compile(allocator, &f));
}

/// How many vector accesses in `code` use the addressing mode `mop`: 00 unit-stride, 01 indexed.
fn countAccesses(code: []const u32, mop: u32) usize {
    var n: usize = 0;
    for (code) |w| {
        const op = w & 0x7F;
        if (op != 0b0000111 and op != 0b0100111) continue;
        if ((w >> 12) & 0b111 != 0b110) continue;
        if ((w >> 26) & 0b11 != mop) continue;
        n += 1;
    }
    return n;
}

test "an index by lane id becomes one contiguous access" {
    // `data[lane] += 7`. The lanes address consecutive words, so the load and the store each take
    // one unit-stride instruction from a scalar start address, and the per-lane address vector is
    // never built: the shift and the add that made it are absorbed.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const old = try f.appendInst(entry, u32_t, .{ .load = .{ .ptr = at } });
    const bumped = try f.appendArithImm(entry, u32_t, .add, old, 7);
    try f.appendStore(entry, bumped, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 2), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b01));
    // The address arithmetic is gone with it. `vsll.vi` is funct6 100101 in the OPIVI form, and
    // the shift by two that scaled the lane index is the only one this kernel could emit.
    for (words) |w| {
        if (w & 0x7F != 0b1010111) continue;
        try std.testing.expect(w >> 26 != 0b100101);
    }

    var data: [8]u32 = @splat(0);
    for (0..8) |i| data[i] = @intCast(i * 10);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i * 10 + 7)), data[i]);
    }
}

test "an index the lowering cannot prove contiguous stays a scatter" {
    // `data[lane ^ 1] = lane`. The lanes swap in pairs, so the words are not a run and the access
    // keeps its address per lane. A miss costs speed and never an answer.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const swapped = try f.appendArithImm(entry, u32_t, .bit_xor, tid, 1);
    const offset = try f.appendArithImm(entry, u32_t, .shl, swapped, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, tid, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b01));

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i ^ 1)), data[i]);
    }
}

test "a lane index scaled by anything but one word stays a scatter" {
    // `data[lane * 2] = lane + 1`, so the byte offset is the lane shifted by THREE. The words are
    // every other one, not a run, and reading the shift amount is the only thing that tells the
    // two apart. Treating this as contiguous writes the first four words instead of every second
    // one, which the values below catch.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 3);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const value = try f.appendArithImm(entry, u32_t, .add, tid, 1);
    try f.appendStore(entry, value, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b01));

    const sentinel: u32 = 0x1234_5678;
    var data: [16]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), data[i * 2]);
        try std.testing.expectEqual(sentinel, data[i * 2 + 1]);
    }
}

test "an address a second reader wants is still built" {
    // `data[lane] = 0x55; table[lane] = &data[lane]`. The address feeds a contiguous store AND
    // goes into memory as a VALUE, so the arithmetic that built it cannot be absorbed: the second
    // store would write a register holding nothing. The addresses in the table are the assertion.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();
    const lanes = glacier.encode.Profile.@"gc1.n".lanes();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const offset = try f.appendArithImm(entry, u32_t, .shl, tid, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const fill = try f.appendInst(entry, u32_t, .{ .iconst = 0x55 });
    try f.appendStore(entry, fill, at);

    // The table sits one warp width along, so its own address is unit-stride from there.
    const table = try f.appendArithImm(entry, ptr_t, .add, base, @intCast(lanes * 4));
    const table_at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = table, .rhs = offset } });
    try f.appendStore(entry, at, table_at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 2), countAccesses(words, 0b00));

    var data: [16]u32 = @splat(0);
    try runOnce(words, &data, 0, 1);
    for (0..lanes) |i| {
        try std.testing.expectEqual(@as(u32, 0x55), data[i]);
        try std.testing.expectEqual(data_base + @as(u32, @intCast(i * 4)), data[lanes + i]);
    }
}

test "a lane term buried under the index scale still becomes one contiguous access" {
    // `w[y * row + x + lane] = colour`, which is how a scanline fill indexes: the whole index is
    // scaled to bytes at once, so the lane term sits INSIDE the shift and the address is
    // `ptr + (uniform + lane) * 4`. Hoisting the lane term out is what lets the access see a run.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const y = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
    const row = try f.appendInst(entry, u32_t, .{ .iconst = 4 });
    const x = try f.appendInst(entry, u32_t, .{ .iconst = 1 });
    const colour = try f.appendInst(entry, u32_t, .{ .iconst = 0x7E });
    const scanline = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .mul, .lhs = y, .rhs = row } });
    const column = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .add, .lhs = scanline, .rhs = x } });
    const index = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .add, .lhs = column, .rhs = tid } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, index, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, colour, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b01));

    const sentinel: u32 = 0x1234_5678;
    var data: [16]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    const first = 1 * 4 + 1;
    for (0..data.len) |i| {
        const inside = i >= first and i < first + glacier.encode.Profile.@"gc1.n".lanes();
        try std.testing.expectEqual(if (inside) @as(u32, 0x7E) else sentinel, data[i]);
    }
}

test "a buried lane term that steps two words stays a scatter" {
    // `w[uniform + lane * 2]`, so the scale under the shift makes the run step eight bytes. The
    // stride has to be carried through the shift to see that, and getting it wrong writes a
    // contiguous block over words the kernel never named.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const start = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const doubled = try f.appendArithImm(entry, u32_t, .mul, tid, 2);
    const index = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .add, .lhs = start, .rhs = doubled } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, index, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const value = try f.appendArithImm(entry, u32_t, .add, tid, 1);
    try f.appendStore(entry, value, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b01));

    const sentinel: u32 = 0x1234_5678;
    var data: [16]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..glacier.encode.Profile.@"gc1.n".lanes()) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), data[2 + i * 2]);
        try std.testing.expectEqual(sentinel, data[2 + i * 2 + 1]);
    }
}

test "a constant offset inside the index reaches the start of the run" {
    // `w[lane + 3] = lane`. The constant is scaled with the index, so it is twelve bytes and not
    // three, and it belongs to the scalar start of the run rather than to any lane.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const index = try f.appendArithImm(entry, u32_t, .add, tid, 3);
    const offset = try f.appendArithImm(entry, u32_t, .shl, index, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    try f.appendStore(entry, tid, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 1), countAccesses(words, 0b00));

    const sentinel: u32 = 0x1234_5678;
    var data: [16]u32 = @splat(sentinel);
    try runOnce(words, &data, 0, 1);
    for (0..data.len) |i| {
        const lane = @as(i64, @intCast(i)) - 3;
        const inside = lane >= 0 and lane < glacier.encode.Profile.@"gc1.n".lanes();
        try std.testing.expectEqual(if (inside) @as(u32, @intCast(lane)) else sentinel, data[i]);
    }
}

test "a buried address read and written takes the contiguous form both ways" {
    // `w[y * row + lane] += 1`. The load and the store share one address, so it is rewritten once
    // and both accesses read the run from the same scalar start. The reuse itself is an
    // instruction-count matter and nothing here can see it; what this pins is that BOTH halves of
    // a read-modify-write take the contiguous form and land on the same words.
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try f.types.ptrGlobal();

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const base = try f.appendGlobalAddr(entry, ptr_t, "data");
    const y = try f.appendInst(entry, u32_t, .{ .iconst = 2 });
    const row = try f.appendInst(entry, u32_t, .{ .iconst = 4 });
    const scanline = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .mul, .lhs = y, .rhs = row } });
    const index = try f.appendInst(entry, u32_t, .{ .arith = .{ .op = .add, .lhs = scanline, .rhs = tid } });
    const offset = try f.appendArithImm(entry, u32_t, .shl, index, 2);
    const at = try f.appendInst(entry, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
    const old = try f.appendInst(entry, u32_t, .{ .load = .{ .ptr = at } });
    const bumped = try f.appendArithImm(entry, u32_t, .add, old, 1);
    try f.appendStore(entry, bumped, at);
    f.setTerminator(entry, .{ .ret = ir.function.Ret.none() });

    const words = try compile(allocator, &f);
    defer allocator.free(words);
    try std.testing.expectEqual(@as(usize, 2), countAccesses(words, 0b00));
    try std.testing.expectEqual(@as(usize, 0), countAccesses(words, 0b01));

    var data: [16]u32 = @splat(0);
    for (0..data.len) |i| data[i] = @intCast(i * 10);
    try runOnce(words, &data, 0, 1);
    for (0..data.len) |i| {
        const lane = @as(i64, @intCast(i)) - 8;
        const inside = lane >= 0 and lane < glacier.encode.Profile.@"gc1.n".lanes();
        const want: u32 = @intCast(i * 10 + @as(usize, if (inside) 1 else 0));
        try std.testing.expectEqual(want, data[i]);
    }
}
