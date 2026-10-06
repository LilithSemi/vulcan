//! The Mandelbrot kernel, assembled with Vulcan's encoders and diffed against Glacier's own.
//!
//! This is the oracle the Glacier backend is built against. The program is a real multi-warp
//! kernel that renders the set into a framebuffer and retires with EBREAK, and it runs on an ECP5
//! FPGA today, so matching it word for word is a statement about silicon rather than about two
//! encoders agreeing with each other.
//!
//! The expected words come from `glacierMandelbrotKernel` in the Glacier repo
//! (packages/glacier_hdl/lib/src/boot/mandelbrot_kernel.dart), dumped for a 64x48 frame at
//! fbBase 0 with the default 16 iterations and `GlacierMandelbrotView.fit(64, 48)`. That
//! function is the ground truth: if the two disagree, this file is wrong.
//!
//! WHAT THIS PROVES is that a Glacier core's instruction words ARE RISC-V words and the existing
//! `riscv64/encode.zig` emits them correctly, so the Glacier backend needs a profile and an
//! instruction selector rather than an encoder of its own. The one thing Glacier adds is `ebreak`.
//!
//! WHAT THIS IS NOT is a test of code generation. The program below is hand-assembled, exactly as
//! the Dart is. `lower.zig` compiles the same image from Vulcan IR and checks it against this one.
//! It compares the FRAME the two render, not the words: a compiler picks other registers and
//! another instruction order, so the two word streams differ by design.

const std = @import("std");
const glacier = @import("vulcan-target").glacier.encode;

const rv = glacier.rv;
const Reg = glacier.Reg;

/// Fixed-point fractional bits: a value is `real * 2^fraction_bits`.
const fraction_bits: u5 = 12;

/// The complex-plane rectangle a render covers, in fixed point. The kernel walks the view by
/// adding `dx` and `dy` to a register, so the pixel grid IS these integers.
pub const View = struct {
    x_min: i32,
    y_min: i32,
    dx: i32,
    dy: i32,

    /// The whole set in a `width` x `height` frame with square pixels and the real axis exactly on
    /// the middle row. Deriving `y_min` as a whole number of `dy` steps is what makes the image
    /// mirror about its middle row, which a render can then be checked against.
    pub fn fit(width: i32, height: i32) View {
        const span_x = toFixed(3.0);
        const centre_x = toFixed(-0.6);
        const dx = @divTrunc(span_x, width);
        return .{
            .x_min = centre_x - @divTrunc(width, 2) * dx,
            .y_min = -@divTrunc(height, 2) * dx,
            .dx = dx,
            .dy = dx,
        };
    }

    fn toFixed(value: f64) i32 {
        return @intFromFloat(@round(value * (1 << fraction_bits)));
    }
};

/// How far to shift the surviving iteration count to make a colour: the largest shift that keeps
/// the brightest pixel inside one byte lane, so a shade cannot carry into the next channel.
pub fn colourShift(iterations: u32) u5 {
    var shift: u5 = 0;
    while ((iterations << (shift + 1)) <= 0xFF) shift += 1;
    return shift;
}

/// Assembles the renderer into `out`, returning the words written.
///
/// The image is warp-count agnostic. GC1 issues its resident warps round-robin, so one warp can
/// only use one slot in that rotation. Rather than build a program per warp, every warp reads
/// `mhartid` and `hart_count` and takes every Nth row, so the SAME image fills whatever core it is
/// launched on and the host launches every warp at one entry point.
///
/// Rows are interleaved rather than split into bands because the expensive pixels are the ones
/// inside the set, and they sit in a band across the middle of the frame. Bands would give one
/// warp all of them.
pub fn kernel(
    out: []u32,
    fb_base: i32,
    width: i32,
    height: i32,
    iterations: u16,
    view: View,
) []const u32 {
    var n: usize = 0;
    var pair: [2]u32 = undefined;
    const put = struct {
        fn one(buf: []u32, at: *usize, w: u32) void {
            buf[at.*] = w;
            at.* += 1;
        }
        fn many(buf: []u32, at: *usize, ws: []const u32) void {
            for (ws) |w| one(buf, at, w);
        }
    };
    // Byte address of an instruction, which is what a branch offset is measured from.
    const here = struct {
        fn at(i: usize) i32 {
            return @intCast(i * 4);
        }
    };

    const row_stride = width * 4;
    // |z|^2 >= 4 is the escape test, and 4.0 in fixed point needs LUI when the fraction is 12
    // bits, so it is loaded as an upper immediate rather than an ADDI.
    const escape: i32 = 4 << fraction_bits;
    const shade_shift = colourShift(@as(u32, iterations));

    // The prologue: each warp asks the hardware who it is, then works out its own share.
    put.one(out, &n, glacier.csrr(.x21, glacier.Csr.mhartid));
    put.one(out, &n, glacier.csrr(.x22, glacier.Csr.hart_count));
    // A row's byte pitch lives in a register, not an immediate: a wide frame does not fit the
    // 12-bit ADDI field, and the start-up multiplies below take it as an operand.
    put.many(out, &n, glacier.loadImmediate(.x26, row_stride, &pair));
    put.one(out, &n, rv.addi(.x18, .x0, @intCast(width)));
    put.one(out, &n, rv.addi(.x19, .x0, @intCast(height)));
    put.one(out, &n, rv.lui(.x17, @intCast(escape >> 12)));
    put.many(out, &n, glacier.loadImmediate(.x20, view.x_min, &pair));
    put.many(out, &n, glacier.loadImmediate(.x1, fb_base, &pair));
    put.many(out, &n, glacier.loadImmediate(.x5, view.y_min, &pair));
    put.one(out, &n, rv.addi(.x3, .x21, 0)); // first row = my warp id
    put.one(out, &n, rv.addi(.x12, .x0, @intCast(view.dy)));

    // Skip ahead to my first row, then derive the per-row strides: x23 is (N-1)*row_stride, which
    // skips the warps behind me once the column loop has walked my own row, and x24 is N*dy.
    put.one(out, &n, rv.mul(.x16, .x21, .x26));
    put.one(out, &n, rv.add(.x1, .x1, .x16));
    put.one(out, &n, rv.mul(.x16, .x21, .x12));
    put.one(out, &n, rv.add(.x5, .x5, .x16));
    put.one(out, &n, rv.addi(.x23, .x22, -1));
    put.one(out, &n, rv.mul(.x23, .x23, .x26));
    put.one(out, &n, rv.mul(.x24, .x22, .x12));

    const row_label = here.at(n);
    put.one(out, &n, rv.addi(.x4, .x20, 0)); // cx = x_min
    put.one(out, &n, rv.addi(.x2, .x0, 0)); // column = 0

    const col_label = here.at(n);
    put.one(out, &n, rv.addi(.x6, .x0, 0)); // zx = 0
    put.one(out, &n, rv.addi(.x7, .x0, 0)); // zy = 0
    put.one(out, &n, rv.addi(.x8, .x0, @intCast(iterations)));

    const iter_label = here.at(n);
    // A square is never negative, so the low word of the product is an unsigned value and a
    // logical shift scales it back down.
    put.one(out, &n, rv.mul(.x9, .x6, .x6));
    put.one(out, &n, rv.srli(.x9, .x9, fraction_bits)); // zx^2
    put.one(out, &n, rv.mul(.x10, .x7, .x7));
    put.one(out, &n, rv.srli(.x10, .x10, fraction_bits)); // zy^2
    put.one(out, &n, rv.add(.x11, .x9, .x10));
    const escape_branch = n;
    put.one(out, &n, 0); // patched below: bge x11, x17, done

    // zy = 2*zx*zy + cy. The product keeps 24 fraction bits and the shift takes it back to 12. It
    // must truncate toward ZERO, not toward minus infinity, so a negative product is shifted as a
    // magnitude and the sign is put back afterwards.
    put.one(out, &n, rv.mul(.x11, .x6, .x7));
    put.one(out, &n, rv.srai(.x16, .x11, 31)); // -1 when the product is negative
    put.one(out, &n, rv.xor_(.x11, .x11, .x16));
    put.one(out, &n, rv.sub(.x11, .x11, .x16));
    put.one(out, &n, rv.srli(.x11, .x11, fraction_bits));
    put.one(out, &n, rv.xor_(.x11, .x11, .x16));
    put.one(out, &n, rv.sub(.x11, .x11, .x16));
    put.one(out, &n, rv.slli(.x11, .x11, 1));
    put.one(out, &n, rv.add(.x7, .x11, .x5));
    // zx = zx^2 - zy^2 + cx
    put.one(out, &n, rv.sub(.x6, .x9, .x10));
    put.one(out, &n, rv.add(.x6, .x6, .x4));
    put.one(out, &n, rv.addi(.x8, .x8, -1));
    const iter_branch = n;
    put.one(out, &n, 0); // patched below: bne x8, x0, iter

    const done_label = here.at(n);
    // Colour from the iterations left: a point that survived every iteration is inside the set and
    // stays black, and the sooner one escaped the brighter it is.
    put.one(out, &n, rv.slli(.x11, .x8, shade_shift));
    put.one(out, &n, rv.slli(.x16, .x11, 8));
    put.one(out, &n, rv.or_(.x11, .x11, .x16)); // the same shade in blue and green
    put.one(out, &n, rv.sw(.x11, .x1, 0));
    put.one(out, &n, rv.addi(.x1, .x1, 4)); // next pixel
    // cx += dx. The step fits an ADDI immediate, so walking the view needs no multiply.
    put.one(out, &n, rv.addi(.x4, .x4, @intCast(view.dx)));
    put.one(out, &n, rv.addi(.x2, .x2, 1)); // column++
    const col_branch = n;
    put.one(out, &n, 0); // patched below: bne x2, x18, col

    put.one(out, &n, rv.add(.x1, .x1, .x23)); // skip the rows the other warps own
    put.one(out, &n, rv.add(.x5, .x5, .x24)); // cy += N*dy
    put.one(out, &n, rv.add(.x3, .x3, .x22)); // row += N
    const row_branch = n;
    put.one(out, &n, 0); // patched below: blt x3, x19, row
    put.one(out, &n, glacier.ebreak);

    // Patch the branches now that every label is known. A branch offset is measured from the
    // branch's own address, so each is the target minus where the branch sits.
    out[escape_branch] = rv.bge(.x11, .x17, @intCast(done_label - here.at(escape_branch)));
    out[iter_branch] = rv.bne(.x8, .x0, @intCast(iter_label - here.at(iter_branch)));
    out[col_branch] = rv.bne(.x2, .x18, @intCast(col_label - here.at(col_branch)));
    out[row_branch] = rv.blt(.x3, .x19, @intCast(row_label - here.at(row_branch)));
    return out[0..n];
}

/// `glacierMandelbrotKernel(fbBase: 0, width: 64, height: 48, view: fit(64, 48))`, dumped from the
/// Glacier repo. Ground truth, not a transcription of what Vulcan happens to emit.
const oracle = [_]u32{
    0xf1402af3,
    0xfc002b73,
    0x10000d13,
    0x04000913,
    0x03000993,
    0x000048b7,
    0xffffea37,
    0xe66a0a13,
    0x00000093,
    0xfffff2b7,
    0xe0028293,
    0x000a8193,
    0x0c000613,
    0x03aa8833,
    0x010080b3,
    0x02ca8833,
    0x010282b3,
    0xfffb0b93,
    0x03ab8bb3,
    0x02cb0c33,
    0x000a0213,
    0x00000113,
    0x00000313,
    0x00000393,
    0x01000413,
    0x026304b3,
    0x00c4d493,
    0x02738533,
    0x00c55513,
    0x00a485b3,
    0x0315dc63,
    0x027305b3,
    0x41f5d813,
    0x0105c5b3,
    0x410585b3,
    0x00c5d593,
    0x0105c5b3,
    0x410585b3,
    0x00159593,
    0x005583b3,
    0x40a48333,
    0x00430333,
    0xfff40413,
    0xfa041ce3,
    0x00341593,
    0x00859813,
    0x0105e5b3,
    0x00b0a023,
    0x00408093,
    0x0c020213,
    0x00110113,
    0xf92116e3,
    0x017080b3,
    0x018282b3,
    0x016181b3,
    0xf731cae3,
    0x00100073,
};

test "the Mandelbrot kernel assembles to Glacier's own word stream" {
    var buf: [128]u32 = undefined;
    const got = kernel(&buf, 0, 64, 48, 16, View.fit(64, 48));
    try std.testing.expectEqualSlices(u32, &oracle, got);
}

test "the kernel ends by retiring the warp and fits the instruction aperture" {
    var buf: [128]u32 = undefined;
    const got = kernel(&buf, 0, 64, 48, 16, View.fit(64, 48));
    // A warp has no caller to return to, so the program ends with EBREAK.
    try std.testing.expectEqual(glacier.ebreak, got[got.len - 1]);
    // Every placeholder was patched. A surviving zero would be a valid-looking word that decodes
    // to nothing useful, which is exactly the bug the patching exists to avoid.
    for (got) |w| try std.testing.expect(w != 0);
    // The instruction aperture a host writes in one command is 8192 words.
    try std.testing.expect(got.len <= 8192);
}

test "a core profile fixes the vector length, which is the warp width" {
    // There is no vsetvl: the lane count is a property of the core a kernel is built against, not
    // a register it sets. These are the GC1 profiles.
    try std.testing.expectEqual(@as(u8, 4), glacier.Profile.@"gc1.n".lanes());
    try std.testing.expectEqual(@as(u8, 4), glacier.Profile.@"gc1.n".warps());
    try std.testing.expectEqual(@as(u8, 4), glacier.Profile.@"gc1.mi".lanes());
    try std.testing.expectEqual(@as(u8, 8), glacier.Profile.@"gc1.mi".warps());
    try std.testing.expectEqual(@as(u8, 8), glacier.Profile.@"gc1.s".lanes());
    try std.testing.expectEqual(@as(u8, 16), glacier.Profile.@"gc1.f".lanes());
    try std.testing.expectEqual(@as(u8, 16), glacier.Profile.@"gc1.ma".warps());
}
