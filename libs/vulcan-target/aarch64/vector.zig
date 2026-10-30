//! The rest of Advanced SIMD: shifts, the wider set of integer and floating point lane operations,
//! permutes, reductions, the by-element forms, and the widening and narrowing ones.
//!
//! Everything here is lane arithmetic done on the registers in `Cpu.v`, a lane at a time and
//! unrolled, so a lane's width is a type and the operation is the same scalar operation the
//! rest of the lifter uses. A result is formed completely before any of it is stored, because the
//! destination may be a source. A 64-bit arrangement writes zero to the top half of its
//! destination, as the architecture does.

const std = @import("std");
const ir = @import("vulcan-ir");
const Decode = @import("decode.zig");
const simd = @import("simd.zig");

const Context = simd.Context;
const Value = ir.function.Value;
const Type = ir.types.Type;
const Error = Decode.Error;

fn bits(word: u32, high: u5, low: u5) u32 {
    return (word >> low) & ((@as(u32, 2) << (high - low)) - 1);
}

pub const ShiftOp = enum { sshr, ushr, ssra, usra, shl, sli, sri };
pub const UnaryOp = enum { rev64, rev32, rev16, clz, cnt, not, rbit, abs, neg, cmgt0, cmeq0, cmlt0, cmge0, cmle0 };
pub const FloatUnary = enum { abs, neg, sqrt, scvtf, ucvtf, fcvtzs, fcvtzu, cmeq0, cmge0, cmgt0, cmle0, cmlt0 };
pub const IntOp = enum { smax, umax, smin, umin, sabd, uabd, saba, uaba, mla, mls, addp, smaxp, umaxp, sminp, uminp, sshl, ushl };
pub const FloatOp = enum { add, sub, mul, div, max, min, maxnm, minnm, abd, mla, mls, cmeq, cmge, cmgt, addp, maxp, minp };
pub const Permute = enum { uzp1, trn1, zip1, uzp2, trn2, zip2 };
pub const Across = enum { addv, saddlv, uaddlv, smaxv, umaxv, sminv, uminv };
pub const ElementOp = enum { mul, mla, mls, fmul, fmla, fmls };
pub const LongOp = enum { saddl, uaddl, ssubl, usubl, sabdl, uabdl, smull, umull, smlal, umlal, smlsl, umlsl };

pub const Op = union(enum) {
    /// Shifts by an immediate, in place of the lane.
    shift: struct { op: ShiftOp, q: bool, size: u2, amount: u7, rd: u5, rn: u5 },
    /// `SSHLL` and `USHLL`, of which `SXTL` and `UXTL` are the shift of zero.
    shift_long: struct { signed: bool, upper: bool, size: u2, amount: u6, rd: u5, rn: u5 },
    /// `XTN` and `XTN2`: each lane narrowed to half its width.
    narrow: struct { upper: bool, size: u2, rd: u5, rn: u5 },
    unary: struct { op: UnaryOp, q: bool, size: u2, rd: u5, rn: u5 },
    float_unary: struct { op: FloatUnary, q: bool, double: bool, rd: u5, rn: u5 },
    int_op: struct { op: IntOp, q: bool, size: u2, rd: u5, rn: u5, rm: u5 },
    float_op: struct { op: FloatOp, q: bool, double: bool, rd: u5, rn: u5, rm: u5 },
    permute: struct { op: Permute, q: bool, size: u2, rd: u5, rn: u5, rm: u5 },
    /// `EXT`: a window of sixteen (or eight) bytes taken from the pair of registers.
    extract: struct { q: bool, index: u4, rd: u5, rn: u5, rm: u5 },
    across: struct { op: Across, q: bool, size: u2, rd: u5, rn: u5 },
    /// `DUP Vd, Vn.T[index]` into a scalar: `MOV Dd, Vn.D[1]`.
    dup_scalar: struct { size: u2, index: u4, rd: u5, rn: u5 },
    /// `ADDP Dd, Vn.2D` and `FADDP`.
    pairwise_scalar: struct { float: bool, double: bool, rd: u5, rn: u5 },
    element: struct { op: ElementOp, q: bool, size: u2, index: u4, rd: u5, rn: u5, rm: u5 },
    long: struct { op: LongOp, upper: bool, size: u2, rd: u5, rn: u5, rm: u5 },
    /// `FCVTL`/`FCVTN`, between half and single, and single and double.
    float_widen: struct { upper: bool, double: bool, rd: u5, rn: u5 },
    float_narrow: struct { upper: bool, double: bool, rd: u5, rn: u5 },
};

pub fn parse(word: u32) Error!Op {
    const q = (word & 0x4000_0000) != 0;
    const u = (word & 0x2000_0000) != 0;
    const size: u2 = @truncate(word >> 22);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    const rm: u5 = @truncate(word >> 16);

    // Shift by immediate: `0 Q U 011110 immh immb opcode 1 Rn Rd`.
    if ((word & 0x9f80_0400) == 0x0f00_0400) {
        const immh = bits(word, 22, 19);
        if (immh == 0) return error.UnsupportedInstruction;
        const lane: u2 = @intCast(31 - @clz(immh) - 0);
        const esize: u7 = @as(u7, 8) << lane;
        const immhb: u7 = @intCast(bits(word, 22, 16));
        const opcode = bits(word, 15, 11);
        if (lane == 3 and !q and opcode != 0b10100) return error.UnsupportedInstruction;
        switch (opcode) {
            0b00000, 0b00010 => {
                const amount: u7 = @intCast(@as(u8, 2) * esize - immhb);
                const op: ShiftOp = if (opcode == 0) (if (u) .ushr else .sshr) else (if (u) .usra else .ssra);
                return .{ .shift = .{ .op = op, .q = q, .size = lane, .amount = amount, .rd = rd, .rn = rn } };
            },
            0b01010 => {
                const amount = immhb - esize;
                return .{ .shift = .{ .op = if (u) .sli else .shl, .q = q, .size = lane, .amount = amount, .rd = rd, .rn = rn } };
            },
            0b01000 => {
                if (!u) return error.UnsupportedInstruction;
                return .{ .shift = .{ .op = .sri, .q = q, .size = lane, .amount = @intCast(@as(u8, 2) * esize - immhb), .rd = rd, .rn = rn } };
            },
            0b10100 => {
                if (lane == 3) return error.UnsupportedInstruction;
                return .{ .shift_long = .{ .signed = !u, .upper = q, .size = lane, .amount = @intCast(immhb - esize), .rd = rd, .rn = rn } };
            },
            else => return error.UnsupportedInstruction,
        }
    }

    // Two-register miscellaneous: `0 Q U 01110 size 10000 opcode 10 Rn Rd`.
    if ((word & 0x9f3e_0c00) == 0x0e20_0800) {
        const opcode = bits(word, 16, 12);
        const float = size >= 2 or opcode >= 0b11000;
        if (!float or opcode < 0b10000) {
            const op: UnaryOp = switch ((@as(u32, @intFromBool(u)) << 7) | opcode) {
                0b000_00000 => .rev64,
                0b100_00000 => .rev32,
                0b000_00001 => .rev16,
                0b100_00100 => .clz,
                0b000_00101 => .cnt,
                0b100_00101 => if (size == 0) .not else if (size == 1) .rbit else return error.UnsupportedInstruction,
                0b000_01011 => .abs,
                0b100_01011 => .neg,
                0b000_01000 => .cmgt0,
                0b000_01001 => .cmeq0,
                0b000_01010 => .cmlt0,
                0b100_01000 => .cmge0,
                0b100_01001 => .cmle0,
                else => return narrowOrFloat(word),
            };
            const bytes_only = op == .cnt or op == .not or op == .rbit;
            if (bytes_only and size > 1) return error.UnsupportedInstruction;
            if (bytes_only and op == .cnt and size != 0) return error.UnsupportedInstruction;
            if (size == 3 and (!q or op == .clz or op == .rev32 or op == .rev16 or op == .rev64)) return error.UnsupportedInstruction;
            if ((op == .rev16 and size != 0) or (op == .rev32 and size > 1) or (op == .rev64 and size > 2)) return error.UnsupportedInstruction;
            if (op == .clz and size == 3) return error.UnsupportedInstruction;
            return .{ .unary = .{ .op = op, .q = q, .size = size, .rd = rd, .rn = rn } };
        }
        return narrowOrFloat(word);
    }

    // Three same, the integer operations the first set left out, and the floating point ones.
    if ((word & 0x9f20_0400) == 0x0e20_0400) return threeSameMore(word);

    // Three different: `0 Q U 01110 size 1 Rm opcode 00 Rn Rd`.
    if ((word & 0x9f20_0c00) == 0x0e20_0000) {
        const opcode = bits(word, 15, 12);
        const op: LongOp = switch ((@as(u32, @intFromBool(u)) << 4) | opcode) {
            0b0_0000 => .saddl,
            0b1_0000 => .uaddl,
            0b0_0010 => .ssubl,
            0b1_0010 => .usubl,
            0b0_0111 => .sabdl,
            0b1_0111 => .uabdl,
            0b0_1100 => .smull,
            0b1_1100 => .umull,
            0b0_1000 => .smlal,
            0b1_1000 => .umlal,
            0b0_1010 => .smlsl,
            0b1_1010 => .umlsl,
            else => return error.UnsupportedInstruction,
        };
        if (size == 3) return error.UnsupportedInstruction;
        return .{ .long = .{ .op = op, .upper = q, .size = size, .rd = rd, .rn = rn, .rm = rm } };
    }

    // Permute: `0 Q 001110 size 0 Rm 0 opcode 10 Rn Rd`.
    if ((word & 0xbf20_8c00) == 0x0e00_0800) {
        const op: Permute = switch (bits(word, 14, 12)) {
            1 => .uzp1,
            2 => .trn1,
            3 => .zip1,
            5 => .uzp2,
            6 => .trn2,
            7 => .zip2,
            else => return error.UnsupportedInstruction,
        };
        if (size == 3 and !q) return error.UnsupportedInstruction;
        return .{ .permute = .{ .op = op, .q = q, .size = size, .rd = rd, .rn = rn, .rm = rm } };
    }

    // `EXT`: `0 Q 101110 00 0 Rm 0 imm4 0 Rn Rd`.
    if ((word & 0xbfe0_8400) == 0x2e00_0000) {
        const index: u4 = @intCast(bits(word, 14, 11));
        if (!q and index >= 8) return error.UnsupportedInstruction;
        return .{ .extract = .{ .q = q, .index = index, .rd = rd, .rn = rn, .rm = rm } };
    }

    // Across lanes: `0 Q U 01110 size 11000 opcode 10 Rn Rd`.
    if ((word & 0x9f3e_0c00) == 0x0e30_0800) {
        const opcode = bits(word, 16, 12);
        const op: Across = switch ((@as(u32, @intFromBool(u)) << 7) | opcode) {
            0b000_11011 => .addv,
            0b000_00011 => .saddlv,
            0b100_00011 => .uaddlv,
            0b000_01010 => .smaxv,
            0b100_01010 => .umaxv,
            0b000_11010 => .sminv,
            0b100_11010 => .uminv,
            else => return error.UnsupportedInstruction,
        };
        if (size == 3 or (size == 2 and !q)) return error.UnsupportedInstruction;
        return .{ .across = .{ .op = op, .q = q, .size = size, .rd = rd, .rn = rn } };
    }

    // `DUP` (scalar, element): `01 0 11110000 imm5 0 0000 1 Rn Rd`.
    if ((word & 0xffe0_fc00) == 0x5e00_0400) {
        const imm5 = bits(word, 20, 16);
        const lane: u2 = if (imm5 & 1 != 0) 0 else if (imm5 & 2 != 0) 1 else if (imm5 & 4 != 0) 2 else if (imm5 & 8 != 0) 3 else return error.UnsupportedInstruction;
        return .{ .dup_scalar = .{ .size = lane, .index = @intCast(imm5 >> (@as(u5, lane) + 1)), .rd = rd, .rn = rn } };
    }

    // Scalar pairwise: `01 U 11110 size 11000 opcode 10 Rn Rd`.
    if ((word & 0xdf3e_0c00) == 0x5e30_0800) {
        const opcode = bits(word, 16, 12);
        if (!u and opcode == 0b11011 and size == 3) return .{ .pairwise_scalar = .{ .float = false, .double = true, .rd = rd, .rn = rn } };
        if (u and opcode == 0b01101) return .{ .pairwise_scalar = .{ .float = true, .double = (size & 1) != 0, .rd = rd, .rn = rn } };
        return error.UnsupportedInstruction;
    }

    // By element: `0 Q U 01111 size L M Rm opcode H 0 Rn Rd`.
    if ((word & 0x9f00_0400) == 0x0f00_0000) {
        const opcode = bits(word, 15, 12);
        const l = bits(word, 21, 21);
        const m = bits(word, 20, 20);
        const h = bits(word, 11, 11);
        const float = opcode == 0b1001 and !u or opcode == 0b0001 and !u or opcode == 0b0101 and !u;
        if (float) {
            const double = (size & 1) != 0;
            // A double has no `L`, and its register number keeps all of `M`.
            if (double and l != 0) return error.UnsupportedInstruction;
            const index: u4 = if (double) @intCast(h) else @intCast((h << 1) | l);
            const reg: u5 = @intCast((m << 4) | bits(word, 19, 16));
            if (double and !q) return error.UnsupportedInstruction;
            const op: ElementOp = switch (opcode) {
                0b1001 => .fmul,
                0b0001 => .fmla,
                else => .fmls,
            };
            return .{ .element = .{ .op = op, .q = q, .size = if (double) 3 else 2, .index = index, .rd = rd, .rn = rn, .rm = reg } };
        }
        const op: ElementOp = switch ((@as(u32, @intFromBool(u)) << 4) | opcode) {
            0b0_1000 => .mul,
            0b1_0000 => .mla,
            0b1_0100 => .mls,
            else => return error.UnsupportedInstruction,
        };
        return switch (size) {
            1 => .{ .element = .{ .op = op, .q = q, .size = 1, .index = @intCast((h << 2) | (l << 1) | m), .rd = rd, .rn = rn, .rm = @intCast(bits(word, 19, 16)) } },
            2 => .{ .element = .{ .op = op, .q = q, .size = 2, .index = @intCast((h << 1) | l), .rd = rd, .rn = rn, .rm = @intCast((m << 4) | bits(word, 19, 16)) } },
            else => error.UnsupportedInstruction,
        };
    }

    return error.UnsupportedInstruction;
}

/// The two-register-miscellaneous operations that are narrowing, widening or floating point.
fn narrowOrFloat(word: u32) Error!Op {
    const q = (word & 0x4000_0000) != 0;
    const u = (word & 0x2000_0000) != 0;
    const size: u2 = @truncate(word >> 22);
    const opcode = bits(word, 16, 12);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    const double = (size & 1) != 0;
    const high = (size >> 1) != 0;
    if (!u and opcode == 0b10010) {
        if (size == 3) return error.UnsupportedInstruction;
        return .{ .narrow = .{ .upper = q, .size = size, .rd = rd, .rn = rn } };
    }
    if (!u and opcode == 0b10110 and !high) return .{ .float_narrow = .{ .upper = q, .double = double, .rd = rd, .rn = rn } };
    if (!u and opcode == 0b10111 and !high) return .{ .float_widen = .{ .upper = q, .double = double, .rd = rd, .rn = rn } };
    if (double and !q) return error.UnsupportedInstruction;
    const op: FloatUnary = switch ((@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(high)) << 5) | opcode) {
        0b0_1_01111 => .abs,
        0b1_1_01111 => .neg,
        0b1_1_11111 => .sqrt,
        0b0_0_11101 => .scvtf,
        0b1_0_11101 => .ucvtf,
        0b0_1_11011 => .fcvtzs,
        0b1_1_11011 => .fcvtzu,
        0b0_1_01101 => .cmeq0,
        0b1_1_01100 => .cmge0,
        0b0_1_01100 => .cmgt0,
        0b1_1_01101 => .cmle0,
        0b0_1_01110 => .cmlt0,
        else => return error.UnsupportedInstruction,
    };
    return .{ .float_unary = .{ .op = op, .q = q, .double = double, .rd = rd, .rn = rn } };
}

fn threeSameMore(word: u32) Error!Op {
    const q = (word & 0x4000_0000) != 0;
    const u = (word & 0x2000_0000) != 0;
    const size: u2 = @truncate(word >> 22);
    const opcode = bits(word, 15, 11);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    const rm: u5 = @truncate(word >> 16);
    if (opcode >= 0b11000) {
        const double = (size & 1) != 0;
        const a = (size >> 1) != 0;
        if (double and !q) return error.UnsupportedInstruction;
        const op: FloatOp = switch ((@as(u32, @intFromBool(u)) << 6) | (@as(u32, @intFromBool(a)) << 5) | opcode) {
            0b0_0_11010 => .add,
            0b0_1_11010 => .sub,
            0b1_1_11010 => .abd,
            0b1_0_11010 => .addp,
            0b1_0_11011 => .mul,
            0b1_0_11111 => .div,
            0b0_0_11110 => .max,
            0b0_1_11110 => .min,
            0b1_0_11110 => .maxp,
            0b1_1_11110 => .minp,
            0b0_0_11000 => .maxnm,
            0b0_1_11000 => .minnm,
            0b0_0_11001 => .mla,
            0b0_1_11001 => .mls,
            0b0_0_11100 => .cmeq,
            0b1_0_11100 => .cmge,
            0b1_1_11100 => .cmgt,
            else => return error.UnsupportedInstruction,
        };
        return .{ .float_op = .{ .op = op, .q = q, .double = double, .rd = rd, .rn = rn, .rm = rm } };
    }
    const op: IntOp = switch ((@as(u32, @intFromBool(u)) << 7) | opcode) {
        0b000_01100 => .smax,
        0b100_01100 => .umax,
        0b000_01101 => .smin,
        0b100_01101 => .umin,
        0b000_01110 => .sabd,
        0b100_01110 => .uabd,
        0b000_01111 => .saba,
        0b100_01111 => .uaba,
        0b000_10010 => .mla,
        0b100_10010 => .mls,
        0b000_10111 => .addp,
        0b000_10100 => .smaxp,
        0b100_10100 => .umaxp,
        0b000_10101 => .sminp,
        0b100_10101 => .uminp,
        0b000_01000 => .sshl,
        0b100_01000 => .ushl,
        else => return error.UnsupportedInstruction,
    };
    if (size == 3 and !q) return error.UnsupportedInstruction;
    const wide_ok = op == .addp or op == .sshl or op == .ushl;
    if (size == 3 and !wide_ok) return error.UnsupportedInstruction;
    return .{ .int_op = .{ .op = op, .q = q, .size = size, .rd = rd, .rn = rn, .rm = rm } };
}

// ---------------------------------------------------------------------------------------------
// Lifting
// ---------------------------------------------------------------------------------------------

fn lanes(q: bool, size: u2) usize {
    return (if (q) @as(usize, 16) else 8) >> size;
}

fn ones(c: *const Context, size: u2) !Value {
    return c.constant(c.uint(size), @as(u64, std.math.maxInt(u64)) >> @intCast(64 - (@as(u7, 8) << size)));
}

fn zeroOf(c: *const Context, size: u2) !Value {
    return c.constant(c.uint(size), 0);
}

fn finish(c: *const Context, rd: u5, q: bool) !void {
    if (!q) try c.store(rd, 8, try c.constant(c.u64_t, 0));
}

fn signedOrNot(c: *const Context, size: u2, signed: bool) Type {
    return if (signed) c.sint(size) else c.uint(size);
}

pub fn emit(c: *const Context, op: Op) !void {
    switch (op) {
        .shift => |v| try shift(c, v.op, v.q, v.size, v.amount, v.rd, v.rn),
        .shift_long => |v| {
            const half: usize = if (v.upper) 8 else 0;
            const count = @as(usize, 8) >> v.size;
            const wide: u2 = v.size + 1;
            var results: [8]Value = undefined;
            for (0..count) |i| {
                const lane = try c.load(v.rn, half + (i << v.size), signedOrNot(c, v.size, v.signed));
                const widened = try c.convert(signedOrNot(c, wide, v.signed), lane);
                const shifted = if (v.amount == 0) widened else try c.arith(signedOrNot(c, wide, v.signed), .shl, widened, try c.constant(signedOrNot(c, wide, v.signed), v.amount));
                results[i] = if (v.signed) try c.convert(c.uint(wide), shifted) else shifted;
            }
            for (0..count) |i| try c.store(v.rd, i << wide, results[i]);
        },
        .narrow => |v| {
            const count = @as(usize, 8) >> v.size;
            const wide: u2 = v.size + 1;
            const half: usize = if (v.upper) 8 else 0;
            var results: [8]Value = undefined;
            for (0..count) |i| results[i] = try c.convert(c.uint(v.size), try c.load(v.rn, i << wide, c.uint(wide)));
            for (0..count) |i| try c.store(v.rd, half + (i << v.size), results[i]);
            if (!v.upper) try c.store(v.rd, 8, try c.constant(c.u64_t, 0));
        },
        .unary => |v| try unary(c, v.op, v.q, v.size, v.rd, v.rn),
        .float_unary => |v| try floatUnary(c, v.op, v.q, v.double, v.rd, v.rn),
        .int_op => |v| try intOp(c, v.op, v.q, v.size, v.rd, v.rn, v.rm),
        .float_op => |v| try floatOp(c, v.op, v.q, v.double, v.rd, v.rn, v.rm),
        .permute => |v| try permute(c, v.op, v.q, v.size, v.rd, v.rn, v.rm),
        .extract => |v| {
            const total: usize = if (v.q) 16 else 8;
            var results: [16]Value = undefined;
            for (0..total) |i| {
                const at = i + v.index;
                results[i] = if (at < total) try c.load(v.rn, at, c.u8_t) else try c.load(v.rm, at - total, c.u8_t);
            }
            for (0..total) |i| try c.store(v.rd, i, results[i]);
            try finish(c, v.rd, v.q);
        },
        .across => |v| try across(c, v.op, v.q, v.size, v.rd, v.rn),
        .dup_scalar => |v| {
            const lane = try c.load(v.rn, @as(usize, v.index) << v.size, c.uint(v.size));
            try c.store(v.rd, 0, lane);
            const used: usize = @as(usize, 1) << v.size;
            if (used < 8) {
                if (used == 4) {
                    try c.store(v.rd, 4, try c.constant(c.u32_t, 0));
                } else if (used == 2) {
                    try c.store(v.rd, 2, try c.constant(c.u16_t, 0));
                    try c.store(v.rd, 4, try c.constant(c.u32_t, 0));
                } else {
                    try c.store(v.rd, 1, try c.constant(c.u8_t, 0));
                    try c.store(v.rd, 2, try c.constant(c.u16_t, 0));
                    try c.store(v.rd, 4, try c.constant(c.u32_t, 0));
                }
            }
            try c.store(v.rd, 8, try c.constant(c.u64_t, 0));
        },
        .pairwise_scalar => |v| {
            if (!v.float) {
                const sum = try c.arith(c.u64_t, .add, try c.load(v.rn, 0, c.u64_t), try c.load(v.rn, 8, c.u64_t));
                try c.store(v.rd, 0, sum);
                try c.store(v.rd, 8, try c.constant(c.u64_t, 0));
            } else {
                const precision: simd.Precision = if (v.double) .double else .single;
                const ty = c.float(precision);
                const step: usize = precision.bytesPublic();
                const sum = try c.arith(ty, .add, try c.load(v.rn, 0, ty), try c.load(v.rn, step, ty));
                try c.putFloat(v.rd, precision, sum);
            }
        },
        .element => |v| try element(c, v),
        .long => |v| try long(c, v),
        .float_widen => |v| {
            const half: usize = if (v.upper) 8 else 0;
            if (v.double) {
                var results: [2]Value = undefined;
                for (0..2) |i| results[i] = try c.convert(c.f64_t, try c.load(v.rn, half + i * 4, c.f32_t));
                for (0..2) |i| try c.store(v.rd, i * 8, results[i]);
            } else {
                var results: [4]Value = undefined;
                for (0..4) |i| results[i] = try simd.halfToSingle(c, try c.convert(c.u32_t, try c.load(v.rn, half + i * 2, c.u16_t)));
                for (0..4) |i| try c.store(v.rd, i * 4, results[i]);
            }
        },
        .float_narrow => |v| {
            const half: usize = if (v.upper) 8 else 0;
            if (v.double) {
                var results: [2]Value = undefined;
                for (0..2) |i| results[i] = try c.convert(c.f32_t, try c.load(v.rn, i * 8, c.f64_t));
                for (0..2) |i| try c.store(v.rd, half + i * 4, results[i]);
            } else {
                var results: [4]Value = undefined;
                for (0..4) |i| results[i] = try simd.singleToHalf(c, try c.load(v.rn, i * 4, c.u32_t));
                for (0..4) |i| try c.store(v.rd, half + i * 2, results[i]);
            }
            if (!v.upper) try c.store(v.rd, 8, try c.constant(c.u64_t, 0));
        },
    }
}

fn shift(c: *const Context, op: ShiftOp, q: bool, size: u2, amount: u7, rd: u5, rn: u5) !void {
    const bits_wide: u7 = @as(u7, 8) << size;
    const signed = op == .sshr or op == .ssra;
    const ty = if (signed) c.sint(size) else c.uint(size);
    const uty = c.uint(size);
    const count = lanes(q, size);
    var results: [16]Value = undefined;
    for (0..count) |i| {
        const at = i << size;
        const source = try c.load(rn, at, ty);
        results[i] = switch (op) {
            .shl => try c.arith(uty, .shl, try c.load(rn, at, uty), try c.constant(uty, amount)),
            .sshr, .ssra, .ushr, .usra => blk: {
                var shifted: Value = undefined;
                if (amount >= bits_wide) {
                    // A shift by the whole width leaves the sign for an arithmetic shift, and
                    // nothing for a logical one.
                    shifted = if (signed) try c.arith(ty, .shr, source, try c.constant(ty, bits_wide - 1)) else try c.constant(ty, 0);
                } else shifted = try c.arith(ty, .shr, source, try c.constant(ty, amount));
                const as_unsigned = if (signed) try c.convert(uty, shifted) else shifted;
                if (op == .ssra or op == .usra) break :blk try c.arith(uty, .add, try c.load(rd, at, uty), as_unsigned);
                break :blk as_unsigned;
            },
            .sli => blk: {
                const mask = (@as(u64, std.math.maxInt(u64)) << @intCast(amount)) & (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - @as(u7, bits_wide)));
                const moved = try c.arith(uty, .shl, try c.load(rn, at, uty), try c.constant(uty, amount));
                const kept = try c.arith(uty, .bit_and, try c.load(rd, at, uty), try c.constant(uty, ~mask & (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - @as(u7, bits_wide)))));
                break :blk try c.arith(uty, .bit_or, kept, moved);
            },
            .sri => blk: {
                const full = @as(u64, std.math.maxInt(u64)) >> @intCast(64 - @as(u7, bits_wide));
                const mask = if (amount >= bits_wide) 0 else full >> @intCast(amount);
                const moved = if (amount >= bits_wide) try c.constant(uty, 0) else try c.arith(uty, .shr, try c.load(rn, at, uty), try c.constant(uty, amount));
                const kept = try c.arith(uty, .bit_and, try c.load(rd, at, uty), try c.constant(uty, ~mask & full));
                break :blk try c.arith(uty, .bit_or, kept, moved);
            },
        };
    }
    for (0..count) |i| try c.store(rd, i << size, results[i]);
    try finish(c, rd, q);
}

fn unary(c: *const Context, op: UnaryOp, q: bool, size: u2, rd: u5, rn: u5) !void {
    // These three work on bytes whatever `size` says; for `NOT` and `RBIT` it is part of the opcode.
    const bytes_only = op == .not or op == .rbit or op == .cnt;
    const count = if (bytes_only) lanes(q, 0) else lanes(q, size);
    const uty = c.uint(size);
    const sty = c.sint(size);
    var results: [16]Value = undefined;
    switch (op) {
        .rev64, .rev32, .rev16 => {
            // The lanes inside each group of 8, 4 or 2 bytes are put in the opposite order.
            const group: usize = switch (op) {
                .rev64 => 8,
                .rev32 => 4,
                else => 2,
            };
            const per = group >> size;
            for (0..count) |i| {
                const base = (i / per) * per;
                results[i] = try c.load(rn, (base + (per - 1 - (i - base))) << size, uty);
            }
        },
        .clz => {
            for (0..count) |i| results[i] = try countLeadingZeros(c, size, try c.load(rn, i << size, uty));
        },
        .cnt => {
            for (0..count) |i| results[i] = try popcount8(c, try c.load(rn, i, c.u8_t));
        },
        .not => {
            // Only bytes exist for this one.
            for (0..count) |i| results[i] = try c.arith(c.u8_t, .bit_xor, try c.load(rn, i, c.u8_t), try c.constant(c.u8_t, 0xff));
        },
        .rbit => {
            for (0..count) |i| results[i] = try reverseBits8(c, try c.load(rn, i, c.u8_t));
        },
        .abs => {
            for (0..count) |i| {
                const v = try c.load(rn, i << size, sty);
                const negative = try c.compare(.lt, v, try c.constant(sty, 0));
                const negated = try c.arith(sty, .sub, try c.constant(sty, 0), v);
                results[i] = try c.convert(uty, try c.select(sty, negative, negated, v));
            }
        },
        .neg => {
            for (0..count) |i| results[i] = try c.arith(uty, .sub, try c.constant(uty, 0), try c.load(rn, i << size, uty));
        },
        .cmgt0, .cmeq0, .cmlt0, .cmge0, .cmle0 => {
            const all = try ones(c, size);
            const none = try zeroOf(c, size);
            for (0..count) |i| {
                const v = try c.load(rn, i << size, sty);
                const zero = try c.constant(sty, 0);
                const holds = try c.compare(switch (op) {
                    .cmgt0 => .gt,
                    .cmeq0 => .eq,
                    .cmlt0 => .lt,
                    .cmge0 => .ge,
                    else => .le,
                }, v, zero);
                results[i] = try c.select(uty, holds, all, none);
            }
        },
    }
    const out_size: u2 = if (bytes_only) 0 else size;
    for (0..count) |i| try c.store(rd, i << out_size, results[i]);
    try finish(c, rd, q);
}

fn countLeadingZeros(c: *const Context, size: u2, value: Value) !Value {
    // Binary search on the width: halve the interesting region until one bit is left.
    const ty = c.uint(size);
    const width: u64 = @as(u64, 8) << size;
    var count = try c.constant(ty, 0);
    var v = value;
    var step: u64 = width / 2;
    while (step >= 1) : (step /= 2) {
        const mask: u64 = (@as(u64, std.math.maxInt(u64)) >> @intCast(64 - step)) << @intCast(width - step);
        const high = try c.arith(ty, .bit_and, v, try c.constant(ty, mask));
        const empty = try c.compare(.eq, high, try c.constant(ty, 0));
        count = try c.select(ty, empty, try c.arith(ty, .add, count, try c.constant(ty, step)), count);
        v = try c.select(ty, empty, try c.arith(ty, .shl, v, try c.constant(ty, step)), v);
        if (step == 1) break;
    }
    // A value that was all zero has consumed the whole width by the search except the last bit.
    const last = try c.compare(.eq, v, try c.constant(ty, 0));
    return c.select(ty, last, try c.arith(ty, .add, count, try c.constant(ty, 1)), count);
}

fn popcount8(c: *const Context, value: Value) !Value {
    const t = c.u8_t;
    var v = value;
    const m1 = try c.constant(t, 0x55);
    const m2 = try c.constant(t, 0x33);
    const m4 = try c.constant(t, 0x0f);
    const a = try c.arith(t, .bit_and, try c.arith(t, .shr, v, try c.constant(t, 1)), m1);
    v = try c.arith(t, .sub, v, a);
    v = try c.arith(t, .add, try c.arith(t, .bit_and, v, m2), try c.arith(t, .bit_and, try c.arith(t, .shr, v, try c.constant(t, 2)), m2));
    return c.arith(t, .bit_and, try c.arith(t, .add, v, try c.arith(t, .shr, v, try c.constant(t, 4))), m4);
}

fn reverseBits8(c: *const Context, value: Value) !Value {
    const t = c.u8_t;
    var v = value;
    v = try c.arith(t, .bit_or, try c.arith(t, .shr, v, try c.constant(t, 4)), try c.arith(t, .shl, v, try c.constant(t, 4)));
    v = try c.arith(t, .bit_or, try c.arith(t, .shr, try c.arith(t, .bit_and, v, try c.constant(t, 0xcc)), try c.constant(t, 2)), try c.arith(t, .shl, try c.arith(t, .bit_and, v, try c.constant(t, 0x33)), try c.constant(t, 2)));
    v = try c.arith(t, .bit_or, try c.arith(t, .shr, try c.arith(t, .bit_and, v, try c.constant(t, 0xaa)), try c.constant(t, 1)), try c.arith(t, .shl, try c.arith(t, .bit_and, v, try c.constant(t, 0x55)), try c.constant(t, 1)));
    return v;
}

fn floatUnary(c: *const Context, op: FloatUnary, q: bool, double: bool, rd: u5, rn: u5) !void {
    const precision: simd.Precision = if (double) .double else .single;
    const size: u2 = if (double) 3 else 2;
    const ty = c.float(precision);
    const uty = c.unsignedOf(precision);
    const count = lanes(q, size);
    var results: [4]Value = undefined;
    for (0..count) |i| {
        const at = i << size;
        results[i] = switch (op) {
            .abs => try c.arith(uty, .bit_and, try c.load(rn, at, uty), try c.constant(uty, if (double) 0x7fff_ffff_ffff_ffff else 0x7fff_ffff)),
            .neg => try c.arith(uty, .bit_xor, try c.load(rn, at, uty), try c.constant(uty, if (double) 0x8000_0000_0000_0000 else 0x8000_0000)),
            .sqrt => try c.reinterpret(uty, try c.unary(ty, .sqrt, try c.load(rn, at, ty))),
            .scvtf => try c.reinterpret(uty, try c.convert(ty, try c.load(rn, at, c.sint(size)))),
            .ucvtf => try c.reinterpret(uty, try c.convert(ty, try c.load(rn, at, uty))),
            .fcvtzs, .fcvtzu => try convertToInt(c, op == .fcvtzs, precision, try c.load(rn, at, ty)),
            .cmeq0, .cmge0, .cmgt0, .cmle0, .cmlt0 => blk: {
                const v = try c.load(rn, at, ty);
                const zero = try c.func.appendInst(c.block, ty, .{ .fconst = 0 });
                const holds = try c.compare(switch (op) {
                    .cmeq0 => .eq,
                    .cmge0 => .ge,
                    .cmgt0 => .gt,
                    .cmle0 => .le,
                    else => .lt,
                }, v, zero);
                break :blk try c.select(uty, holds, try ones(c, size), try zeroOf(c, size));
            },
        };
    }
    for (0..count) |i| try c.store(rd, i << size, results[i]);
    try finish(c, rd, q);
}

/// The saturating conversion of one float to an integer of the same width, rounding toward zero.
fn convertToInt(c: *const Context, signed: bool, precision: simd.Precision, value: Value) !Value {
    const ty = c.float(precision);
    const width: u7 = if (precision == .single) 32 else 64;
    const target = if (signed) (if (width == 64) c.i64_t else c.i32_t) else (if (width == 64) c.u64_t else c.u32_t);
    const truncated = try c.unary(ty, .trunc, value);
    const top: f64 = if (signed) std.math.pow(f64, 2, @floatFromInt(width - 1)) else std.math.pow(f64, 2, @floatFromInt(width));
    const bottom: f64 = if (signed) -top else 0;
    const above = try c.compare(.ge, truncated, try c.func.appendInst(c.block, ty, .{ .fconst = top }));
    const below = try c.compare(.le, truncated, try c.func.appendInst(c.block, ty, .{ .fconst = bottom }));
    const nan = try c.isNan(value, precision);
    const zero = try c.func.appendInst(c.block, ty, .{ .fconst = 0 });
    const safe = try c.select(ty, below, zero, try c.select(ty, above, zero, truncated));
    var result = try c.convert(target, safe);
    const max_bits: u64 = if (signed) (@as(u64, 1) << @intCast(width - 1)) - 1 else if (width == 64) std.math.maxInt(u64) else std.math.maxInt(u32);
    const min_signed: i64 = if (width == 64) std.math.minInt(i64) else std.math.minInt(i32);
    result = try c.select(target, above, try c.constant(target, max_bits), result);
    result = try c.select(target, below, try c.constant(target, if (signed) @bitCast(min_signed) else 0), result);
    result = try c.select(target, nan, try c.constant(target, 0), result);
    return if (signed) try c.convert(c.unsignedOf(precision), result) else result;
}

fn intOp(c: *const Context, op: IntOp, q: bool, size: u2, rd: u5, rn: u5, rm: u5) !void {
    const signed = switch (op) {
        .smax, .smin, .sabd, .saba, .smaxp, .sminp, .sshl => true,
        else => false,
    };
    const ty = signedOrNot(c, size, signed);
    const uty = c.uint(size);
    const count = lanes(q, size);
    var results: [16]Value = undefined;
    switch (op) {
        .addp, .smaxp, .umaxp, .sminp, .uminp => {
            // The pairs of `Rn` then the pairs of `Rm`, concatenated.
            for (0..count) |i| {
                const source = if (i < count / 2) rn else rm;
                const j = if (i < count / 2) i else i - count / 2;
                const a = try c.load(source, (2 * j) << size, ty);
                const b = try c.load(source, (2 * j + 1) << size, ty);
                results[i] = switch (op) {
                    .addp => try c.arith(uty, .add, try c.load(source, (2 * j) << size, uty), try c.load(source, (2 * j + 1) << size, uty)),
                    .smaxp, .umaxp => try c.convertBack(uty, try c.select(ty, try c.compare(.gt, a, b), a, b)),
                    else => try c.convertBack(uty, try c.select(ty, try c.compare(.lt, a, b), a, b)),
                };
            }
        },
        else => for (0..count) |i| {
            const at = i << size;
            const a = try c.load(rn, at, ty);
            const b = try c.load(rm, at, ty);
            results[i] = switch (op) {
                .smax, .umax => try c.convertBack(uty, try c.select(ty, try c.compare(.gt, a, b), a, b)),
                .smin, .umin => try c.convertBack(uty, try c.select(ty, try c.compare(.lt, a, b), a, b)),
                .sabd, .uabd, .saba, .uaba => blk: {
                    const bigger = try c.compare(.gt, a, b);
                    const forward = try c.arith(ty, .sub, a, b);
                    const backward = try c.arith(ty, .sub, b, a);
                    const difference = try c.convertBack(uty, try c.select(ty, bigger, forward, backward));
                    if (op == .saba or op == .uaba) break :blk try c.arith(uty, .add, try c.load(rd, at, uty), difference);
                    break :blk difference;
                },
                .mla => try c.arith(uty, .add, try c.load(rd, at, uty), try c.arith(uty, .mul, try c.load(rn, at, uty), try c.load(rm, at, uty))),
                .mls => try c.arith(uty, .sub, try c.load(rd, at, uty), try c.arith(uty, .mul, try c.load(rn, at, uty), try c.load(rm, at, uty))),
                .sshl, .ushl => try shiftByRegister(c, op == .sshl, size, a, try c.load(rm, at, c.i8_t)),
                else => unreachable,
            };
        },
    }
    for (0..count) |i| try c.store(rd, i << size, results[i]);
    try finish(c, rd, q);
}

/// `SSHL` and `USHL`: the amount is the signed low byte of the second operand, left when it is
/// positive and right (rounding down) when negative. An amount past the width gives zero, or the
/// sign for an arithmetic shift to the right.
fn shiftByRegister(c: *const Context, signed: bool, size: u2, value: Value, amount: Value) !Value {
    const ty = if (signed) c.sint(size) else c.uint(size);
    const uty = c.uint(size);
    const width: u64 = @as(u64, 8) << size;
    const wide = c.i32_t;
    const n = try c.convert(wide, amount);
    const negative = try c.compare(.lt, n, try c.constant(wide, 0));
    const magnitude = try c.select(wide, negative, try c.arith(wide, .sub, try c.constant(wide, 0), n), n);
    const too_far = try c.compare(.ge, magnitude, try c.constant(wide, width));
    // Shifting by a safe amount and then replacing the result when the amount was too large.
    const safe = try c.select(wide, too_far, try c.constant(wide, 0), magnitude);
    const typed = try c.convert(ty, safe);
    const left = try c.arith(ty, .shl, value, typed);
    const right = try c.arith(ty, .shr, value, typed);
    const moved = try c.select(ty, negative, right, left);
    const fallback = if (signed)
        try c.select(ty, negative, try c.arith(ty, .shr, value, try c.constant(ty, width - 1)), try c.constant(ty, 0))
    else
        try c.constant(ty, 0);
    const picked = try c.select(ty, too_far, fallback, moved);
    return if (signed) c.convert(uty, picked) else picked;
}

fn floatLaneOp(c: *const Context, op: FloatOp, precision: simd.Precision, d: Value, a: Value, b: Value) !Value {
    const ty = c.float(precision);
    return switch (op) {
        .add => try c.arith(ty, .add, a, b),
        .sub => try c.arith(ty, .sub, a, b),
        .mul => try c.arith(ty, .mul, a, b),
        .div => try c.arith(ty, .div, a, b),
        .max, .maxp => try simd.minMax(c, .max, ty, precision, a, b),
        .min, .minp => try simd.minMax(c, .min, ty, precision, a, b),
        .maxnm => try simd.minMax(c, .maxnm, ty, precision, a, b),
        .minnm => try simd.minMax(c, .minnm, ty, precision, a, b),
        .abd => blk: {
            const difference = try c.arith(ty, .sub, a, b);
            const uty = c.unsignedOf(precision);
            const mask: u64 = if (precision == .single) 0x7fff_ffff else 0x7fff_ffff_ffff_ffff;
            break :blk try c.reinterpret(ty, try c.arith(uty, .bit_and, try c.reinterpret(uty, difference), try c.constant(uty, mask)));
        },
        .mla => try c.arith(ty, .add, d, try c.arith(ty, .mul, a, b)),
        .mls => try c.arith(ty, .sub, d, try c.arith(ty, .mul, a, b)),
        else => unreachable,
    };
}

fn floatOp(c: *const Context, op: FloatOp, q: bool, double: bool, rd: u5, rn: u5, rm: u5) !void {
    const precision: simd.Precision = if (double) .double else .single;
    const size: u2 = if (double) 3 else 2;
    const ty = c.float(precision);
    const uty = c.unsignedOf(precision);
    const count = lanes(q, size);
    var results: [4]Value = undefined;
    switch (op) {
        .addp, .maxp, .minp => {
            for (0..count) |i| {
                const source = if (i < count / 2) rn else rm;
                const j = if (i < count / 2) i else i - count / 2;
                const a = try c.load(source, (2 * j) << size, ty);
                const b = try c.load(source, (2 * j + 1) << size, ty);
                const lane_op: FloatOp = if (op == .addp) .add else op;
                results[i] = try c.reinterpret(uty, try floatLaneOp(c, lane_op, precision, a, a, b));
            }
        },
        .cmeq, .cmge, .cmgt => {
            for (0..count) |i| {
                const a = try c.load(rn, i << size, ty);
                const b = try c.load(rm, i << size, ty);
                const holds = try c.compare(switch (op) {
                    .cmeq => .eq,
                    .cmge => .ge,
                    else => .gt,
                }, a, b);
                results[i] = try c.select(uty, holds, try ones(c, size), try zeroOf(c, size));
            }
        },
        else => for (0..count) |i| {
            const at = i << size;
            const a = try c.load(rn, at, ty);
            const b = try c.load(rm, at, ty);
            const d = try c.load(rd, at, ty);
            results[i] = try c.reinterpret(uty, try floatLaneOp(c, op, precision, d, a, b));
        },
    }
    for (0..count) |i| try c.store(rd, i << size, results[i]);
    try finish(c, rd, q);
}

fn permute(c: *const Context, op: Permute, q: bool, size: u2, rd: u5, rn: u5, rm: u5) !void {
    const count = lanes(q, size);
    const uty = c.uint(size);
    var results: [16]Value = undefined;
    const second = op == .uzp2 or op == .trn2 or op == .zip2;
    const half = count / 2;
    for (0..count) |i| {
        results[i] = switch (op) {
            .uzp1, .uzp2 => blk: {
                // The even (or odd) lanes of `Rn` and then of `Rm`.
                const k = 2 * (if (i < half) i else i - half) + @intFromBool(second);
                break :blk try c.load(if (i < half) rn else rm, k << size, uty);
            },
            .trn1, .trn2 => blk: {
                const pair = i / 2;
                const k = 2 * pair + @intFromBool(second);
                break :blk try c.load(if (i % 2 == 0) rn else rm, k << size, uty);
            },
            .zip1, .zip2 => blk: {
                const k = (if (second) half else 0) + i / 2;
                break :blk try c.load(if (i % 2 == 0) rn else rm, k << size, uty);
            },
        };
    }
    for (0..count) |i| try c.store(rd, i << size, results[i]);
    try finish(c, rd, q);
}

fn across(c: *const Context, op: Across, q: bool, size: u2, rd: u5, rn: u5) !void {
    const count = lanes(q, size);
    const signed = op == .saddlv or op == .smaxv or op == .sminv;
    const ty = signedOrNot(c, size, signed);
    const uty = c.uint(size);
    switch (op) {
        .saddlv, .uaddlv => {
            const wide: u2 = size + 1;
            var sum = try c.constant(c.uint(wide), 0);
            for (0..count) |i| {
                const lane = try c.load(rn, i << size, ty);
                sum = try c.arith(c.uint(wide), .add, sum, try c.convertBack(c.uint(wide), try c.convert(signedOrNot(c, wide, signed), lane)));
            }
            try c.store(rd, 0, sum);
            try clearAbove(c, rd, @as(usize, 1) << wide);
        },
        else => {
            var best = try c.load(rn, 0, ty);
            var sum = try c.load(rn, 0, uty);
            for (1..count) |i| {
                const lane = try c.load(rn, i << size, ty);
                switch (op) {
                    .smaxv, .umaxv => best = try c.select(ty, try c.compare(.gt, lane, best), lane, best),
                    .sminv, .uminv => best = try c.select(ty, try c.compare(.lt, lane, best), lane, best),
                    else => sum = try c.arith(uty, .add, sum, try c.load(rn, i << size, uty)),
                }
            }
            try c.store(rd, 0, if (op == .addv) sum else try c.convertBack(uty, best));
            try clearAbove(c, rd, @as(usize, 1) << size);
        },
    }
}

/// Zero a register above its first `used` bytes.
fn clearAbove(c: *const Context, reg: u5, used: usize) !void {
    var at = used;
    while (at < 8) {
        const step: usize = if (at % 8 == 0) 8 else if (at % 4 == 0) 4 else if (at % 2 == 0) 2 else 1;
        const ty = switch (step) {
            1 => c.u8_t,
            2 => c.u16_t,
            4 => c.u32_t,
            else => c.u64_t,
        };
        try c.store(reg, at, try c.constant(ty, 0));
        at += step;
    }
    try c.store(reg, 8, try c.constant(c.u64_t, 0));
}

fn element(c: *const Context, v: anytype) !void {
    const count = lanes(v.q, v.size);
    var results: [16]Value = undefined;
    switch (v.op) {
        .mul, .mla, .mls => {
            const uty = c.uint(v.size);
            const scalar = try c.load(v.rm, @as(usize, v.index) << v.size, uty);
            for (0..count) |i| {
                const at = i << v.size;
                const product = try c.arith(uty, .mul, try c.load(v.rn, at, uty), scalar);
                results[i] = switch (v.op) {
                    .mul => product,
                    .mla => try c.arith(uty, .add, try c.load(v.rd, at, uty), product),
                    else => try c.arith(uty, .sub, try c.load(v.rd, at, uty), product),
                };
            }
        },
        else => {
            const precision: simd.Precision = if (v.size == 3) .double else .single;
            const ty = c.float(precision);
            const uty = c.unsignedOf(precision);
            const scalar = try c.load(v.rm, @as(usize, v.index) << v.size, ty);
            for (0..count) |i| {
                const at = i << v.size;
                const product = try c.arith(ty, .mul, try c.load(v.rn, at, ty), scalar);
                results[i] = try c.reinterpret(uty, switch (v.op) {
                    .fmul => product,
                    .fmla => try c.arith(ty, .add, try c.load(v.rd, at, ty), product),
                    else => try c.arith(ty, .sub, try c.load(v.rd, at, ty), product),
                });
            }
        },
    }
    for (0..count) |i| try c.store(v.rd, i << v.size, results[i]);
    try finish(c, v.rd, v.q);
}

fn long(c: *const Context, v: anytype) !void {
    const signed = switch (v.op) {
        .saddl, .ssubl, .sabdl, .smull, .smlal, .smlsl => true,
        else => false,
    };
    const wide: u2 = v.size + 1;
    const count = @as(usize, 8) >> v.size;
    const half: usize = if (v.upper) 8 else 0;
    const wty = signedOrNot(c, wide, signed);
    const wuty = c.uint(wide);
    var results: [8]Value = undefined;
    for (0..count) |i| {
        const at = half + (i << v.size);
        const a = try c.convert(wty, try c.load(v.rn, at, signedOrNot(c, v.size, signed)));
        const b = try c.convert(wty, try c.load(v.rm, at, signedOrNot(c, v.size, signed)));
        const acc = try c.load(v.rd, i << wide, wuty);
        results[i] = switch (v.op) {
            .saddl, .uaddl => try c.convertBack(wuty, try c.arith(wty, .add, a, b)),
            .ssubl, .usubl => try c.convertBack(wuty, try c.arith(wty, .sub, a, b)),
            .sabdl, .uabdl => blk: {
                const bigger = try c.compare(.gt, a, b);
                break :blk try c.convertBack(wuty, try c.select(wty, bigger, try c.arith(wty, .sub, a, b), try c.arith(wty, .sub, b, a)));
            },
            .smull, .umull => try c.convertBack(wuty, try c.arith(wty, .mul, a, b)),
            .smlal, .umlal => try c.arith(wuty, .add, acc, try c.convertBack(wuty, try c.arith(wty, .mul, a, b))),
            .smlsl, .umlsl => try c.arith(wuty, .sub, acc, try c.convertBack(wuty, try c.arith(wty, .mul, a, b))),
        };
    }
    for (0..count) |i| try c.store(v.rd, i << wide, results[i]);
}
