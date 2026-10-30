//! The SIMD and floating point instructions of an AArch64 guest: what they are, and the IR they lift
//! to.
//!
//! The registers are `Cpu.v`, 128 bits each, and every instruction here reads and writes them
//! through memory, lane by lane, at the width of the lane. A lane of a floating point register is
//! loaded as an integer and reinterpreted, so the bits the guest wrote are the bits it reads and
//! nothing is converted unless an instruction says so. A write to a scalar view of a register
//! zeroes the rest of it, which is what the architecture does and what a kernel's `memcpy`
//! relies on.
//!
//! Only what the decoder knows is lifted. Anything else is refused by `parse`, which is how a
//! guest finds out that an instruction is missing: with its address and its word, at the moment it
//! reaches it, and not as a wrong answer later.

const std = @import("std");
const ir = @import("vulcan-ir");
const Cpu = @import("cpu.zig").Cpu;
const Decode = @import("decode.zig");
const lift = @import("lift.zig");
const vector = @import("vector.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = Decode.Error;

/// The precision of a scalar floating point instruction. Half precision is not modelled.
pub const Precision = enum(u2) {
    single = 0,
    double = 1,

    fn bytes(self: Precision) u8 {
        return if (self == .single) 4 else 8;
    }

    pub fn bytesPublic(self: Precision) u8 {
        return self.bytes();
    }
};

pub const Format = enum { half, single, double };
pub const Binary = enum { mul, div, add, sub, max, min, maxnm, minnm, nmul };
pub const Unary = enum { mov, abs, neg, sqrt, rint_nearest, rint_up, rint_down, rint_zero };
pub const Fused = enum { madd, msub, nmadd, nmsub };
/// How a float is rounded on its way to an integer: to nearest even, toward plus infinity, toward
/// minus infinity, toward zero, or to nearest with ties away from zero.
pub const Rounding = enum { nearest, up, down, zero, away };
pub const IntWidth = enum { w32, x64 };

pub const LaneLogic = enum { and_, bic, orr, orn, eor, bsl, bit, bif };
pub const LaneInt = enum { add, sub, mul, cmeq, cmgt, cmge, cmhi, cmhs, cmtst };

pub const Op = union(enum) {
    /// `FADD` and the other two-source scalar operations.
    fp_binary: struct { op: Binary, precision: Precision, rd: u5, rn: u5, rm: u5 },
    /// `FMOV`, `FABS`, `FNEG`, `FSQRT` and `FRINT*` between registers of one precision.
    fp_unary: struct { op: Unary, precision: Precision, rd: u5, rn: u5 },
    /// `FCVT` between half, single and double.
    fp_convert: struct { from: Format, to: Format, rd: u5, rn: u5 },
    /// `FMADD`, `FMSUB`, `FNMADD` and `FNMSUB`: `Ra` plus or minus the product of `Rn` and `Rm`.
    fp_fused: struct { op: Fused, precision: Precision, rd: u5, rn: u5, rm: u5, ra: u5 },
    /// `FCMP` and `FCMPE`, against a register or against zero.
    fp_compare: struct { precision: Precision, rn: u5, rm: u5, zero: bool },
    /// `FCCMP` and `FCCMPE`.
    fp_cond_compare: struct { precision: Precision, rn: u5, rm: u5, cond: Decode.Condition, nzcv: u4 },
    /// `FCSEL`.
    fp_select: struct { precision: Precision, rd: u5, rn: u5, rm: u5, cond: Decode.Condition },
    /// `FMOV` of a constant.
    fp_immediate: struct { precision: Precision, rd: u5, imm8: u8 },
    /// `SCVTF` and `UCVTF` from a general register.
    int_to_fp: struct { signed: bool, precision: Precision, width: IntWidth, rd: u5, rn: u5 },
    /// `FCVT{N,P,M,Z,A}{S,U}` into a general register.
    fp_to_int: struct { signed: bool, rounding: Rounding, precision: Precision, width: IntWidth, rd: u5, rn: u5 },
    /// `FMOV` between a general register and the low 64 bits (or 32) of a vector register.
    fmov_to_fp: struct { precision: Precision, rd: u5, rn: u5 },
    fmov_to_gpr: struct { precision: Precision, rd: u5, rn: u5 },
    /// `FMOV Xd, Vn.D[1]` and `FMOV Vd.D[1], Xn`.
    fmov_high: struct { to_fp: bool, rd: u5, rn: u5 },

    /// The bitwise operations of the vector three-same class.
    vector_logic: struct { op: LaneLogic, q: bool, rd: u5, rn: u5, rm: u5 },
    /// The integer operations of the vector three-same class, at lane size `1 << size` bytes.
    vector_int: struct { op: LaneInt, q: bool, size: u2, rd: u5, rn: u5, rm: u5 },
    /// `DUP Vd.T, Vn.Ts[index]`.
    dup_element: struct { q: bool, size: u2, index: u4, rd: u5, rn: u5 },
    /// `DUP Vd.T, Wn|Xn`.
    dup_general: struct { q: bool, size: u2, rd: u5, rn: u5 },
    /// `INS Vd.Ts[index], Wn|Xn`.
    ins_general: struct { size: u2, index: u4, rd: u5, rn: u5 },
    /// `INS Vd.Ts[di], Vn.Ts[si]`.
    ins_element: struct { size: u2, dst: u4, src: u4, rd: u5, rn: u5 },
    /// `UMOV` and `SMOV` into a general register.
    mov_to_gpr: struct { signed: bool, size: u2, index: u4, wide: bool, rd: u5, rn: u5 },
    /// `MOVI`, `MVNI`, `ORR` and `BIC` with an immediate, and the `FMOV` of a vector constant.
    modified_immediate: struct { kind: ImmediateKind, q: bool, rd: u5, imm64: u64 },
    /// Everything else in Advanced SIMD that is decoded.
    vector: vector.Op,
};

pub const ImmediateKind = enum { movi, orr, bic };

fn bits(word: u32, high: u5, low: u5) u32 {
    return (word >> low) & ((@as(u32, 2) << (high - low)) - 1);
}

fn precisionOf(ftype: u32) Error!Precision {
    return switch (ftype) {
        0 => .single,
        1 => .double,
        else => error.UnsupportedInstruction,
    };
}

/// `VFPExpandImm`: the eight-bit constant of `FMOV` as the bit pattern it names.
pub fn expandImmediate(precision: Precision, imm8: u8) u64 {
    const sign: u64 = imm8 >> 7;
    const b6: u64 = (imm8 >> 6) & 1;
    const rest: u64 = imm8 & 0x3f;
    if (precision == .single) {
        const exponent: u64 = ((b6 ^ 1) << 7) | (if (b6 == 1) @as(u64, 0x1f) << 2 else 0) | (rest >> 4);
        const fraction: u64 = (rest & 0xf) << 19;
        return (sign << 31) | (exponent << 23) | fraction;
    }
    const exponent: u64 = ((b6 ^ 1) << 10) | (if (b6 == 1) @as(u64, 0xff) << 2 else 0) | (rest >> 4);
    const fraction: u64 = (rest & 0xf) << 48;
    return (sign << 63) | (exponent << 52) | fraction;
}

/// The 64-bit constant of the modified-immediate class, which spreads eight bits over a lane
/// pattern according to `cmode`. `op` is the architectural `op` bit. Returns the pattern, and for
/// the inverting forms the pattern is already inverted.
fn modifiedImmediate(cmode: u4, op: u1, abcdefgh: u8) Error!struct { value: u64, kind: ImmediateKind } {
    const imm: u64 = abcdefgh;
    switch (cmode) {
        // 32-bit lanes, the byte in one of four positions. Even `cmode` is the plain move and the
        // inverted one, odd is the `ORR` and `BIC`.
        0b0000, 0b0010, 0b0100, 0b0110, 0b0001, 0b0011, 0b0101, 0b0111 => {
            const shift: u6 = @as(u6, @intCast(cmode >> 1)) * 8;
            const lane: u64 = imm << shift;
            const pattern = lane | (lane << 32);
            if (cmode & 1 == 0) return .{ .value = if (op == 1) ~pattern else pattern, .kind = .movi };
            return .{ .value = pattern, .kind = if (op == 1) .bic else .orr };
        },
        // 16-bit lanes.
        0b1000, 0b1010, 0b1001, 0b1011 => {
            const shift: u6 = @as(u6, @intCast((cmode >> 1) & 1)) * 8;
            const lane: u64 = imm << shift;
            const pattern = lane | (lane << 16) | (lane << 32) | (lane << 48);
            if (cmode & 1 == 0) return .{ .value = if (op == 1) ~pattern else pattern, .kind = .movi };
            return .{ .value = pattern, .kind = if (op == 1) .bic else .orr };
        },
        // 32-bit lanes with ones shifted in: `MSL #8` and `MSL #16`.
        0b1100, 0b1101 => {
            const shift: u6 = if (cmode == 0b1100) 8 else 16;
            const lane: u64 = (imm << shift) | ((@as(u64, 1) << shift) - 1);
            const pattern = lane | (lane << 32);
            return .{ .value = if (op == 1) ~pattern else pattern, .kind = .movi };
        },
        0b1110 => {
            if (op == 0) {
                const byte = imm * 0x0101010101010101;
                return .{ .value = byte, .kind = .movi };
            }
            // Each bit of the immediate becomes a whole byte.
            var value: u64 = 0;
            for (0..8) |index| {
                if ((imm >> @intCast(index)) & 1 == 1) value |= @as(u64, 0xff) << @intCast(index * 8);
            }
            return .{ .value = value, .kind = .movi };
        },
        // Floating point constants: single in 32-bit lanes, double only with `op`.
        0b1111 => {
            if (op == 1) {
                const wide = expandImmediate(.double, abcdefgh);
                return .{ .value = wide, .kind = .movi };
            }
            const single = expandImmediate(.single, abcdefgh);
            return .{ .value = single | (single << 32), .kind = .movi };
        },
    }
}

/// Decode a word of the SIMD and floating point classes, or refuse it.
pub fn parse(word: u32) Error!Op {
    // Floating point data processing: `M S` clear, bits 28:24 are `11110`.
    if ((word & 0x7f00_0000) == 0x1e00_0000) {
        // Scalar float <-> integer, which is the one class here with a width bit.
        if ((word & 0x7f20_fc00) == 0x1e20_0000) return floatInteger(word);
        if ((word & 0xff20_0000) != 0x1e20_0000) return error.UnsupportedInstruction;
        const ftype = bits(word, 23, 22);
        const rd: u5 = @truncate(word);
        const rn: u5 = @truncate(word >> 5);
        const rm: u5 = @truncate(word >> 16);
        // One source.
        if ((word & 0xff20_7c00) == 0x1e20_4000) {
            const opcode = bits(word, 20, 15);
            switch (opcode) {
                0...3, 8...11 => {
                    const precision = try precisionOf(ftype);
                    const op: Unary = switch (opcode) {
                        0 => .mov,
                        1 => .abs,
                        2 => .neg,
                        3 => .sqrt,
                        8 => .rint_nearest,
                        9 => .rint_up,
                        10 => .rint_down,
                        11 => .rint_zero,
                        else => unreachable,
                    };
                    return .{ .fp_unary = .{ .op = op, .precision = precision, .rd = rd, .rn = rn } };
                },
                // The destination is in the opcode and the source in `ftype`; they must differ.
                4, 5, 7 => {
                    const from: Format = switch (ftype) {
                        0 => .single,
                        1 => .double,
                        3 => .half,
                        else => return error.UnsupportedInstruction,
                    };
                    const to: Format = switch (opcode) {
                        4 => .single,
                        5 => .double,
                        else => .half,
                    };
                    if (from == to) return error.UnsupportedInstruction;
                    return .{ .fp_convert = .{ .from = from, .to = to, .rd = rd, .rn = rn } };
                },
                else => {},
            }
            return error.UnsupportedInstruction;
        }
        // Two source.
        if ((word & 0xff20_0c00) == 0x1e20_0800) {
            const precision = try precisionOf(ftype);
            const op: Binary = switch (bits(word, 15, 12)) {
                0 => .mul,
                1 => .div,
                2 => .add,
                3 => .sub,
                4 => .max,
                5 => .min,
                6 => .maxnm,
                7 => .minnm,
                8 => .nmul,
                else => return error.UnsupportedInstruction,
            };
            return .{ .fp_binary = .{ .op = op, .precision = precision, .rd = rd, .rn = rn, .rm = rm } };
        }
        // Compare.
        if ((word & 0xff20_fc07) == 0x1e20_2000) {
            const precision = try precisionOf(ftype);
            const zero = (word & 0x08) != 0;
            if (zero and rm != 0) return error.UnsupportedInstruction;
            return .{ .fp_compare = .{ .precision = precision, .rn = rn, .rm = rm, .zero = zero } };
        }
        // Conditional compare.
        if ((word & 0xff20_0c00) == 0x1e20_0400) {
            const precision = try precisionOf(ftype);
            return .{ .fp_cond_compare = .{
                .precision = precision,
                .rn = rn,
                .rm = rm,
                .cond = @fromBackingInt(@intCast(bits(word, 15, 12))),
                .nzcv = @truncate(word),
            } };
        }
        // Conditional select.
        if ((word & 0xff20_0c00) == 0x1e20_0c00) {
            const precision = try precisionOf(ftype);
            return .{ .fp_select = .{ .precision = precision, .rd = rd, .rn = rn, .rm = rm, .cond = @fromBackingInt(@intCast(bits(word, 15, 12))) } };
        }
        // Immediate.
        if ((word & 0xff20_1fe0) == 0x1e20_1000) {
            const precision = try precisionOf(ftype);
            return .{ .fp_immediate = .{ .precision = precision, .rd = rd, .imm8 = @truncate(word >> 13) } };
        }
        return error.UnsupportedInstruction;
    }
    // Three source: `FMADD` and its three siblings. Bit 21 is `o1` and bit 15 is `o0`.
    if ((word & 0xff00_0000) == 0x1f00_0000) {
        const precision = try precisionOf(bits(word, 23, 22));
        const select = (bits(word, 21, 21) << 1) | bits(word, 15, 15);
        return .{ .fp_fused = .{
            .op = @fromBackingInt(@intCast(select)),
            .precision = precision,
            .rd = @truncate(word),
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .ra = @truncate(word >> 10),
        } };
    }
    // Advanced SIMD three same: `0 Q U 01110 size 1 Rm opcode 1 Rn Rd`.
    if ((word & 0x9f20_0400) == 0x0e20_0400) return threeSame(word) catch |err| return vectorOr(word, err);
    // Advanced SIMD copy: `0 Q op 01110000 imm5 0 imm4 1 Rn Rd`.
    if ((word & 0x9fe0_8400) == 0x0e00_0400) return copy(word);
    // Advanced SIMD modified immediate: `0 Q op 0111100000 abc cmode o2 1 defgh Rd`.
    if ((word & 0x9ff8_0400) == 0x0f00_0400) {
        const cmode: u4 = @truncate(word >> 12);
        const op: u1 = @truncate(word >> 29);
        const abcdefgh: u8 = @intCast((bits(word, 18, 16) << 5) | bits(word, 9, 5));
        if (bits(word, 11, 11) != 0) return error.UnsupportedInstruction;
        const result = try modifiedImmediate(cmode, op, abcdefgh);
        return .{ .modified_immediate = .{
            .kind = result.kind,
            .q = (word & 0x4000_0000) != 0,
            .rd = @truncate(word),
            .imm64 = result.value,
        } };
    }
    return vectorOr(word, error.UnsupportedInstruction);
}

/// Hand a word the first decoder did not know to the second, keeping the first's reason if the
/// second does not know it either.
fn vectorOr(word: u32, first: Error) Error!Op {
    return .{ .vector = vector.parse(word) catch return first };
}

fn floatInteger(word: u32) Error!Op {
    const wide = (word & 0x8000_0000) != 0;
    const ftype = bits(word, 23, 22);
    const rmode = bits(word, 20, 19);
    const opcode = bits(word, 18, 16);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    const width: IntWidth = if (wide) .x64 else .w32;
    // The top half: `FMOV Xd, Vn.D[1]` and `FMOV Vd.D[1], Xn`.
    if (ftype == 2) {
        if (!wide or rmode != 1) return error.UnsupportedInstruction;
        return switch (opcode) {
            7 => .{ .fmov_high = .{ .to_fp = true, .rd = rd, .rn = rn } },
            6 => .{ .fmov_high = .{ .to_fp = false, .rd = rd, .rn = rn } },
            else => error.UnsupportedInstruction,
        };
    }
    const precision = try precisionOf(ftype);
    switch (opcode) {
        2, 3 => {
            if (rmode != 0) return error.UnsupportedInstruction;
            return .{ .int_to_fp = .{ .signed = opcode == 2, .precision = precision, .width = width, .rd = rd, .rn = rn } };
        },
        6, 7 => {
            if (rmode != 0) return error.UnsupportedInstruction;
            // The move is bit for bit, so its width follows the precision: a single moves a `W`
            // register and a double an `X`.
            if ((precision == .single) == wide) return error.UnsupportedInstruction;
            if (opcode == 7) return .{ .fmov_to_fp = .{ .precision = precision, .rd = rd, .rn = rn } };
            return .{ .fmov_to_gpr = .{ .precision = precision, .rd = rd, .rn = rn } };
        },
        0, 1, 4, 5 => {
            const rounding: Rounding = switch (rmode) {
                0 => if (opcode >= 4) .away else .nearest,
                1 => .up,
                2 => .down,
                3 => .zero,
                else => unreachable,
            };
            // Round-away exists only in the rmode 0 column; the other columns take opcode 0 and 1.
            if (rmode != 0 and opcode >= 4) return error.UnsupportedInstruction;
            return .{ .fp_to_int = .{ .signed = opcode & 1 == 0, .rounding = rounding, .precision = precision, .width = width, .rd = rd, .rn = rn } };
        },
        else => return error.UnsupportedInstruction,
    }
}

fn threeSame(word: u32) Error!Op {
    const q = (word & 0x4000_0000) != 0;
    const u = (word & 0x2000_0000) != 0;
    const size: u2 = @truncate(word >> 22);
    const opcode = bits(word, 15, 11);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    const rm: u5 = @truncate(word >> 16);
    if (opcode == 0b00011) {
        const op: LaneLogic = switch ((@as(u32, @intFromBool(u)) << 2) | size) {
            0 => .and_,
            1 => .bic,
            2 => .orr,
            3 => .orn,
            4 => .eor,
            5 => .bsl,
            6 => .bit,
            7 => .bif,
            else => unreachable,
        };
        return .{ .vector_logic = .{ .op = op, .q = q, .rd = rd, .rn = rn, .rm = rm } };
    }
    const op: LaneInt = switch (opcode) {
        0b10000 => if (u) .sub else .add,
        0b10001 => if (u) .cmeq else .cmtst,
        0b00110 => if (u) .cmhi else .cmgt,
        0b00111 => if (u) .cmhs else .cmge,
        0b10011 => if (u) return error.UnsupportedInstruction else .mul,
        else => return error.UnsupportedInstruction,
    };
    // A 64-bit lane exists only in the 128-bit arrangement, and a multiply has none.
    if (size == 3 and (!q or op == .mul)) return error.UnsupportedInstruction;
    return .{ .vector_int = .{ .op = op, .q = q, .size = size, .rd = rd, .rn = rn, .rm = rm } };
}

fn copy(word: u32) Error!Op {
    const q = (word & 0x4000_0000) != 0;
    const op = (word & 0x2000_0000) != 0;
    const imm5 = bits(word, 20, 16);
    const imm4 = bits(word, 14, 11);
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    if (imm5 & 0xf == 0 and imm5 != 0x10) return error.UnsupportedInstruction;
    // The lowest set bit of `imm5` is the lane size, and what is above it is the index.
    const size: u2 = if (imm5 & 1 != 0) 0 else if (imm5 & 2 != 0) 1 else if (imm5 & 4 != 0) 2 else if (imm5 & 8 != 0) 3 else return error.UnsupportedInstruction;
    const index: u4 = @intCast(imm5 >> (@as(u5, size) + 1));
    if (op) {
        // `INS (element)`: the source index is in `imm4`.
        if (!q) return error.UnsupportedInstruction;
        return .{ .ins_element = .{ .size = size, .dst = index, .src = @intCast(imm4 >> size), .rd = rd, .rn = rn } };
    }
    switch (imm4) {
        0 => return .{ .dup_element = .{ .q = q, .size = size, .index = index, .rd = rd, .rn = rn } },
        1 => {
            if (size == 3 and !q) return error.UnsupportedInstruction;
            return .{ .dup_general = .{ .q = q, .size = size, .rd = rd, .rn = rn } };
        },
        3 => {
            if (!q) return error.UnsupportedInstruction;
            return .{ .ins_general = .{ .size = size, .index = index, .rd = rd, .rn = rn } };
        },
        5, 7 => {
            const signed = imm4 == 5;
            // `UMOV` into an `X` register is only for 64-bit lanes (Q set), and into a `W` register
            // for the smaller ones. `SMOV` reaches an `X` register from lanes up to a word.
            if (!signed) {
                if (size == 3 and !q) return error.UnsupportedInstruction;
                if (size != 3 and q) return error.UnsupportedInstruction;
            } else if (size == 3 or (size == 2 and !q)) return error.UnsupportedInstruction;
            return .{ .mov_to_gpr = .{ .signed = signed, .size = size, .index = index, .wide = q, .rd = rd, .rn = rn } };
        },
        else => return error.UnsupportedInstruction,
    }
}

// ---------------------------------------------------------------------------------------------
// Lifting
// ---------------------------------------------------------------------------------------------

/// The types a lowering needs, and where it is writing.
pub const Context = struct {
    func: *Function,
    block: Block,
    cpu_ptr: Value,
    u8_t: Type,
    u16_t: Type,
    u32_t: Type,
    u64_t: Type,
    i8_t: Type,
    i16_t: Type,
    i32_t: Type,
    i64_t: Type,
    f16_t: Type,
    f32_t: Type,
    f64_t: Type,
    bool_t: Type,
    ptr_t: Type,

    pub fn init(func: *Function, block: Block, cpu_ptr: Value) std.mem.Allocator.Error!Context {
        const t = &func.types;
        return .{
            .func = func,
            .block = block,
            .cpu_ptr = cpu_ptr,
            .u8_t = try t.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }),
            .u16_t = try t.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
            .u32_t = try t.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
            .u64_t = try t.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } }),
            .i8_t = try t.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } }),
            .i16_t = try t.intern(.{ .int = .{ .signedness = .signed, .bits = 16 } }),
            .i32_t = try t.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
            .i64_t = try t.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } }),
            .f16_t = try t.intern(.{ .float = .f16 }),
            .f32_t = try t.intern(.{ .float = .f32 }),
            .f64_t = try t.intern(.{ .float = .f64 }),
            .bool_t = try t.intern(.bool),
            .ptr_t = try t.ptrGlobal(),
        };
    }

    /// A value of a signed lane type as the unsigned one of the same width, which is how lanes are
    /// stored. A value that is already unsigned is returned as it is.
    pub fn convertBack(self: *const Context, ty: Type, value: Value) std.mem.Allocator.Error!Value {
        if (self.func.valueType(value) == ty) return value;
        return self.convert(ty, value);
    }

    pub fn uint(self: *const Context, size: u2) Type {
        return switch (size) {
            0 => self.u8_t,
            1 => self.u16_t,
            2 => self.u32_t,
            3 => self.u64_t,
        };
    }

    pub fn sint(self: *const Context, size: u2) Type {
        return switch (size) {
            0 => self.i8_t,
            1 => self.i16_t,
            2 => self.i32_t,
            3 => self.i64_t,
        };
    }

    pub fn float(self: *const Context, precision: Precision) Type {
        return if (precision == .single) self.f32_t else self.f64_t;
    }

    pub fn unsignedOf(self: *const Context, precision: Precision) Type {
        return if (precision == .single) self.u32_t else self.u64_t;
    }

    pub fn constant(self: *const Context, ty: Type, value: u64) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .iconst = @bitCast(value) });
    }

    pub fn arith(self: *const Context, ty: Type, op: ir.function.BinOp, lhs: Value, rhs: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    pub fn compare(self: *const Context, op: ir.function.CmpOp, lhs: Value, rhs: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, self.bool_t, .{ .icmp = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    pub fn select(self: *const Context, ty: Type, cond: Value, then: Value, otherwise: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .select = .{ .cond = cond, .then = then, .@"else" = otherwise } });
    }

    pub fn convert(self: *const Context, ty: Type, value: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .convert = .{ .value = value } });
    }

    pub fn reinterpret(self: *const Context, ty: Type, value: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .unary = .{ .op = .reinterpret, .value = value } });
    }

    pub fn unary(self: *const Context, ty: Type, op: ir.function.UnaryOp, value: Value) std.mem.Allocator.Error!Value {
        return self.func.appendInst(self.block, ty, .{ .unary = .{ .op = op, .value = value } });
    }

    /// Lane `lane` of register `reg`, which is `1 << size` bytes wide, as an integer of `ty`.
    pub fn load(self: *const Context, reg: u5, byte: usize, ty: Type) std.mem.Allocator.Error!Value {
        const at = @offsetOf(Cpu, "v") + @as(usize, reg) * 16 + byte;
        const delta = try self.constant(self.u64_t, at);
        const address = try self.arith(self.ptr_t, .add, self.cpu_ptr, delta);
        return self.func.appendInst(self.block, ty, .{ .load = .{ .ptr = address } });
    }

    pub fn store(self: *const Context, reg: u5, byte: usize, value: Value) std.mem.Allocator.Error!void {
        const at = @offsetOf(Cpu, "v") + @as(usize, reg) * 16 + byte;
        const delta = try self.constant(self.u64_t, at);
        const address = try self.arith(self.ptr_t, .add, self.cpu_ptr, delta);
        try self.func.appendStore(self.block, value, address);
    }

    /// Zero the bytes of register `reg` from `from` up to its top.
    pub fn clearFrom(self: *const Context, reg: u5, from: usize) std.mem.Allocator.Error!void {
        var at = from;
        if (at == 4) {
            try self.store(reg, 4, try self.constant(self.u32_t, 0));
            at = 8;
        }
        if (at == 8) {
            try self.store(reg, 8, try self.constant(self.u64_t, 0));
        }
    }

    /// The scalar view of register `reg`, as the bits of a float of `precision`.
    pub fn scalar(self: *const Context, reg: u5, precision: Precision) std.mem.Allocator.Error!Value {
        return self.load(reg, 0, self.unsignedOf(precision));
    }

    pub fn floatOf(self: *const Context, reg: u5, precision: Precision) std.mem.Allocator.Error!Value {
        return self.reinterpret(self.float(precision), try self.scalar(reg, precision));
    }

    /// Write a scalar result: the bits in the bottom of the register and zero above them.
    pub fn putScalar(self: *const Context, reg: u5, precision: Precision, bits_value: Value) std.mem.Allocator.Error!void {
        try self.store(reg, 0, bits_value);
        try self.clearFrom(reg, precision.bytes());
    }

    pub fn putFloat(self: *const Context, reg: u5, precision: Precision, value: Value) std.mem.Allocator.Error!void {
        try self.putScalar(reg, precision, try self.reinterpret(self.unsignedOf(precision), value));
    }

    /// Whether a float is a NaN, decided on its bits. Comparing a value with itself is the usual
    /// test and the optimizer folds it away as always equal, which is right for integers.
    pub fn isNan(self: *const Context, value: Value, precision: Precision) std.mem.Allocator.Error!Value {
        const int_ty = self.unsignedOf(precision);
        const magnitude: u64 = if (precision == .single) 0x7fff_ffff else 0x7fff_ffff_ffff_ffff;
        const infinity: u64 = if (precision == .single) 0x7f80_0000 else 0x7ff0_0000_0000_0000;
        const bits_value = try self.reinterpret(int_ty, value);
        const cleared = try self.arith(int_ty, .bit_and, bits_value, try self.constant(int_ty, magnitude));
        return self.compare(.gt, cleared, try self.constant(int_ty, infinity));
    }

    /// The value with its sign flipped, which is a bit operation so that a NaN and a zero keep
    /// everything but the sign.
    fn negate(self: *const Context, value: Value, precision: Precision) std.mem.Allocator.Error!Value {
        const int_ty = self.unsignedOf(precision);
        const sign: u64 = if (precision == .single) 0x8000_0000 else 0x8000_0000_0000_0000;
        const flipped = try self.arith(int_ty, .bit_xor, try self.reinterpret(int_ty, value), try self.constant(int_ty, sign));
        return self.reinterpret(self.float(precision), flipped);
    }

    fn gpr(self: *const Context, reg: u5, wide: bool) std.mem.Allocator.Error!Value {
        return lift.loadRegOrZero(self.func, self.block, self.cpu_ptr, reg, if (wide) self.u64_t else self.u32_t, self.u64_t, self.ptr_t);
    }

    fn putGpr(self: *const Context, reg: u5, value: Value) std.mem.Allocator.Error!void {
        if (reg == 31) return;
        try lift.storeReg(self.func, self.block, self.cpu_ptr, reg, value, self.u64_t, self.ptr_t);
    }

    fn putFlags(self: *const Context, nzcv: Value) std.mem.Allocator.Error!void {
        const shifted = try self.arith(self.u32_t, .shl, nzcv, try self.constant(self.u32_t, 28));
        const delta = try self.constant(self.u64_t, @offsetOf(Cpu, "flags"));
        const address = try self.arith(self.ptr_t, .add, self.cpu_ptr, delta);
        try self.func.appendStore(self.block, shifted, address);
    }
};

/// Lift one instruction. Nothing here ends a block.
pub fn emit(c: *const Context, op: Op) !void {
    switch (op) {
        .fp_binary => |b| try binary(c, b.op, b.precision, b.rd, b.rn, b.rm),
        .fp_unary => |u| try unaryOp(c, u.op, u.precision, u.rd, u.rn),
        .fp_convert => |v| {
            switch (v.to) {
                .half => {
                    const half = if (v.from == .single) try singleToHalf(c, try c.load(v.rn, 0, c.u32_t)) else try doubleToHalf(c, try c.load(v.rn, 0, c.u64_t));
                    try c.store(v.rd, 0, half);
                    try c.store(v.rd, 2, try c.constant(c.u16_t, 0));
                    try c.clearFrom(v.rd, 4);
                },
                .single => {
                    if (v.from == .half) {
                        const wide = try c.convert(c.u32_t, try c.load(v.rn, 0, c.u16_t));
                        try c.putScalar(v.rd, .single, try halfToSingle(c, wide));
                    } else {
                        try c.putFloat(v.rd, .single, try c.convert(c.f32_t, try c.floatOf(v.rn, .double)));
                    }
                },
                .double => {
                    if (v.from == .half) {
                        const wide = try c.convert(c.u32_t, try c.load(v.rn, 0, c.u16_t));
                        try c.putScalar(v.rd, .double, try halfToDouble(c, wide));
                    } else {
                        try c.putFloat(v.rd, .double, try c.convert(c.f64_t, try c.floatOf(v.rn, .single)));
                    }
                },
            }
        },
        .fp_fused => |f| try fused(c, f.op, f.precision, f.rd, f.rn, f.rm, f.ra),
        .fp_compare => |f| {
            const lhs = try c.floatOf(f.rn, f.precision);
            const rhs = if (f.zero)
                try c.func.appendInst(c.block, c.float(f.precision), .{ .fconst = 0 })
            else
                try c.floatOf(f.rm, f.precision);
            try c.putFlags(try compareFlags(c, f.precision, lhs, rhs));
        },
        .fp_cond_compare => |f| {
            const lhs = try c.floatOf(f.rn, f.precision);
            const rhs = try c.floatOf(f.rm, f.precision);
            const holds = try lift.conditionValue(c.func, c.block, c.cpu_ptr, f.cond, c.u32_t, c.bool_t, c.ptr_t);
            const computed = try compareFlags(c, f.precision, lhs, rhs);
            const fallback = try c.constant(c.u32_t, f.nzcv);
            try c.putFlags(try c.select(c.u32_t, holds, computed, fallback));
        },
        .fp_select => |f| {
            const holds = try lift.conditionValue(c.func, c.block, c.cpu_ptr, f.cond, c.u32_t, c.bool_t, c.ptr_t);
            const lhs = try c.scalar(f.rn, f.precision);
            const rhs = try c.scalar(f.rm, f.precision);
            try c.putScalar(f.rd, f.precision, try c.select(c.unsignedOf(f.precision), holds, lhs, rhs));
        },
        .fp_immediate => |f| try c.putScalar(f.rd, f.precision, try c.constant(c.unsignedOf(f.precision), expandImmediate(f.precision, f.imm8))),
        .int_to_fp => |f| {
            const source_ty = if (f.signed) (if (f.width == .x64) c.i64_t else c.i32_t) else (if (f.width == .x64) c.u64_t else c.u32_t);
            const source = try lift.loadRegOrZero(c.func, c.block, c.cpu_ptr, f.rn, source_ty, c.u64_t, c.ptr_t);
            try c.putFloat(f.rd, f.precision, try c.convert(c.float(f.precision), source));
        },
        .fp_to_int => |f| try floatToInt(c, f.signed, f.rounding, f.precision, f.width, f.rd, f.rn),
        .fmov_to_fp => |f| {
            const value = try c.gpr(f.rn, f.precision == .double);
            try c.putScalar(f.rd, f.precision, value);
        },
        .fmov_to_gpr => |f| try c.putGpr(f.rd, try c.scalar(f.rn, f.precision)),
        .fmov_high => |f| {
            if (f.to_fp) {
                try c.store(f.rd, 8, try c.gpr(f.rn, true));
            } else {
                try c.putGpr(f.rd, try c.load(f.rn, 8, c.u64_t));
            }
        },
        .vector_logic => |v| try vectorLogic(c, v.op, v.q, v.rd, v.rn, v.rm),
        .vector_int => |v| try vectorInt(c, v.op, v.q, v.size, v.rd, v.rn, v.rm),
        .dup_element => |v| {
            const ty = c.uint(v.size);
            const lane = try c.load(v.rn, @as(usize, v.index) << v.size, ty);
            try fill(c, v.rd, v.q, v.size, lane);
        },
        .dup_general => |v| {
            const source = try c.gpr(v.rn, v.size == 3);
            const lane = if (v.size == 3) source else if (v.size == 2) source else try c.convert(c.uint(v.size), source);
            try fill(c, v.rd, v.q, v.size, lane);
        },
        .ins_general => |v| {
            const source = try c.gpr(v.rn, v.size == 3);
            const lane = if (v.size >= 2) source else try c.convert(c.uint(v.size), source);
            try c.store(v.rd, @as(usize, v.index) << v.size, lane);
        },
        .ins_element => |v| {
            const lane = try c.load(v.rn, @as(usize, v.src) << v.size, c.uint(v.size));
            try c.store(v.rd, @as(usize, v.dst) << v.size, lane);
        },
        .mov_to_gpr => |v| {
            const at = @as(usize, v.index) << v.size;
            if (!v.signed) {
                const lane = try c.load(v.rn, at, c.uint(v.size));
                try c.putGpr(v.rd, if (v.size == 3) lane else try c.convert(c.u32_t, lane));
            } else {
                const lane = try c.load(v.rn, at, c.sint(v.size));
                const target = if (v.wide) c.i64_t else c.i32_t;
                const extended = try c.convert(target, lane);
                // A `W` destination is a 32-bit value that is then zero-extended, not sign-extended.
                try c.putGpr(v.rd, if (v.wide) extended else try c.convert(c.u32_t, extended));
            }
        },
        .modified_immediate => |v| try modified(c, v.kind, v.q, v.rd, v.imm64),
        .vector => |v| try vector.emit(c, v),
    }
}

fn binary(c: *const Context, op: Binary, precision: Precision, rd: u5, rn: u5, rm: u5) !void {
    const ty = c.float(precision);
    const lhs = try c.floatOf(rn, precision);
    const rhs = try c.floatOf(rm, precision);
    const result = switch (op) {
        .mul => try c.arith(ty, .mul, lhs, rhs),
        .div => try c.arith(ty, .div, lhs, rhs),
        .add => try c.arith(ty, .add, lhs, rhs),
        .sub => try c.arith(ty, .sub, lhs, rhs),
        .nmul => try c.negate(try c.arith(ty, .mul, lhs, rhs), precision),
        .max, .min, .maxnm, .minnm => try minMax(c, op, ty, precision, lhs, rhs),
    };
    try c.putFloat(rd, precision, result);
}

/// `FMAX` and friends. A NaN in either operand gives a NaN, a signalling one before a quiet one and
/// the first operand before the second, quieted. The `NM` forms prefer a number to a quiet NaN.
/// Plus zero is greater than minus zero, which a compare cannot see, so ties are settled on the bits.
pub fn minMax(c: *const Context, op: Binary, ty: Type, precision: Precision, lhs_in: Value, rhs_in: Value) !Value {
    const want_max = op == .max or op == .maxnm;
    const prefer_number = op == .maxnm or op == .minnm;
    const int_ty = c.unsignedOf(precision);
    const quiet_bit: u64 = if (precision == .single) 0x40_0000 else 0x8_0000_0000_0000;
    const quiet = try c.constant(int_ty, quiet_bit);
    const zero = try c.constant(int_ty, 0);
    const lhs_bits = try c.reinterpret(int_ty, lhs_in);
    const rhs_bits = try c.reinterpret(int_ty, rhs_in);
    const lhs_nan = try c.isNan(lhs_in, precision);
    const rhs_nan = try c.isNan(rhs_in, precision);
    const lhs_quiet_bit_clear = try c.compare(.eq, try c.arith(int_ty, .bit_and, lhs_bits, quiet), zero);
    const rhs_quiet_bit_clear = try c.compare(.eq, try c.arith(int_ty, .bit_and, rhs_bits, quiet), zero);
    const lhs_signalling = try c.arith(c.bool_t, .bit_and, lhs_nan, lhs_quiet_bit_clear);
    const rhs_signalling = try c.arith(c.bool_t, .bit_and, rhs_nan, rhs_quiet_bit_clear);
    var lhs = lhs_in;
    var rhs = rhs_in;
    var lhs_live = lhs_nan;
    var rhs_live = rhs_nan;
    if (prefer_number) {
        const not = struct {
            fn f(ctx: *const Context, v: Value) !Value {
                return ctx.arith(ctx.bool_t, .bit_xor, v, try ctx.constant(ctx.bool_t, 1));
            }
        }.f;
        const lhs_is_quiet = try c.arith(c.bool_t, .bit_and, lhs_nan, try not(c, lhs_signalling));
        const rhs_is_quiet = try c.arith(c.bool_t, .bit_and, rhs_nan, try not(c, rhs_signalling));
        const ignore_lhs = try c.arith(c.bool_t, .bit_and, lhs_is_quiet, try not(c, rhs_is_quiet));
        const ignore_rhs = try c.arith(c.bool_t, .bit_and, rhs_is_quiet, try not(c, lhs_is_quiet));
        // Ignoring one is standing in for it with the value that loses.
        const loser = try c.func.appendInst(c.block, ty, .{ .fconst = if (want_max) -std.math.inf(f64) else std.math.inf(f64) });
        lhs = try c.select(ty, ignore_lhs, loser, lhs);
        rhs = try c.select(ty, ignore_rhs, loser, rhs);
        lhs_live = try c.arith(c.bool_t, .bit_and, lhs_nan, try not(c, ignore_lhs));
        rhs_live = try c.arith(c.bool_t, .bit_and, rhs_nan, try not(c, ignore_rhs));
    }
    // Equal values differ only in the sign of a zero: for a maximum the one with the sign clear
    // wins, for a minimum the one with it set, which is a bitwise AND for one and OR for the other.
    const tied_bits = try c.arith(int_ty, if (want_max) .bit_and else .bit_or, try c.reinterpret(int_ty, lhs), try c.reinterpret(int_ty, rhs));
    var result = try c.select(ty, try c.compare(if (want_max) .gt else .lt, lhs, rhs), lhs, rhs);
    result = try c.select(ty, try c.compare(.eq, lhs, rhs), try c.reinterpret(ty, tied_bits), result);
    // The NaN that comes out, lowest priority first so that each later one overrides.
    const lhs_out = try c.reinterpret(ty, try c.arith(int_ty, .bit_or, lhs_bits, quiet));
    const rhs_out = try c.reinterpret(ty, try c.arith(int_ty, .bit_or, rhs_bits, quiet));
    result = try c.select(ty, rhs_live, rhs_out, result);
    result = try c.select(ty, lhs_live, lhs_out, result);
    result = try c.select(ty, rhs_signalling, rhs_out, result);
    result = try c.select(ty, lhs_signalling, lhs_out, result);
    return result;
}

fn unaryOp(c: *const Context, op: Unary, precision: Precision, rd: u5, rn: u5) !void {
    const sign: u64 = if (precision == .single) 0x8000_0000 else 0x8000_0000_0000_0000;
    const int_ty = c.unsignedOf(precision);
    switch (op) {
        .mov => try c.putScalar(rd, precision, try c.scalar(rn, precision)),
        // Both leave a NaN's payload alone and only touch the sign, so they are bit operations.
        .abs => try c.putScalar(rd, precision, try c.arith(int_ty, .bit_and, try c.scalar(rn, precision), try c.constant(int_ty, sign - 1))),
        .neg => try c.putScalar(rd, precision, try c.arith(int_ty, .bit_xor, try c.scalar(rn, precision), try c.constant(int_ty, sign))),
        .sqrt => try c.putFloat(rd, precision, try c.unary(c.float(precision), .sqrt, try c.floatOf(rn, precision))),
        .rint_nearest => try c.putFloat(rd, precision, try c.unary(c.float(precision), .nearest, try c.floatOf(rn, precision))),
        .rint_up => try c.putFloat(rd, precision, try c.unary(c.float(precision), .ceil, try c.floatOf(rn, precision))),
        .rint_down => try c.putFloat(rd, precision, try c.unary(c.float(precision), .floor, try c.floatOf(rn, precision))),
        .rint_zero => try c.putFloat(rd, precision, try c.unary(c.float(precision), .trunc, try c.floatOf(rn, precision))),
    }
}

fn fused(c: *const Context, op: Fused, precision: Precision, rd: u5, rn: u5, rm: u5, ra: u5) !void {
    const ty = c.float(precision);
    var addend = try c.floatOf(ra, precision);
    var first = try c.floatOf(rn, precision);
    const second = try c.floatOf(rm, precision);
    // The architecture negates the addend, and the first factor, as the instruction says, and then
    // does one multiply-add. Negating the first factor and not the product is what decides the
    // sign of a NaN that comes out.
    switch (op) {
        .madd => {},
        .msub => first = try c.negate(first, precision),
        .nmadd => {
            addend = try c.negate(addend, precision);
            first = try c.negate(first, precision);
        },
        .nmsub => addend = try c.negate(addend, precision),
    }
    // The product is rounded before the add here, where the architecture rounds once. A single
    // precision product is exact in a double, so for those the sum is formed there and rounded
    // once. Doubles can differ from the hardware in the last place.
    if (precision == .single) {
        const wide_first = try c.convert(c.f64_t, first);
        const wide_second = try c.convert(c.f64_t, second);
        const wide_addend = try c.convert(c.f64_t, addend);
        const sum = try c.arith(c.f64_t, .add, wide_addend, try c.arith(c.f64_t, .mul, wide_first, wide_second));
        const exact = try c.convert(ty, sum);
        // A NaN is handled by the single precision arithmetic, which keeps its payload and its
        // sign in the way the architecture does; the widening would quiet and reshape it.
        const any_nan = try c.arith(c.bool_t, .bit_or, try c.arith(c.bool_t, .bit_or, try c.isNan(first, precision), try c.isNan(second, precision)), try c.isNan(addend, precision));
        const narrow = try c.arith(ty, .add, addend, try c.arith(ty, .mul, first, second));
        try c.putFloat(rd, precision, try c.select(ty, any_nan, narrow, exact));
        return;
    }
    const product = try c.arith(ty, .mul, first, second);
    try c.putFloat(rd, precision, try c.arith(ty, .add, addend, product));
}

fn compareFlags(c: *const Context, precision: Precision, lhs: Value, rhs: Value) !Value {
    const lhs_nan = try c.isNan(lhs, precision);
    const rhs_nan = try c.isNan(rhs, precision);
    const unordered = try c.arith(c.bool_t, .bit_or, lhs_nan, rhs_nan);
    const equal = try c.compare(.eq, lhs, rhs);
    const less = try c.compare(.lt, lhs, rhs);
    // Greater is 0010, equal 0110, less 1000 and unordered 0011.
    var flags = try c.constant(c.u32_t, 0b0010);
    flags = try c.select(c.u32_t, less, try c.constant(c.u32_t, 0b1000), flags);
    flags = try c.select(c.u32_t, equal, try c.constant(c.u32_t, 0b0110), flags);
    flags = try c.select(c.u32_t, unordered, try c.constant(c.u32_t, 0b0011), flags);
    return flags;
}

fn floatToInt(c: *const Context, signed: bool, rounding: Rounding, precision: Precision, width: IntWidth, rd: u5, rn: u5) !void {
    const ty = c.float(precision);
    var value = try c.floatOf(rn, precision);
    switch (rounding) {
        .zero => value = try c.unary(ty, .trunc, value),
        .up => value = try c.unary(ty, .ceil, value),
        .down => value = try c.unary(ty, .floor, value),
        .nearest => value = try c.unary(ty, .nearest, value),
        // Ties away from zero: add one half with the sign of the value, then truncate.
        .away => {
            const half = try c.func.appendInst(c.block, ty, .{ .fconst = 0.5 });
            const negative = try c.compare(.lt, value, try c.func.appendInst(c.block, ty, .{ .fconst = 0 }));
            const offset = try c.select(ty, negative, try c.func.appendInst(c.block, ty, .{ .fconst = -0.5 }), half);
            value = try c.unary(ty, .trunc, try c.arith(ty, .add, value, offset));
        },
    }
    const bits_wide: u7 = if (width == .x64) 64 else 32;
    const target = if (signed) (if (width == .x64) c.i64_t else c.i32_t) else (if (width == .x64) c.u64_t else c.u32_t);
    // The architecture saturates, and NaN is zero. The host conversion does neither, so the
    // bounds are compared here, in the float, and the conversion only sees values it can take.
    const top: f64 = if (signed) std.math.pow(f64, 2, @floatFromInt(bits_wide - 1)) else std.math.pow(f64, 2, @floatFromInt(bits_wide));
    const bottom: f64 = if (signed) -top else 0;
    const above = try c.compare(.ge, value, try c.func.appendInst(c.block, ty, .{ .fconst = top }));
    const below = try c.compare(.le, value, try c.func.appendInst(c.block, ty, .{ .fconst = bottom }));
    const nan = try c.isNan(value, precision);
    const safe_hi = try c.select(ty, above, try c.func.appendInst(c.block, ty, .{ .fconst = 0 }), value);
    const safe = try c.select(ty, below, try c.func.appendInst(c.block, ty, .{ .fconst = 0 }), safe_hi);
    var result = try c.convert(target, safe);
    const max_bits: u64 = if (signed) (@as(u64, 1) << @intCast(bits_wide - 1)) - 1 else if (width == .x64) std.math.maxInt(u64) else std.math.maxInt(u32);
    const min_signed: i64 = if (width == .x64) std.math.minInt(i64) else std.math.minInt(i32);
    const max_value = try c.constant(target, max_bits);
    const min_value = try c.constant(target, if (signed) @bitCast(min_signed) else 0);
    result = try c.select(target, above, max_value, result);
    result = try c.select(target, below, min_value, result);
    result = try c.select(target, nan, try c.constant(target, 0), result);
    // A `W` destination is zero-extended, so the signed 32-bit result is made unsigned first.
    try c.putGpr(rd, if (width == .w32 and signed) try c.convert(c.u32_t, result) else result);
}

fn fill(c: *const Context, rd: u5, q: bool, size: u2, lane: Value) !void {
    const count: usize = (if (q) @as(usize, 16) else 8) >> size;
    for (0..count) |index| try c.store(rd, index << size, lane);
    if (!q) try c.store(rd, 8, try c.constant(c.u64_t, 0));
}

fn vectorLogic(c: *const Context, op: LaneLogic, q: bool, rd: u5, rn: u5, rm: u5) !void {
    const halves: usize = if (q) 2 else 1;
    for (0..halves) |half| {
        const at = half * 8;
        const n = try c.load(rn, at, c.u64_t);
        const m = try c.load(rm, at, c.u64_t);
        const result = switch (op) {
            .and_ => try c.arith(c.u64_t, .bit_and, n, m),
            .orr => try c.arith(c.u64_t, .bit_or, n, m),
            .eor => try c.arith(c.u64_t, .bit_xor, n, m),
            .bic => try c.arith(c.u64_t, .bit_and, n, try c.arith(c.u64_t, .bit_xor, m, try c.constant(c.u64_t, std.math.maxInt(u64)))),
            .orn => try c.arith(c.u64_t, .bit_or, n, try c.arith(c.u64_t, .bit_xor, m, try c.constant(c.u64_t, std.math.maxInt(u64)))),
            // The three bit-select forms differ in which operand is the mask, and the mask is also
            // the thing that is read: the destination's old value.
            .bsl, .bit, .bif => blk: {
                const d = try c.load(rd, at, c.u64_t);
                const mask, const first, const second = switch (op) {
                    .bsl => .{ d, n, m },
                    .bit => .{ m, n, d },
                    .bif => .{ try c.arith(c.u64_t, .bit_xor, m, try c.constant(c.u64_t, std.math.maxInt(u64))), n, d },
                    else => unreachable,
                };
                // (first & mask) | (second & ~mask), written as second ^ ((first ^ second) & mask).
                const delta = try c.arith(c.u64_t, .bit_xor, first, second);
                break :blk try c.arith(c.u64_t, .bit_xor, second, try c.arith(c.u64_t, .bit_and, delta, mask));
            },
        };
        try c.store(rd, at, result);
    }
    if (!q) try c.store(rd, 8, try c.constant(c.u64_t, 0));
}

fn vectorInt(c: *const Context, op: LaneInt, q: bool, size: u2, rd: u5, rn: u5, rm: u5) !void {
    const signed_compare = op == .cmgt or op == .cmge;
    const ty = if (signed_compare) c.sint(size) else c.uint(size);
    const count: usize = (if (q) @as(usize, 16) else 8) >> size;
    const ones = try c.constant(c.uint(size), @as(u64, std.math.maxInt(u64)) >> @intCast(64 - (@as(u7, 8) << size)));
    const zero = try c.constant(c.uint(size), 0);
    // Every lane is read before any is written, because the destination may be a source.
    var results: [16]Value = undefined;
    for (0..count) |index| {
        const at = index << size;
        const n = try c.load(rn, at, ty);
        const m = try c.load(rm, at, ty);
        results[index] = switch (op) {
            .add => try c.arith(ty, .add, n, m),
            .sub => try c.arith(ty, .sub, n, m),
            .mul => try c.arith(ty, .mul, n, m),
            .cmeq => try c.select(c.uint(size), try c.compare(.eq, n, m), ones, zero),
            .cmgt, .cmhi => try c.select(c.uint(size), try c.compare(.gt, n, m), ones, zero),
            .cmge, .cmhs => try c.select(c.uint(size), try c.compare(.ge, n, m), ones, zero),
            .cmtst => try c.select(c.uint(size), try c.compare(.ne, try c.arith(ty, .bit_and, n, m), zero), ones, zero),
        };
    }
    for (0..count) |index| try c.store(rd, index << size, results[index]);
    if (!q) try c.store(rd, 8, try c.constant(c.u64_t, 0));
}

fn modified(c: *const Context, kind: ImmediateKind, q: bool, rd: u5, pattern: u64) !void {
    const halves: usize = if (q) 2 else 1;
    for (0..halves) |half| {
        const at = half * 8;
        const constant = try c.constant(c.u64_t, pattern);
        const result = switch (kind) {
            .movi => constant,
            .orr => try c.arith(c.u64_t, .bit_or, try c.load(rd, at, c.u64_t), constant),
            .bic => try c.arith(c.u64_t, .bit_and, try c.load(rd, at, c.u64_t), try c.constant(c.u64_t, ~pattern)),
        };
        try c.store(rd, at, result);
    }
    if (!q) try c.store(rd, 8, try c.constant(c.u64_t, 0));
}

// ---------------------------------------------------------------------------------------------
// Half precision
//
// The host's half precision type is not used: an f16 in the IR is a widened single on some
// targets, so the bits of one cannot be moved through it. The conversions are done on the bits.
// ---------------------------------------------------------------------------------------------

fn pickBits(c: *const Context, ty: Type, cond: Value, then: Value, otherwise: Value) !Value {
    return c.select(ty, cond, then, otherwise);
}

/// The single precision bits of the half precision value in the low 16 bits of `half`.
pub fn halfToSingle(c: *const Context, half: Value) !Value {
    const t = c.u32_t;
    const sign = try c.arith(t, .shl, try c.arith(t, .bit_and, half, try c.constant(t, 0x8000)), try c.constant(t, 16));
    const exponent = try c.arith(t, .bit_and, try c.arith(t, .shr, half, try c.constant(t, 10)), try c.constant(t, 0x1f));
    const fraction = try c.arith(t, .bit_and, half, try c.constant(t, 0x3ff));
    const zero = try c.constant(t, 0);
    // A normal number: rebias the exponent and widen the fraction.
    const normal = try c.arith(t, .bit_or, try c.arith(t, .shl, try c.arith(t, .add, exponent, try c.constant(t, 112)), try c.constant(t, 23)), try c.arith(t, .shl, fraction, try c.constant(t, 13)));
    // Infinity and NaN: all ones exponent, with a NaN made quiet.
    const special_fraction = try c.arith(t, .bit_or, try c.arith(t, .shl, fraction, try c.constant(t, 13)), try c.select(t, try c.compare(.ne, fraction, zero), try c.constant(t, 0x40_0000), zero));
    const special = try c.arith(t, .bit_or, try c.constant(t, 0x7f80_0000), special_fraction);
    // A subnormal is the fraction times two to the minus twenty-four, which a single holds exactly.
    const scale = try c.func.appendInst(c.block, c.f32_t, .{ .fconst = 5.9604644775390625e-08 });
    const subnormal = try c.reinterpret(t, try c.arith(c.f32_t, .mul, try c.convert(c.f32_t, fraction), scale));
    var result = normal;
    result = try c.select(t, try c.compare(.eq, exponent, zero), subnormal, result);
    result = try c.select(t, try c.compare(.eq, exponent, try c.constant(t, 31)), special, result);
    return c.arith(t, .bit_or, result, sign);
}

/// The double precision bits of a half precision value, through the single.
pub fn halfToDouble(c: *const Context, half: Value) !Value {
    const single = try c.reinterpret(c.f32_t, try halfToSingle(c, half));
    // A NaN keeps its payload through the widening because the conversion preserves it.
    return c.reinterpret(c.u64_t, try c.convert(c.f64_t, single));
}

/// Round the significand of a float to a half. `bits` is the width of the source, 32 or 64.
fn toHalf(c: *const Context, comptime source_bits: u7, value_bits: Value) !Value {
    const t = if (source_bits == 32) c.u32_t else c.u64_t;
    const fraction_bits: u7 = if (source_bits == 32) 23 else 52;
    const bias_difference: u64 = if (source_bits == 32) 112 else 1008;
    const exponent_mask: u64 = if (source_bits == 32) 0xff else 0x7ff;
    const fraction_mask: u64 = (@as(u64, 1) << fraction_bits) - 1;
    const drop: u64 = fraction_bits - 10;
    const k = struct {
        fn n(ctx: *const Context, ty: Type, v: u64) !Value {
            return ctx.constant(ty, v);
        }
    }.n;
    const sign = try c.convert(c.u32_t, try c.arith(t, .bit_and, try c.arith(t, .shr, value_bits, try k(c, t, source_bits - 16)), try k(c, t, 0x8000)));
    const exponent = try c.arith(t, .bit_and, try c.arith(t, .shr, value_bits, try k(c, t, fraction_bits)), try k(c, t, exponent_mask));
    const fraction = try c.arith(t, .bit_and, value_bits, try k(c, t, fraction_mask));
    // Everything below is done in the source width; the result is cut to 16 bits at the end.
    const u = c.u32_t;
    const he = try c.arith(t, .sub, exponent, try k(c, t, bias_difference));
    const he_signed = try c.convert(c.i32_t, he);
    const normal = he_signed;
    _ = normal;
    // Normal range: half exponent 1 to 30 before rounding.
    const half_normal = try c.convert(u, try c.arith(t, .bit_or, try c.arith(t, .shl, he, try k(c, t, 10)), try c.arith(t, .shr, fraction, try k(c, t, drop))));
    const round_bit_n = try c.convert(u, try c.arith(t, .bit_and, try c.arith(t, .shr, fraction, try k(c, t, drop - 1)), try k(c, t, 1)));
    const sticky_n = try c.compare(.ne, try c.arith(t, .bit_and, fraction, try k(c, t, (@as(u64, 1) << @intCast(drop - 1)) - 1)), try k(c, t, 0));
    const one = try c.constant(u, 1);
    const zero = try c.constant(u, 0);
    const up_n = try c.arith(c.bool_t, .bit_and, try c.compare(.ne, round_bit_n, zero), try c.arith(c.bool_t, .bit_or, sticky_n, try c.compare(.ne, try c.arith(u, .bit_and, half_normal, one), zero)));
    const rounded_normal = try c.select(u, up_n, try c.arith(u, .add, half_normal, one), half_normal);
    // Subnormal range: the significand with its hidden bit, shifted right by `14 - he` (or the
    // same count in the double case) and rounded.
    const significand = try c.arith(t, .bit_or, fraction, try k(c, t, @as(u64, 1) << @intCast(fraction_bits)));
    const shift_wide = try c.arith(t, .sub, try k(c, t, if (source_bits == 32) 126 else 1051), exponent);
    const too_small = try c.compare(.gt, shift_wide, try k(c, t, if (source_bits == 32) 25 else 54));
    const shift = try c.select(t, too_small, try k(c, t, 1), shift_wide);
    const half_sub = try c.convert(u, try c.arith(t, .shr, significand, shift));
    const round_sub = try c.convert(u, try c.arith(t, .bit_and, try c.arith(t, .shr, significand, try c.arith(t, .sub, shift, try k(c, t, 1))), try k(c, t, 1)));
    const lower = try c.arith(t, .sub, try c.arith(t, .shl, try k(c, t, 1), try c.arith(t, .sub, shift, try k(c, t, 1))), try k(c, t, 1));
    const sticky_s = try c.compare(.ne, try c.arith(t, .bit_and, significand, lower), try k(c, t, 0));
    const up_s = try c.arith(c.bool_t, .bit_and, try c.compare(.ne, round_sub, zero), try c.arith(c.bool_t, .bit_or, sticky_s, try c.compare(.ne, try c.arith(u, .bit_and, half_sub, one), zero)));
    const rounded_sub = try c.select(u, up_s, try c.arith(u, .add, half_sub, one), half_sub);
    const subnormal = try c.select(u, too_small, zero, rounded_sub);
    // Classify by the source exponent.
    const is_special = try c.compare(.eq, exponent, try k(c, t, exponent_mask));
    const is_small = try c.compare(.le, try c.convert(c.i64_t, exponent), try c.constant(c.i64_t, @intCast(bias_difference)));
    const overflow = try c.compare(.ge, he, try k(c, t, 31));
    const infinity = try c.constant(u, 0x7c00);
    var magnitude = rounded_normal;
    magnitude = try c.select(u, overflow, infinity, magnitude);
    magnitude = try c.select(u, is_small, subnormal, magnitude);
    // Infinity, and a NaN made quiet with the top of its payload kept.
    const payload = try c.convert(u, try c.arith(t, .shr, fraction, try k(c, t, drop)));
    const is_nan = try c.compare(.ne, fraction, try k(c, t, 0));
    const special = try c.select(u, is_nan, try c.arith(u, .bit_or, try c.constant(u, 0x7e00), payload), infinity);
    magnitude = try c.select(u, is_special, special, magnitude);
    return c.convert(c.u16_t, try c.arith(u, .bit_or, magnitude, sign));
}

pub fn singleToHalf(c: *const Context, bits_value: Value) !Value {
    return toHalf(c, 32, bits_value);
}

pub fn doubleToHalf(c: *const Context, bits_value: Value) !Value {
    return toHalf(c, 64, bits_value);
}
