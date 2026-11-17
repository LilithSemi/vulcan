//! Structured RISC-V decoder. `decode` reads one instruction from a byte slice and returns its
//! operation, operands and length, so a CPU model steps a program without parsing text.
//! `disasm.zig` renders the same words to text and acts as the oracle for the tests here.

const std = @import("std");
const encode = @import("encode.zig");
const disasm = @import("disasm.zig");
const compress = @import("compress.zig");

/// One decoded instruction. Fields a given operation does not use are left at zero, because a
/// reader that has the operation knows which ones it reads.
pub const Decoded = struct {
    op: Op,
    /// Two for a compressed instruction and four otherwise. A reader steps the program counter by
    /// this rather than by a guess at the encoding.
    len: u3,
    rd: u5 = 0,
    rs1: u5 = 0,
    rs2: u5 = 0,
    /// Only the fused multiply add operations read a third source.
    rs3: u5 = 0,
    /// Sign extended where the operation's immediate is signed, so a reader adds it without
    /// knowing which form it came from. `lui` and `auipc` hold the value already shifted left by
    /// twelve, which is what the instruction means, so a reader uses it as it is and does not
    /// shift it again.
    imm: i64 = 0,
    csr: u12 = 0,
    /// The rounding mode a float operation names, which is `dyn` for most of them.
    rm: u3 = 0,
    /// The ordering an atomic asks for.
    aq: bool = false,
    rl: bool = false,
};

/// A compressed instruction decodes to the tag of the wide instruction it stands for, so there
/// is no tag for a compressed form. A dot in a mnemonic becomes an underscore in the tag.
pub const Op = enum {
    /// The table does not cover this word. The length is two or four. It is wrong for a word
    /// longer than four bytes, which River never runs. The caller decides whether to trap or to
    /// hand the word elsewhere.
    unknown,

    // RV64I.
    lui,
    auipc,
    jal,
    jalr,
    beq,
    bne,
    blt,
    bge,
    bltu,
    bgeu,
    lb,
    lh,
    lw,
    ld,
    lbu,
    lhu,
    lwu,
    sb,
    sh,
    sw,
    sd,
    addi,
    slti,
    sltiu,
    xori,
    ori,
    andi,
    slli,
    srli,
    srai,
    add,
    sub,
    sll,
    slt,
    sltu,
    xor,
    @"or",
    @"and",
    srl,
    sra,
    addiw,
    slliw,
    srliw,
    sraiw,
    addw,
    subw,
    sllw,
    srlw,
    sraw,

    // SYSTEM, MISC-MEM and privileged.
    ecall,
    ebreak,
    csrrw,
    csrrs,
    csrrc,
    csrrwi,
    csrrsi,
    csrrci,
    mret,
    sret,
    wfi,
    fence,
    fence_i,
    sfence_vma,

    // M.
    mul,
    mulh,
    mulhsu,
    mulhu,
    div,
    divu,
    rem,
    remu,
    mulw,
    divw,
    divuw,
    remw,
    remuw,

    // A, word.
    lr_w,
    sc_w,
    amoswap_w,
    amoadd_w,
    amoxor_w,
    amoand_w,
    amoor_w,
    amomin_w,
    amomax_w,
    amominu_w,
    amomaxu_w,

    // A, double word.
    lr_d,
    sc_d,
    amoswap_d,
    amoadd_d,
    amoxor_d,
    amoand_d,
    amoor_d,
    amomin_d,
    amomax_d,
    amominu_d,
    amomaxu_d,

    // F.
    flw,
    fsw,
    fmadd_s,
    fmsub_s,
    fnmsub_s,
    fnmadd_s,
    fadd_s,
    fsub_s,
    fmul_s,
    fdiv_s,
    fsqrt_s,
    fsgnj_s,
    fsgnjn_s,
    fsgnjx_s,
    fmin_s,
    fmax_s,
    fcvt_w_s,
    fcvt_wu_s,
    fcvt_l_s,
    fcvt_lu_s,
    fmv_x_w,
    feq_s,
    flt_s,
    fle_s,
    fclass_s,
    fcvt_s_w,
    fcvt_s_wu,
    fcvt_s_l,
    fcvt_s_lu,
    fmv_w_x,

    // D.
    fld,
    fsd,
    fmadd_d,
    fmsub_d,
    fnmsub_d,
    fnmadd_d,
    fadd_d,
    fsub_d,
    fmul_d,
    fdiv_d,
    fsqrt_d,
    fsgnj_d,
    fsgnjn_d,
    fsgnjx_d,
    fmin_d,
    fmax_d,
    fcvt_s_d,
    fcvt_d_s,
    feq_d,
    flt_d,
    fle_d,
    fclass_d,
    fcvt_w_d,
    fcvt_wu_d,
    fcvt_l_d,
    fcvt_lu_d,
    fcvt_d_w,
    fcvt_d_wu,
    fcvt_d_l,
    fcvt_d_lu,
    fmv_x_d,
    fmv_d_x,
};

pub const Error = error{
    /// The slice holds fewer bytes than the instruction at its start needs.
    Truncated,
};

/// Decodes the instruction at the start of `bytes`. A word the table does not cover returns
/// `.unknown` with a length of two or four, which is not an error. Only a slice too short for the
/// instruction it starts returns `error.Truncated`.
pub fn decode(bytes: []const u8) Error!Decoded {
    if (bytes.len < 2) return error.Truncated;
    // The length rule here reports two or four and nothing else. A word whose bits 4 to 2 are all
    // set is longer than four bytes in the base specification, and this reports four for it, which
    // is wrong for such a word. River implements none of those encodings, so one cannot appear in
    // an image it runs, and the operation comes back as unknown either way.
    if (bytes[0] & 3 != 3) return decode16(std.mem.readInt(u16, bytes[0..2], .little));
    if (bytes.len < 4) return error.Truncated;
    return decode32(std.mem.readInt(u32, bytes[0..4], .little));
}

/// Decodes a halfword whose two low bits are not both set. Each form takes the tag and operands
/// of the wide instruction it expands to, so `c.mv` is `add` with `rs1` of zero.
fn decode16(half: u16) Decoded {
    const h: u32 = half;
    const unknown: Decoded = .{ .op = .unknown, .len = 2 };
    return switch (@as(u2, @truncate(h))) {
        0b00 => quadrant0(h) orelse unknown,
        0b01 => quadrant1(h) orelse unknown,
        0b10 => quadrant2(h) orelse unknown,
        0b11 => unreachable,
    };
}

fn bit(h: u32, comptime n: u5) u32 {
    return (h >> n) & 1;
}

/// The bits `hi` down to `lo` of `h`, placed with their lowest bit at position `at`.
fn span(h: u32, comptime hi: u5, comptime lo: u5, comptime at: u5) u32 {
    return ((h >> lo) & ((1 << (hi - lo + 1)) - 1)) << at;
}

fn signExtend(value: u32, comptime bits: u5) i64 {
    const shift: u5 = @intCast(32 - @as(u6, bits));
    return @as(i32, @bitCast(value << shift)) >> shift;
}

/// A three bit register field names one of x8 to x15.
fn smallReg(h: u32, comptime lo: u5) u5 {
    return @as(u5, 8) + @as(u3, @truncate(h >> lo));
}

fn fullRd(h: u32) u5 {
    return @truncate(h >> 7);
}

fn fullRs2(h: u32) u5 {
    return @truncate(h >> 2);
}

fn immCi(h: u32) i64 {
    return signExtend(span(h, 12, 12, 5) | span(h, 6, 2, 0), 6);
}

fn shamtCi(h: u32) i64 {
    return span(h, 12, 12, 5) | span(h, 6, 2, 0);
}

fn wordOffsetCl(h: u32) i64 {
    return span(h, 12, 10, 3) | bit(h, 6) << 2 | bit(h, 5) << 6;
}

fn doubleOffsetCl(h: u32) i64 {
    return span(h, 12, 10, 3) | span(h, 6, 5, 6);
}

fn doubleLoadOffsetCi(h: u32) i64 {
    return span(h, 12, 12, 5) | span(h, 6, 5, 3) | span(h, 4, 2, 6);
}

fn doubleStoreOffsetCss(h: u32) i64 {
    return span(h, 12, 10, 3) | span(h, 9, 7, 6);
}

fn quadrant0(h: u32) ?Decoded {
    const rd = smallReg(h, 2);
    const rs1 = smallReg(h, 7);
    return switch (@as(u3, @truncate(h >> 13))) {
        0b000 => blk: {
            const imm = span(h, 12, 11, 4) | span(h, 10, 7, 6) | bit(h, 6) << 2 | bit(h, 5) << 3;
            break :blk if (imm == 0) null else .{ .op = .addi, .len = 2, .rd = rd, .rs1 = 2, .imm = imm };
        },
        0b001 => .{ .op = .fld, .len = 2, .rd = rd, .rs1 = rs1, .imm = doubleOffsetCl(h) },
        0b010 => .{ .op = .lw, .len = 2, .rd = rd, .rs1 = rs1, .imm = wordOffsetCl(h) },
        0b011 => .{ .op = .ld, .len = 2, .rd = rd, .rs1 = rs1, .imm = doubleOffsetCl(h) },
        0b100 => null,
        0b101 => .{ .op = .fsd, .len = 2, .rs1 = rs1, .rs2 = rd, .imm = doubleOffsetCl(h) },
        0b110 => .{ .op = .sw, .len = 2, .rs1 = rs1, .rs2 = rd, .imm = wordOffsetCl(h) },
        0b111 => .{ .op = .sd, .len = 2, .rs1 = rs1, .rs2 = rd, .imm = doubleOffsetCl(h) },
    };
}

fn quadrant1(h: u32) ?Decoded {
    const rd = fullRd(h);
    return switch (@as(u3, @truncate(h >> 13))) {
        0b000 => .{ .op = .addi, .len = 2, .rd = rd, .rs1 = rd, .imm = immCi(h) },
        // This slot is `c.jal` in RV32 and `c.addiw` in RV64.
        0b001 => if (rd == 0) null else .{ .op = .addiw, .len = 2, .rd = rd, .rs1 = rd, .imm = immCi(h) },
        0b010 => .{ .op = .addi, .len = 2, .rd = rd, .rs1 = 0, .imm = immCi(h) },
        0b011 => if (rd == 2) addi16sp(h) else cLui(h, rd),
        0b100 => miscAlu(h),
        0b101 => .{ .op = .jal, .len = 2, .rd = 0, .imm = jumpOffsetCj(h) },
        0b110 => .{ .op = .beq, .len = 2, .rs1 = smallReg(h, 7), .rs2 = 0, .imm = branchOffsetCb(h) },
        0b111 => .{ .op = .bne, .len = 2, .rs1 = smallReg(h, 7), .rs2 = 0, .imm = branchOffsetCb(h) },
    };
}

fn addi16sp(h: u32) ?Decoded {
    const raw = bit(h, 12) << 9 | bit(h, 6) << 4 | bit(h, 5) << 6 | span(h, 4, 3, 7) |
        bit(h, 2) << 5;
    if (raw == 0) return null;
    return .{ .op = .addi, .len = 2, .rd = 2, .rs1 = 2, .imm = signExtend(raw, 10) };
}

fn cLui(h: u32, rd: u5) ?Decoded {
    const raw = bit(h, 12) << 17 | span(h, 6, 2, 12);
    if (raw == 0) return null;
    return .{ .op = .lui, .len = 2, .rd = rd, .imm = signExtend(raw, 18) };
}

fn miscAlu(h: u32) ?Decoded {
    const rd = smallReg(h, 7);
    switch (@as(u2, @truncate(h >> 10))) {
        0b00 => return .{ .op = .srli, .len = 2, .rd = rd, .rs1 = rd, .imm = shamtCi(h) },
        0b01 => return .{ .op = .srai, .len = 2, .rd = rd, .rs1 = rd, .imm = shamtCi(h) },
        0b10 => return .{ .op = .andi, .len = 2, .rd = rd, .rs1 = rd, .imm = immCi(h) },
        0b11 => {},
    }
    const op: Op = switch (bit(h, 12) == 1) {
        false => switch (@as(u2, @truncate(h >> 5))) {
            0b00 => .sub,
            0b01 => .xor,
            0b10 => .@"or",
            0b11 => .@"and",
        },
        true => switch (@as(u2, @truncate(h >> 5))) {
            0b00 => .subw,
            0b01 => .addw,
            0b10, 0b11 => return null,
        },
    };
    return .{ .op = op, .len = 2, .rd = rd, .rs1 = rd, .rs2 = smallReg(h, 2) };
}

fn jumpOffsetCj(h: u32) i64 {
    const raw = bit(h, 12) << 11 | bit(h, 11) << 4 | span(h, 10, 9, 8) | bit(h, 8) << 10 |
        bit(h, 7) << 6 | bit(h, 6) << 7 | span(h, 5, 3, 1) | bit(h, 2) << 5;
    return signExtend(raw, 12);
}

fn branchOffsetCb(h: u32) i64 {
    const raw = bit(h, 12) << 8 | span(h, 11, 10, 3) | span(h, 6, 5, 6) | span(h, 4, 3, 1) |
        bit(h, 2) << 5;
    return signExtend(raw, 9);
}

fn quadrant2(h: u32) ?Decoded {
    const rd = fullRd(h);
    const rs2 = fullRs2(h);
    return switch (@as(u3, @truncate(h >> 13))) {
        0b000 => .{ .op = .slli, .len = 2, .rd = rd, .rs1 = rd, .imm = shamtCi(h) },
        0b001 => .{ .op = .fld, .len = 2, .rd = rd, .rs1 = 2, .imm = doubleLoadOffsetCi(h) },
        0b010 => if (rd == 0) null else .{
            .op = .lw,
            .len = 2,
            .rd = rd,
            .rs1 = 2,
            .imm = span(h, 12, 12, 5) | span(h, 6, 4, 2) | span(h, 3, 2, 6),
        },
        0b011 => if (rd == 0) null else .{
            .op = .ld,
            .len = 2,
            .rd = rd,
            .rs1 = 2,
            .imm = doubleLoadOffsetCi(h),
        },
        0b100 => registerJump(h, rd, rs2),
        0b101 => .{ .op = .fsd, .len = 2, .rs1 = 2, .rs2 = rs2, .imm = doubleStoreOffsetCss(h) },
        0b110 => .{
            .op = .sw,
            .len = 2,
            .rs1 = 2,
            .rs2 = rs2,
            .imm = span(h, 12, 9, 2) | span(h, 8, 7, 6),
        },
        0b111 => .{ .op = .sd, .len = 2, .rs1 = 2, .rs2 = rs2, .imm = doubleStoreOffsetCss(h) },
    };
}

fn registerJump(h: u32, rd: u5, rs2: u5) ?Decoded {
    if (bit(h, 12) == 0) {
        if (rs2 != 0) return .{ .op = .add, .len = 2, .rd = rd, .rs1 = 0, .rs2 = rs2 };
        if (rd == 0) return null;
        return .{ .op = .jalr, .len = 2, .rd = 0, .rs1 = rd };
    }
    if (rs2 != 0) return .{ .op = .add, .len = 2, .rd = rd, .rs1 = rd, .rs2 = rs2 };
    if (rd == 0) return .{ .op = .ebreak, .len = 2 };
    return .{ .op = .jalr, .len = 2, .rd = 1, .rs1 = rd };
}

const opcode_load: u7 = 0b0000011;
const opcode_op_imm: u7 = 0b0010011;
const opcode_auipc: u7 = 0b0010111;
const opcode_op_imm_32: u7 = 0b0011011;
const opcode_store: u7 = 0b0100011;
const opcode_op: u7 = 0b0110011;
const opcode_lui: u7 = 0b0110111;
const opcode_op_32: u7 = 0b0111011;
const opcode_branch: u7 = 0b1100011;
const opcode_jalr: u7 = 0b1100111;
const opcode_jal: u7 = 0b1101111;

fn decode32(word: u32) Decoded {
    const unknown: Decoded = .{ .op = .unknown, .len = 4 };
    return switch (@as(u7, @truncate(word))) {
        opcode_lui => .{ .op = .lui, .len = 4, .rd = rdOf(word), .imm = uImm(word) },
        opcode_auipc => .{ .op = .auipc, .len = 4, .rd = rdOf(word), .imm = uImm(word) },
        opcode_jal => .{ .op = .jal, .len = 4, .rd = rdOf(word), .imm = jImm(word) },
        opcode_jalr => if (funct3(word) == 0)
            .{ .op = .jalr, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .imm = iImm(word) }
        else
            unknown,
        opcode_branch => branch(word) orelse unknown,
        opcode_load => load(word) orelse unknown,
        opcode_store => store(word) orelse unknown,
        opcode_op_imm => opImm(word) orelse unknown,
        opcode_op => opReg(word) orelse unknown,
        opcode_op_imm_32 => opImm32(word) orelse unknown,
        opcode_op_32 => opReg32(word) orelse unknown,
        else => unknown,
    };
}

fn branch(word: u32) ?Decoded {
    const op: Op = switch (funct3(word)) {
        0b000 => .beq,
        0b001 => .bne,
        0b100 => .blt,
        0b101 => .bge,
        0b110 => .bltu,
        0b111 => .bgeu,
        else => return null,
    };
    return .{ .op = op, .len = 4, .rs1 = rs1Of(word), .rs2 = rs2Of(word), .imm = bImm(word) };
}

fn load(word: u32) ?Decoded {
    const op: Op = switch (funct3(word)) {
        0b000 => .lb,
        0b001 => .lh,
        0b010 => .lw,
        0b011 => .ld,
        0b100 => .lbu,
        0b101 => .lhu,
        0b110 => .lwu,
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .imm = iImm(word) };
}

fn store(word: u32) ?Decoded {
    const op: Op = switch (funct3(word)) {
        0b000 => .sb,
        0b001 => .sh,
        0b010 => .sw,
        0b011 => .sd,
        else => return null,
    };
    return .{ .op = op, .len = 4, .rs1 = rs1Of(word), .rs2 = rs2Of(word), .imm = sImm(word) };
}

fn opImm(word: u32) ?Decoded {
    const shift_amount: i64 = (word >> 20) & 0x3F;
    const shift_kind = funct7(word) >> 1;
    const op: Op, const imm: i64 = switch (funct3(word)) {
        0b000 => .{ .addi, iImm(word) },
        0b010 => .{ .slti, iImm(word) },
        0b011 => .{ .sltiu, iImm(word) },
        0b100 => .{ .xori, iImm(word) },
        0b110 => .{ .ori, iImm(word) },
        0b111 => .{ .andi, iImm(word) },
        0b001 => if (shift_kind == 0b000000) .{ .slli, shift_amount } else return null,
        0b101 => switch (shift_kind) {
            0b000000 => .{ .srli, shift_amount },
            0b010000 => .{ .srai, shift_amount },
            else => return null,
        },
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .imm = imm };
}

fn opImm32(word: u32) ?Decoded {
    const shift_amount: i64 = rs2Of(word);
    const op: Op, const imm: i64 = switch (funct3(word)) {
        0b000 => .{ .addiw, iImm(word) },
        0b001 => if (funct7(word) == 0b0000000) .{ .slliw, shift_amount } else return null,
        0b101 => switch (funct7(word)) {
            0b0000000 => .{ .srliw, shift_amount },
            0b0100000 => .{ .sraiw, shift_amount },
            else => return null,
        },
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .imm = imm };
}

fn opReg(word: u32) ?Decoded {
    const op: Op = switch (funct7(word)) {
        0b0000000 => switch (funct3(word)) {
            0b000 => .add,
            0b001 => .sll,
            0b010 => .slt,
            0b011 => .sltu,
            0b100 => .xor,
            0b101 => .srl,
            0b110 => .@"or",
            0b111 => .@"and",
        },
        0b0100000 => switch (funct3(word)) {
            0b000 => .sub,
            0b101 => .sra,
            else => return null,
        },
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .rs2 = rs2Of(word) };
}

fn opReg32(word: u32) ?Decoded {
    const op: Op = switch (funct7(word)) {
        0b0000000 => switch (funct3(word)) {
            0b000 => .addw,
            0b001 => .sllw,
            0b101 => .srlw,
            else => return null,
        },
        0b0100000 => switch (funct3(word)) {
            0b000 => .subw,
            0b101 => .sraw,
            else => return null,
        },
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .rs2 = rs2Of(word) };
}

fn rdOf(word: u32) u5 {
    return @truncate(word >> 7);
}

fn rs1Of(word: u32) u5 {
    return @truncate(word >> 15);
}

fn rs2Of(word: u32) u5 {
    return @truncate(word >> 20);
}

fn funct3(word: u32) u3 {
    return @truncate(word >> 12);
}

fn funct7(word: u32) u7 {
    return @truncate(word >> 25);
}

fn iImm(word: u32) i64 {
    return @as(i32, @bitCast(word)) >> 20;
}

fn sImm(word: u32) i64 {
    const raw = (word >> 25 << 5) | ((word >> 7) & 0x1F);
    return @as(i32, @bitCast(raw << 20)) >> 20;
}

fn bImm(word: u32) i64 {
    const raw = ((word >> 31) << 12) | (((word >> 7) & 1) << 11) |
        (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1);
    return @as(i32, @bitCast(raw << 19)) >> 19;
}

/// The value `lui` and `auipc` add: the 20 bit field already shifted left by twelve.
fn uImm(word: u32) i64 {
    return @as(i32, @bitCast(word & 0xFFFFF000));
}

fn jImm(word: u32) i64 {
    const raw = ((word >> 31) << 20) | (((word >> 12) & 0xFF) << 12) |
        (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1);
    return @as(i32, @bitCast(raw << 11)) >> 11;
}

fn wordBytes(value: u32) [4]u8 {
    var out: [4]u8 = undefined;
    std.mem.writeInt(u32, &out, value, .little);
    return out;
}

fn decodeWord(value: u32) !Decoded {
    return decode(&wordBytes(value));
}

/// What the oracle text says about one instruction, with pseudo instructions turned back into
/// the instruction they stand for.
const Expected = struct {
    op: Op,
    rd: u5 = 0,
    rs1: u5 = 0,
    rs2: u5 = 0,
    imm: i64 = 0,
};

const TextError = error{ UnknownMnemonic, BadOperand };

fn reg(value: i64) u5 {
    return @intCast(value);
}

/// Reads the text `disasm.one` renders. Operands are collected in the order they print.
fn parseText(text: []const u8) TextError!Expected {
    const space = std.mem.indexOfScalar(u8, text, ' ') orelse text.len;
    const mnemonic = text[0..space];
    var values: [3]i64 = @splat(0);
    var count: usize = 0;
    var tokens = std.mem.tokenizeAny(u8, text[space..], ", ()");
    while (tokens.next()) |token| : (count += 1) {
        if (count == values.len) return error.BadOperand;
        values[count] = if (token[0] == 'x' or token[0] == 'f')
            std.fmt.parseInt(i64, token[1..], 10) catch return error.BadOperand
        else if (token[0] == '.')
            std.fmt.parseInt(i64, token[1..], 10) catch return error.BadOperand
        else
            std.fmt.parseInt(i64, token, 0) catch return error.BadOperand;
    }
    const v = values;

    const Pseudo = struct { Op, [3]i64 };
    const pseudo: ?Pseudo = if (std.mem.eql(u8, mnemonic, "j"))
        .{ .jal, .{ 0, v[0], 0 } }
    else if (std.mem.eql(u8, mnemonic, "jal") and count == 1)
        .{ .jal, .{ 1, v[0], 0 } }
    else if (std.mem.eql(u8, mnemonic, "ret"))
        .{ .jalr, .{ 0, 0, 1 } }
    else if (std.mem.eql(u8, mnemonic, "jr"))
        .{ .jalr, .{ 0, 0, v[0] } }
    else if (std.mem.eql(u8, mnemonic, "jalr") and count == 1)
        .{ .jalr, .{ 1, 0, v[0] } }
    else if (std.mem.eql(u8, mnemonic, "li"))
        .{ .addi, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "mv"))
        .{ .addi, .{ v[0], v[1], 0 } }
    else if (std.mem.eql(u8, mnemonic, "not"))
        .{ .xori, .{ v[0], v[1], -1 } }
    else if (std.mem.eql(u8, mnemonic, "seqz"))
        .{ .sltiu, .{ v[0], v[1], 1 } }
    else if (std.mem.eql(u8, mnemonic, "neg"))
        .{ .sub, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "snez"))
        .{ .sltu, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "sext.w"))
        .{ .addiw, .{ v[0], v[1], 0 } }
    else if (std.mem.eql(u8, mnemonic, "negw"))
        .{ .subw, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "beqz"))
        .{ .beq, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "bnez"))
        .{ .bne, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "bltz"))
        .{ .blt, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "bgez"))
        .{ .bge, .{ v[0], 0, v[1] } }
    else if (std.mem.eql(u8, mnemonic, "blez"))
        .{ .bge, .{ 0, v[0], v[1] } }
    else if (std.mem.eql(u8, mnemonic, "bgtz"))
        .{ .blt, .{ 0, v[0], v[1] } }
    else
        null;

    const op: Op, const a = if (pseudo) |p|
        .{ p[0], p[1] }
    else
        .{ std.meta.stringToEnum(Op, mnemonic) orelse return error.UnknownMnemonic, v };

    return switch (op) {
        .lui, .auipc => .{
            .op = op,
            .rd = reg(a[0]),
            .imm = @as(i32, @bitCast(@as(u32, @intCast(a[1])) << 12)),
        },
        .jal => .{ .op = op, .rd = reg(a[0]), .imm = a[1] },
        .jalr, .lb, .lh, .lw, .ld, .lbu, .lhu, .lwu, .fld => .{
            .op = op,
            .rd = reg(a[0]),
            .imm = a[1],
            .rs1 = reg(a[2]),
        },
        .sb, .sh, .sw, .sd, .fsd => .{ .op = op, .rs2 = reg(a[0]), .imm = a[1], .rs1 = reg(a[2]) },
        .beq, .bne, .blt, .bge, .bltu, .bgeu => .{
            .op = op,
            .rs1 = reg(a[0]),
            .rs2 = reg(a[1]),
            .imm = a[2],
        },
        .addi, .slti, .sltiu, .xori, .ori, .andi, .slli, .srli, .srai => .{
            .op = op,
            .rd = reg(a[0]),
            .rs1 = reg(a[1]),
            .imm = a[2],
        },
        .addiw, .slliw, .srliw, .sraiw => .{
            .op = op,
            .rd = reg(a[0]),
            .rs1 = reg(a[1]),
            .imm = a[2],
        },
        .add, .sub, .sll, .slt, .sltu, .xor, .@"or", .@"and", .srl, .sra => .{
            .op = op,
            .rd = reg(a[0]),
            .rs1 = reg(a[1]),
            .rs2 = reg(a[2]),
        },
        .addw, .subw, .sllw, .srlw, .sraw => .{
            .op = op,
            .rd = reg(a[0]),
            .rs1 = reg(a[1]),
            .rs2 = reg(a[2]),
        },
        else => error.UnknownMnemonic,
    };
}

/// Decodes `value` and checks it against what `disasm.one` renders for it.
fn expectAgreesWithDisasm(value: u32) !void {
    const text = try disasm.one(std.testing.allocator, value);
    defer std.testing.allocator.free(text);
    const expected = parseText(text) catch |err| {
        std.debug.print("word 0x{x:0>8} renders as \"{s}\"\n", .{ value, text });
        return err;
    };
    const got = try decodeWord(value);
    errdefer std.debug.print("word 0x{x:0>8} renders as \"{s}\"\n", .{ value, text });
    try std.testing.expectEqual(Decoded{
        .op = expected.op,
        .len = 4,
        .rd = expected.rd,
        .rs1 = expected.rs1,
        .rs2 = expected.rs2,
        .imm = expected.imm,
    }, got);
}

fn rType(f7: u7, f3: u3, opcode: u7, rd: u5, rs1: u5, rs2: u5) u32 {
    return @as(u32, opcode) | (@as(u32, rd) << 7) | (@as(u32, f3) << 12) |
        (@as(u32, rs1) << 15) | (@as(u32, rs2) << 20) | (@as(u32, f7) << 25);
}

fn iType(f3: u3, opcode: u7, rd: u5, rs1: u5, imm: i12) u32 {
    return @as(u32, opcode) | (@as(u32, rd) << 7) | (@as(u32, f3) << 12) |
        (@as(u32, rs1) << 15) | (@as(u32, @as(u12, @bitCast(imm))) << 20);
}

const Reg = encode.Reg;
const reg_triples = [_][3]Reg{
    .{ .x1, .x2, .x3 },
    .{ .x0, .x5, .x6 },
    .{ .x7, .x0, .x8 },
    .{ .x9, .x10, .x0 },
    .{ .x31, .x30, .x29 },
    .{ .x0, .x0, .x0 },
    .{ .x1, .x1, .x1 },
};
const imms_12 = [_]i12{ 0, 1, -1, 5, -5, 1, 2047, -2048, 0x555, -0x556 };

test "register operations agree with the disassembler" {
    inline for (.{
        encode.add, encode.sub, encode.and_, encode.or_, encode.xor_,
        encode.sll, encode.srl, encode.sra,  encode.slt, encode.sltu,
    }) |emit| {
        for (reg_triples) |t| try expectAgreesWithDisasm(emit(t[0], t[1], t[2]));
    }
}

test "immediate operations agree with the disassembler" {
    inline for (.{ encode.addi, encode.xori, encode.sltiu, encode.andi, encode.ori }) |emit| {
        for (reg_triples) |t| {
            for (imms_12) |imm| try expectAgreesWithDisasm(emit(t[0], t[1], imm));
        }
    }
}

test "immediate shifts agree with the disassembler" {
    inline for (.{ encode.slli, encode.srli, encode.srai }) |emit| {
        for (reg_triples) |t| {
            for ([_]u6{ 0, 1, 31, 32, 63 }) |shamt| try expectAgreesWithDisasm(emit(t[0], t[1], shamt));
        }
    }
}

test "loads and jalr agree with the disassembler" {
    inline for (.{
        encode.lb, encode.lh, encode.lw, encode.ld, encode.lbu, encode.lhu, encode.lwu, encode.jalr,
    }) |emit| {
        for (reg_triples) |t| {
            for (imms_12) |imm| try expectAgreesWithDisasm(emit(t[0], t[1], imm));
        }
    }
}

test "stores agree with the disassembler" {
    inline for (.{ encode.sb, encode.sh, encode.sw, encode.sd }) |emit| {
        for (reg_triples) |t| {
            for (imms_12) |imm| try expectAgreesWithDisasm(emit(t[0], t[1], imm));
        }
    }
}

test "branches agree with the disassembler" {
    inline for (.{ encode.beq, encode.bne, encode.blt, encode.bge, encode.bltu, encode.bgeu }) |emit| {
        for (reg_triples) |t| {
            if (t[0] == .x0 or t[1] == .x0) continue;
            for ([_]i13{ 0, 2, -2, 4, -4, 2046, -2048, 4094, -4096, 0x555 * 2, -0x556 * 2 }) |imm| {
                try expectAgreesWithDisasm(emit(t[0], t[1], imm));
            }
        }
    }
}

test "a branch against the zero register keeps its operands in order" {
    // The disassembler prints `beqz x5` for a zero compare whichever side the zero is on, so it
    // cannot say which operand held it. These words name the order instead, which is what catches
    // a decode that swaps the two sources.
    const forward = try decodeWord(encode.beq(.x0, .x5, 8));
    try std.testing.expectEqual(@as(u5, 0), forward.rs1);
    try std.testing.expectEqual(@as(u5, 5), forward.rs2);

    const backward = try decodeWord(encode.beq(.x5, .x0, 8));
    try std.testing.expectEqual(@as(u5, 5), backward.rs1);
    try std.testing.expectEqual(@as(u5, 0), backward.rs2);

    const unequal = try decodeWord(encode.bne(.x0, .x7, -4));
    try std.testing.expectEqual(Op.bne, unequal.op);
    try std.testing.expectEqual(@as(u5, 0), unequal.rs1);
    try std.testing.expectEqual(@as(u5, 7), unequal.rs2);
    try std.testing.expectEqual(@as(i64, -4), unequal.imm);
}

test "jal agrees with the disassembler" {
    for ([_]Reg{ .x0, .x1, .x5, .x31 }) |rd| {
        for ([_]i21{ 0, 2, -2, 4, -4, 2046, -2048, 0xFFFFE, -0x100000, 0x55554, -0x55556 }) |imm| {
            try expectAgreesWithDisasm(encode.jal(rd, imm));
        }
    }
}

test "lui and auipc agree with the disassembler" {
    inline for (.{ encode.lui, encode.auipc }) |emit| {
        for ([_]Reg{ .x0, .x1, .x31 }) |rd| {
            for ([_]u20{ 0, 1, 0x7FFFF, 0x80000, 0xFFFFF, 0x12345 }) |imm| {
                try expectAgreesWithDisasm(emit(rd, imm));
            }
        }
    }
}

// The encoder has no emitter for these, so the words are built here and the disassembler still
// supplies the expected result.
test "set less than immediate and the word operations agree with the disassembler" {
    const op_imm: u7 = opcode_op_imm;
    const op_imm_32: u7 = opcode_op_imm_32;
    const op_32: u7 = opcode_op_32;
    for (reg_triples) |t| {
        const rd: u5 = @backingInt(t[0]);
        const rs1: u5 = @backingInt(t[1]);
        const rs2: u5 = @backingInt(t[2]);
        for (imms_12) |imm| {
            try expectAgreesWithDisasm(iType(0b010, op_imm, rd, rs1, imm));
            try expectAgreesWithDisasm(iType(0b000, op_imm_32, rd, rs1, imm));
        }
        for ([_]u5{ 0, 1, 16, 31 }) |shamt| {
            try expectAgreesWithDisasm(rType(0b0000000, 0b001, op_imm_32, rd, rs1, shamt));
            try expectAgreesWithDisasm(rType(0b0000000, 0b101, op_imm_32, rd, rs1, shamt));
            try expectAgreesWithDisasm(rType(0b0100000, 0b101, op_imm_32, rd, rs1, shamt));
        }
        try expectAgreesWithDisasm(rType(0b0000000, 0b000, op_32, rd, rs1, rs2));
        try expectAgreesWithDisasm(rType(0b0100000, 0b000, op_32, rd, rs1, rs2));
        try expectAgreesWithDisasm(rType(0b0000000, 0b001, op_32, rd, rs1, rs2));
        try expectAgreesWithDisasm(rType(0b0000000, 0b101, op_32, rd, rs1, rs2));
        try expectAgreesWithDisasm(rType(0b0100000, 0b101, op_32, rd, rs1, rs2));
    }
}

test "known words decode to their fields" {
    try std.testing.expectEqual(
        Decoded{ .op = .addi, .len = 4, .rd = 10, .rs1 = 0, .imm = 5 },
        try decodeWord(0x00500513),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .addi, .len = 4, .rd = 1, .rs1 = 2, .imm = -1 },
        try decodeWord(0xFFF10093),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .jalr, .len = 4, .rd = 0, .rs1 = 1 },
        try decodeWord(0x00008067),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .sd, .len = 4, .rs1 = 2, .rs2 = 5, .imm = 8 },
        try decodeWord(0x00513423),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .lui, .len = 4, .rd = 5, .imm = -0x80000000 },
        try decodeWord(0x800002B7),
    );
}

test "shift amounts are unsigned and sized by the instruction" {
    const slli = try decodeWord(encode.slli(.x1, .x2, 63));
    try std.testing.expectEqual(@as(i64, 63), slli.imm);
    const srai = try decodeWord(encode.srai(.x1, .x2, 32));
    try std.testing.expectEqual(Op.srai, srai.op);
    try std.testing.expectEqual(@as(i64, 32), srai.imm);
    const sraiw = try decodeWord(rType(0b0100000, 0b101, opcode_op_imm_32, 1, 2, 31));
    try std.testing.expectEqual(Op.sraiw, sraiw.op);
    try std.testing.expectEqual(@as(i64, 31), sraiw.imm);
}

test "reserved encodings in the covered opcodes decode as unknown" {
    const reserved = [_]u32{
        iType(0b001, opcode_jalr, 1, 2, 0),
        iType(0b111, opcode_load, 1, 2, 0),
        iType(0b100, opcode_store, 1, 2, 0),
        rType(0b0000000, 0b010, opcode_branch, 1, 2, 3),
        rType(0b0000010, 0b001, opcode_op_imm, 1, 2, 3),
        rType(0b0100000, 0b001, opcode_op_imm, 1, 2, 3),
        rType(0b0000000, 0b010, opcode_op_imm_32, 1, 2, 3),
        rType(0b0100000, 0b001, opcode_op_32, 1, 2, 3),
        rType(0b0100000, 0b100, opcode_op, 1, 2, 3),
    };
    for (reserved) |value| {
        const got = try decodeWord(value);
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, got);
    }
}

test "an opcode outside the table decodes as unknown with a length of four" {
    const got = try decodeWord(0x0000000B);
    try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, got);
}

test "a compressed halfword reports a length of two" {
    for ([_][]const u8{ &.{ 0x01, 0x00 }, &.{ 0x00, 0x00 }, &.{ 0x82, 0x80, 0xFF } }) |bytes| {
        const got = try decode(bytes);
        try std.testing.expectEqual(@as(u3, 2), got.len);
    }
}

test "every covered word reports a length of four" {
    const got = try decode(&wordBytes(encode.addi(.x1, .x2, 3)));
    try std.testing.expectEqual(@as(u3, 4), got.len);
}

test "a slice shorter than the instruction is truncated" {
    const wide = wordBytes(encode.addi(.x1, .x2, 3));
    try std.testing.expectError(error.Truncated, decode(&.{}));
    try std.testing.expectError(error.Truncated, decode(wide[0..1]));
    try std.testing.expectError(error.Truncated, decode(wide[0..2]));
    try std.testing.expectError(error.Truncated, decode(wide[0..3]));
    try std.testing.expectError(error.Truncated, decode(&.{0x01}));
}

fn halfBytes(value: u16) [2]u8 {
    var out: [2]u8 = undefined;
    std.mem.writeInt(u16, &out, value, .little);
    return out;
}

fn decodeHalf(value: u16) !Decoded {
    return decode(&halfBytes(value));
}

/// The encodings the oracle renders as an instruction although the specification reserves them.
fn reservedCompressed(half: u16) bool {
    const h: u32 = half;
    const rd = (h >> 7) & 0x1F;
    const rs2 = (h >> 2) & 0x1F;
    const imm6 = ((h >> 12) & 1) << 5 | rs2;
    return switch (h & 3) {
        0b00 => switch (h >> 13) {
            0b000 => (h >> 5) & 0xFF == 0,
            // A later extension uses this slot. `zca` and `zcd` do not.
            0b100 => true,
            else => false,
        },
        0b01 => switch (h >> 13) {
            0b001 => rd == 0,
            // `c.subw` and `c.addw` use funct2 00 and 01. The other two values are reserved.
            0b100 => (h >> 10) & 7 == 0b111 and (h >> 6) & 1 == 1,
            0b011 => if (rd == 2)
                (h & 0x107C) == 0
            else
                imm6 == 0,
            else => false,
        },
        0b10 => switch (h >> 13) {
            0b010, 0b011 => rd == 0,
            0b100 => (h >> 12) & 1 == 0 and rs2 == 0 and rd == 0,
            else => false,
        },
        else => false,
    };
}

/// Decodes `half` and checks it against what `disasm.oneCompressed` renders for it.
fn expectCompressedAgreesWithDisasm(half: u16, unlisted: *usize) !void {
    const text = try disasm.oneCompressed(std.testing.allocator, half);
    defer std.testing.allocator.free(text);
    const got = try decodeHalf(half);
    errdefer std.debug.print("half 0x{x:0>4} renders as \"{s}\"\n", .{ half, text });

    if (text.len == 0 or got.op == .unknown) {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 2 }, got);
        if (!reservedCompressed(half)) {
            std.debug.print("unlisted reserved half 0x{x:0>4}\n", .{half});
            unlisted.* += 1;
        }
        return;
    }
    if (std.mem.eql(u8, text, "nop")) {
        // The oracle prints every `c.addi` on x0 as `nop`, whatever its immediate.
        const imm6: u32 = ((half >> 12) & 1) << 5 | ((half >> 2) & 0x1F);
        const want_imm: i64 = if (imm6 & 0x20 != 0) @as(i64, imm6) - 64 else imm6;
        try std.testing.expectEqual(Decoded{ .op = .addi, .len = 2, .imm = want_imm }, got);
        return;
    }
    if (std.mem.eql(u8, text, "ebreak")) {
        try std.testing.expectEqual(Decoded{ .op = .ebreak, .len = 2 }, got);
        return;
    }
    var expected = try parseText(text);
    if (std.mem.startsWith(u8, text, "mv ")) {
        // The specification expands `c.mv` to `add rd, x0, rs2`, and the oracle prints `mv`.
        const source = expected.rs1;
        expected = .{ .op = .add, .rd = expected.rd, .rs1 = 0, .rs2 = source };
    }
    try std.testing.expectEqual(Decoded{
        .op = expected.op,
        .len = 2,
        .rd = expected.rd,
        .rs1 = expected.rs1,
        .rs2 = expected.rs2,
        .imm = expected.imm,
    }, got);
}

/// What the immediate field of a `c.lui` holds, worked out from the specification rather than
/// from the disassembler, whose own rendering of this field drops one bit. Bit 12 carries
/// `nzimm[17]` and bits 6 to 2 carry `nzimm[16:12]`, sign extended from bit 17.
fn expectedCluiField(half: u16) ?u32 {
    const hi: u32 = (half >> 12) & 1;
    const lo: u32 = (half >> 2) & 0x1F;
    const nzimm: u32 = (hi << 5) | lo;
    // A zero field is reserved, so there is no immediate to expect.
    if (nzimm == 0) return null;
    const sign: u32 = if (hi == 1) 0xFFFFC0 else 0;
    return (sign | nzimm) & 0xFFFFF;
}

/// Checks a `c.lui` halfword against the model. The decoder holds `imm` already shifted left by
/// twelve and sign extended, so the low twenty bits of `imm >> 12` are the model's field.
fn expectCluiAgreesWithModel(half: u16) !void {
    const got = try decodeHalf(half);
    errdefer std.debug.print("c.lui half 0x{x:0>4}\n", .{half});
    const field = expectedCluiField(half) orelse {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 2 }, got);
        return;
    };
    try std.testing.expectEqual(Op.lui, got.op);
    try std.testing.expectEqual(@as(u5, @intCast((half >> 7) & 0x1F)), got.rd);
    try std.testing.expectEqual(@as(u3, 2), got.len);
    try std.testing.expectEqual(@as(i64, @as(i32, @bitCast(field << 12))), got.imm);
}

test "every compressed halfword agrees with the disassembler" {
    var unlisted: usize = 0;
    var half: u32 = 0;
    while (half <= std.math.maxInt(u16)) : (half += 1) {
        if (half & 3 == 3) continue;
        const narrow: u16 = @intCast(half);
        // The oracle drops bit 16 of the `c.lui` immediate, so the model checks these instead.
        const rd = (half >> 7) & 0x1F;
        if (half >> 13 == 0b011 and half & 3 == 1 and rd != 2) {
            try expectCluiAgreesWithModel(narrow);
            continue;
        }
        try expectCompressedAgreesWithDisasm(narrow, &unlisted);
    }
    try std.testing.expectEqual(@as(usize, 0), unlisted);
}

/// Decodes a wide word and the halfword it compressed to, and checks they differ only in `len`.
fn expectSameAsWide(word: u32, half: u16) !void {
    var wide = try decodeWord(word);
    const narrow = try decodeHalf(half);
    errdefer std.debug.print("word 0x{x:0>8} and half 0x{x:0>4}\n", .{ word, half });
    // The specification expands `c.mv` to `add`, while the compressor also takes `addi rd, rs, 0`.
    if (wide.op == .addi and wide.imm == 0 and wide.rs1 != 0 and narrow.op == .add) {
        const source = wide.rs1;
        wide.op = .add;
        wide.rs1 = 0;
        wide.rs2 = source;
    }
    wide.len = 2;
    try std.testing.expectEqual(wide, narrow);
}

test "a compressed word decodes like the wide word it came from" {
    var compressed: usize = 0;
    inline for (.{ encode.addi, encode.andi }) |emit| {
        for (reg_triples) |t| {
            for ([_]i12{ 0, 1, -1, 31, -32, 32, -33, 16, -16, 496, -512, 508, 1020 }) |imm| {
                const word = emit(t[0], t[1], imm);
                const half = compress.tryCompress(word) orelse continue;
                try expectSameAsWide(word, half);
                compressed += 1;
            }
        }
    }
    inline for (.{ encode.slli, encode.srli, encode.srai }) |emit| {
        for (reg_triples) |t| {
            for ([_]u6{ 0, 1, 31, 32, 63 }) |shamt| {
                const word = emit(t[0], t[1], shamt);
                const half = compress.tryCompress(word) orelse continue;
                try expectSameAsWide(word, half);
                compressed += 1;
            }
        }
    }
    inline for (.{ encode.add, encode.sub, encode.xor_, encode.or_, encode.and_ }) |emit| {
        for (reg_triples) |t| {
            const word = emit(t[0], t[1], t[2]);
            const half = compress.tryCompress(word) orelse continue;
            try expectSameAsWide(word, half);
            compressed += 1;
        }
    }
    inline for (.{ encode.lw, encode.ld, encode.jalr }) |emit| {
        for (reg_triples) |t| {
            for ([_]i12{ 0, 4, 8, 24, 124, 128, 252, 256, 504, 508, 512, -8 }) |imm| {
                const word = emit(t[0], t[1], imm);
                const half = compress.tryCompress(word) orelse continue;
                try expectSameAsWide(word, half);
                compressed += 1;
            }
        }
    }
    inline for (.{ encode.sw, encode.sd }) |emit| {
        for (reg_triples) |t| {
            for ([_]i12{ 0, 4, 8, 24, 124, 128, 252, 256, 504, 508, 512, -8 }) |imm| {
                const word = emit(t[0], t[1], imm);
                const half = compress.tryCompress(word) orelse continue;
                try expectSameAsWide(word, half);
                compressed += 1;
            }
        }
    }
    try std.testing.expect(compressed > 50);
}

test "lui words decode like the compressed lui they came from" {
    for ([_]Reg{ .x1, .x5, .x31 }) |rd| {
        for ([_]u20{ 1, 15, 16, 31, 0xFFFE0, 0xFFFEF, 0xFFFFF }) |imm| {
            const word = encode.lui(rd, imm);
            try expectSameAsWide(word, compress.tryCompress(word).?);
        }
    }
}

test "control transfers keep their operands and recompute their offsets" {
    const words = [_]u32{
        encode.beq(.x8, .x0, 12),
        encode.addi(.x10, .x0, 1),
        encode.jal(.x0, 8),
        encode.addi(.x11, .x0, 2),
        encode.bne(.x9, .x0, -16),
        encode.jalr(.x0, .x1, 0),
        encode.jalr(.x1, .x5, 0),
    };
    const targets = [_]?usize{ 3, null, 4, null, 0, null, null };
    const bytes = try compress.compress(std.testing.allocator, &words);
    defer std.testing.allocator.free(bytes);

    var starts: [words.len + 1]usize = undefined;
    var at: usize = 0;
    for (0..words.len) |i| {
        starts[i] = at;
        at += (try decode(bytes[at..])).len;
    }
    starts[words.len] = at;
    try std.testing.expectEqual(bytes.len, at);

    for (words, targets, 0..) |word, target, i| {
        const wide = try decodeWord(word);
        var narrow = try decode(bytes[starts[i]..]);
        try std.testing.expectEqual(starts[i + 1] - starts[i], narrow.len);
        if (target) |t| {
            const offset: i64 = @intCast(starts[t]);
            try std.testing.expectEqual(offset - @as(i64, @intCast(starts[i])), narrow.imm);
            narrow.imm = wide.imm;
        }
        narrow.len = wide.len;
        try std.testing.expectEqual(wide, narrow);
    }
}

test "compressed encodings with scrambled immediates decode to known values" {
    const cases = [_]struct { half: u16, want: Decoded }{
        .{ .half = 0x1141, .want = .{ .op = .addi, .len = 2, .rd = 2, .rs1 = 2, .imm = -16 } },
        .{ .half = 0x6105, .want = .{ .op = .addi, .len = 2, .rd = 2, .rs1 = 2, .imm = 32 } },
        .{ .half = 0x0080, .want = .{ .op = .addi, .len = 2, .rd = 8, .rs1 = 2, .imm = 64 } },
        .{ .half = 0x4501, .want = .{ .op = .addi, .len = 2, .rd = 10, .rs1 = 0, .imm = 0 } },
        .{ .half = 0x2501, .want = .{ .op = .addiw, .len = 2, .rd = 10, .rs1 = 10, .imm = 0 } },
        .{ .half = 0x9001, .want = .{ .op = .srli, .len = 2, .rd = 8, .rs1 = 8, .imm = 32 } },
        .{ .half = 0xE406, .want = .{ .op = .sd, .len = 2, .rs1 = 2, .rs2 = 1, .imm = 8 } },
        .{ .half = 0x60A2, .want = .{ .op = .ld, .len = 2, .rd = 1, .rs1 = 2, .imm = 8 } },
        .{ .half = 0x8082, .want = .{ .op = .jalr, .len = 2, .rd = 0, .rs1 = 1 } },
        .{ .half = 0x9002, .want = .{ .op = .ebreak, .len = 2 } },
        .{ .half = 0x6041, .want = .{ .op = .lui, .len = 2, .rd = 0, .imm = 0x10000 } },
        .{ .half = 0x757D, .want = .{ .op = .lui, .len = 2, .rd = 10, .imm = -0x1000 } },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.want, try decodeHalf(case.half));
    }
}

test "a reserved compressed encoding decodes as unknown with a length of two" {
    // The all zero halfword, then `c.addiw` and `c.lwsp` on x0, `c.jr` on x0, and the reserved
    // `c.addi4spn`, `c.addi16sp` and `c.lui` immediates.
    for ([_]u16{ 0x0000, 0x2001, 0x4002, 0x8002, 0x0004, 0x6101, 0x6081, 0x9C61 }) |half| {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 2 }, try decodeHalf(half));
    }
}
