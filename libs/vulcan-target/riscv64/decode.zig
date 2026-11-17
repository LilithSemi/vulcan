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
    /// shift it again. `fence` holds instruction bits 31 to 20 unsigned: `fm` in bits 11 to 8,
    /// `pred` in bits 7 to 4 and `succ` in bits 3 to 0. `csrrwi`, `csrrsi` and `csrrci` hold their
    /// five bit unsigned value in `rs1`, because that is the field it occupies. The same `rs1` test then decides a write for all six.
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

/// Which register file each operand of an operation names. A float operation may read one file and
/// write the other, and `Decoded` holds a plain number for each operand, so this is how a caller
/// knows which file to read and which to write.
pub const Files = struct {
    pub const File = enum { integer, float };

    rd: File = .integer,
    rs1: File = .integer,
    rs2: File = .integer,
    rs3: File = .float,
};

/// The register file of each operand of `op`. A position the operation does not use still holds a
/// value, which comes from the arm of that operation. A caller does not read such a position, so
/// the value makes no claim. Only the fused operations use `rs3`, and it is float everywhere.
/// The switch has no `else` arm, so a new operation does not compile until it is placed here.
pub fn files(op: Op) Files {
    const float: Files = .{ .rd = .float, .rs1 = .float, .rs2 = .float };
    const float_to_integer: Files = .{ .rd = .integer, .rs1 = .float, .rs2 = .float };
    const integer_to_float: Files = .{ .rd = .float, .rs1 = .integer, .rs2 = .float };
    return switch (op) {
        .unknown,
        .lui,
        .auipc,
        .jal,
        .jalr,
        .beq,
        .bne,
        .blt,
        .bge,
        .bltu,
        .bgeu,
        .lb,
        .lh,
        .lw,
        .ld,
        .lbu,
        .lhu,
        .lwu,
        .sb,
        .sh,
        .sw,
        .sd,
        .addi,
        .slti,
        .sltiu,
        .xori,
        .ori,
        .andi,
        .slli,
        .srli,
        .srai,
        .add,
        .sub,
        .sll,
        .slt,
        .sltu,
        .xor,
        .@"or",
        .@"and",
        .srl,
        .sra,
        .addiw,
        .slliw,
        .srliw,
        .sraiw,
        .addw,
        .subw,
        .sllw,
        .srlw,
        .sraw,
        .ecall,
        .ebreak,
        .csrrw,
        .csrrs,
        .csrrc,
        .csrrwi,
        .csrrsi,
        .csrrci,
        .mret,
        .sret,
        .wfi,
        .fence,
        .fence_i,
        .sfence_vma,
        .mul,
        .mulh,
        .mulhsu,
        .mulhu,
        .div,
        .divu,
        .rem,
        .remu,
        .mulw,
        .divw,
        .divuw,
        .remw,
        .remuw,
        .lr_w,
        .sc_w,
        .amoswap_w,
        .amoadd_w,
        .amoxor_w,
        .amoand_w,
        .amoor_w,
        .amomin_w,
        .amomax_w,
        .amominu_w,
        .amomaxu_w,
        .lr_d,
        .sc_d,
        .amoswap_d,
        .amoadd_d,
        .amoxor_d,
        .amoand_d,
        .amoor_d,
        .amomin_d,
        .amomax_d,
        .amominu_d,
        .amomaxu_d,
        => .{},
        .flw, .fld => .{ .rd = .float },
        .fsw, .fsd => .{ .rs2 = .float },
        .fmadd_s,
        .fmsub_s,
        .fnmsub_s,
        .fnmadd_s,
        .fadd_s,
        .fsub_s,
        .fmul_s,
        .fdiv_s,
        .fsqrt_s,
        .fsgnj_s,
        .fsgnjn_s,
        .fsgnjx_s,
        .fmin_s,
        .fmax_s,
        .fmadd_d,
        .fmsub_d,
        .fnmsub_d,
        .fnmadd_d,
        .fadd_d,
        .fsub_d,
        .fmul_d,
        .fdiv_d,
        .fsqrt_d,
        .fsgnj_d,
        .fsgnjn_d,
        .fsgnjx_d,
        .fmin_d,
        .fmax_d,
        .fcvt_s_d,
        .fcvt_d_s,
        => float,
        .fcvt_w_s,
        .fcvt_wu_s,
        .fcvt_l_s,
        .fcvt_lu_s,
        .fmv_x_w,
        .feq_s,
        .flt_s,
        .fle_s,
        .fclass_s,
        .feq_d,
        .flt_d,
        .fle_d,
        .fclass_d,
        .fcvt_w_d,
        .fcvt_wu_d,
        .fcvt_l_d,
        .fcvt_lu_d,
        .fmv_x_d,
        => float_to_integer,
        .fcvt_s_w,
        .fcvt_s_wu,
        .fcvt_s_l,
        .fcvt_s_lu,
        .fmv_w_x,
        .fcvt_d_w,
        .fcvt_d_wu,
        .fcvt_d_l,
        .fcvt_d_lu,
        .fmv_d_x,
        => integer_to_float,
    };
}

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
    const rd = rdOf(h);
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
    const rd = rdOf(h);
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
const opcode_misc_mem: u7 = 0b0001111;
const opcode_amo: u7 = 0b0101111;
const opcode_system: u7 = 0b1110011;
const opcode_load_fp: u7 = 0b0000111;
const opcode_store_fp: u7 = 0b0100111;
const opcode_fmadd: u7 = 0b1000011;
const opcode_fmsub: u7 = 0b1000111;
const opcode_fnmsub: u7 = 0b1001011;
const opcode_fnmadd: u7 = 0b1001111;
const opcode_op_fp: u7 = 0b1010011;

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
        opcode_misc_mem => miscMem(word) orelse unknown,
        opcode_system => system(word) orelse unknown,
        opcode_amo => amo(word) orelse unknown,
        opcode_load_fp => loadFp(word) orelse unknown,
        opcode_store_fp => storeFp(word) orelse unknown,
        opcode_fmadd, opcode_fmsub, opcode_fnmsub, opcode_fnmadd => fused(word) orelse unknown,
        opcode_op_fp => opFp(word) orelse unknown,
        else => unknown,
    };
}

/// `fence` keeps its twelve immediate bits unsigned in `imm`: `fm` in bits 11 to 8, the
/// predecessor set in bits 7 to 4 and the successor set in bits 3 to 0. The register fields are
/// reserved and a reader ignores them, so they stay zero. `fence.i` ignores `imm`, `rs1` and `rd`
/// in the same way, because a later extension may define them.
/// A reserved `fm` passes through in `imm`, although a float operation with a reserved rounding mode
/// is rejected. The float chapter makes a reserved rounding mode an illegal instruction. This
/// chapter lets an implementation treat a reserved `fm` as a plain fence.
fn miscMem(word: u32) ?Decoded {
    return switch (funct3(word)) {
        0b000 => .{ .op = .fence, .len = 4, .imm = word >> 20 },
        0b001 => .{ .op = .fence_i, .len = 4 },
        else => null,
    };
}

/// The immediate forms of the CSR operations hold their five bit unsigned value in `rs1`, because
/// that is the field it occupies. A CSR write happens when `rs1` is non-zero, for the register forms
/// and for the immediate forms. The `csrrs`, `csrrc`, `csrrsi` and `csrrci` operations do not write
/// the CSR when `rs1` is zero.
fn system(word: u32) ?Decoded {
    const f3 = funct3(word);
    const op: Op = switch (f3) {
        0b001 => .csrrw,
        0b010 => .csrrs,
        0b011 => .csrrc,
        0b101 => .csrrwi,
        0b110 => .csrrsi,
        0b111 => .csrrci,
        0b000 => return privileged(word),
        0b100 => return null,
    };
    const csr: u12 = @truncate(word >> 20);
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .csr = csr };
}

fn privileged(word: u32) ?Decoded {
    if (funct7(word) == 0b0001001 and rdOf(word) == 0) {
        return .{ .op = .sfence_vma, .len = 4, .rs1 = rs1Of(word), .rs2 = rs2Of(word) };
    }
    const op: Op = switch (word) {
        0x00000073 => .ecall,
        0x00100073 => .ebreak,
        0x10200073 => .sret,
        0x30200073 => .mret,
        0x10500073 => .wfi,
        else => return null,
    };
    return .{ .op = op, .len = 4 };
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
        0b0000001 => switch (funct3(word)) {
            0b000 => .mul,
            0b001 => .mulh,
            0b010 => .mulhsu,
            0b011 => .mulhu,
            0b100 => .div,
            0b101 => .divu,
            0b110 => .rem,
            0b111 => .remu,
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
        0b0000001 => switch (funct3(word)) {
            0b000 => .mulw,
            0b100 => .divw,
            0b101 => .divuw,
            0b110 => .remw,
            0b111 => .remuw,
            else => return null,
        },
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .rs2 = rs2Of(word) };
}

/// The atomics. Bit 26 is `aq` and bit 25 is `rl`. `lr` has no second source and the
/// specification fixes its `rs2` field at zero, so a nonzero field is not an `lr`. The other
/// operations take their `funct5` from bits 31 to 27 and the width from `funct3`.
fn amo(word: u32) ?Decoded {
    const double = switch (funct3(word)) {
        0b010 => false,
        0b011 => true,
        else => return null,
    };
    const op: Op = switch (@as(u5, @truncate(word >> 27))) {
        0b00010 => blk: {
            if (rs2Of(word) != 0) return null;
            break :blk if (double) .lr_d else .lr_w;
        },
        0b00011 => if (double) .sc_d else .sc_w,
        0b00001 => if (double) .amoswap_d else .amoswap_w,
        0b00000 => if (double) .amoadd_d else .amoadd_w,
        0b00100 => if (double) .amoxor_d else .amoxor_w,
        0b01100 => if (double) .amoand_d else .amoand_w,
        0b01000 => if (double) .amoor_d else .amoor_w,
        0b10000 => if (double) .amomin_d else .amomin_w,
        0b10100 => if (double) .amomax_d else .amomax_w,
        0b11000 => if (double) .amominu_d else .amominu_w,
        0b11100 => if (double) .amomaxu_d else .amomaxu_w,
        else => return null,
    };
    return .{
        .op = op,
        .len = 4,
        .rd = rdOf(word),
        .rs1 = rs1Of(word),
        .rs2 = rs2Of(word),
        .aq = (word >> 26) & 1 != 0,
        .rl = (word >> 25) & 1 != 0,
    };
}

fn loadFp(word: u32) ?Decoded {
    const op: Op = switch (funct3(word)) {
        0b010 => .flw,
        0b011 => .fld,
        else => return null,
    };
    return .{ .op = op, .len = 4, .rd = rdOf(word), .rs1 = rs1Of(word), .imm = iImm(word) };
}

fn storeFp(word: u32) ?Decoded {
    const op: Op = switch (funct3(word)) {
        0b010 => .fsw,
        0b011 => .fsd,
        else => return null,
    };
    return .{ .op = op, .len = 4, .rs1 = rs1Of(word), .rs2 = rs2Of(word), .imm = sImm(word) };
}

/// The rounding mode in bits 14 to 12. Values 5 and 6 are reserved and the specification makes
/// an instruction that names one illegal, so the decoder reports such a word as unknown. Value 7
/// means the mode in `frm`, which only a running hart knows, so it passes through.
fn roundingMode(word: u32) ?u3 {
    const mode = funct3(word);
    if (mode == 0b101 or mode == 0b110) return null;
    return mode;
}

/// The four fused multiply add forms. Bits 26 and 25 pick single or double precision and bits 31
/// to 27 hold the third source.
fn fused(word: u32) ?Decoded {
    const double = switch (@as(u2, @truncate(word >> 25))) {
        0b00 => false,
        0b01 => true,
        else => return null,
    };
    const op: Op = switch (@as(u7, @truncate(word))) {
        opcode_fmadd => if (double) .fmadd_d else .fmadd_s,
        opcode_fmsub => if (double) .fmsub_d else .fmsub_s,
        opcode_fnmsub => if (double) .fnmsub_d else .fnmsub_s,
        opcode_fnmadd => if (double) .fnmadd_d else .fnmadd_s,
        else => unreachable,
    };
    return .{
        .op = op,
        .len = 4,
        .rd = rdOf(word),
        .rs1 = rs1Of(word),
        .rs2 = rs2Of(word),
        .rs3 = @truncate(word >> 27),
        .rm = roundingMode(word) orelse return null,
    };
}

/// Which fields of an OP-FP word a form reads. Bits 14 to 12 are a rounding mode for some forms
/// and part of the operation for the others, and `rm` stays zero for the others.
const FpForm = enum { binary_rm, unary_rm, binary, unary };

fn fpForm(word: u32, op: Op, form: FpForm) ?Decoded {
    const has_rm = form == .binary_rm or form == .unary_rm;
    const has_rs2 = form == .binary_rm or form == .binary;
    return .{
        .op = op,
        .len = 4,
        .rd = rdOf(word),
        .rs1 = rs1Of(word),
        .rs2 = if (has_rs2) rs2Of(word) else 0,
        .rm = if (has_rm) roundingMode(word) orelse return null else 0,
    };
}

/// OP-FP. `funct7` splits into `funct5` in bits 31 to 27 and the format in bits 26 and 25. The
/// conversions name their operation in `rs2`. `fsqrt`, `fclass` and the moves have an `rs2`
/// field fixed at zero. The specification has no rule to ignore that field, so a word with any
/// other value is not one of these instructions. `rs2` stays zero in the result for all of them.
fn opFp(word: u32) ?Decoded {
    const double = switch (@as(u2, @truncate(word >> 25))) {
        0b00 => false,
        0b01 => true,
        else => return null,
    };
    const rs2 = rs2Of(word);
    const f3 = funct3(word);
    switch (@as(u5, @truncate(word >> 27))) {
        0b00000 => return fpForm(word, if (double) .fadd_d else .fadd_s, .binary_rm),
        0b00001 => return fpForm(word, if (double) .fsub_d else .fsub_s, .binary_rm),
        0b00010 => return fpForm(word, if (double) .fmul_d else .fmul_s, .binary_rm),
        0b00011 => return fpForm(word, if (double) .fdiv_d else .fdiv_s, .binary_rm),
        0b01011 => {
            if (rs2 != 0) return null;
            return fpForm(word, if (double) .fsqrt_d else .fsqrt_s, .unary_rm);
        },
        0b00100 => {
            const op: Op = switch (f3) {
                0b000 => if (double) .fsgnj_d else .fsgnj_s,
                0b001 => if (double) .fsgnjn_d else .fsgnjn_s,
                0b010 => if (double) .fsgnjx_d else .fsgnjx_s,
                else => return null,
            };
            return fpForm(word, op, .binary);
        },
        0b00101 => {
            const op: Op = switch (f3) {
                0b000 => if (double) .fmin_d else .fmin_s,
                0b001 => if (double) .fmax_d else .fmax_s,
                else => return null,
            };
            return fpForm(word, op, .binary);
        },
        0b01000 => {
            const op: Op = switch (rs2) {
                0b00001 => if (double) return null else .fcvt_s_d,
                0b00000 => if (double) .fcvt_d_s else return null,
                else => return null,
            };
            return fpForm(word, op, .unary_rm);
        },
        0b10100 => {
            const op: Op = switch (f3) {
                0b010 => if (double) .feq_d else .feq_s,
                0b001 => if (double) .flt_d else .flt_s,
                0b000 => if (double) .fle_d else .fle_s,
                else => return null,
            };
            return fpForm(word, op, .binary);
        },
        0b11000 => {
            const op: Op = switch (rs2) {
                0 => if (double) .fcvt_w_d else .fcvt_w_s,
                1 => if (double) .fcvt_wu_d else .fcvt_wu_s,
                2 => if (double) .fcvt_l_d else .fcvt_l_s,
                3 => if (double) .fcvt_lu_d else .fcvt_lu_s,
                else => return null,
            };
            return fpForm(word, op, .unary_rm);
        },
        0b11010 => {
            const op: Op = switch (rs2) {
                0 => if (double) .fcvt_d_w else .fcvt_s_w,
                1 => if (double) .fcvt_d_wu else .fcvt_s_wu,
                2 => if (double) .fcvt_d_l else .fcvt_s_l,
                3 => if (double) .fcvt_d_lu else .fcvt_s_lu,
                else => return null,
            };
            return fpForm(word, op, .unary_rm);
        },
        0b11100 => {
            if (rs2 != 0) return null;
            const op: Op = switch (f3) {
                0b000 => if (double) .fmv_x_d else .fmv_x_w,
                0b001 => if (double) .fclass_d else .fclass_s,
                else => return null,
            };
            return fpForm(word, op, .unary);
        },
        0b11110 => {
            if (rs2 != 0 or f3 != 0) return null;
            return fpForm(word, if (double) .fmv_d_x else .fmv_w_x, .unary);
        },
        else => return null,
    }
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
    return signExtend(raw, 12);
}

fn bImm(word: u32) i64 {
    const raw = ((word >> 31) << 12) | (((word >> 7) & 1) << 11) |
        (((word >> 25) & 0x3F) << 5) | (((word >> 8) & 0xF) << 1);
    return signExtend(raw, 13);
}

/// The value `lui` and `auipc` add: the 20 bit field already shifted left by twelve.
fn uImm(word: u32) i64 {
    return @as(i32, @bitCast(word & 0xFFFFF000));
}

fn jImm(word: u32) i64 {
    const raw = ((word >> 31) << 20) | (((word >> 12) & 0xFF) << 12) |
        (((word >> 20) & 1) << 11) | (((word >> 21) & 0x3FF) << 1);
    return signExtend(raw, 21);
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
/// from the disassembler. Bit 12 carries `nzimm[17]` and bits 6 to 2 carry `nzimm[16:12]`,
/// sign extended from bit 17.
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
        // The specification is the authority for the `c.lui` immediate. The oracle gets a
        // second look at the same halfwords below.
        const rd = (half >> 7) & 0x1F;
        if (half >> 13 == 0b011 and half & 3 == 1 and rd != 2) {
            try expectCluiAgreesWithModel(narrow);
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

fn csrWord(csr: u12, f3: u3, rd: u5, rs1: u5) u32 {
    return iType(f3, opcode_system, rd, rs1, @bitCast(csr));
}

test "ecall and ebreak differ in bit 20 alone" {
    // I type, opcode 1110011, funct3 000, rd and rs1 zero, imm[11:0] 0 for ecall and 1 for ebreak.
    try std.testing.expectEqual(Decoded{ .op = .ecall, .len = 4 }, try decodeWord(0x00000073));
    try std.testing.expectEqual(Decoded{ .op = .ebreak, .len = 4 }, try decodeWord(0x00100073));
    try std.testing.expectEqual(@as(u32, 1) << 20, 0x00000073 ^ 0x00100073);
}

test "mret, sret and wfi decode as whole words" {
    // imm[11:0] is 0011000 00010 for mret, 0001000 00010 for sret and 0001000 00101 for wfi.
    // Then rs1 00000, funct3 000, rd 00000 and opcode 1110011.
    try std.testing.expectEqual(Decoded{ .op = .mret, .len = 4 }, try decodeWord(0x30200073));
    try std.testing.expectEqual(Decoded{ .op = .sret, .len = 4 }, try decodeWord(0x10200073));
    try std.testing.expectEqual(Decoded{ .op = .wfi, .len = 4 }, try decodeWord(0x10500073));
}

test "privileged lookalikes with another field set decode as unknown" {
    // rd = x1 on ecall, mret, sret and wfi, then rs1 = x1 on mret, sret and wfi, then three
    // immediates that no instruction uses.
    const words = [_]u32{
        0x000000F3, 0x302000F3, 0x102000F3, 0x105000F3,
        0x30208073, 0x10208073, 0x10508073, 0x30300073,
        0x10600073, 0x20200073,
    };
    for (words) |word| {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(word));
    }
}

test "a system word with funct3 100 decodes as unknown" {
    try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(0x00004073));
    try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(0x30004573));
}

test "the six csr operations read the csr number and the source" {
    // csr[11:0] in bits 31 to 20, rs1 in 19 to 15, funct3 in 14 to 12, rd in 11 to 7, opcode
    // 1110011. Here csr 0x305 (mtvec), rd x5 and rs1 x6, so the base is 0x30500000 | 0x30000 |
    // 0x280 | 0x73 = 0x305302F3 and funct3 adds 0x1000 per step.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrw, .len = 4, .rd = 5, .rs1 = 6, .csr = 0x305 },
        try decodeWord(0x305312F3),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .csrrs, .len = 4, .rd = 5, .rs1 = 6, .csr = 0x305 },
        try decodeWord(0x305322F3),
    );
    try std.testing.expectEqual(
        Decoded{ .op = .csrrc, .len = 4, .rd = 5, .rs1 = 6, .csr = 0x305 },
        try decodeWord(0x305332F3),
    );
    // The immediate forms take funct3 101, 110 and 111, and bits 19 to 15 are a five bit
    // unsigned value. Here the value is 31, so bits 19 to 15 are 0xF8000.
    // csrrwi x7, mip 0x344, 31: 0x34400000 | 0xF8000 | 0x5000 | 0x380 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrwi, .len = 4, .rd = 7, .rs1 = 31, .csr = 0x344 },
        try decodeWord(0x344FD3F3),
    );
    // csrrsi x7, mepc 0x341, 1: 0x34100000 | 0x8000 | 0x6000 | 0x380 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrsi, .len = 4, .rd = 7, .rs1 = 1, .csr = 0x341 },
        try decodeWord(0x3410E3F3),
    );
    // csrrci x7, mcause 0x342, 2: 0x34200000 | 0x10000 | 0x7000 | 0x380 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrci, .len = 4, .rd = 7, .rs1 = 2, .csr = 0x342 },
        try decodeWord(0x342173F3),
    );
}

test "an immediate csr form carries its value in rs1 so one test decides a write" {
    // csrsi mstatus, 8 is 0x30046073. A non-zero rs1 tells a caller the CSR is written, for the
    // immediate forms as much as for the register ones.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrsi, .len = 4, .rd = 0, .rs1 = 8, .csr = 0x300 },
        try decodeWord(0x30046073),
    );
}

test "the csr numbers the firmware uses decode from bits 31 to 20" {
    // Numbers from the privileged specification: mstatus 0x300, mie 0x304, mtvec 0x305,
    // mepc 0x341, mcause 0x342, mip 0x344, satp 0x180 and stimecmp 0x14D. Each word is
    // `csrrw x0, csr, x5`, which is csr << 20 | 0x28000 | 0x1000 | 0x73.
    const cases = [_]struct { word: u32, csr: u12 }{
        .{ .word = 0x30029073, .csr = 0x300 },
        .{ .word = 0x30429073, .csr = 0x304 },
        .{ .word = 0x30529073, .csr = 0x305 },
        .{ .word = 0x34129073, .csr = 0x341 },
        .{ .word = 0x34229073, .csr = 0x342 },
        .{ .word = 0x34429073, .csr = 0x344 },
        .{ .word = 0x18029073, .csr = 0x180 },
        .{ .word = 0x14D29073, .csr = 0x14D },
    };
    for (cases) |case| {
        try std.testing.expectEqual(
            Decoded{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 5, .csr = case.csr },
            try decodeWord(case.word),
        );
        try std.testing.expectEqual(case.word, csrWord(case.csr, 0b001, 0, 5));
    }
}

test "csr pseudo instructions decode as their real encodings" {
    // csrr a0, mstatus is csrrs x10, 0x300, x0: 0x30000000 | 0x2000 | 0x500 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrs, .len = 4, .rd = 10, .rs1 = 0, .csr = 0x300 },
        try decodeWord(0x30002573),
    );
    // csrw mtvec, t0 is csrrw x0, 0x305, x5: 0x30500000 | 0x28000 | 0x1000 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 5, .csr = 0x305 },
        try decodeWord(0x30529073),
    );
    // csrs mie, a0 is csrrs x0, 0x304, x10: 0x30400000 | 0x50000 | 0x2000 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrs, .len = 4, .rd = 0, .rs1 = 10, .csr = 0x304 },
        try decodeWord(0x30452073),
    );
    // csrci mstatus, 8 is csrrci x0, 0x300, 8: 0x30000000 | 0x40000 | 0x7000 | 0x73.
    try std.testing.expectEqual(
        Decoded{ .op = .csrrci, .len = 4, .rd = 0, .rs1 = 8, .csr = 0x300 },
        try decodeWord(0x30047073),
    );
}

test "fence keeps its fields in the immediate and fence.i is a separate word" {
    // imm[11:0] is fm[3:0] pred[3:0] succ[3:0], with the bits i, o, r and w from high to low.
    // Then rs1 00000, funct3 000, rd 00000 and opcode 0001111.
    // fence rw, rw: fm 0000, pred 0011, succ 0011.
    try std.testing.expectEqual(Decoded{ .op = .fence, .len = 4, .imm = 0x033 }, try decodeWord(0x0330000F));
    // fence iorw, iorw: pred 1111, succ 1111.
    try std.testing.expectEqual(Decoded{ .op = .fence, .len = 4, .imm = 0x0FF }, try decodeWord(0x0FF0000F));
    // fence w, r: pred 0001, succ 0010.
    try std.testing.expectEqual(Decoded{ .op = .fence, .len = 4, .imm = 0x012 }, try decodeWord(0x0120000F));
    // fence.tso: fm 1000, pred 0011, succ 0011. The field stays unsigned.
    try std.testing.expectEqual(Decoded{ .op = .fence, .len = 4, .imm = 0x833 }, try decodeWord(0x8330000F));
    // fence.i: funct3 001 and opcode 0001111. Its imm, rs1 and rd are ignored, because a later
    // extension may define them. The words set none of them, then imm, rs1 and rd in turn.
    for ([_]u32{ 0x0000100F, 0x0010100F, 0x0001900F, 0x0000108F }) |word| {
        try std.testing.expectEqual(Decoded{ .op = .fence_i, .len = 4 }, try decodeWord(word));
    }
    // Every funct3 other than 000 and 001 is reserved in this opcode. The words check 010 and 111.
    for ([_]u32{ 0x0000200F, 0x0000700F }) |word| {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(word));
    }
}

test "ecall and ebreak need every other field zero" {
    // The fence pair has a rule that says to ignore the reserved fields. These two have no such
    // rule. The specification fixes their other fields at zero, so a word that sets one is not
    // ecall or ebreak. Do not make them ignore those fields to match fence and fence.i.
    // The words are ebreak with rd 1, ebreak with rs1 1 and ecall with rs1 1.
    for ([_]u32{ 0x001000F3, 0x00108073, 0x00008073 }) |word| {
        try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(word));
    }
}

test "sfence.vma carries its two register operands" {
    // funct7 0001001, rs2, rs1, funct3 000, rd 00000, opcode 1110011.
    // sfence.vma x0, x0: 0x12000073.
    try std.testing.expectEqual(Decoded{ .op = .sfence_vma, .len = 4 }, try decodeWord(0x12000073));
    // sfence.vma a0, x0: rs1 x10 is 0x50000.
    try std.testing.expectEqual(
        Decoded{ .op = .sfence_vma, .len = 4, .rs1 = 10 },
        try decodeWord(0x12050073),
    );
    // sfence.vma a0, a1: rs2 x11 is 0xB00000.
    try std.testing.expectEqual(
        Decoded{ .op = .sfence_vma, .len = 4, .rs1 = 10, .rs2 = 11 },
        try decodeWord(0x12B50073),
    );
    // A nonzero rd is reserved.
    try std.testing.expectEqual(Decoded{ .op = .unknown, .len = 4 }, try decodeWord(0x120500F3));
}

test "words assembled by llvm decode to the expected fields" {
    // Each word comes from assembling the named instruction with
    // `zig cc --target=riscv64-linux-musl -c` and reading the .text bytes of the object file.
    // To regenerate a word, assemble the same line and dump the object with `llvm-objdump -d`.
    const cases = [_]struct { word: u32, want: Decoded }{
        // csrw mtvec, t0
        .{ .word = 0x30529073, .want = .{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 5, .csr = 0x305 } },
        // csrr a0, mstatus
        .{ .word = 0x30002573, .want = .{ .op = .csrrs, .len = 4, .rd = 10, .rs1 = 0, .csr = 0x300 } },
        // csrw mie, zero
        .{ .word = 0x30401073, .want = .{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 0, .csr = 0x304 } },
        // csrw mip, zero
        .{ .word = 0x34401073, .want = .{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 0, .csr = 0x344 } },
        // csrs mstatus, t1
        .{ .word = 0x30032073, .want = .{ .op = .csrrs, .len = 4, .rd = 0, .rs1 = 6, .csr = 0x300 } },
        // csrci mstatus, 8
        .{ .word = 0x30047073, .want = .{ .op = .csrrci, .len = 4, .rd = 0, .rs1 = 8, .csr = 0x300 } },
        // csrwi mtvec, 0
        .{ .word = 0x30505073, .want = .{ .op = .csrrwi, .len = 4, .rd = 0, .rs1 = 0, .csr = 0x305 } },
        // csrrw a1, mepc, a2
        .{ .word = 0x341615f3, .want = .{ .op = .csrrw, .len = 4, .rd = 11, .rs1 = 12, .csr = 0x341 } },
        // csrrc a3, mcause, a4
        .{ .word = 0x342736f3, .want = .{ .op = .csrrc, .len = 4, .rd = 13, .rs1 = 14, .csr = 0x342 } },
        // csrrsi a5, satp, 31
        .{ .word = 0x180fe7f3, .want = .{ .op = .csrrsi, .len = 4, .rd = 15, .rs1 = 31, .csr = 0x180 } },
        // csrw stimecmp, t3
        .{ .word = 0x14de1073, .want = .{ .op = .csrrw, .len = 4, .rd = 0, .rs1 = 28, .csr = 0x14D } },
        // mret, sret, wfi, ecall and ebreak
        .{ .word = 0x30200073, .want = .{ .op = .mret, .len = 4 } },
        .{ .word = 0x10200073, .want = .{ .op = .sret, .len = 4 } },
        .{ .word = 0x10500073, .want = .{ .op = .wfi, .len = 4 } },
        .{ .word = 0x00000073, .want = .{ .op = .ecall, .len = 4 } },
        .{ .word = 0x00100073, .want = .{ .op = .ebreak, .len = 4 } },
        // fence rw, rw
        .{ .word = 0x0330000f, .want = .{ .op = .fence, .len = 4, .imm = 0x033 } },
        // fence iorw, iorw
        .{ .word = 0x0ff0000f, .want = .{ .op = .fence, .len = 4, .imm = 0x0FF } },
        // fence.i
        .{ .word = 0x0000100f, .want = .{ .op = .fence_i, .len = 4 } },
        // sfence.vma
        .{ .word = 0x12000073, .want = .{ .op = .sfence_vma, .len = 4 } },
        // sfence.vma a0, a1
        .{ .word = 0x12b50073, .want = .{ .op = .sfence_vma, .len = 4, .rs1 = 10, .rs2 = 11 } },
        // fence r, w
        .{ .word = 0x0210000f, .want = .{ .op = .fence, .len = 4, .imm = 0x021 } },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.want, try decodeWord(case.word));
    }
}

test "words assembled by llvm for M and A decode to the expected fields" {
    // Each word comes from assembling the named instruction with
    // `zig cc --target=riscv64-linux-musl -c` after `.option norvc`, then reading the .text
    // bytes of the object file. Registers are a0 (x10), a1 (x11) and a2 (x12).
    const r3 = Decoded{ .op = .unknown, .len = 4, .rd = 10, .rs1 = 11, .rs2 = 12 };
    const lr = Decoded{ .op = .unknown, .len = 4, .rd = 10, .rs1 = 11 };
    const cases = [_]struct { word: u32, want: Decoded }{
        .{ .word = 0x02c58533, .want = with(r3, .mul) },
        .{ .word = 0x02c59533, .want = with(r3, .mulh) },
        .{ .word = 0x02c5a533, .want = with(r3, .mulhsu) },
        .{ .word = 0x02c5b533, .want = with(r3, .mulhu) },
        .{ .word = 0x02c5c533, .want = with(r3, .div) },
        .{ .word = 0x02c5d533, .want = with(r3, .divu) },
        .{ .word = 0x02c5e533, .want = with(r3, .rem) },
        .{ .word = 0x02c5f533, .want = with(r3, .remu) },
        .{ .word = 0x02c5853b, .want = with(r3, .mulw) },
        .{ .word = 0x02c5c53b, .want = with(r3, .divw) },
        .{ .word = 0x02c5d53b, .want = with(r3, .divuw) },
        .{ .word = 0x02c5e53b, .want = with(r3, .remw) },
        .{ .word = 0x02c5f53b, .want = with(r3, .remuw) },
        // lr.w, lr.w.aq
        .{ .word = 0x1005a52f, .want = with(lr, .lr_w) },
        .{ .word = 0x1405a52f, .want = aqrl(with(lr, .lr_w), true, false) },
        // sc.w, sc.w.rl
        .{ .word = 0x18c5a52f, .want = with(r3, .sc_w) },
        .{ .word = 0x1ac5a52f, .want = aqrl(with(r3, .sc_w), false, true) },
        .{ .word = 0x08c5a52f, .want = with(r3, .amoswap_w) },
        .{ .word = 0x00c5a52f, .want = with(r3, .amoadd_w) },
        .{ .word = 0x20c5a52f, .want = with(r3, .amoxor_w) },
        .{ .word = 0x60c5a52f, .want = with(r3, .amoand_w) },
        .{ .word = 0x40c5a52f, .want = with(r3, .amoor_w) },
        .{ .word = 0x80c5a52f, .want = with(r3, .amomin_w) },
        .{ .word = 0xa0c5a52f, .want = with(r3, .amomax_w) },
        .{ .word = 0xc0c5a52f, .want = with(r3, .amominu_w) },
        .{ .word = 0xe0c5a52f, .want = with(r3, .amomaxu_w) },
        .{ .word = 0x1005b52f, .want = with(lr, .lr_d) },
        .{ .word = 0x18c5b52f, .want = with(r3, .sc_d) },
        .{ .word = 0x08c5b52f, .want = with(r3, .amoswap_d) },
        .{ .word = 0x00c5b52f, .want = with(r3, .amoadd_d) },
        // amoadd.d.aqrl
        .{ .word = 0x06c5b52f, .want = aqrl(with(r3, .amoadd_d), true, true) },
        .{ .word = 0x20c5b52f, .want = with(r3, .amoxor_d) },
        .{ .word = 0x60c5b52f, .want = with(r3, .amoand_d) },
        .{ .word = 0x40c5b52f, .want = with(r3, .amoor_d) },
        .{ .word = 0x80c5b52f, .want = with(r3, .amomin_d) },
        .{ .word = 0xa0c5b52f, .want = with(r3, .amomax_d) },
        .{ .word = 0xc0c5b52f, .want = with(r3, .amominu_d) },
        .{ .word = 0xe0c5b52f, .want = with(r3, .amomaxu_d) },
        // lr.d.aqrl a0, (a1)
        .{ .word = 0x1605b52f, .want = aqrl(with(lr, .lr_d), true, true) },
        // lr.d.rl a3, (a4)
        .{ .word = 0x120736af, .want = aqrl(.{ .op = .lr_d, .len = 4, .rd = 13, .rs1 = 14 }, false, true) },
        // sc.d.aq a0, a2, (a1)
        .{ .word = 0x1cc5b52f, .want = aqrl(with(r3, .sc_d), true, false) },
        // sc.d.aqrl a5, a6, (a7)
        .{ .word = 0x1f08b7af, .want = aqrl(.{ .op = .sc_d, .len = 4, .rd = 15, .rs1 = 17, .rs2 = 16 }, true, true) },
        // amoswap.w.aq a0, a2, (a1)
        .{ .word = 0x0cc5a52f, .want = aqrl(with(r3, .amoswap_w), true, false) },
        // amoor.w.rl t0, t1, (t2)
        .{ .word = 0x4263a2af, .want = aqrl(.{ .op = .amoor_w, .len = 4, .rd = 5, .rs1 = 7, .rs2 = 6 }, false, true) },
        // amomaxu.w.aqrl a0, a2, (a1)
        .{ .word = 0xe6c5a52f, .want = aqrl(with(r3, .amomaxu_w), true, true) },
        // amomin.d.aq s1, s2, (s3)
        .{ .word = 0x8529b4af, .want = aqrl(.{ .op = .amomin_d, .len = 4, .rd = 9, .rs1 = 19, .rs2 = 18 }, true, false) },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.want, try decodeWord(case.word));
    }
}

fn with(base: Decoded, op: Op) Decoded {
    var out = base;
    out.op = op;
    return out;
}

fn aqrl(base: Decoded, aq: bool, rl: bool) Decoded {
    var out = base;
    out.aq = aq;
    out.rl = rl;
    return out;
}

test "every funct5 value in the atomic opcode decodes or is unknown" {
    const assigned = [_]struct { f5: u5, w: Op, d: Op }{
        .{ .f5 = 0b00010, .w = .lr_w, .d = .lr_d },
        .{ .f5 = 0b00011, .w = .sc_w, .d = .sc_d },
        .{ .f5 = 0b00001, .w = .amoswap_w, .d = .amoswap_d },
        .{ .f5 = 0b00000, .w = .amoadd_w, .d = .amoadd_d },
        .{ .f5 = 0b00100, .w = .amoxor_w, .d = .amoxor_d },
        .{ .f5 = 0b01100, .w = .amoand_w, .d = .amoand_d },
        .{ .f5 = 0b01000, .w = .amoor_w, .d = .amoor_d },
        .{ .f5 = 0b10000, .w = .amomin_w, .d = .amomin_d },
        .{ .f5 = 0b10100, .w = .amomax_w, .d = .amomax_d },
        .{ .f5 = 0b11000, .w = .amominu_w, .d = .amominu_d },
        .{ .f5 = 0b11100, .w = .amomaxu_w, .d = .amomaxu_d },
    };
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    var seen: usize = 0;
    for (0..32) |value| {
        const f5: u5 = @intCast(value);
        var want_w: Op = .unknown;
        var want_d: Op = .unknown;
        for (assigned) |entry| {
            if (entry.f5 == f5) {
                want_w = entry.w;
                want_d = entry.d;
                seen += 1;
            }
        }
        const word_w = rType(@as(u7, f5) << 2, 0b010, opcode_amo, 10, 11, 0);
        const word_d = rType(@as(u7, f5) << 2, 0b011, opcode_amo, 10, 11, 0);
        try std.testing.expectEqual(want_w, (try decodeWord(word_w)).op);
        try std.testing.expectEqual(want_d, (try decodeWord(word_d)).op);
        if (want_w == .unknown) {
            try std.testing.expectEqual(unknown, try decodeWord(word_w));
            try std.testing.expectEqual(unknown, try decodeWord(word_d));
        }
    }
    try std.testing.expectEqual(assigned.len, seen);
}

test "atomic widths other than word and double word decode as unknown" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    for ([_]u3{ 0b000, 0b001, 0b100, 0b101, 0b110, 0b111 }) |width| {
        try std.testing.expectEqual(unknown, try decodeWord(rType(0, width, opcode_amo, 10, 11, 12)));
    }
}

test "lr with a nonzero rs2 decodes as unknown" {
    // The specification fixes the rs2 field of lr at zero and has no rule to ignore it.
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    try std.testing.expectEqual(unknown, try decodeWord(0x1005a52f | (1 << 20)));
    try std.testing.expectEqual(unknown, try decodeWord(0x1005b52f | (31 << 20)));
}

test "multiply and divide words with another funct7 or width decode as unknown" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    for ([_]u3{ 0b001, 0b010, 0b011 }) |f3| {
        try std.testing.expectEqual(unknown, try decodeWord(rType(0b0000001, f3, opcode_op_32, 10, 11, 12)));
    }
    try std.testing.expectEqual(unknown, try decodeWord(rType(0b0000011, 0b000, opcode_op, 10, 11, 12)));
}

fn fpWant(op: Op, rm: u3, rs2: u5, rs3: u5) Decoded {
    return .{ .op = op, .len = 4, .rd = 10, .rs1 = 11, .rs2 = rs2, .rs3 = rs3, .rm = rm };
}

test "words assembled by llvm for F and D decode to the expected fields" {
    // Each word comes from assembling the named instruction with
    // `zig cc --target=riscv64-linux-musl -c` after `.option norvc`. Registers are fa0 (f10),
    // fa1 (f11), fa2 (f12), fa3 (f13), a0 (x10) and a1 (x11). The assembler writes `dyn`, which
    // is 7, for an instruction that takes a rounding mode and names none. It writes 0 for the
    // conversions that are exact.
    const cases = [_]struct { word: u32, want: Decoded }{
        // flw fa0, 8(a1)
        .{ .word = 0x0085a507, .want = .{ .op = .flw, .len = 4, .rd = 10, .rs1 = 11, .imm = 8 } },
        // fld fa0, 8(a1)
        .{ .word = 0x0085b507, .want = .{ .op = .fld, .len = 4, .rd = 10, .rs1 = 11, .imm = 8 } },
        // fsw fa0, 8(a1)
        .{ .word = 0x00a5a427, .want = .{ .op = .fsw, .len = 4, .rs1 = 11, .rs2 = 10, .imm = 8 } },
        // fsd fa0, 8(a1)
        .{ .word = 0x00a5b427, .want = .{ .op = .fsd, .len = 4, .rs1 = 11, .rs2 = 10, .imm = 8 } },
        // flw ft11, -4(s11)
        .{ .word = 0xffcdaf87, .want = .{ .op = .flw, .len = 4, .rd = 31, .rs1 = 27, .imm = -4 } },
        // fsd ft11, -2048(s11)
        .{ .word = 0x81fdb027, .want = .{ .op = .fsd, .len = 4, .rs1 = 27, .rs2 = 31, .imm = -2048 } },
        // fsw fa0, 2047(sp)
        .{ .word = 0x7ea12fa7, .want = .{ .op = .fsw, .len = 4, .rs1 = 2, .rs2 = 10, .imm = 2047 } },
        // fmadd.s ft11, ft10, ft9, ft8
        .{ .word = 0xe1df7fc3, .want = .{ .op = .fmadd_s, .len = 4, .rd = 31, .rs1 = 30, .rs2 = 29, .rs3 = 28, .rm = 7 } },
        .{ .word = 0x00c5f553, .want = fpWant(.fadd_s, 7, 12, 0) },
        .{ .word = 0x08c5f553, .want = fpWant(.fsub_s, 7, 12, 0) },
        .{ .word = 0x10c5f553, .want = fpWant(.fmul_s, 7, 12, 0) },
        .{ .word = 0x18c5f553, .want = fpWant(.fdiv_s, 7, 12, 0) },
        .{ .word = 0x5805f553, .want = fpWant(.fsqrt_s, 7, 0, 0) },
        .{ .word = 0x02c5f553, .want = fpWant(.fadd_d, 7, 12, 0) },
        .{ .word = 0x0ac5f553, .want = fpWant(.fsub_d, 7, 12, 0) },
        .{ .word = 0x12c5f553, .want = fpWant(.fmul_d, 7, 12, 0) },
        .{ .word = 0x1ac5f553, .want = fpWant(.fdiv_d, 7, 12, 0) },
        .{ .word = 0x5a05f553, .want = fpWant(.fsqrt_d, 7, 0, 0) },
        .{ .word = 0x68c5f543, .want = fpWant(.fmadd_s, 7, 12, 13) },
        .{ .word = 0x68c5f547, .want = fpWant(.fmsub_s, 7, 12, 13) },
        .{ .word = 0x68c5f54b, .want = fpWant(.fnmsub_s, 7, 12, 13) },
        .{ .word = 0x68c5f54f, .want = fpWant(.fnmadd_s, 7, 12, 13) },
        .{ .word = 0x6ac5f543, .want = fpWant(.fmadd_d, 7, 12, 13) },
        .{ .word = 0x20c58553, .want = fpWant(.fsgnj_s, 0, 12, 0) },
        .{ .word = 0x20c59553, .want = fpWant(.fsgnjn_s, 0, 12, 0) },
        .{ .word = 0x20c5a553, .want = fpWant(.fsgnjx_s, 0, 12, 0) },
        .{ .word = 0x28c58553, .want = fpWant(.fmin_s, 0, 12, 0) },
        .{ .word = 0x28c59553, .want = fpWant(.fmax_s, 0, 12, 0) },
        .{ .word = 0xa0c5a553, .want = fpWant(.feq_s, 0, 12, 0) },
        .{ .word = 0xa0c59553, .want = fpWant(.flt_s, 0, 12, 0) },
        .{ .word = 0xa0c58553, .want = fpWant(.fle_s, 0, 12, 0) },
        .{ .word = 0xe0059553, .want = fpWant(.fclass_s, 0, 0, 0) },
        .{ .word = 0xc005f553, .want = fpWant(.fcvt_w_s, 7, 0, 0) },
        .{ .word = 0xc015f553, .want = fpWant(.fcvt_wu_s, 7, 0, 0) },
        .{ .word = 0xc025f553, .want = fpWant(.fcvt_l_s, 7, 0, 0) },
        .{ .word = 0xc035f553, .want = fpWant(.fcvt_lu_s, 7, 0, 0) },
        .{ .word = 0xd005f553, .want = fpWant(.fcvt_s_w, 7, 0, 0) },
        .{ .word = 0xd025f553, .want = fpWant(.fcvt_s_l, 7, 0, 0) },
        .{ .word = 0x4015f553, .want = fpWant(.fcvt_s_d, 7, 0, 0) },
        .{ .word = 0x42058553, .want = fpWant(.fcvt_d_s, 0, 0, 0) },
        .{ .word = 0xe0058553, .want = fpWant(.fmv_x_w, 0, 0, 0) },
        .{ .word = 0xf0058553, .want = fpWant(.fmv_w_x, 0, 0, 0) },
        .{ .word = 0xe2058553, .want = fpWant(.fmv_x_d, 0, 0, 0) },
        .{ .word = 0xf2058553, .want = fpWant(.fmv_d_x, 0, 0, 0) },
        .{ .word = 0x00c59553, .want = fpWant(.fadd_s, 1, 12, 0) },
        .{ .word = 0x22c58553, .want = fpWant(.fsgnj_d, 0, 12, 0) },
        .{ .word = 0x22c59553, .want = fpWant(.fsgnjn_d, 0, 12, 0) },
        .{ .word = 0x22c5a553, .want = fpWant(.fsgnjx_d, 0, 12, 0) },
        .{ .word = 0x2ac58553, .want = fpWant(.fmin_d, 0, 12, 0) },
        .{ .word = 0x2ac59553, .want = fpWant(.fmax_d, 0, 12, 0) },
        .{ .word = 0xa2c5a553, .want = fpWant(.feq_d, 0, 12, 0) },
        .{ .word = 0xa2c59553, .want = fpWant(.flt_d, 0, 12, 0) },
        .{ .word = 0xa2c58553, .want = fpWant(.fle_d, 0, 12, 0) },
        .{ .word = 0xe2059553, .want = fpWant(.fclass_d, 0, 0, 0) },
        .{ .word = 0xc205f553, .want = fpWant(.fcvt_w_d, 7, 0, 0) },
        .{ .word = 0xc215f553, .want = fpWant(.fcvt_wu_d, 7, 0, 0) },
        .{ .word = 0xc225f553, .want = fpWant(.fcvt_l_d, 7, 0, 0) },
        .{ .word = 0xc235f553, .want = fpWant(.fcvt_lu_d, 7, 0, 0) },
        .{ .word = 0xd2058553, .want = fpWant(.fcvt_d_w, 0, 0, 0) },
        .{ .word = 0xd2158553, .want = fpWant(.fcvt_d_wu, 0, 0, 0) },
        .{ .word = 0xd225f553, .want = fpWant(.fcvt_d_l, 7, 0, 0) },
        .{ .word = 0xd235f553, .want = fpWant(.fcvt_d_lu, 7, 0, 0) },
        .{ .word = 0xd015f553, .want = fpWant(.fcvt_s_wu, 7, 0, 0) },
        .{ .word = 0xd035f553, .want = fpWant(.fcvt_s_lu, 7, 0, 0) },
        .{ .word = 0x6ac5f547, .want = fpWant(.fmsub_d, 7, 12, 13) },
        .{ .word = 0x6ac5f54b, .want = fpWant(.fnmsub_d, 7, 12, 13) },
        .{ .word = 0x6ac5f54f, .want = fpWant(.fnmadd_d, 7, 12, 13) },
        .{ .word = 0x4015a553, .want = fpWant(.fcvt_s_d, 2, 0, 0) },
        .{ .word = 0x5a05b553, .want = fpWant(.fsqrt_d, 3, 0, 0) },
        .{ .word = 0xc005c553, .want = fpWant(.fcvt_w_s, 4, 0, 0) },
        .{ .word = 0x02c58553, .want = fpWant(.fadd_d, 0, 12, 0) },
        .{ .word = 0x68c5a543, .want = fpWant(.fmadd_s, 2, 12, 13) },
    };
    for (cases) |case| {
        errdefer std.debug.print("word 0x{x:0>8}\n", .{case.word});
        try std.testing.expectEqual(case.want, try decodeWord(case.word));
    }
}

/// Each conversion `funct7` with the operation it gives for `rs2` of 0, 1, 2 and 3. A null entry
/// is a value the specification leaves unassigned.
const conversion_groups = [_]struct { f7: u7, ops: [4]?Op }{
    .{ .f7 = 0b1100000, .ops = .{ .fcvt_w_s, .fcvt_wu_s, .fcvt_l_s, .fcvt_lu_s } },
    .{ .f7 = 0b1100001, .ops = .{ .fcvt_w_d, .fcvt_wu_d, .fcvt_l_d, .fcvt_lu_d } },
    .{ .f7 = 0b1101000, .ops = .{ .fcvt_s_w, .fcvt_s_wu, .fcvt_s_l, .fcvt_s_lu } },
    .{ .f7 = 0b1101001, .ops = .{ .fcvt_d_w, .fcvt_d_wu, .fcvt_d_l, .fcvt_d_lu } },
    .{ .f7 = 0b0100000, .ops = .{ null, .fcvt_s_d, null, null } },
    .{ .f7 = 0b0100001, .ops = .{ .fcvt_d_s, null, null, null } },
};

test "a conversion decodes only for the rs2 values the specification assigns" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    for (conversion_groups) |group| {
        for (0..32) |value| {
            const field: u5 = @intCast(value);
            const word = rType(group.f7, 0b111, opcode_op_fp, 10, 11, field);
            errdefer std.debug.print("funct7 0b{b:0>7} rs2 {d}\n", .{ group.f7, field });
            const want = if (field < 4) group.ops[field] else null;
            if (want) |op| {
                try std.testing.expectEqual(Decoded{ .op = op, .len = 4, .rd = 10, .rs1 = 11, .rm = 7 }, try decodeWord(word));
            } else {
                try std.testing.expectEqual(unknown, try decodeWord(word));
            }
        }
    }
}

test "fsqrt, fclass and the moves decode only with rs2 zero" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    const cases = [_]struct { f7: u7, f3: u3, op: Op }{
        .{ .f7 = 0b0101100, .f3 = 0b111, .op = .fsqrt_s },
        .{ .f7 = 0b0101101, .f3 = 0b111, .op = .fsqrt_d },
        .{ .f7 = 0b1110000, .f3 = 0b001, .op = .fclass_s },
        .{ .f7 = 0b1110001, .f3 = 0b001, .op = .fclass_d },
        .{ .f7 = 0b1110000, .f3 = 0b000, .op = .fmv_x_w },
        .{ .f7 = 0b1110001, .f3 = 0b000, .op = .fmv_x_d },
        .{ .f7 = 0b1111000, .f3 = 0b000, .op = .fmv_w_x },
        .{ .f7 = 0b1111001, .f3 = 0b000, .op = .fmv_d_x },
    };
    for (cases) |case| {
        errdefer std.debug.print("{s}\n", .{@tagName(case.op)});
        try std.testing.expectEqual(case.op, (try decodeWord(rType(case.f7, case.f3, opcode_op_fp, 10, 11, 0))).op);
        for (1..32) |value| {
            const word = rType(case.f7, case.f3, opcode_op_fp, 10, 11, @intCast(value));
            try std.testing.expectEqual(unknown, try decodeWord(word));
        }
    }
}

test "a reserved rounding mode decodes as unknown and the rest reach rm" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    const words = [_]struct { f7: u7, rs2: u5, op: Op }{
        .{ .f7 = 0b0000000, .rs2 = 12, .op = .fadd_s },
        .{ .f7 = 0b0001101, .rs2 = 12, .op = .fdiv_d },
        .{ .f7 = 0b0101101, .rs2 = 0, .op = .fsqrt_d },
        .{ .f7 = 0b1100000, .rs2 = 2, .op = .fcvt_l_s },
        .{ .f7 = 0b1101001, .rs2 = 0, .op = .fcvt_d_w },
        .{ .f7 = 0b0100000, .rs2 = 1, .op = .fcvt_s_d },
        .{ .f7 = 0b0100001, .rs2 = 0, .op = .fcvt_d_s },
    };
    for (words) |entry| {
        for (0..8) |value| {
            const mode: u3 = @intCast(value);
            const word = rType(entry.f7, mode, opcode_op_fp, 10, 11, entry.rs2);
            errdefer std.debug.print("{s} rm {d}\n", .{ @tagName(entry.op), mode });
            const got = try decodeWord(word);
            if (mode == 5 or mode == 6) {
                try std.testing.expectEqual(unknown, got);
            } else {
                try std.testing.expectEqual(entry.op, got.op);
                try std.testing.expectEqual(mode, got.rm);
            }
        }
    }
    for ([_]u7{ opcode_fmadd, opcode_fmsub, opcode_fnmsub, opcode_fnmadd }) |opcode| {
        for (0..8) |value| {
            const mode: u3 = @intCast(value);
            const word = rType(0b0000000, mode, opcode, 10, 11, 12);
            const got = try decodeWord(word);
            if (mode == 5 or mode == 6) {
                try std.testing.expectEqual(unknown, got);
            } else {
                try std.testing.expectEqual(mode, got.rm);
            }
        }
    }
}

test "forms whose funct3 is part of the operation keep rm at zero" {
    for ([_]u32{
        rType(0b0010000, 0b010, opcode_op_fp, 10, 11, 12),
        rType(0b0010101, 0b001, opcode_op_fp, 10, 11, 12),
        rType(0b1010001, 0b010, opcode_op_fp, 10, 11, 12),
        rType(0b1110000, 0b001, opcode_op_fp, 10, 11, 0),
    }) |word| {
        const got = try decodeWord(word);
        try std.testing.expect(got.op != .unknown);
        try std.testing.expectEqual(@as(u3, 0), got.rm);
    }
}

test "float words with an unassigned funct3, format or funct5 decode as unknown" {
    const unknown = Decoded{ .op = .unknown, .len = 4 };
    for ([_]u3{ 0b000, 0b001, 0b100, 0b101, 0b110, 0b111 }) |width| {
        try std.testing.expectEqual(unknown, try decodeWord(iType(width, opcode_load_fp, 10, 11, 8)));
        try std.testing.expectEqual(unknown, try decodeWord(rType(0, width, opcode_store_fp, 10, 11, 12)));
    }
    const unassigned = [_]struct { f7: u7, f3: u3 }{
        .{ .f7 = 0b0010000, .f3 = 0b011 },
        .{ .f7 = 0b0010001, .f3 = 0b111 },
        .{ .f7 = 0b0010100, .f3 = 0b010 },
        .{ .f7 = 0b0010101, .f3 = 0b100 },
        .{ .f7 = 0b1010000, .f3 = 0b011 },
        .{ .f7 = 0b1010001, .f3 = 0b111 },
        .{ .f7 = 0b1110000, .f3 = 0b010 },
        .{ .f7 = 0b1110001, .f3 = 0b111 },
        .{ .f7 = 0b1111000, .f3 = 0b001 },
        .{ .f7 = 0b1111001, .f3 = 0b111 },
    };
    for (unassigned) |case| {
        try std.testing.expectEqual(unknown, try decodeWord(rType(case.f7, case.f3, opcode_op_fp, 10, 11, 0)));
    }
    for ([_]u2{ 0b10, 0b11 }) |format| {
        for (0..32) |value| {
            const f7 = (@as(u7, @intCast(value)) << 2) | format;
            try std.testing.expectEqual(unknown, try decodeWord(rType(f7, 0, opcode_op_fp, 10, 11, 0)));
        }
        for ([_]u7{ opcode_fmadd, opcode_fmsub, opcode_fnmsub, opcode_fnmadd }) |opcode| {
            try std.testing.expectEqual(unknown, try decodeWord(rType(@as(u7, 0b01101 << 2) | format, 0, opcode, 10, 11, 12)));
        }
    }
}

test "every funct5 value in the float opcode decodes or is unknown" {
    // With rs2 of zero and funct3 of zero, `funct5` 0b01000 is `fcvt.d.s` for the double format
    // only. The compare group reads funct3 0 as `fle`.
    const single = [_]u5{ 0b00000, 0b00001, 0b00010, 0b00011, 0b01011, 0b00100, 0b00101, 0b10100, 0b11000, 0b11010, 0b11100, 0b11110 };
    for ([_]u2{ 0b00, 0b01 }) |format| {
        var seen: usize = 0;
        for (0..32) |value| {
            const f5: u5 = @intCast(value);
            const got = try decodeWord(rType((@as(u7, f5) << 2) | format, 0, opcode_op_fp, 10, 11, 0));
            const assigned = std.mem.indexOfScalar(u5, &single, f5) != null or
                (format == 0b01 and f5 == 0b01000);
            try std.testing.expectEqual(assigned, got.op != .unknown);
            if (assigned) seen += 1;
        }
        try std.testing.expectEqual(single.len + format, seen);
    }
}

test "fused words keep rs3 in bits 31 to 27" {
    for ([_]u5{ 0, 1, 13, 31 }) |third| {
        const base = rType(@as(u7, third) << 2, 0b000, opcode_fmadd, 10, 11, 12);
        const got = try decodeWord(base);
        try std.testing.expectEqual(Decoded{ .op = .fmadd_s, .len = 4, .rd = 10, .rs1 = 11, .rs2 = 12, .rs3 = third }, got);
        const double = try decodeWord(base | (1 << 25));
        try std.testing.expectEqual(Op.fmadd_d, double.op);
        try std.testing.expectEqual(third, double.rs3);
    }
}

fn mnemonicOf(op: Op, buffer: []u8) []const u8 {
    const name = @tagName(op);
    @memcpy(buffer[0..name.len], name);
    std.mem.replaceScalar(u8, buffer[0..name.len], '_', '.');
    return buffer[0..name.len];
}

test "float arithmetic, fused and compare words agree with the disassembler" {
    const triples = [_][3]u5{ .{ 1, 2, 3 }, .{ 0, 5, 6 }, .{ 31, 30, 29 }, .{ 10, 11, 12 } };
    const arithmetic = [_]struct { op: Op, f7: u7 }{
        .{ .op = .fadd_s, .f7 = 0b0000000 },
        .{ .op = .fsub_s, .f7 = 0b0000100 },
        .{ .op = .fmul_s, .f7 = 0b0001000 },
        .{ .op = .fdiv_s, .f7 = 0b0001100 },
        .{ .op = .fadd_d, .f7 = 0b0000001 },
        .{ .op = .fsub_d, .f7 = 0b0000101 },
        .{ .op = .fmul_d, .f7 = 0b0001001 },
        .{ .op = .fdiv_d, .f7 = 0b0001101 },
    };
    const compare = [_]struct { op: Op, f7: u7, f3: u3 }{
        .{ .op = .feq_s, .f7 = 0b1010000, .f3 = 0b010 },
        .{ .op = .flt_s, .f7 = 0b1010000, .f3 = 0b001 },
        .{ .op = .fle_s, .f7 = 0b1010000, .f3 = 0b000 },
        .{ .op = .feq_d, .f7 = 0b1010001, .f3 = 0b010 },
        .{ .op = .flt_d, .f7 = 0b1010001, .f3 = 0b001 },
        .{ .op = .fle_d, .f7 = 0b1010001, .f3 = 0b000 },
    };
    const fused_forms = [_]struct { op: Op, opcode: u7, fmt: u7 }{
        .{ .op = .fmadd_s, .opcode = opcode_fmadd, .fmt = 0 },
        .{ .op = .fmsub_s, .opcode = opcode_fmsub, .fmt = 0 },
        .{ .op = .fnmsub_s, .opcode = opcode_fnmsub, .fmt = 0 },
        .{ .op = .fnmadd_s, .opcode = opcode_fnmadd, .fmt = 0 },
        .{ .op = .fmadd_d, .opcode = opcode_fmadd, .fmt = 1 },
        .{ .op = .fmsub_d, .opcode = opcode_fmsub, .fmt = 1 },
        .{ .op = .fnmsub_d, .opcode = opcode_fnmsub, .fmt = 1 },
        .{ .op = .fnmadd_d, .opcode = opcode_fnmadd, .fmt = 1 },
    };
    var name: [16]u8 = undefined;
    var line: [64]u8 = undefined;
    for (triples) |t| {
        for (arithmetic) |entry| {
            const word = rType(entry.f7, 0b111, opcode_op_fp, t[0], t[1], t[2]);
            const want = try std.fmt.bufPrint(&line, "{s} f{d}, f{d}, f{d}", .{ mnemonicOf(entry.op, &name), t[0], t[1], t[2] });
            try expectRendersAs(word, entry.op, want);
        }
        for (compare) |entry| {
            const word = rType(entry.f7, entry.f3, opcode_op_fp, t[0], t[1], t[2]);
            const want = try std.fmt.bufPrint(&line, "{s} x{d}, f{d}, f{d}", .{ mnemonicOf(entry.op, &name), t[0], t[1], t[2] });
            try expectRendersAs(word, entry.op, want);
        }
        for (fused_forms) |entry| {
            const word = rType((@as(u7, 29) << 2) | entry.fmt, 0b111, entry.opcode, t[0], t[1], t[2]);
            const want = try std.fmt.bufPrint(&line, "{s} f{d}, f{d}, f{d}, f29", .{ mnemonicOf(entry.op, &name), t[0], t[1], t[2] });
            try expectRendersAs(word, entry.op, want);
        }
    }
}

fn expectRendersAs(word: u32, op: Op, want: []const u8) !void {
    const text = try disasm.one(std.testing.allocator, word);
    defer std.testing.allocator.free(text);
    errdefer std.debug.print("word 0x{x:0>8} renders as \"{s}\"\n", .{ word, text });
    try std.testing.expectEqualStrings(want, text);
    try std.testing.expectEqual(op, (try decodeWord(word)).op);
}

fn expectFiles(op: Op, rd: Files.File, rs1: Files.File, rs2: Files.File) !void {
    const got = files(op);
    errdefer std.debug.print("operation {s}\n", .{@tagName(op)});
    try std.testing.expectEqual(rd, got.rd);
    try std.testing.expectEqual(rs1, got.rs1);
    try std.testing.expectEqual(rs2, got.rs2);
}

test "an operation that writes an integer register from a float input reports so" {
    const converts = [_]Op{
        .fcvt_w_s, .fcvt_wu_s, .fcvt_l_s, .fcvt_lu_s,
        .fcvt_w_d, .fcvt_wu_d, .fcvt_l_d, .fcvt_lu_d,
    };
    for (converts) |op| try expectFiles(op, .integer, .float, .float);
    try expectFiles(.fmv_x_w, .integer, .float, .float);
    try expectFiles(.fmv_x_d, .integer, .float, .float);
    try expectFiles(.fclass_s, .integer, .float, .float);
    try expectFiles(.fclass_d, .integer, .float, .float);
    const compares = [_]Op{ .feq_s, .flt_s, .fle_s, .feq_d, .flt_d, .fle_d };
    for (compares) |op| try expectFiles(op, .integer, .float, .float);
}

test "an operation that reads an integer register to make a float reports so" {
    const converts = [_]Op{
        .fcvt_s_w, .fcvt_s_wu, .fcvt_s_l, .fcvt_s_lu,
        .fcvt_d_w, .fcvt_d_wu, .fcvt_d_l, .fcvt_d_lu,
    };
    for (converts) |op| try expectFiles(op, .float, .integer, .float);
    try expectFiles(.fmv_w_x, .float, .integer, .float);
    try expectFiles(.fmv_d_x, .float, .integer, .float);
}

test "arithmetic and fused forms name float registers for every operand" {
    const arithmetic = [_]Op{
        .fadd_s,   .fsub_s,   .fmul_s,   .fdiv_s,   .fsqrt_s,
        .fsgnj_s,  .fsgnjn_s, .fsgnjx_s, .fmin_s,   .fmax_s,
        .fadd_d,   .fsub_d,   .fmul_d,   .fdiv_d,   .fsqrt_d,
        .fsgnj_d,  .fsgnjn_d, .fsgnjx_d, .fmin_d,   .fmax_d,
        .fcvt_s_d, .fcvt_d_s, .fmadd_s,  .fmsub_s,  .fnmsub_s,
        .fnmadd_s, .fmadd_d,  .fmsub_d,  .fnmsub_d, .fnmadd_d,
    };
    for (arithmetic) |op| {
        try expectFiles(op, .float, .float, .float);
        try std.testing.expectEqual(Files.File.float, files(op).rs3);
    }
}

test "an integer operation names integer registers" {
    try expectFiles(.addi, .integer, .integer, .integer);
    try expectFiles(.add, .integer, .integer, .integer);
    try expectFiles(.sd, .integer, .integer, .integer);
    try expectFiles(.amoadd_d, .integer, .integer, .integer);
    try expectFiles(.unknown, .integer, .integer, .integer);
}

test "float loads and stores take an integer base and mix files" {
    for ([_]Op{ .flw, .fld }) |op| try expectFiles(op, .float, .integer, .integer);
    for ([_]Op{ .fsw, .fsd }) |op| try expectFiles(op, .integer, .integer, .float);
}

// The words below come from the text of two Weir images built with ReleaseFast for the River
// board, the first stage and the main firmware. Each image was walked from its first byte by the
// lengths `decode` reports. Both walks ended on the last byte with no unknown result. The images
// hold 2789 distinct 16 bit words and 9428 distinct 32 bit words. This table keeps up to eight
// words for each operation and length, spread evenly over the sorted words. To regenerate, dump
// the distinct words of both `.text` sections, repeat that selection and sort each list. The
// The test checks only that each word is not unknown and that its length matches. The table proves
// that these words decode to something and says nothing about whether the fields are right. The
// tables derived from an assembler check the fields.
// words show what one real compiler emitted. They make no claim about the whole instruction set.
const weir_words16 = [_]u16{
    0x0001, 0x00e2, 0x0512, 0x060a, 0x0722, 0x0809, 0x08e2, 0x0e62,
    0x15e2, 0x1a02, 0x1c00, 0x22a1, 0x2501, 0x2505, 0x2681, 0x2689,
    0x2711, 0x2789, 0x2d85, 0x409c, 0x4154, 0x4210, 0x42d4, 0x439c,
    0x4554, 0x4605, 0x4791, 0x4a81, 0x4a88, 0x4e81, 0x5498, 0x6088,
    0x62d5, 0x62f8, 0x6511, 0x65c1, 0x661c, 0x6689, 0x6715, 0x6841,
    0x69fa, 0x6c89, 0x6cae, 0x7179, 0x7406, 0x78fd, 0x7a46, 0x7fee,
    0x8082, 0x8091, 0x80a6, 0x8189, 0x8211, 0x82e1, 0x8582, 0x85e6,
    0x8702, 0x8735, 0x882a, 0x8905, 0x8941, 0x8a05, 0x8a41, 0x8a9d,
    0x8b91, 0x8b96, 0x8c85, 0x8ca9, 0x8cb1, 0x8cd1, 0x8ced, 0x8d0d,
    0x8d31, 0x8d55, 0x8d6d, 0x8d79, 0x8d89, 0x8db5, 0x8dc9, 0x8ddd,
    0x8df1, 0x8dfd, 0x8e09, 0x8e19, 0x8e35, 0x8e59, 0x8e75, 0x8e8d,
    0x8ea5, 0x8ecd, 0x8ee5, 0x8f11, 0x8f2d, 0x8f45, 0x8f7d, 0x8f99,
    0x8fb1, 0x8fc5, 0x9111, 0x91d1, 0x927d, 0x9381, 0x9482, 0x94c2,
    0x9505, 0x957d, 0x9595, 0x9646, 0x9682, 0x96bd, 0x96e1, 0x971d,
    0x977d, 0x9782, 0x9932, 0x9971, 0x9982, 0x9b82, 0x9be1, 0x9d31,
    0x9e35, 0x9f29, 0x9f3d, 0x9fea, 0xa005, 0xa105, 0xa801, 0xa8f9,
    0xaeb5, 0xb745, 0xbd31, 0xbff9, 0xc09d, 0xc10c, 0xc184, 0xc198,
    0xc265, 0xc308, 0xc55d, 0xc710, 0xc8a8, 0xc909, 0xcb25, 0xce75,
    0xd0a8, 0xd5fd, 0xdcb4, 0xdfbd, 0xe022, 0xe105, 0xe23d, 0xe310,
    0xe5b9, 0xe61c, 0xe961, 0xeaca, 0xed45, 0xeea6, 0xf165, 0xf41c,
    0xf8da, 0xf96d, 0xffd6, 0xffe5,
};

const weir_words32 = [_]u32{
    0x00000073, 0x00000097, 0x0000100f, 0x00004f97, 0x00009597, 0x0000b803,
    0x0000e097, 0x00012597, 0x00020837, 0x00028d63, 0x00039563, 0x0003c783,
    0x00048623, 0x00049423, 0x0004a823, 0x0004b023, 0x0004d503, 0x0005059b,
    0x00051603, 0x00052003, 0x00056503, 0x00058497, 0x00058603, 0x0005861b,
    0x00059603, 0x0006059b, 0x0006069b, 0x00065603, 0x00066683, 0x0006c763,
    0x0008a617, 0x000a2423, 0x000b051b, 0x000b059b, 0x001035b3, 0x00130513,
    0x0013f613, 0x00149513, 0x0014b593, 0x00156533, 0x0015b513, 0x0015d49b,
    0x00165693, 0x0017b513, 0x001a4683, 0x001a7513, 0x002080e7, 0x00254513,
    0x00260683, 0x002e869b, 0x0034c913, 0x00350483, 0x00351493, 0x00358583,
    0x00368a03, 0x0039f313, 0x003a8e03, 0x003a9613, 0x003e2713, 0x00471793,
    0x004a5a13, 0x0054c6b3, 0x00556863, 0x00560533, 0x00561733, 0x0056f863,
    0x005a15b3, 0x005cc783, 0x0066f5b3, 0x006bc333, 0x007355b7, 0x0074ca13,
    0x0074cc13, 0x0075d49b, 0x00761633, 0x007e6533, 0x0085151b, 0x0085d59b,
    0x00951b1b, 0x0095d793, 0x0095d89b, 0x0096161b, 0x00965633, 0x00967933,
    0x009b86bb, 0x009d84bb, 0x009d961b, 0x00a000ef, 0x00a035b3, 0x00a48a33,
    0x00a59613, 0x00a67ab3, 0x00a768b3, 0x00aab5b3, 0x00ab1423, 0x00ac86b3,
    0x00b48023, 0x00b49633, 0x00b60bbb, 0x00b61633, 0x00b75733, 0x00b95663,
    0x00b95863, 0x00b9d5bb, 0x00bd87b3, 0x00c52f03, 0x00c579b3, 0x00c595b3,
    0x00c5cd33, 0x00c5d5b3, 0x00c5f713, 0x00c6c963, 0x00c73733, 0x00c75733,
    0x00c78e3b, 0x00c794b3, 0x00c7efb3, 0x00c89723, 0x00c90a33, 0x00ce063b,
    0x00d5cc63, 0x00d65633, 0x00d77633, 0x00d77763, 0x00e587b3, 0x00e60923,
    0x00e7d733, 0x00e7e3b3, 0x00e89023, 0x00f4cc93, 0x00f4d733, 0x00f536b3,
    0x00f71733, 0x0102c5b3, 0x0102f6b3, 0x01056b83, 0x0105d713, 0x01069703,
    0x0106d803, 0x0107971b, 0x0108e833, 0x010bcbb3, 0x012a853b, 0x01348533,
    0x0145a023, 0x01476a63, 0x01503733, 0x015b3c23, 0x015bcab3, 0x016b4603,
    0x0175959b, 0x017c0bbb, 0x017c66b3, 0x0184979b, 0x01855593, 0x0189951b,
    0x018b0c3b, 0x01a4b493, 0x01a73713, 0x01b034b3, 0x01b9cdb3, 0x01c5551b,
    0x01c5d533, 0x01c675b3, 0x01d7f7b3, 0x01e705b3, 0x01ea0823, 0x01f037b3,
    0x01f3d79b, 0x01f50783, 0x01f65e1b, 0x01f6cfb3, 0x01f7d71b, 0x01f9e633,
    0x01fa4993, 0x02000837, 0x0203f393, 0x02051593, 0x0205d493, 0x020a8063,
    0x021d85bb, 0x0230000f, 0x02560733, 0x0256dc63, 0x0266b6b3, 0x02a70533,
    0x02a855bb, 0x02b635b3, 0x02b6f363, 0x02b7b7b3, 0x02c50633, 0x02c53533,
    0x02c6cd63, 0x02c7b7b3, 0x02d4b4b3, 0x02d5553b, 0x02d61c63, 0x02da0b33,
    0x02e63eb3, 0x02f587bb, 0x02f58f3b, 0x02f68633, 0x02f69733, 0x02f6d833,
    0x02f707bb, 0x0304d613, 0x03056513, 0x0305e593, 0x03091693, 0x03096593,
    0x0309e613, 0x030ae513, 0x030ae613, 0x030be613, 0x030c07b3, 0x0310000f,
    0x03350633, 0x03354a63, 0x033cfc33, 0x034c2503, 0x03a53533, 0x03c56583,
    0x03c5c883, 0x03c70613, 0x03ee55bb, 0x03f60333, 0x03f75793, 0x03fb1513,
    0x0456dd63, 0x04b3f763, 0x04c86e63, 0x060b3603, 0x0654a023, 0x06a0106f,
    0x0749e603, 0x0803f2b7, 0x08080363, 0x08575463, 0x0924fd63, 0x0c854583,
    0x0d05a603, 0x0de000ef, 0x0e068693, 0x0e06f713, 0x0f56b713, 0x0f96b713,
    0x0fe6b713, 0x0ff6f813, 0x10500073, 0x10502573, 0x10569073, 0x12000073,
    0x12051363, 0x12da8b63, 0x12ee6163, 0x13575163, 0x14da9073, 0x154000ef,
    0x160725af, 0x18002673, 0x1a0a0423, 0x1a538337, 0x1ac92503, 0x1af7262f,
    0x1cb64563, 0x1d0080e7, 0x1e058963, 0x1ed7db63, 0x1eda2e23, 0x1f70006f,
    0x21313c23, 0x23461063, 0x288a3503, 0x288b0913, 0x2ad77763, 0x2c856b83,
    0x30047073, 0x30200073, 0x30371073, 0x30453073, 0x3045a073, 0x30569073,
    0x30669073, 0x34059073, 0x34102573, 0x341029f3, 0x34179073, 0x34302673,
    0x3445b073, 0x34ac3023, 0x3727b783, 0x3ae080e7, 0x3b069073, 0x4055d933,
    0x4055da33, 0x4056561b, 0x40570e3b, 0x405755b3, 0x40648633, 0x4076d69b,
    0x4087d79b, 0x40978b33, 0x40998dbb, 0x409c09b3, 0x40a6561b, 0x40a80dbb,
    0x40a88533, 0x40b654b3, 0x40b7d79b, 0x40c68933, 0x40e88633, 0x40f699e3,
    0x4117d79b, 0x412b0633, 0x4146063b, 0x41850abb, 0x41898b3b, 0x418a8abb,
    0x41f5551b, 0x41f787b3, 0x41f7d61b, 0x4297a02f, 0x42d6202f, 0x43f65713,
    0x43f8d293, 0x451c2623, 0x46058b63, 0x48c3cf63, 0x4d575637, 0x588080e7,
    0x5a030423, 0x5b00106f, 0x5ed49f23, 0x5ee68693, 0x62b6202f, 0x62c5a02f,
    0x6cec3823, 0x7da0106f, 0x7f591d23, 0x84e080e7, 0x97043503, 0x9bf58d93,
    0xa00044b7, 0xa2b568e3, 0xab8080e7, 0xaeb51fe3, 0xb0c42583, 0xc01024f3,
    0xcab43c23, 0xcc446a03, 0xcc845503, 0xd3043903, 0xd3944503, 0xd4060be3,
    0xd6a080e7, 0xd6d40e23, 0xdff0851b, 0xe0050593, 0xe00b2c23, 0xf0a67fe3,
    0xf1402773, 0xf2845503, 0xf2e43023, 0xf4043903, 0xf5845503, 0xf8bb1823,
    0xf8c697e3, 0xf9042503, 0xf954ede3, 0xfa845503, 0xfbc46603, 0xfcd8e7e3,
    0xfe0658e3, 0xfe5b1523, 0xfe7fe06f, 0xfecfc3e3, 0xff596ce3, 0xff75bc23,
    0xff867713, 0xff883703, 0xffa5e593, 0xffc39be3, 0xffc7a703, 0xffc7f3e3,
    0xffd50be3, 0xffe080e7, 0xffe30fa3, 0xffe4d503, 0xffeb2623, 0xfff48603,
    0xfff54613, 0xfff7c593, 0xfffc4703, 0xfffdc3b7, 0xffff8f13, 0xfffff797,
};

test "every word seen in the Weir images decodes" {
    for (weir_words16) |word| {
        const got = try decodeHalf(word);
        try std.testing.expect(got.op != .unknown);
        try std.testing.expectEqual(@as(u3, 2), got.len);
    }
    for (weir_words32) |word| {
        const got = try decodeWord(word);
        try std.testing.expect(got.op != .unknown);
        try std.testing.expectEqual(@as(u3, 4), got.len);
    }
}
