//! Decoding a Glacier instruction word into what it does.
//!
//! The inverse of what `encode.zig` emits, and the input to `lift.zig`. A Glacier word is a
//! RISC-V word, so this decodes RV32I, RV32M, the A extension word atomics, the one CSR read the
//! core implements, and the three words that end or park a warp.
//!
//! It is written against the RISC-V unprivileged specification rather than against the encoders
//! beside it. A decoder built by reading its own encoder passes whenever the two agree, including
//! where both are wrong, so the tests below pin whole words from the specification's tables and
//! then round trip the encoders for coverage.
//!
//! A word this does not decode is `error.Illegal` and never a word that does nothing. The core
//! deliberately decodes a small set, and a kernel holding something outside it is a kernel nothing
//! can run, which is worth saying at once rather than after a wrong answer.

const std = @import("std");

pub const Error = error{Illegal};

/// A register number. Register zero reads zero and takes no writes, which is the reader's to
/// honour rather than this decoder's.
pub const Reg = u5;

/// What a branch asks about its two registers.
pub const Compare = enum { eq, ne, lt, ge, lt_unsigned, ge_unsigned };

/// How much of a word an access moves.
pub const Width = enum { byte, half, word };

/// The arithmetic the core has. The multiply and divide set is RV32M, and the rest is RV32I.
pub const Alu = enum {
    add,
    sub,
    sll,
    slt,
    slt_unsigned,
    xor,
    srl,
    sra,
    @"or",
    @"and",
    mul,
    mul_high,
    mul_high_signed_unsigned,
    mul_high_unsigned,
    div,
    div_unsigned,
    rem,
    rem_unsigned,
};

/// The atomics the core has. There is no load reserved pair, so there is no compare and swap.
pub const Atomic = enum { add, swap, @"and", @"or" };

pub const Instruction = union(enum) {
    /// The immediate, already in the top twenty bits of a word.
    lui: struct { rd: Reg, value: u32 },
    /// The same immediate, to add to the address of this instruction.
    auipc: struct { rd: Reg, value: u32 },
    /// A jump to this address plus the offset, with the address after it into `rd`.
    jal: struct { rd: Reg, offset: i32 },
    /// A jump to a register plus an offset, with the low bit of the target cleared.
    jalr: struct { rd: Reg, rs1: Reg, offset: i32 },
    branch: struct { on: Compare, rs1: Reg, rs2: Reg, offset: i32 },
    /// A load, which extends a byte or a half from the addressed bytes.
    load: struct { width: Width, signed: bool, rd: Reg, rs1: Reg, offset: i32 },
    /// A store, which drives the byte select of the lanes it addresses and leaves the rest of the
    /// word alone.
    store: struct { width: Width, rs1: Reg, rs2: Reg, offset: i32 },
    /// Arithmetic against an immediate. A shift carries its amount there.
    alu_imm: struct { op: Alu, rd: Reg, rs1: Reg, value: i32 },
    alu: struct { op: Alu, rd: Reg, rs1: Reg, rs2: Reg },
    /// A read and modify of one word, which answers with the value it found. The ordering bits are
    /// accepted and dropped, because one bus with one access in flight already gives the order
    /// they ask for.
    atomic: struct { op: Atomic, rd: Reg, rs1: Reg, rs2: Reg },
    /// A read of a control register. The core decodes `csrrs` with `rs1` of zero and nothing else,
    /// so there is no write to decode.
    csr: struct { rd: Reg, which: u12 },
    /// A plain fence. One bus in order needs nothing of it.
    fence,
    /// The cross warp barrier, which is a fence with a mode of one. It parks a warp until every
    /// other live warp on the core reaches one.
    barrier,
    /// `ebreak`, which retires the warp as finished. A kernel ends with this because a warp has no
    /// caller to return to.
    retire,
    /// `ecall`, which retires it as faulted. The code is the value the register holds at the time
    /// and not the register number.
    fault: struct { rs1: Reg },
};

const opcode_load = 0x03;
const opcode_fence = 0x0f;
const opcode_alu_imm = 0x13;
const opcode_auipc = 0x17;
const opcode_store = 0x23;
const opcode_atomic = 0x2f;
const opcode_alu = 0x33;
const opcode_lui = 0x37;
const opcode_branch = 0x63;
const opcode_jalr = 0x67;
const opcode_jal = 0x6f;
const opcode_system = 0x73;

/// The whole barrier word, so a fence with a mode of one is one comparison. RISC-V gives mode
/// zero to a plain fence and eight to `fence.tso`, which leaves this encoding for a core to claim.
pub const barrier_word: u32 = 0x1000000f;

fn rd(word: u32) Reg {
    return @truncate(word >> 7);
}

fn rs1(word: u32) Reg {
    return @truncate(word >> 15);
}

fn rs2(word: u32) Reg {
    return @truncate(word >> 20);
}

fn funct3(word: u32) u3 {
    return @truncate(word >> 12);
}

fn funct7(word: u32) u7 {
    return @truncate(word >> 25);
}

fn immediateI(word: u32) i32 {
    return @as(i32, @bitCast(word)) >> 20;
}

fn immediateS(word: u32) i32 {
    const value: u12 = @truncate((((word >> 25) & 0x7f) << 5) | ((word >> 7) & 0x1f));
    return @as(i12, @bitCast(value));
}

fn immediateB(word: u32) i32 {
    const value: u13 = @truncate((((word >> 31) & 1) << 12) |
        (((word >> 7) & 1) << 11) |
        (((word >> 25) & 0x3f) << 5) |
        (((word >> 8) & 0xf) << 1));
    return @as(i13, @bitCast(value));
}

fn immediateJ(word: u32) i32 {
    const value: u21 = @truncate((((word >> 31) & 1) << 20) |
        (((word >> 12) & 0xff) << 12) |
        (((word >> 20) & 1) << 11) |
        (((word >> 21) & 0x3ff) << 1));
    return @as(i21, @bitCast(value));
}

/// The arithmetic a word names, for the register form and for the immediate form alike. The
/// immediate form has no second register, so it reaches none of the multiply set and neither
/// subtract.
fn alu(which: u3, seven: u7, immediate: bool) Error!Alu {
    if (!immediate and seven == 0x01) return switch (which) {
        0 => .mul,
        1 => .mul_high,
        2 => .mul_high_signed_unsigned,
        3 => .mul_high_unsigned,
        4 => .div,
        5 => .div_unsigned,
        6 => .rem,
        7 => .rem_unsigned,
    };

    // A shift's immediate form carries the amount in the low five bits, so the rest of the field
    // is what says which shift it is.
    const high = if (immediate) seven & 0x7e else seven;
    return switch (which) {
        0 => if (immediate or high == 0x00)
            .add
        else if (high == 0x20)
            .sub
        else
            error.Illegal,
        1 => if (high == 0x00) .sll else error.Illegal,
        2 => .slt,
        3 => .slt_unsigned,
        4 => .xor,
        5 => switch (high) {
            0x00 => .srl,
            0x20 => .sra,
            else => error.Illegal,
        },
        6 => .@"or",
        7 => .@"and",
    };
}

/// What one word does, or `error.Illegal` where the core decodes nothing for it.
pub fn one(word: u32) Error!Instruction {
    if (word == barrier_word) return .barrier;

    return switch (@as(u7, @truncate(word))) {
        opcode_lui => .{ .lui = .{ .rd = rd(word), .value = word & 0xffff_f000 } },
        opcode_auipc => .{ .auipc = .{ .rd = rd(word), .value = word & 0xffff_f000 } },
        opcode_jal => .{ .jal = .{ .rd = rd(word), .offset = immediateJ(word) } },
        opcode_jalr => if (funct3(word) == 0)
            .{ .jalr = .{ .rd = rd(word), .rs1 = rs1(word), .offset = immediateI(word) } }
        else
            error.Illegal,
        opcode_branch => .{ .branch = .{
            .on = switch (funct3(word)) {
                0 => .eq,
                1 => .ne,
                4 => .lt,
                5 => .ge,
                6 => .lt_unsigned,
                7 => .ge_unsigned,
                else => return error.Illegal,
            },
            .rs1 = rs1(word),
            .rs2 = rs2(word),
            .offset = immediateB(word),
        } },
        opcode_load => .{
            .load = .{
                .width = switch (funct3(word)) {
                    0, 4 => .byte,
                    1, 5 => .half,
                    2 => .word,
                    else => return error.Illegal,
                },
                // The unsigned forms are the signed ones with bit two set, and a word load has no
                // unsigned form because there is nothing above it to extend into.
                .signed = funct3(word) & 0b100 == 0,
                .rd = rd(word),
                .rs1 = rs1(word),
                .offset = immediateI(word),
            },
        },
        opcode_store => .{ .store = .{
            .width = switch (funct3(word)) {
                0 => .byte,
                1 => .half,
                2 => .word,
                else => return error.Illegal,
            },
            .rs1 = rs1(word),
            .rs2 = rs2(word),
            .offset = immediateS(word),
        } },
        opcode_alu_imm => .{
            .alu_imm = .{
                .op = try alu(funct3(word), funct7(word), true),
                .rd = rd(word),
                .rs1 = rs1(word),
                // A shift takes its amount from the low five bits of the same field, which is what the
                // immediate already holds once the rest of it is dropped.
                .value = switch (funct3(word)) {
                    1, 5 => @as(i32, rs2(word)),
                    else => immediateI(word),
                },
            },
        },
        opcode_alu => .{ .alu = .{
            .op = try alu(funct3(word), funct7(word), false),
            .rd = rd(word),
            .rs1 = rs1(word),
            .rs2 = rs2(word),
        } },
        opcode_atomic => blk: {
            // Word atomics only. The core has no doubleword, and the width field says which.
            if (funct3(word) != 0b010) break :blk error.Illegal;
            break :blk .{ .atomic = .{
                .op = switch (funct7(word) >> 2) {
                    0b00000 => .add,
                    0b00001 => .swap,
                    0b01100 => .@"and",
                    0b01000 => .@"or",
                    else => break :blk error.Illegal,
                },
                .rd = rd(word),
                .rs1 = rs1(word),
                .rs2 = rs2(word),
            } };
        },
        opcode_fence => if (funct3(word) == 0) .fence else error.Illegal,
        opcode_system => blk: {
            // `csrrs` with a source of zero is a read. Anything that would write a control
            // register is outside what the core decodes.
            if (funct3(word) == 0b010 and rs1(word) == 0) {
                break :blk .{ .csr = .{ .rd = rd(word), .which = @truncate(word >> 20) } };
            }
            if (funct3(word) != 0) break :blk error.Illegal;
            // The two retires differ in one bit of the immediate, and the core wants a source of
            // zero on the finish.
            break :blk switch (word >> 20) {
                0x000 => .{ .fault = .{ .rs1 = rs1(word) } },
                0x001 => if (rs1(word) == 0) .retire else error.Illegal,
                else => error.Illegal,
            };
        },
        else => error.Illegal,
    };
}

const rv = @import("../riscv64/encode.zig");
const encode = @import("encode.zig");

test "the words that end or park a warp" {
    // All three are pinned in `encode.zig` with their own provenance, and the barrier word is in
    // the programming model as the whole constant.
    try std.testing.expectEqual(@as(Instruction, .retire), try one(0x0010_0073));
    try std.testing.expectEqual(@as(Instruction, .barrier), try one(0x1000_000f));
    try std.testing.expectEqual(Instruction{ .fault = .{ .rs1 = 0 } }, try one(0x0000_0073));
    try std.testing.expectEqual(Instruction{ .fault = .{ .rs1 = 31 } }, try one(0x000f_8073));
}

test "whole words from the specification decode to what they say" {
    // `addi x10, x0, 1`, which is the word every RISC-V tutorial opens with.
    try std.testing.expectEqual(
        Instruction{ .alu_imm = .{ .op = .add, .rd = 10, .rs1 = 0, .value = 1 } },
        try one(0x0010_0513),
    );

    // `lw x5, 8(x6)`.
    try std.testing.expectEqual(
        Instruction{ .load = .{ .width = .word, .signed = true, .rd = 5, .rs1 = 6, .offset = 8 } },
        try one(0x0083_2283),
    );

    // `sw x7, 12(x8)`.
    try std.testing.expectEqual(
        Instruction{ .store = .{ .width = .word, .rs1 = 8, .rs2 = 7, .offset = 12 } },
        try one(0x0074_2623),
    );

    // `beq x1, x2, 16`.
    try std.testing.expectEqual(
        Instruction{ .branch = .{ .on = .eq, .rs1 = 1, .rs2 = 2, .offset = 16 } },
        try one(0x0020_8863),
    );

    // `jal x0, -4`, which is a branch back onto itself and the hardest immediate of the set.
    try std.testing.expectEqual(
        Instruction{ .jal = .{ .rd = 0, .offset = -4 } },
        try one(0xFFDF_F06F),
    );
}

test "every arithmetic word the encoders emit decodes back to the same operation" {
    const pairs = [_]struct { word: u32, op: Alu }{
        .{ .word = rv.add(.x1, .x2, .x3), .op = .add },
        .{ .word = rv.sub(.x1, .x2, .x3), .op = .sub },
        .{ .word = rv.sll(.x1, .x2, .x3), .op = .sll },
        .{ .word = rv.slt(.x1, .x2, .x3), .op = .slt },
        .{ .word = rv.sltu(.x1, .x2, .x3), .op = .slt_unsigned },
        .{ .word = rv.xor_(.x1, .x2, .x3), .op = .xor },
        .{ .word = rv.srl(.x1, .x2, .x3), .op = .srl },
        .{ .word = rv.sra(.x1, .x2, .x3), .op = .sra },
        .{ .word = rv.or_(.x1, .x2, .x3), .op = .@"or" },
        .{ .word = rv.and_(.x1, .x2, .x3), .op = .@"and" },
        .{ .word = rv.mul(.x1, .x2, .x3), .op = .mul },
        .{ .word = rv.mulh(.x1, .x2, .x3), .op = .mul_high },
        .{ .word = rv.mulhu(.x1, .x2, .x3), .op = .mul_high_unsigned },
        .{ .word = rv.div(.x1, .x2, .x3), .op = .div },
        .{ .word = rv.divu(.x1, .x2, .x3), .op = .div_unsigned },
        .{ .word = rv.rem(.x1, .x2, .x3), .op = .rem },
        .{ .word = rv.remu(.x1, .x2, .x3), .op = .rem_unsigned },
    };

    for (pairs) |each| {
        const found = try one(each.word);
        try std.testing.expectEqual(each.op, found.alu.op);
        try std.testing.expectEqual(@as(Reg, 1), found.alu.rd);
        try std.testing.expectEqual(@as(Reg, 2), found.alu.rs1);
        try std.testing.expectEqual(@as(Reg, 3), found.alu.rs2);
    }
}

test "a shift by an immediate carries its amount and not a sign extended field" {
    const word = rv.srai(.x5, .x6, 31);
    try std.testing.expectEqual(
        Instruction{ .alu_imm = .{ .op = .sra, .rd = 5, .rs1 = 6, .value = 31 } },
        try one(word),
    );

    // The right shifts differ in a bit of the same field the amount sits in, so a decoder that
    // read the whole field as the amount would call this one a logical shift by a huge number.
    try std.testing.expectEqual(
        Instruction{ .alu_imm = .{ .op = .srl, .rd = 5, .rs1 = 6, .value = 31 } },
        try one(rv.srli(.x5, .x6, 31)),
    );
}

test "a negative immediate keeps its sign through every format that carries one" {
    try std.testing.expectEqual(@as(i32, -1), (try one(rv.addi(.x1, .x2, -1))).alu_imm.value);
    try std.testing.expectEqual(@as(i32, -2048), (try one(rv.addi(.x1, .x2, -2048))).alu_imm.value);
    try std.testing.expectEqual(@as(i32, -4), (try one(rv.lw(.x1, .x2, -4))).load.offset);
    try std.testing.expectEqual(@as(i32, -8), (try one(rv.sw(.x2, .x1, -8))).store.offset);
    try std.testing.expectEqual(@as(i32, -16), (try one(rv.beq(.x1, .x2, -16))).branch.offset);
}

test "the atomics the core has, and the double word ones it does not" {
    try std.testing.expectEqual(Atomic.add, (try one(rv.amo(.add, .x1, .x2, .x3, false, false))).atomic.op);
    try std.testing.expectEqual(Atomic.swap, (try one(rv.amo(.swap, .x1, .x2, .x3, false, false))).atomic.op);
    try std.testing.expectEqual(Atomic.@"and", (try one(rv.amo(.@"and", .x1, .x2, .x3, false, false))).atomic.op);
    try std.testing.expectEqual(Atomic.@"or", (try one(rv.amo(.@"or", .x1, .x2, .x3, false, false))).atomic.op);

    // The ordering bits are part of the same field as the operation, so a decoder that read the
    // whole field would refuse an acquiring atomic the core accepts.
    try std.testing.expectEqual(Atomic.add, (try one(rv.amo(.add, .x1, .x2, .x3, true, false))).atomic.op);

    // A double word atomic is a width the core does not have.
    try std.testing.expectError(error.Illegal, one((rv.amo(.add, .x1, .x2, .x3, false, false) & ~@as(u32, 0x7000)) | 0x3000));
}

test "a control register read names the register, and a write is refused" {
    const word = rv.csrrs(.x7, encode.Csr.mhartid, .x0);
    try std.testing.expectEqual(
        Instruction{ .csr = .{ .rd = 7, .which = encode.Csr.mhartid } },
        try one(word),
    );

    // The same word with a source register would write, which is not something the core decodes.
    try std.testing.expectError(error.Illegal, one(rv.csrrs(.x7, encode.Csr.mhartid, .x8)));
}

test "a word outside what the core decodes is refused rather than ignored" {
    // A plain fence is decoded and does nothing. The barrier is the same opcode with a mode.
    try std.testing.expectEqual(@as(Instruction, .fence), try one(0x0ff0_000f));
    try std.testing.expectError(error.Illegal, one(0x0000_0000));
    try std.testing.expectError(error.Illegal, one(0xffff_ffff));
    // A vector word. Real, and not decoded here yet, which is a refusal and not a no-op.
    try std.testing.expectError(error.Illegal, one(0x0000_0057));
}
