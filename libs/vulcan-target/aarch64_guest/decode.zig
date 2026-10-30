//! AArch64 instruction classification; encoding masks live only here.

pub const Error = error{UnsupportedInstruction};

const simd = @import("simd.zig");
pub const Width = enum(u8) { w32 = 32, x64 = 64 };
pub const Size = enum(u8) { byte = 1, half = 2, word = 4, double = 8, quad = 16 };
/// How a load extends the bytes it read into its register: not at all, to 32 bits
/// (a `W` destination, whose upper half is then zero), or to the whole 64. Named
/// here because the decoder and the run loop that performs the load must agree.
pub const SignExtend = enum(u8) { none, to32, to64 };

pub const Wide = struct { width: Width, rd: u5, shift: u6, immediate: u16 };
/// Which way an add or subtract goes. Named so the decoder and the lowering
/// agree on one type rather than on two identical anonymous ones.
pub const ArithOp = enum { add, sub };
pub const Arithmetic = struct {
    op: ArithOp,
    flags: bool,
    width: Width,
    rn: u5,
    rd: u5,
    immediate: u32,
};
/// How a load or store finds its address. The two register forms differ from
/// the immediate ones in that the offset is not known until the access runs,
/// and the two indexed forms differ in when the base register is updated.
/// The immediate is signed because the unscaled and indexed forms allow a
/// negative displacement, and it is added to the base as a wrapping value.
pub const Addressing = union(enum) {
    /// `[Xn]`, or `[Xn, #imm]` with the immediate already scaled by the size.
    offset: i64,
    /// `[Xn, Xm]`, `[Xn, Xm, lsl #amount]`, or with a 32-bit index that is extended
    /// first, `[Xn, Wm, uxtw #amount]` and `[Xn, Wm, sxtw #amount]`. The amount is
    /// zero, or the log2 of the size of the access when the encoding scales the index.
    register: struct { rm: u5, extend: Extend, amount: u6 },
    /// `[Xn, #imm]!`, where the base is updated before the access.
    pre_index: i64,
    /// `[Xn], #imm`, where the base is updated after it.
    post_index: i64,
};
pub const Memory = struct {
    op: enum { load, store },
    size: Size,
    rn: u5,
    rt: u5,
    addressing: Addressing,
    /// For a load, whether the value read is sign-extended as it is put in the
    /// register, and to which width: `LDRSB`, `LDRSH` and `LDRSW`.
    signed: SignExtend = .none,
    /// Whether `rt` names a SIMD and floating point register instead of a general one.
    vector: bool = false,
};
/// A load or store of two adjacent registers. The pair is presented to the
/// memory system as two accesses of one element each, so the second register
/// sits at the address of the first plus the element size. `LDPSW` loads two
/// words and sign-extends each to 64 bits, so the extension rides along for
/// both halves.
pub const Pair = struct {
    op: enum { load, store },
    size: Size,
    rn: u5,
    rt: u5,
    rt2: u5,
    addressing: Addressing,
    signed: SignExtend = .none,
    /// Whether `rt` and `rt2` name SIMD and floating point registers.
    vector: bool = false,
};
/// A product, optionally added to or subtracted from an accumulator. Register 31
/// as the accumulator means there is none, which is how `MUL` and `MNEG` are
/// spelled.
pub const Multiply = struct {
    op: ArithOp,
    width: Width,
    rn: u5,
    rm: u5,
    ra: u5,
    rd: u5,
};
/// The widening multiplies, `SMADDL`, `SMSUBL`, `UMADDL` and `UMSUBL`, of which
/// `SMULL`, `UMULL`, `SMNEGL` and `UMNEGL` are the forms with no accumulator: two
/// 32-bit registers, each sign- or zero-extended, multiplied into a 64-bit result
/// that is added to or subtracted from a 64-bit accumulator. They exist only at 64
/// bits. Register 31 as the accumulator is zero.
pub const MultiplyLong = struct {
    signed: bool,
    op: ArithOp,
    rn: u5,
    rm: u5,
    ra: u5,
    rd: u5,
};
/// `SMULH` and `UMULH`: the upper 64 bits of the 128-bit product of two 64-bit
/// registers, which is how a compiler divides by a constant and detects overflow.
pub const MultiplyHigh = struct { signed: bool, rn: u5, rm: u5, rd: u5 };
/// `LDR` (literal): a load from an address relative to the instruction itself, which
/// is how code reaches a constant in its own literal pool, and how the kernel
/// fetches the address it jumps to once translation is on. Only the two plain
/// widths into a general register are decoded. The sign-extending word load needs
/// an extension the run loop does not perform on a load, the prefetch is a hint,
/// and the SIMD form is another register file.
pub const Literal = struct { size: Size, rt: u5, offset: i64, vector: bool = false };
/// A load-exclusive or a store-exclusive: `LDXR` and `STXR` at all four sizes, with
/// the acquire and release variants `LDAXR` and `STLXR`, which order memory and so
/// mean nothing on a coherent machine with one CPU. A store-exclusive reports in
/// `rs` whether it stored: zero if it did, one if it did not.
pub const Exclusive = struct { op: enum { load, store }, size: Size, rn: u5, rt: u5, rs: u5 };
/// The address of an instruction, either exactly (`ADR`) or to a page boundary
/// (`ADRP`), which is how position-independent code reaches its own data.
pub const Address = struct {
    /// True when the result is a page base rather than an exact address.
    page: bool,
    rd: u5,
    offset: i64,
};
/// The system registers this models. An instruction naming anything else is
/// refused rather than silently reading as zero, because a kernel that gets a
/// wrong answer from a control register corrupts itself in ways that are very
/// hard to trace back here.
pub const SystemRegister = enum {
    /// `SP_EL0`, `SP_EL1` and `SP_EL2`. These are not a register each but a
    /// view of the stack pointer belonging to one exception level, so the one
    /// that matches the current level is the live `sp`.
    sp_el0,
    sp_el1,
    sp_el2,
    current_el,
    spsr_el1,
    elr_el1,
    esr_el1,
    far_el1,
    vbar_el1,
    cpacr_el1,
    sctlr_el1,
    ttbr0_el1,
    ttbr1_el1,
    tcr_el1,
    mair_el1,
    tpidr_el0,
    tpidrro_el0,
    cntvct_el0,
    cntv_ctl_el0,
    cntv_cval_el0,
    /// `CNTV_TVAL_EL0`: the compare as a signed 32-bit distance from the counter, which is how a
    /// kernel arms the timer. It has no storage of its own.
    cntv_tval_el0,
    tpidr_el1,
    mdscr_el1,
    cntkctl_el1,
    oslar_el1,
    osdlr_el1,
    oslsr_el1,
    fpcr,
    fpsr,
    daif,
    /// The registers that describe the machine, which are read-only. Their values
    /// are in `Identification`, and writing one is refused.
    midr_el1,
    mpidr_el1,
    revidr_el1,
    ctr_el0,
    dczid_el0,
    cntfrq_el0,
    clidr_el1,
    id_aa64pfr0_el1,
    id_aa64pfr1_el1,
    id_aa64pfr2_el1,
    id_aa64zfr0_el1,
    id_aa64smfr0_el1,
    id_aa64fpfr0_el1,
    id_aa64isar3_el1,
    aidr_el1,
    id_aa64dfr0_el1,
    id_aa64dfr1_el1,
    id_aa64isar0_el1,
    id_aa64isar1_el1,
    id_aa64isar2_el1,
    id_aa64mmfr0_el1,
    id_aa64mmfr1_el1,
    id_aa64mmfr2_el1,
    id_aa64mmfr3_el1,
    id_aa64mmfr4_el1,
    /// The two forms that take a four-bit immediate rather than a register, and
    /// set or clear those bits of `daif` in one step. A kernel enables interrupts
    /// with the clear form, so both are needed to get as far as an interrupt.
    daif_set,
    daif_clear,
    /// The condition flags, which this keeps in `Cpu.flags` rather than in the
    /// control register block, so that it is one value rather than two.
    nzcv,

    /// The registers a write to which means nothing: the machine's description,
    /// the level the guest is running at, and the counter. A write is refused
    /// rather than dropped, because a guest that believes it changed one is wrong.
    pub fn readOnly(self: SystemRegister) bool {
        return switch (self) {
            .current_el,
            .midr_el1,
            .mpidr_el1,
            .revidr_el1,
            .ctr_el0,
            .dczid_el0,
            .cntfrq_el0,
            .clidr_el1,
            .id_aa64pfr0_el1,
            .id_aa64pfr1_el1,
            .id_aa64pfr2_el1,
            .id_aa64zfr0_el1,
            .id_aa64smfr0_el1,
            .id_aa64fpfr0_el1,
            .id_aa64isar3_el1,
            .aidr_el1,
            .id_aa64dfr0_el1,
            .id_aa64dfr1_el1,
            .id_aa64isar0_el1,
            .id_aa64isar1_el1,
            .id_aa64isar2_el1,
            .id_aa64mmfr0_el1,
            .id_aa64mmfr1_el1,
            .id_aa64mmfr2_el1,
            .id_aa64mmfr3_el1,
            .id_aa64mmfr4_el1,
            => true,
            else => false,
        };
    }
};
pub const System = struct {
    /// True for a read (`MRS`), false for a write (`MSR`).
    read: bool,
    register: SystemRegister,
    rt: u5,
    /// The mask for the two immediate forms, which carry it in the field that
    /// names the register for every other one.
    immediate: u4 = 0,
};
/// The register forms of the data-processing instructions take a shifted
/// register, where an immediate form takes a 12-bit constant. A rotate is only
/// a logical shift: no add or subtract names one, and decoding one there is
/// still reserved.
pub const Shift = enum { lsl, lsr, asr, ror };
pub const ArithmeticReg = struct {
    op: ArithOp,
    flags: bool,
    width: Width,
    rn: u5,
    rm: u5,
    rd: u5,
    shift: Shift,
    amount: u6,
};
pub const Condition = enum(u4) { eq, ne, cs, cc, mi, pl, vs, vc, hi, ls, ge, lt, gt, le, al, nv };

/// The bitwise instructions. `invert` is the encoding's N bit, which turns
/// AND into BIC, ORR into ORN, EOR into EON, and ANDS into BICS.
pub const LogicOp = enum { bit_and, bit_or, bit_xor };
pub const Logic = struct {
    op: LogicOp,
    flags: bool,
    width: Width,
    rn: u5,
    rm: u5,
    rd: u5,
    shift: Shift,
    amount: u6,
    invert: bool,
};
pub const LogicImm = struct {
    op: LogicOp,
    flags: bool,
    width: Width,
    rn: u5,
    rd: u5,
    /// The immediate as a full register, already replicated.
    immediate: u64,
};
/// Compare and branch on a register being zero, and test a bit. Neither reads
/// the flags, and neither writes a register.
pub const Test = struct { rt: u5, bit_index: ?u6, negate: bool, width: Width, offset: i64 };
/// A branch that also links, and the indirect branches through a register.
pub const Call = struct { target: i64, link: bool };
pub const Indirect = struct { rn: u5, link: bool };
/// The data-processing instructions with two register sources: divides, and the
/// shifts whose distance is in a register. Register 31 is the zero register in
/// all of them.
pub const VariableOp = enum { udiv, sdiv, lsl, lsr, asr, ror };
pub const Variable = struct { op: VariableOp, width: Width, rn: u5, rm: u5, rd: u5 };
/// `CCMP` and `CCMN`: compare (or compare negative) if the condition holds, and
/// otherwise load the flags from an immediate. `nzcv` is that immediate, N in its
/// top bit. The second operand is a register or a five-bit constant, which is how
/// a compiler chains two comparisons into one branch.
pub const CondCompare = struct {
    op: ArithOp,
    width: Width,
    rn: u5,
    operand: union(enum) { register: u5, immediate: u5 },
    cond: Condition,
    nzcv: u4,
};
/// `ADC`, `ADCS`, `SBC` and `SBCS`: an add or a subtract that also takes the carry
/// flag, which is how a sum wider than a register is carried from one word to the
/// next. `NGC` is `SBC` with the zero register as its first operand. Register 31 is
/// the zero register in every position, and never the stack pointer.
pub const AddCarry = struct { op: ArithOp, flags: bool, width: Width, rn: u5, rm: u5, rd: u5 };
/// `ADD` and `SUB` (extended register): the second operand is the low byte, half,
/// word or all of a register, zero- or sign-extended, and then shifted left by up
/// to four. Its point is the stack pointer, which the shifted-register form cannot
/// name: register 31 as a first operand is `sp`, and as a destination is `sp` unless
/// the flags are set, when it is the zero register. It is also how a 32-bit index
/// becomes a 64-bit offset (`add x0, x1, w2, sxtw #2`).
pub const Extend = enum(u3) { uxtb, uxth, uxtw, uxtx, sxtb, sxth, sxtw, sxtx };
pub const ArithmeticExtended = struct {
    op: ArithOp,
    flags: bool,
    width: Width,
    rn: u5,
    rm: u5,
    rd: u5,
    extend: Extend,
    amount: u3,
};
/// The data-processing instructions with one register source, which rearrange or
/// count its bits: bit reversal, the three byte reversals (of each halfword, of each
/// word, and of the whole register), and counting leading zeros. `CLS` and the
/// pointer-authentication instructions share the class and are refused.
pub const UnaryOp = enum { rbit, rev16, rev32, rev, clz };
pub const Unary = struct { op: UnaryOp, width: Width, rn: u5, rd: u5 };
/// `EXTR`: a register's worth of bits taken from the pair `Rn:Rm` starting `lsb` up
/// from the bottom of `Rm`. With the same register twice it is a rotate right by an
/// immediate, which is how `ROR` by a constant is spelled.
pub const Extract = struct { width: Width, rn: u5, rm: u5, rd: u5, lsb: u6 };
/// `UBFM` zero-extends the field, `SBFM` sign-extends it, and `BFM` inserts it into
/// the destination and leaves the rest of that alone. `extract` says which end of
/// the encoding it was: a field taken from the source at `lsb`, or one taken from
/// the bottom of the source and placed at `lsb` in the result. `len` is 1 to 64.
pub const BitfieldOp = enum { unsigned, signed, insert };
pub const Bitfield = struct {
    op: BitfieldOp,
    extract: bool,
    width: Width,
    rn: u5,
    rd: u5,
    lsb: u6,
    len: u7,
};
pub const Instruction = union(enum) {
    /// A SIMD or floating point instruction, which `simd.zig` knows how to lift.
    simd: simd.Op,
    movz: Wide,
    movn: Wide,
    movk: Wide,
    arith_imm: Arithmetic,
    arith_reg: ArithmeticReg,
    arith_ext: ArithmeticExtended,
    logic_reg: Logic,
    logic_imm: LogicImm,
    test_branch: Test,
    call: Call,
    indirect: Indirect,
    memory: Memory,
    pair: Pair,
    mul: Multiply,
    mul_long: MultiplyLong,
    mul_high: MultiplyHigh,
    literal: Literal,
    exclusive: Exclusive,
    /// `CLREX`: forget any address a load-exclusive claimed.
    clrex,
    adr: Address,
    system: System,
    /// `DC ZVA, Xt`: zero the line holding the address in `Xt`.
    dc_zva: u5,
    /// `ERET`: return from an exception.
    eret,
    /// A cache maintenance instruction (`DC` other than `ZVA`, and `IC`). Memory is
    /// coherent here and a translation is chosen by the bytes it came from, so
    /// there is nothing to clean or invalidate and the instruction does nothing.
    cache_op,
    /// A `TLBI`. It ends the block, so the run loop can drop what it remembers of
    /// the translation tables before the next instruction is fetched.
    tlbi,
    nop,
    /// The barriers, which order nothing here and so do nothing.
    dsb,
    dmb,
    isb,
    sev,
    sevl,
    wfe,
    yield,
    /// A `BRK`: the guest has reached a trap it cannot recover from.
    trap,
    b: i64,
    b_cond: struct { cond: Condition, offset: i64 },
    /// The `CSEL` family. `opc` is the architectural selector over what the
    /// instruction produces when the condition fails: `Rm`, `Rm+1`, `~Rm`, `-Rm`.
    csel: struct { cond: Condition, width: Width, rn: u5, rm: u5, rd: u5, opc: u2 },
    variable: Variable,
    cond_compare: CondCompare,
    arith_carry: AddCarry,
    unary: Unary,
    bitfield: Bitfield,
    extract: Extract,
    svc,
    psci,
    wfi,

    pub fn terminates(self: Instruction) bool {
        return switch (self) {
            .movz, .movn, .movk, .arith_imm, .arith_reg, .arith_ext, .arith_carry, .logic_reg, .logic_imm, .csel, .cond_compare, .variable, .unary, .bitfield, .extract, .mul, .mul_long, .mul_high, .adr, .system, .cache_op, .clrex, .nop, .dsb, .dmb, .sev, .sevl, .yield, .simd => false,
            .memory, .exclusive, .literal, .pair, .b, .b_cond, .test_branch, .call, .indirect, .svc, .psci, .wfi, .wfe, .trap, .isb, .dc_zva, .eret, .tlbi => true,
        };
    }
};

/// Placeholder until the immediate form is fitted against the assembler.
/// Unpack the architecture's "bit pattern immediate" into the value it names.
///
/// The encoding does not store the pattern. It stores three numbers: the width
/// of one element of the pattern, how long a run of ones is, and how far that
/// run is rotated within the element. The pattern is a run of ones that may
/// wrap around the end of its element, and a pattern narrower than the register
/// repeats until the register is full. An element only two bits wide or wider
/// can name a pattern; anything smaller is left to MOVN, which is the same
/// fields describing an inverted pattern instead.
fn bitmask(word: u32, width: Width) Error!u64 {
    const n: u1 = @truncate((word >> 22) & 1);
    const immr: u6 = @truncate((word >> 16) & 0x3f);
    const imms: u6 = @truncate((word >> 10) & 0x3f);

    // Read as one number: N above the inverse of imms. Its highest bit is the
    // element width as a power of two, with N standing in for the bit that only
    // a 64-bit pattern has room for.
    const combined: u7 = (@as(u7, n) << 6) | (~imms & 0x3f);
    if (combined == 0) return error.UnsupportedInstruction;
    const esize: u7 = @as(u7, 1) << @intCast(6 - @clz(combined));
    const levels = esize - 1;
    // A run as long as the element is the whole element, which is not a pattern
    // this form can name.
    if (imms & levels == levels) return error.UnsupportedInstruction;

    // Both fields are read modulo the element width, which is what keeps a
    // rotation from pulling in bits from beyond the element.
    const run = imms & levels;
    const rotation = immr & levels;
    // The run is `run + 1` ones wide, one bit short of the element, and the
    // rotation is taken within the element so a run at the top wraps around to
    // the bottom instead of running off the end. Rotating within the register
    // instead would be wrong for every element narrower than the register,
    // because the bits would cross into the neighbouring element.
    const ones: u64 = (@as(u64, 1) << @intCast(run + 1)) - 1;
    const welem: u64 = if (rotation == 0) ones else blk: {
        // A 64-bit element is the whole register, so there is nothing to mask
        // off and nothing for a rotation to cross into.
        const within: u64 = if (esize == 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(esize)) - 1;
        break :blk ((ones >> @intCast(rotation)) |
            (ones << @intCast(esize - rotation))) & within;
    };

    // A pattern narrower than the register repeats across it, and that repeated
    // form is the immediate itself.
    const d_bits: u7 = if (width == .x64) 64 else 32;
    var immediate: u64 = 0;
    var offset: u7 = 0;
    while (offset < d_bits) : (offset += esize) {
        immediate |= welem << @intCast(offset);
    }
    return immediate;
}

/// The bitwise instructions number their operation 00, 01, 10 and set the flags
/// with 11, which is the same operation with the `S` that `ADD` also carries.
fn logicOp(word: u32) LogicOp {
    return switch (@as(u2, @truncate(word >> 29))) {
        0, 3 => LogicOp.bit_and,
        1 => LogicOp.bit_or,
        2 => LogicOp.bit_xor,
    };
}

fn logicSetsFlags(word: u32) bool {
    return @as(u2, @truncate(word >> 29)) == 3;
}

/// The signed nine-bit displacement that the unscaled and indexed forms share,
/// taken from bits 20:12.
/// The register a system instruction names, or an error when it names one this
/// does not model. The five fields are the architectural ones: a three-bit
/// `op1` that also carries the direction and the exception level, then the two
/// register numbers and the operation within them.
/// The register a system instruction names, keyed on all four architectural
/// fields rather than on a prefix of them. Every entry is a word the assembler
/// emits; a name this does not carry is refused, because a kernel that reads a
/// control register as zero and carries on will fail much later and much less
/// clearly than it would here.
fn systemRegister(op1: u3, crn: u4, crm: u4, op2: u3) Error!SystemRegister {
    // The two immediate forms share this register number with `NZCV` and `DAIF`
    // and are told apart by the operation alone, because the field that would
    // name the register carries the mask instead.
    if (op1 == 3 and crn == 4) {
        if (op2 == 6) return .daif_set;
        if (op2 == 7) return .daif_clear;
    }
    const key: u16 = @as(u16, op1) << 12 | @as(u16, crn) << 8 | @as(u16, crm) << 4 | op2;
    return switch (key) {
        0x0410 => .sp_el0,
        0x4410 => .sp_el1,
        0x6410 => .sp_el2,
        0x0422 => .current_el,
        0x0400 => .spsr_el1,
        0x0401 => .elr_el1,
        0x0520 => .esr_el1,
        0x0600 => .far_el1,
        0x0c00 => .vbar_el1,
        0x0102 => .cpacr_el1,
        0x0100 => .sctlr_el1,
        0x0200 => .ttbr0_el1,
        0x0201 => .ttbr1_el1,
        0x0202 => .tcr_el1,
        0x0a20 => .mair_el1,
        0x3d02 => .tpidr_el0,
        0x3d03 => .tpidrro_el0,
        0x3e02 => .cntvct_el0,
        0x3e31 => .cntv_ctl_el0,
        0x3e32 => .cntv_cval_el0,
        0x3e30 => .cntv_tval_el0,
        0x0d04 => .tpidr_el1,
        0x0022 => .mdscr_el1,
        0x0104 => .oslar_el1,
        0x0134 => .osdlr_el1,
        0x0114 => .oslsr_el1,
        0x0e10 => .cntkctl_el1,
        0x3440 => .fpcr,
        0x3441 => .fpsr,
        0x3421 => .daif,
        0x0000 => .midr_el1,
        0x0005 => .mpidr_el1,
        0x0006 => .revidr_el1,
        0x3001 => .ctr_el0,
        0x3007 => .dczid_el0,
        0x3e00 => .cntfrq_el0,
        0x1001 => .clidr_el1,
        0x0040 => .id_aa64pfr0_el1,
        0x0041 => .id_aa64pfr1_el1,
        0x0042 => .id_aa64pfr2_el1,
        0x0044 => .id_aa64zfr0_el1,
        0x0045 => .id_aa64smfr0_el1,
        0x0047 => .id_aa64fpfr0_el1,
        0x0063 => .id_aa64isar3_el1,
        0x1007 => .aidr_el1,
        0x0050 => .id_aa64dfr0_el1,
        0x0051 => .id_aa64dfr1_el1,
        0x0060 => .id_aa64isar0_el1,
        0x0061 => .id_aa64isar1_el1,
        0x0062 => .id_aa64isar2_el1,
        0x0070 => .id_aa64mmfr0_el1,
        0x0071 => .id_aa64mmfr1_el1,
        0x0072 => .id_aa64mmfr2_el1,
        0x0073 => .id_aa64mmfr3_el1,
        0x0074 => .id_aa64mmfr4_el1,
        0x3420 => .nzcv,

        else => error.UnsupportedInstruction,
    };
}

fn signExtend9(word: u32) i64 {
    return @as(i9, @bitCast(@as(u9, @truncate(word >> 12))));
}

/// The bitfield moves: `UBFM`, `SBFM` and `BFM`, which are what `UBFX`, `SBFX`,
/// `UBFIZ`, `SBFIZ`, `BFI`, `BFXIL`, the sign and zero extensions, and the shifts
/// by a constant are all spelled as. One instruction takes a run of bits from one
/// place and puts it in another, and the two fields `immr` and `imms` say where,
/// which is unpacked here into a position and a length so nothing downstream has
/// to remember the architecture's encoding of them.
fn bitfield(word: u32) Error!Instruction {
    const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
    // The N bit repeats the width, and disagreeing with it is a reserved encoding.
    if (((word >> 22) & 1) != @intFromBool(width == .x64)) return error.UnsupportedInstruction;
    const op: BitfieldOp = switch ((word >> 29) & 3) {
        0 => .signed,
        1 => .insert,
        2 => .unsigned,
        else => return error.UnsupportedInstruction,
    };
    const immr: u7 = @truncate((word >> 16) & 0x3f);
    const imms: u7 = @truncate((word >> 10) & 0x3f);
    const bits: u7 = @intCast(@backingInt(width));
    // A 32-bit form cannot name a bit past 31, and that encoding is reserved.
    if (immr >= bits or imms >= bits) return error.UnsupportedInstruction;

    // A run that ends at or above where it starts is taken from the source at
    // `immr`. One that ends below is a run of `imms + 1` bits from the bottom of
    // the source, placed `bits - immr` up in the destination.
    const extract = imms >= immr;
    return .{ .bitfield = .{
        .op = op,
        .extract = extract,
        .width = width,
        .rn = @truncate(word >> 5),
        .rd = @truncate(word),
        .lsb = @intCast(if (extract) immr else bits - immr),
        .len = if (extract) imms - immr + 1 else imms + 1,
    } };
}

/// The address of a load or store of one register, in all four of its forms. It does not depend on
/// whether the register is a general one or a vector one; the size of the access does.
fn singleAddressing(word: u32, size: Size) Error!Addressing {
    return if ((word & 0x0100_0000) != 0) blk: {
        // Here bits 11:10 belong to the twelve-bit immediate rather than
        // choosing a form, so the whole field is the offset.
        break :blk .{ .offset = @intCast(@as(u64, (word >> 10) & 0xfff) * @backingInt(size)) };
    } else if ((word & 0x0020_0000) != 0) blk: {
        // The offset is a register, and bits 11:10 being `10` is what says so. The
        // option field says how it is extended, and the `S` bit whether it is scaled.
        if ((word & 0x0000_0c00) != 0x0000_0800) return error.UnsupportedInstruction;
        // Four of the eight options are allocated for an index: the 32-bit ones,
        // zero- or sign-extended, and the 64-bit ones, of which `lsl` is UXTX.
        const extend: Extend = switch ((word >> 13) & 7) {
            2 => .uxtw,
            3 => .uxtx,
            6 => .sxtw,
            7 => .sxtx,
            else => return error.UnsupportedInstruction,
        };
        // The `S` bit says the index is scaled by the size of the access, so the
        // shift is the log2 of that size (2 for a word, 3 for a doubleword), not a
        // constant. Without it there is no shift at all.
        const scaled = (word & 0x0000_1000) != 0;
        break :blk .{ .register = .{
            .rm = @truncate(word >> 16),
            .extend = extend,
            .amount = if (scaled) @intCast(@ctz(@backingInt(size))) else 0,
        } };
    } else switch ((word >> 10) & 3) {
        // `2` is the unprivileged form (LDTR/STTR), which Linux uses for user memory. It
        // addresses like the unscaled form. Its access is not yet checked as EL0.
        0, 2 => .{ .offset = signExtend9(word) },
        1 => .{ .post_index = signExtend9(word) },
        3 => .{ .pre_index = signExtend9(word) },
        else => unreachable, // masked to two bits above
    };
}

pub fn decode(word: u32) Error!Instruction {
    // Move wide, in its three forms, which the two bits at 30:29 tell apart: MOVN
    // writes the complement of the shifted constant, MOVZ the constant with zeros
    // around it, and MOVK replaces only its own halfword. The value 1 is unallocated.
    if ((word & 0x1f80_0000) == 0x1280_0000) {
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        const hw: u2 = @truncate(word >> 21);
        if (width == .w32 and hw > 1) return error.UnsupportedInstruction;
        const wide: Wide = .{
            .width = width,
            .rd = @truncate(word),
            .shift = @as(u6, hw) * 16,
            .immediate = @truncate(word >> 5),
        };
        return switch ((word >> 29) & 3) {
            0 => .{ .movn = wide },
            2 => .{ .movz = wide },
            3 => .{ .movk = wide },
            else => error.UnsupportedInstruction,
        };
    }
    // ADD/SUB (extended register). Bit 21 is what separates it from the shifted
    // register form below, and bits 23:22, which are a shift type there, must be
    // zero here. The shift amount is three bits and only 0 to 4 are allocated.
    if ((word & 0x1fe0_0000) == 0x0b20_0000) {
        const amount: u3 = @truncate(word >> 10);
        if (amount > 4) return error.UnsupportedInstruction;
        return .{ .arith_ext = .{
            .op = if ((word & 0x4000_0000) != 0) .sub else .add,
            .flags = (word & 0x2000_0000) != 0,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
            .extend = @fromBackingInt(@intCast(@as(u3, @truncate(word >> 13)))),
            .amount = amount,
        } };
    }
    // ADD/SUB (shifted register), with the flag-setting bit left out of the
    // class so that the `S` forms land here too. Bit 21 is the form: zero here is
    // the shifted register, which is the only one of this class that is decoded.
    if ((word & 0x5f20_0000) == 0x0b00_0000 or (word & 0x5f20_0000) == 0x4b00_0000) {
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        const shift: Shift = switch ((word >> 22) & 3) {
            0 => .lsl,
            1 => .lsr,
            2 => .asr,
            else => return error.UnsupportedInstruction, // a rotate (3) is only a logical shift
        };
        // The field is the shift itself, in every one of the three types. A 32-bit
        // register cannot shift by 32 or more, and a right shift of zero is the
        // full-width shift, which no assembler emits, so both are refused rather
        // than wrapped into something the guest did not ask for.
        const amount: u6 = @truncate(word >> 10);
        if (shift != .lsl and amount == 0) return error.UnsupportedInstruction;
        if (width == .w32 and amount >= 32) return error.UnsupportedInstruction;
        return .{ .arith_reg = .{
            .op = if ((word & 0x4000_0000) != 0) .sub else .add,
            .flags = (word & 0x2000_0000) != 0,
            .width = width,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
            .shift = shift,
            .amount = amount,
        } };
    }
    // The bitfield class, which is the shifts by a constant. It is tested before
    // the bitwise immediates because their class masks must not overlap: the two
    // differ only in bit 24, and testing fewer bits made this one look like that.
    if ((word & 0x1f80_0000) == 0x1300_0000) return bitfield(word);
    // `EXTR`, beside the bitfield class: bits 28:23 are `100111`, one bit above the
    // bitfield's. The width bit and N must agree, and bit 21 and the two bits at
    // 30:29 are zero. A 32-bit form names a bit position below 32, and anything
    // above that is reserved.
    if ((word & 0x7fa0_0000) == 0x1380_0000) {
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        if (((word >> 22) & 1) != @intFromBool(width == .x64)) return error.UnsupportedInstruction;
        const lsb: u6 = @truncate(word >> 10);
        if (width == .w32 and lsb >= 32) return error.UnsupportedInstruction;
        return .{ .extract = .{
            .width = width,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
            .lsb = lsb,
        } };
    }
    // The bitwise instructions, in both their immediate and their shifted
    // register form. The immediate form is a bit pattern that the architecture
    // stores as a run of ones, a rotation, and a length, so it is unpacked here.
    // The class is bits 28:23; the operation is the two bits above them and the
    // flag-setting bit is one of those, so neither may be part of the test.
    // Getting this wrong hides every `ANDS` immediate, which is also how `TST`
    // is spelled. The whole of 28:23 is tested, which also keeps `MOVN` (one bit
    // away) from being read as a bitmask.
    if ((word & 0x1f80_0000) == 0x1200_0000) {
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        return .{ .logic_imm = .{
            .op = logicOp(word),
            .flags = logicSetsFlags(word),
            .immediate = try bitmask(word, width),
            .width = width,
            .rn = @truncate(word >> 5),
            .rd = @truncate(word),
        } };
    }
    if ((word & 0x1f00_0000) == 0x0a00_0000) {
        const shift: Shift = switch ((word >> 22) & 3) {
            0 => .lsl,
            1 => .lsr,
            2 => .asr,
            else => .ror,
        };
        const amount: u6 = @truncate(word >> 10);
        if (shift != .lsl and shift != .ror and amount == 0) return error.UnsupportedInstruction;
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        if (width == .w32 and amount >= 32) return error.UnsupportedInstruction;
        return .{ .logic_reg = .{
            .op = logicOp(word),
            .flags = logicSetsFlags(word),
            .width = width,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
            .shift = shift,
            .amount = amount,
            .invert = (word & 0x0020_0000) != 0,
        } };
    }
    // Compare and branch: bit 4 says whether the test is against zero or not.
    if ((word & 0x7e00_0000) == 0x3400_0000) {
        return .{ .test_branch = .{
            .rt = @truncate(word),
            .bit_index = null,
            .negate = (word & 0x0100_0000) != 0,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .offset = @as(i64, @as(i19, @bitCast(@as(u19, @truncate(word >> 5))))) << 2,
        } };
    }
    // Test a bit: the same class, with the bit number split across two fields.
    if ((word & 0x7e00_0000) == 0x3600_0000) {
        // The bit number is b5 at 31 joined to b40 at 23:19, and bit 31 doubles
        // as the width, so the two always agree about the register.
        const bit: u6 = @as(u6, @intCast((word >> 31) & 1)) << 5 | @as(u6, @truncate((word >> 19) & 0x1f));
        return .{ .test_branch = .{
            .rt = @truncate(word),
            .bit_index = bit,
            .negate = (word & 0x0100_0000) != 0,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .offset = @as(i64, @as(i14, @bitCast(@as(u14, @truncate(word >> 5))))) << 2,
        } };
    }
    if ((word & 0xfc00_0000) == 0x9400_0000) {
        return .{ .call = .{
            .target = @as(i64, @as(i26, @bitCast(@as(u26, @truncate(word))))) << 2,
            .link = true,
        } };
    }
    // The indirect branches. Register 30 is the link register, so `ret` is this
    // class with that operand, and nothing else distinguishes it.
    if ((word & 0xfe00_0000) == 0xd600_0000) {
        if ((word & 0x001f_0000) != 0x001f_0000) return error.UnsupportedInstruction;
        const opc = (word >> 21) & 7;
        return switch (opc) {
            0 => .{ .indirect = .{ .rn = @truncate(word >> 5), .link = false } },
            1 => .{ .indirect = .{ .rn = @truncate(word >> 5), .link = true } },
            2 => .{ .indirect = .{ .rn = @truncate(word >> 5), .link = false } },
            // ERET has no register operand, and the assembler writes 31 there.
            4 => if ((word & 0x0000_03e0) == 0x0000_03e0) .eret else error.UnsupportedInstruction,
            else => error.UnsupportedInstruction,
        };
    }
    if ((word & 0x1f00_0000) == 0x1100_0000) {
        if ((word & 0x0080_0000) != 0) return error.UnsupportedInstruction; // reserved shift
        const shift: u5 = if ((word & 0x0040_0000) != 0) 12 else 0;
        const arithmetic: Arithmetic = .{
            .op = if ((word & 0x4000_0000) != 0) .sub else .add,
            .flags = (word & 0x2000_0000) != 0,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rd = @truncate(word),
            .immediate = ((word >> 10) & 0xfff) << shift,
        };
        return .{ .arith_imm = arithmetic };
    }
    // The prefetches: `PRFM` with an immediate, a register, or an unscaled offset, and
    // with a literal. A prefetch is a hint that changes no state and cannot fault, so
    // it is nothing here. It is tested before the loads because it lives in their
    // opcode space, where a doubleword with opcode 2 would otherwise be refused.
    if ((word & 0xffc0_0000) == 0xf980_0000 or
        (word & 0xffe0_0c00) == 0xf8a0_0800 or
        (word & 0xffe0_0c00) == 0xf880_0000 or
        (word & 0xff00_0000) == 0xd800_0000) return .nop;
    // The literal load. The class is bits 29:24 with the SIMD bit, 26, clear, and the
    // size is the two bits at 31:30: 0 is a word, 1 a doubleword, and the other two
    // are the sign-extending load and the prefetch. The offset is the 19 bits at
    // 23:5, signed, and scaled by four because instructions are that long.
    if ((word & 0x3f00_0000) == 0x1800_0000) {
        const size: Size = switch (word >> 30) {
            0 => .word,
            1 => .double,
            else => return error.UnsupportedInstruction,
        };
        return .{ .literal = .{
            .size = size,
            .rt = @truncate(word),
            .offset = @as(i64, @as(i19, @bitCast(@as(u19, @truncate(word >> 5))))) << 2,
        } };
    }
    // The exclusive and ordered loads and stores, which share a class with the pair
    // and compare-and-swap forms. Bit 21 says one of those two, and neither is
    // decoded. Bit 23 says the access is ordered and not exclusive: that is the
    // `LDAR`, `STLR`, `LDAPR` and `LDAPUR` family, which are plain accesses here
    // because there is nothing to order. Bit 15 is one for the release forms and
    // zero for the acquire ones, and both are accepted. Bit 22 says load. For
    // every form the field for a second register is unused and must be 31, and so
    // must the status register of anything that is not a store-exclusive.
    if ((word & 0x3f00_0000) == 0x0800_0000) {
        const size: Size = @fromBackingInt(@intCast(@as(u8, 1) << @intCast(word >> 30)));
        const load = (word & 0x0040_0000) != 0;
        const rn: u5 = @truncate(word >> 5);
        const rt: u5 = @truncate(word);
        const rs: u5 = @truncate(word >> 16);
        if (@as(u5, @truncate(word >> 10)) != 31) return error.UnsupportedInstruction;
        if ((word & 0x0080_0000) != 0) {
            if (rs != 31) return error.UnsupportedInstruction;
            // Bit 15 clear with bit 23 set on a load is the limited-ordering
            // `LDLAR`, which stays refused. Clear with bit 23 clear is the
            // acquire `LDAPR`, which is the same plain access.
            if (load and (word & 0x0000_8000) == 0 and ((word >> 23) & 1) == 1) return error.UnsupportedInstruction;
            return .{ .memory = .{
                .op = if (load) .load else .store,
                .size = size,
                .rn = rn,
                .rt = rt,
                .addressing = .{ .offset = 0 },
            } };
        }
        if (load and rs != 31) return error.UnsupportedInstruction;
        return .{ .exclusive = .{ .op = if (load) .load else .store, .size = size, .rn = rn, .rt = rt, .rs = rs } };
    }
    // The SIMD and floating point loads and stores of one register. They share every
    // addressing form with the general ones. The size is the two bits at 31:30, except
    // that a 128-bit access has size 0 and the top bit of `opc` set, which is why `opc`
    // above 1 is a quad here and a sign extension in the general class.
    if ((word & 0x3f00_0000) == 0x3d00_0000 or (word & 0x3f00_0000) == 0x3c00_0000) {
        const opc = (word >> 22) & 3;
        const size: Size = if (opc >= 2)
            (if ((word >> 30) == 0) .quad else return error.UnsupportedInstruction)
        else
            @fromBackingInt(@intCast(@as(u8, 1) << @intCast(word >> 30)));
        return .{ .memory = .{
            .op = if (opc & 1 == 0) .store else .load,
            .size = size,
            .rn = @truncate(word >> 5),
            .rt = @truncate(word),
            .addressing = try singleAddressing(word, size),
            .vector = true,
        } };
    }
    // The pair forms, and the literal form, of the same registers.
    if ((word & 0x3e00_0000) == 0x2c00_0000) {
        const opc: u2 = @truncate(word >> 30);
        const size: Size = switch (opc) {
            0 => .word,
            1 => .double,
            2 => .quad,
            else => return error.UnsupportedInstruction,
        };
        const scale: i64 = @backingInt(size);
        const raw: i7 = @bitCast(@as(u7, @truncate(word >> 15)));
        const displacement = @as(i64, raw) * scale;
        const addressing: Addressing = switch ((word >> 23) & 7) {
            0, 2 => .{ .offset = displacement },
            3 => .{ .pre_index = displacement },
            1 => .{ .post_index = displacement },
            else => return error.UnsupportedInstruction,
        };
        return .{ .pair = .{
            .op = if ((word & 0x0040_0000) != 0) .load else .store,
            .size = size,
            .rn = @truncate(word >> 5),
            .rt = @truncate(word),
            .rt2 = @truncate(word >> 10),
            .addressing = addressing,
            .vector = true,
        } };
    }
    if ((word & 0x3f00_0000) == 0x1c00_0000) {
        const size: Size = switch (word >> 30) {
            0 => .word,
            1 => .double,
            2 => .quad,
            else => return error.UnsupportedInstruction,
        };
        return .{ .literal = .{
            .size = size,
            .rt = @truncate(word),
            .offset = @as(i64, @as(i19, @bitCast(@as(u19, @truncate(word >> 5))))) << 2,
            .vector = true,
        } };
    }
    // The load and store class for one register, in all four of its addressing
    // forms. Bit 24 picks the scaled twelve-bit offset from the rest, and within
    // those bit 21 picks a register offset, leaving the three immediate forms to
    // be told apart by bits 11:10.
    if ((word & 0x3e00_0000) == 0x3800_0000) {
        const opc = (word >> 22) & 3;
        const size: Size = @fromBackingInt(@intCast(@as(u8, 1) << @intCast(word >> 30)));
        // `opc` 2 with a doubleword is the prefetch (`PRFM`, `PRFUM`), which is a
        // hint and not a load.
        if (opc == 2 and size == .double) return .nop;
        // `opc` 0 is a store and 1 a load. 2 and 3 are the sign-extending loads, to a
        // 64-bit and a 32-bit register, except that a word can only be extended to 64
        // bits. The extension is a property of the access, not of the address, so
        // it is settled here before the addressing form is.
        const signed: SignExtend = switch (opc) {
            0, 1 => .none,
            2 => .to64,
            3 => if (size == .byte or size == .half) .to32 else return error.UnsupportedInstruction,
            else => unreachable,
        };
        const addressing = try singleAddressing(word, size);
        return .{ .memory = .{
            .op = if (opc == 0) .store else .load,
            .size = size,
            .rn = @truncate(word >> 5),
            .rt = @truncate(word),
            .addressing = addressing,
            .signed = signed,
        } };
    }
    // The hint instructions. A hint has no effect on the machine state, so `NOP`
    // falls through as nothing at all, and the rest are where a compiler puts a
    // breakpoint or waits for an interrupt. The hint number is the seven bits at
    // 11:5, and the architecture's numbering is the one used here: YIELD 1, WFE 2,
    // WFI 3, SEV 4, SEVL 5.
    //
    // Every other number is a `NOP`, and the architecture says so in as many words:
    // a hint that is unallocated, or that belongs to a feature this machine does not
    // have, executes as nothing. That covers the branch target identification hints
    // (32, 34, 36, 38), the pointer authentication ones (7 and 24 to 31), error and
    // trace synchronisation (16 to 18), and the speculation barrier (20), and every
    // one of them is a feature the guest is told is not there.
    if ((word & 0xffff_f01f) == 0xd503_201f) {
        return switch ((word >> 5) & 0x7f) {
            0x01 => .yield,
            0x02 => .wfe,
            0x03 => .wfi,
            0x04 => .sev,
            0x05 => .sevl,
            else => .nop,
        };
    }
    // The barriers. A kernel issues these after changing a control register, and
    // they ask the hardware to order memory against that change. A translating
    // machine has nothing to order, so DSB and DMB are nothing here. The operand
    // in bits 11:8 is which domain and which accesses, and `op2` says which barrier,
    // so the operand is left out of the test: `dsb nsh` and `dmb ishst` are as much
    // these as `dsb sy` is. The hints above use register number 2, these use 3.
    // `CLREX` shares the class, with `op2` 2, and takes an operand that says nothing
    // here. `SB` shares it too and is refused until something models it.
    if ((word & 0xffff_f01f) == 0xd503_301f) {
        return switch ((word >> 5) & 7) {
            2 => .clrex,
            4 => .dsb,
            5 => .dmb,
            6 => .isb,
            else => error.UnsupportedInstruction,
        };
    }
    // `DC ZVA` is `SYS #3, C7, C4, #1, Xt`. The other cache instructions share
    // the class and differ only in these fields; they only ask for ordering,
    // which nothing here needs, so they stay refused. It is tested first
    // because the class mask below would otherwise claim it.
    if ((word & 0xffff_ffe0) == 0xd50b_7420) return .{ .dc_zva = @truncate(word) };
    // The other system instructions (`SYS`, which is a write with no result). Cache
    // maintenance is the seven `DC` and `IC` operations a kernel issues around page
    // tables and DMA, and every one of them does nothing here. `TLBI` is the whole
    // of the class numbered 8. Address translation instructions (`AT`) and anything
    // else in the class are refused. It is tested before the register moves below
    // because that class mask would otherwise claim these too.
    if ((word & 0xfff8_0000) == 0xd508_0000) {
        const crn: u4 = @truncate(word >> 12);
        const crm: u4 = @truncate(word >> 8);
        if (crn == 8) return .tlbi;
        if (crn == 7) switch (crm) {
            1, 5, 6, 10, 11, 14 => return .cache_op,
            else => {},
        };
        return error.UnsupportedInstruction;
    }
    // The system register moves. The register is named by five fields, and
    // `systemRegister` is the whole of what is modelled; anything else is
    // refused. Reading one this does not know must not quietly yield zero,
    // because a kernel told its own page table base is zero will simply stop.
    if ((word & 0xffc0_0000) == 0xd500_0000) {
        const read = (word & 0x0020_0000) != 0;
        const register = try systemRegister(@truncate(word >> 16), @truncate(word >> 12), @truncate(word >> 8), @truncate(word >> 5));
        if (!read and register.readOnly()) return error.UnsupportedInstruction;
        // The first of the five fields, which the register key leaves out, is part of
        // the name: the debug registers are 2, the immediate forms of `MSR` are 0
        // and are writes, and everything else here is 3. Without it a word in one
        // space reads as a register in another that shares the rest of its fields.
        const op0: u2 = @truncate(word >> 19);
        const expected_op0: u2 = switch (register) {
            .daif_set, .daif_clear => 0,
            .mdscr_el1, .oslar_el1, .osdlr_el1, .oslsr_el1 => 2,
            else => 3,
        };
        if (op0 != expected_op0) return error.UnsupportedInstruction;
        if (read and expected_op0 == 0) return error.UnsupportedInstruction;
        return .{
            .system = .{
                // The two instructions differ only in this bit: a read has it set
                // and a write does not.
                .read = read,
                .register = register,
                .rt = @truncate(word),
                // Only the immediate forms use it, and for them it is this field.
                .immediate = @truncate(word >> 8),
            },
        };
    }
    if ((word & 0xffe0_f000) == 0xd420_0000) {
        return switch (word & 0x0000_ffe0) {
            0x0000_0020 => .trap,
            else => error.UnsupportedInstruction,
        };
    }
    // `ADR` and `ADRP`. The displacement is one 21-bit field, split across the
    // word as the two ends of the high half, and scaled by a byte or a page.
    if ((word & 0x1f00_0000) == 0x1000_0000) {
        const page = (word & 0x8000_0000) != 0;
        // The halves are joined into a 21-bit field: the low two bits sit at 30:29
        // and the high nineteen at 23:5. The high half is widened to the full
        // field before the shift, because shifting it in its own nineteen-bit
        // width would drop the bits the shift moves up. The low half is read
        // through an explicit mask, because at 30:29 the next bit up is the one
        // that says which of the two forms this is.
        const high: u21 = @truncate(word >> 5);
        const low: u21 = @truncate((word >> 29) & 0b11);
        const wide: u21 = (high << 2) | low;
        const signed: i21 = @bitCast(wide);
        const scale: i64 = if (page) 4096 else 1;
        return .{ .adr = .{
            .page = page,
            .rd = @truncate(word),
            .offset = @as(i64, signed) * scale,
        } };
    }
    // The widening and high-half multiplies. They share the class of the plain ones
    // and are told apart by bits 23:21: 001 and 101 are the signed and unsigned long
    // multiply-adds, and 010 and 110 the high halves. All of them are 64-bit only, so
    // the width bit is part of the test. The high halves have no accumulator, and
    // that is spelled by register 31 in its field and a clear subtract bit, so
    // anything else there is not one of them.
    if ((word & 0xff00_0000) == 0x9b00_0000) {
        const form = (word >> 21) & 7;
        const rn: u5 = @truncate(word >> 5);
        const rm: u5 = @truncate(word >> 16);
        const ra: u5 = @truncate(word >> 10);
        const rd: u5 = @truncate(word);
        switch (form) {
            0b001, 0b101 => return .{ .mul_long = .{
                .signed = form == 0b001,
                .op = if ((word & 0x0000_8000) != 0) .sub else .add,
                .rn = rn,
                .rm = rm,
                .ra = ra,
                .rd = rd,
            } },
            0b010, 0b110 => {
                if ((word & 0x0000_8000) != 0 or ra != 31) return error.UnsupportedInstruction;
                return .{ .mul_high = .{ .signed = form == 0b010, .rn = rn, .rm = rm, .rd = rd } };
            },
            else => {},
        }
    }
    // The multiply and multiply-accumulate family, in its plain form: the low half
    // of the product, added to or subtracted from an accumulator. The two bits at
    // 30:29 are zero in all of it, and are part of the test so that a reserved
    // encoding is not read as one.
    if ((word & 0x7fe0_0000) == 0x1b00_0000) {
        if ((word & 0x0020_0000) != 0) return error.UnsupportedInstruction; // the wide forms
        // Register 31 in the accumulator field is how MUL and MNEG are spelled,
        // so the accumulator itself is always read.
        return .{ .mul = .{
            .op = if ((word & 0x0000_8000) != 0) .sub else .add,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .ra = @truncate(word >> 10),
            .rd = @truncate(word),
        } };
    }
    // The pair class. Bit 25:23 says which of the three forms this is, and bit
    // 22 says which direction. The displacement is a signed seven-bit field
    // scaled by the element size, in all three forms.
    //
    // The pair class numbers its element width one step below the single-register
    // class: two here is a 64-bit register, and so eight bytes, where the same
    // two there is four bytes. A pair of 16-bit elements is the floating point
    // class, which needs registers this does not model. A one in the width field
    // is `LDPSW`: two words, each sign-extended to 64 bits, with no store form.
    if ((word & 0x3e00_0000) == 0x2800_0000) {
        const is_load = (word & 0x0040_0000) != 0;
        const opc: u2 = @truncate(word >> 30);
        if (opc == 1 and !is_load) return error.UnsupportedInstruction;
        const scale: i64 = switch (opc) {
            0 => 4, // two 32-bit registers
            1 => 4, // two words, sign-extended
            2 => 8, // two 64-bit registers
            else => return error.UnsupportedInstruction,
        };
        const size: Size = @fromBackingInt(@intCast(@as(u8, @intCast(scale))));
        const raw: i7 = @bitCast(@as(u7, @truncate(word >> 15)));
        const displacement = @as(i64, raw) * scale;
        const addressing: Addressing = switch ((word >> 23) & 7) {
            // 0 is the non-temporal hint form (STNP/LDNP), which addresses like 2 (offset).
            0, 2 => .{ .offset = displacement },
            3 => .{ .pre_index = displacement },
            1 => .{ .post_index = displacement },
            else => return error.UnsupportedInstruction,
        };
        return .{ .pair = .{
            .op = if (is_load) .load else .store,
            .size = size,
            .rn = @truncate(word >> 5),
            .rt = @truncate(word),
            .rt2 = @truncate(word >> 10),
            .addressing = addressing,
            .signed = if (opc == 1) .to64 else .none,
        } };
    }
    if ((word & 0xfc00_0000) == 0x1400_0000)
        return .{ .b = @as(i64, @as(i32, @bitCast(word << 6))) >> 4 };
    if ((word & 0xff00_0010) == 0x5400_0000) {
        const cond: Condition = @fromBackingInt(@intCast(@as(u4, @truncate(word))));
        if (cond == .al or cond == .nv) return error.UnsupportedInstruction;
        // The offset is the 19 bits at 23:5, sign extended and scaled by four.
        // Shifting the whole word would fold the condition in at the bottom, so
        // the field is taken on its own.
        return .{ .b_cond = .{
            .cond = cond,
            .offset = @as(i64, @as(i19, @bitCast(@as(u19, @truncate(word >> 5))))) << 2,
        } };
    }
    // One-source data processing. Bit 29 is the flag-setting bit, which this class
    // does not have, and bit 30 separates it from the two-source class next door.
    // The five bits at 20:16 are a second opcode that is zero for everything here.
    // The first opcode is the six bits at 15:10, and its meaning depends on the
    // width: 2 is the 32-bit byte reversal but the 64-bit one is 3, and 2 at 64 bits
    // is the reversal within each word.
    if ((word & 0x7fff_0000) == 0x5ac0_0000) {
        const width: Width = if ((word & 0x8000_0000) != 0) .x64 else .w32;
        const op: UnaryOp = switch ((word >> 10) & 0x3f) {
            0 => .rbit,
            1 => .rev16,
            2 => if (width == .x64) .rev32 else .rev,
            3 => if (width == .x64) .rev else return error.UnsupportedInstruction,
            4 => .clz,
            else => return error.UnsupportedInstruction,
        };
        return .{ .unary = .{ .op = op, .width = width, .rn = @truncate(word >> 5), .rd = @truncate(word) } };
    }
    // Add and subtract with carry. Bits 28:21 are `11010000`, and the six bits at
    // 15:10 are a second opcode that is zero for all four of these; the conditional
    // select and compare classes next door differ in bits 23:21, which are zero here.
    // Bit 30 says subtract and bit 29 says the flags are set.
    if ((word & 0x1fe0_fc00) == 0x1a00_0000) {
        return .{ .arith_carry = .{
            .op = if ((word & 0x4000_0000) != 0) .sub else .add,
            .flags = (word & 0x2000_0000) != 0,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
        } };
    }
    // The conditional compares. Bit 29 (the flag-setting bit) is set in both, bit 30
    // says compare against negate, and the two bits that are clear in every one of
    // them, 10 and 4, keep this apart from the neighbouring selects. Bit 11 says
    // whether the second operand is a five-bit constant or a register.
    if ((word & 0x3fe0_0410) == 0x3a40_0000) {
        return .{ .cond_compare = .{
            .op = if ((word & 0x4000_0000) != 0) .sub else .add,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .operand = if ((word & 0x0000_0800) != 0)
                .{ .immediate = @truncate(word >> 16) }
            else
                .{ .register = @truncate(word >> 16) },
            .cond = @fromBackingInt(@intCast(@as(u4, @truncate(word >> 12)))),
            .nzcv = @truncate(word),
        } };
    }
    // The `CSEL` family shares one class. What a failed condition leaves behind
    // is spelled as two bits in different places: bit 30 picks the inverted pair
    // (CSEL, CSINC against CSINV, CSNEG) and bit 10 picks the other half of it.
    // Bit 29 and bit 11 are zero in all four, so an encoding that sets them is
    // not one of these instructions.
    if ((word & 0x1fe0_0000) == 0x1a80_0000) {
        if ((word & 0x2000_0800) != 0) return error.UnsupportedInstruction;
        const opc: u2 = (@as(u2, @intFromBool((word & 0x4000_0000) != 0)) << 1) | @intFromBool((word & 0x400) != 0);
        return .{ .csel = .{
            .cond = @fromBackingInt(@intCast(@as(u4, @truncate(word >> 12)))),
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
            .opc = opc,
        } };
    }
    // Two-source data processing. Bit 29 is the flag-setting bit, which this class
    // does not have, and bit 30 separates it from the one-source class next door,
    // so both are part of the test. The operation is the six bits at 15:10; the
    // rest of that space is the CRC and pointer-authentication instructions, which
    // are refused rather than read as something they are not.
    if ((word & 0x7fe0_0000) == 0x1ac0_0000) {
        const op: VariableOp = switch ((word >> 10) & 0x3f) {
            0x02 => .udiv,
            0x03 => .sdiv,
            0x08 => .lsl,
            0x09 => .lsr,
            0x0a => .asr,
            0x0b => .ror,
            else => return error.UnsupportedInstruction,
        };
        return .{ .variable = .{
            .op = op,
            .width = if ((word & 0x8000_0000) != 0) .x64 else .w32,
            .rn = @truncate(word >> 5),
            .rm = @truncate(word >> 16),
            .rd = @truncate(word),
        } };
    }
    return switch (word) {
        0xd400_0001 => .svc,
        0xd400_0002 => .psci,
        0xd503_207f => .wfi,
        // What is left is the SIMD and floating point classes, or nothing at all.
        else => .{ .simd = try simd.parse(word) },
    };
}
