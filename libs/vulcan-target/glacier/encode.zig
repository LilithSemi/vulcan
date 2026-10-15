//! Glacier instruction encoding.
//!
//! A Glacier core decodes RV32I plus RV32M, the A extension atomics, a scalar FP divide, and an
//! RVV v1.0 subset, so the instruction words ARE RISC-V words and `riscv64/encode.zig` already
//! emits them. This file does not repeat those encoders; it re-exports them as `rv` and adds only
//! what is specific to Glacier: the CSRs a kernel reads to find out which warp it is, the core
//! profiles that fix the vector length, and `ebreak`, which retires a warp.
//!
//! What a Glacier core deliberately does NOT decode shapes code generation more than what it
//! does, so it is recorded here rather than left to be rediscovered:
//!
//!   - No `vsetvl`. The vector length is fixed by the core profile and is not a runtime register,
//!     so the lane count is a compile-time target parameter. See `Profile`.
//!   - No LR/SC. The A extension atomics are there, the load-reserved pair is not, so there is no
//!     compare-and-swap and a barrier built on one is not available.
//!   - An AMO's `aq` and `rl` bits are accepted and IGNORED. The lowering still emits what the
//!     program asked for, and today that is correct and inert: one bus, one outstanding access and
//!     in-order completion already give the ordering the bits ask for, so a `seq_cst` atomic and a
//!     relaxed one execute identically and both are right. THAT EXPIRES WITH THE MOVE TO AXI4,
//!     which has independent read and write channels and more than one transaction in flight. At
//!     that point every clause of the justification goes at once and the bits become a promise the
//!     core has to keep.
//!   - No FP square root, and every FP32 result has its subnormals flushed to a signed zero. The
//!     rounding is IEEE nearest even except in the widening MAC, which truncates its accumulate.
//!
//! Sub-word access is NOT one of the absences. Both halves work: LB, LBU, LH and LHU extend from
//! the addressed byte or halfword, and SB and SH drive the bus byte select for the lane they
//! address, so a neighbouring byte in the same word is untouched. This lowering emits word access
//! only, which is a limit of the lowering and not of the core: a sub-word value needs every
//! operation on it truncated to its width, and skipping that is a wrong answer rather than a
//! refusal. Packing into a word and using SW stays correct, it is just no longer forced.

const std = @import("std");

/// The RISC-V encoders, unchanged. A Glacier word is a RISC-V word.
pub const rv = @import("../riscv64/encode.zig");

pub const Reg = rv.Reg;

/// EBREAK. Retires the warp as FINISHED: a Glacier kernel ends with this rather than returning,
/// because a warp has no caller to return to. The core requires rs1 = 0 here, deliberately, so the
/// rest of the encoding stays free to be given meaning later.
pub const ebreak: u32 = 0x00100073;

/// ECALL. Retires the warp as FAULTED, with the code taken from THE VALUE `rs1` HOLDS AT RUN TIME.
///
/// NOT from the register number. The core reads the register, so a constant code has to be
/// materialised into one first and a code is never an immediate. This takes a `Reg` rather than an
/// integer for exactly that reason: the two readings of this word, "the rs1 field IS the code" and
/// "the rs1 register holds the code", produce the same instruction and differ in what the hardware
/// reports, and nothing but the type can tell them apart. A register-valued code is also what makes
/// the fault useful to a KERNEL rather than only to the compiler, because the interesting thing to
/// report is usually a computed value and not an enum.
///
/// The host reads the code at FAULT (CSR offset 0x1C) and the valid bit at STATUS bit 4, and it
/// needs both: code zero is a legal report and says nothing on its own. The FIRST fault wins, so
/// with several warps failing the code that explains the run is not overwritten by whichever
/// failed last.
///
/// This is how a kernel says something went wrong AT ALL. Retiring with EBREAK is indistinguishable
/// from finishing, so without this a kernel that discovers its own bad state reports success.
///
/// It is separate from STATUS bit 3, which the FABRIC raises when an address no slave decodes is
/// accessed. Bit 3 means the bus objected, bit 4 means the kernel did.
pub fn ecall(rs1: Reg) u32 {
    return 0x73 | (@as(u32, @backingInt(rs1)) << 15);
}

/// The fault code for a state the compiler proved impossible, as a VALUE to materialise. Nothing
/// emits it on a path a kernel can take: it marks the places where a wrong answer from a compiler
/// analysis would otherwise retire the warp and read as success. A host seeing this has found a
/// compiler bug, not a kernel bug, which is why it is reserved rather than left to a kernel.
///
/// The top of the range on purpose, so a kernel numbering its own codes from one never meets it.
/// 28 through 31 are reserved for the toolchain by agreement; 1 through 27 belong to the kernel.
pub const fault_compiler_bug: i12 = 31;

/// The fault code for a scratch region smaller than the kernel was built for, as a VALUE to
/// materialise.
///
/// The spill slots and the workgroup window are placed against the PROFILE's region size, which
/// this compiler mirrors from the SoC's configuration. A core whose slice is smaller would have
/// the kernel writing into the hart underneath it, which corrupts a neighbour and reports nothing.
/// So a kernel that uses the region checks it first and faults with this.
pub const fault_scratch_too_small: i12 = 30;

test "the fault retire is an ECALL naming the register its code is in" {
    // `x0` reads zero, so the architectural ECALL word is already the zero-code fault and needs no
    // special case. The second one names x31, which is the register this lowering keeps for itself.
    try std.testing.expectEqual(@as(u32, 0x00000073), ecall(.x0));
    try std.testing.expectEqual(@as(u32, 0x000F8073), ecall(.x31));
    // A fault and a finish differ, which is the whole point of the pair.
    try std.testing.expect(ecall(.x0) != ebreak);
}

/// The CSRs a Glacier core decodes. Reading one is the only CSR access it implements, and it is
/// how a kernel asks the hardware which warp is running it.
pub const Csr = struct {
    /// Standard machine-information CSR: the id of the warp doing the read, unique across the
    /// whole SoC. A warp owns a PC and a register file, so it is what a hart id identifies here,
    /// and a second core continues the numbering rather than repeating it.
    pub const mhartid: u12 = 0xF14;

    /// Custom read-only CSR: how many harts the whole SoC runs, across every core. A kernel
    /// strides its work by this and offsets it by `mhartid`, so one entry point fills whatever
    /// core it is launched on without being told the shape by its host.
    pub const hart_count: u12 = 0xFC0;

    /// How many launch arguments a GC1 core carries.
    pub const launch_args: usize = 8;

    /// The custom read-only CSR holding launch argument `i`, 0xFC1 through 0xFC8, or null when the
    /// core has no such argument.
    ///
    /// These belong to the CORE and not to a warp: one dispatch gives every warp the same values,
    /// so a launch argument is WARP UNIFORM BY CONSTRUCTION rather than because an analysis proved
    /// it. A host writes them through the core's CSR window before the launch strobe, and the
    /// STROBE LATCHES THEM into a second set that the kernel reads, so a write during a dispatch
    /// lands on the next launch and a read is constant for the whole of this one.
    ///
    /// The honest edge, since the launch is one strobe per warp: a host that writes different
    /// values BETWEEN the strobes of one dispatch gives different warps different values. Writing
    /// once and then strobing is safe; interleaving is not, and the hardware does not forbid it.
    ///
    /// This is what lets a kernel reach a buffer whose address nothing knew at compile time. The
    /// alternative, resolving an address to a constant in the instruction stream, means a new
    /// BUILD to bind a different buffer, which no graphics driver can do between draws.
    pub fn launchArg(i: usize) ?u12 {
        if (i >= launch_args) return null;
        return @intCast(0xFC1 + i);
    }

    /// Standard machine scratch CSR, read only here: the INITIAL STACK POINTER of the hart doing
    /// the read, one past the top of its own scratch region. It is the top and not the base so a
    /// downward stack never needs the region size, which lets the size change with the core
    /// generation without a kernel changing. Reads zero on a core built with no scratch.
    pub const mscratch: u12 = 0x340;

    /// Custom read-only CSR: how many bytes the reading hart's own scratch slice holds. Unlike
    /// `mscratch` it does not depend on which warp reads it, and it reads zero on a core built
    /// with no scratch, so the floor of a slice is `mscratch - scratchsize`.
    ///
    /// Every profile reserves 2048 bytes today, which is why a kernel built for one runs on all of
    /// them. That is a per-profile CONFIGURATION FIELD and not an architectural constant, so a
    /// kernel that bakes the number mis-sizes itself the day a profile changes. Reading it is how
    /// generated code checks that the region it was built for is the region it got.
    pub const scratchsize: u12 = 0xFD2;
};

/// The cross-warp barrier: FENCE with fm = 0001, which has no operands, so it is one constant.
///
/// The scope is ONE CORE. A warp parks until every other live warp OF ITS OWN CORE arrives, and a
/// warp that has already retired does not hold the rest. A second core's warps are not involved, so
/// a barrier across cores has to be built from the atomics instead.
///
/// That scope has no name in Vulcan's IR, which offers only a workgroup and a subgroup. Both of
/// those are one warp here, so this instruction is wider than either and `isel.zig` emits it for
/// neither. See there for why a wider barrier is not the safe choice.
///
/// fm = 0001 is a RESERVED FENCE encoding. A future RISC-V spec version could claim it and give
/// this word another meaning.
pub const warp_barrier: u32 = 0x1000000F;

/// `fdiv.s rd, rs1, rs2`, with the register fields naming X REGISTERS.
///
/// Glacier has no f register file, so every field of an FP instruction names a scalar register.
/// The bit layout is the standard one, which is why the test pins it against an assembler's
/// `fdiv.s f8, f9, f10`: the same word, read as a different register file.
///
/// It is the ONLY scalar FP operation the core decodes, and the only FP operation of any kind that
/// is not per-lane. The rounding mode in `funct3` is accepted and ignored, because the core ties
/// the divider's mode to nearest even.
///
/// A divide whose operands are the same in every lane belongs here rather than on `vfdiv_vv`: both
/// give the same quotient, and this one does the work once where the vector divider would walk
/// every lane and park the warp for all of them. A renderer still divides per vertex or per
/// scanline and not per pixel, because the walk is what a per-lane divide costs.
pub fn fdiv(rd: Reg, rs1: Reg, rs2: Reg) u32 {
    return 0b1010011 | (@as(u32, @backingInt(rd)) << 7) | (0b111 << 12) |
        (@as(u32, @backingInt(rs1)) << 15) | (@as(u32, @backingInt(rs2)) << 20) |
        (@as(u32, 0b0001100) << 25);
}

/// Read a CSR into `rd`. CSRRS with x0 as the set mask, so it reads without writing.
pub fn csrr(rd: Reg, csr: u12) u32 {
    return rv.csrrs(rd, csr, .x0);
}

/// A GC1 core profile. The warp width IS the vector length and both are baked into the core, so
/// a kernel is built against a profile rather than discovering the shape at run time.
pub const Profile = enum {
    @"gc1.n",
    @"gc1.mi",
    @"gc1.s",
    @"gc1.f",
    @"gc1.ma",

    /// Thread lanes per warp, which is also the vector length in 32-bit elements.
    pub fn lanes(self: Profile) u8 {
        return switch (self) {
            .@"gc1.n", .@"gc1.mi" => 4,
            .@"gc1.s" => 8,
            .@"gc1.f", .@"gc1.ma" => 16,
        };
    }

    /// Warps resident on the core. They issue round-robin, one instruction per cycle across the
    /// whole core, so this is how much latency hiding is available rather than a parallelism knob
    /// a kernel chooses.
    pub fn warps(self: Profile) u8 {
        return switch (self) {
            .@"gc1.n" => 4,
            .@"gc1.mi", .@"gc1.s", .@"gc1.f" => 8,
            .@"gc1.ma" => 16,
        };
    }

    /// Bytes of private scratch one hart owns, which is the whole stack a kernel gets.
    ///
    /// A kernel never reads this number: `mscratch` gives it one past the top of its own region
    /// and the stack grows down from there, so the size can change without a kernel changing. A
    /// COMPILER does read it, because it must know when a spill would walk out of the region and
    /// into the hart below, and the generation is in the profile name so a later core cannot
    /// quietly move it.
    /// Two kilobytes on every GC1 profile. The region is carved from the top of the largest memory
    /// device, SDRAM or DDR3 and not on-chip SRAM, so the size costs address space and no logic:
    /// eight resident warps on gc1.mi reserve 16 KiB out of 32 MiB. A board whose largest memory
    /// IS on-chip pays visibly, and the answer there is fewer resident warps rather than a smaller
    /// region.
    ///
    /// THIS NUMBER MUST MATCH `GlacierConfig.scratchBytesPerHart` IN THE CORE, and the two
    /// directions are not equally bad. A core that reserves MORE than this wastes address space. A
    /// core that reserves LESS lets a spill walk into the hart below and corrupt a neighbour,
    /// which surfaces as that hart's wrong answer somewhere else. So the core is raised first and
    /// this follows, never the other way round.
    ///
    /// That makes a NEW compiler against an OLD bitstream the dangerous pairing, and nothing in
    /// either half detects it: the number is fixed at compile time, so no CSR a kernel could read
    /// would help. Pairing a kernel with the core generation it was built for is the host's job.
    pub fn scratchBytes(self: Profile) u32 {
        return switch (self) {
            .@"gc1.n", .@"gc1.mi", .@"gc1.s", .@"gc1.f", .@"gc1.ma" => 2048,
        };
    }
};

/// Load an arbitrary 32-bit constant into `rd`, returning the one or two words it takes.
///
/// ADDI alone when the value fits its immediate, else LUI plus ADDI. ADDI SIGN EXTENDS, so a low
/// half with bit 11 set borrows one from the upper half. Getting that wrong lands the constant
/// 0x1000 away, which is the classic hand-assembly bug this exists to prevent.
pub fn loadImmediate(rd: Reg, value: i32, out: *[2]u32) []const u32 {
    if (value >= -2048 and value <= 2047) {
        out[0] = rv.addi(rd, .x0, @intCast(value));
        return out[0..1];
    }
    const low: i32 = @as(i32, @bitCast(@as(u32, @bitCast(value)) & 0xFFF));
    const signed_low: i32 = if (low >= 0x800) low - 0x1000 else low;
    const upper: u20 = @truncate(@as(u32, @bitCast(value -% signed_low)) >> 12);
    out[0] = rv.lui(rd, upper);
    out[1] = rv.addi(rd, rd, @intCast(signed_low));
    return out[0..2];
}
