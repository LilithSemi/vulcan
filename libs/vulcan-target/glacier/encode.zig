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
//!   - No LR/SC. The A extension atomics are there, the load-reserved pair is not.
//!   - No FP square root, and FP32 truncates toward zero with subnormals flushed rather than
//!     rounding to nearest. Anything that assumes IEEE default rounding differs in the last bit.
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

/// EBREAK. Retires the warp: a Glacier kernel ends with this rather than returning, because a
/// warp has no caller to return to.
pub const ebreak: u32 = 0x00100073;

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

    /// Standard machine scratch CSR, read only here: the INITIAL STACK POINTER of the hart doing
    /// the read, one past the top of its own scratch region. It is the top and not the base so a
    /// downward stack never needs the region size, which lets the size change with the core
    /// generation without a kernel changing. Reads zero on a core built with no scratch.
    pub const mscratch: u12 = 0x340;
};

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
    pub fn scratchBytes(self: Profile) u32 {
        return switch (self) {
            .@"gc1.n", .@"gc1.mi" => 256,
            .@"gc1.s" => 512,
            .@"gc1.f", .@"gc1.ma" => 1024,
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
