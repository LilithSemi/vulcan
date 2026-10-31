//! What the guest is told about the machine it is running on.
//!
//! A kernel reads these once, early, and then decides which of its code paths to
//! take from them: which instructions it may use, how large an address it may
//! form, whether a controller is reached by system register or by memory. So
//! every value here is a claim about what the translator can do, and a claim
//! that is too generous sends the guest down a path that stops on an instruction
//! this does not model. The rule is to say nothing is implemented unless it is.
//!
//! All of these are read-only and the same for every access. A field that reads
//! zero says "not implemented" for a feature field, except where the architecture
//! numbers the other way, which is noted where it is.
const Decode = @import("decode.zig");

/// The frequency of the system counter. The device tree gives the architected
/// timer no frequency of its own, so the guest reads it from `CNTFRQ_EL0`.
pub const counter_hz: u64 = 62_500_000;

/// An Arm implementer and a part number that no erratum in the guest's tables
/// names, so that no workaround is applied for hardware this is not.
const midr: u64 = 0x410f_d0f0;

/// Bit 31 is reserved as one. The affinity fields are zero: there is one CPU.
const mpidr: u64 = 0x8000_0000;

/// Both instruction and data caches are 64-byte lines, and neither has to be
/// maintained for code to become visible: memory is coherent here, and a
/// translation is chosen by the bytes it was made from, not by a cached copy.
/// So IDC (bit 28) and DIC (bit 29) are set, which tells the guest it may skip
/// the clean and invalidate it would otherwise do before running new code.
/// Bit 31 is reserved as one; the line sizes are log2 of a count of words.
const ctr: u64 = (1 << 31) | (1 << 29) | (1 << 28) | // reserved, DIC, IDC
    (4 << 24) | (4 << 20) | // CWG and ERG: 64 bytes
    (4 << 16) | (0b11 << 14) | (4 << 0); // DminLine, physically indexed instruction cache, IminLine

/// Zeroing is allowed, and the block is 64 bytes: log2 of a count of words.
/// This must agree with the line the run loop zeroes for `DC ZVA`.
const dczid: u64 = 4;

/// AArch64 at EL0 and EL1 and nothing above. FP and ASIMD are not implemented
/// by the translator yet, so report both fields as 0xf rather than selecting
/// kernel paths that execute unsupported vector instructions. No GIC system
/// register interface, so the guest reaches its controller through memory.
const id_aa64pfr0: u64 = (0xf << 20) | (0xf << 16) | (1 << 4) | (1 << 0);

/// Architecture debug version 8.0. One breakpoint and one watchpoint, and no
/// performance monitor, trace, or statistical profiling.
const id_aa64dfr0: u64 = 6;

/// A 40-bit physical address, which covers everything the guest can be given.
/// 4 KiB granules only: the field for 64 KiB says not implemented with 0xf, and
/// the one for 16 KiB says it with zero. 4 KiB is implemented, at 48 bits of
/// virtual address, so a kernel built for more falls back to that.
const id_aa64mmfr0: u64 = (0xf << 24) | 2;

/// The value a read of `register` returns, or null when it is not one of the
/// registers that describe the machine.
pub fn value(register: Decode.SystemRegister) ?u64 {
    return switch (register) {
        .midr_el1 => midr,
        .mpidr_el1 => mpidr,
        .ctr_el0 => ctr,
        .dczid_el0 => dczid,
        .cntfrq_el0 => counter_hz,
        .id_aa64pfr0_el1 => id_aa64pfr0,
        .id_aa64dfr0_el1 => id_aa64dfr0,
        .id_aa64mmfr0_el1 => id_aa64mmfr0,
        // Not implemented, or nothing to report: no caches to describe, and none
        // of the optional features these registers list.
        .revidr_el1,
        .clidr_el1,
        .id_aa64pfr1_el1,
        .id_aa64pfr2_el1,
        .id_aa64zfr0_el1,
        .id_aa64smfr0_el1,
        .id_aa64fpfr0_el1,
        .id_aa64isar3_el1,
        .aidr_el1,
        .id_aa64dfr1_el1,
        .id_aa64isar0_el1,
        .id_aa64isar1_el1,
        .id_aa64isar2_el1,
        .id_aa64mmfr1_el1,
        .id_aa64mmfr2_el1,
        .id_aa64mmfr3_el1,
        .id_aa64mmfr4_el1,
        => 0,
        else => null,
    };
}
