//! AArch64 (A64) target: encoding and codegen. Bring-up in progress. Host is
//! aarch64, so generated code is validated by native in-process execution.
//!
//! It is also an instruction set to run: `decode` and `lift` turn the words of an AArch64 guest
//! into IR over a `GuestCpu`, the way `glacier.lift` does for Glacier kernels, so a host of any
//! architecture can compile and run that guest. Everything about the guest side is in `decode`,
//! `lift`, `simd`, `vector`, `identification` and `cpu`; the rest of this directory emits code
//! for an AArch64 host.

const std = @import("std");

pub const encode = @import("aarch64/encode.zig");
pub const disasm = @import("aarch64/disasm.zig");
pub const isel = @import("aarch64/isel.zig");
pub const peephole = @import("aarch64/peephole.zig");
pub const link = @import("aarch64/link.zig");
pub const object = @import("aarch64/object.zig");
pub const jit = @import("aarch64/jit.zig");

// The guest side.
pub const decode = @import("aarch64/decode.zig");
pub const identification = @import("aarch64/identification.zig");
pub const lift = @import("aarch64/lift.zig");
/// The state of one guest cpu, as lifted code addresses it.
pub const GuestCpu = @import("aarch64/cpu.zig").Cpu;

/// Execution-test runners, one per backend (see each file). Skip when the
/// backend is unavailable or incompatible.
const tests = struct {
    pub const native = @import("aarch64/tests/native.zig");
    pub const wimmer_native = @import("aarch64/tests/wimmer_native.zig");
    pub const qemu = @import("aarch64/tests/qemu.zig");
    pub const global_addr = @import("aarch64/tests/global_addr.zig");
    pub const link_archive = @import("aarch64/tests/link_archive.zig");
    pub const link_script = @import("aarch64/tests/link_script.zig");
    pub const dynamic = @import("aarch64/tests/dynamic.zig");
};

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(tests);
}
