//! The Glacier GPU target: the LilithSemi GPU core generation inside the Chenega platform.
//!
//! A Glacier core is a device, not a CPU. It comes out of reset HALTED and runs only when a host
//! writes its CSR window over a transport, so generated code is a kernel launched into an
//! instruction aperture rather than a function something calls.
//!
//! Its instruction words ARE RISC-V words, RV32I plus RV32M, the A extension atomics, a scalar FP
//! divide and an RVV v1.0 subset, so this target reuses `riscv64/encode.zig` for encoding and adds
//! only what is Glacier's own. See `encode.zig` for the CSRs, the core profiles, and the list of
//! what the core deliberately does NOT decode, which shapes code generation more than what it does.
//!
//! The execution model is a barrel/IMT core: resident warps issue round-robin, one instruction per
//! cycle across the core, and a warp parks on a long operation while another issues, so latency is
//! hidden by other warps rather than by scheduling inside one. Divergence is by RVV masks under v0
//! and not by a hardware reconvergence stack, so control flow lowers to mask arithmetic. Warp
//! identity comes from `mhartid` and the resident count from a custom CSR, which is the closest
//! thing to a workgroup id the hardware has.

pub const encode = @import("glacier/encode.zig");
pub const isel = @import("glacier/isel.zig");
