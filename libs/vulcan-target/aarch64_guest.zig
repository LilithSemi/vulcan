//! An AArch64 guest, for a host that runs it as native code.
//!
//! The counterpart of `glacier.zig`'s `lift`: the words of a guest become Vulcan IR here, and
//! whoever runs the guest compiles that for its own cpu, keeps the result, and decides what a
//! stop of the compiled code means. `aarch64.zig` is something else, the backend that emits code
//! for an AArch64 host.

const std = @import("std");

pub const decode = @import("aarch64_guest/decode.zig");
pub const identification = @import("aarch64_guest/identification.zig");
pub const Cpu = @import("aarch64_guest/cpu.zig").Cpu;
pub const lift = @import("aarch64_guest/lift.zig");

test {
    std.testing.refAllDecls(@This());
}
