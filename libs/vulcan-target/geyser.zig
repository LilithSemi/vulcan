//! The Geyser Forth co-processor target: a J1-style 16-bit stack CPU.
//!
//! See `encode.zig` for the instruction set and how it differs from J1.

const std = @import("std");

pub const encode = @import("geyser/encode.zig");
pub const decode = @import("geyser/decode.zig");
pub const lift = @import("geyser/lift.zig");
pub const disasm = @import("geyser/disasm.zig");
pub const isel = @import("geyser/isel.zig");

test {
    // A root that imports this as a module analyses only what it references, and a test that is
    // not collected reads like a test that passed.
    std.testing.refAllDecls(@This());
}
