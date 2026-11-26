//! Decoding a Geyser instruction word. The inverse of `encode.zig`.
//!
//! An ALU word with bit 4 set makes the core fault, so decode rejects it.

const std = @import("std");
const testing = std.testing;
const encode = @import("encode.zig");

pub const Error = error{Reserved};

pub fn decode(word: u16) Error!encode.Insn {
    if (word & 0x8000 != 0) return .{ .lit = word & encode.max_literal };
    const target = word & encode.max_target;
    switch (@as(u2, @truncate(word >> 13))) {
        0 => return .{ .jmp = target },
        1 => return .{ .jz = target },
        2 => return .{ .call = target },
        3 => {
            if (word & 0x10 != 0) return error.Reserved;
            return .{ .alu = .{
                .ret = word & 0x1000 != 0,
                .t_op = @fromBackingInt(@intCast(@as(u4, @truncate(word >> 8)))),
                .t_to_n = word & 0x80 != 0,
                .t_to_r = word & 0x40 != 0,
                .store = word & 0x20 != 0,
                .rdelta = @fromBackingInt(@intCast(@as(u2, @truncate(word >> 2)))),
                .ddelta = @fromBackingInt(@intCast(@as(u2, @truncate(word)))),
            } };
        },
    }
}

test "every word is reserved or round trips to itself" {
    var reserved: usize = 0;
    var word: u32 = 0;
    while (word <= 0xFFFF) : (word += 1) {
        const w: u16 = @intCast(word);
        const insn = decode(w) catch |err| {
            try testing.expectEqual(error.Reserved, err);
            try testing.expect(w & 0xE010 == 0x6010);
            reserved += 1;
            continue;
        };
        try testing.expectEqual(w, try encode.encode(insn));
    }
    // ALU words are 0x6000 to 0x7FFF, and bit 4 is set in half of them.
    try testing.expectEqual(@as(usize, 0x2000 / 2), reserved);
}

test "named words decode back to their constructors" {
    const names = .{ "dup", "drop", "swap", "over", "nip", "plus", "and", "or", "xor", "invert", "eq", "lt", "ult", "rshift", "lshift", "decrement", "depth", "fetch", "store", "to_r", "r_from", "r_fetch", "exit" };
    inline for (names) |name| {
        const insn = @field(encode, name);
        try testing.expectEqual(insn, try decode(try encode.encode(insn)));
    }
}

test "fields decode to the right variant" {
    try testing.expectEqual(encode.Insn{ .lit = 0x7FFF }, try decode(0xFFFF));
    try testing.expectEqual(encode.Insn{ .jmp = 0x1FFF }, try decode(0x1FFF));
    try testing.expectEqual(encode.Insn{ .jz = 0x0042 }, try decode(0x2042));
    try testing.expectEqual(encode.Insn{ .call = 0x0001 }, try decode(0x4001));
    try testing.expectEqual(encode.exit, try decode(0x700C));
}

test "an ALU word with bit 4 set is reserved" {
    try testing.expectError(error.Reserved, decode(0x6010));
    try testing.expectError(error.Reserved, decode(0x7F1F));
}
