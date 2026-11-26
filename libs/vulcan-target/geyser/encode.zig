//! Encoding for the Geyser Forth co-processor, a J1-style 16-bit stack CPU.
//!
//! The instruction set is the one in the `GeyserCpu` doc comment in the Geyser repository. It
//! follows the J1 encoding with these changes: the return stack holds word addresses, a shift by
//! 16 or more gives 0, byte addresses from 0xF000 up are I/O, and ALU bit 4 is reserved and must
//! be 0.

const std = @import("std");
const testing = std.testing;

pub const Error = error{OutOfRange};

/// Literals hold 15 bits. Jump, branch and call targets hold 13 bits of word address.
pub const max_literal: u16 = 0x7FFF;
pub const max_target: u16 = 0x1FFF;

/// The image limit, in words. It matches the largest memory the core accepts.
pub const max_image_words: usize = 4096;

/// The stack depth of the Geyser core on the boards.
pub const default_stack_depth: u32 = 16;

/// The T-op field of an ALU word, bits 11:8. It selects the new top of stack.
pub const TOp = enum(u4) {
    t = 0,
    n = 1,
    add = 2,
    @"and" = 3,
    @"or" = 4,
    xor = 5,
    invert = 6,
    eq = 7,
    lt = 8,
    rshift = 9,
    decrement = 10,
    r = 11,
    fetch = 12,
    lshift = 13,
    depth = 14,
    ult = 15,
};

/// A stack pointer change. The tag value is the two bit field in the word.
pub const Delta = enum(u2) {
    zero = 0,
    plus_one = 1,
    minus_two = 2,
    minus_one = 3,
};

pub const Alu = struct {
    ret: bool = false,
    t_op: TOp,
    t_to_n: bool = false,
    t_to_r: bool = false,
    store: bool = false,
    rdelta: Delta = .zero,
    ddelta: Delta = .zero,
};

/// Literal values and targets are wide enough to hold a bad value, so `encode` can reject it.
pub const Insn = union(enum) {
    lit: u16,
    jmp: u16,
    jz: u16,
    call: u16,
    alu: Alu,
};

pub fn encode(insn: Insn) Error!u16 {
    switch (insn) {
        .lit => |value| {
            if (value > max_literal) return error.OutOfRange;
            return 0x8000 | value;
        },
        .jmp => |target| return try encodeBranch(0x0000, target),
        .jz => |target| return try encodeBranch(0x2000, target),
        .call => |target| return try encodeBranch(0x4000, target),
        .alu => |alu| return encodeAlu(alu),
    }
}

fn encodeBranch(base: u16, target: u16) Error!u16 {
    if (target > max_target) return error.OutOfRange;
    return base | target;
}

fn encodeAlu(alu: Alu) u16 {
    var word: u16 = 0x6000;
    if (alu.ret) word |= 0x1000;
    word |= @as(u16, @backingInt(alu.t_op)) << 8;
    if (alu.t_to_n) word |= 0x80;
    if (alu.t_to_r) word |= 0x40;
    if (alu.store) word |= 0x20;
    word |= @as(u16, @backingInt(alu.rdelta)) << 2;
    word |= @backingInt(alu.ddelta);
    return word;
}

fn aluInsn(alu: Alu) Insn {
    return .{ .alu = alu };
}

pub const dup = aluInsn(.{ .t_op = .t, .t_to_n = true, .ddelta = .plus_one });
pub const drop = aluInsn(.{ .t_op = .n, .ddelta = .minus_one });
pub const swap = aluInsn(.{ .t_op = .n, .t_to_n = true });
pub const over = aluInsn(.{ .t_op = .n, .t_to_n = true, .ddelta = .plus_one });
pub const nip = aluInsn(.{ .t_op = .t, .ddelta = .minus_one });
pub const plus = aluInsn(.{ .t_op = .add, .ddelta = .minus_one });
pub const @"and" = aluInsn(.{ .t_op = .@"and", .ddelta = .minus_one });
pub const @"or" = aluInsn(.{ .t_op = .@"or", .ddelta = .minus_one });
pub const xor = aluInsn(.{ .t_op = .xor, .ddelta = .minus_one });
pub const invert = aluInsn(.{ .t_op = .invert });
pub const eq = aluInsn(.{ .t_op = .eq, .ddelta = .minus_one });
pub const lt = aluInsn(.{ .t_op = .lt, .ddelta = .minus_one });
pub const ult = aluInsn(.{ .t_op = .ult, .ddelta = .minus_one });
pub const rshift = aluInsn(.{ .t_op = .rshift, .ddelta = .minus_one });
pub const lshift = aluInsn(.{ .t_op = .lshift, .ddelta = .minus_one });
pub const decrement = aluInsn(.{ .t_op = .decrement });
pub const depth = aluInsn(.{ .t_op = .depth, .t_to_n = true, .ddelta = .plus_one });
pub const fetch = aluInsn(.{ .t_op = .fetch });
/// Stores N at address T and pops T. The stored value stays on T.
pub const store = aluInsn(.{ .t_op = .n, .store = true, .ddelta = .minus_one });
pub const to_r = aluInsn(.{ .t_op = .n, .t_to_r = true, .ddelta = .minus_one, .rdelta = .plus_one });
pub const r_from = aluInsn(.{ .t_op = .r, .t_to_n = true, .ddelta = .plus_one, .rdelta = .minus_one });
pub const r_fetch = aluInsn(.{ .t_op = .r, .t_to_n = true, .ddelta = .plus_one });
pub const exit = aluInsn(.{ .t_op = .t, .ret = true, .rdelta = .minus_one });

/// Writes the words little endian, two bytes each.
pub fn writeImage(words: []const u16, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    for (words) |w| try writer.writeInt(u16, w, .little);
}

pub const ImageError = error{ OddLength, ImageTooLarge } || std.mem.Allocator.Error;

/// Reads a little endian image. The caller frees the result.
pub fn readImage(allocator: std.mem.Allocator, bytes: []const u8) ImageError![]u16 {
    if (bytes.len % 2 != 0) return error.OddLength;
    if (bytes.len / 2 > max_image_words) return error.ImageTooLarge;
    const words = try allocator.alloc(u16, bytes.len / 2);
    for (words, 0..) |*w, i| w.* = std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
    return words;
}

test "named words encode to the exact J1 words" {
    const cases = [_]struct { Insn, u16 }{
        .{ dup, 0x6081 },
        .{ drop, 0x6103 },
        .{ swap, 0x6180 },
        .{ over, 0x6181 },
        .{ nip, 0x6003 },
        .{ plus, 0x6203 },
        .{ @"and", 0x6303 },
        .{ @"or", 0x6403 },
        .{ xor, 0x6503 },
        .{ invert, 0x6600 },
        .{ eq, 0x6703 },
        .{ lt, 0x6803 },
        .{ rshift, 0x6903 },
        .{ decrement, 0x6A00 },
        .{ fetch, 0x6C00 },
        .{ lshift, 0x6D03 },
        .{ depth, 0x6E81 },
        .{ ult, 0x6F03 },
        .{ store, 0x6123 },
        .{ to_r, 0x6147 },
        .{ r_from, 0x6B8D },
        .{ r_fetch, 0x6B81 },
        .{ exit, 0x700C },
    };
    for (cases) |c| try testing.expectEqual(c[1], try encode(c[0]));
}

test "literals and branches put their field in the low bits" {
    try testing.expectEqual(@as(u16, 0x8000), try encode(.{ .lit = 0 }));
    try testing.expectEqual(@as(u16, 0xFFFF), try encode(.{ .lit = 0x7FFF }));
    try testing.expectEqual(@as(u16, 0x0000), try encode(.{ .jmp = 0 }));
    try testing.expectEqual(@as(u16, 0x1FFF), try encode(.{ .jmp = 0x1FFF }));
    try testing.expectEqual(@as(u16, 0x2123), try encode(.{ .jz = 0x123 }));
    try testing.expectEqual(@as(u16, 0x5FFF), try encode(.{ .call = 0x1FFF }));
}

test "a literal above 15 bits and a target above 13 bits are rejected" {
    try testing.expectError(error.OutOfRange, encode(.{ .lit = 0x8000 }));
    try testing.expectError(error.OutOfRange, encode(.{ .jmp = 0x2000 }));
    try testing.expectError(error.OutOfRange, encode(.{ .jz = 0x2000 }));
    try testing.expectError(error.OutOfRange, encode(.{ .call = 0xFFFF }));
}

test "delta tags use the 0, +1, -2, -1 field order" {
    const d = Alu{ .t_op = .t, .rdelta = .minus_two, .ddelta = .plus_one };
    try testing.expectEqual(@as(u16, 0x6009), try encode(.{ .alu = d }));
}

test "writeImage writes little endian words" {
    var buf: [4]u8 = undefined;
    var w = std.Io.Writer.fixed(&buf);
    try writeImage(&.{ 0x6081, 0x8001 }, &w);
    try testing.expectEqualSlices(u8, &.{ 0x81, 0x60, 0x01, 0x80 }, &buf);
}

test "readImage reads little endian words" {
    const words = try readImage(testing.allocator, &.{ 0x81, 0x60, 0x01, 0x80 });
    defer testing.allocator.free(words);
    try testing.expectEqualSlices(u16, &.{ 0x6081, 0x8001 }, words);
}

test "readImage rejects an odd length" {
    try testing.expectError(error.OddLength, readImage(testing.allocator, &.{ 1, 2, 3 }));
}

test "readImage accepts 4096 words and rejects 4097" {
    const full = try testing.allocator.alloc(u8, max_image_words * 2);
    defer testing.allocator.free(full);
    @memset(full, 0);
    const words = try readImage(testing.allocator, full);
    testing.allocator.free(words);
    const over_full = try testing.allocator.alloc(u8, max_image_words * 2 + 2);
    defer testing.allocator.free(over_full);
    @memset(over_full, 0);
    try testing.expectError(error.ImageTooLarge, readImage(testing.allocator, over_full));
}
