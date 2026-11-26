//! Printing a Geyser image as text, one word to a line.

const std = @import("std");
const testing = std.testing;
const encode = @import("encode.zig");
const decode = @import("decode.zig");

const named = .{
    "dup",   "drop",   "swap",  "over", "nip",    "plus",    "and",    "or",
    "xor",   "invert", "eq",    "lt",   "ult",    "rshift",  "lshift", "decrement",
    "depth", "fetch",  "store", "to_r", "r_from", "r_fetch", "exit",
};

fn deltaText(delta: encode.Delta) []const u8 {
    return switch (delta) {
        .zero => "0",
        .plus_one => "+1",
        .minus_two => "-2",
        .minus_one => "-1",
    };
}

/// Write the mnemonic of one word. A reserved word prints as `.word`.
pub fn writeMnemonic(w: *std.Io.Writer, word: u16) std.Io.Writer.Error!void {
    const insn = decode.decode(word) catch return w.print(".word 0x{x:0>4}", .{word});
    switch (insn) {
        .lit => |value| try w.print("lit {d}", .{value}),
        .jmp => |to| try w.print("jmp 0x{x:0>4}", .{to}),
        .jz => |to| try w.print("jz 0x{x:0>4}", .{to}),
        .call => |to| try w.print("call 0x{x:0>4}", .{to}),
        .alu => |alu| {
            inline for (named) |name| {
                if (std.meta.eql(encode.Insn{ .alu = alu }, @field(encode, name))) return w.writeAll(name);
            }
            try w.print("alu {t}", .{alu.t_op});
            if (alu.t_to_n) try w.writeAll(" t>n");
            if (alu.t_to_r) try w.writeAll(" t>r");
            if (alu.store) try w.writeAll(" store");
            if (alu.ret) try w.writeAll(" ret");
            try w.print(" d{s} r{s}", .{ deltaText(alu.ddelta), deltaText(alu.rdelta) });
        },
    }
}

/// Write each word of `words` as `address: word  mnemonic`. The address counts words.
pub fn writeListing(w: *std.Io.Writer, words: []const u16) std.Io.Writer.Error!void {
    for (words, 0..) |word, at| {
        try w.print("{x:0>4}: {x:0>4}  ", .{ at, word });
        try writeMnemonic(w, word);
        try w.writeByte('\n');
    }
}

test "a listing names words, prints targets and keeps reserved words" {
    var buffer: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    const words = [_]u16{ 0x8005, 0x0003, 0x2004, 0x4010, 0x6203, 0x700C, 0x6010, 0x6A40 };
    try writeListing(&w, &words);
    try testing.expectEqualStrings(
        \\0000: 8005  lit 5
        \\0001: 0003  jmp 0x0003
        \\0002: 2004  jz 0x0004
        \\0003: 4010  call 0x0010
        \\0004: 6203  plus
        \\0005: 700c  exit
        \\0006: 6010  .word 0x6010
        \\0007: 6a40  alu decrement t>r d0 r0
        \\
    , w.buffered());
}
