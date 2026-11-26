//! The primitive words: names, data stack effects and compile-time evaluation.

const std = @import("std");

pub const Prim = enum {
    dup,
    drop,
    swap,
    over,
    rot,
    nip,
    tuck,
    two_dup,
    two_drop,
    to_r,
    r_from,
    r_fetch,
    add,
    sub,
    mul,
    bit_and,
    bit_or,
    bit_xor,
    invert,
    negate,
    lshift,
    rshift,
    eq,
    ne,
    lt,
    gt,
    ult,
    zero_eq,
    zero_lt,
    fetch,
    store,
    plus_store,
    i,
    j,
    unloop,
    leave,
    exit,
    recurse,
};

const names = std.enums.directEnumArray(Prim, []const u8, 0, .{
    .dup = "dup",
    .drop = "drop",
    .swap = "swap",
    .over = "over",
    .rot = "rot",
    .nip = "nip",
    .tuck = "tuck",
    .two_dup = "2dup",
    .two_drop = "2drop",
    .to_r = ">r",
    .r_from = "r>",
    .r_fetch = "r@",
    .add = "+",
    .sub = "-",
    .mul = "*",
    .bit_and = "and",
    .bit_or = "or",
    .bit_xor = "xor",
    .invert = "invert",
    .negate = "negate",
    .lshift = "lshift",
    .rshift = "rshift",
    .eq = "=",
    .ne = "<>",
    .lt = "<",
    .gt = ">",
    .ult = "u<",
    .zero_eq = "0=",
    .zero_lt = "0<",
    .fetch = "@",
    .store = "!",
    .plus_store = "+!",
    .i = "i",
    .j = "j",
    .unloop = "unloop",
    .leave = "leave",
    .exit = "exit",
    .recurse = "recurse",
});

pub fn name(prim: Prim) []const u8 {
    return names[@backingInt(prim)];
}

pub fn lookup(word: []const u8) ?Prim {
    for (names, 0..) |n, idx| {
        if (std.ascii.eqlIgnoreCase(n, word)) return @fromBackingInt(@intCast(idx));
    }
    return null;
}

/// How many cells a primitive takes from the data stack and puts back. Control words
/// (`exit`, `leave`, `unloop`, `recurse`) and loop indexes are handled by the analysis itself.
pub const Effect = struct { pops: u8, pushes: u8 };

pub fn dataEffect(prim: Prim) Effect {
    return switch (prim) {
        .dup => .{ .pops = 1, .pushes = 2 },
        .drop => .{ .pops = 1, .pushes = 0 },
        .swap => .{ .pops = 2, .pushes = 2 },
        .over => .{ .pops = 2, .pushes = 3 },
        .rot => .{ .pops = 3, .pushes = 3 },
        .nip => .{ .pops = 2, .pushes = 1 },
        .tuck => .{ .pops = 2, .pushes = 3 },
        .two_dup => .{ .pops = 2, .pushes = 4 },
        .two_drop => .{ .pops = 2, .pushes = 0 },
        .to_r => .{ .pops = 1, .pushes = 0 },
        .r_from, .r_fetch, .i, .j => .{ .pops = 0, .pushes = 1 },
        .add, .sub, .mul, .bit_and, .bit_or, .bit_xor, .lshift, .rshift, .eq, .ne, .lt, .gt, .ult => .{ .pops = 2, .pushes = 1 },
        .invert, .negate, .zero_eq, .zero_lt, .fetch => .{ .pops = 1, .pushes = 1 },
        .store, .plus_store => .{ .pops = 2, .pushes = 0 },
        .unloop, .leave, .exit, .recurse => .{ .pops = 0, .pushes = 0 },
    };
}

pub const flag_true: u16 = 0xFFFF;

pub fn flag(cond: bool) u16 {
    return if (cond) flag_true else 0;
}

fn signed(x: u16) i16 {
    return @bitCast(x);
}

/// A two operand word on cells: `a` is the deeper operand. Shared by the compile-time
/// evaluator and the tests that mirror the generated code.
pub fn binary(prim: Prim, a: u16, b: u16) u16 {
    return switch (prim) {
        .add => a +% b,
        .sub => a -% b,
        .mul => a *% b,
        .bit_and => a & b,
        .bit_or => a | b,
        .bit_xor => a ^ b,
        .lshift => if (b < 16) a << @intCast(b) else 0,
        .rshift => if (b < 16) a >> @intCast(b) else 0,
        .eq => flag(a == b),
        .ne => flag(a != b),
        .lt => flag(signed(a) < signed(b)),
        .gt => flag(signed(a) > signed(b)),
        .ult => flag(a < b),
        else => unreachable,
    };
}

pub fn unary(prim: Prim, a: u16) u16 {
    return switch (prim) {
        .invert => ~a,
        .negate => 0 -% a,
        .zero_eq => flag(a == 0),
        .zero_lt => flag(signed(a) < 0),
        else => unreachable,
    };
}

pub const CtError = error{ Underflow, Overflow, NotConstant };

/// A small stack the compiler uses to evaluate constant expressions.
pub const CtStack = struct {
    items: [32]u16 = undefined,
    len: usize = 0,

    pub fn push(self: *CtStack, v: u16) CtError!void {
        if (self.len == self.items.len) return error.Overflow;
        self.items[self.len] = v;
        self.len += 1;
    }

    pub fn pop(self: *CtStack) CtError!u16 {
        if (self.len == 0) return error.Underflow;
        self.len -= 1;
        return self.items[self.len];
    }

    fn need(self: *const CtStack, n: usize) CtError!void {
        if (self.len < n) return error.Underflow;
    }

    /// Run a primitive on the stack. Words that touch memory, the return stack or the
    /// control flow are not constant and return `NotConstant`.
    pub fn apply(self: *CtStack, prim: Prim) CtError!void {
        switch (prim) {
            .dup => {
                try self.need(1);
                try self.push(self.items[self.len - 1]);
            },
            .drop => _ = try self.pop(),
            .swap => {
                try self.need(2);
                std.mem.swap(u16, &self.items[self.len - 1], &self.items[self.len - 2]);
            },
            .over => {
                try self.need(2);
                try self.push(self.items[self.len - 2]);
            },
            .rot => {
                try self.need(3);
                const x = self.items[self.len - 3];
                self.items[self.len - 3] = self.items[self.len - 2];
                self.items[self.len - 2] = self.items[self.len - 1];
                self.items[self.len - 1] = x;
            },
            .nip => {
                const top = try self.pop();
                _ = try self.pop();
                try self.push(top);
            },
            .tuck => {
                try self.need(2);
                const top = self.items[self.len - 1];
                try self.push(top);
                self.items[self.len - 2] = self.items[self.len - 3];
                self.items[self.len - 3] = top;
            },
            .two_dup => {
                try self.need(2);
                const a = self.items[self.len - 2];
                const b = self.items[self.len - 1];
                try self.push(a);
                try self.push(b);
            },
            .two_drop => {
                _ = try self.pop();
                _ = try self.pop();
            },
            .add, .sub, .mul, .bit_and, .bit_or, .bit_xor, .lshift, .rshift, .eq, .ne, .lt, .gt, .ult => {
                const b = try self.pop();
                const a = try self.pop();
                try self.push(binary(prim, a, b));
            },
            .invert, .negate, .zero_eq, .zero_lt => try self.push(unary(prim, try self.pop())),
            .to_r, .r_from, .r_fetch, .fetch, .store, .plus_store, .i, .j, .unloop, .leave, .exit, .recurse => return error.NotConstant,
        }
    }
};

test "names round trip" {
    for (std.enums.values(Prim)) |prim| {
        try std.testing.expectEqual(prim, lookup(name(prim)).?);
    }
    try std.testing.expectEqual(Prim.ult, lookup("U<").?);
    try std.testing.expectEqual(@as(?Prim, null), lookup("2swap"));
}

test "compile-time stack words" {
    var s: CtStack = .{};
    try s.push(1);
    try s.push(2);
    try s.push(3);
    try s.apply(.rot);
    try std.testing.expectEqualSlices(u16, &.{ 2, 3, 1 }, s.items[0..s.len]);
    try s.apply(.tuck);
    try std.testing.expectEqualSlices(u16, &.{ 2, 1, 3, 1 }, s.items[0..s.len]);
}

test "shifts of 16 or more give zero" {
    try std.testing.expectEqual(@as(u16, 0), binary(.lshift, 1, 16));
    try std.testing.expectEqual(@as(u16, 0x8000), binary(.lshift, 1, 15));
    try std.testing.expectEqual(@as(u16, 0), binary(.rshift, 0x8000, 16));
}
