const std = @import("std");

pub const Format = enum(u8) {
    bf16 = 0,
    f8_e4m3 = 1,
    f8_e5m2 = 2,

    pub fn payloadBits(self: Format) u16 {
        return switch (self) {
            .bf16 => 16,
            .f8_e4m3, .f8_e5m2 => 8,
        };
    }
};

pub const Error = error{InvalidPayload};

pub fn decode(format: Format, payload: u16) Error!f32 {
    return switch (format) {
        .bf16 => @bitCast(@as(u32, payload) << 16),
        .f8_e4m3 => decodeE4M3(try payload8(payload)),
        .f8_e5m2 => decodeE5M2(try payload8(payload)),
    };
}

pub fn encode(format: Format, value: f32) u16 {
    return switch (format) {
        .bf16 => encodeBf16(value),
        .f8_e4m3 => encodeE4M3(value),
        .f8_e5m2 => encodeE5M2(value),
    };
}

fn payload8(payload: u16) Error!u8 {
    if (payload > std.math.maxInt(u8)) return error.InvalidPayload;
    return @intCast(payload);
}

fn encodeBf16(value: f32) u16 {
    const source: u32 = @bitCast(value);
    const exponent = source & 0x7f80_0000;
    const fraction = source & 0x007f_ffff;
    var retained: u16 = @truncate(source >> 16);
    if (exponent == 0x7f80_0000) {
        if (fraction != 0) retained |= 0x0040;
        return retained;
    }
    return @truncate((source + 0x7fff + (retained & 1)) >> 16);
}

fn encodeE4M3(value: f32) u8 {
    const source: u32 = @bitCast(value);
    const sign: u8 = @truncate((source >> 24) & 0x80);
    const source_exponent: u8 = @truncate((source >> 23) & 0xff);
    const source_fraction = source & 0x007f_ffff;
    if (source_exponent == 0xff) {
        if (source_fraction != 0) return sign | 0x7f;
        return sign | 0x7e;
    }
    if (source_exponent == 0) return sign;

    const exponent: i16 = @as(i16, source_exponent) - 127;
    const significand = 0x0080_0000 | source_fraction;
    var magnitude: u32 = undefined;
    if (exponent < -6) {
        const shift: u8 = @intCast(-9 - exponent + 23);
        magnitude = roundRight(significand, shift);
    } else {
        var rounded = roundRight(significand, 20);
        var target_exponent = exponent;
        if (rounded == 16) {
            rounded = 8;
            target_exponent += 1;
        }
        magnitude = @as(u32, @intCast(target_exponent + 7)) * 8 + (rounded - 8);
    }
    return sign | @as(u8, @intCast(@min(magnitude, 0x7e)));
}

fn decodeE4M3(payload: u8) f32 {
    const sign = @as(u32, payload & 0x80) << 24;
    const exponent = (payload >> 3) & 0x0f;
    const fraction = payload & 0x07;
    if (exponent == 0x0f and fraction == 0x07) return @bitCast(sign | 0x7fc0_0000);
    if (exponent == 0) {
        if (fraction == 0) return @bitCast(sign);
        const leading: u5 = @intCast(7 - @clz(fraction));
        const f32_exponent = @as(u32, 118) + leading;
        const remainder = @as(u32, fraction) - (@as(u32, 1) << leading);
        return @bitCast(sign | (f32_exponent << 23) | (remainder << @intCast(23 - leading)));
    }
    const f32_exponent = @as(u32, exponent) + 120;
    return @bitCast(sign | (f32_exponent << 23) | (@as(u32, fraction) << 20));
}

fn encodeE5M2(value: f32) u8 {
    const source: u32 = @bitCast(value);
    const sign: u8 = @truncate((source >> 24) & 0x80);
    const source_exponent: u8 = @truncate((source >> 23) & 0xff);
    const source_fraction = source & 0x007f_ffff;
    if (source_exponent == 0xff) {
        if (source_fraction == 0) return sign | 0x7c;
        const payload: u8 = @truncate(source_fraction >> 21);
        return sign | 0x7c | payload | 0x02;
    }
    if (source_exponent == 0) return sign;

    const exponent: i16 = @as(i16, source_exponent) - 127;
    const significand = 0x0080_0000 | source_fraction;
    var magnitude: u32 = undefined;
    if (exponent < -14) {
        const shift: u8 = @intCast(-16 - exponent + 23);
        magnitude = roundRight(significand, shift);
    } else {
        var rounded = roundRight(significand, 21);
        var target_exponent = exponent;
        if (rounded == 8) {
            rounded = 4;
            target_exponent += 1;
        }
        magnitude = @as(u32, @intCast(target_exponent + 15)) * 4 + (rounded - 4);
    }
    return sign | @as(u8, @intCast(@min(magnitude, 0x7b)));
}

fn decodeE5M2(payload: u8) f32 {
    const sign = @as(u32, payload & 0x80) << 24;
    const exponent = (payload >> 2) & 0x1f;
    const fraction = payload & 0x03;
    if (exponent == 0x1f) {
        if (fraction == 0) return @bitCast(sign | 0x7f80_0000);
        return @bitCast(sign | 0x7f80_0000 | (@as(u32, fraction | 0x02) << 21));
    }
    if (exponent == 0) {
        if (fraction == 0) return @bitCast(sign);
        const leading: u5 = @intCast(7 - @clz(fraction));
        const f32_exponent = @as(u32, 111) + leading;
        const remainder = @as(u32, fraction) - (@as(u32, 1) << leading);
        return @bitCast(sign | (f32_exponent << 23) | (remainder << @intCast(23 - leading)));
    }
    const f32_exponent = @as(u32, exponent) + 112;
    return @bitCast(sign | (f32_exponent << 23) | (@as(u32, fraction) << 21));
}

fn roundRight(value: u32, shift: u8) u32 {
    if (shift > 24) return 0;
    const retained = value >> @intCast(shift);
    const mask = (@as(u32, 1) << @intCast(shift)) - 1;
    const discarded = value & mask;
    const halfway = @as(u32, 1) << @intCast(shift - 1);
    return retained + @intFromBool(discarded > halfway or
        (discarded == halfway and retained & 1 != 0));
}

test "BF16 payloads round trip with NaNs quieted" {
    for (0..@as(usize, std.math.maxInt(u16)) + 1) |raw| {
        const payload: u16 = @intCast(raw);
        const expected = if (payload & 0x7f80 == 0x7f80 and payload & 0x007f != 0)
            payload | 0x0040
        else
            payload;
        try std.testing.expectEqual(expected, encode(.bf16, try decode(.bf16, payload)));
    }
}

test "BF16 encode uses ties to even and preserves special signs" {
    const cases = [_]struct { source: u32, expected: u16 }{
        .{ .source = 0x0000_0000, .expected = 0x0000 },
        .{ .source = 0x8000_0000, .expected = 0x8000 },
        .{ .source = 0x3f80_0000, .expected = 0x3f80 },
        .{ .source = 0x3f80_7fff, .expected = 0x3f80 },
        .{ .source = 0x3f80_8000, .expected = 0x3f80 },
        .{ .source = 0x3f80_8001, .expected = 0x3f81 },
        .{ .source = 0x3f81_7fff, .expected = 0x3f81 },
        .{ .source = 0x3f81_8000, .expected = 0x3f82 },
        .{ .source = 0x3f81_8001, .expected = 0x3f82 },
        .{ .source = 0x7f7f_ffff, .expected = 0x7f80 },
        .{ .source = 0xff7f_ffff, .expected = 0xff80 },
        .{ .source = 0x7f80_0000, .expected = 0x7f80 },
        .{ .source = 0xff80_0000, .expected = 0xff80 },
        .{ .source = 0x7f81_2345, .expected = 0x7fc1 },
        .{ .source = 0xffa5_4321, .expected = 0xffe5 },
    };
    for (cases) |case| try std.testing.expectEqual(case.expected, encode(.bf16, @bitCast(case.source)));
}

test "FP8 payloads round trip with format-specific NaNs" {
    for (0..256) |raw| {
        const payload: u8 = @intCast(raw);
        try std.testing.expectEqual(@as(u16, payload), encode(.f8_e4m3, try decode(.f8_e4m3, payload)));
        const e5_expected = if (payload & 0x7c == 0x7c and payload & 0x03 != 0)
            payload | 0x02
        else
            payload;
        try std.testing.expectEqual(e5_expected, encode(.f8_e5m2, try decode(.f8_e5m2, payload)));
    }
}

test "FP8 normal and subnormal boundaries round nearest ties to even" {
    const e4_cases = [_]struct { source: u32, expected: u16 }{
        .{ .source = 0x3f87_ffff, .expected = 0x38 },
        .{ .source = 0x3f88_0000, .expected = 0x38 },
        .{ .source = 0x3f88_0001, .expected = 0x39 },
        .{ .source = 0x3f97_ffff, .expected = 0x39 },
        .{ .source = 0x3f98_0000, .expected = 0x3a },
        .{ .source = 0x3f98_0001, .expected = 0x3a },
        .{ .source = 0x3a7f_ffff, .expected = 0x00 },
        .{ .source = 0x3a80_0000, .expected = 0x00 },
        .{ .source = 0x3a80_0001, .expected = 0x01 },
        .{ .source = 0x3b3f_ffff, .expected = 0x01 },
        .{ .source = 0x3b40_0000, .expected = 0x02 },
        .{ .source = 0x3b40_0001, .expected = 0x02 },
        .{ .source = 0x3c6f_ffff, .expected = 0x07 },
        .{ .source = 0x3c70_0000, .expected = 0x08 },
        .{ .source = 0x3c70_0001, .expected = 0x08 },
    };
    for (e4_cases) |case| {
        try std.testing.expectEqual(case.expected, encode(.f8_e4m3, @bitCast(case.source)));
        try std.testing.expectEqual(case.expected | 0x80, encode(.f8_e4m3, @bitCast(case.source | 0x8000_0000)));
    }

    const e5_cases = [_]struct { source: u32, expected: u16 }{
        .{ .source = 0x3f8f_ffff, .expected = 0x3c },
        .{ .source = 0x3f90_0000, .expected = 0x3c },
        .{ .source = 0x3f90_0001, .expected = 0x3d },
        .{ .source = 0x3faf_ffff, .expected = 0x3d },
        .{ .source = 0x3fb0_0000, .expected = 0x3e },
        .{ .source = 0x3fb0_0001, .expected = 0x3e },
        .{ .source = 0x36ff_ffff, .expected = 0x00 },
        .{ .source = 0x3700_0000, .expected = 0x00 },
        .{ .source = 0x3700_0001, .expected = 0x01 },
        .{ .source = 0x37bf_ffff, .expected = 0x01 },
        .{ .source = 0x37c0_0000, .expected = 0x02 },
        .{ .source = 0x37c0_0001, .expected = 0x02 },
        .{ .source = 0x385f_ffff, .expected = 0x03 },
        .{ .source = 0x3860_0000, .expected = 0x04 },
        .{ .source = 0x3860_0001, .expected = 0x04 },
    };
    for (e5_cases) |case| {
        try std.testing.expectEqual(case.expected, encode(.f8_e5m2, @bitCast(case.source)));
        try std.testing.expectEqual(case.expected | 0x80, encode(.f8_e5m2, @bitCast(case.source | 0x8000_0000)));
    }
}

test "FP8 encode covers rounding saturation infinity and NaN" {
    const cases = [_]struct { source: u32, e4: u16, e5: u16 }{
        .{ .source = 0x0000_0000, .e4 = 0x00, .e5 = 0x00 },
        .{ .source = 0x8000_0000, .e4 = 0x80, .e5 = 0x80 },
        .{ .source = 0x3f88_0000, .e4 = 0x38, .e5 = 0x3c },
        .{ .source = 0x3f88_0001, .e4 = 0x39, .e5 = 0x3c },
        .{ .source = 0x3a80_0000, .e4 = 0x00, .e5 = 0x14 },
        .{ .source = 0x3a80_0001, .e4 = 0x01, .e5 = 0x14 },
        .{ .source = 0x3700_0000, .e4 = 0x00, .e5 = 0x00 },
        .{ .source = 0x3700_0001, .e4 = 0x00, .e5 = 0x01 },
        .{ .source = 0x7f7f_ffff, .e4 = 0x7e, .e5 = 0x7b },
        .{ .source = 0xff7f_ffff, .e4 = 0xfe, .e5 = 0xfb },
        .{ .source = 0x7f80_0000, .e4 = 0x7e, .e5 = 0x7c },
        .{ .source = 0xff80_0000, .e4 = 0xfe, .e5 = 0xfc },
        .{ .source = 0x7f80_0001, .e4 = 0x7f, .e5 = 0x7e },
        .{ .source = 0xffa0_0001, .e4 = 0xff, .e5 = 0xff },
    };
    for (cases) |case| {
        const value: f32 = @bitCast(case.source);
        try std.testing.expectEqual(case.e4, encode(.f8_e4m3, value));
        try std.testing.expectEqual(case.e5, encode(.f8_e5m2, value));
    }
    try std.testing.expect(@as(u32, @bitCast(try decode(.f8_e4m3, 0x7c))) !=
        @as(u32, @bitCast(try decode(.f8_e5m2, 0x7c))));
    try std.testing.expect(@as(u32, @bitCast(try decode(.f8_e4m3, 0x7e))) !=
        @as(u32, @bitCast(try decode(.f8_e5m2, 0x7e))));
}

test "FP8 rejects upper payload bits while BF16 accepts all payloads" {
    try std.testing.expectError(error.InvalidPayload, decode(.f8_e4m3, 0x0100));
    try std.testing.expectError(error.InvalidPayload, decode(.f8_e5m2, 0xffff));
    for (0..@as(usize, std.math.maxInt(u16)) + 1) |raw| {
        _ = try decode(.bf16, @intCast(raw));
    }
}
