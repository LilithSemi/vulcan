const std = @import("std");
const low_float = @import("low_float.zig");

pub const ScaleApplication = enum(u8) {
    multiply = 0,
    divide = 1,
};

comptime {
    if (@backingInt(ScaleApplication.multiply) != 0) @compileError("multiply tag changed");
    if (@backingInt(ScaleApplication.divide) != 1) @compileError("divide tag changed");
}

pub fn decodeE2M1(payload: u8) f32 {
    const nibble = payload & 0x0f;
    const sign = @as(u32, nibble & 0x08) << 28;
    const magnitude = [_]u32{
        0x0000_0000,
        0x3f00_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x4000_0000,
        0x4040_0000,
        0x4080_0000,
        0x40c0_0000,
    };
    return @bitCast(sign | magnitude[nibble & 0x07]);
}

pub fn encodeE2M1(value: f32) u8 {
    const source: u32 = @bitCast(value);
    const sign: u8 = @truncate((source >> 28) & 0x08);
    const magnitude = source & 0x7fff_ffff;
    if (magnitude > 0x7f80_0000) return 0x07;
    if (magnitude == 0x7f80_0000) return sign | 0x07;

    const result: u8 = if (magnitude <= 0x3e80_0000)
        0x00
    else if (magnitude < 0x3f40_0000)
        0x01
    else if (magnitude <= 0x3fa0_0000)
        0x02
    else if (magnitude < 0x3fe0_0000)
        0x03
    else if (magnitude <= 0x4020_0000)
        0x04
    else if (magnitude < 0x4060_0000)
        0x05
    else if (magnitude <= 0x40a0_0000)
        0x06
    else
        0x07;
    return sign | result;
}

fn apply(value: f32, scale: f32, application: ScaleApplication) f32 {
    return switch (application) {
        .multiply => value * scale,
        .divide => value / scale,
    };
}

fn undo(value: f32, scale: f32, application: ScaleApplication) f32 {
    return switch (application) {
        .multiply => value / scale,
        .divide => value * scale,
    };
}

pub fn dequantize(
    payload: u8,
    block_scale: u8,
    global_scale: f32,
    block_application: ScaleApplication,
    global_application: ScaleApplication,
) f32 {
    const decoded_scale = low_float.decode(.f8_e4m3, block_scale) catch unreachable;
    const local: f32 = apply(decodeE2M1(payload), decoded_scale, block_application);
    return apply(local, global_scale, global_application);
}

pub fn quantize(
    value: f32,
    block_scale: u8,
    global_scale: f32,
    block_application: ScaleApplication,
    global_application: ScaleApplication,
) u8 {
    const decoded_scale = low_float.decode(.f8_e4m3, block_scale) catch unreachable;
    const global_unscaled: f32 = undo(value, global_scale, global_application);
    const local_unscaled: f32 = undo(global_unscaled, decoded_scale, block_application);
    return encodeE2M1(local_unscaled);
}

test "all E2M1 payloads decode exactly and round trip" {
    const positive = [_]u32{ 0, 0x3f00_0000, 0x3f80_0000, 0x3fc0_0000, 0x4000_0000, 0x4040_0000, 0x4080_0000, 0x40c0_0000 };
    for (0..16) |payload| {
        const expected = positive[payload & 7] | @as(u32, @intFromBool(payload & 8 != 0)) << 31;
        try std.testing.expectEqual(expected, @as(u32, @bitCast(decodeE2M1(@intCast(payload)))));
        try std.testing.expectEqual(@as(u8, @intCast(payload)), encodeE2M1(decodeE2M1(@intCast(payload))));
    }
}

test "E2M1 midpoint neighbors round to the adjacent even payload" {
    const cases = [_]struct { midpoint: u32, lower: u8, tie: u8, upper: u8 }{
        .{ .midpoint = 0x3e80_0000, .lower = 0, .tie = 0, .upper = 1 },
        .{ .midpoint = 0x3f40_0000, .lower = 1, .tie = 2, .upper = 2 },
        .{ .midpoint = 0x3fa0_0000, .lower = 2, .tie = 2, .upper = 3 },
        .{ .midpoint = 0x3fe0_0000, .lower = 3, .tie = 4, .upper = 4 },
        .{ .midpoint = 0x4020_0000, .lower = 4, .tie = 4, .upper = 5 },
        .{ .midpoint = 0x4060_0000, .lower = 5, .tie = 6, .upper = 6 },
        .{ .midpoint = 0x40a0_0000, .lower = 6, .tie = 6, .upper = 7 },
    };
    for (cases) |case| {
        try std.testing.expectEqual(case.lower, encodeE2M1(@bitCast(case.midpoint - 1)));
        try std.testing.expectEqual(case.tie, encodeE2M1(@bitCast(case.midpoint)));
        try std.testing.expectEqual(case.upper, encodeE2M1(@bitCast(case.midpoint + 1)));
        try std.testing.expectEqual(case.lower | 8, encodeE2M1(@bitCast((case.midpoint - 1) | 0x8000_0000)));
        try std.testing.expectEqual(case.tie | 8, encodeE2M1(@bitCast(case.midpoint | 0x8000_0000)));
        try std.testing.expectEqual(case.upper | 8, encodeE2M1(@bitCast((case.midpoint + 1) | 0x8000_0000)));
    }
}

test "E2M1 special values preserve signs and canonicalize NaNs" {
    try std.testing.expectEqual(@as(u8, 0), encodeE2M1(@bitCast(@as(u32, 1))));
    try std.testing.expectEqual(@as(u8, 8), encodeE2M1(@bitCast(@as(u32, 0x8000_0001))));
    try std.testing.expectEqual(@as(u8, 7), encodeE2M1(@bitCast(@as(u32, 0x7f7f_ffff))));
    try std.testing.expectEqual(@as(u8, 15), encodeE2M1(@bitCast(@as(u32, 0xff7f_ffff))));
    try std.testing.expectEqual(@as(u8, 7), encodeE2M1(std.math.inf(f32)));
    try std.testing.expectEqual(@as(u8, 15), encodeE2M1(-std.math.inf(f32)));
    for ([_]u32{ 0x7f80_0001, 0x7fc0_0000, 0xff80_0001, 0xffc0_0000 }) |bits| {
        try std.testing.expectEqual(@as(u8, 7), encodeE2M1(@bitCast(bits)));
    }
}

test "E2M1 carrier masks and clears its high nibble" {
    for (0..256) |carrier| {
        try std.testing.expectEqual(@as(u32, @bitCast(decodeE2M1(@intCast(carrier & 0x0f)))), @as(u32, @bitCast(decodeE2M1(@intCast(carrier)))));
        try std.testing.expectEqual(@as(u8, 0), encodeE2M1(@bitCast(@as(u32, @intCast(carrier)) << 23)) & 0xf0);
    }
}

test "packed bytes decode both ordered nibbles" {
    for (0..256) |packed_value| {
        try std.testing.expectEqual(@as(u32, @bitCast(decodeE2M1(@intCast(packed_value & 0x0f)))), @as(u32, @bitCast(decodeE2M1(@intCast(packed_value)))));
        try std.testing.expectEqual(@as(u32, @bitCast(decodeE2M1(@intCast((packed_value >> 4) & 0x0f)))), @as(u32, @bitCast(decodeE2M1(@intCast(packed_value >> 4)))));
    }
}

test "scale policies use local then global order and reverse undo order" {
    const cases = [_]struct {
        block: ScaleApplication,
        global: ScaleApplication,
        expected: f32,
    }{
        .{ .block = .multiply, .global = .multiply, .expected = 24 },
        .{ .block = .multiply, .global = .divide, .expected = 1.5 },
        .{ .block = .divide, .global = .multiply, .expected = 6 },
        .{ .block = .divide, .global = .divide, .expected = 0.375 },
    };
    for (cases) |case| {
        const decoded = dequantize(5, 0x40, 4, case.block, case.global);
        try std.testing.expectEqual(@as(u32, @bitCast(case.expected)), @as(u32, @bitCast(decoded)));
        try std.testing.expectEqual(@as(u8, 5), quantize(decoded, 0x40, 4, case.block, case.global));
    }

    const dequantize_order_cases = [_]struct {
        payload: u8,
        block_scale: u8,
        global_scale_bits: u32,
        block_application: ScaleApplication,
        global_application: ScaleApplication,
        expected_bits: u32,
        reordered_bits: u32,
    }{
        .{ .payload = 0x03, .block_scale = 0x03, .global_scale_bits = 0x3e00_0001, .block_application = .multiply, .global_application = .multiply, .expected_bits = 0x3a90_0001, .reordered_bits = 0x3a90_0002 },
        .{ .payload = 0x01, .block_scale = 0x03, .global_scale_bits = 0x3e00_0001, .block_application = .multiply, .global_application = .divide, .expected_bits = 0x3cbf_ffff, .reordered_bits = 0x3cbf_fffe },
        .{ .payload = 0x01, .block_scale = 0x05, .global_scale_bits = 0x3e00_0001, .block_application = .divide, .global_application = .multiply, .expected_bits = 0x40cc_cccf, .reordered_bits = 0x40cc_ccce },
        .{ .payload = 0x01, .block_scale = 0x03, .global_scale_bits = 0x3e00_0001, .block_application = .divide, .global_application = .divide, .expected_bits = 0x442a_aaaa, .reordered_bits = 0x442a_aaa9 },
    };
    for (dequantize_order_cases) |case| {
        const global_scale: f32 = @bitCast(case.global_scale_bits);
        const block_scale = low_float.decode(.f8_e4m3, case.block_scale) catch unreachable;
        const ordered = dequantize(case.payload, case.block_scale, global_scale, case.block_application, case.global_application);
        const reordered = apply(
            apply(decodeE2M1(case.payload), global_scale, case.global_application),
            block_scale,
            case.block_application,
        );
        try std.testing.expectEqual(case.expected_bits, @as(u32, @bitCast(ordered)));
        try std.testing.expectEqual(case.reordered_bits, @as(u32, @bitCast(reordered)));
        try std.testing.expect(case.expected_bits != case.reordered_bits);
    }

    const quantize_order_cases = [_]struct {
        value_bits: u32,
        block_scale: u8,
        global_scale_bits: u32,
        block_application: ScaleApplication,
        global_application: ScaleApplication,
        expected: u8,
        reordered: u8,
    }{
        .{ .value_bits = 0x3e00_0522, .block_scale = 0x23, .global_scale_bits = 0x3e54_d004, .block_application = .multiply, .global_application = .multiply, .expected = 0x06, .reordered = 0x05 },
        .{ .value_bits = 0x3e00_154f, .block_scale = 0x0f, .global_scale_bits = 0x3ed1_dd10, .block_application = .multiply, .global_application = .divide, .expected = 0x04, .reordered = 0x03 },
        .{ .value_bits = 0x3e00_154f, .block_scale = 0x47, .global_scale_bits = 0x3f20_1aa3, .block_application = .divide, .global_application = .multiply, .expected = 0x01, .reordered = 0x02 },
        .{ .value_bits = 0x3e00_399b, .block_scale = 0x55, .global_scale_bits = 0x3eeb_e49b, .block_application = .divide, .global_application = .divide, .expected = 0x01, .reordered = 0x02 },
    };
    for (quantize_order_cases) |case| {
        const value: f32 = @bitCast(case.value_bits);
        const global_scale: f32 = @bitCast(case.global_scale_bits);
        const block_scale = low_float.decode(.f8_e4m3, case.block_scale) catch unreachable;
        const ordered = quantize(value, case.block_scale, global_scale, case.block_application, case.global_application);
        const reordered = encodeE2M1(undo(
            undo(value, block_scale, case.block_application),
            global_scale,
            case.global_application,
        ));
        try std.testing.expectEqual(case.expected, ordered);
        try std.testing.expectEqual(case.reordered, reordered);
        try std.testing.expect(case.expected != case.reordered);
    }
}

test "scale special values follow IEEE arithmetic" {
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(dequantize(4, 0x00, 2, .multiply, .multiply))));
    try std.testing.expectEqual(@as(u32, 0xc000_0000), @as(u32, @bitCast(dequantize(2, 0xb8, 2, .multiply, .multiply))));
    try std.testing.expectEqual(@as(u32, 0x43e0_0000), @as(u32, @bitCast(dequantize(2, 0x7e, 1, .multiply, .multiply))));
    try std.testing.expect(std.math.isNan(dequantize(2, 0x7f, 1, .multiply, .multiply)));
    try std.testing.expectEqual(@as(u8, 7), quantize(1, 0x00, 1, .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 10), quantize(1, 0xb8, 1, .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 0), quantize(1, 0x7e, 1, .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 7), quantize(1, 0x7f, 1, .multiply, .multiply));

    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(dequantize(2, 0x38, 0, .multiply, .multiply))));
    try std.testing.expectEqual(@as(u32, 0xbf80_0000), @as(u32, @bitCast(dequantize(2, 0x38, -1, .multiply, .multiply))));
    try std.testing.expect(std.math.isInf(dequantize(2, 0x38, std.math.inf(f32), .multiply, .multiply)));
    try std.testing.expect(std.math.isNan(dequantize(2, 0x38, std.math.nan(f32), .multiply, .multiply)));
    try std.testing.expectEqual(@as(u8, 7), quantize(1, 0x38, 0, .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 10), quantize(1, 0x38, -1, .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 0), quantize(1, 0x38, std.math.inf(f32), .multiply, .multiply));
    try std.testing.expectEqual(@as(u8, 7), quantize(1, 0x38, std.math.nan(f32), .multiply, .multiply));
}

test "canonical blocks and padded row and column layouts use scalar formulas" {
    var payloads: [16]u8 = undefined;
    for (&payloads, 0..) |*payload, i| payload.* = @intCast(i);
    for (payloads) |payload| {
        const expected = decodeE2M1(payload) * 2 * 3;
        try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(dequantize(payload, 0x40, 3, .multiply, .multiply))));
    }

    const rows: usize = 3;
    const columns: usize = 17;
    const padded_columns = std.mem.alignForward(usize, columns, 16);
    const padded_rows = std.mem.alignForward(usize, rows, 16);
    const row_scales: [6]u8 = .{ 0x38, 0x40, 0x39, 0x38, 0x40, 0x39 };
    var column_scales: [34]u8 = undefined;
    for (&column_scales, 0..) |*scale, i| scale.* = if (i % 2 == 0) 0x38 else 0x40;
    for (0..rows) |row| for (0..columns) |column| {
        const row_physical = row * padded_columns + column;
        const column_physical = column * padded_rows + row;
        const payload: u8 = @intCast((row + column) & 0x0f);
        const row_expected = decodeE2M1(payload) * (low_float.decode(.f8_e4m3, row_scales[row_physical / 16]) catch unreachable);
        const column_expected = decodeE2M1(payload) * (low_float.decode(.f8_e4m3, column_scales[column_physical / 16]) catch unreachable);
        try std.testing.expectEqual(@as(u32, @bitCast(row_expected)), @as(u32, @bitCast(dequantize(payload, row_scales[row_physical / 16], 1, .multiply, .multiply))));
        try std.testing.expectEqual(@as(u32, @bitCast(column_expected)), @as(u32, @bitCast(dequantize(payload, column_scales[column_physical / 16], 1, .multiply, .multiply))));
    };
}
