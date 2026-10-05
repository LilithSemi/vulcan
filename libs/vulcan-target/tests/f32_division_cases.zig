const std = @import("std");

pub const Pair = struct {
    numerator: u32,
    denominator: u32,
};

pub const Halfway = struct {
    numerator: u32,
    denominator: u32,
    expected: u32,
};

pub const halfway_cases = [_]Halfway{
    .{ .numerator = 0x0000_0001, .denominator = 0x4000_0000, .expected = 0x0000_0000 },
    .{ .numerator = 0x0000_0005, .denominator = 0x4000_0000, .expected = 0x0000_0002 },
    .{ .numerator = 0x0000_0009, .denominator = 0x4000_0000, .expected = 0x0000_0004 },
    .{ .numerator = 0x0000_000d, .denominator = 0x4000_0000, .expected = 0x0000_0006 },
    .{ .numerator = 0x0000_0011, .denominator = 0x4000_0000, .expected = 0x0000_0008 },
    .{ .numerator = 0x0000_0015, .denominator = 0x4000_0000, .expected = 0x0000_000a },
    .{ .numerator = 0x0000_0019, .denominator = 0x4000_0000, .expected = 0x0000_000c },
    .{ .numerator = 0x0000_001d, .denominator = 0x4000_0000, .expected = 0x0000_000e },
    .{ .numerator = 0x0000_0021, .denominator = 0x4000_0000, .expected = 0x0000_0010 },
    .{ .numerator = 0x0000_0025, .denominator = 0x4000_0000, .expected = 0x0000_0012 },
    .{ .numerator = 0x0000_0029, .denominator = 0x4000_0000, .expected = 0x0000_0014 },
    .{ .numerator = 0x0000_002d, .denominator = 0x4000_0000, .expected = 0x0000_0016 },
    .{ .numerator = 0x0000_0031, .denominator = 0x4000_0000, .expected = 0x0000_0018 },
    .{ .numerator = 0x0000_0035, .denominator = 0x4000_0000, .expected = 0x0000_001a },
    .{ .numerator = 0x0000_0039, .denominator = 0x4000_0000, .expected = 0x0000_001c },
    .{ .numerator = 0x0000_003d, .denominator = 0x4000_0000, .expected = 0x0000_001e },
    .{ .numerator = 0x0000_0003, .denominator = 0x4000_0000, .expected = 0x0000_0002 },
    .{ .numerator = 0x0000_0007, .denominator = 0x4000_0000, .expected = 0x0000_0004 },
    .{ .numerator = 0x0000_000b, .denominator = 0x4000_0000, .expected = 0x0000_0006 },
    .{ .numerator = 0x0000_000f, .denominator = 0x4000_0000, .expected = 0x0000_0008 },
    .{ .numerator = 0x0000_0013, .denominator = 0x4000_0000, .expected = 0x0000_000a },
    .{ .numerator = 0x0000_0017, .denominator = 0x4000_0000, .expected = 0x0000_000c },
    .{ .numerator = 0x0000_001b, .denominator = 0x4000_0000, .expected = 0x0000_000e },
    .{ .numerator = 0x0000_001f, .denominator = 0x4000_0000, .expected = 0x0000_0010 },
    .{ .numerator = 0x0000_0023, .denominator = 0x4000_0000, .expected = 0x0000_0012 },
    .{ .numerator = 0x0000_0027, .denominator = 0x4000_0000, .expected = 0x0000_0014 },
    .{ .numerator = 0x0000_002b, .denominator = 0x4000_0000, .expected = 0x0000_0016 },
    .{ .numerator = 0x0000_002f, .denominator = 0x4000_0000, .expected = 0x0000_0018 },
    .{ .numerator = 0x0000_0033, .denominator = 0x4000_0000, .expected = 0x0000_001a },
    .{ .numerator = 0x0000_0037, .denominator = 0x4000_0000, .expected = 0x0000_001c },
    .{ .numerator = 0x0000_003b, .denominator = 0x4000_0000, .expected = 0x0000_001e },
    .{ .numerator = 0x0000_003f, .denominator = 0x4000_0000, .expected = 0x0000_0020 },
};

pub const named_cases = [_]Pair{
    .{ .numerator = 0x0000_0000, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x8000_0000, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x0000_0001, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x0000_0001, .denominator = 0x4000_0000 },
    .{ .numerator = 0x0000_0001, .denominator = 0x3fff_ffff },
    .{ .numerator = 0x0000_0001, .denominator = 0x4000_0001 },
    .{ .numerator = 0x007f_ffff, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x0080_0000, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x0080_0001, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x0080_0000, .denominator = 0x4000_0000 },
    .{ .numerator = 0x0100_0000, .denominator = 0x4000_0000 },
    .{ .numerator = 0x0100_0001, .denominator = 0x4000_0000 },
    .{ .numerator = 0x7f7f_ffff, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x7f7f_ffff, .denominator = 0x3f7f_ffff },
    .{ .numerator = 0x7f7f_fffe, .denominator = 0x3f7f_ffff },
    .{ .numerator = 0x3f80_0000, .denominator = 0x7f7f_ffff },
    .{ .numerator = 0x7f80_0000, .denominator = 0x3f80_0000 },
    .{ .numerator = 0xff80_0000, .denominator = 0x3f80_0000 },
    .{ .numerator = 0x3f80_0000, .denominator = 0x0000_0000 },
    .{ .numerator = 0xbf80_0000, .denominator = 0x0000_0000 },
};

pub const Corpus = struct {
    allocator: std.mem.Allocator,
    pairs: []Pair,

    pub fn deinit(self: *Corpus) void {
        self.allocator.free(self.pairs);
        self.* = undefined;
    }
};

fn nextRandom(state: *u64) u32 {
    state.* ^= state.* >> 12;
    state.* ^= state.* << 25;
    state.* ^= state.* >> 27;
    return @truncate(state.* *% 0x2545_f491_4f6c_dd1d);
}

fn appendUnique(allocator: std.mem.Allocator, seen: *std.AutoHashMapUnmanaged(Pair, void), pairs: *std.ArrayList(Pair), pair: Pair) !void {
    const result = try seen.getOrPut(allocator, pair);
    if (!result.found_existing) try pairs.append(allocator, pair);
}

pub fn build(allocator: std.mem.Allocator) !Corpus {
    var seen: std.AutoHashMapUnmanaged(Pair, void) = .empty;
    defer seen.deinit(allocator);
    var pairs: std.ArrayList(Pair) = .empty;
    errdefer pairs.deinit(allocator);

    const classes = [_]u32{
        0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001,
        0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000,
        0x3f80_0000, 0xbf80_0000, 0x7f7f_ffff, 0xff7f_ffff,
        0x7f80_0000, 0xff80_0000, 0x7f80_0001, 0xff80_0001,
        0x7fc0_0001, 0xffc0_0001,
    };
    for (classes) |numerator| for (classes) |denominator| {
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = numerator, .denominator = denominator });
    };

    const mantissas = [_]u32{ 0, 1, 0x003f_ffff, 0x007f_fffe, 0x007f_ffff };
    for (0..256) |numerator_exponent| for (0..256) |denominator_exponent| {
        for (mantissas, 0..) |numerator_mantissa, template_index| {
            const denominator_mantissa = mantissas[(template_index + numerator_exponent + denominator_exponent) % mantissas.len];
            const numerator_sign: u32 = if ((numerator_exponent + denominator_exponent + template_index) & 1 == 0) 0 else 0x8000_0000;
            const denominator_sign: u32 = if ((numerator_exponent * 3 + denominator_exponent + template_index) & 2 == 0) 0 else 0x8000_0000;
            try appendUnique(allocator, &seen, &pairs, .{
                .numerator = numerator_sign | (@as(u32, @intCast(numerator_exponent)) << 23) | numerator_mantissa,
                .denominator = denominator_sign | (@as(u32, @intCast(denominator_exponent)) << 23) | denominator_mantissa,
            });
        }
    };

    for (named_cases) |case| try appendUnique(allocator, &seen, &pairs, case);
    for (halfway_cases) |case| {
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = case.numerator, .denominator = case.denominator });
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = case.numerator - 1, .denominator = case.denominator });
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = case.numerator + 1, .denominator = case.denominator });
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = case.numerator | 0x8000_0000, .denominator = case.denominator });
    }

    var random_state: u64 = 0x6e76_6670_3464_6976;
    var random_count: usize = 0;
    while (random_count < 65_536) : (random_count += 1) {
        try appendUnique(allocator, &seen, &pairs, .{ .numerator = nextRandom(&random_state), .denominator = nextRandom(&random_state) });
    }
    try std.testing.expectEqual(@as(usize, 65_536), random_count);

    var exponent_pairs: [65_536]bool = @splat(false);
    var numerator_coverage: [256][5][2]bool = @splat(@splat(.{ false, false }));
    var denominator_coverage: [256][5][2]bool = @splat(@splat(.{ false, false }));
    for (pairs.items) |pair| {
        const numerator_exponent: usize = @intCast((pair.numerator >> 23) & 0xff);
        const denominator_exponent: usize = @intCast((pair.denominator >> 23) & 0xff);
        exponent_pairs[numerator_exponent * 256 + denominator_exponent] = true;
        for (mantissas, 0..) |mantissa, template_index| {
            if (pair.numerator & 0x007f_ffff == mantissa) numerator_coverage[numerator_exponent][template_index][pair.numerator >> 31] = true;
            if (pair.denominator & 0x007f_ffff == mantissa) denominator_coverage[denominator_exponent][template_index][pair.denominator >> 31] = true;
        }
    }
    for (exponent_pairs) |covered| try std.testing.expect(covered);
    for (numerator_coverage) |by_template| for (by_template) |by_sign| for (by_sign) |covered| try std.testing.expect(covered);
    for (denominator_coverage) |by_template| for (by_template) |by_sign| for (by_sign) |covered| try std.testing.expect(covered);
    for (named_cases) |case| try std.testing.expect(seen.contains(case));
    for (halfway_cases) |case| {
        try std.testing.expect(seen.contains(.{ .numerator = case.numerator, .denominator = case.denominator }));
        try std.testing.expect(seen.contains(.{ .numerator = case.numerator - 1, .denominator = case.denominator }));
        try std.testing.expect(seen.contains(.{ .numerator = case.numerator + 1, .denominator = case.denominator }));
        try std.testing.expect(seen.contains(.{ .numerator = case.numerator | 0x8000_0000, .denominator = case.denominator }));
    }
    random_state = 0x6e76_6670_3464_6976;
    random_count = 0;
    while (random_count < 65_536) : (random_count += 1) {
        try std.testing.expect(seen.contains(.{ .numerator = nextRandom(&random_state), .denominator = nextRandom(&random_state) }));
    }
    try std.testing.expectEqual(@as(usize, 65_536), random_count);

    return .{ .allocator = allocator, .pairs = try pairs.toOwnedSlice(allocator) };
}

test "division corpus retains every required generated source" {
    var corpus = try build(std.testing.allocator);
    defer corpus.deinit();
    try std.testing.expect(corpus.pairs.len >= 256 * 256 * 5);
    for (halfway_cases) |case| {
        const denominator: f32 = @bitCast(case.denominator);
        try std.testing.expectEqual(case.expected, @as(u32, @bitCast(@as(f32, @bitCast(case.numerator)) / denominator)));
        const below: u32 = @bitCast(@as(f32, @bitCast(case.numerator - 1)) / denominator);
        const above: u32 = @bitCast(@as(f32, @bitCast(case.numerator + 1)) / denominator);
        try std.testing.expect(below <= case.expected);
        try std.testing.expect(above >= case.expected);
        try std.testing.expect(below != above);
    }
}
