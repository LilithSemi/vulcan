const std = @import("std");
const warp = @import("warp.zig");
test "row partition preserves raster records and per pixel draw order" {
    const a = std.testing.allocator;
    const raw = [_][]const u8{ @embedFile("raster-baseline.bin"), @embedFile("raster-multiwarp.bin") };
    const reference = try a.alloc(u32, 768 * 128);
    defer a.free(reference);
    @memset(reference, 0);
    var reference_counts: [384]u32 = @splat(0);
    for (0..2) |mode| {
        const code = try a.alloc(u32, raw[mode].len / 4);
        defer a.free(code);
        for (code, 0..) |*word, i| word.* = std.mem.readInt(u32, raw[mode][i * 4 ..][0..4], .little);
        const scratch_word = 4096 + 1024 * 128;
        const data = try a.alloc(u32, scratch_word + 8 * 4096);
        defer a.free(data);
        @memset(data, 0);
        const locations = [_]u32{ 1024, 128, 256, 2048, 4096 };
        const lengths = [_]u32{ 128, 64, 384, 1024, 1024 * 128 };
        for (locations, lengths, 0..) |location, length, i| {
            data[i * 4] = 0x1000 + location * 4;
            data[i * 4 + 1] = length * 4;
            data[i * 4 + 2] = 0x1000 + 800 * 4;
            data[i * 4 + 3] = 0x1000 + 804 * 4;
        }
        data[1024] = 0;
        data[1025] = 15;
        data[1026] = 512;
        data[1027] = 512;
        data[1028] = 1024;
        data[1029] = 1;
        data[1030] = @bitCast(@as(f32, 1));
        data[1031] = 1;
        data[1032] = 0x100 | (1 << 9) | (2 << 16);
        for (0..16) |y| data[1024 + 72 + y] = 23 << 16;
        const xy = [_][2]f32{ .{ 0, 0 }, .{ 0, 16 }, .{ 24, 0 } };
        for (xy, 0..) |v, i| {
            const off = 256 + i * 128 + 28;
            data[off] = @bitCast(v[0]);
            data[off + 1] = @bitCast(v[1]);
            data[off + 2] = @bitCast(@as(f32, 0.5));
            data[off + 3] = @bitCast(@as(f32, 1));
        }
        const groups: u32 = if (mode == 0) 1 else 8;
        const scratch_base = 0x1000 + scratch_word * 4;
        var mem: warp.Memory = .{ .base = 0x1000, .words = data, .scratch = .{ .base = scratch_base, .stride = 16384, .harts = 8 } };
        for (0..2) |draw| {
            for (0..3) |v| data[256 + v * 128] = @bitCast(@as(f32, @floatFromInt(1 + v + draw * 10)));
            for (0..groups) |h| {
                const hart: u32 = @intCast(h);
                var w: warp.Warp = .{ .hart_id = hart, .hart_count = 8, .grid_dim = groups, .lanes = 8, .stack_top = scratch_base + (hart + 1) * 16384, .scratch_bytes = 16384 };
                for (0..5) |i| w.launch_args[i] = 0x1000 + @as(u32, @intCast(i)) * 16;
                _ = try warp.run(code, &mem, &w, 5000000);
            }
        }
        var seen: [384]u32 = @splat(0);
        const channels = groups * 8;
        const per_channel = 1024 / channels;
        for (0..channels) |channel| {
            const count = data[128 + channel];
            try std.testing.expect(count <= per_channel);
            for (0..count) |n| {
                const slot = channel * per_channel + n;
                const coord = data[2048 + slot];
                const x = coord & 65535;
                const y = coord >> 16;
                try std.testing.expect(x < 24 and y < 16);
                const pixel = y * 24 + x;
                try std.testing.expect(seen[pixel] < 2);
                const start = (pixel * 2 + seen[pixel]) * 128;
                const record = data[4096 + slot * 128 ..][0..128];
                if (mode == 0) @memcpy(reference[start..][0..128], record) else try std.testing.expectEqualSlices(u32, reference[start..][0..128], record);
                seen[pixel] += 1;
            }
        }
        if (mode == 0) {
            reference_counts = seen;
            var covered: usize = 0;
            for (seen) |n| {
                try std.testing.expect(n == 0 or n == 2);
                if (n == 2) covered += 1;
            }
            try std.testing.expect(covered > 150 and covered < 220);
        } else try std.testing.expectEqualSlices(u32, &reference_counts, &seen);
    }
}
