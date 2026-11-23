const std = @import("std");
const warp = @import("warp.zig");
const programs = @import("programs.zig").programs;
test "bounded compute loops preserve two dimensional grid and subgroup indices" {
    inline for (.{ .{ @as(u32, 8), @as(u32, 23), @as(u32, 7) }, .{ @as(u32, 1024), @as(u32, 3072), @as(u32, 100) }, .{ @as(u32, 0), @as(u32, 60000), @as(u32, 100) } }, 0..) |range, k| {
        const start = range[0];
        const end = range[1];
        const grid = range[2];
        const data = try std.testing.allocator.alloc(u32, end + 512);
        defer std.testing.allocator.free(data);
        @memset(data, 0xffffffff);
        const counts = try std.testing.allocator.alloc(u64, data.len);
        defer std.testing.allocator.free(counts);
        @memset(counts, 0);
        const scratch_base = 0x1000 + end * 4;
        var mem: warp.Memory = .{ .base = 0x1000, .words = data, .access_words = counts, .scratch = .{ .base = scratch_base, .stride = 256, .harts = 8 } };
        for (0..8) |i| {
            const hart: u32 = @intCast(i);
            var w: warp.Warp = .{ .hart_id = hart, .hart_count = 8, .workgroup_base = start, .grid_dim = grid, .lanes = 8, .stack_top = scratch_base + (hart + 1) * 256, .scratch_bytes = 256 };
            var code = programs[k];
            _ = try warp.run(&code, &mem, &w, 1000000);
        }
        for (0..end) |i| {
            try std.testing.expectEqual(if (i < start) @as(u32, 0xffffffff) else grid, data[i]);
            try std.testing.expectEqual(if (i < start) @as(u64, 0) else 1, counts[i]);
        }
    }
}

test "compiled 2D compute shader matches individual and bounded launches" {
    const allocator = std.testing.allocator;
    const raw = [_][]const u8{ @embedFile("grid.bin"), @embedFile("grid24.bin"), @embedFile("grid48.bin") };
    var code: [3][]u32 = undefined;
    for (raw, 0..) |bytes, k| {
        code[k] = try allocator.alloc(u32, bytes.len / 4);
        for (code[k], 0..) |*word, i| word.* = std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little);
    }
    defer for (code) |words| allocator.free(words);
    for ([_]bool{ false, true }) |batched| {
        const data = try allocator.alloc(u32, 4096 + 16384 * 8 / 4);
        defer allocator.free(data);
        @memset(data, 0);
        const counts = try allocator.alloc(u64, data.len);
        defer allocator.free(counts);
        @memset(counts, 0);
        data[0] = 0x1000 + 256 * 4;
        data[1] = 384 * 4;
        data[2] = 0x1000 + 20 * 4;
        data[3] = 0x1000 + 21 * 4;
        const scratch_base = 0x1000 + 4096 * 4;
        var mem: warp.Memory = .{ .base = 0x1000, .words = data, .access_words = counts, .scratch = .{ .base = scratch_base, .stride = 16384, .harts = 8 } };
        const launch_count: usize = if (batched) 2 else 6;
        for (0..launch_count) |launch| {
            for (0..8) |h| {
                const hart: u32 = @intCast(h);
                var w: warp.Warp = .{ .hart_id = hart, .hart_count = 8, .workgroup_base = @intCast(launch * (if (batched) @as(usize, 24) else 8)), .grid_dim = 3, .lanes = 8, .stack_top = scratch_base + (hart + 1) * 16384, .scratch_bytes = 16384 };
                w.launch_args[0] = 0x1000;
                _ = try warp.run(code[if (batched) launch + 1 else 0], &mem, &w, 1000000);
            }
        }
        for (0..384) |i| {
            const x: u32 = @intCast(i % 24);
            const y: u32 = @intCast(i / 24);
            var expected = x + 101 * y;
            for (0..(x & 3) + 1) |j| expected = (expected *% 17) ^ @as(u32, @intCast(j));
            try std.testing.expectEqual(expected, data[256 + i]);
            try std.testing.expectEqual(@as(u64, 1), counts[256 + i]);
        }
    }
}

fn imageDescriptor(d: []u32, base: u32, width: u32, height: u32) void {
    @memset(d[0..36], 0);
    d[0] = base;
    d[1] = width * 4;
    d[2] = width;
    d[3] = height;
    d[4] = @bitCast(@as(f32, @floatFromInt(width)));
    d[5] = @bitCast(@as(f32, @floatFromInt(height)));
    d[6] = 2;
    d[7] = 2;
    d[12] = 4;
    d[13] = 1;
    d[14] = @bitCast(@as(f32, 1) / @as(f32, @floatFromInt(width)));
    d[15] = @bitCast(@as(f32, 1) / @as(f32, @floatFromInt(height)));
    d[18] = 3;
    d[19] = 4;
    d[20] = 5;
    d[21] = 6;
    d[23] = width * height * 4;
    d[24] = 1;
    d[25] = 1;
    d[26] = 0x1000 + 720 * 4;
    d[27] = 0x1000 + 724 * 4;
    d[28] = 1;
    d[29] = @bitCast(@as(f32, 1));
    d[30] = @bitCast(@as(f32, 1));
    d[31] = 2;
}
