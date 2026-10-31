//! Static IR score for microarchitecture-aware transforms.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const schedule = @import("schedule.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const UnitClass = mm.UnitClass;

pub const Score = struct {
    cycles: u64 = 0,
    instructions: u64 = 0,
    peak_live: u32 = 0,
    idle_slots: u64 = 0,

    pub fn accepts(candidate: Score, baseline: Score) bool {
        if (candidate.cycles != baseline.cycles) return candidate.cycles < baseline.cycles;
        if (candidate.peak_live != baseline.peak_live) return candidate.peak_live < baseline.peak_live;
        if (candidate.instructions != baseline.instructions) return candidate.instructions < baseline.instructions;
        return candidate.idle_slots <= baseline.idle_slots;
    }
};

pub fn function(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) std.mem.Allocator.Error!Score {
    var result: Score = .{};
    for (0..func.blockCount()) |bi| {
        const block_result = try block(allocator, func, @fromBackingInt(@intCast(bi)), model);
        result.cycles += block_result.cycles;
        result.instructions += block_result.instructions;
        result.peak_live = @max(result.peak_live, block_result.peak_live);
        result.idle_slots += block_result.idle_slots;
    }
    return result;
}

pub fn block(allocator: std.mem.Allocator, func: *const Function, block_id: Block, model: *const mm.Model) std.mem.Allocator.Error!Score {
    const insts = func.blockInsts(block_id);
    if (insts.len == 0) return .{};

    const class_count = @typeInfo(UnitClass).@"enum".field_names.len;
    var demand: [class_count]u64 = @splat(0);
    var total_demand: u64 = 0;
    var critical_path: u64 = 0;

    const ready = try allocator.alloc(u64, func.valueCount());
    defer allocator.free(ready);
    @memset(ready, 0);

    const def_at = try allocator.alloc(usize, func.valueCount());
    defer allocator.free(def_at);
    @memset(def_at, std.math.maxInt(usize));
    const last_use = try allocator.alloc(usize, func.valueCount());
    defer allocator.free(last_use);
    @memset(last_use, 0);

    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);

    for (insts, 0..) |inst, index| {
        try schedule.collectOperands(allocator, func, inst, &operands);
        var starts_at: u64 = 0;
        for (operands.items) |operand| {
            starts_at = @max(starts_at, ready[@backingInt(operand)]);
            if (def_at[@backingInt(operand)] != std.math.maxInt(usize))
                last_use[@backingInt(operand)] = index;
        }

        const opcode = func.opcode(inst);
        const latency: u64 = model.latency(opcode);
        const finishes_at = starts_at + latency;
        critical_path = @max(critical_path, finishes_at);
        if (func.instResult(inst)) |result| {
            ready[@backingInt(result)] = finishes_at;
            def_at[@backingInt(result)] = index;
            last_use[@backingInt(result)] = index;
        }

        const weight: u64 = model.throughput(opcode, elementIsFloat(func, inst));
        demand[@backingInt(model.unitOf(opcode))] += weight;
        total_demand += weight;
    }

    const width: u64 = @max(1, model.issue_width);
    var resource_cycles = ceilDiv(total_demand, width);
    inline for (0..class_count) |class_index| {
        const class: UnitClass = @fromBackingInt(@intCast(class_index));
        const ports = schedule.classPorts(model, class);
        if (ports != std.math.maxInt(u32))
            resource_cycles = @max(resource_cycles, ceilDiv(demand[class_index], ports));
    }

    const ends = try allocator.alloc(u32, insts.len);
    defer allocator.free(ends);
    @memset(ends, 0);
    for (insts, 0..) |inst, index| {
        if (func.instResult(inst)) |result| {
            const end = last_use[@backingInt(result)];
            if (end > index) ends[end] += 1;
        }
    }

    var live: u32 = 0;
    var peak_live: u32 = 0;
    for (insts, 0..) |inst, index| {
        if (func.instResult(inst)) |result| {
            if (last_use[@backingInt(result)] > index) live += 1;
        }
        peak_live = @max(peak_live, live);
        live -= ends[index];
    }

    const cycles = @max(critical_path, resource_cycles);
    return .{
        .cycles = cycles,
        .instructions = insts.len,
        .peak_live = peak_live,
        .idle_slots = cycles * width -| total_demand,
    };
}

fn ceilDiv(numerator: u64, denominator: anytype) u64 {
    const divisor: u64 = denominator;
    return (numerator + divisor - 1) / divisor;
}

fn elementIsFloat(func: *const Function, inst: ir.function.Inst) bool {
    const value = switch (func.opcode(inst)) {
        .arith => |op| op.lhs,
        .arith_imm => |op| op.lhs,
        else => return false,
    };
    return switch (func.types.type_kind(func.valueType(value))) {
        .float => true,
        .vector => |vector| switch (func.types.type_kind(vector.elem)) {
            .float => true,
            else => false,
        },
        else => false,
    };
}

test "score prefers lower cycles before instruction count" {
    const baseline = Score{ .cycles = 8, .instructions = 4, .peak_live = 2 };
    try std.testing.expect((Score{ .cycles = 7, .instructions = 8, .peak_live = 3 }).accepts(baseline));
    try std.testing.expect(!(Score{ .cycles = 9, .instructions = 2, .peak_live = 1 }).accepts(baseline));
}

test "score uses pressure and size to break cycle ties" {
    const baseline = Score{ .cycles = 8, .instructions = 6, .peak_live = 4 };
    try std.testing.expect((Score{ .cycles = 8, .instructions = 8, .peak_live = 3 }).accepts(baseline));
    try std.testing.expect((Score{ .cycles = 8, .instructions = 5, .peak_live = 4 }).accepts(baseline));
}
