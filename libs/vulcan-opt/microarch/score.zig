//! Static IR score for microarchitecture-aware transforms.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const schedule = @import("schedule.zig");
const registry = @import("registry.zig");
const cfg_mod = @import("../cfg.zig");
const dominators = @import("../dominators.zig");
const loops = @import("../loops.zig");

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
        if (candidate.instructions != baseline.instructions) return candidate.instructions < baseline.instructions;
        if (candidate.peak_live != baseline.peak_live) return candidate.peak_live < baseline.peak_live;
        return candidate.idle_slots <= baseline.idle_slots;
    }
};

/// Owns a speculative pre-selection rewrite until its static score improves.
pub const IRGate = struct {
    allocator: std.mem.Allocator,
    model: *const mm.Model,
    baseline: Score,
    candidate: ?Function,

    pub fn init(allocator: std.mem.Allocator, original: *const Function, model: *const mm.Model) std.mem.Allocator.Error!IRGate {
        return .{
            .allocator = allocator,
            .model = model,
            .baseline = try function(allocator, original, model),
            .candidate = try original.clone(allocator),
        };
    }

    pub fn deinit(self: *IRGate) void {
        if (self.candidate) |*candidate| candidate.deinit();
        self.candidate = null;
    }

    pub fn candidateFunction(self: *IRGate) *Function {
        return &self.candidate.?;
    }

    pub fn commit(self: *IRGate, original: *Function) std.mem.Allocator.Error!bool {
        const candidate_score = try function(self.allocator, &self.candidate.?, self.model);
        if (!candidate_score.accepts(self.baseline)) return false;

        original.deinit();
        original.* = self.candidate.?;
        self.candidate = null;
        return true;
    }
};

pub fn function(allocator: std.mem.Allocator, func: *const Function, model: *const mm.Model) std.mem.Allocator.Error!Score {
    var graph = try cfg_mod.build(allocator, func);
    defer graph.deinit(allocator);
    var doms = try dominators.computeFromCfg(allocator, &graph);
    defer doms.deinit(allocator);
    var loop_info = try loops.analyzeFromCfg(allocator, &graph, &doms);
    defer loop_info.deinit(allocator);

    var result: Score = .{};
    for (0..func.blockCount()) |bi| {
        const block_id: Block = @fromBackingInt(@intCast(bi));
        const block_result = try block(allocator, func, block_id, model);
        const weight = executionWeight(&graph, &doms, &loop_info, bi);
        result.cycles += block_result.cycles * weight;
        result.instructions += block_result.instructions * weight;
        result.peak_live = @max(result.peak_live, block_result.peak_live);
        result.idle_slots += block_result.idle_slots * weight;

        if (func.terminator(block_id)) |term| switch (term) {
            .jump => |jump| if (@backingInt(jump.target) <= bi) {
                result.cycles += weight;
                result.instructions += weight;
                result.idle_slots += (@as(u64, @max(1, model.issue_width)) - 1) * weight;
            },
            .ret => {},
        };
    }
    result.peak_live = @max(result.peak_live, try cfgPeakLive(allocator, func, &graph));
    return result;
}

fn cfgPeakLive(allocator: std.mem.Allocator, func: *const Function, graph: *const cfg_mod.Cfg) std.mem.Allocator.Error!u32 {
    const block_count = func.blockCount();
    const value_count = func.valueCount();
    const cells = block_count * value_count;
    const defined = try allocator.alloc(bool, cells);
    defer allocator.free(defined);
    const used = try allocator.alloc(bool, cells);
    defer allocator.free(used);
    const live_in = try allocator.alloc(bool, cells);
    defer allocator.free(live_in);
    const live_out = try allocator.alloc(bool, cells);
    defer allocator.free(live_out);
    @memset(defined, false);
    @memset(used, false);
    @memset(live_in, false);
    @memset(live_out, false);

    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);
    for (0..block_count) |bi| {
        const block_id: Block = @fromBackingInt(@intCast(bi));
        const base = bi * value_count;
        for (func.blockParams(block_id)) |parameter| defined[base + @backingInt(parameter)] = true;
        for (func.blockInsts(block_id)) |inst| {
            try schedule.collectOperands(allocator, func, inst, &operands);
            for (operands.items) |operand| {
                const at = base + @backingInt(operand);
                if (!defined[at]) used[at] = true;
            }
            if (func.instResult(inst)) |result| defined[base + @backingInt(result)] = true;
        }
        if (func.terminator(block_id)) |term| switch (term) {
            .ret => |ret| for (ret.slice()) |operand| {
                const at = base + @backingInt(operand);
                if (!defined[at]) used[at] = true;
            },
            .jump => |jump| for (func.blockArgs(jump)) |operand| {
                const at = base + @backingInt(operand);
                if (!defined[at]) used[at] = true;
            },
        };
    }

    var changed = true;
    while (changed) {
        changed = false;
        var bi = block_count;
        while (bi > 0) {
            bi -= 1;
            const base = bi * value_count;
            for (0..value_count) |vi| {
                var new_out = false;
                for (graph.successors(bi)) |successor| {
                    if (live_in[@as(usize, successor) * value_count + vi]) {
                        new_out = true;
                        break;
                    }
                }
                const new_in = used[base + vi] or (new_out and !defined[base + vi]);
                if (live_out[base + vi] != new_out or live_in[base + vi] != new_in) changed = true;
                live_out[base + vi] = new_out;
                live_in[base + vi] = new_in;
            }
        }
    }

    const live = try allocator.alloc(bool, value_count);
    defer allocator.free(live);
    var peak: u32 = 0;
    for (0..block_count) |bi| {
        const block_id: Block = @fromBackingInt(@intCast(bi));
        const base = bi * value_count;
        @memcpy(live, live_out[base..][0..value_count]);
        var count: u32 = 0;
        for (live) |is_live| count += @intFromBool(is_live);
        peak = @max(peak, count);

        if (func.terminator(block_id)) |term| switch (term) {
            .ret => |ret| for (ret.slice()) |operand| markLive(live, operand, &count),
            .jump => |jump| for (func.blockArgs(jump)) |operand| markLive(live, operand, &count),
        };
        peak = @max(peak, count);

        const insts = func.blockInsts(block_id);
        var index = insts.len;
        while (index > 0) {
            index -= 1;
            const inst = insts[index];
            if (func.instResult(inst)) |result| {
                const at = @backingInt(result);
                if (live[at]) {
                    live[at] = false;
                    count -= 1;
                }
            }
            try schedule.collectOperands(allocator, func, inst, &operands);
            for (operands.items) |operand| markLive(live, operand, &count);
            peak = @max(peak, count);
        }
    }
    return peak;
}

fn markLive(live: []bool, value: Value, count: *u32) void {
    const at = @backingInt(value);
    if (live[at]) return;
    live[at] = true;
    count.* += 1;
}

fn executionWeight(graph: *const cfg_mod.Cfg, doms: *const dominators.Dominators, loop_info: *const loops.LoopInfo, block_id: usize) u64 {
    var weight: u64 = 1;
    for (loop_info.loops) |loop| {
        if (!loop.contains(block_id)) continue;
        var found_latch = false;
        var dominates_every_latch = true;
        for (0..graph.blockCount()) |candidate| {
            if (!loop.contains(candidate)) continue;
            for (graph.successors(candidate)) |successor| {
                if (successor != loop.header) continue;
                found_latch = true;
                if (!doms.dominates(@intCast(block_id), candidate)) dominates_every_latch = false;
            }
        }
        if (found_latch and dominates_every_latch)
            weight = std.math.mul(u64, weight, 8) catch std.math.maxInt(u64);
    }
    return weight;
}

pub fn block(allocator: std.mem.Allocator, func: *const Function, block_id: Block, model: *const mm.Model) std.mem.Allocator.Error!Score {
    const insts = func.blockInsts(block_id);
    if (insts.len == 0) return .{};

    const class_count = @typeInfo(UnitClass).@"enum".field_names.len;
    var demand: [class_count]u64 = @splat(0);
    var total_demand: u64 = 0;
    var result_instruction_count: u64 = 0;
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
    const live_in = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(live_in);
    @memset(live_in, false);

    var operands: std.ArrayList(Value) = .empty;
    defer operands.deinit(allocator);

    for (insts, 0..) |inst, index| {
        try schedule.collectOperands(allocator, func, inst, &operands);
        var starts_at: u64 = 0;
        for (operands.items) |operand| {
            starts_at = @max(starts_at, ready[@backingInt(operand)]);
            if (def_at[@backingInt(operand)] != std.math.maxInt(usize)) {
                last_use[@backingInt(operand)] = index;
            } else {
                live_in[@backingInt(operand)] = true;
                last_use[@backingInt(operand)] = index;
            }
        }

        const opcode = func.opcode(inst);
        const micro_ops = estimatedInstructions(func, inst, model);
        const latency: u64 = model.latency(opcode) + micro_ops - 1;
        const finishes_at = starts_at + latency;
        critical_path = @max(critical_path, finishes_at);
        if (func.instResult(inst)) |result| {
            ready[@backingInt(result)] = finishes_at;
            def_at[@backingInt(result)] = index;
            last_use[@backingInt(result)] = index;
        }

        const weight: u64 = model.throughput(opcode, elementIsFloat(func, inst)) + micro_ops - 1;
        demand[@backingInt(model.unitOf(opcode))] += weight;
        total_demand += weight;
        result_instruction_count += micro_ops;
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
    var live: u32 = 0;
    for (live_in, 0..) |is_live, value_index| {
        if (!is_live) continue;
        live += 1;
        ends[last_use[value_index]] += 1;
    }
    for (insts, 0..) |inst, index| {
        if (func.instResult(inst)) |result| {
            const end = last_use[@backingInt(result)];
            if (end > index) ends[end] += 1;
        }
    }

    var peak_live: u32 = live;
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
        .instructions = result_instruction_count,
        .peak_live = peak_live,
        .idle_slots = cycles * width -| total_demand,
    };
}

fn estimatedInstructions(func: *const Function, inst: ir.function.Inst, model: *const mm.Model) u64 {
    if (model.arch != .aarch64) return 1;
    const opcode = func.opcode(inst);
    return switch (opcode) {
        .iconst => |value| constantInstructions(func, inst, value),
        .arith_imm => |arith| if ((arith.op == .add or arith.op == .sub) and
            (arith.imm < -4095 or arith.imm > 4095))
            1 + constantInstructions(func, inst, arith.imm)
        else
            1,
        else => 1,
    };
}

fn constantInstructions(func: *const Function, inst: ir.function.Inst, value: i64) u64 {
    const result = func.instResult(inst) orelse return 1;
    const bits = switch (func.types.type_kind(func.valueType(result))) {
        .int => |int| int.bits,
        else => return 1,
    };
    const raw: u64 = @bitCast(value);
    if (bits <= 32) return 1 + @as(u64, @intFromBool(@as(u16, @truncate(raw >> 16)) != 0));
    var count: u64 = 1;
    inline for (1..4) |part| count += @as(u64, @intFromBool(@as(u16, @truncate(raw >> (part * 16))) != 0));
    return count;
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

test "score uses size then pressure to break cycle ties" {
    const baseline = Score{ .cycles = 8, .instructions = 6, .peak_live = 4 };
    try std.testing.expect((Score{ .cycles = 8, .instructions = 5, .peak_live = 4 }).accepts(baseline));
    try std.testing.expect(!(Score{ .cycles = 8, .instructions = 8, .peak_live = 3 }).accepts(baseline));
    try std.testing.expect((Score{ .cycles = 8, .instructions = 6, .peak_live = 3 }).accepts(baseline));
}

test "score counts values live through an otherwise empty block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const middle = try func.appendBlock();
    const done = try func.appendBlock();
    const lhs = try func.appendInst(entry, i32_t, .{ .iconst = 11 });
    const rhs = try func.appendInst(entry, i32_t, .{ .iconst = 17 });
    try func.setJump(entry, middle, &.{});
    try func.setJump(middle, done, &.{});
    const sum = try func.appendInst(done, i32_t, .{ .arith = .{ .op = .add, .lhs = lhs, .rhs = rhs } });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(sum) });

    const result = try function(allocator, &func, registry.modelFor(.@"ampere-altra"));
    try std.testing.expect(result.peak_live >= 2);
}

test "IR gate commits a faster candidate" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const value = try func.appendBlockParam(entry, i32_t);
    const product = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = value, .rhs = value } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(product) });

    var gate = try IRGate.init(allocator, &func, registry.modelFor(.@"ampere-altra"));
    defer gate.deinit();
    gate.candidateFunction().opcodeMut(gate.candidateFunction().definingInst(product).?).arith.op = .add;

    try std.testing.expect(try gate.commit(&func));
    try std.testing.expectEqual(ir.function.BinOp.add, func.opcode(func.definingInst(product).?).arith.op);
}

test "IR gate keeps the baseline when a candidate is slower" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const value = try func.appendBlockParam(entry, i32_t);
    const sum = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .add, .lhs = value, .rhs = value } });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.one(sum) });

    var gate = try IRGate.init(allocator, &func, registry.modelFor(.@"ampere-altra"));
    defer gate.deinit();
    gate.candidateFunction().opcodeMut(gate.candidateFunction().definingInst(sum).?).arith.op = .mul;

    try std.testing.expect(!try gate.commit(&func));
    try std.testing.expectEqual(ir.function.BinOp.add, func.opcode(func.definingInst(sum).?).arith.op);
}

test "IR gate accepts hoisting an expanded immediate from a loop" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const pass = @import("../pass.zig");
    const loopimm = @import("loopimm.zig");

    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_ty = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const done = try func.appendBlock();
    const seed = try func.appendInst(entry, ty, .{ .iconst = 1 });
    try func.setJump(entry, header, &.{seed});
    const state = try func.appendBlockParam(header, ty);
    const next = try func.appendInst(header, ty, .{ .arith_imm = .{ .op = .add, .lhs = state, .imm = 1013904223 } });
    const keep_going = try func.appendInst(header, bool_ty, .{ .icmp = .{ .op = .ne, .lhs = next, .rhs = state } });
    try func.appendIf(header, keep_going, .{ .target = header, .args = &.{next} }, .{ .target = done });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(next) });

    var gate = try IRGate.init(allocator, &func, registry.modelFor(.@"ampere-altra"));
    defer gate.deinit();
    try std.testing.expect(try pass.run(allocator, gate.candidateFunction(), &.{loopimm.pass_def}));
    try std.testing.expect(try gate.commit(&func));
    try std.testing.expect(func.opcode(func.definingInst(next).?) == .arith);
}
