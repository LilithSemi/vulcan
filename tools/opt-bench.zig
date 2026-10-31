//! Repeatable wall-clock benchmark for target-independent optimizer compile time.
//! It builds a function with many independent e-graph candidates, clones it outside
//! the timed region, and measures only the requested optimization work. Timing uses
//! the injected `std.Io`, matching the rest of Vulcan's hosted tooling.

const std = @import("std");
const ir = @import("vulcan-ir");
const opt = @import("vulcan-opt");

const Function = ir.function.Function;
const BenchFn = *const fn (std.mem.Allocator, *Function) anyerror!bool;

const blocks: usize = 2048;
const samples: usize = 9;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var factor = try buildFactorWorkload(allocator, blocks);
    defer factor.deinit();
    var nested = try buildNestedWorkload(allocator, blocks);
    defer nested.deinit();
    var rootless = try buildRootlessWorkload(allocator, blocks);
    defer rootless.deinit();
    var unrelated = try buildUnrelatedWorkload(allocator, blocks);
    defer unrelated.deinit();

    var buffer: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(init.io, &buffer);
    const writer = &stdout.interface;

    const factor_egraph_ns = try measure(init.io, allocator, &factor, runEgraph);
    const factor_pipeline_ns = try measure(init.io, allocator, &factor, runPipeline);
    const nested_egraph_ns = try measure(init.io, allocator, &nested, runEgraph);
    const nested_pipeline_ns = try measure(init.io, allocator, &nested, runPipeline);
    const rootless_egraph_ns = try measure(init.io, allocator, &rootless, runEgraph);
    const unrelated_egraph_ns = try measure(init.io, allocator, &unrelated, runEgraph);
    try writer.print(
        "optimizer benchmark ({d} candidate blocks, best of {d})\n" ++
            "  factor egraph:          {d} ns/block\n" ++
            "  factor early pipeline:  {d} ns/block\n" ++
            "  nested egraph:          {d} ns/block\n" ++
            "  nested early pipeline:  {d} ns/block\n" ++
            "  rootless egraph:        {d} ns/block\n" ++
            "  unrelated egraph:       {d} ns/block\n",
        .{
            blocks,
            samples,
            factor_egraph_ns / blocks,
            factor_pipeline_ns / blocks,
            nested_egraph_ns / blocks,
            nested_pipeline_ns / blocks,
            rootless_egraph_ns / blocks,
            unrelated_egraph_ns / blocks,
        },
    );
    try writer.flush();
}

fn measure(io: std.Io, allocator: std.mem.Allocator, template: *const Function, bench: BenchFn) !u64 {
    var best: u64 = std.math.maxInt(u64);
    for (0..samples) |_| {
        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();
        const scratch = arena.allocator();
        var func = try template.clone(scratch);
        defer func.deinit();

        const start = std.Io.Timestamp.now(io, .awake);
        std.mem.doNotOptimizeAway(try bench(scratch, &func));
        const finish = std.Io.Timestamp.now(io, .awake);
        const elapsed: u64 = @intCast(start.durationTo(finish).nanoseconds);
        best = @min(best, elapsed);
    }
    return best;
}

fn runEgraph(allocator: std.mem.Allocator, func: *Function) !bool {
    return opt.egraph.runWithLimits(allocator, func, .{});
}

fn runPipeline(allocator: std.mem.Allocator, func: *Function) !bool {
    return opt.optimizeEarly(allocator, func);
}

fn buildFactorWorkload(allocator: std.mem.Allocator, count: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    for (0..count) |_| {
        const block = try func.appendBlock();
        const a = try func.appendBlockParam(block, ty);
        const b = try func.appendBlockParam(block, ty);
        const c = try func.appendBlockParam(block, ty);
        const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
        const sum = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
        func.setTerminator(block, .{ .ret = ir.function.Ret.one(sum) });
    }
    return func;
}

fn buildNestedWorkload(allocator: std.mem.Allocator, count: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    for (0..count) |_| {
        const block = try func.appendBlock();
        const a = try func.appendBlockParam(block, ty);
        const b = try func.appendBlockParam(block, ty);
        const c = try func.appendBlockParam(block, ty);
        const d = try func.appendBlockParam(block, ty);
        const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
        const ad = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = d } });
        const partial = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
        const total = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = partial, .rhs = ad } });
        func.setTerminator(block, .{ .ret = ir.function.Ret.one(total) });
    }
    return func;
}

fn buildRootlessWorkload(allocator: std.mem.Allocator, count: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    for (0..count) |_| {
        const block = try func.appendBlock();
        const a = try func.appendBlockParam(block, ty);
        const b = try func.appendBlockParam(block, ty);
        const c = try func.appendBlockParam(block, ty);
        const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
        _ = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
        func.setTerminator(block, .{ .ret = ir.function.Ret.none() });
    }
    return func;
}

fn buildUnrelatedWorkload(allocator: std.mem.Allocator, count: usize) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    for (0..count) |_| {
        const block = try func.appendBlock();
        const a = try func.appendBlockParam(block, ty);
        const b = try func.appendBlockParam(block, ty);
        const c = try func.appendBlockParam(block, ty);
        const ab = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const ac = try func.appendInst(block, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = c } });
        _ = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = ab, .rhs = ac } });
        func.setTerminator(block, .{ .ret = ir.function.Ret.one(a) });
    }
    return func;
}
