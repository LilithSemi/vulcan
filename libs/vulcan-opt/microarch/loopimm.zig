//! Hoist AArch64 immediates that need a register out of loop backbones. The generic IR keeps
//! constants inside `arith_imm`, but AArch64 can encode only a small add/sub immediate. A larger
//! value becomes one or more MOV instructions on every iteration unless it is made explicit in
//! the loop entry.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");
const dominators = @import("../dominators.zig");
const loops = @import("../loops.zig");
const cfg_mod = @import("../cfg.zig");

const Function = ir.function.Function;
const Inst = ir.function.Inst;
const Value = ir.function.Value;
const Type = ir.types.Type;

pub const pass_def = pass.Pass{ .name = "aarch64-loop-immediates", .run = run, .preserves_cfg = true };

const Candidate = struct {
    inst: Inst,
    preheader: u32,
    lhs: Value,
    ty: Type,
    op: ir.function.BinOp,
    imm: i64,
};

const Materialized = struct {
    preheader: u32,
    ty: Type,
    imm: i64,
    value: Value,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const loop_info = try analyses.loops();
    if (loop_info.loops.len == 0) return false;
    const graph = try analyses.cfg();
    const doms = try analyses.dominators();

    var candidates: std.ArrayList(Candidate) = .empty;
    defer candidates.deinit(allocator);
    const selected = try allocator.alloc(?usize, func.instCount());
    defer allocator.free(selected);
    @memset(selected, null);

    for (loop_info.loops) |loop| {
        const preheader = loop.preheader orelse loop.entry orelse continue;
        for (0..func.blockCount()) |bi| {
            if (!loop.contains(bi) or !executesEveryIteration(&loop, graph, doms, bi)) continue;
            for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
                const arith = switch (func.opcode(inst)) {
                    .arith_imm => |arith| arith,
                    else => continue,
                };
                if ((arith.op != .add and arith.op != .sub) or
                    encodableAddSub(arith.imm)) continue;
                const result = func.instResult(inst).?;
                switch (func.types.type_kind(func.valueType(result))) {
                    .int => {},
                    else => continue,
                }

                if (selected[@backingInt(inst)]) |candidate_index| {
                    const old_preheader = candidates.items[candidate_index].preheader;
                    if (doms.dominates(preheader, old_preheader))
                        candidates.items[candidate_index].preheader = preheader;
                    continue;
                }
                selected[@backingInt(inst)] = candidates.items.len;
                try candidates.append(allocator, .{
                    .inst = inst,
                    .preheader = preheader,
                    .lhs = arith.lhs,
                    .ty = func.valueType(result),
                    .op = arith.op,
                    .imm = arith.imm,
                });
            }
        }
    }
    if (candidates.items.len == 0) return false;

    var materialized: std.ArrayList(Materialized) = .empty;
    defer materialized.deinit(allocator);
    for (candidates.items) |candidate| {
        const constant = findMaterialized(materialized.items, candidate.preheader, candidate.ty, candidate.imm) orelse create: {
            const value = try func.appendInst(
                @fromBackingInt(@intCast(candidate.preheader)),
                candidate.ty,
                .{ .iconst = candidate.imm },
            );
            beforeBranch(func, @fromBackingInt(@intCast(candidate.preheader)));
            try materialized.append(allocator, .{
                .preheader = candidate.preheader,
                .ty = candidate.ty,
                .imm = candidate.imm,
                .value = value,
            });
            break :create value;
        };
        func.opcodeMut(candidate.inst).* = .{ .arith = .{
            .op = candidate.op,
            .lhs = candidate.lhs,
            .rhs = constant,
        } };
    }
    return true;
}

fn encodableAddSub(imm: i64) bool {
    const magnitude = @abs(imm);
    return magnitude <= 4095 or (magnitude & 4095 == 0 and magnitude >> 12 <= 4095);
}

fn executesEveryIteration(loop: *const loops.Loop, graph: *const cfg_mod.Cfg, doms: *const dominators.Dominators, block: usize) bool {
    var found_latch = false;
    for (0..graph.blockCount()) |candidate| {
        if (!loop.contains(candidate)) continue;
        for (graph.successors(candidate)) |successor| {
            if (successor != loop.header) continue;
            found_latch = true;
            if (!doms.dominates(@intCast(block), candidate)) return false;
        }
    }
    return found_latch;
}

/// Move the instruction just appended to `block` in front of the `if` that ends it, since control
/// leaves the block at that `if`.
fn beforeBranch(func: *Function, block: ir.function.Block) void {
    const insts = func.blockInstsMut(block).items;
    if (insts.len < 2 or func.opcode(insts[insts.len - 2]) != .@"if") return;
    std.mem.swap(Inst, &insts[insts.len - 1], &insts[insts.len - 2]);
}

fn findMaterialized(values: []const Materialized, preheader: u32, ty: Type, imm: i64) ?Value {
    for (values) |value| {
        if (value.preheader == preheader and value.ty == ty and value.imm == imm) return value.value;
    }
    return null;
}

test "hoists a wide add immediate from a loop backbone" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
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

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    const rewritten = func.opcode(func.definingInst(next).?).arith;
    try std.testing.expectEqual(state, rewritten.lhs);
    try std.testing.expectEqual(@as(i64, 1013904223), func.opcode(func.definingInst(rewritten.rhs).?).iconst);
    try std.testing.expect(std.mem.indexOfScalar(Inst, func.blockInsts(entry), func.definingInst(rewritten.rhs).?) != null);
}

test "a hoisted immediate goes before the branch that ends the preheader" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_ty = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const done = try func.appendBlock();
    const seed = try func.appendInst(entry, ty, .{ .iconst = 1 });
    const enter = try func.appendBlockParam(entry, bool_ty);
    try func.appendIf(entry, enter, .{ .target = header, .args = &.{seed} }, .{ .target = done, .args = &.{seed} });
    func.setTerminator(entry, .{ .ret = ir.function.Ret.none() });
    const state = try func.appendBlockParam(header, ty);
    const next = try func.appendInst(header, ty, .{ .arith_imm = .{ .op = .add, .lhs = state, .imm = 12345 } });
    const keep_going = try func.appendInst(header, bool_ty, .{ .icmp = .{ .op = .ne, .lhs = next, .rhs = state } });
    try func.appendIf(header, keep_going, .{ .target = header, .args = &.{next} }, .{ .target = done, .args = &.{next} });
    func.setTerminator(header, .{ .ret = ir.function.Ret.none() });
    const result = try func.appendBlockParam(done, ty);
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(result) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    const insts = func.blockInsts(entry);
    try std.testing.expect(func.opcode(insts[insts.len - 1]) == .@"if");
}

test "keeps a small loop immediate encoded in the operation" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const seed = try func.appendInst(entry, ty, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{seed});
    const index = try func.appendBlockParam(header, ty);
    const next = try func.appendInst(header, ty, .{ .arith_imm = .{ .op = .add, .lhs = index, .imm = 1 } });
    try func.setJump(header, header, &.{next});

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(!try run(allocator, &func, &analyses));
    try std.testing.expect(func.opcode(func.definingInst(next).?) == .arith_imm);
}
