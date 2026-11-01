//! Structural loop vectorizer. Unrolls a counted loop's main body by the SIMD width V into one block,
//! regenerating each array access as `base_i + m*stride` (the `arith_imm add base, CONST` form the SLP
//! pass recognizes), and keeps the original loop as a scalar remainder. The subsequent SLP pass
//! (`vectorize.runModel`) then fuses the V-wide body into vector loads/ops/stores. So this pass owns
//! only the structure and the strided-address analysis. SLP owns the arithmetic widening.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const cost = @import("cost.zig");
const loops = @import("../loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Opcode = ir.function.Opcode;
const BinOp = ir.function.BinOp;
const CmpOp = ir.function.CmpOp;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error;

/// One contiguous (unit-stride) array access in the body: `addr = base + i*stride`, stride == the
/// element's byte size. `addr_inst` is the `base + off` instruction (regenerated per copy);
/// `scale_inst` is the `i*stride` computation (skipped when cloning, since addresses are regenerated).
const ArrayAccess = struct { base: Value, stride: i64, offset: i64, addr_inst: Inst, scale_inst: Inst, is_store: bool };
const AccessBase = struct { base: Value, stride: i64 };

const MapReduction = struct {
    accumulator: usize,
    update_inst: Inst,
    increment: Value,
    op: BinOp,
};

const Plan = struct {
    header: Block,
    body: Block,
    latch: Block,
    preheader: Block,
    exit_cond: CmpOp,
    exit_offset: i64,
    bound: Value,
    induction: usize,
    induction_alias: Value,
    induction_update: Inst,
    step: i64,
    accesses: []ArrayAccess, // owned
    reduction: ?MapReduction,
    factor: u32, // V
    groups: u32,
    /// Scalar element type of a vector induction carried through the widened loop.
    index_elem_type: ?Type,
};

fn carriedArgs(storage: []Value, index: Value, vector_index: ?Value, accumulators: []const ?Value) []const Value {
    var len: usize = 0;
    storage[len] = index;
    len += 1;
    if (vector_index) |value| {
        storage[len] = value;
        len += 1;
    }
    for (accumulators) |accumulator| if (accumulator) |value| {
        storage[len] = value;
        len += 1;
    };
    return storage[0..len];
}

pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);
    if (info.loops.len == 0) return false;
    const fast_math = functionHasFastMath(func);

    var plans: std.ArrayList(Plan) = .empty;
    defer {
        for (plans.items) |*p| allocator.free(p.accesses);
        plans.deinit(allocator);
    }
    var reductions: std.ArrayList(RedPlan) = .empty;
    defer reductions.deinit(allocator);
    for (info.loops) |*loop| {
        if (try recognize(allocator, func, model, loop)) |plan| {
            try plans.append(allocator, plan);
        } else if (recognizeReduction(func, model, loop, fast_math)) |rplan| {
            try reductions.append(allocator, rplan);
        }
    }
    for (plans.items) |*plan| try apply(allocator, func, plan);
    for (reductions.items) |*plan| try applyReduction(allocator, func, plan);
    return plans.items.len != 0 or reductions.items.len != 0;
}

fn functionHasFastMath(func: *const Function) bool {
    var it = func.attributesOf(.func);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| if (std.mem.eql(u8, c.namespace, "vulcan") and std.mem.eql(u8, c.key, "fast_math")) return true,
        else => {},
    };
    return false;
}

fn apply(allocator: std.mem.Allocator, func: *Function, plan: *const Plan) Error!void {
    const V = plan.factor;
    const U = V * plan.groups;
    const ind_ty = func.valueType(func.blockParams(plan.header)[plan.induction]);
    const reduction_ty = if (plan.reduction) |red| func.valueType(func.blockParams(plan.header)[red.accumulator]) else null;
    const reduction_vec_ty = if (reduction_ty) |elem|
        try func.types.intern(.{ .vector = .{ .len = @intCast(V), .elem = elem } })
    else
        null;
    const bool_t = try func.types.intern(.bool);
    const ind_int = switch (func.types.type_kind(ind_ty)) {
        .int => |int| int,
        else => unreachable,
    };
    const index_elem_ty = plan.index_elem_type;
    const index_vec_ty = if (index_elem_ty) |elem|
        try func.types.intern(.{ .vector = .{ .len = @intCast(V), .elem = elem } })
    else
        null;
    const group_offset: ?i64 = std.math.add(i64, @as(i64, @intCast(U - 1)), plan.exit_offset) catch null;
    const offset_fits = group_offset != null and group_offset.? >= 0 and ind_int.bits < 64 and
        @as(u64, @intCast(group_offset.?)) < (@as(u64, 1) << @intCast(ind_int.bits));
    const rotated = ind_int.signedness == .unsigned and offset_fits;

    const mainheader = try func.appendBlock();
    const mainbody = if (rotated) mainheader else try func.appendBlock();
    const combine = try func.appendBlock();
    const mi = try func.appendBlockParam(mainheader, ind_ty);
    const mvi = if (index_vec_ty) |vec_ty| try func.appendBlockParam(mainheader, vec_ty) else null;
    var maccs: [2]?Value = @splat(null);
    if (reduction_vec_ty) |vec_ty| {
        for (0..plan.groups) |group| maccs[group] = try func.appendBlockParam(mainheader, vec_ty);
    }

    // Preheader -> main header, carrying the induction's initial value.
    const pre_args = try allocator.dupe(Value, func.blockArgs(func.terminator(plan.preheader).?.jump));
    defer allocator.free(pre_args);
    const init_i = pre_args[plan.induction];
    const init_vi = if (index_vec_ty) |vec_ty| init: {
        const scalar = try func.appendInst(plan.preheader, index_elem_ty.?, .{ .convert = .{ .value = init_i } });
        var lanes: [MAX_LANES]Value = undefined;
        for (lanes[0..V], 0..) |*lane, index| lane.* = if (index == 0)
            scalar
        else
            try func.appendArithImm(plan.preheader, index_elem_ty.?, .add, scalar, @intCast(index));
        break :init try func.appendInst(plan.preheader, vec_ty, .{ .struct_new = .{ .fields = try func.internValues(lanes[0..V]) } });
    } else null;
    const index_step = if (index_vec_ty) |vec_ty| step: {
        const scalar = try func.appendInst(plan.preheader, index_elem_ty.?, .{ .iconst = @intCast(U) });
        break :step try func.appendSplat(plan.preheader, vec_ty, scalar);
    } else null;
    const init_acc = if (plan.reduction) |red| pre_args[red.accumulator] else null;
    const vector_identity = if (plan.reduction) |red| identity: {
        const scalar = try identityConst(func, plan.preheader, red.op, reduction_ty.?);
        break :identity try func.appendSplat(plan.preheader, reduction_vec_ty.?, scalar);
    } else null;
    var initial_accs: [2]?Value = @splat(null);
    for (0..plan.groups) |group| initial_accs[group] = vector_identity;
    const threshold = if (ind_int.signedness == .unsigned and offset_fits)
        try func.appendArithImm(plan.preheader, ind_ty, .sub, plan.bound, group_offset.?)
    else
        null;
    if (threshold != null) {
        const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
        const wide_bound = try func.appendInst(plan.preheader, u64_t, .{ .convert = .{ .value = plan.bound } });
        const wide_highest = if (constOf(func, init_i)) |constant| constant: {
            const mask = (@as(u64, 1) << @intCast(ind_int.bits)) - 1;
            const init_bits = @as(u64, @bitCast(constant)) & mask;
            const highest = init_bits + @as(u64, @intCast(group_offset.?));
            break :constant try func.appendInst(plan.preheader, u64_t, .{ .iconst = @bitCast(highest) });
        } else highest: {
            const wide_i = try func.appendInst(plan.preheader, u64_t, .{ .convert = .{ .value = init_i } });
            break :highest try func.appendArithImm(plan.preheader, u64_t, .add, wide_i, group_offset.?);
        };
        const reverse: CmpOp = if (plan.exit_cond == .lt) .gt else .ge;
        const can_enter = try func.appendInst(plan.preheader, bool_t, .{ .icmp = .{ .op = reverse, .lhs = wide_bound, .rhs = wide_highest } });
        var enter_storage: [4]Value = undefined;
        const enter_args = carriedArgs(&enter_storage, init_i, init_vi, &initial_accs);
        try func.appendIf(plan.preheader, can_enter, .{ .target = mainheader, .args = enter_args }, .{ .target = plan.header, .args = pre_args });
    } else {
        var enter_storage: [4]Value = undefined;
        try func.setJump(plan.preheader, mainheader, carriedArgs(&enter_storage, init_i, init_vi, &initial_accs));
    }

    // Main header: run a full group only while the highest index in it still passes the test.
    const cont = if (rotated)
        null
    else if (ind_int.signedness == .unsigned and ind_int.bits < 64 and group_offset != null) guard: {
        // Compare in u64 so one bound check also proves both additions cannot wrap in the
        // induction type. The old form needed a compare for the bound and one for each add.
        const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
        const wide_i = try func.appendInst(mainheader, u64_t, .{ .convert = .{ .value = mi } });
        // The bound is loop invariant. Widen it in the preheader instead of on every group.
        const wide_bound = try func.appendInst(plan.preheader, u64_t, .{ .convert = .{ .value = plan.bound } });
        const wide_highest = try func.appendArithImm(mainheader, u64_t, .add, wide_i, group_offset.?);
        break :guard try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = wide_highest, .rhs = wide_bound } });
    } else guard: {
        const highest_lane = try func.appendArithImm(mainheader, ind_ty, .add, mi, @as(i64, @intCast(U - 1)));
        const highest = if (plan.exit_offset == 0)
            highest_lane
        else
            try func.appendArithImm(mainheader, ind_ty, .add, highest_lane, plan.exit_offset);
        var guarded = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = highest, .rhs = plan.bound } });
        if (ind_int.signedness == .unsigned) {
            const lane_nowrap = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = .ge, .lhs = highest_lane, .rhs = mi } });
            guarded = try func.appendInst(mainheader, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = guarded, .rhs = lane_nowrap } });
            if (plan.exit_offset > 0) {
                const offset_nowrap = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = .ge, .lhs = highest, .rhs = highest_lane } });
                guarded = try func.appendInst(mainheader, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = guarded, .rhs = offset_nowrap } });
            }
        }
        break :guard guarded;
    };
    if (cont) |condition| {
        var body_storage: [4]Value = undefined;
        var combine_storage: [3]Value = undefined;
        try func.appendIf(mainheader, condition, .{ .target = mainbody, .args = carriedArgs(&body_storage, mi, mvi, &maccs) }, .{ .target = combine, .args = carriedArgs(&combine_storage, mi, null, &maccs) });
    }

    // Main body: U copies emitted operation-major in V-wide groups, which gives the SLP pass one or more
    // contiguous vector groups. Per unique address,
    // base_i = base + bmi*stride computed once; copy m reads/writes base_i + m*stride. Each copy's
    // induction maps to bmi+m, so pure body computation may observe the raw index too. The original
    // address/scale/induction-update instructions are skipped when cloning.
    const bmi = if (rotated) mi else try func.appendBlockParam(mainbody, ind_ty);
    const bvi = if (rotated)
        mvi
    else if (index_vec_ty) |vec_ty|
        try func.appendBlockParam(mainbody, vec_ty)
    else
        null;
    var baccs: [2]?Value = @splat(null);
    if (rotated) {
        baccs = maccs;
    } else if (reduction_vec_ty) |vec_ty| {
        for (0..plan.groups) |group| baccs[group] = try func.appendBlockParam(mainbody, vec_ty);
    }
    const ind_update = plan.induction_update;
    const base_ty = func.valueType(plan.accesses[0].base);

    // Per-copy value maps, pre-seeded with the induction and every regenerated address. Deduplicate by
    // the address instruction (a load and a store to the same slot share it).
    const vmaps = try allocator.alloc(std.AutoHashMapUnmanaged(Value, Value), U);
    defer {
        for (vmaps) |*vm| vm.deinit(allocator);
        allocator.free(vmaps);
    }
    for (vmaps) |*vm| vm.* = .empty;
    const i_alias = plan.induction_alias;
    var skip: std.AutoHashMapUnmanaged(Inst, void) = .empty;
    defer skip.deinit(allocator);
    var access_bases: std.AutoHashMapUnmanaged(AccessBase, Value) = .empty;
    defer access_bases.deinit(allocator);
    try skip.put(allocator, ind_update, {});
    if (plan.reduction) |red| try skip.put(allocator, red.update_inst, {});
    var vector_indices: [MAX_LANES]?Value = @splat(null);
    if (bvi) |vector_index| {
        for (0..plan.groups) |group| {
            const group_index = if (group == 0) vector_index else shifted: {
                const offset = try func.appendInst(mainbody, index_elem_ty.?, .{ .iconst = @intCast(group * V) });
                const offsets = try func.appendSplat(mainbody, index_vec_ty.?, offset);
                break :shifted try func.appendInst(mainbody, index_vec_ty.?, .{ .arith = .{ .op = .add, .lhs = vector_index, .rhs = offsets } });
            };
            for (0..V) |lane| {
                vector_indices[group * V + lane] = try func.appendInst(mainbody, index_elem_ty.?, .{ .extract = .{ .aggregate = group_index, .index = @intCast(lane) } });
            }
        }
    }
    for (vmaps, 0..) |*vm, m| {
        const lane_i = vector_indices[m] orelse if (m == 0)
            bmi
        else
            try func.appendArithImm(mainbody, ind_ty, .add, bmi, @intCast(m));
        try vm.put(allocator, i_alias, lane_i);
    }
    for (plan.accesses) |acc| {
        try skip.put(allocator, acc.addr_inst, {});
        try skip.put(allocator, acc.scale_inst, {});
        const addr_res = func.instResult(acc.addr_inst).?;
        if (vmaps[0].contains(addr_res)) continue; // already handled via a shared address instruction
        const entry = try access_bases.getOrPut(allocator, .{ .base = acc.base, .stride = acc.stride });
        if (!entry.found_existing) {
            const scaled = try func.appendArithImm(mainbody, base_ty, .mul, bmi, acc.stride);
            entry.value_ptr.* = try func.appendInst(mainbody, base_ty, .{ .arith = .{ .op = .add, .lhs = acc.base, .rhs = scaled } });
        }
        const unadjusted = entry.value_ptr.*;
        const base_i = if (acc.offset == 0)
            unadjusted
        else
            try func.appendArithImm(mainbody, base_ty, .add, unadjusted, acc.offset);
        for (vmaps, 0..) |*vm, m| {
            const addr = try func.appendArithImm(mainbody, base_ty, .add, base_i, @as(i64, @intCast(m)) * acc.stride);
            try vm.put(allocator, addr_res, addr);
        }
    }

    // Clone the body op-major: for each instruction, emit its U copies (one per value map) adjacently.
    //
    // Each copy keeps the same type and operation as the original, so every attribute of
    // the original travels with it. This is a scalar unroll by U, not a widening: an `endian` on a
    // 4-byte load still describes a 4-byte load in every copy. (The reduction path in
    // `applyReduction` below is the widening one, and it deliberately carries nothing.)
    const body_insts = try allocator.dupe(Inst, func.blockInsts(plan.body));
    defer allocator.free(body_insts);

    // The `original -> clone` record for the attribute copy, one entry per emitted copy.
    var value_pairs: std.ArrayList(ir.function.Function.ValuePair) = .empty;
    defer value_pairs.deinit(allocator);
    var inst_pairs: std.ArrayList(ir.function.Function.InstPair) = .empty;
    defer inst_pairs.deinit(allocator);

    for (body_insts) |inst| {
        if (skip.contains(inst)) continue;
        if (func.instResult(inst)) |result| {
            if (vmaps[0].contains(result)) continue; // a pre-seeded address
            if (index_elem_ty != null and func.opcode(inst) == .convert and func.opcode(inst).convert.value == i_alias and func.valueType(result) == index_elem_ty.?) {
                for (vmaps, 0..) |*vm, lane| try vm.put(allocator, result, vector_indices[lane].?);
                continue;
            }
            for (vmaps) |*vm| {
                const op = try remapOp(func, func.opcode(inst), vm, allocator);
                const clone = try func.appendInst(mainbody, func.valueType(result), op);
                try vm.put(allocator, result, clone);
                try value_pairs.append(allocator, .{ .old = result, .new = clone });
                try inst_pairs.append(allocator, .{ .old = inst, .new = func.definingInst(clone).? });
            }
        } else {
            for (vmaps) |*vm| {
                const op = try remapOp(func, func.opcode(inst), vm, allocator);
                const clone = try func.appendStmtRaw(mainbody, op);
                try inst_pairs.append(allocator, .{ .old = inst, .new = clone });
            }
        }
    }

    try func.cloneAttrs(value_pairs.items, inst_pairs.items);
    var next_accs: [2]?Value = @splat(null);
    if (plan.reduction) |red| for (0..plan.groups) |group_index| {
        var increments: [MAX_LANES]Value = undefined;
        for (0..V) |lane| {
            const vm = &vmaps[group_index * V + lane];
            increments[lane] = vm.get(red.increment) orelse red.increment;
        }
        const increment_vector = try func.appendInst(mainbody, reduction_vec_ty.?, .{ .struct_new = .{ .fields = try func.internValues(increments[0..V]) } });
        next_accs[group_index] = try func.appendInst(mainbody, reduction_vec_ty.?, .{ .arith = .{ .op = red.op, .lhs = baccs[group_index].?, .rhs = increment_vector } });
    };
    const nmi = try func.appendArithImm(mainbody, ind_ty, .add, bmi, @as(i64, @intCast(U)));
    const next_vi = if (bvi) |vi| try func.appendInst(mainbody, index_vec_ty.?, .{ .arith = .{ .op = .add, .lhs = vi, .rhs = index_step.? } }) else null;
    if (rotated) {
        const next_cont = try func.appendInst(mainbody, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = nmi, .rhs = threshold.? } });
        var loop_storage: [4]Value = undefined;
        var exit_storage: [3]Value = undefined;
        try func.appendIf(mainbody, next_cont, .{ .target = mainbody, .args = carriedArgs(&loop_storage, nmi, next_vi, &next_accs) }, .{ .target = combine, .args = carriedArgs(&exit_storage, nmi, null, &next_accs) });
    } else {
        var loop_storage: [4]Value = undefined;
        try func.setJump(mainbody, mainheader, carriedArgs(&loop_storage, nmi, next_vi, &next_accs));
    }

    // Continue in the original loop at the current index and accumulator value.
    const cmi = try func.appendBlockParam(combine, ind_ty);
    if (plan.reduction) |red| {
        var scalar_acc = init_acc.?;
        for (0..plan.groups) |_| {
            const cacc = try func.appendBlockParam(combine, reduction_vec_ty.?);
            const horizontal = try func.appendReduce(combine, red.op, cacc);
            scalar_acc = try func.appendInst(combine, reduction_ty.?, .{ .arith = .{ .op = red.op, .lhs = scalar_acc, .rhs = horizontal } });
        }
        var rem_args: [2]Value = undefined;
        rem_args[plan.induction] = cmi;
        rem_args[red.accumulator] = scalar_acc;
        try func.setJump(combine, plan.header, &rem_args);
    } else {
        try func.setJump(combine, plan.header, &.{cmi});
    }
}

fn rv(vmap: *const std.AutoHashMapUnmanaged(Value, Value), v: Value) Value {
    return vmap.get(v) orelse v;
}

fn remapOp(func: *Function, op: Opcode, vmap: *const std.AutoHashMapUnmanaged(Value, Value), allocator: std.mem.Allocator) Error!Opcode {
    return switch (op) {
        .iconst, .fconst, .fconst128, .alloca, .global_addr => op,
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = rv(vmap, a.lhs), .rhs = rv(vmap, a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = rv(vmap, a.lhs), .imm = a.imm } },
        .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = rv(vmap, c.lhs), .rhs = rv(vmap, c.rhs) } },
        .select => |s| .{ .select = .{ .cond = rv(vmap, s.cond), .then = rv(vmap, s.then), .@"else" = rv(vmap, s.@"else") } },
        .convert => |c| .{ .convert = .{ .value = rv(vmap, c.value) } },
        .unary => |u| .{ .unary = .{ .op = u.op, .value = rv(vmap, u.value) } },
        .extract => |e| .{ .extract = .{ .aggregate = rv(vmap, e.aggregate), .index = e.index } },
        .load => |l| .{ .load = .{
            .ptr = rv(vmap, l.ptr),
            .@"volatile" = l.@"volatile",
            .mask = if (l.mask) |m| rv(vmap, m) else null,
        } },
        .store => |s| .{ .store = .{
            .value = rv(vmap, s.value),
            .ptr = rv(vmap, s.ptr),
            .@"volatile" = s.@"volatile",
            .mask = if (s.mask) |m| rv(vmap, m) else null,
        } },
        .prefetch => |p| .{ .prefetch = .{ .ptr = rv(vmap, p.ptr) } },
        .struct_new => |sn| blk: {
            var fields: std.ArrayList(Value) = .empty;
            defer fields.deinit(allocator);
            for (func.valueList(sn.fields)) |v| try fields.append(allocator, rv(vmap, v));
            break :blk .{ .struct_new = .{ .fields = try func.internValues(fields.items) } };
        },
        else => unreachable, // recognition excludes call/dot/matmul/if
    };
}

/// SLP fuses 32-bit elements, so loopvec targets the same width and gates on the same vector feature.
const ELEM_BITS: u16 = 32;
const MAX_LANES: u32 = 8;

fn recognize(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model, loop: *const loops.Loop) Error!?Plan {
    const vec_ok = switch (model.features) {
        .aarch64 => |f| f.neon,
        .riscv64 => |f| f.v or f.vpu,
        .x86_64 => false,
        // See vectorize.zig: a warp lane is already the vector lane on an SM, so there is no IR
        // vector op for the NVIDIA backend to lower.
        .nvidia => false,
    };
    if (!vec_ok or model.vector_bits < 2 * ELEM_BITS) return null;
    const V: u32 = @min(model.vector_bits / ELEM_BITS, MAX_LANES);
    if (V < 2) return null;

    const header: Block = @fromBackingInt(@intCast(loop.header));
    const preheader = loop.preheader orelse return null;

    // Straight-line body with either an integrated latch (the hand-built IR shape) or one separate
    // trivial latch block (the shape emitted by the C frontend after mem2reg).
    var latch: ?Block = null;
    var in_loop_blocks: usize = 0;
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi]) continue;
        in_loop_blocks += 1;
        const b: Block = @fromBackingInt(@intCast(bi));
        if (b == header) continue;
        switch (func.terminator(b) orelse return null) {
            .jump => |j| if (j.target == header) {
                if (latch != null) return null;
                latch = b;
            },
            .ret => return null,
        }
    }
    if (in_loop_blocks != 2 and in_loop_blocks != 3) return null;
    const latchb = latch orelse return null;

    // Header: pure test ending in an `if`, body on the then edge, passing the header params through.
    var if_inst: ?Inst = null;
    for (func.blockInsts(header)) |inst| switch (func.opcode(inst)) {
        .@"if" => {
            if (if_inst != null) return null;
            if_inst = inst;
        },
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .convert, .unary => {},
        else => return null,
    };
    const cf = func.opcode(if_inst orelse return null).@"if";
    const bodyb = cf.then.target;
    if (bodyb == header) return null;
    if (bodyb != latchb) {
        if (in_loop_blocks != 3) return null;
        const body_term = func.terminator(bodyb) orelse return null;
        if (body_term != .jump or body_term.jump.target != latchb) return null;
    } else if (in_loop_blocks != 2) return null;
    const hparams = func.blockParams(header);
    const body_args = func.blockArgs(cf.then);
    if (body_args.len != 0 and body_args.len != hparams.len) return null;
    if (body_args.len != 0) for (body_args, hparams) |arg, hp| if (arg != hp) return null;

    // A map may carry only its induction, or an integer reduction accumulator whose per-lane
    // increment is computed by the same body. The latter covers map+store+checksum loops without
    // dropping the store as the reduction-only path would.
    if (hparams.len != 1 and hparams.len != 2) return null;
    const cmp = switch (func.opcode(func.definingInst(cf.cond) orelse return null)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .lt and cmp.op != .le) return null;
    var induction: ?usize = null;
    var exit_offset: i64 = 0;
    for (hparams, 0..) |param, index| {
        if (cmp.lhs == param) {
            induction = index;
            break;
        }
        if (constAddend(func, cmp.lhs, param)) |offset| {
            induction = index;
            exit_offset = offset;
            break;
        }
    }
    const induction_index = induction orelse return null;
    if (exit_offset < 0) return null;
    if (definedInLoop(func, loop, cmp.rhs)) return null;
    const bound = cmp.rhs;

    const bparams = func.blockParams(bodyb);
    if (body_args.len == 0 and bparams.len != 0) return null;
    const i_alias = if (body_args.len == 0) hparams[induction_index] else bparams[induction_index];
    const back_args = func.blockArgs(func.terminator(latchb).?.jump);
    const step = constAddend(func, back_args[induction_index], i_alias) orelse return null;
    if (step != 1) return null; // unit induction step: consecutive iterations touch consecutive elements
    const ind_update = func.definingInst(back_args[induction_index]) orelse return null;
    if (latchb != bodyb) {
        const latch_insts = func.blockInsts(latchb);
        if (latch_insts.len == 0) {
            var in_header = false;
            for (func.blockInsts(header)) |inst| if (inst == ind_update) {
                in_header = true;
                break;
            };
            if (!in_header) return null;
        } else if (latch_insts.len != 1 or latch_insts[0] != ind_update) return null;
    }
    const reduction: ?MapReduction = if (hparams.len == 2) red: {
        const accumulator = 1 - induction_index;
        const acc_alias = if (body_args.len == 0) hparams[accumulator] else bparams[accumulator];
        const update_inst = func.definingInst(back_args[accumulator]) orelse return null;
        const update = switch (func.opcode(update_inst)) {
            .arith => |a| a,
            else => return null,
        };
        const increment = if (update.lhs == acc_alias)
            update.rhs
        else if (update.rhs == acc_alias)
            update.lhs
        else
            return null;
        if (countUses(func, bodyb, acc_alias) != 1) return null;
        const elem_kind = func.types.type_kind(func.valueType(acc_alias));
        if (elem_kind != .int or byteSize(func, func.valueType(acc_alias)) * 8 != ELEM_BITS) return null;
        if (!reorderable(func, back_args[accumulator], update.op, false)) return null;
        break :red .{
            .accumulator = accumulator,
            .update_inst = update_inst,
            .increment = increment,
            .op = update.op,
        };
    } else null;
    // Every load/store address must be unit-stride contiguous; other ops must be pure compute.
    var accesses: std.ArrayList(ArrayAccess) = .empty;
    errdefer accesses.deinit(allocator);
    for (func.blockInsts(bodyb)) |inst| {
        switch (func.opcode(inst)) {
            .load => |l| {
                // A `volatile` access is observable (see `Load.@"volatile"`). Widening the loop
                // turns V iterations of this scalar load into ONE vector load, which coalesces V
                // observable accesses into one and drops the flag. Refuse the loop.
                //
                // An `endian` tag is NOT refused here. This path REPLICATES: `apply` emits V
                // scalar copies of this load, each of the same width and each keeping the tag
                // (see `apply`'s clone comment). The widening happens in the SLP pass that runs
                // after, and that pass refuses a tagged load on its own.
                if (l.@"volatile") return bail(&accesses, allocator);
                const sa = stridedAddress(func, l.ptr, i_alias) orelse return bail(&accesses, allocator);
                const elem_bytes = byteSize(func, func.valueType(func.instResult(inst).?));
                const interleaved = model.arch == .aarch64 and elem_bytes == 4 and sa.stride == 16;
                if (sa.stride != elem_bytes and !interleaved) return bail(&accesses, allocator);
                try accesses.append(allocator, .{ .base = sa.base, .stride = sa.stride, .offset = sa.offset, .addr_inst = sa.addr_inst, .scale_inst = sa.scale_inst, .is_store = false });
            },
            .store => |s| {
                // Same reasoning as the `load` arm above: V observable stores must not become one.
                // A masked store cannot be widened either: one vector store writes every element,
                // and a masked lane writes nothing.
                if (s.@"volatile" or s.mask != null) return bail(&accesses, allocator);
                const sa = stridedAddress(func, s.ptr, i_alias) orelse return bail(&accesses, allocator);
                const elem_bytes = byteSize(func, func.valueType(s.value));
                const interleaved = model.arch == .aarch64 and elem_bytes == 4 and sa.stride == 16;
                if (sa.stride != elem_bytes and !interleaved) return bail(&accesses, allocator);
                try accesses.append(allocator, .{ .base = sa.base, .stride = sa.stride, .offset = sa.offset, .addr_inst = sa.addr_inst, .scale_inst = sa.scale_inst, .is_store = true });
            },
            .arith, .arith_imm, .iconst, .fconst, .fconst128, .icmp, .select, .convert, .unary => {},
            else => return bail(&accesses, allocator), // call/alloca/dot/matmul/etc: not handled
        }
    }
    if (accesses.items.len == 0) return bail(&accesses, allocator);
    var interleaved_base: ?Value = null;
    var interleaved_loads: u4 = 0;
    var interleaved_stores: u4 = 0;
    var has_interleaved = false;
    for (accesses.items) |access| {
        if (access.stride == 4) continue;
        has_interleaved = true;
        if (access.stride != 16 or access.offset < 0 or access.offset >= 16 or @rem(access.offset, 4) != 0)
            return bail(&accesses, allocator);
        if (interleaved_base) |base| {
            if (access.base != base) return bail(&accesses, allocator);
        } else interleaved_base = access.base;
        const bit: u4 = @as(u4, 1) << @intCast(@divExact(access.offset, 4));
        if (access.is_store) interleaved_stores |= bit else interleaved_loads |= bit;
    }
    // Native ld4/st4 vectorization is worthwhile only for a complete four-field record. Keeping
    // partial records out avoids the scalar lane-pack fallback that is slower than the original.
    if (has_interleaved and (interleaved_loads != 0b1111 or interleaved_stores != 0b1111 or reduction != null))
        return bail(&accesses, allocator);
    // Carry a same-width integer conversion as one vector induction value only when something
    // besides the regenerated address chain consumes it. Pointer lowering commonly inserts a
    // same-width signedness conversion before its scale; carrying that dead-after-rewrite value
    // adds one SIMD loop dependency for no result.
    var index_elem_type: ?Type = null;
    const induction_kind = func.types.type_kind(func.valueType(i_alias));
    const induction_bits = switch (induction_kind) {
        .int => |int| int.bits,
        else => unreachable,
    };
    for (func.blockInsts(bodyb)) |inst| {
        const conversion = switch (func.opcode(inst)) {
            .convert => |cv| if (cv.value == i_alias) cv else continue,
            else => continue,
        };
        _ = conversion;
        const result = func.instResult(inst).?;
        const result_int = switch (func.types.type_kind(func.valueType(result))) {
            .int => |int| int,
            else => continue,
        };
        if (result_int.bits == induction_bits and hasNonAddressUse(func, bodyb, result, accesses.items)) {
            index_elem_type = func.valueType(result);
            break;
        }
    }
    // A raw induction used by body arithmetic is itself a vector chain. Carry it through the main
    // loop so SLP can reuse the lanes instead of rebuilding each group with a scalar broadcast and
    // offsets. Address-generation uses are regenerated separately below, and the latch increment is
    // replaced by the wide step, so neither one alone justifies the extra vector loop parameter.
    if (model.arch == .aarch64 and index_elem_type == null) body_use: for (func.blockInsts(bodyb)) |inst| {
        if (inst == ind_update) continue;
        for (accesses.items) |access| {
            if (inst == access.addr_inst or inst == access.scale_inst) continue :body_use;
        }
        if (func.opcode(inst) == .convert and func.opcode(inst).convert.value == i_alias) {
            const result = func.instResult(inst).?;
            if (!hasNonAddressUse(func, bodyb, result, accesses.items)) continue;
        }
        if (usesValue(func, inst, i_alias)) {
            index_elem_type = func.valueType(i_alias);
            break;
        }
    };
    // Retain the signed fallback for hand-built interleaved IR without an explicit conversion.
    if (has_interleaved and index_elem_type == null)
        index_elem_type = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = induction_bits } });

    const parallel_reduction = reduction == null or
        cost.reductionGroupsProfitable(model, reduction.?.op, @intCast(accesses.items.len), 2);
    const groups: u32 = if (model.arch == .aarch64 and parallel_reduction and accesses.items.len <= 2 and V * 2 <= MAX_LANES) 2 else 1;
    return Plan{
        .header = header,
        .body = bodyb,
        .latch = latchb,
        .preheader = @fromBackingInt(@intCast(preheader)),
        .exit_cond = cmp.op,
        .exit_offset = exit_offset,
        .bound = bound,
        .induction = induction_index,
        .induction_alias = i_alias,
        .induction_update = ind_update,
        .step = step,
        .accesses = try accesses.toOwnedSlice(allocator),
        .reduction = reduction,
        .factor = V,
        .groups = groups,
        .index_elem_type = index_elem_type,
    };
}

fn hasNonAddressUse(func: *const Function, body: Block, value: Value, accesses: []const ArrayAccess) bool {
    use_scan: for (func.blockInsts(body)) |inst| {
        for (accesses) |access| {
            if (inst == access.addr_inst or inst == access.scale_inst) continue :use_scan;
        }
        if (usesValue(func, inst, value)) return true;
    }
    return false;
}

fn bail(accesses: *std.ArrayList(ArrayAccess), allocator: std.mem.Allocator) ?Plan {
    accesses.deinit(allocator);
    return null;
}

fn paramIndex(params: []const Value, v: Value) ?usize {
    for (params, 0..) |p, i| if (p == v) return i;
    return null;
}

const Strided = struct { base: Value, stride: i64, offset: i64, addr_inst: Inst, scale_inst: Inst };

/// Decompose `addr` as `base + i_alias*stride` (base loop-invariant), or null.
fn stridedAddress(func: *const Function, addr: Value, i_alias: Value) ?Strided {
    const addr_inst = func.definingInst(addr) orelse return null;
    if (func.opcode(addr_inst) == .arith_imm) {
        const field = func.opcode(addr_inst).arith_imm;
        if (field.op == .add) {
            var inner = stridedAddress(func, field.lhs, i_alias) orelse return null;
            inner.offset = std.math.add(i64, inner.offset, field.imm) catch return null;
            inner.addr_inst = addr_inst;
            return inner;
        }
    }
    const add = switch (func.opcode(addr_inst)) {
        .arith => |a| if (a.op == .add) a else return null,
        else => return null,
    };
    if (scaleOf(func, add.rhs, i_alias)) |sc| return .{ .base = add.lhs, .stride = sc.stride, .offset = sc.offset, .addr_inst = addr_inst, .scale_inst = sc.inst };
    if (scaleOf(func, add.lhs, i_alias)) |sc| return .{ .base = add.rhs, .stride = sc.stride, .offset = sc.offset, .addr_inst = addr_inst, .scale_inst = sc.inst };
    return null;
}

/// Recognize `v == i_alias * C` (via `mul` or `shl`), returning the constant stride and the inst.
fn scaleOf(func: *const Function, v: Value, i_alias: Value) ?struct { stride: i64, offset: i64, inst: Inst } {
    const di = func.definingInst(v) orelse return null;
    switch (func.opcode(di)) {
        .arith_imm => |a| {
            const index_offset = inductionOffset(func, a.lhs, i_alias) orelse return null;
            return switch (a.op) {
                .mul => .{ .stride = a.imm, .offset = std.math.mul(i64, index_offset, a.imm) catch return null, .inst = di },
                .shl => blk: {
                    if (a.imm < 0 or a.imm >= 63) break :blk null;
                    const stride = @as(i64, 1) << @intCast(a.imm);
                    break :blk .{ .stride = stride, .offset = std.math.mul(i64, index_offset, stride) catch return null, .inst = di };
                },
                else => null,
            };
        },
        .arith => |a| {
            if (a.op == .mul) {
                if (inductionOffset(func, a.lhs, i_alias)) |index_offset| {
                    if (constOf(func, a.rhs)) |c| return .{ .stride = c, .offset = std.math.mul(i64, index_offset, c) catch return null, .inst = di };
                } else if (inductionOffset(func, a.rhs, i_alias)) |index_offset| {
                    if (constOf(func, a.lhs)) |c| return .{ .stride = c, .offset = std.math.mul(i64, index_offset, c) catch return null, .inst = di };
                }
            } else if (a.op == .shl) {
                if (inductionOffset(func, a.lhs, i_alias)) |index_offset| if (constOf(func, a.rhs)) |c| {
                    if (c < 0 or c >= 63) return null;
                    const stride = @as(i64, 1) << @intCast(c);
                    return .{ .stride = stride, .offset = std.math.mul(i64, index_offset, stride) catch return null, .inst = di };
                };
            }
            return null;
        },
        else => return null,
    }
}

/// The constant offset in `v == i_alias + offset`, through the integer-width conversion C pointer
/// arithmetic inserts. This admits stencil neighbours such as `(i - 1) * 4` while preserving their
/// byte displacement when the vector body regenerates addresses.
fn inductionOffset(func: *const Function, v: Value, i_alias: Value) ?i64 {
    if (v == i_alias) return 0;
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .convert => |c| inductionOffset(func, c.value, i_alias),
        .arith_imm => |a| blk: {
            const base = inductionOffset(func, a.lhs, i_alias) orelse break :blk null;
            break :blk switch (a.op) {
                .add => std.math.add(i64, base, a.imm) catch null,
                .sub => std.math.sub(i64, base, a.imm) catch null,
                else => null,
            };
        },
        .arith => |a| blk: {
            if (a.op == .add) {
                if (inductionOffset(func, a.lhs, i_alias)) |base| if (constOf(func, a.rhs)) |c| break :blk std.math.add(i64, base, c) catch null;
                if (inductionOffset(func, a.rhs, i_alias)) |base| if (constOf(func, a.lhs)) |c| break :blk std.math.add(i64, base, c) catch null;
            } else if (a.op == .sub) {
                if (inductionOffset(func, a.lhs, i_alias)) |base| if (constOf(func, a.rhs)) |c| break :blk std.math.sub(i64, base, c) catch null;
            }
            break :blk null;
        },
        else => null,
    };
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .iconst => |c| c,
        else => null,
    };
}

fn constAddend(func: *const Function, v: Value, base: Value) ?i64 {
    const di = func.definingInst(v) orelse return null;
    return switch (func.opcode(di)) {
        .arith_imm => |a| if (a.op == .add and a.lhs == base) a.imm else null,
        .arith => |a| if (a.op == .add and a.lhs == base) constOf(func, a.rhs) else if (a.op == .add and a.rhs == base) constOf(func, a.lhs) else null,
        else => null,
    };
}

/// The element byte size for a scalar type (unit stride equals this).
fn byteSize(func: *const Function, ty: ir.types.Type) i64 {
    return switch (func.types.type_kind(ty)) {
        .int => |i| @intCast((i.bits + 7) / 8),
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .f128 => 16,
        },
        .ptr => 8,
        else => 0,
    };
}

/// Whether `inst` reads `v` in any operand slot. The callers count uses to prove a value is
/// dead or is read only where they expect, so a missed reader answers "unused" and lets an
/// unsound rewrite through. The switch is exhaustive for that reason: a trailing `else`
/// answered "no" for every call, every `matmul`, the whole `va_*` family and the `@"if"` edge
/// arguments.
fn usesValue(func: *const Function, inst: Inst, v: Value) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.lhs == v or a.rhs == v,
        .arith_imm => |a| a.lhs == v,
        .icmp => |c| c.lhs == v or c.rhs == v,
        .select => |s| s.cond == v or s.then == v or s.@"else" == v,
        .convert => |c| c.value == v,
        .decode_low_float, .encode_low_float => |c| c.value == v,
        .dequantize_nvfp4, .quantize_nvfp4 => |c| c.value == v or c.block_scale == v or c.global_scale == v,
        .unary => |u| u.value == v,
        .load => |l| l.ptr == v,
        .store => |s| s.value == v or s.ptr == v or s.mask == v,
        .atomic_rmw => |a| a.ptr == v or a.value == v or (a.compare != null and a.compare.? == v),
        .prefetch => |p| p.ptr == v,
        .dot => |d| d.acc == v or d.a == v or d.b == v,
        .reduce => |red| red.vector == v,
        .splat => |sp| sp.scalar == v,
        .matmul => |m| m.a == v or m.b == v or m.c == v,
        .va_start => |s| s.list == v,
        .va_arg => |a| a.list == v,
        .va_end => |e| e.list == v,
        .extract => |e| e.aggregate == v,
        .struct_new => |sn| blk: {
            for (func.valueList(sn.fields)) |f| if (f == v) break :blk true;
            break :blk false;
        },
        .call => |c| blk: {
            for (func.valueList(c.args)) |a| if (a == v) break :blk true;
            break :blk false;
        },
        .call_indirect => |c| blk: {
            if (c.target == v) break :blk true;
            for (func.valueList(c.args)) |a| if (a == v) break :blk true;
            break :blk false;
        },
        .@"if" => |cf| blk: {
            if (cf.cond == v) break :blk true;
            for (func.blockArgs(cf.then)) |a| if (a == v) break :blk true;
            for (func.blockArgs(cf.@"else")) |a| if (a == v) break :blk true;
            break :blk false;
        },
        // Reads no Value operand at all.
        .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => false,
    };
}

fn definedInLoop(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    if (func.definingInst(v)) |di| {
        for (0..func.blockCount()) |bi| {
            for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
                if (inst == di) return bi < loop.body.len and loop.body[bi];
            }
        }
        return false;
    }
    for (0..func.blockCount()) |bi| {
        for (func.blockParams(@fromBackingInt(@intCast(bi)))) |p| {
            if (p == v) return bi < loop.body.len and loop.body[bi];
        }
    }
    return false;
}

/// A recognized `s op= a[i]` reduction over contiguous memory, vectorizable to a vector accumulator.
const RedPlan = struct {
    header: Block,
    preheader: Block,
    exit_cond: CmpOp,
    bound: Value,
    induction: usize,
    accumulator: usize,
    op: BinOp,
    elem_ty: ir.types.Type, // the reduction/element type (32-bit)
    load_base: Value, // the contiguous load's base pointer
    stride: i64, // element byte size (unit stride)
    factor: u32, // V
};

fn recognizeReduction(func: *const Function, model: *const mm.Model, loop: *const loops.Loop, fast_math: bool) ?RedPlan {
    const vec_ok = switch (model.features) {
        .aarch64 => |f| f.neon,
        .riscv64 => |f| f.v or f.vpu,
        .x86_64 => false,
        // See vectorize.zig: a warp lane is already the vector lane on an SM, so there is no IR
        // vector op for the NVIDIA backend to lower.
        .nvidia => false,
    };
    if (!vec_ok or model.vector_bits < 2 * ELEM_BITS) return null;
    const V: u32 = @min(model.vector_bits / ELEM_BITS, MAX_LANES);
    if (V < 2) return null;

    const header: Block = @fromBackingInt(@intCast(loop.header));
    const preheader = loop.preheader orelse return null;

    // Single-block straight-line body, only latch, no if/matmul.
    var body: ?Block = null;
    var in_loop_blocks: usize = 0;
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi]) continue;
        in_loop_blocks += 1;
        const b: Block = @fromBackingInt(@intCast(bi));
        if (b == header) continue;
        switch (func.terminator(b) orelse return null) {
            .jump => |j| if (j.target == header) {
                if (body != null) return null;
                body = b;
            } else return null,
            .ret => return null,
        }
    }
    if (in_loop_blocks != 2) return null;
    const bodyb = body orelse return null;
    for (func.blockInsts(bodyb)) |inst| switch (func.opcode(inst)) {
        // An atomic joins them, for both reasons at once: vectorizing by V would run V
        // times fewer read-modify-writes, and `applyReduction` does not clone this body at
        // all, so the write would happen only in the scalar remainder.
        .atomic_rmw,
        .decode_low_float,
        .encode_low_float,
        .dequantize_nvfp4,
        .quantize_nvfp4,
        // A barrier joins them: vectorizing by V divides the trip count by V, so the loop
        // would meet V times fewer than the source says.
        //
        // `applyReduction` BUILDS a new main body (a wide load plus one vector arith). It does
        // not clone this one, so any side effect here would run only in the scalar remainder.
        // A trailing `else` let a `store`, a call and the `va_*` family through, and the
        // rewrite then executed them for the remainder iterations alone.
        .@"if",
        .matmul,
        .barrier,
        .store,
        .call,
        .call_indirect,
        .va_start,
        .va_arg,
        .va_end,
        => return null,
        // A `volatile` load is an observable access (see `Load.@"volatile"`). The new main body
        // reads V elements with ONE wide load, so V observable accesses would become one and the
        // flag would be lost. A plain load is pure and may be widened.
        //
        // An `endian`-tagged load is refused for the byte-order reason (see `Attribute.endian`).
        // This path WIDENS: `applyReduction` builds one `<V x elem>` load and carries no
        // attribute onto it. Even carrying the tag would be wrong, because reversing a V-element
        // vector is not the same as reversing each element. The map path above REPLICATES
        // instead, and carries the tag onto every copy, which is correct there.
        .load => |l| if (l.@"volatile" or func.isByteOrderTagged(inst)) return null,
        // Pure, or a hint with no observable effect. `applyReduction` reads the load and the
        // reduction arith it recognized below and needs nothing else from this body.
        .prefetch,
        .iconst,
        .fconst,
        .fconst128,
        .arith,
        .arith_imm,
        .icmp,
        .select,
        .struct_new,
        .extract,
        .convert,
        .unary,
        .alloca,
        .global_addr,
        .dot,
        .reduce,
        .splat,
        => {},
    };

    // Header: pure test, body on then, params passed through.
    var if_inst: ?Inst = null;
    for (func.blockInsts(header)) |inst| switch (func.opcode(inst)) {
        .@"if" => {
            if (if_inst != null) return null;
            if_inst = inst;
        },
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .convert, .unary => {},
        else => return null,
    };
    const cf = func.opcode(if_inst orelse return null).@"if";
    if (cf.then.target != bodyb) return null;
    const hparams = func.blockParams(header);
    if (hparams.len != 2) return null; // exactly [induction, accumulator]
    const body_args = func.blockArgs(cf.then);
    if (body_args.len != hparams.len) return null;
    for (body_args, hparams) |arg, hp| if (arg != hp) return null;

    const cmp = switch (func.opcode(func.definingInst(cf.cond) orelse return null)) {
        .icmp => |c| c,
        else => return null,
    };
    if (cmp.op != .lt and cmp.op != .le) return null;
    const induction = paramIndex(hparams, cmp.lhs) orelse return null;
    if (definedInLoop(func, loop, cmp.rhs)) return null;
    const accumulator = 1 - induction; // the other of the two params
    const bparams = func.blockParams(bodyb);
    const back_args = func.blockArgs(func.terminator(bodyb).?.jump);

    // Induction steps by 1.
    if ((constAddend(func, back_args[induction], bparams[induction]) orelse return null) != 1) return null;

    // The accumulator's next value is `acc_body op load(a[i])`, op associative & reorderable, and the
    // increment is a contiguous unit-stride load of a 32-bit element.
    const acc_body = bparams[accumulator];
    const upd = func.definingInst(back_args[accumulator]) orelse return null;
    const ar = switch (func.opcode(upd)) {
        .arith => |a| a,
        else => return null,
    };
    const incr = if (ar.lhs == acc_body) ar.rhs else if (ar.rhs == acc_body) ar.lhs else return null;
    if (!reorderable(func, back_args[accumulator], ar.op, fast_math)) return null;
    const load_inst = func.definingInst(incr) orelse return null;
    const load = switch (func.opcode(load_inst)) {
        .load => |l| l,
        else => return null,
    };
    const elem_ty = func.valueType(incr);
    if (byteSize(func, elem_ty) * 8 != ELEM_BITS) return null; // 32-bit element for the vector path
    const sa = stridedAddress(func, load.ptr, bparams[induction]) orelse return null;
    if (sa.stride != byteSize(func, elem_ty)) return null;
    // The accumulator body-alias is used only by its update.
    if (countUses(func, bodyb, acc_body) != 1) return null;

    return RedPlan{
        .header = header,
        .preheader = @fromBackingInt(@intCast(preheader)),
        .exit_cond = cmp.op,
        .bound = cmp.rhs,
        .induction = induction,
        .accumulator = accumulator,
        .op = ar.op,
        .elem_ty = elem_ty,
        .load_base = sa.base,
        .stride = sa.stride,
        .factor = V,
    };
}

fn countUses(func: *const Function, block: Block, v: Value) usize {
    var c: usize = 0;
    for (func.blockInsts(block)) |inst| {
        if (usesValue(func, inst, v)) c += 1;
    }
    if (func.terminator(block)) |t| switch (t) {
        .jump => |j| for (func.blockArgs(j)) |a| {
            if (a == v) c += 1;
        },
        .ret => |r| for (r.slice()) |xx| {
            if (xx == v) c += 1;
        },
    };
    return c;
}

fn reorderable(func: *const Function, v: Value, op: BinOp, fast_math: bool) bool {
    const is_float = switch (func.types.type_kind(func.valueType(v))) {
        .float => true,
        else => false,
    };
    return switch (op) {
        .add, .mul => if (is_float) fast_math else true,
        .bit_and, .bit_or, .bit_xor => !is_float,
        .sub, .div, .rem, .shl, .shr, .mulh => false,
    };
}

fn identityConst(func: *Function, block: Block, op: BinOp, ty: ir.types.Type) Error!Value {
    const is_float = switch (func.types.type_kind(ty)) {
        .float => true,
        else => false,
    };
    return switch (op) {
        .add, .bit_or, .bit_xor => if (is_float) func.appendInst(block, ty, .{ .fconst = 0 }) else func.appendInst(block, ty, .{ .iconst = 0 }),
        .mul => if (is_float) func.appendInst(block, ty, .{ .fconst = 1 }) else func.appendInst(block, ty, .{ .iconst = 1 }),
        .bit_and => func.appendInst(block, ty, .{ .iconst = -1 }),
        else => unreachable,
    };
}

fn applyReduction(allocator: std.mem.Allocator, func: *Function, plan: *const RedPlan) Error!void {
    const V = plan.factor;
    const hparams = try allocator.dupe(Value, func.blockParams(plan.header));
    defer allocator.free(hparams);
    const ind_ty = func.valueType(hparams[plan.induction]);
    const base_ty = func.valueType(plan.load_base);
    const vt = try func.types.intern(.{ .vector = .{ .len = @intCast(V), .elem = plan.elem_ty } });

    const mainheader = try func.appendBlock();
    const mainbody = try func.appendBlock();
    const combine = try func.appendBlock();

    // Preheader: build a splat-identity vector accumulator, redirect into the main header.
    const pre_args = try allocator.dupe(Value, func.blockArgs(func.terminator(plan.preheader).?.jump));
    defer allocator.free(pre_args);
    const init_i = pre_args[plan.induction];
    const init_acc = pre_args[plan.accumulator];
    const ident = try identityConst(func, plan.preheader, plan.op, plan.elem_ty);
    var idlanes: std.ArrayList(Value) = .empty;
    defer idlanes.deinit(allocator);
    for (0..V) |_| try idlanes.append(allocator, ident);
    const vacc0 = try func.appendInst(plan.preheader, vt, .{ .struct_new = .{ .fields = try func.internValues(idlanes.items) } });
    try func.setJump(plan.preheader, mainheader, &.{ init_i, vacc0 });

    // Main header: [mi, vacc]. Run a full V-group while the highest index passes the test.
    const mi = try func.appendBlockParam(mainheader, ind_ty);
    const vacc = try func.appendBlockParam(mainheader, vt);
    const highest = try func.appendArithImm(mainheader, ind_ty, .add, mi, @as(i64, @intCast(V - 1)));
    const bool_t = try func.types.intern(.bool);
    const cont = try func.appendInst(mainheader, bool_t, .{ .icmp = .{ .op = plan.exit_cond, .lhs = highest, .rhs = plan.bound } });
    try func.appendIf(mainheader, cont, .{ .target = mainbody, .args = &.{ mi, vacc } }, .{ .target = combine, .args = &.{ mi, vacc } });

    // Main body: wide-load V contiguous elements at base + mi*stride, add into the vector accumulator.
    const bmi = try func.appendBlockParam(mainbody, ind_ty);
    const bvacc = try func.appendBlockParam(mainbody, vt);
    const scaled = try func.appendArithImm(mainbody, base_ty, .mul, bmi, plan.stride);
    const base_i = try func.appendInst(mainbody, base_ty, .{ .arith = .{ .op = .add, .lhs = plan.load_base, .rhs = scaled } });
    const wv = try func.appendInst(mainbody, vt, .{ .load = .{ .ptr = base_i } });
    const nvacc = try func.appendInst(mainbody, vt, .{ .arith = .{ .op = plan.op, .lhs = bvacc, .rhs = wv } });
    const nmi = try func.appendArithImm(mainbody, ind_ty, .add, bmi, @as(i64, @intCast(V)));
    try func.setJump(mainbody, mainheader, &.{ nmi, nvacc });

    // Combine: horizontally reduce the vector accumulator (extract each lane, tree-sum), fold in the
    // original init, then fall into the ORIGINAL loop (remainder) for the leftover scalar iterations.
    const cmi = try func.appendBlockParam(combine, ind_ty);
    const cvacc = try func.appendBlockParam(combine, vt);
    var lanes: std.ArrayList(Value) = .empty;
    defer lanes.deinit(allocator);
    for (0..V) |k| try lanes.append(allocator, try func.appendInst(combine, plan.elem_ty, .{ .extract = .{ .aggregate = cvacc, .index = @intCast(k) } }));
    const hsum = try buildTree(func, combine, plan.op, plan.elem_ty, lanes.items, allocator);
    const sacc = try func.appendInst(combine, plan.elem_ty, .{ .arith = .{ .op = plan.op, .lhs = init_acc, .rhs = hsum } });
    const rem_args = try allocator.alloc(Value, hparams.len);
    defer allocator.free(rem_args);
    rem_args[plan.induction] = cmi;
    rem_args[plan.accumulator] = sacc;
    try func.setJump(combine, plan.header, rem_args);
}

fn buildTree(func: *Function, block: Block, op: BinOp, ty: ir.types.Type, items: []const Value, allocator: std.mem.Allocator) Error!Value {
    var cur: std.ArrayList(Value) = .empty;
    defer cur.deinit(allocator);
    try cur.appendSlice(allocator, items);
    while (cur.items.len > 1) {
        var next: std.ArrayList(Value) = .empty;
        var i: usize = 0;
        while (i + 1 < cur.items.len) : (i += 2) {
            try next.append(allocator, try func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = cur.items[i], .rhs = cur.items[i + 1] } }));
        }
        if (cur.items.len % 2 == 1) try next.append(allocator, cur.items[cur.items.len - 1]);
        cur.deinit(allocator);
        cur = next;
    }
    return cur.items[0];
}

const testing = std.testing;
const registry = @import("registry.zig");

/// Which access of `buildSaxpy`'s body carries the `volatile` flag.
const SaxpyVol = enum { none, load, store };

/// `for (i = 0; i < n; i += 1) y[i] = a*x[i] + y[i];` over f32 arrays. Induction only used in the two
/// contiguous addresses; the classic saxpy the vectorizer should recognize. `vol` marks the `x` load
/// or the `y` store `volatile`, which must make the loop unrecognizable.
fn buildSaxpy(func: *Function, vol: SaxpyVol) !void {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const x = try func.appendBlockParam(entry, ptr_t);
    const y = try func.appendBlockParam(entry, ptr_t);
    const a = try func.appendBlockParam(entry, f32_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4); // i*4 (f32 elem bytes)
    const xaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = off } });
    const xv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = xaddr, .@"volatile" = vol == .load } });
    const yaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = off } });
    const yv = try func.appendInst(body, f32_t, .{ .load = .{ .ptr = yaddr } });
    const ax = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = xv } });
    const res = try func.appendInst(body, f32_t, .{ .arith = .{ .op = .add, .lhs = ax, .rhs = yv } });
    try func.appendStoreVol(body, res, yaddr, vol == .store);
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

test "recognizes a saxpy map loop: two contiguous loads, one contiguous store, V lanes" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func, .none);

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    const plan = (try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(plan.accesses);

    try testing.expectEqual(@as(u32, 4), plan.factor); // 128-bit / 32-bit f32 = 4 lanes
    try testing.expectEqual(@as(u32, 1), plan.groups);
    var loads: usize = 0;
    var stores: usize = 0;
    for (plan.accesses) |acc| {
        if (acc.is_store) stores += 1 else loads += 1;
        try testing.expectEqual(@as(i64, 4), acc.stride); // unit stride for f32
    }
    try testing.expectEqual(@as(usize, 2), loads);
    try testing.expectEqual(@as(usize, 1), stores);
}

test "frontend-shaped map loop may use the induction value as stored data" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const out = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, u32_t);
    const zero = try func.appendInst(entry, u32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, u32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, u32_t);
    const off = try func.appendArithImm(body, u32_t, .mul, bi, 4);
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = out, .rhs = off } });
    const stored = try func.appendInst(body, i32_t, .{ .convert = .{ .value = bi } });
    try func.appendStore(body, stored, addr);
    try func.setJump(body, latch, &.{});
    const ni = try func.appendArithImm(latch, u32_t, .add, bi, 1);
    try func.setJump(latch, loop, &.{ni});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    const plan = (try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(plan.accesses);
    try testing.expectEqual(i32_t, plan.index_elem_type.?);
    try testing.expectEqual(@as(u32, 2), plan.groups);

    try apply(allocator, &func, &plan);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "recognizes a frontend-shaped stencil map with xor reduction and affine neighbours" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const input = try func.appendBlockParam(entry, ptr_t);
    const output = try func.appendBlockParam(entry, ptr_t);
    const count = try func.appendBlockParam(entry, u32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    const one = try func.appendInst(entry, u32_t, .{ .iconst = 1 });
    try func.setJump(entry, header, &.{ one, zero });

    const index = try func.appendBlockParam(header, u32_t);
    const checksum = try func.appendBlockParam(header, i32_t);
    const next = try func.appendArithImm(header, u32_t, .add, index, 1);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next, .rhs = count } });
    try func.appendIf(header, cond, .{ .target = body }, .{ .target = done });

    var values: [3]Value = undefined;
    for ([_]i64{ -1, 0, 1 }, 0..) |bias, lane| {
        const neighbour = if (bias == 0) index else try func.appendArithImm(body, u32_t, .add, index, bias);
        const wide = try func.appendInst(body, i64_t, .{ .convert = .{ .value = neighbour } });
        const offset = try func.appendArithImm(body, i64_t, .shl, wide, 2);
        const address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = input, .rhs = offset } });
        values[lane] = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = address } });
    }
    const doubled = try func.appendArithImm(body, i32_t, .shl, values[1], 1);
    const left_center = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = values[0], .rhs = doubled } });
    const value = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = left_center, .rhs = values[2] } });
    const out_wide = try func.appendInst(body, i64_t, .{ .convert = .{ .value = index } });
    const out_offset = try func.appendArithImm(body, i64_t, .shl, out_wide, 2);
    const out_address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = output, .rhs = out_offset } });
    try func.appendStore(body, value, out_address);
    const updated = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_xor, .lhs = checksum, .rhs = value } });
    try func.setJump(body, latch, &.{});
    try func.setJump(latch, header, &.{ next, updated });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(checksum) });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    const plan = (try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(plan.accesses);
    try testing.expectEqual(@as(i64, 1), plan.exit_offset);
    try testing.expectEqual(BinOp.bit_xor, plan.reduction.?.op);
    var saw_left = false;
    var saw_center = false;
    var saw_right = false;
    for (plan.accesses) |access| {
        saw_left = saw_left or access.offset == -4;
        saw_center = saw_center or access.offset == 0;
        saw_right = saw_right or access.offset == 4;
    }
    try testing.expect(saw_left and saw_center and saw_right);

    try testing.expect(try run(allocator, &func, registry.modelFor(.@"ampere-altra")));
    const vector_header: Block = @fromBackingInt(5);
    var compares: usize = 0;
    var boolean_ands: usize = 0;
    var saw_wide_group_offset = false;
    for (func.blockInsts(vector_header)) |inst| switch (func.opcode(inst)) {
        .icmp => compares += 1,
        .arith => |arith| if (arith.op == .bit_and) {
            boolean_ands += 1;
        },
        .arith_imm => |arith| if (arith.op == .add and arith.imm == 4) {
            const result = func.instResult(inst).?;
            if (func.types.type_kind(func.valueType(result)) == .int) {
                const int = func.types.type_kind(func.valueType(result)).int;
                saw_wide_group_offset = int.signedness == .unsigned and int.bits == 64;
            }
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), compares);
    try testing.expectEqual(@as(usize, 0), boolean_ands);
    try testing.expect(!saw_wide_group_offset);
    const vector_body = vector_header;
    var scaled_bases: usize = 0;
    for (func.blockInsts(vector_body)) |inst| switch (func.opcode(inst)) {
        .arith_imm => |arith| if (arith.op == .mul) {
            scaled_bases += 1;
        },
        else => {},
    };
    try testing.expectEqual(@as(usize, 2), scaled_bases); // one each for input and output
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "declines a non-unit-stride access (a[2*i])" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const arr = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{zero});
    const i = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{i} }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 8); // 2*i for a 4-byte element: stride 8 != 4
    const addr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = arr, .rhs = off } });
    _ = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = addr } });
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ni});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    try testing.expect((try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) == null);
}

/// `for (i = 0; i < n; i += 1) { s += a[i]; if (extra_store) b[i] = 0; }` over i32 arrays.
/// The reduction shape `recognizeReduction` looks for, with an optional SECOND memory effect
/// in the body.
fn buildIntReduction(func: *Function, extra_store: bool) !void {
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const loop = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const a = try func.appendBlockParam(entry, ptr_t);
    const b = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, loop, &.{ zero, zero });
    const i = try func.appendBlockParam(loop, i32_t);
    const s = try func.appendBlockParam(loop, i32_t);
    const cmp = try func.appendInst(loop, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(loop, cmp, .{ .target = body, .args = &.{ i, s } }, .{ .target = done });
    const bi = try func.appendBlockParam(body, i32_t);
    const bs = try func.appendBlockParam(body, i32_t);
    const off = try func.appendArithImm(body, i32_t, .mul, bi, 4);
    const aaddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = off } });
    const av = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = aaddr } });
    const ns = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = bs, .rhs = av } });
    if (extra_store) {
        const baddr = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = b, .rhs = off } });
        const zero_b = try func.appendInst(body, i32_t, .{ .iconst = 0 });
        try func.appendStore(body, zero_b, baddr);
    }
    const ni = try func.appendArithImm(body, i32_t, .add, bi, 1);
    try func.setJump(body, loop, &.{ ni, ns });
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

test "recognizes a reduction-only loop with a computed increment" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const left = try func.appendBlockParam(entry, ptr_t);
    const right = try func.appendBlockParam(entry, ptr_t);
    const count = try func.appendBlockParam(entry, u32_t);
    const zero_i = try func.appendInst(entry, u32_t, .{ .iconst = 0 });
    const zero_sum = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, header, &.{ zero_i, zero_sum });

    const index = try func.appendBlockParam(header, u32_t);
    const checksum = try func.appendBlockParam(header, i32_t);
    const cond = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = index, .rhs = count } });
    try func.appendIf(header, cond, .{ .target = body, .args = &.{ index, checksum } }, .{ .target = done });

    const body_index = try func.appendBlockParam(body, u32_t);
    const body_checksum = try func.appendBlockParam(body, i32_t);
    const offset = try func.appendArithImm(body, u32_t, .mul, body_index, 4);
    const left_address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = left, .rhs = offset } });
    const right_address = try func.appendInst(body, ptr_t, .{ .arith = .{ .op = .add, .lhs = right, .rhs = offset } });
    const left_value = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = left_address } });
    const right_value = try func.appendInst(body, i32_t, .{ .load = .{ .ptr = right_address } });
    const product = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .mul, .lhs = left_value, .rhs = right_value } });
    const signed_index = try func.appendInst(body, i32_t, .{ .convert = .{ .value = body_index } });
    const index_term = try func.appendArithImm(body, i32_t, .mul, signed_index, 17);
    const increment = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .add, .lhs = product, .rhs = index_term } });
    const next_checksum = try func.appendInst(body, i32_t, .{ .arith = .{ .op = .bit_xor, .lhs = body_checksum, .rhs = increment } });
    const next_index = try func.appendArithImm(body, u32_t, .add, body_index, 1);
    try func.setJump(body, header, &.{ next_index, next_checksum });
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(checksum) });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    const plan = (try recognize(allocator, &func, registry.modelFor(.@"ampere-altra"), &info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(plan.accesses);
    try testing.expectEqual(@as(usize, 2), plan.accesses.len);
    try testing.expectEqual(BinOp.bit_xor, plan.reduction.?.op);
    try testing.expectEqual(increment, plan.reduction.?.increment);
    try testing.expectEqual(i32_t, plan.index_elem_type.?);
    try testing.expectEqual(@as(u32, 2), plan.groups);

    try apply(allocator, &func, &plan);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a reduction body with a second memory effect is declined" {
    // `applyReduction` BUILDS a new main body (one wide load plus one vector arith) instead of
    // cloning this one, and it divides the trip count by V. Any other side effect in the body
    // would then run only in the scalar remainder, so `s += a[i]; b[i] = 0;` would leave most
    // of `b` unwritten. A trailing `else => {}` in the body filter let the store through.
    //
    // The same loop WITHOUT the store is still recognized, so the filter declines the store
    // and nothing else.
    const allocator = testing.allocator;

    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildIntReduction(&plain, false);
    var plain_info = try loops.analyze(allocator, &plain);
    defer plain_info.deinit(allocator);
    const scored_plan = (try recognize(allocator, &plain, registry.modelFor(.@"ampere-altra"), &plain_info.loops[0])) orelse return error.NotRecognized;
    defer allocator.free(scored_plan.accesses);
    try testing.expectEqual(@as(u32, 2), scored_plan.groups);
    try testing.expect(recognizeReduction(&plain, registry.modelFor(.@"ampere-altra"), &plain_info.loops[0], false) != null);

    var stored = Function.init(allocator);
    defer stored.deinit();
    try buildIntReduction(&stored, true);
    var stored_info = try loops.analyze(allocator, &stored);
    defer stored_info.deinit(allocator);
    try testing.expect(recognizeReduction(&stored, registry.modelFor(.@"ampere-altra"), &stored_info.loops[0], false) == null);
}

test "usesValue sees every operand slot, not only the arithmetic ones" {
    // `countUses` calls this to prove a value is read only where expected. A missed reader answers
    // "unused" and lets an unsound rewrite
    // through. A trailing `else => false` hid every call argument, both `matmul` operands
    // and the `va_*` list.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const block = try func.appendBlock();
    const other = try func.appendBlock();
    const p = try func.appendBlockParam(block, ptr_t);
    const c = try func.appendBlockParam(block, bool_t);

    try func.appendMatmul(block, p, p, p, 4, 4, 4, .int8, false);
    try func.appendPrefetch(block, p);
    try func.appendVaStart(block, p);
    _ = try func.appendVaArg(block, p, i32_t);
    try func.appendVaEnd(block, p);
    _ = try func.appendCall(block, i32_t, "f", &.{p});
    _ = try func.appendInst(block, i32_t, .{ .dot = .{ .acc = p, .a = p, .b = p } });
    try func.appendIf(block, c, .{ .target = other, .args = &.{p} }, .{ .target = other, .args = &.{p} });
    _ = try func.appendBlockParam(other, ptr_t);
    func.setTerminator(other, .{ .ret = ir.function.Ret.none() });

    for (func.blockInsts(block)) |inst| try testing.expect(usesValue(&func, inst, p));
}

test "volatile: a map loop whose access is volatile is not widened" {
    // Widening runs V source iterations per main-body copy, so the V copies of one scalar access
    // become the run the SLP pass fuses into a single wide vector load or store. That coalesces V
    // observable accesses into one and loses the flag (see `Load.@"volatile"`), so the loop must
    // not be recognized at all.
    const allocator = testing.allocator;
    const model = registry.modelFor(.@"ampere-altra");

    // NEGATIVE CONTROL: the identical loop with plain accesses IS recognized, so the refusal below
    // is a volatile rule and not a dead pass.
    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildSaxpy(&plain, .none);
    var plain_info = try loops.analyze(allocator, &plain);
    defer plain_info.deinit(allocator);
    const plan = (try recognize(allocator, &plain, model, &plain_info.loops[0])) orelse return error.NotRecognized;
    allocator.free(plan.accesses);

    for ([_]SaxpyVol{ .load, .store }) |vol| {
        var func = Function.init(allocator);
        defer func.deinit();
        try buildSaxpy(&func, vol);
        var info = try loops.analyze(allocator, &func);
        defer info.deinit(allocator);
        try testing.expect((try recognize(allocator, &func, model, &info.loops[0])) == null);

        // The access is still in the body, still volatile: refusing kept it, it was not rewritten.
        var seen = false;
        for (func.blockInsts(@fromBackingInt(@intCast(2)))) |inst| switch (func.opcode(inst)) {
            .load => |l| if (l.@"volatile") {
                seen = true;
            },
            .store => |s| if (s.@"volatile") {
                seen = true;
            },
            else => {},
        };
        try testing.expect(seen);
    }
}

test "volatile: a reduction over a volatile load is not widened" {
    // `applyReduction` builds a new main body reading V elements with ONE wide load. A volatile
    // source load must not be coalesced that way, so the reduction is declined and the scalar loop
    // (with its flag) survives.
    const allocator = testing.allocator;
    const model = registry.modelFor(.@"ampere-altra");

    // NEGATIVE CONTROL: the same reduction over a plain load is still recognized.
    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildIntReduction(&plain, false);
    var plain_info = try loops.analyze(allocator, &plain);
    defer plain_info.deinit(allocator);
    try testing.expect(recognizeReduction(&plain, model, &plain_info.loops[0], false) != null);

    var func = Function.init(allocator);
    defer func.deinit();
    try buildIntReduction(&func, false);
    // Mark the body's one load volatile, changing nothing else about the shape.
    const body: Block = @fromBackingInt(@intCast(2));
    var marked: usize = 0;
    for (func.blockInsts(body)) |inst| {
        if (func.opcode(inst) != .load) continue;
        func.opcodeMut(inst).load.@"volatile" = true;
        marked += 1;
    }
    try testing.expectEqual(@as(usize, 1), marked);

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    try testing.expect(recognizeReduction(&func, model, &info.loops[0], false) == null);
    try testing.expect(func.opcode(func.blockInsts(body)[2]).load.@"volatile");
}

test "the map path carries a body instruction's endian attribute onto every lane copy" {
    // The map path is a scalar unroll by U, not a widening: each copy loads the same 4 bytes as
    // the original, so it keeps the original's `endian`. A copy that lost it would stop being
    // byte-swapped while the source iteration it stands for still is.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func, .none);

    // Tag the `x` load's result. Body block 2 holds off, xaddr, xv, yaddr, yv, mul, add, store.
    const body: Block = @fromBackingInt(@intCast(2));
    const xv = func.instResult(func.blockInsts(body)[2]).?;
    try func.addAttr(.{ .value = xv }, .{ .endian = .big });
    try testing.expectEqual(@as(usize, 1), countBigEndianValues(&func));

    try testing.expect(try run(allocator, &func, registry.modelFor(.@"ampere-altra")));

    // The original scalar body stays as the remainder loop, so its tag survives too: one
    // original plus one per unrolled iteration.
    try testing.expectEqual(@as(usize, 1 + 4), countBigEndianValues(&func));
}

test "the map path does not carry a block attribute onto a rewritten block" {
    // Suspicious case, the other direction. A block attribute names a block of the ORIGINAL
    // region, and this pass builds new blocks around it.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildSaxpy(&func, .none);
    const body: Block = @fromBackingInt(@intCast(2));
    try func.addAttr(.{ .block = body }, .cold);

    try testing.expect(try run(allocator, &func, registry.modelFor(.@"ampere-altra")));

    var block_attrs: usize = 0;
    for (func.attributeEntries()) |entry| switch (entry.target) {
        .block => |b| {
            try testing.expectEqual(body, b);
            block_attrs += 1;
        },
        .func, .inst, .value => {},
    };
    try testing.expectEqual(@as(usize, 1), block_attrs);
}

/// How many values carry a big-endian tag.
fn countBigEndianValues(func: *const Function) usize {
    var n: usize = 0;
    for (func.attributeEntries()) |entry| {
        if (entry.target != .value) continue;
        switch (entry.attr) {
            .endian => |e| if (e == .big) {
                n += 1;
            },
            .@"inline", .noreturn, .cold, .@"align", .custom => {},
        }
    }
    return n;
}

test "endian: a reduction over a byte-order-tagged load is not widened" {
    // `applyReduction` builds a new main body reading V elements with ONE `<V x i32>` load. That
    // is a WIDENING, so a tagged source load cannot come along: reversing a V-element vector is
    // not the same as reversing each element, and the wide load carries no tag at all. The
    // reduction is declined and the scalar loop keeps its tag. Contrast the map path above, which
    // REPLICATES and does carry the tag.
    const allocator = testing.allocator;
    const model = registry.modelFor(.@"ampere-altra");

    // NEGATIVE CONTROL: the same reduction over an untagged load is still recognized.
    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildIntReduction(&plain, false);
    var plain_info = try loops.analyze(allocator, &plain);
    defer plain_info.deinit(allocator);
    try testing.expect(recognizeReduction(&plain, model, &plain_info.loops[0], false) != null);

    var func = Function.init(allocator);
    defer func.deinit();
    try buildIntReduction(&func, false);
    // Tag the body's one load, changing nothing else about the shape.
    const body: Block = @fromBackingInt(@intCast(2));
    var marked: usize = 0;
    for (func.blockInsts(body)) |inst| {
        if (func.opcode(inst) != .load) continue;
        try func.addAttr(.{ .value = func.instResult(inst).? }, .{ .endian = .big });
        marked += 1;
    }
    try testing.expectEqual(@as(usize, 1), marked);

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    try testing.expect(recognizeReduction(&func, model, &info.loops[0], false) == null);
    try testing.expect(func.isByteOrderTagged(func.blockInsts(body)[2]));
}

test "a reduction body holding an atomic is declined" {
    // `applyReduction` BUILDS a new main body instead of cloning this one, and it divides the
    // trip count by V. An atomic in the body would then run only in the scalar remainder,
    // which is V times fewer read-modify-writes than the source asks for.
    //
    // NEGATIVE CONTROL: the same loop without the atomic IS recognized, so the filter
    // declines the atomic and nothing else.
    const allocator = testing.allocator;
    const model = registry.modelFor(.@"ampere-altra");

    var plain = Function.init(allocator);
    defer plain.deinit();
    try buildIntReduction(&plain, false);
    var plain_info = try loops.analyze(allocator, &plain);
    defer plain_info.deinit(allocator);
    try testing.expect(recognizeReduction(&plain, model, &plain_info.loops[0], false) != null);

    var func = Function.init(allocator);
    defer func.deinit();
    try buildIntReduction(&func, false);
    // The same body plus one atomic increment of the second pointer, changing nothing else.
    const body: Block = @fromBackingInt(@intCast(2));
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry: Block = @fromBackingInt(@intCast(0));
    const b_ptr = func.blockParams(entry)[1];
    const one = try func.appendInst(body, i32_t, .{ .iconst = 1 });
    try func.appendAtomicRmwStmt(body, .{ .op = .add, .ptr = b_ptr, .value = one, .ordering = .relaxed, .scope = .device });

    var info = try loops.analyze(allocator, &func);
    defer info.deinit(allocator);
    try testing.expect(recognizeReduction(&func, model, &info.loops[0], false) == null);
}
