//! Glacier instruction selection: a Vulcan IR kernel to a Glacier word stream.
//!
//! A Glacier kernel is not a function. It has no caller, no arguments and no stack, so this
//! lowering is narrower than a CPU backend on purpose:
//!
//!   - A kernel takes NO parameters. Its entry-block parameters must all be hardware builtins
//!     (`vulcan-gpu`'s `Builtin`), and everything else a kernel needs is a constant the host
//!     already knew when it built the program. A launch shape that changes rebuilds the kernel.
//!   - A kernel RETIRES. An empty `ret` becomes EBREAK, and a `ret` with a value is refused:
//!     there is nowhere for the value to go.
//!   - A kernel does NOT spill. A warp has no stack, so this refuses an allocation that needs
//!     one rather than inventing a memory region the hardware has not promised.
//!
//! Register allocation is the shared Wimmer-Franz allocator, same as every other backend, so
//! liveness across a loop and the parallel copies on a control-flow edge are not reimplemented
//! here. This file owns the register model and the encoding.
//!
//! Floating point is admitted narrowly, because the core decodes five float instructions and no
//! more (see `encode.zig` and `glacier.fdiv`). An f32 and an f16 are values here, a float constant
//! is its bit pattern, the three converts the core has are selected, and an f32 divide becomes the
//! one scalar divide. Everything else is refused: there is no float add, no float compare, and no
//! per-lane divide. An f16 never reaches memory, since it is only the low half of a lane.
//!
//! One thing it deliberately does NOT do yet is fuse a compare into its branch. A fused branch
//! reads the COMPARE's operands at the branch, and the allocator only knows the branch reads the
//! boolean, so fusing needs `wimmer.RegDescription.fusedOperands` to tell it. Until that is
//! wired, a compare materializes 0 or 1 into a register and the branch tests it against x0. The
//! cost is one word per branch, which is correct and slower, not wrong and fast.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const wimmer = @import("../wimmer.zig");
const opt = @import("vulcan-opt");
const glacier = @import("encode.zig");

const rv = glacier.rv;
const Reg = glacier.Reg;
const VReg = rv.VReg;
const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error || error{Unsupported};

/// The register this lowering keeps for itself. It holds a constant too wide for an immediate
/// field, and the Wimmer resolver routes a parallel-copy cycle through it. It is never
/// allocatable, so no value ever lives here across an instruction boundary.
const scratch: Reg = .x31;

/// The stack pointer. Reserved by the Glacier ABI whether or not a kernel spills, and loaded from
/// `mscratch`, which reads one past the top of the reading hart's own scratch region.
const sp: Reg = .x2;

/// x1 and x3 to x30. A Glacier kernel has no call, so the return-address register carries data
/// like any other. `x2` is the ABI stack pointer and `x31` is the lowering's scratch.
const allocatable: [29]u16 = blk: {
    var regs: [29]u16 = undefined;
    var n: usize = 0;
    for (1..31) |i| {
        if (i == @backingInt(sp)) continue;
        regs[n] = i;
        n += 1;
    }
    break :blk regs;
};

/// The vector register this lowering keeps for itself, to splat a warp-uniform value into a lane
/// vector and to break a parallel-copy cycle. Never allocatable.
const vscratch: VReg = .v31;

/// v1 to v30. `v0` is reserved because it is the only register that PREDICATES: a mask has to be
/// moved into it to be applied. Nothing here predicates yet, so reserving it costs one register
/// and buys the guarantee that no value can be sitting in it when masks arrive.
const allocatable_v: [30]u16 = blk: {
    var regs: [30]u16 = undefined;
    for (&regs, 1..) |*r, i| r.* = i;
    break :blk regs;
};

const scratch_set: [2]u16 = .{ @backingInt(scratch), @backingInt(vscratch) };

/// Where a global the kernel names lives. A Glacier kernel reaches memory by absolute address:
/// the host knows where it put the framebuffer, and the kernel is built for that address.
pub const Global = struct { name: []const u8, address: u32 };

pub const Options = struct {
    /// The core the kernel is built for. It fixes the lane count, which is what `block_dim_x`
    /// and `subgroup_size` read.
    profile: glacier.Profile,
    /// Absolute addresses for the symbols the kernel's `global_addr` instructions name. A name
    /// that is not here is refused rather than guessed.
    globals: []const Global = &.{},
};

/// Lower `func` to a Glacier word stream. The caller owns the result.
pub fn compileKernel(allocator: std.mem.Allocator, func: *const Function, opts: Options) Error![]u32 {
    if (func.blockCount() == 0) return error.Unsupported;
    if (func.is_variadic or func.sret) return error.Unsupported;

    // The Wimmer resolver needs a block on every critical edge to put its shuffle moves on, and
    // splitting mutates the function, so split a clone and leave the caller's input alone.
    var work = try func.clone(allocator);
    defer work.deinit();

    // Promote every stack slot to an SSA value BEFORE anything else looks at the function. A
    // warp has no stack, so an `alloca` that survives is refused below, and a frontend that keeps
    // its loop counters in memory (Zig's SPIR-V backend does) would otherwise be rejected for a
    // reason that has nothing to do with the hardware. This is also why the promotion belongs
    // here rather than in a frontend: it needs real phis, which only an SSA construction gives.
    {
        var analyses: opt.pass.Analyses = .{ .allocator = allocator, .func = &work };
        defer analyses.deinit();
        _ = try opt.mem2reg.run(allocator, &work, &analyses);
    }
    try ir.critical_edge.splitCriticalEdges(allocator, &work);
    const reachable = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable);

    const divergent = try analyzeDivergence(allocator, &work);
    defer allocator.free(divergent);
    var ctx: Ctx = .{ .divergent = divergent };

    try checkSupported(&work, opts, divergent);

    const classes = [_]wimmer.RegClass{
        .{
            .name = "gpr",
            .allocatable = &allocatable,
            .callee_saved = &.{},
            .slot_bytes = 4,
        },
        .{
            .name = "vreg",
            .allocatable = &allocatable_v,
            .callee_saved = &.{},
            .slot_bytes = @intCast(opts.profile.lanes() * 4),
        },
    };
    var desc: wimmer.RegDescription = .{
        .classes = &classes,
        .classOf = classOf,
        .useKind = useKind,
        .entry_fixed = &.{},
        .call_sites = &.{},
        .scratch = &scratch_set,
        .ctx = &ctx,
        .copySource = copySource,
        .coalesce_block_params = true,
    };
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);

    // A spill goes into the hart's own scratch region, so the kernel must fit there. The region
    // belongs to one hart, which is why no other hart's slots are counted against it.
    const slots = walloc.slot_count_per_class[0];
    if (slots > opts.profile.scratchBytes() / 4) return error.Unsupported;
    // The deepest slot is reached by one signed 12-bit displacement from `sp`.
    if (slots * 4 > 2048) return error.Unsupported;
    const uses = try allocator.alloc(u32, work.valueCount());
    defer allocator.free(uses);
    opt.dce.countUses(&work, uses);

    // A spilled VECTOR would need a vector store to the scratch region and a reload from it.
    // Nothing emits those yet, so refuse rather than drop the spill and miscompile.
    if (walloc.slot_count_per_class[1] != 0) return error.Unsupported;

    var e: Emitter = .{
        .allocator = allocator,
        .func = &work,
        .opts = opts,
        .alloc = &walloc,
        .divergent = divergent,
        .uses = uses,
        .spills = slots != 0,
    };
    defer e.deinit();
    try e.run();
    return e.words.toOwnedSlice(allocator);
}

// ===========================================================================
// Divergence: which values are per-lane.
//
// A Glacier warp is one workgroup and its LANES are the threads, so a value derived from the lane
// index varies across the lanes and a value that is not is the same in all of them. That split is
// the whole of the vector lowering: a DIVERGENT value lives in a vector register and its
// instructions are the RVV forms, a UNIFORM value lives in a scalar register exactly as before.
//
// The IR carries no vector type for this. It does not have to: the shape is a property of where a
// value came from, not of its type, and `thread_id_x` already says a thread is a lane.

/// What the analysis and the emitter share. `desc.ctx` points at it.
const Ctx = struct { divergent: []const bool };

/// Whether a builtin differs between the lanes of one warp. A warp index is the same in every
/// lane; a lane or thread index is the point of the exercise.
fn laneVarying(b: gpu.builtin.Builtin) bool {
    return switch (b) {
        .thread_id_x, .lane_id, .global_id_x => true,
        else => false,
    };
}

/// Visit every value `inst` reads.
///
/// Every opcode `checkSupported` accepts must have an arm here. The trailing `else` cannot be
/// `unreachable`, because this runs BEFORE admission and so sees opcodes that are about to be
/// refused. That makes the two lists a pair that has to be kept in step by hand: an opcode added
/// to the accepted set and not added here reads as warp-uniform, and its result then names a
/// scalar register with a vector register's number. `convert` did exactly that.
fn forEachOperand(func: *const Function, inst: Inst, ctx: anytype, comptime f: fn (@TypeOf(ctx), Value) void) void {
    switch (func.opcode(inst)) {
        .iconst, .fconst, .global_addr, .barrier => {},
        .unary => |u| f(ctx, u.value),
        .arith => |a| {
            f(ctx, a.lhs);
            f(ctx, a.rhs);
        },
        .arith_imm => |a| f(ctx, a.lhs),
        .convert => |cv| f(ctx, cv.value),
        .icmp => |c| {
            f(ctx, c.lhs);
            f(ctx, c.rhs);
        },
        .select => |s| {
            f(ctx, s.cond);
            f(ctx, s.then);
            f(ctx, s.@"else");
        },
        .load => |l| f(ctx, l.ptr),
        .store => |st| {
            f(ctx, st.value);
            f(ctx, st.ptr);
        },
        .@"if" => |cf| f(ctx, cf.cond),
        else => {},
    }
}

/// Which values are per-lane. Seeded from the lane-varying entry builtins and propagated forward
/// to a fixpoint, because a loop can carry divergence backwards into a block parameter.
///
/// This propagates through OPERANDS only. That is sound ONLY BECAUSE `checkSupported` refuses a
/// branch on a per-lane condition: with no divergent branch there is no control divergence, so a
/// value cannot become per-lane by being defined on one side of a split. ADDING MASK SUPPORT
/// WITHOUT ADDING THE CONTROL RULE HERE WOULD BE A SILENT MISCOMPILE.
///
/// `vulcan-opt`'s `uniform.zig` already answers a similar question and does model control
/// dependence, but it answers a different one: it seeds EVERY load divergent, because it exists to
/// decide barrier legality and a load could differ per thread. That is the right conservatism for a
/// barrier and the wrong one here, where it would put every loaded value in a vector register.
fn analyzeDivergence(allocator: std.mem.Allocator, func: *const Function) Error![]bool {
    const div = try allocator.alloc(bool, func.valueCount());
    errdefer allocator.free(div);
    @memset(div, false);

    for (func.blockParams(@fromBackingInt(0))) |p| {
        const b = gpu.attrs.builtinOf(func, p) orelse continue;
        if (laneVarying(b)) div[@backingInt(p)] = true;
    }

    const Spread = struct {
        div: []bool,
        any: bool = false,
        fn see(self: *@This(), v: Value) void {
            if (self.div[@backingInt(v)]) self.any = true;
        }
    };
    var changed = true;
    while (changed) {
        changed = false;
        for (0..func.blockCount()) |bi| {
            const block: Block = @fromBackingInt(@intCast(bi));
            for (func.blockInsts(block)) |inst| {
                var s: Spread = .{ .div = div };
                forEachOperand(func, inst, &s, Spread.see);
                // An operand that varies per lane makes the result vary too.
                if (func.instResult(inst)) |r| {
                    if (s.any and !div[@backingInt(r)]) {
                        div[@backingInt(r)] = true;
                        changed = true;
                    }
                }
                // An edge argument that varies makes the parameter it feeds vary.
                if (func.opcode(inst) == .@"if") {
                    const cf = func.opcode(inst).@"if";
                    for ([_]ir.function.Jump{ cf.then, cf.@"else" }) |j| {
                        if (spreadToParams(func, div, j)) changed = true;
                    }
                }
            }
            if (func.terminator(block)) |term| switch (term) {
                .jump => |j| if (spreadToParams(func, div, j)) {
                    changed = true;
                },
                .ret => {},
            };
        }
    }
    return div;
}

/// Mark each of `j`'s target parameters divergent when the argument feeding it is. Returns whether
/// anything changed.
fn spreadToParams(func: *const Function, div: []bool, j: ir.function.Jump) bool {
    const args = func.blockArgs(j);
    const params = func.blockParams(j.target);
    var changed = false;
    for (args, 0..) |arg, i| {
        if (i >= params.len) break;
        if (div[@backingInt(arg)] and !div[@backingInt(params[i])]) {
            div[@backingInt(params[i])] = true;
            changed = true;
        }
    }
    return changed;
}

// ===========================================================================
// What this backend accepts.

fn classOf(ctx: *const anyopaque, _: *const Function, v: Value) u16 {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return if (c.divergent[@backingInt(v)]) 1 else 0;
}

fn useKind(_: *const anyopaque, _: *const Function, _: Inst, _: Value) wimmer.UseKind {
    return .must_have_register;
}

/// A `reinterpret` between two 32-bit values moves no bits, so the allocator may put its result
/// and its source in one register and leave the move out. Nothing else here is a pure copy.
fn copySource(_: *const anyopaque, func: *const Function, v: Value) ?Value {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .unary => |u| if (u.op == .reinterpret) u.value else null,
        .convert => |cv| if (isBitPreservingConvert(func, inst, cv)) cv.value else null,
        else => null,
    };
}

/// Whether `ty` is a value a Glacier register holds. One 32-bit word, nothing narrower.
///
/// The core has sub-word loads AND sub-word stores (see `encode.zig`), so what blocks a narrower
/// value here is this lowering and not the hardware: every operation on one would have to truncate
/// to its width, and silently skipping that truncation is a wrong answer rather than a refusal.
/// An f16 is admitted here even though it is half a word, because it never reaches memory. See
/// `isHalf`.
fn isWord(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .bool, .ptr => true,
        .int => |i| i.bits == 32,
        .float => |f| f == .f32 or f == .f16,
        else => false,
    };
}

/// Whether `ty` is an f16.
///
/// A Glacier f16 occupies the LOW HALF of a 32-bit lane: that is where `vfcvt.f.x.v` and
/// `vfncvt.f.f.w` write one and where `vfwmacc.vv` reads one. The upper half is not part of the
/// value, so an f16 cannot travel through a word load or store and `checkSupported` refuses both.
fn isHalf(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .float => |f| f == .f16,
        else => false,
    };
}

fn isFloat(func: *const Function, ty: Type) bool {
    return func.types.type_kind(ty) == .float;
}

fn isSigned(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.signedness == .signed,
        else => false,
    };
}

/// The register value of a float constant: its bit pattern, with a subnormal flushed to zero.
///
/// The lane flushes subnormals on every operation, so a constant that stayed subnormal here would
/// not equal the same number the hardware computes, and a folded path would disagree with an
/// unfolded one near zero.
fn floatBits(func: *const Function, inst: Inst, c: f64) ?u32 {
    const result = func.instResult(inst) orelse return null;
    return switch (func.types.type_kind(func.valueType(result))) {
        .float => |k| switch (k) {
            .f32 => @as(u32, @bitCast(flushToZero(f32, @floatCast(c)))),
            .f16 => @as(u32, @as(u16, @bitCast(flushToZero(f16, @floatCast(c))))),
            else => null,
        },
        else => null,
    };
}

/// `v` with a subnormal replaced by a zero of the same sign. A NaN and an infinity pass through.
fn flushToZero(comptime T: type, v: T) T {
    if (v != 0 and @abs(v) < std.math.floatMin(T)) return std.math.copysign(@as(T, 0.0), v);
    return v;
}

/// What a `convert` costs on Glacier, or null when the core has no instruction for it.
///
/// Glacier decodes three converts and no scalar form of any of them, so each one is a vector
/// instruction even when both sides are warp-uniform. Nothing here WIDENS an f16: the core has no
/// `vfwcvt`, and the only instruction that reads an f16 as an f32 is the widening MAC.
const ConvertKind = enum {
    /// Same width and same kind, so the bits already are the value and a move is the whole
    /// lowering.
    move,
    /// `vfcvt.f.x.v`: a SIGNED 32-bit integer to f16, rounding to nearest even.
    int_to_half,
    /// `vfcvt.rtz.xu.f.v`: an f32 to an UNSIGNED 32-bit integer, truncating toward zero.
    float_to_uint,
    /// `vfncvt.f.f.w`: an f32 to an f16, rounding to nearest even, so an overflow gives infinity
    /// rather than the largest finite value.
    narrow_half,
};

fn convertKind(func: *const Function, inst: Inst, cv: ir.function.Convert) ?ConvertKind {
    const result = func.instResult(inst) orelse return null;
    const from = func.types.type_kind(func.valueType(cv.value));
    const to = func.types.type_kind(func.valueType(result));
    if (from == .int and to == .int) {
        // Everything here is one word wide, so this changes the signedness and nothing else.
        return if (from.int.bits == to.int.bits) .move else null;
    }
    if (from == .int and to == .float) {
        // The instruction reads its source as signed, so an unsigned value above the signed range
        // would come out negative.
        if (from.int.bits != 32 or from.int.signedness != .signed) return null;
        return if (to.float == .f16) .int_to_half else null;
    }
    if (from == .float and to == .int) {
        // The instruction writes an unsigned result, so a signed destination would read a large
        // value as negative.
        if (from.float != .f32 or to.int.bits != 32 or to.int.signedness != .unsigned) return null;
        return .float_to_uint;
    }
    if (from == .float and to == .float) {
        return if (from.float == .f32 and to.float == .f16) .narrow_half else null;
    }
    return null;
}

/// The encoder for a convert the vector unit performs. `move` has none, because it is not an
/// instruction.
fn convertEncoder(kind: ConvertKind) *const fn (VReg, VReg) u32 {
    return switch (kind) {
        .move => unreachable,
        .int_to_half => rv.vfcvt_f_x_v,
        .float_to_uint => rv.vfcvt_rtz_xu_f_v,
        .narrow_half => rv.vfncvt_f_f_w,
    };
}

/// Whether a `convert` only changes how the bits are read, so it needs no instruction. This is
/// what lets the allocator coalesce the result with the source; see `copySource`.
fn isBitPreservingConvert(func: *const Function, inst: Inst, cv: ir.function.Convert) bool {
    return convertKind(func, inst, cv) == .move;
}

/// Whether `inst`'s result is per-lane, which is what decides the emitter's path. Mirrors
/// `Emitter.isVector` for an instruction that has a result.
fn resultDivergent(func: *const Function, inst: Inst, divergent: []const bool) bool {
    const r = func.instResult(inst) orelse return false;
    return divergent[@backingInt(r)];
}

/// Reject every shape this lowering does not handle, before anything is emitted. A refusal here
/// is a clear error; a refusal halfway through emission is a truncated program.
fn checkSupported(func: *const Function, opts: Options, divergent: []const bool) Error!void {
    for (0..func.valueCount()) |vi| {
        const v: Value = @fromBackingInt(@intCast(vi));
        if (!isWord(func, func.valueType(v))) return error.Unsupported;
    }
    for (func.blockParams(@fromBackingInt(0))) |p| {
        const b = gpu.attrs.builtinOf(func, p) orelse return error.Unsupported;
        // A lane-varying builtin comes from `vid.v` rather than from a CSR, so it has no
        // `builtinSource`. See `lanewiseBuiltin`.
        if (!laneVarying(b)) _ = builtinSource(b) orelse return error.Unsupported;
    }
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, i| switch (func.opcode(inst)) {
            // A constant must fit a 32-bit register. An unsigned type reaches 0xFFFFFFFF and a
            // signed one reaches the most negative value, so the window spans both.
            .iconst => |c| if (c < std.math.minInt(i32) or c > std.math.maxInt(u32)) return error.Unsupported,
            // The result type is swept above, so it is already an f32 or an f16 here.
            .fconst => |c| _ = floatBits(func, inst, c) orelse return error.Unsupported,
            // The emitter picks the scalar or the vector table from the result's divergence, so
            // admission has to ask the same table it will use. The integer tables are both
            // complete, but the float ones are not: a divide is scalar only.
            .arith => |a| if (resultDivergent(func, inst, divergent)) {
                _ = vectorBinOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported;
            } else {
                _ = binOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported;
            },
            .arith_imm => |a| {
                // The immediate is an integer, so this opcode never carries a float operand.
                if (isFloat(func, func.valueType(a.lhs))) return error.Unsupported;
                _ = binOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported;
            },
            // A per-lane compare writes a MASK, and a per-lane select reads one. Masks are the
            // next slice of this lowering, so refuse rather than compare lane 0 and call it the
            // answer for the whole warp.
            .icmp => |c| {
                // Glacier has no float compare in any form. An integer compare over float bits
                // answers a different relation, and it is wrong for a negative value and for NaN.
                if (isFloat(func, func.valueType(c.lhs))) return error.Unsupported;
                if (divergent[@backingInt(c.lhs)] or divergent[@backingInt(c.rhs)]) return error.Unsupported;
            },
            .select => |s| if (divergent[@backingInt(s.cond)]) return error.Unsupported,
            .load => |l| {
                if (!isWord(func, func.valueType(l.ptr))) return error.Unsupported;
                const r = func.instResult(inst) orelse return error.Unsupported;
                // An f16 is the low half of a lane, not a word in memory. See `isHalf`.
                if (isHalf(func, func.valueType(r))) return error.Unsupported;
            },
            .store => |st| {
                if (!isWord(func, func.valueType(st.value))) return error.Unsupported;
                if (isHalf(func, func.valueType(st.value))) return error.Unsupported;
            },
            .global_addr => |g| _ = addressOf(opts, func.symbolName(g.symbol)) orelse return error.Unsupported,
            .unary => |u| if (u.op != .reinterpret) return error.Unsupported,
            .convert => |cv| _ = convertKind(func, inst, cv) orelse return error.Unsupported,
            .barrier => |b| switch (b.scope) {
                .workgroup, .subgroup => {},
            },
            // A conditional transfers control, so nothing may follow it in its block, and its
            // block must have no terminator: a terminator there is an edge the allocator would
            // resolve and this emits nothing for.
            .@"if" => |cf| {
                if (i + 1 != insts.len or func.terminator(block) != null) return error.Unsupported;
                // A branch on a per-lane condition is DIVERGENCE: the lanes would need to take
                // both arms under masks. A scalar branch here would send the whole warp whichever
                // way lane 0 went, which is a wrong answer and not a slow one.
                if (divergent[@backingInt(cf.cond)]) return error.Unsupported;
            },
            else => return error.Unsupported,
        };
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| if (r.count != 0) return error.Unsupported,
            .jump => {},
        };
    }
}

/// Where a WARP-UNIFORM hardware builtin comes from, or null when the core cannot answer.
///
/// A warp is the workgroup: it owns a program counter and a register file, and `mhartid` is
/// unique across the whole SoC, which is what a workgroup index is. The lanes inside a warp are
/// the threads, so a thread or lane index differs between the lanes and comes from `vid.v`
/// instead. Those are answered by `lanewiseBuiltin`, not here.
fn builtinSource(b: gpu.builtin.Builtin) ?union(enum) { csr: u12, lanes } {
    return switch (b) {
        .block_id_x => .{ .csr = glacier.Csr.mhartid },
        .grid_dim_x => .{ .csr = glacier.Csr.hart_count },
        .block_dim_x, .subgroup_size => .lanes,
        else => null,
    };
}

/// The low 32 bits of an IR constant, as the signed value a register holds. An unsigned constant
/// above the signed range and the negative value with the same bits are one register value.
fn lowWord(c: i64) i32 {
    return @bitCast(@as(u32, @truncate(@as(u64, @bitCast(c)))));
}

fn addressOf(opts: Options, name: []const u8) ?u32 {
    for (opts.globals) |g| {
        if (std.mem.eql(u8, g.name, name)) return g.address;
    }
    return null;
}

/// The register-form encoder for a binary operation on an operand of type `ty`, or null when
/// Glacier has no instruction for it. Signedness comes from the operand type, so one `shr`
/// becomes a logical or an arithmetic shift depending on it.
fn binOp(func: *const Function, op: ir.function.BinOp, ty: Type) ?*const fn (Reg, Reg, Reg) u32 {
    // Glacier decodes ONE float operation that is not per-lane, and it is a divide of two f32.
    // There is no fadd, no fsub and no non-widening fmul in any width, so everything else is a
    // refusal rather than a sequence. See `encode.zig`.
    switch (func.types.type_kind(ty)) {
        .float => |f| return if (op == .div and f == .f32) glacier.fdiv else null,
        else => {},
    }
    const signed = isSigned(func, ty);
    return switch (op) {
        .add => rv.add,
        .sub => rv.sub,
        .mul => rv.mul,
        .div => if (signed) rv.div else rv.divu,
        .rem => if (signed) rv.rem else rv.remu,
        .mulh => if (signed) rv.mulh else rv.mulhu,
        .bit_and => rv.and_,
        .bit_or => rv.or_,
        .bit_xor => rv.xor_,
        .shl => rv.sll,
        .shr => if (signed) rv.sra else rv.srl,
    };
}

/// The vector-vector encoder for a binary operation, or null when the vector unit has no form of
/// it. The divides are present but NOT worth vectorizing: the integer divider is the one unit the
/// lanes share and it walks them one at a time, so a vector divide costs what the scalar ones
/// would. It is still correct, and emitting it keeps a kernel that divides from being refused.
fn vectorBinOp(func: *const Function, op: ir.function.BinOp, ty: Type) ?*const fn (VReg, VReg, VReg) u32 {
    // There is no per-lane float arithmetic. The widening MAC is the only vector float operation
    // and it is not a binary op, because it accumulates into its own destination. A per-lane
    // divide would have to feed the single scalar divider one lane at a time.
    if (isFloat(func, ty)) return null;
    const signed = isSigned(func, ty);
    return switch (op) {
        .add => rv.vadd_vv,
        .sub => rv.vsub_vv,
        .mul => rv.vmul_vv,
        .div => if (signed) rv.vdiv_vv else rv.vdivu_vv,
        .rem => if (signed) rv.vrem_vv else rv.vremu_vv,
        .mulh => if (signed) rv.vmulh_vv else rv.vmulhu_vv,
        .bit_and => rv.vand_vv,
        .bit_or => rv.vor_vv,
        .bit_xor => rv.vxor_vv,
        .shl => rv.vsll_vv,
        .shr => if (signed) rv.vsra_vv else rv.vsrl_vv,
    };
}

/// The vector-scalar encoder for a binary operation, for the shape where one operand is the same
/// in every lane. Only the forms that exist; the rest broadcast and use the vector-vector form.
fn vectorBinOpVx(func: *const Function, op: ir.function.BinOp, ty: Type) ?*const fn (VReg, VReg, Reg) u32 {
    if (isFloat(func, ty)) return null;
    return switch (op) {
        .add => rv.vadd_vx,
        .mul => rv.vmul_vx,
        .shl => rv.vsll_vx,
        else => null,
    };
}

// ===========================================================================
// Emission.

/// A jump whose target word is not known yet. Every forward reference is a J-type jump, never a
/// branch: a branch reaches 4 KiB and the instruction aperture is 32 KiB, so a branch can fall
/// out of range inside one legal kernel while a jump cannot.
const Fixup = struct { at: usize, target: Block };

const Emitter = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    opts: Options,
    alloc: *const wimmer.Allocation,
    /// How many times each value is read, so a pure instruction nobody reads is not emitted.
    uses: []const u32,
    /// Which values are per-lane, so live in a vector register. See `analyzeDivergence`.
    divergent: []const bool,
    /// Whether the allocation uses the scratch region, so the prologue must load `sp`.
    spills: bool,
    words: std.ArrayList(u32) = .empty,
    block_start: std.ArrayList(u32) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,

    fn deinit(self: *Emitter) void {
        self.words.deinit(self.allocator);
        self.block_start.deinit(self.allocator);
        self.fixups.deinit(self.allocator);
    }

    fn put(self: *Emitter, word: u32) Error!void {
        try self.words.append(self.allocator, word);
    }

    fn putMany(self: *Emitter, ws: []const u32) Error!void {
        try self.words.appendSlice(self.allocator, ws);
    }

    /// Where `v` lives at position `pos`, or null when the allocator placed it nowhere because
    /// nothing reads it.
    fn locAt(self: *const Emitter, v: Value, pos: u32) ?wimmer.Location {
        const segs = self.alloc.segments.get(v) orelse return null;
        var found: ?wimmer.Location = null;
        for (segs) |s| {
            if (s.from > pos) break;
            found = s.loc;
        }
        return found;
    }

    /// The register holding operand `v` at `pos`.
    ///
    /// Every operand here is `must_have_register`, so the allocator reloads a spilled one into a
    /// register before the instruction that reads it. A slot here would mean that contract broke,
    /// so refuse rather than invent a reload the liveness never accounted for.
    fn reg(self: *const Emitter, v: Value, pos: u32) Error!Reg {
        return switch (self.locAt(v, pos) orelse return error.Unsupported) {
            .reg => |r| regOf(r),
            .slot => error.Unsupported,
        };
    }

    fn isDivergent(self: *const Emitter, v: Value) bool {
        return self.divergent[@backingInt(v)];
    }

    /// The vector register holding per-lane value `v` at `pos`.
    fn vreg(self: *const Emitter, v: Value, pos: u32) Error!VReg {
        return switch (self.locAt(v, pos) orelse return error.Unsupported) {
            .reg => |r| @as(VReg, @fromBackingInt(@as(u5, @intCast(r)))),
            .slot => error.Unsupported, // a spilled vector; refused before emission
        };
    }

    /// `v` as a vector operand. A warp-uniform value is broadcast into the vector scratch first,
    /// since one operand of a per-lane operation may well be the same in every lane.
    ///
    /// At most one operand of an instruction can need this: if every operand were uniform the
    /// result would be uniform too and the scalar path would have taken it.
    fn vectorOperand(self: *Emitter, v: Value, pos: u32) Error!VReg {
        if (self.isDivergent(v)) return self.vreg(v, pos);
        try self.put(rv.vmv_v_x(vscratch, try self.reg(v, pos)));
        return vscratch;
    }

    /// Materialize `value` into `dst`, using one or two words.
    fn constant(self: *Emitter, dst: Reg, value: i32) Error!void {
        var pair: [2]u32 = undefined;
        try self.putMany(glacier.loadImmediate(dst, value, &pair));
    }

    fn move(self: *Emitter, dst: Reg, src: Reg) Error!void {
        if (dst != src) try self.put(rv.addi(dst, src, 0));
    }

    fn run(self: *Emitter) Error!void {
        const nblocks = self.func.blockCount();
        try self.block_start.appendNTimes(self.allocator, 0, nblocks);

        var pos: u32 = 0;
        for (0..nblocks) |bi| {
            const block: Block = @fromBackingInt(@intCast(bi));
            self.block_start.items[bi] = @intCast(self.words.items.len);
            if (bi == 0) try self.prologue(block) else try self.edgeMovesInto(block);

            try self.actionsAt(pos); // the block-parameter position
            pos += 1;
            const insts = self.func.blockInsts(block);
            var ended = false;
            for (insts) |inst| {
                try self.actionsAt(pos);
                if (self.func.opcode(inst) == .@"if") {
                    try self.emitIf(inst, pos, bi);
                    ended = true;
                } else {
                    try self.emitInst(inst, pos);
                }
                pos += 1;
            }
            if (!ended) {
                try self.actionsAt(pos);
                try self.emitTerminator(block, bi);
            }
            pos += 1;
        }
        try self.patch();
    }

    /// The entry prologue: every entry parameter is a hardware builtin, so each one is a CSR read
    /// or a core constant written into the register the allocator gave it.
    ///
    /// A kernel that spills loads `sp` here. A kernel that does not leaves it alone: nothing else
    /// reads it, because this target has no calls, and a core built with no scratch region reads
    /// zero from `mscratch` anyway.
    fn prologue(self: *Emitter, block: Block) Error!void {
        if (self.spills) try self.put(glacier.csrr(sp, glacier.Csr.mscratch));
        for (self.func.blockParams(block)) |p| {
            const home = self.locAt(p, 0) orelse continue;
            const b = gpu.attrs.builtinOf(self.func, p) orelse return error.Unsupported;
            if (self.isDivergent(p)) {
                try self.lanewiseBuiltin(b, try self.vreg(p, 0));
                continue;
            }
            const dst: Reg = switch (home) {
                .reg => |r| regOf(r),
                .slot => scratch,
            };
            switch (builtinSource(b) orelse return error.Unsupported) {
                .csr => |c| try self.put(glacier.csrr(dst, c)),
                .lanes => try self.constant(dst, self.opts.profile.lanes()),
            }
            switch (home) {
                .slot => |n| try self.put(rv.sw(dst, sp, try self.slotOffset(n))),
                .reg => {},
            }
        }
    }

    /// A builtin that differs between lanes, into `dst`.
    ///
    /// `vid.v` numbers the lanes the launch mask names and leaves the rest holding whatever the
    /// previous kernel put there. That is safe to build an address from: the vector load and store
    /// units are gated by the same mask, so a dead lane's address is computed and dropped. It is
    /// NOT safe to reduce across, which is why a lane count comes from the profile.
    fn lanewiseBuiltin(self: *Emitter, b: gpu.builtin.Builtin, dst: VReg) Error!void {
        try self.put(rv.vid_v(dst));
        if (b != .global_id_x) return;
        // A global index is this warp's base plus the lane: `mhartid * lanes + vid`. The lane
        // count is a power of two, so the multiply is a shift.
        const lanes = self.opts.profile.lanes();
        try self.put(glacier.csrr(scratch, glacier.Csr.mhartid));
        try self.put(rv.slli(scratch, scratch, @intCast(std.math.log2_int(u8, lanes))));
        try self.put(rv.vmv_v_x(vscratch, scratch));
        try self.put(rv.vadd_vv(dst, dst, vscratch));
    }

    /// The parallel copies on the edges that END at `block`, for the edges the predecessor cannot
    /// host. See `edgeMovesOutOf` for which side hosts which edge.
    fn edgeMovesInto(self: *Emitter, block: Block) Error!void {
        for (self.alloc.edge_moves) |set| {
            if (set.succ != block or set.moves.len == 0) continue;
            if (!endsInIf(self.func, set.pred)) continue;
            if (predCount(self.func, block) != 1) return error.Unsupported;
            try self.emitMoves(set.moves);
        }
    }

    /// The parallel copies on the edges that LEAVE `block`, for the edges this block hosts.
    ///
    /// A block that leaves by a jump hosts its one edge, because the moves belong to that edge
    /// alone. A block that leaves by a conditional cannot: it has one place to put them and two
    /// paths out, so moves there would also run on the sibling path. Those sit at the top of the
    /// successor instead, which is sound only when the successor has one predecessor, and
    /// splitting the critical edges is what makes that true.
    fn edgeMovesOutOf(self: *Emitter, block: Block) Error!void {
        for (self.alloc.edge_moves) |set| {
            if (set.pred != block or set.moves.len == 0) continue;
            try self.emitMoves(set.moves);
        }
    }

    /// Realize one parallel-move set: register moves, spills, reloads, and the slot-to-slot step
    /// the Wimmer resolver already routes through the class scratch.
    fn emitMoves(self: *Emitter, moves: []const wimmer.Move) Error!void {
        for (moves) |m| switch (m.src) {
            .reg => |s| switch (m.dst) {
                .reg => |d| try self.move(regOf(d), regOf(s)),
                .slot => |d| try self.put(rv.sw(regOf(s), sp, try self.slotOffset(d))),
            },
            .slot => |s| switch (m.dst) {
                .reg => |d| try self.put(rv.lw(regOf(d), sp, try self.slotOffset(s))),
                .slot => |d| {
                    try self.put(rv.lw(scratch, sp, try self.slotOffset(s)));
                    try self.put(rv.sw(scratch, sp, try self.slotOffset(d)));
                },
            },
        };
    }

    /// The displacement of spill slot `n` from `sp`. The stack grows DOWN from `mscratch`, which
    /// reads one past the top of the hart's region, so every slot sits below the pointer.
    ///
    /// Refusing when `spills` is clear is not redundant. `spills` decides whether the prologue
    /// loads `sp` at all, so a slot reference without it would address off an uninitialized
    /// register, and that is a silent wrong answer rather than a refusal.
    fn slotOffset(self: *const Emitter, n: u32) Error!i12 {
        if (!self.spills) return error.Unsupported;
        const off: i64 = -(@as(i64, n) + 1) * 4;
        if (off < std.math.minInt(i12)) return error.Unsupported;
        return @intCast(off);
    }

    /// The allocator's own spills, reloads and moves at `pos`.
    fn actionsAt(self: *Emitter, pos: u32) Error!void {
        for (self.alloc.actions) |a| {
            if (a.at != pos) continue;
            try self.emitMoves(&[_]wimmer.Move{.{ .src = a.src, .dst = a.dst, .class = a.class, .value = a.value }});
        }
    }

    /// Whether `inst` operates on lanes rather than on one warp-wide value. A result that varies
    /// per lane decides it; a store has no result, so its address and its value decide instead.
    fn isVector(self: *const Emitter, inst: Inst) bool {
        if (self.func.instResult(inst)) |r| return self.isDivergent(r);
        return switch (self.func.opcode(inst)) {
            .store => |st| self.isDivergent(st.ptr) or self.isDivergent(st.value),
            else => false,
        };
    }

    /// The per-lane form of an instruction. Every op is unmasked, so it writes every lane the
    /// launch mask names.
    fn emitVectorInst(self: *Emitter, inst: Inst, pos: u32) Error!void {
        switch (self.func.opcode(inst)) {
            // A gather and a scatter take a scalar base plus a per-lane byte offset. The offset
            // vector carries the whole address and the base is x0, so no analysis has to prove
            // the lanes are contiguous. A unit-stride access is the faster form when they are,
            // and recognizing that is the next optimization rather than a correctness matter.
            .load => |l| try self.put(rv.vluxei32(
                try self.vreg(self.func.instResult(inst).?, pos),
                .x0,
                try self.vectorOperand(l.ptr, pos),
            )),
            .store => |st| {
                const addr = try self.vectorOperand(st.ptr, pos);
                // Both operands cannot need the scratch: the address is per-lane here whenever
                // the value is not, because a uniform store to a uniform address is not a vector
                // instruction at all.
                const val = if (self.isDivergent(st.value)) try self.vreg(st.value, pos) else blk: {
                    if (addr == vscratch) return error.Unsupported;
                    try self.put(rv.vmv_v_x(vscratch, try self.reg(st.value, pos)));
                    break :blk vscratch;
                };
                try self.put(rv.vsuxei32(val, .x0, addr));
            },
            .unary => |u| try self.put(rv.vmv_v_v(
                try self.vreg(self.func.instResult(inst).?, pos),
                try self.vectorOperand(u.value, pos),
            )),
            .convert => |cv| {
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                const src = try self.vectorOperand(cv.value, pos);
                const kind = convertKind(self.func, inst, cv) orelse return error.Unsupported;
                try self.put(if (kind == .move) rv.vmv_v_v(dst, src) else convertEncoder(kind)(dst, src));
            },
            .arith => |a| {
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                const f = vectorBinOp(self.func, a.op, self.func.valueType(a.lhs)) orelse return error.Unsupported;
                // The scalar-operand form when only the right side is warp-uniform, which is the
                // common shape: a per-lane value combined with a loop invariant.
                if (self.isDivergent(a.lhs) and !self.isDivergent(a.rhs)) {
                    if (vectorBinOpVx(self.func, a.op, self.func.valueType(a.lhs))) |g| {
                        return self.put(g(dst, try self.vreg(a.lhs, pos), try self.reg(a.rhs, pos)));
                    }
                }
                const lhs = try self.vectorOperand(a.lhs, pos);
                const rhs = try self.vectorOperand(a.rhs, pos);
                try self.put(f(dst, lhs, rhs));
            },
            .arith_imm => |a| try self.emitVectorArithImm(a, inst, pos),
            else => return error.Unsupported,
        }
    }

    fn emitVectorArithImm(self: *Emitter, a: ir.function.ArithImm, inst: Inst, pos: u32) Error!void {
        const dst = try self.vreg(self.func.instResult(inst).?, pos);
        const ty = self.func.valueType(a.lhs);
        const lhs = try self.vectorOperand(a.lhs, pos);
        if (a.op == .shl or a.op == .shr) {
            if (a.imm < 0 or a.imm > 31) return error.Unsupported;
            const shamt: u5 = @intCast(a.imm);
            return self.put(switch (a.op) {
                .shl => rv.vsll_vi(dst, lhs, shamt),
                else => if (isSigned(self.func, ty)) rv.vsra_vi(dst, lhs, shamt) else rv.vsrl_vi(dst, lhs, shamt),
            });
        }
        // Everything else takes the constant through a scalar register. Subtraction has no
        // immediate form, so it is an add of the negated constant, same as the scalar path.
        const op: ir.function.BinOp = if (a.op == .sub) .add else a.op;
        const imm: i64 = if (a.op == .sub) -a.imm else a.imm;
        try self.constant(scratch, lowWord(imm));
        // The `.vx` form when the vector unit has one. Only a few ops do, so everything else
        // broadcasts the constant into a vector first: a bitwise AND against a mask and a modulo
        // by a constant are both common per-lane shapes, and refusing them would send a kernel
        // back to scalar for the sake of one instruction.
        if (vectorBinOpVx(self.func, op, ty)) |g| return self.put(g(dst, lhs, scratch));
        const f = vectorBinOp(self.func, op, ty) orelse return error.Unsupported;
        // `lhs` is never the vector scratch here: an `arith_imm` whose result is per-lane has a
        // per-lane operand, so `vectorOperand` returned that operand's own register.
        try self.put(rv.vmv_v_x(vscratch, scratch));
        try self.put(f(dst, lhs, vscratch));
    }

    fn emitInst(self: *Emitter, inst: Inst, pos: u32) Error!void {
        // A pure instruction nothing reads is dropped. The allocator still places its result, so
        // the home lookup below would not catch it: a frontend that emits a constant whose only
        // consumer was removed (an eliminated bounds check leaves the array length behind) would
        // otherwise cost a word per launch for a register nobody reads.
        if (self.func.instResult(inst)) |r| {
            if (self.uses[@backingInt(r)] == 0) return;
        }
        if (self.isVector(inst)) return self.emitVectorInst(inst, pos);
        // A pure instruction nothing reads has no home, so it is dropped. An instruction with no
        // result at all (a store) has no home either, and must NOT be dropped.
        const home: ?wimmer.Location = if (self.func.instResult(inst)) |r|
            (self.locAt(r, pos) orelse return)
        else
            null;
        // A result that lives in a spill slot is computed in the scratch register and stored
        // afterwards, so the opcode below never has to know where its result goes.
        const dst: Reg = switch (home orelse wimmer.Location{ .reg = @backingInt(scratch) }) {
            .reg => |r| regOf(r),
            .slot => scratch,
        };
        switch (self.func.opcode(inst)) {
            .iconst => |c| try self.constant(dst, lowWord(c)),
            .fconst => |c| try self.constant(dst, @bitCast(floatBits(self.func, inst, c) orelse return error.Unsupported)),
            .global_addr => |g| try self.constant(
                dst,
                @bitCast(addressOf(self.opts, self.func.symbolName(g.symbol)) orelse return error.Unsupported),
            ),
            .unary => |u| try self.move(dst, try self.reg(u.value, pos)),
            .convert => |cv| try self.emitConvert(cv, inst, dst, pos),
            .arith => |a| {
                const f = binOp(self.func, a.op, self.func.valueType(a.lhs)) orelse return error.Unsupported;
                try self.put(f(dst, try self.reg(a.lhs, pos), try self.reg(a.rhs, pos)));
            },
            .arith_imm => |a| try self.emitArithImm(a, dst, pos),
            .icmp => |c| try self.emitCompare(c, dst, pos),
            .select => |s| try self.emitSelect(s, dst, pos),
            .load => |l| try self.put(rv.lw(dst, try self.reg(l.ptr, pos), 0)),
            .store => |st| try self.put(rv.sw(try self.reg(st.value, pos), try self.reg(st.ptr, pos), 0)),
            // A barrier over a workgroup or a subgroup emits NOTHING, and that is the answer
            // rather than a shortcut. A workgroup here is one warp, and the lanes of a warp are
            // elements of one instruction stream, so they cannot reach different points in the
            // program and have nothing to wait for. The memory half needs nothing either: one
            // stream, one bus, and in-order completion already order the accesses.
            //
            // Glacier's own `warp_barrier` is deliberately NOT emitted here. It synchronizes every
            // warp of one core, which is wider than this opcode asks for, and a wider barrier is
            // NOT the safe choice: a warp parks until its siblings arrive, so a warp that took a
            // different amount of work waits for warps that had no reason to meet it. The core can
            // time that out and raise a sticky error instead of waiting, but the watchdog budget
            // defaults to zero, which disables it, so a build as generated today waits forever.
            // Either way the barrier does not do what the IR asked. The IR has no scope for one
            // core's warps, so there is nothing to spell it with.
            .barrier => {},
            else => return error.Unsupported,
        }
        if (home) |h| switch (h) {
            .slot => |n| try self.put(rv.sw(dst, sp, try self.slotOffset(n))),
            .reg => {},
        };
    }

    /// A warp-uniform convert. Glacier has no scalar convert of any kind, so the value goes out
    /// to the vector unit and comes back: a splat, the per-lane convert, then a read of lane
    /// zero. Three words rather than one, which still beats refusing the kernel.
    ///
    /// Source and destination are the same vector register here. That is the ordinary shape for
    /// every op in this file, because a Glacier lane is a fixed 32 bits and no vector op moves a
    /// value between lanes.
    fn emitConvert(self: *Emitter, cv: ir.function.Convert, inst: Inst, dst: Reg, pos: u32) Error!void {
        const kind = convertKind(self.func, inst, cv) orelse return error.Unsupported;
        if (kind == .move) return self.move(dst, try self.reg(cv.value, pos));
        try self.put(rv.vmv_v_x(vscratch, try self.reg(cv.value, pos)));
        try self.put(convertEncoder(kind)(vscratch, vscratch));
        try self.put(rv.vmv_x_s(dst, vscratch));
    }

    fn emitArithImm(self: *Emitter, a: ir.function.ArithImm, dst: Reg, pos: u32) Error!void {
        const lhs = try self.reg(a.lhs, pos);
        const ty = self.func.valueType(a.lhs);
        const signed = isSigned(self.func, ty);
        // A shift amount above the register width is not an encodable shift on RV32.
        if (a.op == .shl or a.op == .shr) {
            if (a.imm < 0 or a.imm > 31) return error.Unsupported;
            const shamt: u6 = @intCast(a.imm);
            return self.put(switch (a.op) {
                .shl => rv.slli(dst, lhs, shamt),
                else => if (signed) rv.srai(dst, lhs, shamt) else rv.srli(dst, lhs, shamt),
            });
        }
        // Subtraction has no immediate form, so it is an add of the negated constant. Negating
        // the most negative value overflows, so that one goes the register route.
        const imm: i64 = if (a.op == .sub) -a.imm else a.imm;
        const fits = imm >= -2048 and imm <= 2047;
        if (fits) switch (a.op) {
            .add, .sub => return self.put(rv.addi(dst, lhs, @intCast(imm))),
            .bit_and => return self.put(rv.andi(dst, lhs, @intCast(imm))),
            .bit_or => return self.put(rv.ori(dst, lhs, @intCast(imm))),
            .bit_xor => return self.put(rv.xori(dst, lhs, @intCast(imm))),
            else => {},
        };
        const f = binOp(self.func, a.op, ty) orelse return error.Unsupported;
        try self.constant(scratch, lowWord(a.imm));
        try self.put(f(dst, lhs, scratch));
    }

    /// A comparison into a 0-or-1 register. Each form writes `dst` before it reads it again, so
    /// a destination that aliases a source is safe.
    fn emitCompare(self: *Emitter, c: ir.function.Compare, dst: Reg, pos: u32) Error!void {
        const lhs = try self.reg(c.lhs, pos);
        const rhs = try self.reg(c.rhs, pos);
        const signed = isSigned(self.func, self.func.valueType(c.lhs));
        const less: *const fn (Reg, Reg, Reg) u32 = if (signed) rv.slt else rv.sltu;
        switch (c.op) {
            .eq => {
                try self.put(rv.xor_(dst, lhs, rhs));
                try self.put(rv.sltiu(dst, dst, 1));
            },
            .ne => {
                try self.put(rv.xor_(dst, lhs, rhs));
                try self.put(rv.sltu(dst, .x0, dst));
            },
            .lt => try self.put(less(dst, lhs, rhs)),
            .gt => try self.put(less(dst, rhs, lhs)),
            .ge => {
                try self.put(less(dst, lhs, rhs));
                try self.put(rv.xori(dst, dst, 1));
            },
            .le => {
                try self.put(less(dst, rhs, lhs));
                try self.put(rv.xori(dst, dst, 1));
            },
        }
    }

    /// A value-producing conditional. Glacier has no conditional move, so each arm moves its own
    /// value and the branch picks the arm.
    ///
    /// The branch comes FIRST so no temporary is needed. Every source is read either by the branch
    /// or by the one move that runs, and `dst` is written once after that read, so a destination
    /// that aliases the condition or either source is still correct. A temporary here would have
    /// to be the scratch register, and that is where a spilled result is computed.
    fn emitSelect(self: *Emitter, s: ir.function.Select, dst: Reg, pos: u32) Error!void {
        const cond = try self.reg(s.cond, pos);
        const then = try self.reg(s.then, pos);
        const other = try self.reg(s.@"else", pos);
        // Always four words, even when a move is onto its own register, so the branch offsets do
        // not depend on which registers these turn out to be.
        try self.put(rv.beq(cond, .x0, 12)); // false takes the else arm
        try self.put(rv.addi(dst, then, 0));
        try self.put(rv.jal(.x0, 8)); // over the else arm
        try self.put(rv.addi(dst, other, 0));
    }

    fn emitIf(self: *Emitter, inst: Inst, pos: u32, bi: usize) Error!void {
        const cf = self.func.opcode(inst).@"if";
        const cond = try self.reg(cf.cond, pos);
        // The boolean is 0 or 1, so testing it against x0 is the whole condition. A false
        // condition skips the then jump, so that jump is always emitted even when its target
        // follows: eliding it would make the branch skip the else jump instead.
        try self.put(rv.beq(cond, .x0, 8));
        try self.jumpAlways(cf.then.target);
        try self.jumpTo(cf.@"else".target, bi);
    }

    fn emitTerminator(self: *Emitter, block: Block, bi: usize) Error!void {
        const term = self.func.terminator(block) orelse {
            // A neutralized unreachable block has no terminator. Retiring is the safe word to
            // leave behind: nothing branches here, and EBREAK cannot run on into the next block.
            return self.put(glacier.ebreak);
        };
        switch (term) {
            .ret => try self.put(glacier.ebreak),
            .jump => |j| {
                try self.edgeMovesOutOf(block);
                try self.jumpTo(j.target, bi);
            },
        }
    }

    /// A jump to `target`, elided when the target is the block emitted next.
    fn jumpTo(self: *Emitter, target: Block, bi: usize) Error!void {
        if (@backingInt(target) == bi + 1) return;
        try self.jumpAlways(target);
    }

    fn jumpAlways(self: *Emitter, target: Block) Error!void {
        try self.fixups.append(self.allocator, .{ .at = self.words.items.len, .target = target });
        try self.put(0);
    }

    fn patch(self: *Emitter) Error!void {
        for (self.fixups.items) |f| {
            const to: i64 = @as(i64, self.block_start.items[@backingInt(f.target)]) * 4;
            const from: i64 = @as(i64, @intCast(f.at)) * 4;
            const delta = to - from;
            if (delta < std.math.minInt(i21) or delta > std.math.maxInt(i21)) return error.Unsupported;
            self.words.items[f.at] = rv.jal(.x0, @intCast(delta));
        }
    }
};

/// The register a Wimmer class-relative index names. The numbering is the register's own, so
/// index n is x_n.
fn regOf(index: u16) Reg {
    return @fromBackingInt(@as(u5, @intCast(index)));
}

/// Whether `block` leaves by a conditional rather than by its terminator. A conditional transfers
/// control itself, so the terminator after one is never reached and never emitted.
fn endsInIf(func: *const Function, block: Block) bool {
    const insts = func.blockInsts(block);
    return insts.len != 0 and func.opcode(insts[insts.len - 1]) == .@"if";
}

fn predCount(func: *const Function, block: Block) usize {
    var n: usize = 0;
    for (0..func.blockCount()) |bi| {
        const pred: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(pred);
        var counted = false;
        if (endsInIf(func, pred)) {
            const cf = func.opcode(insts[insts.len - 1]).@"if";
            if (cf.then.target == block or cf.@"else".target == block) n += 1;
            counted = true;
        }
        if (counted) continue;
        if (func.terminator(pred)) |term| switch (term) {
            .jump => |j| if (j.target == block) {
                n += 1;
            },
            .ret => {},
        };
    }
    return n;
}
