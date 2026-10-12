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
//! is its bit pattern, the three converts the core has are selected, an f32 divide becomes the one
//! scalar divide. The widening MAC covers all other float arithmetic, in three shapes: the MAC
//! itself, a product with a zeroed destination, and an addition against a factor of 1.0. See
//! `findMacs`. A warp-uniform MAC runs on the vector unit as well, since that is the only float
//! multiplier the core has, and comes back out of lane zero. Everything else is refused: no float
//! compare, no per-lane divide, and no float arithmetic that is not over f16 factors or a constant
//! an f16 holds exactly. An f16 never reaches memory, since it is only the low
//! half of a lane.
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

/// The scratch pool the widening MAC draws from, which is the only thing that needs more than one
/// at a time. Three is the most one MAC can want: a warp-uniform MAC accumulates in one and
/// broadcasts both operands into the others, and a per-lane MAC can broadcast two warp-uniform
/// operands and still have to stage a factor out of its own destination. See `Emitter.mac`.
const vscratch2: VReg = .v30;
const vscratch3: VReg = .v29;
const vscratch_pool: [3]VReg = .{ vscratch, vscratch2, vscratch3 };

/// Hands out `vscratch_pool` in order, so two users in one instruction cannot pick the same one.
const Pool = struct {
    next: usize = 0,

    /// The next free scratch. Running out is a lowering bug rather than an unsupported kernel,
    /// since no shape needs four, but refusing beats naming a register another value is in.
    fn take(self: *Pool) Error!VReg {
        if (self.next >= vscratch_pool.len) return error.Unsupported;
        defer self.next += 1;
        return vscratch_pool[self.next];
    }
};

/// v1 to v28. `v0` is reserved because it is the only register that PREDICATES: a mask has to be
/// moved into it to be applied. Nothing here predicates yet, so reserving it costs one register
/// and buys the guarantee that no value can be sitting in it when masks arrive.
const allocatable_v: [28]u16 = blk: {
    var regs: [28]u16 = undefined;
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
        // A SPIR-V module constant lands at the top of the entry block, so a kernel that writes a
        // table of them holds every one at once and spills the lot. A warp has little scratch, so
        // that is a refusal rather than slow code.
        _ = try opt.sink.run(allocator, &work, &analyses);
    }
    try ir.critical_edge.splitCriticalEdges(allocator, &work);
    const reachable = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable);

    // If-convert the divergent diamonds, then analyse again: converting one turns a join parameter
    // into a select, which can make an enclosing condition per-lane that operand propagation had
    // called uniform, and that can expose another diamond. So it iterates to a fixpoint.
    while (true) {
        const d = try analyzeDivergence(allocator, &work);
        defer allocator.free(d);
        if (!try ifConvert(allocator, &work, d)) break;
    }
    const reachable_after = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable_after);

    // Put every one-word-per-lane address into the shape `findStrides` reads, then clear out the
    // arithmetic that is left over. The divergence analysis runs again below, because the rewrite
    // adds values and moves work from the vector side to the scalar side.
    {
        const d = try analyzeDivergence(allocator, &work);
        defer allocator.free(d);
        if (try hoistLaneAddresses(allocator, &work, d)) {
            var analyses: opt.pass.Analyses = .{ .allocator = allocator, .func = &work };
            defer analyses.deinit();
            _ = try opt.dce.run(allocator, &work, &analyses);
        }
    }

    const divergent = try analyzeDivergence(allocator, &work);
    defer allocator.free(divergent);

    const uses = try allocator.alloc(u32, work.valueCount());
    defer allocator.free(uses);
    opt.dce.countUses(&work, uses);

    var macs = try findMacs(allocator, &work, uses, divergent);
    defer macs.deinit(allocator);

    var strides = try findStrides(allocator, &work, uses, divergent);
    defer strides.deinit(allocator);

    var ctx: Ctx = .{ .divergent = divergent, .macs = &macs, .strides = &strides };

    try checkSupported(&work, opts, divergent, &macs);

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
        .fusedOperands = fusedOperands,
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
        .macs = &macs,
        .strides = &strides,
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
const Ctx = struct { divergent: []const bool, macs: *const Macs, strides: *const Strides };

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
        .atomic_rmw => |a| {
            f(ctx, a.ptr);
            f(ctx, a.value);
            if (a.compare) |c| f(ctx, c);
        },
        .store => |st| {
            f(ctx, st.value);
            f(ctx, st.ptr);
            if (st.mask) |m| f(ctx, m);
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
///
/// A MAC is reported too, which is not a copy. It accumulates into its own destination, so the
/// result and the accumulator want one register, and this hint is how the allocator is asked for
/// that. A different register is still correct and costs one `vmv.v.v`.
fn copySource(ctx: *const anyopaque, func: *const Function, v: Value) ?Value {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    const inst = func.definingInst(v) orelse return null;
    // A MAC reports its accumulator, or nothing when it has none and zeroes its destination
    // instead. Either way the switch below must not read the instruction as a plain arithmetic op.
    const src = if (c.macs.at(inst)) |m| (m.acc orelse return null) else switch (func.opcode(inst)) {
        .unary => |u| if (u.op == .reinterpret) u.value else return null,
        .convert => |cv| if (isBitPreservingConvert(func, inst, cv)) cv.value else return null,
        else => return null,
    };
    // A MAC can report an accumulator in the other register file, which happens when the result
    // is per-lane and the accumulator is not. That needs no guard here: `wimmer.buildIntervals`
    // drops a hint whose source is in another class, because the two share no register pool.
    return src;
}

/// What a MAC's machine instruction really reads, which is not what the IR says.
///
/// An absorbed multiply or convert emits nothing, so the IR's operand walk is wrong in both
/// directions: the f16 factors appear to die at the converts that no longer exist, and the
/// products appear to be read by a multiply that no longer exists. Left unreported, the allocator
/// hands a factor's register to a value born between the convert and the MAC, and the MAC
/// multiplies whatever landed there. There is no diagnostic for that; it is a wrong frame.
fn fusedOperands(ctx: *const anyopaque, func: *const Function, inst: Inst, out: *[wimmer.max_fused_operands]Value) ?u8 {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    // A unit-stride access reads a SCALAR base and not the per-lane address the IR names, and the
    // arithmetic that built that address is not emitted at all. Both halves have to be reported:
    // the base so it is live at the access, and nothing for the absorbed arithmetic.
    if (c.strides.isFolded(inst)) return 0;
    if (c.strides.at(inst)) |base| {
        out[0] = base;
        var n: u8 = 1;
        switch (func.opcode(inst)) {
            .store => |st| {
                out[n] = st.value;
                n += 1;
                if (st.mask) |m| {
                    out[n] = m;
                    n += 1;
                }
            },
            else => {},
        }
        return n;
    }
    // An absorbed instruction reads nothing, because it is not emitted. Reporting that is worth
    // real code: a vertex pass of 1840 words came out at 1582 once the suppressed multiplies and
    // converts stopped being counted as readers. It is not a register-pressure win, since the
    // factors now live LONGER by as much as the products live less.
    if (c.macs.isFolded(inst)) return 0;
    const m = c.macs.at(inst) orelse return null;
    var n: u8 = 0;
    if (m.acc) |acc| {
        out[n] = acc;
        n += 1;
    }
    for ([_]Factor{ m.a, m.b }) |f| switch (f) {
        // A constant is materialized at the MAC and is no one's value.
        .constant => {},
        .value => |v| {
            out[n] = v;
            n += 1;
        },
    };
    return n;
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
    /// `vfwcvt.f.f.v`: an f16 to an f32. Exact in every case, including a subnormal, so there is
    /// no rounding to describe.
    widen_half,
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
        if (from.float == .f32 and to.float == .f16) return .narrow_half;
        if (from.float == .f16 and to.float == .f32) return .widen_half;
        return null;
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
        .widen_half => rv.vfwcvt_f_f_v,
    };
}

/// Whether a `convert` only changes how the bits are read, so it needs no instruction. This is
/// what lets the allocator coalesce the result with the source; see `copySource`.
fn isBitPreservingConvert(func: *const Function, inst: Inst, cv: ir.function.Convert) bool {
    return convertKind(func, inst, cv) == .move;
}

// ===========================================================================
// If-conversion: a divergent branch becomes straight-line code with selects.

/// Rewrite every divergent if-then-else diamond into one block with selects at the join, and
/// report whether anything changed.
///
/// WHY THIS AND NOT A CONTROL-DEPENDENCE RULE. `analyzeDivergence` propagates through operands
/// only, and the case it gets wrong once lanes can take different sides is a join parameter whose
/// two incoming values are both warp-uniform: operand propagation calls the parameter uniform and
/// the lanes disagree about it. Adding a rule for that means the analysis and the emitter have to
/// agree about control dependence forever, and my own warning said getting it wrong is a silent
/// miscompile.
///
/// Converting instead makes the hazard structural. The join parameter becomes a `select` whose
/// CONDITION is the per-lane value, so the divergence arrives as an operand and the analysis that
/// already exists is already right. Nothing is control-dependent on anything, because no branch is
/// left.
///
/// And the refusal stays. `checkSupported` still rejects a divergent `if`, so a shape this cannot
/// transform is refused rather than mis-emitted: the safety comes from never lifting the refusal
/// rather than from a rule that has to be kept in step.
///
/// Both arms then execute for every lane, which is why only PURE arms convert. An instruction with
/// a side effect would take effect in lanes that did not choose it, and the answer for that is a
/// masked form of the instruction rather than this.
fn ifConvert(allocator: std.mem.Allocator, func: *Function, divergent: []const bool) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const a: Block = @fromBackingInt(@intCast(bi));
        const found = candidate(func, a, divergent) orelse continue;
        try convert(allocator, func, a, found);
        changed = true;
    }
    return changed;
}

/// One convertible diamond: the branch, its two arms and the join they share.
const Diamond = struct {
    cond: Value,
    then_block: Block,
    else_block: Block,
    join: Block,
    then_args: ir.function.ValueList,
    else_args: ir.function.ValueList,
};

/// Whether the `if` ending `a` is a diamond this can convert. Every condition here is a reason it
/// would otherwise be wrong, not a simplification of convenience.
fn candidate(func: *const Function, a: Block, divergent: []const bool) ?Diamond {
    const insts = func.blockInsts(a);
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return null,
    };
    // A uniform branch is left alone: it costs one jump and runs one side, where converting it
    // would run both.
    if (!divergent[@backingInt(cf.cond)]) return null;
    const t = cf.then.target;
    const e = cf.@"else".target;
    if (t == e) return null;
    // An arm with another predecessor is reached from somewhere this branch does not control, so
    // its instructions cannot move into `a`. Critical-edge splitting runs first and gives every arm
    // its own predecessor, so this holds structurally today and stands against a change of order.
    if (predCount(func, t) != 1 or predCount(func, e) != 1) return null;
    // An arm taking parameters receives them from the branch, and moving its body out would leave
    // them undefined.
    if (func.blockParams(t).len != 0 or func.blockParams(e).len != 0) return null;
    // Both arms must leave by a plain jump to one join, and nothing may follow the branch.
    if (func.blockArgs(cf.then).len != 0 or func.blockArgs(cf.@"else").len != 0) return null;
    const t_jump = soleJump(func, t) orelse return null;
    const e_jump = soleJump(func, e) orelse return null;
    if (t_jump.target != e_jump.target) return null;
    const join = t_jump.target;
    if (join == t or join == e or join == a) return null;
    // The entry block is reachable without a predecessor, so it cannot be absorbed into another.
    if (@backingInt(join) == 0) return null;
    // A join reached from anywhere else takes parameters from a path this does not control, so
    // they cannot all become selects.
    if (predCount(func, join) != 2) return null;
    if (func.blockArgs(t_jump).len != func.blockParams(join).len) return null;
    if (func.blockArgs(e_jump).len != func.blockParams(join).len) return null;
    // Both arms run for every lane after this, so a side effect in one would reach lanes that did
    // not choose it.
    for ([_]Block{ t, e }) |arm| {
        for (func.blockInsts(arm)) |inst| if (!convertible(func.opcode(inst))) return null;
    }
    // A store in the ELSE arm runs where the condition is FALSE, so it needs the complement as a
    // real value. The only exact complement here is an inverted integer compare: `!(a < b)` is
    // `a >= b` over a total order, and is NOT over floats, where a NaN leaves both false. A
    // condition this cannot invert means the else arm keeps its store and the diamond is refused.
    if (hasStore(func, e) and invertedCompare(func, cf.cond) == null) return null;
    return .{
        .cond = cf.cond,
        .then_block = t,
        .else_block = e,
        .join = join,
        .then_args = t_jump.args,
        .else_args = e_jump.args,
    };
}

/// The jump leaving `block`, when that is all it does by way of control flow.
fn soleJump(func: *const Function, block: Block) ?ir.function.Jump {
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .@"if") return null;
    }
    return switch (func.terminator(block) orelse return null) {
        .jump => |j| j,
        .ret => null,
    };
}

fn hasStore(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .store) return true;
    }
    return false;
}

/// The compare that answers the opposite of `cond`, when `cond` is an integer compare. Null for
/// anything else, including a FLOAT compare: an unordered pair leaves `lt` and `ge` both false, so
/// flipping the operator there changes the answer rather than negating it.
fn invertedCompare(func: *const Function, cond: Value) ?ir.function.Compare {
    const inst = func.definingInst(cond) orelse return null;
    const c = switch (func.opcode(inst)) {
        .icmp => |x| x,
        else => return null,
    };
    if (isFloat(func, func.valueType(c.lhs))) return null;
    return .{
        .op = switch (c.op) {
            .eq => .ne,
            .ne => .eq,
            .lt => .ge,
            .le => .gt,
            .gt => .le,
            .ge => .lt,
        },
        .lhs = c.lhs,
        .rhs = c.rhs,
    };
}

/// Whether an instruction may run for lanes that did not choose its arm. Anything that writes
/// memory, or reads an address a lane never meant to compute, may not.
fn convertible(code: ir.function.Opcode) bool {
    return switch (code) {
        .iconst, .fconst, .global_addr, .arith, .arith_imm, .icmp, .select, .convert, .unary => true,
        // A store converts because the core can predicate one. An inactive lane makes no bus
        // access at all, so a store spliced out of an arm writes exactly the lanes that chose it.
        // A store that already carries a mask converts too: the two masks are ANDed, which is a
        // mask op and exact at any lane count. See `maskBinOp`.
        .store => true,
        else => false,
    };
}

fn convert(allocator: std.mem.Allocator, func: *Function, a: Block, d: Diamond) Error!void {
    // Drop the branch first, so the arms' instructions land where it was.
    const a_insts = func.blockInstsMut(a);
    _ = a_insts.pop();

    // The complement of the condition, materialized ONLY when the else arm stores. It goes in
    // before either body so it dominates every store it masks.
    const else_mask: ?Value = if (hasStore(func, d.else_block)) blk: {
        const inverted = invertedCompare(func, d.cond).?;
        break :blk try func.appendInst(a, func.valueType(d.cond), .{ .icmp = inverted });
    } else null;

    for ([_]Block{ d.then_block, d.else_block }) |arm| {
        const mask = if (arm == d.else_block) else_mask else d.cond;
        const body = func.blockInstsMut(arm);
        const moved = try allocator.dupe(Inst, body.items);
        defer allocator.free(moved);
        body.clearRetainingCapacity();
        // Both arms run for every lane now, so a store that was under the branch becomes a store
        // under the mask. The lanes the mask excludes make no access at all.
        //
        // A store that arrives already masked came from a diamond this converted earlier, and it
        // has to run where BOTH conditions hold, so the two masks are ANDed. The AND is appended
        // before the store it feeds, which is why the body moves one instruction at a time rather
        // than in one slice.
        for (moved) |inst| {
            const carried: ?Value = switch (func.opcode(inst)) {
                .store => |st| st.mask,
                else => null,
            };
            if (carried) |inner| {
                const both = try func.appendInst(a, func.valueType(inner), .{ .arith = .{
                    .op = .bit_and,
                    .lhs = inner,
                    .rhs = mask.?,
                } });
                try func.blockInstsMut(a).append(allocator, inst);
                func.opcodeMut(inst).store.mask = both;
                continue;
            }
            try func.blockInstsMut(a).append(allocator, inst);
            switch (func.opcodeMut(inst).*) {
                .store => |*st| st.mask = mask,
                else => {},
            }
        }
    }

    // Each join parameter becomes a select between what the two arms would have passed it.
    const count = func.blockParams(d.join).len;
    var args: std.ArrayList(Value) = .empty;
    defer args.deinit(allocator);
    for (0..count) |i| {
        const then_v = func.valueList(d.then_args)[i];
        const else_v = func.valueList(d.else_args)[i];
        const ty = func.valueType(func.blockParams(d.join)[i]);
        try args.append(allocator, try func.appendInst(a, ty, .{ .select = .{
            .cond = d.cond,
            .then = then_v,
            .@"else" = else_v,
        } }));
    }
    for (0..count) |i| {
        func.replaceAllUses(func.blockParams(d.join)[i], args.items[i]);
    }
    try func.setBlockParams(d.join, &.{});

    // Absorb the join as well. Both arms came from `a` and nothing else reaches the join, so `a`
    // dominates it and taking its body leaves no jump anywhere in the diamond. Its terminator
    // comes over as it is, which is null when the join itself ends in an `if`: that branch moves
    // across with the body and the next round of this pass sees it.
    const join_body = func.blockInstsMut(d.join);
    try func.blockInstsMut(a).appendSlice(allocator, join_body.items);
    join_body.clearRetainingCapacity();
    func.terminatorPtr(a).* = func.terminatorPtr(d.join).*;

    // The arms and the join are now unreachable; `neutralizeUnreachable` takes them out of the
    // picture.
    for ([_]Block{ d.then_block, d.else_block, d.join }) |dead| {
        func.setTerminator(dead, .{ .ret = ir.function.Ret.none() });
    }
}
// ===========================================================================
// Hoisting the lane term out of an address.

/// An address taken apart as a warp-uniform sum plus a multiple of the lane index.
const Affine = struct {
    /// A per-lane address with more independent terms than this is refused rather than grown. Six
    /// covers a base, a row product, a column and a constant with room to spare.
    const max_terms = 6;

    const Term = struct { value: Value, scale: i64 };

    terms: [max_terms]Term = undefined,
    count: usize = 0,
    constant: i64 = 0,
    /// Bytes between the address of one lane and the next.
    stride: i64 = 0,
    /// The lane index itself, so the rewrite can scale it again.
    lane: ?Value = null,

    fn push(self: *Affine, value: Value, scale: i64) bool {
        if (self.count == max_terms) return false;
        self.terms[self.count] = .{ .value = value, .scale = scale };
        self.count += 1;
        return true;
    }
};

/// Take `v` apart into `out` as `uniform terms + constant + lane * stride`, scaling everything by
/// `scale`. False when it is not that shape.
///
/// Arithmetic here WRAPS, like the address arithmetic it describes. A term scaled past 32 bits
/// addresses the same word either way, so wrapping is the answer rather than a refusal.
fn peelAffine(func: *const Function, divergent: []const bool, v: Value, scale: i64, out: *Affine) bool {
    if (!divergent[@backingInt(v)]) {
        // A constant goes into the constant rather than becoming a term to multiply at run time.
        if (constantOf(func, v)) |c| {
            out.constant +%= c *% scale;
            return true;
        }
        return out.push(v, scale);
    }
    if (gpu.attrs.builtinOf(func, v)) |b| {
        if (b != .thread_id_x and b != .lane_id) return false;
        if (out.lane) |already| {
            if (already != v) return false;
        }
        out.lane = v;
        out.stride +%= scale;
        return true;
    }
    const inst = func.definingInst(v) orelse return false;
    switch (func.opcode(inst)) {
        // The scale arrives as a CONSTANT OPERAND and not as an immediate. `OpAccessChain` emits
        // `mul(index, iconst)` for the element size, so reading only the immediate form meant no
        // kernel that reached the IR through SPIR-V was ever recognized, which is every kernel
        // written in Zig.
        .arith => |a| switch (a.op) {
            .add => {
                if (!peelAffine(func, divergent, a.lhs, scale, out)) return false;
                return peelAffine(func, divergent, a.rhs, scale, out);
            },
            .mul => {
                if (constantOf(func, a.rhs)) |k| return peelAffine(func, divergent, a.lhs, scale *% k, out);
                if (constantOf(func, a.lhs)) |k| return peelAffine(func, divergent, a.rhs, scale *% k, out);
                return false;
            },
            .shl => {
                const k = constantOf(func, a.rhs) orelse return false;
                if (k < 0 or k > 31) return false;
                return peelAffine(func, divergent, a.lhs, scale *% (@as(i64, 1) << @intCast(k)), out);
            },
            else => return false,
        },
        // An index is coerced to the offset type on its way into an access chain, and neither
        // spelling of that coercion moves a bit of the address. `reinterpret` is the one a SPIR-V
        // module actually produces, for the unsigned index of a signed offset.
        .unary => |u| {
            if (u.op != .reinterpret) return false;
            return peelAffine(func, divergent, u.value, scale, out);
        },
        .convert => |cv| {
            if (!isBitPreservingConvert(func, inst, cv)) return false;
            return peelAffine(func, divergent, cv.value, scale, out);
        },
        .arith_imm => |a| switch (a.op) {
            .add => {
                out.constant +%= a.imm *% scale;
                return peelAffine(func, divergent, a.lhs, scale, out);
            },
            .shl => {
                if (a.imm < 0 or a.imm > 31) return false;
                const factor = @as(i64, 1) << @intCast(a.imm);
                return peelAffine(func, divergent, a.lhs, scale *% factor, out);
            },
            .mul => return peelAffine(func, divergent, a.lhs, scale *% a.imm, out),
            else => return false,
        },
        else => return false,
    }
}

/// Rewrite every per-lane address that steps ONE WORD per lane into `uniform + lane * 4`, and
/// report whether anything changed.
///
/// An index by lane reaches the IR with the lane term buried: `w[base + y * row + x + lane]`
/// scales the whole index at once, so the address is `ptr + (uniform + lane) * 4` and the lane
/// term is inside the shift. `findStrides` reads a shape and cannot see through that, so this
/// turns the address into the shape it reads, which also moves the uniform arithmetic from the
/// vector side to the scalar side. The old address instructions are left for `dce`.
///
/// Nothing is rewritten unless the stride comes out at exactly one word, so this never fires
/// speculatively: the shape it produces is the shape the unit-stride access wants.
fn hoistLaneAddresses(allocator: std.mem.Allocator, func: *Function, divergent: []const bool) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        // One address can feed several accesses, a read-modify-write of a pixel for instance. The
        // first one rewrites it and the rest reuse that, which holds because they all sit in this
        // block after the arithmetic that was put in for the first.
        var done: std.AutoHashMapUnmanaged(Value, Value) = .empty;
        defer done.deinit(allocator);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const before = switch (func.opcode(inst)) {
                .load => |l| l.ptr,
                .store => |st| st.ptr,
                else => {
                    try func.blockInstsMut(block).append(allocator, inst);
                    continue;
                },
            };
            const rebuilt = done.get(before) orelse try hoistOne(func, block, divergent, inst);
            try func.blockInstsMut(block).append(allocator, inst);
            if (rebuilt) |ptr| {
                switch (func.opcodeMut(inst).*) {
                    .load => |*l| l.ptr = ptr,
                    .store => |*st| st.ptr = ptr,
                    else => unreachable,
                }
                try done.put(allocator, before, ptr);
                changed = true;
            }
        }
    }
    return changed;
}

/// The rewritten address for `inst`, appended to `block`, or null when it is left alone. The
/// caller appends `inst` itself afterwards, which is what puts the new arithmetic in front of it.
fn hoistOne(func: *Function, block: Block, divergent: []const bool, inst: Inst) Error!?Value {
    const ptr = switch (func.opcode(inst)) {
        .load => |l| l.ptr,
        .store => |st| st.ptr,
        else => return null,
    };
    if (!divergent[@backingInt(ptr)]) return null;
    // Already the shape `findStrides` reads, so rewriting it would only churn.
    if (laneRun(func, ptr) != null) return null;
    var aff: Affine = .{};
    if (!peelAffine(func, divergent, ptr, 1, &aff)) return null;
    if (aff.stride != 4) return null;
    const lane = aff.lane orelse return null;

    // One term carries the address and the rest are offsets into it. Two pointers in one sum is
    // not an address this builds, and neither is a scaled one.
    var carrier: ?usize = null;
    for (aff.terms[0..aff.count], 0..) |t, i| {
        if (func.types.type_kind(func.valueType(t.value)) != .ptr) continue;
        if (carrier != null) return null;
        if (t.scale != 1) return null;
        carrier = i;
    }
    const ci = carrier orelse return null;
    const ptr_ty = func.valueType(aff.terms[ci].value);

    var sum = aff.terms[ci].value;
    for (aff.terms[0..aff.count], 0..) |t, i| {
        if (i == ci) continue;
        const scaled = if (t.scale == 1) t.value else try func.appendInst(
            block,
            func.valueType(t.value),
            .{ .arith_imm = .{ .op = .mul, .lhs = t.value, .imm = t.scale } },
        );
        sum = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = scaled } });
    }
    if (aff.constant != 0) {
        sum = try func.appendInst(block, ptr_ty, .{ .arith_imm = .{ .op = .add, .lhs = sum, .imm = aff.constant } });
    }
    const offset = try func.appendInst(
        block,
        func.valueType(lane),
        .{ .arith_imm = .{ .op = .shl, .lhs = lane, .imm = 2 } },
    );
    return try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = sum, .rhs = offset } });
}

// ===========================================================================
// Unit-stride accesses.

/// Which per-lane accesses read or write a CONTIGUOUS run of words, and the scalar address the run
/// starts at.
///
/// Every per-lane access is a gather or a scatter otherwise: an address per lane, which costs the
/// address vector to build and makes the load-store unit visit one word at a time. A run of
/// consecutive words is the common case and the hardware has one instruction for it, so this is
/// where that case is recognized.
///
/// `folded` marks the arithmetic the access absorbs. The emitter writes nothing for it and
/// `fusedOperands` tells the allocator it reads nothing, exactly as for an absorbed multiply.
const Strides = struct {
    base: []const ?Value,
    folded: []const bool,

    fn deinit(self: *Strides, allocator: std.mem.Allocator) void {
        allocator.free(self.base);
        allocator.free(self.folded);
    }

    fn at(self: *const Strides, inst: Inst) ?Value {
        return self.base[@backingInt(inst)];
    }

    fn isFolded(self: *const Strides, inst: Inst) bool {
        return self.folded[@backingInt(inst)];
    }
};

/// Find every unit-stride access. The caller owns the result.
fn findStrides(
    allocator: std.mem.Allocator,
    func: *const Function,
    uses: []const u32,
    divergent: []const bool,
) Error!Strides {
    const base = try allocator.alloc(?Value, func.instCount());
    errdefer allocator.free(base);
    @memset(base, null);
    const folded = try allocator.alloc(bool, func.instCount());
    errdefer allocator.free(folded);
    @memset(folded, false);

    // How many unit-stride accesses read each address. The arithmetic can only be absorbed when
    // that accounts for EVERY reader: an absorbed instruction is not emitted, so a reader left
    // over would find the register holding nothing.
    const recognized = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(recognized);
    @memset(recognized, 0);

    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const ptr = switch (func.opcode(inst)) {
                .load => |l| l.ptr,
                .store => |st| st.ptr,
                else => continue,
            };
            if (!divergent[@backingInt(ptr)]) continue;
            const run = laneRun(func, ptr) orelse continue;
            base[@backingInt(inst)] = run.base;
            recognized[@backingInt(ptr)] += 1;
        }
    }

    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const result = func.instResult(inst) orelse continue;
            if (recognized[@backingInt(result)] != uses[@backingInt(result)]) continue;
            const run = laneRun(func, result) orelse continue;
            folded[@backingInt(inst)] = true;
            // The scale can only go with it when the add above was its one reader.
            if (uses[@backingInt(run.offset)] != 1) continue;
            folded[@backingInt(func.definingInst(run.offset).?)] = true;
        }
    }
    return .{ .base = base, .folded = folded };
}

/// `base + lane * 4` taken apart, which is the address an index by lane id makes.
const LaneRun = struct {
    /// The warp-uniform start of the run, which the access names as a scalar register.
    base: Value,
    /// The per-lane byte offset, `lane * 4`. Absorbed with the add above it.
    offset: Value,
};

/// Whether `ptr` addresses consecutive words, one per lane.
///
/// `global_id_x` is deliberately NOT accepted even though its lanes are consecutive too. Its value
/// is `mhartid * lanes + vid`, so the run would start at the address PLUS this warp's own offset,
/// and that sum is not a value the function holds: naming it would mean emitting arithmetic from
/// here rather than recognizing something already there.
fn laneRun(func: *const Function, ptr: Value) ?LaneRun {
    const inst = func.definingInst(ptr) orelse return null;
    const a = switch (func.opcode(inst)) {
        .arith => |x| x,
        else => return null,
    };
    if (a.op != .add) return null;
    for ([_][2]Value{ .{ a.lhs, a.rhs }, .{ a.rhs, a.lhs } }) |pair| {
        if (!isLaneWordOffset(func, pair[1])) continue;
        return .{ .base = pair[0], .offset = pair[1] };
    }
    return null;
}

/// The constant behind `v`, when it is one.
fn constantOf(func: *const Function, v: Value) ?i64 {
    const inst = func.definingInst(v) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |c| c,
        else => null,
    };
}

/// Whether `v` is the lane index scaled to a word offset, as a shift by two or a multiply by four.
fn isLaneWordOffset(func: *const Function, v: Value) bool {
    const inst = func.definingInst(v) orelse return false;
    const a = switch (func.opcode(inst)) {
        .arith_imm => |x| x,
        else => return false,
    };
    const scaled = (a.op == .shl and a.imm == 2) or (a.op == .mul and a.imm == 4);
    if (!scaled) return false;
    const b = gpu.attrs.builtinOf(func, a.lhs) orelse return false;
    return b == .thread_id_x or b == .lane_id;
}

// ===========================================================================
// The widening multiply-accumulate.

/// One factor of a MAC. The core multiplies f16 lanes, so a factor is an f16 value in a vector
/// register or a constant an f16 holds exactly.
///
/// A constant factor is not a corner case. `@as(f32, a) * @as(f32, b)` with a comptime `b` reaches
/// the IR as a multiply by an f32 CONSTANT, because the widening of a constant folds in the
/// frontend, so only one side arrives as a convert.
const Factor = union(enum) {
    value: Value,
    /// An f16 bit pattern, which the emitter splats into a scratch register.
    constant: u16,
};

/// A `vfwmacc.vv` found in the IR: `result = acc + a * b`, where the factors are f16 and `acc` and
/// the result are f32. The accumulator can be missing, because the core has no float multiply on
/// its own and this instruction stands in for one. See `findMacs`.
const Mac = struct {
    /// What the product adds to. Null when the IR asked for the product alone, and the destination
    /// is zeroed first.
    acc: ?Value,
    a: Factor,
    b: Factor,
};

/// Where the MACs are. `mac` is indexed by instruction and holds the operands of the `add` that
/// becomes the instruction; `folded` marks the multiply and the two widening converts that the
/// same instruction absorbs, so the emitter writes nothing for them.
const Macs = struct {
    mac: []const ?Mac,
    folded: []const bool,

    fn deinit(self: *Macs, allocator: std.mem.Allocator) void {
        allocator.free(self.mac);
        allocator.free(self.folded);
    }

    fn at(self: *const Macs, inst: Inst) ?Mac {
        return self.mac[@backingInt(inst)];
    }

    fn isFolded(self: *const Macs, inst: Inst) bool {
        return self.folded[@backingInt(inst)];
    }
};

/// Find every shape the widening MAC answers. There are three, because the core has no float add
/// and no float multiply of its own, so `vfwmacc.vv` has to stand in for both:
///
///   - `acc + f32(a) * f32(b)`, the instruction's own shape.
///   - `f32(a) * f32(b)` alone, which accumulates into a zeroed destination. The sum of a zero and
///     the exact product is the product, so this costs one extra word and no accuracy.
///   - `acc + f32(a)`, which is a MAC whose second factor is the constant 1.0.
///
/// The IR has no fused multiply-add and no widening multiply, so the hardware instruction only
/// reaches a kernel through these trees. Nothing is rewritten: the multiply and the converts stay
/// in the function and keep their registers, because a Glacier `arith` takes two operands and this
/// instruction reads three values. The registers cost little, since each intermediate lives for one
/// instruction and a chain reuses the same ones.
///
/// This needs no consent attribute. The product of two f16 is exact in f32, so the fusion loses
/// nothing; the core truncates the sum where the IR asks for nearest even, and that is the rounding
/// every Glacier float operation has, recorded in `encode.zig`.
fn findMacs(
    allocator: std.mem.Allocator,
    func: *const Function,
    uses: []const u32,
    divergent: []const bool,
) Error!Macs {
    const mac = try allocator.alloc(?Mac, func.instCount());
    errdefer allocator.free(mac);
    @memset(mac, null);
    const folded = try allocator.alloc(bool, func.instCount());
    errdefer allocator.free(folded);
    @memset(folded, false);

    // The function has no instruction to block map, and the match needs one to keep an absorbed
    // value inside one block.
    const block_of = try allocator.alloc(u32, func.instCount());
    defer allocator.free(block_of);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| block_of[@backingInt(inst)] = @intCast(bi);
    }

    var m: Matcher = .{ .func = func, .uses = uses, .divergent = divergent, .block_of = block_of };

    // How many MACs took each instruction's result as an operand of their own.
    const absorbed = try allocator.alloc(u32, func.instCount());
    defer allocator.free(absorbed);
    @memset(absorbed, 0);

    // The additions come first, so a multiply that feeds one is absorbed by it rather than being
    // claimed as a product of its own.
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            const found = m.matchAdd(inst) orelse continue;
            record(mac, absorbed, inst, found);
        }
    }
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            if (absorbed[@backingInt(inst)] != 0) continue;
            const found = m.matchProduct(inst) orelse continue;
            record(mac, absorbed, inst, found);
        }
    }

    // An absorbed instruction is suppressed only when EVERY reader absorbed it. A multiply is read
    // once by construction, so it always is. A widening convert can be read several times, and a
    // reader that is not a MAC still needs the f32, so it stays and is widened below. The MACs
    // read the f16 source either way, which is sound because `fusedOperands` reports it.
    for (0..func.instCount()) |ii| {
        const inst: Inst = @fromBackingInt(@intCast(ii));
        if (absorbed[ii] == 0) continue;
        const result = func.instResult(inst) orelse continue;
        if (absorbed[ii] == uses[@backingInt(result)]) folded[ii] = true;
    }

    // A widening convert nothing suppressed needs no MAC: `vfwcvt.f.f.v` does it in one
    // instruction, so `convertKind` handles it and this pass has nothing to add.
    return .{ .mac = mac, .folded = folded };
}

fn record(mac: []?Mac, absorbed: []u32, inst: Inst, found: Match) void {
    mac[@backingInt(inst)] = found.mac;
    for (found.folded) |f| {
        if (f) |taken| absorbed[@backingInt(taken)] += 1;
    }
}

/// One recognized MAC: its operands, and the instructions it absorbs.
const Match = struct { mac: Mac, folded: [3]?Inst };

const Matcher = struct {
    func: *const Function,
    uses: []const u32,
    divergent: []const bool,
    block_of: []const u32,

    /// An `add` read as a full MAC. An addition of a single widened f16 is NOT one: with a native
    /// float add it is a widening convert and an add, two instructions, where a MAC against a
    /// constant 1.0 would be four once the constant is materialized and broadcast.
    fn matchAdd(self: *const Matcher, inst: Inst) ?Match {
        const a = switch (self.func.opcode(inst)) {
            .arith => |x| x,
            else => return null,
        };
        if (a.op != .add) return null;
        if (!self.resultIsF32(inst)) return null;

        // Either side can hold the product, so both orders are tried.
        for ([_][2]Value{ .{ a.rhs, a.lhs }, .{ a.lhs, a.rhs } }) |pair| {
            const term = pair[0];
            // A positive zero is not an accumulator. `0.0 + x` IS `x`, apart from a minus zero
            // coming back as plus zero, and zeroing the destination does exactly that anyway. A
            // hand-written `mac(0.0, a, b)` reaches the bare-product form this way rather than
            // needing an optimiser pass to delete the addition first. A MINUS zero is kept, since
            // it is the one accumulator value that changes the answer.
            const acc: ?Value = if (self.isPositiveZero(pair[1])) null else pair[1];

            // `acc + f32(a) * f32(b)`.
            if (self.soleUseInBlock(inst, term)) |defining| {
                if (self.product(inst, defining)) |p| {
                    return .{
                        .mac = .{ .acc = acc, .a = p.a, .b = p.b },
                        .folded = .{ defining, p.folded[0], p.folded[1] },
                    };
                }
            }
        }
        return null;
    }

    /// Whether `v` is the float constant positive zero.
    fn isPositiveZero(self: *const Matcher, v: Value) bool {
        const inst = self.func.definingInst(v) orelse return false;
        const c = switch (self.func.opcode(inst)) {
            .fconst => |x| x,
            else => return false,
        };
        return c == 0.0 and !std.math.signbit(c);
    }

    /// A product with nothing to accumulate into, which zeroes its destination and then
    /// accumulates.
    fn matchProduct(self: *const Matcher, inst: Inst) ?Match {
        if (!self.resultIsF32(inst)) return null;
        const p = self.product(inst, inst) orelse return null;
        return .{
            .mac = .{ .acc = null, .a = p.a, .b = p.b },
            .folded = .{ p.folded[0], p.folded[1], null },
        };
    }

    /// `mul` read as a product of two f16 factors, with any convert absorbable at `at`.
    fn product(self: *const Matcher, at: Inst, mul: Inst) ?struct { a: Factor, b: Factor, folded: [2]?Inst } {
        const m = switch (self.func.opcode(mul)) {
            .arith => |x| x,
            else => return null,
        };
        if (m.op != .mul or !isF32(self.func, self.func.valueType(m.lhs))) return null;
        const fa = self.factor(at, m.lhs) orelse return null;
        const fb = self.factor(at, m.rhs) orelse return null;
        return .{ .a = fa.factor, .b = fb.factor, .folded = .{ fa.folded, fb.folded } };
    }

    /// `v` as one factor of a MAC: a widening convert the instruction at `at` can absorb, or a
    /// float constant an f16 holds exactly.
    /// A widening convert does NOT have to be read once to be absorbed. The MAC reads the f16
    /// source, not the f32, and reusing a widened value is the ordinary thing to write: one
    /// `sphi` feeds two products in any sphere. Whether the convert is then suppressed is decided
    /// afterwards, from whether every reader absorbed it.
    fn factor(self: *const Matcher, at: Inst, v: Value) ?struct { factor: Factor, folded: ?Inst } {
        if (self.sameBlock(at, v)) |inst| {
            if (self.widened(inst)) |ha| {
                return .{ .factor = .{ .value = ha }, .folded = inst };
            }
        }
        if (self.halfConstant(v)) |bits| return .{ .factor = .{ .constant = bits }, .folded = null };
        return null;
    }

    /// The instruction defining `v`, when it shares a block with `at`. One block bounds how far
    /// the f16 source has to stay live, which matters because a spilled vector is refused.
    fn sameBlock(self: *const Matcher, at: Inst, v: Value) ?Inst {
        const inst = self.func.definingInst(v) orelse return null;
        if (self.block_of[@backingInt(inst)] != self.block_of[@backingInt(at)]) return null;
        return inst;
    }

    /// The f16 bit pattern of the float constant `v`, when an f16 holds its value exactly.
    ///
    /// A value the round trip changes is refused, because the core would multiply by a different
    /// number than the IR named. So is one an f16 holds only as a subnormal, since the lane
    /// flushes those and the constant would behave as a zero.
    fn halfConstant(self: *const Matcher, v: Value) ?u16 {
        const inst = self.func.definingInst(v) orelse return null;
        const c = switch (self.func.opcode(inst)) {
            .fconst => |x| x,
            else => return null,
        };
        if (!isWord(self.func, self.func.valueType(v))) return null;
        const wide: f32 = @floatCast(c);
        const narrow: f16 = @floatCast(wide);
        if (@as(f32, narrow) != wide) return null;
        if (flushToZero(f16, narrow) != narrow) return null;
        return @bitCast(narrow);
    }
    /// Whether `inst` produces an f32, which everything a MAC produces does. Divergence does NOT
    /// come into it: a warp-uniform MAC runs on the vector unit too, because that is the only
    /// float multiplier the core has. See `Emitter.emitScalarMac`.
    fn resultIsF32(self: *const Matcher, inst: Inst) bool {
        const result = self.func.instResult(inst) orelse return false;
        return isF32(self.func, self.func.valueType(result));
    }

    /// The instruction defining `v`, when `v` is read exactly once and shares a block with `at`.
    /// One use is what makes absorbing the instruction safe, and one block keeps the absorbed
    /// value from being live across an edge the resolver would then shuffle.
    fn soleUseInBlock(self: *const Matcher, at: Inst, v: Value) ?Inst {
        if (self.uses[@backingInt(v)] != 1) return null;
        const inst = self.func.definingInst(v) orelse return null;
        if (self.block_of[@backingInt(inst)] != self.block_of[@backingInt(at)]) return null;
        return inst;
    }

    /// The f16 that `inst` widens to f32, or null when `inst` is not that convert.
    fn widened(self: *const Matcher, inst: Inst) ?Value {
        const cv = switch (self.func.opcode(inst)) {
            .convert => |c| c,
            else => return null,
        };
        const result = self.func.instResult(inst) orelse return null;
        if (!isF32(self.func, self.func.valueType(result))) return null;
        if (!isHalf(self.func, self.func.valueType(cv.value))) return null;
        return cv.value;
    }
};

fn isF32(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .float => |f| f == .f32,
        else => false,
    };
}

/// Whether `inst`'s result is per-lane, which is what decides the emitter's path. Mirrors
/// `Emitter.isVector` for an instruction that has a result.
fn resultDivergent(func: *const Function, inst: Inst, divergent: []const bool) bool {
    const r = func.instResult(inst) orelse return false;
    return divergent[@backingInt(r)];
}

/// Reject every shape this lowering does not handle, before anything is emitted. A refusal here
/// is a clear error; a refusal halfway through emission is a truncated program.
fn checkSupported(func: *const Function, opts: Options, divergent: []const bool, macs: *const Macs) Error!void {
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
        for (insts, 0..) |inst, i| {
            // The multiply and the two converts a MAC absorbs are emitted by the `add`, so the
            // rules below do not apply to them: on their own the core has neither instruction.
            if (macs.isFolded(inst)) continue;
            try checkInst(func, opts, divergent, macs, block, insts, inst, i);
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| if (r.count != 0) return error.Unsupported,
            .jump => {},
        };
    }
}

fn checkInst(
    func: *const Function,
    opts: Options,
    divergent: []const bool,
    macs: *const Macs,
    block: Block,
    insts: []const Inst,
    inst: Inst,
    i: usize,
) Error!void {
    switch (func.opcode(inst)) {
        // A constant must fit a 32-bit register. An unsigned type reaches 0xFFFFFFFF and a
        // signed one reaches the most negative value, so the window spans both.
        .iconst => |c| if (c < std.math.minInt(i32) or c > std.math.maxInt(u32)) return error.Unsupported,
        // The result type is swept above, so it is already an f32 or an f16 here.
        .fconst => |c| _ = floatBits(func, inst, c) orelse return error.Unsupported,
        // The emitter picks the scalar or the vector table from the result's divergence, so
        // admission has to ask the same table it will use. The integer tables are both
        // complete, but the float ones are not: a divide is scalar only.
        // An `add` the recognizer claimed becomes the widening MAC, which is one instruction where
        // the plain float ops are two or more.
        .arith => |a| if (macs.at(inst) == null) {
            const ty = func.valueType(a.lhs);
            // Two per-lane bools combine with a mask op. Both sides have to BE masks: a
            // warp-uniform bool sits in a scalar register as a 0 or a 1, and broadcasting that
            // gives one value per element rather than one bit per lane, so it is refused rather
            // than quietly read as a mask.
            if (isBool(func, ty) and resultDivergent(func, inst, divergent)) {
                _ = maskBinOp(a.op) orelse return error.Unsupported;
                if (!divergent[@backingInt(a.lhs)] or !divergent[@backingInt(a.rhs)]) {
                    return error.Unsupported;
                }
                return;
            }
            // Float arithmetic runs on the vector unit either way, so its table is the same one
            // whether or not the operands vary per lane; only the plumbing around it differs.
            if (floatVectorBinOp(func, a.op, ty) == null) {
                if (resultDivergent(func, inst, divergent)) {
                    _ = vectorBinOp(func, a.op, ty) orelse return error.Unsupported;
                } else {
                    _ = binOp(func, a.op, ty) orelse return error.Unsupported;
                }
            }
        },
        .arith_imm => |a| {
            // The immediate is an integer, so this opcode never carries a float operand.
            if (isFloat(func, func.valueType(a.lhs))) return error.Unsupported;
            _ = binOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported;
        },
        // A per-lane compare writes a MASK, one bit per lane, and a per-lane select reads one.
        .icmp => |c| {
            const ty = func.valueType(c.lhs);
            // A WARP-UNIFORM float compare has nothing to do it: the compares write a mask, so
            // they are vector instructions, and the scalar path has no float compare at all. An
            // integer compare over float bits answers a different relation and is wrong for a
            // negative value and for NaN, so it is a refusal rather than a substitution.
            if (resultDivergent(func, inst, divergent)) {
                _ = vectorCmpOp(func, c.op, ty) orelse return error.Unsupported;
            } else if (isFloat(func, ty)) {
                return error.Unsupported;
            }
        },
        .select => {},
        .load => |l| {
            if (!isWord(func, func.valueType(l.ptr))) return error.Unsupported;
            const r = func.instResult(inst) orelse return error.Unsupported;
            // An f16 is the low half of a lane, not a word in memory. See `isHalf`.
            if (isHalf(func, func.valueType(r))) return error.Unsupported;
        },
        .store => |st| {
            if (!isWord(func, func.valueType(st.value))) return error.Unsupported;
            if (isHalf(func, func.valueType(st.value))) return error.Unsupported;
            // A mask has to be a real per-lane predicate. A WARP-UNIFORM one says every lane
            // agrees, and the answer for that is a branch over the store rather than a predicate
            // on it, so it is refused rather than widened into a mask of all ones or all zeros.
            if (st.mask) |m| {
                if (!isWord(func, func.valueType(m))) return error.Unsupported;
                if (!divergent[@backingInt(m)]) return error.Unsupported;
            }
        },
        .atomic_rmw => |a| {
            _ = amoOp(func, a.op, func.valueType(a.value)) orelse return error.Unsupported;
            // Every lane would need its own read-modify-write, and the core has one bus. A per-lane
            // atomic is a loop over the lanes, which is not this. The vector emitter would refuse
            // it too; this is here because admission is where a refusal belongs.
            if (divergent[@backingInt(a.ptr)] or divergent[@backingInt(a.value)]) return error.Unsupported;
            if (!isWord(func, func.valueType(a.value))) return error.Unsupported;
        },
        .global_addr => |g| _ = addressOf(opts, func.symbolName(g.symbol)) orelse return error.Unsupported,
        .unary => |u| if (u.op != .reinterpret) return error.Unsupported,
        // A widening convert has no instruction of its own; the recognizer turns one into a MAC.
        .convert => |cv| if (macs.at(inst) == null) {
            _ = convertKind(func, inst, cv) orelse return error.Unsupported;
        },
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

/// The AMO that performs an IR atomic, or null when the core has none for it.
///
/// `compare_exchange` is the absence that matters: it is the one operation that needs LR and SC,
/// and Glacier decodes the AMOs without the load-reserved pair. See `encode.zig`.
fn amoOp(func: *const Function, op: ir.function.AtomicOp, ty: Type) ?rv.AmoOp {
    // The compare is against the OPERAND type, so an unsigned value takes the unsigned form.
    const signed = isSigned(func, ty);
    return switch (op) {
        .add => .add,
        .exchange => .swap,
        .bit_and => .@"and",
        .bit_or => .@"or",
        .bit_xor => .xor,
        .min => if (signed) .min else .minu,
        .max => if (signed) .max else .maxu,
        .compare_exchange => null,
    };
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
    //
    if (isFloat(func, ty)) return floatVectorBinOp(func, op, ty);
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

/// The per-lane float arithmetic the core decodes, or null when it has none for this shape.
///
/// BEFORE ADDING TO THIS, THREE THINGS HAVE TO BE TRUE OF THE CORE, and they are separate. A LANE
/// that can perform the operation. A DECODER that decodes the instruction and routes it to that
/// lane, with the element width, the format select and the rounding mode wired. A BITSTREAM built
/// from both. A lane that can add is not a core that can be told to.
///
/// Each fails differently and none of them fails as a compile error. Without the lane, a multiply
/// whose format does not fit the significand product width returns the canonical NaN and raises the
/// invalid flag. Without the decode, the instruction is never executed at all. Without the
/// bitstream, neither of the first two is in the hardware anyone loads.
///
/// So this list is NOT the lane's op list. The divide is the one that will look missing: the lane
/// reaches its divider through a separate port set with its own handshake, because it iterates, so
/// `vfdiv.vv` is not an arm on the lane's op input and does not decode. `FDIV.S` stays the only
/// float divide, and it is warp-uniform; see `binOp`.
///
/// f16 arithmetic is absent for a different reason: the element width is fixed at 32 bits, so these
/// read whole lanes. An f16 lives in the low half of one, which is the MAC's shape and not this.
fn floatVectorBinOp(func: *const Function, op: ir.function.BinOp, ty: Type) ?*const fn (VReg, VReg, VReg) u32 {
    if (!isF32(func, ty)) return null;
    return switch (op) {
        .add => rv.vfadd_vv,
        .sub => rv.vfsub_vv,
        .mul => rv.vfmul_vv,
        else => null,
    };
}

/// The per-lane compare that answers `op`, and whether its operands must be swapped to get it.
///
/// A compare writes one BIT per lane into a mask register. RVV has no vector-vector greater-than,
/// and needs none: greater-than is less-than with the operands the other way round.
///
/// The float set is three instructions, not six. There is no `vmfne`, and it needs none either:
/// `not (a == b)` and `a != b` agree on EVERY input including NaN, because a NaN equals nothing and
/// differs from everything, so the negation is exact rather than an approximation. The compares are
/// ordered, which is all the lowering can ask for: `vulcan-spirv` maps only the `FOrd` predicates
/// and refuses the unordered ones.
///
/// f16 has no compare for the same reason it has no arithmetic: the element width is 32 bits, so
/// these read whole lanes.
fn vectorCmpOp(func: *const Function, op: ir.function.CmpOp, ty: Type) ?struct {
    encode: *const fn (VReg, VReg, VReg) u32,
    swap: bool = false,
    /// Whether the mask has to be inverted afterwards, which only `ne` on a float needs.
    negate: bool = false,
} {
    if (isFloat(func, ty)) {
        if (!isF32(func, ty)) return null;
        return switch (op) {
            .eq => .{ .encode = rv.vmfeq_vv },
            .ne => .{ .encode = rv.vmfeq_vv, .negate = true },
            .lt => .{ .encode = rv.vmflt_vv },
            .le => .{ .encode = rv.vmfle_vv },
            .gt => .{ .encode = rv.vmflt_vv, .swap = true },
            .ge => .{ .encode = rv.vmfle_vv, .swap = true },
        };
    }
    const signed = isSigned(func, ty);
    return switch (op) {
        .eq => .{ .encode = rv.vmseq_vv },
        .ne => .{ .encode = rv.vmsne_vv },
        .lt => .{ .encode = if (signed) rv.vmslt_vv else rv.vmsltu_vv },
        .le => .{ .encode = if (signed) rv.vmsle_vv else rv.vmsleu_vv },
        .gt => .{ .encode = if (signed) rv.vmslt_vv else rv.vmsltu_vv, .swap = true },
        .ge => .{ .encode = if (signed) rv.vmsle_vv else rv.vmsleu_vv, .swap = true },
    };
}

/// Whether `ty` is a bool. A PER-LANE bool is a mask: one bit per lane, packed into the first
/// element of the register. It is not a word per lane, so nothing may read one as a value and the
/// ops that combine two of them are the mask ops.
fn isBool(func: *const Function, ty: Type) bool {
    return func.types.type_kind(ty) == .bool;
}

/// The mask op for a bitwise operation between two per-lane bools.
///
/// `vand.vv` gives the same answer today, because sixteen lanes of mask fit inside the first
/// element and a bitwise AND of that element is a bitwise AND of the mask. That stops at
/// thirty-two lanes, where the mask spills into the second element and an element-wise op reads
/// the lanes of one register against the wrong lanes of the other. These ops are right at any
/// width, so the coincidence is not used.
fn maskBinOp(op: ir.function.BinOp) ?*const fn (VReg, VReg, VReg) u32 {
    return switch (op) {
        .bit_and => rv.vmand_mm,
        .bit_or => rv.vmor_mm,
        .bit_xor => rv.vmxor_mm,
        else => null,
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
    /// Which `add` instructions are a widening MAC, and which instructions one absorbs.
    macs: *const Macs,
    /// Which accesses are unit-stride, and which address arithmetic one absorbs.
    strides: *const Strides,
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

    fn vmove(self: *Emitter, dst: VReg, src: VReg) Error!void {
        if (dst != src) try self.put(rv.vmv_v_v(dst, src));
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
            // A masked store is per-lane by definition: the mask decides which lanes write, so it
            // goes out as a predicated scatter even when the address and the value are uniform.
            .store => |st| st.mask != null or self.isDivergent(st.ptr) or self.isDivergent(st.value),
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
            .load => |l| {
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                // A contiguous run is one instruction and one bus run. See `Strides`.
                if (self.strides.at(inst)) |b| {
                    return self.put(rv.vle32(dst, try self.reg(b, pos)));
                }
                try self.put(rv.vluxei32(dst, .x0, try self.vectorOperand(l.ptr, pos)));
            },
            .store => |st| {
                // A masked store can have a uniform address AND a uniform value, since the mask
                // alone makes it per-lane, so both go through the pool and each gets its own
                // scratch register rather than sharing one.
                var pool: Pool = .{};
                // A contiguous run takes a scalar start address, so the per-lane address is not
                // read at all and the arithmetic that built it is not emitted. See `Strides`.
                if (self.strides.at(inst)) |b| {
                    const start = try self.reg(b, pos);
                    const words = try self.maskOperand(st.value, &pool, pos);
                    const mask = st.mask orelse return self.put(rv.vse32(words, start));
                    const bits = try self.vreg(mask, pos);
                    try self.put(rv.vmand_mm(.v0, bits, bits));
                    return self.put(rv.vse32_m(words, start));
                }
                const addr = try self.maskOperand(st.ptr, &pool, pos);
                const val = try self.maskOperand(st.value, &pool, pos);
                const mask = st.mask orelse {
                    try self.put(rv.vsuxei32(val, .x0, addr));
                    return;
                };
                // The predicate reads from v0 and nowhere else. A lane whose bit is clear makes no
                // bus access at all, which is what lets a store survive if-conversion.
                const bits = try self.vreg(mask, pos);
                try self.put(rv.vmand_mm(.v0, bits, bits));
                try self.put(rv.vsuxei32_m(val, .x0, addr));
            },
            .unary => |u| try self.vmove(
                try self.vreg(self.func.instResult(inst).?, pos),
                try self.vectorOperand(u.value, pos),
            ),
            .convert => |cv| {
                if (self.macs.at(inst)) |m| return self.emitVectorMac(m, inst, pos);
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                const src = try self.vectorOperand(cv.value, pos);
                const kind = convertKind(self.func, inst, cv) orelse return error.Unsupported;
                if (kind == .move) return self.vmove(dst, src);
                try self.put(convertEncoder(kind)(dst, src));
            },
            .arith => |a| {
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                if (self.macs.at(inst)) |m| return self.emitVectorMac(m, inst, pos);
                if (isBool(self.func, self.func.valueType(a.lhs))) {
                    const mask_op = maskBinOp(a.op) orelse return error.Unsupported;
                    return self.put(mask_op(dst, try self.vreg(a.lhs, pos), try self.vreg(a.rhs, pos)));
                }
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
            // A per-lane compare writes a MASK into its destination: one bit per lane, not a word
            // per lane. Nothing else may read it as a value.
            .icmp => |c| {
                var pool: Pool = .{};
                const dst = try self.vreg(self.func.instResult(inst).?, pos);
                const cmp = vectorCmpOp(self.func, c.op, self.func.valueType(c.lhs)) orelse return error.Unsupported;
                const lhs = try self.maskOperand(c.lhs, &pool, pos);
                const rhs = try self.maskOperand(c.rhs, &pool, pos);
                // Greater-than is less-than the other way round, which is why there is no encoder
                // for it.
                try self.put(if (cmp.swap) cmp.encode(dst, rhs, lhs) else cmp.encode(dst, lhs, rhs));
                // A float `ne` is the equality inverted, which is exact: a NaN equals nothing and
                // differs from everything, so the two agree on every input.
                if (cmp.negate) try self.put(rv.vmnand_mm(dst, dst, dst));
            },
            .select => |s| try self.emitVectorSelect(s, inst, pos),
            else => return error.Unsupported,
        }
    }

    /// `vfwmacc.vv`: the destination accumulates the f32 product of two f16, so the accumulator
    /// has to be in the destination before the instruction runs. The allocator is asked for that
    /// register in `copySource`, and a `vmv.v.v` covers the case where it could not give it.
    ///
    /// A factor dies where the result is born, so the allocator may give the result that factor's
    /// register, and the accumulator has to be in the result's register before the instruction
    /// runs. A factor sitting there is therefore staged aside first. This is not theoretical: a
    /// bare product gets `vfwmacc.vv v2, v1, v2` and would read a zeroed destination as its own
    /// factor.
    /// A per-lane MAC, whose result stays in a vector register.
    fn emitVectorMac(self: *Emitter, m: Mac, inst: Inst, pos: u32) Error!void {
        var pool: Pool = .{};
        try self.mac(m, try self.vreg(self.func.instResult(inst).?, pos), &pool, pos);
    }

    /// An operand of a per-lane instruction, broadcast from a scalar register when it is
    /// warp-uniform. Unlike `vectorOperand` this draws from the pool, so several operands of one
    /// instruction can each need one.
    fn maskOperand(self: *Emitter, v: Value, pool: *Pool, pos: u32) Error!VReg {
        if (self.isDivergent(v)) return self.vreg(v, pos);
        const into = try pool.take();
        try self.put(rv.vmv_v_x(into, try self.reg(v, pos)));
        return into;
    }

    /// A per-lane select: `vmerge.vvm`, which reads its mask from v0 and nowhere else.
    ///
    /// The mask is moved into v0 with a mask AND against itself. `vmv.v.v` would also work on this
    /// core, because sixteen lanes of mask fit inside the first 32-bit element and it copies that
    /// element, but it is a move of ELEMENTS and this is a move of BITS; the mask op says what is
    /// meant and stays right at any lane count. v0 is never allocatable, so writing it displaces
    /// nothing. NOTE the warp model cannot tell the two apart, for the same reason.
    fn emitVectorSelect(self: *Emitter, s: ir.function.Select, inst: Inst, pos: u32) Error!void {
        var pool: Pool = .{};
        const dst = try self.vreg(self.func.instResult(inst).?, pos);
        const then_v = try self.maskOperand(s.then, &pool, pos);
        const else_v = try self.maskOperand(s.@"else", &pool, pos);
        if (!self.isDivergent(s.cond)) {
            // A warp-uniform condition picks the same side in every lane, but the sides are
            // per-lane, so it still goes through the merge. The mask is built by comparing the
            // broadcast condition against zero, which gives all ones exactly where it is FALSE, so
            // the two sides go in the other order below.
            const tmp = try pool.take();
            try self.put(rv.vmv_v_x(tmp, try self.reg(s.cond, pos)));
            try self.put(rv.vmseq_vx(.v0, tmp, .x0));
            try self.put(rv.vmerge_vvm(dst, then_v, else_v));
            return;
        }
        const mask = try self.vreg(s.cond, pos);
        try self.put(rv.vmand_mm(.v0, mask, mask));
        try self.put(rv.vmerge_vvm(dst, else_v, then_v));
    }

    /// A warp-uniform float binary operation. There is no scalar form of any of these, so both
    /// operands are broadcast, the vector unit does the work and lane zero carries the answer back.
    /// Four words where a scalar float unit would take one.
    fn emitScalarFloatBinOp(
        self: *Emitter,
        encode: *const fn (VReg, VReg, VReg) u32,
        a: ir.function.Arith,
        dst: Reg,
        pos: u32,
    ) Error!void {
        var pool: Pool = .{};
        const into = try pool.take();
        const lhs = try pool.take();
        const rhs = try pool.take();
        try self.put(rv.vmv_v_x(lhs, try self.reg(a.lhs, pos)));
        try self.put(rv.vmv_v_x(rhs, try self.reg(a.rhs, pos)));
        try self.put(encode(into, lhs, rhs));
        try self.put(rv.vmv_x_s(dst, into));
    }

    /// A warp-uniform MAC. It still runs on the vector unit, because that is the only float
    /// multiplier the core has, so it accumulates in a scratch and the answer comes back out of
    /// lane zero. Around six words where a scalar float unit would take one, which is the price of
    /// a core that has no scalar float multiply at all.
    fn emitScalarMac(self: *Emitter, m: Mac, dst: Reg, pos: u32) Error!void {
        var pool: Pool = .{};
        const at = try pool.take();
        try self.mac(m, at, &pool, pos);
        try self.put(rv.vmv_x_s(dst, at));
    }

    fn mac(self: *Emitter, m: Mac, dst: VReg, pool: *Pool, pos: u32) Error!void {
        var a = try self.macOperand(m.a, pool, pos);
        var b = try self.macOperand(m.b, pool, pos);
        // A factor dies where the result is born, so the allocator may give the result that
        // factor's register, and the accumulator has to be in the result's register before the
        // instruction runs. A factor sitting there is therefore staged aside first. This is not
        // theoretical: a bare product gets `vfwmacc.vv v2, v1, v2` and would otherwise read a
        // zeroed destination as its own factor.
        if (a == dst or b == dst) {
            const staged = try pool.take();
            try self.put(rv.vmv_v_v(staged, dst));
            if (a == dst) a = staged;
            if (b == dst) b = staged;
        }
        if (m.acc) |acc| {
            if (self.isDivergent(acc)) {
                const from = try self.vreg(acc, pos);
                if (dst != from) try self.put(rv.vmv_v_v(dst, from));
            } else {
                try self.put(rv.vmv_v_x(dst, try self.reg(acc, pos)));
            }
        } else {
            // A product with nothing to add to accumulates into zero. The product of two f16 is
            // exact in f32, so this answers the product itself. The one difference is a product of
            // minus zero, which comes back as plus zero: adding zeroes of opposite sign gives plus
            // zero in every rounding mode this core has.
            try self.put(rv.vmv_v_x(dst, .x0));
        }
        try self.put(rv.vfwmacc_vv(dst, a, b));
    }

    /// The vector register holding one MAC factor. A warp-uniform value and a constant both live
    /// outside the vector file, so each is broadcast into a scratch first.
    fn macOperand(self: *Emitter, f: Factor, pool: *Pool, pos: u32) Error!VReg {
        switch (f) {
            .value => |v| {
                if (self.isDivergent(v)) return self.vreg(v, pos);
                const into = try pool.take();
                try self.put(rv.vmv_v_x(into, try self.reg(v, pos)));
                return into;
            },
            .constant => |bits| {
                const into = try pool.take();
                try self.constant(scratch, bits);
                try self.put(rv.vmv_v_x(into, scratch));
                return into;
            },
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
        // The multiply and the converts a MAC absorbed. Their registers stay allocated and hold
        // nothing, because a three-operand instruction cannot be spelled as one `arith`.
        if (self.macs.isFolded(inst)) return;
        // The address arithmetic a unit-stride access absorbed. Same reasoning as a MAC's folded
        // multiply: the instruction is not emitted and its register holds nothing.
        if (self.strides.isFolded(inst)) return;
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
            .convert => |cv| if (self.macs.at(inst)) |m|
                try self.emitScalarMac(m, dst, pos)
            else
                try self.emitConvert(cv, inst, dst, pos),
            .arith => |a| {
                const ty = self.func.valueType(a.lhs);
                if (self.macs.at(inst)) |m| {
                    try self.emitScalarMac(m, dst, pos);
                } else if (floatVectorBinOp(self.func, a.op, ty)) |encode| {
                    try self.emitScalarFloatBinOp(encode, a, dst, pos);
                } else {
                    const f = binOp(self.func, a.op, ty) orelse return error.Unsupported;
                    try self.put(f(dst, try self.reg(a.lhs, pos), try self.reg(a.rhs, pos)));
                }
            },
            .arith_imm => |a| try self.emitArithImm(a, dst, pos),
            .icmp => |c| try self.emitCompare(c, dst, pos),
            .select => |s| try self.emitSelect(s, dst, pos),
            .load => |l| try self.put(rv.lw(dst, try self.reg(l.ptr, pos), 0)),
            // A masked store is never scalar: `isVector` reports one as per-lane whatever its
            // operands are. The refusal is here because a scalar store cannot predicate, so a
            // change that routed one here would write every lane.
            .store => |st| if (st.mask != null)
                return error.Unsupported
            else
                try self.put(rv.sw(try self.reg(st.value, pos), try self.reg(st.ptr, pos), 0)),
            // The old value goes to `dst`, or to the lowering's scratch when nobody reads it: the
            // core has no reduction form that writes no register, so the read-modify-write happens
            // either way and the answer is thrown away.
            .atomic_rmw => |a| try self.put(rv.amo(
                amoOp(self.func, a.op, self.func.valueType(a.value)) orelse return error.Unsupported,
                dst,
                try self.reg(a.ptr, pos),
                try self.reg(a.value, pos),
                a.ordering == .acquire or a.ordering == .acq_rel or a.ordering == .seq_cst,
                a.ordering == .release or a.ordering == .acq_rel or a.ordering == .seq_cst,
            )),
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
