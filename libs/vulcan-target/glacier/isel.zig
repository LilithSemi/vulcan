//! Glacier ruction selection: a Vulcan IR kernel to a Glacier word stream.
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

/// Why a kernel was refused, for a caller that asks for it.
///
/// `error.Unsupported` on its own sends a reader bisecting the COMPILER for a construct that is in
/// their own kernel. That happened: a refusal with no detail cost one session four compiles and
/// another an hour, and a one-line print of this information settled it in three runs.
///
/// `block` indexes the function the caller passed, for every block it had: the passes that run
/// before admission APPEND blocks (critical-edge splitting) rather than renumbering, so an
/// original block keeps its index. `index` is the instruction's position within that block in the
/// LOWERING'S OWN order, which if-conversion and the address rewrite both change, so it is for
/// reading beside an IR dump rather than for mapping back to source.
pub const Refusal = struct {
    /// A fixed phrase naming what was refused. Empty when nothing filled it in.
    reason: []const u8 = "",
    /// The opcode it was on, by name, or empty for a refusal about the whole function.
    opcode: []const u8 = "",
    block: u32 = unknown,
    index: u32 = unknown,
    /// For a refusal about a SIZE, what the kernel asked for and what it may have, in bytes.
    /// Both `unknown` otherwise. A size refusal that does not name the size leaves the reader
    /// guessing how much to cut.
    wanted: u32 = unknown,
    limit: u32 = unknown,

    pub const unknown: u32 = std.math.maxInt(u32);

    /// Record `reason` and refuse. Returns the error so a caller writes one statement.
    fn at(self: ?*Refusal, func: *const Function, block: Block, index: usize, inst: Inst, reason: []const u8) error{Unsupported} {
        if (self) |r| r.* = .{
            .reason = reason,
            .opcode = @tagName(func.opcode(inst)),
            .block = @backingInt(block),
            .index = @intCast(index),
        };
        return error.Unsupported;
    }

    /// Record a refusal about the function rather than about one instruction.
    fn whole(self: ?*Refusal, reason: []const u8) error{Unsupported} {
        if (self) |r| r.* = .{ .reason = reason };
        return error.Unsupported;
    }

    /// Record a refusal about a size, naming what was asked for and what fits.
    fn tooBig(self: ?*Refusal, reason: []const u8, wanted: u64, limit: u32) error{Unsupported} {
        if (self) |r| r.* = .{
            .reason = reason,
            .wanted = std.math.cast(u32, wanted) orelse unknown,
            .limit = limit,
        };
        return error.Unsupported;
    }
};

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
    /// How many words the core's instruction aperture holds. A program longer than this is refused
    /// rather than truncated at load.
    ///
    /// The default is the ARCHITECTURAL CEILING, 8192, and a given SoC may hold FEWER:
    /// `GlacierPipelinedCoreSoc.programWords` defaults to the whole aperture and a board may lower
    /// it to pay for less block RAM. That number is a per-BOARD area decision rather than a
    /// property of the core's shape, so it is not in the profile and this compiler cannot derive
    /// it: nothing is attached when a kernel is built. A caller that knows the device reads the
    /// core's `PROG_WORDS` register and passes the real figure here; the ceiling is the honest
    /// default for everyone else, and the runtime refuses a program the core cannot hold.
    program_words: u32 = 8192,
    /// Words ALREADY RESIDENT in the aperture beside this program, which this one has to fit
    /// around.
    ///
    /// A Glacier core holds ONE program, and uploading one costs a PCIe round trip per word, so a
    /// pipeline keeps both stages resident and a draw is launch arguments plus a launch. Nothing
    /// inside a single compile can see the other stage, so the caller that composes the pipeline
    /// passes the first stage's length when it compiles the second.
    resident_words: u32 = 0,
};

pub const vertex_input_stride = 256;
pub const vertex_output_stride = 256;
pub const vertex_input_vertex_index_offset = 0x60;
pub const vertex_input_instance_index_offset = 0x64;
pub const fragment_record_stride = 512;
pub const fragment_output_depth_offset = 0x40;
pub const fragment_output_discard_offset = 0x44;
pub const fragment_input_point_coord_offset = 0x60;
pub const fragment_input_front_facing_offset = 0x68;
pub const fragment_input_coord_offset = 0x70;
pub const fragment_input_gradient_offset = 0x100;

/// Lower `func` to a Glacier word stream. The caller owns the result.
pub fn compileKernel(allocator: std.mem.Allocator, func: *const Function, opts: Options) Error![]u32 {
    return compileKernelInternal(allocator, func, opts, null);
}

pub fn compileVertex(allocator: std.mem.Allocator, func: *const Function, opts: Options) Error![]u32 {
    var work = try func.clone(allocator);
    defer work.deinit();
    try lowerVertexIO(allocator, &work);
    return compileKernelInternal(allocator, &work, opts, null);
}

pub fn compileFragment(allocator: std.mem.Allocator, func: *const Function, opts: Options) Error![]u32 {
    var work = try func.clone(allocator);
    defer work.deinit();
    try lowerFragmentIO(allocator, &work);
    return compileKernelInternal(allocator, &work, opts, null);
}

/// `compileFragment`, with a generated rasteriser around the body. See `lowerFragmentRaster`.
pub fn compileFragmentRaster(allocator: std.mem.Allocator, func: *const Function, opts: Options) Error![]u32 {
    return compileFragmentRasterExplained(allocator, func, opts, null);
}

/// `compileFragmentRaster`, and on a refusal it says what it refused. See `Refusal`.
pub fn compileFragmentRasterExplained(
    allocator: std.mem.Allocator,
    func: *const Function,
    opts: Options,
    why: ?*Refusal,
) Error![]u32 {
    if (why) |r| r.* = .{};
    var work = try func.clone(allocator);
    defer work.deinit();
    try lowerFragmentRaster(allocator, &work, why);
    return compileKernelInternal(allocator, &work, opts, why);
}

/// `compileKernel`, and on a refusal it says what it refused. See `Refusal`.
pub fn compileKernelExplained(
    allocator: std.mem.Allocator,
    func: *const Function,
    opts: Options,
    why: *Refusal,
) Error![]u32 {
    why.* = .{};
    return compileKernelInternal(allocator, func, opts, why);
}

fn compileKernelInternal(
    allocator: std.mem.Allocator,
    func: *const Function,
    opts: Options,
    why: ?*Refusal,
) Error![]u32 {
    if (func.blockCount() == 0) return Refusal.whole(why, "a kernel with no blocks");
    if (func.is_variadic) return Refusal.whole(why, "a variadic kernel");
    if (func.sret) return Refusal.whole(why, "a kernel that returns a value through a pointer");

    // The Wimmer resolver needs a block on every critical edge to put its shuffle moves on, and
    // splitting mutates the function, so split a clone and leave the caller's input alone.
    var work = try func.clone(allocator);
    defer work.deinit();

    try lowerImageSample(allocator, &work);
    try lowerTexelFetch(allocator, &work);
    try lowerStorageImageWrite(allocator, &work);
    try lowerImageSize(allocator, &work);
    try lowerSquareRoot(allocator, &work);
    try unifyDiscardReturns(allocator, &work);

    // Promote every stack slot to an SSA value BEFORE anything else looks at the function. A
    // warp has no stack, so an `alloca` that survives is refused below, and a frontend that keeps
    // its loop counters in memory (Zig's SPIR-V backend does) would otherwise be rejected for a
    // reason that has nothing to do with the hardware. This is also why the promotion belongs
    // here rather than in a frontend: it needs real phis, which only an SSA construction gives.
    {
        var analyses: opt.pass.Analyses = .{ .allocator = allocator, .func = &work };
        defer analyses.deinit();
        _ = try opt.mem2reg.run(allocator, &work, &analyses);
        // A scale by a power of two is a SHIFT. An address computation reaches the IR as a
        // multiply by the element size, and this core has a multiplier, so nothing downstream was
        // obliged to notice: it emitted `mul` by four for every indexed access. A shift is cheaper
        // on any core and it keeps an address computation out of the multiplier entirely.
        // The SHIFT phase only: the magic-number phase widens to 64 bits for a division by a
        // non-power-of-two, and a warp register holds 32.
        _ = try opt.strength.shiftPhase(allocator, &work);
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
    // A loop a flattened region decided the mask for, rather than a branch above it.
    var loop_masks: std.ArrayList(LoopMask) = .empty;
    defer loop_masks.deinit(allocator);
    while (true) {
        const d = try analyzeDivergence(allocator, &work);
        defer allocator.free(d);
        // Merging first, because a block boundary with nothing on either side of it hides both a
        // region and a loop latch. Flattening then makes new boundaries, so the two iterate
        // together rather than one after the other.
        const merged = try mergeStraightLine(allocator, &work);
        // A branch that chooses between two identical arrivals goes first: it is not a region to
        // flatten, it is a branch that should not be there.
        const dropped = dropPointlessBranches(&work);
        const flattened = try ifConvert(allocator, &work, d, &loop_masks);
        if (!merged and !dropped and !flattened) break;
    }
    const reachable_after = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable_after);

    // Flattening a region makes new edges, and a loop whose body it collapsed comes out as a
    // SELF-LOOP with its back edge critical again. Splitting once before if-conversion is
    // therefore not enough: the allocator needs a block on every critical edge to put its shuffle
    // moves on, and the loop recognizer below reads the split block as the latch.
    try ir.critical_edge.splitCriticalEdges(allocator, &work);

    // A loop whose lanes disagree about when to stop: carry an active mask round it, mask the
    // stores with it, and freeze the carried values of a lane that has finished. This runs AFTER
    // if-conversion, which is what flattens an inner diamond into the straight-line body this
    // needs, and before the analysis below, because it adds values of its own.
    var latches: std.ArrayList(Inst) = .empty;
    defer latches.deinit(allocator);
    // A loop whose test sits at the TOP becomes the do-while the conversion reads, and then the
    // conversion runs. Both are analysed again each round, because each adds values of its own and
    // because A LOOP INSIDE ANOTHER ONE only looks divergent after the outer one is converted: an
    // inner bound that comes from an outer parameter is the same in every lane until that
    // parameter is frozen per lane.
    while (true) {
        var moved = false;
        {
            const d = try analyzeDivergence(allocator, &work);
            defer allocator.free(d);
            if (try rotateWhileLoops(allocator, &work, d)) moved = true;
        }
        {
            const d = try analyzeDivergence(allocator, &work);
            defer allocator.free(d);
            if (try convertDivergentLoops(allocator, &work, d, &loop_masks, &latches)) moved = true;
        }
        if (!moved) break;
    }
    // A loop whose body spans blocks puts a branch where a jump was, and the back edge is critical
    // again. The allocator needs a block on it to put its shuffle moves on.
    try ir.critical_edge.splitCriticalEdges(allocator, &work);

    // A loop that absorbed its guard leaves the guard's failing side unreachable. Emptying it now
    // keeps the passes below off a block nothing reaches, and the allocator off a live range in
    // one: a value there is live nowhere and the allocation verifier reads that as unsound.
    {
        const rest = try ir.reachable.neutralizeUnreachable(allocator, &work);
        allocator.free(rest);
    }
    const any_lane_at = try allocator.alloc(bool, work.instCount());
    defer allocator.free(any_lane_at);
    @memset(any_lane_at, false);
    for (latches.items) |l| any_lane_at[@backingInt(l)] = true;
    const any_lane: AnyLane = .{ .at = any_lane_at };

    // No per-lane parameter may be fed a warp-uniform argument: see `broadcastUniformArgs` for
    // what the shared allocator does with a move that crosses the two register files.
    {
        const d = try analyzeDivergence(allocator, &work);
        defer allocator.free(d);
        _ = try broadcastUniformArgs(allocator, &work, d);
    }

    // Put every one-word-per-lane address into the shape `findStrides` reads, then clear out the
    // arithmetic that is left over. The divergence analysis runs again below, because the rewrite
    // adds values and moves work from the vector side to the scalar side.
    {
        const d = try analyzeDivergence(allocator, &work);
        defer allocator.free(d);
        _ = try hoistLaneAddresses(allocator, &work, d);
    }

    // Unconditionally, because the passes above all leave dead values behind: a condition whose
    // branch is gone, a predicate for a block that turned out to hold nothing, the arithmetic of
    // an address that was rebuilt. The emitter drops an instruction nothing reads, but a dead
    // value still holds a register until then, and a dead MASK is a vector register.
    {
        var analyses: opt.pass.Analyses = .{ .allocator = allocator, .func = &work };
        defer analyses.deinit();
        _ = try opt.dce.run(allocator, &work, &analyses);
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

    try checkSupported(&work, opts, divergent, &macs, &any_lane, why);

    // Workgroup shared memory lives in the hart's own scratch region, directly below `mscratch`,
    // with the spill slots under it. A workgroup is ONE WARP on this core and a hart owns its
    // slice of the region, so each workgroup gets its own window with nothing to arrange: the
    // per-core `shmaddr` region would hand every warp of the core the SAME base, which is right
    // for a wider scope and wrong for a workgroup.
    var frame = gpu.abi.layoutSharedFrame(allocator, &work, .{
        .param_base = 0,
        .pointer_bytes = 4,
        .param_align = 4,
        // Placed against no limit, then checked against the real one below, so the refusal can
        // name what the kernel ASKED FOR. The placement rule is the shared one either way.
        .max_shared_bytes = std.math.maxInt(u32),
        .linear_thread_id = true,
    }) catch |e| switch (e) {
        error.OutOfMemory => return error.OutOfMemory,
        error.SharedMemoryTooLarge => return Refusal.tooBig(why, "more workgroup memory than a 32-bit offset can name", 0, opts.profile.scratchBytes()),
        error.UnsupportedSharedType => return Refusal.whole(why, "a workgroup object this lowering cannot place"),
        error.SharedFrameMismatch => return Refusal.whole(why, "a declared workgroup size that disagrees with the slots"),
        error.SharedFrameAliasesParameter => return Refusal.whole(why, "both a workgroup slot and a workgroup pointer parameter"),
        error.UnsupportedParamType => return Refusal.whole(why, "a parameter type this lowering cannot place"),
    };
    defer frame.deinit(allocator);
    const shared_bytes = frame.bytes;
    if (shared_bytes > opts.profile.scratchBytes()) {
        return Refusal.tooBig(
            why,
            "more workgroup memory than the hart's scratch region holds",
            shared_bytes,
            opts.profile.scratchBytes(),
        );
    }

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
    if (shared_bytes + slots * 4 > opts.profile.scratchBytes()) {
        return Refusal.tooBig(
            why,
            "more spill slots and workgroup memory than the hart's scratch region holds",
            shared_bytes + @as(u64, slots) * 4,
            opts.profile.scratchBytes(),
        );
    }
    // The deepest slot is reached by one signed 12-bit displacement from `sp`, and the stack grows
    // down, so 2048 bytes is all one displacement can name.
    //
    // At a 2048-byte region the two ceilings now MEET: 512 slots either way, so the check above
    // never fires on its own. Raising the region again buys nothing until a deep slot is addressed
    // with more than one instruction, which is the thing to change first rather than the number.
    if (shared_bytes + slots * 4 > 2048) {
        return Refusal.tooBig(why, "a spill slot past the one displacement that reaches it", shared_bytes + @as(u64, slots) * 4, 2048);
    }

    // A spilled per-lane value goes under the scalar slots, one whole vector register each, and
    // both regions have to fit the hart's scratch. A mask spends a register's worth of space on
    // one bit per lane, which is the price of not having a narrower store.
    const vector_bytes = walloc.slot_count_per_class[1] * @as(u32, opts.profile.lanes()) * 4;
    if (shared_bytes + slots * 4 + vector_bytes > opts.profile.scratchBytes()) {
        return Refusal.tooBig(
            why,
            "more spill space and workgroup memory than the hart's scratch region holds",
            @as(u64, shared_bytes) + slots * 4 + vector_bytes,
            opts.profile.scratchBytes(),
        );
    }
    // The deepest per-lane slot is addressed by one signed 12-bit displacement from `sp`, the same
    // ceiling the scalar slots have.
    if (shared_bytes + slots * 4 + vector_bytes > 2048) {
        return Refusal.tooBig(why, "a spill slot past the one displacement that reaches it", @as(u64, shared_bytes) + slots * 4 + vector_bytes, 2048);
    }

    // Recomputed here rather than reused from above: the passes between have emptied blocks and
    // rewired edges, so what was reachable then is not what is reachable now. Emission skips the
    // rest, which is worth a word each: a dead block otherwise leaves an `ebreak` behind.
    const live = try ir.reachable.neutralizeUnreachable(allocator, &work);
    defer allocator.free(live);

    const remat = try rematTable(allocator, &work);
    var e: Emitter = .{
        .allocator = allocator,
        .func = &work,
        .live = live,
        .opts = opts,
        .alloc = &walloc,
        .divergent = divergent,
        .uses = uses,
        .macs = &macs,
        .strides = &strides,
        .any_lane = &any_lane,
        .why = why,
        .spills = slots != 0 or walloc.slot_count_per_class[1] != 0 or shared_bytes != 0,
        .shared = frame.slots,
        .shared_bytes = shared_bytes,
        .remat = remat,
        .elide = try elidableSlots(allocator, &walloc, remat, slots),
        .vector_slot_base = slots * 4,
    };
    defer e.deinit();
    try e.run();

    // A cluster barrier rides on an INSTRUCTION attribute, which a pass that clones or rebuilds
    // the instruction can drop. Losing one would emit the narrower barrier, which on this target
    // is no instruction at all, so a kernel would run with its handshake silently gone. The count
    // the frontend recorded before any pass ran is what turns that into a refusal.
    if (gpu.attrs.clusterBarriers(func)) |declared| {
        if (e.cluster_barriers < declared) {
            return Refusal.tooBig(why, "fewer core-wide barriers than the kernel declared, so a pass dropped one", e.cluster_barriers, declared);
        }
    }
    // The aperture holds ONE program and a host uploads it a word at a time, so a program that
    // does not fit is refused here rather than truncated at load. `resident_words` carries the
    // other stage of a pipeline, which a single compile cannot see.
    const total: u64 = @as(u64, opts.resident_words) + e.words.items.len;
    if (total > opts.program_words) {
        return Refusal.tooBig(why, "a program longer than the core's instruction aperture holds", total, opts.program_words);
    }
    return e.words.toOwnedSlice(allocator);
}

fn hasGpuKey(func: *const Function, value: Value, key: []const u8) bool {
    var attrs = func.attributesOf(.{ .value = value });
    while (attrs.next()) |attr| switch (attr) {
        .custom => |custom| if (std.mem.eql(u8, custom.namespace, "vulcan.gpu") and std.mem.eql(u8, custom.key, key)) return true,
        else => {},
    };
    return false;
}

fn hasDiscardParam(func: *const Function) bool {
    for (func.blockParams(@fromBackingInt(0))) |param| {
        if (hasGpuKey(func, param, "discard_fn")) return true;
    }
    return false;
}

fn unifyDiscardReturns(allocator: std.mem.Allocator, func: *Function) Error!void {
    if (!hasDiscardParam(func)) return;
    var returns = std.ArrayList(Block).empty;
    defer returns.deinit(allocator);
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const terminator = func.terminator(block) orelse continue;
        if (terminator != .ret) continue;
        if (terminator.ret.count != 0) return error.Unsupported;
        try returns.append(allocator, block);
    }
    if (returns.items.len <= 1) return;
    const exit = try func.appendBlock();
    func.setTerminator(exit, .{ .ret = ir.function.Ret.none() });
    for (returns.items) |block| try func.setJump(block, exit, &.{});
}

fn gpuIntAttr(func: *const Function, value: Value, key: []const u8) ?u32 {
    var attrs = func.attributesOf(.{ .value = value });
    while (attrs.next()) |attr| switch (attr) {
        .custom => |custom| if (std.mem.eql(u8, custom.namespace, "vulcan.gpu") and std.mem.eql(u8, custom.key, key)) {
            return switch (custom.value) {
                .int => |number| @intCast(number),
                else => null,
            };
        },
        else => {},
    };
    return null;
}

fn hasGpuStage(func: *const Function, stage: []const u8) bool {
    var attrs = func.attributesOf(.func);
    while (attrs.next()) |attr| switch (attr) {
        .custom => |custom| if (std.mem.eql(u8, custom.namespace, "vulcan.gpu") and std.mem.eql(u8, custom.key, "stage")) {
            return switch (custom.value) {
                .string => |name| std.mem.eql(u8, name, stage),
                else => false,
            };
        },
        else => {},
    };
    return false;
}

fn lowerVertexIO(allocator: std.mem.Allocator, func: *Function) Error!void {
    if (!hasGpuStage(func, "vertex") or func.blockCount() == 0) return error.Unsupported;
    const entry: Block = @fromBackingInt(0);
    const original_params = try allocator.dupe(Value, func.blockParams(entry));
    defer allocator.free(original_params);
    const index_type = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const invocation_index = try func.appendBlockParam(entry, index_type);
    try gpu.attrs.setBuiltin(func, invocation_index, .global_id_x);
    const input = try func.appendBlockParam(entry, try func.types.ptrGlobal());
    const output = try func.appendBlockParam(entry, try func.types.ptrGlobal());

    const original_entry = try allocator.dupe(Inst, func.blockInsts(entry));
    defer allocator.free(original_entry);
    func.blockInstsMut(entry).clearRetainingCapacity();
    for (original_params) |param| {
        const slot: u32 = if (gpuIntAttr(func, param, "attr")) |attribute|
            attribute
        else if (gpu.attrs.builtinOf(func, param)) |builtin| switch (builtin) {
            .vertex_index => vertex_input_vertex_index_offset,
            .instance_index => vertex_input_instance_index_offset,
            else => continue,
        } else continue;
        try func.addAttr(.{ .value = param }, .{ .custom = .{
            .namespace = "vulcan.gpu",
            .key = "lowered_graphics_input",
            .value = .flag,
        } });
        const stride = try func.appendInst(entry, index_type, .{ .iconst = vertex_input_stride });
        const row = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .mul, .lhs = invocation_index, .rhs = stride } });
        const offset = try func.appendInst(entry, index_type, .{ .iconst = slot });
        const byte_offset = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = offset } });
        const source = try func.appendInst(entry, func.valueType(input), .{ .arith = .{ .op = .add, .lhs = input, .rhs = byte_offset } });
        const value = try func.appendInst(entry, func.valueType(param), .{ .load = .{ .ptr = source } });
        func.replaceAllUses(param, value);
    }
    try func.blockInstsMut(entry).appendSlice(func.allocator, original_entry);

    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation != .store) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const store = operation.store;
            const slot = gpuIntAttr(func, store.ptr, "out_attr") orelse {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            };
            const stride = try func.appendInst(block, index_type, .{ .iconst = vertex_output_stride });
            const row = try func.appendInst(block, index_type, .{ .arith = .{ .op = .mul, .lhs = invocation_index, .rhs = stride } });
            const offset = try func.appendInst(block, index_type, .{ .iconst = slot });
            const byte_offset = try func.appendInst(block, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = offset } });
            const destination = try func.appendInst(block, func.valueType(output), .{ .arith = .{ .op = .add, .lhs = output, .rhs = byte_offset } });
            if (store.mask) |mask| {
                try func.appendStoreMaskedVol(block, store.value, destination, mask, store.@"volatile");
            } else {
                try func.appendStoreVol(block, store.value, destination, store.@"volatile");
            }
        }
    }
}

fn lowerFragmentIO(allocator: std.mem.Allocator, func: *Function) Error!void {
    if (!hasGpuStage(func, "fragment") or func.blockCount() == 0) return error.Unsupported;
    const entry: Block = @fromBackingInt(0);
    const original_params = try allocator.dupe(Value, func.blockParams(entry));
    defer allocator.free(original_params);
    const index_type = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const fragment_index = try func.appendBlockParam(entry, index_type);
    try gpu.attrs.setBuiltin(func, fragment_index, .global_id_x);
    const input = try func.appendBlockParam(entry, try func.types.ptrGlobal());
    const output = try func.appendBlockParam(entry, try func.types.ptrGlobal());

    const original_entry = try allocator.dupe(Inst, func.blockInsts(entry));
    defer allocator.free(original_entry);
    func.blockInstsMut(entry).clearRetainingCapacity();
    for (original_params) |param| {
        if (hasGpuKey(func, param, "grad_buf")) {
            try func.addAttr(.{ .value = param }, .{ .custom = .{
                .namespace = "vulcan.gpu",
                .key = "lowered_graphics_input",
                .value = .flag,
            } });
            const stride = try func.appendInst(entry, index_type, .{ .iconst = fragment_record_stride });
            const row = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .mul, .lhs = fragment_index, .rhs = stride } });
            const offset = try func.appendInst(entry, index_type, .{ .iconst = fragment_input_gradient_offset });
            const byte_offset = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = offset } });
            const source = try func.appendInst(entry, func.valueType(input), .{ .arith = .{ .op = .add, .lhs = input, .rhs = byte_offset } });
            func.replaceAllUses(param, source);
            continue;
        }
        if (hasGpuKey(func, param, "discard_fn")) {
            try func.addAttr(.{ .value = param }, .{ .custom = .{
                .namespace = "vulcan.gpu",
                .key = "lowered_graphics_input",
                .value = .flag,
            } });
            continue;
        }
        const offset = if (gpuIntAttr(func, param, "attr")) |slot|
            slot
        else if (gpu.attrs.builtinOf(func, param)) |builtin| switch (builtin) {
            .frag_coord => fragment_input_coord_offset + 4 * (gpuIntAttr(func, param, "bicomp") orelse 0),
            .point_coord => fragment_input_point_coord_offset + 4 * (gpuIntAttr(func, param, "bicomp") orelse 0),
            .front_facing => fragment_input_front_facing_offset,
            else => continue,
        } else continue;
        try func.addAttr(.{ .value = param }, .{ .custom = .{
            .namespace = "vulcan.gpu",
            .key = "lowered_graphics_input",
            .value = .flag,
        } });
        const stride = try func.appendInst(entry, index_type, .{ .iconst = fragment_record_stride });
        const row = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .mul, .lhs = fragment_index, .rhs = stride } });
        const slot = try func.appendInst(entry, index_type, .{ .iconst = offset });
        const byte_offset = try func.appendInst(entry, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = slot } });
        const source = try func.appendInst(entry, func.valueType(input), .{ .arith = .{ .op = .add, .lhs = input, .rhs = byte_offset } });
        const value = try func.appendInst(entry, func.valueType(param), .{ .load = .{ .ptr = source } });
        func.replaceAllUses(param, value);
    }
    try func.blockInstsMut(entry).appendSlice(func.allocator, original_entry);

    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation == .call_indirect and hasGpuKey(func, operation.call_indirect.target, "discard_fn")) {
                const stride = try func.appendInst(block, index_type, .{ .iconst = fragment_record_stride });
                const row = try func.appendInst(block, index_type, .{ .arith = .{ .op = .mul, .lhs = fragment_index, .rhs = stride } });
                const offset = try func.appendInst(block, index_type, .{ .iconst = fragment_output_discard_offset });
                const byte_offset = try func.appendInst(block, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = offset } });
                const destination = try func.appendInst(block, func.valueType(output), .{ .arith = .{ .op = .add, .lhs = output, .rhs = byte_offset } });
                const discarded = try func.appendInst(block, index_type, .{ .iconst = 1 });
                try func.appendStore(block, discarded, destination);
                continue;
            }
            if (operation != .store) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const store = operation.store;
            const offset = if (gpuIntAttr(func, store.ptr, "color_out")) |component|
                if (component < 16) component * 4 else return error.Unsupported
            else if (gpuIntAttr(func, store.ptr, "frag_depth") != null)
                fragment_output_depth_offset
            else {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            };
            const stride = try func.appendInst(block, index_type, .{ .iconst = fragment_record_stride });
            const row = try func.appendInst(block, index_type, .{ .arith = .{ .op = .mul, .lhs = fragment_index, .rhs = stride } });
            const slot = try func.appendInst(block, index_type, .{ .iconst = offset });
            const byte_offset = try func.appendInst(block, index_type, .{ .arith = .{ .op = .add, .lhs = row, .rhs = slot } });
            const destination = try func.appendInst(block, func.valueType(output), .{ .arith = .{ .op = .add, .lhs = output, .rhs = byte_offset } });
            if (store.mask) |mask| {
                try func.appendStoreMaskedVol(block, store.value, destination, mask, store.@"volatile");
            } else {
                try func.appendStoreVol(block, store.value, destination, store.@"volatile");
            }
        }
    }
}

/// The framebuffer byte a constant colour component holds, or null when the component is per lane.
///
/// This has to agree with the clamp and the convert the per-lane path emits, and
/// `vfcvt.rtz.xu.f.v` truncates toward zero.
fn constantColourByte(func: *const Function, value: Value) ?u32 {
    const inst = func.definingInst(value) orelse return null;
    const constant = switch (func.opcode(inst)) {
        .fconst => |x| x,
        else => return null,
    };
    if (std.math.isNan(constant)) return 0;
    return @intFromFloat(@min(1.0, @max(0.0, constant)) * 255.0);
}

/// A screen coordinate as a float, for a core that decodes no integer-to-f32 convert.
///
/// The value goes in the mantissa of 2 raised to the 23rd and a subtraction takes the exponent
/// away again, which is exact for every input below that. A screen coordinate is far below it. An
/// f16 destination takes the convert the core does have.
fn smallIntToFloat(func: *Function, block: Block, value: Value, ty: Type, u32_t: Type) Error!Value {
    if (func.types.type_kind(ty).float != .f32) {
        const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
        const signed = try func.appendInst(block, i32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
        return func.appendInst(block, ty, .{ .convert = .{ .value = signed } });
    }
    const biased = try func.appendArithImm(block, u32_t, .bit_or, value, 0x4B000000);
    const mantissa = try func.appendInst(block, ty, .{ .unary = .{ .op = .reinterpret, .value = biased } });
    const exponent = try func.appendInst(block, ty, .{ .fconst = 8388608 });
    return func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = mantissa, .rhs = exponent } });
}

/// Where the generated rasteriser reads a vertex, inside the vertex output record. The vertex
/// stage writes attribute slot 0 and slot 4, which is where `lowerVertexIO` puts them, and both
/// hold SCREEN SPACE integers: a viewport transform belongs to the vertex stage.
pub const raster_vertex_x_offset = 0;
pub const raster_vertex_y_offset = 4;

/// Where each colour component sits in a framebuffer word. Red is the third byte and blue is the
/// first, the way the display reads them, and alpha goes in the byte above, which it ignores.
const raster_colour_shift: [4]i64 = .{ 16, 8, 0, 24 };

/// Wrap a fragment body in a generated rasteriser, so the core walks the triangle itself.
///
/// `lowerFragmentIO` needs the host to write one fragment record per pixel. A screen-filling
/// triangle at 1080p is two million records and the link moves one word per transaction, so the
/// host cannot feed that path at a useful rate. The core can: each edge of the triangle has an
/// integer edge function, the sign of a cross product, and a pixel is covered when all three are
/// non-negative. One pixel in x adds a constant to each edge and one row adds another, so the
/// edges are stepped rather than evaluated and the per-pixel path holds no multiply at all.
///
/// The loop takes five launch arguments after the shader's own: the framebuffer, the vertex
/// buffer, the width and height to walk, and the framebuffer pitch in bytes. A lane owns one
/// pixel and the warp steps along a row, so the pixels past the end of a short final step drop out
/// with the coverage.
///
/// The body writes its colour as four floats, which this packs, or as one 32-bit word, which goes
/// straight to the framebuffer.
///
/// ONE BLOCK for now. A body with control flow of its own has to be if-converted into the span,
/// which is the next step.
fn lowerFragmentRaster(allocator: std.mem.Allocator, func: *Function, why: ?*Refusal) Error!void {
    if (!hasGpuStage(func, "fragment")) {
        return Refusal.whole(why, "a rasteriser around a function that is not a fragment stage");
    }
    if (func.blockCount() != 1) {
        return Refusal.whole(why, "a rasteriser around a fragment body with control flow of its own");
    }
    const entry: Block = @fromBackingInt(0);
    switch (func.terminator(entry) orelse return Refusal.whole(why, "a fragment body with no terminator")) {
        .ret => |r| if (r.count != 0) return Refusal.whole(why, "a fragment body that returns a value"),
        .jump => return Refusal.whole(why, "a fragment body whose only block jumps"),
    }

    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);

    // Everything this lowering cannot do, named before the function is touched, so a refusal
    // leaves the caller's clone the way it arrived.
    const original_params = try allocator.dupe(Value, func.blockParams(entry));
    defer allocator.free(original_params);
    for (original_params) |param| {
        if (isSamplerFunction(func, param)) continue;
        if (hasGpuKey(func, param, "grad_buf")) {
            return Refusal.whole(why, "a rasteriser for a fragment body that reads derivatives");
        }
        if (hasGpuKey(func, param, "discard_fn")) {
            return Refusal.whole(why, "a rasteriser for a fragment body that discards");
        }
        if (gpuIntAttr(func, param, "attr") != null) {
            return Refusal.whole(why, "a rasteriser for a fragment body that reads an interpolated attribute");
        }
        if (gpu.attrs.builtinOf(func, param)) |builtin| switch (builtin) {
            .frag_coord => if ((gpuIntAttr(func, param, "bicomp") orelse 0) > 1) {
                return Refusal.whole(why, "a rasteriser that interpolates depth or the perspective divide");
            },
            else => return Refusal.whole(why, "a rasteriser for a fragment body that reads a builtin other than the coordinate"),
        };
    }
    var written: [4]bool = @splat(false);
    var one_word = false;
    for (func.blockInsts(entry), 0..) |inst, index| {
        const operation = func.opcode(inst);
        if (operation != .store) continue;
        const store = operation.store;
        if (gpuIntAttr(func, store.ptr, "frag_depth") != null) {
            return Refusal.at(why, func, entry, index, inst, "a rasteriser for a fragment body that writes depth");
        }
        const component = gpuIntAttr(func, store.ptr, "color_out") orelse continue;
        if (component > 3 or store.mask != null or written[component]) {
            return Refusal.at(why, func, entry, index, inst, "a rasteriser for a colour the fragment body does not write once per component");
        }
        written[component] = true;
        switch (func.types.type_kind(func.valueType(store.value))) {
            .int => |i| {
                if (component != 0 or i.bits != 32) {
                    return Refusal.at(why, func, entry, index, inst, "a rasteriser for an integer colour that is not one whole word");
                }
                one_word = true;
            },
            .float => |f| if (f != .f32) {
                return Refusal.at(why, func, entry, index, inst, "a rasteriser for a colour component that is not f32");
            },
            else => return Refusal.at(why, func, entry, index, inst, "a rasteriser for a colour that is neither a word nor four floats"),
        }
    }
    var components: usize = 0;
    for (written) |one| components += @intFromBool(one);
    if (components != @as(usize, if (one_word) 1 else 4)) {
        return Refusal.whole(why, "a rasteriser for a fragment body that writes neither one word nor four colour components");
    }

    // The body moves into the span, which leaves the entry block free to become the setup.
    const body = try allocator.dupe(Inst, func.blockInsts(entry));
    defer allocator.free(body);
    func.blockInstsMut(entry).clearRetainingCapacity();

    const lane = try func.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(func, lane, .thread_id_x);
    const lanes = try func.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(func, lanes, .block_dim_x);
    const ptr_t = try func.types.ptrGlobal();
    const frame = try func.appendBlockParam(entry, ptr_t);
    const vertices = try func.appendBlockParam(entry, ptr_t);
    const width = try func.appendBlockParam(entry, u32_t);
    const height = try func.appendBlockParam(entry, u32_t);
    const pitch = try func.appendBlockParam(entry, u32_t);

    const row_loop = try func.appendBlock();
    const span = try func.appendBlock();
    const next_row = try func.appendBlock();
    const done = try func.appendBlock();

    // Triangle setup. Edge i runs from vertex i to vertex i+1, and its value at (x, y) is
    //   (x - vx[i]) * dy[i] - (y - vy[i]) * dx[i]
    // so one pixel in x adds dy[i] and one row adds -dx[i].
    var vx: [3]Value = undefined;
    var vy: [3]Value = undefined;
    for (0..3) |i| {
        const slot: i64 = @intCast(i * vertex_output_stride);
        const at_x = try appendOffset(func, entry, vertices, slot + raster_vertex_x_offset, u32_t);
        const at_y = try appendOffset(func, entry, vertices, slot + raster_vertex_y_offset, u32_t);
        vx[i] = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = at_x } });
        vy[i] = try func.appendInst(entry, i32_t, .{ .load = .{ .ptr = at_y } });
    }
    var dx: [3]Value = undefined;
    var dy: [3]Value = undefined;
    for (0..3) |i| {
        const j = (i + 1) % 3;
        dx[i] = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = vx[j], .rhs = vx[i] } });
        dy[i] = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = vy[j], .rhs = vy[i] } });
    }
    // Each edge at the first pixel this lane owns, and the step for one whole warp along a row.
    const lane_signed = try func.appendInst(entry, i32_t, .{ .unary = .{ .op = .reinterpret, .value = lane } });
    const lanes_signed = try func.appendInst(entry, i32_t, .{ .unary = .{ .op = .reinterpret, .value = lanes } });
    const origin = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    var first: [3]Value = undefined;
    var step_x: [3]Value = undefined;
    for (0..3) |i| {
        const x_rel = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = lane_signed, .rhs = vx[i] } });
        const y_rel = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = origin, .rhs = vy[i] } });
        const along = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = x_rel, .rhs = dy[i] } });
        const down = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = y_rel, .rhs = dx[i] } });
        first[i] = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .sub, .lhs = along, .rhs = down } });
        step_x[i] = try func.appendInst(entry, i32_t, .{ .arith = .{ .op = .mul, .lhs = lanes_signed, .rhs = dy[i] } });
    }
    const top = try func.appendInst(entry, u32_t, .{ .iconst = 0 });
    // The span tests after it writes, so a draw with no rows has to turn back here. The width
    // needs no such test: a pixel past the end of a row drops out with the coverage.
    const rows = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = height, .rhs = top } });
    // A conditional carries both edges itself, so the block it ends may hold no terminator, and
    // the body left a `ret` on this one.
    func.terminatorPtr(entry).* = null;
    try func.appendIf(
        entry,
        rows,
        .{ .target = row_loop, .args = &.{ top, first[0], first[1], first[2], frame } },
        .{ .target = done, .args = &.{} },
    );

    // One iteration per row: the row, each edge at the row's first pixel, and the row's base.
    const y = try func.appendBlockParam(row_loop, u32_t);
    const row0 = try func.appendBlockParam(row_loop, i32_t);
    const row1 = try func.appendBlockParam(row_loop, i32_t);
    const row2 = try func.appendBlockParam(row_loop, i32_t);
    const row_base = try func.appendBlockParam(row_loop, ptr_t);
    const row_start = try func.appendInst(row_loop, u32_t, .{ .iconst = 0 });
    try func.setJump(row_loop, span, &.{ row_start, row0, row1, row2 });

    // One iteration per warp step along the row. Coverage is every edge non-negative, and the
    // lowering has `lt` and not `ge`, so it is three compares, two ORs and a NOT.
    const x = try func.appendBlockParam(span, u32_t);
    const edge0 = try func.appendBlockParam(span, i32_t);
    const edge1 = try func.appendBlockParam(span, i32_t);
    const edge2 = try func.appendBlockParam(span, i32_t);
    const zero = try func.appendInst(span, i32_t, .{ .iconst = 0 });
    const left0 = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = edge0, .rhs = zero } });
    const left1 = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = edge1, .rhs = zero } });
    const left2 = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = edge2, .rhs = zero } });
    const left01 = try func.appendInst(span, bool_t, .{ .arith = .{ .op = .bit_or, .lhs = left0, .rhs = left1 } });
    const outside = try func.appendInst(span, bool_t, .{ .arith = .{ .op = .bit_or, .lhs = left01, .rhs = left2 } });
    const inside = try func.appendArithImm(span, bool_t, .bit_xor, outside, 1);
    const px = try func.appendInst(span, u32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = lane } });
    const on_screen = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = px, .rhs = width } });
    const covered = try func.appendInst(span, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = inside, .rhs = on_screen } });
    const offset = try func.appendArithImm(span, u32_t, .mul, px, 4);
    const pixel = try func.appendInst(span, ptr_t, .{ .arith = .{ .op = .add, .lhs = row_base, .rhs = offset } });

    // The coordinate the body reads, before the body, since it is what the body is written around.
    for (original_params) |param| {
        const builtin = gpu.attrs.builtinOf(func, param) orelse continue;
        if (builtin != .frag_coord) continue;
        try func.addAttr(.{ .value = param }, .{ .custom = .{
            .namespace = "vulcan.gpu",
            .key = "lowered_graphics_input",
            .value = .flag,
        } });
        const whole = if ((gpuIntAttr(func, param, "bicomp") orelse 0) == 0) px else y;
        const ty = func.valueType(param);
        const coord = if (isFloat(func, ty))
            try smallIntToFloat(func, span, whole, ty, u32_t)
        else
            whole;
        func.replaceAllUses(param, coord);
    }

    // The body, with its colour store held back so the pixel it writes is this lane's.
    var colour: [4]Value = undefined;
    for (body) |inst| {
        const operation = func.opcode(inst);
        if (operation == .store) {
            const store = operation.store;
            if (gpuIntAttr(func, store.ptr, "color_out")) |component| {
                if (one_word) {
                    try func.appendStoreMasked(span, store.value, pixel, covered);
                } else {
                    colour[component] = store.value;
                }
                continue;
            }
        }
        try func.blockInstsMut(span).append(func.allocator, inst);
    }
    if (!one_word) {
        var constants: u32 = 0;
        var word: ?Value = null;
        var scale: ?Value = null;
        var range: ?[2]Value = null;
        for (colour, 0..) |component, i| {
            // A constant component folds here. The core has no scalar float unit, so the clamp
            // on a warp-uniform component would be a float compare it cannot do, and a colour
            // the shader states outright should cost nothing per pixel anyway.
            if (constantColourByte(func, component)) |byte| {
                constants |= byte << @intCast(raster_colour_shift[i]);
                continue;
            }
            if (range == null) {
                range = .{
                    try func.appendInst(span, f32_t, .{ .fconst = 0 }),
                    try func.appendInst(span, f32_t, .{ .fconst = 1 }),
                };
                scale = try func.appendInst(span, f32_t, .{ .fconst = 255 });
            }
            const low = range.?[0];
            const high = range.?[1];
            const under = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = component, .rhs = low } });
            const lifted = try func.appendInst(span, f32_t, .{ .select = .{ .cond = under, .then = low, .@"else" = component } });
            // The compare unit has `lt` and `le` and no `gt`, so the operands swap instead.
            const over = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = high, .rhs = lifted } });
            const clamped = try func.appendInst(span, f32_t, .{ .select = .{ .cond = over, .then = high, .@"else" = lifted } });
            const scaled = try func.appendInst(span, f32_t, .{ .arith = .{ .op = .mul, .lhs = clamped, .rhs = scale.? } });
            const whole = try func.appendInst(span, u32_t, .{ .convert = .{ .value = scaled } });
            const placed = try func.appendArithImm(span, u32_t, .shl, whole, raster_colour_shift[i]);
            word = if (word) |so_far|
                try func.appendInst(span, u32_t, .{ .arith = .{ .op = .bit_or, .lhs = so_far, .rhs = placed } })
            else
                placed;
        }
        const pixel_word = if (word) |per_lane|
            if (constants == 0) per_lane else try func.appendArithImm(span, u32_t, .bit_or, per_lane, constants)
        else
            try func.appendInst(span, u32_t, .{ .iconst = constants });
        try func.appendStoreMasked(span, pixel_word, pixel, covered);
    }
    // Step the span by one warp, and the row by one pitch.
    const next_x = try func.appendInst(span, u32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = lanes } });
    const next0 = try func.appendInst(span, i32_t, .{ .arith = .{ .op = .add, .lhs = edge0, .rhs = step_x[0] } });
    const next1 = try func.appendInst(span, i32_t, .{ .arith = .{ .op = .add, .lhs = edge1, .rhs = step_x[1] } });
    const next2 = try func.appendInst(span, i32_t, .{ .arith = .{ .op = .add, .lhs = edge2, .rhs = step_x[2] } });
    const more_x = try func.appendInst(span, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next_x, .rhs = width } });
    try func.appendIf(
        span,
        more_x,
        .{ .target = span, .args = &.{ next_x, next0, next1, next2 } },
        .{ .target = next_row, .args = &.{} },
    );

    const next_y = try func.appendArithImm(next_row, u32_t, .add, y, 1);
    const down0 = try func.appendInst(next_row, i32_t, .{ .arith = .{ .op = .sub, .lhs = row0, .rhs = dx[0] } });
    const down1 = try func.appendInst(next_row, i32_t, .{ .arith = .{ .op = .sub, .lhs = row1, .rhs = dx[1] } });
    const down2 = try func.appendInst(next_row, i32_t, .{ .arith = .{ .op = .sub, .lhs = row2, .rhs = dx[2] } });
    const next_base = try func.appendInst(next_row, ptr_t, .{ .arith = .{ .op = .add, .lhs = row_base, .rhs = pitch } });
    const more_y = try func.appendInst(next_row, bool_t, .{ .icmp = .{ .op = .lt, .lhs = next_y, .rhs = height } });
    try func.appendIf(
        next_row,
        more_y,
        .{ .target = row_loop, .args = &.{ next_y, down0, down1, down2, next_base } },
        .{ .target = done, .args = &.{} },
    );
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

fn isSamplerFunction(func: *const Function, value: Value) bool {
    return hasGpuKey(func, value, "sampler_fn") or
        hasGpuKey(func, value, "sampler_fetch_fn") or
        hasGpuKey(func, value, "sampler_fetch_array_fn") or
        hasGpuKey(func, value, "sampler_fetch_3d_fn") or
        hasGpuKey(func, value, "sampler_gather_fn") or
        hasGpuKey(func, value, "sampler_shadow_fn") or
        hasGpuKey(func, value, "sampler_cube_fn") or
        hasGpuKey(func, value, "sampler_3d_fn") or
        hasGpuKey(func, value, "sampler_2darray_fn") or
        hasGpuKey(func, value, "sampler_cube_shadow_fn") or
        hasGpuKey(func, value, "sampler_2darray_shadow_fn") or
        hasGpuKey(func, value, "storage_image_write_fn") or
        hasGpuKey(func, value, "image_size_fn");
}

fn isIgnoredEntryParam(func: *const Function, value: Value) bool {
    return isSamplerFunction(func, value) or hasGpuKey(func, value, "lowered_graphics_input");
}

fn appendOffset(func: *Function, block: Block, ptr: Value, offset: i64, int_ty: Type) Error!Value {
    if (offset == 0) return ptr;
    const amount = try func.appendInst(block, int_ty, .{ .iconst = offset });
    return func.appendInst(block, func.valueType(ptr), .{ .arith = .{ .op = .add, .lhs = ptr, .rhs = amount } });
}

fn nullableImagePointer(func: *Function, block: Block, descriptor: Value, pointer: Value, write: bool, component_offset: i64, int_ty: Type) Error!Value {
    if (!hasGpuKey(func, descriptor, "nullable_desc")) return pointer;
    const ptr_ty = func.valueType(descriptor);
    const valid_ptr = try appendOffset(func, block, descriptor, 100, int_ty);
    const valid = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = valid_ptr } });
    const zero = try func.appendInst(block, int_ty, .{ .iconst = 0 });
    const present = try func.appendInst(block, try func.types.intern(.bool), .{
        .icmp = .{ .op = .ne, .lhs = valid, .rhs = zero },
    });
    const fallback_ptr = try appendOffset(func, block, descriptor, if (write) 108 else 104, int_ty);
    var fallback = try func.appendInst(block, ptr_ty, .{ .load = .{ .ptr = fallback_ptr } });
    fallback = try appendOffset(func, block, fallback, component_offset, int_ty);
    return func.appendInst(block, ptr_ty, .{
        .select = .{ .cond = present, .then = pointer, .@"else" = fallback },
    });
}

const sampler_row_pitch_offset = 4;
const sampler_width_offset = 8;
const sampler_height_offset = 12;
const sampler_width_float_offset = 16;
const sampler_height_float_offset = 20;
const sampler_address_u_offset = 24;
const sampler_address_v_offset = 28;
const sampler_border_offset = 32;
const sampler_texel_bytes_offset = 48;
const sampler_format_offset = 52;
const sampler_width_reciprocal_offset = 56;
const sampler_height_reciprocal_offset = 60;
const sampler_mag_filter_offset = 64;
const sampler_min_filter_offset = 68;
const sampler_swizzle_offset = 72;
const sampler_compare_op_offset = 88;
const sampler_array_pitch_offset = 92;
const sampler_level_count_offset = 96;
const sampler_descriptor_size = 100;
const nullable_sampler_descriptor_size = 112;
const rgba32f_texel_bytes = 16;

const SampleCoordinate = struct {
    index: Value,
    border: Value,
};

const LinearCoordinate = struct {
    first: SampleCoordinate,
    second: SampleCoordinate,
    weight: Value,
};

const SampleImage = struct {
    desc: Value,
    base: Value,
    row_pitch: Value,
    texel_bytes: Value,
    float_format: Value,
    depth_format: Value,
    bgra_format: Value,
    integer_bias: Value,
    unorm_scale: Value,
    int_ty: Type,
    ptr_ty: Type,
    f32_t: Type,
    bool_t: Type,
};

fn lowerFloorF32(func: *Function, block: Block, value: Value) Error!Value {
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f32_t = func.valueType(value);
    const zero_f = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const one_f = try func.appendInst(block, f32_t, .{ .fconst = 1 });
    const negative_one_f = try func.appendInst(block, f32_t, .{ .fconst = -1 });
    const bits = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    const signless = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .bit_and, .lhs = bits, .imm = 0x7fff_ffff } });
    const exponent_shifted = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .shr, .lhs = signless, .imm = 23 } });
    const exponent = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .bit_and, .lhs = exponent_shifted, .imm = 0xff } });
    const bias = try func.appendInst(block, u32_t, .{ .iconst = 127 });
    const whole = try func.appendInst(block, u32_t, .{ .iconst = 150 });
    const at_least_one = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = exponent, .rhs = bias } });
    const already_whole = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = exponent, .rhs = whole } });
    const upper_bounded = try func.appendInst(block, u32_t, .{ .select = .{ .cond = already_whole, .then = whole, .@"else" = exponent } });
    const bounded_exponent = try func.appendInst(block, u32_t, .{ .select = .{ .cond = at_least_one, .then = upper_bounded, .@"else" = bias } });
    const shift = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .sub, .lhs = whole, .rhs = bounded_exponent } });
    const one = try func.appendInst(block, u32_t, .{ .iconst = 1 });
    const mask_top = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .shl, .lhs = one, .rhs = shift } });
    const mask = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .sub, .lhs = mask_top, .imm = 1 } });
    const inverse_mask = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .bit_xor, .lhs = mask, .imm = 0xffff_ffff } });
    const truncated_bits = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_and, .lhs = bits, .rhs = inverse_mask } });
    const truncated = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = truncated_bits } });
    const fraction_bits = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_and, .lhs = signless, .rhs = mask } });
    const zero_u = try func.appendInst(block, u32_t, .{ .iconst = 0 });
    const has_fraction = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ne, .lhs = fraction_bits, .rhs = zero_u } });
    const negative = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = value, .rhs = zero_f } });
    const negative_fraction = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = negative, .rhs = has_fraction } });
    const lower = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = truncated, .rhs = one_f } });
    const middle = try func.appendInst(block, f32_t, .{ .select = .{ .cond = negative_fraction, .then = lower, .@"else" = truncated } });
    const nonzero = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ne, .lhs = signless, .rhs = zero_u } });
    const negative_nonzero = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = negative, .rhs = nonzero } });
    const small = try func.appendInst(block, f32_t, .{ .select = .{ .cond = negative_nonzero, .then = negative_one_f, .@"else" = zero_f } });
    const finite_floor = try func.appendInst(block, f32_t, .{ .select = .{ .cond = at_least_one, .then = middle, .@"else" = small } });
    return func.appendInst(block, f32_t, .{ .select = .{ .cond = already_whole, .then = value, .@"else" = finite_floor } });
}

fn lowerSampleCoordinate(func: *Function, block: Block, desc: Value, coord: Value, size_offset: i64, size_float_offset: i64, mode_offset: i64) Error!SampleCoordinate {
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f32_t = func.valueType(coord);
    const size_ptr = try appendOffset(func, block, desc, size_offset, u32_t);
    const size = try func.appendInst(block, u32_t, .{ .load = .{ .ptr = size_ptr } });
    const size_float_ptr = try appendOffset(func, block, desc, size_float_offset, u32_t);
    const size_float = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = size_float_ptr } });
    const zero_f = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const one_f = try func.appendInst(block, f32_t, .{ .fconst = 1 });
    const two_f = try func.appendInst(block, f32_t, .{ .fconst = 2 });
    const half_f = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
    const negative = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = coord, .rhs = zero_f } });
    const past_end = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = coord, .rhs = one_f } });
    const border = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_or, .lhs = negative, .rhs = past_end } });
    const floor = try lowerFloorF32(func, block, coord);
    const repeat = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = coord, .rhs = floor } });
    const half_coord = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = coord, .rhs = half_f } });
    const half_floor = try lowerFloorF32(func, block, half_coord);
    const period_base = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = half_floor, .rhs = two_f } });
    const phase = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = coord, .rhs = period_base } });
    const mirror_second_half = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = phase, .rhs = one_f } });
    const reflected = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = two_f, .rhs = phase } });
    const mirrored = try func.appendInst(block, f32_t, .{ .select = .{ .cond = mirror_second_half, .then = reflected, .@"else" = phase } });
    const low_clamped = try func.appendInst(block, f32_t, .{ .select = .{ .cond = negative, .then = zero_f, .@"else" = coord } });
    const clamped = try func.appendInst(block, f32_t, .{ .select = .{ .cond = past_end, .then = one_f, .@"else" = low_clamped } });

    const mode_ptr = try appendOffset(func, block, desc, mode_offset, u32_t);
    const mode = try func.appendInst(block, u32_t, .{ .load = .{ .ptr = mode_ptr } });
    const zero_u = try func.appendInst(block, u32_t, .{ .iconst = 0 });
    const repeat_mode = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = mode, .rhs = zero_u } });
    const mirror_value = try func.appendInst(block, u32_t, .{ .iconst = 1 });
    const mirror_mode = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = mode, .rhs = mirror_value } });
    const border_value = try func.appendInst(block, u32_t, .{ .iconst = 3 });
    const border_mode = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = mode, .rhs = border_value } });
    const repeat_or_clamp = try func.appendInst(block, f32_t, .{ .select = .{ .cond = repeat_mode, .then = repeat, .@"else" = clamped } });
    const addressed = try func.appendInst(block, f32_t, .{ .select = .{ .cond = mirror_mode, .then = mirrored, .@"else" = repeat_or_clamp } });
    const scaled = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = addressed, .rhs = size_float } });
    const converted = try func.appendInst(block, u32_t, .{ .convert = .{ .value = scaled } });
    const one_u = try func.appendInst(block, u32_t, .{ .iconst = 1 });
    const size_last = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .sub, .lhs = size, .rhs = one_u } });
    const index_past_end = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = converted, .rhs = size } });
    const index = try func.appendInst(block, u32_t, .{ .select = .{ .cond = index_past_end, .then = size_last, .@"else" = converted } });
    const no_border = try func.appendInst(block, bool_t, .{ .iconst = 0 });
    const effective_border = try func.appendInst(block, bool_t, .{ .select = .{ .cond = border_mode, .then = border, .@"else" = no_border } });
    return .{ .index = index, .border = effective_border };
}

fn lowerLinearCoordinate(func: *Function, block: Block, desc: Value, coord: Value, size_offset: i64, size_float_offset: i64, reciprocal_offset: i64, mode_offset: i64) Error!LinearCoordinate {
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = func.valueType(coord);
    const size_ptr = try appendOffset(func, block, desc, size_float_offset, u32_t);
    const size = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = size_ptr } });
    const reciprocal_ptr = try appendOffset(func, block, desc, reciprocal_offset, u32_t);
    const reciprocal = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = reciprocal_ptr } });
    const half = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
    const scaled = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = coord, .rhs = size } });
    const centered = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = scaled, .rhs = half } });
    const floor = try lowerFloorF32(func, block, centered);
    const weight = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = centered, .rhs = floor } });
    const first_center = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = floor, .rhs = half } });
    const first_normalized = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = first_center, .rhs = reciprocal } });
    const second_normalized = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = first_normalized, .rhs = reciprocal } });
    return .{
        .first = try lowerSampleCoordinate(func, block, desc, first_normalized, size_offset, size_float_offset, mode_offset),
        .second = try lowerSampleCoordinate(func, block, desc, second_normalized, size_offset, size_float_offset, mode_offset),
        .weight = weight,
    };
}

fn selectSampleCoordinate(func: *Function, block: Block, linear: Value, filtered: SampleCoordinate, nearest: SampleCoordinate) Error!SampleCoordinate {
    return .{
        .index = try func.appendInst(block, func.valueType(filtered.index), .{ .select = .{ .cond = linear, .then = filtered.index, .@"else" = nearest.index } }),
        .border = try func.appendInst(block, func.valueType(filtered.border), .{ .select = .{ .cond = linear, .then = filtered.border, .@"else" = nearest.border } }),
    };
}

fn lowerSampleTexel(func: *Function, block: Block, image: SampleImage, x: SampleCoordinate, y: SampleCoordinate) Error![4]Value {
    const outside = try func.appendInst(block, image.bool_t, .{ .arith = .{ .op = .bit_or, .lhs = x.border, .rhs = y.border } });
    const row = try func.appendInst(block, image.int_ty, .{ .arith = .{ .op = .mul, .lhs = y.index, .rhs = image.row_pitch } });
    const column = try func.appendInst(block, image.int_ty, .{ .arith = .{ .op = .mul, .lhs = x.index, .rhs = image.texel_bytes } });
    const byte_offset = try func.appendInst(block, image.int_ty, .{ .arith = .{ .op = .add, .lhs = row, .rhs = column } });
    const texel = try func.appendInst(block, image.ptr_ty, .{ .arith = .{ .op = .add, .lhs = image.base, .rhs = byte_offset } });
    const packed_texel = try func.appendInst(block, image.int_ty, .{ .load = .{ .ptr = texel } });
    const zero = try func.appendInst(block, image.int_ty, .{ .iconst = 0 });
    var values: [4]Value = undefined;
    for (0..4) |component| {
        const offset: i64 = @intCast(component * 4);
        const float_offset = try func.appendInst(block, image.int_ty, .{ .iconst = offset });
        const safe_float_offset = try func.appendInst(block, image.int_ty, .{ .select = .{ .cond = image.float_format, .then = float_offset, .@"else" = zero } });
        const source = try func.appendInst(block, image.ptr_ty, .{ .arith = .{ .op = .add, .lhs = texel, .rhs = safe_float_offset } });
        const float_value = try func.appendInst(block, image.f32_t, .{ .load = .{ .ptr = source } });
        const rgba_shift = try func.appendInst(block, image.int_ty, .{ .iconst = @as(i64, @intCast(component * 8)) });
        const bgra_shifts = [4]i64{ 16, 8, 0, 24 };
        const bgra_shift = try func.appendInst(block, image.int_ty, .{ .iconst = bgra_shifts[component] });
        const shift = try func.appendInst(block, image.int_ty, .{ .select = .{ .cond = image.bgra_format, .then = bgra_shift, .@"else" = rgba_shift } });
        const shifted = try func.appendInst(block, image.int_ty, .{ .arith = .{ .op = .shr, .lhs = packed_texel, .rhs = shift } });
        const byte = try func.appendInst(block, image.int_ty, .{ .arith_imm = .{ .op = .bit_and, .lhs = shifted, .imm = 0xff } });
        const biased_bits = try func.appendInst(block, image.int_ty, .{ .arith_imm = .{ .op = .bit_or, .lhs = byte, .imm = 0x4b00_0000 } });
        const biased = try func.appendInst(block, image.f32_t, .{ .unary = .{ .op = .reinterpret, .value = biased_bits } });
        const integer_float = try func.appendInst(block, image.f32_t, .{ .arith = .{ .op = .sub, .lhs = biased, .rhs = image.integer_bias } });
        const unorm_value = try func.appendInst(block, image.f32_t, .{ .arith = .{ .op = .mul, .lhs = integer_float, .rhs = image.unorm_scale } });
        const color_value = try func.appendInst(block, image.f32_t, .{ .select = .{ .cond = image.float_format, .then = float_value, .@"else" = unorm_value } });
        const depth_default = try func.appendInst(block, image.f32_t, .{ .fconst = if (component == 3) 1 else 0 });
        const depth_value = if (component == 0) float_value else depth_default;
        const value = try func.appendInst(block, image.f32_t, .{ .select = .{ .cond = image.depth_format, .then = depth_value, .@"else" = color_value } });
        const border_ptr = try appendOffset(func, block, image.desc, sampler_border_offset + offset, image.int_ty);
        const border = try func.appendInst(block, image.f32_t, .{ .load = .{ .ptr = border_ptr } });
        values[component] = try func.appendInst(block, image.f32_t, .{ .select = .{ .cond = outside, .then = border, .@"else" = value } });
    }
    return values;
}

fn lowerShadowCompare(func: *Function, block: Block, compare_op: Value, reference: Value, depth: Value) Error!Value {
    const int_ty = func.valueType(compare_op);
    const bool_t = try func.types.intern(.bool);
    const f32_t = func.valueType(reference);
    const zero = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const one = try func.appendInst(block, f32_t, .{ .fconst = 1 });
    var result = zero;
    const relations = [_]ir.function.CmpOp{ .lt, .eq, .le, .gt, .ne, .ge };
    for (relations, 1..) |relation, code| {
        const expected = try func.appendInst(block, int_ty, .{ .iconst = @intCast(code) });
        const selected = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = compare_op, .rhs = expected } });
        const compared = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = relation, .lhs = reference, .rhs = depth } });
        const comparison = try func.appendInst(block, f32_t, .{ .select = .{ .cond = compared, .then = one, .@"else" = zero } });
        result = try func.appendInst(block, f32_t, .{ .select = .{ .cond = selected, .then = comparison, .@"else" = result } });
    }
    const always_code = try func.appendInst(block, int_ty, .{ .iconst = 7 });
    const always = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = compare_op, .rhs = always_code } });
    return func.appendInst(block, f32_t, .{ .select = .{ .cond = always, .then = one, .@"else" = result } });
}

const CubeCoordinate = struct {
    u: Value,
    v: Value,
    face: Value,
};

fn lowerCubeCoordinate(func: *Function, block: Block, x: Value, y: Value, z: Value) Error!CubeCoordinate {
    const f32_t = func.valueType(x);
    const int_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const lane = laneValue(func) orelse return error.Unsupported;
    const lane_zero_bits = try func.appendInst(block, func.valueType(lane), .{ .arith_imm = .{ .op = .bit_and, .lhs = lane, .imm = 0 } });
    const lane_zero = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = lane_zero_bits } });
    const lane_x = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = lane_zero } });
    const lane_y = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = y, .rhs = lane_zero } });
    const lane_z = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = z, .rhs = lane_zero } });
    const zero = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const half = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
    const one = try func.appendInst(block, f32_t, .{ .fconst = 1 });
    const negative_x = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = lane_x } });
    const negative_y = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = lane_y } });
    const negative_z = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = lane_z } });
    const x_positive = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = lane_x, .rhs = zero } });
    const y_positive = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = lane_y, .rhs = zero } });
    const z_positive = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = lane_z, .rhs = zero } });
    const ax = try func.appendInst(block, f32_t, .{ .select = .{ .cond = x_positive, .then = lane_x, .@"else" = negative_x } });
    const ay = try func.appendInst(block, f32_t, .{ .select = .{ .cond = y_positive, .then = lane_y, .@"else" = negative_y } });
    const az = try func.appendInst(block, f32_t, .{ .select = .{ .cond = z_positive, .then = lane_z, .@"else" = negative_z } });
    const x_over_y = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = ax, .rhs = ay } });
    const x_over_z = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = ax, .rhs = az } });
    const x_major = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = x_over_y, .rhs = x_over_z } });
    const y_major = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = ay, .rhs = az } });

    const x_sc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = x_positive, .then = negative_z, .@"else" = lane_z } });
    const x_tc = negative_y;
    const y_sc = lane_x;
    const y_tc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = y_positive, .then = lane_z, .@"else" = negative_z } });
    const z_sc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = z_positive, .then = lane_x, .@"else" = negative_x } });
    const z_tc = negative_y;
    const yz_sc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = y_major, .then = y_sc, .@"else" = z_sc } });
    const yz_tc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = y_major, .then = y_tc, .@"else" = z_tc } });
    const yz_major = try func.appendInst(block, f32_t, .{ .select = .{ .cond = y_major, .then = ay, .@"else" = az } });
    const sc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = x_major, .then = x_sc, .@"else" = yz_sc } });
    const tc = try func.appendInst(block, f32_t, .{ .select = .{ .cond = x_major, .then = x_tc, .@"else" = yz_tc } });
    const major = try func.appendInst(block, f32_t, .{ .select = .{ .cond = x_major, .then = ax, .@"else" = yz_major } });
    const sc_normalized = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = sc, .rhs = major } });
    const tc_normalized = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = tc, .rhs = major } });
    const u_shifted = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = sc_normalized, .rhs = one } });
    const v_shifted = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = tc_normalized, .rhs = one } });

    const x_positive_face = try func.appendInst(block, int_ty, .{ .iconst = 0 });
    const x_negative_face = try func.appendInst(block, int_ty, .{ .iconst = 1 });
    const y_positive_face = try func.appendInst(block, int_ty, .{ .iconst = 2 });
    const y_negative_face = try func.appendInst(block, int_ty, .{ .iconst = 3 });
    const z_positive_face = try func.appendInst(block, int_ty, .{ .iconst = 4 });
    const z_negative_face = try func.appendInst(block, int_ty, .{ .iconst = 5 });
    const x_face = try func.appendInst(block, int_ty, .{ .select = .{ .cond = x_positive, .then = x_positive_face, .@"else" = x_negative_face } });
    const y_face = try func.appendInst(block, int_ty, .{ .select = .{ .cond = y_positive, .then = y_positive_face, .@"else" = y_negative_face } });
    const z_face = try func.appendInst(block, int_ty, .{ .select = .{ .cond = z_positive, .then = z_positive_face, .@"else" = z_negative_face } });
    const yz_face = try func.appendInst(block, int_ty, .{ .select = .{ .cond = y_major, .then = y_face, .@"else" = z_face } });
    return .{
        .u = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = u_shifted, .rhs = half } }),
        .v = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = v_shifted, .rhs = half } }),
        .face = try func.appendInst(block, int_ty, .{ .select = .{ .cond = x_major, .then = x_face, .@"else" = yz_face } }),
    };
}

fn lerp(func: *Function, block: Block, from: Value, to: Value, amount: Value) Error!Value {
    const difference = try func.appendInst(block, func.valueType(from), .{ .arith = .{ .op = .sub, .lhs = to, .rhs = from } });
    const scaled = try func.appendInst(block, func.valueType(from), .{ .arith = .{ .op = .mul, .lhs = difference, .rhs = amount } });
    return func.appendInst(block, func.valueType(from), .{ .arith = .{ .op = .add, .lhs = from, .rhs = scaled } });
}

fn lowerSampleSwizzle(func: *Function, block: Block, desc: Value, input: [4]Value) Error![4]Value {
    const int_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const f32_t = func.valueType(input[0]);
    const zero = try func.appendInst(block, f32_t, .{ .fconst = 0 });
    const one = try func.appendInst(block, f32_t, .{ .fconst = 1 });
    var output: [4]Value = undefined;
    for (0..4) |component| {
        const swizzle_ptr = try appendOffset(func, block, desc, sampler_swizzle_offset + @as(i64, @intCast(component * 4)), int_ty);
        const swizzle = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = swizzle_ptr } });
        var value = input[component];
        for (0..4) |source| {
            const source_code = try func.appendInst(block, int_ty, .{ .iconst = @as(i64, @intCast(source + 3)) });
            const select_source = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = swizzle, .rhs = source_code } });
            value = try func.appendInst(block, f32_t, .{ .select = .{ .cond = select_source, .then = input[source], .@"else" = value } });
        }
        const zero_code = try func.appendInst(block, int_ty, .{ .iconst = 1 });
        const select_zero = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = swizzle, .rhs = zero_code } });
        value = try func.appendInst(block, f32_t, .{ .select = .{ .cond = select_zero, .then = zero, .@"else" = value } });
        const one_code = try func.appendInst(block, int_ty, .{ .iconst = 2 });
        const select_one = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = swizzle, .rhs = one_code } });
        output[component] = try func.appendInst(block, f32_t, .{ .select = .{ .cond = select_one, .then = one, .@"else" = value } });
    }
    return output;
}

fn lowerInverseSquareRoot(func: *Function, block: Block, value: Value) Error!Value {
    const f32_t = func.valueType(value);
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bits = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    const half_bits = try func.appendInst(block, u32_t, .{ .arith_imm = .{ .op = .shr, .lhs = bits, .imm = 1 } });
    const magic = try func.appendInst(block, u32_t, .{ .iconst = 0x5f375a86 });
    const seed_bits = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .sub, .lhs = magic, .rhs = half_bits } });
    var estimate = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = seed_bits } });
    const half = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
    const one_and_half = try func.appendInst(block, f32_t, .{ .fconst = 1.5 });
    const value_half = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = value, .rhs = half } });
    for (0..3) |_| {
        const squared = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = estimate, .rhs = estimate } });
        const scaled = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = value_half, .rhs = squared } });
        const correction = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .sub, .lhs = one_and_half, .rhs = scaled } });
        estimate = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .mul, .lhs = estimate, .rhs = correction } });
    }
    return estimate;
}

fn lowerSquareRoot(allocator: std.mem.Allocator, func: *Function) Error!void {
    for (0..func.blockCount()) |block_index| {
        const block: Block = @fromBackingInt(@intCast(block_index));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation != .unary or operation.unary.op != .sqrt) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const inverse = try lowerInverseSquareRoot(func, block, operation.unary.value);
            const result = try func.appendInst(block, func.valueType(operation.unary.value), .{ .arith = .{
                .op = .mul,
                .lhs = operation.unary.value,
                .rhs = inverse,
            } });
            func.replaceAllUses(func.instResult(inst) orelse return error.Unsupported, result);
        }
    }
}

fn lowerImageSample(allocator: std.mem.Allocator, func: *Function) Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var sampled = std.AutoHashMapUnmanaged(Value, [4]Value){};
        defer sampled.deinit(allocator);
        const Element = struct { base: Value, component: u2 };
        var elements = std.AutoHashMapUnmanaged(Value, Element){};
        defer elements.deinit(allocator);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation == .arith and operation.arith.op == .add) {
                if (elements.get(operation.arith.lhs)) |element| {
                    if (func.definingInst(operation.arith.rhs)) |offset_inst| {
                        const offset = func.opcode(offset_inst);
                        if (offset == .iconst and offset.iconst >= 0 and offset.iconst <= 12 and @mod(offset.iconst, 4) == 0) {
                            const result = func.instResult(inst).?;
                            try elements.put(allocator, result, .{ .base = element.base, .component = @intCast(@divExact(offset.iconst, 4)) });
                            continue;
                        }
                    }
                }
            }
            if (operation == .load) {
                if (elements.get(operation.load.ptr)) |element| {
                    const values = sampled.get(element.base) orelse return error.Unsupported;
                    func.replaceAllUses(func.instResult(inst).?, values[element.component]);
                    continue;
                }
            }
            const regular_sample = operation == .call_indirect and hasGpuKey(func, operation.call_indirect.target, "sampler_fn");
            const array_sample = operation == .call_indirect and hasGpuKey(func, operation.call_indirect.target, "sampler_2darray_fn");
            const cube_sample = operation == .call_indirect and hasGpuKey(func, operation.call_indirect.target, "sampler_cube_fn");
            const shadow_sample = operation == .call_indirect and hasGpuKey(func, operation.call_indirect.target, "sampler_shadow_fn");
            if (!regular_sample and !array_sample and !cube_sample and !shadow_sample) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const args = func.valueList(operation.call_indirect.args);
            const expected_arg_count: usize = if (array_sample or cube_sample) 6 else 5;
            if (args.len != expected_arg_count) return error.Unsupported;
            var desc = args[0];
            const nullable = hasGpuKey(func, desc, "nullable_desc");
            const int_ty = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
            const bool_t = try func.types.intern(.bool);
            const ptr_ty = func.valueType(desc);
            const f32_t = func.valueType(args[1]);
            const lod_index: usize = if (array_sample or cube_sample) 4 else 3;
            const lod = args[lod_index];
            const zero_f = try func.appendInst(block, f32_t, .{ .fconst = 0 });
            const constant_lod: ?f64 = if (func.definingInst(lod)) |lod_inst| blk: {
                const lod_op = func.opcode(lod_inst);
                break :blk if (lod_op == .fconst) lod_op.fconst else null;
            } else null;
            const base_lod = if (constant_lod) |value| value <= 0 else false;
            if (!base_lod) {
                const requested_level = if (constant_lod) |value|
                    try func.appendInst(block, int_ty, .{ .iconst = @intFromFloat(@floor(value + 0.5)) })
                else blk: {
                    const lane = laneValue(func) orelse return error.Unsupported;
                    const lane_zero_bits = try func.appendInst(block, func.valueType(lane), .{ .arith_imm = .{ .op = .bit_and, .lhs = lane, .imm = 0 } });
                    const lane_zero = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = lane_zero_bits } });
                    const lane_lod = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = lod, .rhs = lane_zero } });
                    const negative_lod = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = lane_lod, .rhs = zero_f } });
                    const nonnegative_lod = try func.appendInst(block, f32_t, .{ .select = .{ .cond = negative_lod, .then = zero_f, .@"else" = lane_lod } });
                    const half_f = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
                    const rounded_lod = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = nonnegative_lod, .rhs = half_f } });
                    break :blk try func.appendInst(block, int_ty, .{ .convert = .{ .value = rounded_lod } });
                };
                const count_ptr = try appendOffset(func, block, desc, sampler_level_count_offset, int_ty);
                const count = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = count_ptr } });
                const one_u = try func.appendInst(block, int_ty, .{ .iconst = 1 });
                const last_level = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .sub, .lhs = count, .rhs = one_u } });
                const lod_past_end = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = requested_level, .rhs = count } });
                const level = try func.appendInst(block, int_ty, .{ .select = .{ .cond = lod_past_end, .then = last_level, .@"else" = requested_level } });
                const stride = try func.appendInst(block, int_ty, .{
                    .iconst = if (nullable) nullable_sampler_descriptor_size else sampler_descriptor_size,
                });
                const descriptor_offset = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = level, .rhs = stride } });
                desc = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = desc, .rhs = descriptor_offset } });
            }
            const cube = if (cube_sample) try lowerCubeCoordinate(func, block, args[1], args[2], args[3]) else null;
            const u = if (cube) |coordinate| coordinate.u else args[1];
            const v = if (cube) |coordinate| coordinate.v else args[2];
            const nearest_x = try lowerSampleCoordinate(func, block, desc, u, sampler_width_offset, sampler_width_float_offset, sampler_address_u_offset);
            const nearest_y = try lowerSampleCoordinate(func, block, desc, v, sampler_height_offset, sampler_height_float_offset, sampler_address_v_offset);
            const linear_x = try lowerLinearCoordinate(func, block, desc, u, sampler_width_offset, sampler_width_float_offset, sampler_width_reciprocal_offset, sampler_address_u_offset);
            const linear_y = try lowerLinearCoordinate(func, block, desc, v, sampler_height_offset, sampler_height_float_offset, sampler_height_reciprocal_offset, sampler_address_v_offset);
            const mag_filter_ptr = try appendOffset(func, block, desc, sampler_mag_filter_offset, int_ty);
            const mag_filter = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = mag_filter_ptr } });
            const min_filter_ptr = try appendOffset(func, block, desc, sampler_min_filter_offset, int_ty);
            const min_filter = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = min_filter_ptr } });
            const filter = if (constant_lod) |value|
                if (value > 0) min_filter else mag_filter
            else blk: {
                const minifying = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = zero_f, .rhs = lod } });
                break :blk try func.appendInst(block, int_ty, .{ .select = .{ .cond = minifying, .then = min_filter, .@"else" = mag_filter } });
            };
            const linear_value = try func.appendInst(block, int_ty, .{ .iconst = 1 });
            const linear = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = filter, .rhs = linear_value } });
            const x0 = try selectSampleCoordinate(func, block, linear, linear_x.first, nearest_x);
            const x1 = try selectSampleCoordinate(func, block, linear, linear_x.second, nearest_x);
            const y0 = try selectSampleCoordinate(func, block, linear, linear_y.first, nearest_y);
            const y1 = try selectSampleCoordinate(func, block, linear, linear_y.second, nearest_y);
            const x_weight = try func.appendInst(block, f32_t, .{ .select = .{ .cond = linear, .then = linear_x.weight, .@"else" = zero_f } });
            const y_weight = try func.appendInst(block, f32_t, .{ .select = .{ .cond = linear, .then = linear_y.weight, .@"else" = zero_f } });
            var base = try func.appendInst(block, ptr_ty, .{ .load = .{ .ptr = desc } });
            if (array_sample) {
                const half = try func.appendInst(block, f32_t, .{ .fconst = 0.5 });
                const rounded = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = args[3], .rhs = half } });
                const layer = try func.appendInst(block, int_ty, .{ .convert = .{ .value = rounded } });
                const pitch_ptr = try appendOffset(func, block, desc, sampler_array_pitch_offset, int_ty);
                const pitch = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = pitch_ptr } });
                const layer_offset = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = layer, .rhs = pitch } });
                base = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = layer_offset } });
            } else if (cube) |coordinate| {
                const pitch_ptr = try appendOffset(func, block, desc, sampler_array_pitch_offset, int_ty);
                const pitch = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = pitch_ptr } });
                const face_offset = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = coordinate.face, .rhs = pitch } });
                base = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = face_offset } });
            }
            const row_ptr = try appendOffset(func, block, desc, sampler_row_pitch_offset, int_ty);
            const row_pitch = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = row_ptr } });
            const texel_bytes_ptr = try appendOffset(func, block, desc, sampler_texel_bytes_offset, int_ty);
            const texel_bytes = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = texel_bytes_ptr } });
            const format_ptr = try appendOffset(func, block, desc, sampler_format_offset, int_ty);
            const format = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = format_ptr } });
            const float_format_value = try func.appendInst(block, int_ty, .{ .iconst = 0 });
            const float_format = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = format, .rhs = float_format_value } });
            const bgra_format_value = try func.appendInst(block, int_ty, .{ .iconst = 2 });
            const bgra_format = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = format, .rhs = bgra_format_value } });
            const depth_format_value = try func.appendInst(block, int_ty, .{ .iconst = 3 });
            const depth_format = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = format, .rhs = depth_format_value } });
            const integer_bias = try func.appendInst(block, f32_t, .{ .fconst = 8388608 });
            const unorm_scale = try func.appendInst(block, f32_t, .{ .fconst = 1.0 / 255.0 });
            const image = SampleImage{
                .desc = desc,
                .base = base,
                .row_pitch = row_pitch,
                .texel_bytes = texel_bytes,
                .float_format = float_format,
                .depth_format = depth_format,
                .bgra_format = bgra_format,
                .integer_bias = integer_bias,
                .unorm_scale = unorm_scale,
                .int_ty = int_ty,
                .ptr_ty = ptr_ty,
                .f32_t = f32_t,
                .bool_t = bool_t,
            };
            const texel00 = try lowerSampleTexel(func, block, image, x0, y0);
            const texel10 = try lowerSampleTexel(func, block, image, x1, y0);
            const texel01 = try lowerSampleTexel(func, block, image, x0, y1);
            const texel11 = try lowerSampleTexel(func, block, image, x1, y1);
            if (shadow_sample) {
                const compare_ptr = try appendOffset(func, block, desc, sampler_compare_op_offset, int_ty);
                const compare_op = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = compare_ptr } });
                const compared00 = try lowerShadowCompare(func, block, compare_op, args[4], texel00[0]);
                const compared10 = try lowerShadowCompare(func, block, compare_op, args[4], texel10[0]);
                const compared01 = try lowerShadowCompare(func, block, compare_op, args[4], texel01[0]);
                const compared11 = try lowerShadowCompare(func, block, compare_op, args[4], texel11[0]);
                const top = try lerp(func, block, compared00, compared10, x_weight);
                const bottom = try lerp(func, block, compared01, compared11, x_weight);
                var value = try lerp(func, block, top, bottom, y_weight);
                if (nullable) {
                    const valid_ptr = try appendOffset(func, block, desc, 100, int_ty);
                    const valid = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = valid_ptr } });
                    const zero_u = try func.appendInst(block, int_ty, .{ .iconst = 0 });
                    const present = try func.appendInst(block, bool_t, .{
                        .icmp = .{ .op = .ne, .lhs = valid, .rhs = zero_u },
                    });
                    value = try func.appendInst(block, f32_t, .{
                        .select = .{ .cond = present, .then = value, .@"else" = zero_f },
                    });
                }
                func.replaceAllUses(func.instResult(inst) orelse return error.Unsupported, value);
                continue;
            }
            var values: [4]Value = undefined;
            for (0..4) |component| {
                const top = try lerp(func, block, texel00[component], texel10[component], x_weight);
                const bottom = try lerp(func, block, texel01[component], texel11[component], x_weight);
                values[component] = try lerp(func, block, top, bottom, y_weight);
            }
            const out_index: usize = if (array_sample or cube_sample) 5 else 4;
            const out = args[out_index];
            try sampled.put(allocator, out, try lowerSampleSwizzle(func, block, desc, values));
            try elements.put(allocator, out, .{ .base = out, .component = 0 });
        }
    }
}

fn lowerTexelFetch(allocator: std.mem.Allocator, func: *Function) Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var fetched = std.AutoHashMapUnmanaged(Value, [4]Value){};
        defer fetched.deinit(allocator);
        const Element = struct { base: Value, component: u2 };
        var elements = std.AutoHashMapUnmanaged(Value, Element){};
        defer elements.deinit(allocator);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const op = func.opcode(inst);
            if (op == .arith) {
                const arith = op.arith;
                if (arith.op == .add) {
                    if (elements.get(arith.lhs)) |element| {
                        if (func.definingInst(arith.rhs)) |offset_inst| {
                            const offset_op = func.opcode(offset_inst);
                            if (offset_op == .iconst and offset_op.iconst >= 0 and offset_op.iconst <= 12 and @mod(offset_op.iconst, 4) == 0) {
                                const result = func.instResult(inst).?;
                                try elements.put(allocator, result, .{ .base = element.base, .component = @intCast(@divExact(offset_op.iconst, 4)) });
                                continue;
                            }
                        }
                    }
                }
            }
            if (op == .load) {
                if (elements.get(op.load.ptr)) |element| {
                    const values = fetched.get(element.base) orelse return error.Unsupported;
                    func.replaceAllUses(func.instResult(inst).?, values[element.component]);
                    continue;
                }
            }
            if (op != .call_indirect or !hasGpuKey(func, op.call_indirect.target, "sampler_fetch_fn")) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const args = func.valueList(op.call_indirect.args);
            if (args.len != 5) return error.Unsupported;
            const lod_inst = func.definingInst(args[3]) orelse return error.Unsupported;
            if (func.opcode(lod_inst) != .iconst or func.opcode(lod_inst).iconst != 0) return error.Unsupported;

            const desc = args[0];
            const x = args[1];
            const y = args[2];
            const out = args[4];
            const int_ty = func.valueType(x);
            const ptr_ty = func.valueType(desc);
            const float_ty = try func.types.intern(.{ .float = .f32 });
            const base = try func.appendInst(block, ptr_ty, .{ .load = .{ .ptr = desc } });
            const row_ptr = try appendOffset(func, block, desc, sampler_row_pitch_offset, int_ty);
            const row_pitch = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = row_ptr } });
            const row = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = y, .rhs = row_pitch } });
            const texel_bytes = try func.appendInst(block, int_ty, .{ .iconst = rgba32f_texel_bytes });
            const column = try func.appendInst(block, int_ty, .{ .arith = .{
                .op = .mul,
                .lhs = x,
                .rhs = texel_bytes,
            } });
            const offset = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .add, .lhs = row, .rhs = column } });
            const texel = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
            var values: [4]Value = undefined;
            for (0..4) |component| {
                const byte_offset: i64 = @intCast(component * 4);
                const source = try appendOffset(func, block, texel, byte_offset, int_ty);
                const read_ptr = try nullableImagePointer(func, block, desc, source, false, byte_offset, int_ty);
                values[component] = try func.appendInst(block, float_ty, .{ .load = .{ .ptr = read_ptr } });
            }
            try fetched.put(allocator, out, values);
            try elements.put(allocator, out, .{ .base = out, .component = 0 });
        }
    }
}

fn lowerStorageImageWrite(allocator: std.mem.Allocator, func: *Function) Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation != .call_indirect or !hasGpuKey(func, operation.call_indirect.target, "storage_image_write_fn")) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const args = func.valueList(operation.call_indirect.args);
            if (args.len != 7) return error.Unsupported;
            const desc = args[0];
            const int_ty = func.valueType(args[1]);
            const ptr_ty = func.valueType(desc);
            const base = try func.appendInst(block, ptr_ty, .{ .load = .{ .ptr = desc } });
            const row_ptr = try appendOffset(func, block, desc, sampler_row_pitch_offset, int_ty);
            const row_pitch = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = row_ptr } });
            const row = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = args[2], .rhs = row_pitch } });
            const texel_bytes = try func.appendInst(block, int_ty, .{ .iconst = rgba32f_texel_bytes });
            const column = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .mul, .lhs = args[1], .rhs = texel_bytes } });
            const offset = try func.appendInst(block, int_ty, .{ .arith = .{ .op = .add, .lhs = row, .rhs = column } });
            const texel = try func.appendInst(block, ptr_ty, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
            for (0..4) |component| {
                const byte_offset: i64 = @intCast(component * 4);
                const destination = try appendOffset(func, block, texel, byte_offset, int_ty);
                const write_ptr = try nullableImagePointer(func, block, desc, destination, true, byte_offset, int_ty);
                try func.appendStore(block, args[3 + component], write_ptr);
            }
        }
    }
}

fn lowerImageSize(allocator: std.mem.Allocator, func: *Function) Error!void {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var sizes = std.AutoHashMapUnmanaged(Value, [2]Value){};
        defer sizes.deinit(allocator);
        const Element = struct { base: Value, component: u1 };
        var elements = std.AutoHashMapUnmanaged(Value, Element){};
        defer elements.deinit(allocator);
        func.blockInstsMut(block).clearRetainingCapacity();
        for (original) |inst| {
            const operation = func.opcode(inst);
            if (operation == .arith and operation.arith.op == .add) {
                if (elements.get(operation.arith.lhs)) |element| {
                    if (func.definingInst(operation.arith.rhs)) |offset_inst| {
                        const offset = func.opcode(offset_inst);
                        if (offset == .iconst and offset.iconst == 4) {
                            const result = func.instResult(inst).?;
                            try elements.put(allocator, result, .{ .base = element.base, .component = 1 });
                            continue;
                        }
                    }
                }
            }
            if (operation == .load) {
                if (elements.get(operation.load.ptr)) |element| {
                    const values = sizes.get(element.base) orelse return error.Unsupported;
                    func.replaceAllUses(func.instResult(inst).?, values[element.component]);
                    continue;
                }
            }
            if (operation != .call_indirect or !hasGpuKey(func, operation.call_indirect.target, "image_size_fn")) {
                try func.blockInstsMut(block).append(func.allocator, inst);
                continue;
            }
            const args = func.valueList(operation.call_indirect.args);
            if (args.len != 3) return error.Unsupported;
            const lod_inst = func.definingInst(args[1]) orelse return error.Unsupported;
            if (func.opcode(lod_inst) != .iconst or func.opcode(lod_inst).iconst != 0) return error.Unsupported;
            const desc = args[0];
            const out = args[2];
            const int_ty = func.valueType(args[1]);
            const width_ptr = try appendOffset(func, block, desc, 8, int_ty);
            const height_ptr = try appendOffset(func, block, desc, 12, int_ty);
            var width = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = width_ptr } });
            var height = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = height_ptr } });
            if (hasGpuKey(func, desc, "nullable_desc")) {
                const valid_ptr = try appendOffset(func, block, desc, 100, int_ty);
                const valid = try func.appendInst(block, int_ty, .{ .load = .{ .ptr = valid_ptr } });
                const zero = try func.appendInst(block, int_ty, .{ .iconst = 0 });
                const present = try func.appendInst(block, try func.types.intern(.bool), .{
                    .icmp = .{ .op = .ne, .lhs = valid, .rhs = zero },
                });
                width = try func.appendInst(block, int_ty, .{
                    .select = .{ .cond = present, .then = width, .@"else" = zero },
                });
                height = try func.appendInst(block, int_ty, .{
                    .select = .{ .cond = present, .then = height, .@"else" = zero },
                });
            }
            try sizes.put(allocator, out, .{ width, height });
            try elements.put(allocator, out, .{ .base = out, .component = 0 });
        }
    }
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
        .thread_id_x, .lane_id, .global_id_x, .vertex_index => true,
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
        .load => |l| {
            f(ctx, l.ptr);
            if (l.mask) |m| f(ctx, m);
        },
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

/// The two register files, as the allocator indexes them: scalar first, per-lane second.
const scalar_class: u16 = 0;
const vector_class: u16 = 1;

fn classOf(ctx: *const anyopaque, _: *const Function, v: Value) u16 {
    const c: *const Ctx = @ptrCast(@alignCast(ctx));
    return if (c.divergent[@backingInt(v)]) vector_class else scalar_class;
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
fn ifConvert(
    allocator: std.mem.Allocator,
    func: *Function,
    divergent: []const bool,
    masks: *std.ArrayList(LoopMask),
) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const a: Block = @fromBackingInt(@intCast(bi));
        const found = region(func, a, divergent) orelse continue;
        if (try flatten(allocator, func, found, divergent, masks)) changed = true;
    }
    return changed;
}

/// The most blocks one region may hold. The bound keeps the compile bounded rather than the code
/// fast: a divergent branch has no branchy alternative, so there is no faster version to protect.
const max_region_blocks = 64;

/// A region one divergent branch opens and one block closes: single entry, single exit, acyclic,
/// and every path from the head leaves only through the exit.
///
/// A diamond is the two-block case of this. A chain of `else if`s is not, which is why the
/// diamond-only version refused the shape a real rasteriser writes: the arms of the first branch
/// do not rejoin immediately, and the block they all end at has more than two predecessors.
/// A loop a region holds, which stays a loop. See `Region.loop`.
const LoopNode = struct {
    header: Block,
    latch: Block,
    exit: Block,
    pre: Block,
};

/// The mask a loop runs its first iteration under, decided by the region flattened around it
/// rather than by a branch above it. `divergentLoop` reads it.
const LoopMask = struct { header: Block, mask: Value };

const Region = struct {
    head: Block,
    exit: Block,
    /// The blocks strictly inside, ordered so a block comes after every predecessor of it.
    inside: [max_region_blocks]Block = undefined,
    /// Whether `inside[i]` runs after the loop, which decides where it is flattened to.
    after: [max_region_blocks]bool = @splat(false),
    count: usize = 0,
    /// The one loop inside, left as a loop. Its blocks are not in `inside`: nothing moves them,
    /// and what the flattening gives it is the mask its first iteration runs under.
    loop: ?LoopNode = null,

    fn holds(self: *const Region, b: Block) bool {
        if (self.loop) |l| {
            if (b == l.header or b == l.latch) return true;
        }
        for (self.inside[0..self.count]) |x| {
            if (x == b) return true;
        }
        return false;
    }

    /// Where `b`'s instructions go: the head, or the block the loop leaves by for everything that
    /// runs after the loop.
    fn destOf(self: *const Region, b: Block) Block {
        const l = self.loop orelse return self.head;
        for (self.inside[0..self.count], 0..) |x, i| {
            if (x == b) return if (self.after[i]) l.exit else self.head;
        }
        return self.head;
    }
};

/// The region the `if` ending `head` opens, or null when this cannot flatten it.
fn region(func: *const Function, head: Block, divergent: []const bool) ?Region {
    const insts = func.blockInsts(head);
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return null,
    };
    // A uniform branch is left alone: it costs one jump and runs one side, where flattening would
    // run every block of the region.
    if (!varies(divergent, cf.cond)) return null;
    if (cf.then.target == head or cf.@"else".target == head) return null;

    var r: Region = .{ .head = head, .exit = head };
    // Candidates are the blocks an edge out of the region's known part reaches. One of them ends
    // up being the exit and the rest end up inside.
    var seen: [max_region_blocks * 2]Block = undefined;
    var seen_count: usize = 0;
    const push = struct {
        fn f(list: []Block, n: *usize, b: Block) bool {
            for (list[0..n.*]) |x| {
                if (x == b) return true;
            }
            if (n.* == list.len) return false;
            list[n.*] = b;
            n.* += 1;
            return true;
        }
    }.f;
    if (!push(&seen, &seen_count, cf.then.target)) return null;
    if (!push(&seen, &seen_count, cf.@"else".target)) return null;

    // Take blocks into the region while more than one is still outstanding, always one whose
    // predecessors are all accounted for, which builds a topological order. The region CLOSES
    // when exactly one is left: that block is where every path meets again, so it is the exit.
    // Absorbing it instead would swallow the rest of the function.
    while (outstanding(&r, seen[0..seen_count]) > 1) {
        var took = false;
        for (seen[0..seen_count]) |b| {
            if (r.holds(b)) continue;
            // A LOOP inside the region stays a loop: nothing can move its blocks, because the
            // warp has to go round them. It is taken as one node, and what the flattening gives
            // it is the mask its first iteration runs under, which is the predicate of the block
            // that enters it. The walk carries on from the block the loop leaves by.
            if (r.loop == null) {
                if (loopNodeAt(func, &r, b, divergent)) |l| {
                    r.loop = l;
                    if (!push(&seen, &seen_count, l.exit)) return null;
                    took = true;
                    break;
                }
            }
            if (!readyIn(func, &r, b)) continue;
            // A block that leaves by `ret` takes a path out that the exit does not cover, so it
            // cannot go INSIDE. It may still be the exit, which is the ordinary end of a kernel.
            if (func.terminator(b)) |t| {
                if (t == .ret) continue;
            }
            // Everything inside runs for every lane, so it has to be something a lane that did
            // not choose it may run. A store and a load qualify: both take the mask.
            var pure = true;
            for (func.blockInsts(b)) |in_b| {
                if (func.opcode(in_b) == .@"if") continue; // dropped: the predicates carry it
                if (!convertible(func.opcode(in_b))) pure = false;
            }
            if (!pure) continue;
            if (r.count == max_region_blocks) return null;
            r.inside[r.count] = b;
            r.count += 1;
            for (successorsOf(func, b)) |s| {
                const t = s orelse continue;
                if (t == head) return null; // a cycle: this is a loop and not a region
                if (r.holds(t)) return null; // a cycle inside the region
                if (!push(&seen, &seen_count, t)) return null;
            }
            took = true;
            break;
        }
        if (!took) return null; // nothing was ready, so the shape is not single-exit
    }
    var exit: ?Block = null;
    for (seen[0..seen_count]) |b| {
        if (r.holds(b)) continue;
        exit = b;
    }
    r.exit = exit orelse return null;
    if (r.count == 0) return null;
    if (r.exit == head) return null;
    if (@backingInt(r.exit) == 0) return null;
    // Every way into the region is the head's branch, and every way out is the exit. A block with
    // an outside predecessor would be reached on a path this does not control.
    if (!readyIn(func, &r, r.exit)) return null;
    // What runs after the loop: the block it leaves by, and everything inside that block reaches.
    // Those are flattened into it rather than into the head, because the head runs BEFORE the
    // loop and they run after it.
    if (r.loop) |l| {
        var growing = true;
        while (growing) {
            growing = false;
            for (r.inside[0..r.count], 0..) |b, i| {
                if (r.after[i]) continue;
                var reached = b == l.exit;
                for (r.inside[0..r.count], 0..) |src, j| {
                    if (!r.after[j]) continue;
                    for (successorsOf(func, src)) |s| {
                        if (s == b) reached = true;
                    }
                }
                if (!reached) continue;
                r.after[i] = true;
                growing = true;
            }
        }
    }
    return r;
}

/// The loop `b` heads, when `b` heads one this region can hold, and null otherwise.
fn loopNodeAt(func: *const Function, r: *const Region, b: Block, divergent: []const bool) ?LoopNode {
    const dl = divergentLoop(func, b, divergent, &.{}) orelse return null;
    // The block that enters the loop has to be accounted for already, because the mask the loop
    // runs under is that block's predicate. The latch is the other way in and belongs to the loop.
    if (dl.preheader != r.head and !r.holds(dl.preheader)) return null;
    // Only the loop reaches the block it leaves by, so the predicate there is the loop's own.
    if (predCount(func, dl.exit) != 1) return null;
    if (dl.exit == r.head or dl.exit == dl.header) return null;
    return .{ .header = b, .latch = dl.latch, .exit = dl.exit, .pre = dl.preheader };
}

/// How many candidate blocks are not inside the region yet.
fn outstanding(r: *const Region, seen: []const Block) usize {
    var n: usize = 0;
    for (seen) |b| {
        if (!r.holds(b)) n += 1;
    }
    return n;
}

/// Whether every predecessor of `b` is the head or already inside the region.
fn readyIn(func: *const Function, r: *const Region, b: Block) bool {
    for (0..func.blockCount()) |pi| {
        const p: Block = @fromBackingInt(@intCast(pi));
        if (!jumpsTo(func, p, b)) continue;
        if (p == r.head or r.holds(p)) continue;
        return false;
    }
    return true;
}

/// The blocks `b` can go to, as two optional slots: an `if`'s two targets or a jump's one.
fn successorsOf(func: *const Function, b: Block) [2]?Block {
    const insts = func.blockInsts(b);
    if (insts.len != 0) {
        if (func.opcode(insts[insts.len - 1]) == .@"if") {
            const cf = func.opcode(insts[insts.len - 1]).@"if";
            return .{ cf.then.target, cf.@"else".target };
        }
    }
    return switch (func.terminator(b) orelse return .{ null, null }) {
        .jump => |j| .{ j.target, null },
        .ret => .{ null, null },
    };
}

/// What a block did by way of control flow, captured BEFORE anything moves. Flattening empties
/// the blocks it would otherwise read this from, and a predicate built later needs the condition
/// of a branch whose block is already gone.
const Flow = struct {
    block: Block,
    /// Null for a block that leaves by a jump.
    cond: ?Value = null,
    /// The then edge and the else edge, or just the first for a jump.
    edges: [2]?ir.function.Jump = .{ null, null },

    fn edgeTo(self: Flow, b: Block) ?ir.function.Jump {
        for (self.edges) |e| {
            const j = e orelse continue;
            if (j.target == b) return j;
        }
        return null;
    }

    /// Which side of a branch goes to `b`: 0 the then edge, 1 the else edge, null neither.
    fn sideTo(self: Flow, b: Block) ?usize {
        for (self.edges, 0..) |e, i| {
            const j = e orelse continue;
            if (j.target == b) return i;
        }
        return null;
    }
};

/// The control flow of the head and of every block inside a region.
const Flows = struct {
    items: [max_region_blocks + 1]Flow = undefined,
    count: usize = 0,

    fn add(self: *Flows, func: *const Function, b: Block) void {
        var f: Flow = .{ .block = b };
        const insts = func.blockInsts(b);
        if (insts.len != 0 and func.opcode(insts[insts.len - 1]) == .@"if") {
            const cf = func.opcode(insts[insts.len - 1]).@"if";
            f.cond = cf.cond;
            f.edges = .{ cf.then, cf.@"else" };
        } else if (func.terminator(b)) |t| switch (t) {
            .jump => |j| f.edges[0] = j,
            .ret => {},
        };
        self.items[self.count] = f;
        self.count += 1;
    }

    fn of(self: *const Flows, b: Block) ?Flow {
        for (self.items[0..self.count]) |f| {
            if (f.block == b) return f;
        }
        return null;
    }
};

/// Flatten `r` into its head: every block's instructions move in under the predicate of the block
/// they came from, every block parameter becomes a select over the paths that reach it, and the
/// branches disappear. Reports false when a predicate it would need cannot be built, leaving the
/// function as it was.
///
/// The cost is the SUM of the region's blocks rather than the longest path through it, because
/// every block now runs for every lane. That is the price of a warp having one program counter,
/// and the alternative is refusing the kernel.
///
/// A block is predicated and moved in one step, in topological order. The two cannot be separated:
/// a predicate reads the branch conditions of the blocks above it, and those are values that only
/// reach the head when their block moves.
fn flatten(
    allocator: std.mem.Allocator,
    func: *Function,
    r: Region,
    divergent: []const bool,
    masks: *std.ArrayList(LoopMask),
) Error!bool {
    const bool_ty = try func.types.intern(.bool);
    const head = r.head;

    // Captured first, because moving a block destroys what it says about its own edges.
    var flows: Flows = .{};
    flows.add(func, head);
    for (r.inside[0..r.count]) |b| flows.add(func, b);

    // How the loop is entered, read before the branch above it goes away.
    const enter: ?ir.function.Jump = if (r.loop) |l| blk: {
        if (l.pre != head) break :blk switch (func.terminator(l.pre) orelse return false) {
            .jump => |j| j,
            .ret => return false,
        };
        const insts = func.blockInsts(head);
        const cf = func.opcode(insts[insts.len - 1]).@"if";
        const edge = if (cf.then.target == l.header) cf.then else cf.@"else";
        break :blk .{ .target = l.header, .args = edge.args };
    } else null;

    // Nothing is touched until every predicate is known to be usable as a mask, so a refusal
    // leaves the function alone.
    for (r.inside[0..r.count]) |b| {
        for (flows.items[0..flows.count]) |src| {
            if (src.sideTo(b)) |side| {
                const cond = (src.cond orelse continue);
                if (!maskable(func, cond, divergent)) return false;
                if (side == 1 and varies(divergent, cond) and
                    complementOf(func, cond) == null and !hasDiscardParam(func))
                    return false;
            }
        }
    }

    _ = func.blockInstsMut(head).pop().?;

    var preds: [max_region_blocks]?Value = @splat(null);
    var negated: [max_region_blocks + 1]?Value = @splat(null);
    var masked: [max_region_blocks + 1]?Value = @splat(null);

    // The mask the loop's first iteration runs under: the predicate of the block that enters it.
    // `predicateOf` gives exactly that, because the latch is not one of the flows, so the only
    // path it finds into the header is the one from above.
    var entry: ?Value = null;
    if (r.loop) |l| {
        if (l.pre == head) {
            entry = try predicateOf(func, head, head, r, &flows, &preds, &negated, &masked, divergent, l.header, bool_ty);
        }
    }

    for (0..r.count) |i| {
        const b = r.inside[i];
        const dest = r.destOf(b);
        if (r.loop) |l| {
            if (b == l.exit) {
                // The loop still jumps here with the values it carries, so the parameters stay and
                // only the instructions take the predicate. This block is where everything after
                // the loop is flattened to.
                preds[i] = entry;
                try maskInPlace(allocator, func, b, entry, true);
                continue;
            }
        }
        preds[i] = try predicateOf(func, head, dest, r, &flows, &preds, &negated, &masked, divergent, b, bool_ty);
        try moveBlock(allocator, func, head, dest, r, &flows, &preds, b, preds[i], bool_ty, .drop_branch);
        if (r.loop) |l| {
            if (b == l.pre) {
                entry = try predicateOf(func, head, head, r, &flows, &preds, &negated, &masked, divergent, l.header, bool_ty);
            }
        }
    }

    // Everything after the loop was flattened into the block the loop leaves by, so that is where
    // the region ends and where the exit is absorbed.
    const last = if (r.loop) |l| l.exit else head;
    // The exit runs for every lane that entered the region, so it needs no predicate of its own.
    // Its own branch STAYS: the region ends there, so whatever it does next is not this region's
    // business. For a loop that branch is the latch, and dropping it would lose the loop.
    try moveBlock(allocator, func, head, last, r, &flows, &preds, r.exit, null, bool_ty, .keep_branch);

    // The last destination takes the exit's terminator, so nothing branches anywhere inside.
    // Everything else is unreachable and `neutralizeUnreachable` takes it out.
    func.terminatorPtr(last).* = func.terminatorPtr(r.exit).*;
    // With a loop in the region the head no longer branches: it enters the loop, and the lanes the
    // predicate excludes enter it inactive.
    if (enter) |j| {
        func.terminatorPtr(head).* = .{ .jump = j };
        if (entry) |m| try recordLoopMask(func, head, masks, r.loop.?.header, m, bool_ty);
    }
    for (r.inside[0..r.count]) |b| {
        if (b == last) continue;
        func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
    }
    if (r.exit != last) func.setTerminator(r.exit, .{ .ret = ir.function.Ret.none() });
    return true;
}

/// Record the mask `header`'s loop runs its first iteration under, ANDing with a mask an inner
/// region already decided: a lane has to pass both to take part.
fn recordLoopMask(
    func: *Function,
    at: Block,
    masks: *std.ArrayList(LoopMask),
    header: Block,
    mask: Value,
    bool_ty: Type,
) Error!void {
    for (masks.items) |*m| {
        if (m.header != header) continue;
        m.mask = try maskAnd(func, at, m.mask, mask, bool_ty);
        return;
    }
    try masks.append(func.allocator, .{ .header = header, .mask = mask });
}

/// Put `mask` over every access `b` holds, in place. With `drop_branch` a branch it ends with goes,
/// which is what a flattened region wants: its branches are gone and their conditions are inside
/// the predicates already. A loop's blocks keep theirs.
fn maskInPlace(
    allocator: std.mem.Allocator,
    func: *Function,
    b: Block,
    mask: ?Value,
    drop_branch: bool,
) Error!void {
    const body = try allocator.dupe(Inst, func.blockInsts(b));
    defer allocator.free(body);
    func.blockInstsMut(b).clearRetainingCapacity();
    for (body) |inst| {
        if (drop_branch and func.opcode(inst) == .@"if") continue;
        if (mask) |m| {
            try appendMasked(allocator, func, b, inst, m);
        } else {
            try func.blockInstsMut(b).append(allocator, inst);
        }
    }
}

/// The predicate for `b`: the paths into it, each the predicate of its source ANDed with what the
/// edge tests, ORed together. Null means every lane that reached the region.
fn predicateOf(
    func: *Function,
    head: Block,
    dest: Block,
    r: Region,
    flows: *const Flows,
    preds: *const [max_region_blocks]?Value,
    negated: *[max_region_blocks + 1]?Value,
    masked: *[max_region_blocks + 1]?Value,
    divergent: []const bool,
    b: Block,
    bool_ty: Type,
) Error!?Value {
    var total: ?Value = null;
    var any = false;
    for (flows.items[0..flows.count], 0..) |src, si| {
        const side = src.sideTo(b) orelse continue;
        const from: ?Value = if (src.block == head) null else preds[indexIn(r, src.block).?];
        // What the edge itself tests. A jump tests nothing, a branch its condition one way and the
        // complement of it the other, and a branch whose two edges both land here tests nothing.
        const edge: ?Value = blk: {
            const cond = src.cond orelse break :blk null;
            if (src.edges[0] != null and src.edges[1] != null and
                src.edges[0].?.target == src.edges[1].?.target) break :blk null;
            // Where the CONDITION is, which is where its block was flattened to.
            const at = r.destOf(src.block);
            if (masked[si] == null) masked[si] = (try asMask(func, at, cond, divergent, bool_ty)).?;
            if (side == 0) break :blk masked[si];
            if (negated[si] == null) {
                negated[si] = (try appendComplement(func, at, masked[si].?, bool_ty)).?;
            }
            break :blk negated[si];
        };
        const path: ?Value = if (from) |f|
            (if (edge) |e| try maskAnd(func, dest, f, e, bool_ty) else f)
        else
            edge;
        if (!any) {
            total = path;
        } else if (total) |a_mask| {
            // A path with no predicate of its own covers every lane, so the OR is everything.
            const b_mask = path orelse break;
            total = try maskOr(func, dest, a_mask, b_mask, bool_ty);
        }
        any = true;
    }
    return total;
}

fn maskAnd(func: *Function, at: Block, x: Value, y: Value, bool_ty: Type) Error!Value {
    return func.appendInst(at, bool_ty, .{ .arith = .{ .op = .bit_and, .lhs = x, .rhs = y } });
}

fn maskOr(func: *Function, at: Block, x: Value, y: Value, bool_ty: Type) Error!Value {
    return func.appendInst(at, bool_ty, .{ .arith = .{ .op = .bit_or, .lhs = x, .rhs = y } });
}

/// Where `b` sits in the region's order, or null when it is not inside.
fn indexIn(r: Region, b: Block) ?usize {
    for (r.inside[0..r.count], 0..) |x, i| {
        if (x == b) return i;
    }
    return null;
}

/// Move `b`'s parameters and instructions into `head`, under `mask`.
fn moveBlock(
    allocator: std.mem.Allocator,
    func: *Function,
    head: Block,
    dest: Block,
    r: Region,
    flows: *const Flows,
    preds: *const [max_region_blocks]?Value,
    b: Block,
    mask: ?Value,
    bool_ty: Type,
    branch: enum { drop_branch, keep_branch },
) Error!void {
    // A parameter becomes a select down the paths that reach it. The paths are mutually exclusive,
    // because a lane takes one way out of a branch, so the last one stands as the default and the
    // order between them does not matter.
    const params = try allocator.dupe(Value, func.blockParams(b));
    defer allocator.free(params);
    for (params, 0..) |p, pi| {
        var value: ?Value = null;
        for (flows.items[0..flows.count]) |src| {
            const edge = src.edgeTo(b) orelse continue;
            const args = func.blockArgs(edge);
            if (pi >= args.len) continue;
            const arg = args[pi];
            if (value == null) {
                value = arg;
                continue;
            }
            const gate = if (src.block == head) null else preds[indexIn(r, src.block).?];
            const g = gate orelse {
                value = arg; // a path that covers every lane wins outright
                continue;
            };
            value = try func.appendInst(dest, func.valueType(p), .{ .select = .{
                .cond = g,
                .then = arg,
                .@"else" = value.?,
            } });
        }
        if (value) |v| func.replaceAllUses(p, v);
    }
    try func.setBlockParams(b, &.{});

    const body = try allocator.dupe(Inst, func.blockInsts(b));
    defer allocator.free(body);
    func.blockInstsMut(b).clearRetainingCapacity();
    for (body) |inst| {
        // A branch INSIDE the region is gone: its condition is already inside every predicate
        // that depended on it. The exit keeps its own branch, because the region ends before it,
        // and for a loop that branch is the latch.
        if (func.opcode(inst) == .@"if" and branch == .drop_branch) continue;
        const carried: ?Value = switch (func.opcode(inst)) {
            .store => |st| st.mask,
            .load => |l| l.mask,
            else => null,
        };
        const combined: ?Value = if (mask) |m| blk: {
            if (carried) |inner| break :blk try maskAnd(func, dest, inner, m, bool_ty);
            break :blk m;
        } else carried;
        try func.blockInstsMut(dest).append(allocator, inst);
        switch (func.opcodeMut(inst).*) {
            .store => |*st| st.mask = combined,
            .load => |*l| l.mask = combined,
            else => {},
        }
    }
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

/// Whether `block` holds an access that would need the complement of a condition to run under.
fn hasStore(func: *const Function, block: Block) bool {
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .store, .load => return true,
        else => {},
    };
    return false;
}

/// The compare that answers the opposite of `cond`, when `cond` is an integer compare. Null for
/// anything else, including a FLOAT compare: an unordered pair leaves `lt` and `ge` both false, so
/// flipping the operator there changes the answer rather than negating it.
/// Whether `v` varies per lane, for a value that may have been MADE since the analysis ran.
///
/// The passes that flatten and convert create values of their own, and the analysis they were
/// handed is as long as the function was when it started. A value past the end is one of those, so
/// it is read as per-lane: every one of them is a mask, a predicate or a select over masks, and
/// reading it the other way would hand a mask op a scalar bool.
fn varies(divergent: []const bool, v: Value) bool {
    const i = @backingInt(v);
    return i >= divergent.len or divergent[i];
}

/// Whether `cond` can be used as a mask in a predicate: it already varies per lane, or it is a
/// compare this can rebuild per lane. See `asMask`.
fn maskable(func: *const Function, cond: Value, divergent: []const bool) bool {
    if (varies(divergent, cond)) return true;
    const inst = func.definingInst(cond) orelse return false;
    switch (func.opcode(inst)) {
        .icmp => |c| return !isFloat(func, func.valueType(c.lhs)),
        // A NOT of something maskable: mask that and flip it.
        .arith, .arith_imm => switch (complementOf(func, cond) orelse return false) {
            .have => |v| return maskable(func, v, divergent),
            .compare => return false,
        },
        else => return false,
    }
}

/// `cond` as a MASK, one bit per lane, rebuilt when it is the same in every lane.
///
/// A warp-uniform bool sits in a scalar register as a 0 or a 1, and a mask op reads one BIT per
/// lane, so the two cannot be combined as they stand. A region mixes the two whenever a test that
/// does not vary per lane sits inside a branch that does, which a rasteriser writes every time it
/// asks something about the triangle rather than about the pixel.
///
/// Rebuilding the compare with one operand made per-lane gives the same answer in every lane and
/// arrives as a real mask. Adding a per-lane zero is what makes the operand per-lane: the value is
/// unchanged and the divergence analysis reads it off the operands.
fn asMask(func: *Function, at: Block, cond: Value, divergent: []const bool, bool_ty: Type) Error!?Value {
    if (varies(divergent, cond)) return cond;
    const inst = func.definingInst(cond) orelse return null;
    const c = switch (func.opcode(inst)) {
        .icmp => |x| x,
        // A NOT: the mask of what it negates, flipped.
        .arith, .arith_imm => switch (complementOf(func, cond) orelse return null) {
            .have => |v| {
                const inner = (try asMask(func, at, v, divergent, bool_ty)) orelse return null;
                return try appendComplement(func, at, inner, bool_ty);
            },
            .compare => return null,
        },
        else => return null,
    };
    const ty = func.valueType(c.lhs);
    if (isFloat(func, ty)) return null;
    const lane = laneValue(func) orelse return null;
    const zero = try func.appendInst(at, func.valueType(lane), .{ .arith_imm = .{
        .op = .bit_and,
        .lhs = lane,
        .imm = 0,
    } });
    const spread = try func.appendInst(at, ty, .{ .arith = .{ .op = .add, .lhs = c.lhs, .rhs = zero } });
    return try func.appendInst(at, bool_ty, .{ .icmp = .{ .op = c.op, .lhs = spread, .rhs = c.rhs } });
}

/// How the complement of a bool can be had, when it can.
const Complement = union(enum) {
    /// A value that already IS the complement, so it costs nothing.
    have: Value,
    /// An integer compare to emit, which answers the opposite of the one given.
    compare: ir.function.Compare,
};

/// The complement of `cond`, or null when there is none to be had.
///
/// An integer compare has one: flip the operator. A NOT has one for free, and a frontend writes a
/// NOT as `b ^ 1`, in either the immediate form or with a constant of its own, so the complement
/// of that is `b`.
fn complementOf(func: *const Function, cond: Value) ?Complement {
    if (invertedCompare(func, cond)) |c| return .{ .compare = c };
    const inst = func.definingInst(cond) orelse return null;
    switch (func.opcode(inst)) {
        .arith_imm => |a| {
            if (a.op != .bit_xor or a.imm != 1) return null;
            if (!isBool(func, func.valueType(a.lhs))) return null;
            return .{ .have = a.lhs };
        },
        .arith => |a| {
            if (a.op != .bit_xor) return null;
            if (!isBool(func, func.valueType(a.lhs))) return null;
            // One side is the constant one, and the other is what this negates.
            if (isOne(func, a.rhs)) return .{ .have = a.lhs };
            if (isOne(func, a.lhs)) return .{ .have = a.rhs };
            return null;
        },
        else => return null,
    }
}

/// Whether `v` is the constant one.
fn isOne(func: *const Function, v: Value) bool {
    const inst = func.definingInst(v) orelse return false;
    return switch (func.opcode(inst)) {
        .iconst => |c| c == 1,
        else => false,
    };
}

/// The complement of `cond` as a value, emitted in `at` when it has to be.
fn appendComplement(func: *Function, at: Block, cond: Value, bool_ty: Type) Error!?Value {
    const complement = complementOf(func, cond) orelse
        return try func.appendInst(at, bool_ty, .{ .arith_imm = .{ .op = .bit_xor, .lhs = cond, .imm = 1 } });
    return switch (complement) {
        .have => |v| v,
        .compare => |c| try func.appendInst(at, bool_ty, .{ .icmp = c }),
    };
}

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
        // A store or a load converts because the core can predicate either. An inactive lane makes
        // no bus access at all, so an access spliced out of an arm touches exactly the lanes that
        // chose it, and a lane that did not read gets a destination lane it was never going to
        // look at. An access that already carries a mask converts too: the two masks are ANDed,
        // which is a mask op and exact at any lane count. See `maskBinOp`.
        .store, .load => true,
        else => false,
    };
}

// ===========================================================================
// Merging a block into the predecessor that only jumps to it.

/// Where control really ends up when a block only passes values on: through a chain of empty
/// parameterless blocks, to the first block that holds something, with the arguments that arrive
/// there.
fn arrivalOf(func: *const Function, edge: ir.function.Jump) ir.function.Jump {
    var cur = edge;
    var hops: usize = 0;
    while (hops < max_forward_hops) : (hops += 1) {
        const b = cur.target;
        if (func.blockInsts(b).len != 0) return cur;
        if (func.blockParams(b).len != 0) return cur;
        if (func.blockArgs(cur).len != 0) return cur;
        cur = switch (func.terminator(b) orelse return cur) {
            .jump => |j| j,
            .ret => return cur,
        };
    }
    return cur;
}

/// How far a chain of empty forwarding blocks is followed. A bound, not a limit on anything real.
const max_forward_hops = 8;

/// Replace a branch whose two sides end up at the same block with the same values by a jump, and
/// report whether anything changed.
///
/// A structurizer leaves these behind: `if (code == X) {} else {}` with both arms emptied by the
/// optimizer, so the branch chooses between two identical arrivals. It is still a BRANCH, so a
/// per-lane condition on it would be refused for a choice that makes no difference, and the arms
/// cannot be flattened when the block they meet at is reached from elsewhere as well.
fn dropPointlessBranches(func: *Function) bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        if (insts.len == 0) continue;
        const cf = switch (func.opcode(insts[insts.len - 1])) {
            .@"if" => |x| x,
            else => continue,
        };
        const a = arrivalOf(func, cf.then);
        const b = arrivalOf(func, cf.@"else");
        if (a.target != b.target) continue;
        if (!std.mem.eql(Value, func.blockArgs(a), func.blockArgs(b))) continue;
        _ = func.blockInstsMut(block).pop().?;
        func.terminatorPtr(block).* = .{ .jump = a };
        changed = true;
    }
    return changed;
}

/// Merge every block whose one predecessor reaches it by an unconditional jump, and report whether
/// anything changed.
///
/// A frontend leaves these everywhere. A `while` loop arrives as a header that holds the
/// loop-carried parameters and does nothing else, jumping to a second block that holds the test,
/// so the test is not in the header and `divergentLoop` does not see a latch. Merging the two puts
/// the test where the header is, which is the shape the rest of this reads.
///
/// It is also what lets region if-conversion see a whole region: a block boundary with nothing on
/// either side of it is a block the region has to account for.
fn mergeStraightLine(allocator: std.mem.Allocator, func: *Function) Error!bool {
    var changed = false;
    for (1..func.blockCount()) |bi| {
        const b: Block = @fromBackingInt(@intCast(bi));
        if (predCount(func, b) != 1) continue;
        const pred = solePredecessor(func, b) orelse continue;
        if (pred == b) continue;
        // The predecessor must LEAVE by a jump. A block ending in a branch keeps its two edges,
        // and critical-edge splitting put some of those blocks there on purpose.
        if (endsInIf(func, pred)) continue;
        const jump = switch (func.terminator(pred) orelse continue) {
            .jump => |j| j,
            .ret => continue,
        };
        if (jump.target != b) continue;

        // The parameters take the one set of arguments that reaches them.
        const params = try allocator.dupe(Value, func.blockParams(b));
        defer allocator.free(params);
        const args = try allocator.dupe(Value, func.blockArgs(jump));
        defer allocator.free(args);
        if (args.len != params.len) continue;
        for (params, args) |p, a| func.replaceAllUses(p, a);
        try func.setBlockParams(b, &.{});

        const body = try allocator.dupe(Inst, func.blockInsts(b));
        defer allocator.free(body);
        func.blockInstsMut(b).clearRetainingCapacity();
        try func.blockInstsMut(pred).appendSlice(allocator, body);
        func.terminatorPtr(pred).* = func.terminatorPtr(b).*;
        func.setTerminator(b, .{ .ret = ir.function.Ret.none() });
        changed = true;
    }
    return changed;
}

/// Where `b` really goes: through an empty forwarding block that passes nothing, or `b` itself.
///
/// A block that carries ARGUMENTS is not forwarding, because the values it passes belong to the
/// path that reaches it and cannot be read as the same arrival.
fn forwardsTo(func: *const Function, b: Block) Block {
    if (func.blockInsts(b).len != 0) return b;
    if (func.blockParams(b).len != 0) return b;
    return switch (func.terminator(b) orelse return b) {
        .jump => |j| if (func.blockArgs(j).len == 0) j.target else b,
        .ret => b,
    };
}

/// Whether the guard's failing side rejoins the loop's path AFTER the block the loop leaves by,
/// and what that side carries. Setting `fail` says so, and the values may be none.
///
/// Two shapes, and both mean a lane that failed the guard does NOT run the exit block in the
/// program as written:
///
///   `if (c) { while (...) } else { x }` where both sides give a merge block a value, so the
///   failing side arrives WHERE the exit arrives;
///
///   `if (c) { while (...); tail }` where the tail IS the exit block, so the failing side arrives
///   where the exit GOES.
///
/// Once the guard is a mask every lane runs the exit block, so each access there takes the guard
/// as its mask and the values the failing side carried are selected over the loop's at the exit.
/// An instruction there that cannot take a mask keeps the branch: see `convertible`.
fn guardRejoins(
    func: *const Function,
    missed_edge: ir.function.Jump,
    exit: Block,
    fail: *?ir.function.ValueList,
) bool {
    // Only the loop reaches the exit, which puts the exit below the guard and the guard's own
    // values within reach of it.
    if (predCount(func, exit) != 1) return false;
    const taken = switch (func.terminator(exit) orelse return false) {
        .jump => |j| j,
        .ret => return false,
    };
    for (func.blockInsts(exit)) |inst| {
        if (!convertible(func.opcode(inst))) return false;
    }
    const out = forwardsTo(func, missed_edge.target);
    if (out == taken.target) {
        // The failing side IS where the exit goes, so what it carries is on the guard's own edge.
        if (func.blockArgs(missed_edge).len != func.blockArgs(taken).len) return false;
        fail.* = missed_edge.args;
        return true;
    }
    // The failing side is a block of its own that arrives where the exit arrives. It must hold
    // nothing, so the values it passes are defined above the guard and reach the exit.
    if (func.blockInsts(out).len != 0) return false;
    if (func.blockParams(out).len != 0) return false;
    if (predCount(func, out) != 1) return false;
    const missed = switch (func.terminator(out) orelse return false) {
        .jump => |j| j,
        .ret => return false,
    };
    if (missed.target != taken.target) return false;
    if (func.blockArgs(missed).len != func.blockArgs(taken).len) return false;
    fail.* = missed.args;
    return true;
}

/// The one block that reaches `b`, when exactly one does.
fn solePredecessor(func: *const Function, b: Block) ?Block {
    var found: ?Block = null;
    for (0..func.blockCount()) |pi| {
        const p: Block = @fromBackingInt(@intCast(pi));
        if (!jumpsTo(func, p, b)) continue;
        if (found != null) return null;
        found = p;
    }
    return found;
}

// ===========================================================================
// Broadcasting a warp-uniform argument into a per-lane parameter.

/// Give every PER-LANE block parameter a per-lane argument, and report whether anything changed.
///
/// A parameter and the argument that feeds it can sit in different register files here: the
/// parameter is per-lane and the argument is the same in every lane, a loop's initial accumulator
/// for instance. The shared allocator builds that move with the PARAMETER's class and the
/// ARGUMENT's location, and `locEql` compares the two by index without their file, so the move is
/// emitted into the wrong register file or DROPPED ENTIRELY when the two indices coincide. Neither
/// has a diagnostic.
///
/// So the crossing is removed instead of being handled: the argument is broadcast into lanes in
/// the predecessor, and the parameter then receives a per-lane value like any other. Adding the
/// lane-wise zero is what makes it per-lane, and it costs one instruction once plus one per
/// argument fixed.
fn broadcastUniformArgs(allocator: std.mem.Allocator, func: *Function, divergent: []const bool) Error!bool {
    const lane = laneValue(func) orelse return false;
    var zero: ?Value = null;
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const pred: Block = @fromBackingInt(@intCast(bi));
        var edge: usize = 0;
        while (edge < 3) : (edge += 1) {
            const target = edgeTarget(func, pred, edge) orelse continue;
            const params = func.blockParams(target);
            const args = edgeArgsOf(func, pred, edge);
            if (args.len != params.len) continue;
            var fixed: std.ArrayList(Value) = .empty;
            defer fixed.deinit(allocator);
            var any = false;
            for (params, args) |p, a| {
                if (!divergent[@backingInt(p)] or divergent[@backingInt(a)]) {
                    try fixed.append(allocator, a);
                    continue;
                }
                if (zero == null) {
                    zero = try appendBeforeBranch(allocator, func, @fromBackingInt(0), func.valueType(lane), .{
                        .arith_imm = .{ .op = .bit_and, .lhs = lane, .imm = 0 },
                    });
                }
                try fixed.append(allocator, try appendBeforeBranch(allocator, func, pred, func.valueType(a), .{
                    .arith = .{ .op = .add, .lhs = a, .rhs = zero.? },
                }));
                any = true;
            }
            if (!any) continue;
            try setEdgeArgs(func, pred, edge, try func.internValueList(fixed.items));
            changed = true;
        }
    }
    return changed;
}

/// The target of edge `n` leaving `block`: 0 and 1 are an `if`'s then and else, 2 is a jump.
fn edgeTarget(func: *const Function, block: Block, n: usize) ?Block {
    if (endsInIf(func, block)) {
        const insts = func.blockInsts(block);
        const cf = func.opcode(insts[insts.len - 1]).@"if";
        return switch (n) {
            0 => cf.then.target,
            1 => cf.@"else".target,
            else => null,
        };
    }
    if (n != 2) return null;
    return switch (func.terminator(block) orelse return null) {
        .jump => |j| j.target,
        .ret => null,
    };
}

fn edgeArgsOf(func: *const Function, block: Block, n: usize) []const Value {
    if (endsInIf(func, block)) {
        const insts = func.blockInsts(block);
        const cf = func.opcode(insts[insts.len - 1]).@"if";
        return func.blockArgs(if (n == 0) cf.then else cf.@"else");
    }
    return switch (func.terminator(block).?) {
        .jump => |j| func.blockArgs(j),
        .ret => &.{},
    };
}

fn setEdgeArgs(func: *Function, block: Block, n: usize, list: ir.function.ValueList) Error!void {
    if (endsInIf(func, block)) {
        const insts = func.blockInsts(block);
        const cf = &func.opcodeMut(insts[insts.len - 1]).@"if";
        if (n == 0) cf.then.args = list else cf.@"else".args = list;
        return;
    }
    func.terminatorPtr(block).*.?.jump.args = list;
}

/// Append an instruction to `block` so that it lands BEFORE the conditional that ends it, which is
/// where a value the branch's arguments read has to be.
fn appendBeforeBranch(
    allocator: std.mem.Allocator,
    func: *Function,
    block: Block,
    ty: Type,
    op: ir.function.Opcode,
) Error!Value {
    if (!endsInIf(func, block)) return func.appendInst(block, ty, op);
    const insts = func.blockInstsMut(block);
    const branch = insts.pop().?;
    const v = try func.appendInst(block, ty, op);
    try func.blockInstsMut(block).append(allocator, branch);
    return v;
}

// ===========================================================================
// Divergent loops.

/// Which `if` instructions branch on whether ANY LANE is still active, rather than on a boolean.
///
/// A warp has one program counter, so a loop whose lanes disagree about when to stop runs until
/// the last of them is done, with the finished ones carried along inactive. The condition of such
/// a branch is a MASK and the branch is a reduction of it, which is not what an `if` means
/// anywhere else. `checkSupported` still refuses every other divergent branch, so this set is the
/// whole of the exception and `convertDivergentLoops` is its only producer.
const AnyLane = struct {
    at: []const bool,

    fn deinit(self: *AnyLane, allocator: std.mem.Allocator) void {
        allocator.free(self.at);
    }

    fn has(self: *const AnyLane, inst: Inst) bool {
        return self.at[@backingInt(inst)];
    }
};

/// A loop this can convert: a header left by a per-lane condition, with the back edge going
/// through the empty block critical-edge splitting put on it.
///
/// A self-loop is NOT the shape that arrives. The latch has two successors and the header has two
/// predecessors, so the back edge is critical and `splitCriticalEdges` always puts a block on it
/// before this runs. That block holds nothing, takes no parameters and jumps straight back, so it
/// is where the values that go round are named.
const DivergentLoop = struct {
    header: Block,
    /// The empty block on the back edge, which carries the arguments round.
    latch: Block,
    /// The one predecessor of the header that is not the latch.
    preheader: Block,
    cond: Value,
    /// Whether the TRUE edge continues the loop.
    continue_on_true: bool,
    exit: Block,
    /// The condition of a per-lane GUARD around the loop, which becomes the mask the first
    /// iteration runs under. Null when every lane that reached the preheader enters.
    guard: ?Value = null,
    /// Set when the lanes the guard turns away rejoin AFTER the block the loop leaves by, holding
    /// what that side carries, which may be nothing. Then the exit block runs only for the lanes
    /// the guard admitted: its accesses take the guard as a mask and its outgoing values are
    /// selected against these. Null when the failing lanes land on the exit block itself, which
    /// they run as they always did. See `guardRejoins`.
    guard_fail: ?ir.function.ValueList = null,
    /// Whether the guard's TRUE side is the one that enters the loop.
    guard_on_true: bool = true,
    /// The block the guard branches in, which is one above the preheader when critical-edge
    /// splitting put a block between them. Null when the mask came from a region flattened around
    /// the loop, where there is no branch left to absorb.
    guard_block: ?Block = null,
};

/// Whether `block` is a loop this converts. Every condition is a reason it would otherwise be
/// wrong rather than a simplification of convenience.
fn divergentLoop(
    func: *const Function,
    block: Block,
    divergent: []const bool,
    masks: []const LoopMask,
) ?DivergentLoop {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return null,
    };
    if (!varies(divergent, cf.cond)) return null;
    const then_latch = isLatchOf(func, cf.then.target, block);
    const else_latch = isLatchOf(func, cf.@"else".target, block);
    // One edge round and the other out. Both or neither is not a loop this reads.
    if (then_latch == else_latch) return null;
    const latch = if (then_latch) cf.then.target else cf.@"else".target;
    const exit = if (then_latch) cf.@"else".target else cf.then.target;
    if (exit == block or exit == latch) return null;
    // The branch passes nothing to the latch: the latch is empty and takes no parameters, so the
    // values that go round are the ones IT names.
    if (func.blockArgs(if (then_latch) cf.then else cf.@"else").len != 0) return null;
    const round = switch (func.terminator(latch).?) {
        .jump => |j| j,
        .ret => return null,
    };
    // The carried values arrive as parameters and go round from the latch, so the two line up.
    if (func.blockArgs(round).len != func.blockParams(block).len) return null;
    if (predCount(func, block) != 2) return null;
    var preheader: ?Block = null;
    for (0..func.blockCount()) |bi| {
        const p: Block = @fromBackingInt(@intCast(bi));
        if (p == block or p == latch) continue;
        if (!jumpsTo(func, p, block)) continue;
        if (preheader != null) return null;
        preheader = p;
    }
    const pre = preheader orelse return null;
    // A GUARDED loop: the preheader leaves by a per-lane branch, one way into the loop and the
    // other to the block the loop exits to. That is `if (c) while (...)`, and it means the loop
    // starts with only the lanes `c` names. The guard's condition becomes the loop's first mask
    // and the branch goes away: a lane that fails it enters inactive, runs nothing, and leaves by
    // the loop's own exit edge, which is where the guard would have sent it.
    //
    // The guard must send the failing lanes to THE LOOP'S EXIT. One that sends them anywhere else
    // is a different shape and stays refused.
    // The mask for the first iteration is appended to the preheader, which only lands before the
    // jump when the preheader LEAVES by a jump.
    if (func.terminator(pre) == null) return null;
    for (func.blockInsts(pre)) |inst| {
        if (func.opcode(inst) == .@"if") return null;
    }
    // Where the guard sits, if there is one. Critical-edge splitting puts an EMPTY block between
    // a branch and a loop header, so the guard is one level above the preheader rather than being
    // the preheader. Looking only at the immediate predecessor finds nothing.
    var guard: ?Value = null;
    var guard_fail: ?ir.function.ValueList = null;
    var guard_on_true = true;
    var guard_block: ?Block = null;
    // A region flattened around this loop already decided which lanes enter it, and left no
    // branch: the mask is a value in a block above. See `flatten`.
    for (masks) |m| {
        if (m.header == block) guard = m.mask;
    }
    if (guard == null and func.blockInsts(pre).len == 0 and func.blockParams(pre).len == 0) {
        if (solePredecessor(func, pre)) |up| {
            if (endsInIf(func, up)) {
                const up_insts = func.blockInsts(up);
                const g = func.opcode(up_insts[up_insts.len - 1]).@"if";
                const into_then = g.then.target == pre;
                const into_else = g.@"else".target == pre;
                // One way in and the other OUT, to the block the loop itself exits to. A guard
                // that sends the failing lanes anywhere else is a different shape.
                const out = if (into_then) g.@"else".target else g.then.target;
                var fail: ?ir.function.ValueList = null;
                // Both edges leave for the same place, but critical-edge splitting puts a
                // DIFFERENT empty block on each, so comparing the immediate targets finds two
                // blocks where the source had one. Compare where they end up.
                const to_the_exit = forwardsTo(func, out) == forwardsTo(func, exit) or
                    guardRejoins(func, if (into_then) g.@"else" else g.then, exit, &fail);
                if (into_then != into_else and
                    to_the_exit and
                    varies(divergent, g.cond) and
                    (into_then or complementOf(func, g.cond) != null))
                {
                    guard = g.cond;
                    guard_fail = fail;
                    guard_on_true = into_then;
                    guard_block = up;
                }
            }
        }
    }
    // Every instruction in the body runs for a lane that has already finished, so the body must be
    // instructions that cost nothing when their answer is thrown away. A store and a load both
    // qualify because both take the mask: a finished lane makes no access at all.
    for (insts[0 .. insts.len - 1]) |inst| {
        if (!convertible(func.opcode(inst))) return null;
    }
    return .{
        .header = block,
        .latch = latch,
        .preheader = pre,
        .cond = cf.cond,
        .continue_on_true = then_latch,
        .exit = exit,
        .guard = guard,
        .guard_fail = guard_fail,
        .guard_on_true = guard_on_true,
        .guard_block = guard_block,
    };
}

/// Whether `candidate` is the block critical-edge splitting put on the back edge to `header`:
/// empty, parameterless, reached only from the header, and leaving straight back to it.
fn isLatchOf(func: *const Function, maybe: Block, header: Block) bool {
    if (maybe == header) return false;
    if (func.blockInsts(maybe).len != 0) return false;
    if (func.blockParams(maybe).len != 0) return false;
    if (predCount(func, maybe) != 1) return false;
    return switch (func.terminator(maybe) orelse return false) {
        .jump => |j| j.target == header,
        .ret => false,
    };
}

/// Convert every divergent loop, recording each rewritten latch in `latches`.
///
/// A warp has ONE program counter, so the lanes cannot leave a loop at different times. They leave
/// together, at the last iteration any of them needs, and a lane that is finished is carried along
/// INACTIVE: an active mask says which lanes are still going, the body's stores take that mask, and
/// every carried value is frozen for a lane the mask has dropped. The branch then asks whether any
/// lane is left rather than what one lane thinks.
///
/// The mask is a loop-carried value like any other, which is what makes the rest fall out: it
/// arrives as a block parameter, so the divergence analysis sees it as an operand and nothing here
/// depends on a control-dependence rule.
// ===========================================================================
// A loop whose test is at the TOP.

/// Turn every `while` loop into the do-while shape `divergentLoop` reads, and report whether
/// anything changed.
///
/// A SPIR-V producer puts the test in the HEADER and the body in a block of its own, so the
/// header's branch goes to the body one way and out of the loop the other. `divergentLoop` reads
/// the other shape, where the body IS the header and the branch at the end of it either goes round
/// or leaves. The two are the same loop once the body runs UNDER THE TEST: a lane the test excludes
/// does nothing and carries its values unchanged, which is what a mask and a select say. So the
/// body moves into the header under the test and the body's block, now empty, becomes the latch.
///
/// Only a divergent loop needs this. A warp-uniform one branches and costs nothing.
fn rotateWhileLoops(allocator: std.mem.Allocator, func: *Function, divergent: []const bool) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        if (try rotateWhileLoop(allocator, func, block, divergent)) changed = true;
    }
    return changed;
}

/// Whether `maybe` is the body of a `while` loop whose header is `header`: reached only from the
/// header, holding no parameters of its own, and leaving straight back to the header with the
/// values that go round.
fn whileBodyOf(func: *const Function, maybe: Block, header: Block) bool {
    if (maybe == header) return false;
    // An EMPTY block here is the latch of a loop that is already a do-while, where the body sits
    // in the header and ran before the test. Rotating that would put this iteration's work behind
    // the test that ends the loop, and a lane leaving now would carry the value it had BEFORE the
    // iteration it just ran.
    if (func.blockInsts(maybe).len == 0) return false;
    if (func.blockParams(maybe).len != 0) return false;
    if (predCount(func, maybe) != 1) return false;
    return switch (func.terminator(maybe) orelse return false) {
        .jump => |j| j.target == header,
        .ret => false,
    };
}

/// Rotate the loop `header` heads, or report false when it is not a `while` loop this reads. Every
/// condition is a reason it would otherwise be wrong rather than a simplification of convenience.
fn rotateWhileLoop(
    allocator: std.mem.Allocator,
    func: *Function,
    header: Block,
    divergent: []const bool,
) Error!bool {
    const insts = func.blockInsts(header);
    if (insts.len == 0) return false;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return false,
    };
    if (!varies(divergent, cf.cond)) return false;
    // One edge into the body and the other out. Both or neither is not a loop this reads.
    const then_body = whileBodyOf(func, cf.then.target, header);
    const else_body = whileBodyOf(func, cf.@"else".target, header);
    if (then_body == else_body) return false;
    const body = if (then_body) cf.then.target else cf.@"else".target;
    const exit = if (then_body) cf.@"else".target else cf.then.target;
    if (exit == header or exit == body) return false;
    if (predCount(func, header) != 2) return false;
    // The body takes no parameters, so the branch passes it nothing and the values that go round
    // are the ones the back edge names. They line up with the header's parameters.
    if (func.blockArgs(if (then_body) cf.then else cf.@"else").len != 0) return false;
    const round = func.terminator(body).?.jump;
    if (func.blockArgs(round).len != func.blockParams(header).len) return false;
    // Every instruction in the body runs for every lane once it moves up, so each has to cost
    // nothing when its answer is thrown away. An access qualifies because it takes the test as a
    // mask: a lane the test excludes makes no access at all.
    for (func.blockInsts(body)) |inst| {
        if (!convertible(func.opcode(inst))) return false;
    }
    // The body on the FALSE side needs the complement of the test. See `complementOf`.
    if (!then_body and complementOf(func, cf.cond) == null) return false;

    const bool_ty = try func.types.intern(.bool);
    const pred = if (then_body) cf.cond else switch (complementOf(func, cf.cond).?) {
        .have => |v| v,
        .compare => |c| try appendBeforeBranch(allocator, func, header, bool_ty, .{ .icmp = c }),
    };

    const moved = try allocator.dupe(Inst, func.blockInsts(body));
    defer allocator.free(moved);
    const back = try allocator.dupe(Value, func.blockArgs(round));
    defer allocator.free(back);
    // The edge out, kept before the branch is rewritten, arguments and all.
    const leaving = if (then_body) cf.@"else" else cf.then;

    // The branch comes off while the body goes under it, and goes back on at the end.
    const branch = func.blockInstsMut(header).pop().?;
    for (moved) |inst| try appendMasked(allocator, func, header, inst, pred);

    // What goes round: the body's value for a lane the test admitted, and the parameter it already
    // had for a lane the test excluded. Without this a lane that stopped early would carry a value
    // computed from an iteration it took no part in.
    var carried: std.ArrayList(Value) = .empty;
    defer carried.deinit(allocator);
    for (func.blockParams(header), back) |param, arg| {
        try carried.append(allocator, try func.appendInst(header, func.valueType(param), .{ .select = .{
            .cond = pred,
            .then = arg,
            .@"else" = param,
        } }));
    }
    try func.blockInstsMut(header).append(allocator, branch);

    // The body is the latch now: empty, taking nothing, carrying the selected values round.
    func.blockInstsMut(body).clearRetainingCapacity();
    func.terminatorPtr(body).* = .{ .jump = .{
        .target = header,
        .args = try func.internValueList(carried.items),
    } };
    // The test enters the latch on the TRUE side whatever its polarity was, which is the shape
    // `divergentLoop` reads.
    func.opcodeMut(branch).* = .{ .@"if" = .{
        .cond = pred,
        .then = .{ .target = body, .args = try func.internValueList(&.{}) },
        .@"else" = leaving,
    } };
    return true;
}

/// Put `inst` in `block` with `mask` over the access it makes, ANDed with a mask it already
/// carries. An instruction that makes no access goes in unchanged.
fn appendMasked(
    allocator: std.mem.Allocator,
    func: *Function,
    block: Block,
    inst: Inst,
    mask: Value,
) Error!void {
    const carried: ?Value = switch (func.opcode(inst)) {
        .store => |st| st.mask,
        .load => |l| l.mask,
        else => null,
    };
    const both = if (carried) |inner| try func.appendInst(block, try func.types.intern(.bool), .{ .arith = .{
        .op = .bit_and,
        .lhs = inner,
        .rhs = mask,
    } }) else mask;
    try func.blockInstsMut(block).append(allocator, inst);
    switch (func.opcodeMut(inst).*) {
        .store => |*st| st.mask = both,
        .load => |*l| l.mask = both,
        else => {},
    }
}

// ===========================================================================
// A loop whose body is more than one block.

/// The most blocks one loop body may hold. A body is a chain here, so this bounds the compile
/// rather than the code.
const max_body_blocks = 64;

/// A divergent loop whose body spans blocks, which is what a loop INSIDE the body makes: the warp
/// has to go round the inner one, so nothing can flatten the body into the header.
///
/// The test sits at the top, so the mask the body runs under is the mask of the lanes still going
/// ANDed with the test. The iteration ends in the tail, which is where the freezing and the
/// reduction of that mask go.
const MultiLoop = struct {
    header: Block,
    preheader: Block,
    /// The body in order, from the block the test enters to the one that goes round. An inner
    /// loop's own blocks are NOT here: its own conversion masks those.
    body: [max_body_blocks]Block = undefined,
    count: usize = 0,
    /// The inner loop headers, and the block that enters each.
    inner: [max_body_blocks]Block = undefined,
    inner_pre: [max_body_blocks]Block = undefined,
    inner_count: usize = 0,
    /// Every block the inner loops own. They keep their own shape, and they take the body's mask
    /// like the rest of the body: an inner loop that is warp-uniform gets no conversion of its
    /// own, and its accesses still belong to the lanes the test admitted.
    owned: [max_body_blocks]Block = undefined,
    owned_count: usize = 0,
    /// The body block the iteration ends in. With the test at the top it goes round by a jump this
    /// turns into the reduction branch; with the test at the bottom it already branches.
    tail: Block,
    /// The empty block on the back edge, when the test is at the bottom and there is one.
    latch: Block,
    exit: Block,
    cond: Value,
    /// Whether the test is in the HEADER. A `while` loop reads that way and a `do`-`while` does
    /// not, and the difference is which mask the body runs under: with the test at the top a lane
    /// the test turns away runs no body, so the body's mask is the test ANDed in, and with the test
    /// at the bottom every lane that entered the iteration ran the body already.
    test_at_top: bool,
    /// With the test at the top, whether its TRUE side enters the body. With it at the bottom,
    /// whether its TRUE side goes round.
    enter_on_true: bool,
    /// The two edges of the test's branch, kept before it is rewritten.
    entering: ir.function.Jump,
    leaving: ir.function.Jump,
};

/// Whether `header` heads a loop whose body spans blocks. Every condition is a reason it would
/// otherwise be wrong rather than a simplification of convenience.
fn multiBlockLoop(
    func: *const Function,
    header: Block,
    divergent: []const bool,
    masks: []const LoopMask,
) ?MultiLoop {
    if (predCount(func, header) != 2) return null;
    for (func.blockInsts(header)) |inst| {
        if (func.opcode(inst) == .@"if") continue; // the test, read below
        if (!convertible(func.opcode(inst))) return null;
    }
    const m = if (endsInIf(func, header))
        topTestLoop(func, header, divergent, masks) orelse return null
    else
        bottomTestLoop(func, header, divergent, masks) orelse return null;
    // Where the loop is entered from: the other way into the header.
    var found = m;
    var pre: ?Block = null;
    for (0..func.blockCount()) |pi| {
        const p: Block = @fromBackingInt(@intCast(pi));
        if (p == found.tail or p == header or p == found.latch) continue;
        if (!jumpsTo(func, p, header)) continue;
        if (pre != null) return null;
        pre = p;
    }
    found.preheader = pre orelse return null;
    if (func.terminator(found.preheader) == null) return null;
    for (func.blockInsts(found.preheader)) |inst| {
        if (func.opcode(inst) == .@"if") return null;
    }
    if (found.exit == header or found.exit == found.tail) return null;
    return found;
}

/// The loop `header` heads when the test is in the header, which is a `while` loop.
fn topTestLoop(
    func: *const Function,
    header: Block,
    divergent: []const bool,
    masks: []const LoopMask,
) ?MultiLoop {
    const insts = func.blockInsts(header);
    const cf = func.opcode(insts[insts.len - 1]).@"if";
    if (!varies(divergent, cf.cond)) return null;
    // One side enters the body and the other leaves. The body is the side that comes back.
    for ([_]bool{ true, false }) |on_true| {
        const entering = if (on_true) cf.then else cf.@"else";
        const leaving = if (on_true) cf.@"else" else cf.then;
        var m: MultiLoop = .{
            .header = header,
            .preheader = header,
            .tail = header,
            .latch = header,
            .exit = leaving.target,
            .cond = cf.cond,
            .test_at_top = true,
            .enter_on_true = on_true,
            .entering = entering,
            .leaving = leaving,
        };
        if (!walkBody(func, &m, entering.target, divergent, masks)) continue;
        // The values that go round are named by the tail, so they line up with the parameters.
        const round = switch (func.terminator(m.tail) orelse continue) {
            .jump => |j| j,
            .ret => continue,
        };
        if (func.blockArgs(round).len != func.blockParams(header).len) continue;
        // The branch passes the body nothing, so the body names what it reads.
        if (func.blockArgs(entering).len != func.blockParams(entering.target).len) continue;
        return m;
    }
    return null;
}

/// The loop `header` heads when the test is in the block the iteration ends in, which is a
/// `do`-`while`. The header only carries the values round, so the body starts where it jumps.
fn bottomTestLoop(
    func: *const Function,
    header: Block,
    divergent: []const bool,
    masks: []const LoopMask,
) ?MultiLoop {
    const first = switch (func.terminator(header) orelse return null) {
        .jump => |j| j,
        .ret => return null,
    };
    var w: MultiLoop = .{
        .header = header,
        .preheader = header,
        .tail = header,
        .latch = header,
        .exit = header,
        .cond = undefined,
        .test_at_top = false,
        .enter_on_true = true,
        .entering = first,
        .leaving = first,
    };
    if (!walkBody(func, &w, first.target, divergent, masks)) return null;
    // The tail is where the iteration ends, so that is where the test is.
    const insts = func.blockInsts(w.tail);
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return null,
    };
    if (!varies(divergent, cf.cond)) return null;
    // One side goes round, through the empty block the edge split put there, and the other leaves.
    const then_round = isLatchOf(func, cf.then.target, header);
    const else_round = isLatchOf(func, cf.@"else".target, header);
    if (then_round == else_round) return null;
    w.cond = cf.cond;
    w.enter_on_true = then_round;
    w.latch = if (then_round) cf.then.target else cf.@"else".target;
    w.leaving = if (then_round) cf.@"else" else cf.then;
    w.exit = w.leaving.target;
    // The branch passes the latch nothing, so what goes round is what the latch names.
    if (func.blockArgs(if (then_round) cf.then else cf.@"else").len != 0) return null;
    const round = switch (func.terminator(w.latch).?) {
        .jump => |j| j,
        .ret => return null,
    };
    if (func.blockArgs(round).len != func.blockParams(header).len) return null;
    if (func.blockArgs(first).len != func.blockParams(first.target).len) return null;
    return w;
}

/// A loop inside another loop's body, read from its shape alone. The blocks are the header and the
/// chain that comes back to it, which is one block before the `while` rotation and two after.
const InnerLoop = struct {
    pre: Block,
    exit: Block,
    blocks: [max_inner_blocks]Block = undefined,
    count: usize = 0,
};

/// The most blocks one inner loop may own. A deeper nest than this is refused.
const max_inner_blocks = 8;

/// The loop `header` heads, read without reference to divergence, or null when it heads none.
fn innerLoop(func: *const Function, header: Block) ?InnerLoop {
    const insts = func.blockInsts(header);
    if (insts.len == 0) return null;
    const cf = switch (func.opcode(insts[insts.len - 1])) {
        .@"if" => |x| x,
        else => return null,
    };
    if (predCount(func, header) != 2) return null;
    var m: InnerLoop = .{ .pre = header, .exit = header };
    m.blocks[0] = header;
    m.count = 1;
    // One side comes back and the other leaves.
    const then_back = chainReaches(func, cf.then.target, header, &m);
    const back_count = m.count;
    const else_back = chainReaches(func, cf.@"else".target, header, &m);
    if (then_back == else_back) return null;
    if (then_back) m.count = back_count;
    m.exit = if (then_back) cf.@"else".target else cf.then.target;
    if (m.exit == header) return null;
    // Where it is entered from: the way in that is not the way round.
    var pre: ?Block = null;
    for (0..func.blockCount()) |pi| {
        const p: Block = @fromBackingInt(@intCast(pi));
        var own = false;
        for (m.blocks[0..m.count]) |b| {
            if (b == p) own = true;
        }
        if (own) continue;
        if (!jumpsTo(func, p, header)) continue;
        if (pre != null) return null;
        pre = p;
    }
    m.pre = pre orelse return null;
    return m;
}

/// Whether the chain from `from` comes back to `header`, recording the blocks it passes on the way.
fn chainReaches(func: *const Function, from: Block, header: Block, m: *InnerLoop) bool {
    var cur = from;
    while (true) {
        if (cur == header) return true;
        if (m.count == max_inner_blocks) return false;
        if (predCount(func, cur) != 1) return false;
        if (endsInIf(func, cur)) return false;
        m.blocks[m.count] = cur;
        m.count += 1;
        switch (func.terminator(cur) orelse return false) {
            .jump => |j| cur = j.target,
            .ret => return false,
        }
    }
}

/// Follow the body from `first` to the block that goes round. False when the body is not a chain
/// this reads.
fn walkBody(
    func: *const Function,
    m: *MultiLoop,
    first: Block,
    divergent: []const bool,
    masks: []const LoopMask,
) bool {
    _ = divergent;
    _ = masks;
    var cur = first;
    while (true) {
        if (cur == m.header) return false;
        if (m.count == max_body_blocks) return false;
        // A block that branches inside the body can only be an inner LOOP. The warp has one
        // program counter, so anything else was flattened already or is refused.
        //
        // Read WITHOUT asking whether the inner test varies per lane, because it does not yet: an
        // inner bound that comes from the outer loop's parameter looks the same in every lane
        // until the outer conversion freezes that parameter per lane. The conversion iterates for
        // the same reason.
        if (endsInIf(func, cur)) {
            // The block the iteration ends in when the test is at the BOTTOM: one of its edges
            // goes round, through the block the edge split left there. Its own code belongs to the
            // body like the rest.
            const cf = func.opcode(func.blockInsts(cur)[func.blockInsts(cur).len - 1]).@"if";
            if (isLatchOf(func, cf.then.target, m.header) or isLatchOf(func, cf.@"else".target, m.header)) {
                if (predCount(func, cur) != 1) return false;
                m.tail = cur;
                m.body[m.count] = cur;
                m.count += 1;
                return m.inner_count != 0;
            }
            const inner = innerLoop(func, cur) orelse return false;
            if (m.inner_count == max_body_blocks) return false;
            if (m.owned_count + inner.count > max_body_blocks) return false;
            m.inner[m.inner_count] = cur;
            m.inner_pre[m.inner_count] = inner.pre;
            m.inner_count += 1;
            for (inner.blocks[0..inner.count]) |b| {
                m.owned[m.owned_count] = b;
                m.owned_count += 1;
            }
            cur = inner.exit;
            continue;
        }
        // One way in, so the body is a chain and the mask of a block is the mask of the body.
        if (predCount(func, cur) != 1) return false;
        for (func.blockInsts(cur)) |inst| {
            if (!convertible(func.opcode(inst))) return false;
        }
        m.body[m.count] = cur;
        m.count += 1;
        switch (func.terminator(cur) orelse return false) {
            .jump => |j| {
                if (j.target == m.header) {
                    m.tail = cur;
                    // Without an inner loop the body would have been flattened into the header,
                    // and `divergentLoop` reads that shape. This path is for the one it cannot.
                    return m.inner_count != 0;
                }
                cur = j.target;
            },
            .ret => return false,
        }
    }
}

/// Rewrite a loop whose body spans blocks, or report false when a condition only the rewrite can
/// see refuses it.
fn convertMultiBlockLoop(
    allocator: std.mem.Allocator,
    func: *Function,
    m: MultiLoop,
    masks: *std.ArrayList(LoopMask),
    latches: *std.ArrayList(Inst),
) Error!bool {
    const bool_ty = try func.types.intern(.bool);
    const lane = laneValue(func) orelse return false;
    // A loop the lanes leave on the FALSE edge needs the complement of its condition. See
    // `complementOf`.
    if (!m.enter_on_true and complementOf(func, m.cond) == null) return false;
    const header = m.header;
    const carried = func.blockParams(header).len;

    // The mask the first iteration runs under: a region flattened around this loop may have
    // decided it, and otherwise it is every lane that reached the loop.
    const all = blk: {
        for (masks.items) |mm| {
            if (mm.header == header) break :blk mm.mask;
        }
        break :blk try func.appendInst(m.preheader, bool_ty, .{ .icmp = .{
            .op = .eq,
            .lhs = lane,
            .rhs = lane,
        } });
    };
    const active = try func.appendBlockParam(header, bool_ty);
    try appendJumpArg(allocator, func, m.preheader, all);

    // The header's own code runs for every lane still going. With the test there its branch comes
    // off, so the body follows unconditionally and the lanes the test turns away run it with the
    // mask shut.
    var next: Value = active;
    if (m.test_at_top) {
        const head_insts = try allocator.dupe(Inst, func.blockInsts(header));
        defer allocator.free(head_insts);
        func.blockInstsMut(header).clearRetainingCapacity();
        for (head_insts[0 .. head_insts.len - 1]) |inst| {
            try appendMasked(allocator, func, header, inst, active);
        }
        const again = if (m.enter_on_true)
            m.cond
        else
            (try appendComplement(func, header, m.cond, bool_ty)).?;
        next = try maskAnd(func, header, active, again, bool_ty);
        func.terminatorPtr(header).* = .{ .jump = m.entering };
    } else {
        try maskInPlace(allocator, func, header, active, false);
    }

    // The body, under the mask of the iteration it runs in: with the test at the bottom every lane
    // that entered the iteration ran the body, so that is the mask the loop arrived with. An inner
    // loop's own blocks take it too: a uniform inner loop gets no mask of its own, and a divergent
    // one ANDs what it adds with this.
    const body_mask = next;
    for (m.body[0..m.count]) |b| try maskInPlace(allocator, func, b, body_mask, false);
    for (m.owned[0..m.owned_count]) |b| try maskInPlace(allocator, func, b, body_mask, false);

    // An inner loop takes the body's mask ON TOP of the one it already had, which is what says a
    // lane the test turned away takes no part in that loop either. The AND goes where the inner
    // loop is entered, because that is the one place both masks are in reach.
    for (m.inner[0..m.inner_count], m.inner_pre[0..m.inner_count]) |h, pre| {
        var had = false;
        for (masks.items) |*mm| {
            if (mm.header != h) continue;
            mm.mask = try maskAnd(func, pre, mm.mask, body_mask, bool_ty);
            had = true;
            break;
        }
        if (!had) try masks.append(allocator, .{ .header = h, .mask = body_mask });
    }

    // With the test at the bottom the mask that goes round is only known once the body has run,
    // and the tail is where that is. The branch comes off while everything else goes in, because
    // a branch is the last thing its block does.
    const held: ?Inst = if (m.test_at_top) null else func.blockInstsMut(m.tail).pop().?;
    if (!m.test_at_top) {
        const again = if (m.enter_on_true)
            m.cond
        else
            (try appendComplement(func, m.tail, m.cond, bool_ty)).?;
        next = try maskAnd(func, m.tail, active, again, bool_ty);
    }

    // What goes round is named by the block the back edge leaves from.
    const goes_round = if (m.test_at_top) m.tail else m.latch;
    const back_args = try allocator.dupe(Value, func.valueList(func.terminator(goes_round).?.jump.args));
    defer allocator.free(back_args);
    const exit_args = try allocator.dupe(Value, func.valueList(m.leaving.args));
    defer allocator.free(exit_args);

    // A lane the mask has dropped keeps the value it had when it was dropped. A lane leaving now
    // ran the HEADER but not the body, so what it keeps is the parameter and not this pass.
    var frozen: std.ArrayList(Value) = .empty;
    defer frozen.deinit(allocator);
    // With the test at the bottom the lane ran the whole body, so what it keeps is this pass.
    const freeze_mask = if (m.test_at_top) next else active;
    for (0..carried) |i| {
        const param = func.blockParams(header)[i];
        try frozen.append(allocator, try func.appendInst(m.tail, func.valueType(param), .{ .select = .{
            .cond = freeze_mask,
            .then = back_args[i],
            .@"else" = param,
        } }));
    }

    // A value the loop computes that something outside reads needs a frozen copy, the same as for
    // a one-block body. The header's code runs under `active` and the body's under `next`, so the
    // two are asked separately.
    var slots: std.ArrayList(CarriedValue) = .empty;
    defer slots.deinit(allocator);
    var inside: std.ArrayList(Block) = .empty;
    defer inside.deinit(allocator);
    try inside.append(allocator, header);
    try inside.appendSlice(allocator, m.body[0..m.count]);
    try inside.appendSlice(allocator, m.owned[0..m.owned_count]);
    if (!m.test_at_top) try inside.append(allocator, m.latch);
    if (!try carrySlots(allocator, func, inside.items, &.{header}, header, m.preheader, m.tail, active, &slots)) return false;
    if (!try carrySlots(allocator, func, inside.items, m.body[0..m.count], header, m.preheader, m.tail, body_mask, &slots)) return false;

    var out: std.ArrayList(Value) = .empty;
    defer out.deinit(allocator);
    for (exit_args) |v| {
        const slotted = blk: {
            for (slots.items) |s| {
                if (s.value == v) break :blk s.frozen;
            }
            break :blk exitValue(func, header, v, back_args, frozen.items) orelse return false;
        };
        try out.append(allocator, slotted);
    }

    var round: std.ArrayList(Value) = .empty;
    defer round.deinit(allocator);
    try round.appendSlice(allocator, frozen.items);
    try round.append(allocator, next);
    for (slots.items) |s| try round.append(allocator, s.frozen);

    // The loop runs again while ANY lane is left, which is a reduction of the mask and not a
    // per-lane branch.
    if (m.test_at_top) {
        // The jump round becomes that branch, which makes the back edge critical: the edges are
        // split again after this pass.
        try func.appendIf(m.tail, next, .{
            .target = header,
            .args = round.items,
        }, .{
            .target = m.exit,
            .args = out.items,
        });
        // A block that ends in a branch has NO terminator: the branch is the edge, and a
        // terminator beside it is one the allocator would resolve and the emitter would drop.
        func.terminatorPtr(m.tail).* = null;
    } else {
        // The branch goes back on, and the back edge already has its block, so what goes round is
        // written on that block's jump.
        const branch = held.?;
        try func.blockInstsMut(m.tail).append(allocator, branch);
        func.opcodeMut(branch).* = .{ .@"if" = .{
            .cond = next,
            .then = .{ .target = m.latch, .args = try func.internValueList(&.{}) },
            .@"else" = .{ .target = m.exit, .args = try func.internValueList(out.items) },
        } };
        func.terminatorPtr(m.latch).*.?.jump.args = try func.internValueList(round.items);
    }
    const tail_insts = func.blockInsts(m.tail);
    try latches.append(allocator, tail_insts[tail_insts.len - 1]);
    return true;
}

fn convertDivergentLoops(
    allocator: std.mem.Allocator,
    func: *Function,
    divergent: []const bool,
    masks: *std.ArrayList(LoopMask),
    latches: *std.ArrayList(Inst),
) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        // A loop this already converted, which the rounds would otherwise convert again: the
        // reduction branch is both the marker and the thing that would be rewritten.
        if (alreadyConverted(func, block, latches.items)) continue;
        if (divergentLoop(func, block, divergent, masks.items)) |loop| {
            if (try convertLoop(allocator, func, loop, latches)) changed = true;
            continue;
        }
        // A body that spans blocks, which an inner loop makes. The outer one goes first, so the
        // mask it hands the inner one is in place before the inner one is rewritten.
        const wide = multiBlockLoop(func, block, divergent, masks.items) orelse continue;
        // With the test at the bottom the marker sits on the TAIL, so the check above looked at
        // the wrong block. Without this the rounds convert the same loop again and again.
        if (alreadyConverted(func, wide.tail, latches.items)) continue;
        if (try convertMultiBlockLoop(allocator, func, wide, masks, latches)) changed = true;
    }
    return changed;
}

/// Whether the branch `block` ends with is one a conversion already made.
fn alreadyConverted(func: *const Function, block: Block, latches: []const Inst) bool {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return false;
    for (latches) |l| {
        if (l == insts[insts.len - 1]) return true;
    }
    return false;
}

/// Rewrite one loop, or report false when a condition only the rewrite can see refuses it.
fn convertLoop(
    allocator: std.mem.Allocator,
    func: *Function,
    loop: DivergentLoop,
    latches: *std.ArrayList(Inst),
) Error!bool {
    const bool_ty = try func.types.intern(.bool);
    const lane = laneValue(func) orelse return false;
    // A loop the lanes leave on the FALSE edge needs the complement of its condition. See
    // `complementOf`.
    if (!loop.continue_on_true and complementOf(func, loop.cond) == null) return false;
    const header = loop.header;

    // The mask the first iteration runs under. Without a guard that is every lane which reached
    // the loop, and comparing the lane index against itself says so: true in every lane, and a
    // per-lane value by its operands, so it reaches the hardware as a real mask rather than a
    // broadcast 1. With a guard it is the guard's condition, or the complement of it when the
    // loop is entered on the false side.
    const all = if (loop.guard) |g| blk: {
        if (loop.guard_on_true) break :blk g;
        // Before the guard's branch, which is still there until below.
        break :blk switch (complementOf(func, g).?) {
            .have => |v| v,
            .compare => |c| try appendBeforeBranch(allocator, func, loop.guard_block.?, bool_ty, .{ .icmp = c }),
        };
    } else try func.appendInst(loop.preheader, bool_ty, .{ .icmp = .{
        .op = .eq,
        .lhs = lane,
        .rhs = lane,
    } });
    const carried = func.blockParams(header).len;
    const active = try func.appendBlockParam(header, bool_ty);
    try appendJumpArg(allocator, func, loop.preheader, all);
    if (loop.guard_block) |guard_block| {
        // The guard is the mask now, so the branch it was goes away and control simply enters the
        // loop. The lanes it would have turned away enter INACTIVE instead, run no iteration of
        // the body, and leave by the loop's own exit edge, which is where the guard sent them.
        const g_inst = func.blockInstsMut(guard_block).pop().?;
        const g = func.opcode(g_inst).@"if";
        const enter = if (loop.guard_on_true) g.then else g.@"else";
        func.terminatorPtr(guard_block).* = .{ .jump = .{
            .target = loop.preheader,
            .args = enter.args,
        } };
        if (loop.guard_fail) |missed| {
            // The lanes the guard turned away did not run the exit block and now they reach it, so
            // every access there takes the guard as its mask. The arithmetic around them is left
            // alone: it reads values from an iteration those lanes never ran, and the selects
            // below throw the answers away.
            const tail = try allocator.dupe(Inst, func.blockInsts(loop.exit));
            defer allocator.free(tail);
            func.blockInstsMut(loop.exit).clearRetainingCapacity();
            for (tail) |inst| try appendMasked(allocator, func, loop.exit, inst, all);

            // A lane that failed the guard leaves with the values the guard's own side carried.
            // It runs the exit block now, so the choice between the two belongs there.
            const theirs = try allocator.dupe(Value, func.valueList(missed));
            defer allocator.free(theirs);
            const ours = try allocator.dupe(Value, func.blockArgs(func.terminator(loop.exit).?.jump));
            defer allocator.free(ours);
            var merged: std.ArrayList(Value) = .empty;
            defer merged.deinit(allocator);
            for (ours, theirs) |took, missed_val| {
                try merged.append(allocator, try func.appendInst(loop.exit, func.valueType(took), .{ .select = .{
                    .cond = g.cond,
                    .then = if (loop.guard_on_true) took else missed_val,
                    .@"else" = if (loop.guard_on_true) missed_val else took,
                } }));
            }
            func.terminatorPtr(loop.exit).*.?.jump.args = try func.internValueList(merged.items);
        }
    }

    // The body, with every store put under the mask of the iteration it runs in. A lane that is
    // finished takes no part: an inactive lane makes no bus access at all.
    const body = func.blockInstsMut(header);
    const original = try allocator.dupe(Inst, body.items);
    defer allocator.free(original);
    const branch = original[original.len - 1];
    body.clearRetainingCapacity();
    const cf = func.opcode(branch).@"if";
    const exit_edge = if (loop.continue_on_true) cf.@"else" else cf.then;
    // What goes round is named by the LATCH, which is where the back edge's arguments live.
    const back_args = try allocator.dupe(Value, func.valueList(func.terminator(loop.latch).?.jump.args));
    defer allocator.free(back_args);
    const exit_args = try allocator.dupe(Value, func.valueList(exit_edge.args));
    defer allocator.free(exit_args);

    for (original[0 .. original.len - 1]) |inst| try appendMasked(allocator, func, header, inst, active);

    // Which lanes go round again: the ones still active that also want another iteration.
    //
    // The AND with `active` cannot be observed TODAY and is still right to keep. A finished lane
    // has its values frozen, so the body recomputes the same condition for it and it stays out
    // with or without the AND. That holds only while the body reads no memory: once a masked load
    // is admitted, another lane's store could change what a frozen lane loads and bring it back
    // to life. The AND is what makes that impossible rather than unlikely.
    const again = if (loop.continue_on_true)
        loop.cond
    else
        (try appendComplement(func, header, loop.cond, bool_ty)).?;
    const next = try func.appendInst(header, bool_ty, .{ .arith = .{
        .op = .bit_and,
        .lhs = active,
        .rhs = again,
    } });

    // A lane the mask has dropped keeps the value it had when it was dropped, which is this
    // iteration's for the lane that is leaving now and the frozen one for a lane that left
    // earlier. The select on the mask AT THE TOP of the iteration says exactly that.
    var frozen: std.ArrayList(Value) = .empty;
    defer frozen.deinit(allocator);
    for (0..carried) |i| {
        const param = func.blockParams(header)[i];
        try frozen.append(allocator, try func.appendInst(header, func.valueType(param), .{ .select = .{
            .cond = active,
            .then = back_args[i],
            .@"else" = param,
        } }));
    }

    // A value the body computes that something outside the loop reads needs a frozen copy of its
    // own. This runs before the exit edge is rebuilt, because an argument on that edge is one of
    // the reads it answers for.
    var slots: std.ArrayList(CarriedValue) = .empty;
    defer slots.deinit(allocator);
    if (!try carrySlots(
        allocator,
        func,
        &.{ header, loop.latch },
        &.{header},
        header,
        loop.preheader,
        header,
        active,
        &slots,
    )) return false;

    // The exit carries the frozen values too: the loop is left when NO lane is active, so this
    // iteration's results belong to no one.
    var out: std.ArrayList(Value) = .empty;
    defer out.deinit(allocator);
    for (exit_args) |v| {
        const slotted = blk: {
            for (slots.items) |s| {
                if (s.value == v) break :blk s.frozen;
            }
            break :blk exitValue(func, header, v, back_args, frozen.items) orelse return false;
        };
        try out.append(allocator, slotted);
    }

    // The mask goes round with the rest, and it needs no select of its own: a lane the AND has
    // dropped stays dropped, so the mask freezes itself.
    var round: std.ArrayList(Value) = .empty;
    defer round.deinit(allocator);
    try round.appendSlice(allocator, frozen.items);
    try round.append(allocator, next);
    // In the order the slots were given their parameters.
    for (slots.items) |s| try round.append(allocator, s.frozen);

    // What goes round is written on the LATCH's jump, and the mask goes with it.
    func.terminatorPtr(loop.latch).*.?.jump.args = try func.internValueList(round.items);

    // The back edge is put on the THEN side whatever the original polarity was. The condition is
    // now a MASK and the branch is a reduction of it, so which way the lanes used to go has no
    // meaning here: the loop runs again while any lane is left.
    func.opcodeMut(branch).* = .{ .@"if" = .{
        .cond = next,
        .then = .{ .target = loop.latch, .args = try func.internValueList(&.{}) },
        .@"else" = .{ .target = loop.exit, .args = try func.internValueList(out.items) },
    } };
    try func.blockInstsMut(header).append(allocator, branch);
    try latches.append(allocator, branch);
    return true;
}

/// A value the body computes that something outside the loop reads, and the frozen copy of it the
/// outside reads instead. See `carrySlots`.
const CarriedValue = struct { value: Value, frozen: Value };

/// Give every value the body computes that something OUTSIDE the loop reads a frozen copy, and
/// report false when one of them cannot have it.
///
/// A lane that stops early keeps its parameters frozen, so the body goes on recomputing from them
/// and lands ONE ITERATION PAST the one that lane last took part in. Inside the loop that is
/// harmless, because nothing reads the result of an iteration a lane sat out. Outside it is a
/// wrong answer: the block the loop leaves by would read the value as though that iteration had
/// happened. The frozen copy is the same select the carried values get, so the lane leaving now
/// contributes this iteration and a lane that left earlier keeps what it had.
///
/// The slot starts at zero from the preheader. Only a lane that runs no iteration at all reads
/// that, and such a lane took no part in anything the value feeds. A BOOL slot starts at a per-lane
/// false rather than a zero: it is a mask, and the broadcast that gives a warp-uniform value to a
/// per-lane parameter adds a per-lane zero, which is one value per element where a mask is one BIT
/// per lane. Comparing the lane index against itself is false in every lane and per-lane by its
/// operands, so it arrives as a real mask.
fn carrySlots(
    allocator: std.mem.Allocator,
    func: *Function,
    inside: []const Block,
    produce: []const Block,
    header: Block,
    preheader: Block,
    dest: Block,
    took_part: Value,
    slots: *std.ArrayList(CarriedValue),
) Error!bool {
    var computed: std.ArrayList(Value) = .empty;
    defer computed.deinit(allocator);
    for (produce) |b| {
        for (func.blockInsts(b)) |inst| {
            if (func.instResult(inst)) |v| try computed.append(allocator, v);
        }
    }
    if (computed.items.len == 0) return true;

    // Counted with the loop's own code OUT OF SIGHT, so what is left is the reads from outside.
    // The header's branch is not in the block at this point, so the exit edge's arguments are not
    // counted either: `exitValue` answers for those.
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    var hidden_insts: std.ArrayList([]Inst) = .empty;
    defer {
        for (hidden_insts.items) |h| allocator.free(h);
        hidden_insts.deinit(allocator);
    }
    var hidden_args: std.ArrayList(ir.function.ValueList) = .empty;
    defer hidden_args.deinit(allocator);
    const none = try func.internValueList(&.{});
    for (inside) |b| {
        try hidden_insts.append(allocator, try allocator.dupe(Inst, func.blockInsts(b)));
        func.blockInstsMut(b).clearRetainingCapacity();
        // A block whose last instruction is a branch has no terminator of its own, and the branch
        // went out of sight with the instructions.
        if (func.terminatorPtr(b).*) |*t| switch (t.*) {
            .jump => |*j| {
                try hidden_args.append(allocator, j.args);
                j.args = none;
            },
            .ret => try hidden_args.append(allocator, none),
        } else try hidden_args.append(allocator, none);
    }
    opt.dce.countUses(func, uses);
    for (inside, 0..) |b, i| {
        try func.blockInstsMut(b).appendSlice(allocator, hidden_insts.items[i]);
        if (func.terminatorPtr(b).*) |*t| switch (t.*) {
            .jump => |*j| j.args = hidden_args.items[i],
            .ret => {},
        };
    }

    for (computed.items) |v| {
        if (uses[@backingInt(v)] == 0) continue;
        const ty = func.valueType(v);
        const zero = switch (func.types.type_kind(ty)) {
            .int => try func.appendInst(preheader, ty, .{ .iconst = 0 }),
            .float => try func.appendInst(preheader, ty, .{ .fconst = 0 }),
            .bool => blk: {
                const lane = laneValue(func) orelse return false;
                break :blk try func.appendInst(preheader, ty, .{ .icmp = .{
                    .op = .ne,
                    .lhs = lane,
                    .rhs = lane,
                } });
            },
            else => return false,
        };
        const slot = try func.appendBlockParam(header, ty);
        const frozen = try func.appendInst(dest, ty, .{ .select = .{
            .cond = took_part,
            .then = v,
            .@"else" = slot,
        } });
        // Every block but the loop's own: inside, the body reads this iteration's value.
        for (0..func.blockCount()) |bi| {
            const b: Block = @fromBackingInt(@intCast(bi));
            var own = false;
            for (inside) |x| {
                if (x == b) own = true;
            }
            if (own) continue;
            func.replaceUsesIn(b, v, frozen);
        }
        try appendJumpArg(allocator, func, preheader, zero);
        try slots.append(allocator, .{ .value = v, .frozen = frozen });
    }
    return true;
}

/// What the exit edge passes in place of `v`: the frozen value when it is one the loop carries,
/// and `v` itself when it comes from outside. Null when it is neither, which is a value computed
/// in the body that only the exit reads, and that would need its own carried slot.
fn exitValue(
    func: *const Function,
    block: Block,
    v: Value,
    back_args: []const Value,
    frozen: []const Value,
) ?Value {
    for (back_args, 0..) |arg, i| {
        if (arg == v) return frozen[i];
    }
    for (func.blockParams(block), 0..) |p, i| {
        if (p != v) continue;
        if (i >= frozen.len) return null; // the mask parameter itself
        return frozen[i];
    }
    const inst = func.definingInst(v) orelse return v; // a parameter of another block
    for (func.blockInsts(block)) |in_body| {
        if (in_body == inst) return null;
    }
    return v;
}

/// The lane index a kernel reads, whichever builtin it asked for.
fn laneValue(func: *const Function) ?Value {
    for (func.blockParams(@fromBackingInt(0))) |p| {
        const b = gpu.attrs.builtinOf(func, p) orelse continue;
        if (laneVarying(b)) return p;
    }
    return null;
}

/// Append `v` to the argument list of the jump leaving `block`.
fn appendJumpArg(allocator: std.mem.Allocator, func: *Function, block: Block, v: Value) Error!void {
    const term = func.terminatorPtr(block);
    const old = term.*.?.jump;
    var args: std.ArrayList(Value) = .empty;
    defer args.deinit(allocator);
    try args.appendSlice(allocator, func.valueList(old.args));
    try args.append(allocator, v);
    term.*.?.jump.args = try func.internValueList(args.items);
}

/// Whether `block` leaves by a jump to `target`.
fn jumpsTo(func: *const Function, block: Block, target: Block) bool {
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .@"if" => |cf| return cf.then.target == target or cf.@"else".target == target,
        else => {},
    };
    return switch (func.terminator(block) orelse return false) {
        .jump => |j| j.target == target,
        .ret => false,
    };
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
        // Every scale a real index produces is a POWER OF TWO, so the rebuild uses a SHIFT and
        // never a multiply. The scale of a term is the product of the constants between the
        // address and the term, and the only ones this peels through are on the path to the lane,
        // whose total has to come out at one element. So a term's scale divides the element size.
        //
        // That is an argument about the shapes an index takes and not a proof, and a contrived one
        // (`lane + 3 * (u + lane)`) reaches a scale of three with a stride of four. Refusing that
        // is what makes the shift GUARANTEED rather than usual, and it matters past the lowering:
        // a 64-bit multiply is expensive on an FPGA and a 64-bit shift by a constant is not, so
        // the core need not decode one for an address.
        const shift = powerOfTwoShift(t.scale) orelse return null;
        const scaled = if (shift == 0) t.value else try func.appendInst(
            block,
            func.valueType(t.value),
            .{ .arith_imm = .{ .op = .shl, .lhs = t.value, .imm = shift } },
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

/// The shift that multiplies by `scale`, or null when `scale` is not a positive power of two.
fn powerOfTwoShift(scale: i64) ?i64 {
    if (scale <= 0) return null;
    if (scale & (scale - 1) != 0) return null;
    return std.math.log2_int(u64, @intCast(scale));
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
fn checkSupported(
    func: *const Function,
    opts: Options,
    divergent: []const bool,
    macs: *const Macs,
    any_lane: *const AnyLane,
    why: ?*Refusal,
) Error!void {
    for (0..func.valueCount()) |vi| {
        const v: Value = @fromBackingInt(@intCast(vi));
        if (!isWord(func, func.valueType(v))) {
            return Refusal.whole(why, "a value wider than the 32-bit word a register holds");
        }
    }
    var args: usize = 0;
    for (func.blockParams(@fromBackingInt(0))) |p| {
        if (isIgnoredEntryParam(func, p)) continue;
        const b = gpu.attrs.builtinOf(func, p) orelse {
            // An entry parameter with no builtin is a KERNEL PARAMETER, read from a launch
            // argument CSR. See `launchArgOf` for the ordering rule.
            //
            // Its TYPE needs no check here: the loop above refuses any value this core cannot
            // hold in a register, so a parameter wider than a word is already gone.
            if (launchArgOf(args) == null) {
                return Refusal.whole(why, "more kernel parameters than the core has launch arguments");
            }
            args += 1;
            continue;
        };
        // A lane-varying builtin comes from `vid.v` rather than from a CSR, so it has no
        // `builtinSource`. See `lanewiseBuiltin`.
        if (!laneVarying(b)) _ = builtinSource(b) orelse
            return Refusal.whole(why, "a builtin the core cannot answer");
    }
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, i| {
            // The multiply and the two converts a MAC absorbs are emitted by the `add`, so the
            // rules below do not apply to them: on their own the core has neither instruction.
            if (macs.isFolded(inst)) continue;
            try checkInst(func, opts, divergent, macs, any_lane, block, insts, inst, i, why);
        }
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| if (r.count != 0) {
                return Refusal.whole(why, "a kernel that returns a value, which has no caller to return it to");
            },
            .jump => {},
        };
    }
}

fn checkInst(
    func: *const Function,
    opts: Options,
    divergent: []const bool,
    macs: *const Macs,
    any_lane: *const AnyLane,
    block: Block,
    insts: []const Inst,
    inst: Inst,
    i: usize,
    why: ?*Refusal,
) Error!void {
    switch (func.opcode(inst)) {
        // A constant must fit a 32-bit register. An unsigned type reaches 0xFFFFFFFF and a
        // signed one reaches the most negative value, so the window spans both.
        .iconst => |c| if (c < std.math.minInt(i32) or c > std.math.maxInt(u32)) return Refusal.at(why, func, block, i, inst, "a constant too wide for a 32-bit register"),
        // The result type is swept above, so it is already an f32 or an f16 here.
        .fconst => |c| _ = floatBits(func, inst, c) orelse return Refusal.at(why, func, block, i, inst, "a float constant in a format the core has no value for"),
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
                _ = maskBinOp(a.op) orelse return Refusal.at(why, func, block, i, inst, "a bitwise operation between per-lane bools that has no mask op");
                if (!divergent[@backingInt(a.lhs)] or !divergent[@backingInt(a.rhs)]) {
                    return Refusal.at(why, func, block, i, inst, "a mask combined with a warp-uniform bool");
                }
                return;
            }
            // Float arithmetic runs on the vector unit either way, so the table is nearly the same
            // whether or not the operands vary per lane. The divide is the exception: see
            // `floatVectorBinOp`.
            if (floatVectorBinOp(func, a.op, ty, resultDivergent(func, inst, divergent)) == null) {
                if (resultDivergent(func, inst, divergent)) {
                    _ = vectorBinOp(func, a.op, ty) orelse
                        return Refusal.at(why, func, block, i, inst, "a per-lane arithmetic operation the core has no instruction for");
                } else {
                    _ = binOp(func, a.op, ty) orelse
                        return Refusal.at(why, func, block, i, inst, "a warp-uniform arithmetic operation the core has no instruction for");
                }
            }
        },
        .arith_imm => |a| {
            // The immediate is an integer, so this opcode never carries a float operand.
            if (isFloat(func, func.valueType(a.lhs))) {
                return Refusal.at(why, func, block, i, inst, "float arithmetic against an immediate");
            }
            // A per-lane bool is a mask, so the only immediate that means anything against one is
            // the 1 of a NOT. Anything else would be an integer operation on the register a mask
            // lives in, which touches lane zero and no other lane.
            if (isBool(func, func.valueType(a.lhs)) and resultDivergent(func, inst, divergent)) {
                if (a.op != .bit_xor or a.imm != 1) {
                    return Refusal.at(why, func, block, i, inst, "an immediate against a per-lane mask that is not a NOT");
                }
                return;
            }
            _ = binOp(func, a.op, func.valueType(a.lhs)) orelse
                return Refusal.at(why, func, block, i, inst, "an operation against an immediate the core has no instruction for");
        },
        // A per-lane compare writes a MASK, one bit per lane, and a per-lane select reads one.
        .icmp => |c| {
            const ty = func.valueType(c.lhs);
            // A WARP-UNIFORM float compare has nothing to do it: the compares write a mask, so
            // they are vector instructions, and the scalar path has no float compare at all. An
            // integer compare over float bits answers a different relation and is wrong for a
            // negative value and for NaN, so it is a refusal rather than a substitution.
            if (resultDivergent(func, inst, divergent)) {
                _ = vectorCmpOp(func, c.op, ty) orelse
                    return Refusal.at(why, func, block, i, inst, "a per-lane compare the core has no instruction for");
            } else if (isFloat(func, ty)) {
                return Refusal.at(why, func, block, i, inst, "a warp-uniform float compare, which the core has only per lane");
            }
        },
        .select => {},
        .load => |l| {
            if (!isWord(func, func.valueType(l.ptr))) {
                return Refusal.at(why, func, block, i, inst, "a load through an address a register cannot hold");
            }
            const r = func.instResult(inst) orelse
                return Refusal.at(why, func, block, i, inst, "a load whose result nothing names");
            // An f16 is the low half of a lane, not a word in memory. See `isHalf`.
            if (isHalf(func, func.valueType(r))) {
                return Refusal.at(why, func, block, i, inst, "an f16 load, which is half a lane and never a word in memory");
            }
            // A mask has to be a real per-lane predicate, the same rule as a masked store: a
            // warp-uniform one says every lane agrees, and the answer for that is a branch over
            // the load rather than a predicate on it.
            if (l.mask) |m| {
                if (!isWord(func, func.valueType(m))) {
                    return Refusal.at(why, func, block, i, inst, "a load mask a register cannot hold");
                }
                if (!divergent[@backingInt(m)]) {
                    return Refusal.at(why, func, block, i, inst, "a load under a warp-uniform mask, which is a branch and not a predicate");
                }
                // A masked load leaves the lanes its mask excludes holding whatever the
                // destination held, so the result is only defined where the mask is set. That is
                // true of every per-lane op in a masked region, and the per-lane path is the only
                // one that can express it.
                if (!resultDivergent(func, inst, divergent)) {
                    return Refusal.at(why, func, block, i, inst, "a masked load whose result is not per-lane");
                }
            }
        },
        .store => |st| {
            if (!isWord(func, func.valueType(st.value))) {
                return Refusal.at(why, func, block, i, inst, "a store of a value wider than a word");
            }
            if (isHalf(func, func.valueType(st.value))) {
                return Refusal.at(why, func, block, i, inst, "an f16 store, which is half a lane and never a word in memory");
            }
            // A mask has to be a real per-lane predicate. A WARP-UNIFORM one says every lane
            // agrees, and the answer for that is a branch over the store rather than a predicate
            // on it, so it is refused rather than widened into a mask of all ones or all zeros.
            if (st.mask) |m| {
                if (!isWord(func, func.valueType(m))) {
                    return Refusal.at(why, func, block, i, inst, "a store mask a register cannot hold");
                }
                if (!divergent[@backingInt(m)]) {
                    return Refusal.at(why, func, block, i, inst, "a store under a warp-uniform mask, which is a branch and not a predicate");
                }
            }
        },
        .atomic_rmw => |a| {
            _ = amoOp(func, a.op, func.valueType(a.value)) orelse
                return Refusal.at(why, func, block, i, inst, "an atomic the core has no AMO for, compare-and-swap among them");
            // Every lane would need its own read-modify-write, and the core has one bus. A per-lane
            // atomic is a loop over the lanes, which is not this. The vector emitter would refuse
            // it too; this is here because admission is where a refusal belongs.
            if (divergent[@backingInt(a.ptr)] or divergent[@backingInt(a.value)]) {
                return Refusal.at(why, func, block, i, inst, "a per-lane atomic, which would be a loop over the lanes");
            }
            if (!isWord(func, func.valueType(a.value))) {
                return Refusal.at(why, func, block, i, inst, "an atomic on a value wider than a word");
            }
        },
        .global_addr => |g| _ = addressOf(opts, func.symbolName(g.symbol)) orelse
            return Refusal.at(why, func, block, i, inst, "a global the host did not place at an address"),
        .unary => |u| if (u.op != .reinterpret)
            return Refusal.at(why, func, block, i, inst, "a unary operation that is not a reinterpret"),
        // A widening convert has no instruction of its own; the recognizer turns one into a MAC.
        .convert => |cv| if (macs.at(inst) == null) {
            _ = convertKind(func, inst, cv) orelse
                return Refusal.at(why, func, block, i, inst, "a convert between two types the core has no instruction for");
        },
        .barrier => |b| switch (b.scope) {
            .workgroup, .subgroup => {},
        },
        // A workgroup object: a window in the hart's scratch region, which a warp owns. A warp has
        // no stack, so every OTHER address space is refused here, and `private` is the one a
        // frontend reaches for by mistake.
        .alloca => {
            const res = func.instResult(inst) orelse
                return Refusal.at(why, func, block, i, inst, "an allocation whose address nothing reads");
            switch (func.types.type_kind(func.valueType(res))) {
                .ptr => |space| switch (space) {
                    .shared => {},
                    .global, .constant, .private => return Refusal.at(why, func, block, i, inst, "an allocation outside workgroup memory, which a warp has no stack for"),
                },
                else => return Refusal.at(why, func, block, i, inst, "an allocation that is not a pointer"),
            }
            if (divergent[@backingInt(res)]) {
                return Refusal.at(why, func, block, i, inst, "a per-lane workgroup address, which every lane of a workgroup shares");
            }
        },
        // A conditional transfers control, so nothing may follow it in its block, and its
        // block must have no terminator: a terminator there is an edge the allocator would
        // resolve and this emits nothing for.
        .@"if" => |cf| {
            if (i + 1 != insts.len or func.terminator(block) != null) {
                return Refusal.at(why, func, block, i, inst, "a branch that is not the last thing its block does");
            }
            // A branch on a per-lane condition is DIVERGENCE: the lanes would need to take
            // both arms under masks. A scalar branch here would send the whole warp whichever
            // way lane 0 went, which is a wrong answer and not a slow one.
            //
            // The ONE exception is a loop latch this lowering rewrote itself, where the condition
            // is the active mask and the branch is a reduction of it. See `AnyLane`.
            if (divergent[@backingInt(cf.cond)] and !any_lane.has(inst)) {
                return Refusal.at(why, func, block, i, inst, "a branch on a per-lane condition that if-conversion could not flatten");
            }
        },
        else => return Refusal.at(why, func, block, i, inst, "an opcode this lowering has no rule for"),
    }
}

/// The CSR a kernel parameter is read from, by its position among the entry parameters that are
/// NOT builtins.
///
/// That ordering is not invented here: `gpu.abi.layoutParams` places a kernel's parameters by
/// walking the entry parameters and skipping the builtins, so the i'th parameter a host writes is
/// the i'th non-builtin entry parameter on every target. NVIDIA delivers it from constant memory
/// and this core delivers it from a CSR, which is the only difference.
///
/// A parameter is WARP UNIFORM, and by construction rather than by analysis: one dispatch gives
/// every warp the same values. `analyzeDivergence` seeds divergence only from lane-varying
/// builtins, so nothing has to be told.
fn launchArgOf(index: usize) ?u12 {
    return glacier.Csr.launchArg(index);
}

/// Where a WARP-UNIFORM hardware builtin comes from, or null when the core cannot answer.
///
/// A warp is the workgroup: it owns a program counter and a register file. The dispatch CSR adds
/// the virtual batch offset to the physical hart id. The lanes inside a warp are the threads, so
/// a thread or lane index differs between the lanes and comes from `vid.v` instead.
fn builtinSource(b: gpu.builtin.Builtin) ?union(enum) { csr: u12, lanes } {
    return switch (b) {
        .block_id_x => .{ .csr = glacier.Csr.workgroup_id },
        .grid_dim_x => .{ .csr = glacier.Csr.grid_dim },
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
    if (isFloat(func, ty)) return floatVectorBinOp(func, op, ty, true);
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
/// So this list is NOT the lane's op list.
///
/// THE DIVIDE IS PER-LANE ONLY, and that is not an oversight of the table. The core has TWO float
/// dividers, deliberately unshared so a vector divide never waits on a scalar one, and both have
/// their mode tied to nearest even. So the quotient cannot say which one ran and the choice is
/// cost alone: `vfdiv.vv` walks the lanes, about 31 beats of recurrence for each active one, and
/// the warp parks for the whole walk, where `FDIV.S` divides once and leaves the vector write port
/// alone. A warp-uniform divide therefore stays scalar rather than compute the same quotient in
/// every lane, and `per_lane` is what keeps the two apart. The wider the core, the further apart
/// they are.
///
/// A lane the mask excludes is SKIPPED rather than divided and discarded, so a divide under a
/// narrow mask costs in proportion. A divide per span and a multiply per pixel is still the shape
/// to write: in an inner loop the walk is paid every time round.
///
/// f16 arithmetic is absent for a different reason: the element width is fixed at 32 bits, so these
/// read whole lanes. An f16 lives in the low half of one, which is the MAC's shape and not this.
fn floatVectorBinOp(
    func: *const Function,
    op: ir.function.BinOp,
    ty: Type,
    per_lane: bool,
) ?*const fn (VReg, VReg, VReg) u32 {
    if (!isF32(func, ty)) return null;
    return switch (op) {
        .add => rv.vfadd_vv,
        .sub => rv.vfsub_vv,
        .mul => rv.vfmul_vv,
        .div => if (per_lane) rv.vfdiv_vv else null,
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

/// Which scalar spill slots hold nothing, because every value that reaches one rematerialises.
///
/// A slot belongs to an INTERVAL and not to a value, and a move names the value whose bits it
/// carries, which are not always the same thing: the entry edge of a loop writes a CONSTANT into
/// the slot of the loop's parameter, and the body then reads that slot under the PARAMETER's name.
/// Eliding by value drops that write and leaves the read on a slot nothing wrote. So the question
/// has to be asked of the slot: a slot every one of whose values rematerialises is never written
/// and never read, and any other slot is left alone.
fn elidableSlots(
    allocator: std.mem.Allocator,
    alloc: *const wimmer.Allocation,
    remat: []const ?i32,
    slots: u32,
) Error![]const bool {
    const plain = try allocator.alloc(bool, slots);
    defer allocator.free(plain);
    @memset(plain, false);
    const touch = struct {
        fn at(p: []bool, r: []const ?i32, class: u16, loc: wimmer.Location, v: Value) void {
            if (class != scalar_class) return;
            switch (loc) {
                .slot => |n| if (n < p.len and r[@backingInt(v)] == null) {
                    p[n] = true;
                },
                .reg => {},
            }
        }
    };
    for (alloc.actions) |a| {
        touch.at(plain, remat, a.class, a.src, a.value);
        touch.at(plain, remat, a.class, a.dst, a.value);
    }
    for (alloc.edge_moves) |set| for (set.moves) |m| {
        touch.at(plain, remat, m.class, m.src, m.value);
        touch.at(plain, remat, m.class, m.dst, m.value);
    };
    const out = try allocator.alloc(bool, slots);
    for (out, plain) |*o, p| o.* = !p;
    return out;
}

/// The one-word constant each value rematerialises to, or null where it has to go through a spill
/// slot.
///
/// A constant that spills costs a store at its definition and a load at every reload. One that
/// fits ADDI's immediate takes ONE word to re-emit, which is what the load costs, so re-emitting
/// is never the worse trade and the store goes away with the slot. A wider constant takes two
/// words and keeps its slot. The single-use case is already gone by here: `opt.sink` moved those
/// next to their reader, so what is left is a constant several instructions read.
fn rematTable(allocator: std.mem.Allocator, func: *const Function) Error![]const ?i32 {
    const table = try allocator.alloc(?i32, func.valueCount());
    @memset(table, null);
    for (0..func.blockCount()) |bi| {
        const b: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(b)) |inst| {
            const res = func.instResult(inst) orelse continue;
            const value: i32 = switch (func.opcode(inst)) {
                .iconst => |c| lowWord(c),
                .fconst => |c| @bitCast(floatBits(func, inst, c) orelse continue),
                else => continue,
            };
            if (value < -2048 or value > 2047) continue;
            table[@backingInt(res)] = value;
        }
    }
    return table;
}

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
    /// Which branches ask whether any lane is still active. See `AnyLane`.
    any_lane: *const AnyLane,
    /// Which blocks the entry can still reach. The passes above leave emptied blocks behind and
    /// nothing branches to them, so emitting anything for one is a word wasted per block.
    live: []const bool,
    /// Where a refusal is recorded, when the caller asked for one. See `Refusal`.
    why: ?*Refusal,
    /// Whether the allocation uses the scratch region, so the prologue must load `sp`.
    spills: bool,
    /// What each value rematerialises to. See `rematTable`.
    remat: []const ?i32,
    /// Which spill slots hold nothing. See `elidableSlots`.
    elide: []const bool,
    /// The value whose mask bits are in `v0`, or null when the contents are unknown. `v0` is
    /// never allocatable, so the predicate installs are the only writes to it, and a value never
    /// changes after it is computed: a second install of the same mask in one block is dead.
    v0_mask: ?Value = null,
    /// How many cluster barriers were emitted, checked against the count the kernel carried
    /// before any pass ran. See `gpu.attrs.setClusterBarrier` for why an instruction attribute
    /// needs that check.
    cluster_barriers: u32 = 0,
    /// The placed workgroup slots, each an `alloca` and its offset in the window.
    shared: []const gpu.abi.SharedSlot,
    /// The bytes the workgroup window takes directly below `sp`. The spill slots start under it.
    shared_bytes: u32,
    /// Where the per-lane spill slots start, in bytes below `sp`. The scalar slots take the top of
    /// the region one word each, so the vector slots sit under them, one whole register each.
    vector_slot_base: u32,
    words: std.ArrayList(u32) = .empty,
    block_start: std.ArrayList(u32) = .empty,
    fixups: std.ArrayList(Fixup) = .empty,

    fn deinit(self: *Emitter) void {
        self.allocator.free(self.remat);
        self.allocator.free(self.elide);
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
            // A per-lane value the allocator put in a slot at its own definition. The scalar path
            // computes such a result in the scratch register and stores it after, which this
            // cannot copy: the vector scratch already carries a broadcast warp-uniform operand on
            // several paths here. A move to or from a slot between two instructions is handled,
            // and that is what the allocator emits for a value that spills after it is defined.
            .slot => error.Unsupported,
        };
    }

    /// Re-emit `value` into `dst`, in place of a load from a slot that holds nothing.
    ///
    /// A value with no constant here means a slot was elided that something plain reads, which is
    /// this lowering disagreeing with itself. It REFUSES rather than loading a word nobody wrote.
    fn rematerialise(self: *Emitter, dst: Reg, value: Value) Error!void {
        const c = self.remat[@backingInt(value)] orelse return error.Unsupported;
        try self.constant(dst, c);
    }

    /// Put the mask bits of `cond` in `v0`, which is the only register that predicates. A masked
    /// operation reads them from there and nowhere else.
    ///
    /// The install is spelled `vmand.mm v0, b, b`, so a reader counting mask operations in the
    /// stream sees an AND. A destination of `v0` makes it an install.
    fn predicate(self: *Emitter, cond: Value, pos: u32) Error!void {
        if (self.v0_mask) |have| {
            if (have == cond) return;
        }
        const bits = try self.vreg(cond, pos);
        try self.put(rv.vmand_mm(.v0, bits, bits));
        self.v0_mask = cond;
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

    /// Attach a position to a refusal the emitter raised, and pass anything else through.
    fn explain(self: *Emitter, e: Error, block: Block, idx: usize, inst: Inst) Error {
        if (e != error.Unsupported) return e;
        return Refusal.at(
            self.why,
            self.func,
            block,
            idx,
            inst,
            "an instruction admission accepted and emission could not express",
        );
    }

    fn run(self: *Emitter) Error!void {
        const nblocks = self.func.blockCount();
        try self.block_start.appendNTimes(self.allocator, 0, nblocks);

        var pos: u32 = 0;
        for (0..nblocks) |bi| {
            const block: Block = @fromBackingInt(@intCast(bi));
            self.block_start.items[bi] = @intCast(self.words.items.len);
            // A block can be entered from several predecessors, each leaving its own mask behind.
            self.v0_mask = null;
            // Nothing reaches this block, so nothing needs a word here. Its POSITIONS still
            // advance below, because the allocator numbered them whether or not they are emitted,
            // and an action it placed at one of them belongs to a value no longer read here.
            if (!self.live[bi]) {
                pos += 2 + @as(u32, @intCast(self.func.blockInsts(block).len));
                continue;
            }
            if (bi == 0) try self.prologue(block) else try self.edgeMovesInto(block);

            try self.actionsAt(pos); // the block-parameter position
            pos += 1;
            const insts = self.func.blockInsts(block);
            var ended = false;
            for (insts, 0..) |inst, idx| {
                try self.actionsAt(pos);
                // Admission accepted this instruction and emission could not express it, which is
                // the lowering disagreeing with itself rather than a kernel asking for too much.
                // Naming it here covers every refusal in the emitter from one place.
                if (self.func.opcode(inst) == .@"if") {
                    self.emitIf(inst, pos, bi) catch |e| return self.explain(e, block, idx, inst);
                    ended = true;
                } else {
                    self.emitInst(inst, pos) catch |e| return self.explain(e, block, idx, inst);
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
        if (self.spills) {
            try self.put(glacier.csrr(sp, glacier.Csr.mscratch));
            try self.checkScratch();
        }
        // Counted over EVERY non-builtin parameter, including one the allocator gave no home,
        // because the position is the host's argument number and not a register index.
        var args: usize = 0;
        for (self.func.blockParams(block)) |p| {
            if (isIgnoredEntryParam(self.func, p)) continue;
            const b = gpu.attrs.builtinOf(self.func, p);
            const arg: ?u12 = if (b == null) blk: {
                defer args += 1;
                break :blk launchArgOf(args) orelse return error.Unsupported;
            } else null;
            // A parameter nothing reads costs no instruction. The allocator still gives a dead
            // definition a register, so the home lookup below would not catch it, and a kernel
            // that declares eight arguments and reads one would pay eight CSR reads per launch.
            // The ARGUMENT NUMBER still advances: it is the host's, not the register file's.
            if (arg != null and self.uses[@backingInt(p)] == 0) continue;
            const home = self.locAt(p, 0) orelse continue;
            if (self.isDivergent(p)) {
                try self.lanewiseBuiltin(b orelse return error.Unsupported, try self.vreg(p, 0));
                continue;
            }
            const dst: Reg = switch (home) {
                .reg => |r| regOf(r),
                .slot => scratch,
            };
            if (arg) |csr| {
                try self.put(glacier.csrr(dst, csr));
            } else switch (builtinSource(b.?) orelse return error.Unsupported) {
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
        if (b != .global_id_x and b != .vertex_index) return;
        // A global index is this workgroup's base plus the lane: `workgroup_id * lanes + vid`. The lane
        // count is a power of two, so the multiply is a shift.
        const lanes = self.opts.profile.lanes();
        try self.put(glacier.csrr(scratch, glacier.Csr.workgroup_id));
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
            try self.emitMoves(set.moves, set.pred, set.succ);
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
            try self.emitMoves(set.moves, set.pred, set.succ);
        }
    }

    /// Realize one parallel-move set: register moves, spills, reloads, and the slot-to-slot step
    /// the Wimmer resolver already routes through the class scratch.
    /// The parallel copies the resolver asked for, in the register file each one belongs to.
    ///
    /// A PER-LANE value is moved with a vector instruction, and a per-lane BOOL with a mask one: a
    /// mask is bits and not elements, and `vmv.v.v` is only right for one while the whole mask fits
    /// inside the first element. A scalar `addi` on a vector register pair writes neither, which is
    /// what this used to do for every class: the moves are the one place a class is not implied by
    /// the opcode, and ignoring `m.class` was a wrong answer waiting for a per-lane block
    /// parameter to cross an edge.
    fn emitMoves(self: *Emitter, moves: []const wimmer.Move, pred: ?Block, succ: ?Block) Error!void {
        for (moves) |m| {
            if (m.class == vector_class) {
                const uniform_source = if (pred) |p| self.argIsUniform(p, succ.?, m.value) else false;
                if (uniform_source) return error.Unsupported;
                // A per-lane value that had to spill. The access takes its address from a register,
                // so the slot's address goes in the scalar scratch first. Nothing else is using the
                // scratch between the two words.
                switch (m.src) {
                    .reg => |r| switch (m.dst) {
                        .reg => {},
                        .slot => |n| {
                            try self.vectorSlotAddress(n);
                            try self.put(rv.vse32(vregOf(r), scratch));
                            continue;
                        },
                    },
                    .slot => |n| switch (m.dst) {
                        .reg => |d| {
                            try self.vectorSlotAddress(n);
                            try self.put(rv.vle32(vregOf(d), scratch));
                            continue;
                        },
                        // Slot to slot goes through the vector scratch, which no value occupies
                        // across an instruction boundary.
                        .slot => |d| {
                            try self.vectorSlotAddress(n);
                            try self.put(rv.vle32(vscratch, scratch));
                            try self.vectorSlotAddress(d);
                            try self.put(rv.vse32(vscratch, scratch));
                            continue;
                        },
                    },
                }
                const s = switch (m.src) {
                    .reg => |r| vregOf(r),
                    .slot => unreachable,
                };
                const d = switch (m.dst) {
                    .reg => |r| vregOf(r),
                    .slot => unreachable,
                };
                // A WARP-UNIFORM value feeding a per-lane parameter would cross the register
                // files: the source index names a SCALAR register and the destination a vector
                // one. `broadcastUniformArgs` removes every such edge before this runs, so
                // reaching here means it missed one, and the answer is a refusal rather than a
                // move that reads the wrong file (refused above). Note the allocator can also DROP
                // such a move before emission, which nothing here can catch; that is what the test
                // pins.
                if (s == d) continue;
                if (isBool(self.func, self.func.valueType(m.value))) {
                    try self.put(rv.vmand_mm(d, s, s));
                } else {
                    try self.put(rv.vmv_v_v(d, s));
                }
                continue;
            }
            // A constant this can re-emit never reaches its slot, so a move that would write the
            // slot is dropped and a move that would read it re-emits instead. Every scalar read
            // of a slot goes through here, which is what makes dropping the write sound.
            // A write to an elided slot goes nowhere and a read of one re-emits the constant
            // instead. Every value that reaches such a slot rematerialises, which is what makes
            // dropping the write sound. See `elidableSlots`.
            switch (m.src) {
                .reg => |s| switch (m.dst) {
                    .reg => |d| try self.move(regOf(d), regOf(s)),
                    .slot => |d| if (!self.elide[d]) {
                        try self.put(rv.sw(regOf(s), sp, try self.slotOffset(d)));
                    },
                },
                .slot => |s| switch (m.dst) {
                    .reg => |d| if (self.elide[s]) {
                        try self.rematerialise(regOf(d), m.value);
                    } else {
                        try self.put(rv.lw(regOf(d), sp, try self.slotOffset(s)));
                    },
                    .slot => |d| if (!self.elide[d]) {
                        if (self.elide[s]) {
                            try self.rematerialise(scratch, m.value);
                        } else {
                            try self.put(rv.lw(scratch, sp, try self.slotOffset(s)));
                        }
                        try self.put(rv.sw(scratch, sp, try self.slotOffset(d)));
                    },
                },
            }
        }
    }

    /// Whether the value a per-lane parameter receives on the edge `pred` to `succ` is warp
    /// UNIFORM, which makes the move a broadcast across the register files.
    ///
    /// The allocator builds a parameter move with the PARAMETER's class and the ARGUMENT's
    /// location, because it assumes the two share a register file. Here they need not: a loop's
    /// initial accumulator arrives as a constant and goes round as a lane value. `m.value` names
    /// the parameter, so the argument is the one at the same index on this edge.
    fn argIsUniform(self: *const Emitter, pred: Block, succ: Block, param: Value) bool {
        const args = self.edgeArgs(pred, succ) orelse return false;
        const params = self.func.blockParams(succ);
        if (args.len != params.len) return false;
        for (params, args) |p, a| {
            if (p != param) continue;
            return !self.isDivergent(a);
        }
        return false;
    }

    /// The arguments `pred` passes on its edge to `succ`, or null when it has two edges there and
    /// the one this move belongs to cannot be told apart.
    fn edgeArgs(self: *const Emitter, pred: Block, succ: Block) ?[]const Value {
        const insts = self.func.blockInsts(pred);
        if (endsInIf(self.func, pred)) {
            const cf = self.func.opcode(insts[insts.len - 1]).@"if";
            if (cf.then.target == succ and cf.@"else".target == succ) return null;
            if (cf.then.target == succ) return self.func.blockArgs(cf.then);
            if (cf.@"else".target == succ) return self.func.blockArgs(cf.@"else");
            return null;
        }
        return switch (self.func.terminator(pred) orelse return null) {
            .jump => |j| if (j.target == succ) self.func.blockArgs(j) else null,
            .ret => null,
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
        const off: i64 = -(@as(i64, self.shared_bytes) + (@as(i64, n) + 1) * 4);
        if (off < std.math.minInt(i12)) return error.Unsupported;
        return @intCast(off);
    }

    /// Fault unless the hart's scratch slice is as large as this kernel was built for.
    ///
    /// The spill slots and the workgroup window are placed against the PROFILE's region size,
    /// which this compiler mirrors from the SoC's configuration rather than reading from the
    /// hardware. A core whose slice is smaller would have the kernel writing into the hart
    /// underneath it, which corrupts a neighbour and reports nothing. Five words at launch buy a
    /// loud failure instead, and only a kernel that uses the region pays them.
    ///
    /// `scratch` is the only register this needs: the subtraction turns the comparison into a
    /// sign test, and the whole sequence runs before any value is live.
    fn checkScratch(self: *Emitter) Error!void {
        const need = self.scratchNeeded();
        if (need == 0) return;
        try self.put(glacier.csrr(scratch, glacier.Csr.scratchsize));
        try self.put(rv.addi(scratch, scratch, -@as(i12, @intCast(need))));
        // Two words ahead: past the code and past the fault itself.
        try self.put(rv.bge(scratch, .x0, 12));
        try self.put(rv.addi(scratch, .x0, glacier.fault_scratch_too_small));
        try self.put(glacier.ecall(scratch));
    }

    /// The bytes of the hart's scratch region this kernel uses: the workgroup window plus both
    /// classes of spill slot.
    fn scratchNeeded(self: *const Emitter) u32 {
        const vector_bytes = self.alloc.slot_count_per_class[1] * @as(u32, self.opts.profile.lanes()) * 4;
        return self.shared_bytes + self.alloc.slot_count_per_class[0] * 4 + vector_bytes;
    }

    /// The window offset of the workgroup slot `value` names, or null when it names none, which
    /// is this lowering disagreeing with the placement it asked for.
    fn sharedOffset(self: *const Emitter, value: Value) ?u32 {
        for (self.shared) |slot| {
            if (slot.value == value) return slot.offset;
        }
        return null;
    }

    /// The offset of PER-LANE spill slot `n` from `sp`, below the scalar slots. A whole vector
    /// register wide, because a spill of one stores every lane: a MASK only uses the first element,
    /// and storing the rest costs nothing but the space.
    fn vectorSlotOffset(self: *const Emitter, n: u32) Error!i12 {
        if (!self.spills) return error.Unsupported;
        const width: i64 = @as(i64, self.opts.profile.lanes()) * 4;
        const off: i64 = -(@as(i64, self.shared_bytes) + @as(i64, self.vector_slot_base) + (@as(i64, n) + 1) * width);
        if (off < std.math.minInt(i12)) return error.Unsupported;
        return @intCast(off);
    }

    /// Put the address of per-lane slot `n` in the scalar scratch, because a vector access takes
    /// its address from a register and has no displacement field of its own.
    fn vectorSlotAddress(self: *Emitter, n: u32) Error!void {
        try self.put(rv.addi(scratch, sp, try self.vectorSlotOffset(n)));
    }

    /// The allocator's own spills, reloads and moves at `pos`.
    fn actionsAt(self: *Emitter, pos: u32) Error!void {
        for (self.alloc.actions) |a| {
            if (a.at != pos) continue;
            // An intra-block action moves one value to itself, so there is no edge and no
            // parameter: source and destination are always the same register file.
            try self.emitMoves(&[_]wimmer.Move{.{ .src = a.src, .dst = a.dst, .class = a.class, .value = a.value }}, null, null);
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
                const mask = l.mask;
                if (mask) |m| try self.predicate(m, pos);
                // A contiguous run is one instruction and one bus run. See `Strides`.
                if (self.strides.at(inst)) |b| {
                    const start = try self.reg(b, pos);
                    if (mask != null) return self.put(rv.vle32_m(dst, start));
                    return self.put(rv.vle32(dst, start));
                }
                const addr = try self.vectorOperand(l.ptr, pos);
                if (mask != null) return self.put(rv.vluxei32_m(dst, .x0, addr));
                try self.put(rv.vluxei32(dst, .x0, addr));
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
                    try self.predicate(mask, pos);
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
                try self.predicate(mask, pos);
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
            // A mask built by comparing, not installed from a value, so nothing can match it.
            self.v0_mask = null;
            try self.put(rv.vmerge_vvm(dst, then_v, else_v));
            return;
        }
        try self.predicate(s.cond, pos);
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
        // A per-lane bool is a MASK, one bit per lane packed into the first element, and the only
        // immediate a frontend writes against one is `b ^ 1`, which is a NOT. `vmnand` against one
        // source twice IS that NOT, in one instruction and with no all-ones mask to materialise
        // first. An integer xor against an immediate would flip lane zero and leave the rest.
        if (isBool(self.func, ty)) {
            if (a.op != .bit_xor or a.imm != 1) return error.Unsupported;
            return self.put(rv.vmnand_mm(dst, lhs, lhs));
        }
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
        // A constant whose home is a slot is neither computed nor stored, because every read of
        // that slot re-emits it instead. See `rematTable`.
        if (home) |h| switch (h) {
            .slot => |n| if (self.elide[n]) return,
            .reg => {},
        };
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
                } else if (floatVectorBinOp(self.func, a.op, ty, false)) |encode| {
                    try self.emitScalarFloatBinOp(encode, a, dst, pos);
                } else {
                    const f = binOp(self.func, a.op, ty) orelse return error.Unsupported;
                    try self.put(f(dst, try self.reg(a.lhs, pos), try self.reg(a.rhs, pos)));
                }
            },
            .arith_imm => |a| try self.emitArithImm(a, dst, pos),
            .icmp => |c| try self.emitCompare(c, dst, pos),
            .select => |s| try self.emitSelect(s, dst, pos),
            // A masked load is never scalar: admission requires its result to be per-lane. The
            // refusal is here because a scalar load cannot predicate, so a change that routed one
            // here would read every lane.
            .load => |l| if (l.mask != null)
                return error.Unsupported
            else
                try self.put(rv.lw(dst, try self.reg(l.ptr, pos), 0)),
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
            // One widened to the CORE is a different question and does emit, because the warps of
            // a core are separate instruction streams that do drift apart. `gpu.attrs` carries
            // that width as an attribute rather than as a scope on the opcode, so the opcode here
            // reads `workgroup` either way and the attribute decides.
            //
            // The barrier needs NO fence beside it. A memory operation parks its own warp until
            // the bus completes it, so there is no store buffer to drain, and the barrier releases
            // only once every live warp has arrived: the two together are release and acquire.
            // THAT EXPIRES WITH AXI4, where a write is accepted on one channel long before its
            // response, so completion has to mean the response.
            .barrier => {
                // Widened to every warp of the core, which is what `warp_barrier` does: the warp
                // parks until its siblings arrive and NOTHING touches the bus, where the same
                // handshake written as a spin on a word costs one access an iteration on a bus
                // that is already the limit. Only LIVE warps hold it, so a warp that has exited
                // releases rather than hanging the rest.
                if (gpu.attrs.isClusterBarrier(self.func, inst)) {
                    self.cluster_barriers += 1;
                    try self.put(glacier.warp_barrier);
                }
            },
            // A workgroup object is an ADDRESS in the hart's scratch region. The window sits
            // directly below `mscratch`, so a slot at window offset `off` is `sp - bytes + off`,
            // and that is one instruction with no load behind it. The lanes of the warp all
            // compute the same address, which is what makes the object shared between them.
            .alloca => {
                const res = self.func.instResult(inst).?;
                const off = self.sharedOffset(res) orelse return error.Unsupported;
                const from_sp: i64 = @as(i64, off) - @as(i64, self.shared_bytes);
                if (from_sp < std.math.minInt(i12)) return error.Unsupported;
                try self.put(rv.addi(dst, sp, @intCast(from_sp)));
            },
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
        // A loop latch branches on whether ANY LANE is still active. The condition is a mask, so
        // it is reduced to a lane count in a scalar register first, and the warp goes round again
        // while that count is not zero. See `AnyLane`.
        if (self.any_lane.has(inst)) {
            try self.put(rv.vcpop_m(scratch, try self.vreg(cf.cond, pos)));
            try self.put(rv.beq(scratch, .x0, 8));
            try self.jumpAlways(cf.then.target);
            return self.jumpTo(cf.@"else".target, bi);
        }
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
            // A LIVE block with no terminator, which is a compiler bug and not a kernel one: a
            // block nothing reaches is skipped entirely and emits no word at all, so reaching this
            // means an analysis called a block live and left it no way out.
            //
            // It FAULTS rather than retires. Retiring is indistinguishable from finishing, so the
            // host would read a successful launch that did part of the work, and the only evidence
            // would be a wrong frame. See `encode.fault_compiler_bug`.
            //
            // The code is a VALUE the core reads out of a register, so it is materialised first.
            // `scratch` is never allocatable, so writing it here disturbs nothing, and nothing runs
            // after a fault in any case. That it is x31 while the code is 31 is a coincidence of
            // two unrelated numbers: the register is the one this lowering reserves, and the code
            // is the top of the range the toolchain reserved.
            try self.put(rv.addi(scratch, .x0, glacier.fault_compiler_bug));
            return self.put(glacier.ecall(scratch));
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

/// The same for the per-lane file: index n is v_n.
fn vregOf(index: u16) VReg {
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
