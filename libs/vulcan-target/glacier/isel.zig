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
//! One thing it deliberately does NOT do yet is fuse a compare into its branch. A fused branch
//! reads the COMPARE's operands at the branch, and the allocator only knows the branch reads the
//! boolean, so fusing needs `wimmer.RegDescription.fusedOperands` to tell it. Until that is
//! wired, a compare materializes 0 or 1 into a register and the branch tests it against x0. The
//! cost is one word per branch, which is correct and slower, not wrong and fast.

const std = @import("std");
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");
const wimmer = @import("../wimmer.zig");
const glacier = @import("encode.zig");

const rv = glacier.rv;
const Reg = glacier.Reg;
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

/// x1 to x30. A Glacier kernel has neither a call nor a stack, so the usual return-address and
/// stack-pointer reservations buy nothing and both registers carry data instead.
const allocatable: [30]u16 = blk: {
    var regs: [30]u16 = undefined;
    for (&regs, 1..) |*r, i| r.* = i;
    break :blk regs;
};

const scratch_set: [1]u16 = .{@backingInt(scratch)};

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
    try ir.critical_edge.splitCriticalEdges(allocator, &work);
    const reachable = try ir.reachable.neutralizeUnreachable(allocator, &work);
    allocator.free(reachable);

    try checkSupported(&work, opts);

    const classes = [_]wimmer.RegClass{.{
        .name = "gpr",
        .allocatable = &allocatable,
        .callee_saved = &.{},
        .slot_bytes = 4,
    }};
    var desc: wimmer.RegDescription = .{
        .classes = &classes,
        .classOf = classOf,
        .useKind = useKind,
        .entry_fixed = &.{},
        .call_sites = &.{},
        .scratch = &scratch_set,
        .ctx = undefined,
        .copySource = copySource,
        .coalesce_block_params = true,
    };
    var walloc = try wimmer.allocate(allocator, &work, &desc);
    defer walloc.deinit(allocator);

    // A spill needs a stack and a warp has no stack. Refuse here rather than emit a store to an
    // address the hardware never promised.
    for (walloc.slot_count_per_class) |n| if (n != 0) return error.Unsupported;

    var e: Emitter = .{
        .allocator = allocator,
        .func = &work,
        .opts = opts,
        .alloc = &walloc,
    };
    defer e.deinit();
    try e.run();
    return e.words.toOwnedSlice(allocator);
}

// ===========================================================================
// What this backend accepts.

fn classOf(_: *const anyopaque, _: *const Function, _: Value) u16 {
    return 0;
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
        else => null,
    };
}

/// Whether `ty` is a value a Glacier register holds. One 32-bit word, nothing narrower.
///
/// A narrow store is refused because the hardware has none: a byte store writes a whole word and
/// destroys its neighbours (see `encode.zig`). A narrow LOAD does exist, so what blocks a sub-word
/// value here is this lowering and not the core: every operation on one would have to truncate to
/// its width, and silently skipping that truncation is a wrong answer rather than a refusal.
fn isWord(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .bool, .ptr => true,
        .int => |i| i.bits == 32,
        else => false,
    };
}

fn isSigned(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.signedness == .signed,
        else => false,
    };
}

/// Reject every shape this lowering does not handle, before anything is emitted. A refusal here
/// is a clear error; a refusal halfway through emission is a truncated program.
fn checkSupported(func: *const Function, opts: Options) Error!void {
    for (0..func.valueCount()) |vi| {
        const v: Value = @fromBackingInt(@intCast(vi));
        if (!isWord(func, func.valueType(v))) return error.Unsupported;
    }
    for (func.blockParams(@fromBackingInt(0))) |p| {
        const b = gpu.attrs.builtinOf(func, p) orelse return error.Unsupported;
        _ = builtinSource(b) orelse return error.Unsupported;
    }
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        for (insts, 0..) |inst, i| switch (func.opcode(inst)) {
            // A constant must fit a 32-bit register. An unsigned type reaches 0xFFFFFFFF and a
            // signed one reaches the most negative value, so the window spans both.
            .iconst => |c| if (c < std.math.minInt(i32) or c > std.math.maxInt(u32)) return error.Unsupported,
            .arith => |a| _ = binOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported,
            .arith_imm => |a| _ = binOp(func, a.op, func.valueType(a.lhs)) orelse return error.Unsupported,
            .icmp, .select => {},
            .load => |l| if (!isWord(func, func.valueType(l.ptr))) return error.Unsupported,
            .store => |st| if (!isWord(func, func.valueType(st.value))) return error.Unsupported,
            .global_addr => |g| _ = addressOf(opts, func.symbolName(g.symbol)) orelse return error.Unsupported,
            .unary => |u| if (u.op != .reinterpret) return error.Unsupported,
            // A conditional transfers control, so nothing may follow it in its block, and its
            // block must have no terminator: a terminator there is an edge the allocator would
            // resolve and this emits nothing for.
            .@"if" => if (i + 1 != insts.len or func.terminator(block) != null) return error.Unsupported,
            else => return error.Unsupported,
        };
        if (func.terminator(block)) |term| switch (term) {
            .ret => |r| if (r.count != 0) return error.Unsupported,
            .jump => {},
        };
    }
}

/// Where a hardware builtin comes from on a Glacier core, or null when the core cannot answer.
///
/// A warp is the workgroup: it owns a program counter and a register file, and `mhartid` is
/// unique across the whole SoC, which is what a workgroup index is. The lanes inside a warp are
/// the threads, and a lane index is a VECTOR value (`vid.v`), so `thread_id_*` and `lane_id` are
/// not scalar builtins and are not answered here.
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

    /// The register holding `v` at position `pos`, or null when the allocator placed nothing
    /// there because nothing reads it.
    fn regAt(self: *Emitter, v: Value, pos: u32) Error!?Reg {
        const segs = self.alloc.segments.get(v) orelse return null;
        var found: ?wimmer.Location = null;
        for (segs) |s| {
            if (s.from > pos) break;
            found = s.loc;
        }
        const loc = found orelse return null;
        return switch (loc) {
            .reg => |r| @as(Reg, @fromBackingInt(@as(u5, @intCast(r)))),
            .slot => error.Unsupported,
        };
    }

    fn reg(self: *Emitter, v: Value, pos: u32) Error!Reg {
        return (try self.regAt(v, pos)) orelse error.Unsupported;
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
    fn prologue(self: *Emitter, block: Block) Error!void {
        for (self.func.blockParams(block)) |p| {
            const dst = (try self.regAt(p, 0)) orelse continue;
            const b = gpu.attrs.builtinOf(self.func, p) orelse return error.Unsupported;
            switch (builtinSource(b) orelse return error.Unsupported) {
                .csr => |c| try self.put(glacier.csrr(dst, c)),
                .lanes => try self.constant(dst, self.opts.profile.lanes()),
            }
        }
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

    fn emitMoves(self: *Emitter, moves: []const wimmer.Move) Error!void {
        for (moves) |m| {
            const src = switch (m.src) {
                .reg => |r| @as(Reg, @fromBackingInt(@as(u5, @intCast(r)))),
                .slot => return error.Unsupported,
            };
            const dst = switch (m.dst) {
                .reg => |r| @as(Reg, @fromBackingInt(@as(u5, @intCast(r)))),
                .slot => return error.Unsupported,
            };
            try self.move(dst, src);
        }
    }

    /// The allocator's own moves at `pos`. A split with no stack can only be a register-to-
    /// register move, so a store or a reload here means something spilled after all.
    fn actionsAt(self: *Emitter, pos: u32) Error!void {
        for (self.alloc.actions) |a| {
            if (a.at != pos) continue;
            if (a.kind != .move) return error.Unsupported;
            try self.emitMoves(&[_]wimmer.Move{.{ .src = a.src, .dst = a.dst, .class = a.class, .value = a.value }});
        }
    }

    fn emitInst(self: *Emitter, inst: Inst, pos: u32) Error!void {
        const result = self.func.instResult(inst);
        // A pure instruction nothing reads has no register to write to, so it is dropped.
        const dst: ?Reg = if (result) |r| try self.regAt(r, pos) else null;
        switch (self.func.opcode(inst)) {
            .iconst => |c| try self.constant(dst orelse return, lowWord(c)),
            .global_addr => |g| try self.constant(
                dst orelse return,
                @bitCast(addressOf(self.opts, self.func.symbolName(g.symbol)) orelse return error.Unsupported),
            ),
            .unary => |u| try self.move(dst orelse return, try self.reg(u.value, pos)),
            .arith => |a| {
                const d = dst orelse return;
                const f = binOp(self.func, a.op, self.func.valueType(a.lhs)) orelse return error.Unsupported;
                try self.put(f(d, try self.reg(a.lhs, pos), try self.reg(a.rhs, pos)));
            },
            .arith_imm => |a| try self.emitArithImm(a, dst orelse return, pos),
            .icmp => |c| try self.emitCompare(c, dst orelse return, pos),
            .select => |s| try self.emitSelect(s, dst orelse return, pos),
            .load => |l| try self.put(rv.lw(dst orelse return, try self.reg(l.ptr, pos), 0)),
            .store => |st| try self.put(rv.sw(try self.reg(st.value, pos), try self.reg(st.ptr, pos), 0)),
            else => return error.Unsupported,
        }
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

    /// A value-producing conditional. Glacier has no conditional move, so this is a move, a
    /// branch over one word, and a move. The taken value goes through the scratch register first
    /// so that a destination aliasing either source still reads the right bits.
    fn emitSelect(self: *Emitter, s: ir.function.Select, dst: Reg, pos: u32) Error!void {
        const cond = try self.reg(s.cond, pos);
        const then = try self.reg(s.then, pos);
        const other = try self.reg(s.@"else", pos);
        try self.move(scratch, then);
        try self.move(dst, other);
        try self.put(rv.beq(cond, .x0, 8));
        try self.move(dst, scratch);
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
