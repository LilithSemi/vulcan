//! Instruction selection for the Geyser Forth co-processor: Vulcan IR from the Forth frontend
//! becomes J1 words.
//!
//! Every IR value that is not a constant lives in a RAM slot. An operation loads its operands
//! from their slots and a result goes back to its slot. One exception keeps the code small: a
//! result with a single use, in the operation right after the one that made it, stays on the
//! data stack and has no slot.
//!
//! Calls follow the Forth convention. The caller pushes the arguments, deepest first, and the
//! callee leaves its results on the data stack. The frontend returns the top result and puts the
//! others in the context at `ctx_results + 4 * k`. This code reads that pattern and keeps all of
//! the results on the data stack.
//!
//! Slots are static, so a word that can call itself saves the slots that are live across the
//! call on a stack in RAM. Addresses from 0xF000 up are I/O and go to plain `@` and `!`.
//!
//! The image has an entry jump in word 0, then the data area, then the code. The data area starts
//! at byte address `data_base`, so every RAM address of the program is moved up by that.

const std = @import("std");
const ir = @import("vulcan-ir");
const encode = @import("encode.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Allocator = std.mem.Allocator;

pub const Error = Allocator.Error || encode.Error || error{
    /// The IR has a shape this selector does not handle.
    Unsupported,
    NoMain,
    /// The image, its slots and its save stack do not fit in the memory of the core.
    ImageTooLarge,
    /// A call chain or a word needs more stack than the core has.
    StackTooDeep,
};

/// What the compiler knows about the core.
pub const Options = struct {
    /// Cells in each of the two stacks. A stack of N cells holds N - 1 values, because the
    /// pointer of an empty stack still names a cell.
    stack_depth: u32 = encode.default_stack_depth,
    memory_words: u32 = 4096,
    diag: ?*Diagnostic = null,
};

/// The reason for an `ImageTooLarge` or `StackTooDeep` error, with the sizes in it.
pub const Diagnostic = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buf[0..self.len];
    }
};

fn fail(diag: ?*Diagnostic, err: Error, comptime fmt: []const u8, args: anytype) Error {
    if (diag) |d| {
        const text = std.fmt.bufPrint(&d.buf, fmt, args) catch &d.buf;
        d.len = text.len;
    }
    return err;
}

pub const Item = struct { name: []const u8, func: *const Function };

pub const Image = struct {
    words: []u16,
    /// The number of RAM slots the program uses. They sit right after the image.
    slots: u32,
};

/// The byte address of the first byte of the data area. Word 0 holds the entry jump.
pub const data_base: u16 = 2;

/// Byte offsets in the context the frontend generates code against.
const ctx_mem = 0;
const ctx_io_read = 8;
const ctx_io_write = 16;
const ctx_results = 24;
const max_extra = 15;
const io_base: u16 = 0xF000;

const Opd = union(enum) {
    imm: u16,
    val: Value,

    fn isVal(self: Opd, v: Value) bool {
        return self == .val and self.val == v;
    }

    fn isImm(self: Opd, n: u16) bool {
        return self == .imm and self.imm == n;
    }
};

const Def = union(enum) {
    unset,
    imm: u16,
    /// The context pointer plus a byte offset.
    ctx: u16,
    io_read,
    io_write,
    real,
    cmp: struct { op: ir.function.CmpOp, lhs: Opd, rhs: Opd },
    /// The value is the operand xor 0x8000.
    xor_sign: Opd,
    shift: Opd,

    fn isReal(self: Def) bool {
        return switch (self) {
            .real, .cmp, .xor_sign, .shift => true,
            .unset, .imm, .ctx, .io_read, .io_write => false,
        };
    }
};

const Kind = enum { alu, sub, invert, mul, select, fetch, store, call };

const Op = struct {
    kind: Kind,
    t_op: encode.TOp = .t,
    post_invert: bool = false,
    result: ?Value = null,
    plan: []const Opd,
    callee: u32 = 0,
    /// The values the frontend loads from the context after a call, by result index.
    extras: [max_extra]?Value = @splat(null),
    dead: bool = false,
    outs: u8 = 0,
    saves: []const Value = &.{},
    /// The call can reach its own caller again.
    cyclic: bool = false,

    fn pure(self: Op) bool {
        return switch (self.kind) {
            .alu, .sub, .invert, .mul, .select => true,
            .fetch, .store, .call => false,
        };
    }
};

const Copy = struct { param: Value, arg: Opd };
const Edge = struct { target: u32, copies: []Copy };
const Branch = struct { cond: Opd, then: Edge, @"else": Edge };
const Term = union(enum) {
    jump: Edge,
    branch: Branch,
    ret: []const Opd,
};

const BlockCode = struct {
    ir_block: u32,
    ops: std.ArrayList(Op) = .empty,
    term: Term = undefined,
};

const Routine = enum { mul, save, restore, entry, enter, leave, trap };

const Target = union(enum) {
    block: u32,
    func: u32,
    routine: Routine,
    slot: u32,
    stack_base,
    stack_limit,
};

const FixKind = enum { jmp, jz, call, lit };
const Fix = struct { at: usize, kind: FixKind, target: Target };

fn index(v: Value) usize {
    return @backingInt(v);
}

fn trunc16(v: i64) u16 {
    return @truncate(@as(u64, @bitCast(v)));
}

fn flag(b: bool) u16 {
    return if (b) 0xFFFF else 0;
}

/// State shared by every function of the image.
const Gen = struct {
    arena: Allocator,
    words: std.ArrayList(u16) = .empty,
    fixes: std.ArrayList(Fix) = .empty,
    func_addr: []usize = &.{},
    routine_addr: [7]?usize = @splat(null),
    slot_count: u32 = 0,
    sp_slot: ?u32 = null,
    mul_slots: ?[3]u32 = null,
    need_save: bool = false,
    need_enter: bool = false,
    rd_slot: ?u32 = null,
    /// The most values a recursive call saves, and the most recursive calls that can be active.
    max_saves: u32 = 0,
    max_frames: u32 = 0,
    save_cells: u32 = 0,
    need_mul: bool = false,

    fn here(self: *Gen) usize {
        return self.words.items.len;
    }

    fn word(self: *Gen, insn: encode.Insn) Error!void {
        try self.words.append(self.arena, try encode.encode(insn));
    }

    fn literal(self: *Gen, v: u16) Error!void {
        if (v <= encode.max_literal) {
            try self.word(.{ .lit = v });
        } else {
            try self.word(.{ .lit = ~v & encode.max_literal });
            try self.word(encode.invert);
        }
    }

    fn fixed(self: *Gen, kind: FixKind, target: Target) Error!void {
        try self.fixes.append(self.arena, .{ .at = self.here(), .kind = kind, .target = target });
        try self.words.append(self.arena, 0);
    }

    fn fixedRoutine(self: *Gen, kind: FixKind, routine: Routine) Error!void {
        try self.fixed(kind, .{ .routine = routine });
    }

    fn newSlot(self: *Gen) u32 {
        self.slot_count += 1;
        return self.slot_count - 1;
    }

    fn rdSlot(self: *Gen) u32 {
        if (self.rd_slot == null) self.rd_slot = self.newSlot();
        return self.rd_slot.?;
    }

    fn spSlot(self: *Gen) u32 {
        if (self.sp_slot == null) self.sp_slot = self.newSlot();
        return self.sp_slot.?;
    }
};

/// The state of one function while it is selected.
const F = struct {
    gen: *Gen,
    arena: Allocator,
    func: *const Function,
    item: u32,
    index_of: *const std.StringHashMapUnmanaged(u32),
    defs: []Def,
    alias: []Value,
    blocks: std.ArrayList(BlockCode) = .empty,
    discovered: []bool,
    stack: std.ArrayList(u32) = .empty,
    callees: std.ArrayList(u32) = .empty,
    uses: []u32 = &.{},
    transient: []bool = &.{},
    slot: []?u32 = &.{},
    outs: u8 = 0,
    block_addr: []usize = &.{},
    local: std.ArrayList(Fix) = .empty,
    pending: ?Value = null,

    // Pass 1: IR to operations.

    fn canon(self: *F, v: Value) Value {
        return self.alias[index(v)];
    }

    fn opd(self: *F, v: Value) Error!Opd {
        const c = self.canon(v);
        return switch (self.defs[index(c)]) {
            .imm => |x| .{ .imm = x },
            .real, .cmp, .xor_sign, .shift => .{ .val = c },
            .unset, .ctx, .io_read, .io_write => error.Unsupported,
        };
    }

    fn setImm(self: *F, res: Value, v: u16) void {
        self.defs[index(res)] = .{ .imm = v };
    }

    fn bind(self: *F, res: Value, o: Opd) void {
        switch (o) {
            .imm => |v| self.setImm(res, v),
            .val => |v| self.alias[index(res)] = v,
        }
    }

    fn dupePlan(self: *F, plan: []const Opd) Error![]const Opd {
        return self.arena.dupe(Opd, plan);
    }

    fn addOp(self: *F, bc: *BlockCode, op: Op) Error!void {
        var o = op;
        o.plan = try self.dupePlan(op.plan);
        if (o.result) |r| self.defs[index(r)] = .real;
        try bc.ops.append(self.arena, o);
    }

    fn alu(self: *F, bc: *BlockCode, res: Value, t_op: encode.TOp, plan: []const Opd, post_invert: bool) Error!void {
        try self.addOp(bc, .{ .kind = .alu, .t_op = t_op, .post_invert = post_invert, .result = res, .plan = plan });
    }

    fn immOf(self: *F, v: Value) ?u16 {
        const c = self.canon(v);
        return switch (self.defs[index(c)]) {
            .imm => |x| x,
            else => null,
        };
    }

    fn ctxOf(self: *F, v: Value) ?u16 {
        const c = self.canon(v);
        return switch (self.defs[index(c)]) {
            .ctx => |x| x,
            else => null,
        };
    }

    fn fold(op: ir.function.BinOp, a: u16, b: u16) ?u16 {
        return switch (op) {
            .add => a +% b,
            .sub => a -% b,
            .mul => a *% b,
            .bit_and => a & b,
            .bit_or => a | b,
            .bit_xor => a ^ b,
            .shl => if (b >= 16) 0 else a << @intCast(b),
            .shr => if (b >= 16) 0 else a >> @intCast(b),
            .div, .rem, .mulh => null,
        };
    }

    fn arith(self: *F, bc: *BlockCode, res: Value, op: ir.function.BinOp, a: Opd, b: Opd) Error!void {
        if (a == .imm and b == .imm) {
            self.setImm(res, fold(op, a.imm, b.imm) orelse return error.Unsupported);
            return;
        }
        switch (op) {
            .add => {
                if (b.isImm(0)) return self.bind(res, a);
                if (a.isImm(0)) return self.bind(res, b);
                try self.alu(bc, res, .add, &.{ a, b }, false);
            },
            .sub => {
                if (b.isImm(0)) return self.bind(res, a);
                if (b == .imm) return self.alu(bc, res, .add, &.{ a, .{ .imm = 0 -% b.imm } }, false);
                try self.addOp(bc, .{ .kind = .sub, .result = res, .plan = &.{ a, b } });
            },
            .mul => {
                if (a.isImm(0) or b.isImm(0)) return self.setImm(res, 0);
                if (b.isImm(1)) return self.bind(res, a);
                if (a.isImm(1)) return self.bind(res, b);
                self.gen.need_mul = true;
                try self.addOp(bc, .{ .kind = .mul, .result = res, .plan = &.{ a, b } });
            },
            .bit_and => {
                if (b.isImm(0xFFFF)) return self.bind(res, a);
                if (a.isImm(0xFFFF)) return self.bind(res, b);
                if (a.isImm(0) or b.isImm(0)) return self.setImm(res, 0);
                try self.alu(bc, res, .@"and", &.{ a, b }, false);
            },
            .bit_or => {
                if (b.isImm(0)) return self.bind(res, a);
                if (a.isImm(0)) return self.bind(res, b);
                if (a.isImm(0xFFFF) or b.isImm(0xFFFF)) return self.setImm(res, 0xFFFF);
                try self.alu(bc, res, .@"or", &.{ a, b }, false);
            },
            .bit_xor => {
                if (b.isImm(0)) return self.bind(res, a);
                if (a.isImm(0)) return self.bind(res, b);
                if (b.isImm(0xFFFF) or a.isImm(0xFFFF)) {
                    const x = if (b.isImm(0xFFFF)) a else b;
                    return self.addOp(bc, .{ .kind = .invert, .result = res, .plan = &.{x} });
                }
                try self.alu(bc, res, .xor, &.{ a, b }, false);
                if (b.isImm(0x8000)) self.defs[index(res)] = .{ .xor_sign = a };
                if (a.isImm(0x8000)) self.defs[index(res)] = .{ .xor_sign = b };
            },
            .shl, .shr => {
                if (b.isImm(0)) return self.bind(res, a);
                if (a.isImm(0)) return self.setImm(res, 0);
                try self.alu(bc, res, if (op == .shl) .lshift else .rshift, &.{ a, b }, false);
                self.defs[index(res)] = .{ .shift = b };
            },
            .div, .rem, .mulh => return error.Unsupported,
        }
    }

    fn signedForm(self: *F, o: Opd) ?Opd {
        return switch (o) {
            .imm => |v| .{ .imm = v ^ 0x8000 },
            .val => |v| switch (self.defs[index(v)]) {
                .xor_sign => |inner| inner,
                else => null,
            },
        };
    }

    fn isSignFlip(self: *F, o: Opd) bool {
        return o == .val and self.defs[index(o.val)] == .xor_sign;
    }

    fn compare(self: *F, bc: *BlockCode, res: Value, op: ir.function.CmpOp, a0: Opd, b0: Opd) Error!void {
        if (a0 == .imm and b0 == .imm) {
            const x = a0.imm;
            const y = b0.imm;
            return self.setImm(res, flag(switch (op) {
                .eq => x == y,
                .ne => x != y,
                .lt => x < y,
                .le => x <= y,
                .gt => x > y,
                .ge => x >= y,
            }));
        }
        var a = a0;
        var b = b0;
        var t_op: encode.TOp = .ult;
        const ordered = op != .eq and op != .ne;
        if (ordered and (self.isSignFlip(a) or self.isSignFlip(b))) {
            if (self.signedForm(a)) |sa| {
                if (self.signedForm(b)) |sb| {
                    a = sa;
                    b = sb;
                    t_op = .lt;
                }
            }
        }
        switch (op) {
            .eq => try self.alu(bc, res, .eq, &.{ a, b }, false),
            .ne => try self.alu(bc, res, .eq, &.{ a, b }, true),
            .lt => try self.alu(bc, res, t_op, &.{ a, b }, false),
            .gt => try self.alu(bc, res, t_op, &.{ b, a }, false),
            .ge => try self.alu(bc, res, t_op, &.{ a, b }, true),
            .le => try self.alu(bc, res, t_op, &.{ b, a }, true),
        }
        self.defs[index(res)] = .{ .cmp = .{ .op = op, .lhs = a0, .rhs = b0 } };
    }

    fn select(self: *F, bc: *BlockCode, res: Value, cond: Value, then: Value, other: Value) Error!void {
        const c = try self.opd(cond);
        const t = try self.opd(then);
        const e = try self.opd(other);
        if (c == .imm) return self.bind(res, if (c.imm != 0) t else e);
        if (std.meta.eql(t, e)) return self.bind(res, t);
        if (t.isImm(0xFFFF) and e.isImm(0)) return self.bind(res, c);
        if (t.isImm(0) and e.isImm(0xFFFF)) return self.addOp(bc, .{ .kind = .invert, .result = res, .plan = &.{c} });
        if (e.isImm(0) and t == .val and self.shiftGuard(c, t)) return self.bind(res, t);
        try self.addOp(bc, .{ .kind = .select, .result = res, .plan = &.{ t, e, c, e } });
    }

    /// A select between a shift and zero, guarded by "count below 16". The core already gives 0
    /// for a shift by 16 or more.
    fn shiftGuard(self: *F, c: Opd, t: Opd) bool {
        if (c != .val) return false;
        const guard = switch (self.defs[index(c.val)]) {
            .cmp => |x| x,
            else => return false,
        };
        const shifted = switch (self.defs[index(t.val)]) {
            .shift => |count| count,
            else => return false,
        };
        return guard.op == .lt and guard.rhs.isImm(16) and std.meta.eql(guard.lhs, shifted);
    }

    fn edge(self: *F, jump: ir.function.Jump) Error!Edge {
        const target = jump.target;
        const params = self.func.blockParams(target);
        const args = self.func.blockArgs(jump);
        if (params.len != args.len) return error.Unsupported;
        var copies: std.ArrayList(Copy) = .empty;
        for (params, args) |p, a| {
            const arg = try self.opd(a);
            if (arg.isVal(p)) continue;
            try copies.append(self.arena, .{ .param = p, .arg = arg });
        }
        const t: u32 = @backingInt(target);
        if (!self.discovered[t]) {
            self.discovered[t] = true;
            try self.stack.append(self.arena, t);
        }
        return .{ .target = t, .copies = copies.items };
    }

    fn lowerBlock(self: *F, bi: u32) Error!void {
        const block: Block = @fromBackingInt(bi);
        var bc: BlockCode = .{ .ir_block = bi };
        var last_call: ?usize = null;
        var stores: [max_extra + 1]?Opd = @splat(null);
        var store_count: usize = 0;
        var term: ?Term = null;

        const insts = self.func.blockInsts(block);
        for (insts, 0..) |inst, at| {
            const res = self.func.instResult(inst);
            switch (self.func.opcode(inst)) {
                .iconst => |v| self.setImm(res.?, trunc16(v)),
                .arith => |a| {
                    if (self.ctxOf(a.lhs)) |off| {
                        if (a.op != .add) return error.Unsupported;
                        const n = self.immOf(a.rhs) orelse return error.Unsupported;
                        self.defs[index(res.?)] = .{ .ctx = off +% n };
                    } else {
                        try self.arith(&bc, res.?, a.op, try self.opd(a.lhs), try self.opd(a.rhs));
                    }
                },
                .arith_imm => |a| {
                    if (self.ctxOf(a.lhs)) |off| {
                        if (a.op != .add) return error.Unsupported;
                        self.defs[index(res.?)] = .{ .ctx = off +% trunc16(a.imm) };
                    } else {
                        try self.arith(&bc, res.?, a.op, try self.opd(a.lhs), .{ .imm = trunc16(a.imm) });
                    }
                },
                .icmp => |c| try self.compare(&bc, res.?, c.op, try self.opd(c.lhs), try self.opd(c.rhs)),
                .select => |s| try self.select(&bc, res.?, s.cond, s.then, s.@"else"),
                .load => |l| {
                    if (self.ctxOf(l.ptr)) |off| {
                        switch (off) {
                            ctx_mem => self.setImm(res.?, data_base),
                            ctx_io_read => self.defs[index(res.?)] = .io_read,
                            ctx_io_write => self.defs[index(res.?)] = .io_write,
                            else => {
                                if (off < ctx_results or (off - ctx_results) % 4 != 0) return error.Unsupported;
                                const k = (off - ctx_results) / 4;
                                if (k >= max_extra) return error.Unsupported;
                                const call_at = last_call orelse return error.Unsupported;
                                bc.ops.items[call_at].extras[k] = res.?;
                                self.defs[index(res.?)] = .real;
                            },
                        }
                    } else {
                        try self.addOp(&bc, .{ .kind = .fetch, .result = res.?, .plan = &.{try self.opd(l.ptr)} });
                    }
                },
                .store => |s| {
                    if (self.ctxOf(s.ptr)) |off| {
                        if (off < ctx_results or (off - ctx_results) % 4 != 0) return error.Unsupported;
                        const k = (off - ctx_results) / 4;
                        if (k >= max_extra) return error.Unsupported;
                        stores[k] = try self.opd(s.value);
                        store_count = @max(store_count, k + 1);
                    } else {
                        try self.addOp(&bc, .{ .kind = .store, .plan = &.{ try self.opd(s.value), try self.opd(s.ptr) } });
                    }
                },
                .call => |c| {
                    const name = self.func.symbolName(c.symbol);
                    const callee = self.index_of.get(name) orelse return error.Unsupported;
                    const args = self.func.valueList(c.args);
                    if (args.len == 0) return error.Unsupported;
                    const plan = try self.arena.alloc(Opd, args.len - 1);
                    for (args[1..], plan) |a, *p| p.* = try self.opd(a);
                    last_call = bc.ops.items.len;
                    try self.addOp(&bc, .{ .kind = .call, .result = res, .plan = plan, .callee = callee });
                    try self.callees.append(self.arena, callee);
                },
                .call_indirect => |c| {
                    const target = self.canon(c.target);
                    const args = self.func.valueList(c.args);
                    switch (self.defs[index(target)]) {
                        .io_read => {
                            if (args.len != 2 or res == null) return error.Unsupported;
                            try self.addOp(&bc, .{ .kind = .fetch, .result = res, .plan = &.{try self.opd(args[1])} });
                        },
                        .io_write => {
                            if (args.len != 3) return error.Unsupported;
                            try self.addOp(&bc, .{ .kind = .store, .plan = &.{ try self.opd(args[2]), try self.opd(args[1]) } });
                        },
                        else => return error.Unsupported,
                    }
                },
                .@"if" => |c| {
                    if (at + 1 != insts.len) return error.Unsupported;
                    term = try self.branch(c);
                },
                else => return error.Unsupported,
            }
        }

        if (term == null) {
            const t = self.func.terminator(block) orelse return error.Unsupported;
            term = switch (t) {
                .jump => |j| .{ .jump = try self.edge(j) },
                .ret => |r| blk: {
                    const values = r.slice();
                    if (values.len > 1) return error.Unsupported;
                    const plan = try self.arena.alloc(Opd, store_count + values.len);
                    for (plan[0..store_count], stores[0..store_count]) |*p, s| p.* = s orelse return error.Unsupported;
                    if (values.len == 1) plan[store_count] = try self.opd(values[0]);
                    break :blk .{ .ret = plan };
                },
            };
        }
        bc.term = term.?;
        try self.blocks.append(self.arena, bc);
    }

    fn branch(self: *F, c: ir.function.If) Error!Term {
        var then = c.then;
        var other = c.@"else";
        const cond = self.canon(c.cond);
        var test_opd: Opd = undefined;
        switch (self.defs[index(cond)]) {
            .imm => |v| return .{ .jump = try self.edge(if (v != 0) then else other) },
            .cmp => |cmp| {
                const zero_rhs = cmp.rhs.isImm(0) and cmp.lhs == .val;
                const zero_lhs = cmp.lhs.isImm(0) and cmp.rhs == .val;
                if ((cmp.op == .ne or cmp.op == .eq) and (zero_rhs or zero_lhs)) {
                    test_opd = if (zero_rhs) cmp.lhs else cmp.rhs;
                    if (cmp.op == .eq) std.mem.swap(ir.function.Jump, &then, &other);
                } else {
                    test_opd = .{ .val = cond };
                }
            },
            else => test_opd = try self.opd(cond),
        }
        return .{ .branch = .{ .cond = test_opd, .then = try self.edge(then), .@"else" = try self.edge(other) } };
    }

    fn lowerAll(self: *F) Error!void {
        const nblocks = self.func.blockCount();
        for (0..nblocks) |bi| {
            const b: Block = @fromBackingInt(@intCast(bi));
            for (self.func.blockParams(b)) |p| self.defs[index(p)] = .real;
        }
        const entry: Block = @fromBackingInt(0);
        const params = self.func.blockParams(entry);
        if (params.len == 0) return error.Unsupported;
        self.defs[index(params[0])] = .{ .ctx = 0 };
        self.discovered[0] = true;
        try self.stack.append(self.arena, 0);
        while (self.stack.pop()) |bi| try self.lowerBlock(bi);
    }

    // Pass 2: dead code, stack-resident values, slots.

    fn countUses(self: *F) void {
        @memset(self.uses, 0);
        for (self.blocks.items) |*bc| {
            for (bc.ops.items) |op| {
                if (op.dead) continue;
                self.countPlan(op.plan);
            }
            switch (bc.term) {
                .jump => |e| self.countCopies(e.copies),
                .branch => |b| {
                    self.countPlan(&.{b.cond});
                    self.countCopies(b.then.copies);
                    self.countCopies(b.@"else".copies);
                },
                .ret => |plan| self.countPlan(plan),
            }
        }
    }

    fn countPlan(self: *F, plan: []const Opd) void {
        for (plan) |o| if (o == .val) {
            self.uses[index(o.val)] += 1;
        };
    }

    fn countCopies(self: *F, copies: []const Copy) void {
        for (copies) |c| self.countPlan(&.{c.arg});
    }

    fn compactCopies(self: *F, e: *Edge) bool {
        var n: usize = 0;
        for (e.copies) |c| {
            if (self.uses[index(c.param)] == 0) continue;
            e.copies[n] = c;
            n += 1;
        }
        const changed = n != e.copies.len;
        e.copies = e.copies[0..n];
        return changed;
    }

    fn removeDead(self: *F) void {
        var changed = true;
        while (changed) {
            changed = false;
            self.countUses();
            for (self.blocks.items) |*bc| {
                for (bc.ops.items) |*op| {
                    const r = op.result orelse continue;
                    if (!op.dead and op.pure() and self.uses[index(r)] == 0) {
                        op.dead = true;
                        changed = true;
                    }
                }
                switch (bc.term) {
                    .jump => |*e| changed = self.compactCopies(e) or changed,
                    .branch => |*b| {
                        changed = self.compactCopies(&b.then) or changed;
                        changed = self.compactCopies(&b.@"else") or changed;
                    },
                    .ret => {},
                }
            }
        }
    }

    fn eligible(plan: []const Opd, p: Value) bool {
        if (plan.len >= 1 and plan[0].isVal(p)) return true;
        return plan.len == 2 and plan[1].isVal(p);
    }

    fn termPlan(self: *F, term: *const Term) Error![]const Opd {
        switch (term.*) {
            .jump => |e| return self.copyPlan(e.copies),
            .branch => |b| return self.dupePlan(&.{b.cond}),
            .ret => |plan| return plan,
        }
    }

    fn copyPlan(self: *F, copies: []const Copy) Error![]const Opd {
        const plan = try self.arena.alloc(Opd, copies.len);
        for (plan, copies) |*p, c| p.* = c.arg;
        return plan;
    }

    fn findTransients(self: *F) Error!void {
        for (self.blocks.items) |*bc| {
            for (bc.ops.items, 0..) |op, i| {
                const r = op.result orelse continue;
                if (op.dead or self.uses[index(r)] != 1) continue;
                if (op.kind == .call and op.outs > 1) continue;
                var next: ?[]const Opd = null;
                for (bc.ops.items[i + 1 ..]) |later| {
                    if (later.dead) continue;
                    next = later.plan;
                    break;
                }
                const plan = next orelse try self.termPlan(&bc.term);
                if (eligible(plan, r)) self.transient[index(r)] = true;
            }
        }
    }

    fn assignSlots(self: *F) void {
        for (self.slot, 0..) |*s, v| {
            const value: Value = @fromBackingInt(@intCast(v));
            if (!self.defs[v].isReal() or self.alias[v] != value) continue;
            if (self.uses[v] == 0 or self.transient[v]) continue;
            s.* = self.gen.newSlot();
        }
    }

    /// The most values this word can have on the data stack at once.
    fn dataPeak(self: *F) u32 {
        var peak: u32 = @intCast(self.func.blockParams(@fromBackingInt(0)).len);
        for (self.blocks.items) |*bc| {
            for (bc.ops.items) |op| {
                if (op.dead) continue;
                const plan: u32 = @intCast(op.plan.len);
                peak = @max(peak, plan + 3);
                if (op.kind == .call) peak = @max(peak, op.outs + 2);
            }
            const plan: u32 = switch (bc.term) {
                .jump => |e| @intCast(e.copies.len),
                .branch => |b| @intCast(@max(b.then.copies.len, b.@"else".copies.len, 1)),
                .ret => |p| @intCast(p.len),
            };
            peak = @max(peak, plan + 2);
        }
        return peak;
    }

    fn usesHelper(self: *F) bool {
        for (self.blocks.items) |*bc| for (bc.ops.items) |op| {
            if (!op.dead and (op.kind == .mul or op.cyclic)) return true;
        };
        return false;
    }

    // Liveness, for the slots a recursive call has to save.

    fn successors(bc: *const BlockCode, out: *[2]u32) []const u32 {
        switch (bc.term) {
            .jump => |e| {
                out[0] = e.target;
                return out[0..1];
            },
            .branch => |b| {
                out[0] = b.then.target;
                out[1] = b.@"else".target;
                return out[0..2];
            },
            .ret => return out[0..0],
        }
    }

    fn markPlan(self: *F, live: []bool, plan: []const Opd) void {
        for (plan) |o| if (o == .val and self.slot[index(o.val)] != null) {
            live[index(o.val)] = true;
        };
    }

    /// Walk block `li` backwards from `live` (its live-out set). When `record` is set, every call
    /// that needs it gets the slots live after it.
    fn walkBlock(self: *F, li: usize, live: []bool, record: ?*const Reach) Error!void {
        const bc = &self.blocks.items[li];
        switch (bc.term) {
            .jump => |e| for (e.copies) |c| self.markPlan(live, &.{c.arg}),
            .branch => |b| {
                self.markPlan(live, &.{b.cond});
                for (b.then.copies) |c| self.markPlan(live, &.{c.arg});
                for (b.@"else".copies) |c| self.markPlan(live, &.{c.arg});
            },
            .ret => |plan| self.markPlan(live, plan),
        }
        var i = bc.ops.items.len;
        while (i > 0) {
            i -= 1;
            const op = &bc.ops.items[i];
            if (op.dead) continue;
            if (op.result) |r| live[index(r)] = false;
            for (op.extras) |e| if (e) |v| {
                live[index(v)] = false;
            };
            if (record) |reach| {
                if (op.kind == .call and reach.needs(self.item, op.callee)) {
                    var saves: std.ArrayList(Value) = .empty;
                    for (live, 0..) |is_live, v| {
                        if (is_live) try saves.append(self.arena, @fromBackingInt(@intCast(v)));
                    }
                    op.saves = saves.items;
                    op.cyclic = true;
                    self.gen.need_enter = true;
                    self.gen.max_saves = @max(self.gen.max_saves, @as(u32, @intCast(saves.items.len)));
                    if (saves.items.len != 0) self.gen.need_save = true;
                }
            }
            self.markPlan(live, op.plan);
        }
        for (self.func.blockParams(@fromBackingInt(bc.ir_block))) |p| live[index(p)] = false;
    }

    fn computeSaves(self: *F, reach: *const Reach) Error!void {
        const nb = self.blocks.items.len;
        const nv = self.defs.len;
        const live_in = try self.arena.alloc(bool, nb * nv);
        @memset(live_in, false);
        const live = try self.arena.alloc(bool, nv);
        const layout = try self.arena.alloc(u32, self.discovered.len);
        for (self.blocks.items, 0..) |bc, li| layout[bc.ir_block] = @intCast(li);

        var changed = true;
        while (changed) {
            changed = false;
            var li = nb;
            while (li > 0) {
                li -= 1;
                @memset(live, false);
                var buf: [2]u32 = undefined;
                for (successors(&self.blocks.items[li], &buf)) |s| {
                    const sin = live_in[layout[s] * nv ..][0..nv];
                    for (live, sin) |*l, x| l.* = l.* or x;
                }
                try self.walkBlock(li, live, null);
                const mine = live_in[li * nv ..][0..nv];
                for (mine, live) |*m, l| {
                    if (l and !m.*) {
                        m.* = true;
                        changed = true;
                    }
                }
            }
        }
        for (0..nb) |li| {
            @memset(live, false);
            var buf: [2]u32 = undefined;
            for (successors(&self.blocks.items[li], &buf)) |s| {
                const sin = live_in[layout[s] * nv ..][0..nv];
                for (live, sin) |*l, x| l.* = l.* or x;
            }
            try self.walkBlock(li, live, reach);
        }
    }

    // Pass 3: words.

    fn word(self: *F, insn: encode.Insn) Error!void {
        try self.gen.word(insn);
    }

    fn slotLit(self: *F, v: Value) Error!void {
        const s = self.slot[index(v)] orelse unreachable;
        try self.gen.fixed(.lit, .{ .slot = s });
    }

    fn pushOpd(self: *F, o: Opd) Error!void {
        switch (o) {
            .imm => |v| try self.gen.literal(v),
            .val => |v| {
                try self.slotLit(v);
                try self.word(encode.fetch);
            },
        }
    }

    /// Store the top of stack in the slot of `v` and drop it.
    fn spill(self: *F, v: Value) Error!void {
        try self.slotLit(v);
        try self.word(encode.store);
        try self.word(encode.drop);
    }

    /// Settle the pending value against `plan`. Gives the number of leading operands that are
    /// already on the stack.
    fn begin(self: *F, plan: []const Opd) Error!usize {
        const p = self.pending orelse return 0;
        self.pending = null;
        if (plan.len >= 1 and plan[0].isVal(p)) return 1;
        if (plan.len == 2 and plan[1].isVal(p)) {
            try self.pushOpd(plan[0]);
            try self.word(encode.swap);
            return 2;
        }
        std.debug.assert(self.slot[index(p)] != null);
        try self.word(encode.drop);
        return 0;
    }

    fn pushPlan(self: *F, plan: []const Opd) Error!void {
        const n = try self.begin(plan);
        for (plan[n..]) |o| try self.pushOpd(o);
    }

    fn finishResult(self: *F, result: ?Value) Error!void {
        const r = result orelse return;
        if (self.uses[index(r)] == 0) return self.word(encode.drop);
        if (self.transient[index(r)]) {
            self.pending = r;
            return;
        }
        try self.slotLit(r);
        try self.word(encode.store);
        self.pending = r;
    }

    fn spillOrDrop(self: *F, v: ?Value) Error!void {
        if (v) |x| {
            if (self.uses[index(x)] != 0) return self.spill(x);
        }
        try self.word(encode.drop);
    }

    fn aluWord(t_op: encode.TOp) encode.Insn {
        return switch (t_op) {
            .add => encode.plus,
            .@"and" => encode.@"and",
            .@"or" => encode.@"or",
            .xor => encode.xor,
            .eq => encode.eq,
            .lt => encode.lt,
            .ult => encode.ult,
            .lshift => encode.lshift,
            .rshift => encode.rshift,
            .t, .n, .invert, .decrement, .r, .fetch, .depth => unreachable,
        };
    }

    fn emitOp(self: *F, op: *const Op) Error!void {
        const g = self.gen;
        switch (op.kind) {
            .alu => {
                try self.pushPlan(op.plan);
                try self.word(aluWord(op.t_op));
                if (op.post_invert) try self.word(encode.invert);
            },
            .sub => {
                try self.pushPlan(op.plan);
                try self.word(encode.invert);
                try g.literal(1);
                try self.word(encode.plus);
                try self.word(encode.plus);
            },
            .invert => {
                try self.pushPlan(op.plan);
                try self.word(encode.invert);
            },
            .mul => {
                try self.pushPlan(op.plan);
                try g.fixed(.call, .{ .routine = .mul });
            },
            .select => {
                const n = try self.begin(op.plan);
                std.debug.assert(n <= 1);
                for (op.plan[n..2]) |o| try self.pushOpd(o);
                try self.word(encode.xor);
                try self.pushOpd(op.plan[2]);
                try self.word(encode.@"and");
                try self.pushOpd(op.plan[3]);
                try self.word(encode.xor);
            },
            .fetch => {
                try self.pushPlan(op.plan);
                try self.word(encode.fetch);
            },
            .store => {
                try self.pushPlan(op.plan);
                try self.word(encode.store);
                try self.word(encode.drop);
            },
            .call => return self.emitCall(op),
        }
        try self.finishResult(op.result);
    }

    fn emitCall(self: *F, op: *const Op) Error!void {
        const g = self.gen;
        if (op.cyclic) try g.fixed(.call, .{ .routine = .enter });
        for (op.saves) |v| {
            try self.pushOpd(.{ .val = v });
            try g.fixed(.call, .{ .routine = .save });
        }
        try self.pushPlan(op.plan);
        try g.fixed(.call, .{ .func = op.callee });
        var i = op.saves.len;
        while (i > 0) {
            i -= 1;
            try g.fixed(.call, .{ .routine = .restore });
            try self.spill(op.saves[i]);
        }
        if (op.cyclic) try g.fixed(.call, .{ .routine = .leave });
        const result = op.result orelse {
            for (0..op.outs) |_| try self.word(encode.drop);
            return;
        };
        switch (op.outs) {
            0 => return error.Unsupported,
            1 => try self.finishResult(result),
            else => {
                try self.spillOrDrop(result);
                var k: usize = op.outs - 1;
                while (k > 0) {
                    k -= 1;
                    try self.spillOrDrop(op.extras[k]);
                }
            },
        }
    }

    fn localFix(self: *F, kind: FixKind, target: u32) Error!void {
        try self.local.append(self.arena, .{ .at = self.gen.here(), .kind = kind, .target = .{ .block = target } });
        try self.gen.words.append(self.gen.arena, 0);
    }

    fn jumpTo(self: *F, target: u32, next: ?u32) Error!void {
        if (next != null and next.? == target) return;
        try self.localFix(.jmp, target);
    }

    fn emitCopies(self: *F, copies: []const Copy) Error!void {
        try self.pushPlan(try self.copyPlan(copies));
        var i = copies.len;
        while (i > 0) {
            i -= 1;
            try self.spill(copies[i].param);
        }
    }

    fn emitTerm(self: *F, term: *const Term, next: ?u32) Error!void {
        const g = self.gen;
        switch (term.*) {
            .jump => |e| {
                try self.emitCopies(e.copies);
                try self.jumpTo(e.target, next);
            },
            .branch => |b| {
                try self.pushPlan(&.{b.cond});
                const stub = b.@"else".copies.len != 0;
                const jz_at = g.here();
                try g.words.append(g.arena, 0);
                if (!stub) try self.local.append(self.arena, .{ .at = jz_at, .kind = .jz, .target = .{ .block = b.@"else".target } });
                try self.emitCopies(b.then.copies);
                try self.jumpTo(b.then.target, if (stub) null else next);
                if (stub) {
                    g.words.items[jz_at] = try encode.encode(.{ .jz = @intCast(g.here()) });
                    try self.emitCopies(b.@"else".copies);
                    try self.jumpTo(b.@"else".target, next);
                }
            },
            .ret => |plan| {
                try self.pushPlan(plan);
                try self.word(encode.exit);
            },
        }
    }

    fn emit(self: *F) Error!void {
        const g = self.gen;
        g.func_addr[self.item] = g.here();
        const entry = self.func.blockParams(@fromBackingInt(0));
        var i = entry.len;
        while (i > 1) {
            i -= 1;
            try self.spillOrDrop(entry[i]);
        }
        for (self.blocks.items, 0..) |*bc, li| {
            self.block_addr[bc.ir_block] = g.here();
            self.pending = null;
            for (bc.ops.items) |*op| {
                if (!op.dead) try self.emitOp(op);
            }
            const next: ?u32 = if (li + 1 < self.blocks.items.len) self.blocks.items[li + 1].ir_block else null;
            try self.emitTerm(&bc.term, next);
        }
        for (self.local.items) |fix| {
            const target = self.block_addr[fix.target.block];
            g.words.items[fix.at] = try encode.encode(switch (fix.kind) {
                .jmp => .{ .jmp = @intCast(target) },
                .jz => .{ .jz = @intCast(target) },
                .call, .lit => unreachable,
            });
        }
    }
};

/// Which words can call which, for the words that are in the program.
const Reach = struct {
    n: usize,
    bits: []bool,

    fn needs(self: *const Reach, caller: u32, callee: u32) bool {
        return self.bits[callee * self.n + caller];
    }
};

fn emitRoutines(g: *Gen) Error!void {
    if (g.need_mul) try emitMul(g);
    if (g.need_enter) try emitFrames(g);
    if (g.need_save) {
        g.routine_addr[@backingInt(Routine.save)] = g.here();
        // ( v -- ) trap when the save stack is full, store at the stack pointer, then move the
        // pointer up one cell.
        const sp = g.spSlot();
        try g.fixed(.lit, .{ .slot = sp });
        try g.word(encode.fetch);
        try g.word(encode.dup);
        try g.fixed(.lit, .stack_limit);
        try g.word(encode.ult);
        try g.fixedRoutine(.jz, .trap);
        try g.word(encode.store);
        try g.word(encode.drop);
        try g.fixed(.lit, .{ .slot = sp });
        try g.word(encode.fetch);
        try g.literal(2);
        try g.word(encode.plus);
        try storeSlot(g, sp);
        try g.word(encode.exit);

        g.routine_addr[@backingInt(Routine.restore)] = g.here();
        // ( -- v ) move the pointer down one cell, then read the cell.
        try g.fixed(.lit, .{ .slot = sp });
        try g.word(encode.fetch);
        try g.literal(0xFFFE);
        try g.word(encode.plus);
        try g.word(encode.dup);
        try storeSlot(g, sp);
        try g.word(encode.fetch);
        try g.word(encode.exit);
    }
}

fn storeSlot(g: *Gen, slot: u32) Error!void {
    try g.fixed(.lit, .{ .slot = slot });
    try g.word(encode.store);
    try g.word(encode.drop);
}

fn loadSlot(g: *Gen, slot: u32) Error!void {
    try g.fixed(.lit, .{ .slot = slot });
    try g.word(encode.fetch);
}

/// ( a b -- a*b ) by shift and add.
fn emitMul(g: *Gen) Error!void {
    const a = g.newSlot();
    const b = g.newSlot();
    const p = g.newSlot();
    g.routine_addr[@backingInt(Routine.mul)] = g.here();
    try storeSlot(g, b);
    try storeSlot(g, a);
    try g.literal(0);
    try storeSlot(g, p);

    const loop = g.here();
    try loadSlot(g, b);
    const done_at = g.here();
    try g.words.append(g.arena, 0);
    try loadSlot(g, b);
    try g.literal(1);
    try g.word(encode.@"and");
    const skip_at = g.here();
    try g.words.append(g.arena, 0);
    try loadSlot(g, a);
    try loadSlot(g, p);
    try g.word(encode.plus);
    try storeSlot(g, p);
    g.words.items[skip_at] = try encode.encode(.{ .jz = @intCast(g.here()) });
    try loadSlot(g, a);
    try g.literal(1);
    try g.word(encode.lshift);
    try storeSlot(g, a);
    try loadSlot(g, b);
    try g.literal(1);
    try g.word(encode.rshift);
    try storeSlot(g, b);
    try g.word(.{ .jmp = @intCast(loop) });
    g.words.items[done_at] = try encode.encode(.{ .jz = @intCast(g.here()) });
    try loadSlot(g, p);
    try g.word(encode.exit);
}

fn patch(g: *Gen) Error!void {
    const slot_base: usize = 2 * g.here();
    for (g.fixes.items) |fix| {
        const insn: encode.Insn = switch (fix.kind) {
            .lit => .{ .lit = switch (fix.target) {
                .slot => |s| std.math.cast(u16, slot_base + 2 * s) orelse return error.ImageTooLarge,
                .stack_base => std.math.cast(u16, slot_base + 2 * g.slot_count) orelse return error.ImageTooLarge,
                .stack_limit => std.math.cast(u16, slot_base + 2 * (g.slot_count + g.save_cells)) orelse return error.ImageTooLarge,
                else => unreachable,
            } },
            .jmp, .jz, .call => blk: {
                const address: usize = switch (fix.target) {
                    .func => |f| g.func_addr[f],
                    .routine => |r| g.routine_addr[@backingInt(r)].?,
                    else => unreachable,
                };
                const to = std.math.cast(u16, address) orelse return error.ImageTooLarge;
                break :blk switch (fix.kind) {
                    .jmp => .{ .jmp = to },
                    .jz => .{ .jz = to },
                    else => .{ .call = to },
                };
            },
        };
        g.words.items[fix.at] = try encode.encode(insn);
    }
}

/// Select instructions for every word the program can reach from `items[main]`. `data` is the
/// initial data area. Calls name their callee with the item name.
pub fn compile(allocator: Allocator, items: []const Item, data: []const u8, main: usize, options: Options) Error!Image {
    var scratch = std.heap.ArenaAllocator.init(allocator);
    defer scratch.deinit();
    const arena = scratch.allocator();
    if (main >= items.len) return error.NoMain;

    var index_of: std.StringHashMapUnmanaged(u32) = .empty;
    for (items, 0..) |item, i| try index_of.put(arena, item.name, @intCast(i));

    var g: Gen = .{ .arena = arena };
    g.func_addr = try arena.alloc(usize, items.len);
    const fs = try arena.alloc(?*F, items.len);
    @memset(fs, null);
    var order: std.ArrayList(u32) = .empty;
    var work: std.ArrayList(u32) = .empty;
    try work.append(arena, @intCast(main));
    while (work.pop()) |i| {
        if (fs[i] != null) continue;
        const func = items[i].func;
        const nv = func.valueCount();
        const f = try arena.create(F);
        f.* = .{
            .gen = &g,
            .arena = arena,
            .func = func,
            .item = i,
            .defs = try arena.alloc(Def, nv),
            .alias = try arena.alloc(Value, nv),
            .discovered = try arena.alloc(bool, func.blockCount()),
            .uses = try arena.alloc(u32, nv),
            .transient = try arena.alloc(bool, nv),
            .slot = try arena.alloc(?u32, nv),
            .block_addr = try arena.alloc(usize, func.blockCount()),
            .index_of = &index_of,
        };
        @memset(f.defs, .unset);
        for (f.alias, 0..) |*a, v| a.* = @fromBackingInt(@intCast(v));
        @memset(f.discovered, false);
        @memset(f.transient, false);
        @memset(f.slot, null);
        try f.lowerAll();
        fs[i] = f;
        try order.append(arena, i);
        for (f.callees.items) |c| try work.append(arena, c);
    }

    for (order.items) |i| {
        const f = fs[i].?;
        var outs: ?usize = null;
        for (f.blocks.items) |bc| {
            if (bc.term != .ret) continue;
            if (outs != null and outs.? != bc.term.ret.len) return error.Unsupported;
            outs = bc.term.ret.len;
        }
        f.outs = @intCast(outs orelse 0);
    }
    for (order.items) |i| {
        const f = fs[i].?;
        for (f.blocks.items) |*bc| for (bc.ops.items) |*op| {
            if (op.kind != .call) continue;
            const callee = fs[op.callee].?;
            if (callee.func.blockParams(@fromBackingInt(0)).len != op.plan.len + 1) return error.Unsupported;
            op.outs = callee.outs;
        };
    }

    const n = items.len;
    const reach: Reach = .{ .n = n, .bits = try arena.alloc(bool, n * n) };
    @memset(reach.bits, false);
    for (order.items) |i| for (fs[i].?.callees.items) |c| {
        reach.bits[i * n + c] = true;
    };
    for (order.items) |k| for (order.items) |i| {
        if (!reach.bits[i * n + k]) continue;
        for (order.items) |j| {
            if (reach.bits[k * n + j]) reach.bits[i * n + j] = true;
        }
    };

    for (order.items) |i| {
        const f = fs[i].?;
        f.removeDead();
        try f.findTransients();
        f.assignSlots();
        try f.computeSaves(&reach);
    }

    const usable = options.stack_depth - 1;
    for (order.items) |i| {
        const peak = fs[i].?.dataPeak();
        if (peak > usable) return fail(options.diag, error.StackTooDeep, "word {s} needs {d} data stack cells, the stack holds {d}", .{ items[i].name, peak, usable });
    }
    const memo = try arena.alloc(?u32, n);
    @memset(memo, null);
    const base = 1 + longestChain(fs, &reach, order.items, @intCast(main), memo);
    if (base > usable) return fail(options.diag, error.StackTooDeep, "a call chain needs {d} return stack cells, the stack holds {d}", .{ base, usable });
    if (g.need_enter) {
        if (base >= usable) return fail(options.diag, error.StackTooDeep, "recursion needs a return stack cell and the call chain above it takes {d} of {d}", .{ base, usable });
        g.max_frames = usable - base;
        g.save_cells = g.max_frames * g.max_saves;
    }

    // Layout: the entry jump, the data area, the words, the helper routines and the start code.
    try g.words.append(arena, 0);
    var at: usize = 0;
    while (at < data.len) : (at += 2) {
        const hi: u16 = if (at + 1 < data.len) data[at + 1] else 0;
        try g.words.append(arena, data[at] | hi << 8);
    }
    for (order.items) |i| try fs[i].?.emit();
    try emitRoutines(&g);

    g.routine_addr[@backingInt(Routine.entry)] = g.here();
    if (g.need_enter) {
        try g.literal(0);
        try storeSlot(&g, g.rdSlot());
    }
    if (g.need_save) {
        try g.fixed(.lit, .stack_base);
        try storeSlot(&g, g.spSlot());
    }
    try g.fixed(.call, .{ .func = @intCast(main) });
    try g.word(.{ .jmp = @intCast(g.here()) });
    g.words.items[0] = try encode.encode(.{ .jmp = @intCast(g.routine_addr[@backingInt(Routine.entry)].?) });

    const total = g.here() + g.slot_count + g.save_cells;
    if (g.here() > encode.max_image_words or total > options.memory_words) {
        return fail(options.diag, error.ImageTooLarge, "image {d} words + {d} slots + {d} save stack cells = {d} words, memory has {d}", .{ g.here(), g.slot_count, g.save_cells, total, options.memory_words });
    }
    try patch(&g);
    return .{ .words = try allocator.dupe(u16, g.words.items), .slots = g.slot_count };
}

/// The trap word has the reserved ALU bit set, so the core faults on it.
const trap_word: u16 = 0x6010;

fn emitFrames(g: *Gen) Error!void {
    const rd = g.rdSlot();
    g.routine_addr[@backingInt(Routine.trap)] = g.here();
    try g.words.append(g.arena, trap_word);

    g.routine_addr[@backingInt(Routine.enter)] = g.here();
    // ( -- ) count one more active recursive call and trap when there are too many.
    try loadSlot(g, rd);
    try g.literal(1);
    try g.word(encode.plus);
    try g.word(encode.dup);
    try g.literal(@intCast(g.max_frames + 1));
    try g.word(encode.ult);
    try g.fixedRoutine(.jz, .trap);
    try storeSlot(g, rd);
    try g.word(encode.exit);

    g.routine_addr[@backingInt(Routine.leave)] = g.here();
    try loadSlot(g, rd);
    try g.word(encode.decrement);
    try storeSlot(g, rd);
    try g.word(encode.exit);
}

/// The most return stack cells the calls made by `f` and by the words that can call it back
/// can take, not counting calls inside the cycle. A call that stays in the cycle counts at run
/// time, in the frame counter.
fn longestChain(fs: []const ?*F, reach: *const Reach, order: []const u32, f: u32, memo: []?u32) u32 {
    if (memo[f]) |known| return known;
    var best: u32 = 0;
    for (order) |m| {
        if (!inCycle(reach, f, m)) continue;
        const member = fs[m].?;
        if (member.usesHelper()) best = @max(best, 1);
        for (member.callees.items) |callee| {
            if (inCycle(reach, f, callee)) continue;
            best = @max(best, 1 + longestChain(fs, reach, order, callee, memo));
        }
    }
    memo[f] = best;
    return best;
}

fn inCycle(reach: *const Reach, f: u32, g: u32) bool {
    return f == g or (reach.bits[f * reach.n + g] and reach.bits[g * reach.n + f]);
}
