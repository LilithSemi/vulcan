//! Lifting Geyser instruction words into Vulcan IR, so a host can run a Forth program as native
//! code.
//!
//! The words say where they go, so a program lifts into one IR function with a block for each
//! jump target and each word after a call. The data stack, the return stack and their depths
//! live in the context. The top of stack and the two depths live in stack slots, so `mem2reg`
//! makes them values and every pass after it can see through them.
//!
//! A return goes to an address read from the return stack. The lifted code compares that address
//! with the word after each call and jumps to the block that matches. An address that matches
//! none leaves the function with `Exit.jumped`.
//!
//! A word that makes the core fault leaves the function with `Exit.faulted`. The state stays as
//! it was before that word, and `Core.fault_cause` has the same bits as the hardware.
//!
//! Lifted code does not see a store into the words of the program. The program that runs is the
//! program that was lifted.

const std = @import("std");
const ir = @import("vulcan-ir");
const decode = @import("decode.zig");
const encode = @import("encode.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = std.mem.Allocator.Error || error{
    /// A word named a target outside the program, or the program ran off its end.
    Outside,
    /// The stack depth is not 8, 16 or 32.
    BadDepth,
};

/// Why compiled code gave the core back.
pub const Exit = enum(u32) {
    /// A jump to its own word. Nothing can change after it.
    halted = 0,
    /// A stack leaving its range, or an illegal word. `Core.fault_cause` has the bits.
    faulted = 1,
    /// The host asked the core to stop. `Core.pc` is where to carry on from.
    stopped = 2,
    /// A return to an address that is not the word after a call. `Core.pc` is that address.
    jumped = 3,
};

/// Bits of `Core.fault_cause`.
pub const cause = struct {
    pub const stack: u32 = 1;
    pub const illegal: u32 = 2;
};

/// The largest stack the core has. A smaller stack uses the start of each array.
pub const max_stack = 32;

/// How `Core.io` is called.
pub const io = struct {
    pub const read: u32 = 0;
    pub const write: u32 = 1;
};

/// What a compiled program reads and writes. Lifted code and the code that runs it agree on this
/// layout, so it is `extern`.
pub const Core = extern struct {
    d: [max_stack]u16 = @splat(0),
    r: [max_stack]u16 = @splat(0),
    /// The top of stack, then the two stack depths.
    t: u32 = 0,
    dsp: u32 = 0,
    rsp: u32 = 0,
    /// The word the core stopped at.
    pc: u32 = 0,
    fault_cause: u32 = 0,
    /// Set by the host to take the core down at the next backward jump or return.
    stop: u32 = 0,

    /// Memory, and its size in words minus one. The size is a power of two.
    mem: ?[*]u16 = null,
    mem_mask: u32 = 0,

    /// Answers a byte address of 0xF000 and up. A read takes the answer and a write takes the
    /// value. `kind` is `io.read` or `io.write`.
    io: ?*const fn (ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u32 = null,
    io_ctx: ?*anyopaque = null,
};

const Lifter = struct {
    allocator: std.mem.Allocator,
    func: *Function,
    program: []const u16,
    depth: u32,

    u16_t: Type,
    u32_t: Type,
    u64_t: Type,
    bool_t: Type,
    ptr_t: Type,

    ctx: Value,
    t_slot: Value,
    dsp_slot: Value,
    rsp_slot: Value,
    res_slot: Value,

    leaders: []bool,
    reached: []bool,
    blocks: []?Block,
    /// The word after each call that is reached.
    sites: std.ArrayList(u32) = .empty,
    at: Block,

    fault_block: ?Block = null,
    ret_block: ?Block = null,

    fn constant(self: *Lifter, value: u32) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .iconst = @as(i64, value) });
    }

    fn arith(self: *Lifter, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = op, .lhs = a, .rhs = b } });
    }

    fn imm(self: *Lifter, op: ir.function.BinOp, a: Value, value: i64) Error!Value {
        return self.func.appendArithImm(self.at, self.u32_t, op, a, value);
    }

    fn compare(self: *Lifter, op: ir.function.CmpOp, a: Value, b: Value) Error!Value {
        return self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = op, .lhs = a, .rhs = b } });
    }

    fn select(self: *Lifter, cond: Value, then: Value, other: Value) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = cond, .then = then, .@"else" = other } });
    }

    fn signedLess(self: *Lifter, a: Value, b: Value) Error!Value {
        // Flipping the sign bit of both turns a signed order into an unsigned one.
        const x = try self.imm(.bit_xor, a, 0x8000);
        const y = try self.imm(.bit_xor, b, 0x8000);
        return self.compare(.lt, x, y);
    }

    fn all(self: *Lifter, cond: Value) Error!Value {
        return self.select(cond, try self.constant(0xFFFF), try self.constant(0));
    }

    fn load(self: *Lifter, slot: Value) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = slot } });
    }

    fn field(self: *Lifter, comptime name: []const u8) Error!Value {
        const offset = @offsetOf(Core, name);
        if (offset == 0) return self.ctx;
        return self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, offset);
    }

    fn read(self: *Lifter, comptime name: []const u8, ty: Type) Error!Value {
        return self.func.appendInst(self.at, ty, .{ .load = .{ .ptr = try self.field(name) } });
    }

    fn writeField(self: *Lifter, comptime name: []const u8, value: Value) Error!void {
        try self.func.appendStore(self.at, value, try self.field(name));
    }

    /// The address of one 16 bit cell of `base`, by cell index.
    fn cellAt(self: *Lifter, base: Value, index: Value) Error!Value {
        const bytes = try self.imm(.shl, index, 1);
        const wide = try self.func.appendInst(self.at, self.u64_t, .{ .convert = .{ .value = bytes } });
        return self.func.appendInst(self.at, self.ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = wide } });
    }

    fn stackAt(self: *Lifter, comptime name: []const u8, index: Value) Error!Value {
        return self.cellAt(try self.field(name), index);
    }

    fn loadCell(self: *Lifter, at: Value) Error!Value {
        const cell = try self.func.appendInst(self.at, self.u16_t, .{ .load = .{ .ptr = at } });
        return self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = cell } });
    }

    fn storeCell(self: *Lifter, value: Value, at: Value) Error!void {
        const cell = try self.func.appendInst(self.at, self.u16_t, .{ .convert = .{ .value = value } });
        try self.func.appendStore(self.at, cell, at);
    }

    fn memAt(self: *Lifter, t: Value) Error!Value {
        const word = try self.imm(.shr, t, 1);
        const wrapped = try self.arith(.bit_and, word, try self.read("mem_mask", self.u32_t));
        return self.cellAt(try self.read("mem", self.ptr_t), wrapped);
    }

    fn isIo(self: *Lifter, t: Value) Error!Value {
        return self.compare(.eq, try self.imm(.shr, t, 12), try self.constant(0xF));
    }

    fn askIo(self: *Lifter, t: Value, value: Value, kind: u32) Error!Value {
        const handler = try self.read("io", self.ptr_t);
        const handler_ctx = try self.read("io_ctx", self.ptr_t);
        const at = try self.imm(.bit_and, t, 0xFFFE);
        const got = try self.func.appendCallIndirect(self.at, self.u32_t, handler, &.{
            handler_ctx, at, value, try self.constant(kind),
        });
        return self.imm(.bit_and, got, 0xFFFF);
    }

    /// Write the state back and give the core back.
    fn leave(self: *Lifter, why: Exit, pc: Value) Error!void {
        try self.writeField("pc", pc);
        try self.writeField("t", try self.load(self.t_slot));
        try self.writeField("dsp", try self.load(self.dsp_slot));
        try self.writeField("rsp", try self.load(self.rsp_slot));
        const reason = try self.constant(@backingInt(why));
        self.func.setTerminator(self.at, .{ .ret = ir.function.Ret.one(reason) });
    }

    fn faultBlock(self: *Lifter) Error!Block {
        if (self.fault_block) |block| return block;
        const block = try self.func.appendBlock();
        _ = try self.func.appendBlockParam(block, self.u32_t);
        _ = try self.func.appendBlockParam(block, self.u32_t);
        self.fault_block = block;
        return block;
    }

    fn retBlock(self: *Lifter) Error!Block {
        if (self.ret_block) |block| return block;
        const block = try self.func.appendBlock();
        _ = try self.func.appendBlockParam(block, self.u32_t);
        self.ret_block = block;
        return block;
    }

    /// Jump to the block of a word. A jump backwards reads the host's stop first, because a
    /// program that loops forever is the usual program.
    fn goto(self: *Lifter, from: u32, to: u32) Error!void {
        if (to > from) return self.func.setJump(self.at, self.blocks[to].?, &.{});
        const stop = try self.read("stop", self.u32_t);
        const asked = try self.compare(.ne, stop, try self.constant(0));
        const leaving = try self.func.appendBlock();
        try self.func.appendIf(self.at, asked, .{ .target = leaving }, .{ .target = self.blocks[to].? });
        self.at = leaving;
        try self.leave(.stopped, try self.constant(to));
    }

    /// The stack changes of a word, read from its bits. A reserved word still has them.
    fn deltas(word: u16) struct { d: i32, r: i32 } {
        const table = [4]i32{ 0, 1, -2, -1 };
        if (word & 0x8000 != 0) return .{ .d = 1, .r = 0 };
        return switch (@as(u2, @truncate(word >> 13))) {
            0 => .{ .d = 0, .r = 0 },
            1 => .{ .d = -1, .r = 0 },
            2 => .{ .d = 0, .r = 1 },
            3 => .{ .d = table[word & 3], .r = table[(word >> 2) & 3] },
        };
    }

    /// Branch to the fault block when a stack would leave its range or the word is illegal.
    /// The code after this runs only when none of that happened.
    fn guard(self: *Lifter, index: u32, d: i32, r: i32, illegal: bool, io_clash: ?Value, dsp: Value, rsp: Value) Error!void {
        if (d == 0 and r == 0 and !illegal and io_clash == null) return;
        var bits = try self.constant(0);
        if (d != 0) {
            const next = try self.imm(.add, dsp, d);
            const bad = try self.compare(.ge, next, try self.constant(self.depth));
            bits = try self.select(bad, try self.constant(cause.stack), bits);
        }
        if (r != 0) {
            const next = try self.imm(.add, rsp, r);
            const bad = try self.compare(.ge, next, try self.constant(self.depth));
            bits = try self.select(bad, try self.constant(cause.stack), bits);
        }
        const fault = try self.faultBlock();
        const pc = try self.constant(index);
        if (illegal) {
            bits = try self.imm(.bit_or, bits, cause.illegal);
            return self.func.setJump(self.at, fault, &.{ bits, pc });
        }
        if (io_clash) |clash| {
            const with = try self.imm(.bit_or, bits, cause.illegal);
            bits = try self.select(clash, with, bits);
        }
        const bad = try self.compare(.ne, bits, try self.constant(0));
        const next = try self.func.appendBlock();
        try self.func.appendIf(self.at, bad, .{ .target = fault, .args = &.{ bits, pc } }, .{ .target = next });
        self.at = next;
    }

    fn discover(self: *Lifter, entry: u32) Error!void {
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(self.allocator);
        try stack.append(self.allocator, entry);

        while (stack.pop()) |at| {
            if (at >= self.program.len) return error.Outside;
            if (self.reached[at]) continue;
            self.reached[at] = true;
            const next = try self.successors(at);
            for (next.items()) |to| try stack.append(self.allocator, to);
        }

        self.leaders[entry] = true;
        for (0..self.program.len) |at| {
            if (!self.reached[at]) continue;
            const insn = decode.decode(self.program[at]) catch continue;
            switch (insn) {
                .jmp => |to| self.leaders[to] = true,
                .jz => |to| {
                    self.leaders[to] = true;
                    self.leaders[at + 1] = true;
                },
                .call => |to| {
                    self.leaders[to] = true;
                    self.leaders[at + 1] = true;
                    try self.sites.append(self.allocator, @intCast(at + 1));
                },
                else => {},
            }
        }
    }

    const Next = struct {
        list: [2]u32 = undefined,
        len: usize = 0,

        fn items(self: *const Next) []const u32 {
            return self.list[0..self.len];
        }

        fn add(self: *Next, to: u32) void {
            self.list[self.len] = to;
            self.len += 1;
        }
    };

    /// The words control can reach from one word without a return.
    fn successors(self: *Lifter, at: u32) Error!Next {
        var next: Next = .{};
        const insn = decode.decode(self.program[at]) catch return next;
        switch (insn) {
            .lit => next.add(at + 1),
            .jmp => |to| if (to != at) next.add(to),
            .jz => |to| {
                next.add(to);
                next.add(at + 1);
            },
            .call => |to| {
                next.add(to);
                next.add(at + 1);
            },
            .alu => |alu| if (!alu.ret) next.add(at + 1),
        }
        for (next.items()) |to| if (to >= self.program.len) return error.Outside;
        return next;
    }

    fn emit(self: *Lifter, start: u32) Error!void {
        self.at = self.blocks[start].?;
        var index = start;
        while (true) {
            const word = self.program[index];
            const t = try self.load(self.t_slot);
            const dsp = try self.load(self.dsp_slot);
            const rsp = try self.load(self.rsp_slot);
            const moves = deltas(word);
            const insn = decode.decode(word) catch {
                return self.guard(index, moves.d, moves.r, true, null, dsp, rsp);
            };
            const dnew = try self.imm(.add, dsp, moves.d);
            const rnew = try self.imm(.add, rsp, moves.r);

            switch (insn) {
                .lit => |value| {
                    try self.guard(index, moves.d, moves.r, false, null, dsp, rsp);
                    try self.storeCell(t, try self.stackAt("d", dnew));
                    try self.finish(try self.constant(value), dnew, rsp);
                },
                .jmp => |to| {
                    if (to == index) return self.leave(.halted, try self.constant(index));
                    return self.goto(index, to);
                },
                .jz => |to| {
                    try self.guard(index, moves.d, moves.r, false, null, dsp, rsp);
                    const n = try self.loadCell(try self.stackAt("d", dsp));
                    const taken = try self.compare(.eq, t, try self.constant(0));
                    try self.finish(n, dnew, rsp);
                    return self.branch(index, taken, to);
                },
                .call => |to| {
                    try self.guard(index, moves.d, moves.r, false, null, dsp, rsp);
                    try self.storeCell(try self.constant(index + 1), try self.stackAt("r", rnew));
                    try self.finish(t, dsp, rnew);
                    return self.goto(index, to);
                },
                .alu => |alu| {
                    const returned = try self.aluWord(index, alu, t, dsp, rsp, dnew, rnew, moves.d, moves.r);
                    if (returned) |address| {
                        const block = try self.retBlock();
                        return self.func.setJump(self.at, block, &.{address});
                    }
                },
            }

            index += 1;
            if (index >= self.program.len) return error.Outside;
            if (self.leaders[index]) {
                try self.func.setJump(self.at, self.blocks[index].?, &.{});
                return;
            }
        }
    }

    /// Set the top of stack and the two depths.
    fn finish(self: *Lifter, t: Value, dsp: Value, rsp: Value) Error!void {
        try self.func.appendStore(self.at, t, self.t_slot);
        try self.func.appendStore(self.at, dsp, self.dsp_slot);
        try self.func.appendStore(self.at, rsp, self.rsp_slot);
    }

    /// A conditional jump, to a block of the program or to the word after.
    fn branch(self: *Lifter, index: u32, taken: Value, to: u32) Error!void {
        if (to > index) {
            return self.func.appendIf(self.at, taken, .{ .target = self.blocks[to].? }, .{ .target = self.blocks[index + 1].? });
        }
        const back = try self.func.appendBlock();
        try self.func.appendIf(self.at, taken, .{ .target = back }, .{ .target = self.blocks[index + 1].? });
        self.at = back;
        try self.goto(index, to);
    }

    /// One ALU word. Answers the return address when the word returns.
    fn aluWord(
        self: *Lifter,
        index: u32,
        word: encode.Alu,
        t: Value,
        dsp: Value,
        rsp: Value,
        dnew: Value,
        rnew: Value,
        d: i32,
        r: i32,
    ) Error!?Value {
        const reads_io = word.t_op == .fetch;
        const clash = if (reads_io and word.store) try self.isIo(t) else null;
        try self.guard(index, d, r, false, clash, dsp, rsp);

        const uses_n = switch (word.t_op) {
            .t, .invert, .decrement, .r, .depth, .fetch => false,
            else => true,
        } or word.store;
        const n = if (uses_n) try self.loadCell(try self.stackAt("d", dsp)) else undefined;
        const uses_r = word.t_op == .r or word.ret;
        const top = if (uses_r) try self.loadCell(try self.stackAt("r", rsp)) else undefined;

        const result: Value = switch (word.t_op) {
            .t => t,
            .n => n,
            .add => try self.imm(.bit_and, try self.arith(.add, t, n), 0xFFFF),
            .@"and" => try self.arith(.bit_and, t, n),
            .@"or" => try self.arith(.bit_or, t, n),
            .xor => try self.arith(.bit_xor, t, n),
            .invert => try self.imm(.bit_xor, t, 0xFFFF),
            .eq => try self.all(try self.compare(.eq, n, t)),
            .lt => try self.all(try self.signedLess(n, t)),
            .ult => try self.all(try self.compare(.lt, n, t)),
            .rshift => try self.shift(.shr, n, t),
            .lshift => try self.shift(.shl, n, t),
            .decrement => try self.imm(.bit_and, try self.imm(.sub, t, 1), 0xFFFF),
            .r => top,
            .depth => try self.arith(.bit_or, try self.imm(.shl, rsp, 8), dsp),
            .fetch => try self.fetch(t),
        };

        if (word.store) try self.memStore(t, n);
        if (word.t_to_n) try self.storeCell(t, try self.stackAt("d", dnew));
        if (word.t_to_r) try self.storeCell(t, try self.stackAt("r", rnew));
        try self.finish(result, dnew, rnew);
        return if (word.ret) top else null;
    }

    /// A shift by 16 or more gives zero.
    fn shift(self: *Lifter, op: ir.function.BinOp, n: Value, t: Value) Error!Value {
        const small = try self.compare(.lt, t, try self.constant(16));
        const amount = try self.imm(.bit_and, t, 15);
        const moved = try self.imm(.bit_and, try self.arith(op, n, amount), 0xFFFF);
        return self.select(small, moved, try self.constant(0));
    }

    fn fetch(self: *Lifter, t: Value) Error!Value {
        const device = try self.func.appendBlock();
        const memory = try self.func.appendBlock();
        const merge = try self.func.appendBlock();
        try self.func.appendIf(self.at, try self.isIo(t), .{ .target = device }, .{ .target = memory });

        self.at = device;
        try self.func.appendStore(self.at, try self.askIo(t, try self.constant(0), io.read), self.res_slot);
        try self.func.setJump(self.at, merge, &.{});

        self.at = memory;
        try self.func.appendStore(self.at, try self.loadCell(try self.memAt(t)), self.res_slot);
        try self.func.setJump(self.at, merge, &.{});

        self.at = merge;
        return self.load(self.res_slot);
    }

    fn memStore(self: *Lifter, t: Value, n: Value) Error!void {
        const device = try self.func.appendBlock();
        const memory = try self.func.appendBlock();
        const merge = try self.func.appendBlock();
        try self.func.appendIf(self.at, try self.isIo(t), .{ .target = device }, .{ .target = memory });

        self.at = device;
        _ = try self.askIo(t, n, io.write);
        try self.func.setJump(self.at, merge, &.{});

        self.at = memory;
        try self.storeCell(n, try self.memAt(t));
        try self.func.setJump(self.at, merge, &.{});

        self.at = merge;
    }

    fn finishFault(self: *Lifter) Error!void {
        const block = self.fault_block orelse return;
        self.at = block;
        const params = self.func.blockParams(block);
        try self.writeField("fault_cause", params[0]);
        try self.leave(.faulted, params[1]);
    }

    fn finishReturn(self: *Lifter) Error!void {
        const block = self.ret_block orelse return;
        const address = self.func.blockParams(block)[0];
        self.at = block;

        const stop = try self.read("stop", self.u32_t);
        const asked = try self.compare(.ne, stop, try self.constant(0));
        const leaving = try self.func.appendBlock();
        const dispatch = try self.func.appendBlock();
        try self.func.appendIf(self.at, asked, .{ .target = leaving }, .{ .target = dispatch });
        self.at = leaving;
        try self.leave(.stopped, address);

        self.at = dispatch;
        for (self.sites.items) |site| {
            const hit = try self.compare(.eq, address, try self.constant(site));
            const miss = try self.func.appendBlock();
            try self.func.appendIf(self.at, hit, .{ .target = self.blocks[site].? }, .{ .target = miss });
            self.at = miss;
        }
        try self.leave(.jumped, address);
    }
};

/// Lift the program a core reaches from the word `entry` into `func`.
///
/// The function takes one argument, the address of a `Core`, and answers with an `Exit`.
/// `stack_depth` is the depth of both stacks and is 8, 16 or 32.
pub fn region(allocator: std.mem.Allocator, func: *Function, program: []const u16, entry: u32, stack_depth: u32) Error!void {
    if (stack_depth != 8 and stack_depth != 16 and stack_depth != 32) return error.BadDepth;
    if (entry >= program.len) return error.Outside;

    const leaders = try allocator.alloc(bool, program.len);
    defer allocator.free(leaders);
    const reached = try allocator.alloc(bool, program.len);
    defer allocator.free(reached);
    const blocks = try allocator.alloc(?Block, program.len);
    defer allocator.free(blocks);
    @memset(leaders, false);
    @memset(reached, false);
    @memset(blocks, null);

    const prologue = try func.appendBlock();
    var self: Lifter = .{
        .allocator = allocator,
        .func = func,
        .program = program,
        .depth = stack_depth,
        .u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
        .u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
        .u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } }),
        .bool_t = try func.types.intern(.bool),
        .ptr_t = try func.types.ptrGlobal(),
        .ctx = undefined,
        .t_slot = undefined,
        .dsp_slot = undefined,
        .rsp_slot = undefined,
        .res_slot = undefined,
        .leaders = leaders,
        .reached = reached,
        .blocks = blocks,
        .at = prologue,
    };
    defer self.sites.deinit(allocator);
    self.ctx = try func.appendBlockParam(prologue, self.ptr_t);

    try self.discover(entry);

    self.t_slot = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    self.dsp_slot = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    self.rsp_slot = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    self.res_slot = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    try func.appendStore(prologue, try self.read("t", self.u32_t), self.t_slot);
    try func.appendStore(prologue, try self.read("dsp", self.u32_t), self.dsp_slot);
    try func.appendStore(prologue, try self.read("rsp", self.u32_t), self.rsp_slot);

    for (0..program.len) |at| {
        if (self.leaders[at] and self.reached[at]) blocks[at] = try func.appendBlock();
    }
    try func.setJump(prologue, blocks[entry].?, &.{});

    for (0..program.len) |at| {
        if (blocks[at] == null) continue;
        try self.emit(@intCast(at));
    }
    try self.finishFault();
    try self.finishReturn();
}
