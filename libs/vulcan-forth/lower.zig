//! Lowers Forth words to Vulcan IR. Each word becomes one function. Inside a function the
//! data stack, the return stack and the loop parameters are SSA values, and every join
//! carries them as block parameters. A cell is a 32 bit unsigned value kept in the range
//! 0 to 0xFFFF, so each operation that can leave that range masks its result.
//!
//! A function takes a context pointer and then the `in` cells of the word, deepest first.
//! It returns the top result. Any further results go to the context before the return.

const std = @import("std");
const ir = @import("vulcan-ir");
const ast = @import("ast.zig");
const prims = @import("prims.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Type = ir.types.Type;
const Node = ast.Node;
const Word = ast.Word;

pub const Error = std.mem.Allocator.Error;

/// Byte offsets in the context the generated code reads.
pub const ctx_mem = 0;
pub const ctx_io_read = 8;
pub const ctx_io_write = 16;
pub const ctx_results = 24;
pub const ctx_size = ctx_results + 4 * ast.max_cells;

const cell_mask: i64 = 0xFFFF;

/// A block that several paths reach. It takes the whole stack state as parameters.
const Join = struct {
    block: ?Block = null,
    params: []const Value = &.{},
    data_len: usize = 0,
    rstack_len: usize = 0,
};

const Snapshot = struct {
    data: []const Value,
    rstack: []const Value,
    loops: []const Value,
};

const L = struct {
    arena: std.mem.Allocator,
    func: *Function,
    block: Block,
    words: []const Word,
    symbols: []const []const u8,
    word: *const Word,
    word_index: usize,
    cell_t: Type,
    ptr_t: Type,
    bool_t: Type,
    u16_t: Type,
    ctx: Value,
    mem: Value,
    data: std.ArrayList(Value) = .empty,
    rstack: std.ArrayList(Value) = .empty,
    /// Pairs of (limit, index), innermost last.
    loops: std.ArrayList(Value) = .empty,
    dead: bool = false,
    exits: std.ArrayList(*Join) = .empty,

    fn iconst(self: *L, v: i64) Error!Value {
        return self.func.appendInst(self.block, self.cell_t, .{ .iconst = v });
    }

    fn arith(self: *L, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
        return self.func.appendInst(self.block, self.cell_t, .{ .arith = .{ .op = op, .lhs = a, .rhs = b } });
    }

    fn imm(self: *L, op: ir.function.BinOp, a: Value, v: i64) Error!Value {
        return self.func.appendArithImm(self.block, self.cell_t, op, a, v);
    }

    fn masked(self: *L, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
        return self.imm(.bit_and, try self.arith(op, a, b), cell_mask);
    }

    fn cmp(self: *L, op: ir.function.CmpOp, a: Value, b: Value) Error!Value {
        return self.func.appendInst(self.block, self.bool_t, .{ .icmp = .{ .op = op, .lhs = a, .rhs = b } });
    }

    fn flag(self: *L, cond: Value) Error!Value {
        const yes = try self.iconst(cell_mask);
        const no = try self.iconst(0);
        return self.func.appendInst(self.block, self.cell_t, .{ .select = .{ .cond = cond, .then = yes, .@"else" = no } });
    }

    fn select(self: *L, cond: Value, a: Value, b: Value) Error!Value {
        return self.func.appendInst(self.block, self.cell_t, .{ .select = .{ .cond = cond, .then = a, .@"else" = b } });
    }

    fn pop(self: *L) Value {
        return self.data.pop().?;
    }

    fn push(self: *L, v: Value) Error!void {
        try self.data.append(self.arena, v);
    }

    fn top(self: *L, depth: usize) Value {
        return self.data.items[self.data.items.len - 1 - depth];
    }

    // State handling around control flow.

    fn snapshot(self: *L) Error!Snapshot {
        return .{
            .data = try self.arena.dupe(Value, self.data.items),
            .rstack = try self.arena.dupe(Value, self.rstack.items),
            .loops = try self.arena.dupe(Value, self.loops.items),
        };
    }

    fn restore(self: *L, s: Snapshot) Error!void {
        self.data.clearRetainingCapacity();
        try self.data.appendSlice(self.arena, s.data);
        self.rstack.clearRetainingCapacity();
        try self.rstack.appendSlice(self.arena, s.rstack);
        self.loops.clearRetainingCapacity();
        try self.loops.appendSlice(self.arena, s.loops);
        self.dead = false;
    }

    fn flat(self: *L) Error![]Value {
        const out = try self.arena.alloc(Value, self.data.items.len + self.rstack.items.len + self.loops.items.len);
        @memcpy(out[0..self.data.items.len], self.data.items);
        @memcpy(out[self.data.items.len..][0..self.rstack.items.len], self.rstack.items);
        @memcpy(out[self.data.items.len + self.rstack.items.len ..], self.loops.items);
        return out;
    }

    fn setFlat(self: *L, values: []const Value, data_len: usize, rstack_len: usize) Error!void {
        self.data.clearRetainingCapacity();
        self.rstack.clearRetainingCapacity();
        self.loops.clearRetainingCapacity();
        try self.data.appendSlice(self.arena, values[0..data_len]);
        try self.rstack.appendSlice(self.arena, values[data_len..][0..rstack_len]);
        try self.loops.appendSlice(self.arena, values[data_len + rstack_len ..]);
        self.dead = false;
    }

    /// End the current block with a jump to `join`, creating it on the first jump.
    fn jumpTo(self: *L, join: *Join) Error!void {
        const args = try self.flat();
        if (join.block == null) {
            const block = try self.func.appendBlock();
            const params = try self.arena.alloc(Value, args.len);
            for (params) |*p| p.* = try self.func.appendBlockParam(block, self.cell_t);
            join.* = .{ .block = block, .params = params, .data_len = self.data.items.len, .rstack_len = self.rstack.items.len };
        }
        std.debug.assert(join.params.len == args.len);
        try self.func.setJump(self.block, join.block.?, args);
        self.dead = true;
    }

    /// Continue in `join`, or mark the code dead when nothing reached it.
    fn resumeAt(self: *L, join: *Join) Error!void {
        const block = join.block orelse {
            self.dead = true;
            return;
        };
        self.block = block;
        try self.setFlat(join.params, join.data_len, join.rstack_len);
    }

    /// Jump from here to a new block that takes the whole state, and continue in it.
    fn enterHeader(self: *L) Error!Block {
        const args = try self.flat();
        const block = try self.func.appendBlock();
        const params = try self.arena.alloc(Value, args.len);
        for (params) |*p| p.* = try self.func.appendBlockParam(block, self.cell_t);
        try self.func.setJump(self.block, block, args);
        self.block = block;
        try self.setFlat(params, self.data.items.len, self.rstack.items.len);
        return block;
    }

    // Memory.

    fn loadCell(self: *L, addr: Value) Error!Value {
        const a = try self.imm(.bit_and, addr, 0xFFFE);
        const limit = try self.iconst(ast.io_base);
        const is_io = try self.cmp(.ge, a, limit);
        const io_b = try self.func.appendBlock();
        const ram_b = try self.func.appendBlock();
        const join_b = try self.func.appendBlock();
        const result = try self.func.appendBlockParam(join_b, self.cell_t);
        try self.func.appendIf(self.block, is_io, .{ .target = io_b }, .{ .target = ram_b });

        const handler = try self.contextSlot(io_b, ctx_io_read, self.ptr_t);
        const io_value = try self.func.appendCallIndirect(io_b, self.cell_t, handler, &.{ self.ctx, a });
        try self.func.setJump(io_b, join_b, &.{io_value});

        const ptr = try self.func.appendInst(ram_b, self.ptr_t, .{ .arith = .{ .op = .add, .lhs = self.mem, .rhs = a } });
        const raw = try self.func.appendInst(ram_b, self.u16_t, .{ .load = .{ .ptr = ptr } });
        const ram_value = try self.func.appendArithImm(ram_b, self.cell_t, .bit_or, raw, 0);
        try self.func.setJump(ram_b, join_b, &.{ram_value});

        self.block = join_b;
        return result;
    }

    fn storeCell(self: *L, addr: Value, value: Value) Error!void {
        const a = try self.imm(.bit_and, addr, 0xFFFE);
        const limit = try self.iconst(ast.io_base);
        const is_io = try self.cmp(.ge, a, limit);
        const io_b = try self.func.appendBlock();
        const ram_b = try self.func.appendBlock();
        const join_b = try self.func.appendBlock();
        try self.func.appendIf(self.block, is_io, .{ .target = io_b }, .{ .target = ram_b });

        const handler = try self.contextSlot(io_b, ctx_io_write, self.ptr_t);
        try self.func.appendVoidCallIndirect(io_b, handler, &.{ self.ctx, a, value });
        try self.func.setJump(io_b, join_b, &.{});

        const ptr = try self.func.appendInst(ram_b, self.ptr_t, .{ .arith = .{ .op = .add, .lhs = self.mem, .rhs = a } });
        const narrow = try self.func.appendArithImm(ram_b, self.u16_t, .bit_or, value, 0);
        try self.func.appendStore(ram_b, narrow, ptr);
        try self.func.setJump(ram_b, join_b, &.{});

        self.block = join_b;
    }

    fn contextSlot(self: *L, block: Block, offset: i64, ty: Type) Error!Value {
        const p = if (offset == 0) self.ctx else try self.func.appendArithImm(block, self.ptr_t, .add, self.ctx, offset);
        return self.func.appendInst(block, ty, .{ .load = .{ .ptr = p } });
    }

    // Words.

    fn resultSlot(self: *L, k: usize) Error!Value {
        return self.func.appendArithImm(self.block, self.ptr_t, .add, self.ctx, @intCast(ctx_results + 4 * k));
    }

    fn emitReturn(self: *L) Error!void {
        const out = self.word.effect.out;
        std.debug.assert(self.data.items.len == out);
        if (out == 0) {
            self.func.setTerminator(self.block, .{ .ret = ir.function.Ret.none() });
        } else {
            for (self.data.items[0 .. out - 1], 0..) |v, k| try self.func.appendStore(self.block, v, try self.resultSlot(k));
            self.func.setTerminator(self.block, .{ .ret = ir.function.Ret.one(self.data.items[out - 1]) });
        }
        self.dead = true;
    }

    fn call(self: *L, index: usize) Error!void {
        const eff = self.words[index].effect;
        var args: [1 + ast.max_cells]Value = undefined;
        args[0] = self.ctx;
        const base = self.data.items.len - eff.in;
        @memcpy(args[1..][0..eff.in], self.data.items[base..]);
        self.data.shrinkRetainingCapacity(base);
        const name = self.symbols[index];
        if (eff.out == 0) {
            try self.func.appendVoidCall(self.block, name, args[0 .. 1 + eff.in]);
            return;
        }
        const ret = try self.func.appendCall(self.block, self.cell_t, name, args[0 .. 1 + eff.in]);
        for (0..eff.out - 1) |k| {
            const v = try self.func.appendInst(self.block, self.cell_t, .{ .load = .{ .ptr = try self.resultSlot(k) } });
            try self.push(v);
        }
        try self.push(ret);
    }

    fn seq(self: *L, nodes: []const Node) Error!void {
        for (nodes) |n| {
            if (self.dead) return;
            try self.node(n);
        }
    }

    fn node(self: *L, n: Node) Error!void {
        switch (n) {
            .lit => |v| try self.push(try self.iconst(v)),
            .call => |idx| try self.call(idx),
            .prim => |p| try self.prim(p),
            .@"if" => |branch| try self.ifThen(branch),
            .until => |body| {
                const header = try self.enterHeader();
                try self.seq(body);
                if (self.dead) return;
                const cond = try self.cmp(.ne, self.pop(), try self.iconst(0));
                const exit_b = try self.func.appendBlock();
                try self.func.appendIf(self.block, cond, .{ .target = exit_b }, .{ .target = header, .args = try self.flat() });
                self.block = exit_b;
            },
            .again => |body| {
                const header = try self.enterHeader();
                try self.seq(body);
                if (self.dead) return;
                try self.func.setJump(self.block, header, try self.flat());
                self.dead = true;
            },
            .@"while" => |loop| {
                const header = try self.enterHeader();
                try self.seq(loop.cond);
                if (self.dead) return;
                const cond = try self.cmp(.ne, self.pop(), try self.iconst(0));
                const body_b = try self.func.appendBlock();
                const exit_b = try self.func.appendBlock();
                try self.func.appendIf(self.block, cond, .{ .target = body_b }, .{ .target = exit_b });
                const leaving = try self.snapshot();
                self.block = body_b;
                try self.seq(loop.body);
                if (!self.dead) try self.func.setJump(self.block, header, try self.flat());
                try self.restore(leaving);
                self.block = exit_b;
            },
            .do => |loop| try self.doLoop(loop),
        }
    }

    fn ifThen(self: *L, branch: Node.If) Error!void {
        const cond = try self.cmp(.ne, self.pop(), try self.iconst(0));
        const then_b = try self.func.appendBlock();
        const else_b = try self.func.appendBlock();
        try self.func.appendIf(self.block, cond, .{ .target = then_b }, .{ .target = else_b });
        const before = try self.snapshot();
        var join: Join = .{};

        self.block = then_b;
        try self.seq(branch.then);
        if (!self.dead) try self.jumpTo(&join);

        try self.restore(before);
        self.block = else_b;
        if (branch.has_else) try self.seq(branch.@"else");
        if (!self.dead) try self.jumpTo(&join);

        try self.resumeAt(&join);
    }

    fn doLoop(self: *L, loop: Node.Do) Error!void {
        const index = self.pop();
        const limit = self.pop();
        try self.loops.append(self.arena, limit);
        try self.loops.append(self.arena, index);
        const header = try self.enterHeader();
        var exit_join: Join = .{};
        try self.exits.append(self.arena, &exit_join);
        defer _ = self.exits.pop();

        try self.seq(loop.body);
        if (!self.dead) {
            const loop_limit = self.loops.items[self.loops.items.len - 2];
            const loop_index = self.loops.items[self.loops.items.len - 1];
            var done: Value = undefined;
            var next_index: Value = undefined;
            if (loop.plus) {
                const step = self.pop();
                const before = try self.masked(.sub, loop_index, loop_limit);
                const after = try self.masked(.add, before, step);
                const crossed = try self.imm(.bit_and, try self.arith(.bit_xor, before, after), 0x8000);
                done = try self.cmp(.ne, crossed, try self.iconst(0));
                next_index = try self.masked(.add, loop_index, step);
            } else {
                next_index = try self.imm(.bit_and, try self.imm(.add, loop_index, 1), cell_mask);
                done = try self.cmp(.eq, next_index, loop_limit);
            }
            self.loops.items[self.loops.items.len - 1] = next_index;
            const args = try self.flat();
            self.loops.shrinkRetainingCapacity(self.loops.items.len - 2);
            const exit_b = try self.func.appendBlock();
            try self.func.appendIf(self.block, done, .{ .target = exit_b }, .{ .target = header, .args = args });
            self.block = exit_b;
            try self.jumpTo(&exit_join);
        }
        try self.resumeAt(&exit_join);
    }

    fn leave(self: *L) Error!void {
        const join = self.exits.items[self.exits.items.len - 1];
        const keep = try self.snapshot();
        self.loops.shrinkRetainingCapacity(self.loops.items.len - 2);
        try self.jumpTo(join);
        try self.restore(keep);
        self.dead = true;
    }

    fn prim(self: *L, p: ast.Prim) Error!void {
        switch (p) {
            .dup => try self.push(self.top(0)),
            .drop => _ = self.pop(),
            .swap => std.mem.swap(Value, &self.data.items[self.data.items.len - 1], &self.data.items[self.data.items.len - 2]),
            .over => try self.push(self.top(1)),
            .rot => {
                const items = self.data.items;
                const x = items[items.len - 3];
                items[items.len - 3] = items[items.len - 2];
                items[items.len - 2] = items[items.len - 1];
                items[items.len - 1] = x;
            },
            .nip => {
                const t = self.pop();
                _ = self.pop();
                try self.push(t);
            },
            .tuck => {
                const b = self.pop();
                const a = self.pop();
                try self.push(b);
                try self.push(a);
                try self.push(b);
            },
            .two_dup => {
                try self.push(self.top(1));
                try self.push(self.top(1));
            },
            .two_drop => {
                _ = self.pop();
                _ = self.pop();
            },
            .to_r => try self.rstack.append(self.arena, self.pop()),
            .r_from => try self.push(self.rstack.pop().?),
            .r_fetch => try self.push(self.rstack.items[self.rstack.items.len - 1]),
            .add, .sub, .mul => {
                const b = self.pop();
                const a = self.pop();
                try self.push(try self.masked(switch (p) {
                    .add => .add,
                    .sub => .sub,
                    else => .mul,
                }, a, b));
            },
            .bit_and, .bit_or, .bit_xor => {
                const b = self.pop();
                const a = self.pop();
                try self.push(try self.arith(switch (p) {
                    .bit_and => .bit_and,
                    .bit_or => .bit_or,
                    else => .bit_xor,
                }, a, b));
            },
            .lshift, .rshift => {
                const count = self.pop();
                const x = self.pop();
                const in_range = try self.cmp(.lt, count, try self.iconst(16));
                const shifted = if (p == .lshift) try self.masked(.shl, x, count) else try self.arith(.shr, x, count);
                try self.push(try self.select(in_range, shifted, try self.iconst(0)));
            },
            .eq, .ne, .ult => {
                const b = self.pop();
                const a = self.pop();
                try self.push(try self.flag(try self.cmp(switch (p) {
                    .eq => .eq,
                    .ne => .ne,
                    else => .lt,
                }, a, b)));
            },
            .lt, .gt => {
                const b = try self.imm(.bit_xor, self.pop(), 0x8000);
                const a = try self.imm(.bit_xor, self.pop(), 0x8000);
                try self.push(try self.flag(try self.cmp(if (p == .lt) .lt else .gt, a, b)));
            },
            .invert => try self.push(try self.imm(.bit_xor, self.pop(), cell_mask)),
            .negate => try self.push(try self.masked(.sub, try self.iconst(0), self.pop())),
            .zero_eq => try self.push(try self.flag(try self.cmp(.eq, self.pop(), try self.iconst(0)))),
            .zero_lt => {
                const sign = try self.imm(.bit_and, self.pop(), 0x8000);
                try self.push(try self.flag(try self.cmp(.ne, sign, try self.iconst(0))));
            },
            .fetch => try self.push(try self.loadCell(self.pop())),
            .store => {
                const addr = self.pop();
                try self.storeCell(addr, self.pop());
            },
            .plus_store => {
                const addr = self.pop();
                const n = self.pop();
                const old = try self.loadCell(addr);
                try self.storeCell(addr, try self.masked(.add, old, n));
            },
            .i => try self.push(self.loops.items[self.loops.items.len - 1]),
            .j => try self.push(self.loops.items[self.loops.items.len - 3]),
            .unloop => self.loops.shrinkRetainingCapacity(self.loops.items.len - 2),
            .leave => try self.leave(),
            .exit => try self.emitReturn(),
            .recurse => try self.call(self.word_index),
        }
    }
};

/// Lower word `index`. `symbols` names every word's function. The IR function belongs to the caller.
pub fn lowerWord(gpa: std.mem.Allocator, scratch: std.mem.Allocator, words: []const Word, symbols: []const []const u8, index: usize) Error!Function {
    const word = &words[index];
    var func = Function.init(gpa);
    errdefer func.deinit();
    const cell_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const entry = try func.appendBlock();
    const ptr_t = try func.types.ptrGlobal();
    const ctx = try func.appendBlockParam(entry, ptr_t);
    var l: L = .{
        .arena = scratch,
        .func = &func,
        .block = entry,
        .words = words,
        .symbols = symbols,
        .word = word,
        .word_index = index,
        .cell_t = cell_t,
        .ptr_t = ptr_t,
        .bool_t = try func.types.intern(.bool),
        .u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
        .ctx = ctx,
        .mem = try func.appendInst(entry, ptr_t, .{ .load = .{ .ptr = ctx } }),
    };
    for (0..word.effect.in) |_| try l.data.append(scratch, try func.appendBlockParam(entry, cell_t));
    try l.seq(word.body);
    if (!l.dead) try l.emitReturn();
    return func;
}

/// The IR for every word, in word order.
pub fn lowerAll(gpa: std.mem.Allocator, scratch: std.mem.Allocator, words: []const Word) Error![]ast.Lowered {
    const symbols = try scratch.alloc([]const u8, words.len);
    for (symbols, 0..) |*s, i| s.* = try std.fmt.allocPrint(scratch, "w{d}", .{i});
    const out = try scratch.alloc(ast.Lowered, words.len);
    var done: usize = 0;
    errdefer for (out[0..done]) |*lowered| lowered.func.deinit();
    for (out, 0..) |*slot, i| {
        slot.* = .{ .name = symbols[i], .func = try lowerWord(gpa, scratch, words, symbols, i) };
        done += 1;
    }
    return out;
}
