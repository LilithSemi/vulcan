//! Accumulator tail recursion. A function that returns `y op self(args)` for an associative and
//! commutative integer `op` loops instead: the call becomes a jump back to the function start with
//! `args` and an accumulator `acc op y`, and every other return gives `acc op value`. Each
//! remaining call does half the work, so `fib(n - 1) + fib(n - 2)` makes about half the calls.
//!
//! The recursive call must be the last call before its exit, with only its `op` after it, so the
//! jump skips nothing that has an effect. Integer `add`, `mul`, `and`, `or` and `xor` wrap, so
//! the reordering keeps every result.

const std = @import("std");
const ir = @import("vulcan-ir");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const BinOp = ir.function.BinOp;

pub const Error = std.mem.Allocator.Error;

/// An exit of the function: the block that returns, or that passes the result to the single
/// return block, and the value it returns.
const Exit = struct { block: Block, value: Value, through_if: ?bool };

const Recursion = struct { block: Block, call: Inst, op_inst: Inst, other: Value, args: []const Value };

pub fn run(allocator: std.mem.Allocator, func: *Function, self_name: []const u8) Error!bool {
    if (func.is_variadic or func.blockCount() == 0) return false;
    for (func.blockInsts(@fromBackingInt(0))) |inst| if (func.opcode(inst) == .alloca) return false;
    const entry: Block = @fromBackingInt(0);
    var exits: std.ArrayList(Exit) = .empty;
    defer exits.deinit(allocator);
    const join = returnJoin(func);
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        const insts = func.blockInsts(block);
        if (insts.len != 0 and func.opcode(insts[insts.len - 1]) == .@"if") {
            const cf = func.opcode(insts[insts.len - 1]).@"if";
            for ([_]bool{ true, false }) |is_then| {
                const edge = if (is_then) cf.then else cf.@"else";
                if (join != null and edge.target == join.?) try exits.append(allocator, .{ .block = block, .value = func.blockArgs(edge)[0], .through_if = is_then });
            }
            continue;
        }
        const term = func.terminator(block) orelse continue;
        switch (term) {
            .jump => |j| if (join != null and j.target == join.?) try exits.append(allocator, .{ .block = block, .value = func.blockArgs(j)[0], .through_if = null }),
            .ret => |r| {
                if (join != null and block == join.?) continue;
                if (r.count != 1) return false;
                try exits.append(allocator, .{ .block = block, .value = r.values[0], .through_if = null });
            },
        }
    }
    if (exits.items.len < 2) return false;

    // Every exit through a self call must use one `op`.
    var op: ?BinOp = null;
    var any = false;
    for (exits.items) |exit| {
        const rec = recursion(func, exit, self_name) orelse continue;
        const this_op = func.opcode(rec.op_inst).arith.op;
        if (op != null and op.? != this_op) return false;
        op = this_op;
        any = true;
    }
    if (!any) return false;
    // A self call anywhere else could still need the entry as it was, which is fine, but the
    // function must take only scalar integer parameters for the loop header to carry them.
    const ty = func.valueType(exits.items[0].value);
    if (func.types.type_kind(ty) != .int) return false;
    for (exits.items) |exit| if (func.valueType(exit.value) != ty) return false;

    // The entry's code moves into a loop header that also carries the accumulator.
    const header = try func.appendBlock();
    const params = try allocator.dupe(Value, func.blockParams(entry));
    defer allocator.free(params);
    const replacements = try allocator.alloc(Value, func.valueCount() + params.len + 1);
    defer allocator.free(replacements);
    for (replacements, 0..) |*r, index| r.* = @fromBackingInt(@intCast(index));
    var header_params: std.ArrayList(Value) = .empty;
    defer header_params.deinit(allocator);
    for (params) |p| {
        const hp = try func.appendBlockParam(header, func.valueType(p));
        try header_params.append(allocator, hp);
        replacements[@backingInt(p)] = hp;
    }
    const acc = try func.appendBlockParam(header, ty);
    const entry_insts = try allocator.dupe(Inst, func.blockInsts(entry));
    defer allocator.free(entry_insts);
    try func.setBlockInsts(header, entry_insts);
    if (func.terminator(entry)) |term| func.setTerminator(header, term);
    try func.setBlockInsts(entry, &.{});
    func.rewriteAllUses(replacements[0..func.valueCount()]);
    const identity = try func.appendInst(entry, ty, .{ .iconst = identityOf(op.?) });
    var start_args: std.ArrayList(Value) = .empty;
    defer start_args.deinit(allocator);
    try start_args.appendSlice(allocator, params);
    try start_args.append(allocator, identity);
    try func.setJump(entry, header, start_args.items);

    for (exits.items) |raw_exit| {
        var exit = raw_exit;
        if (exit.block == entry) exit.block = header;
        if (@backingInt(exit.value) < replacements.len) exit.value = replacements[@backingInt(exit.value)];
        if (recursion(func, exit, self_name)) |rec| {
            // `return other op self(args)` becomes `acc' = acc op other; goto start(args, acc')`.
            const args = try allocator.dupe(Value, rec.args);
            defer allocator.free(args);
            const list = func.blockInstsMut(rec.block);
            const call_at = std.mem.indexOfScalar(Inst, list.items, rec.call).?;
            list.shrinkRetainingCapacity(call_at);
            const keep = try allocator.dupe(Inst, list.items);
            defer allocator.free(keep);
            try func.setBlockInsts(rec.block, keep);
            const next = try func.appendInst(rec.block, ty, .{ .arith = .{ .op = op.?, .lhs = acc, .rhs = rec.other } });
            var loop_args: std.ArrayList(Value) = .empty;
            defer loop_args.deinit(allocator);
            try loop_args.appendSlice(allocator, args);
            try loop_args.append(allocator, next);
            try func.setJump(rec.block, header, loop_args.items);
            continue;
        }
        // Any other exit folds the accumulator into its value.
        const folded = try func.createInst(ty, .{ .arith = .{ .op = op.?, .lhs = acc, .rhs = exit.value } });
        const list = func.blockInstsMut(exit.block);
        const at = if (exit.through_if != null) list.items.len - 1 else list.items.len;
        try list.insert(func.allocator, at, func.definingInst(folded).?);
        if (exit.through_if) |is_then| {
            const cf = &func.opcodeMut(list.items[list.items.len - 1]).@"if";
            const edge = if (is_then) &cf.then else &cf.@"else";
            edge.args = try func.internValues(&.{folded});
        } else switch (func.terminator(exit.block).?) {
            .jump => |j| try func.setJump(exit.block, j.target, &.{folded}),
            .ret => func.setTerminator(exit.block, .{ .ret = ir.function.Ret.one(folded) }),
        }
    }
    return true;
}

/// The block that only returns its single parameter, when the function has one.
fn returnJoin(func: *const Function) ?Block {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        if (func.blockInsts(block).len != 0 or func.blockParams(block).len != 1) continue;
        const term = func.terminator(block) orelse continue;
        switch (term) {
            .ret => |r| if (r.count == 1 and r.values[0] == func.blockParams(block)[0]) return block,
            .jump => {},
        }
    }
    return null;
}

fn recursion(func: *const Function, exit: Exit, self_name: []const u8) ?Recursion {
    if (exit.through_if != null) return null;
    const op_inst = func.definingInst(exit.value) orelse return null;
    const a = switch (func.opcode(op_inst)) {
        .arith => |a| a,
        else => return null,
    };
    switch (a.op) {
        .add, .mul, .bit_and, .bit_or, .bit_xor => {},
        else => return null,
    }
    const insts = func.blockInsts(exit.block);
    if (insts.len < 2 or insts[insts.len - 1] != op_inst) return null;
    const call_inst = insts[insts.len - 2];
    const call = switch (func.opcode(call_inst)) {
        .call => |c| c,
        else => return null,
    };
    if (!std.mem.eql(u8, func.symbolName(call.symbol), self_name)) return null;
    if (call.is_variadic or call.sret or call.ret_dest != null or call.ret_regs != 0) return null;
    const result = func.instResult(call_inst) orelse return null;
    const other = if (a.rhs == result and a.lhs != result) a.lhs else if (a.lhs == result and a.rhs != result) a.rhs else return null;
    return .{ .block = exit.block, .call = call_inst, .op_inst = op_inst, .other = other, .args = func.valueList(call.args) };
}

fn identityOf(op: BinOp) i64 {
    return switch (op) {
        .mul => 1,
        .bit_and => -1,
        else => 0,
    };
}

const testing = std.testing;

test "fib's second call becomes a loop that carries the sum" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const recurse = try func.appendBlock();
    const join = try func.appendBlock();
    // return n < 2 ? n : fib(n - 1) + fib(n - 2);
    const n = try func.appendBlockParam(entry, u32_t);
    const two = try func.appendInst(entry, u32_t, .{ .iconst = 2 });
    const small = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .lt, .lhs = n, .rhs = two } });
    try func.appendIf(entry, small, .{ .target = join, .args = &.{n} }, .{ .target = recurse });
    const fib = try func.internSymbol("fib");
    const n1 = try func.appendArithImm(recurse, u32_t, .sub, n, 1);
    const f1 = try func.appendInst(recurse, u32_t, .{ .call = .{ .symbol = fib, .args = try func.internValues(&.{n1}) } });
    const n2 = try func.appendArithImm(recurse, u32_t, .sub, n, 2);
    const f2 = try func.appendInst(recurse, u32_t, .{ .call = .{ .symbol = fib, .args = try func.internValues(&.{n2}) } });
    const sum = try func.appendInst(recurse, u32_t, .{ .arith = .{ .op = .add, .lhs = f1, .rhs = f2 } });
    try func.setJump(recurse, join, &.{sum});
    const result = try func.appendBlockParam(join, u32_t);
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(result) });

    try testing.expect(try run(allocator, &func, "fib"));
    var calls: usize = 0;
    for (0..func.blockCount()) |bi| for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
        if (func.opcode(inst) == .call) calls += 1;
    };
    try testing.expectEqual(@as(usize, 1), calls);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a call whose result is used again stays a call" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const recurse = try func.appendBlock();
    const join = try func.appendBlock();
    // return n == 0 ? 1 : f(n - 1) - 3; subtraction does not reorder.
    const n = try func.appendBlockParam(entry, u32_t);
    const zero = try func.appendInst(entry, u32_t, .{ .iconst = 0 });
    const done = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .eq, .lhs = n, .rhs = zero } });
    const one = try func.appendInst(entry, u32_t, .{ .iconst = 1 });
    try func.appendIf(entry, done, .{ .target = join, .args = &.{one} }, .{ .target = recurse });
    const n1 = try func.appendArithImm(recurse, u32_t, .sub, n, 1);
    const f1 = try func.appendInst(recurse, u32_t, .{ .call = .{ .symbol = try func.internSymbol("f"), .args = try func.internValues(&.{n1}) } });
    const three = try func.appendInst(recurse, u32_t, .{ .iconst = 3 });
    const less = try func.appendInst(recurse, u32_t, .{ .arith = .{ .op = .sub, .lhs = f1, .rhs = three } });
    try func.setJump(recurse, join, &.{less});
    const result = try func.appendBlockParam(join, u32_t);
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(result) });

    try testing.expect(!try run(allocator, &func, "f"));
}
