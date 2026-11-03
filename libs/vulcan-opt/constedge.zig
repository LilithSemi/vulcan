//! Constants passed on an edge are rebuilt in the block that passes them. An edge move of a
//! constant then writes an immediate, which costs the same as a register copy, and the constant
//! no longer holds a register across the whole function. This matters for a loop that picks one
//! of several constants per path, such as a token kind.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "constedge", .run = run, .preserves_cfg = true };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const def_block = try allocator.alloc(?u32, func.valueCount());
    defer allocator.free(def_block);
    @memset(def_block, null);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            if (func.opcode(inst) != .iconst) continue;
            const result = func.instResult(inst).?;
            if (func.types.type_kind(func.valueType(result)) != .int) continue;
            def_block[@backingInt(result)] = @intCast(bi);
        }
    }

    var changed = false;
    var copies: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer copies.deinit(allocator);
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        copies.clearRetainingCapacity();
        const insts = func.blockInsts(block);
        const last_if: ?Inst = if (insts.len != 0 and func.opcode(insts[insts.len - 1]) == .@"if") insts[insts.len - 1] else null;
        var made: std.ArrayList(Inst) = .empty;
        defer made.deinit(allocator);
        if (last_if) |inst| {
            const cf = func.opcode(inst).@"if";
            const then_args = try localArgs(allocator, func, def_block, @intCast(bi), func.blockArgs(cf.then), &copies, &made);
            defer allocator.free(then_args);
            const else_args = try localArgs(allocator, func, def_block, @intCast(bi), func.blockArgs(cf.@"else"), &copies, &made);
            defer allocator.free(else_args);
            if (made.items.len == 0) continue;
            const then_list = try func.internValues(then_args);
            const else_list = try func.internValues(else_args);
            const cf_mut = &func.opcodeMut(inst).@"if";
            cf_mut.then.args = then_list;
            cf_mut.@"else".args = else_list;
            // The copies go right before the `if`.
            const list = func.blockInstsMut(block);
            try list.insertSlice(func.allocator, list.items.len - 1, made.items);
        } else if (func.terminator(block)) |term| switch (term) {
            .jump => |j| {
                const args = try localArgs(allocator, func, def_block, @intCast(bi), func.blockArgs(j), &copies, &made);
                defer allocator.free(args);
                if (made.items.len == 0) continue;
                try func.blockInstsMut(block).appendSlice(func.allocator, made.items);
                try func.setJump(block, j.target, args);
            },
            .ret => continue,
        } else continue;
        changed = true;
    }
    return changed;
}

/// `args` with each constant defined in another block replaced by a copy made for this block.
fn localArgs(allocator: std.mem.Allocator, func: *Function, def_block: []const ?u32, block: u32, args: []const Value, copies: *std.AutoHashMapUnmanaged(Value, Value), made: *std.ArrayList(Inst)) ![]Value {
    const out = try allocator.dupe(Value, args);
    for (out) |*arg| {
        const index = @backingInt(arg.*);
        if (index >= def_block.len) continue;
        const home = def_block[index] orelse continue;
        if (home == block) continue;
        const entry = try copies.getOrPut(allocator, arg.*);
        if (!entry.found_existing) {
            const value = func.opcode(func.definingInst(arg.*).?).iconst;
            const copy = try func.createInst(func.valueType(arg.*), .{ .iconst = value });
            try made.append(allocator, func.definingInst(copy).?);
            entry.value_ptr.* = copy;
        }
        arg.* = entry.value_ptr.*;
    }
    return out;
}

const testing = std.testing;

test "a constant passed on a loop edge is rebuilt in the passing block" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const pick = try func.appendBlock();
    const join = try func.appendBlock();
    const flag = try func.appendBlockParam(entry, bool_t);
    const two = try func.appendInst(entry, i32_t, .{ .iconst = 2 });
    const three = try func.appendInst(entry, i32_t, .{ .iconst = 3 });
    try func.setJump(entry, pick, &.{});
    try func.appendIf(pick, flag, .{ .target = join, .args = &.{two} }, .{ .target = join, .args = &.{three} });
    func.setTerminator(pick, .{ .ret = ir.function.Ret.none() });
    const kind = try func.appendBlockParam(join, i32_t);
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(kind) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    const cf = func.opcode(func.blockInsts(pick)[func.blockInsts(pick).len - 1]).@"if";
    for ([_]ir.function.Jump{ cf.then, cf.@"else" }) |edge| {
        const arg = func.blockArgs(edge)[0];
        try testing.expect(arg != two and arg != three);
        try testing.expect(std.mem.indexOfScalar(Inst, func.blockInsts(pick), func.definingInst(arg).?) != null);
    }
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}
