//! Same-width integer converts keep the bits and change only the signedness. A use that does not
//! read the signedness can take either side. This pass moves such uses to one side, so the other
//! side dies at the convert and the allocator can turn the convert into no code.
//!
//! A sign-blind use is an `eq` or `ne` compare against a small constant, the integer offset of a
//! pointer add, or the value of a store.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const dce = @import("dce.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const pass_def = pass.Pass{ .name = "samewidth", .run = run };

const Use = struct { inst: Inst, block: u32, index: u32 };

const Candidate = struct { convert: Inst, block: u32, index: u32, source: Value, result: Value };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    var candidates: std.ArrayList(Candidate) = .empty;
    defer candidates.deinit(allocator);
    const consts = try allocator.alloc(?i64, func.valueCount());
    defer allocator.free(consts);
    @memset(consts, null);
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi))), 0..) |inst, index| switch (func.opcode(inst)) {
            .iconst => |c| consts[@backingInt(func.instResult(inst).?)] = c,
            .convert => |c| {
                const result = func.instResult(inst).?;
                if (!sameWidthInts(func, c.value, result)) continue;
                try candidates.append(allocator, .{ .convert = inst, .block = @intCast(bi), .index = @intCast(index), .source = c.value, .result = result });
            },
            else => {},
        };
    }
    if (candidates.items.len == 0) return false;

    const tracked = try allocator.alloc(bool, func.valueCount());
    defer allocator.free(tracked);
    @memset(tracked, false);
    for (candidates.items) |c| {
        tracked[@backingInt(c.source)] = true;
        tracked[@backingInt(c.result)] = true;
    }
    var uses: std.AutoHashMapUnmanaged(Value, std.ArrayList(Use)) = .empty;
    defer {
        var it = uses.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        uses.deinit(allocator);
    }
    for (0..func.blockCount()) |bi| {
        for (func.blockInsts(@fromBackingInt(@intCast(bi))), 0..) |inst, index| {
            var operands: [3]Value = undefined;
            for (operandsOf(func, inst, &operands)) |v| {
                if (!tracked[@backingInt(v)]) continue;
                const entry = try uses.getOrPut(allocator, v);
                if (!entry.found_existing) entry.value_ptr.* = .empty;
                try entry.value_ptr.append(allocator, .{ .inst = inst, .block = @intCast(bi), .index = @intCast(index) });
            }
        }
    }
    // A value read anywhere this pass does not list keeps its uses where they are.
    const counts = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(counts);
    dce.countUses(func, counts);

    const doms = try analyses.dominators();
    var made: std.AutoHashMapUnmanaged(struct { Type, i64 }, Value) = .empty;
    defer made.deinit(allocator);
    // A rewrite makes the use lists of both its values stale, so each value moves once per run.
    var touched: std.AutoHashMapUnmanaged(Value, void) = .empty;
    defer touched.deinit(allocator);
    const entry_block: Block = @fromBackingInt(0);
    var changed = false;
    for (candidates.items) |c| {
        if (touched.contains(c.result) or touched.contains(c.source)) continue;
        const result_uses = if (uses.get(c.result)) |list| list.items else &.{};
        const source_uses = if (uses.get(c.source)) |list| list.items else &.{};
        // Move every use of the result onto the source, so the convert is dead.
        if (result_uses.len != 0 and counts[@backingInt(c.result)] == result_uses.len and
            allSignBlind(func, result_uses, consts, c.result, c.source, null, doms, c))
        {
            for (result_uses) |use| try rewrite(allocator, func, use.inst, c.result, c.source, consts, &made, entry_block);
        } else if (source_uses.len > 1 and counts[@backingInt(c.source)] == source_uses.len and
            allSignBlind(func, source_uses, consts, c.source, c.result, c.convert, doms, c))
        {
            // Or move the other uses of the source onto the result, so the source dies at the convert.
            for (source_uses) |use| {
                if (use.inst == c.convert) continue;
                try rewrite(allocator, func, use.inst, c.source, c.result, consts, &made, entry_block);
            }
        } else continue;
        try touched.put(allocator, c.result, {});
        try touched.put(allocator, c.source, {});
        changed = true;
    }
    if (changed) analyses.invalidate();
    return changed;
}

fn sameWidthInts(func: *const Function, a: Value, b: Value) bool {
    const from = switch (func.types.type_kind(func.valueType(a))) {
        .int => |int| int,
        else => return false,
    };
    const to = switch (func.types.type_kind(func.valueType(b))) {
        .int => |int| int,
        else => return false,
    };
    return from.bits == to.bits;
}

/// The value operands of the instructions this pass can read or rewrite.
fn operandsOf(func: *const Function, inst: Inst, out: *[3]Value) []const Value {
    return switch (func.opcode(inst)) {
        .icmp => |c| blk: {
            out[0] = c.lhs;
            out[1] = c.rhs;
            break :blk out[0..2];
        },
        .arith => |a| blk: {
            out[0] = a.lhs;
            out[1] = a.rhs;
            break :blk out[0..2];
        },
        .store => |s| blk: {
            out[0] = s.value;
            out[1] = s.ptr;
            if (s.mask) |m| {
                out[2] = m;
                break :blk out[0..3];
            }
            break :blk out[0..2];
        },
        .convert => |c| blk: {
            out[0] = c.value;
            break :blk out[0..1];
        },
        else => out[0..0],
    };
}

/// Whether every listed use of `from`, other than `skip`, reads no signedness and may read `to`
/// instead. `to` must be available at each use: the source always is, and the convert result is
/// only after the convert.
fn allSignBlind(func: *const Function, list: []const Use, consts: []const ?i64, from: Value, to: Value, skip: ?Inst, doms: anytype, c: Candidate) bool {
    for (list) |use| {
        if (skip != null and use.inst == skip.?) continue;
        if (!signBlind(func, use.inst, from, consts)) return false;
        if (to == c.result) {
            const after = if (use.block == c.block) use.index > c.index else doms.strictlyDominates(c.block, use.block);
            if (!after) return false;
        }
    }
    return true;
}

fn signBlind(func: *const Function, inst: Inst, v: Value, consts: []const ?i64) bool {
    return switch (func.opcode(inst)) {
        .icmp => |c| blk: {
            if (c.op != .eq and c.op != .ne) break :blk false;
            const other = if (c.lhs == v) c.rhs else c.lhs;
            if (other == v) break :blk false;
            const k = consts[@backingInt(other)] orelse break :blk false;
            break :blk fitsBoth(func, v, k);
        },
        .arith => |a| a.op == .add and func.types.type_kind(func.valueType(func.instResult(inst).?)) == .ptr and
            (a.lhs == v) != (a.rhs == v),
        .store => |s| s.value == v and s.ptr != v and s.mask != v,
        else => false,
    };
}

/// A constant that keeps one meaning under both signednesses of `v`'s width.
fn fitsBoth(func: *const Function, v: Value, k: i64) bool {
    const bits = func.types.type_kind(func.valueType(v)).int.bits;
    if (k < 0) return false;
    if (bits >= 64) return true;
    return k < (@as(i64, 1) << @intCast(bits - 1));
}

fn rewrite(allocator: std.mem.Allocator, func: *Function, inst: Inst, from: Value, to: Value, consts: []const ?i64, made: anytype, entry_block: Block) !void {
    const op = func.opcodeMut(inst);
    switch (op.*) {
        .icmp => |*c| {
            const from_is_lhs = c.lhs == from;
            const other = if (from_is_lhs) c.rhs else c.lhs;
            const k = consts[@backingInt(other)].?;
            const ty = func.valueType(to);
            const entry = try made.getOrPut(allocator, .{ ty, k });
            if (!entry.found_existing) entry.value_ptr.* = try func.appendInst(entry_block, ty, .{ .iconst = k });
            // `appendInst` may move the opcode storage, so the compare is read again.
            const cmp = &func.opcodeMut(inst).icmp;
            if (from_is_lhs) {
                cmp.lhs = to;
                cmp.rhs = entry.value_ptr.*;
            } else {
                cmp.rhs = to;
                cmp.lhs = entry.value_ptr.*;
            }
        },
        .arith => |*a| {
            if (a.lhs == from) a.lhs = to else a.rhs = to;
        },
        .store => |*s| s.value = to,
        else => unreachable,
    }
}

const testing = std.testing;

test "a pointer offset reads the index before its same-width convert" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const index = try func.appendBlockParam(b, u64_t);
    const signed = try func.appendInst(b, i64_t, .{ .convert = .{ .value = index } });
    const address = try func.appendInst(b, ptr_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = signed } });
    const byte = try func.appendInst(b, u8_t, .{ .load = .{ .ptr = address } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(byte) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    try testing.expectEqual(index, func.opcode(func.definingInst(address).?).arith.rhs);
}

test "eq compares of a widened byte move to its same-width alias, an order compare does not" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const then_b = try func.appendBlock();
    const else_b = try func.appendBlock();
    const x = try func.appendBlockParam(b, i32_t);
    const space = try func.appendInst(b, i32_t, .{ .iconst = 32 });
    const alias = try func.appendInst(b, u32_t, .{ .convert = .{ .value = x } });
    const lowered = try func.appendArithImm(b, u32_t, .sub, alias, 97);
    const is_space = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = space } });
    try func.appendIf(b, is_space, .{ .target = then_b }, .{ .target = else_b });
    func.setTerminator(then_b, .{ .ret = ir.function.Ret.one(lowered) });
    const below = try func.appendInst(else_b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = space } });
    const below_int = try func.appendInst(else_b, u32_t, .{ .convert = .{ .value = below } });
    func.setTerminator(else_b, .{ .ret = ir.function.Ret.one(below_int) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    // The signed order compare reads `x` as signed, so `x` keeps its other uses.
    try testing.expect(!try run(allocator, &func, &analyses));

    func.opcodeMut(func.definingInst(below).?).icmp.op = .ne;
    var fresh = pass.Analyses{ .allocator = allocator, .func = &func };
    defer fresh.deinit();
    try testing.expect(try run(allocator, &func, &fresh));
    const cmp = func.opcode(func.definingInst(is_space).?).icmp;
    try testing.expectEqual(alias, cmp.lhs);
    try testing.expectEqual(u32_t, func.valueType(cmp.rhs));
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
}
