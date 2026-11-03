//! Loop register promotion of one memory location. In a loop that reads and then writes the same
//! loop-invariant address once per iteration, the read takes the value the previous iteration
//! wrote, carried in a header parameter. The write stays, so memory is correct after the loop.
//!
//! The first value is loaded before the loop, behind a copy of the header test, so the load
//! runs only when the loop body runs. That body always reaches the original load, because the
//! loop exits only from its header and the load's block dominates every back edge. Every other
//! store in the loop must provably miss the location: the same base pointer at known constant
//! offsets.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const loops = @import("loops.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;

pub const pass_def = pass.Pass{ .name = "looppromote", .run = run };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    const info = try analyses.loops();
    const doms = try analyses.dominators();
    for (info.loops) |*loop| {
        const found = try findLocation(allocator, func, loop, doms) orelse continue;
        try promote(allocator, func, loop, found);
        analyses.invalidate();
        return true;
    }
    return false;
}

const Location = struct {
    load: Inst,
    store: Inst,
    ptr: Value,
    ty: Type,
    preheader: Block,
};

/// A pointer as a base value plus a small set of possible constant byte offsets.
const Root = struct {
    base: Value,
    offsets: [max_offsets]i64 = undefined,
    len: usize = 0,
};

const max_offsets = 8;

fn findLocation(allocator: std.mem.Allocator, func: *const Function, loop: *const loops.Loop, doms: anytype) !?Location {
    const header: Block = @fromBackingInt(loop.header);
    const preheader_index = loop.preheader orelse return null;
    const preheader: Block = @fromBackingInt(preheader_index);
    if (lastIf(func, preheader) != null) return null;
    const pre_term = func.terminator(preheader) orelse return null;
    if (pre_term != .jump or pre_term.jump.target != header) return null;
    if (!pureTest(func, header)) return null;

    // The loop exits only from its header, and every back edge is a plain jump.
    var latches: std.ArrayList(u32) = .empty;
    defer latches.deinit(allocator);
    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        const block: Block = @fromBackingInt(@intCast(bi));
        var targets: [2]Block = undefined;
        for (successors(func, block, &targets)) |succ| {
            if (!loop.contains(@backingInt(succ))) {
                if (block != header) return null;
            } else if (succ == header) {
                if (lastIf(func, block) != null) return null;
                try latches.append(allocator, @intCast(bi));
            }
        }
    }
    if (latches.items.len == 0) return null;

    // Every memory effect in the loop is a plain load or store.
    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| switch (func.opcode(inst)) {
            .call, .call_indirect, .atomic_rmw, .barrier, .matmul, .va_start, .va_arg, .va_end => return null,
            .load => |l| if (l.@"volatile" or l.mask != null) return null,
            .store => |s| if (s.@"volatile" or s.mask != null) return null,
            else => {},
        };
    }

    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block), 0..) |inst, index| {
            const load = switch (func.opcode(inst)) {
                .load => |l| l,
                else => continue,
            };
            if (definedIn(func, loop, load.ptr)) continue;
            const ty = func.valueType(func.instResult(inst).?);
            if (!scalarInt(func, ty)) continue;
            // The block runs on every iteration that returns to the header.
            for (latches.items) |latch| if (!doms.dominates(bi, latch)) break else {
                if (try matchStore(func, loop, block, index, inst, load.ptr, ty)) |store| {
                    return .{ .load = inst, .store = store, .ptr = load.ptr, .ty = ty, .preheader = preheader };
                }
            };
        }
    }
    return null;
}

/// The single store to `ptr` after the load in the same block, when it is the only other access
/// to that location in the loop and every other store misses it.
fn matchStore(func: *const Function, loop: *const loops.Loop, block: Block, load_index: usize, load_inst: Inst, ptr: Value, ty: Type) !?Inst {
    var store: ?Inst = null;
    for (func.blockInsts(block)[load_index + 1 ..]) |inst| switch (func.opcode(inst)) {
        .store => |s| if (s.ptr == ptr) {
            if (store != null) return null;
            if (func.valueType(s.value) != ty) return null;
            store = inst;
        },
        else => {},
    };
    const the_store = store orelse return null;
    const here = rootOf(func, ptr) orelse return null;
    const size = byteSize(func, ty);
    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            if (inst == the_store or inst == load_inst) continue;
            switch (func.opcode(inst)) {
                .load => |l| if (l.ptr == ptr) return null,
                .store => |s| {
                    if (s.ptr == ptr) return null;
                    const other = rootOf(func, s.ptr) orelse return null;
                    if (!disjoint(here, size, other, byteSize(func, func.valueType(s.value)))) return null;
                },
                else => {},
            }
        }
    }
    return the_store;
}

fn promote(allocator: std.mem.Allocator, func: *Function, loop: *const loops.Loop, found: Location) !void {
    _ = allocator;
    const header: Block = @fromBackingInt(loop.header);
    const entry_args = try func.allocator.dupe(Value, func.blockArgs(func.terminator(found.preheader).?.jump));
    defer func.allocator.free(entry_args);
    const header_if = lastIf(func, header).?;
    const cf = func.opcode(header_if).@"if";
    const stays_then = loop.contains(@backingInt(cf.then.target));

    // The preheader repeats the header test on the entry values.
    var map: std.AutoHashMapUnmanaged(Value, Value) = .empty;
    defer map.deinit(func.allocator);
    for (func.blockParams(header), entry_args) |param, arg| try map.put(func.allocator, param, arg);
    const header_insts = try func.allocator.dupe(Inst, func.blockInsts(header));
    defer func.allocator.free(header_insts);
    for (header_insts) |inst| {
        if (inst == header_if) break;
        const result = func.instResult(inst).?;
        const copy = try func.appendInst(found.preheader, func.valueType(result), remap(func.opcode(inst), &map));
        try map.put(func.allocator, result, copy);
    }
    const enters = map.get(cf.cond) orelse cf.cond;

    const first = try func.appendBlock();
    const value = try func.appendBlockParam(header, found.ty);
    const unused = try func.appendInst(found.preheader, found.ty, .{ .iconst = 0 });
    const with_unused = try appendArg(func, entry_args, unused);
    defer func.allocator.free(with_unused);
    if (stays_then)
        try func.appendIf(found.preheader, enters, .{ .target = first }, .{ .target = header, .args = with_unused })
    else
        try func.appendIf(found.preheader, enters, .{ .target = header, .args = with_unused }, .{ .target = first });
    func.setTerminator(found.preheader, .{ .ret = ir.function.Ret.none() });
    const initial = try func.appendInst(first, found.ty, .{ .load = .{ .ptr = found.ptr } });
    const with_initial = try appendArg(func, entry_args, initial);
    defer func.allocator.free(with_initial);
    try func.setJump(first, header, with_initial);

    // Each back edge carries the value its iteration stored. An edge from a dead block passes
    // the unused constant, so every edge keeps the header's argument count.
    const stored = func.opcode(found.store).store.value;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        if (block == found.preheader or block == first) continue;
        const extra = if (bi < loop.body.len and loop.contains(bi)) stored else unused;
        if (lastIf(func, block)) |inst| {
            const edges = func.opcode(inst).@"if";
            for ([_]bool{ true, false }) |is_then| {
                const edge = if (is_then) edges.then else edges.@"else";
                if (edge.target != header) continue;
                const args = try appendArg(func, func.blockArgs(edge), extra);
                defer func.allocator.free(args);
                const list = try func.internValues(args);
                const cf_mut = &func.opcodeMut(inst).@"if";
                if (is_then) cf_mut.then.args = list else cf_mut.@"else".args = list;
            }
            continue;
        }
        const term = func.terminator(block) orelse continue;
        if (term != .jump or term.jump.target != header) continue;
        const args = try appendArg(func, func.blockArgs(term.jump), extra);
        defer func.allocator.free(args);
        try func.setJump(block, header, args);
    }

    // The load reads the carried value.
    const loaded = func.instResult(found.load).?;
    const replacements = try func.allocator.alloc(Value, func.valueCount());
    defer func.allocator.free(replacements);
    for (replacements, 0..) |*r, index| r.* = @fromBackingInt(@intCast(index));
    replacements[@backingInt(loaded)] = value;
    func.rewriteAllUses(replacements);
    func.opcodeMut(found.load).* = .{ .iconst = 0 };
}

fn appendArg(func: *Function, args: []const Value, extra: Value) ![]Value {
    const out = try func.allocator.alloc(Value, args.len + 1);
    @memcpy(out[0..args.len], args);
    out[args.len] = extra;
    return out;
}

fn remap(op: ir.function.Opcode, map: *const std.AutoHashMapUnmanaged(Value, Value)) ir.function.Opcode {
    const m = struct {
        fn v(mp: *const std.AutoHashMapUnmanaged(Value, Value), x: Value) Value {
            return mp.get(x) orelse x;
        }
    }.v;
    return switch (op) {
        .iconst, .fconst, .fconst128 => op,
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = m(map, a.lhs), .rhs = m(map, a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = m(map, a.lhs), .imm = a.imm } },
        .icmp => |c| .{ .icmp = .{ .op = c.op, .lhs = m(map, c.lhs), .rhs = m(map, c.rhs) } },
        .convert => |c| .{ .convert = .{ .value = m(map, c.value) } },
        .unary => |u| .{ .unary = .{ .op = u.op, .value = m(map, u.value) } },
        .select => |s| .{ .select = .{ .cond = m(map, s.cond), .then = m(map, s.then), .@"else" = m(map, s.@"else") } },
        else => unreachable,
    };
}

/// A header that holds only pure computations and one closing `if`.
fn pureTest(func: *const Function, header: Block) bool {
    const insts = func.blockInsts(header);
    if (insts.len == 0 or func.opcode(insts[insts.len - 1]) != .@"if") return false;
    for (insts[0 .. insts.len - 1]) |inst| switch (func.opcode(inst)) {
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .convert, .unary, .select => {},
        else => return false,
    };
    // A division can trap, so the copy before the loop must not run one the loop would skip.
    for (insts[0 .. insts.len - 1]) |inst| switch (func.opcode(inst)) {
        .arith => |a| if (a.op == .div or a.op == .rem) return false,
        .arith_imm => |a| if (a.op == .div or a.op == .rem) return false,
        else => {},
    };
    return true;
}

fn lastIf(func: *const Function, block: Block) ?Inst {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const last = insts[insts.len - 1];
    return if (func.opcode(last) == .@"if") last else null;
}

fn successors(func: *const Function, block: Block, out: *[2]Block) []const Block {
    if (lastIf(func, block)) |inst| {
        const cf = func.opcode(inst).@"if";
        out[0] = cf.then.target;
        out[1] = cf.@"else".target;
        return out[0..2];
    }
    const term = func.terminator(block) orelse return out[0..0];
    return switch (term) {
        .jump => |j| blk: {
            out[0] = j.target;
            break :blk out[0..1];
        },
        .ret => out[0..0],
    };
}

fn definedIn(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    for (0..func.blockCount()) |bi| {
        if (!loop.contains(bi)) continue;
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block)) |p| if (p == v) return true;
        for (func.blockInsts(block)) |inst| if (func.instResult(inst) == v) return true;
    }
    return false;
}

fn scalarInt(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |int| int.bits % 8 == 0 and int.bits <= 64,
        .ptr => true,
        else => false,
    };
}

fn byteSize(func: *const Function, ty: Type) i64 {
    return switch (func.types.type_kind(ty)) {
        .int => |int| @divFloor(@as(i64, int.bits) + 7, 8),
        .float => |f| switch (f) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .f128 => 16,
        },
        .ptr => 8,
        .vector => |v| @as(i64, v.len) * byteSize(func, v.elem),
        else => std.math.maxInt(i32),
    };
}

fn disjoint(a: Root, a_size: i64, b: Root, b_size: i64) bool {
    if (a.base != b.base) return false;
    for (a.offsets[0..a.len]) |ao| for (b.offsets[0..b.len]) |bo| {
        if (ao < bo + b_size and bo < ao + a_size) return false;
    };
    return true;
}

fn rootOf(func: *const Function, ptr: Value) ?Root {
    const def = func.definingInst(ptr) orelse return single(ptr);
    switch (func.opcode(def)) {
        .arith_imm => |a| if (a.op == .add) {
            var r = rootOf(func, a.lhs) orelse return null;
            for (r.offsets[0..r.len]) |*o| o.* = std.math.add(i64, o.*, a.imm) catch return null;
            return r;
        },
        .arith => |a| if (a.op == .add) {
            const lhs_ptr = func.types.type_kind(func.valueType(a.lhs)) == .ptr;
            const base_side = if (lhs_ptr) a.lhs else a.rhs;
            const index_side = if (lhs_ptr) a.rhs else a.lhs;
            const r = rootOf(func, base_side) orelse return null;
            var set: IntSet = .{};
            if (!intSet(func, index_side, &set, 6)) return null;
            var out = Root{ .base = r.base };
            for (r.offsets[0..r.len]) |o| for (set.values[0..set.len]) |s| {
                if (out.len == max_offsets) return null;
                out.offsets[out.len] = std.math.add(i64, o, s) catch return null;
                out.len += 1;
            };
            return out;
        },
        else => {},
    }
    return single(ptr);
}

const IntSet = struct {
    values: [max_offsets]i64 = undefined,
    len: usize = 0,

    fn add(self: *IntSet, v: i64) bool {
        for (self.values[0..self.len]) |existing| if (existing == v) return true;
        if (self.len == max_offsets) return false;
        self.values[self.len] = v;
        self.len += 1;
        return true;
    }
};

/// The possible values of an integer as a small set: a constant, a block parameter fed only
/// constants, or a shift, multiply, add or widening of one of those.
fn intSet(func: *const Function, v: Value, out: *IntSet, depth: u8) bool {
    if (depth == 0) return false;
    if (func.definingInst(v)) |def| return switch (func.opcode(def)) {
        .iconst => |c| out.add(c),
        .convert => |c| blk: {
            const from = switch (func.types.type_kind(func.valueType(c.value))) {
                .int => |int| int,
                else => break :blk false,
            };
            const to = switch (func.types.type_kind(func.valueType(v))) {
                .int => |int| int,
                else => break :blk false,
            };
            if (to.bits < from.bits) break :blk false;
            var inner: IntSet = .{};
            if (!intSet(func, c.value, &inner, depth - 1)) break :blk false;
            for (inner.values[0..inner.len]) |x| {
                // A value outside the source range would mean the set is not exact.
                if (from.signedness == .unsigned and x < 0) break :blk false;
                if (!out.add(x)) break :blk false;
            }
            break :blk true;
        },
        .arith_imm => |a| blk: {
            var inner: IntSet = .{};
            if (!intSet(func, a.lhs, &inner, depth - 1)) break :blk false;
            for (inner.values[0..inner.len]) |x| {
                const y = switch (a.op) {
                    .add => std.math.add(i64, x, a.imm) catch break :blk false,
                    .mul => std.math.mul(i64, x, a.imm) catch break :blk false,
                    .shl => if (a.imm >= 0 and a.imm < 32) std.math.shlExact(i64, x, @intCast(a.imm)) catch break :blk false else break :blk false,
                    else => break :blk false,
                };
                if (!out.add(y)) break :blk false;
            }
            break :blk true;
        },
        else => false,
    };
    // A block parameter: every incoming argument must be in a known set. A function argument is
    // unknown, and a parameter of a block nothing jumps to never holds a value.
    const param = paramPosition(func, v) orelse return false;
    if (@backingInt(param.block) == 0) return false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        if (lastIf(func, block)) |inst| {
            const cf = func.opcode(inst).@"if";
            for ([_]ir.function.Jump{ cf.then, cf.@"else" }) |edge| if (edge.target == param.block) {
                if (!constArg(func, func.blockArgs(edge), param.index, out, depth - 1)) return false;
            };
        } else if (func.terminator(block)) |term| switch (term) {
            .jump => |j| if (j.target == param.block) {
                if (!constArg(func, func.blockArgs(j), param.index, out, depth - 1)) return false;
            },
            .ret => {},
        };
    }
    return true;
}


fn paramPosition(func: *const Function, v: Value) ?struct { block: Block, index: u32 } {
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block), 0..) |p, index| if (p == v) return .{ .block = block, .index = @intCast(index) };
    }
    return null;
}

const testing = std.testing;

/// `while (i < n) { kind = i odd ? 1 : 3; c[kind] += 1; c->hash = c->hash * 31 + kind; ++i; }`
/// over `u32 c[8]`, with the hash at byte 24. `unknown_index` makes the counter store use a
/// loaded index instead, which may reach the hash.
fn buildCounter(func: *Function, unknown_index: bool) !void {
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const header = try func.appendBlock();
    const odd = try func.appendBlock();
    const even = try func.appendBlock();
    const latch = try func.appendBlock();
    const done = try func.appendBlock();
    const c = try func.appendBlockParam(entry, ptr_t);
    const n = try func.appendBlockParam(entry, u64_t);
    const zero = try func.appendInst(entry, u64_t, .{ .iconst = 0 });
    const hash_at = try func.appendArithImm(entry, ptr_t, .add, c, 24);
    try func.setJump(entry, header, &.{zero});
    const i = try func.appendBlockParam(header, u64_t);
    const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(header, more, .{ .target = odd }, .{ .target = done });
    func.setTerminator(header, .{ .ret = ir.function.Ret.none() });
    const low = try func.appendArithImm(odd, u64_t, .bit_and, i, 1);
    const one = try func.appendInst(odd, u64_t, .{ .iconst = 1 });
    const is_odd = try func.appendInst(odd, bool_t, .{ .icmp = .{ .op = .eq, .lhs = low, .rhs = one } });
    const k1 = try func.appendInst(odd, u32_t, .{ .iconst = 1 });
    try func.appendIf(odd, is_odd, .{ .target = latch, .args = &.{k1} }, .{ .target = even });
    func.setTerminator(odd, .{ .ret = ir.function.Ret.none() });
    const k3 = try func.appendInst(even, u32_t, .{ .iconst = 3 });
    try func.setJump(even, latch, &.{k3});
    const kind = try func.appendBlockParam(latch, u32_t);
    const index = if (unknown_index)
        try func.appendInst(latch, u32_t, .{ .load = .{ .ptr = c } })
    else
        kind;
    const wide = try func.appendInst(latch, i64_t, .{ .convert = .{ .value = index } });
    const offset = try func.appendArithImm(latch, i64_t, .shl, wide, 2);
    const slot = try func.appendInst(latch, ptr_t, .{ .arith = .{ .op = .add, .lhs = c, .rhs = offset } });
    const count = try func.appendInst(latch, u32_t, .{ .load = .{ .ptr = slot } });
    const bumped = try func.appendArithImm(latch, u32_t, .add, count, 1);
    try func.appendStore(latch, bumped, slot);
    const hash = try func.appendInst(latch, u32_t, .{ .load = .{ .ptr = hash_at } });
    const scaled = try func.appendArithImm(latch, u32_t, .mul, hash, 31);
    const mixed = try func.appendInst(latch, u32_t, .{ .arith = .{ .op = .add, .lhs = scaled, .rhs = kind } });
    try func.appendStore(latch, mixed, hash_at);
    const next = try func.appendArithImm(latch, u64_t, .add, i, 1);
    try func.setJump(latch, header, &.{next});
    func.setTerminator(done, .{ .ret = ir.function.Ret.none() });
}

test "a hash read and written once per iteration rides in a header parameter" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildCounter(&func, false);
    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(try run(allocator, &func, &analyses));
    try testing.expectEqual(@as(usize, 2), func.blockParams(@fromBackingInt(1)).len);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
    // Only the counter load stays in the loop, and the hash store stays for memory.
    var loads: usize = 0;
    var stores: usize = 0;
    for (func.blockInsts(@fromBackingInt(4))) |inst| switch (func.opcode(inst)) {
        .load => loads += 1,
        .store => stores += 1,
        else => {},
    };
    try testing.expectEqual(@as(usize, 1), loads);
    try testing.expectEqual(@as(usize, 2), stores);
}

test "a store at an unknown offset keeps the hash in memory" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildCounter(&func, true);
    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try testing.expect(!try run(allocator, &func, &analyses));
}

fn single(base: Value) Root {
    var root = Root{ .base = base };
    root.offsets[0] = 0;
    root.len = 1;
    return root;
}

fn constArg(func: *const Function, args: []const Value, index: u32, out: *IntSet, depth: u8) bool {
    if (index >= args.len) return false;
    return intSet(func, args[index], out, depth);
}
