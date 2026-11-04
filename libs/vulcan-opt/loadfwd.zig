//! Redundant-load elimination: within a block, forward the value of a store to a later load of the
//! same address, and reuse an earlier load of an address for a later one, when no aliasing store (or
//! call) sits between them. This is the in-memory analogue of mem2reg: mem2reg promotes the slots
//! whose address never escapes, this handles the memory that stays in memory (escaped locals, heap,
//! incoming pointers). A light alias oracle keeps it cheap and correct: two distinct allocas never
//! alias, two distinct globals never alias, and an alloca and a global never alias. Constant-offset
//! accesses from the same root also cannot alias when their known primitive byte ranges are
//! disjoint; unresolved roots and widths remain conservative. A call, a matmul, a `va_list`
//! operation or a barrier may touch any memory, so each of them clears everything. An atomic is a
//! store of its address, and an atomic ordered above `relaxed` is a barrier as well.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Inst = ir.function.Inst;

pub const pass_def = pass.Pass{ .name = "loadfwd", .run = run, .preserves_cfg = true };

/// One address with an available value: `ptr` currently holds `value` (from a store or a prior
/// load), valid until an aliasing store or a barrier invalidates it.
const Avail = struct { ptr: Value, value: Value };

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    var changed = false;
    var avail: std.ArrayList(Avail) = .empty;
    defer avail.deinit(allocator);
    var keep: std.ArrayList(Inst) = .empty;
    defer keep.deinit(allocator);
    // Addresses every predecessor holds, each with its own value. A load of one takes a new
    // block param that each predecessor fills on its edge.
    var merges: std.ArrayList(Avail) = .empty;
    defer merges.deinit(allocator);

    // A block starts with the entries available at the end of every predecessor. A value that is
    // available in all of them is defined above all of them, so it dominates the block. Reverse
    // postorder visits the predecessors first, except along a back edge, which gives nothing.
    const graph = try analyses.cfg();
    const rpo = try analyses.reversePostorder();
    const ends = try allocator.alloc(?[]const Avail, func.blockCount());
    defer {
        for (ends) |end| if (end) |items| allocator.free(items);
        allocator.free(ends);
    }
    @memset(ends, null);
    const order = try allocator.alloc(u32, func.blockCount());
    defer allocator.free(order);
    const reached = try allocator.alloc(bool, func.blockCount());
    defer allocator.free(reached);
    @memset(reached, false);
    for (rpo, 0..) |b, k| {
        order[k] = b;
        reached[b] = true;
    }
    var next = rpo.len;
    for (reached, 0..) |r, b| if (!r) {
        order[next] = @intCast(b);
        next += 1;
    };

    // A loop that writes no memory leaves every value it was entered with in place, so its header
    // takes what its entry edges hold and skips its back edges.
    const entry_preds = try allocator.alloc(?[]u32, func.blockCount());
    defer {
        for (entry_preds) |list| if (list) |l| allocator.free(l);
        allocator.free(entry_preds);
    }
    @memset(entry_preds, null);
    for ((try analyses.loops()).loops) |*loop| {
        if (writesMemory(func, loop)) continue;
        var list: std.ArrayList(u32) = .empty;
        errdefer list.deinit(allocator);
        for (graph.predecessors(loop.header)) |p| if (reached[p] and !loop.contains(p)) try list.append(allocator, p);
        if (entry_preds[loop.header]) |old| allocator.free(old);
        entry_preds[loop.header] = try list.toOwnedSlice(allocator);
    }

    for (order) |bi| {
        avail.clearRetainingCapacity();
        keep.clearRetainingCapacity();
        merges.clearRetainingCapacity();
        var preds: []const u32 = &.{};
        if (bi < graph.blockCount()) {
            preds = entry_preds[bi] orelse graph.predecessors(bi);
            const all_seen = for (preds) |p| {
                if (p == bi or ends[p] == null) break false;
            } else preds.len != 0;
            if (all_seen) {
                for (ends[preds[0]].?) |entry| {
                    const everywhere = for (preds[1..]) |p| {
                        if (!holds(ends[p].?, entry)) break false;
                    } else true;
                    if (everywhere) {
                        try avail.append(allocator, entry);
                        continue;
                    }
                    const ty = func.valueType(entry.value);
                    const anywhere = for (preds[1..]) |p| {
                        if (exact(ends[p].?, func, entry.ptr, ty) == null) break false;
                    } else true;
                    if (anywhere and entry_preds[bi] == null) try merges.append(allocator, entry);
                }
            }
        }
        var block_changed = false;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
            switch (func.opcode(inst)) {
                .load => |ld| {
                    // A `volatile` load (SM9 Plan 2 Task 4) must observably re-read memory every
                    // time: it is never satisfied from a prior availability entry, and it is never
                    // itself recorded as one (so no later load can be forwarded from it either).
                    //
                    // An `endian`-tagged load gets the same treatment, for a different reason (see
                    // `Attribute.endian`). Its result is the bytes in memory REVERSED, so it does
                    // not hold the value a plain store put there, and a later plain load must not
                    // be answered from it either. Forwarding across the tag silently deletes the
                    // byte swap the backend would have emitted.
                    //
                    // A MASKED load is kept out of it in both directions: its result holds the
                    // memory only for the lanes its mask names, so it may not be answered from an
                    // earlier store and no later load may be answered from it.
                    if (!ld.@"volatile" and ld.mask == null and !func.isByteOrderTagged(inst)) {
                        const result = func.instResult(inst).?;
                        if (find(avail.items, func, ld.ptr, func.valueType(result))) |v| {
                            // This pass never reattaches instructions orphaned by an earlier
                            // transform, so avoid scanning that historical instruction storage.
                            func.replaceAllLiveUses(result, v);
                            block_changed = true;
                            continue; // drop the now-redundant load
                        }
                        if (try joinValue(allocator, func, merges.items, ends, preds, @intCast(bi), ld.ptr, func.valueType(result))) |v| {
                            func.replaceAllLiveUses(result, v);
                            try avail.append(allocator, .{ .ptr = ld.ptr, .value = v });
                            block_changed = true;
                            continue;
                        }
                        try avail.append(allocator, .{ .ptr = ld.ptr, .value = result });
                    }
                },
                .store => |st| {
                    // A `volatile` store is still a barrier to aliasing loads (invalidate as usual),
                    // but its value must not be handed out to satisfy a later load - so it is not
                    // recorded as an availability entry. An `endian`-tagged store is barred from
                    // the table for the byte-order reason: it writes its value REVERSED, so what
                    // the address now holds is not `st.value`.
                    //
                    // A MASKED store is kept out of the table for a third reason: the lanes whose
                    // mask bit is clear did not write, so the address does not hold `st.value`
                    // afterwards. It still invalidates, because the lanes that did write changed
                    // what is there.
                    const same = sameValueStore(func, inst, st);
                    invalidateStore(&avail, func, st.ptr, st.value, same);
                    invalidateStore(&merges, func, st.ptr, st.value, same);
                    if (!st.@"volatile" and st.mask == null and !func.isByteOrderTagged(inst)) {
                        try avail.append(allocator, .{ .ptr = st.ptr, .value = st.value });
                    }
                },
                // An atomic is a STORE of `ptr`, so a relaxed one invalidates every entry
                // that may alias it, exactly like the `store` arm above. It is NEVER
                // recorded as an entry, because its result is the OLD value: the address
                // does not hold that value afterwards.
                //
                // Any ordering above `relaxed` also makes memory another thread wrote
                // before it visible after it, which is what a barrier does, so it drops the
                // whole set the way the `barrier` arm below does.
                .atomic_rmw => |a| switch (a.ordering) {
                    .relaxed => {
                        invalidateAliasing(&avail, func, a.ptr, func.valueType(a.value));
                        invalidateAliasing(&merges, func, a.ptr, func.valueType(a.value));
                    },
                    .acquire, .release, .acq_rel, .seq_cst => {
                        avail.clearRetainingCapacity();
                        merges.clearRetainingCapacity();
                    },
                },
                // A call or a matmul may write any memory, so everything recorded is dropped.
                .call, .call_indirect, .matmul => {
                    avail.clearRetainingCapacity();
                    merges.clearRetainingCapacity();
                },
                // A barrier makes memory another thread wrote before it visible after it, so
                // every value this block believes it holds may now be stale. Forwarding a
                // store to a load across a barrier is exactly the bug the barrier exists to
                // prevent, so drop the whole availability set.
                .barrier => {
                    avail.clearRetainingCapacity();
                    merges.clearRetainingCapacity();
                },
                // These three read and mutate the `va_list` object through `list`, so they
                // may write memory a recorded entry names. Drop the set, like a call.
                .va_start, .va_arg, .va_end => {
                    avail.clearRetainingCapacity();
                    merges.clearRetainingCapacity();
                },
                // The rest are pure (no memory effect) and leave availability intact. A
                // prefetch is a hint and writes nothing. The switch is exhaustive with no
                // `else` prong on purpose: an `else` reads a newly added effectful opcode as
                // pure, which is how a barrier gets silently forwarded across.
                .iconst,
                .fconst,
                .fconst128,
                .arith,
                .arith_imm,
                .icmp,
                .select,
                .struct_new,
                .extract,
                .convert,
                .decode_low_float,
                .encode_low_float,
                .dequantize_nvfp4,
                .quantize_nvfp4,
                .unary,
                .alloca,
                .global_addr,
                .prefetch,
                .dot,
                .reduce,
                .splat,
                .@"if",
                => {},
            }
            try keep.append(allocator, inst);
        }
        ends[bi] = try allocator.dupe(Avail, avail.items);
        if (block_changed) {
            try func.setBlockInsts(@fromBackingInt(@intCast(bi)), keep.items);
            changed = true;
        }
    }
    return changed;
}

/// The available value for `ptr`, if some entry holds exactly that address AND was recorded
/// at the same IR type as the load being satisfied. The type check matters because Vulcan
/// pointers are untyped (no pointee type carried at the IR level): the same address can
/// legitimately be stored/loaded at DIFFERENT widths at different points in a function - e.g.
/// a struct field access at offset 0 (its own narrower scalar type) versus a whole-struct
/// blob copy through that same base address (a wider word type covering the whole struct).
/// Forwarding a value of one type to satisfy a load of another would hand a mistyped operand
/// to every use of the load's result, which the IR verifier (rightly) rejects afterward - so
/// a type mismatch here is treated as "not available", not a match, falling through to a
/// fresh load exactly as if no prior access exists.
fn find(items: []const Avail, func: *const Function, ptr: Value, want_ty: ir.types.Type) ?Value {
    for (items) |a| {
        if (sameAddress(func, a.ptr, ptr) and func.valueType(a.value) == want_ty) return a.value;
    }
    return null;
}

/// The value an entry holds for exactly `ptr` at type `ty`.
fn exact(items: []const Avail, func: *const Function, ptr: Value, ty: ir.types.Type) ?Value {
    for (items) |item| {
        if (item.ptr == ptr and func.valueType(item.value) == ty) return item.value;
    }
    return null;
}

/// A new param of `block` for a merged address. Each predecessor passes the value it holds there.
fn joinValue(allocator: std.mem.Allocator, func: *Function, merges: []const Avail, ends: []const ?[]const Avail, preds: []const u32, block: u32, ptr: Value, ty: ir.types.Type) !?Value {
    const merge = for (merges) |m| {
        if (sameAddress(func, m.ptr, ptr) and func.valueType(m.value) == ty) break m;
    } else return null;
    const param = try func.appendBlockParam(@fromBackingInt(block), ty);
    for (preds, 0..) |p, k| {
        if (std.mem.indexOfScalar(u32, preds[0..k], p) != null) continue;
        const value = exact(ends[p].?, func, merge.ptr, ty).?;
        try passOnEdges(allocator, func, @fromBackingInt(p), @fromBackingInt(block), value);
    }
    return param;
}

/// Append `value` to the args of every edge from `from` to `to`.
fn passOnEdges(allocator: std.mem.Allocator, func: *Function, from: Block, to: Block, value: Value) !void {
    for (func.blockInsts(from)) |inst| {
        if (func.opcode(inst) != .@"if") continue;
        const cf = func.opcode(inst).@"if";
        const then_args = if (cf.then.target == to) try withArg(allocator, func, func.blockArgs(cf.then), value) else null;
        const else_args = if (cf.@"else".target == to) try withArg(allocator, func, func.blockArgs(cf.@"else"), value) else null;
        const cf_mut = &func.opcodeMut(inst).@"if";
        if (then_args) |args| cf_mut.then.args = args;
        if (else_args) |args| cf_mut.@"else".args = args;
    }
    if (func.terminator(from)) |term| switch (term) {
        .jump => |j| if (j.target == to) {
            const args = try allocator.alloc(Value, func.blockArgs(j).len + 1);
            defer allocator.free(args);
            @memcpy(args[0 .. args.len - 1], func.blockArgs(j));
            args[args.len - 1] = value;
            try func.setJump(from, to, args);
        },
        .ret => {},
    };
}

fn withArg(allocator: std.mem.Allocator, func: *Function, args: []const Value, value: Value) !ir.function.ValueList {
    const list = try allocator.alloc(Value, args.len + 1);
    defer allocator.free(list);
    @memcpy(list[0..args.len], args);
    list[args.len] = value;
    return func.internValues(list);
}

fn writesMemory(func: *const Function, loop: anytype) bool {
    for (loop.body, 0..) |member, b| {
        if (!member) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(b)))) |inst| switch (func.opcode(inst)) {
            .store, .call, .call_indirect, .atomic_rmw, .matmul, .va_start, .va_arg, .va_end, .barrier => return true,
            else => {},
        };
    }
    return false;
}

fn holds(items: []const Avail, entry: Avail) bool {
    for (items) |item| {
        if (item.ptr == entry.ptr and item.value == entry.value) return true;
    }
    return false;
}

/// A store of the value an entry already holds leaves that entry true whether the addresses alias or
/// not. This needs whole overlap or none, which holds for pointer-typed values, since C never places
/// two pointer objects partly over each other.
fn sameValueStore(func: *const Function, inst: Inst, st: ir.function.Store) bool {
    return st.mask == null and !func.isByteOrderTagged(inst) and
        func.types.type_kind(func.valueType(st.value)) == .ptr;
}

fn invalidateStore(avail: *std.ArrayList(Avail), func: *const Function, ptr: Value, value: Value, same: bool) void {
    var i: usize = 0;
    while (i < avail.items.len) {
        const item = avail.items[i];
        if (!(same and item.value == value) and mayAlias(func, item.ptr, func.valueType(item.value), ptr, func.valueType(value))) {
            _ = avail.swapRemove(i);
        } else i += 1;
    }
}

/// Drop every availability entry whose address may alias `ptr` (a store to `ptr` may have
/// overwritten it). Entries at provably-distinct addresses survive.
fn invalidateAliasing(avail: *std.ArrayList(Avail), func: *const Function, ptr: Value, stored_ty: ir.types.Type) void {
    var i: usize = 0;
    while (i < avail.items.len) {
        if (mayAlias(func, avail.items[i].ptr, func.valueType(avail.items[i].value), ptr, stored_ty)) {
            _ = avail.swapRemove(i);
        } else i += 1;
    }
}

fn sameAddress(func: *const Function, a: Value, b: Value) bool {
    if (a == b) return true;
    const ba = baseOf(func, a);
    const bb = baseOf(func, b);
    return ba.offset == bb.offset and std.meta.eql(ba.root, bb.root);
}

const Root = union(enum) { alloca: Inst, global: u32, unknown: Value };
const Addr = struct { root: Root, offset: i64 };

/// Resolve `v` to a base address plus a constant offset, following `arith_imm` add/sub chains to an
/// `alloca` or `global_addr` root when possible. An unknown root carries the value itself, so two
/// unknown roots are equal only when they are the same SSA value.
fn baseOf(func: *const Function, v: Value) Addr {
    var cur = v;
    var offset: i64 = 0;
    // Bounded by the def chain length; each step moves to a strictly earlier-defined value.
    var steps: usize = 0;
    while (steps < 64) : (steps += 1) {
        const inst = func.definingInst(cur) orelse return .{ .root = .{ .unknown = cur }, .offset = offset };
        switch (func.opcode(inst)) {
            .alloca => return .{ .root = .{ .alloca = inst }, .offset = offset },
            .global_addr => |g| return .{ .root = .{ .global = g.symbol }, .offset = offset },
            .arith_imm => |a| switch (a.op) {
                .add => {
                    offset += a.imm;
                    cur = a.lhs;
                },
                .sub => {
                    offset -= a.imm;
                    cur = a.lhs;
                },
                else => return .{ .root = .{ .unknown = cur }, .offset = offset },
            },
            else => return .{ .root = .{ .unknown = cur }, .offset = offset },
        }
    }
    return .{ .root = .{ .unknown = cur }, .offset = offset };
}

/// Whether a store to `b` may alias the address `a`. Distinct identified roots (two allocas, two
/// globals, an alloca vs a global) never alias. Disjoint constant-offset ranges from the same root
/// also cannot alias; unresolved roots or access widths remain conservative.
fn mayAlias(func: *const Function, a: Value, a_ty: ir.types.Type, b: Value, b_ty: ir.types.Type) bool {
    if (a == b) return true;
    const aa = baseOf(func, a);
    const ab = baseOf(func, b);
    const same_root = switch (aa.root) {
        .alloca => |ia| switch (ab.root) {
            .alloca => |ib| ia == ib,
            .global => false,
            .unknown => null,
        },
        .global => |ga| switch (ab.root) {
            .global => |gb| ga == gb,
            .alloca => false,
            .unknown => null,
        },
        .unknown => |ua| switch (ab.root) {
            .unknown => |ub| if (ua == ub) true else null,
            else => null,
        },
    };
    if (same_root == false) return false;
    if (same_root == null) return true;
    const a_size = accessSize(func, a_ty) orelse return true;
    const b_size = accessSize(func, b_ty) orelse return true;
    const a_end = std.math.add(i64, aa.offset, @intCast(a_size)) catch return true;
    const b_end = std.math.add(i64, ab.offset, @intCast(b_size)) catch return true;
    return aa.offset < b_end and ab.offset < a_end;
}

/// The exact byte width of primitive memory operations. Aggregate layout and pointer width are
/// target-dependent, so those remain unknown and conservatively alias.
fn accessSize(func: *const Function, ty: ir.types.Type) ?u64 {
    return switch (func.types.type_kind(ty)) {
        .int => |int| (@as(u64, int.bits) + 7) / 8,
        .float => |float| switch (float) {
            .f16 => 2,
            .f32 => 4,
            .f64 => 8,
            .f128 => 16,
        },
        .vector => |vector| std.math.mul(u64, accessSize(func, vector.elem) orelse return null, vector.len) catch null,
        .bool, .ptr, .@"struct", .array, .slice => null,
    };
}

const testing = std.testing;

fn runOnce(allocator: std.mem.Allocator, func: *Function) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();
    return run(allocator, func, &analyses);
}

fn i32Ty(func: *Function) !ir.types.Type {
    return func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
}

test "a store forwards its value to a later load of the same address" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t); // an incoming pointer (mem2reg won't touch it)
    const v = try func.appendBlockParam(b, t);
    try func.appendStore(b, v, p);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(v, func.terminator(b).?.ret.values[0]); // load forwarded the stored value
}

test "a block with one predecessor reuses a load from it, a join block does not" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const only = try func.appendBlock();
    const other = try func.appendBlock();
    const join = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const first = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const positive = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = first, .rhs = zero } });
    try func.appendIf(entry, positive, .{ .target = only }, .{ .target = other });
    const second = try func.appendInst(only, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(only, .{ .ret = ir.function.Ret.one(second) });
    try func.setJump(other, join, &.{});
    const third = try func.appendInst(join, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(third) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(first, func.terminator(only).?.ret.values[0]);
    // `join` has one predecessor too, so it is answered through `other`.
    try testing.expectEqual(first, func.terminator(join).?.ret.values[0]);
}

test "a join block takes a param for an address each predecessor holds with its own value" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const join = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const v = try func.appendBlockParam(entry, t);
    const first = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const positive = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = first, .rhs = zero } });
    try func.appendIf(entry, positive, .{ .target = left }, .{ .target = right });
    try func.appendStore(left, v, p);
    try func.setJump(left, join, &.{});
    try func.setJump(right, join, &.{});
    const again = try func.appendInst(join, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(again) });

    _ = try runOnce(allocator, &func);
    const merged = func.terminator(join).?.ret.values[0];
    try testing.expectEqual(func.blockParams(join)[0], merged);
    try testing.expectEqual(v, func.blockArgs(func.terminator(left).?.jump)[0]);
    try testing.expectEqual(first, func.blockArgs(func.terminator(right).?.jump)[0]);
    var diags = try ir.verify.verify(allocator, &func, .high);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a join block reuses a load that every predecessor still holds" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const left = try func.appendBlock();
    const right = try func.appendBlock();
    const join = try func.appendBlock();
    const p = try func.appendBlockParam(entry, ptr_t);
    const first = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
    const zero = try func.appendInst(entry, t, .{ .iconst = 0 });
    const positive = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = first, .rhs = zero } });
    try func.appendIf(entry, positive, .{ .target = left }, .{ .target = right });
    _ = try func.appendArithImm(left, t, .add, first, 1);
    try func.setJump(left, join, &.{});
    try func.setJump(right, join, &.{});
    const again = try func.appendInst(join, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(join, .{ .ret = ir.function.Ret.one(again) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(first, func.terminator(join).?.ret.values[0]);
}

test "a second load of an address reuses the first" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const y1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const y2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = y1, .rhs = y2 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(y1, add.lhs);
    try testing.expectEqual(y1, add.rhs); // the second load reused the first
}

test "a store to a distinct alloca does not kill an available load" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);
    const a = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    const c = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = c } });
    try func.appendStore(b, v, a); // store to a distinct alloca
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = c } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(first, add.rhs); // the store to `a` did not kill the load of `c`
}

test "a store to a possibly-aliasing pointer forces a reload" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const q = try func.appendBlockParam(b, ptr_t); // unknown base: may alias p
    const v = try func.appendBlockParam(b, t);
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    try func.appendStore(b, v, q);
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(!try runOnce(allocator, &func)); // second load cannot be forwarded
}

test "a non-overlapping constant-offset store preserves an available load" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const value = try func.appendBlockParam(b, t);
    const field12 = try func.appendArithImm(b, ptr_t, .add, base, 12);
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = field12 } });
    try func.appendStore(b, value, base);
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = field12 } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(first, add.rhs);
}

test "an overlapping constant-offset store invalidates an available load" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const base = try func.appendBlockParam(b, ptr_t);
    const value = try func.appendBlockParam(b, t);
    const field12 = try func.appendArithImm(b, ptr_t, .add, base, 12);
    const overlap = try func.appendArithImm(b, ptr_t, .add, base, 14);
    const first = try func.appendInst(b, t, .{ .load = .{ .ptr = field12 } });
    try func.appendStore(b, value, overlap);
    const second = try func.appendInst(b, t, .{ .load = .{ .ptr = field12 } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(!try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(second, add.rhs);
}

// SM6 Task 2 (vcc struct copy) surfaced this: a struct's whole-blob copy stores/loads a WIDE
// word directly through the same base pointer a narrower scalar field also loads/stores
// through (a field at offset 0 shares the struct's own base address, no arithmetic in
// between) - two different widths, same address, same block. Forwarding the wide store's
// value to satisfy the narrow load (or vice versa) would hand a mistyped operand to every use
// of the narrow load's result.
test "a store and a load of different widths at the same address do not forward" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t32 = try i32Ty(&func);
    const t64 = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const wide = try func.appendBlockParam(b, t64);
    try func.appendStore(b, wide, p); // a wide (64-bit) store to p
    const narrow = try func.appendInst(b, t32, .{ .load = .{ .ptr = p } }); // a narrower (32-bit) load of the SAME address
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(narrow) });

    _ = try runOnce(allocator, &func); // may still run (nothing else to fold), but must not forward
    const term = func.terminator(b).?;
    try testing.expectEqual(narrow, term.ret.values[0]); // the narrow load must survive, not be replaced by `wide`
}

test "a store does not forward to a load across a barrier" {
    // The point of a barrier: memory another thread wrote before it becomes visible after
    // it, so the value this block stored is not necessarily what the load reads.
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, t);
    try func.appendStore(b, v, p);
    try func.appendBarrier(b, .workgroup);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });

    try testing.expect(!try runOnce(allocator, &func));
    // The load survives and still feeds the return: nothing was forwarded.
    try testing.expectEqual(y, func.terminator(b).?.ret.values[0]);
    try testing.expectEqual(@as(usize, 3), func.blockInsts(b).len);
}

test "a load does not satisfy a later load across a barrier" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const y1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    try func.appendBarrier(b, .workgroup);
    const y2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = y1, .rhs = y2 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(!try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(y1, add.lhs);
    try testing.expectEqual(y2, add.rhs); // the second load was NOT replaced by the first
}

/// `*p = v; y = *p; return y;` in one block. `tag` marks the LOAD `endian(big)`. A tagged load
/// reads the bytes at `p` reversed, so it does not hold `v` and must not be answered from the
/// store. Returns the stored value, the loaded value and the block.
fn buildStoreThenLoad(func: *Function, tag: bool) !struct { stored: Value, loaded: Value, block: Block } {
    const t = try i32Ty(func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t); // an incoming pointer (mem2reg won't touch it)
    const v = try func.appendBlockParam(b, t);
    try func.appendStore(b, v, p);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    if (tag) try func.addAttr(.{ .value = y }, .{ .endian = .big });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });
    return .{ .stored = v, .loaded = y, .block = b };
}

test "endian: a tagged load is not answered from an earlier store of the same address" {
    const allocator = testing.allocator;
    // The tagged load must survive, still read memory, and still carry its tag: the backend
    // emits the byte swap from the tag, and forwarding `v` here would delete the swap.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildStoreThenLoad(&func, true);
        try testing.expect(!try runOnce(allocator, &func));
        try testing.expectEqual(@as(usize, 2), func.blockInsts(s.block).len); // store + load
        try testing.expectEqual(s.loaded, func.terminator(s.block).?.ret.values[0]);
        try testing.expect(func.isByteOrderTagged(func.definingInst(s.loaded).?));
    }
    // The identical shape WITHOUT the tag still forwards, so the refusal is about the tag and
    // not about the shape.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildStoreThenLoad(&func, false);
        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(s.stored, func.terminator(s.block).?.ret.values[0]);
    }
}

/// `*p = v; y = *p; return y;` in one block. `tag` marks the STORE `endian(big)`. A tagged store
/// writes `v` reversed, so what `p` holds afterwards is not `v` and the later plain load must not
/// be answered from it. Returns the stored value, the loaded value and the block.
fn buildTaggedStoreThenLoad(func: *Function, tag: bool) !struct { stored: Value, loaded: Value, block: Block } {
    const t = try i32Ty(func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, t);
    const st = try func.appendStmtRaw(b, .{ .store = .{ .value = v, .ptr = p } });
    if (tag) try func.addAttr(.{ .inst = st }, .{ .endian = .big });
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });
    return .{ .stored = v, .loaded = y, .block = b };
}

test "endian: a tagged store does not forward its value to a later load" {
    const allocator = testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildTaggedStoreThenLoad(&func, true);
        try testing.expect(!try runOnce(allocator, &func));
        try testing.expectEqual(@as(usize, 2), func.blockInsts(s.block).len);
        try testing.expectEqual(s.loaded, func.terminator(s.block).?.ret.values[0]);
        try testing.expect(func.isByteOrderTagged(func.blockInsts(s.block)[0]));
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildTaggedStoreThenLoad(&func, false);
        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(s.stored, func.terminator(s.block).?.ret.values[0]);
    }
}

/// `y1 = *p; y2 = *p; return y1 + y2;` in one block. `tag` marks the FIRST load `endian(big)`.
/// A tagged load holds the reversed bytes, so the later plain load must not reuse its result.
fn buildTwoLoads(func: *Function, tag: bool) !struct { first: Value, second: Value, sum: Value } {
    const t = try i32Ty(func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const y1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    if (tag) try func.addAttr(.{ .value = y1 }, .{ .endian = .big });
    const y2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = y1, .rhs = y2 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });
    return .{ .first = y1, .second = y2, .sum = sum };
}

test "endian: a tagged load is not reused for a later plain load of the same address" {
    const allocator = testing.allocator;
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildTwoLoads(&func, true);
        try testing.expect(!try runOnce(allocator, &func));
        const add = func.opcode(func.definingInst(s.sum).?).arith;
        try testing.expectEqual(s.first, add.lhs);
        try testing.expectEqual(s.second, add.rhs); // the plain load was NOT replaced by the tagged one
        try testing.expect(func.isByteOrderTagged(func.definingInst(s.first).?));
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildTwoLoads(&func, false);
        try testing.expect(try runOnce(allocator, &func));
        const add = func.opcode(func.definingInst(s.sum).?).arith;
        try testing.expectEqual(s.first, add.lhs);
        try testing.expectEqual(s.first, add.rhs); // the second load reused the first
    }
}

test "a store does not forward to a load across an atomic on the same address" {
    // An atomic is a store of its address, so what the address holds afterwards is not the
    // value the earlier store put there. The no-atomic control at the end proves the pass
    // does forward this exact shape when nothing sits between.
    const allocator = testing.allocator;
    const build = struct {
        fn f(a: std.mem.Allocator, func: *Function, with_atomic: bool) !struct { load: ir.function.Value, block: ir.function.Block } {
            _ = a;
            const t = try i32Ty(func);
            const ptr_t = try func.types.ptrGlobal();
            const b = try func.appendBlock();
            const p = try func.appendBlockParam(b, ptr_t);
            const v = try func.appendBlockParam(b, t);
            try func.appendStore(b, v, p);
            if (with_atomic) {
                try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
            }
            const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
            func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });
            return .{ .load = y, .block = b };
        }
    }.f;

    {
        var func = Function.init(allocator);
        defer func.deinit();
        const built = try build(allocator, &func, true);
        try testing.expect(!try runOnce(allocator, &func));
        // The load survives and still feeds the return: nothing was forwarded.
        try testing.expectEqual(built.load, func.terminator(built.block).?.ret.values[0]);
    }
    {
        // The control: the same shape without the atomic DOES forward, so the refusal above
        // comes from the atomic and not from the alias oracle giving up.
        var func = Function.init(allocator);
        defer func.deinit();
        const built = try build(allocator, &func, false);
        try testing.expect(try runOnce(allocator, &func));
        try testing.expect(func.terminator(built.block).?.ret.values[0] != built.load);
    }
}

test "an atomic ordered above relaxed drops availability for an UNRELATED address too" {
    // A relaxed atomic is only a store of its own address. Anything stronger also makes
    // memory another thread wrote before it visible after it, which is what a barrier does,
    // so the whole availability set goes. The relaxed control forwards the same shape.
    const allocator = testing.allocator;
    const build = struct {
        fn f(func: *Function, ordering: ir.function.AtomicOrdering) !struct { load: ir.function.Value, block: ir.function.Block } {
            const t = try i32Ty(func);
            const ptr_t = try func.types.ptrGlobal();
            const b = try func.appendBlock();
            const v = try func.appendBlockParam(b, t);
            // Two DISTINCT allocas: the alias oracle knows they never alias, so only the
            // whole-set drop can stop the forward.
            const slot = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
            const counter = try func.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
            try func.appendStore(b, v, slot);
            try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = counter, .value = v, .ordering = ordering, .scope = .device });
            const y = try func.appendInst(b, t, .{ .load = .{ .ptr = slot } });
            func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });
            return .{ .load = y, .block = b };
        }
    }.f;

    {
        var func = Function.init(allocator);
        defer func.deinit();
        const built = try build(&func, .seq_cst);
        try testing.expect(!try runOnce(allocator, &func));
        try testing.expectEqual(built.load, func.terminator(built.block).?.ret.values[0]);
    }
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const built = try build(&func, .relaxed);
        try testing.expect(try runOnce(allocator, &func));
        try testing.expect(func.terminator(built.block).?.ret.values[0] != built.load);
    }
}

test "a load does not satisfy a later load across an atomic on the same address" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try i32Ty(&func);
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, t);
    const y1 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    try func.appendAtomicRmwStmt(b, .{ .op = .add, .ptr = p, .value = v, .ordering = .relaxed, .scope = .device });
    const y2 = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    const sum = try func.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = y1, .rhs = y2 } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(sum) });

    try testing.expect(!try runOnce(allocator, &func));
    const add = func.opcode(func.definingInst(sum).?).arith;
    try testing.expectEqual(y1, add.lhs);
    try testing.expectEqual(y2, add.rhs); // the second load was NOT replaced by the first
}

/// `*p = v if m; y = *p; return y;` in one block. `mask` predicates the store. Returns the stored
/// value, the loaded value and the block.
fn buildMaskedStoreThenLoad(func: *Function, mask: bool) !struct { stored: Value, loaded: Value, block: Block } {
    const t = try i32Ty(func);
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const p = try func.appendBlockParam(b, ptr_t);
    const v = try func.appendBlockParam(b, t);
    const m = try func.appendBlockParam(b, bool_t);
    if (mask) try func.appendStoreMasked(b, v, p, m) else try func.appendStore(b, v, p);
    const y = try func.appendInst(b, t, .{ .load = .{ .ptr = p } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(y) });
    return .{ .stored = v, .loaded = y, .block = b };
}

test "a masked store does not forward its value to a later load" {
    const allocator = testing.allocator;
    // The lanes the mask excluded did not write, so the address does not hold `v` afterwards and
    // the load has to stay.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildMaskedStoreThenLoad(&func, true);
        try testing.expect(!try runOnce(allocator, &func));
        try testing.expectEqual(@as(usize, 2), func.blockInsts(s.block).len);
        try testing.expectEqual(s.loaded, func.terminator(s.block).?.ret.values[0]);
        try testing.expect(func.opcode(func.blockInsts(s.block)[0]).store.mask != null);
    }
    // The identical shape with no mask still forwards, so the refusal is about the mask.
    {
        var func = Function.init(allocator);
        defer func.deinit();
        const s = try buildMaskedStoreThenLoad(&func, false);
        try testing.expect(try runOnce(allocator, &func));
        try testing.expectEqual(s.stored, func.terminator(s.block).?.ret.values[0]);
    }
}

test "storing the pointer an address already holds keeps that address known" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const l = try func.appendBlockParam(b, ptr_t);
    const other = try func.appendBlockParam(b, ptr_t);
    const next = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = l } });
    try func.appendStore(b, next, other);
    const again = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = l } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(again) });

    try testing.expect(try runOnce(allocator, &func));
    try testing.expectEqual(next, func.terminator(b).?.ret.values[0]);
}

test "storing a different pointer to an unknown address drops what an address holds" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const b = try func.appendBlock();
    const l = try func.appendBlockParam(b, ptr_t);
    const other = try func.appendBlockParam(b, ptr_t);
    const next = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = l } });
    try func.appendStore(b, other, other);
    const again = try func.appendInst(b, ptr_t, .{ .load = .{ .ptr = l } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(again) });

    _ = try runOnce(allocator, &func);
    try testing.expect(func.terminator(b).?.ret.values[0] != next);
}

test "a load reaches past a loop that only reads memory, and not past one that stores" {
    const allocator = testing.allocator;
    for ([_]bool{ false, true }) |stores| {
        var func = Function.init(allocator);
        defer func.deinit();
        const t = try i32Ty(&func);
        const bool_t = try func.types.intern(.bool);
        const ptr_t = try func.types.ptrGlobal();
        const entry = try func.appendBlock();
        const header = try func.appendBlock();
        const body = try func.appendBlock();
        const done = try func.appendBlock();
        const p = try func.appendBlockParam(entry, ptr_t);
        const q = try func.appendBlockParam(entry, ptr_t);
        const first = try func.appendInst(entry, t, .{ .load = .{ .ptr = p } });
        try func.setJump(entry, header, &.{first});
        const acc = try func.appendBlockParam(header, t);
        const other = try func.appendInst(header, t, .{ .load = .{ .ptr = q } });
        const more = try func.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = acc, .rhs = other } });
        try func.appendIf(header, more, .{ .target = body }, .{ .target = done });
        const next = try func.appendArithImm(body, t, .add, acc, 1);
        if (stores) try func.appendStore(body, next, q);
        try func.setJump(body, header, &.{next});
        const again = try func.appendInst(done, t, .{ .load = .{ .ptr = p } });
        func.setTerminator(done, .{ .ret = ir.function.Ret.one(again) });

        _ = try runOnce(allocator, &func);
        const result = func.terminator(done).?.ret.values[0];
        try testing.expectEqual(!stores, result == first);
    }
}
