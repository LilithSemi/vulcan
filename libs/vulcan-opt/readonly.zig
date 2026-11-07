//! Hoist an invariant read-only leaf call from a loop known to enter its body.
const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("pass.zig");
const inlining = @import("inline.zig");
const schedule = @import("microarch/schedule.zig");
const Function = ir.function.Function;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Block = ir.function.Block;

fn readable(f: *const Function, inst: Inst) bool {
    return switch (f.opcode(inst)) {
        .load => |l| !l.@"volatile" and l.mask == null and !f.isByteOrderTagged(inst),
        .iconst, .fconst, .fconst128, .arith, .arith_imm, .icmp, .select, .convert, .unary, .global_addr, .struct_new, .extract, .dot, .reduce, .splat => true,
        else => false,
    };
}

fn readOnlyLeaf(f: *const Function) bool {
    if (f.is_variadic) return false;
    for (0..f.blockCount()) |bi| for (f.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
        if (f.opcode(inst) != .@"if" and !readable(f, inst)) return false;
    };
    return true;
}

fn literal(f: *const Function, value: Value) ?i64 {
    const inst = f.definingInst(value) orelse return null;
    return if (f.opcode(inst) == .iconst) f.opcode(inst).iconst else null;
}

const Gather = struct {
    allocator: std.mem.Allocator,
    f: *Function,
    analyses: *pass.Analyses,
    body: []const bool,
    pre: u32,
    state: []u8,
    order: std.ArrayList(Inst) = .empty,

    fn visit(g: *Gather, value: Value, depth: u32) pass.Error!bool {
        if (depth > 128) return false;
        const raw = @backingInt(value);
        if (g.state[raw] == 2) return true;
        if (g.state[raw] == 1) return false;
        const definitions = try g.analyses.definingBlocks();
        const block = definitions[raw];
        if (block >= g.body.len) return false;
        if (!g.body[block]) return (try g.analyses.dominators()).dominates(block, g.pre);
        const inst = g.f.definingInst(value) orelse return false; // loop-carried parameter
        if (!readable(g.f, inst)) return false;
        g.state[raw] = 1;
        var operands: std.ArrayList(Value) = .empty;
        defer operands.deinit(g.allocator);
        try schedule.collectOperands(g.allocator, g.f, inst, &operands);
        for (operands.items) |operand| if (!try g.visit(operand, depth + 1)) return false;
        try g.order.append(g.allocator, inst);
        g.state[raw] = 2;
        return true;
    }
};

/// Moves at most one call; callers can recompute analyses and repeat. No memory
/// writes or unknown calls may occur in the loop. The only call must execute on
/// the first iteration: its block dominates every latch, no body edge exits,
/// there is no nested loop, and the entry comparison is provably true. Thus
/// neither the call nor its argument loads are speculated onto a zero-trip path.
pub fn run(allocator: std.mem.Allocator, f: *Function, lookup: inlining.Lookup) !bool {
    var analyses = pass.Analyses{ .allocator = allocator, .func = f };
    defer analyses.deinit();
    const info = try analyses.loops();
    const graph = try analyses.cfg();
    const doms = try analyses.dominators();
    for (info.loops) |loop| {
        const pre = loop.preheader orelse continue;
        const header: Block = @fromBackingInt(loop.header);
        const entry = f.terminator(@fromBackingInt(pre)) orelse continue;
        if (entry != .jump or entry.jump.target != header) continue;
        var has_entry_branch = false;
        for (f.blockInsts(@fromBackingInt(pre))) |inst| if (f.opcode(inst) == .@"if") {
            has_entry_branch = true;
        };
        if (has_entry_branch) continue;
        const header_insts = f.blockInsts(header);
        if (header_insts.len != 2) continue;
        const branch = switch (f.opcode(header_insts[header_insts.len - 1])) {
            .@"if" => |cf| cf,
            else => continue,
        };
        if (!loop.contains(@backingInt(branch.then.target)) or loop.contains(@backingInt(branch.@"else".target))) continue;
        const compare_inst = f.definingInst(branch.cond) orelse continue;
        const compare = switch (f.opcode(compare_inst)) {
            .icmp => |c| c,
            else => continue,
        };
        if (compare.op != .lt) continue;
        const int = switch (f.types.type_kind(f.valueType(compare.lhs))) {
            .int => |i| i,
            else => continue,
        };
        if (int.bits != 32 and int.bits != 64) continue;
        const params = f.blockParams(header);
        const index = std.mem.indexOfScalar(Value, params, compare.lhs) orelse continue;
        const args = f.blockArgs(entry.jump);
        if (args.len != params.len) continue;
        const initial = literal(f, args[index]) orelse continue;
        const limit = literal(f, compare.rhs) orelse continue;
        if (initial < 0 or initial >= limit or limit > std.math.maxInt(i32)) continue;

        var site: ?Inst = null;
        var site_block: u32 = 0;
        var safe = true;
        for (info.loops) |other| if (other.header != loop.header and loop.contains(other.header)) {
            safe = false;
        };
        for (0..f.blockCount()) |bi| {
            if (!loop.contains(bi)) continue;
            if (bi < pre) {
                safe = false;
                break;
            }
            for (graph.successors(bi)) |succ| if (bi != loop.header and !loop.contains(succ)) {
                safe = false;
            };
            for (f.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| {
                switch (f.opcode(inst)) {
                    .call => |c| {
                        if (site != null or c.is_variadic or c.sret or c.ret_dest != null or f.instResult(inst) == null) {
                            safe = false;
                            break;
                        }
                        const callee = lookup.get(f.symbolName(c.symbol)) orelse {
                            safe = false;
                            break;
                        };
                        if (!readOnlyLeaf(callee)) {
                            safe = false;
                            break;
                        }
                        site = inst;
                        site_block = @intCast(bi);
                    },
                    .@"if" => {},
                    .arith => |a| if (a.op == .div or a.op == .rem) {
                        safe = false;
                        break;
                    },
                    .arith_imm => |a| if (a.op == .div or a.op == .rem) {
                        safe = false;
                        break;
                    },
                    else => if (!readable(f, inst)) {
                        safe = false;
                        break;
                    },
                }
            }
        }
        const call = site orelse continue;
        if (!safe or site_block != @backingInt(branch.then.target)) continue;
        for (graph.predecessors(loop.header)) |pred| if (loop.contains(pred) and !doms.dominates(site_block, pred)) {
            safe = false;
        };
        if (!safe) continue;
        const state = try allocator.alloc(u8, f.valueCount());
        defer allocator.free(state);
        @memset(state, 0);
        var gather = Gather{ .allocator = allocator, .f = f, .analyses = &analyses, .body = loop.body, .pre = pre, .state = state };
        defer gather.order.deinit(allocator);
        for (f.valueList(f.opcode(call).call.args)) |arg| if (!try gather.visit(arg, 0)) {
            safe = false;
            break;
        };
        if (!safe) continue;
        try gather.order.append(allocator, call);
        for (0..f.blockCount()) |bi| {
            if (!loop.contains(bi)) continue;
            const insts = f.blockInstsMut(@fromBackingInt(@intCast(bi)));
            var write: usize = 0;
            for (insts.items) |inst| {
                if (std.mem.indexOfScalar(Inst, gather.order.items, inst) != null) continue;
                insts.items[write] = inst;
                write += 1;
            }
            insts.shrinkRetainingCapacity(write);
        }
        try f.blockInstsMut(@fromBackingInt(pre)).appendSlice(allocator, gather.order.items);
        return true;
    }
    return false;
}

const TestLookup = struct {
    callee: *const Function,
    fn get(context: *anyopaque, name: []const u8) ?*const Function {
        const self: *const TestLookup = @ptrCast(@alignCast(context));
        return if (std.mem.eql(u8, name, "read")) self.callee else null;
    }
};

const TestCase = enum { invariant, zero_trip, variant, volatile_callee, volatile_arg, writes, writes_callee, unknown, unknown_bound, conditional };

fn checkHoist(case: TestCase) !void {
    const allocator = std.testing.allocator;
    var callee = Function.init(allocator);
    defer callee.deinit();
    const callee_ptr = try callee.types.ptrGlobal();
    const callee_int = try callee.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
    const cb = try callee.appendBlock();
    const cp = try callee.appendBlockParam(cb, callee_ptr);
    const read = try callee.appendInst(cb, callee_int, .{ .load = .{ .ptr = cp, .@"volatile" = case == .volatile_callee } });
    if (case == .writes_callee) try callee.appendStore(cb, read, cp);
    callee.setTerminator(cb, .{ .ret = ir.function.Ret.one(read) });
    var context = TestLookup{ .callee = &callee };
    const lookup = inlining.Lookup{ .context = &context, .func = TestLookup.get };
    var f = Function.init(allocator);
    defer f.deinit();
    const ptr_t = try f.types.ptrGlobal();
    const int_t = try f.types.intern(.{ .int = .{ .bits = 32, .signedness = .signed } });
    const bool_t = try f.types.intern(.bool);
    const entry = try f.appendBlock();
    const header = try f.appendBlock();
    const body = try f.appendBlock();
    const call_block = if (case == .conditional) try f.appendBlock() else body;
    const latch = try f.appendBlock();
    const done = try f.appendBlock();
    const base = try f.appendBlockParam(entry, ptr_t);
    const zero = try f.appendInst(entry, int_t, .{ .iconst = 0 });
    const limit = if (case == .unknown_bound) try f.appendBlockParam(entry, int_t) else try f.appendInst(entry, int_t, .{ .iconst = if (case == .zero_trip) 0 else 4 });
    const index = try f.appendBlockParam(header, int_t);
    try f.setJump(entry, header, &.{zero});
    const more = try f.appendInst(header, bool_t, .{ .icmp = .{ .op = .lt, .lhs = index, .rhs = limit } });
    try f.appendIf(header, more, .{ .target = body }, .{ .target = done });
    if (case == .conditional) {
        const conditional = try f.appendInst(body, bool_t, .{ .icmp = .{ .op = .eq, .lhs = index, .rhs = zero } });
        try f.appendIf(body, conditional, .{ .target = call_block }, .{ .target = latch });
    }
    const arg_ptr = if (case == .variant)
        try f.appendInst(call_block, ptr_t, .{ .convert = .{ .value = index } })
    else
        base;
    const arg = try f.appendInst(call_block, ptr_t, .{ .load = .{ .ptr = arg_ptr, .@"volatile" = case == .volatile_arg } });
    const call = try f.appendCall(call_block, int_t, if (case == .unknown) "unknown" else "read", &.{arg});
    if (case == .writes) try f.appendStore(call_block, call, base);
    try f.setJump(call_block, latch, &.{});
    const next = try f.appendArithImm(latch, int_t, .add, index, 1);
    try f.setJump(latch, header, &.{next});
    f.setTerminator(done, .{ .ret = ir.function.Ret.one(index) });
    const did_hoist = try run(allocator, &f, lookup);
    try std.testing.expectEqual(case == .invariant, did_hoist);
    if (did_hoist) {
        const insts = f.blockInsts(entry);
        try std.testing.expectEqual(f.definingInst(arg).?, insts[insts.len - 2]);
        try std.testing.expectEqual(f.definingInst(call).?, insts[insts.len - 1]);
    }
    var diagnostics = try ir.verify.verify(allocator, &f, .high);
    defer diagnostics.deinit();
    try std.testing.expect(diagnostics.ok());
}

test "readonly loop call hoisting requires invariant arguments and a guaranteed first execution" {
    inline for (std.meta.tags(TestCase)) |case| try checkHoist(case);
}
