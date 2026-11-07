//! Outer-loop vectorization. A loop over `j` whose body runs an inner loop, such as the column
//! loop of a matrix multiply, computes V columns at once: each lane runs the inner loop for its
//! own `j`, so loads that walk along `j` become vector loads, values that do not depend on `j`
//! become splats, and an inner reduction becomes a vector accumulator. Every lane keeps its own
//! evaluation order, so the result is exact, floats included.
//!
//! The vector loop runs while V more columns remain. The original loop then finishes the rest.
//! A store must not share a root pointer with any load. The AArch64 path also proves that
//! roots denote distinct allocation objects; unknown pointer parameters remain scalar.
//! The x86 path retains the existing distinct-root policy.

const std = @import("std");
const ir = @import("vulcan-ir");
const mm = @import("model.zig");
const loops = @import("../loops.zig");
const dce = @import("../dce.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;
const Type = ir.types.Type;
const BinOp = ir.function.BinOp;

pub const Error = std.mem.Allocator.Error;

const lanes = 4;

/// How a body value relates to `j`. `index` is the counter widened and scaled by `1 << shift`
/// elements, `address` is a uniform pointer plus such an index, and `vector` holds one value
/// per lane.
const Kind = union(enum) {
    uniform,
    counter,
    index: u6,
    address: u6,
    vector,
};

const Plan = struct {
    header: Block,
    preheader: Block,
    counter: Value,
    bound: Value,
    step: Inst,
    body: []bool,
};

pub fn run(allocator: std.mem.Allocator, func: *Function, model: *const mm.Model) Error!bool {
    if ((model.arch != .x86_64 and model.arch != .aarch64) or model.simdBits() < 128) return false;
    var changed = false;
    // Each rewrite adds blocks, so the loop structure is read again after one.
    for (0..8) |_| {
        if (!try runOnce(allocator, func, model.arch == .aarch64)) break;
        changed = true;
    }
    return changed;
}

fn runOnce(allocator: std.mem.Allocator, func: *Function, prove_disjoint: bool) Error!bool {
    var info = try loops.analyze(allocator, func);
    defer info.deinit(allocator);
    for (info.loops) |*loop| {
        var plan = try recognize(allocator, func, &info, loop) orelse continue;
        defer allocator.free(plan.body);
        const kinds = try allocator.alloc(Kind, func.valueCount());
        defer allocator.free(kinds);
        const entry = func.opcode(func.blockInsts(plan.header)[1]).@"if".then.target;
        const order = try bodyOrder(allocator, func, &plan, entry);
        defer allocator.free(order);
        if (!try classify(func, &plan, order, kinds)) continue;
        if (!memorySafe(func, &plan, kinds, prove_disjoint)) continue;
        try rewrite(allocator, func, &plan, kinds);
        return true;
    }
    return false;
}

fn recognize(allocator: std.mem.Allocator, func: *const Function, info: *const loops.LoopInfo, loop: *const loops.Loop) Error!?Plan {
    const preheader_index = loop.preheader orelse return null;
    const header: Block = @fromBackingInt(@intCast(loop.header));
    const preheader: Block = @fromBackingInt(@intCast(preheader_index));
    switch (func.terminator(preheader) orelse return null) {
        .jump => |j| if (j.target != header) return null,
        .ret => return null,
    }
    const params = func.blockParams(header);
    if (params.len != 1) return null;
    const counter = params[0];
    if (func.types.type_kind(func.valueType(counter)) != .int) return null;
    const insts = func.blockInsts(header);
    if (insts.len != 2) return null;
    const cmp = switch (func.opcode(insts[0])) {
        .icmp => |c| c,
        else => return null,
    };
    const cf = switch (func.opcode(insts[1])) {
        .@"if" => |c| c,
        else => return null,
    };
    if (cf.cond != func.instResult(insts[0]).? or cmp.op != .lt or cmp.lhs != counter) return null;
    if (!loop.contains(@backingInt(cf.then.target)) or loop.contains(@backingInt(cf.@"else".target))) return null;
    if (func.blockArgs(cf.then).len != 0) return null;
    if (definedIn(func, loop, cmp.rhs)) return null;

    // The one latch steps the counter by one, and nothing else reads that step.
    var step: ?Inst = null;
    var has_inner = false;
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi] or bi == loop.header) continue;
        const block: Block = @fromBackingInt(@intCast(bi));
        if (func.terminator(block)) |term| switch (term) {
            .jump => |j| if (j.target == header) {
                if (step != null or func.blockArgs(j).len != 1) return null;
                const arg = func.blockArgs(j)[0];
                const def = func.definingInst(arg) orelse return null;
                switch (func.opcode(def)) {
                    .arith_imm => |a| if (a.op != .add or a.lhs != counter or a.imm != 1) return null,
                    else => return null,
                }
                if (try useCount(allocator, func, arg) != 1) return null;
                step = def;
            },
            .ret => return null,
        };
        // Every edge stays inside the loop or returns to its header.
        for (successors(func, block)) |maybe| {
            const s = maybe orelse continue;
            if (!loop.contains(@backingInt(s))) return null;
        }
        for (info.loops) |*other| if (other.header == bi) {
            has_inner = true;
        };
    }
    if (!has_inner) return null;
    const body = try allocator.alloc(bool, func.blockCount());
    for (body, 0..) |*b, bi| b.* = bi < loop.body.len and loop.body[bi] and bi != loop.header;
    return .{ .header = header, .preheader = preheader, .counter = counter, .bound = cmp.rhs, .step = step orelse {
        allocator.free(body);
        return null;
    }, .body = body };
}

/// Fill `kinds` for every value, to a fixpoint over the body's block parameters.
fn classify(func: *const Function, plan: *const Plan, order: []const Block, kinds: []Kind) Error!bool {
    @memset(kinds, .uniform);
    kinds[@backingInt(plan.counter)] = .counter;
    var stores: usize = 0;
    var work: usize = 0;
    var again = true;
    var rounds: usize = 0;
    while (again) : (rounds += 1) {
        if (rounds > 16) return false;
        again = false;
        stores = 0;
        work = 0;
        for (order) |block| {
            for (func.blockInsts(block)) |inst| {
                if (inst == plan.step) continue;
                const kind = instKind(func, inst, kinds) orelse return false;
                switch (func.opcode(inst)) {
                    .store => stores += 1,
                    .load => if (kind == .vector) {
                        work += 1;
                    },
                    else => {},
                }
                const result = func.instResult(inst) orelse continue;
                kinds[@backingInt(result)] = kind;
            }
            for (successors(func, block), edgeArgs(func, block)) |maybe, args| {
                const s = maybe orelse continue;
                if (!plan.body[@backingInt(s)]) continue;
                const params = func.blockParams(s);
                if (args.len != params.len) return false;
                for (params, args) |param, arg| {
                    const k = kinds[@backingInt(arg)];
                    switch (k) {
                        .uniform => {},
                        .vector => if (kinds[@backingInt(param)] != .vector) {
                            if (!vectorElem(func, func.valueType(param))) return false;
                            kinds[@backingInt(param)] = .vector;
                            again = true;
                        },
                        else => return false,
                    }
                }
            }
        }
    }
    return stores != 0 and work != 0;
}

/// The kind of one body instruction's result, or null when the loop cannot be vectorized.
fn instKind(func: *const Function, inst: Inst, kinds: []const Kind) ?Kind {
    const k = struct {
        fn of(ks: []const Kind, v: Value) Kind {
            return ks[@backingInt(v)];
        }
    }.of;
    switch (func.opcode(inst)) {
        .iconst, .fconst => return .uniform,
        .convert => |c| return switch (k(kinds, c.value)) {
            .uniform => .uniform,
            .counter => if (intBits(func, func.instResult(inst).?) == 64) .{ .index = 0 } else null,
            else => null,
        },
        .arith_imm => |a| return switch (k(kinds, a.lhs)) {
            .uniform => .uniform,
            .index => |s| switch (a.op) {
                .shl => if (a.imm >= 0 and s + a.imm <= 8) .{ .index = @intCast(s + a.imm) } else null,
                else => null,
            },
            .address => |s| if (a.op == .add) .{ .address = s } else null,
            .vector => if (vectorOp(func, a.op, func.instResult(inst).?)) .vector else null,
            .counter => null,
        },
        .arith => |a| {
            const l = k(kinds, a.lhs);
            const r = k(kinds, a.rhs);
            if (l == .uniform and r == .uniform) return .uniform;
            if (a.op == .add) {
                if (isPtr(func, a.lhs) and l == .uniform and r == .index) return .{ .address = r.index };
                if (isPtr(func, a.rhs) and r == .uniform and l == .index) return .{ .address = l.index };
                if (l == .address and r == .uniform) return .{ .address = l.address };
                if (r == .address and l == .uniform) return .{ .address = r.address };
            }
            if ((l == .vector or l == .uniform) and (r == .vector or r == .uniform) and vectorOp(func, a.op, func.instResult(inst).?))
                return .vector;
            return null;
        },
        .load => |l| {
            if (l.@"volatile" or l.mask != null) return null;
            const result = func.instResult(inst).?;
            return switch (k(kinds, l.ptr)) {
                .uniform => .uniform,
                .address => |s| if (vectorElem(func, func.valueType(result)) and s == 2) .vector else null,
                else => null,
            };
        },
        .store => |s| {
            if (s.@"volatile" or s.mask != null) return null;
            const address_ok = switch (k(kinds, s.ptr)) {
                .address => |shift| shift == 2,
                else => false,
            };
            const value_ok = switch (k(kinds, s.value)) {
                .uniform, .vector => vectorElem(func, func.valueType(s.value)),
                else => false,
            };
            return if (address_ok and value_ok) .uniform else null;
        },
        .icmp => |c| return if (k(kinds, c.lhs) == .uniform and k(kinds, c.rhs) == .uniform) .uniform else null,
        .@"if" => |c| return if (k(kinds, c.cond) == .uniform) .uniform else null,
        .select => |s| return if (k(kinds, s.cond) == .uniform and k(kinds, s.then) == .uniform and k(kinds, s.@"else") == .uniform) .uniform else null,
        else => return null,
    }
}

/// Every store goes through a root pointer no load in the loop also uses.
fn memorySafe(func: *const Function, plan: *const Plan, kinds: []const Kind, prove_disjoint: bool) bool {
    _ = kinds;
    var store_roots: [8]Value = undefined;
    var store_count: usize = 0;
    for (0..func.blockCount()) |bi| {
        if (!plan.body[bi]) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| switch (func.opcode(inst)) {
            .store => |s| {
                if (store_count == store_roots.len) return false;
                store_roots[store_count] = pointerRoot(func, s.ptr);
                store_count += 1;
            },
            else => {},
        };
    }
    for (0..func.blockCount()) |bi| {
        if (!plan.body[bi]) continue;
        for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| switch (func.opcode(inst)) {
            .load => |l| {
                const root = pointerRoot(func, l.ptr);
                for (store_roots[0..store_count]) |s| {
                    if (s == root or (prove_disjoint and !distinctObjects(func, s, root))) return false;
                }
            },
            else => {},
        };
    }
    return true;
}

fn distinctObjects(func: *const Function, a: Value, b: Value) bool {
    const ai = func.definingInst(a) orelse return false;
    const bi = func.definingInst(b) orelse return false;
    const ao = func.opcode(ai);
    const bo = func.opcode(bi);
    if (ao == .alloca) return (bo == .alloca and ai != bi) or bo == .global_addr;
    if (ao == .global_addr) return bo == .alloca or (bo == .global_addr and ao.global_addr.symbol != bo.global_addr.symbol);
    return false;
}

fn pointerRoot(func: *const Function, ptr: Value) Value {
    var v = ptr;
    for (0..32) |_| {
        const inst = func.definingInst(v) orelse return v;
        switch (func.opcode(inst)) {
            .arith => |a| if (a.op == .add or a.op == .sub) {
                v = if (isPtr(func, a.lhs)) a.lhs else if (isPtr(func, a.rhs)) a.rhs else return v;
                continue;
            },
            .arith_imm => |a| if (a.op == .add or a.op == .sub) {
                v = a.lhs;
                continue;
            },
            else => {},
        }
        return v;
    }
    return v;
}

const Clone = struct {
    func: *Function,
    kinds: []const Kind,
    /// Scalar clone of each body value: uniform values, the counter chain, and addresses.
    scalar: []?Value,
    /// Vector clone of each vector value.
    vector: []?Value,

    fn scalarOf(self: *const Clone, v: Value) Value {
        return self.scalar[@backingInt(v)] orelse v;
    }

    /// `v` as a vector in `block`: its vector clone, or a splat of its scalar.
    fn vectorOf(self: *Clone, block: Block, v: Value) Error!Value {
        if (self.vector[@backingInt(v)]) |vv| return vv;
        const s = self.scalarOf(v);
        const vt = try vectorType(self.func, self.func.valueType(v));
        const fields: [lanes]Value = @splat(s);
        return self.func.appendInst(block, vt, .{ .struct_new = .{ .fields = try self.func.internValues(&fields) } });
    }
};

fn rewrite(allocator: std.mem.Allocator, func: *Function, plan: *const Plan, kinds: []const Kind) Error!void {
    const n = func.blockCount();
    const clones = try allocator.alloc(?Block, n);
    defer allocator.free(clones);
    @memset(clones, null);
    const scalar = try allocator.alloc(?Value, func.valueCount() + 4096);
    defer allocator.free(scalar);
    @memset(scalar, null);
    const vector = try allocator.alloc(?Value, func.valueCount() + 4096);
    defer allocator.free(vector);
    @memset(vector, null);
    var c = Clone{ .func = func, .kinds = kinds, .scalar = scalar, .vector = vector };

    const counter_ty = func.valueType(plan.counter);
    const i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);

    // The vector loop header: continue while `j + lanes <= bound`, compared in 64 bits.
    const main = try func.appendBlock();
    const j = try func.appendBlockParam(main, counter_ty);
    c.scalar[@backingInt(plan.counter)] = j;
    for (0..n) |bi| if (plan.body[bi]) {
        clones[bi] = try func.appendBlock();
    };
    for (0..n) |bi| if (plan.body[bi]) {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block)) |p| {
            const cl = clones[bi].?;
            if (kinds[@backingInt(p)] == .vector) {
                c.vector[@backingInt(p)] = try func.appendBlockParam(cl, try vectorType(func, func.valueType(p)));
            } else {
                c.scalar[@backingInt(p)] = try func.appendBlockParam(cl, func.valueType(p));
            }
        }
    };
    const header_if = func.blockInsts(plan.header)[1];
    const entry_target = func.opcode(header_if).@"if".then.target;
    {
        const wide_j = if (intBits(func, j) == 64) j else try func.appendInst(main, i64_t, .{ .convert = .{ .value = j } });
        const wide_bound = if (intBits(func, plan.bound) == 64) plan.bound else try func.appendInst(main, i64_t, .{ .convert = .{ .value = plan.bound } });
        const last = try func.appendArithImm(main, i64_t, .add, wide_j, lanes);
        const fits = try func.appendInst(main, bool_t, .{ .icmp = .{ .op = .le, .lhs = last, .rhs = wide_bound } });
        try func.appendIf(main, fits, .{ .target = clones[@backingInt(entry_target)].?, .args = &.{} }, .{ .target = plan.header, .args = &.{j} });
    }

    // Instructions, in an order where each block follows its dominators.
    const order = try bodyOrder(allocator, func, plan, entry_target);
    defer allocator.free(order);
    for (order) |block| {
        const cl = clones[@backingInt(block)].?;
        for (func.blockInsts(block)) |inst| {
            if (inst == plan.step) {
                c.scalar[@backingInt(func.instResult(inst).?)] = try func.appendArithImm(cl, counter_ty, .add, j, lanes);
                continue;
            }
            try cloneInst(&c, cl, inst);
        }
    }
    // Terminators and edges, once every value has its clone.
    for (order) |block| {
        const cl = clones[@backingInt(block)].?;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .@"if" => |cf| {
                const then_args = try edgeClone(allocator, &c, cl, cf.then);
                defer allocator.free(then_args);
                const else_args = try edgeClone(allocator, &c, cl, cf.@"else");
                defer allocator.free(else_args);
                try func.appendIf(cl, c.scalarOf(cf.cond), .{ .target = clones[@backingInt(cf.then.target)].?, .args = then_args }, .{ .target = clones[@backingInt(cf.@"else".target)].?, .args = else_args });
            },
            else => {},
        };
        // A block that ends in an `if` keeps only the placeholder terminator.
        if (endsInIf(func, block)) {
            func.setTerminator(cl, .{ .ret = ir.function.Ret.none() });
            continue;
        }
        const jmp = func.terminator(block).?.jump;
        if (jmp.target == plan.header) {
            try func.setJump(cl, main, &.{c.scalarOf(func.blockArgs(jmp)[0])});
        } else {
            const args = try edgeClone(allocator, &c, cl, jmp);
            defer allocator.free(args);
            try func.setJump(cl, clones[@backingInt(jmp.target)].?, args);
        }
    }

    // The preheader now enters the vector loop, which hands the rest to the original loop.
    const pre_jump = func.terminator(plan.preheader).?.jump;
    try func.setJump(plan.preheader, main, func.blockArgs(pre_jump));
}

fn cloneInst(c: *Clone, block: Block, inst: Inst) Error!void {
    const func = c.func;
    const op = func.opcode(inst);
    switch (op) {
        .@"if" => return,
        .store => |s| {
            const value = try c.vectorOf(block, s.value);
            try func.appendStore(block, value, c.scalarOf(s.ptr));
            return;
        },
        else => {},
    }
    const result = func.instResult(inst).?;
    const ty = func.valueType(result);
    switch (c.kinds[@backingInt(result)]) {
        .vector => {
            const vt = try vectorType(func, ty);
            const v = switch (op) {
                .load => |l| try func.appendInst(block, vt, .{ .load = .{ .ptr = c.scalarOf(l.ptr) } }),
                .arith => |a| blk: {
                    const lhs = try c.vectorOf(block, a.lhs);
                    const rhs = try c.vectorOf(block, a.rhs);
                    break :blk try func.appendInst(block, vt, .{ .arith = .{ .op = a.op, .lhs = lhs, .rhs = rhs } });
                },
                .arith_imm => |a| blk: {
                    const lhs = try c.vectorOf(block, a.lhs);
                    const k = try func.appendInst(block, ty, .{ .iconst = a.imm });
                    const fields: [lanes]Value = @splat(k);
                    const rhs = try func.appendInst(block, vt, .{ .struct_new = .{ .fields = try func.internValues(&fields) } });
                    break :blk try func.appendInst(block, vt, .{ .arith = .{ .op = a.op, .lhs = lhs, .rhs = rhs } });
                },
                else => unreachable,
            };
            c.vector[@backingInt(result)] = v;
        },
        else => {
            const mapped = try remapScalar(c, op);
            c.scalar[@backingInt(result)] = try func.appendInst(block, ty, mapped);
        },
    }
}

fn remapScalar(c: *const Clone, op: ir.function.Opcode) Error!ir.function.Opcode {
    return switch (op) {
        .iconst, .fconst => op,
        .convert => |cv| .{ .convert = .{ .value = c.scalarOf(cv.value) } },
        .arith => |a| .{ .arith = .{ .op = a.op, .lhs = c.scalarOf(a.lhs), .rhs = c.scalarOf(a.rhs) } },
        .arith_imm => |a| .{ .arith_imm = .{ .op = a.op, .lhs = c.scalarOf(a.lhs), .imm = a.imm } },
        .load => |l| .{ .load = .{ .ptr = c.scalarOf(l.ptr) } },
        .icmp => |cmp| .{ .icmp = .{ .op = cmp.op, .lhs = c.scalarOf(cmp.lhs), .rhs = c.scalarOf(cmp.rhs) } },
        .select => |s| .{ .select = .{ .cond = c.scalarOf(s.cond), .then = c.scalarOf(s.then), .@"else" = c.scalarOf(s.@"else") } },
        else => unreachable,
    };
}

fn edgeClone(allocator: std.mem.Allocator, c: *Clone, block: Block, jump: ir.function.Jump) Error![]Value {
    const func = c.func;
    const args = func.blockArgs(jump);
    const params = func.blockParams(jump.target);
    const out = try allocator.alloc(Value, args.len);
    for (args, params, out) |arg, param, *o| {
        o.* = if (c.kinds[@backingInt(param)] == .vector) try c.vectorOf(block, arg) else c.scalarOf(arg);
    }
    return out;
}

/// Body blocks in reverse postorder from the loop's first body block, so dominators come first.
fn bodyOrder(allocator: std.mem.Allocator, func: *const Function, plan: *const Plan, entry: Block) Error![]Block {
    const n = func.blockCount();
    const seen = try allocator.alloc(bool, n);
    defer allocator.free(seen);
    @memset(seen, false);
    var post: std.ArrayList(Block) = .empty;
    errdefer post.deinit(allocator);
    const Frame = struct { block: Block, next: u8 };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);
    seen[@backingInt(entry)] = true;
    try stack.append(allocator, .{ .block = entry, .next = 0 });
    while (stack.items.len != 0) {
        const top = &stack.items[stack.items.len - 1];
        const succs = successors(func, top.block);
        if (top.next < succs.len) {
            const s = succs[top.next];
            top.next += 1;
            const target = s orelse continue;
            const ti = @backingInt(target);
            if (!plan.body[ti] or seen[ti]) continue;
            seen[ti] = true;
            try stack.append(allocator, .{ .block = target, .next = 0 });
        } else {
            try post.append(allocator, top.block);
            _ = stack.pop();
        }
    }
    std.mem.reverse(Block, post.items);
    return post.toOwnedSlice(allocator);
}

fn successors(func: *const Function, block: Block) [3]?Block {
    var out: [3]?Block = .{ null, null, null };
    if (endsInIfInst(func, block)) |inst| {
        out[0] = func.opcode(inst).@"if".then.target;
        out[1] = func.opcode(inst).@"if".@"else".target;
    }
    if (func.terminator(block)) |term| switch (term) {
        .jump => |j| out[2] = j.target,
        .ret => {},
    };
    return out;
}

fn edgeArgs(func: *const Function, block: Block) [3][]const Value {
    var out: [3][]const Value = .{ &.{}, &.{}, &.{} };
    if (endsInIfInst(func, block)) |inst| {
        out[0] = func.blockArgs(func.opcode(inst).@"if".then);
        out[1] = func.blockArgs(func.opcode(inst).@"if".@"else");
    }
    if (func.terminator(block)) |term| switch (term) {
        .jump => |j| out[2] = func.blockArgs(j),
        .ret => {},
    };
    return out;
}

fn endsInIfInst(func: *const Function, block: Block) ?Inst {
    const insts = func.blockInsts(block);
    if (insts.len == 0) return null;
    const last = insts[insts.len - 1];
    return if (func.opcode(last) == .@"if") last else null;
}

fn endsInIf(func: *const Function, block: Block) bool {
    return endsInIfInst(func, block) != null;
}

fn definedIn(func: *const Function, loop: *const loops.Loop, v: Value) bool {
    for (0..func.blockCount()) |bi| {
        if (bi >= loop.body.len or !loop.body[bi]) continue;
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block)) |p| if (p == v) return true;
        for (func.blockInsts(block)) |inst| if (func.instResult(inst) == v) return true;
    }
    return false;
}

fn useCount(allocator: std.mem.Allocator, func: *const Function, v: Value) Error!u32 {
    const uses = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(uses);
    dce.countUses(func, uses);
    return uses[@backingInt(v)];
}

fn intBits(func: *const Function, v: Value) u16 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i.bits,
        .ptr => 64,
        else => 0,
    };
}

fn isPtr(func: *const Function, v: Value) bool {
    return func.types.type_kind(func.valueType(v)) == .ptr;
}

/// A 32-bit integer or f32 element, the width the vector backends lower four lanes of.
fn vectorElem(func: *const Function, ty: Type) bool {
    return switch (func.types.type_kind(ty)) {
        .int => |i| i.bits == 32,
        .float => |f| f == .f32,
        else => false,
    };
}

fn vectorType(func: *Function, elem: Type) Error!Type {
    return func.types.intern(.{ .vector = .{ .len = lanes, .elem = elem } });
}

fn vectorOp(func: *const Function, op: BinOp, result: Value) bool {
    return switch (func.types.type_kind(func.valueType(result))) {
        .int => |i| i.bits == 32 and switch (op) {
            .add, .sub, .mul, .bit_and, .bit_or, .bit_xor => true,
            else => false,
        },
        .float => |f| f == .f32 and switch (op) {
            .add, .sub, .mul, .div => true,
            else => false,
        },
        else => false,
    };
}
