//! Dominator analysis over a function's CFG. Block A dominates B when every path
//! from entry to B passes through A. Immediate dominators use the Cooper-Harvey-
//! Kennedy iteration, and tree intervals answer dominance queries in constant time.

const std = @import("std");
const ir = @import("vulcan-ir");
const cfg_mod = @import("cfg.zig");

const Function = ir.function.Function;

pub const Dominators = struct {
    n: usize,
    /// The immediate dominator of each block (itself for the entry and for
    /// unreachable blocks).
    idom: []u32,
    /// Preorder entry and exit positions in the dominator tree.
    pre: []u32,
    post: []u32,
    /// Whether each block is reachable from the entry. Unreachable blocks have a
    /// degenerate dominance relation (dominated by everything), so consumers that
    /// walk the CFG (e.g. loop detection) must exclude them via `isReachable`.
    reachable: []bool,

    pub fn deinit(self: *Dominators, allocator: std.mem.Allocator) void {
        allocator.free(self.idom);
        allocator.free(self.pre);
        allocator.free(self.post);
        allocator.free(self.reachable);
    }

    pub fn dominates(self: *const Dominators, a: usize, b: usize) bool {
        if (!self.reachable[a] or !self.reachable[b]) return a == b;
        return self.pre[a] <= self.pre[b] and self.post[b] <= self.post[a];
    }

    pub fn isReachable(self: *const Dominators, b: usize) bool {
        return self.reachable[b];
    }

    /// Whether `a` strictly dominates `b` (dominates and is not `b`).
    pub fn strictlyDominates(self: *const Dominators, a: usize, b: usize) bool {
        return a != b and self.dominates(a, b);
    }

    pub fn immediateDominator(self: *const Dominators, b: usize) u32 {
        return self.idom[b];
    }
};

/// Compute the dominators of `func` (entry is block 0). The caller owns the
/// result (`deinit`).
pub fn compute(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!Dominators {
    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    return computeFromCfg(allocator, &cfg);
}

/// Compute dominators from an existing CFG. The caller owns the result.
pub fn computeFromCfg(allocator: std.mem.Allocator, cfg: *const cfg_mod.Cfg) std.mem.Allocator.Error!Dominators {
    const n = cfg.blockCount();
    const rpo = try cfg.reversePostorder(allocator);
    defer allocator.free(rpo);
    const reachable = try allocator.alloc(bool, n);
    errdefer allocator.free(reachable);
    @memset(reachable, false);
    for (rpo) |block| reachable[block] = true;

    const rpo_index = try allocator.alloc(u32, n);
    defer allocator.free(rpo_index);
    @memset(rpo_index, std.math.maxInt(u32));
    for (rpo, 0..) |block, index| rpo_index[block] = @intCast(index);

    const idom = try allocator.alloc(u32, n);
    errdefer allocator.free(idom);
    for (idom, 0..) |*parent, block| parent.* = @intCast(block);
    if (rpo.len > 0) idom[rpo[0]] = rpo[0];
    const unresolved = std.math.maxInt(u32);
    for (rpo, 0..) |block, index| if (index != 0) {
        idom[block] = unresolved;
    };
    var changed = true;
    while (changed) {
        changed = false;
        for (rpo, 0..) |block, index| {
            if (index == 0) continue;
            var parent: ?u32 = null;
            for (cfg.predecessors(block)) |pred| {
                if (!reachable[pred]) continue;
                if (idom[pred] == unresolved) continue;
                parent = if (parent) |current| intersect(idom, rpo_index, pred, current) else pred;
            }
            const next = parent orelse continue;
            if (idom[block] != next) {
                idom[block] = next;
                changed = true;
            }
        }
    }

    const pre = try allocator.alloc(u32, n);
    errdefer allocator.free(pre);
    const post = try allocator.alloc(u32, n);
    errdefer allocator.free(post);
    @memset(pre, 0);
    @memset(post, 0);
    try numberTree(allocator, idom, reachable, pre, post);
    return .{ .n = n, .idom = idom, .pre = pre, .post = post, .reachable = reachable };
}

fn intersect(idom: []const u32, rpo_index: []const u32, a: u32, b: u32) u32 {
    var left = a;
    var right = b;
    while (left != right) {
        while (rpo_index[left] > rpo_index[right]) left = idom[left];
        while (rpo_index[right] > rpo_index[left]) right = idom[right];
    }
    return left;
}

fn numberTree(allocator: std.mem.Allocator, idom: []const u32, reachable: []const bool, pre: []u32, post: []u32) std.mem.Allocator.Error!void {
    if (idom.len == 0) return;
    const children = try allocator.alloc(std.ArrayList(u32), idom.len);
    defer {
        for (children) |*list| list.deinit(allocator);
        allocator.free(children);
    }
    for (children) |*list| list.* = .empty;
    for (idom, 0..) |parent, block| {
        if (block == 0 or !reachable[block]) continue;
        try children[parent].append(allocator, @intCast(block));
    }

    const Frame = struct { block: u32, next_child: usize };
    var stack: std.ArrayList(Frame) = .empty;
    defer stack.deinit(allocator);
    var clock: u32 = 0;
    pre[0] = clock;
    clock += 1;
    try stack.append(allocator, .{ .block = 0, .next_child = 0 });
    while (stack.items.len > 0) {
        const frame = &stack.items[stack.items.len - 1];
        if (frame.next_child < children[frame.block].items.len) {
            const child = children[frame.block].items[frame.next_child];
            frame.next_child += 1;
            pre[child] = clock;
            clock += 1;
            try stack.append(allocator, .{ .block = child, .next_child = 0 });
        } else {
            post[frame.block] = clock;
            _ = stack.pop();
        }
    }
}

/// Post-dominator analysis: the mirror of `Dominators` over the reversed CFG. Block A
/// post-dominates B when every path from B to a function exit passes through A. A divergence
/// analysis reads this to find where a conditional branch's two arms meet again.
pub const PostDominators = struct {
    n: usize,
    /// `pdom[b * n + a]` is true when block `a` post-dominates block `b`.
    pdom: []bool,
    /// The immediate post-dominator of each block, or null when it has none. An exit block has
    /// none, and so does a block with no path to an exit, for example a block in an endless
    /// loop. Such a block post-dominates nothing that helps a caller, so the caller must treat
    /// null as "the arms never meet again" and not as "meet at block 0".
    ipdom: []?u32,
    /// Whether each block has a path to a function exit.
    reaches_exit: []bool,

    pub fn deinit(self: *PostDominators, allocator: std.mem.Allocator) void {
        allocator.free(self.pdom);
        allocator.free(self.ipdom);
        allocator.free(self.reaches_exit);
    }

    pub fn postDominates(self: *const PostDominators, a: usize, b: usize) bool {
        return self.pdom[b * self.n + a];
    }

    /// Whether `a` strictly post-dominates `b` (post-dominates and is not `b`).
    pub fn strictlyPostDominates(self: *const PostDominators, a: usize, b: usize) bool {
        return a != b and self.postDominates(a, b);
    }

    pub fn reachesExit(self: *const PostDominators, b: usize) bool {
        return self.reaches_exit[b];
    }

    pub fn immediatePostDominator(self: *const PostDominators, b: usize) ?u32 {
        return self.ipdom[b];
    }
};

/// Compute the post-dominators of `func`. An exit block is a block with no successors, which
/// covers both a `ret` terminator and an unset one (an implicit `ret void`). The caller owns
/// the result (`deinit`).
pub fn computePost(allocator: std.mem.Allocator, func: *const Function) std.mem.Allocator.Error!PostDominators {
    var cfg = try cfg_mod.build(allocator, func);
    defer cfg.deinit(allocator);
    const n = cfg.blockCount();

    // Which blocks can still reach an exit, by walking the CFG backwards from every exit. A
    // block that cannot is inside an endless loop, and its post-dominator set never shrinks
    // below the universe, so its immediate post-dominator has to stay null.
    const reaches_exit = try allocator.alloc(bool, n);
    errdefer allocator.free(reaches_exit);
    @memset(reaches_exit, false);
    {
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(allocator);
        for (0..n) |b| {
            if (cfg.successors(b).len != 0) continue;
            reaches_exit[b] = true;
            try stack.append(allocator, @intCast(b));
        }
        while (stack.pop()) |b| {
            for (cfg.predecessors(b)) |p| {
                if (reaches_exit[p]) continue;
                reaches_exit[p] = true;
                try stack.append(allocator, p);
            }
        }
    }

    const pdom = try allocator.alloc(bool, n * n);
    errdefer allocator.free(pdom);
    @memset(pdom, true);
    // An exit block is post-dominated only by itself.
    for (0..n) |b| {
        if (cfg.successors(b).len != 0) continue;
        for (0..n) |a| pdom[b * n + a] = (a == b);
    }

    const tmp = try allocator.alloc(bool, n);
    defer allocator.free(tmp);

    var changed = true;
    while (changed) {
        changed = false;
        for (0..n) |b| {
            if (cfg.successors(b).len == 0) continue; // an exit block is already final
            // pdom(b) = {b} U (intersection of pdom(s) over successors s).
            for (0..n) |a| {
                var all = true;
                for (cfg.successors(b)) |s| {
                    if (!pdom[@as(usize, s) * n + a]) {
                        all = false;
                        break;
                    }
                }
                tmp[a] = all;
            }
            tmp[b] = true;
            for (0..n) |a| {
                if (pdom[b * n + a] != tmp[a]) {
                    pdom[b * n + a] = tmp[a];
                    changed = true;
                }
            }
        }
    }

    // Immediate post-dominators: the strict post-dominator of b that every other strict
    // post-dominator of b also post-dominates (the closest one). This mirrors the immediate
    // dominator loop above with the relation reversed.
    const ipdom = try allocator.alloc(?u32, n);
    errdefer allocator.free(ipdom);
    for (0..n) |b| {
        ipdom[b] = null;
        if (!reaches_exit[b]) continue;
        for (0..n) |d| {
            if (d == b or !pdom[b * n + d]) continue; // d must strictly post-dominate b
            var lowest = true;
            for (0..n) |o| {
                if (o == b or o == d or !pdom[b * n + o]) continue;
                // Another strict post-dominator o that d does not post-dominate => d is not it.
                if (!pdom[d * n + o]) {
                    lowest = false;
                    break;
                }
            }
            if (lowest) {
                ipdom[b] = @intCast(d);
                break;
            }
        }
    }

    return .{ .n = n, .pdom = pdom, .ipdom = ipdom, .reaches_exit = reaches_exit };
}

test "dominators of a diamond" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const c = try func.appendBlockParam(b0, bool_t);
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    const b3 = try func.appendBlock();
    const v = try func.appendBlockParam(b3, i32_t);
    try func.appendIf(b0, c, .{ .target = b1 }, .{ .target = b2 });
    const x = try func.appendInst(b1, i32_t, .{ .iconst = 1 });
    try func.setJump(b1, b3, &.{x});
    const y = try func.appendInst(b2, i32_t, .{ .iconst = 2 });
    try func.setJump(b2, b3, &.{y});
    func.setTerminator(b3, .{ .ret = ir.function.Ret.one(v) });

    var doms = try compute(allocator, &func);
    defer doms.deinit(allocator);

    // b0 dominates all. b3 is dominated by b0 and b3 only (not b1 or b2).
    try std.testing.expect(doms.dominates(0, 3));
    try std.testing.expect(!doms.dominates(1, 3));
    try std.testing.expect(!doms.dominates(2, 3));
    try std.testing.expect(doms.strictlyDominates(0, 3));
    try std.testing.expect(!doms.strictlyDominates(3, 3));
    // The dominator tree: idom(b1)=idom(b2)=idom(b3)=b0.
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(1));
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(2));
    try std.testing.expectEqual(@as(u32, 0), doms.immediateDominator(3));
}

test "an empty function has an empty dominator tree" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    var doms = try compute(allocator, &func);
    defer doms.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 0), doms.n);
}

test "unreachable blocks are flagged and map their idom to themselves" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try func.appendBlock();
    const exit = try func.appendBlock();
    const orphan = try func.appendBlock(); // no incoming edges
    const x = try func.appendBlockParam(entry, t);
    const ev = try func.appendBlockParam(exit, t);
    const ov = try func.appendBlockParam(orphan, t);
    try func.setJump(entry, exit, &.{x});
    func.setTerminator(exit, .{ .ret = ir.function.Ret.one(ev) });
    try func.setJump(orphan, exit, &.{ov}); // dead edge into exit

    var doms = try compute(allocator, &func);
    defer doms.deinit(allocator);
    try std.testing.expect(doms.isReachable(0)); // entry
    try std.testing.expect(doms.isReachable(1)); // exit (via entry)
    try std.testing.expect(!doms.isReachable(2)); // orphan
    try std.testing.expectEqual(@as(u32, 2), doms.immediateDominator(2)); // orphan -> itself, not garbage
}

test "post-dominators of a diamond meet at the merge block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();

    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b0 = try func.appendBlock();
    const c = try func.appendBlockParam(b0, bool_t);
    const b1 = try func.appendBlock();
    const b2 = try func.appendBlock();
    const b3 = try func.appendBlock();
    const v = try func.appendBlockParam(b3, i32_t);
    try func.appendIf(b0, c, .{ .target = b1 }, .{ .target = b2 });
    const x = try func.appendInst(b1, i32_t, .{ .iconst = 1 });
    try func.setJump(b1, b3, &.{x});
    const y = try func.appendInst(b2, i32_t, .{ .iconst = 2 });
    try func.setJump(b2, b3, &.{y});
    func.setTerminator(b3, .{ .ret = ir.function.Ret.one(v) });

    var pdoms = try computePost(allocator, &func);
    defer pdoms.deinit(allocator);

    // The merge post-dominates every block, and neither arm post-dominates the branch.
    try std.testing.expect(pdoms.postDominates(3, 0));
    try std.testing.expect(!pdoms.postDominates(1, 0));
    try std.testing.expect(!pdoms.postDominates(2, 0));
    try std.testing.expect(pdoms.strictlyPostDominates(3, 0));
    try std.testing.expect(!pdoms.strictlyPostDominates(3, 3));
    try std.testing.expectEqual(@as(?u32, 3), pdoms.immediatePostDominator(0));
    try std.testing.expectEqual(@as(?u32, 3), pdoms.immediatePostDominator(1));
    try std.testing.expectEqual(@as(?u32, 3), pdoms.immediatePostDominator(2));
    try std.testing.expectEqual(@as(?u32, null), pdoms.immediatePostDominator(3)); // the exit
}

test "a block with no path to an exit reports no immediate post-dominator" {
    // Suspicious case: an endless loop never reaches an exit, so its post-dominator set stays at
    // the universe. Reading an immediate post-dominator out of that would name block 0, which is
    // wrong. The analysis must report null instead.
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const entry = try func.appendBlock();
    const spin = try func.appendBlock();
    try func.setJump(entry, spin, &.{});
    try func.setJump(spin, spin, &.{});

    var pdoms = try computePost(allocator, &func);
    defer pdoms.deinit(allocator);
    try std.testing.expect(!pdoms.reachesExit(0));
    try std.testing.expect(!pdoms.reachesExit(1));
    try std.testing.expectEqual(@as(?u32, null), pdoms.immediatePostDominator(0));
    try std.testing.expectEqual(@as(?u32, null), pdoms.immediatePostDominator(1));
}

test "a loop's exit branch post-dominates into the block after the loop" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);
    const entry = try func.appendBlock();
    const head = try func.appendBlock();
    const body = try func.appendBlock();
    const done = try func.appendBlock();
    const n = try func.appendBlockParam(entry, i32_t);
    const i = try func.appendBlockParam(head, i32_t);
    const zero = try func.appendInst(entry, i32_t, .{ .iconst = 0 });
    try func.setJump(entry, head, &.{zero});
    const c = try func.appendInst(head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = n } });
    try func.appendIf(head, c, .{ .target = body }, .{ .target = done });
    const next = try func.appendArithImm(body, i32_t, .add, i, 1);
    try func.setJump(body, head, &.{next});
    func.setTerminator(done, .{ .ret = ir.function.Ret.one(n) });

    var pdoms = try computePost(allocator, &func);
    defer pdoms.deinit(allocator);
    try std.testing.expectEqual(@as(?u32, 3), pdoms.immediatePostDominator(1)); // head -> done
    try std.testing.expectEqual(@as(?u32, 1), pdoms.immediatePostDominator(2)); // body -> head
}
