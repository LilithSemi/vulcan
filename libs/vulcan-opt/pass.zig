//! Pass manager: runs a sequence of function passes, lazily computing and caching
//! analyses, invalidating them whenever a pass reports a change.

const std = @import("std");
const ir = @import("vulcan-ir");
const cfg_mod = @import("cfg.zig");
const dom = @import("dominators.zig");
const loops_mod = @import("loops.zig");

const Function = ir.function.Function;

pub const Error = std.mem.Allocator.Error;

/// Lazily computes and caches a function's analyses. A pass queries it for what
/// it needs. The manager recomputes on demand after an invalidation.
pub const Analyses = struct {
    allocator: std.mem.Allocator,
    func: *const Function,
    cfg_cache: ?cfg_mod.Cfg = null,
    rpo_cache: ?[]u32 = null,
    doms: ?dom.Dominators = null,
    loop_info: ?loops_mod.LoopInfo = null,

    pub fn deinit(self: *Analyses) void {
        self.invalidate();
    }

    /// The control-flow graph, computed once and cached until invalidation.
    pub fn cfg(self: *Analyses) Error!*const cfg_mod.Cfg {
        if (self.cfg_cache == null) self.cfg_cache = try cfg_mod.build(self.allocator, self.func);
        return &self.cfg_cache.?;
    }

    /// Reachable blocks in reverse postorder, shared by forward dataflow passes.
    pub fn reversePostorder(self: *Analyses) Error![]const u32 {
        if (self.rpo_cache == null) {
            if (self.cfg_cache == null) self.cfg_cache = try cfg_mod.build(self.allocator, self.func);
            self.rpo_cache = try self.cfg_cache.?.reversePostorder(self.allocator);
        }
        return self.rpo_cache.?;
    }

    /// The dominator analysis, computed once and cached until invalidation.
    pub fn dominators(self: *Analyses) Error!*const dom.Dominators {
        if (self.doms == null) {
            if (self.cfg_cache == null) self.cfg_cache = try cfg_mod.build(self.allocator, self.func);
            self.doms = try dom.computeFromCfg(self.allocator, &self.cfg_cache.?);
        }
        return &self.doms.?;
    }

    /// The natural-loop analysis, computed once and cached until invalidation.
    pub fn loops(self: *Analyses) Error!*const loops_mod.LoopInfo {
        if (self.loop_info == null) {
            if (self.cfg_cache == null) self.cfg_cache = try cfg_mod.build(self.allocator, self.func);
            if (self.doms == null) self.doms = try dom.computeFromCfg(self.allocator, &self.cfg_cache.?);
            self.loop_info = try loops_mod.analyzeFromCfg(self.allocator, &self.cfg_cache.?, &self.doms.?);
        }
        return &self.loop_info.?;
    }

    /// Drop every cached analysis (call after the function is modified).
    pub fn invalidate(self: *Analyses) void {
        if (self.rpo_cache) |order| self.allocator.free(order);
        self.rpo_cache = null;
        if (self.doms) |*d| d.deinit(self.allocator);
        self.doms = null;
        if (self.loop_info) |*l| l.deinit(self.allocator);
        self.loop_info = null;
        if (self.cfg_cache) |*graph| graph.deinit(self.allocator);
        self.cfg_cache = null;
    }
};

/// A function pass: transforms `func`, returning whether it changed it, and may
/// query `analyses` for cached analyses.
pub const Pass = struct {
    name: []const u8,
    run: *const fn (allocator: std.mem.Allocator, func: *Function, analyses: *Analyses) Error!bool,
    /// A successful rewrite leaves all CFG-derived analyses valid.
    preserves_cfg: bool = false,
};

/// Run each pass once in order. Analyses are invalidated after any modifying
/// pass. Returns whether anything changed.
pub fn run(allocator: std.mem.Allocator, func: *Function, passes: []const Pass) Error!bool {
    var analyses = Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();

    var changed = false;
    for (passes) |pass| {
        if (try pass.run(allocator, func, &analyses)) {
            if (!pass.preserves_cfg) analyses.invalidate();
            changed = true;
        }
    }
    return changed;
}

/// Run the passes repeatedly until a full sweep makes no change (a fixpoint), up
/// to `max_iters` sweeps. Returns whether anything changed overall.
pub fn runToFixpoint(allocator: std.mem.Allocator, func: *Function, passes: []const Pass, max_iters: usize) Error!bool {
    var analyses = Analyses{ .allocator = allocator, .func = func };
    defer analyses.deinit();

    var changed_overall = false;
    var iter: usize = 0;
    while (iter < max_iters) : (iter += 1) {
        var changed = false;
        for (passes) |pass| {
            if (try pass.run(allocator, func, &analyses)) {
                if (!pass.preserves_cfg) analyses.invalidate();
                changed = true;
            }
        }
        if (!changed) break;
        changed_overall = true;
    }
    return changed_overall;
}

const CacheTestPasses = struct {
    fn prime(_: std.mem.Allocator, _: *Function, analyses: *Analyses) Error!bool {
        _ = try analyses.loops();
        _ = try analyses.reversePostorder();
        return true;
    }

    fn check(_: std.mem.Allocator, _: *Function, analyses: *Analyses) Error!bool {
        std.debug.assert(analyses.cfg_cache != null);
        std.debug.assert(analyses.rpo_cache != null);
        std.debug.assert(analyses.doms != null);
        std.debug.assert(analyses.loop_info != null);
        return false;
    }
};

test "a CFG-preserving pass keeps cached analyses for the next pass" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    _ = try func.appendBlock();

    const passes = [_]Pass{
        .{ .name = "prime", .run = CacheTestPasses.prime, .preserves_cfg = true },
        .{ .name = "check", .run = CacheTestPasses.check },
    };
    try std.testing.expect(try run(allocator, &func, &passes));
}
