//! Target-independent optimization framework: pass manager with cached analyses,
//! plus the analyses and transforms over Vulcan IR. Freestanding (no libc/OS,
//! allocator per call).

const std = @import("std");
const ir = @import("vulcan-ir");

pub const cfg = @import("vulcan-opt/cfg.zig");
pub const dominators = @import("vulcan-opt/dominators.zig");
pub const loops = @import("vulcan-opt/loops.zig");
pub const uniform = @import("vulcan-opt/uniform.zig");
pub const pass = @import("vulcan-opt/pass.zig");
pub const mem2reg = @import("vulcan-opt/mem2reg.zig");
pub const knownbits = @import("vulcan-opt/knownbits.zig");
pub const loadfwd = @import("vulcan-opt/loadfwd.zig");
pub const jumpthread = @import("vulcan-opt/jumpthread.zig");
pub const constfold = @import("vulcan-opt/constfold.zig");
pub const simplify = @import("vulcan-opt/simplify.zig");
pub const strength = @import("vulcan-opt/strength.zig");
pub const immform = @import("vulcan-opt/immform.zig");
pub const branchfold = @import("vulcan-opt/branchfold.zig");
pub const gvn = @import("vulcan-opt/gvn.zig");
pub const licm = @import("vulcan-opt/licm.zig");
pub const ivsr = @import("vulcan-opt/ivsr.zig");
pub const inlining = @import("vulcan-opt/inline.zig");
pub const dce = @import("vulcan-opt/dce.zig");
pub const sink = @import("vulcan-opt/sink.zig");
pub const lto = @import("vulcan-opt/lto.zig");
pub const pgo = @import("vulcan-opt/pgo.zig");
pub const lowerdiv = @import("vulcan-opt/lowerdiv.zig");
pub const vectorize = @import("vulcan-opt/vectorize.zig");
pub const blocklayout = @import("vulcan-opt/blocklayout.zig");
pub const microarch = @import("vulcan-opt/microarch.zig");

/// Default pipeline: constant folding, algebraic simplification, strength reduction, GVN/CSE, LICM,
/// then DCE, to a fixpoint.
pub const default_pipeline = [_]pass.Pass{
    mem2reg.pass_def,
    loadfwd.pass_def,
    constfold.pass_def,
    simplify.pass_def,
    knownbits.pass_def,
    strength.pass_def,
    immform.pass_def,
    branchfold.pass_def,
    jumpthread.pass_def,
    gvn.pass_def,
    licm.pass_def,
    dce.pass_def,
};

/// The late pipeline: induction-variable strength reduction, then just enough clean-up to settle
/// what it leaves behind. It is SEPARATE from the default pipeline for a phase-ordering reason.
///
/// `ivsr` rewrites a loop's addresses into loop-carried pointers, which grows the header's
/// parameter list and turns index arithmetic into a back-edge step. Four microarch idiom
/// recognizers read the SHAPE it replaces and match on an exact parameter count:
/// `loopvec.recognize` wants exactly one header parameter, `loopvec.recognizeReduction` and
/// `splitunroll` want the induction plus reduction accumulators and nothing else, and `dotprod`
/// wants exactly four. Running `ivsr` before them costs vectorization, which is worth more than
/// the multiply it removes. So it runs LAST, after those recognizers have had their look. See
/// `optimizeEarly` and `optimizeLate`, and `frontends/vcc.zig`, which is where both halves meet.
pub const late_pipeline = [_]pass.Pass{
    ivsr.pass_def,
    constfold.pass_def,
    simplify.pass_def,
    strength.pass_def,
    immform.pass_def,
    dce.pass_def,
};

/// Optimize `func` in place and lay out its blocks after both cleanup pipelines.
pub fn optimize(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    const early = try optimizeEarly(allocator, func);
    const late = try optimizeLate(allocator, func);
    const laid_out = try layout(allocator, func, null);
    return early or late or laid_out;
}

/// The default pipeline to a fixpoint. It leaves every loop in the shape the microarchitecture
/// idiom recognizers read.
pub fn optimizeEarly(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    return pass.runToFixpoint(allocator, func, &default_pipeline, 16);
}

/// The late pipeline to a fixpoint.
pub fn optimizeLate(allocator: std.mem.Allocator, func: *ir.function.Function) pass.Error!bool {
    return pass.runToFixpoint(allocator, func, &late_pipeline, 16);
}

/// Lay out blocks after every CFG-changing and instruction-scheduling pass has finished.
pub fn layout(allocator: std.mem.Allocator, func: *ir.function.Function, profile: ?[]const u64) pass.Error!bool {
    return blocklayout.layout(allocator, func, profile);
}

test {
    std.testing.refAllDecls(@This());
}
