//! SPIR-V to Glacier, end to end: a compute module through the SPIR-V lowering, through the
//! Glacier instruction selector, executed on the warp model.
//!
//! This is the path `glacierctl exec kernel.spv` takes, and it needs two things a Vulkan dispatch
//! does not. A storage buffer becomes a `global_addr` the host places, because a Glacier kernel
//! has no parameter block to receive a pointer in. And the kernel divides work by WORKGROUP, not
//! by thread: one warp is the workgroup, so a per-thread index would be a vector while a warp
//! index is a scalar. `gl_WorkGroupID` and `gl_NumWorkGroups` are the scalar pair that carries it,
//! and they land on `mhartid` and the hart count.

const std = @import("std");
const spirv = @import("vulcan-spirv");
const glacier = @import("vulcan-target").glacier;
const ir = @import("vulcan-ir");
const gpu = @import("vulcan-gpu");

const warp = @import("warp.zig");
const op = spirv.opcodes;

/// Where the host put the buffer. The kernel is built for this address.
const data_base: u32 = 0x1000;

/// `void main() { out[gl_WorkGroupID.x] = gl_NumWorkGroups.x; }`
///
/// Index and value are deliberately different things, so a lowering that swapped them fails. Each
/// warp writes the hart count into its own slot, which checks both builtins at once: the slot
/// proves `mhartid` arrived and the value proves the hart count did.
fn fillModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 int=2 uint=3 v3uint=4 pInV3=5 pInU=6 arr=7 struct=8 pSbStruct=9 pSbInt=10
    //      voidfn=11 c0=12 elemptr=13 wid=14 out=15 main=16 entry=17 xptr=18 i=19 nwg=20
    //      nptr=21 n=22.
    var b = try spirv.binary.Builder.init(allocator, 23);
    errdefer b.deinit(allocator);
    // "main" packs into one word plus a word for the terminator; "out" fits one word with it.
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 16, 0x6E69616D, 0, 14, 15, 20 });
    try b.emit(allocator, op.Name, &.{ 15, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.Decorate, &.{ 20, op.Decoration.builtin, op.BuiltIn.num_workgroups });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 1 });
    try b.emit(allocator, op.TypeInt, &.{ 3, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 3, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 12, 0 });
    try b.emit(allocator, op.Variable, &.{ 5, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Variable, &.{ 5, 20, op.StorageClass.input });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.AccessChain, &.{ 6, 18, 14, 12 }); // &gl_WorkGroupID.x
    try b.emit(allocator, op.Load, &.{ 3, 19, 18 });
    try b.emit(allocator, op.AccessChain, &.{ 6, 21, 20, 12 }); // &gl_NumWorkGroups.x
    try b.emit(allocator, op.Load, &.{ 3, 22, 21 });
    try b.emit(allocator, op.AccessChain, &.{ 10, 13, 15, 12, 19 }); // &out[i]
    try b.emit(allocator, op.Store, &.{ 13, 22 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

fn compileFill(allocator: std.mem.Allocator) ![]u32 {
    var b = try fillModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    return glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
}

test "a SPIR-V compute kernel runs on Glacier" {
    const allocator = std.testing.allocator;
    const words = try compileFill(allocator);
    defer allocator.free(words);

    for ([_]u32{ 1, 2, 4, 8 }) |warps| {
        var data: [8]u32 = @splat(0xFFFF_FFFF);
        var mem: warp.Memory = .{ .base = data_base, .words = &data };
        for (0..warps) |w| {
            var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = warps };
            _ = try warp.run(words, &mem, &one, 10_000);
        }
        // Each warp wrote the hart count into its own slot, and no warp touched another's.
        for (data, 0..) |got, i| {
            const want: u32 = if (i < warps) warps else 0xFFFF_FFFF;
            try std.testing.expectEqual(want, got);
        }
    }
}

test "the storage buffer became an address and not a parameter" {
    const allocator = std.testing.allocator;
    var b = try fillModule(allocator);
    defer b.deinit(allocator);

    // Under the option, the buffer is a `global_addr` and the entry block takes only the two
    // builtin parameters. Without it the buffer is a third, pointer-typed parameter, which a
    // Glacier kernel has no way to receive.
    var as_global = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer as_global.deinit();
    try std.testing.expectEqual(@as(usize, 2), as_global.blockParams(@fromBackingInt(0)).len);

    var as_param = try spirv.lowerModule(allocator, b.words.items);
    defer as_param.deinit();
    try std.testing.expectEqual(@as(usize, 3), as_param.blockParams(@fromBackingInt(0)).len);
    try std.testing.expectError(
        error.Unsupported,
        glacier.isel.compileKernel(allocator, &as_param, .{
            .profile = .@"gc1.n",
            .globals = &.{.{ .name = "out", .address = data_base }},
        }),
    );
}

test "a buffer the host did not place is refused, not guessed" {
    const allocator = std.testing.allocator;
    var b = try fillModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    try std.testing.expectError(
        error.Unsupported,
        glacier.isel.compileKernel(allocator, &func, .{ .profile = .@"gc1.n" }),
    );
}

/// The same kernel in the shape ZIG's SPIR-V backend emits, which differs from a hand-built or
/// glslang module in ways that each broke the lowering:
///
///   - `OpInBoundsAccessChain` (66) rather than `OpAccessChain` (65), and SPLIT in two: one chain
///     to the struct member, a second to index the array.
///   - the builtin read as a WHOLE uvec3 with `OpLoad`, narrowed by `OpCompositeExtract`, where
///     glslang reads `.x` through an access chain so the loaded value is never vector-typed.
///   - `OpLine` debug annotations interleaved between real instructions.
///   - a fixed `OpTypeArray` rather than `OpTypeRuntimeArray`.
///
/// `out.elems[id] = id * 3 + 1`, from `@workGroupId(0)`.
fn zigShapedModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 v3u32=3 pInV3=4 c4096=5 arr=6 struct=7 pSbStruct=8 pSbArr=9 pSbU32=10
    //      voidfn=11 c0=12 c3=13 c1=14 wid=15 out=16 main=17 entry=18 loaded=19 id=20
    //      pElems=21 pElem=22 scaled=23 sum=24.
    var b = try spirv.binary.Builder.init(allocator, 25);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 15, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 7, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.Constant, &.{ 2, 5, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 6, 2, 5 });
    try b.emit(allocator, op.TypeStruct, &.{ 7, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 12, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 3 });
    try b.emit(allocator, op.Constant, &.{ 2, 14, 1 });
    try b.emit(allocator, op.Variable, &.{ 4, 15, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 8, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 11 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.Line, &.{ 1, 7, 5 });
    try b.emit(allocator, op.Load, &.{ 3, 19, 15 }); // the whole uvec3
    try b.emit(allocator, op.CompositeExtract, &.{ 2, 20, 19, 0 }); // .x
    try b.emit(allocator, op.Line, &.{ 1, 8, 5 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 9, 21, 16, 12 }); // &out.elems
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 10, 22, 21, 20 }); // &out.elems[id]
    try b.emit(allocator, op.IMul, &.{ 2, 23, 20, 13 });
    try b.emit(allocator, op.IAdd, &.{ 2, 24, 23, 14 });
    try b.emit(allocator, op.Store, &.{ 22, 24 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a kernel in Zig's SPIR-V shape runs on Glacier" {
    const allocator = std.testing.allocator;
    var b = try zigShapedModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    for (0..4) |w| {
        var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = 4 };
        _ = try warp.run(words, &mem, &one, 10_000);
    }
    for (data, 0..) |got, i| {
        const want: u32 = if (i < 4) @as(u32, @intCast(i)) * 3 + 1 else 0xFFFF_FFFF;
        try std.testing.expectEqual(want, got);
    }
}

/// The shape `std.spirv` produces, which is what a Zig kernel written the supported way emits.
/// Verified against real `zig build-obj -target spirv32-vulkan` output for:
///
///     out.elems[std.spirv.workgroup_id[0]] = std.spirv.num_workgroups[0];
///
/// `std.spirv` declares each builtin as an `extern const @Vector(3, u32) addrspace(.input)`, so
/// indexing it emits a chain to `.x` and a SCALAR load rather than a whole-vector load. The two
/// wrinkles a hand-built module would not think to include are the memory operands trailing
/// `OpLoad`/`OpStore` and the unreachable block Zig leaves after the return.
fn stdSpirvShapedModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 v3u32=3 pInV3=4 pInU=5 c4096=6 arr=7 struct=8 pSbStruct=9 pSbArr=10
    //      pSbU32=11 voidfn=12 c0=13 wid=14 nwg=15 out=16 main=17 entry=18 widx=19 i=20
    //      cast=21 pElems=22 pElem=23 nidx=24 n=25 dead=26.
    var b = try spirv.binary.Builder.init(allocator, 27);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 14, 15, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.num_workgroups });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 2 });
    try b.emit(allocator, op.Constant, &.{ 2, 6, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 7, 2, 6 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 0 });
    try b.emit(allocator, op.Variable, &.{ 4, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 4, 15, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 12 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.Line, &.{ 1, 6, 5 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 19, 14, 13 }); // &workgroup_id[0]
    try b.emit(allocator, op.Load, &.{ 2, 20, 19, 2, 4 }); // scalar, with memory operands
    try b.emit(allocator, op.Bitcast, &.{ 2, 21, 20 }); // u32 to u32, a no-op Zig still emits
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 10, 22, 16, 13 }); // &out.elems
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 11, 23, 22, 21 }); // &out.elems[i]
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 24, 15, 13 }); // &num_workgroups[0]
    try b.emit(allocator, op.Load, &.{ 2, 25, 24, 2, 4 });
    try b.emit(allocator, op.Store, &.{ 23, 25, 2, 4 });
    try b.emit(allocator, op.Return, &.{});
    // Zig leaves an unreachable block after the return. Nothing branches to it.
    try b.emit(allocator, op.Label, &.{26});
    try b.emit(allocator, op.Unreachable, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a kernel in std.spirv's shape runs on Glacier" {
    const allocator = std.testing.allocator;
    var b = try stdSpirvShapedModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    // The same answer the hand-built module gives, which is the point: each warp writes the hart
    // count into its own slot. Checked against real Zig output, which lowers to 13 words.
    for ([_]u32{ 1, 4 }) |warps| {
        var data: [8]u32 = @splat(0xFFFF_FFFF);
        var mem: warp.Memory = .{ .base = data_base, .words = &data };
        for (0..warps) |w| {
            var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = warps };
            _ = try warp.run(words, &mem, &one, 10_000);
        }
        for (data, 0..) |got, i| {
            try std.testing.expectEqual(if (i < warps) warps else 0xFFFF_FFFF, got);
        }
    }
}

/// `out.elems[gl_GlobalInvocationID.x] = id * 3 + 1`.
///
/// A global invocation id is per-THREAD, and a thread on Glacier is a LANE, so this is the kernel
/// shape that has to vectorize: the index differs between the lanes of one warp, which makes the
/// address and the store per-lane too. Compare `fillModule`, which divides work by warp and stays
/// scalar.
fn perLaneModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 v3u32=3 pInV3=4 pInU=5 c4096=6 arr=7 struct=8 pSbStruct=9 pSbArr=10
    //      pSbU32=11 voidfn=12 c0=13 gid=14 c3=15 out=16 main=17 entry=18 gidx=19 id=20
    //      pElems=21 pElem=22 scaled=23 sum=24 c1=25.
    var b = try spirv.binary.Builder.init(allocator, 26);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 14, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.global_invocation_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 2 });
    try b.emit(allocator, op.Constant, &.{ 2, 6, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 7, 2, 6 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 15, 3 });
    try b.emit(allocator, op.Constant, &.{ 2, 25, 1 });
    try b.emit(allocator, op.Variable, &.{ 4, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 12 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 19, 14, 13 });
    try b.emit(allocator, op.Load, &.{ 2, 20, 19 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 10, 21, 16, 13 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 11, 22, 21, 20 });
    try b.emit(allocator, op.IMul, &.{ 2, 23, 20, 15 });
    try b.emit(allocator, op.IAdd, &.{ 2, 24, 23, 25 });
    try b.emit(allocator, op.Store, &.{ 22, 24 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a per-thread kernel fills every lane in one pass" {
    const allocator = std.testing.allocator;
    var b = try perLaneModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n", // 4 lanes
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    // One warp of four lanes covers four elements, so two warps cover eight. A scalar lowering
    // would write one element per warp and leave three of every four untouched.
    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    for ([_]u32{ 1, 2 }) |warps| {
        var data: [8]u32 = @splat(0xFFFF_FFFF);
        var mem: warp.Memory = .{ .base = data_base, .words = &data };
        for (0..warps) |w| {
            var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = warps, .lanes = lanes };
            _ = try warp.run(words, &mem, &one, 10_000);
        }
        const covered = warps * lanes;
        for (data, 0..) |got, i| {
            const id: u32 = @intCast(i);
            try std.testing.expectEqual(if (i < covered) id * 3 + 1 else 0xFFFF_FFFF, got);
        }
    }

    // The store really is a scatter, not a scalar store repeated.
    var scatters: usize = 0;
    for (words) |w| {
        if (w & 0x7F == 0x27 and (w >> 26) & 0b11 == 0b01) scatters += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), scatters);
}

test "a per-lane branch is refused rather than sent whichever way lane zero went" {
    // Divergence needs masks, which this lowering does not emit yet. The wrong answer it would
    // otherwise give is the whole warp taking one arm, so this refuses instead.
    const allocator = std.testing.allocator;
    var f = ir.function.Function.init(allocator);
    defer f.deinit();
    const u32_t = try f.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try f.types.intern(.bool);

    const entry = try f.appendBlock();
    const tid = try f.appendBlockParam(entry, u32_t);
    try gpu.attrs.setBuiltin(&f, tid, .thread_id_x);
    const then_b = try f.appendBlock();
    const else_b = try f.appendBlock();
    const zero = try f.appendInst(entry, u32_t, .{ .iconst = 0 });
    const c = try f.appendInst(entry, bool_t, .{ .icmp = .{ .op = .gt, .lhs = tid, .rhs = zero } });
    try f.appendIf(entry, c, .{ .target = then_b }, .{ .target = else_b });
    f.setTerminator(then_b, .{ .ret = ir.function.Ret.none() });
    f.setTerminator(else_b, .{ .ret = ir.function.Ret.none() });

    try std.testing.expectError(
        error.Unsupported,
        glacier.isel.compileKernel(allocator, &f, .{ .profile = .@"gc1.n" }),
    );
}

test "every lowered module is valid IR" {
    // The lowering mixes types the module declared with i32 values it synthesizes itself, so an
    // `arith` between them used to be invalid IR that `ir.verify` rejected. Nothing downstream
    // noticed, because the Glacier selector reads signedness from the left operand and add, sub
    // and mul have identical bits either way. A consumer that verifies, or a pass that trusts
    // operand types, would not have been so forgiving.
    const allocator = std.testing.allocator;
    const builders = [_]*const fn (std.mem.Allocator) anyerror!spirv.binary.Builder{
        fillModule,
        zigShapedModule,
        stdSpirvShapedModule,
        perLaneModule,
    };
    for (builders) |build| {
        var b = try build(allocator);
        defer b.deinit(allocator);
        for ([_]bool{ false, true }) |as_globals| {
            var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = as_globals });
            defer func.deinit();
            var diags = try ir.verify.verify(allocator, &func, .low);
            defer diags.deinit();
            try std.testing.expectEqualSlices(ir.verify.Diagnostic, &.{}, diags.items());
        }
    }
}

/// `out.elems[gl_WorkGroupID.x * lanes + gl_LocalInvocationID.x] = gl_LocalInvocationID.x * 10`.
///
/// `gl_LocalInvocationID` is the thread's index INSIDE its workgroup, and a workgroup here is one
/// warp, so this is the lane index. It is the builtin a kernel reaches for when it wants to place
/// its own lane, and it was unmapped until this test existed. Verified against real Zig output for
/// `std.spirv.local_invocation_id[0]`.
fn laneIdModule(allocator: std.mem.Allocator, lanes: u32) !spirv.binary.Builder {
    // ids: void=1 u32=2 v3u32=3 pInV3=4 pInU=5 c4096=6 arr=7 struct=8 pSbStruct=9 pSbArr=10
    //      pSbU32=11 voidfn=12 c0=13 wid=14 lid=15 out=16 main=17 entry=18 widx=19 w=20
    //      lidx=21 l=22 base=23 idx=24 pElems=25 pElem=26 val=27 cLanes=28 cTen=29.
    var b = try spirv.binary.Builder.init(allocator, 30);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 14, 15, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.local_invocation_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 2 });
    try b.emit(allocator, op.Constant, &.{ 2, 6, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 7, 2, 6 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 28, lanes });
    try b.emit(allocator, op.Constant, &.{ 2, 29, 10 });
    try b.emit(allocator, op.Variable, &.{ 4, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 4, 15, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 12 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 19, 14, 13 });
    try b.emit(allocator, op.Load, &.{ 2, 20, 19 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 21, 15, 13 });
    try b.emit(allocator, op.Load, &.{ 2, 22, 21 });
    try b.emit(allocator, op.IMul, &.{ 2, 23, 20, 28 }); // warp * lanes
    try b.emit(allocator, op.IAdd, &.{ 2, 24, 23, 22 }); // + lane
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 10, 25, 16, 13 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 11, 26, 25, 24 });
    try b.emit(allocator, op.IMul, &.{ 2, 27, 22, 29 }); // lane * 10
    try b.emit(allocator, op.Store, &.{ 26, 27 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "gl_LocalInvocationID is the lane index" {
    const allocator = std.testing.allocator;
    const lanes = glacier.encode.Profile.@"gc1.n".lanes();
    var b = try laneIdModule(allocator, lanes);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    for (0..2) |w| {
        var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = 2, .lanes = lanes };
        _ = try warp.run(words, &mem, &one, 10_000);
    }
    // Each warp writes its own four slots, and the value is the LANE's index times ten, so the
    // pattern repeats per warp. A lowering that confused the lane with the warp would write a
    // rising sequence instead.
    for (data, 0..) |got, i| try std.testing.expectEqual(@as(u32, @intCast(i % lanes)) * 10, got);
}
