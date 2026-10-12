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

test "a storage buffer arrives either as an address or as a launch argument" {
    const allocator = std.testing.allocator;
    var b = try fillModule(allocator);
    defer b.deinit(allocator);

    // Under the option the buffer is a `global_addr`, baked into the instruction stream, and the
    // entry block takes only the two builtin parameters.
    var as_global = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer as_global.deinit();
    try std.testing.expectEqual(@as(usize, 2), as_global.blockParams(@fromBackingInt(0)).len);

    // Without it the buffer is a third, POINTER-TYPED parameter, which is a kernel parameter like
    // any other and comes from the first launch argument. This is the path that matters for a
    // driver: the same words bind a different buffer, with no rebuild.
    var as_param = try spirv.lowerModule(allocator, b.words.items);
    defer as_param.deinit();
    try std.testing.expectEqual(@as(usize, 3), as_param.blockParams(@fromBackingInt(0)).len);
    const words = try glacier.isel.compileKernel(allocator, &as_param, .{ .profile = .@"gc1.n" });
    defer allocator.free(words);

    // Nothing in the stream names an address, and 0xFC1 is read once.
    var reads: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b1110011) continue;
        if ((w >> 12) & 0b111 != 2) continue;
        if (w >> 20 == 0xFC1) reads += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), reads);

    // And the buffer the host passes is the buffer it writes.
    var data: [8]u32 = @splat(0xFFFF_FFFF);
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1 };
    one.launch_args[0] = data_base;
    _ = try warp.run(words, &mem, &one, 10_000);
    try std.testing.expectEqual(@as(u32, 1), data[0]);
    for (data[1..]) |got| try std.testing.expectEqual(@as(u32, 0xFFFF_FFFF), got);
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
        floatDivideModule,
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

    // The warp writes a CONTIGUOUS run, so the store is the unit-stride form and not a scatter.
    // This is the shape every Zig kernel reaches memory through, an access chain into a storage
    // buffer, and it is the one that has to be recognized: an address built from pointer
    // arithmetic by hand proves nothing about what a frontend emits.
    var unit: usize = 0;
    var indexed: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b0100111) continue;
        if ((w >> 12) & 0b111 != 0b110) continue;
        switch ((w >> 26) & 0b11) {
            0b00 => unit += 1,
            0b01 => indexed += 1,
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 1), unit);
    try std.testing.expectEqual(@as(usize, 0), indexed);
}

/// A kernel that keeps a value in a FUNCTION-storage local across a branch, which is the shape
/// Zig's SPIR-V backend emits for any local variable. The two arms store different values and the
/// merge block loads it, so promoting the slot needs a real phi: tracking "the last value stored"
/// cannot answer what the merge sees. The shared `mem2reg` does that, which is why the selector
/// runs it before anything else.
///
/// `out.elems[0] = if (gl_WorkGroupID.x < 1) 10 else 20`
fn functionLocalModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 bool=3 v3u32=4 pInV3=5 pInU=6 pFnU=7 c4096=8 arr=9 struct=10
    //      pSbStruct=11 pSbArr=12 pSbU32=13 voidfn=14 c0=15 c1=16 c10=17 c20=18 wid=19 out=20
    //      main=21 entry=22 slot=23 widx=24 w=25 cmp=26 then=27 els=28 merge=29 got=30
    //      pElems=31 pElem=32.
    var b = try spirv.binary.Builder.init(allocator, 33);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 21, 0x6E69616D, 0, 19, 20 });
    try b.emit(allocator, op.Name, &.{ 20, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 19, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 10, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeBool, &.{3});
    try b.emit(allocator, op.TypeVector, &.{ 4, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 2 });
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.function, 2 });
    try b.emit(allocator, op.Constant, &.{ 2, 8, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 9, 2, 8 });
    try b.emit(allocator, op.TypeStruct, &.{ 10, 9 });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.storage_buffer, 10 });
    try b.emit(allocator, op.TypePointer, &.{ 12, op.StorageClass.storage_buffer, 9 });
    try b.emit(allocator, op.TypePointer, &.{ 13, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 14, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 15, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 16, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 17, 10 });
    try b.emit(allocator, op.Constant, &.{ 2, 18, 20 });
    try b.emit(allocator, op.Variable, &.{ 5, 19, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 11, 20, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 21, 0, 14 });
    try b.emit(allocator, op.Label, &.{22});
    try b.emit(allocator, op.Variable, &.{ 7, 23, op.StorageClass.function });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 6, 24, 19, 15 });
    try b.emit(allocator, op.Load, &.{ 2, 25, 24 });
    try b.emit(allocator, op.ULessThan, &.{ 3, 26, 25, 16 });
    try b.emit(allocator, op.SelectionMerge, &.{ 29, 0 });
    try b.emit(allocator, op.BranchConditional, &.{ 26, 27, 28 });
    try b.emit(allocator, op.Label, &.{27});
    try b.emit(allocator, op.Store, &.{ 23, 17 });
    try b.emit(allocator, op.Branch, &.{29});
    try b.emit(allocator, op.Label, &.{28});
    try b.emit(allocator, op.Store, &.{ 23, 18 });
    try b.emit(allocator, op.Branch, &.{29});
    try b.emit(allocator, op.Label, &.{29});
    try b.emit(allocator, op.Load, &.{ 2, 30, 23 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 12, 31, 20, 15 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 13, 32, 31, 15 });
    try b.emit(allocator, op.Store, &.{ 32, 30 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a function-storage local survives a branch" {
    const allocator = std.testing.allocator;
    var b = try functionLocalModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    // No `alloca` reaches the selector: a warp has no stack, so a slot that survived promotion
    // would have been refused rather than compiled.
    for ([_]struct { u32, u32 }{ .{ 0, 10 }, .{ 1, 20 } }) |case| {
        var data: [2]u32 = @splat(0xFFFF_FFFF);
        var mem: warp.Memory = .{ .base = data_base, .words = &data };
        var one: warp.Warp = .{ .hart_id = case[0], .hart_count = 2, .lanes = 4 };
        _ = try warp.run(words, &mem, &one, 10_000);
        try std.testing.expectEqual(case[1], data[0]);
    }
}

test "gl_WorkGroupSize comes from the profile, not from the module" {
    // A kernel reads the workgroup width to stride its own inner loop. Answering it from the
    // profile rather than folding the module's declared LocalSize is what lets one kernel fit a
    // wider or narrower core without being rebuilt.
    const allocator = std.testing.allocator;
    var b = try widthModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();

    for ([_]glacier.encode.Profile{ .@"gc1.n", .@"gc1.s", .@"gc1.f" }) |profile| {
        const words = try glacier.isel.compileKernel(allocator, &func, .{
            .profile = profile,
            .globals = &.{.{ .name = "out", .address = data_base }},
        });
        defer allocator.free(words);
        var data: [1]u32 = .{0xFFFF_FFFF};
        var mem: warp.Memory = .{ .base = data_base, .words = &data };
        var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = profile.lanes() };
        _ = try warp.run(words, &mem, &one, 10_000);
        try std.testing.expectEqual(@as(u32, profile.lanes()), data[0]);
    }
}

/// `out.elems[0] = gl_WorkGroupSize.x`
fn widthModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 v3u32=3 pInV3=4 pInU=5 c4096=6 arr=7 struct=8 pSbStruct=9 pSbArr=10
    //      pSbU32=11 voidfn=12 c0=13 size=14 out=15 main=16 entry=17 sidx=18 s=19 pElems=20
    //      pElem=21.
    var b = try spirv.binary.Builder.init(allocator, 22);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 16, 0x6E69616D, 0, 14, 15 });
    try b.emit(allocator, op.Name, &.{ 15, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.workgroup_size });
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
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 12 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 18, 14, 13 });
    try b.emit(allocator, op.Load, &.{ 2, 19, 18 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 10, 20, 15, 13 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 11, 21, 20, 13 });
    try b.emit(allocator, op.Store, &.{ 21, 19 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

/// A FUNCTION-storage local typed as a two-field struct of u32, the shape Zig's SPIR-V backend
/// emits for a small local struct. Each field is stored through a constant-index access chain,
/// then the whole struct is loaded back and its fields extracted and summed, which exercises the
/// scalar replacement that turns the local into one alloca per field.
///
/// `out.elems[0] = local.a + local.b` where `local.a = 10`, `local.b = 20`.
fn structLocalModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 u32=2 structT=3 pFnStruct=4 pFnU32=5 c0=6 c1=7 c10=8 c20=9 c4096=10 arr=11
    //      structSB=12 pSbStruct=13 pSbArr=14 pSbU32=15 voidfn=16 out=17 main=18 entry=19
    //      local=20 chain0=21 chain1=22 loaded=23 ext0=24 ext1=25 sum=26 pElems=27 pElem=28.
    var b = try spirv.binary.Builder.init(allocator, 29);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 18, 0x6E69616D, 0, 17 });
    try b.emit(allocator, op.Name, &.{ 17, 0x0074756F });
    try b.emit(allocator, op.MemberDecorate, &.{ 12, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeStruct, &.{ 3, 2, 2 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.function, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.function, 2 });
    try b.emit(allocator, op.Constant, &.{ 2, 6, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 7, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 8, 10 });
    try b.emit(allocator, op.Constant, &.{ 2, 9, 20 });
    try b.emit(allocator, op.Constant, &.{ 2, 10, 4096 });
    try b.emit(allocator, op.TypeArray, &.{ 11, 2, 10 });
    try b.emit(allocator, op.TypeStruct, &.{ 12, 11 });
    try b.emit(allocator, op.TypePointer, &.{ 13, op.StorageClass.storage_buffer, 12 });
    try b.emit(allocator, op.TypePointer, &.{ 14, op.StorageClass.storage_buffer, 11 });
    try b.emit(allocator, op.TypePointer, &.{ 15, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 16, 1 });
    try b.emit(allocator, op.Variable, &.{ 13, 17, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 18, 0, 16 });
    try b.emit(allocator, op.Label, &.{19});
    try b.emit(allocator, op.Variable, &.{ 4, 20, op.StorageClass.function });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 21, 20, 6 });
    try b.emit(allocator, op.Store, &.{ 21, 8 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 5, 22, 20, 7 });
    try b.emit(allocator, op.Store, &.{ 22, 9 });
    try b.emit(allocator, op.Load, &.{ 3, 23, 20 });
    try b.emit(allocator, op.CompositeExtract, &.{ 2, 24, 23, 0 });
    try b.emit(allocator, op.CompositeExtract, &.{ 2, 25, 23, 1 });
    try b.emit(allocator, op.IAdd, &.{ 2, 26, 24, 25 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 14, 27, 17, 6 });
    try b.emit(allocator, op.InBoundsAccessChain, &.{ 15, 28, 27, 6 });
    try b.emit(allocator, op.Store, &.{ 28, 26 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a function-storage struct local scalarizes to one alloca per field" {
    const allocator = std.testing.allocator;
    var b = try structLocalModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    var data: [1]u32 = .{0xFFFF_FFFF};
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 4 };
    _ = try warp.run(words, &mem, &one, 10_000);
    try std.testing.expectEqual(@as(u32, 30), data[0]);
}

/// The perspective divide, which is what Glacier's one scalar float instruction is for: a divide
/// per warp, then an integer screen coordinate. The buffer holds the denominator first and the
/// results after it.
fn floatDivideModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 uint=2 float=3 v3uint=4 pInV3=5 pInU=6 arr=7 struct=8 pSbStruct=9 pSbU32=10
    //      voidfn=11 c0=12 c1=13 wid=14 out=15 main=16 entry=17 widptr=18 w=19 pDenom=20
    //      denom=21 c100=22 q=23 n=24 pElem=26 pSbF32=27.
    var b = try spirv.binary.Builder.init(allocator, 28);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 16, 0x6E69616D, 0, 14, 15 });
    try b.emit(allocator, op.Name, &.{ 15, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 14, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 1, op.Decoration.offset, 4 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeVector, &.{ 4, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 4 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 2 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 3, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypePointer, &.{ 27, op.StorageClass.storage_buffer, 3 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 12, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 22, @as(u32, @bitCast(@as(f32, 100.0))) });
    try b.emit(allocator, op.Variable, &.{ 5, 14, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 9, 15, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 16, 0, 11 });
    try b.emit(allocator, op.Label, &.{17});
    try b.emit(allocator, op.AccessChain, &.{ 6, 18, 14, 12 }); // &gl_WorkGroupID.x
    try b.emit(allocator, op.Load, &.{ 2, 19, 18 });
    try b.emit(allocator, op.AccessChain, &.{ 27, 20, 15, 12 }); // &out.denominator
    try b.emit(allocator, op.Load, &.{ 3, 21, 20 });
    try b.emit(allocator, op.FDiv, &.{ 3, 23, 22, 21 });
    try b.emit(allocator, op.ConvertFToU, &.{ 2, 24, 23 });
    try b.emit(allocator, op.AccessChain, &.{ 10, 26, 15, 13, 19 }); // &out.results[warp]
    try b.emit(allocator, op.Store, &.{ 26, 24 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a float divide and a float to integer convert run on Glacier" {
    const allocator = std.testing.allocator;
    var b = try floatDivideModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.n",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    const warps: u32 = 3;
    var data: [8]u32 = @splat(0xFFFF_FFFF);
    data[0] = @bitCast(@as(f32, 8.0));
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    for (0..warps) |w| {
        var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = warps };
        _ = try warp.run(words, &mem, &one, 10_000);
    }
    // 100.0 / 8.0 is exact, so the truncation has nothing to remove.
    for (0..warps) |w| try std.testing.expectEqual(@as(u32, 12), data[1 + w]);

    // 100.0 / 3.0 is not exact, and the divider truncates rather than rounding, so the integer
    // result is 33 either way. The float quotient is what shows the truncation; see `lower.zig`.
    data = @splat(0xFFFF_FFFF);
    data[0] = @bitCast(@as(f32, 3.0));
    var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1 };
    _ = try warp.run(words, &mem, &one, 10_000);
    try std.testing.expectEqual(@as(u32, 33), data[1]);
}

/// `out.results[warp] = u32(f16(signed(out.denominator_slot)) * 2.0)`, where the integer comes from
/// a LOAD and so keeps the module's signless integer type. The point is `OpConvertSToF`: a SPIR-V
/// integer type carries no signedness, so the opcode is the only place that says signed, and a
/// selector that reads the operand type sees unsigned and has no instruction for it.
fn signlessConvertModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    // ids: void=1 uint=2 float=3 half=4 v3uint=5 pInV3=6 pInU=7 arr=8 struct=9 pSbStruct=10
    //      pSbU32=11 voidfn=12 c0=13 c1=14 wid=15 out=16 main=17 entry=18 widptr=19 w=20
    //      pSrc=21 src=22 asfloat=23 scaled=24 narrowed=25 n=26 pElem=27 c2=28.
    var b = try spirv.binary.Builder.init(allocator, 29);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 15, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 9, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.MemberDecorate, &.{ 9, 1, op.Decoration.offset, 4 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    // SIGNLESS, which is the only integer type Zig's SPIR-V backend ever emits.
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeFloat, &.{ 4, 16 });
    try b.emit(allocator, op.TypeVector, &.{ 5, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 6, op.StorageClass.input, 5 });
    try b.emit(allocator, op.TypePointer, &.{ 7, op.StorageClass.input, 2 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 8, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 9, 2, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 9 });
    try b.emit(allocator, op.TypePointer, &.{ 11, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 12, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 13, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 14, 1 });
    try b.emit(allocator, op.Constant, &.{ 3, 28, @as(u32, @bitCast(@as(f32, 2.0))) });
    try b.emit(allocator, op.Variable, &.{ 6, 15, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 10, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 12 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.AccessChain, &.{ 7, 19, 15, 13 }); // &gl_WorkGroupID.x
    try b.emit(allocator, op.Load, &.{ 2, 20, 19 });
    try b.emit(allocator, op.AccessChain, &.{ 11, 21, 16, 13 }); // &out.source
    try b.emit(allocator, op.Load, &.{ 2, 22, 21 });
    try b.emit(allocator, op.ConvertSToF, &.{ 4, 23, 22 }); // f16(signed(source))
    try b.emit(allocator, op.FConvert, &.{ 3, 24, 23 }); // widen to f32
    try b.emit(allocator, op.FMul, &.{ 3, 25, 24, 28 }); // times 2.0
    try b.emit(allocator, op.ConvertFToU, &.{ 2, 26, 25 });
    try b.emit(allocator, op.AccessChain, &.{ 11, 27, 16, 14, 20 }); // &out.results[warp]
    try b.emit(allocator, op.Store, &.{ 27, 26 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "an integer to float convert reaches the hardware from a signless module" {
    const allocator = std.testing.allocator;
    var b = try signlessConvertModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.mi",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    // `vfcvt.f.x.v` is the only instruction that converts an integer to a float on this core.
    var converts: usize = 0;
    for (words) |w| {
        if (w & 0x7F != 0b1010111) continue;
        if ((w >> 12) & 0x7 != 0b001) continue;
        if (w >> 26 != 0b010010 or (w >> 15) & 0x1F != 0b00011) continue;
        converts += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), converts);

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    data[0] = 21;
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1 };
    _ = try warp.run(words, &mem, &one, 10_000);
    try std.testing.expectEqual(@as(u32, 42), data[1]);
}

/// `out.results[warp] = out.results[warp]`, with the load and the store optionally marked volatile
/// by the module. A spin loop polling a flag word is written exactly this way.
fn volatileCopyModule(allocator: std.mem.Allocator, mark: bool) !spirv.binary.Builder {
    // ids: void=1 uint=2 v3uint=3 pInV3=4 pInU=5 arr=6 struct=7 pSbStruct=8 pSbU32=9 voidfn=10
    //      c0=11 wid=12 out=13 main=14 entry=15 widptr=16 w=17 pElem=18 v=19.
    var b = try spirv.binary.Builder.init(allocator, 20);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 14, 0x6E69616D, 0, 12, 13 });
    try b.emit(allocator, op.Name, &.{ 13, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 12, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 7, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 2 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 6, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 7, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 10, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 11, 0 });
    try b.emit(allocator, op.Variable, &.{ 4, 12, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 8, 13, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 14, 0, 10 });
    try b.emit(allocator, op.Label, &.{15});
    try b.emit(allocator, op.AccessChain, &.{ 5, 16, 12, 11 });
    try b.emit(allocator, op.Load, &.{ 2, 17, 16 });
    try b.emit(allocator, op.AccessChain, &.{ 9, 18, 13, 11, 17 });
    if (mark) {
        try b.emit(allocator, op.Load, &.{ 2, 19, 18, op.MemoryOperand.@"volatile" });
        try b.emit(allocator, op.Store, &.{ 18, 19, op.MemoryOperand.@"volatile" });
    } else {
        try b.emit(allocator, op.Load, &.{ 2, 19, 18 });
        try b.emit(allocator, op.Store, &.{ 18, 19 });
    }
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "a volatile load and store keep their flag through the lowering" {
    // Nothing in the Glacier pipeline moves memory, so a poll survives a spin loop either way
    // today. Carrying the flag is what keeps that true the day a pass that does move memory is
    // added, and dropping it would be invisible until then.
    const allocator = std.testing.allocator;
    for ([_]bool{ false, true }) |mark| {
        var b = try volatileCopyModule(allocator, mark);
        defer b.deinit(allocator);
        var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
        defer func.deinit();

        var loads: usize = 0;
        var stores: usize = 0;
        for (0..func.blockCount()) |bi| {
            for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |inst| switch (func.opcode(inst)) {
                .load => |l| {
                    if (l.@"volatile") loads += 1;
                },
                .store => |st| {
                    if (st.@"volatile") stores += 1;
                },
                else => {},
            };
        }
        try std.testing.expectEqual(@as(usize, if (mark) 1 else 0), loads);
        try std.testing.expectEqual(@as(usize, if (mark) 1 else 0), stores);

        // Either way it still reaches the hardware, because a Glacier load is a load.
        const words = try glacier.isel.compileKernel(allocator, &func, .{
            .profile = .@"gc1.mi",
            .globals = &.{.{ .name = "out", .address = data_base }},
        });
        defer allocator.free(words);
        try std.testing.expect(words.len > 0);
    }
}

/// Every warp does `out.results[warp] = atomicAdd(&out.counter, 1)`, which is the arrival counter a
/// software barrier across warps is built from: each warp learns how many arrived before it.
fn atomicCounterModule(allocator: std.mem.Allocator, opcode: u16) !spirv.binary.Builder {
    // ids: void=1 uint=2 v3uint=3 pInV3=4 pInU=5 arr=6 struct=7 pSbStruct=8 pSbU32=9 voidfn=10
    //      c0=11 c1=12 scope=13 semantics=14 wid=15 out=16 main=17 entry=18 widptr=19 w=20
    //      pCounter=21 old=22 pElem=23.
    var b = try spirv.binary.Builder.init(allocator, 24);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 17, 0x6E69616D, 0, 15, 16 });
    try b.emit(allocator, op.Name, &.{ 16, 0x0074756F });
    try b.emit(allocator, op.Decorate, &.{ 15, op.Decoration.builtin, op.BuiltIn.workgroup_id });
    try b.emit(allocator, op.MemberDecorate, &.{ 7, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.MemberDecorate, &.{ 7, 1, op.Decoration.offset, 4 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeVector, &.{ 3, 2, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 4, op.StorageClass.input, 3 });
    try b.emit(allocator, op.TypePointer, &.{ 5, op.StorageClass.input, 2 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 6, 2 });
    try b.emit(allocator, op.TypeStruct, &.{ 7, 2, 6 });
    try b.emit(allocator, op.TypePointer, &.{ 8, op.StorageClass.storage_buffer, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 2 });
    try b.emit(allocator, op.TypeFunction, &.{ 10, 1 });
    try b.emit(allocator, op.Constant, &.{ 2, 11, 0 });
    try b.emit(allocator, op.Constant, &.{ 2, 12, 1 });
    // The scope and the semantics are constants, not literals: device scope, relaxed.
    try b.emit(allocator, op.Constant, &.{ 2, 13, op.Scope.device });
    try b.emit(allocator, op.Constant, &.{ 2, 14, 0 });
    try b.emit(allocator, op.Variable, &.{ 4, 15, op.StorageClass.input });
    try b.emit(allocator, op.Variable, &.{ 8, 16, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 17, 0, 10 });
    try b.emit(allocator, op.Label, &.{18});
    try b.emit(allocator, op.AccessChain, &.{ 5, 19, 15, 11 });
    try b.emit(allocator, op.Load, &.{ 2, 20, 19 });
    try b.emit(allocator, op.AccessChain, &.{ 9, 21, 16, 11 }); // &out.counter
    try b.emit(allocator, opcode, &.{ 2, 22, 21, 13, 14, 12 });
    try b.emit(allocator, op.AccessChain, &.{ 9, 23, 16, 12, 20 }); // &out.results[warp]
    try b.emit(allocator, op.Store, &.{ 23, 22 });
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "an atomic add gives every warp a different arrival number" {
    const allocator = std.testing.allocator;
    var b = try atomicCounterModule(allocator, op.AtomicIAdd);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.mi",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    const warps: u32 = 5;
    var data: [8]u32 = @splat(0xFFFF_FFFF);
    data[0] = 0;
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    for (0..warps) |w| {
        var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = warps };
        _ = try warp.run(words, &mem, &one, 10_000);
    }
    // Each warp took the OLD value, so they are 0, 1, 2, ... and the counter ends at the count.
    try std.testing.expectEqual(warps, data[0]);
    for (0..warps) |w| try std.testing.expectEqual(@as(u32, @intCast(w)), data[1 + w]);
}

test "an atomic exchange claims a word and hands back what it replaced" {
    const allocator = std.testing.allocator;
    var b = try atomicCounterModule(allocator, op.AtomicExchange);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.mi",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    var data: [8]u32 = @splat(0xFFFF_FFFF);
    data[0] = 77;
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    for (0..2) |w| {
        var one: warp.Warp = .{ .hart_id = @intCast(w), .hart_count = 2 };
        _ = try warp.run(words, &mem, &one, 10_000);
    }
    // The first warp swapped 1 in and took 77 out; the second took the 1 the first left.
    try std.testing.expectEqual(@as(u32, 1), data[0]);
    try std.testing.expectEqual(@as(u32, 77), data[1]);
    try std.testing.expectEqual(@as(u32, 1), data[2]);
}

/// How many words are `vfwmacc.vv`, and how many are a plain float add, subtract or multiply.
fn countKind(code: []const u32, mac: bool) usize {
    var n: usize = 0;
    for (code) |w| {
        if (w & 0x7F != 0b1010111 or (w >> 12) & 0x7 != 0b001) continue;
        const f6 = w >> 26;
        if (mac) {
            if (f6 == 0x3C) n += 1;
        } else switch (f6) {
            0x00, 0x02, 0x24 => n += 1,
            else => {},
        }
    }
    return n;
}

/// The selection half of the fp32 acceptance test. The execution half lives in Glacier's own
/// harness, driving hand-assembled words into the RTL: it proves the lane computes correctly given
/// the instructions. This proves the path from a module to those instructions PICKS them, which the
/// other half cannot see, and it is what would catch a selector regression.
///
/// Every operand is kept live by a runtime load, because an operation between two float constants
/// is folded by any real frontend long before a selector sees it.
fn fp32AcceptanceModule(allocator: std.mem.Allocator) !spirv.binary.Builder {
    var b = try spirv.binary.Builder.init(allocator, 50);
    errdefer b.deinit(allocator);
    try b.emit(allocator, op.EntryPoint, &.{ op.ExecutionModel.gl_compute, 26, 0x6E69616D, 0, 25 });
    try b.emit(allocator, op.Name, &.{ 25, 0x0074756F });
    try b.emit(allocator, op.MemberDecorate, &.{ 8, 0, op.Decoration.offset, 0 });
    try b.emit(allocator, op.TypeVoid, &.{1});
    try b.emit(allocator, op.TypeInt, &.{ 2, 32, 0 });
    try b.emit(allocator, op.TypeFloat, &.{ 3, 32 });
    try b.emit(allocator, op.TypeFloat, &.{ 4, 16 });
    try b.emit(allocator, op.TypeRuntimeArray, &.{ 7, 3 });
    try b.emit(allocator, op.TypeStruct, &.{ 8, 7 });
    try b.emit(allocator, op.TypePointer, &.{ 9, op.StorageClass.storage_buffer, 8 });
    try b.emit(allocator, op.TypePointer, &.{ 10, op.StorageClass.storage_buffer, 3 });
    try b.emit(allocator, op.TypeFunction, &.{ 11, 1 });
    for (0..7) |i| try b.emit(allocator, op.Constant, &.{ 2, @intCast(12 + i), @intCast(i) });
    // 3 * 2^-25: three quarters of an f32 ulp at 1.0, and NOT representable in f16, so it cannot
    // become a MAC factor and the addition stays a plain float add.
    try b.emit(allocator, op.Constant, &.{ 3, 19, @as(u32, @bitCast(@as(f32, @floatCast(@as(f64, 3.0) / 33554432.0)))) });
    // 3 * 2^-13 and 2^-12, normal halves whose product is exactly the quantity above.
    try b.emit(allocator, op.Constant, &.{ 4, 20, @as(u16, @bitCast(@as(f16, @floatCast(@as(f64, 3.0) / 8192.0)))) });
    try b.emit(allocator, op.Constant, &.{ 4, 21, @as(u16, @bitCast(@as(f16, @floatCast(@as(f64, 1.0) / 4096.0)))) });
    try b.emit(allocator, op.Constant, &.{ 3, 22, @as(u32, @bitCast(@as(f32, 0.1))) });
    try b.emit(allocator, op.Constant, &.{ 3, 23, @as(u32, @bitCast(@as(f32, 2.0))) });
    // 2^20+1: exact in f32 and no f16 holds it, so the MAC cannot claim the multiply below and it
    // stays a widening convert followed by a plain one. A constant that IS an exact half, 4.0 say,
    // turns the whole thing back into a MAC and the slot stops testing the convert.
    try b.emit(allocator, op.Constant, &.{ 3, 24, @as(u32, @bitCast(@as(f32, 1048577.0))) });
    try b.emit(allocator, op.Variable, &.{ 9, 25, op.StorageClass.storage_buffer });
    try b.emit(allocator, op.Function, &.{ 1, 26, 0, 11 });
    try b.emit(allocator, op.Label, &.{27});
    try b.emit(allocator, op.AccessChain, &.{ 10, 28, 25, 12, 12 }); // &out[0]
    try b.emit(allocator, op.Load, &.{ 3, 29, 28 }); // x, the only runtime value
    try b.emit(allocator, op.FConvert, &.{ 3, 30, 20 }); // f32(3 * 2^-13)
    try b.emit(allocator, op.FConvert, &.{ 3, 31, 21 }); // f32(2^-12)
    try b.emit(allocator, op.FMul, &.{ 3, 32, 30, 31 });
    try b.emit(allocator, op.FAdd, &.{ 3, 33, 29, 19 }); // x + 3*2^-25, nearest even
    try b.emit(allocator, op.FAdd, &.{ 3, 34, 29, 32 }); // x + the same quantity as a MAC, truncated
    try b.emit(allocator, op.FMul, &.{ 3, 35, 29, 22 }); // x * 0.1, full fp32
    try b.emit(allocator, op.FSub, &.{ 3, 36, 29, 23 }); // x - 2
    try b.emit(allocator, op.FSub, &.{ 3, 37, 23, 29 }); // 2 - x
    try b.emit(allocator, op.FConvert, &.{ 4, 38, 29 }); // narrow to f16
    try b.emit(allocator, op.FConvert, &.{ 3, 39, 38 }); // and widen back
    try b.emit(allocator, op.FMul, &.{ 3, 40, 39, 24 });
    for ([_]u32{ 33, 34, 35, 36, 37, 40 }, 0..) |v, i| {
        try b.emit(allocator, op.AccessChain, &.{ 10, @intCast(41 + i), 25, 12, @intCast(13 + i) });
        try b.emit(allocator, op.Store, &.{ @intCast(41 + i), v });
    }
    try b.emit(allocator, op.Return, &.{});
    try b.emit(allocator, op.FunctionEnd, &.{});
    return b;
}

test "the fp32 acceptance kernel selects the right instructions from a module" {
    const allocator = std.testing.allocator;
    var b = try fp32AcceptanceModule(allocator);
    defer b.deinit(allocator);
    var func = try spirv.lowerModuleFor(allocator, b.words.items, .{ .buffers_as_globals = true });
    defer func.deinit();
    const words = try glacier.isel.compileKernel(allocator, &func, .{
        .profile = .@"gc1.mi",
        .globals = &.{.{ .name = "out", .address = data_base }},
    });
    defer allocator.free(words);

    // The selection claim, made explicit: ONE MAC, which claimed the widening converts and the
    // multiply feeding the second addition, and five plain float ops for everything else.
    try std.testing.expectEqual(@as(usize, 1), countKind(words, true));
    try std.testing.expectEqual(@as(usize, 5), countKind(words, false));

    var data: [8]u32 = @splat(0xDEAD_BEEF);
    data[0] = @bitCast(@as(f32, 1.0));
    var mem: warp.Memory = .{ .base = data_base, .words = &data };
    var one: warp.Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 4 };
    _ = try warp.run(words, &mem, &one, 10_000);

    // The two rounding modes on ONE quantity: the add carries three quarters of an ulp up, the MAC
    // truncates it away. Equal answers would mean one mux feeds both, which no single-value check
    // can see. The hardware gave these same two words from the instruction stream directly.
    try std.testing.expectEqual(@as(u32, 0x3f800001), data[1]);
    try std.testing.expectEqual(@as(u32, 0x3f800000), data[2]);
    // Full fp32, not the 0x3dccc000 that narrowing 0.1 to an f16 would give.
    try std.testing.expectEqual(@as(u32, 0x3dcccccd), data[3]);
    // Subtract is not commutative, so the order reaches the instruction.
    try std.testing.expectEqual(@as(u32, 0xbf800000), data[4]);
    try std.testing.expectEqual(@as(u32, 0x3f800000), data[5]);
    // Narrowed to f16 and widened back, then multiplied: both converts and a plain multiply.
    try std.testing.expectEqual(@as(u32, 0x49800008), data[6]);
}
