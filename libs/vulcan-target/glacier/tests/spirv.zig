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
