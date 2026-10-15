//! Typed access to the `vulcan.gpu` attribute namespace. Every layer reads and writes kernel
//! metadata through this file, so the namespace and key strings appear once.
//!
//! The IR stores attributes in a flat list and `addAttr` appends without replacing, so a key
//! can appear more than once on the same target. Every getter here returns the FIRST match,
//! which is the value the first writer set. There is no remove operation in the IR.

const std = @import("std");
const ir = @import("vulcan-ir");
const builtin_mod = @import("builtin.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Builtin = builtin_mod.Builtin;

pub const Error = std.mem.Allocator.Error;

/// The attribute namespace every key below lives in.
pub const namespace = "vulcan.gpu";

const key_builtin = "builtin";
const key_local_size = [3][]const u8{ "local_size_x", "local_size_y", "local_size_z" };
const key_shared_bytes = "shared_bytes";
const key_barrier_scope = "scope";
const key_cluster_barriers = "cluster_barriers";
const scope_cluster = "cluster";

/// The first `vulcan.gpu` integer attribute with `key` on `target`, or null when absent.
fn intAttr(func: *const Function, target: ir.function.AttrTarget, key: []const u8) ?i64 {
    var it = func.attributesOf(target);
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, namespace)) continue;
            if (!std.mem.eql(u8, c.key, key)) continue;
            return switch (c.value) {
                .int => |n| n,
                .flag, .string => null,
            };
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };
    return null;
}

fn addInt(func: *Function, target: ir.function.AttrTarget, key: []const u8, n: i64) Error!void {
    try func.addAttr(target, .{ .custom = .{ .namespace = namespace, .key = key, .value = .{ .int = n } } });
}

/// Tag `value` as sourced from hardware rather than from the parameter block.
pub fn setBuiltin(func: *Function, value: Value, b: Builtin) Error!void {
    try addInt(func, .{ .value = value }, key_builtin, @backingInt(b));
}

/// The builtin `value` carries, or null when it is an ordinary parameter.
///
/// The stored payload is UNTRUSTED: it survives a text round trip and any frontend can write
/// it, so an out-of-range number must not become an invalid enum. `std.enums.fromInt` returns
/// null instead, and the caller treats that as "not a builtin".
pub fn builtinOf(func: *const Function, value: Value) ?Builtin {
    const raw = intAttr(func, .{ .value = value }, key_builtin) orelse return null;
    if (raw < 0 or raw > std.math.maxInt(u16)) return null;
    return std.enums.fromInt(Builtin, @as(u16, @intCast(raw)));
}

/// Record the workgroup size the kernel declares.
pub fn setLocalSize(func: *Function, size: [3]u32) Error!void {
    for (size, key_local_size) |n, key| try addInt(func, .func, key, n);
}

/// The declared workgroup size. Each axis defaults to 1 when the frontend recorded none,
/// which is the correct identity for a kernel that declares no size.
pub fn localSize(func: *const Function) [3]u32 {
    var out: [3]u32 = .{ 1, 1, 1 };
    for (&out, key_local_size) |*slot, key| {
        const raw = intAttr(func, .func, key) orelse continue;
        if (raw > 0 and raw <= std.math.maxInt(u32)) slot.* = @intCast(raw);
    }
    return out;
}

/// Record the workgroup shared memory the kernel needs, in bytes.
pub fn setSharedBytes(func: *Function, bytes: u32) Error!void {
    try addInt(func, .func, key_shared_bytes, bytes);
}

/// The workgroup shared memory the kernel needs, in bytes. Zero when it needs none.
pub fn sharedBytes(func: *const Function) u32 {
    const raw = intAttr(func, .func, key_shared_bytes) orelse return 0;
    if (raw < 0 or raw > std.math.maxInt(u32)) return 0;
    return @intCast(raw);
}

test "a builtin tag round-trips through the attribute store" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try setBuiltin(&func, v, .block_id_y);
    try std.testing.expectEqual(@as(?Builtin, .block_id_y), builtinOf(&func, v));
}

test "an untagged parameter reports no builtin" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "an out-of-range builtin payload reports null instead of an invalid enum" {
    // Suspicious case: the payload is untrusted, so a bad number must not become a Builtin.
    // @enumFromInt here would be undefined behaviour.
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = namespace,
        .key = "builtin",
        .value = .{ .int = 9999 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "a negative builtin payload reports null" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = namespace,
        .key = "builtin",
        .value = .{ .int = -1 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

test "local size round-trips all three axes" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try setLocalSize(&func, .{ 64, 2, 1 });
    try std.testing.expectEqual([3]u32{ 64, 2, 1 }, localSize(&func));
}

test "local size defaults to one on every axis when unset" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, localSize(&func));
}

test "shared bytes round-trips and defaults to zero" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expectEqual(@as(u32, 0), sharedBytes(&func));
    try setSharedBytes(&func, 4096);
    try std.testing.expectEqual(@as(u32, 4096), sharedBytes(&func));
}

test "a foreign namespace with the same key is ignored" {
    // Suspicious case: the attribute bag is open, so another layer may use the key "builtin".
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try func.appendBlock();
    const v = try func.appendBlockParam(b, t);

    try func.addAttr(.{ .value = v }, .{ .custom = .{
        .namespace = "other.layer",
        .key = "builtin",
        .value = .{ .int = 0 },
    } });
    try std.testing.expectEqual(@as(?Builtin, null), builtinOf(&func, v));
}

/// Widen a `barrier` to every warp of one core, which is a scope above a workgroup on a target
/// whose workgroup is one warp.
///
/// It is an ATTRIBUTE and not a `BarrierScope` value, which is a deliberate trade and carries a
/// cost worth stating. A scope value makes every exhaustive switch on it fail to compile until it
/// is answered; an attribute does not, so a backend that does not read this emits the NARROWER
/// barrier the opcode names and nothing reports it. Two things hold that down:
///
///   - every backend that emits a real barrier refuses one carrying this. NVIDIA is the only other
///     one today, where `BAR.SYNC` would synchronise a CTA for a kernel that asked for more.
///   - `setClusterBarriers` records how many the kernel had before any pass ran, and a backend
///     compares that with how many it emitted. An attribute keyed to an INSTRUCTION is fragile in
///     a way a flag in the opcode is not: a pass that clones or rebuilds the instruction can drop
///     it, and the count is what turns that into a refusal rather than a narrower barrier.
pub fn setClusterBarrier(func: *Function, inst: ir.function.Inst) Error!void {
    try func.addAttr(.{ .inst = inst }, .{ .custom = .{
        .namespace = namespace,
        .key = key_barrier_scope,
        .value = .{ .string = scope_cluster },
    } });
}

/// Whether `inst` is a barrier widened to every warp of one core.
pub fn isClusterBarrier(func: *const Function, inst: ir.function.Inst) bool {
    var it = func.attributesOf(.{ .inst = inst });
    while (it.next()) |attr| switch (attr) {
        .custom => |c| {
            if (!std.mem.eql(u8, c.namespace, namespace)) continue;
            if (!std.mem.eql(u8, c.key, key_barrier_scope)) continue;
            switch (c.value) {
                .string => |s| if (std.mem.eql(u8, s, scope_cluster)) return true,
                .int, .flag => {},
            }
        },
        .@"inline", .noreturn, .cold, .@"align", .endian => {},
    };
    return false;
}

/// Record how many cluster barriers the kernel had before any pass ran. See `setClusterBarrier`
/// for why the count exists.
pub fn setClusterBarriers(func: *Function, n: u32) Error!void {
    try addInt(func, .func, key_cluster_barriers, n);
}

/// How many cluster barriers the kernel had before any pass ran, or null when nothing recorded it.
/// A backend that finds a count and emits fewer has lost one to a pass.
pub fn clusterBarriers(func: *const Function) ?u32 {
    const raw = intAttr(func, .func, key_cluster_barriers) orelse return null;
    if (raw < 0 or raw > std.math.maxInt(u32)) return null;
    return @intCast(raw);
}

/// Count the cluster barriers in `func` as it stands.
pub fn countClusterBarriers(func: *const Function) u32 {
    var n: u32 = 0;
    for (0..func.blockCount()) |bi| {
        const b: ir.function.Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(b)) |inst| {
            if (func.opcode(inst) != .barrier) continue;
            if (isClusterBarrier(func, inst)) n += 1;
        }
    }
    return n;
}

test "a cluster barrier and its count round-trip through the attribute store" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    const b = try func.appendBlock();
    const plain = try func.appendStmtRaw(b, .{ .barrier = .{ .scope = .workgroup } });
    const wide = try func.appendStmtRaw(b, .{ .barrier = .{ .scope = .workgroup } });
    try setClusterBarrier(&func, wide);

    try std.testing.expect(!isClusterBarrier(&func, plain));
    try std.testing.expect(isClusterBarrier(&func, wide));
    try std.testing.expectEqual(@as(u32, 1), countClusterBarriers(&func));

    try std.testing.expectEqual(@as(?u32, null), clusterBarriers(&func));
    try setClusterBarriers(&func, countClusterBarriers(&func));
    try std.testing.expectEqual(@as(?u32, 1), clusterBarriers(&func));
}
