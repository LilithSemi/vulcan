//! Known C library calls with constant inputs. `strlen` of a read-only string becomes its length,
//! and `memcmp(a, b, n)` with a small constant `n`, read only by compares with zero, becomes
//! loads, xors and ors. The driver enables this only for a hosted C library, whose functions
//! have the standard meaning.

const std = @import("std");
const ir = @import("vulcan-ir");
const dce = @import("dce.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Value = ir.function.Value;
const Inst = ir.function.Inst;

pub const Error = std.mem.Allocator.Error;

/// The bytes of a read-only data object, by symbol name, or null when it is not constant.
pub const Constants = struct {
    context: *const anyopaque,
    find: *const fn (context: *const anyopaque, name: []const u8) ?[]const u8,
};

/// The longest `memcmp` this expands. Longer compares stay library calls.
const max_compare_bytes = 16;

pub fn run(allocator: std.mem.Allocator, func: *Function, constants: Constants) Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| {
            const call = switch (func.opcode(inst)) {
                .call => |c| c,
                else => continue,
            };
            if (!std.mem.eql(u8, func.symbolName(call.symbol), "strlen")) continue;
            const args = func.valueList(call.args);
            if (args.len != 1) continue;
            const bytes = constantBytes(func, args[0], constants) orelse continue;
            const len = std.mem.indexOfScalar(u8, bytes, 0) orelse continue;
            func.opcodeMut(inst).* = .{ .iconst = @intCast(len) };
            changed = true;
        }
    }
    var bi: usize = 0;
    while (bi < func.blockCount()) : (bi += 1) {
        const block: Block = @fromBackingInt(@intCast(bi));
        var index: usize = 0;
        while (index < func.blockInsts(block).len) : (index += 1) {
            if (try expandCompare(allocator, func, block, index)) changed = true;
        }
    }
    // A plain integer load from read-only data at a constant offset reads a constant. The data
    // holds target bytes, and every target this runs for is little-endian.
    for (0..func.blockCount()) |load_block| {
        for (func.blockInsts(@fromBackingInt(@intCast(load_block)))) |inst| {
            const load = switch (func.opcode(inst)) {
                .load => |l| l,
                else => continue,
            };
            if (load.@"volatile" or load.mask != null or func.isByteOrderTagged(inst)) continue;
            const result = func.instResult(inst) orelse continue;
            const int = switch (func.types.type_kind(func.valueType(result))) {
                .int => |int| int,
                else => continue,
            };
            if (int.bits % 8 != 0 or int.bits > 64) continue;
            var offset: i64 = 0;
            var ptr = load.ptr;
            if (func.definingInst(ptr)) |def| switch (func.opcode(def)) {
                .arith_imm => |a| if (a.op == .add) {
                    offset = a.imm;
                    ptr = a.lhs;
                },
                else => {},
            };
            const bytes = constantBytes(func, ptr, constants) orelse continue;
            const size = int.bits / 8;
            if (offset < 0 or offset + size > bytes.len) continue;
            const start: usize = @intCast(offset);
            var raw: u64 = 0;
            for (bytes[start .. start + size], 0..) |byte, shift| raw |= @as(u64, byte) << @intCast(shift * 8);
            const value: i64 = if (int.signedness == .signed and int.bits < 64)
                @as(i64, @bitCast(raw << @intCast(64 - int.bits))) >> @intCast(64 - int.bits)
            else
                @bitCast(raw);
            func.opcodeMut(inst).* = .{ .iconst = value };
            changed = true;
        }
    }
    return changed;
}

fn constantBytes(func: *const Function, ptr: Value, constants: Constants) ?[]const u8 {
    const def = func.definingInst(ptr) orelse return null;
    const global = switch (func.opcode(def)) {
        .global_addr => |g| g,
        else => return null,
    };
    if (global.via_got) return null;
    return constants.find(constants.context, func.symbolName(global.symbol));
}

fn constOf(func: *const Function, v: Value) ?i64 {
    const def = func.definingInst(v) orelse return null;
    return switch (func.opcode(def)) {
        .iconst => |c| c,
        else => null,
    };
}

/// Expand the `memcmp` at `index` of `block` when every reader compares it with zero for
/// equality. The difference of each chunk is `a ^ b`, and the ors of all chunks are zero only
/// when every byte matches.
fn expandCompare(allocator: std.mem.Allocator, func: *Function, block: Block, index: usize) Error!bool {
    const inst = func.blockInsts(block)[index];
    const call = switch (func.opcode(inst)) {
        .call => |c| c,
        else => return false,
    };
    if (!std.mem.eql(u8, func.symbolName(call.symbol), "memcmp")) return false;
    const args = func.valueList(call.args);
    if (args.len != 3) return false;
    const n = constOf(func, args[2]) orelse return false;
    if (n < 1 or n > max_compare_bytes) return false;
    const result = func.instResult(inst) orelse return false;

    const counts = try allocator.alloc(u32, func.valueCount());
    defer allocator.free(counts);
    dce.countUses(func, counts);
    var readers: std.ArrayList(Inst) = .empty;
    defer readers.deinit(allocator);
    for (0..func.blockCount()) |bi| for (func.blockInsts(@fromBackingInt(@intCast(bi)))) |user| switch (func.opcode(user)) {
        .icmp => |c| if (c.lhs == result or c.rhs == result) {
            const other = if (c.lhs == result) c.rhs else c.lhs;
            if ((c.op != .eq and c.op != .ne) or other == result or constOf(func, other) != 0) return false;
            try readers.append(allocator, user);
        },
        else => {},
    };
    if (readers.items.len == 0 or readers.items.len != counts[@backingInt(result)]) return false;

    const a = args[0];
    const b = args[1];
    const ptr_t = func.valueType(a);
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    var made: std.ArrayList(Inst) = .empty;
    defer made.deinit(allocator);
    var diff: ?Value = null;
    var offset: i64 = 0;
    while (offset < n) {
        const remaining = n - offset;
        const bytes: u16 = if (remaining >= 8) 8 else if (remaining >= 4) 4 else if (remaining >= 2) 2 else 1;
        const chunk_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = bytes * 8 } });
        var chunk: [2]Value = undefined;
        for ([_]Value{ a, b }, &chunk) |base, *loaded| {
            const at = if (offset == 0) base else try create(func, &made, allocator, ptr_t, .{ .arith_imm = .{ .op = .add, .lhs = base, .imm = offset } });
            loaded.* = try create(func, &made, allocator, chunk_t, .{ .load = .{ .ptr = at } });
        }
        var d = try create(func, &made, allocator, chunk_t, .{ .arith = .{ .op = .bit_xor, .lhs = chunk[0], .rhs = chunk[1] } });
        if (bytes != 8) d = try create(func, &made, allocator, u64_t, .{ .convert = .{ .value = d } });
        diff = if (diff) |prev| try create(func, &made, allocator, u64_t, .{ .arith = .{ .op = .bit_or, .lhs = prev, .rhs = d } }) else d;
        offset += bytes;
    }
    const zero = try create(func, &made, allocator, u64_t, .{ .iconst = 0 });
    for (readers.items) |user| {
        const cmp = &func.opcodeMut(user).icmp;
        cmp.lhs = diff.?;
        cmp.rhs = zero;
    }
    const list = func.blockInstsMut(block);
    _ = list.orderedRemove(index);
    try list.insertSlice(func.allocator, index, made.items);
    return true;
}

fn create(func: *Function, made: *std.ArrayList(Inst), allocator: std.mem.Allocator, ty: ir.types.Type, op: ir.function.Opcode) Error!Value {
    const value = try func.createInst(ty, op);
    try made.append(allocator, func.definingInst(value).?);
    return value;
}

const testing = std.testing;

const TestStrings = struct {
    fn find(_: *const anyopaque, name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, "word")) "value\x00" else null;
    }
};

test "strlen of a constant string folds, and the memcmp it sizes becomes loads" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const text = try func.appendBlockParam(b, ptr_t);
    const word = try func.appendInst(b, ptr_t, .{ .global_addr = .{ .symbol = try func.internSymbol("word") } });
    const len = try func.appendInst(b, u64_t, .{ .call = .{ .symbol = try func.internSymbol("strlen"), .args = try func.internValues(&.{word}) } });
    const diff = try func.appendInst(b, i32_t, .{ .call = .{ .symbol = try func.internSymbol("memcmp"), .args = try func.internValues(&.{ text, word, len }) } });
    const zero = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    const same = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .eq, .lhs = diff, .rhs = zero } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(same) });

    const constants = Constants{ .context = undefined, .find = TestStrings.find };
    try testing.expect(try run(allocator, &func, constants));
    try testing.expectEqual(@as(?i64, 5), constOf(&func, len));
    var loads: usize = 0;
    for (func.blockInsts(b)) |inst| switch (func.opcode(inst)) {
        .call => |c| try testing.expect(!std.mem.eql(u8, func.symbolName(c.symbol), "memcmp")),
        .load => loads += 1,
        else => {},
    };
    // A 4-byte and a 1-byte chunk of the text. The chunks of the constant word fold.
    try testing.expectEqual(@as(usize, 2), loads);
    var folded = false;
    for (func.blockInsts(b)) |inst| switch (func.opcode(inst)) {
        .iconst => |c| folded = folded or c == 0x756c6176, // "valu"
        else => {},
    };
    try testing.expect(folded);
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try testing.expect(diags.ok());
}

test "a memcmp whose sign is read stays a call" {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const b = try func.appendBlock();
    const x = try func.appendBlockParam(b, ptr_t);
    const y = try func.appendBlockParam(b, ptr_t);
    const four = try func.appendInst(b, u64_t, .{ .iconst = 4 });
    const diff = try func.appendInst(b, i32_t, .{ .call = .{ .symbol = try func.internSymbol("memcmp"), .args = try func.internValues(&.{ x, y, four }) } });
    const zero = try func.appendInst(b, i32_t, .{ .iconst = 0 });
    const below = try func.appendInst(b, bool_t, .{ .icmp = .{ .op = .lt, .lhs = diff, .rhs = zero } });
    func.setTerminator(b, .{ .ret = ir.function.Ret.one(below) });

    const constants = Constants{ .context = undefined, .find = TestStrings.find };
    try testing.expect(!try run(allocator, &func, constants));
}
