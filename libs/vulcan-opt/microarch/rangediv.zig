//! Range-proven narrowing of widened unsigned division sequences.

const std = @import("std");
const ir = @import("vulcan-ir");
const pass = @import("../pass.zig");

const Function = ir.function.Function;
const Block = ir.function.Block;
const Inst = ir.function.Inst;
const Value = ir.function.Value;

pub const pass_def = pass.Pass{ .name = "rangediv", .run = run, .preserves_cfg = true };

const Range = struct { min: i64, max: i64 };
const Candidate = struct { source: Value };
const max_reduced_dividend = 81_919;

pub fn run(allocator: std.mem.Allocator, func: *Function, analyses: *pass.Analyses) pass.Error!bool {
    _ = analyses;
    const uses = try useCounts(allocator, func);
    defer allocator.free(uses);
    var changed = false;
    for (0..func.blockCount()) |bi| {
        var block_changed = false;
        const block: Block = @fromBackingInt(@intCast(bi));
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        for (original) |inst| {
            const candidate = narrowDivideCandidate(func, block, inst, uses) orelse {
                try out.append(allocator, inst);
                continue;
            };
            const signed_t = func.valueType(candidate.source);
            const unsigned_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
            const magic = try func.createInst(signed_t, .{ .iconst = 52_429 });
            try out.append(allocator, func.definingInst(magic).?);
            const product = try func.createInst(signed_t, .{ .arith = .{ .op = .mul, .lhs = candidate.source, .rhs = magic } });
            try out.append(allocator, func.definingInst(product).?);
            const unsigned_product = try func.createInst(unsigned_t, .{ .convert = .{ .value = product } });
            try out.append(allocator, func.definingInst(unsigned_product).?);
            const quotient = try func.createInst(unsigned_t, .{ .arith_imm = .{ .op = .shr, .lhs = unsigned_product, .imm = 18 } });
            try out.append(allocator, func.definingInst(quotient).?);
            func.opcodeMut(inst).* = .{ .convert = .{ .value = quotient } };
            try out.append(allocator, inst);
            changed = true;
            block_changed = true;
        }
        if (block_changed) try func.setBlockInsts(block, out.items);
    }
    return changed;
}

fn narrowDivideCandidate(func: *const Function, block: Block, final_inst: Inst, uses: []const u32) ?Candidate {
    const final = func.instResult(final_inst) orelse return null;
    if (!intType(func, final, .signed, 32)) return null;
    const quotient32 = switch (func.opcode(final_inst)) {
        .convert => |convert| convert.value,
        else => return null,
    };
    if (!intType(func, quotient32, .unsigned, 32)) return null;
    const quotient64 = conversionSource(func, quotient32, .unsigned, 64) orelse return null;
    const post_shift = shiftValue(func, quotient64, 2, .unsigned, 64) orelse return null;
    const product = shiftValue(func, post_shift, 32, .unsigned, 64) orelse return null;
    const multiply = switch (func.opcode(func.definingInst(product) orelse return null)) {
        .arith => |arith| if (arith.op == .mul) arith else return null,
        else => return null,
    };
    const magic: i64 = 3_435_973_837;
    const widened = if (constantValue(func, multiply.rhs) == magic)
        multiply.lhs
    else if (constantValue(func, multiply.lhs) == magic)
        multiply.rhs
    else
        return null;
    const unsigned_source = conversionSource(func, widened, .unsigned, 32) orelse return null;
    const source = convertedValueAnySign(func, unsigned_source, 32) orelse return null;
    if (!intType(func, source, .signed, 32)) return null;
    const range = rangeOf(func, source, 0) orelse return null;
    if (range.max > max_reduced_dividend or uses[@backingInt(final)] != 1 or !guardedPositiveUse(func, block, final, source)) return null;
    return .{ .source = source };
}

fn useCounts(allocator: std.mem.Allocator, func: *const Function) ![]u32 {
    const counts = try allocator.alloc(u32, func.valueCount());
    @memset(counts, 0);
    const bump = struct {
        fn f(values: []u32, value: Value) void {
            values[@backingInt(value)] += 1;
        }
    }.f;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .iconst, .fconst, .fconst128, .alloca, .global_addr, .barrier => {},
            .arith => |op| {
                bump(counts, op.lhs);
                bump(counts, op.rhs);
            },
            .arith_imm => |op| bump(counts, op.lhs),
            .icmp => |op| {
                bump(counts, op.lhs);
                bump(counts, op.rhs);
            },
            .select => |op| {
                bump(counts, op.cond);
                bump(counts, op.then);
                bump(counts, op.@"else");
            },
            .extract => |op| bump(counts, op.aggregate),
            .convert => |op| bump(counts, op.value),
            .decode_low_float, .encode_low_float => |op| bump(counts, op.value),
            .dequantize_nvfp4, .quantize_nvfp4 => |op| {
                bump(counts, op.value);
                bump(counts, op.block_scale);
                bump(counts, op.global_scale);
            },
            .unary => |op| bump(counts, op.value),
            .load => |op| {
                bump(counts, op.ptr);
                if (op.mask) |mask| bump(counts, mask);
            },
            .store => |op| {
                bump(counts, op.value);
                bump(counts, op.ptr);
                if (op.mask) |mask| bump(counts, mask);
            },
            .prefetch => |op| bump(counts, op.ptr),
            .va_start => |op| bump(counts, op.list),
            .va_arg => |op| bump(counts, op.list),
            .va_end => |op| bump(counts, op.list),
            .dot => |op| {
                bump(counts, op.acc);
                bump(counts, op.a);
                bump(counts, op.b);
            },
            .reduce => |op| bump(counts, op.vector),
            .splat => |op| bump(counts, op.scalar),
            .matmul => |op| {
                bump(counts, op.a);
                bump(counts, op.b);
                bump(counts, op.c);
            },
            .atomic_rmw => |op| {
                bump(counts, op.ptr);
                bump(counts, op.value);
                if (op.compare) |compare| bump(counts, compare);
            },
            .struct_new => |op| for (func.valueList(op.fields)) |value| bump(counts, value),
            .call => |op| {
                if (op.ret_dest) |dest| bump(counts, dest);
                for (func.valueList(op.args)) |value| bump(counts, value);
            },
            .call_indirect => |op| {
                bump(counts, op.target);
                if (op.ret_dest) |dest| bump(counts, dest);
                for (func.valueList(op.args)) |value| bump(counts, value);
            },
            .@"if" => |op| {
                bump(counts, op.cond);
                for (func.valueList(op.then.args)) |value| bump(counts, value);
                for (func.valueList(op.@"else".args)) |value| bump(counts, value);
            },
        };
        if (func.terminator(block)) |term| switch (term) {
            .ret => |ret| for (ret.slice()) |value| bump(counts, value),
            .jump => |jump| for (func.blockArgs(jump)) |value| bump(counts, value),
        };
    }
    return counts;
}

fn guardedPositiveUse(func: *const Function, block: Block, quotient: Value, source: Value) bool {
    for (func.blockInsts(block)) |inst| {
        const select = switch (func.opcode(inst)) {
            .select => |select| select,
            else => continue,
        };
        if (select.then != quotient) continue;
        const cmp = switch (func.opcode(func.definingInst(select.cond) orelse continue)) {
            .icmp => |cmp| cmp,
            else => continue,
        };
        if (cmp.lhs == source) {
            const bound = constantValue(func, cmp.rhs) orelse continue;
            if ((cmp.op == .gt and bound >= -1) or (cmp.op == .ge and bound >= 0)) return true;
        }
        if (cmp.rhs == source) {
            const bound = constantValue(func, cmp.lhs) orelse continue;
            if ((cmp.op == .lt and bound >= -1) or (cmp.op == .le and bound >= 0)) return true;
        }
    }
    return false;
}

fn rangeOf(func: *const Function, value: Value, depth: u8) ?Range {
    if (depth >= 24) return null;
    const inst = func.definingInst(value) orelse return incomingRange(func, value, depth + 1);
    return switch (func.opcode(inst)) {
        .iconst => |constant| .{ .min = constant, .max = constant },
        .convert => |convert| convertedRange(func, value, rangeOf(func, convert.value, depth + 1) orelse return null),
        .arith => |arith| arithmeticRange(func, arith.op, arith.lhs, arith.rhs, depth + 1),
        .arith_imm => |arith| arithmeticImmediateRange(func, arith.op, arith.lhs, arith.imm, depth + 1),
        else => null,
    };
}

fn incomingRange(func: *const Function, value: Value, depth: u8) ?Range {
    var target: ?Block = null;
    var parameter_index: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockParams(block), 0..) |parameter, index| if (parameter == value) {
            target = block;
            parameter_index = index;
        };
    }
    const destination = target orelse return null;
    var result: ?Range = null;
    var incoming: usize = 0;
    for (0..func.blockCount()) |bi| {
        const block: Block = @fromBackingInt(@intCast(bi));
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .@"if" => |cf| {
                if (!mergeIncomingRange(func, cf.then, destination, parameter_index, depth, &result, &incoming)) return null;
                if (!mergeIncomingRange(func, cf.@"else", destination, parameter_index, depth, &result, &incoming)) return null;
            },
            else => {},
        };
        if (func.terminator(block)) |term| switch (term) {
            .jump => |jump| if (!mergeIncomingRange(func, jump, destination, parameter_index, depth, &result, &incoming)) return null,
            .ret => {},
        };
    }
    return if (incoming == 0) null else result;
}

fn mergeIncomingRange(func: *const Function, jump: ir.function.Jump, target: Block, index: usize, depth: u8, result: *?Range, incoming: *usize) bool {
    if (jump.target != target) return true;
    const args = func.blockArgs(jump);
    if (index >= args.len) return false;
    const next = rangeOf(func, args[index], depth) orelse return false;
    result.* = if (result.*) |current| .{ .min = @min(current.min, next.min), .max = @max(current.max, next.max) } else next;
    incoming.* += 1;
    return true;
}

fn convertedRange(func: *const Function, result: Value, source: Range) ?Range {
    const int = switch (func.types.type_kind(func.valueType(result))) {
        .int => |int| int,
        else => return null,
    };
    if (int.bits > 63) return source;
    const max: i64 = if (int.signedness == .signed)
        (@as(i64, 1) << @intCast(int.bits - 1)) - 1
    else
        (@as(i64, 1) << @intCast(int.bits)) - 1;
    const min: i64 = if (int.signedness == .signed) -(@as(i64, 1) << @intCast(int.bits - 1)) else 0;
    if (source.min < min or source.max > max) return null;
    return source;
}

fn arithmeticRange(func: *const Function, op: ir.function.BinOp, lhs: Value, rhs: Value, depth: u8) ?Range {
    if (op == .bit_and) {
        if (constantValue(func, rhs)) |mask| if (mask >= 0) return .{ .min = 0, .max = mask };
        if (constantValue(func, lhs)) |mask| if (mask >= 0) return .{ .min = 0, .max = mask };
        return null;
    }
    const left = rangeOf(func, lhs, depth) orelse return null;
    const right = rangeOf(func, rhs, depth) orelse return null;
    return switch (op) {
        .add => .{ .min = std.math.add(i64, left.min, right.min) catch return null, .max = std.math.add(i64, left.max, right.max) catch return null },
        .sub => .{ .min = std.math.sub(i64, left.min, right.max) catch return null, .max = std.math.sub(i64, left.max, right.min) catch return null },
        else => null,
    };
}

fn arithmeticImmediateRange(func: *const Function, op: ir.function.BinOp, lhs: Value, immediate: i64, depth: u8) ?Range {
    if (op == .bit_and and immediate >= 0) return .{ .min = 0, .max = immediate };
    const left = rangeOf(func, lhs, depth) orelse return null;
    return switch (op) {
        .add => .{ .min = std.math.add(i64, left.min, immediate) catch return null, .max = std.math.add(i64, left.max, immediate) catch return null },
        .sub => .{ .min = std.math.sub(i64, left.min, immediate) catch return null, .max = std.math.sub(i64, left.max, immediate) catch return null },
        else => null,
    };
}

fn constantValue(func: *const Function, value: Value) ?i64 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |constant| constant,
        else => null,
    };
}

fn conversionSource(func: *const Function, value: Value, signedness: std.builtin.Signedness, bits: u16) ?Value {
    const inst = func.definingInst(value) orelse return null;
    const source = switch (func.opcode(inst)) {
        .convert => |convert| convert.value,
        else => null,
    };
    return if (source) |candidate| if (intType(func, candidate, signedness, bits)) candidate else null else null;
}

fn convertedValueAnySign(func: *const Function, value: Value, bits: u16) ?Value {
    const inst = func.definingInst(value) orelse return null;
    const source = switch (func.opcode(inst)) {
        .convert => |convert| convert.value,
        else => return null,
    };
    return switch (func.types.type_kind(func.valueType(source))) {
        .int => |int| if (int.bits == bits) source else null,
        else => null,
    };
}

fn shiftValue(func: *const Function, value: Value, amount: i64, signedness: std.builtin.Signedness, bits: u16) ?Value {
    if (!intType(func, value, signedness, bits)) return null;
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .arith_imm => |arith| if (arith.op == .shr and arith.imm == amount) arith.lhs else null,
        else => null,
    };
}

fn intType(func: *const Function, value: Value, signedness: std.builtin.Signedness, bits: u16) bool {
    return switch (func.types.type_kind(func.valueType(value))) {
        .int => |int| int.signedness == signedness and int.bits == bits,
        else => false,
    };
}

test "narrows guarded division by five under a proved bound" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const bool_t = try func.types.intern(.bool);
    const block = try func.appendBlock();
    const input = try func.appendBlockParam(block, i32_t);
    const bounded = try func.appendInst(block, i32_t, .{ .arith_imm = .{ .op = .bit_and, .lhs = input, .imm = 2047 } });
    const source = try func.appendInst(block, i32_t, .{ .arith_imm = .{ .op = .sub, .lhs = bounded, .imm = 1024 } });
    const hundred = try func.appendInst(block, i32_t, .{ .iconst = 100 });
    const condition = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .gt, .lhs = source, .rhs = hundred } });
    const unsigned_source = try func.appendInst(block, u32_t, .{ .convert = .{ .value = source } });
    const widened = try func.appendInst(block, u64_t, .{ .convert = .{ .value = unsigned_source } });
    const wide_magic = try func.appendInst(block, u64_t, .{ .iconst = 3_435_973_837 });
    const product = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = widened, .rhs = wide_magic } });
    const high = try func.appendInst(block, u64_t, .{ .arith_imm = .{ .op = .shr, .lhs = product, .imm = 32 } });
    const shifted = try func.appendInst(block, u64_t, .{ .arith_imm = .{ .op = .shr, .lhs = high, .imm = 2 } });
    const quotient_u32 = try func.appendInst(block, u32_t, .{ .convert = .{ .value = shifted } });
    const quotient = try func.appendInst(block, i32_t, .{ .convert = .{ .value = quotient_u32 } });
    const times_five = try func.appendInst(block, i32_t, .{ .arith_imm = .{ .op = .mul, .lhs = source, .imm = 5 } });
    const selected = try func.appendInst(block, i32_t, .{ .select = .{ .cond = condition, .then = quotient, .@"else" = times_five } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(selected) });

    var analyses = pass.Analyses{ .allocator = allocator, .func = &func };
    defer analyses.deinit();
    try std.testing.expect(try run(allocator, &func, &analyses));
    var found_reduced_magic = false;
    for (func.blockInsts(block)) |inst| {
        if (func.opcode(inst) == .iconst and func.opcode(inst).iconst == 52_429) found_reduced_magic = true;
    }
    try std.testing.expect(found_reduced_magic);
}

test "reduced reciprocal fits its narrowed product throughout the proved domain" {
    var value: u32 = 0;
    while (value <= max_reduced_dividend) : (value += 1) {
        try std.testing.expectEqual(value / 5, @as(u32, @intCast((@as(u64, value) * 52_429) >> 18)));
    }
    const first_truncation: u32 = max_reduced_dividend + 1;
    const low_product: u32 = @truncate(@as(u64, first_truncation) * 52_429);
    try std.testing.expect(low_product >> 18 != first_truncation / 5);
}
