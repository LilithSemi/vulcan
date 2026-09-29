//! IR-to-IR expansions of operations a backend cannot lower, each run once before instruction
//! selection so that backend's isel never meets one. `expandMulh` rewrites the `mulh` BinOp into
//! half-width limbs. `expandMatmul` rewrites the et-soc tensor-tile `matmul` into a scalar loop
//! nest, which is what lets a matmul execute anywhere at all.
//!
//! `expandMulh` expands the high half of a full-width product into plain multiplies, shifts, and
//! masks, for backends that have no high-multiply instruction (wasm/spirv/x86/x86_64/nvidia). The
//! two native scalar backends (aarch64 smulh/umulh, riscv64 mulh/mulhu) lower `mulh` directly and
//! never call this. Producing `mulh` is the magic-number divide lowering's job (`strength.zig`); a
//! backend without native support runs this once before instruction selection so its isel never
//! meets a `mulh`.
//!
//! The high half is computed from half-width limbs. For a W-bit value the limbs are H = W/2 bits:
//! splitting a = ahi*2^H + alo and b likewise, the full product's high W bits are
//!   hihi + (lohi >> H) + (hilo >> H) + (((lolo >> H) + (lohi & m) + (hilo & m)) >> H)
//! where lolo = alo*blo, lohi = alo*bhi, hilo = ahi*blo, hihi = ahi*bhi, m = 2^H - 1. Every shift is
//! masked back to H bits, so an arithmetic right shift is fine even on a signed type (the sign fill
//! lands above bit H and is masked away), which is why no unsigned reinterpret is needed. For a
//! signed `mulh` the unsigned high half is corrected by `- (a<0 ? b : 0) - (b<0 ? a : 0)`.

const std = @import("std");
const function = @import("function.zig");
const types = @import("types.zig");
const verify = @import("verify.zig");
const bitcode = @import("bitcode.zig");
const nvfp4 = @import("nvfp4.zig");

const Function = function.Function;
const Value = function.Value;
const Block = function.Block;
const Inst = function.Inst;
const BinOp = function.BinOp;
const MatMul = function.MatMul;

const NumericBuilder = struct {
    func: *Function,
    out: *std.ArrayList(Inst),
    allocator: std.mem.Allocator,
    u8_t: types.Type,
    u16_t: types.Type,
    u32_t: types.Type,
    f32_t: types.Type,
    bool_t: types.Type,

    fn emit(self: NumericBuilder, ty: types.Type, opcode: function.Opcode) std.mem.Allocator.Error!Value {
        const value = try self.func.createInst(ty, opcode);
        try self.out.append(self.allocator, self.func.definingInst(value).?);
        return value;
    }

    fn constant(self: NumericBuilder, value: u32) std.mem.Allocator.Error!Value {
        return self.emit(self.u32_t, .{ .iconst = value });
    }

    fn binary(self: NumericBuilder, op: BinOp, lhs: Value, rhs: Value) std.mem.Allocator.Error!Value {
        return self.emit(self.u32_t, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    fn floatBinary(self: NumericBuilder, op: BinOp, lhs: Value, rhs: Value) std.mem.Allocator.Error!Value {
        std.debug.assert(op == .mul or op == .div);
        std.debug.assert(self.func.valueType(lhs) == self.f32_t);
        std.debug.assert(self.func.valueType(rhs) == self.f32_t);
        return self.emit(self.f32_t, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    fn compare(self: NumericBuilder, op: function.CmpOp, lhs: Value, rhs: Value) std.mem.Allocator.Error!Value {
        return self.emit(self.bool_t, .{ .icmp = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    }

    fn select(self: NumericBuilder, cond: Value, then_value: Value, else_value: Value) std.mem.Allocator.Error!Value {
        return self.emit(self.u32_t, .{ .select = .{ .cond = cond, .then = then_value, .@"else" = else_value } });
    }

    fn convert(self: NumericBuilder, ty: types.Type, value: Value) std.mem.Allocator.Error!Value {
        return self.emit(ty, .{ .convert = .{ .value = value } });
    }

    fn reinterpret(self: NumericBuilder, ty: types.Type, value: Value) std.mem.Allocator.Error!Value {
        return self.emit(ty, .{ .unary = .{ .op = .reinterpret, .value = value } });
    }

    fn masked(self: NumericBuilder, value: Value, mask: u32) std.mem.Allocator.Error!Value {
        return self.binary(.bit_and, value, try self.constant(mask));
    }

    fn shifted(self: NumericBuilder, op: BinOp, value: Value, amount: u32) std.mem.Allocator.Error!Value {
        return self.binary(op, value, try self.constant(amount));
    }

    fn equalConstant(self: NumericBuilder, value: Value, expected: u32) std.mem.Allocator.Error!Value {
        return self.compare(.eq, value, try self.constant(expected));
    }

    fn roundRight(self: NumericBuilder, value: Value, shift: Value) std.mem.Allocator.Error!Value {
        const one = try self.constant(1);
        const retained = try self.binary(.shr, value, shift);
        const shifted_one = try self.binary(.shl, one, shift);
        const mask = try self.binary(.sub, shifted_one, one);
        const discarded = try self.binary(.bit_and, value, mask);
        const shift_minus_one = try self.binary(.sub, shift, one);
        const halfway = try self.binary(.shl, one, shift_minus_one);
        const greater = try self.compare(.gt, discarded, halfway);
        const equal = try self.compare(.eq, discarded, halfway);
        const odd_bits = try self.binary(.bit_and, retained, one);
        const odd = try self.compare(.ne, odd_bits, try self.constant(0));
        const odd_as_int = try self.select(odd, one, try self.constant(0));
        const tie_and_odd = try self.select(equal, odd_as_int, try self.constant(0));
        const increment = try self.select(greater, one, tie_and_odd);
        return self.binary(.add, retained, increment);
    }
};

/// Replace all low-float storage conversions with target-independent integer operations.
pub fn expandLowFloat(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var has_low_float = false;
    for (0..func.blockCount()) |block_index| {
        const block: Block = @enumFromInt(block_index);
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .decode_low_float, .encode_low_float => {
                has_low_float = true;
                break;
            },
            else => {},
        };
        if (has_low_float) break;
    }
    if (!has_low_float) return false;

    var changed = false;
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);

    for (0..func.blockCount()) |block_index| {
        const block: Block = @enumFromInt(block_index);
        var contains_low_float = false;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .decode_low_float, .encode_low_float => contains_low_float = true,
            else => {},
        };
        if (!contains_low_float) continue;
        changed = true;

        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        const builder: NumericBuilder = .{
            .func = func,
            .out = &out,
            .allocator = allocator,
            .u8_t = u8_t,
            .u16_t = u16_t,
            .u32_t = u32_t,
            .f32_t = f32_t,
            .bool_t = bool_t,
        };

        for (original) |inst| switch (func.opcode(inst)) {
            .decode_low_float => |conversion| {
                const old_result = func.instResult(inst).?;
                const expanded = try expandDecodeLowFloat(builder, conversion);
                func.replaceAllUses(old_result, expanded);
                func.retargetAttrs(.{ .value = old_result }, .{ .value = expanded });
                func.retargetAttrs(.{ .inst = inst }, .{ .inst = func.definingInst(expanded).? });
            },
            .encode_low_float => |conversion| {
                const old_result = func.instResult(inst).?;
                const expanded = try expandEncodeLowFloat(builder, conversion, inst);
                func.replaceAllUses(old_result, expanded);
                func.retargetAttrs(.{ .value = old_result }, .{ .value = expanded });
            },
            else => try out.append(allocator, inst),
        };
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

/// Replace scalar NVFP4 conversions with integer classification and ordered f32 scale operations.
pub fn expandNvFp4(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var has_nvfp4 = false;
    for (0..func.blockCount()) |block_index| {
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .dequantize_nvfp4, .quantize_nvfp4 => {
                has_nvfp4 = true;
                break;
            },
            else => {},
        };
        if (has_nvfp4) break;
    }
    if (!has_nvfp4) return false;

    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);

    for (0..func.blockCount()) |block_index| {
        const block: Block = @enumFromInt(block_index);
        var contains_nvfp4 = false;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .dequantize_nvfp4, .quantize_nvfp4 => contains_nvfp4 = true,
            else => {},
        };
        if (!contains_nvfp4) continue;

        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        const builder: NumericBuilder = .{
            .func = func,
            .out = &out,
            .allocator = allocator,
            .u8_t = u8_t,
            .u16_t = u16_t,
            .u32_t = u32_t,
            .f32_t = f32_t,
            .bool_t = bool_t,
        };

        for (original) |inst| switch (func.opcode(inst)) {
            .dequantize_nvfp4 => |conversion| {
                const old_result = func.instResult(inst).?;
                const expanded = try expandDequantizeNvFp4(builder, conversion);
                func.replaceAllUses(old_result, expanded.result);
                func.retargetAttrs(.{ .inst = inst }, .{ .inst = expanded.semantic_inst });
                func.retargetAttrs(.{ .value = old_result }, .{ .value = expanded.result });
            },
            .quantize_nvfp4 => |conversion| {
                const old_result = func.instResult(inst).?;
                const expanded = try expandQuantizeNvFp4(builder, conversion);
                func.replaceAllUses(old_result, expanded.result);
                func.retargetAttrs(.{ .inst = inst }, .{ .inst = expanded.semantic_inst });
                func.retargetAttrs(.{ .value = old_result }, .{ .value = expanded.result });
            },
            else => try out.append(allocator, inst),
        };
        try func.setBlockInsts(block, out.items);
    }
    return true;
}

/// Replace `reduce` and `splat` with operations every backend already lowers. `reduce` becomes
/// one `extract` per lane plus a balanced tree of `arith` with the reduce's own op, exactly the
/// shape `vulcan-opt.microarch.loopvec`'s own reduction already builds. `splat` becomes a
/// `struct_new` whose fields are all the same value, which the aarch64 backend already
/// recognizes (its `struct_new` isel arm) and lowers to one `dup`.
///
/// A backend with no native lowering of its own runs this unconditionally. aarch64 lowers
/// add-reduce and every splat directly (one `addv`/`faddp` pair, one `dup`), so it calls
/// `expandVectorLanesExcept` instead, keeping only the reduce ops (`mul`, `bit_and`, `bit_or`,
/// `bit_xor`) that still have no matching instruction. Runs once per function.
pub fn expandVectorLanes(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    return expandVectorLanesExcept(allocator, func, keepNothing);
}

fn keepNothing(func: *const Function, inst: Inst) bool {
    _ = func;
    _ = inst;
    return false;
}

/// Same rewrite as `expandVectorLanes`, but leaves a `reduce`/`splat` instruction untouched when
/// `keep` reports the caller's own isel already lowers it. `keep` sees the instruction before any
/// rewriting, so it can inspect the reduce op or the vector's element type to decide.
pub fn expandVectorLanesExcept(
    allocator: std.mem.Allocator,
    func: *Function,
    keep: *const fn (*const Function, Inst) bool,
) std.mem.Allocator.Error!bool {
    var changed = false;
    for (0..func.blockCount()) |block_index| {
        const block: Block = @enumFromInt(block_index);
        var has = false;
        for (func.blockInsts(block)) |inst| {
            if (isVectorLaneOp(func, inst) and !keep(func, inst)) {
                has = true;
                break;
            }
        }
        if (!has) continue;
        changed = true;

        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        for (original) |inst| {
            if (isVectorLaneOp(func, inst) and keep(func, inst)) {
                try out.append(allocator, inst);
                continue;
            }
            switch (func.opcode(inst)) {
                .reduce => |red| {
                    const result = func.instResult(inst).?;
                    const elem_ty = func.valueType(result);
                    const len = vectorLenOf(func, red.vector);

                    var lanes: std.ArrayList(Value) = .empty;
                    defer lanes.deinit(allocator);
                    for (0..len) |lane| {
                        const v = try func.createInst(elem_ty, .{ .extract = .{ .aggregate = red.vector, .index = @intCast(lane) } });
                        try out.append(allocator, func.definingInst(v).?);
                        try lanes.append(allocator, v);
                    }
                    const tree = try buildLaneTree(func, &out, allocator, red.op, elem_ty, lanes.items);
                    func.replaceAllUses(result, tree);
                },
                .splat => |sp| {
                    const result = func.instResult(inst).?;
                    const vec_ty = func.valueType(result);
                    const len = vectorLenOf(func, result);

                    const fields = try allocator.alloc(Value, len);
                    defer allocator.free(fields);
                    for (fields) |*f| f.* = sp.scalar;
                    const list = try func.internValues(fields);
                    const v = try func.createInst(vec_ty, .{ .struct_new = .{ .fields = list } });
                    try out.append(allocator, func.definingInst(v).?);
                    func.replaceAllUses(result, v);
                },
                else => try out.append(allocator, inst),
            }
        }
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

fn isVectorLaneOp(func: *const Function, inst: Inst) bool {
    return switch (func.opcode(inst)) {
        .reduce, .splat => true,
        else => false,
    };
}

/// `v`'s vector length. `v` must be a vector value; `verify` guarantees it for well-formed IR,
/// and this expansion only ever reads it from a `reduce`'s vector operand or a `splat`'s result,
/// both of which `verify` already requires to be a vector.
fn vectorLenOf(func: *const Function, v: Value) u32 {
    return switch (func.types.type_kind(func.valueType(v))) {
        .vector => |vec| vec.len,
        else => unreachable,
    };
}

/// Fold `items` pairwise into a balanced tree of `arith op`, the same shape
/// `loopvec.zig`'s `buildTree` produces for a vectorized reduction.
fn buildLaneTree(func: *Function, out: *std.ArrayList(Inst), allocator: std.mem.Allocator, op: BinOp, ty: types.Type, items: []const Value) std.mem.Allocator.Error!Value {
    var cur: std.ArrayList(Value) = .empty;
    defer cur.deinit(allocator);
    try cur.appendSlice(allocator, items);
    while (cur.items.len > 1) {
        var next: std.ArrayList(Value) = .empty;
        var i: usize = 0;
        while (i + 1 < cur.items.len) : (i += 2) {
            const v = try func.createInst(ty, .{ .arith = .{ .op = op, .lhs = cur.items[i], .rhs = cur.items[i + 1] } });
            try out.append(allocator, func.definingInst(v).?);
            try next.append(allocator, v);
        }
        if (cur.items.len % 2 == 1) try next.append(allocator, cur.items[cur.items.len - 1]);
        cur.deinit(allocator);
        cur = next;
    }
    return cur.items[0];
}

/// Replace scalar f32 division with target-independent u32 operations.
pub fn expandF32Div(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var has_division = false;
    for (0..func.blockCount()) |block_index| {
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .arith => |arith| if (arith.op == .div) switch (func.types.type_kind(func.valueType(func.instResult(inst).?))) {
                .float => |kind| if (kind == .f32) {
                    has_division = true;
                    break;
                },
                else => {},
            },
            else => {},
        };
        if (has_division) break;
    }
    if (!has_division) return false;

    const f32_t = try func.types.intern(.{ .float = .f32 });
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const bool_t = try func.types.intern(.bool);

    for (0..func.blockCount()) |block_index| {
        const block: Block = @enumFromInt(block_index);
        var contains_division = false;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .arith => |arith| if (arith.op == .div and func.valueType(func.instResult(inst).?) == f32_t) {
                contains_division = true;
                break;
            },
            else => {},
        };
        if (!contains_division) continue;

        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        const builder: NumericBuilder = .{
            .func = func,
            .out = &out,
            .allocator = allocator,
            .u8_t = u8_t,
            .u16_t = u16_t,
            .u32_t = u32_t,
            .f32_t = f32_t,
            .bool_t = bool_t,
        };

        for (original) |inst| switch (func.opcode(inst)) {
            .arith => |arith| if (arith.op == .div and func.valueType(func.instResult(inst).?) == f32_t) {
                const old_result = func.instResult(inst).?;
                const expanded = try expandF32Division(builder, arith.lhs, arith.rhs);
                func.replaceAllUses(old_result, expanded);
                func.retargetAttrs(.{ .value = old_result }, .{ .value = expanded });
                func.retargetAttrs(.{ .inst = inst }, .{ .inst = func.definingInst(expanded).? });
            } else try out.append(allocator, inst),
            else => try out.append(allocator, inst),
        };
        try func.setBlockInsts(block, out.items);
    }
    return true;
}

const NormalizedF32 = struct {
    significand: Value,
    exponent: Value,
};

fn normalizeF32(builder: NumericBuilder, magnitude: Value) std.mem.Allocator.Error!NormalizedF32 {
    const fraction = try builder.masked(magnitude, 0x007f_ffff);
    const raw_exponent = try builder.masked(try builder.shifted(.shr, magnitude, 23), 0xff);
    const is_subnormal = try builder.equalConstant(raw_exponent, 0);
    var significand = try builder.select(
        is_subnormal,
        fraction,
        try builder.binary(.bit_or, fraction, try builder.constant(0x0080_0000)),
    );
    var exponent = try builder.select(
        is_subnormal,
        try builder.constant(257),
        try builder.binary(.add, raw_exponent, try builder.constant(256)),
    );
    for ([_]u32{ 16, 8, 4, 2, 1 }) |shift| {
        const needs_shift = try builder.compare(.lt, significand, try builder.constant(@as(u32, 0x0100_0000) >> @intCast(shift)));
        significand = try builder.select(needs_shift, try builder.shifted(.shl, significand, shift), significand);
        exponent = try builder.select(needs_shift, try builder.binary(.sub, exponent, try builder.constant(shift)), exponent);
    }
    return .{ .significand = significand, .exponent = exponent };
}

fn boolAsU32(builder: NumericBuilder, condition: Value) std.mem.Allocator.Error!Value {
    return builder.select(condition, try builder.constant(1), try builder.constant(0));
}

fn flagSet(builder: NumericBuilder, flag: Value) std.mem.Allocator.Error!Value {
    return builder.compare(.ne, flag, try builder.constant(0));
}

fn roundDivisionQuotient(builder: NumericBuilder, quotient: Value, remainder: Value, shift: Value) std.mem.Allocator.Error!Value {
    const one = try builder.constant(1);
    const retained = try builder.binary(.shr, quotient, shift);
    const shifted_one = try builder.binary(.shl, one, shift);
    const discarded = try builder.binary(.bit_and, quotient, try builder.binary(.sub, shifted_one, one));
    const halfway = try builder.binary(.shl, one, try builder.binary(.sub, shift, one));
    const above_half = try boolAsU32(builder, try builder.compare(.gt, discarded, halfway));
    const at_half = try boolAsU32(builder, try builder.compare(.eq, discarded, halfway));
    const has_remainder = try boolAsU32(builder, try builder.compare(.ne, remainder, try builder.constant(0)));
    const odd = try boolAsU32(builder, try builder.compare(.ne, try builder.binary(.bit_and, retained, one), try builder.constant(0)));
    const half_increment = try builder.select(try flagSet(builder, has_remainder), one, odd);
    const tie_increment = try builder.select(try flagSet(builder, at_half), half_increment, try builder.constant(0));
    const increment = try builder.select(try flagSet(builder, above_half), one, tie_increment);
    return builder.binary(.add, retained, increment);
}

fn expandF32Division(builder: NumericBuilder, numerator: Value, denominator: Value) std.mem.Allocator.Error!Value {
    const numerator_bits = try builder.reinterpret(builder.u32_t, numerator);
    const denominator_bits = try builder.reinterpret(builder.u32_t, denominator);
    const numerator_magnitude = try builder.masked(numerator_bits, 0x7fff_ffff);
    const denominator_magnitude = try builder.masked(denominator_bits, 0x7fff_ffff);
    const sign = try builder.masked(try builder.binary(.bit_xor, numerator_bits, denominator_bits), 0x8000_0000);
    const normalized_numerator = try normalizeF32(builder, numerator_magnitude);
    const normalized_denominator = try normalizeF32(builder, denominator_magnitude);

    const ratio_below_one = try boolAsU32(builder, try builder.compare(.lt, normalized_numerator.significand, normalized_denominator.significand));
    const dividend = try builder.select(
        try flagSet(builder, ratio_below_one),
        try builder.shifted(.shl, normalized_numerator.significand, 1),
        normalized_numerator.significand,
    );
    const exponent_bias = try builder.constant(383);
    var result_exponent = try builder.binary(
        .add,
        try builder.binary(.sub, normalized_numerator.exponent, normalized_denominator.exponent),
        exponent_bias,
    );
    result_exponent = try builder.binary(.sub, result_exponent, ratio_below_one);

    var quotient = try builder.constant(0);
    var remainder = dividend;
    for (0..27) |step| {
        const bit = try builder.compare(.ge, remainder, normalized_denominator.significand);
        const subtrahend = try builder.select(bit, normalized_denominator.significand, try builder.constant(0));
        remainder = try builder.binary(.sub, remainder, subtrahend);
        quotient = try builder.binary(.bit_or, try builder.shifted(.shl, quotient, 1), try boolAsU32(builder, bit));
        if (step != 26) remainder = try builder.shifted(.shl, remainder, 1);
    }

    const is_tiny = try boolAsU32(builder, try builder.compare(.le, result_exponent, try builder.constant(256)));
    const raw_tiny_shift = try builder.binary(.sub, try builder.constant(260), result_exponent);
    const shift_too_large = try boolAsU32(builder, try builder.compare(.gt, raw_tiny_shift, try builder.constant(31)));
    const tiny_shift = try builder.select(try flagSet(builder, shift_too_large), try builder.constant(31), raw_tiny_shift);
    const rounding_shift = try builder.select(try flagSet(builder, is_tiny), tiny_shift, try builder.constant(3));
    const rounded = try roundDivisionQuotient(builder, quotient, remainder, rounding_shift);

    const rounded_carry = try boolAsU32(builder, try builder.compare(.ge, rounded, try builder.constant(0x0100_0000)));
    const normal_significand = try builder.select(try flagSet(builder, rounded_carry), try builder.shifted(.shr, rounded, 1), rounded);
    const normal_exponent = try builder.binary(
        .add,
        try builder.binary(.sub, result_exponent, try builder.constant(256)),
        rounded_carry,
    );
    const normal_bits = try builder.binary(
        .bit_or,
        try builder.shifted(.shl, normal_exponent, 23),
        try builder.masked(normal_significand, 0x007f_ffff),
    );
    const finite_bits = try builder.select(try flagSet(builder, is_tiny), rounded, normal_bits);
    const overflow_before_round = try boolAsU32(builder, try builder.compare(.ge, result_exponent, try builder.constant(511)));
    const overflow_after_round = try boolAsU32(builder, try builder.compare(.ge, normal_exponent, try builder.constant(255)));
    const rounded_overflow = try builder.select(try flagSet(builder, is_tiny), try builder.constant(0), overflow_after_round);
    const overflow = try builder.select(try flagSet(builder, overflow_before_round), try builder.constant(1), rounded_overflow);
    var result_bits = try builder.select(
        try builder.compare(.ne, overflow, try builder.constant(0)),
        try builder.constant(0x7f80_0000),
        finite_bits,
    );

    const numerator_zero = try boolAsU32(builder, try builder.equalConstant(numerator_magnitude, 0));
    const denominator_zero = try boolAsU32(builder, try builder.equalConstant(denominator_magnitude, 0));
    const numerator_infinity = try boolAsU32(builder, try builder.equalConstant(numerator_magnitude, 0x7f80_0000));
    const denominator_infinity = try boolAsU32(builder, try builder.equalConstant(denominator_magnitude, 0x7f80_0000));
    const numerator_nan = try boolAsU32(builder, try builder.compare(.gt, numerator_magnitude, try builder.constant(0x7f80_0000)));
    const denominator_nan = try boolAsU32(builder, try builder.compare(.gt, denominator_magnitude, try builder.constant(0x7f80_0000)));
    result_bits = try builder.select(try flagSet(builder, denominator_infinity), try builder.constant(0), result_bits);
    result_bits = try builder.select(try flagSet(builder, numerator_infinity), try builder.constant(0x7f80_0000), result_bits);
    result_bits = try builder.select(try flagSet(builder, denominator_zero), try builder.constant(0x7f80_0000), result_bits);
    result_bits = try builder.select(try flagSet(builder, numerator_zero), try builder.constant(0), result_bits);
    result_bits = try builder.binary(.bit_or, sign, result_bits);

    const both_zero = try builder.select(try flagSet(builder, numerator_zero), denominator_zero, try builder.constant(0));
    const both_infinite = try builder.select(try flagSet(builder, numerator_infinity), denominator_infinity, try builder.constant(0));
    var invalid = try builder.select(try flagSet(builder, numerator_nan), try builder.constant(1), denominator_nan);
    invalid = try builder.binary(.bit_or, invalid, both_zero);
    invalid = try builder.binary(.bit_or, invalid, both_infinite);
    result_bits = try builder.select(
        try builder.compare(.ne, invalid, try builder.constant(0)),
        try builder.constant(0x7fc0_0000),
        result_bits,
    );
    return builder.reinterpret(builder.f32_t, result_bits);
}

fn appendF32DivisionFixture(func: *Function, block: Block) std.mem.Allocator.Error!Value {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const numerator = try func.appendBlockParam(block, f32_t);
    const denominator = try func.appendBlockParam(block, f32_t);
    return func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = numerator, .rhs = denominator } });
}

test "expandF32Div replaces uses attributes and every scalar division in program order" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const first = try appendF32DivisionFixture(&func, entry);
    const first_inst = func.definingInst(first).?;
    try func.addAttr(.{ .inst = first_inst }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 45 } } });
    try func.addAttr(.{ .value = first }, .{ .custom = .{ .namespace = "test", .key = "division", .value = .flag } });
    const second = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .div, .lhs = first, .rhs = func.blockParams(entry)[1] } });
    const exit = try func.appendBlock();
    try func.setJump(entry, exit, &.{second});
    const forwarded = try func.appendBlockParam(exit, f32_t);
    func.setTerminator(exit, .{ .ret = function.Ret.one(forwarded) });

    try std.testing.expect(try expandF32Div(allocator, &func));
    try std.testing.expect(!(try expandF32Div(allocator, &func)));
    var diagnostics = try verify.verify(allocator, &func, .low);
    defer diagnostics.deinit();
    try std.testing.expect(diagnostics.ok());
    for (0..func.blockCount()) |block_index| {
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .arith => |arithmetic| if (arithmetic.op == .div and
                func.valueType(func.instResult(inst).?) == f32_t)
            {
                return error.TestUnexpectedResult;
            },
            else => {},
        };
    }
    const replacement = func.blockArgs(func.terminator(entry).?.jump)[0];
    try std.testing.expect(replacement != second);
    var migrated_value_attrs: usize = 0;
    var migrated_inst_attrs: usize = 0;
    for (func.blockInsts(entry)) |inst| {
        const live_result = func.instResult(inst).?;
        var value_attrs = func.attributesOf(.{ .value = live_result });
        if (value_attrs.next()) |attribute| {
            try std.testing.expectEqualStrings("division", attribute.custom.key);
            migrated_value_attrs += 1;
        }
        var inst_attrs = func.attributesOf(.{ .inst = inst });
        if (inst_attrs.next()) |attribute| {
            try std.testing.expectEqualStrings("line", attribute.custom.key);
            migrated_inst_attrs += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 1), migrated_value_attrs);
    try std.testing.expectEqual(@as(usize, 1), migrated_inst_attrs);
    var old_value_attrs = func.attributesOf(.{ .value = first });
    var old_inst_attrs = func.attributesOf(.{ .inst = first_inst });
    try std.testing.expect(old_value_attrs.next() == null);
    try std.testing.expect(old_inst_attrs.next() == null);
}

test "expandF32Div emits only bounded ordinary integer operations" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const block = try func.appendBlock();
    const result = try appendF32DivisionFixture(&func, block);
    func.setTerminator(block, .{ .ret = function.Ret.one(result) });
    try std.testing.expect(try expandF32Div(allocator, &func));

    var restoring_compares: usize = 0;
    var dynamic_shifts: usize = 0;
    var clamped_shift_uses: usize = 0;
    var count_minus_one_uses: usize = 0;
    var max_predicate_distance: usize = 0;
    const insts = func.blockInsts(block);
    for (insts, 0..) |inst, inst_index| switch (func.opcode(inst)) {
        .arith => |arithmetic| {
            const result_kind = func.types.type_kind(func.valueType(func.instResult(inst).?));
            try std.testing.expect(std.meta.activeTag(result_kind) == .int);
            try std.testing.expectEqual(@as(u16, 32), result_kind.int.bits);
            if (arithmetic.op == .shl or arithmetic.op == .shr) switch (func.opcode(func.definingInst(arithmetic.rhs).?)) {
                .iconst => |amount| try std.testing.expect(amount <= 31),
                .select => |amount| {
                    dynamic_shifts += 1;
                    clamped_shift_uses += 1;
                    switch (func.opcode(func.definingInst(amount.@"else").?)) {
                        .iconst => |normal_count| try std.testing.expectEqual(@as(i64, 3), normal_count),
                        else => return error.TestUnexpectedResult,
                    }
                    switch (func.opcode(func.definingInst(amount.then).?)) {
                        .select => |clamp| {
                            switch (func.opcode(func.definingInst(clamp.then).?)) {
                                .iconst => |maximum| try std.testing.expectEqual(@as(i64, 31), maximum),
                                else => return error.TestUnexpectedResult,
                            }
                            try std.testing.expect(func.opcode(func.definingInst(clamp.@"else").?) == .arith);
                        },
                        else => return error.TestUnexpectedResult,
                    }
                },
                .arith => |count| {
                    dynamic_shifts += 1;
                    try std.testing.expectEqual(function.BinOp.sub, count.op);
                    switch (func.opcode(func.definingInst(count.rhs).?)) {
                        .iconst => |one| try std.testing.expectEqual(@as(i64, 1), one),
                        else => return error.TestUnexpectedResult,
                    }
                    try std.testing.expect(func.opcode(func.definingInst(count.lhs).?) == .select);
                    count_minus_one_uses += 1;
                },
                else => return error.TestUnexpectedResult,
            };
        },
        .icmp => |comparison| {
            if (comparison.op == .ge) restoring_compares += 1;
            try std.testing.expect(std.meta.activeTag(func.types.type_kind(func.valueType(comparison.lhs))) == .int);
            try std.testing.expect(std.meta.activeTag(func.types.type_kind(func.valueType(comparison.rhs))) == .int);
        },
        .select => |selection| {
            try std.testing.expect(std.meta.activeTag(func.types.type_kind(func.valueType(selection.then))) == .int);
            try std.testing.expect(std.meta.activeTag(func.types.type_kind(func.valueType(selection.@"else"))) == .int);
            const predicate_inst = func.definingInst(selection.cond).?;
            var predicate_index: ?usize = null;
            for (insts[0..inst_index], 0..) |candidate, candidate_index| {
                if (candidate == predicate_inst) predicate_index = candidate_index;
            }
            const distance = inst_index - (predicate_index orelse return error.TestUnexpectedResult);
            max_predicate_distance = @max(max_predicate_distance, distance);
        },
        .unary => |unary| try std.testing.expectEqual(function.UnaryOp.reinterpret, unary.op),
        .call, .call_indirect, .convert => return error.TestUnexpectedResult,
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 30), restoring_compares);
    try std.testing.expectEqual(@as(usize, 3), dynamic_shifts);
    try std.testing.expectEqual(@as(usize, 2), clamped_shift_uses);
    try std.testing.expectEqual(@as(usize, 1), count_minus_one_uses);
    // The restoring predicate has the longest span: compare, guarded subtract, quotient
    // shift, then conversion to a u32 flag. Freeze that eight-instruction ceiling so a
    // refactor cannot hoist transient predicates across the unrolled division.
    try std.testing.expectEqual(@as(usize, 8), max_predicate_distance);
}

test "expandF32Div leaves division-free logical IR byte-for-byte unchanged" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const value = try func.appendInst(block, u32_t, .{ .iconst = 45 });
    try func.addAttr(.{ .value = value }, .{ .custom = .{ .namespace = "test", .key = "unchanged", .value = .flag } });
    func.setTerminator(block, .{ .ret = function.Ret.one(value) });
    const text_before = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text_before);
    const bits_before = try bitcode.encode(allocator, &func);
    defer allocator.free(bits_before);
    try std.testing.expect(!(try expandF32Div(allocator, &func)));
    const text_after = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text_after);
    const bits_after = try bitcode.encode(allocator, &func);
    defer allocator.free(bits_after);
    try std.testing.expectEqualStrings(text_before, text_after);
    try std.testing.expectEqualSlices(u8, bits_before, bits_after);
}

const NvFp4Expansion = struct {
    result: Value,
    semantic_inst: Inst,
};

fn scaleOp(application: nvfp4.ScaleApplication, inverse: bool) BinOp {
    return switch (application) {
        .multiply => if (inverse) .div else .mul,
        .divide => if (inverse) .mul else .div,
    };
}

fn expandDequantizeNvFp4(builder: NumericBuilder, conversion: function.NvFp4Convert) std.mem.Allocator.Error!NvFp4Expansion {
    const carrier = try builder.convert(builder.u32_t, conversion.value);
    const nibble = try builder.masked(carrier, 0x0f);
    const sign = try builder.shifted(.shl, try builder.masked(nibble, 0x08), 28);
    const magnitude_index = try builder.masked(nibble, 0x07);
    var magnitude = try builder.constant(0);
    const magnitude_bits = [_]u32{
        0x3f00_0000,
        0x3f80_0000,
        0x3fc0_0000,
        0x4000_0000,
        0x4040_0000,
        0x4080_0000,
        0x40c0_0000,
    };
    for (magnitude_bits, 1..) |bits, index| {
        magnitude = try builder.select(
            try builder.equalConstant(magnitude_index, @intCast(index)),
            try builder.constant(bits),
            magnitude,
        );
    }
    const value_bits = try builder.binary(.bit_or, sign, magnitude);
    const value = try builder.reinterpret(builder.f32_t, value_bits);

    const block_payload = try builder.convert(builder.u32_t, conversion.block_scale);
    const block_bits = try decodeE4M3(builder, block_payload);
    const block_scale = try builder.reinterpret(builder.f32_t, block_bits);
    const local = try builder.floatBinary(scaleOp(conversion.block_application, false), value, block_scale);
    const scaled = try builder.floatBinary(scaleOp(conversion.global_application, false), local, conversion.global_scale);
    const semantic_inst = builder.func.definingInst(scaled).?;

    // This bit round trip blocks a backend from fusing the final scale with an original consumer.
    const scaled_bits = try builder.reinterpret(builder.u32_t, scaled);
    const result = try builder.reinterpret(builder.f32_t, scaled_bits);
    return .{ .result = result, .semantic_inst = semantic_inst };
}

fn expandQuantizeNvFp4(builder: NumericBuilder, conversion: function.NvFp4Convert) std.mem.Allocator.Error!NvFp4Expansion {
    const block_payload = try builder.convert(builder.u32_t, conversion.block_scale);
    const block_bits = try decodeE4M3(builder, block_payload);
    const block_scale = try builder.reinterpret(builder.f32_t, block_bits);
    const global_unscaled = try builder.floatBinary(scaleOp(conversion.global_application, true), conversion.value, conversion.global_scale);
    const local_unscaled = try builder.floatBinary(scaleOp(conversion.block_application, true), global_unscaled, block_scale);
    const bits = try builder.reinterpret(builder.u32_t, local_unscaled);
    const semantic_inst = builder.func.definingInst(bits).?;

    const sign = try builder.masked(try builder.shifted(.shr, bits, 28), 0x08);
    const magnitude = try builder.masked(bits, 0x7fff_ffff);
    var payload = try builder.constant(0x07);
    const thresholds = [_]struct { relation: function.CmpOp, bits: u32, payload: u32 }{
        .{ .relation = .le, .bits = 0x40a0_0000, .payload = 0x06 },
        .{ .relation = .lt, .bits = 0x4060_0000, .payload = 0x05 },
        .{ .relation = .le, .bits = 0x4020_0000, .payload = 0x04 },
        .{ .relation = .lt, .bits = 0x3fe0_0000, .payload = 0x03 },
        .{ .relation = .le, .bits = 0x3fa0_0000, .payload = 0x02 },
        .{ .relation = .lt, .bits = 0x3f40_0000, .payload = 0x01 },
        .{ .relation = .le, .bits = 0x3e80_0000, .payload = 0x00 },
    };
    for (thresholds) |threshold| {
        payload = try builder.select(
            try builder.compare(threshold.relation, magnitude, try builder.constant(threshold.bits)),
            try builder.constant(threshold.payload),
            payload,
        );
    }
    const signed_payload = try builder.binary(.bit_or, sign, payload);
    const is_nan = try builder.compare(.gt, magnitude, try builder.constant(0x7f80_0000));
    const canonical = try builder.select(is_nan, try builder.constant(0x07), signed_payload);
    return .{
        .result = try builder.convert(builder.u8_t, canonical),
        .semantic_inst = semantic_inst,
    };
}

fn expandDecodeLowFloat(builder: NumericBuilder, conversion: function.LowFloatConvert) std.mem.Allocator.Error!Value {
    const payload = try builder.convert(builder.u32_t, conversion.value);
    const bits = switch (conversion.format) {
        .bf16 => try builder.shifted(.shl, payload, 16),
        .f8_e4m3 => try decodeE4M3(builder, payload),
        .f8_e5m2 => try decodeE5M2(builder, payload),
    };
    return builder.reinterpret(builder.f32_t, bits);
}

fn decodeE4M3(builder: NumericBuilder, payload: Value) std.mem.Allocator.Error!Value {
    const sign = try builder.shifted(.shl, try builder.masked(payload, 0x80), 24);
    const exponent = try builder.masked(try builder.shifted(.shr, payload, 3), 0x0f);
    const fraction = try builder.masked(payload, 0x07);
    const normal_exponent = try builder.shifted(.shl, try builder.binary(.add, exponent, try builder.constant(120)), 23);
    const normal_fraction = try builder.shifted(.shl, fraction, 20);
    const normal = try builder.binary(.bit_or, sign, try builder.binary(.bit_or, normal_exponent, normal_fraction));

    var subnormal = try builder.constant(0);
    const subnormal_bits = [_]u32{ 0x3b00_0000, 0x3b80_0000, 0x3bc0_0000, 0x3c00_0000, 0x3c20_0000, 0x3c40_0000, 0x3c60_0000 };
    for (subnormal_bits, 1..) |bits, raw_fraction| {
        subnormal = try builder.select(try builder.equalConstant(fraction, @intCast(raw_fraction)), try builder.constant(bits), subnormal);
    }
    subnormal = try builder.binary(.bit_or, sign, subnormal);
    const finite = try builder.select(try builder.equalConstant(exponent, 0), subnormal, normal);
    const nan = try builder.binary(.bit_or, sign, try builder.constant(0x7fc0_0000));
    const exponent_is_max = try builder.equalConstant(exponent, 0x0f);
    const fraction_is_nan = try builder.equalConstant(fraction, 0x07);
    const fraction_is_nan_int = try builder.select(fraction_is_nan, try builder.constant(1), try builder.constant(0));
    const is_nan_int = try builder.select(exponent_is_max, fraction_is_nan_int, try builder.constant(0));
    const is_nan = try builder.compare(.ne, is_nan_int, try builder.constant(0));
    return builder.select(is_nan, nan, finite);
}

fn decodeE5M2(builder: NumericBuilder, payload: Value) std.mem.Allocator.Error!Value {
    const sign = try builder.shifted(.shl, try builder.masked(payload, 0x80), 24);
    const exponent = try builder.masked(try builder.shifted(.shr, payload, 2), 0x1f);
    const fraction = try builder.masked(payload, 0x03);
    const normal_exponent = try builder.shifted(.shl, try builder.binary(.add, exponent, try builder.constant(112)), 23);
    const normal_fraction = try builder.shifted(.shl, fraction, 21);
    const normal = try builder.binary(.bit_or, sign, try builder.binary(.bit_or, normal_exponent, normal_fraction));

    var subnormal = try builder.constant(0);
    const subnormal_bits = [_]u32{ 0x3780_0000, 0x3800_0000, 0x3840_0000 };
    for (subnormal_bits, 1..) |bits, raw_fraction| {
        subnormal = try builder.select(try builder.equalConstant(fraction, @intCast(raw_fraction)), try builder.constant(bits), subnormal);
    }
    subnormal = try builder.binary(.bit_or, sign, subnormal);
    const finite = try builder.select(try builder.equalConstant(exponent, 0), subnormal, normal);
    const quiet_fraction = try builder.shifted(.shl, try builder.binary(.bit_or, fraction, try builder.constant(2)), 21);
    const nan = try builder.binary(.bit_or, sign, try builder.binary(.bit_or, try builder.constant(0x7f80_0000), quiet_fraction));
    const infinity = try builder.binary(.bit_or, sign, try builder.constant(0x7f80_0000));
    const special = try builder.select(try builder.equalConstant(fraction, 0), infinity, nan);
    return builder.select(try builder.equalConstant(exponent, 0x1f), special, finite);
}

fn expandEncodeLowFloat(builder: NumericBuilder, conversion: function.LowFloatConvert, old_inst: Inst) std.mem.Allocator.Error!Value {
    const bits = try builder.reinterpret(builder.u32_t, conversion.value);
    builder.func.retargetAttrs(.{ .inst = old_inst }, .{ .inst = builder.func.definingInst(bits).? });
    const payload = switch (conversion.format) {
        .bf16 => try encodeBf16(builder, bits),
        .f8_e4m3 => try encodeFp8(builder, bits, true),
        .f8_e5m2 => try encodeFp8(builder, bits, false),
    };
    return builder.convert(if (conversion.format == .bf16) builder.u16_t else builder.u8_t, payload);
}

fn encodeBf16(builder: NumericBuilder, bits: Value) std.mem.Allocator.Error!Value {
    const exponent = try builder.masked(bits, 0x7f80_0000);
    const fraction = try builder.masked(bits, 0x007f_ffff);
    const retained = try builder.shifted(.shr, bits, 16);
    const retained_lsb = try builder.masked(retained, 1);
    const rounded = try builder.shifted(.shr, try builder.binary(.add, bits, try builder.binary(.add, try builder.constant(0x7fff), retained_lsb)), 16);
    const nan = try builder.binary(.bit_or, retained, try builder.constant(0x40));
    const special = try builder.select(try builder.compare(.ne, fraction, try builder.constant(0)), nan, retained);
    return builder.select(try builder.equalConstant(exponent, 0x7f80_0000), special, rounded);
}

fn encodeFp8(builder: NumericBuilder, bits: Value, e4m3: bool) std.mem.Allocator.Error!Value {
    const sign = try builder.masked(try builder.shifted(.shr, bits, 24), 0x80);
    const source_exponent = try builder.masked(try builder.shifted(.shr, bits, 23), 0xff);
    const source_fraction = try builder.masked(bits, 0x007f_ffff);
    const significand = try builder.binary(.bit_or, source_fraction, try builder.constant(0x0080_0000));
    const threshold: u32 = if (e4m3) 121 else 113;
    const shift_bias: u32 = if (e4m3) 141 else 134;
    const original_shift = try builder.binary(.sub, try builder.constant(shift_bias), source_exponent);
    const shift_too_large = try builder.compare(.gt, original_shift, try builder.constant(24));
    const bounded_shift = try builder.select(shift_too_large, try builder.constant(24), original_shift);
    const uses_subnormal_path = try builder.compare(.lt, source_exponent, try builder.constant(threshold));
    const safe_shift = try builder.select(uses_subnormal_path, bounded_shift, try builder.constant(24));
    const subnormal_rounded = try builder.roundRight(significand, safe_shift);
    const subnormal = try builder.select(shift_too_large, try builder.constant(0), subnormal_rounded);

    const normal_shift = try builder.constant(if (e4m3) 20 else 21);
    const normal_rounded = try builder.roundRight(significand, normal_shift);
    const carry_value: u32 = if (e4m3) 16 else 8;
    const retained_value: u32 = if (e4m3) 8 else 4;
    const carry = try builder.equalConstant(normal_rounded, carry_value);
    const rounded = try builder.select(carry, try builder.constant(retained_value), normal_rounded);
    const exponent_base = try builder.binary(.sub, source_exponent, try builder.constant(if (e4m3) 120 else 112));
    const carry_increment = try builder.select(carry, try builder.constant(1), try builder.constant(0));
    const target_exponent = try builder.binary(.add, exponent_base, carry_increment);
    const scaled_exponent = try builder.shifted(.shl, target_exponent, if (e4m3) 3 else 2);
    const normal = try builder.binary(.add, scaled_exponent, try builder.binary(.sub, rounded, try builder.constant(retained_value)));
    const finite_magnitude = try builder.select(try builder.compare(.lt, source_exponent, try builder.constant(threshold)), subnormal, normal);
    const max_finite: u32 = if (e4m3) 0x7e else 0x7b;
    const saturated = try builder.select(try builder.compare(.gt, finite_magnitude, try builder.constant(max_finite)), try builder.constant(max_finite), finite_magnitude);
    const finite = try builder.binary(.bit_or, sign, saturated);

    const nan = if (e4m3)
        try builder.binary(.bit_or, sign, try builder.constant(0x7f))
    else blk: {
        const retained_nan = try builder.masked(try builder.shifted(.shr, source_fraction, 21), 0x03);
        const payload = try builder.binary(.bit_or, try builder.constant(0x7e), retained_nan);
        break :blk try builder.binary(.bit_or, sign, payload);
    };
    const infinity = try builder.binary(.bit_or, sign, try builder.constant(if (e4m3) 0x7e else 0x7c));
    const special = try builder.select(try builder.compare(.ne, source_fraction, try builder.constant(0)), nan, infinity);
    const non_special = try builder.select(try builder.equalConstant(source_exponent, 0), sign, finite);
    return builder.select(try builder.equalConstant(source_exponent, 0xff), special, non_special);
}

fn appendNvFp4Fixture(
    func: *Function,
    block: Block,
    dequantize_direction: bool,
    block_application: nvfp4.ScaleApplication,
    global_application: nvfp4.ScaleApplication,
) !Value {
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const payload = try func.appendBlockParam(block, u8_t);
    const block_scale = try func.appendBlockParam(block, u8_t);
    const value = try func.appendBlockParam(block, f32_t);
    const global_scale = try func.appendBlockParam(block, f32_t);
    const conversion: function.NvFp4Convert = .{
        .value = if (dequantize_direction) payload else value,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = block_application,
        .global_application = global_application,
    };
    return func.appendInst(block, if (dequantize_direction) f32_t else u8_t, if (dequantize_direction)
        .{ .dequantize_nvfp4 = conversion }
    else
        .{ .quantize_nvfp4 = conversion });
}

fn constantU32(func: *const Function, value: Value) ?u32 {
    const inst = func.definingInst(value) orelse return null;
    return switch (func.opcode(inst)) {
        .iconst => |constant| @bitCast(@as(i32, @truncate(constant))),
        else => null,
    };
}

test "expandNvFp4 emits ordered integer classification and two f32 scale operations" {
    const allocator = std.testing.allocator;
    inline for (.{ true, false }) |dequantize_direction| {
        inline for (.{ nvfp4.ScaleApplication.multiply, nvfp4.ScaleApplication.divide }) |block_application| {
            inline for (.{ nvfp4.ScaleApplication.multiply, nvfp4.ScaleApplication.divide }) |global_application| {
                var func = Function.init(allocator);
                defer func.deinit();
                const block = try func.appendBlock();
                const original = try appendNvFp4Fixture(&func, block, dequantize_direction, block_application, global_application);
                func.setTerminator(block, .{ .ret = function.Ret.one(original) });
                const original_params = try allocator.dupe(Value, func.blockParams(block));
                defer allocator.free(original_params);

                try std.testing.expect(try expandNvFp4(allocator, &func));
                try std.testing.expect(!(try expandNvFp4(allocator, &func)));
                var diags = try verify.verify(allocator, &func, .low);
                defer diags.deinit();
                try std.testing.expect(diags.ok());

                var float_arith: [2]Inst = undefined;
                var float_arith_count: usize = 0;
                var carrier_masked = false;
                for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
                    .dequantize_nvfp4, .quantize_nvfp4, .decode_low_float, .encode_low_float, .call, .call_indirect => return error.TestUnexpectedResult,
                    .arith => |arith| {
                        const result = func.instResult(inst).?;
                        if (func.valueType(result) == try func.types.intern(.{ .float = .f32 })) {
                            try std.testing.expect(float_arith_count < float_arith.len);
                            float_arith[float_arith_count] = inst;
                            float_arith_count += 1;
                        } else if (dequantize_direction and arith.op == .bit_and and
                            constantU32(&func, arith.rhs) == 0x0f)
                        {
                            const widened_inst = func.definingInst(arith.lhs) orelse continue;
                            carrier_masked = carrier_masked or switch (func.opcode(widened_inst)) {
                                .convert => |conversion| conversion.value == original_params[0],
                                else => false,
                            };
                        }
                    },
                    .icmp => |compare| {
                        try std.testing.expect(func.types.type_kind(func.valueType(compare.lhs)) == .int);
                        try std.testing.expect(func.types.type_kind(func.valueType(compare.rhs)) == .int);
                    },
                    .select => |select| {
                        try std.testing.expect(func.types.type_kind(func.valueType(select.then)) == .int);
                        try std.testing.expect(func.types.type_kind(func.valueType(select.@"else")) == .int);
                    },
                    .convert => |conversion| {
                        try std.testing.expect(func.types.type_kind(func.valueType(conversion.value)) == .int);
                        try std.testing.expect(func.types.type_kind(func.valueType(func.instResult(inst).?)) == .int);
                    },
                    .unary => |unary| try std.testing.expectEqual(function.UnaryOp.reinterpret, unary.op),
                    else => {},
                };
                try std.testing.expectEqual(@as(usize, 2), float_arith_count);
                const first_result = func.instResult(float_arith[0]).?;
                const first = func.opcode(float_arith[0]).arith;
                const second = func.opcode(float_arith[1]).arith;
                try std.testing.expectEqual(first_result, second.lhs);
                try std.testing.expectEqual(
                    scaleOp(if (dequantize_direction) block_application else global_application, !dequantize_direction),
                    first.op,
                );
                try std.testing.expectEqual(
                    scaleOp(if (dequantize_direction) global_application else block_application, !dequantize_direction),
                    second.op,
                );

                const replacement = func.terminator(block).?.ret.values[0];
                if (dequantize_direction) {
                    try std.testing.expect(carrier_masked);
                    const final_reinterpret = func.opcode(func.definingInst(replacement).?).unary;
                    const bits_reinterpret = func.opcode(func.definingInst(final_reinterpret.value).?).unary;
                    try std.testing.expectEqual(first_result, second.lhs);
                    try std.testing.expectEqual(func.instResult(float_arith[1]).?, bits_reinterpret.value);
                } else {
                    const narrow = func.opcode(func.definingInst(replacement).?).convert;
                    try std.testing.expectEqual(.int, std.meta.activeTag(func.types.type_kind(func.valueType(narrow.value))));
                    try std.testing.expectEqual(@as(u16, 8), func.types.type_kind(func.valueType(replacement)).int.bits);
                }
            }
        }
    }
}

test "expandNvFp4 preserves program order and replaces uses across blocks" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const entry = try func.appendBlock();
    const next = try func.appendBlock();
    const first = try appendNvFp4Fixture(&func, entry, true, .multiply, .divide);
    const entry_params = func.blockParams(entry);
    const addend = entry_params[2];
    const sum = try func.appendInst(entry, f32_t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = addend } });
    try func.setJump(entry, next, &.{ sum, entry_params[1], entry_params[3] });
    const forwarded = try func.appendBlockParam(next, f32_t);
    const next_block_scale = try func.appendBlockParam(next, func.valueType(entry_params[1]));
    const next_global_scale = try func.appendBlockParam(next, f32_t);
    const second = try func.appendInst(next, try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }), .{ .quantize_nvfp4 = .{
        .value = forwarded,
        .block_scale = next_block_scale,
        .global_scale = next_global_scale,
        .block_application = .divide,
        .global_application = .multiply,
    } });
    func.setTerminator(next, .{ .ret = function.Ret.one(second) });

    try std.testing.expect(try expandNvFp4(allocator, &func));
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
    for (0..func.blockCount()) |block_index| {
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .dequantize_nvfp4, .quantize_nvfp4 => return error.TestUnexpectedResult,
            else => {},
        };
    }
    const jump_arg = func.blockArgs(func.terminator(entry).?.jump)[0];
    try std.testing.expectEqual(sum, jump_arg);
    try std.testing.expect(func.opcode(func.definingInst(sum).?).arith.lhs != first);
}

test "expandNvFp4 preserves program order and replaces uses for multiple conversions in one block" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const first = try appendNvFp4Fixture(&func, block, true, .multiply, .divide);
    const second = try appendNvFp4Fixture(&func, block, true, .divide, .multiply);
    const sum = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .add, .lhs = first, .rhs = second } });
    func.setTerminator(block, .{ .ret = function.Ret.one(sum) });

    try std.testing.expect(try expandNvFp4(allocator, &func));
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var float_ops: [5]BinOp = undefined;
    var float_op_count: usize = 0;
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .dequantize_nvfp4, .quantize_nvfp4 => return error.TestUnexpectedResult,
        .arith => |arith| if (func.valueType(func.instResult(inst).?) == f32_t) {
            try std.testing.expect(float_op_count < float_ops.len);
            float_ops[float_op_count] = arith.op;
            float_op_count += 1;
        },
        else => {},
    };
    try std.testing.expectEqual(float_ops.len, float_op_count);
    try std.testing.expectEqualSlices(BinOp, &.{ .mul, .div, .div, .mul, .add }, &float_ops);

    const rewritten_sum = func.opcode(func.definingInst(sum).?).arith;
    try std.testing.expect(rewritten_sum.lhs != first);
    try std.testing.expect(rewritten_sum.rhs != second);
    const instructions = func.blockInsts(block);
    const first_at = std.mem.indexOfScalar(Inst, instructions, func.definingInst(rewritten_sum.lhs).?).?;
    const second_at = std.mem.indexOfScalar(Inst, instructions, func.definingInst(rewritten_sum.rhs).?).?;
    const sum_at = std.mem.indexOfScalar(Inst, instructions, func.definingInst(sum).?).?;
    try std.testing.expect(first_at < second_at);
    try std.testing.expect(second_at < sum_at);
}

test "expandNvFp4 migrates attributes to semantic boundaries" {
    const allocator = std.testing.allocator;
    inline for (.{ true, false }) |dequantize_direction| {
        var func = Function.init(allocator);
        defer func.deinit();
        const block = try func.appendBlock();
        const original = try appendNvFp4Fixture(&func, block, dequantize_direction, .multiply, .divide);
        const original_inst = func.definingInst(original).?;
        try func.addAttr(.{ .inst = original_inst }, .{ .custom = .{
            .namespace = "debug",
            .key = "line",
            .value = .{ .int = 43 },
        } });
        try func.addAttr(.{ .value = original }, .{ .custom = .{
            .namespace = "test",
            .key = "semantic",
            .value = .flag,
        } });
        func.setTerminator(block, .{ .ret = function.Ret.one(original) });

        try std.testing.expect(try expandNvFp4(allocator, &func));
        const replacement = func.terminator(block).?.ret.values[0];
        var value_attrs = func.attributesOf(.{ .value = replacement });
        try std.testing.expectEqualStrings("semantic", value_attrs.next().?.custom.key);
        var old_value_attrs = func.attributesOf(.{ .value = original });
        var old_inst_attrs = func.attributesOf(.{ .inst = original_inst });
        try std.testing.expect(old_value_attrs.next() == null);
        try std.testing.expect(old_inst_attrs.next() == null);

        var boundary: ?Inst = null;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .arith => |arith| if (dequantize_direction and
                func.valueType(func.instResult(inst).?) == try func.types.intern(.{ .float = .f32 }) and
                (arith.op == .mul or arith.op == .div))
            {
                boundary = inst;
            },
            .unary => |unary| if (!dequantize_direction and unary.op == .reinterpret and
                func.types.type_kind(func.valueType(func.instResult(inst).?)) == .int)
            {
                boundary = inst;
            },
            else => {},
        };
        try std.testing.expect(boundary != null);
        var inst_attrs = func.attributesOf(.{ .inst = boundary.? });
        try std.testing.expectEqualStrings("line", inst_attrs.next().?.custom.key);
    }
}

test "expandNvFp4 keeps reinterpret barriers before original add and subtract consumers" {
    const allocator = std.testing.allocator;
    inline for (.{ BinOp.add, BinOp.sub }) |consumer_op| {
        var func = Function.init(allocator);
        defer func.deinit();
        const f32_t = try func.types.intern(.{ .float = .f32 });
        const block = try func.appendBlock();
        const decoded = try appendNvFp4Fixture(&func, block, true, .multiply, .multiply);
        const addend = func.blockParams(block)[2];
        const consumed = try func.appendInst(block, f32_t, .{ .arith = .{ .op = consumer_op, .lhs = decoded, .rhs = addend } });
        func.setTerminator(block, .{ .ret = function.Ret.one(consumed) });

        try std.testing.expect(try expandNvFp4(allocator, &func));
        const consumer = func.opcode(func.definingInst(consumed).?).arith;
        const final_reinterpret = func.opcode(func.definingInst(consumer.lhs).?).unary;
        const bits_reinterpret = func.opcode(func.definingInst(final_reinterpret.value).?).unary;
        const final_scale = func.opcode(func.definingInst(bits_reinterpret.value).?).arith;
        try std.testing.expectEqual(function.UnaryOp.reinterpret, final_reinterpret.op);
        try std.testing.expectEqual(function.UnaryOp.reinterpret, bits_reinterpret.op);
        try std.testing.expectEqual(BinOp.mul, final_scale.op);
    }
}

test "expandNvFp4 leaves unrelated logical IR unchanged" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const value = try func.appendInst(block, u32_t, .{ .iconst = 43 });
    try func.addAttr(.{ .value = value }, .{ .custom = .{
        .namespace = "test",
        .key = "unchanged",
        .value = .flag,
    } });
    func.setTerminator(block, .{ .ret = function.Ret.one(value) });
    const before = try bitcode.encode(allocator, &func);
    defer allocator.free(before);
    const text_before = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text_before);

    try std.testing.expect(!(try expandNvFp4(allocator, &func)));
    const after = try bitcode.encode(allocator, &func);
    defer allocator.free(after);
    const text_after = try std.fmt.allocPrint(allocator, "{f}", .{func});
    defer allocator.free(text_after);
    try std.testing.expectEqualSlices(u8, before, after);
    try std.testing.expectEqualStrings(text_before, text_after);
}

fn appendLowFloatFixture(func: *Function, block: Block, format: function.LowFloatFormat) ![2]Value {
    const payload_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = format.payloadBits() } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const payload = try func.appendBlockParam(block, payload_t);
    const source_bits = try func.appendBlockParam(block, u32_t);
    const source = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = source_bits } });
    const decoded = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .format = format, .value = payload } });
    const decoded_bits = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = decoded } });
    const encoded = try func.appendInst(block, payload_t, .{ .encode_low_float = .{ .format = format, .value = source } });
    const encoded_wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = encoded } });
    return .{ decoded_bits, encoded_wide };
}

test "expandLowFloat rewrites all formats and directions into integer operations" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const block = try func.appendBlock();
    var results: [6]Value = undefined;
    var index: usize = 0;
    for (std.enums.values(function.LowFloatFormat)) |format| {
        const pair = try appendLowFloatFixture(&func, block, format);
        results[index] = pair[0];
        results[index + 1] = pair[1];
        index += 2;
    }
    var combined = results[0];
    for (results[1..]) |result| {
        combined = try func.appendInst(block, func.valueType(result), .{ .arith = .{ .op = .bit_xor, .lhs = combined, .rhs = result } });
    }
    func.setTerminator(block, .{ .ret = function.Ret.one(combined) });

    try std.testing.expect(try expandLowFloat(allocator, &func));
    try std.testing.expect(!(try expandLowFloat(allocator, &func)));
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .decode_low_float, .encode_low_float, .call => return error.TestUnexpectedResult,
        .arith => |op| {
            try std.testing.expect(func.types.type_kind(func.valueType(func.instResult(inst).?)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.lhs)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.rhs)) == .int);
        },
        .icmp => |op| {
            try std.testing.expect(func.types.type_kind(func.valueType(op.lhs)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.rhs)) == .int);
        },
        .select => |op| {
            try std.testing.expect(func.types.type_kind(func.valueType(func.instResult(inst).?)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.then)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.@"else")) == .int);
        },
        .convert => |op| {
            try std.testing.expect(func.types.type_kind(func.valueType(func.instResult(inst).?)) == .int);
            try std.testing.expect(func.types.type_kind(func.valueType(op.value)) == .int);
        },
        .unary => |op| {
            try std.testing.expectEqual(function.UnaryOp.reinterpret, op.op);
            const result_kind = func.types.type_kind(func.valueType(func.instResult(inst).?));
            const source_kind = func.types.type_kind(func.valueType(op.value));
            try std.testing.expect((result_kind == .float and source_kind == .int) or
                (result_kind == .int and source_kind == .float));
        },
        else => {},
    };
}

test "expandLowFloat replaces uses across blocks in program order" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const entry = try func.appendBlock();
    const next = try func.appendBlock();
    const payload = try func.appendBlockParam(entry, u8_t);
    const decoded = try func.appendInst(entry, f32_t, .{ .decode_low_float = .{ .format = .f8_e4m3, .value = payload } });
    try func.setJump(entry, next, &.{decoded});
    const forwarded = try func.appendBlockParam(next, f32_t);
    const encoded = try func.appendInst(next, u8_t, .{ .encode_low_float = .{ .format = .f8_e5m2, .value = forwarded } });
    func.setTerminator(next, .{ .ret = function.Ret.one(encoded) });

    try std.testing.expect(try expandLowFloat(allocator, &func));
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
    for (0..func.blockCount()) |block_index| {
        for (func.blockInsts(@enumFromInt(block_index))) |inst| switch (func.opcode(inst)) {
            .decode_low_float, .encode_low_float => return error.TestUnexpectedResult,
            else => {},
        };
    }
}

test "expandLowFloat migrates live value and debug attributes" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } });
    const block = try func.appendBlock();
    const payload = try func.appendBlockParam(block, u16_t);
    const decoded = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .format = .bf16, .value = payload } });
    const original_inst = func.definingInst(decoded).?;
    try func.addAttr(.{ .value = decoded }, .{ .custom = .{ .namespace = "test", .key = "semantic", .value = .flag } });
    try func.addAttr(.{ .inst = original_inst }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 41 } } });
    func.setTerminator(block, .{ .ret = function.Ret.one(decoded) });

    try std.testing.expect(try expandLowFloat(allocator, &func));
    const replacement = func.terminator(block).?.ret.values[0];
    const boundary = func.definingInst(replacement).?;
    var value_attrs = func.attributesOf(.{ .value = replacement });
    var inst_attrs = func.attributesOf(.{ .inst = boundary });
    var old_value_attrs = func.attributesOf(.{ .value = decoded });
    var old_inst_attrs = func.attributesOf(.{ .inst = original_inst });
    try std.testing.expect(value_attrs.next() != null);
    try std.testing.expect(inst_attrs.next() != null);
    try std.testing.expect(old_value_attrs.next() == null);
    try std.testing.expect(old_inst_attrs.next() == null);
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "expandLowFloat moves encode attributes to the result and f32 boundary" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const block = try func.appendBlock();
    const source = try func.appendBlockParam(block, f32_t);
    const encoded = try func.appendInst(block, u8_t, .{ .encode_low_float = .{ .format = .f8_e5m2, .value = source } });
    const original_inst = func.definingInst(encoded).?;
    try func.addAttr(.{ .value = encoded }, .{ .custom = .{ .namespace = "test", .key = "semantic", .value = .flag } });
    try func.addAttr(.{ .inst = original_inst }, .{ .custom = .{ .namespace = "debug", .key = "line", .value = .{ .int = 42 } } });
    func.setTerminator(block, .{ .ret = function.Ret.one(encoded) });

    try std.testing.expect(try expandLowFloat(allocator, &func));
    const replacement = func.terminator(block).?.ret.values[0];
    var boundary: ?Inst = null;
    for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
        .unary => |op| if (op.op == .reinterpret and op.value == source) {
            boundary = inst;
        },
        else => {},
    };
    try std.testing.expect(boundary != null);
    var value_attrs = func.attributesOf(.{ .value = replacement });
    var inst_attrs = func.attributesOf(.{ .inst = boundary.? });
    var old_value_attrs = func.attributesOf(.{ .value = encoded });
    var old_inst_attrs = func.attributesOf(.{ .inst = original_inst });
    try std.testing.expect(value_attrs.next() != null);
    try std.testing.expect(inst_attrs.next() != null);
    try std.testing.expect(old_value_attrs.next() == null);
    try std.testing.expect(old_inst_attrs.next() == null);
    var diags = try verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());
}

test "expandLowFloat clamps eager FP8 subnormal shifts outside their path" {
    const allocator = std.testing.allocator;
    for ([_]struct { format: function.LowFloatFormat, shift_bias: u32 }{
        .{ .format = .f8_e4m3, .shift_bias = 141 },
        .{ .format = .f8_e5m2, .shift_bias = 134 },
    }) |case| {
        var func = Function.init(allocator);
        defer func.deinit();
        const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
        const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
        const f32_t = try func.types.intern(.{ .float = .f32 });
        const block = try func.appendBlock();
        const bits = try func.appendInst(block, u32_t, .{ .iconst = @as(i64, case.shift_bias << 23) });
        const source = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = bits } });
        const encoded = try func.appendInst(block, u8_t, .{ .encode_low_float = .{ .format = case.format, .value = source } });
        func.setTerminator(block, .{ .ret = function.Ret.one(encoded) });

        try std.testing.expect(try expandLowFloat(allocator, &func));
        var guarded_shift_count: usize = 0;
        for (func.blockInsts(block)) |inst| switch (func.opcode(inst)) {
            .arith => |op| if (op.op == .shr or op.op == .shl) {
                const amount_inst = func.definingInst(op.rhs) orelse continue;
                const amount_select = switch (func.opcode(amount_inst)) {
                    .select => |select| select,
                    else => continue,
                };
                const fallback_inst = func.definingInst(amount_select.@"else").?;
                try std.testing.expectEqual(@as(i64, 24), func.opcode(fallback_inst).iconst);
                guarded_shift_count += 1;
            },
            else => {},
        };
        try std.testing.expectEqual(@as(usize, 2), guarded_shift_count);
    }
}

test "expandLowFloat leaves a function without conversions byte-for-byte unchanged" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const block = try func.appendBlock();
    const value = try func.appendInst(block, u32_t, .{ .iconst = 41 });
    func.setTerminator(block, .{ .ret = function.Ret.one(value) });
    const before = try bitcode.encode(allocator, &func);
    defer allocator.free(before);

    try std.testing.expect(!(try expandLowFloat(allocator, &func)));
    const after = try bitcode.encode(allocator, &func);
    defer allocator.free(after);
    try std.testing.expectEqualSlices(u8, before, after);
}

/// Rewrite every `arith` with op `.mulh` in `func` into an equivalent limb sequence. Returns
/// whether anything was rewritten.
pub fn expandMulh(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    var changed = false;
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        // Does this block hold a mulh? Rebuild its instruction list only if so.
        var has = false;
        for (func.blockInsts(block)) |inst| {
            if (isMulh(func, inst)) {
                has = true;
                break;
            }
        }
        if (!has) continue;
        changed = true;

        var out: std.ArrayList(Inst) = .empty;
        defer out.deinit(allocator);
        // Snapshot the original list: appending new insts to the function's pool must not perturb
        // the sequence we are iterating.
        const original = try allocator.dupe(Inst, func.blockInsts(block));
        defer allocator.free(original);
        for (original) |inst| {
            if (!isMulh(func, inst)) {
                try out.append(allocator, inst);
                continue;
            }
            const a = func.opcode(inst).arith;
            const result = func.instResult(inst).?;
            const high = try emitLimbs(func, &out, allocator, a.lhs, a.rhs, func.valueType(result));
            func.replaceAllUses(result, high);
        }
        try func.setBlockInsts(block, out.items);
    }
    return changed;
}

fn isMulh(func: *const Function, inst: Inst) bool {
    return switch (func.opcode(inst)) {
        .arith => |a| a.op == .mulh,
        else => false,
    };
}

/// Emit the limb sequence for `high half of (lhs * rhs)` at type `ty`, appending each instruction
/// to `out`, and return the value holding the high half.
fn emitLimbs(func: *Function, out: *std.ArrayList(Inst), allocator: std.mem.Allocator, lhs: Value, rhs: Value, ty: types.Type) std.mem.Allocator.Error!Value {
    const info = switch (func.types.type_kind(ty)) {
        .int => |i| i,
        else => unreachable, // mulh is integer-only (verify/strength guarantee it)
    };
    const w: u16 = info.bits;
    const h: i64 = @intCast(w / 2);
    const mask_h: i64 = (@as(i64, 1) << @intCast(w / 2)) - 1;

    const b = struct {
        f: *Function,
        o: *std.ArrayList(Inst),
        a: std.mem.Allocator,
        ty: types.Type,
        fn konst(self: @This(), c: i64) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .iconst = c });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
        fn op(self: @This(), o: BinOp, x: Value, y: Value) std.mem.Allocator.Error!Value {
            const v = try self.f.createInst(self.ty, .{ .arith = .{ .op = o, .lhs = x, .rhs = y } });
            try self.o.append(self.a, self.f.definingInst(v).?);
            return v;
        }
    }{ .f = func, .o = out, .a = allocator, .ty = ty };

    const m = try b.konst(mask_h);
    const hs = try b.konst(h);

    const alo = try b.op(.bit_and, lhs, m);
    const ahi_s = try b.op(.shr, lhs, hs);
    const ahi = try b.op(.bit_and, ahi_s, m);
    const blo = try b.op(.bit_and, rhs, m);
    const bhi_s = try b.op(.shr, rhs, hs);
    const bhi = try b.op(.bit_and, bhi_s, m);

    const lolo = try b.op(.mul, alo, blo);
    const lohi = try b.op(.mul, alo, bhi);
    const hilo = try b.op(.mul, ahi, blo);
    const hihi = try b.op(.mul, ahi, bhi);

    const lolo_hi_s = try b.op(.shr, lolo, hs);
    const lolo_hi = try b.op(.bit_and, lolo_hi_s, m);
    const lohi_lo = try b.op(.bit_and, lohi, m);
    const hilo_lo = try b.op(.bit_and, hilo, m);
    const cross0 = try b.op(.add, lolo_hi, lohi_lo);
    const cross = try b.op(.add, cross0, hilo_lo);

    const lohi_hi_s = try b.op(.shr, lohi, hs);
    const lohi_hi = try b.op(.bit_and, lohi_hi_s, m);
    const hilo_hi_s = try b.op(.shr, hilo, hs);
    const hilo_hi = try b.op(.bit_and, hilo_hi_s, m);
    const cross_hi_s = try b.op(.shr, cross, hs);
    const cross_hi = try b.op(.bit_and, cross_hi_s, m);

    const s0 = try b.op(.add, hihi, lohi_hi);
    const s1 = try b.op(.add, s0, hilo_hi);
    const unsigned_high = try b.op(.add, s1, cross_hi);
    if (info.signedness == .unsigned) return unsigned_high;

    // Signed correction: subtract b where a is negative, and a where b is negative. The sign mask is
    // an arithmetic shift of the ORIGINAL signed operand by W-1 (all ones when negative, else zero).
    const wm1 = try b.konst(@as(i64, @intCast(w - 1)));
    const amask = try b.op(.shr, lhs, wm1);
    const bmask = try b.op(.shr, rhs, wm1);
    const ca = try b.op(.bit_and, amask, rhs);
    const cb = try b.op(.bit_and, bmask, lhs);
    const c0 = try b.op(.sub, unsigned_high, ca);
    return b.op(.sub, c0, cb);
}

/// Why `expandMatmul` leaves one `matmul` in place. Each value names a feature the scalar nest
/// does not model, so a caller that wants a hard failure instead of a surviving `matmul` can ask
/// for the reason and raise its own error. A `null` reason means the expansion handles the op.
pub const Unsupported = enum {
    /// A `quant` epilogue. Requantization is a rounding decision: the fp32 scale multiply, the
    /// relu, the saturation to int8 or uint8, and the zero-point add must happen in the et-soc
    /// order and with the et-soc rounding mode, or the expansion answers a different question than
    /// the tensor unit does. A wrong requantization is worse than no expansion, so this rejects.
    quant,
};

/// The reason `expandMatmul` cannot rewrite `mm`, or null when it can.
///
/// The epilogue is the only rejection. All four dtypes expand: `fp32` multiplies its loaded
/// elements directly, and `fp16`, `int8` and `uint8` widen each element to the 32-bit accumulator
/// first. `input_signs` needs no entry of its own either: `verify` accepts it only when
/// `dtype == .int8`, and the int8 nest reads the per-operand signedness straight out of it. Nor
/// does `embedded`. That flag tells the et-soc backend to save and restore the registers its tensor
/// lowering clobbers, and a scalar nest clobbers nothing, so it is not addressed to this pass.
pub fn matmulUnsupported(mm: MatMul) ?Unsupported {
    if (mm.quant != null) return .quant;
    return switch (mm.dtype) {
        .fp32, .fp16, .int8, .uint8 => null,
    };
}

/// The element shape one `matmul` dtype gives the nest.
const Shape = struct {
    /// The A element type, as loaded from memory.
    a_elem: types.Type,
    /// The B element type, as loaded from memory.
    b_elem: types.Type,
    /// The accumulator and C element type. Always 32 bits: f32 for `fp32`, i32 for `int8`/`uint8`.
    acc: types.Type,
    /// The A/B element size in bytes, which is every A/B pointer stride's unit.
    elem_bytes: i64,
    /// Whether each loaded element passes through a `convert` to the accumulator type first. The
    /// 8-bit dtypes convert, fp32 multiplies the loaded values directly.
    convert: bool,
    /// Whether the accumulator is a float, which picks `fconst` over `iconst` for its zero.
    acc_is_float: bool,
};

/// C is always a 32-bit element, whatever the A/B dtype is (see `MatMul`'s doc comment).
const c_elem_bytes: i64 = 4;

/// The element shape of `mm`. Only called after `matmulUnsupported` returned null.
fn shapeOf(func: *Function, mm: MatMul) std.mem.Allocator.Error!Shape {
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    switch (mm.dtype) {
        .fp32 => return .{
            .a_elem = f32_t,
            .b_elem = f32_t,
            .acc = f32_t,
            .elem_bytes = 4,
            .convert = false,
            .acc_is_float = true,
        },
        .int8, .uint8 => {
            // `input_signs`, when set, is AUTHORITATIVE per operand and `dtype` then only names the
            // hardware element type (see `InputSigns`). Without it the dtype decides both operands.
            const default_unsigned = mm.dtype == .uint8;
            const a_unsigned = if (mm.input_signs) |s| s.a_unsigned else default_unsigned;
            const b_unsigned = if (mm.input_signs) |s| s.b_unsigned else default_unsigned;
            return .{
                .a_elem = try func.types.intern(.{ .int = .{ .signedness = if (a_unsigned) .unsigned else .signed, .bits = 8 } }),
                .b_elem = try func.types.intern(.{ .int = .{ .signedness = if (b_unsigned) .unsigned else .signed, .bits = 8 } }),
                .acc = i32_t,
                .elem_bytes = 1,
                .convert = true,
                .acc_is_float = false,
            };
        },
        // fp16 loads half-width elements and widens each to the f32 accumulator, which is the
        // et-soc tensor unit's own `fp16 -> fp32` type. C stays 32-bit, so only A and B narrow.
        .fp16 => return .{
            .a_elem = try func.types.intern(.{ .float = .f16 }),
            .b_elem = try func.types.intern(.{ .float = .f16 }),
            .acc = f32_t,
            .elem_bytes = 2,
            .convert = true,
            .acc_is_float = true,
        },
    }
}

/// One `matmul` to rewrite: where it sits and what it says.
const Site = struct { block: Block, index: usize, mm: MatMul };

/// The first `matmul` this pass can rewrite, in block order then instruction order, or null when
/// the function holds none.
fn findSite(func: *const Function) ?Site {
    for (0..func.blockCount()) |bi| {
        const block: Block = @enumFromInt(bi);
        var branched = false;
        for (func.blockInsts(block), 0..) |inst, index| {
            switch (func.opcode(inst)) {
                // An `if` does not terminate its block, so both of its edges leave the block and
                // any instruction after it is unreachable. Splitting such a block would move the
                // `if` into the continuation and change where those edges are taken from, so a
                // matmul behind one is left alone. No builder emits this shape.
                .@"if" => branched = true,
                .matmul => |mm| {
                    if (branched) continue;
                    if (matmulUnsupported(mm) != null) continue;
                    return .{ .block = block, .index = index, .mm = mm };
                },
                else => {},
            }
        }
    }
    return null;
}

/// Whether any attribute is keyed by a block. `reorderBlocks` does not remap a block id held in an
/// attribute payload, and this pass finishes with `reorderBlocks`, so a function carrying such an
/// attribute has no safe rewrite here. `vulcan-opt.blocklayout` skips the same functions for the
/// same reason.
fn hasBlockAttribute(func: *const Function) bool {
    for (func.attributeEntries()) |entry| {
        if (entry.target == .block) return true;
    }
    return false;
}

/// Rewrite every `matmul` in `func` that this pass supports into a scalar loop nest over plain
/// loads, multiplies, adds and one store. Returns whether anything was rewritten.
///
/// `matmul` is an et-soc tensor-tile op that only the et-soc VPU backend lowers, so a function
/// holding one cannot execute anywhere else. After this pass runs, every op left in the function is
/// an ordinary scalar op, so any backend compiles it. That makes the expansion the reference answer
/// a tensor lowering gets checked against.
///
/// The nest is the row-major definition of the product, with `a` an m by k matrix, `b` a k by n
/// matrix, and `c` an m by n matrix of 32-bit elements:
///
/// ```
/// for (i in 0..m)
///   for (j in 0..n) {
///     acc = if (accumulate) c[i][j] else 0;
///     for (p in 0..k) acc += a[i][p] * b[p][j];
///     c[i][j] = acc;
///   }
/// ```
///
/// Every pointer walks by a constant stride rather than by a computed index, which is the same
/// shape `vulcan-opt.microarch.matmul_recog` reads when it raises a nest back to a `matmul`: A
/// steps one element per p and one k-element row per i, B steps one n-element row per p and one
/// element per j, and C steps one 32-bit element per j across the whole nest.
///
/// PRECONDITIONS: none. `MatMul`'s doc comment lists 64-byte alignment of `a`, `b` and `c`, one
/// matrix row per cache line, and backend ownership of x31, x6 and the TenC registers across the
/// operation. Those are et-soc HARDWARE requirements of the tensor unit, not properties of the
/// operation. This expansion is plain IR: it reads and writes tightly packed row-major memory at
/// any alignment, it clobbers no register, and the register allocator sees the loads and stores it
/// emits like any others. That difference is the point. A matmul that only runs where those
/// hardware preconditions hold cannot be checked against anything.
///
/// A `matmul` this pass does not support is LEFT IN PLACE and the function is unchanged around it,
/// so a backend that cannot lower it still reports its own error. `matmulUnsupported` names the
/// reason for a caller that wants to raise that error earlier.
pub fn expandMatmul(allocator: std.mem.Allocator, func: *Function) std.mem.Allocator.Error!bool {
    // Guard first: the layout step at the end of every rewrite cannot keep a block-keyed attribute
    // valid, so such a function keeps its matmul rather than getting a stale attribute.
    if (hasBlockAttribute(func)) return false;

    var changed = false;
    // Each rewrite renumbers the blocks, so the next site is found from the top. A rewrite always
    // removes the matmul it was found for, so this terminates.
    while (findSite(func)) |site| {
        try expandSite(allocator, func, site);
        changed = true;
    }
    return changed;
}

/// The blocks one rewrite adds, in the order they are laid out.
const Nest = struct {
    i_head: Block,
    j_head: Block,
    j_body: Block,
    p_head: Block,
    p_body: Block,
    j_latch: Block,
    i_latch: Block,
    /// What the matmul's own block held after it, plus that block's terminator.
    cont: Block,
};

/// Rewrite one `matmul` into the nest. The block holding it becomes the preheader: the
/// instructions before the matmul stay there, and the instructions after it, along with the
/// block's terminator, move to the continuation the nest falls out to.
fn expandSite(allocator: std.mem.Allocator, func: *Function, site: Site) std.mem.Allocator.Error!void {
    const mm = site.mm;
    const shape = try shapeOf(func, mm);
    const preheader = site.block;
    const old_block_count = func.blockCount();

    const ptr_t = try func.types.ptrGlobal();
    const bool_t = try func.types.intern(.bool);
    const i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });

    // Snapshot the block's instruction list: `setBlockInsts` clears the list it would then read
    // from, so the two halves must be copied out first.
    const original = try allocator.dupe(Inst, func.blockInsts(preheader));
    defer allocator.free(original);
    const before = original[0..site.index];
    const after = original[site.index + 1 ..];

    var nest: Nest = undefined;
    nest.i_head = try func.appendBlock();
    nest.j_head = try func.appendBlock();
    nest.j_body = try func.appendBlock();
    nest.p_head = try func.appendBlock();
    nest.p_body = try func.appendBlock();
    nest.j_latch = try func.appendBlock();
    nest.i_latch = try func.appendBlock();
    nest.cont = try func.appendBlock();

    try func.setBlockInsts(nest.cont, after);
    func.terminatorPtr(nest.cont).* = func.terminator(preheader);
    try func.setBlockInsts(preheader, before);
    func.terminatorPtr(preheader).* = null;

    // The preheader: the loop bounds and the fresh accumulator, all loop-invariant, in the one
    // block that dominates the whole nest.
    const zero = try func.appendInst(preheader, i32_t, .{ .iconst = 0 });
    const m_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.m });
    const n_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.n });
    const k_bound = try func.appendInst(preheader, i32_t, .{ .iconst = mm.k });
    const acc_zero = if (shape.acc_is_float)
        try func.appendInst(preheader, shape.acc, .{ .fconst = 0.0 })
    else
        try func.appendInst(preheader, shape.acc, .{ .iconst = 0 });
    try func.setJump(preheader, nest.i_head, &.{ zero, mm.a, mm.c });

    // The i-loop carries the row of A and the running write pointer into C. C is one advance
    // across the whole nest, so the i-loop never resets it.
    const i = try func.appendBlockParam(nest.i_head, i32_t);
    const a_row = try func.appendBlockParam(nest.i_head, ptr_t);
    const c_row = try func.appendBlockParam(nest.i_head, ptr_t);
    const i_lt = try func.appendInst(nest.i_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = i, .rhs = m_bound } });
    try func.appendIf(
        nest.i_head,
        i_lt,
        .{ .target = nest.j_head, .args = &.{ zero, mm.b, c_row } },
        .{ .target = nest.cont },
    );

    // The j-loop carries the column pointer into B and the element pointer into C. The A row is
    // invariant across j, so it is read straight from the i-header, which dominates every block
    // below.
    const j = try func.appendBlockParam(nest.j_head, i32_t);
    const b_col = try func.appendBlockParam(nest.j_head, ptr_t);
    const c_elem = try func.appendBlockParam(nest.j_head, ptr_t);
    const j_lt = try func.appendInst(nest.j_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = j, .rhs = n_bound } });
    try func.appendIf(nest.j_head, j_lt, .{ .target = nest.j_body }, .{ .target = nest.i_latch });

    // The p-loop preheader. The `accumulate` load lives HERE and not in the j-header, because the
    // j-header also runs with `j == n`, where `c_elem` is one past the end of the C row.
    const acc_init = if (mm.accumulate)
        try func.appendInst(nest.j_body, shape.acc, .{ .load = .{ .ptr = c_elem } })
    else
        acc_zero;
    try func.setJump(nest.j_body, nest.p_head, &.{ zero, acc_init, a_row, b_col });

    // The p-loop: the reduction. Its accumulator param holds the finished sum on the exit edge, and
    // the p-header dominates the j-latch, so the store reads it there with no block argument.
    const p = try func.appendBlockParam(nest.p_head, i32_t);
    const acc = try func.appendBlockParam(nest.p_head, shape.acc);
    const a_elem = try func.appendBlockParam(nest.p_head, ptr_t);
    const b_elem = try func.appendBlockParam(nest.p_head, ptr_t);
    const p_lt = try func.appendInst(nest.p_head, bool_t, .{ .icmp = .{ .op = .lt, .lhs = p, .rhs = k_bound } });
    try func.appendIf(nest.p_head, p_lt, .{ .target = nest.p_body }, .{ .target = nest.j_latch });

    const a_val = try func.appendInst(nest.p_body, shape.a_elem, .{ .load = .{ .ptr = a_elem } });
    const b_val = try func.appendInst(nest.p_body, shape.b_elem, .{ .load = .{ .ptr = b_elem } });
    // The 8-bit dtypes widen each element to the accumulator before the multiply, which is where
    // the per-operand signedness of `input_signs` takes effect: the load type decides whether the
    // convert extends the sign or zero-fills.
    const a_op = if (shape.convert) try func.appendInst(nest.p_body, shape.acc, .{ .convert = .{ .value = a_val } }) else a_val;
    const b_op = if (shape.convert) try func.appendInst(nest.p_body, shape.acc, .{ .convert = .{ .value = b_val } }) else b_val;
    const product = try func.appendInst(nest.p_body, shape.acc, .{ .arith = .{ .op = .mul, .lhs = a_op, .rhs = b_op } });
    const next_acc = try func.appendInst(nest.p_body, shape.acc, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = product } });
    const next_p = try func.appendArithImm(nest.p_body, i32_t, .add, p, 1);
    const next_a_elem = try func.appendArithImm(nest.p_body, ptr_t, .add, a_elem, shape.elem_bytes);
    const next_b_elem = try func.appendArithImm(nest.p_body, ptr_t, .add, b_elem, @as(i64, mm.n) * shape.elem_bytes);
    try func.setJump(nest.p_body, nest.p_head, &.{ next_p, next_acc, next_a_elem, next_b_elem });

    // The j-latch: write the element, then step j, the B column, and C by one element each.
    try func.appendStore(nest.j_latch, acc, c_elem);
    const next_j = try func.appendArithImm(nest.j_latch, i32_t, .add, j, 1);
    const next_b_col = try func.appendArithImm(nest.j_latch, ptr_t, .add, b_col, shape.elem_bytes);
    const next_c_elem = try func.appendArithImm(nest.j_latch, ptr_t, .add, c_elem, c_elem_bytes);
    try func.setJump(nest.j_latch, nest.j_head, &.{ next_j, next_b_col, next_c_elem });

    // The i-latch: step i and advance A by one k-element row. C carries on from where the j-loop
    // left it, which is the first element of the next C row.
    const next_i = try func.appendArithImm(nest.i_latch, i32_t, .add, i, 1);
    const next_a_row = try func.appendArithImm(nest.i_latch, ptr_t, .add, a_row, @as(i64, mm.k) * shape.elem_bytes);
    try func.setJump(nest.i_latch, nest.i_head, &.{ next_i, next_a_row, c_elem });

    try layOutNest(allocator, func, @intFromEnum(preheader), old_block_count, nest);
}

/// Put the blocks in an order where every block follows its immediate dominator.
///
/// The nest is built by appending, so its eight blocks land after every block the function already
/// had. That order is legal IR and `ir.verify` accepts it, but the machine backends number
/// linear-scan liveness by block INDEX and need a definition's block to come before every block it
/// dominates. A continuation placed after the blocks it dominates makes the register allocator read
/// a use before its definition and produce an unsound allocation, which surfaces far away as
/// `wimmer.zig`'s allocation verifier firing. `vulcan-opt.blocklayout` states the same rule for the
/// same reason.
///
/// The nest is spliced in where its preheader already was: the blocks before the preheader keep
/// their places, then the preheader, the i-header, the j-header, the j-body, the p-header, the
/// p-body, the j-latch, the i-latch, the continuation, then the blocks that followed the preheader.
/// Each header's immediate dominator is the header outside it, each latch's is a header, and the
/// continuation's is the i-header, so every added block follows the block that dominates it. Every
/// block the function came in with keeps its relative place, and every one of those the preheader
/// dominated is now dominated by the continuation, which still precedes it.
fn layOutNest(
    allocator: std.mem.Allocator,
    func: *Function,
    preheader_index: usize,
    old_block_count: usize,
    nest: Nest,
) std.mem.Allocator.Error!void {
    const order = try allocator.alloc(Block, func.blockCount());
    defer allocator.free(order);

    for (0..preheader_index + 1) |bi| order[bi] = @enumFromInt(bi);
    const added = [_]Block{
        nest.i_head, nest.j_head,  nest.j_body,  nest.p_head,
        nest.p_body, nest.j_latch, nest.i_latch, nest.cont,
    };
    for (added, 0..) |block, offset| order[preheader_index + 1 + offset] = block;

    var next = preheader_index + 1 + added.len;
    for (preheader_index + 1..old_block_count) |bi| {
        order[next] = @enumFromInt(bi);
        next += 1;
    }
    std.debug.assert(next == order.len); // every block placed exactly once
    try func.reorderBlocks(allocator, order);
}

const testing = std.testing;

fn intTy(func: *Function, bits: u16, signedness: std.builtin.Signedness) !types.Type {
    return func.types.intern(.{ .int = .{ .signedness = signedness, .bits = bits } });
}

/// The i128 oracle: the true high `bits` of the full-width product, matching `mulh` semantics.
fn oracleHigh(a: i64, b: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    const shift: u7 = @intCast(bits);
    return switch (signedness) {
        .signed => @truncate(@as(i128, a) * @as(i128, b) >> shift),
        .unsigned => blk: {
            const au: u128 = @as(u64, @bitCast(a)) & maskBits(bits);
            const bu: u128 = @as(u64, @bitCast(b)) & maskBits(bits);
            break :blk @bitCast(@as(u64, @truncate((au * bu) >> shift)));
        },
    };
}

fn maskBits(bits: u16) u64 {
    return if (bits >= 64) ~@as(u64, 0) else (@as(u64, 1) << @intCast(bits)) - 1;
}

/// Sign- or zero-extend the low `bits` of `v` to a canonical i64, matching how a W-bit register
/// value reads back. Used by the test evaluator to model each op at its declared width.
fn wrapTo(v: i64, bits: u16, signedness: std.builtin.Signedness) i64 {
    if (bits >= 64) return v;
    const low: u64 = @as(u64, @bitCast(v)) & maskBits(bits);
    return switch (signedness) {
        .unsigned => @bitCast(low),
        .signed => blk: {
            const sign = @as(u64, 1) << @intCast(bits - 1);
            break :blk @bitCast(if (low & sign != 0) low | ~maskBits(bits) else low);
        },
    };
}

/// Evaluate a value whose whole dataflow is constants (iconst leaves, arith nodes), modelling each
/// op at its result type's width and signedness. Only for the tests below (inputs are constants).
fn evalConst(func: *const Function, v: Value) i64 {
    const inst = func.definingInst(v).?;
    const info = switch (func.types.type_kind(func.valueType(v))) {
        .int => |i| i,
        else => unreachable,
    };
    return switch (func.opcode(inst)) {
        .iconst => |c| wrapTo(c, info.bits, info.signedness),
        .arith => |a| blk: {
            const l = evalConst(func, a.lhs);
            const r = evalConst(func, a.rhs);
            const raw: i64 = switch (a.op) {
                .add => l +% r,
                .sub => l -% r,
                .mul => l *% r,
                .bit_and => l & r,
                .shr => switch (info.signedness) {
                    .signed => l >> @intCast(@as(u64, @bitCast(r)) & 63),
                    .unsigned => @bitCast((@as(u64, @bitCast(l)) & maskBits(info.bits)) >> @intCast(@as(u64, @bitCast(r)) & 63)),
                },
                else => unreachable, // the expansion only emits the ops above
            };
            break :blk wrapTo(raw, info.bits, info.signedness);
        },
        else => unreachable,
    };
}

fn expectMulhExpands(bits: u16, signedness: std.builtin.Signedness, a: i64, b: i64) !void {
    const allocator = testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const t = try intTy(&func, bits, signedness);
    const e = try func.appendBlock();
    const av = try func.appendInst(e, t, .{ .iconst = a });
    const bv = try func.appendInst(e, t, .{ .iconst = b });
    const r = try func.appendInst(e, t, .{ .arith = .{ .op = .mulh, .lhs = av, .rhs = bv } });
    func.setTerminator(e, .{ .ret = function.Ret.one(r) });

    try testing.expect(try expandMulh(allocator, &func));
    for (func.blockInsts(e)) |inst| try testing.expect(!isMulh(&func, inst)); // no mulh survives
    const got = evalConst(&func, func.terminator(e).?.ret.values[0]);
    try testing.expectEqual(oracleHigh(a, b, bits, signedness), got);
}

test "expandMulh matches the i128 oracle for signed 64-bit" {
    try expectMulhExpands(64, .signed, 0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -0x123456789, 0x9876543);
    try expectMulhExpands(64, .signed, -3, -7);
    try expectMulhExpands(64, .signed, std.math.maxInt(i64), std.math.maxInt(i64));
    try expectMulhExpands(64, .signed, std.math.minInt(i64), 2);
}

test "expandMulh matches the i128 oracle for unsigned 64-bit" {
    try expectMulhExpands(64, .unsigned, @bitCast(@as(u64, 0xFFFFFFFF00000000)), @bitCast(@as(u64, 0x2)));
    try expectMulhExpands(64, .unsigned, @bitCast(~@as(u64, 0)), @bitCast(~@as(u64, 0)));
    try expectMulhExpands(64, .unsigned, 0x123456789, 0x9876543);
}

test "expandMulh matches the i128 oracle for 32-bit widths" {
    try expectMulhExpands(32, .signed, 100000, 100000);
    try expectMulhExpands(32, .signed, -100000, 100000);
    try expectMulhExpands(32, .unsigned, @bitCast(@as(u64, 0xFFFF0000)), 0x30000);
}
