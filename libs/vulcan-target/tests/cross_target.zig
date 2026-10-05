//! This test proves that `native.writeObjectDataFor` really cross-emits a runnable
//! object for EACH of the 4 targets, regardless of which arch this test binary itself
//! runs on (aarch64, given the dev host). The test compiles the same arch-independent
//! vulcan-ir `int main(void) { return 42; }` once. Then, for every target, it emits
//! the `.o` file with `writeObjectDataFor`, links it with `vulcan-link.linkObjects`,
//! and prepends a tiny hand-assembled entry stub. The stub calls `main` and exits with
//! its return value, the same stub shape each backend's own `link_native`/`native`
//! tests already use. The test wraps the result in a runnable ELF with
//! `vulcan-link.writeElfExec`, then executes it: natively for aarch64 (the host), and
//! under `qemu-<arch>` for the other three. Each of the 4 must actually RUN, not skip,
//! and exit 42 on a host with all 3 qemu binaries present. An arch whose qemu is
//! missing skips cleanly instead of failing.

const std = @import("std");
const builtin = @import("builtin");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const ld = @import("vulcan-link");

const Function = ir.function.Function;

/// `int main(void) { return 42; }` as vulcan-ir: one block, no params, `ret iconst 42`.
/// This is arch-independent. The same `Function` feeds every backend's
/// `writeObjectDataFor` branch below.
fn buildMain(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try f.appendBlock();
    const c = try f.appendInst(b, t, .{ .iconst = 42 });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(c) });
    return f;
}

fn buildNvFp4(
    allocator: std.mem.Allocator,
    dequantize: bool,
    block_application: ir.nvfp4.ScaleApplication,
    global_application: ir.nvfp4.ScaleApplication,
    later_barrier: bool,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const payload = try func.appendBlockParam(block, u8_t);
    const block_scale = try func.appendBlockParam(block, u8_t);
    const value = try func.appendBlockParam(block, f32_t);
    const global_scale = try func.appendBlockParam(block, f32_t);
    const conversion: ir.function.NvFp4Convert = .{
        .value = if (dequantize) payload else value,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = block_application,
        .global_application = global_application,
    };
    const result = try func.appendInst(block, if (dequantize) f32_t else u8_t, if (dequantize) .{ .dequantize_nvfp4 = conversion } else .{ .quantize_nvfp4 = conversion });
    try func.addAttr(.{ .inst = func.definingInst(result).? }, .{ .custom = .{ .namespace = "debug", .key = "nvfp4", .value = .{ .int = 42 } } });
    try func.addAttr(.{ .value = result }, .{ .custom = .{ .namespace = "test", .key = "live", .value = .flag } });
    if (later_barrier) try func.appendBarrier(block, .workgroup);
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    return func;
}

test "x86 rejects both nvfp4 directions without mutating caller IR" {
    const allocator = std.testing.allocator;
    inline for (.{ true, false }) |dequantize| {
        var func = try buildNvFp4(allocator, dequantize, .multiply, .divide, false);
        defer func.deinit();
        const before = try ir.bitcode.encode(allocator, &func);
        defer allocator.free(before);
        try std.testing.expectError(error.Unsupported, target.native.writeObjectDataFor(allocator, .x86, &.{.{ .name = "convert", .func = &func }}, &.{}));
        const after = try ir.bitcode.encode(allocator, &func);
        defer allocator.free(after);
        try std.testing.expectEqualSlices(u8, before, after);
    }
}

test "supported scalar targets compile every nvfp4 policy and direction without mutation" {
    const allocator = std.testing.allocator;
    inline for ([_]ir.nvfp4.ScaleApplication{ .multiply, .divide }) |block_application| {
        inline for ([_]ir.nvfp4.ScaleApplication{ .multiply, .divide }) |global_application| {
            inline for (.{ true, false }) |dequantize| {
                var func = try buildNvFp4(allocator, dequantize, block_application, global_application, false);
                defer func.deinit();
                const text_before = try std.fmt.allocPrint(allocator, "{f}", .{func});
                defer allocator.free(text_before);
                const bits_before = try ir.bitcode.encode(allocator, &func);
                defer allocator.free(bits_before);
                inline for (.{ .x86_64, .aarch64, .riscv64 }) |arch| {
                    const object = try target.native.writeObjectDataFor(allocator, arch, &.{.{ .name = "convert", .func = &func }}, &.{});
                    allocator.free(object);
                    const text_after = try std.fmt.allocPrint(allocator, "{f}", .{func});
                    defer allocator.free(text_after);
                    const bits_after = try ir.bitcode.encode(allocator, &func);
                    defer allocator.free(bits_after);
                    try std.testing.expectEqualStrings(text_before, text_after);
                    try std.testing.expectEqualSlices(u8, bits_before, bits_after);
                }
            }
        }
    }
}

test "supported nvfp4 expansion reaches a later backend error without caller mutation" {
    const allocator = std.testing.allocator;
    inline for (.{ .x86_64, .aarch64, .riscv64 }) |arch| {
        var func = try buildNvFp4(allocator, true, .multiply, .divide, true);
        defer func.deinit();
        const text_before = try std.fmt.allocPrint(allocator, "{f}", .{func});
        defer allocator.free(text_before);
        const bits_before = try ir.bitcode.encode(allocator, &func);
        defer allocator.free(bits_before);
        try std.testing.expectError(error.Unsupported, target.native.writeObjectDataFor(allocator, arch, &.{.{ .name = "convert", .func = &func }}, &.{}));
        const text_after = try std.fmt.allocPrint(allocator, "{f}", .{func});
        defer allocator.free(text_after);
        const bits_after = try ir.bitcode.encode(allocator, &func);
        defer allocator.free(bits_after);
        try std.testing.expectEqualStrings(text_before, text_after);
        try std.testing.expectEqualSlices(u8, bits_before, bits_after);
    }
}

fn buildNvFp4IntegerAbi(
    allocator: std.mem.Allocator,
    dequantize: bool,
    block_application: ir.nvfp4.ScaleApplication,
    global_application: ir.nvfp4.ScaleApplication,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const input_bits = try func.appendBlockParam(block, u32_t);
    const block_bits = try func.appendBlockParam(block, u32_t);
    const global_bits = try func.appendBlockParam(block, u32_t);
    const carrier = try func.appendInst(block, u8_t, .{ .convert = .{ .value = input_bits } });
    const block_scale = try func.appendInst(block, u8_t, .{ .convert = .{ .value = block_bits } });
    const input_f32 = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = input_bits } });
    const global_scale = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = global_bits } });
    const conversion: ir.function.NvFp4Convert = .{
        .value = if (dequantize) carrier else input_f32,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = block_application,
        .global_application = global_application,
    };
    const converted = try func.appendInst(block, if (dequantize) f32_t else u8_t, if (dequantize) .{ .dequantize_nvfp4 = conversion } else .{ .quantize_nvfp4 = conversion });
    const result = try func.appendInst(block, u32_t, if (dequantize) .{ .unary = .{ .op = .reinterpret, .value = converted } } else .{ .convert = .{ .value = converted } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    return func;
}

fn buildReorderedNvFp4IntegerAbi(
    allocator: std.mem.Allocator,
    dequantize: bool,
    block_application: ir.nvfp4.ScaleApplication,
    global_application: ir.nvfp4.ScaleApplication,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const input_bits = try func.appendBlockParam(block, u32_t);
    const block_bits = try func.appendBlockParam(block, u32_t);
    const global_bits = try func.appendBlockParam(block, u32_t);
    const carrier = try func.appendInst(block, u8_t, .{ .convert = .{ .value = input_bits } });
    const block_payload = try func.appendInst(block, u8_t, .{ .convert = .{ .value = block_bits } });
    const input_f32 = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = input_bits } });
    const global_scale = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = global_bits } });
    const block_scale = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .format = .f8_e4m3, .value = block_payload } });
    const unit_payload = try func.appendInst(block, u8_t, .{ .iconst = 0x38 });
    const unit_scale = try func.appendInst(block, f32_t, .{ .fconst = 1.0 });

    const result = if (dequantize) result: {
        const global_first = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = .{
            .value = carrier,
            .block_scale = unit_payload,
            .global_scale = global_scale,
            .block_application = .multiply,
            .global_application = global_application,
        } });
        const reordered = try func.appendInst(block, f32_t, .{ .arith = .{
            .op = if (block_application == .multiply) .mul else .div,
            .lhs = global_first,
            .rhs = block_scale,
        } });
        break :result try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = reordered } });
    } else result: {
        const block_first = try func.appendInst(block, f32_t, .{ .arith = .{
            .op = if (block_application == .multiply) .div else .mul,
            .lhs = input_f32,
            .rhs = block_scale,
        } });
        const global_second = try func.appendInst(block, f32_t, .{ .arith = .{
            .op = if (global_application == .multiply) .div else .mul,
            .lhs = block_first,
            .rhs = global_scale,
        } });
        const payload = try func.appendInst(block, u8_t, .{ .quantize_nvfp4 = .{
            .value = global_second,
            .block_scale = unit_payload,
            .global_scale = unit_scale,
            .block_application = .multiply,
            .global_application = .multiply,
        } });
        break :result try func.appendInst(block, u32_t, .{ .convert = .{ .value = payload } });
    };
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    return func;
}

fn buildNvFp4ArithmeticBoundary(allocator: std.mem.Allocator, op: ir.function.BinOp) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const carrier_bits = try func.appendBlockParam(block, u32_t);
    const block_bits = try func.appendBlockParam(block, u32_t);
    const global_bits = try func.appendBlockParam(block, u32_t);
    const rhs_bits = try func.appendBlockParam(block, u32_t);
    const carrier = try func.appendInst(block, u8_t, .{ .convert = .{ .value = carrier_bits } });
    const block_scale = try func.appendInst(block, u8_t, .{ .convert = .{ .value = block_bits } });
    const global_scale = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = global_bits } });
    const rhs = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = rhs_bits } });
    const decoded = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = .{
        .value = carrier,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = .multiply,
        .global_application = .multiply,
    } });
    const combined = try func.appendInst(block, f32_t, .{ .arith = .{ .op = op, .lhs = decoded, .rhs = rhs } });
    const result = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = combined } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    return func;
}

fn buildNvFp4ArithmeticBoundaryMain(
    allocator: std.mem.Allocator,
    op: ir.function.BinOp,
    carrier_bits: u32,
    block_bits: u32,
    global_bits: u32,
    rhs_bits: u32,
    expected_bits: u32,
) !Function {
    var func = Function.init(allocator);
    errdefer func.deinit();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const carrier = try func.appendInst(block, u8_t, .{ .iconst = carrier_bits });
    const block_scale = try func.appendInst(block, u8_t, .{ .iconst = block_bits });
    const global_scale = try func.appendInst(block, f32_t, .{ .fconst = @as(f32, @bitCast(global_bits)) });
    const rhs = try func.appendInst(block, f32_t, .{ .fconst = @as(f32, @bitCast(rhs_bits)) });
    const decoded = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = .{
        .value = carrier,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = .multiply,
        .global_application = .multiply,
    } });
    const combined = try func.appendInst(block, f32_t, .{ .arith = .{ .op = op, .lhs = decoded, .rhs = rhs } });
    const actual = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = combined } });
    const expected = try func.appendInst(block, u32_t, .{ .iconst = expected_bits });
    const difference = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = actual, .rhs = expected } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(difference) });
    return func;
}

test "native nvfp4 integer-ABI wrappers execute exact reference conversions" {
    if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64 and builtin.cpu.arch != .riscv64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const ConvertFn = *const fn (u32, u32, u32) callconv(.c) u32;
    const policies = [_]ir.nvfp4.ScaleApplication{ .multiply, .divide };
    const globals = [_]u32{ 0x0000_0000, 0xbf80_0000, 0x3f00_0000, 0x7f80_0000, 0x7fc0_1234 };
    const dequantize_order_cases = [_]struct { payload: u8, block_scale: u8, global_bits: u32, block: ir.nvfp4.ScaleApplication, global: ir.nvfp4.ScaleApplication, expected: u32, reordered: u32 }{
        .{ .payload = 0x03, .block_scale = 0x03, .global_bits = 0x3e00_0001, .block = .multiply, .global = .multiply, .expected = 0x3a90_0001, .reordered = 0x3a90_0002 },
        .{ .payload = 0x01, .block_scale = 0x03, .global_bits = 0x3e00_0001, .block = .multiply, .global = .divide, .expected = 0x3cbf_ffff, .reordered = 0x3cbf_fffe },
        .{ .payload = 0x01, .block_scale = 0x05, .global_bits = 0x3e00_0001, .block = .divide, .global = .multiply, .expected = 0x40cc_cccf, .reordered = 0x40cc_ccce },
        .{ .payload = 0x01, .block_scale = 0x03, .global_bits = 0x3e00_0001, .block = .divide, .global = .divide, .expected = 0x442a_aaaa, .reordered = 0x442a_aaa9 },
    };
    const quantize_order_cases = [_]struct { value_bits: u32, block_scale: u8, global_bits: u32, block: ir.nvfp4.ScaleApplication, global: ir.nvfp4.ScaleApplication, expected: u8, reordered: u8 }{
        .{ .value_bits = 0x3e00_0522, .block_scale = 0x23, .global_bits = 0x3e54_d004, .block = .multiply, .global = .multiply, .expected = 0x06, .reordered = 0x05 },
        .{ .value_bits = 0x3e00_154f, .block_scale = 0x0f, .global_bits = 0x3ed1_dd10, .block = .multiply, .global = .divide, .expected = 0x04, .reordered = 0x03 },
        .{ .value_bits = 0x3e00_154f, .block_scale = 0x47, .global_bits = 0x3f20_1aa3, .block = .divide, .global = .multiply, .expected = 0x01, .reordered = 0x02 },
        .{ .value_bits = 0x3e00_399b, .block_scale = 0x55, .global_bits = 0x3eeb_e49b, .block = .divide, .global = .divide, .expected = 0x01, .reordered = 0x02 },
    };

    inline for (policies) |block_application| inline for (policies) |global_application| {
        var dequantize = try buildNvFp4IntegerAbi(allocator, true, block_application, global_application);
        defer dequantize.deinit();
        var dequant_code = try target.native.jitFunction(allocator, &dequantize);
        defer dequant_code.deinit();
        const dequant_fn = dequant_code.entry(ConvertFn, 0);
        for (globals) |global_bits| for (0..256) |carrier| for (0..256) |block_scale| {
            const expected = ir.nvfp4.dequantize(@intCast(carrier), @intCast(block_scale), @bitCast(global_bits), block_application, global_application);
            const actual: f32 = @bitCast(dequant_fn(@intCast(carrier), @intCast(block_scale), global_bits));
            if (std.math.isNan(expected)) {
                try std.testing.expect(std.math.isNan(actual));
            } else {
                try std.testing.expectEqual(@as(u32, @bitCast(expected)), @as(u32, @bitCast(actual)));
            }
        };
        for (dequantize_order_cases) |case| if (case.block == block_application and case.global == global_application) {
            try std.testing.expectEqual(case.expected, dequant_fn(case.payload, case.block_scale, case.global_bits));
        };
        var reordered_dequantize = try buildReorderedNvFp4IntegerAbi(allocator, true, block_application, global_application);
        defer reordered_dequantize.deinit();
        var reordered_dequant_code = try target.native.jitFunction(allocator, &reordered_dequantize);
        defer reordered_dequant_code.deinit();
        const reordered_dequant_fn = reordered_dequant_code.entry(ConvertFn, 0);
        for (dequantize_order_cases) |case| if (case.block == block_application and case.global == global_application) {
            const actual = reordered_dequant_fn(case.payload, case.block_scale, case.global_bits);
            try std.testing.expectEqual(case.reordered, actual);
            try std.testing.expect(actual != case.expected);
        };

        var quantize = try buildNvFp4IntegerAbi(allocator, false, block_application, global_application);
        defer quantize.deinit();
        var quant_code = try target.native.jitFunction(allocator, &quantize);
        defer quant_code.deinit();
        const quant_fn = quant_code.entry(ConvertFn, 0);

        // Unit scales make every policy pair classify the supplied value itself.
        // Pin every exact E2M1 value and both signs of every midpoint with its two
        // adjacent bit patterns, so a changed tie comparison fails generated code.
        for (0..16) |payload| {
            const value_bits: u32 = @bitCast(ir.nvfp4.decodeE2M1(@intCast(payload)));
            try std.testing.expectEqual(@as(u32, @intCast(payload)), quant_fn(value_bits, 0x38, 0x3f80_0000));
        }
        const midpoint_cases = [_]struct { bits: u32, lower: u8, tie: u8, upper: u8 }{
            .{ .bits = 0x3e80_0000, .lower = 0, .tie = 0, .upper = 1 },
            .{ .bits = 0x3f40_0000, .lower = 1, .tie = 2, .upper = 2 },
            .{ .bits = 0x3fa0_0000, .lower = 2, .tie = 2, .upper = 3 },
            .{ .bits = 0x3fe0_0000, .lower = 3, .tie = 4, .upper = 4 },
            .{ .bits = 0x4020_0000, .lower = 4, .tie = 4, .upper = 5 },
            .{ .bits = 0x4060_0000, .lower = 5, .tie = 6, .upper = 6 },
            .{ .bits = 0x40a0_0000, .lower = 6, .tie = 6, .upper = 7 },
        };
        for (midpoint_cases) |case| inline for (.{ @as(u32, 0), @as(u32, 0x8000_0000) }) |sign| {
            try std.testing.expectEqual(@as(u32, case.lower | @as(u8, @truncate(sign >> 28))), quant_fn((case.bits - 1) | sign, 0x38, 0x3f80_0000));
            try std.testing.expectEqual(@as(u32, case.tie | @as(u8, @truncate(sign >> 28))), quant_fn(case.bits | sign, 0x38, 0x3f80_0000));
            try std.testing.expectEqual(@as(u32, case.upper | @as(u8, @truncate(sign >> 28))), quant_fn((case.bits + 1) | sign, 0x38, 0x3f80_0000));
        };

        const values = [_]u32{
            0x0000_0000, 0x8000_0000, 0x0000_0001, 0x8000_0001,
            0x3e80_0000, 0x3f40_0000, 0x3fa0_0000, 0x3fe0_0000,
            0x4020_0000, 0x4060_0000, 0x40a0_0000, 0x7f7f_ffff,
            0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7f80_0001,
            0xff80_0001, 0x7fc0_1234, 0xffc0_1234,
        };
        const block_scales = [_]u8{ 0x00, 0x38, 0xb8, 0x7e, 0x7f };
        for (values) |value_bits| for (block_scales) |block_scale| for (globals) |global_bits| {
            const expected = ir.nvfp4.quantize(@bitCast(value_bits), block_scale, @bitCast(global_bits), block_application, global_application);
            try std.testing.expectEqual(@as(u32, expected), quant_fn(value_bits, block_scale, global_bits));
        };
        for (quantize_order_cases) |case| if (case.block == block_application and case.global == global_application) {
            try std.testing.expectEqual(@as(u32, case.expected), quant_fn(case.value_bits, case.block_scale, case.global_bits));
        };
        var reordered_quantize = try buildReorderedNvFp4IntegerAbi(allocator, false, block_application, global_application);
        defer reordered_quantize.deinit();
        var reordered_quant_code = try target.native.jitFunction(allocator, &reordered_quantize);
        defer reordered_quant_code.deinit();
        const reordered_quant_fn = reordered_quant_code.entry(ConvertFn, 0);
        for (quantize_order_cases) |case| if (case.block == block_application and case.global == global_application) {
            const actual = reordered_quant_fn(case.value_bits, case.block_scale, case.global_bits);
            try std.testing.expectEqual(@as(u32, case.reordered), actual);
            try std.testing.expect(actual != case.expected);
        };
    };
}

test "native nvfp4 dequantize keeps the rounding boundary before add and subtract" {
    if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64 and builtin.cpu.arch != .riscv64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const BoundaryFn = *const fn (u32, u32, u32, u32) callconv(.c) u32;
    const cases = [_]struct { op: ir.function.BinOp, carrier: u32, block: u32, global: u32, rhs: u32, expected: u32, fused: u32 }{
        .{ .op = .add, .carrier = 0x01, .block = 0x34, .global = 0x3e00_0001, .rhs = 0x3e00_0001, .expected = 0x3e30_0002, .fused = 0x3e30_0001 },
        .{ .op = .sub, .carrier = 0x01, .block = 0x31, .global = 0x3e00_0001, .rhs = 0x3e00_0001, .expected = 0xbdb8_0002, .fused = 0xbdb8_0001 },
    };
    for (cases) |case| {
        var func = try buildNvFp4ArithmeticBoundary(allocator, case.op);
        defer func.deinit();
        var code = try target.native.jitFunction(allocator, &func);
        defer code.deinit();
        const actual = code.entry(BoundaryFn, 0)(case.carrier, case.block, case.global, case.rhs);
        try std.testing.expectEqual(case.expected, actual);
        try std.testing.expect(actual != case.fused);
    }
}

test "native nvfp4 packed memory crosses a scale block and preserves its tail sibling" {
    if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .aarch64 and builtin.cpu.arch != .riscv64) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const element_count = 19;
    const packed_count = (element_count + 1) / 2;
    var func = Function.init(allocator);
    defer func.deinit();
    const ptr_t = try func.types.ptrGlobal();
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const packed_input = try func.appendBlockParam(block, ptr_t);
    const scales = try func.appendBlockParam(block, ptr_t);
    const global_bits = try func.appendBlockParam(block, u32_t);
    const decoded_output = try func.appendBlockParam(block, ptr_t);
    const float_input = try func.appendBlockParam(block, ptr_t);
    const packed_output = try func.appendBlockParam(block, ptr_t);
    const global_scale = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = global_bits } });
    for (0..element_count) |index| {
        const packed_at = if (index < 2) packed_input else try func.appendArithImm(block, ptr_t, .add, packed_input, @intCast(index / 2));
        const scale_at = if (index < 16) scales else try func.appendArithImm(block, ptr_t, .add, scales, @intCast(index / 16));
        const packed_byte = try func.appendInst(block, u8_t, .{ .load = .{ .ptr = packed_at } });
        const packed_wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = packed_byte } });
        const positioned = if (index & 1 == 0)
            packed_wide
        else
            try func.appendArithImm(block, u32_t, .shr, packed_wide, 4);
        const nibble_wide = try func.appendArithImm(block, u32_t, .bit_and, positioned, 0x0f);
        const nibble = try func.appendInst(block, u8_t, .{ .convert = .{ .value = nibble_wide } });
        const block_scale = try func.appendInst(block, u8_t, .{ .load = .{ .ptr = scale_at } });
        const decoded = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = .{
            .value = nibble,
            .block_scale = block_scale,
            .global_scale = global_scale,
            .block_application = .multiply,
            .global_application = .divide,
        } });
        const decoded_at = if (index == 0) decoded_output else try func.appendArithImm(block, ptr_t, .add, decoded_output, @intCast(index * 4));
        try func.appendStore(block, decoded, decoded_at);

        const float_at = if (index == 0) float_input else try func.appendArithImm(block, ptr_t, .add, float_input, @intCast(index * 4));
        const source = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = float_at } });
        const encoded = try func.appendInst(block, u8_t, .{ .quantize_nvfp4 = .{
            .value = source,
            .block_scale = block_scale,
            .global_scale = global_scale,
            .block_application = .divide,
            .global_application = .multiply,
        } });
        const encoded_wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = encoded } });
        const output_at = if (index < 2) packed_output else try func.appendArithImm(block, ptr_t, .add, packed_output, @intCast(index / 2));
        const old_byte = try func.appendInst(block, u8_t, .{ .load = .{ .ptr = output_at } });
        const old_wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = old_byte } });
        const preserved = try func.appendArithImm(block, u32_t, .bit_and, old_wide, if (index & 1 == 0) 0xf0 else 0x0f);
        const placed = if (index & 1 == 0)
            encoded_wide
        else
            try func.appendArithImm(block, u32_t, .shl, encoded_wide, 4);
        const combined = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_or, .lhs = preserved, .rhs = placed } });
        const output_byte = try func.appendInst(block, u8_t, .{ .convert = .{ .value = combined } });
        try func.appendStore(block, output_byte, output_at);
    }
    func.setTerminator(block, .{ .ret = ir.function.Ret.none() });

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();
    const run = code.entry(*const fn (*const u8, *const u8, u32, *u32, *const u32, *u8) callconv(.c) void, 0);
    const packed_values = [_]u8{ 0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe, 0x21, 0x43 };
    const scale_values = [_]u8{ 0x39, 0xb8 };
    const global_scale_bits: u32 = 0x3f40_0000;
    const source_bits = [_]u32{
        0x0000_0000, 0x8000_0000, 0x3e80_0000, 0x3e80_0001, 0x3f40_0000,
        0x3f40_0001, 0x3fa0_0000, 0x3fa0_0001, 0x3fe0_0000, 0x4020_0000,
        0x4060_0000, 0x40a0_0000, 0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000,
        0xff80_0000, 0x7f80_0001, 0xff80_0001, 0x7fc0_1234,
    };
    var decoded_bits: [element_count]u32 = @splat(0);
    var output: [packed_count]u8 = @splat(0xa5);
    var expected_output = output;
    run(&packed_values[0], &scale_values[0], global_scale_bits, &decoded_bits[0], &source_bits[0], &output[0]);
    const global_scale_value: f32 = @bitCast(global_scale_bits);
    for (0..element_count) |index| {
        const payload = if (index & 1 == 0) packed_values[index / 2] & 0x0f else packed_values[index / 2] >> 4;
        const scale = scale_values[index / 16];
        const expected_decoded = ir.nvfp4.dequantize(payload, scale, global_scale_value, .multiply, .divide);
        if (std.math.isNan(expected_decoded)) {
            try std.testing.expect(std.math.isNan(@as(f32, @bitCast(decoded_bits[index]))));
        } else {
            try std.testing.expectEqual(@as(u32, @bitCast(expected_decoded)), decoded_bits[index]);
        }
        const encoded = ir.nvfp4.quantize(@bitCast(source_bits[index]), scale, global_scale_value, .divide, .multiply);
        if (index & 1 == 0) {
            expected_output[index / 2] = (expected_output[index / 2] & 0xf0) | encoded;
        } else {
            expected_output[index / 2] = (expected_output[index / 2] & 0x0f) | (encoded << 4);
        }
    }
    try std.testing.expectEqualSlices(u8, &expected_output, &output);
    try std.testing.expectEqual(@as(u8, 0xa0), output[packed_count - 1] & 0xf0);
}

/// Run `argv[0]`, already on PATH, or a native `./a.elf`, against `elf` written to a
/// fresh tmp dir, and return its exit code. Returns `error.SkipZigTest` when the
/// runner, `qemu-<arch>` when `argv[0]` names one, is not installed.
fn runElf(allocator: std.mem.Allocator, io: std.Io, elf: []const u8, argv: []const []const u8) !u8 {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "a.elf", .data = elf, .flags = .{ .permissions = .executable_file } });

    const proc = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .dir = tmp.dir },
    }) catch |e| switch (e) {
        error.FileNotFound, error.AccessDenied => return error.SkipZigTest,
        else => return e,
    };
    defer allocator.free(proc.stdout);
    defer allocator.free(proc.stderr);
    return switch (proc.term) {
        .exited => |code| code,
        else => {
            std.debug.print("cross_target run term: {any}\nstderr: {s}\n", .{ proc.term, proc.stderr });
            return error.BackendFailed;
        },
    };
}

test "cross-target: writeObjectDataFor(.aarch64, ...) emits+links+runs to exit 42" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Executes the AArch64 ELF directly, which needs a Linux host: the image is a
    // Linux ELF with a Linux svc exit, and a darwin aarch64 host cannot exec it.
    if (builtin.cpu.arch != .aarch64 or builtin.os.tag != .linux) return error.SkipZigTest;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .aarch64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.aarch64.encode;
    const base: u64 = 0x400000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + 12);
    defer image.deinit(allocator);

    // stub: bl main, then movz x8, #93, then svc #0. This exits with exit(x0), where x0
    // already holds main's return value (AAPCS64). The stub is 12 bytes and sits right
    // before `image.code`. `bl` is the very first instruction (pc == base), so its offset
    // to `main` is simply `main_addr - base`. This needs no extra stub-length term, unlike
    // a `bl` sitting later in the stub.
    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const stub = [_]u32{
        encode.bl(@intCast(main_addr - @as(i64, @intCast(base)))),
        encode.movz(.x8, 93, 0),
        encode.svc(0),
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.aarch64, allocator, program.items, program.items.len, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{"./a.elf"}) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.x86_64, ...) emits+links+runs to exit 42 (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .x86_64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.x86_64.encode;
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice()); // rdi = main's return (rax)
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice()); // rax = 60 (exit)
    try exitseq.appendSlice(allocator, encode.syscall().slice());
    const stub_len: u64 = 5 + exitseq.items.len; // call rel32 (5) ++ exitseq

    const base: u64 = 0x10000000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + 5)));
    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(allocator);
    try stub.appendSlice(allocator, encode.callRel(rel).slice());
    try stub.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, stub.items.len);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, stub.items);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86_64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-x86_64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.x86, ...) emits+links+runs to exit 42 (qemu-i386)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .x86, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.x86.encode;
    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.ebx, .eax).slice()); // ebx = main's return (eax)
    try exitseq.appendSlice(allocator, encode.movImm(.eax, 1).slice()); // eax = 1 (exit)
    try exitseq.appendSlice(allocator, encode.int80().slice());
    const stub_len: u64 = 5 + exitseq.items.len;

    const base: u64 = 0x08048000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + 5)));
    var stub: std.ArrayList(u8) = .empty;
    defer stub.deinit(allocator);
    try stub.appendSlice(allocator, encode.callRel(rel).slice());
    try stub.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, stub.items.len);

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, stub.items);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-i386", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

test "cross-target: writeObjectDataFor(.riscv64, ...) emits+links+runs to exit 42 (qemu-riscv64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildMain(allocator);
    defer main_fn.deinit();
    const obj = try target.native.writeObjectDataFor(allocator, .riscv64, &.{.{ .name = "main", .func = &main_fn }}, &.{});
    defer allocator.free(obj);

    const encode = target.riscv64.encode;
    const stub_len: u64 = 12; // jal main (4 bytes), li a7,93 (4 bytes), ecall (4 bytes)

    const base: u64 = 0x10000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    // main's return (i32) is already in a0 (x10), the RISC-V calling convention's
    // integer return register. This matches `exit(a0)`'s expectation directly, so no
    // register move is needed. This mirrors aarch64's x0.
    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const jal_off: i21 = @intCast(main_addr - @as(i64, @intCast(base)));
    const stub = [_]u32{
        encode.jal(.x1, jal_off),
        encode.addi(.x17, .x0, 93), // a7 = 93 (exit)
        encode.ecall(),
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);
    try std.testing.expectEqual(stub_len, @as(u64, 12));

    const elf = try ld.writeElfExec(.riscv64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-riscv64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 42), code);
}

fn runRiscvMain(allocator: std.mem.Allocator, func: *const Function) !u8 {
    const obj = try target.native.writeObjectDataFor(allocator, .riscv64, &.{.{ .name = "main", .func = func }}, &.{});
    defer allocator.free(obj);
    const encode = target.riscv64.encode;
    const stub_len: u64 = 12;
    const base: u64 = 0x10000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);
    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const stub = [_]u32{
        encode.jal(.x1, @intCast(main_addr - @as(i64, @intCast(base)))),
        encode.addi(.x17, .x0, 93),
        encode.ecall(),
    };
    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, std.mem.sliceAsBytes(&stub));
    try program.appendSlice(allocator, image.code);
    const elf = try ld.writeElfExec(.riscv64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);
    return runElf(allocator, std.testing.io, elf, &.{ "qemu-riscv64", "./a.elf" });
}

test "qemu-riscv64 nvfp4 retains separate rounding before add and subtract" {
    const allocator = std.testing.allocator;
    const cases = [_]struct { op: ir.function.BinOp, carrier: u32, block: u32, global: u32, rhs: u32, expected: u32 }{
        .{ .op = .add, .carrier = 0x01, .block = 0x34, .global = 0x3e00_0001, .rhs = 0x3e00_0001, .expected = 0x3e30_0002 },
        .{ .op = .sub, .carrier = 0x01, .block = 0x31, .global = 0x3e00_0001, .rhs = 0x3e00_0001, .expected = 0xbdb8_0002 },
    };
    for (cases) |case| {
        var func = try buildNvFp4ArithmeticBoundaryMain(allocator, case.op, case.carrier, case.block, case.global, case.rhs, case.expected);
        defer func.deinit();
        const exit_code = runRiscvMain(allocator, &func) catch |err| switch (err) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expectEqual(@as(u8, 0), exit_code);
    }
}

/// `main(x)` in the shape that made the x86_64 register allocator refuse a function: `keep` is an
/// argument of the call AND is read after the call, twelve other values are live across the same
/// call, and six more arguments die at it. Every gpr the pool holds is then wanted at the call
/// position, so the blocked-register pick must take a register the call does NOT clobber, or `keep`
/// loses its value across the call. With `x = 1`: live is 2..13 (sum 90), the six arguments are
/// 2..7 (sum 27), `keep` is 2, `callee` answers 27 + 2 = 29, and main answers 29 + 2 + 90 = 121.
fn buildCrossCallArg(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);

    const keep = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = x } });

    const nlive = 12;
    var live: [nlive]ir.function.Value = undefined;
    for (&live, 0..) |*v, i| {
        v.* = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = if (i == 0) x else live[i - 1] } });
    }

    const nargs = 6;
    var args: [nargs + 1]ir.function.Value = undefined;
    for (args[0..nargs], 0..) |*a, i| {
        a.* = try f.appendInst(b, t, .{ .arith = .{ .op = .mul, .lhs = x, .rhs = live[i] } });
    }
    args[nargs] = keep;
    const called = try f.appendCall(b, t, "callee", &args);

    var acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = called, .rhs = keep } });
    for (live) |v| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = v } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });
    return f;
}

/// `callee(a0..a6)` answers the sum of its seven arguments.
fn buildSum7(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const b = try f.appendBlock();
    var ps: [7]ir.function.Value = undefined;
    for (&ps) |*p| p.* = try f.appendBlockParam(b, t);
    var acc = ps[0];
    for (ps[1..]) |p| acc = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(acc) });
    return f;
}

test "cross-target: an x86_64 argument that is also live across its own call runs to 121 (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildCrossCallArg(allocator);
    defer main_fn.deinit();
    var callee_fn = try buildSum7(allocator);
    defer callee_fn.deinit();

    const obj = try target.native.writeObjectDataFor(allocator, .x86_64, &.{
        .{ .name = "main", .func = &main_fn },
        .{ .name = "callee", .func = &callee_fn },
    }, &.{});
    defer allocator.free(obj);

    const encode = target.x86_64.encode;
    // The stub passes x = 1 in rdi, calls main, and exits with its return value.
    var pre: std.ArrayList(u8) = .empty;
    defer pre.deinit(allocator);
    try pre.appendSlice(allocator, encode.movImm(.rdi, 1, true).slice());

    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());

    const stub_len: u64 = pre.items.len + 5 + exitseq.items.len;
    const base: u64 = 0x10000000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + pre.items.len + 5)));

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, pre.items);
    try program.appendSlice(allocator, encode.callRel(rel).slice());
    try program.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, program.items.len);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86_64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-x86_64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 121), code);
}

/// `main(x)` in the shape the x86_64 register allocator used to REFUSE: fourteen i32 block params,
/// all of them live across ONE call, and the first SEVEN of them are the arguments of that call.
///
/// Each of those seven needs a register AT the call and needs the same value again AFTER the call.
/// x86_64 keeps only five callee-saved gpr, so two of the seven cannot stay in a register across the
/// call. The allocator answered `error.Unsupported` for them. The reload-at-use split serves them
/// now: the value waits in a slot, returns to a caller-saved register for the one position the call
/// reads it at, and goes back to the slot in front of the clobber. That placement is only CORRECT if
/// the store lands ahead of the call, so this test runs the code and checks the number.
///
/// With `x = 1` the params are 2..15. `callee` adds the first seven, 2..8, and answers 35. `main`
/// then adds every one of the fourteen, 2 + 3 + ... + 15 = 119, so it answers 35 + 119 = 154.
fn buildFourteenParamCall(allocator: std.mem.Allocator) !Function {
    var f = Function.init(allocator);
    errdefer f.deinit();
    const t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } });
    const entry = try f.appendBlock();
    const body = try f.appendBlock();

    // The entry block holds ONE parameter, so the stub below passes `x` in rdi and needs no stack
    // arguments. It makes the fourteen values from `x` and hands them to `body` as block params.
    const x = try f.appendBlockParam(entry, t);
    const nparams = 14;
    var seed: [nparams]ir.function.Value = undefined;
    for (&seed, 0..) |*v, i| {
        const k = try f.appendInst(entry, t, .{ .iconst = @intCast(i + 1) });
        v.* = try f.appendInst(entry, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = k } });
    }
    try f.setJump(entry, body, &seed);

    var ps: [nparams]ir.function.Value = undefined;
    for (&ps) |*p| p.* = try f.appendBlockParam(body, t);
    const called = try f.appendCall(body, t, "callee", ps[0..7]);

    // The reduction reads every param after the call, so all fourteen are live across it.
    var acc = called;
    for (ps) |p| acc = try f.appendInst(body, t, .{ .arith = .{ .op = .add, .lhs = acc, .rhs = p } });
    f.setTerminator(body, .{ .ret = ir.function.Ret.one(acc) });
    return f;
}

test "cross-target: fourteen params with seven of them call arguments run to 154 (qemu-x86_64)" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;

    var main_fn = try buildFourteenParamCall(allocator);
    defer main_fn.deinit();
    var callee_fn = try buildSum7(allocator);
    defer callee_fn.deinit();

    const obj = try target.native.writeObjectDataFor(allocator, .x86_64, &.{
        .{ .name = "main", .func = &main_fn },
        .{ .name = "callee", .func = &callee_fn },
    }, &.{});
    defer allocator.free(obj);

    const encode = target.x86_64.encode;
    // The stub passes x = 1 in rdi, calls main, and exits with its return value.
    var pre: std.ArrayList(u8) = .empty;
    defer pre.deinit(allocator);
    try pre.appendSlice(allocator, encode.movImm(.rdi, 1, true).slice());

    var exitseq: std.ArrayList(u8) = .empty;
    defer exitseq.deinit(allocator);
    try exitseq.appendSlice(allocator, encode.movReg(.rdi, .rax).slice());
    try exitseq.appendSlice(allocator, encode.movImm(.rax, 60, true).slice());
    try exitseq.appendSlice(allocator, encode.syscall().slice());

    const stub_len: u64 = pre.items.len + 5 + exitseq.items.len;
    const base: u64 = 0x10000000;
    var image = try ld.linkObjects(allocator, &.{obj}, base + stub_len);
    defer image.deinit(allocator);

    const main_addr: i64 = @intCast(image.addressOf("main").?);
    const rel: i32 = @intCast(main_addr - @as(i64, @intCast(base + pre.items.len + 5)));

    var program: std.ArrayList(u8) = .empty;
    defer program.deinit(allocator);
    try program.appendSlice(allocator, pre.items);
    try program.appendSlice(allocator, encode.callRel(rel).slice());
    try program.appendSlice(allocator, exitseq.items);
    try std.testing.expectEqual(stub_len, program.items.len);
    try program.appendSlice(allocator, image.code);

    const elf = try ld.writeElfExec(.x86_64, allocator, program.items, stub_len + image.memsz, base, base);
    defer allocator.free(elf);

    const code = runElf(allocator, io, elf, &.{ "qemu-x86_64", "./a.elf" }) catch |e| switch (e) {
        error.SkipZigTest => return error.SkipZigTest,
        else => return e,
    };
    try std.testing.expectEqual(@as(u8, 154), code);
}
