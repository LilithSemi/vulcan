//! QEMU user-mode runner: `qemu-riscv64`. Runs the shared codegen/optimization case corpus as plain
//! Linux static ELFs (syscall write/exit), so the RISC-V backend executes on any machine with qemu
//! even when River and Spike are absent. Skips when qemu-riscv64 is not on PATH.

const std = @import("std");
const ir = @import("vulcan-ir");
const cases = @import("cases.zig");
const division_cases = @import("../../tests/f32_division_cases.zig");
const emit = @import("../emit.zig");
const encode = @import("../encode.zig");
const harness = @import("harness.zig");
const isel = @import("../isel.zig");

const Function = ir.function.Function;

const low_float_pressure_lanes = 24;
const select_pressure_lanes = 20;
const reinterpret_pressure_lanes = 24;

fn buildLowFloatSpillPressure(func: *Function) !void {
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const input = try func.appendBlockParam(block, u8_t);
    const anchor_bits = try func.appendInst(block, u32_t, .{ .convert = .{ .value = input } });
    var reinterpret_sources: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_sources, 0..) |*source, index| {
        const mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x3000 + index * 37) });
        source.* = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = mask } });
    }
    const bool_t = try func.types.intern(.bool);
    var select_conds: [select_pressure_lanes]ir.function.Value = undefined;
    var select_thens: [select_pressure_lanes]ir.function.Value = undefined;
    var select_elses: [select_pressure_lanes]ir.function.Value = undefined;
    for (0..select_pressure_lanes) |index| {
        const match = try func.appendInst(block, u8_t, .{ .iconst = @intCast(index) });
        select_conds[index] = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = input, .rhs = match } });
        const then_mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x1000 + index * 17) });
        const else_mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x2000 + index * 29) });
        select_thens[index] = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = then_mask } });
        select_elses[index] = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = else_mask } });
    }

    // Build every payload before decoding any of them. They remain simultaneously live into the
    // decode phase and force the expanded integer selects and the u32->f32 boundary through slots.
    var payloads: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&payloads, 0..) |*payload, index| payload.* = try func.appendArithImm(block, u8_t, .add, input, @intCast(index));
    var select_results: [select_pressure_lanes]ir.function.Value = undefined;
    for (&select_results, select_conds, select_thens, select_elses) |*result, cond, then_value, else_value| {
        result.* = try func.appendInst(block, u32_t, .{ .select = .{ .cond = cond, .then = then_value, .@"else" = else_value } });
    }
    var reinterpret_floats: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_floats, reinterpret_sources) |*value, source| value.* = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = source } });

    // Keep more decoded floats live than the 22-register scalar-float pool. Encoding starts only
    // after all decodes, so both reinterpret directions encounter real float and integer pressure.
    var decoded: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&decoded, payloads) |*value, payload| value.* = try func.appendInst(block, f32_t, .{ .decode_low_float = .{ .format = .f8_e4m3, .value = payload } });
    var reinterpret_roundtrips: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_roundtrips, reinterpret_floats) |*value, source| value.* = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = source } });

    var encoded: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&encoded, decoded) |*payload, value| payload.* = try func.appendInst(block, u8_t, .{ .encode_low_float = .{ .format = .f8_e4m3, .value = value } });

    var result = try func.appendInst(block, u32_t, .{ .convert = .{ .value = encoded[0] } });
    for (encoded[1..]) |payload| {
        const wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = payload } });
        result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = wide } });
    }
    for (reinterpret_roundtrips) |roundtrip| result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = roundtrip } });
    for (select_results) |selected| result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = selected } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
}

fn pressureReference(input: u8) u32 {
    var result: u32 = 0;
    for (0..select_pressure_lanes) |index| {
        const mask: u32 = if (input == index) @intCast(0x1000 + index * 17) else @intCast(0x2000 + index * 29);
        result ^= @as(u32, input) ^ mask;
    }
    for (0..reinterpret_pressure_lanes) |index| result ^= @as(u32, input) ^ @as(u32, @intCast(0x3000 + index * 37));
    for (0..low_float_pressure_lanes) |index| {
        const payload: u8 = input + @as(u8, @intCast(index));
        const value = ir.low_float.decode(.f8_e4m3, payload) catch unreachable;
        result ^= ir.low_float.encode(.f8_e4m3, value);
    }
    return result;
}

fn buildNvFp4SpillPressure(func: *Function) !void {
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const bool_t = try func.types.intern(.bool);
    const block = try func.appendBlock();
    const input = try func.appendBlockParam(block, u8_t);
    const anchor_bits = try func.appendInst(block, u32_t, .{ .convert = .{ .value = input } });
    var reinterpret_sources: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_sources, 0..) |*source, index| {
        const mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x3000 + index * 37) });
        source.* = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = mask } });
    }
    var select_conds: [select_pressure_lanes]ir.function.Value = undefined;
    var select_thens: [select_pressure_lanes]ir.function.Value = undefined;
    var select_elses: [select_pressure_lanes]ir.function.Value = undefined;
    for (0..select_pressure_lanes) |index| {
        const match = try func.appendInst(block, u8_t, .{ .iconst = @intCast(index) });
        select_conds[index] = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = input, .rhs = match } });
        const then_mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x1000 + index * 17) });
        const else_mask = try func.appendInst(block, u32_t, .{ .iconst = @intCast(0x2000 + index * 29) });
        select_thens[index] = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = then_mask } });
        select_elses[index] = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = anchor_bits, .rhs = else_mask } });
    }
    var payloads: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&payloads, 0..) |*payload, index| payload.* = try func.appendArithImm(block, u8_t, .add, input, @intCast(index));
    var select_results: [select_pressure_lanes]ir.function.Value = undefined;
    for (&select_results, select_conds, select_thens, select_elses) |*result, cond, then_value, else_value| {
        result.* = try func.appendInst(block, u32_t, .{ .select = .{ .cond = cond, .then = then_value, .@"else" = else_value } });
    }
    var reinterpret_floats: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_floats, reinterpret_sources) |*value, source| value.* = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = source } });

    const block_scale = try func.appendInst(block, u8_t, .{ .iconst = 0x38 });
    const global_scale = try func.appendInst(block, f32_t, .{ .fconst = 1.0 });
    var decoded: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&decoded, payloads) |*value, payload| value.* = try func.appendInst(block, f32_t, .{ .dequantize_nvfp4 = .{
        .value = payload,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = .multiply,
        .global_application = .multiply,
    } });
    var reinterpret_roundtrips: [reinterpret_pressure_lanes]ir.function.Value = undefined;
    for (&reinterpret_roundtrips, reinterpret_floats) |*value, source| value.* = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = source } });
    var encoded: [low_float_pressure_lanes]ir.function.Value = undefined;
    for (&encoded, decoded) |*payload, value| payload.* = try func.appendInst(block, u8_t, .{ .quantize_nvfp4 = .{
        .value = value,
        .block_scale = block_scale,
        .global_scale = global_scale,
        .block_application = .multiply,
        .global_application = .multiply,
    } });

    var result = try func.appendInst(block, u32_t, .{ .convert = .{ .value = encoded[0] } });
    for (encoded[1..]) |payload| {
        const wide = try func.appendInst(block, u32_t, .{ .convert = .{ .value = payload } });
        result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = wide } });
    }
    for (reinterpret_roundtrips) |roundtrip| result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = roundtrip } });
    for (select_results) |selected| result = try func.appendInst(block, u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = result, .rhs = selected } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
}

fn nvfp4PressureReference(input: u8) u32 {
    var result: u32 = 0;
    for (0..select_pressure_lanes) |index| {
        const mask: u32 = if (input == index) @intCast(0x1000 + index * 17) else @intCast(0x2000 + index * 29);
        result ^= @as(u32, input) ^ mask;
    }
    for (0..reinterpret_pressure_lanes) |index| result ^= @as(u32, input) ^ @as(u32, @intCast(0x3000 + index * 37));
    for (0..low_float_pressure_lanes) |index| result ^= (@as(u32, input) + @as(u32, @intCast(index))) & 0x0f;
    return result;
}

fn buildIntToFloatBits(func: *Function) !void {
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const block = try func.appendBlock();
    const bits = try func.appendBlockParam(block, u32_t);
    const value = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = bits } });
    const slot = try func.appendInst(block, ptr_t, .{ .alloca = .{ .elem = u32_t } });
    try func.appendStore(block, value, slot);
    const result = try func.appendInst(block, u32_t, .{ .load = .{ .ptr = slot } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
}

fn buildFloatToIntBits(func: *Function) !void {
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const ptr_t = try func.types.ptrGlobal();
    const block = try func.appendBlock();
    const bits = try func.appendBlockParam(block, u32_t);
    const slot = try func.appendInst(block, ptr_t, .{ .alloca = .{ .elem = u32_t } });
    try func.appendStore(block, bits, slot);
    const value = try func.appendInst(block, f32_t, .{ .load = .{ .ptr = slot } });
    const result = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
}

test "qemu-user-riscv scalar u32/f32 reinterpret preserves every low bit" {
    const patterns = [_]u32{
        0x0000_0000,
        0x8000_0000,
        0x3f80_0000,
        0x7f80_0000,
        0x7f80_0001,
        0x7fc1_2345,
        0xffa5_4321,
        0xffff_ffff,
    };
    inline for (.{ buildIntToFloatBits, buildFloatToIntBits }) |build| {
        var func = Function.init(std.testing.allocator);
        defer func.deinit();
        try build(&func);
        for (patterns) |pattern| {
            const arg: i64 = @as(i32, @bitCast(pattern));
            const got = harness.runFunc(std.testing.io, std.testing.allocator, &func, &.{arg}, harness.qemu_user) catch |err| switch (err) {
                error.SkipZigTest => return error.SkipZigTest,
                else => return err,
            };
            try std.testing.expectEqual(@as(i64, pattern), got);
        }
    }
}

test "qemu-user-riscv low float production expansion executes through spilled selects and reinterprets" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildLowFloatSpillPressure(&func);

    var evidence: isel.LowFloatSpillEvidence = .{};
    var compiled = try isel.compileFunctionWithLowFloatSpillEvidence(allocator, &func, .{}, &evidence);
    defer compiled.deinit(allocator);
    try std.testing.expect(evidence.int_spill_count > 0);
    try std.testing.expect(evidence.float_spill_count > 0);
    try std.testing.expect(evidence.split_action_count > 0);
    try std.testing.expect(evidence.select_condition_spilled_at_use);
    try std.testing.expect(evidence.select_then_spilled_at_use);
    try std.testing.expect(evidence.select_else_spilled_at_use);
    try std.testing.expect(evidence.select_result_has_slot);
    try std.testing.expect(evidence.select_operands_and_result);
    try std.testing.expect(evidence.u32_to_f32_source_and_result);
    try std.testing.expect(evidence.f32_to_u32_source_and_result);

    for ([_]u8{ 0, 1, 17, 64, 127, 191 }) |input| {
        const got = harness.runCode(std.testing.io, allocator, compiled.code, &.{input}, harness.qemu_user) catch |err| switch (err) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expectEqual(@as(i64, pressureReference(input)), got);
    }
}

test "qemu-user-riscv nvfp4 production expansion executes with integer and float spills" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try buildNvFp4SpillPressure(&func);

    var evidence: isel.LowFloatSpillEvidence = .{};
    var compiled = try isel.compileFunctionWithLowFloatSpillEvidence(allocator, &func, .{}, &evidence);
    defer compiled.deinit(allocator);
    try std.testing.expect(evidence.int_spill_count > 0);
    try std.testing.expect(evidence.float_spill_count > 0);
    try std.testing.expect(evidence.split_action_count > 0);
    try std.testing.expect(evidence.select_condition_spilled_at_use);
    try std.testing.expect(evidence.select_then_spilled_at_use);
    try std.testing.expect(evidence.select_else_spilled_at_use);
    try std.testing.expect(evidence.select_result_has_slot);
    try std.testing.expect(evidence.select_operands_and_result);
    try std.testing.expect(evidence.u32_to_f32_source_and_result);
    try std.testing.expect(evidence.f32_to_u32_source_and_result);

    for ([_]u8{ 0, 1, 17, 64, 127, 191 }) |input| {
        const got = harness.runCode(std.testing.io, allocator, compiled.code, &.{input}, harness.qemu_user) catch |err| switch (err) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return err,
        };
        try std.testing.expectEqual(@as(i64, nvfp4PressureReference(input)), got);
    }
}

fn appendU32Immediate(allocator: std.mem.Allocator, words: *std.ArrayList(u32), register: encode.Reg, value: u32) !void {
    const high: u20 = @truncate((value +% 0x800) >> 12);
    const low: i12 = @bitCast(@as(u12, @truncate(value)));
    try words.append(allocator, encode.lui(register, high));
    try words.append(allocator, encode.addi(register, register, low));
}

fn instructionOffset(from: usize, to: usize) i13 {
    return @intCast((@as(isize, @intCast(to)) - @as(isize, @intCast(from))) * 4);
}

fn runDivisionBatch(
    io: std.Io,
    allocator: std.mem.Allocator,
    division_code: []const u32,
    pairs: []const division_cases.Pair,
    first_corpus_index: u32,
) !u64 {
    std.debug.assert(pairs.len > 0);
    std.debug.assert(pairs.len <= 16 * 1024);

    var wrapper: std.ArrayList(u32) = .empty;
    defer wrapper.deinit(allocator);

    const data_address_index = wrapper.items.len;
    try wrapper.append(allocator, encode.auipc(.x8, 0));
    try wrapper.append(allocator, encode.addi(.x8, .x8, 0));
    try appendU32Immediate(allocator, &wrapper, .x9, @intCast(pairs.len));
    try appendU32Immediate(allocator, &wrapper, .x21, first_corpus_index);
    try wrapper.append(allocator, encode.addi(.x18, .x0, 0));
    try wrapper.append(allocator, encode.addi(.x19, .x0, 0));
    try appendU32Immediate(allocator, &wrapper, .x22, 0x7f80_0000);
    try appendU32Immediate(allocator, &wrapper, .x24, 0x7fc0_0000);

    const loop_index = wrapper.items.len;
    try wrapper.append(allocator, encode.lwu(.x10, .x8, 0));
    try wrapper.append(allocator, encode.lwu(.x11, .x8, 4));
    try wrapper.append(allocator, encode.lwu(.x20, .x8, 8));
    const call_index = wrapper.items.len;
    try wrapper.append(allocator, encode.jal(.x1, 0));
    try wrapper.append(allocator, encode.slli(.x5, .x10, 33));
    try wrapper.append(allocator, encode.srli(.x5, .x5, 33));
    const nan_branch_index = wrapper.items.len;
    try wrapper.append(allocator, encode.bltu(.x22, .x5, 0));
    const mismatch_branch_index = wrapper.items.len;
    try wrapper.append(allocator, encode.bne(.x10, .x20, 0));

    const continue_index = wrapper.items.len;
    try wrapper.append(allocator, encode.addi(.x8, .x8, 12));
    try wrapper.append(allocator, encode.addi(.x9, .x9, -1));
    try wrapper.append(allocator, encode.addi(.x21, .x21, 1));
    try wrapper.append(allocator, encode.bne(.x9, .x0, instructionOffset(wrapper.items.len, loop_index)));
    try wrapper.append(allocator, encode.slli(.x19, .x19, 32));
    try wrapper.append(allocator, encode.or_(.x10, .x19, .x18));
    try wrapper.append(allocator, encode.addi(.x2, .x2, -16));
    try wrapper.append(allocator, encode.sd(.x10, .x2, 0));
    try wrapper.append(allocator, encode.addi(.x10, .x0, 1));
    try wrapper.append(allocator, encode.addi(.x11, .x2, 0));
    try wrapper.append(allocator, encode.addi(.x12, .x0, 8));
    try wrapper.append(allocator, encode.addi(.x17, .x0, 64));
    try wrapper.append(allocator, encode.ecall());
    try wrapper.append(allocator, encode.addi(.x10, .x0, 0));
    try wrapper.append(allocator, encode.addi(.x17, .x0, 93));
    try wrapper.append(allocator, encode.ecall());

    const mismatch_index = wrapper.items.len;
    try wrapper.append(allocator, encode.addi(.x18, .x18, 1));
    try wrapper.append(allocator, encode.bne(.x19, .x0, instructionOffset(wrapper.items.len, continue_index)));
    try wrapper.append(allocator, encode.addi(.x19, .x21, 0));
    try wrapper.append(allocator, encode.jal(.x0, instructionOffset(wrapper.items.len, continue_index)));

    const nan_index = wrapper.items.len;
    try wrapper.append(allocator, encode.addi(.x10, .x24, 0));
    try wrapper.append(allocator, encode.jal(.x0, instructionOffset(wrapper.items.len, mismatch_branch_index)));

    const function_index = wrapper.items.len;
    wrapper.items[call_index] = encode.jal(.x1, instructionOffset(call_index, function_index));
    wrapper.items[nan_branch_index] = encode.bltu(.x22, .x5, instructionOffset(nan_branch_index, nan_index));
    wrapper.items[mismatch_branch_index] = encode.bne(.x10, .x20, instructionOffset(mismatch_branch_index, mismatch_index));

    var words: std.ArrayList(u32) = .empty;
    defer words.deinit(allocator);
    try words.appendSlice(allocator, wrapper.items);
    try words.appendSlice(allocator, division_code);
    const data_index = words.items.len;
    const data_offset: i64 = @as(i64, @intCast((data_index - data_address_index) * 4));
    const data_high: u20 = @intCast((data_offset + 0x800) >> 12);
    const data_low: i12 = @intCast(data_offset - (@as(i64, data_high) << 12));
    words.items[data_address_index] = encode.auipc(.x8, data_high);
    words.items[data_address_index + 1] = encode.addi(.x8, .x8, data_low);

    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(allocator);
    const code_bytes = try emit.emitBytes(allocator, words.items);
    defer allocator.free(code_bytes);
    try image.appendSlice(allocator, code_bytes);
    for (pairs) |pair| {
        const numerator: f32 = @bitCast(pair.numerator);
        const denominator: f32 = @bitCast(pair.denominator);
        const reference = numerator / denominator;
        const expected: u32 = if (std.math.isNan(reference)) 0x7fc0_0000 else @bitCast(reference);
        var record: [12]u8 = undefined;
        std.mem.writeInt(u32, record[0..4], pair.numerator, .little);
        std.mem.writeInt(u32, record[4..8], pair.denominator, .little);
        std.mem.writeInt(u32, record[8..12], expected, .little);
        try image.appendSlice(allocator, &record);
    }

    const user_base: u64 = 0x10000;
    const elf = try (@import("vulcan-link")).writeElfExec(.riscv64, allocator, image.items, image.items.len, user_base, user_base);
    defer allocator.free(elf);
    var temporary = std.testing.tmpDir(.{});
    defer temporary.cleanup();
    try temporary.dir.writeFile(io, .{ .sub_path = "firmware.elf", .data = elf, .flags = .{ .permissions = .executable_file } });
    const arguments = try harness.qemu_user.buildArgv(allocator, "firmware.elf");
    defer allocator.free(arguments);
    const result = std.process.run(allocator, io, .{ .argv = arguments, .cwd = .{ .dir = temporary.dir } }) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);
    if (result.stdout.len < 8) return error.BackendFailed;
    return std.mem.readInt(u64, result.stdout[result.stdout.len - 8 ..][0..8], .little);
}

test "qemu-user-riscv executes shared audited exact f32 division corpus in bounded batches" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const f32_t = try func.types.intern(.{ .float = .f32 });
    const block = try func.appendBlock();
    const numerator_bits = try func.appendBlockParam(block, u32_t);
    const denominator_bits = try func.appendBlockParam(block, u32_t);
    const numerator = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = numerator_bits } });
    const denominator = try func.appendInst(block, f32_t, .{ .unary = .{ .op = .reinterpret, .value = denominator_bits } });
    const quotient = try func.appendInst(block, f32_t, .{ .arith = .{ .op = .div, .lhs = numerator, .rhs = denominator } });
    const result = try func.appendInst(block, u32_t, .{ .unary = .{ .op = .reinterpret, .value = quotient } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    try std.testing.expect(try ir.expand.expandF32Div(allocator, &func));
    var division_code = try harness.compileFunc(allocator, &func);
    defer division_code.deinit(allocator);
    var corpus = try division_cases.build(allocator);
    defer corpus.deinit();
    const batch_size = 16 * 1024;
    var batch_start: usize = 0;
    while (batch_start < corpus.pairs.len) : (batch_start += batch_size) {
        const batch_end = @min(batch_start + batch_size, corpus.pairs.len);
        const evidence = runDivisionBatch(
            std.testing.io,
            allocator,
            division_code.items,
            corpus.pairs[batch_start..batch_end],
            @intCast(batch_start + 1),
        ) catch |err| switch (err) {
            error.SkipZigTest => return error.SkipZigTest,
            else => return err,
        };
        const MismatchEvidence = struct { count: u32, first_corpus_index: u32 };
        try std.testing.expectEqual(
            MismatchEvidence{ .count = 0, .first_corpus_index = 0 },
            MismatchEvidence{ .count = @truncate(evidence), .first_corpus_index = @truncate(evidence >> 32) },
        );
    }
}

test "qemu-user-riscv: shared codegen and optimization cases" {
    try cases.runAll(std.testing.io, std.testing.allocator, harness.qemu_user);
}

// binary128 (f128) DATA MOVEMENT on lp64d: an f128 has no register form (it is 16 bytes in a GPR
// PAIR or memory), so it lives in a 16-byte stack slot and materializes into an aligned a-register
// pair only at ABI boundaries. f128 arithmetic/compare/convert are soft-fp libcalls (the shared
// softfp pass), so none appears here; these cases prove the 16-byte value survives the pair ABI and
// the memory paths intact. Each runs under qemu-riscv64 and asserts all 128 bits.

/// The two 64-bit halves (low, high) of an f128's bit pattern.
fn halves(v: f128) [2]u64 {
    const bits: u128 = @bitCast(v);
    return .{ @truncate(bits), @truncate(bits >> 64) };
}

test "qemu-user-riscv f128: identity carries all 128 bits through the a0:a1 pair" {
    const allocator = std.testing.allocator;
    const v: f128 = 0.1;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    const h = halves(v);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

test "qemu-user-riscv f128: return the second argument (a2:a3 -> a0:a1)" {
    const allocator = std.testing.allocator;
    const v0: f128 = 2.5;
    const v1: f128 = 1.0 / 3.0;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    _ = try f.appendBlockParam(b, t);
    const bb = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(bb) });
    const h0 = halves(v0);
    const h1 = halves(v1);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h0[0], h0[1], h1[0], h1[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v1)), got);
}

test "qemu-user-riscv f128: an int arg before an f128 forces the aligned pair (a2:a3, a1 skipped)" {
    const allocator = std.testing.allocator;
    const v: f128 = 3.141592653589793238462643383279502884;
    var f = Function.init(allocator);
    defer f.deinit();
    const i64_t = try f.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    _ = try f.appendBlockParam(b, i64_t); // consumes a0; the f128 must then align to a2:a3
    const a = try f.appendBlockParam(b, t);
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(a) });
    const h = halves(v);
    // argHalves: a0 = x (0x1234), a1 = filler (skipped by alignment), a2:a3 = the f128.
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ 0x1234, 0xdead, h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

test "qemu-user-riscv f128: constant materialized from its two 64-bit halves" {
    const allocator = std.testing.allocator;
    const c: f128 = 2.718281828459045235360287471352662497;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const k = try f.appendInst(b, t, .{ .fconst128 = @bitCast(c) });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(k) });
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{}, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(c)), got);
}

test "qemu-user-riscv f128: alloca store then load round-trips all 16 bytes" {
    const allocator = std.testing.allocator;
    const v: f128 = -0.7;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const ptr_t = try f.types.ptrGlobal();
    const b = try f.appendBlock();
    const a = try f.appendBlockParam(b, t);
    const slot = try f.appendInst(b, ptr_t, .{ .alloca = .{ .elem = t } });
    try f.appendStore(b, a, slot);
    const r = try f.appendInst(b, t, .{ .load = .{ .ptr = slot } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });
    const h = halves(v);
    const got = try harness.runFuncQuad(std.testing.io, allocator, &f, &.{ h[0], h[1] }, harness.qemu_user);
    try std.testing.expectEqual(@as(u128, @bitCast(v)), got);
}

// An f128 add has no riscv64 instruction; the shared softfp pass lowers it to a call, and the
// backend must place the two f128 arguments in the a0:a1 and a2:a3 register pairs and emit a call
// relocation against the undefined `__addtf3`. Compile-only (no riscv64 libgcc in the harness to
// execute the arithmetic), but it exercises the real call-argument pair placement.
test "riscv64: an f128 add compiles to an undefined __addtf3 soft-fp call relocation" {
    const allocator = std.testing.allocator;
    var f = Function.init(allocator);
    defer f.deinit();
    const t = try f.types.intern(.{ .float = .f128 });
    const b = try f.appendBlock();
    const x = try f.appendBlockParam(b, t);
    const y = try f.appendBlockParam(b, t);
    const r = try f.appendInst(b, t, .{ .arith = .{ .op = .add, .lhs = x, .rhs = y } });
    f.setTerminator(b, .{ .ret = ir.function.Ret.one(r) });

    var compiled = try isel.compileFunction(allocator, &f, .{});
    defer compiled.deinit(allocator);
    var addtf3: usize = 0;
    for (compiled.relocs) |rel| {
        if (std.mem.eql(u8, rel.symbol, "__addtf3")) addtf3 += 1;
    }
    try std.testing.expectEqual(@as(usize, 1), addtf3);
}
