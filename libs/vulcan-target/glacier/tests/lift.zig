//! Running a Glacier kernel as native code: the words lifted into IR, compiled for the host, and
//! checked against the warp model beside this file.
//!
//! The pair is the point. A kernel that only ran through the lifter would prove that the lifter
//! agrees with itself, so every program here runs on both and the registers and the memory are
//! diffed. The model decodes the words directly and shares no code with the lifter, so the two
//! disagree whenever either is wrong.
//!
//! The model takes only word loads and stores, so the narrower accesses have expectations of their
//! own further down rather than a model to be diffed against.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");
const warp = @import("warp.zig");

const lift = target.glacier.lift;
const rv = target.glacier.encode.rv;
const Function = ir.function.Function;

/// The address the test programs reach memory at, and how much of it they are given.
const base: u32 = 0x8000_0000;
const words = 16;

/// `lui x1, 0x80000` puts the base of that memory in a register, which is how every program here
/// starts. A Glacier kernel reaches memory by absolute address.
const base_high: u20 = base >> 12;

const Run = *const fn (state: *lift.Warp) callconv(.c) u32;

/// Lift a program, compile it for this machine, and run it.
fn native(allocator: std.mem.Allocator, program: []const u32, entry: u32, state: *lift.Warp) !lift.Exit {
    var func = Function.init(allocator);
    defer func.deinit();
    try lift.region(allocator, &func, program, entry);

    // A function the backend would refuse is worth catching here, where the diagnostic says which
    // instruction is wrong.
    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();
    const run = code.entry(Run, 0);
    return @fromBackingInt(@intCast(run(state)));
}

/// What the host answers for an access the compiled code does not serve itself, which in these
/// programs is every atomic: neither host backend lowers one yet, so the read and the write are a
/// real atomic against the same array here instead.
const Served = struct {
    words: []u32,
    base: u32,

    fn answer(ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u64 {
        const self: *Served = @ptrCast(@alignCast(ctx.?));
        if (kind & lift.away.atomic == 0) return 0;
        if (at < self.base or at % 4 != 0) return 0;
        const index = (at - self.base) / 4;
        if (index >= self.words.len) return 0;
        const cell = &self.words[index];
        const old = switch ((kind >> lift.away.atomic_shift) & 0b11) {
            0 => @atomicRmw(u32, cell, .Add, value, .seq_cst),
            1 => @atomicRmw(u32, cell, .Xchg, value, .seq_cst),
            2 => @atomicRmw(u32, cell, .And, value, .seq_cst),
            3 => @atomicRmw(u32, cell, .Or, value, .seq_cst),
            else => unreachable,
        };
        return lift.away.answered | old;
    }
};

/// Run one program both ways and diff everything either of them could have changed.
fn expectSame(program: []const u32) !void {
    const allocator = std.testing.allocator;

    var model_memory: [words]u32 = @splat(0);
    var memory: warp.Memory = .{ .base = base, .words = &model_memory };
    var model: warp.Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 1 };
    _ = try warp.run(program, &memory, &model, 10_000);

    var native_memory: [words]u32 = @splat(0);
    var served: Served = .{ .words = &native_memory, .base = base };
    var state: lift.Warp = .{
        .hart_count = 1,
        .fabric = @ptrCast(&native_memory),
        .fabric_at = base,
        .fabric_size = native_memory.len * 4,
        .away = Served.answer,
        .away_ctx = &served,
    };
    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, program, 0, &state));

    // Register zero is left out: it reads zero either way and neither side keeps it.
    for (1..32) |which| {
        if (model.x[which] == state.x[which]) continue;
        std.debug.print(
            "register x{d}: the model says {x}, the compiled code says {x}\n",
            .{ which, model.x[which], state.x[which] },
        );
        return error.TestExpectedEqual;
    }
    try std.testing.expectEqualSlices(u32, &model_memory, &native_memory);
    try std.testing.expectEqual(@as(u32, 0), state.bus_error);
}

test "arithmetic a compiler emits, run both ways" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, 7),
        rv.addi(.x3, .x0, -3),
        rv.mul(.x4, .x2, .x3),
        rv.mulh(.x5, .x2, .x3),
        rv.mulhu(.x6, .x2, .x3),
        rv.slt(.x7, .x3, .x2),
        rv.sltu(.x8, .x3, .x2),
        rv.sub(.x9, .x2, .x3),
        rv.xor_(.x10, .x2, .x3),
        rv.or_(.x11, .x2, .x3),
        rv.and_(.x12, .x2, .x3),
        rv.sll(.x13, .x2, .x3),
        rv.srl(.x14, .x3, .x2),
        rv.sra(.x15, .x3, .x2),
        rv.sw(.x4, .x1, 0),
        rv.sw(.x15, .x1, 4),
        target.glacier.encode.ebreak,
    });
}

test "the immediate forms, including the shifts that share a field" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, -1),
        rv.slli(.x3, .x2, 31),
        rv.srli(.x4, .x2, 31),
        rv.srai(.x5, .x2, 31),
        rv.andi(.x6, .x2, 0x0f0),
        rv.ori(.x7, .x0, 0x123),
        rv.xori(.x8, .x2, 0x7ff),
        rv.sltiu(.x10, .x2, 1),
        rv.sw(.x3, .x1, 0),
        rv.sw(.x5, .x1, 4),
        target.glacier.encode.ebreak,
    });
}

test "a loop with a branch backwards, which is where a kernel spends its time" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, 0),
        rv.addi(.x3, .x0, 0),
        rv.addi(.x4, .x0, 10),
        // The body: add the counter in, step it, and go back while there is more.
        rv.add(.x3, .x3, .x2),
        rv.addi(.x2, .x2, 1),
        rv.blt(.x2, .x4, -8),
        rv.sw(.x3, .x1, 0),
        target.glacier.encode.ebreak,
    });
}

test "every branch relation, taken and not" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, 5),
        rv.addi(.x3, .x0, -5),
        rv.addi(.x4, .x0, 0),
        rv.beq(.x2, .x2, 8),
        rv.addi(.x4, .x4, 1),
        rv.bne(.x2, .x3, 8),
        rv.addi(.x4, .x4, 2),
        rv.blt(.x3, .x2, 8),
        rv.addi(.x4, .x4, 4),
        rv.bge(.x2, .x3, 8),
        rv.addi(.x4, .x4, 8),
        // Unsigned, where the negative number is the larger one and the branch is not taken.
        rv.bltu(.x3, .x2, 8),
        rv.addi(.x4, .x4, 16),
        rv.bgeu(.x3, .x2, 8),
        rv.addi(.x4, .x4, 32),
        rv.sw(.x4, .x1, 0),
        target.glacier.encode.ebreak,
    });
}

test "the divides, with the two answers a host divide would trap on" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, -8),
        rv.addi(.x3, .x0, 3),
        rv.div(.x4, .x2, .x3),
        rv.rem(.x5, .x2, .x3),
        rv.divu(.x6, .x2, .x3),
        rv.remu(.x7, .x2, .x3),
        // By zero: all ones for a divide, and the dividend back for a remainder.
        rv.div(.x8, .x2, .x0),
        rv.rem(.x9, .x2, .x0),
        rv.divu(.x10, .x2, .x0),
        rv.remu(.x11, .x2, .x0),
        // The most negative value over minus one, which wraps to itself.
        rv.lui(.x12, 0x80000),
        rv.addi(.x13, .x0, -1),
        rv.div(.x14, .x12, .x13),
        rv.rem(.x15, .x12, .x13),
        rv.sw(.x14, .x1, 0),
        rv.sw(.x15, .x1, 4),
        target.glacier.encode.ebreak,
    });
}

test "a word carried out to memory and read back in" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.lui(.x2, 0xabcde),
        rv.addi(.x2, .x2, 0x123),
        rv.sw(.x2, .x1, 8),
        rv.lw(.x3, .x1, 8),
        rv.add(.x4, .x3, .x3),
        rv.sw(.x4, .x1, 12),
        target.glacier.encode.ebreak,
    });
}

test "an atomic gives back the value it found" {
    try expectSame(&.{
        rv.lui(.x1, base_high),
        rv.addi(.x2, .x0, 100),
        rv.sw(.x2, .x1, 0),
        rv.addi(.x3, .x0, 11),
        rv.amo(.add, .x4, .x1, .x3, false, false),
        rv.amo(.add, .x5, .x1, .x3, false, false),
        rv.amo(.swap, .x6, .x1, .x3, false, false),
        target.glacier.encode.ebreak,
    });
}

test "a kernel reads which warp it is, and what its host gave it" {
    const allocator = std.testing.allocator;
    const csr = target.glacier.encode.Csr;
    const program = [_]u32{
        rv.lui(.x1, base_high),
        rv.csrrs(.x2, csr.mhartid, .x0),
        rv.csrrs(.x3, csr.hart_count, .x0),
        rv.csrrs(.x4, csr.launchArg(0).?, .x0),
        rv.csrrs(.x5, csr.launchArg(7).?, .x0),
        rv.csrrs(.x6, csr.workgroup_id, .x0),
        rv.csrrs(.x7, csr.grid_dim, .x0),
        rv.sw(.x4, .x1, 0),
        target.glacier.encode.ebreak,
    };

    var memory: [words]u32 = @splat(0);
    var state: lift.Warp = .{
        .hart_id = 3,
        .hart_count = 8,
        .workgroup_id = 2,
        .grid_dim = 16,
        .fabric = @ptrCast(&memory),
        .fabric_at = base,
        .fabric_size = memory.len * 4,
    };
    state.args[0] = 0x8014_0000;
    state.args[7] = 0x0000_0640;

    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 3), state.x[2]);
    try std.testing.expectEqual(@as(u32, 8), state.x[3]);
    try std.testing.expectEqual(@as(u32, 0x8014_0000), state.x[4]);
    try std.testing.expectEqual(@as(u32, 0x0000_0640), state.x[5]);
    try std.testing.expectEqual(@as(u32, 2), state.x[6]);
    try std.testing.expectEqual(@as(u32, 16), state.x[7]);
    // The argument reached memory, so a kernel can use what it read rather than only hold it.
    try std.testing.expectEqual(@as(u32, 0x8014_0000), memory[0]);
}

test "the narrow accesses, which extend from the byte they address" {
    const allocator = std.testing.allocator;
    const program = [_]u32{
        rv.lui(.x1, base_high),
        // 0xff80ff80: both halves negative, and both bytes.
        rv.lui(.x2, 0xff810),
        rv.addi(.x2, .x2, -128),
        rv.sw(.x2, .x1, 0),
        rv.lb(.x3, .x1, 0),
        rv.lbu(.x4, .x1, 0),
        rv.lh(.x5, .x1, 0),
        rv.lhu(.x6, .x1, 0),
        // A byte store leaves its neighbours in the word alone.
        rv.addi(.x7, .x0, 0x11),
        rv.sb(.x7, .x1, 5),
        rv.lw(.x8, .x1, 4),
        target.glacier.encode.ebreak,
    };

    var memory: [words]u32 = @splat(0);
    var state: lift.Warp = .{
        .hart_count = 1,
        .fabric = @ptrCast(&memory),
        .fabric_at = base,
        .fabric_size = memory.len * 4,
    };

    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 0xff80_ff80), memory[0]);
    try std.testing.expectEqual(@as(u32, 0xffff_ff80), state.x[3]);
    try std.testing.expectEqual(@as(u32, 0x0000_0080), state.x[4]);
    try std.testing.expectEqual(@as(u32, 0xffff_ff80), state.x[5]);
    try std.testing.expectEqual(@as(u32, 0x0000_ff80), state.x[6]);
    try std.testing.expectEqual(@as(u32, 0x0000_1100), state.x[8]);
    try std.testing.expectEqual(@as(u32, 0x0000_1100), memory[1]);
}

test "a kernel that says it failed, and the code it reports" {
    const allocator = std.testing.allocator;
    const program = [_]u32{
        rv.addi(.x5, .x0, 17),
        target.glacier.encode.ecall(.x5),
        target.glacier.encode.ebreak,
    };

    var state: lift.Warp = .{ .hart_count = 1 };
    try std.testing.expectEqual(lift.Exit.faulted, try native(allocator, &program, 0, &state));
    // The code is the value the register held and not the register number, which is the whole
    // reason a kernel can report something it computed.
    try std.testing.expectEqual(@as(u32, 17), state.fault_code);
    try std.testing.expectEqual(@as(u32, 4), state.pc);
}

test "a barrier gives the warp back, and the word after it is where it carries on" {
    const allocator = std.testing.allocator;
    const program = [_]u32{
        rv.addi(.x1, .x0, 1),
        target.glacier.encode.warp_barrier,
        rv.addi(.x1, .x1, 1),
        target.glacier.encode.ebreak,
    };

    var state: lift.Warp = .{ .hart_count = 1 };
    try std.testing.expectEqual(lift.Exit.parked, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 8), state.pc);
    try std.testing.expectEqual(@as(u32, 1), state.x[1]);

    // Released, it carries on from the word the park left in the program counter.
    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, state.pc, &state));
    try std.testing.expectEqual(@as(u32, 2), state.x[1]);
}

test "a host can take a spinning kernel down" {
    const allocator = std.testing.allocator;
    // The shape a resident kernel has: wait for a word to change, then do the work. Nothing here
    // changes it, so the loop is the whole program.
    const program = [_]u32{
        rv.lui(.x1, base_high),
        rv.lw(.x2, .x1, 0),
        rv.beq(.x2, .x0, -4),
        rv.addi(.x3, .x0, 7),
        target.glacier.encode.ebreak,
    };

    var memory: [words]u32 = @splat(0);
    var state: lift.Warp = .{
        .hart_count = 1,
        .fabric = @ptrCast(&memory),
        .fabric_at = base,
        .fabric_size = memory.len * 4,
        // Already asked, because a test cannot ask while the code is running inside it.
        .stop = 1,
    };

    try std.testing.expectEqual(lift.Exit.stopped, try native(allocator, &program, 0, &state));
    // It stopped at the top of the loop rather than wherever the check sits, so a core that is
    // started again waits for the same word.
    try std.testing.expectEqual(@as(u32, 4), state.pc);
    try std.testing.expectEqual(@as(u32, 0), state.x[3]);

    // With the word written, the same program runs to its end.
    memory[0] = 1;
    state.stop = 0;
    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 7), state.x[3]);
}

test "an access to an address nothing claims reaches the host and writes no register" {
    const allocator = std.testing.allocator;
    const program = [_]u32{
        rv.lui(.x1, 0x10003),
        rv.addi(.x2, .x0, 99),
        rv.lw(.x3, .x1, 0),
        rv.lw(.x4, .x1, 4),
        rv.sw(.x2, .x1, 8),
        target.glacier.encode.ebreak,
    };

    const Block = struct {
        reads: u32 = 0,
        writes: u32 = 0,
        last_at: u32 = 0,
        last_value: u32 = 0,

        fn answer(ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u64 {
            const self: *@This() = @ptrCast(@alignCast(ctx.?));
            self.last_at = at;
            if (kind & lift.away.store != 0) {
                self.writes += 1;
                self.last_value = value;
                return lift.away.answered;
            }
            self.reads += 1;
            // The first word is a register the card answers. The second is an address nothing
            // claims, and nothing is what it answers with.
            if (at == 0x1000_3000) return lift.away.answered | 0x47444953;
            return 0;
        }
    };

    var block: Block = .{};
    var state: lift.Warp = .{
        .hart_count = 1,
        .x = @splat(0xdead_0000),
        .away = Block.answer,
        .away_ctx = &block,
    };

    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 2), block.reads);
    try std.testing.expectEqual(@as(u32, 1), block.writes);
    try std.testing.expectEqual(@as(u32, 0x1000_3008), block.last_at);
    try std.testing.expectEqual(@as(u32, 99), block.last_value);
    try std.testing.expectEqual(@as(u32, 0x47444953), state.x[3]);
    // The access nothing answered left the register as it was rather than writing a zero that
    // reads like an answer.
    try std.testing.expectEqual(@as(u32, 0xdead_0000), state.x[4]);
}

test "a jump to a register leaves the region and says where it was going" {
    const allocator = std.testing.allocator;
    const program = [_]u32{
        rv.addi(.x1, .x0, 12),
        rv.jalr(.x2, .x1, 0),
        target.glacier.encode.ebreak,
        rv.addi(.x3, .x0, 5),
        target.glacier.encode.ebreak,
    };

    var state: lift.Warp = .{ .hart_count = 1 };
    try std.testing.expectEqual(lift.Exit.jumped, try native(allocator, &program, 0, &state));
    try std.testing.expectEqual(@as(u32, 12), state.pc);
    // The address after the jump, which is what a caller would return to.
    try std.testing.expectEqual(@as(u32, 8), state.x[2]);

    // Lifted again at where it pointed, which is how a caller carries on.
    try std.testing.expectEqual(lift.Exit.retired, try native(allocator, &program, state.pc, &state));
    try std.testing.expectEqual(@as(u32, 5), state.x[3]);
}

test "a jump outside the program is refused rather than compiled" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // A jump forwards past the end of what the build holds.
    const program = [_]u32{ rv.jal(.x0, 64), target.glacier.encode.ebreak };
    try std.testing.expectError(error.Outside, lift.region(allocator, &func, &program, 0));
}

test "a word the core does not decode is refused rather than ignored" {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    // A vector word, which is real and not lifted yet.
    const program = [_]u32{ 0x0000_0057, target.glacier.encode.ebreak };
    try std.testing.expectError(error.Illegal, lift.region(allocator, &func, &program, 0));
}
