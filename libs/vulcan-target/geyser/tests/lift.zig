//! Running Geyser programs as native code: the words lifted into IR, compiled for the host, and
//! checked against results worked out by hand.

const std = @import("std");
const ir = @import("vulcan-ir");
const target = @import("vulcan-target");

const lift = target.geyser.lift;
const encode = target.geyser.encode;
const Function = ir.function.Function;

const Run = *const fn (state: *lift.Core) callconv(.c) u32;

const memory_words = 64;

const Memory = struct {
    words: [memory_words]u16 = @splat(0),

    fn attach(self: *Memory, core: *lift.Core) void {
        core.mem = &self.words;
        core.mem_mask = memory_words - 1;
    }
};

fn word(insn: encode.Insn) u16 {
    return encode.encode(insn) catch unreachable;
}

fn lit(value: u15) u16 {
    return word(.{ .lit = value });
}

fn jmp(to: u13) u16 {
    return word(.{ .jmp = to });
}

fn jz(to: u13) u16 {
    return word(.{ .jz = to });
}

fn call(to: u13) u16 {
    return word(.{ .call = to });
}

/// Lift a program, compile it for this machine and run it.
fn native(program: []const u16, entry: u32, depth: u32, core: *lift.Core) !lift.Exit {
    const allocator = std.testing.allocator;
    var func = Function.init(allocator);
    defer func.deinit();
    try lift.region(allocator, &func, program, entry, depth);

    var diags = try ir.verify.verify(allocator, &func, .low);
    defer diags.deinit();
    try std.testing.expect(diags.ok());

    var code = try target.native.jitFunction(allocator, &func);
    defer code.deinit();
    const run = code.entry(Run, 0);
    return @fromBackingInt(@intCast(run(core)));
}

fn halt(at: u13) u16 {
    return jmp(at);
}

test "a literal sum stored in memory, with the address wrapping" {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{
        lit(5),
        lit(7),
        word(encode.plus),
        // Byte address 0x82 is word 65, which wraps to word 1.
        lit(0x82),
        word(encode.store),
        word(encode.drop),
        halt(6),
    };
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 16, &core));
    try std.testing.expectEqual(@as(u16, 12), memory.words[1]);
    try std.testing.expectEqual(@as(u32, 0), core.dsp);
    try std.testing.expectEqual(@as(u32, 0), core.rsp);
    try std.testing.expectEqual(@as(u32, 6), core.pc);
}

const counting_loop = [_]u16{
    lit(5),
    word(encode.dup),
    jz(12),
    lit(0x40),
    word(encode.fetch),
    lit(1),
    word(encode.plus),
    lit(0x40),
    word(encode.store),
    word(encode.drop),
    word(encode.decrement),
    jmp(1),
    halt(12),
};

test "a loop with jz counts five times" {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    try std.testing.expectEqual(lift.Exit.halted, try native(&counting_loop, 0, 8, &core));
    try std.testing.expectEqual(@as(u16, 5), memory.words[32]);
    try std.testing.expectEqual(@as(u32, 0), core.t);
    try std.testing.expectEqual(@as(u32, 12), core.pc);
}

test "a loop gives the core back when the host asks it to stop, and carries on" {
    var memory: Memory = .{};
    var core: lift.Core = .{ .stop = 1 };
    memory.attach(&core);
    try std.testing.expectEqual(lift.Exit.stopped, try native(&counting_loop, 0, 8, &core));
    try std.testing.expectEqual(@as(u32, 1), core.pc);
    try std.testing.expectEqual(@as(u16, 1), memory.words[32]);

    core.stop = 0;
    try std.testing.expectEqual(lift.Exit.halted, try native(&counting_loop, core.pc, 8, &core));
    try std.testing.expectEqual(@as(u16, 5), memory.words[32]);
}

test "call and exit return to the word after the call" {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{
        lit(3),
        call(7),
        lit(0x20),
        word(encode.store),
        word(encode.drop),
        halt(5),
        halt(6),
        word(encode.dup),
        word(encode.plus),
        word(encode.exit),
    };
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 8, &core));
    try std.testing.expectEqual(@as(u16, 6), memory.words[16]);
    try std.testing.expectEqual(@as(u32, 0), core.rsp);
}

test "the return stack carries a word address through to_r and r_from" {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{
        lit(4),
        word(encode.to_r),
        word(encode.r_from),
        lit(0x20),
        word(encode.store),
        word(encode.drop),
        halt(6),
    };
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 8, &core));
    try std.testing.expectEqual(@as(u16, 4), memory.words[16]);
    try std.testing.expectEqual(@as(u32, 0), core.rsp);
}

test "a return to an address that is not after a call leaves with jumped" {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{
        lit(3),
        word(encode.to_r),
        word(encode.exit),
        halt(3),
    };
    try std.testing.expectEqual(lift.Exit.jumped, try native(&program, 0, 8, &core));
    try std.testing.expectEqual(@as(u32, 3), core.pc);
}

fn expectFault(program: []const u16, depth: u32, pc: u32, bits: u32, dsp: u32) !void {
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    try std.testing.expectEqual(lift.Exit.faulted, try native(program, 0, depth, &core));
    try std.testing.expectEqual(pc, core.pc);
    try std.testing.expectEqual(bits, core.fault_cause);
    try std.testing.expectEqual(dsp, core.dsp);
}

test "a data stack that leaves its range faults" {
    try expectFault(&.{ word(encode.drop), halt(1) }, 16, 0, lift.cause.stack, 0);
    var pushes: [9]u16 = @splat(lit(1));
    pushes[8] = halt(8);
    try expectFault(&pushes, 8, 7, lift.cause.stack, 7);
}

test "a return stack that leaves its range faults" {
    try expectFault(&.{ word(encode.exit), halt(1) }, 16, 0, lift.cause.stack, 0);
    try expectFault(&.{ call(0), halt(1) }, 8, 0, lift.cause.stack, 0);
}

test "a reserved word faults, with the stack bit when the stack leaves its range too" {
    try expectFault(&.{ 0x6010, halt(1) }, 16, 0, lift.cause.illegal, 0);
    try expectFault(&.{ 0x6013, halt(1) }, 16, 0, lift.cause.illegal | lift.cause.stack, 0);
}

test "an I/O read with a store in the same word faults, and a memory read does not" {
    const clash = word(.{ .alu = .{ .t_op = .fetch, .store = true } });
    try expectFault(&.{ lit(0xF), lit(12), word(encode.lshift), clash, halt(4) }, 16, 3, lift.cause.illegal, 1);

    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{ lit(0x7), lit(0x20), clash, halt(3) };
    memory.words[16] = 9;
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 16, &core));
    try std.testing.expectEqual(@as(u16, 7), memory.words[16]);
}

const Device = struct {
    wrote_at: u32 = 0,
    wrote: u32 = 0,
    reads: u32 = 0,

    fn answer(ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u32 {
        const self: *Device = @ptrCast(@alignCast(ctx.?));
        if (kind == lift.io.write) {
            self.wrote_at = at;
            self.wrote = value;
            return 0;
        }
        self.reads += 1;
        return 0x1200 + (at & 0xFF);
    }
};

test "addresses from 0xF000 up go to the device" {
    var memory: Memory = .{};
    var device: Device = .{};
    var core: lift.Core = .{ .io = Device.answer, .io_ctx = &device };
    memory.attach(&core);
    const program = [_]u16{
        lit(0x55),
        lit(0xF),
        lit(12),
        word(encode.lshift),
        word(encode.store),
        word(encode.drop),
        lit(0xF),
        lit(12),
        word(encode.lshift),
        lit(4),
        word(encode.plus),
        word(encode.fetch),
        lit(0x20),
        word(encode.store),
        word(encode.drop),
        halt(15),
    };
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 16, &core));
    try std.testing.expectEqual(@as(u32, 0xF000), device.wrote_at);
    try std.testing.expectEqual(@as(u32, 0x55), device.wrote);
    try std.testing.expectEqual(@as(u32, 1), device.reads);
    try std.testing.expectEqual(@as(u16, 0x1204), memory.words[16]);
}

fn push(list: []u16, at: *usize, value: u16) void {
    if (value < 0x8000) {
        list[at.*] = lit(@intCast(value));
        at.* += 1;
    } else {
        list[at.*] = lit(@intCast(~value & 0x7FFF));
        list[at.* + 1] = word(encode.invert);
        at.* += 2;
    }
}

/// Run `n op t` and answer the word it leaves in memory.
fn binary(op: encode.Insn, n: u16, t: u16) !u16 {
    var program: [10]u16 = undefined;
    var at: usize = 0;
    push(&program, &at, n);
    push(&program, &at, t);
    program[at] = word(op);
    program[at + 1] = lit(0);
    program[at + 2] = word(encode.store);
    program[at + 3] = halt(@intCast(at + 3));
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    try std.testing.expectEqual(lift.Exit.halted, try native(program[0 .. at + 4], 0, 16, &core));
    return memory.words[0];
}

fn expected(name: []const u8, n: u16, t: u16) u16 {
    const ns: i16 = @bitCast(n);
    const ts: i16 = @bitCast(t);
    const yes: u16 = 0xFFFF;
    if (std.mem.eql(u8, name, "plus")) return n +% t;
    if (std.mem.eql(u8, name, "and")) return n & t;
    if (std.mem.eql(u8, name, "or")) return n | t;
    if (std.mem.eql(u8, name, "xor")) return n ^ t;
    if (std.mem.eql(u8, name, "eq")) return if (n == t) yes else 0;
    if (std.mem.eql(u8, name, "lt")) return if (ns < ts) yes else 0;
    if (std.mem.eql(u8, name, "ult")) return if (n < t) yes else 0;
    if (std.mem.eql(u8, name, "rshift")) return if (t >= 16) 0 else n >> @intCast(t);
    if (std.mem.eql(u8, name, "lshift")) return if (t >= 16) 0 else n << @intCast(t);
    unreachable;
}

test "every two operand word, over values that tell signed from unsigned and big shifts apart" {
    const names = .{ "plus", "and", "or", "xor", "eq", "lt", "ult", "rshift", "lshift" };
    const values = [_]u16{ 0, 1, 3, 15, 16, 17, 0x7FFF, 0x8000, 0xFFFE, 0xFFFF };
    inline for (names) |name| {
        for (values) |n| {
            for (values) |t| {
                const got = try binary(@field(encode, name), n, t);
                const want = expected(name, n, t);
                if (got != want) {
                    std.debug.print("{s} {x} {x}: got {x}, want {x}\n", .{ name, n, t, got, want });
                    return error.TestExpectedEqual;
                }
            }
        }
    }
}

test "the one operand words and the stack shufflers" {
    // invert, decrement and depth, then swap and over on a known pair.
    var memory: Memory = .{};
    var core: lift.Core = .{};
    memory.attach(&core);
    const program = [_]u16{
        lit(1),
        lit(2),
        word(encode.over),
        word(encode.depth),
        lit(0),
        word(encode.store),
        word(encode.drop),
        word(encode.swap),
        word(encode.decrement),
        lit(2),
        word(encode.store),
        word(encode.drop),
        halt(12),
    };
    try std.testing.expectEqual(lift.Exit.halted, try native(&program, 0, 16, &core));
    // Depth reads 3 after the over, and the second store writes the decremented swap result.
    try std.testing.expectEqual(@as(u16, 3), memory.words[0]);
    try std.testing.expectEqual(@as(u16, 1), memory.words[1]);
}
