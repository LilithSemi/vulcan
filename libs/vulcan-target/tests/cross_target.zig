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

fn buildNvFp4(allocator: std.mem.Allocator, dequantize: bool) !Function {
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
        .block_application = .multiply,
        .global_application = .divide,
    };
    const result = try func.appendInst(block, if (dequantize) f32_t else u8_t, if (dequantize) .{ .dequantize_nvfp4 = conversion } else .{ .quantize_nvfp4 = conversion });
    try func.addAttr(.{ .inst = func.definingInst(result).? }, .{ .custom = .{ .namespace = "debug", .key = "nvfp4", .value = .{ .int = 42 } } });
    try func.addAttr(.{ .value = result }, .{ .custom = .{ .namespace = "test", .key = "live", .value = .flag } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(result) });
    return func;
}

test "all scalar machine targets reject both nvfp4 directions without mutating caller IR" {
    const allocator = std.testing.allocator;
    inline for (.{ true, false }) |dequantize| {
        var func = try buildNvFp4(allocator, dequantize);
        defer func.deinit();
        const before = try ir.bitcode.encode(allocator, &func);
        defer allocator.free(before);
        inline for (.{ .x86, .x86_64, .aarch64, .riscv64 }) |arch| {
            try std.testing.expectError(error.Unsupported, target.native.writeObjectDataFor(allocator, arch, &.{.{ .name = "convert", .func = &func }}, &.{}));
            const after = try ir.bitcode.encode(allocator, &func);
            defer allocator.free(after);
            try std.testing.expectEqualSlices(u8, before, after);
        }
    }
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
