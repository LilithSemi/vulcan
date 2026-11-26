//! Execution tests for the Forth frontend: compile a small program, lower it to IR, JIT it
//! for the host and compare the data stack `main` leaves behind.

const std = @import("std");
const forth = @import("vulcan-forth");
const geyser_run = @import("geyser_run.zig");

const t = std.testing;

const true_flag: u16 = 0xFFFF;

fn load(source: []const u8) !forth.Machine {
    var diag: forth.Diagnostic = .{};
    var program = forth.compile(t.allocator, source, &diag) catch |err| {
        std.debug.print("compile error: {s}\n", .{diag.message()});
        return err;
    };
    errdefer program.deinit();
    return forth.Machine.init(t.allocator, program);
}

fn expectHostStack(source: []const u8, expected: []const u16) !void {
    var machine = try load(source);
    defer machine.deinit();
    const stack = try machine.runMain();
    defer t.allocator.free(stack);
    try t.expectEqualSlices(u16, expected, stack);
}

fn expectGeyserStack(source: []const u8, expected: []const u16) !void {
    var result = try geyser_run.run(t.allocator, source);
    defer result.deinit(t.allocator);
    try t.expectEqual(@import("vulcan-target").geyser.lift.Exit.halted, result.exit);
    try t.expectEqualSlices(u16, expected, result.stack);
}

/// The program runs on the host JIT and as Geyser code, and both leave the same stack.
fn expectStack(source: []const u8, expected: []const u16) !void {
    try expectHostStack(source, expected);
    try expectGeyserStack(source, expected);
}

fn expectError(source: []const u8, needle: []const u8) !void {
    var diag: forth.Diagnostic = .{};
    if (forth.compile(t.allocator, source, &diag)) |program| {
        var p = program;
        p.deinit();
        return error.ExpectedCompileError;
    } else |err| {
        try t.expectEqual(error.Compile, err);
    }
    if (std.mem.indexOf(u8, diag.message(), needle) == null) {
        std.debug.print("message was: {s}\n", .{diag.message()});
        return error.WrongMessage;
    }
}

test "stack words" {
    try expectStack(": main 1 2 3 ;", &.{ 1, 2, 3 });
    try expectStack(": main 1 2 dup ;", &.{ 1, 2, 2 });
    try expectStack(": main 1 2 drop ;", &.{1});
    try expectStack(": main 1 2 swap ;", &.{ 2, 1 });
    try expectStack(": main 1 2 over ;", &.{ 1, 2, 1 });
    try expectStack(": main 1 2 3 rot ;", &.{ 2, 3, 1 });
    try expectStack(": main 1 2 nip ;", &.{2});
    try expectStack(": main 1 2 tuck ;", &.{ 2, 1, 2 });
    try expectStack(": main 1 2 2dup ;", &.{ 1, 2, 1, 2 });
    try expectStack(": main 1 2 3 2drop ;", &.{1});
}

test "return stack words" {
    try expectStack(": main 1 2 >r 3 r> ;", &.{ 1, 3, 2 });
    try expectStack(": main 5 >r r@ r@ + r> + ;", &.{15});
}

test "arithmetic wraps at 16 bits" {
    try expectStack(": main 7 5 + ;", &.{12});
    try expectStack(": main $ffff 1 + ;", &.{0});
    try expectStack(": main 7 5 - ;", &.{2});
    try expectStack(": main 5 7 - ;", &.{0xFFFE});
    try expectStack(": main 300 300 * ;", &.{24464});
    try expectStack(": main $f0f0 $ff00 and $f0f0 $ff00 or $f0f0 $ff00 xor ;", &.{ 0xF000, 0xFFF0, 0x0FF0 });
    try expectStack(": main $00ff invert ;", &.{0xFF00});
    try expectStack(": main 3 negate 0 negate ;", &.{ 0xFFFD, 0 });
    try expectStack(": main -1 -32768 ;", &.{ 0xFFFF, 0x8000 });
}

test "shifts give zero from 16 up" {
    try expectStack(": main 1 4 lshift 1 15 lshift 1 16 lshift 1 17 lshift ;", &.{ 16, 0x8000, 0, 0 });
    try expectStack(": main $8000 15 rshift $8000 16 rshift $ffff 4 rshift 5 0 rshift ;", &.{ 1, 0, 0x0FFF, 5 });
    try expectStack(": main 1 32 lshift 1 48 rshift ;", &.{ 0, 0 });
}

test "comparisons give a full-width flag" {
    try expectStack(": main 3 3 = 3 4 = ;", &.{ true_flag, 0 });
    try expectStack(": main 3 4 <> 3 3 <> ;", &.{ true_flag, 0 });
    try expectStack(": main 1 2 < 2 1 < 1 1 < ;", &.{ true_flag, 0, 0 });
    try expectStack(": main 2 1 > 1 2 > ;", &.{ true_flag, 0 });
    try expectStack(": main -1 1 < -1 1 > ;", &.{ true_flag, 0 });
    try expectStack(": main -1 1 u< 1 -1 u< ;", &.{ 0, true_flag });
    try expectStack(": main 0 0= 5 0= ;", &.{ true_flag, 0 });
    try expectStack(": main -1 0< 1 0< 0 0< ;", &.{ true_flag, 0, 0 });
}

test "if else then" {
    try expectStack(": main 1 if 10 else 20 then ;", &.{10});
    try expectStack(": main 0 if 10 else 20 then ;", &.{20});
    try expectStack(": main 5 0 if 1 + then ;", &.{5});
    try expectStack(": main 5 1 if 1 + then ;", &.{6});
    try expectStack(": pick 0< if 1 else 2 then ; : main -4 pick 4 pick ;", &.{ 1, 2 });
    try expectStack(": main 1 if 0 if 1 else 2 then else 3 then ;", &.{2});
}

test "begin loops" {
    try expectStack(": main 0 begin 1 + dup 5 = until ;", &.{5});
    try expectStack(": main 0 begin dup 10 < while 3 + repeat ;", &.{12});
    try expectStack(": main 0 begin 1 + dup 7 = if exit then again ;", &.{7});
}

test "do loops" {
    try expectStack(": main 0 10 0 do i + loop ;", &.{45});
    try expectStack(": main 0 10 0 do i + 2 +loop ;", &.{20});
    try expectStack(": main 0 0 10 do i + -3 +loop ;", &.{22});
    try expectStack(": main 0 3 0 do 3 0 do j 10 * i + + loop loop ;", &.{@as(u16, 0 + 1 + 2 + 10 + 11 + 12 + 20 + 21 + 22)});
    try expectStack(": main 0 100 0 do i 5 = if leave then 1 + loop ;", &.{5});
    try expectStack(": find 10 0 do i 4 = if i unloop exit then loop 99 ; : main find ;", &.{4});
    try expectStack(": main 0 0 0 do 1 + loop ;", &.{0});
    try expectStack(": main 3 0 do i drop loop 7 ;", &.{7});
}

test "variables and memory" {
    try expectStack("variable v : main 7 v ! v @ ;", &.{7});
    try expectStack("variable v : main 7 v ! 5 v +! v @ ;", &.{12});
    try expectStack("variable a variable b : main 1 a ! 2 b ! a @ b @ ;", &.{ 1, 2 });
    try expectStack("variable a variable b : main a b - ;", &.{0xFFFE});
    try expectStack("create tab 10 , 20 , 30 , : main tab @ tab 2 + @ tab 4 + @ ;", &.{ 10, 20, 30 });
    try expectStack("create buf 8 allot : main $1234 buf 6 + ! buf 6 + @ buf @ ;", &.{ 0x1234, 0 });
    try expectStack("5 constant five : main five five + ;", &.{10});
    try expectStack("3 constant a a 4 + constant b : main b ;", &.{7});
    try expectStack(": main [ 2 3 + 4 * ] literal 1 + ;", &.{21});
    try expectStack("$100 constant base : main [ base 2 + ] literal ;", &.{0x102});
}

test "words call words" {
    try expectStack(": sq dup * ; : main 7 sq ;", &.{49});
    try expectStack(": two 1 2 ; : main two + ;", &.{3});
    try expectStack(": three 1 2 3 ; : main three ;", &.{ 1, 2, 3 });
    try expectStack(": ab swap ; : main 1 2 ab ab ;", &.{ 1, 2 });
    try expectStack(": nothing ; : main 4 nothing ;", &.{4});
    try expectStack(": dropper drop drop ; : main 1 2 3 dropper ;", &.{1});
    try expectStack(": five 1 2 3 4 5 ; : main five + + + + ;", &.{15});
    try expectStack(": sum4 + + + ; : main 1 2 3 4 sum4 ;", &.{10});
    try expectStack(": f 1 ; : f f 1 + ; : main f ;", &.{2});
}

test "recursion" {
    try expectStack(": fact dup 1 > if dup 1 - recurse * then ; : main 5 fact ;", &.{120});
    try expectStack(": fib dup 2 < if exit then dup 1 - recurse swap 2 - recurse + ; : main 10 fib ;", &.{55});
    try expectHostStack(": count dup if 1 - recurse then ; : main 100 count ;", &.{0});
    try expectStack(": count dup if 1 - recurse then ; : main 12 count ;", &.{0});
}

test "last definition named main runs" {
    try expectStack(": main 1 ; : main 2 ;", &.{2});
}

test "io window records writes and reads zero" {
    var machine = try load("$f000 constant out : main 42 out ! 7 $ff00 ! $f002 @ ;");
    defer machine.deinit();
    const stack = try machine.runMain();
    defer t.allocator.free(stack);
    try t.expectEqualSlices(u16, &.{0}, stack);
    try t.expectEqual(@as(usize, 2), machine.io_writes.items.len);
    try t.expectEqual(@as(u16, 0xF000), machine.io_writes.items[0].addr);
    try t.expectEqual(@as(u16, 42), machine.io_writes.items[0].value);
    try t.expectEqual(@as(u16, 0xFF00), machine.io_writes.items[1].addr);
    try t.expectEqual(@as(u16, 7), machine.io_writes.items[1].value);
}

test "io writes never reach ram" {
    var machine = try load("variable v : main 9 v ! 5 $f000 ! v @ ;");
    defer machine.deinit();
    const stack = try machine.runMain();
    defer t.allocator.free(stack);
    try t.expectEqualSlices(u16, &.{9}, stack);
}

test "stack effects are inferred" {
    var diag: forth.Diagnostic = .{};
    var program = try forth.compile(t.allocator, ": a drop ; : b 1 2 3 ; : c + dup ; : main ;", &diag);
    defer program.deinit();
    try t.expectEqual(forth.Effect{ .in = 1, .out = 0 }, program.words[0].effect);
    try t.expectEqual(forth.Effect{ .in = 0, .out = 3 }, program.words[1].effect);
    try t.expectEqual(forth.Effect{ .in = 2, .out = 2 }, program.words[2].effect);
}

test "unbalanced if is a compile error naming the word" {
    try expectError(": f if 1 then ;", "'f'");
    try expectError(": f if 1 then ;", "if");
    try expectError(": g 0 if 1 else 1 2 then ;", "'g'");
}

test "unknown word is a compile error" {
    try expectError(": f frobnicate ;", "frobnicate");
    try expectError(": f 1 ; : main g ;", "'g'");
}

test "unbalanced return stack is a compile error" {
    try expectError(": f 1 >r ;", ">r");
    try expectError(": f r> ;", "r>");
    try expectError(": f 1 >r if r> then ;", "'f'");
}

test "unbalanced loops are compile errors" {
    try expectError(": f begin 1 2 until ;", "'f'");
    try expectError(": f 3 0 do 1 loop ;", "'f'");
    try expectError(": f begin dup while drop drop repeat ;", "'f'");
    try expectError(": f 3 0 do i 1 = if 5 leave then loop ;", "leave");
}

test "structure errors name the word" {
    try expectError(": f then ;", "then");
    try expectError(": f 1 if 2 ;", "'f'");
    try expectError(": f 1 begin ;", "'f'");
    try expectError(": f ( never ends", "comment");
    try expectError(": f 1", "'f'");
}

test "recursion with no consistent effect is an error" {
    try expectError(": f dup if 1 recurse then ;", "'f'");
}

test "main must exist to run" {
    var machine = try load(": f 1 ;");
    defer machine.deinit();
    try t.expectError(error.NoMain, machine.runMain());
}

test "exit inside a do loop needs unloop" {
    try expectError(": f 3 0 do exit loop ;", "unloop");
}

test "literal range is checked" {
    try expectError(": f 70000 ;", "70000");
    try expectError(": f $zz ;", "$zz");
}

test "words that take many cells" {
    try expectStack(": sum10 + + + + + + + + + ; : main 1 2 3 4 5 6 7 8 9 10 sum10 ;", &.{55});
    try expectStack(": ten 1 2 3 4 5 6 7 8 9 10 ; : main ten ;", &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 });
    try expectStack(": rev3 swap rot ; : main 1 2 3 rev3 ;", &.{ 3, 2, 1 });
}

test "state flows through nested control flow" {
    try expectStack(": main 0 5 0 do i 2 > if i + else 100 + then loop ;", &.{307});
    try expectStack(": main 1 2 >r 5 0 do i r@ + + loop r> ;", &.{ 21, 2 });
}

test "loops inside words called from loops" {
    try expectStack(": tri 0 swap 0 do i + loop ; : main 0 4 1 do i tri + loop ;", &.{4});
    try expectError(": two 7 8 ; : main 3 0 do two loop ;", "do ... loop");
}

test "begin loops carry several cells" {
    try expectStack(": main 0 1 begin over 10 < while swap 1 + swap 2 * repeat ;", &.{ 10, 1024 });
}

test "small programs fit in the image limit" {
    const sources = [_][]const u8{
        ": main 1 2 + ;",
        ": sq dup * ; : main 7 sq ;",
        ": fib dup 2 < if exit then dup 1 - recurse swap 2 - recurse + ; : main 10 fib ;",
        ": main 0 10 0 do i + loop ;",
    };
    for (sources) |source| {
        var result = try geyser_run.run(t.allocator, source);
        defer result.deinit(t.allocator);
        try t.expect(result.image_words + result.slots < 4096);
    }
}

// Values read from memory cannot be folded, so these reach the code for every operation.
test "operations on values that are not constants" {
    const pre = "variable a variable b ";
    try expectStack(pre ++ ": main 7 a ! 5 b ! a @ b @ - b @ a @ - ;", &.{ 2, 0xFFFE });
    try expectStack(pre ++ ": main 300 a ! 300 b ! a @ b @ * 7 a ! 0 b ! a @ b @ * ;", &.{ 24464, 0 });
    try expectStack(pre ++ ": main 3 a ! 20 b ! 1 a @ lshift 1 b @ lshift $8000 a @ rshift $8000 b @ rshift ;", &.{ 8, 0, 0x1000, 0 });
    try expectStack(pre ++ ": main -1 a ! 1 b ! a @ b @ < a @ b @ > a @ b @ u< b @ a @ u< a @ b @ = a @ b @ <> ;", &.{ 0xFFFF, 0, 0, 0xFFFF, 0, 0xFFFF });
    try expectStack(pre ++ ": main -5 a ! a @ 0< a @ 0= 0 b ! b @ 0= b @ negate a @ negate ;", &.{ 0xFFFF, 0, 0xFFFF, 0, 5 });
    try expectStack(pre ++ ": main 5 a ! 0 a @ 0 do i + loop -3 b ! 0 0 a @ do i + b @ +loop ;", &.{ 10, 7 });
    try expectStack(pre ++ ": main $f0f0 a ! $ff00 b ! a @ b @ and a @ b @ or a @ b @ xor a @ invert ;", &.{ 0xF000, 0xFFF0, 0x0FF0, 0x0F0F });
    try expectStack(pre ++ ": pair a @ b @ ; : main 4 a ! 9 b ! pair pair + + + ;", &.{26});
    try expectStack(pre ++ ": f dup 0= if drop 0 else dup 1 - recurse + then ; : main 6 a ! a @ f ;", &.{21});
    try expectStack(pre ++ ": main 1 a ! 2 b ! a @ b @ over over swap 2dup rot ;", &.{ 1, 2, 2, 2, 1, 1 });
}

fn expectGeyserTrap(source: []const u8) !void {
    var result = try geyser_run.run(t.allocator, source);
    defer result.deinit(t.allocator);
    try t.expectEqual(@import("vulcan-target").geyser.lift.Exit.faulted, result.exit);
}

fn expectGeyserCompileError(source: []const u8, err: anyerror, needle: []const u8) !void {
    var diag: forth.Diagnostic = .{};
    var program = try forth.compile(t.allocator, source, &diag);
    defer program.deinit();
    var geyser_diag: forth.geyser.Diagnostic = .{};
    try t.expectError(err, forth.compileGeyser(t.allocator, &program, .{ .diag = &geyser_diag }));
    if (std.mem.indexOf(u8, geyser_diag.message(), needle) == null) {
        std.debug.print("message was: {s}\n", .{geyser_diag.message()});
        return error.WrongMessage;
    }
}

test "recursion deeper than the return stack holds traps instead of overflowing" {
    // The entry call, the call of count and the frame counter take 3 of the 15 cells.
    try expectGeyserTrap(": count dup if 1 - recurse then ; : main 13 count ;");
    try expectStack(": fib dup 2 < if exit then dup 1 - recurse swap 2 - recurse + ; : main 12 fib ;", &.{144});
}

test "the save stack traps when it is full" {
    var diag: forth.Diagnostic = .{};
    var program = try forth.compile(t.allocator, ": f dup if dup 1 - recurse + then ; : main 12 f ;", &diag);
    defer program.deinit();
    const image = try forth.compileGeyser(t.allocator, &program, .{});
    defer t.allocator.free(image.words);
    const words = try t.allocator.dupe(u16, image.words);
    defer t.allocator.free(words);
    // Make the limit literal in SAVE too small: find `lit limit` as the word before `ult` in it.
    var patched = false;
    for (words[1..], 1..) |w, i| {
        if (w == 0x6F03 and i >= 2 and words[i - 1] & 0x8000 != 0 and words[i - 2] == 0x6081 and words[i - 3] == 0x6C00) {
            words[i - 1] = 0x8000 | (words[i - 1] & 0x7FFF) - 2;
            patched = true;
            break;
        }
    }
    try t.expect(patched);
    var result = try geyser_run.execute(t.allocator, words, image.slots);
    defer result.deinit(t.allocator);
    try t.expectEqual(@import("vulcan-target").geyser.lift.Exit.faulted, result.exit);
}

test "a call chain deeper than the return stack is a compile error" {
    var source: std.ArrayList(u8) = .empty;
    defer source.deinit(t.allocator);
    try source.appendSlice(t.allocator, ": w0 1 ; ");
    for (1..16) |k| try source.print(t.allocator, ": w{d} w{d} ; ", .{ k, k - 1 });
    try source.appendSlice(t.allocator, ": main w15 ;");
    try expectGeyserCompileError(source.items, error.StackTooDeep, "return stack");
}

test "a word with more values than the data stack holds is a compile error" {
    try expectGeyserCompileError(": f 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ; : main f ;", error.StackTooDeep, "data stack");
}

test "an image that does not fit in memory is a compile error naming the sizes" {
    try expectGeyserCompileError("create big 9000 allot : main 1 ;", error.ImageTooLarge, "memory has 4096");
}

test "begin again with exit returns from the word" {
    try expectStack(": f 0 begin 1 + dup 3 = if exit then again ; : main f f + ;", &.{6});
}

test "a word that never returns compiles" {
    const source = ": main 1 drop begin 5 drop again ;";
    var diag: forth.Diagnostic = .{};
    var program = forth.compile(t.allocator, source, &diag) catch |err| {
        std.debug.print("compile error: {s}\n", .{diag.message()});
        return err;
    };
    try t.expectEqual(@as(u8, 0), program.words[0].effect.in);
    try t.expectEqual(@as(u8, 0), program.words[0].effect.out);
    const image = try forth.compileGeyser(t.allocator, &program, .{});
    t.allocator.free(image.words);
    var machine = try forth.Machine.init(t.allocator, program);
    machine.deinit();
}

test "errors name the source and the line inside it" {
    var diag: forth.Diagnostic = .{};
    const prelude = ": ok 1 ;\n: two 2 ;\n";
    const user = "\n\n: bad frobnicate ;\n";
    try t.expectError(error.Compile, forth.compileSources(t.allocator, &.{
        .{ .name = "prelude.fs", .text = prelude },
        .{ .name = "user.fs", .text = user },
    }, &diag));
    try t.expect(std.mem.startsWith(u8, diag.message(), "user.fs: "));
    try t.expect(std.mem.indexOf(u8, diag.message(), "(line 3)") != null);

    try t.expectError(error.Compile, forth.compileSources(t.allocator, &.{
        .{ .name = "prelude.fs", .text = ": f if 1 then ;\n" },
        .{ .name = "user.fs", .text = ": main 1 ;\n" },
    }, &diag));
    try t.expect(std.mem.startsWith(u8, diag.message(), "prelude.fs: "));
    try t.expect(std.mem.indexOf(u8, diag.message(), "(line 1)") != null);

    var program = try forth.compileSources(t.allocator, &.{
        .{ .name = "prelude.fs", .text = prelude },
        .{ .name = "user.fs", .text = ": main ok two + ;" },
    }, &diag);
    program.deinit();
}
