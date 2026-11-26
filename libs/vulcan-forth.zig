//! Forth frontend: read Forth source with 16 bit cells, infer the stack effect of every
//! word and lower each word to a Vulcan IR function. `engine.zig` JITs the result for the
//! host so programs can run as tests.

const std = @import("std");
const vt = @import("vulcan-target");

pub const ast = @import("vulcan-forth/ast.zig");
pub const lex = @import("vulcan-forth/lex.zig");
pub const prims = @import("vulcan-forth/prims.zig");
pub const parse = @import("vulcan-forth/parse.zig");
pub const effect = @import("vulcan-forth/effect.zig");
pub const lower = @import("vulcan-forth/lower.zig");
pub const engine = @import("vulcan-forth/engine.zig");

pub const Program = ast.Program;
pub const Word = ast.Word;
pub const Effect = ast.Effect;
pub const Diagnostic = ast.Diagnostic;
pub const Machine = engine.Machine;
pub const IoWrite = engine.IoWrite;

pub const Error = error{ Compile, OutOfMemory };

pub const Source = parse.Source;

/// Compile `source`. On `error.Compile` the diagnostic holds the reason.
pub fn compile(allocator: std.mem.Allocator, source: []const u8, diag: *Diagnostic) Error!Program {
    return compileSources(allocator, &.{.{ .name = "", .text = source }}, diag);
}

/// Compile `sources` in order as one program, for example a prelude and then a user file.
/// Diagnostics name the source and the line inside it.
pub fn compileSources(allocator: std.mem.Allocator, sources: []const Source, diag: *Diagnostic) Error!Program {
    defer diag.file = "";
    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const scratch = arena.allocator();

    const parsed = try parse.parseSources(scratch, sources, diag);
    const words = parsed.words;
    for (words, 0..) |*word, i| word.effect = try effect.infer(words[0..i], word, diag);

    var main: ?u32 = null;
    for (words, 0..) |word, i| {
        if (std.ascii.eqlIgnoreCase(word.name, "main")) main = @intCast(i);
    }
    if (main) |m| {
        diag.file = words[m].file;
        if (words[m].effect.in != 0) return diag.fail("word 'main' takes {d} cells, it must take none", .{words[m].effect.in});
    }

    const functions = try lower.lowerAll(allocator, scratch, words);
    return .{ .arena = arena, .words = words, .data = parsed.data, .main = main, .functions = functions };
}

/// Select Geyser instructions for the program. The caller frees `words` of the image.
pub fn compileGeyser(allocator: std.mem.Allocator, program: *const Program, options: vt.geyser.isel.Options) vt.geyser.isel.Error!vt.geyser.isel.Image {
    const main = program.main orelse return error.NoMain;
    const items = try allocator.alloc(vt.geyser.isel.Item, program.functions.len);
    defer allocator.free(items);
    for (items, program.functions) |*item, *lowered| item.* = .{ .name = lowered.name, .func = &lowered.func };
    return vt.geyser.isel.compile(allocator, items, program.data, main, options);
}

/// Write an image as little endian 16 bit words.
pub const geyserWriteImage = vt.geyser.encode.writeImage;
pub const geyser = vt.geyser.isel;

test {
    std.testing.refAllDecls(@This());
    std.testing.refAllDecls(lex);
    std.testing.refAllDecls(prims);
    std.testing.refAllDecls(parse);
}
