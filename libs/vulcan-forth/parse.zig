//! Forth parser. Reads colon definitions into node trees and runs the defining words
//! (`constant`, `variable`, `create`, `allot`, `,`) as the source is read, so constants
//! and addresses are known by the time a word uses them.

const std = @import("std");
const ast = @import("ast.zig");
const lex = @import("lex.zig");
const prims = @import("prims.zig");

const Node = ast.Node;
const Word = ast.Word;
const Diagnostic = ast.Diagnostic;

pub const Error = error{ Compile, OutOfMemory };

pub const Parsed = struct {
    words: []Word,
    data: []const u8,
};

const Entry = struct {
    name: []const u8,
    kind: union(enum) { word: u32, constant: u16 },
};

const Closers = []const []const u8;

const Seq = struct { nodes: []const Node, closer: usize };

const Parser = struct {
    arena: std.mem.Allocator,
    lexer: lex.Lexer,
    diag: *Diagnostic,
    dict: std.ArrayList(Entry) = .empty,
    words: std.ArrayList(Word) = .empty,
    data: std.ArrayList(u8) = .empty,
    ct: prims.CtStack = .{},
    word_name: []const u8 = "",
    saw_recurse: bool = false,
    line: u32 = 0,
    file: []const u8 = "",

    fn next(self: *Parser) Error!?lex.Token {
        const tok = self.lexer.next() catch return self.diag.fail("unterminated comment: missing ')'", .{});
        if (tok) |t| self.line = t.line;
        return tok;
    }

    fn lookup(self: *const Parser, text: []const u8) ?Entry {
        var i = self.dict.items.len;
        while (i > 0) {
            i -= 1;
            if (std.ascii.eqlIgnoreCase(self.dict.items[i].name, text)) return self.dict.items[i];
        }
        return null;
    }

    fn define(self: *Parser, name: []const u8, kind: @FieldType(Entry, "kind")) Error!void {
        try self.dict.append(self.arena, .{ .name = try self.arena.dupe(u8, name), .kind = kind });
    }

    const Number = union(enum) { none, ok: u16, range };

    fn number(text: []const u8) Number {
        var digits = text;
        var negative = false;
        if (digits.len > 0 and digits[0] == '-') {
            negative = true;
            digits = digits[1..];
        }
        var radix: u8 = 10;
        if (digits.len > 0 and digits[0] == '$') {
            radix = 16;
            digits = digits[1..];
        }
        if (digits.len == 0 or digits[0] == '+' or digits[0] == '-') return .none;
        const magnitude = std.fmt.parseInt(u32, digits, radix) catch |err| switch (err) {
            error.InvalidCharacter => return .none,
            error.Overflow => return .range,
        };
        if (negative) {
            if (magnitude > 0x8000) return .range;
            return .{ .ok = @truncate(0 -% magnitude) };
        }
        if (magnitude > 0xFFFF) return .range;
        return .{ .ok = @intCast(magnitude) };
    }

    fn isCloser(text: []const u8, closers: Closers) ?usize {
        for (closers, 0..) |c, idx| if (std.ascii.eqlIgnoreCase(c, text)) return idx;
        return null;
    }

    fn isKeyword(text: []const u8) bool {
        const keywords = [_][]const u8{ ":", ";", "if", "else", "then", "begin", "until", "again", "while", "repeat", "do", "loop", "+loop", "[", "]", "literal", "variable", "create", "constant", "allot", "," };
        for (keywords) |k| if (std.ascii.eqlIgnoreCase(k, text)) return true;
        return false;
    }

    /// Read nodes until one of `closers`. `open` names the structure for messages.
    fn sequence(self: *Parser, closers: Closers, open: []const u8) Error!Seq {
        var nodes: std.ArrayList(Node) = .empty;
        while (try self.next()) |tok| {
            const text = tok.text;
            if (isCloser(text, closers)) |idx| {
                return .{ .nodes = try nodes.toOwnedSlice(self.arena), .closer = idx };
            }
            if (std.ascii.eqlIgnoreCase(text, ";")) {
                return self.diag.fail("word '{s}': '{s}' is not closed before ';' (line {d})", .{ self.word_name, open, tok.line });
            }
            if (std.ascii.eqlIgnoreCase(text, "if")) {
                const then_seq = try self.sequence(&.{ "else", "then" }, "if");
                var else_nodes: []const Node = &.{};
                if (then_seq.closer == 0) else_nodes = (try self.sequence(&.{"then"}, "else")).nodes;
                try nodes.append(self.arena, .{ .@"if" = .{ .then = then_seq.nodes, .@"else" = else_nodes, .has_else = then_seq.closer == 0 } });
            } else if (std.ascii.eqlIgnoreCase(text, "begin")) {
                const first = try self.sequence(&.{ "until", "again", "while" }, "begin");
                switch (first.closer) {
                    0 => try nodes.append(self.arena, .{ .until = first.nodes }),
                    1 => try nodes.append(self.arena, .{ .again = first.nodes }),
                    else => {
                        const body = try self.sequence(&.{"repeat"}, "while");
                        try nodes.append(self.arena, .{ .@"while" = .{ .cond = first.nodes, .body = body.nodes } });
                    },
                }
            } else if (std.ascii.eqlIgnoreCase(text, "do")) {
                const body = try self.sequence(&.{ "loop", "+loop" }, "do");
                try nodes.append(self.arena, .{ .do = .{ .body = body.nodes, .plus = body.closer == 1 } });
            } else if (std.ascii.eqlIgnoreCase(text, "[")) {
                try self.compileTime();
            } else if (std.ascii.eqlIgnoreCase(text, "literal")) {
                const v = self.ct.pop() catch return self.diag.fail("word '{s}': 'literal' needs a value from '[ ]' (line {d})", .{ self.word_name, tok.line });
                try nodes.append(self.arena, .{ .lit = v });
            } else if (isKeyword(text) or isStrayCloser(text)) {
                return self.diag.fail("word '{s}': unexpected '{s}' (line {d})", .{ self.word_name, text, tok.line });
            } else {
                try nodes.append(self.arena, try self.simple(tok));
            }
        }
        if (std.mem.eql(u8, open, "definition")) {
            return self.diag.fail("word '{s}': missing ';'", .{self.word_name});
        }
        return self.diag.fail("word '{s}': '{s}' is not closed", .{ self.word_name, open });
    }

    fn isStrayCloser(text: []const u8) bool {
        const closers = [_][]const u8{ "else", "then", "until", "again", "while", "repeat", "loop", "+loop" };
        for (closers) |c| if (std.ascii.eqlIgnoreCase(c, text)) return true;
        return false;
    }

    /// A token that is a literal, a call or a primitive.
    fn simple(self: *Parser, tok: lex.Token) Error!Node {
        const text = tok.text;
        if (self.lookup(text)) |entry| return switch (entry.kind) {
            .word => |idx| .{ .call = idx },
            .constant => |v| .{ .lit = v },
        };
        switch (number(text)) {
            .ok => |v| return .{ .lit = v },
            .range => return self.diag.fail("word '{s}': number '{s}' does not fit in 16 bits (line {d})", .{ self.word_name, text, tok.line }),
            .none => {},
        }
        if (prims.lookup(text)) |prim| {
            if (prim == .recurse) self.saw_recurse = true;
            return .{ .prim = prim };
        }
        return self.diag.fail("word '{s}': unknown word '{s}' (line {d})", .{ self.word_name, text, tok.line });
    }

    /// Run words at compile time until `]`.
    fn compileTime(self: *Parser) Error!void {
        while (try self.next()) |tok| {
            if (std.mem.eql(u8, tok.text, "]")) return;
            try self.evaluate(tok);
        }
        return self.diag.fail("word '{s}': '[' is not closed by ']'", .{self.word_name});
    }

    /// Run one token on the compile-time stack.
    fn evaluate(self: *Parser, tok: lex.Token) Error!void {
        const text = tok.text;
        const where = if (self.word_name.len == 0) "top level" else self.word_name;
        if (self.lookup(text)) |entry| switch (entry.kind) {
            .constant => |v| return self.ct.push(v) catch self.diag.fail("{s}: compile-time stack is full (line {d})", .{ where, tok.line }),
            .word => return self.diag.fail("{s}: '{s}' cannot run at compile time (line {d})", .{ where, text, tok.line }),
        };
        switch (number(text)) {
            .ok => |v| return self.ct.push(v) catch self.diag.fail("{s}: compile-time stack is full (line {d})", .{ where, tok.line }),
            .range => return self.diag.fail("{s}: number '{s}' does not fit in 16 bits (line {d})", .{ where, text, tok.line }),
            .none => {},
        }
        if (prims.lookup(text)) |prim| {
            self.ct.apply(prim) catch |err| switch (err) {
                error.NotConstant => return self.diag.fail("{s}: '{s}' cannot run at compile time (line {d})", .{ where, text, tok.line }),
                error.Underflow => return self.diag.fail("{s}: '{s}' finds too few values at compile time (line {d})", .{ where, text, tok.line }),
                error.Overflow => return self.diag.fail("{s}: compile-time stack is full (line {d})", .{ where, tok.line }),
            };
            return;
        }
        return self.diag.fail("{s}: unknown word '{s}' (line {d})", .{ where, text, tok.line });
    }

    fn here(self: *const Parser) usize {
        return self.data.items.len;
    }

    fn reserve(self: *Parser, bytes: usize, line: u32) Error!void {
        if (self.here() + bytes > ast.io_base) {
            return self.diag.fail("data area is full: it must stay below 0x{X} (line {d})", .{ ast.io_base, line });
        }
        try self.data.appendNTimes(self.arena, 0, bytes);
    }

    fn alignHere(self: *Parser, line: u32) Error!void {
        if (self.here() % 2 != 0) try self.reserve(1, line);
    }

    fn nameToken(self: *Parser, after: []const u8) Error!lex.Token {
        const tok = (try self.next()) orelse return self.diag.fail("'{s}' needs a name", .{after});
        if (isKeyword(tok.text) or isStrayCloser(tok.text)) return self.diag.fail("'{s}' cannot be used as a name (line {d})", .{ tok.text, tok.line });
        return tok;
    }

    fn popCt(self: *Parser, word: []const u8, line: u32) Error!u16 {
        return self.ct.pop() catch self.diag.fail("'{s}' needs a value on the compile-time stack (line {d})", .{ word, line });
    }

    fn colon(self: *Parser) Error!void {
        const name_tok = try self.nameToken(":");
        self.word_name = try self.arena.dupe(u8, name_tok.text);
        self.saw_recurse = false;
        const body = try self.sequence(&.{";"}, "definition");
        const index: u32 = @intCast(self.words.items.len);
        try self.words.append(self.arena, .{ .name = self.word_name, .body = body.nodes, .has_recurse = self.saw_recurse, .line = name_tok.line, .file = self.file });
        try self.define(self.word_name, .{ .word = index });
        self.word_name = "";
    }

    fn topLevel(self: *Parser) Error!void {
        while (try self.next()) |tok| {
            const text = tok.text;
            if (std.ascii.eqlIgnoreCase(text, ":")) {
                try self.colon();
            } else if (std.ascii.eqlIgnoreCase(text, "variable")) {
                const name_tok = try self.nameToken("variable");
                try self.alignHere(tok.line);
                const address: u16 = @intCast(self.here());
                try self.reserve(2, tok.line);
                try self.define(name_tok.text, .{ .constant = address });
            } else if (std.ascii.eqlIgnoreCase(text, "create")) {
                const name_tok = try self.nameToken("create");
                try self.alignHere(tok.line);
                try self.define(name_tok.text, .{ .constant = @intCast(self.here()) });
            } else if (std.ascii.eqlIgnoreCase(text, "constant")) {
                const name_tok = try self.nameToken("constant");
                const v = try self.popCt("constant", tok.line);
                try self.define(name_tok.text, .{ .constant = v });
            } else if (std.ascii.eqlIgnoreCase(text, "allot")) {
                try self.reserve(try self.popCt("allot", tok.line), tok.line);
            } else if (std.mem.eql(u8, text, ",")) {
                const v = try self.popCt(",", tok.line);
                try self.alignHere(tok.line);
                try self.reserve(2, tok.line);
                std.mem.writeInt(u16, self.data.items[self.here() - 2 ..][0..2], v, .little);
            } else if (isKeyword(text) or isStrayCloser(text)) {
                return self.diag.fail("'{s}' is only allowed inside a definition (line {d})", .{ text, tok.line });
            } else {
                try self.evaluate(tok);
            }
        }
    }
};

pub const Source = struct { name: []const u8, text: []const u8 };

/// Parse `sources` in order as one program. A definition cannot continue into the next
/// source. All memory comes from `arena`.
pub fn parseSources(arena: std.mem.Allocator, sources: []const Source, diag: *Diagnostic) Error!Parsed {
    var parser: Parser = .{ .arena = arena, .lexer = lex.Lexer.init(""), .diag = diag };
    for (sources) |source| {
        parser.file = try arena.dupe(u8, source.name);
        parser.lexer = lex.Lexer.init(source.text);
        diag.file = parser.file;
        try parser.topLevel();
    }
    return .{ .words = parser.words.items, .data = parser.data.items };
}

/// Parse `source`. All memory comes from `arena`.
pub fn parse(arena: std.mem.Allocator, source: []const u8, diag: *Diagnostic) Error!Parsed {
    return parseSources(arena, &.{.{ .name = "", .text = source }}, diag);
}

test "numbers" {
    const N = Parser.Number;
    try std.testing.expectEqual(N{ .ok = 42 }, Parser.number("42"));
    try std.testing.expectEqual(N{ .ok = 0xFFFF }, Parser.number("-1"));
    try std.testing.expectEqual(N{ .ok = 0x8000 }, Parser.number("-32768"));
    try std.testing.expectEqual(N{ .ok = 0xBEEF }, Parser.number("$beef"));
    try std.testing.expectEqual(N{ .ok = 0xFFFF }, Parser.number("-$1"));
    try std.testing.expectEqual(N.range, Parser.number("65536"));
    try std.testing.expectEqual(N.range, Parser.number("-32769"));
    try std.testing.expectEqual(N.none, Parser.number("-"));
    try std.testing.expectEqual(N.none, Parser.number("+"));
    try std.testing.expectEqual(N.none, Parser.number("+!"));
    try std.testing.expectEqual(N.none, Parser.number("$"));
    try std.testing.expectEqual(N.none, Parser.number("1+"));
}

test "defining words build the data image" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var diag: Diagnostic = .{};
    const parsed = try parse(arena.allocator(), "variable a create t 1 , $0203 , 2 allot variable b", &diag);
    try std.testing.expectEqualSlices(u8, &.{ 0, 0, 1, 0, 3, 2, 0, 0, 0, 0 }, parsed.data);
}
