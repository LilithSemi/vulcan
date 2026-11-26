//! Forth lexer: whitespace separated tokens with `( ... )` and `\` comments removed.

const std = @import("std");

pub const Token = struct {
    text: []const u8,
    line: u32,
};

pub const Error = error{UnterminatedComment};

pub const Lexer = struct {
    source: []const u8,
    pos: usize = 0,
    line: u32 = 1,

    pub fn init(source: []const u8) Lexer {
        return .{ .source = source };
    }

    /// The next token, or null at the end of the source.
    pub fn next(self: *Lexer) Error!?Token {
        while (true) {
            self.skipSpace();
            if (self.pos >= self.source.len) return null;
            const start = self.pos;
            const line = self.line;
            while (self.pos < self.source.len and !std.ascii.isWhitespace(self.source[self.pos])) self.pos += 1;
            const text = self.source[start..self.pos];
            if (std.mem.eql(u8, text, "\\")) {
                while (self.pos < self.source.len and self.source[self.pos] != '\n') self.pos += 1;
                continue;
            }
            if (std.mem.eql(u8, text, "(")) {
                try self.skipParenComment();
                continue;
            }
            return .{ .text = text, .line = line };
        }
    }

    fn skipSpace(self: *Lexer) void {
        while (self.pos < self.source.len and std.ascii.isWhitespace(self.source[self.pos])) {
            if (self.source[self.pos] == '\n') self.line += 1;
            self.pos += 1;
        }
    }

    fn skipParenComment(self: *Lexer) Error!void {
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            self.pos += 1;
            if (c == '\n') self.line += 1;
            if (c == ')') return;
        }
        return error.UnterminatedComment;
    }
};

test "tokens skip comments and track lines" {
    var lexer = Lexer.init(": sq ( n -- n ) dup * ; \\ square\n 3 sq");
    const expected = [_][]const u8{ ":", "sq", "dup", "*", ";", "3", "sq" };
    for (expected) |want| {
        const tok = (try lexer.next()).?;
        try std.testing.expectEqualStrings(want, tok.text);
    }
    try std.testing.expectEqual(@as(?Token, null), try lexer.next());
}

test "line numbers count newlines" {
    var lexer = Lexer.init("a\nb\n\nc");
    try std.testing.expectEqual(@as(u32, 1), (try lexer.next()).?.line);
    try std.testing.expectEqual(@as(u32, 2), (try lexer.next()).?.line);
    try std.testing.expectEqual(@as(u32, 4), (try lexer.next()).?.line);
}

test "an open parenthesis comment is an error" {
    var lexer = Lexer.init("1 ( never closed");
    _ = try lexer.next();
    try std.testing.expectError(error.UnterminatedComment, lexer.next());
}

test "a paren inside a word is not a comment" {
    var lexer = Lexer.init("a(b c");
    try std.testing.expectEqualStrings("a(b", (try lexer.next()).?.text);
}
