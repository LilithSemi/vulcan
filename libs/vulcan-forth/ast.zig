//! Shared types: the parsed program, stack effects and compile diagnostics.

const std = @import("std");
const ir = @import("vulcan-ir");
const prims = @import("prims.zig");

pub const Prim = prims.Prim;

/// The first byte address of the memory mapped I/O window.
pub const io_base: u16 = 0xF000;

/// The largest number of cells a word may take or give back.
pub const max_cells: u8 = 16;

pub const Effect = struct { in: u8, out: u8 };

pub const Node = union(enum) {
    lit: u16,
    /// Call of the word with this index in `Program.words`.
    call: u32,
    prim: Prim,
    @"if": If,
    until: []const Node,
    again: []const Node,
    @"while": While,
    do: Do,

    pub const If = struct { then: []const Node, @"else": []const Node, has_else: bool };
    pub const While = struct { cond: []const Node, body: []const Node };
    pub const Do = struct { body: []const Node, plus: bool };
};

pub const Word = struct {
    name: []const u8,
    body: []const Node,
    has_recurse: bool,
    line: u32,
    /// The name of the source the word was defined in.
    file: []const u8 = "",
    effect: Effect = .{ .in = 0, .out = 0 },
};

pub const Lowered = struct {
    /// The symbol the word has in the JIT module.
    name: []const u8,
    func: ir.function.Function,
};

pub const Diagnostic = struct {
    buf: [256]u8 = undefined,
    len: usize = 0,
    /// The source being compiled. A message names it when it is not empty.
    file: []const u8 = "",

    pub fn message(self: *const Diagnostic) []const u8 {
        return self.buf[0..self.len];
    }

    /// Record a message and return the compile error.
    pub fn fail(self: *Diagnostic, comptime fmt: []const u8, args: anytype) error{Compile} {
        const text = if (self.file.len == 0)
            std.fmt.bufPrint(&self.buf, fmt, args) catch &self.buf
        else
            std.fmt.bufPrint(&self.buf, "{s}: " ++ fmt, .{self.file} ++ args) catch &self.buf;
        self.len = text.len;
        return error.Compile;
    }
};

pub const Program = struct {
    arena: std.heap.ArenaAllocator,
    words: []Word,
    /// The initial contents of the data area, from byte address 0.
    data: []const u8,
    /// Index of the last word named `main`.
    main: ?u32,
    functions: []Lowered,

    pub fn deinit(self: *Program) void {
        for (self.functions) |*lowered| lowered.func.deinit();
        self.arena.deinit();
    }
};
