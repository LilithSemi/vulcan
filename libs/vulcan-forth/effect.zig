//! Static stack effect inference. A word has the effect (in, out) when it takes `in`
//! cells from the data stack and leaves `out` cells in their place. Every path through a
//! word must agree, so the lowering can keep the stacks as SSA values.

const std = @import("std");
const ast = @import("ast.zig");
const prims = @import("prims.zig");

const Node = ast.Node;
const Word = ast.Word;
const Effect = ast.Effect;
const Diagnostic = ast.Diagnostic;

pub const Error = error{Compile};

const max_loop_depth = 16;

/// How many cells a recursive call may take or give when the effect is searched for.
const max_recursive_cells = 8;

const State = struct {
    d: i32 = 0,
    r: i32 = 0,
    loops: u8 = 0,
    dead: bool = false,

    fn sameShape(a: State, b: State) bool {
        return a.d == b.d and a.r == b.r and a.loops == b.loops;
    }
};

const Leave = struct { d: i32, r: i32 };

const Sim = struct {
    word: *const Word,
    words: []const Word,
    recursive: ?Effect,
    diag: *Diagnostic,
    min: i32 = 0,
    exit_d: ?i32 = null,
    leaves: [max_loop_depth]?Leave = @splat(null),
    depth: usize = 0,

    fn pop(self: *Sim, st: *State, n: u8) void {
        st.d -= n;
        self.min = @min(self.min, st.d);
    }

    fn push(st: *State, n: u8) void {
        st.d += n;
    }

    fn fail(self: *Sim, comptime fmt: []const u8, args: anytype) Error {
        var buf: [200]u8 = undefined;
        const detail = std.fmt.bufPrint(&buf, fmt, args) catch &buf;
        return self.diag.fail("word '{s}' (line {d}): {s}", .{ self.word.name, self.word.line, detail });
    }

    fn seq(self: *Sim, nodes: []const Node, st: *State) Error!void {
        for (nodes) |n| {
            if (st.dead) return;
            try self.node(n, st);
        }
    }

    fn merge(self: *Sim, a: State, b: State, what: []const u8, a_name: []const u8, b_name: []const u8, st: *State) Error!void {
        if (a.dead or b.dead) {
            st.* = if (a.dead) b else a;
            st.dead = a.dead and b.dead;
            return;
        }
        if (!a.sameShape(b)) {
            return self.fail("'{s}' branches leave different stack depths ({s} {d}, {s} {d}, return stack {d} and {d})", .{ what, a_name, a.d, b_name, b.d, a.r, b.r });
        }
        st.* = a;
    }

    fn balanced(self: *Sim, start: State, end: State, what: []const u8) Error!void {
        if (end.dead or start.sameShape(end)) return;
        return self.fail("'{s}' body changes the stack depth by {d} per pass (return stack {d})", .{ what, end.d - start.d, end.r - start.r });
    }

    fn node(self: *Sim, n: Node, st: *State) Error!void {
        switch (n) {
            .lit => push(st, 1),
            .call => |idx| {
                const eff = self.words[idx].effect;
                self.pop(st, eff.in);
                push(st, eff.out);
            },
            .prim => |p| try self.prim(p, st),
            .@"if" => |branch| {
                self.pop(st, 1);
                var a = st.*;
                var b = st.*;
                try self.seq(branch.then, &a);
                if (branch.has_else) {
                    try self.seq(branch.@"else", &b);
                    try self.merge(a, b, "if", "then", "else", st);
                } else {
                    if (!a.dead and !a.sameShape(b)) {
                        return self.fail("'if' without 'else' changes the stack depth by {d} (return stack {d})", .{ a.d - b.d, a.r - b.r });
                    }
                    st.* = b;
                }
            },
            .until => |body| {
                const start = st.*;
                try self.seq(body, st);
                if (st.dead) return;
                self.pop(st, 1);
                try self.balanced(start, st.*, "begin ... until");
            },
            .again => |body| {
                const start = st.*;
                try self.seq(body, st);
                try self.balanced(start, st.*, "begin ... again");
                st.dead = true;
            },
            .@"while" => |loop| {
                const start = st.*;
                try self.seq(loop.cond, st);
                if (st.dead) return;
                self.pop(st, 1);
                const leaving = st.*;
                try self.seq(loop.body, st);
                try self.balanced(start, st.*, "begin ... while ... repeat");
                st.* = leaving;
            },
            .do => |loop| try self.doLoop(loop, st),
        }
    }

    fn doLoop(self: *Sim, loop: Node.Do, st: *State) Error!void {
        if (self.depth == max_loop_depth) return self.fail("do loops nest deeper than {d}", .{max_loop_depth});
        self.pop(st, 2);
        const outer = st.*;
        var inner = outer;
        inner.loops += 1;
        const slot = self.depth;
        self.depth += 1;
        self.leaves[slot] = null;
        defer self.depth -= 1;
        const start = inner;
        try self.seq(loop.body, &inner);
        if (!inner.dead) {
            if (loop.plus) self.pop(&inner, 1);
            try self.balanced(start, inner, if (loop.plus) "do ... +loop" else "do ... loop");
            inner.loops -= 1;
            if (self.leaves[slot]) |lv| {
                if (lv.d != inner.d or lv.r != inner.r) return self.fail("'leave' leaves a different stack depth than the end of the loop", .{});
            }
            st.* = inner;
        } else if (self.leaves[slot]) |lv| {
            st.* = .{ .d = lv.d, .r = lv.r, .loops = outer.loops };
        } else {
            st.* = inner;
        }
    }

    fn prim(self: *Sim, p: ast.Prim, st: *State) Error!void {
        switch (p) {
            .to_r => {
                self.pop(st, 1);
                st.r += 1;
            },
            .r_from => {
                if (st.r == 0) return self.fail("'r>' has no matching '>r' in this word", .{});
                st.r -= 1;
                push(st, 1);
            },
            .r_fetch => {
                if (st.r == 0) return self.fail("'r@' has no matching '>r' in this word", .{});
                push(st, 1);
            },
            .i => {
                if (st.loops < 1) return self.fail("'i' is used outside a 'do' loop", .{});
                push(st, 1);
            },
            .j => {
                if (st.loops < 2) return self.fail("'j' needs two nested 'do' loops", .{});
                push(st, 1);
            },
            .unloop => {
                if (st.loops < 1) return self.fail("'unloop' is used outside a 'do' loop", .{});
                st.loops -= 1;
            },
            .leave => {
                if (self.depth == 0 or st.loops != self.depth) return self.fail("'leave' is used outside a 'do' loop", .{});
                const here: Leave = .{ .d = st.d, .r = st.r };
                if (self.leaves[self.depth - 1]) |prev| {
                    if (prev.d != here.d or prev.r != here.r) return self.fail("two 'leave' words leave different stack depths", .{});
                }
                self.leaves[self.depth - 1] = here;
                st.dead = true;
            },
            .exit => {
                if (st.loops != 0) return self.fail("'exit' inside a 'do' loop needs 'unloop' first", .{});
                if (st.r != 0) return self.fail("'exit' leaves a '>r' value on the return stack", .{});
                if (self.exit_d) |d| {
                    if (d != st.d) return self.fail("two 'exit' words leave different stack depths", .{});
                }
                self.exit_d = st.d;
                st.dead = true;
            },
            .recurse => {
                const eff = self.recursive orelse return self.fail("'recurse' has no stack effect yet", .{});
                self.pop(st, eff.in);
                push(st, eff.out);
            },
            else => {
                const eff = prims.dataEffect(p);
                self.pop(st, eff.pops);
                push(st, eff.pushes);
            },
        }
    }

    fn finish(self: *Sim, st: State) Error!Effect {
        var final_d: i32 = undefined;
        if (st.dead) {
            // A word that never returns gives no cells.
            final_d = self.exit_d orelse self.min;
        } else {
            if (st.r != 0) return self.fail("'>r' is not matched by 'r>' before the word ends", .{});
            if (self.exit_d) |d| {
                if (d != st.d) return self.fail("'exit' leaves a different stack depth than the end of the word", .{});
            }
            final_d = st.d;
        }
        const in = -self.min;
        const out = final_d - self.min;
        if (in > ast.max_cells or out > ast.max_cells) {
            return self.fail("takes {d} and gives {d} cells, the limit is {d}", .{ in, out, ast.max_cells });
        }
        return .{ .in = @intCast(in), .out = @intCast(out) };
    }
};

fn run(words: []const Word, word: *const Word, recursive: ?Effect, diag: *Diagnostic) Error!Effect {
    var sim: Sim = .{ .word = word, .words = words, .recursive = recursive, .diag = diag };
    var st: State = .{};
    try sim.seq(word.body, &st);
    return sim.finish(st);
}

/// The stack effect of `word`. `words` holds every word before it, with effects set.
pub fn infer(words: []const Word, word: *const Word, diag: *Diagnostic) Error!Effect {
    diag.file = word.file;
    if (!word.has_recurse) return run(words, word, null, diag);
    var total: u8 = 0;
    while (total <= 2 * max_recursive_cells) : (total += 1) {
        var in: u8 = 0;
        while (in <= @min(total, max_recursive_cells)) : (in += 1) {
            const out = total - in;
            if (out > max_recursive_cells) continue;
            const guess: Effect = .{ .in = in, .out = out };
            const got = run(words, word, guess, diag) catch continue;
            if (got.in == guess.in and got.out == guess.out) return got;
        }
    }
    return diag.fail("word '{s}' (line {d}): no stack effect fits its recursive calls", .{ word.name, word.line });
}
