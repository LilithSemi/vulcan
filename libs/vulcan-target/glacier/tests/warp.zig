//! A model of one Glacier warp's scalar execution, so two word streams can be compared by what
//! they RENDER rather than by how they are spelled.
//!
//! A compiler and a hand assembler pick different registers and a different instruction order, so
//! a word diff says nothing about whether they agree. Running both and diffing the framebuffer
//! does. The FPGA is the real answer, and this is what makes a disagreement visible before a
//! kernel ever reaches it.
//!
//! It models only what a Glacier core decodes and only what a scalar kernel uses: RV32I, RV32M,
//! and a CSR read. An unknown word is an error, never a no-op, so a lowering that emits something
//! this model does not know about is caught rather than silently skipped.

const std = @import("std");

pub const Error = error{
    /// A word this model does not decode. Either the kernel is wrong or the model is behind.
    Illegal,
    /// An access outside the framebuffer the test gave the warp.
    OutOfRange,
    Unaligned,
    /// The warp ran longer than the caller allowed, so it is in a loop that does not end.
    Budget,
};

/// The CSRs a Glacier core answers. Anything else is an illegal word here, because a kernel that
/// reads one is relying on hardware this model cannot stand in for.
const csr_mhartid: u12 = 0xF14;
const csr_hart_count: u12 = 0xFC0;

const ebreak: u32 = 0x00100073;

/// The memory a kernel writes, as words at `base`. A Glacier kernel reaches memory by absolute
/// address, so the base is part of the program's contract with its host.
pub const Memory = struct {
    base: u32,
    words: []u32,

    fn index(self: Memory, addr: u32) Error!usize {
        if (addr % 4 != 0) return error.Unaligned;
        if (addr < self.base) return error.OutOfRange;
        const at = (addr - self.base) / 4;
        if (at >= self.words.len) return error.OutOfRange;
        return at;
    }
};

/// One warp: a register file, a program counter, and the identity the hardware gives it.
pub const Warp = struct {
    x: [32]u32 = @splat(0),
    pc: u32 = 0,
    hart_id: u32,
    hart_count: u32,

    fn set(self: *Warp, rd: u5, value: u32) void {
        if (rd != 0) self.x[rd] = value;
    }
};

fn signExtend(value: u32, bits: u5) i32 {
    const shift: u5 = @intCast(32 - @as(u6, bits));
    return @as(i32, @bitCast(value << shift)) >> shift;
}

fn immI(word: u32) i32 {
    return signExtend(word >> 20, 12);
}

fn immS(word: u32) i32 {
    const low = (word >> 7) & 0x1F;
    const high = word >> 25;
    return signExtend((high << 5) | low, 12);
}

fn immB(word: u32) i32 {
    const bit11 = (word >> 7) & 1;
    const low = (word >> 8) & 0xF;
    const mid = (word >> 25) & 0x3F;
    const sign = (word >> 31) & 1;
    return signExtend((sign << 12) | (bit11 << 11) | (mid << 5) | (low << 1), 13);
}

fn immJ(word: u32) i32 {
    const bit11 = (word >> 20) & 1;
    const low = (word >> 21) & 0x3FF;
    const high = (word >> 12) & 0xFF;
    const sign = (word >> 31) & 1;
    return signExtend((sign << 20) | (high << 12) | (bit11 << 11) | (low << 1), 21);
}

/// Run `code` from word zero until it retires, and return how many instructions it took.
///
/// `code` sits at address zero of its own aperture, which is separate from `mem`: a Glacier core
/// takes its program through the instruction aperture and its data over the bus, so nothing a
/// kernel stores can rewrite the kernel.
pub fn run(code: []const u32, mem: *Memory, warp: *Warp, budget: usize) Error!usize {
    var steps: usize = 0;
    while (true) {
        if (steps == budget) return error.Budget;
        if (warp.pc % 4 != 0) return error.Unaligned;
        const at = warp.pc / 4;
        if (at >= code.len) return error.OutOfRange;
        const word = code[at];
        if (word == ebreak) return steps;
        steps += 1;
        warp.pc += 4;
        try step(word, mem, warp);
    }
}

fn step(word: u32, mem: *Memory, warp: *Warp) Error!void {
    const opcode = word & 0x7F;
    const rd: u5 = @intCast((word >> 7) & 0x1F);
    const rs1: u5 = @intCast((word >> 15) & 0x1F);
    const rs2: u5 = @intCast((word >> 20) & 0x1F);
    const funct3 = (word >> 12) & 0x7;
    const funct7 = word >> 25;
    const a = warp.x[rs1];
    const b = warp.x[rs2];

    switch (opcode) {
        // LUI
        0x37 => warp.set(rd, word & 0xFFFF_F000),
        // AUIPC. The program counter has already advanced, so the base is one word back.
        0x17 => warp.set(rd, (warp.pc -% 4) +% (word & 0xFFFF_F000)),
        // JAL
        0x6F => {
            const target = (warp.pc -% 4) +% @as(u32, @bitCast(immJ(word)));
            warp.set(rd, warp.pc);
            warp.pc = target;
        },
        // JALR
        0x67 => {
            if (funct3 != 0) return error.Illegal;
            const target = (a +% @as(u32, @bitCast(immI(word)))) & ~@as(u32, 1);
            warp.set(rd, warp.pc);
            warp.pc = target;
        },
        // Conditional branches.
        0x63 => {
            const taken = switch (funct3) {
                0 => a == b,
                1 => a != b,
                4 => @as(i32, @bitCast(a)) < @as(i32, @bitCast(b)),
                5 => @as(i32, @bitCast(a)) >= @as(i32, @bitCast(b)),
                6 => a < b,
                7 => a >= b,
                else => return error.Illegal,
            };
            if (taken) warp.pc = (warp.pc -% 4) +% @as(u32, @bitCast(immB(word)));
        },
        // Loads.
        0x03 => {
            const addr = a +% @as(u32, @bitCast(immI(word)));
            if (funct3 != 2) return error.Illegal; // only LW, see encode.zig on narrow access
            warp.set(rd, mem.words[try mem.index(addr)]);
        },
        // Stores.
        0x23 => {
            const addr = a +% @as(u32, @bitCast(immS(word)));
            if (funct3 != 2) return error.Illegal; // only SW, see encode.zig on narrow stores
            mem.words[try mem.index(addr)] = b;
        },
        // Register-immediate arithmetic.
        0x13 => {
            const imm = immI(word);
            const uimm: u32 = @bitCast(imm);
            const shamt: u5 = @intCast(rs2);
            warp.set(rd, switch (funct3) {
                0 => a +% uimm,
                1 => a << shamt,
                2 => if (@as(i32, @bitCast(a)) < imm) 1 else 0,
                3 => if (a < uimm) 1 else 0,
                4 => a ^ uimm,
                5 => switch (funct7) {
                    0x00 => a >> shamt,
                    0x20 => @bitCast(@as(i32, @bitCast(a)) >> shamt),
                    else => return error.Illegal,
                },
                6 => a | uimm,
                7 => a & uimm,
                else => unreachable,
            });
        },
        // Register-register arithmetic, including the M extension.
        0x33 => warp.set(rd, try arith(funct3, funct7, a, b)),
        // CSR read. CSRRS with x0 as the set mask is the only CSR access a Glacier core decodes.
        0x73 => {
            if (funct3 != 2 or rs1 != 0) return error.Illegal;
            warp.set(rd, switch (@as(u12, @intCast(word >> 20))) {
                csr_mhartid => warp.hart_id,
                csr_hart_count => warp.hart_count,
                else => return error.Illegal,
            });
        },
        else => return error.Illegal,
    }
}

fn arith(funct3: u32, funct7: u32, a: u32, b: u32) Error!u32 {
    const sa: i32 = @bitCast(a);
    const sb: i32 = @bitCast(b);
    const shamt: u5 = @truncate(b);
    if (funct7 == 0x01) return switch (funct3) {
        0 => a *% b,
        1 => @bitCast(@as(i32, @truncate((@as(i64, sa) * @as(i64, sb)) >> 32))),
        2 => @bitCast(@as(i32, @truncate((@as(i64, sa) * @as(i64, b)) >> 32))),
        3 => @truncate((@as(u64, a) * @as(u64, b)) >> 32),
        // Division by zero gives all ones, and the most negative value over minus one wraps.
        4 => if (b == 0)
            @bitCast(@as(i32, -1))
        else if (sa == std.math.minInt(i32) and sb == -1)
            @bitCast(sa)
        else
            @bitCast(@divTrunc(sa, sb)),
        5 => if (b == 0) std.math.maxInt(u32) else a / b,
        6 => if (b == 0)
            a
        else if (sa == std.math.minInt(i32) and sb == -1)
            0
        else
            @bitCast(@rem(sa, sb)),
        7 => if (b == 0) a else a % b,
        else => unreachable,
    };
    return switch (funct3) {
        0 => switch (funct7) {
            0x00 => a +% b,
            0x20 => a -% b,
            else => error.Illegal,
        },
        1 => a << shamt,
        2 => if (sa < sb) 1 else 0,
        3 => if (a < b) 1 else 0,
        4 => a ^ b,
        5 => switch (funct7) {
            0x00 => a >> shamt,
            0x20 => @bitCast(sa >> shamt),
            else => error.Illegal,
        },
        6 => a | b,
        7 => a & b,
        else => unreachable,
    };
}

/// Run `code` on `warps` warps that all share one framebuffer, and return the frame.
///
/// Every warp runs the SAME program and finds its own share from `mhartid`, which is how a
/// Glacier kernel is launched. The warps run one after another here. A Glacier core issues them
/// round-robin, and the two orders give the same frame only because each warp owns the pixels it
/// writes, which is a property of the kernel this is checking.
pub fn render(
    allocator: std.mem.Allocator,
    code: []const u32,
    fb_base: u32,
    pixels: usize,
    warps: u32,
    budget: usize,
) (Error || std.mem.Allocator.Error)![]u32 {
    const frame = try allocator.alloc(u32, pixels);
    errdefer allocator.free(frame);
    @memset(frame, 0);
    var mem: Memory = .{ .base = fb_base, .words = frame };
    for (0..warps) |w| {
        var warp: Warp = .{ .hart_id = @intCast(w), .hart_count = warps };
        _ = try run(code, &mem, &warp, budget);
    }
    return frame;
}
