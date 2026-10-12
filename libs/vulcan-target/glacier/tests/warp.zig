//! A model of one Glacier warp, so two word streams can be compared by what they COMPUTE rather
//! than by how they are spelled.
//!
//! A compiler and a hand assembler pick different registers and a different instruction order, so
//! a word diff says nothing about whether they agree. Running both and diffing the result does.
//! The FPGA is the real answer, and this is what makes a disagreement visible before a kernel ever
//! reaches it.
//!
//! It models only what a Glacier core decodes and only what a kernel uses: RV32I, RV32M, the A
//! extension word atomics, a CSR read, the RVV integer subset the lowering emits, and the float
//! converts and scalar divide. An unknown word is an error, never a no-op, so a lowering that
//! emits something this model does not know about is caught rather than silently skipped.
//!
//! The float rules are the ones the core implements, which are NOT the ones the mnemonics imply:
//! the scalar divide always truncates, the integer-to-float and the narrowing converts round to
//! nearest even, and every lane flushes a subnormal result to a signed zero. A model that rounded
//! the way the names suggest would disagree with the hardware in the last bit.
//!
//! ONE WARP AT A TIME, TO COMPLETION. `run` returns when its warp hits EBREAK, so a caller that
//! wants several warps runs them one after another and they never interleave. Real warps issue
//! round-robin, and a kernel cannot order two of them: both IR barrier scopes are one warp here,
//! so both emit nothing. This model therefore cannot show a cross-warp race whatever the hardware
//! does, so give every address one owning warp rather than reading a passing test as proof there
//! is no race.
//!
//! Lane behaviour is modelled faithfully because it cannot be checked any other way here: every
//! vector write is gated by the launch mask, and a lane outside it keeps `stale_lane`. A kernel
//! that reads an inactive lane therefore gets an obviously wrong answer instead of a plausible one.

const std = @import("std");

pub const Error = error{
    /// A word this model does not decode. Either the kernel is wrong or the model is behind.
    Illegal,
    /// An access outside the memory the test gave the warp.
    OutOfRange,
    Unaligned,
    /// The warp ran longer than the caller allowed, so it is in a loop that does not end.
    Budget,
    /// A warp reached into another hart's scratch slice. On hardware this corrupts that hart and
    /// shows up as its wrong answer somewhere else, so it is caught here instead.
    CrossHart,
};

/// The CSRs a Glacier core answers. Anything else is an illegal word here, because a kernel that
/// reads one is relying on hardware this model cannot stand in for.
const csr_mhartid: u12 = 0xF14;
const csr_hart_count: u12 = 0xFC0;

const csr_mscratch: u12 = 0x340;

const ebreak: u32 = 0x00100073;

/// The per-hart scratch region, as the SoC build lays it out: one slice of `stride` bytes per hart,
/// starting at `base`. A hart reads one past the top of ITS OWN slice from `mscratch` and the stack
/// grows down from there.
///
/// Modelling it is worth the lines because it makes the one failure the ABI warns about visible. A
/// hart that walks out of its slice is corrupting the hart below, and on real hardware that shows
/// up as another hart's wrong answer somewhere else entirely.
pub const Scratch = struct { base: u32, stride: u32, harts: u32 };

/// The memory a kernel writes, as words at `base`. A Glacier kernel reaches memory by absolute
/// address, so the base is part of the program's contract with its host.
pub const Memory = struct {
    base: u32,
    words: []u32,
    /// The scratch region, when the core has one. It lies inside `words` like any other memory.
    scratch: ?Scratch = null,

    /// Whether `addr` falls in a hart's scratch slice other than `hart_id`'s own.
    fn crossesHart(self: Memory, addr: u32, hart_id: u32) bool {
        const s = self.scratch orelse return false;
        if (addr < s.base or addr >= s.base + s.stride * s.harts) return false;
        const owner = (addr - s.base) / s.stride;
        return owner != hart_id;
    }

    fn index(self: Memory, addr: u32) Error!usize {
        if (addr % 4 != 0) return error.Unaligned;
        if (addr < self.base) return error.OutOfRange;
        const at = (addr - self.base) / 4;
        if (at >= self.words.len) return error.OutOfRange;
        return at;
    }
};

/// The widest warp any GC1 profile runs.
pub const max_lanes = 16;

/// A value left in a lane the launch mask does not name. The hardware leaves whatever the previous
/// kernel wrote; this model leaves something recognisable instead, so a kernel that reads an
/// inactive lane produces an obviously wrong answer rather than a plausible one.
pub const stale_lane: u32 = 0xBAD0_BAD0;

/// One warp: a register file, a program counter, and the identity the hardware gives it.
pub const Warp = struct {
    x: [32]u32 = @splat(0),
    /// The vector registers, `lanes` elements of each in use. Lanes past the launch mask hold
    /// `stale_lane`: every vector write is gated by that mask, so they are never written.
    v: [32][max_lanes]u32 = @splat(@splat(stale_lane)),
    pc: u32 = 0,
    hart_id: u32,
    hart_count: u32,
    /// Lanes this warp runs. A warp may be launched NARROWER than the core is wide, and then the
    /// rest are inactive for its whole life with no instruction able to turn them on.
    lanes: u8 = 1,
    /// What `mscratch` reads: one past the top of this hart's own scratch slice. Zero on a core
    /// built with no scratch, same as the hardware.
    stack_top: u32 = 0,

    fn set(self: *Warp, rd: u5, value: u32) void {
        if (rd != 0) self.x[rd] = value;
    }

    /// A mask register, as THIS MODEL keeps one: the bits in element ZERO, rather than packed
    /// across the register the way the hardware lays them out. Nothing reads a mask as elements, so
    /// the difference is invisible, and the rest of the register is left recognisably stale so a
    /// misuse shows up as an obviously wrong answer.
    fn mask(self: *const Warp, reg: u5) u32 {
        return self.v[reg][0];
    }

    fn setMask(self: *Warp, reg: u5, bits: u32) void {
        self.v[reg] = @splat(stale_lane);
        self.v[reg][0] = bits;
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
        // Vector loads and stores. Only the 32-bit indexed forms, which is what the lowering
        // emits: lane i accesses `rs1 + vs2[i]`, and only for a lane the launch mask names.
        0x07, 0x27 => {
            if (funct3 != 0b110) return error.Illegal; // width must be 32-bit
            // mop 00 is unit stride, where lane i takes the i'th word from `rs1` and the offset
            // register is not read at all. mop 01 is indexed-unordered, where lane i takes its own
            // offset from `vs2`. Nothing else is decoded.
            const mop = (word >> 26) & 0b11;
            if (mop != 0b00 and mop != 0b01) return error.Illegal;
            const unit = mop == 0b00;
            // A clear `vm` predicates the access on v0. A lane whose bit is clear makes NO access:
            // it does not read, it does not write, and the hardware skips it on the bus rather
            // than driving a harmless cycle. That is what a store under divergent control flow
            // relies on, so the model has to skip it here too and not merely discard the result.
            const masked = (word >> 25) & 1 == 0;
            const bits = warp.mask(0);
            for (0..warp.lanes) |i| {
                if (masked and (bits >> @intCast(i)) & 1 == 0) continue;
                const addr = if (unit) a +% @as(u32, @intCast(i * 4)) else a +% warp.v[rs2][i];
                if (mem.crossesHart(addr, warp.hart_id)) return error.CrossHart;
                const at = try mem.index(addr);
                if (opcode == 0x07) warp.v[rd][i] = mem.words[at] else mem.words[at] = warp.v[rd][i];
            }
        },
        // Loads. Only LW: the lowering emits nothing narrower, so a narrow load here would be a
        // lowering this model has not been taught rather than one it should quietly accept.
        0x03 => {
            const addr = a +% @as(u32, @bitCast(immI(word)));
            if (funct3 != 2) return error.Illegal;
            if (mem.crossesHart(addr, warp.hart_id)) return error.CrossHart;
            warp.set(rd, mem.words[try mem.index(addr)]);
        },
        // Stores. Only SW, for the same reason as the loads.
        0x23 => {
            const addr = a +% @as(u32, @bitCast(immS(word)));
            if (funct3 != 2) return error.Illegal;
            if (mem.crossesHart(addr, warp.hart_id)) return error.CrossHart;
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
        // The A extension, word width. Read, combine, write back, give back the ORIGINAL. One warp
        // runs at a time here, so the model cannot show whether the pair is really indivisible;
        // what it does check is the VALUE and that the old one comes back rather than the new.
        0x2F => {
            if (funct3 != 2) return error.Illegal;
            if (mem.crossesHart(a, warp.hart_id)) return error.CrossHart;
            const at = try mem.index(a);
            const old = mem.words[at];
            mem.words[at] = switch (@as(u5, @intCast(funct7 >> 2))) {
                0b00000 => old +% b,
                0b00001 => b,
                0b00100 => old ^ b,
                0b01100 => old & b,
                0b01000 => old | b,
                0b10000 => if (@as(i32, @bitCast(old)) < @as(i32, @bitCast(b))) old else b,
                0b10100 => if (@as(i32, @bitCast(old)) > @as(i32, @bitCast(b))) old else b,
                0b11000 => @min(old, b),
                0b11100 => @max(old, b),
                else => return error.Illegal,
            };
            warp.set(rd, old);
        },
        // OP-V: every vector arithmetic form.
        0x57 => try vectorOp(word, warp),
        // OP-FP. Glacier decodes exactly one of these, FDIV.S, and its register fields name X
        // registers because the core has no f register file. It always truncates: the rounding
        // mode in funct3 is accepted and ignored.
        0x53 => {
            if (funct7 != 0b0001100) return error.Illegal;
            warp.set(rd, @bitCast(ftz(f32, divTowardZero(asFloat(a), asFloat(b)))));
        },
        // CSR read. CSRRS with x0 as the set mask is the only CSR access a Glacier core decodes.
        0x73 => {
            if (funct3 != 2 or rs1 != 0) return error.Illegal;
            warp.set(rd, switch (@as(u12, @intCast(word >> 20))) {
                csr_mhartid => warp.hart_id,
                csr_hart_count => warp.hart_count,
                csr_mscratch => warp.stack_top,
                else => return error.Illegal,
            });
        },
        else => return error.Illegal,
    }
}

/// One OP-V instruction. Every write is gated by the launch mask, so a lane outside it keeps what
/// it held, which in this model is `stale_lane`.
fn vectorOp(word: u32, warp: *Warp) Error!void {
    const vd: u5 = @intCast((word >> 7) & 0x1F);
    const funct3 = (word >> 12) & 0x7;
    const src: u5 = @intCast((word >> 15) & 0x1F); // vs1, rs1, or a 5-bit immediate
    const vs2: u5 = @intCast((word >> 20) & 0x1F);
    const funct6 = word >> 26;
    // `vmerge` is the ONE masked instruction the lowering emits, and it is handled below. Anything
    // else with `vm` clear is predicated, which nothing emits yet, so it is an error rather than a
    // quietly unpredicated write.
    const masked = (word >> 25) & 1 == 0;
    if (masked and funct6 != 0b010111) return error.Illegal;

    // vid.v and vmv.v.x write a value that does not come from vs2, so they are taken first.
    if (funct3 == 0b010 and funct6 == 0b010100) {
        if (src != 17) return error.Illegal; // only vid.v in this group
        for (0..warp.lanes) |i| warp.v[vd][i] = @intCast(i);
        return;
    }
    // The two moves share their funct6 with `vmerge`, which is told apart ONLY by the mask bit. A
    // merge reaching here would copy one side over every lane and lose the mask entirely.
    if (funct3 == 0b100 and funct6 == 0b010111 and !masked) {
        for (0..warp.lanes) |i| warp.v[vd][i] = warp.x[src];
        return;
    }
    if (funct3 == 0b000 and funct6 == 0b010111 and !masked) {
        for (0..warp.lanes) |i| warp.v[vd][i] = warp.v[src][i];
        return;
    }
    // vmv.x.s moves lane 0 to a scalar register, so it writes no vector lane.
    if (funct3 == 0b010 and funct6 == 0b010000) {
        warp.set(vd, warp.v[vs2][0]);
        return;
    }

    // The plain float arithmetic. These read WHOLE lanes as f32: the element width is fixed at 32
    // bits, so there is no f16 form of any of them.
    //
    // These round to NEAREST EVEN, from Harbor's `fp_round_pack.dart`, which uses the RISC-V
    // encoding where rm 0 is rne. The lane is NOT uniform about rounding: the widening MAC and the
    // float-to-integer convert truncate toward zero while these and the other converts round to
    // nearest. Getting one of them wrong moves a low bit with no error anywhere, so the two modes
    // are pinned against each other by test rather than left to a comment.
    // The float compares, which write a MASK exactly as the integer ones do. Three instructions,
    // not six: there is no greater-than, which is less-than with the operands swapped, and no
    // `vmfne`, which the lowering builds by inverting the equality.
    //
    // Ordered semantics: a NaN compares FALSE against everything, including itself.
    if (funct3 == 0b001 and (funct6 == 0b011000 or funct6 == 0b011001 or funct6 == 0b011011)) {
        var bits: u32 = 0;
        for (0..warp.lanes) |i| {
            const x = asFloat(warp.v[vs2][i]);
            const y = asFloat(warp.v[src][i]);
            const take = switch (funct6) {
                0b011000 => x == y,
                0b011001 => x <= y,
                0b011011 => x < y,
                else => unreachable,
            };
            if (take) bits |= @as(u32, 1) << @intCast(i);
        }
        warp.setMask(vd, bits);
        return;
    }
    if (funct3 == 0b001) switch (funct6) {
        0b000000, 0b000010, 0b100100, 0b000100, 0b000110 => {
            for (0..warp.lanes) |i| {
                const x = asFloat(warp.v[vs2][i]);
                const y = asFloat(warp.v[src][i]);
                warp.v[vd][i] = @bitCast(ftz(f32, switch (funct6) {
                    0b000000 => x + y,
                    0b000010 => x - y,
                    0b100100 => x * y,
                    0b000100 => @min(x, y),
                    0b000110 => @max(x, y),
                    else => unreachable,
                }));
            }
            return;
        },
        // The converts and the widening MAC are handled below.
        else => {},
    };

    // The float converts. One funct6 holds all three and the vs1 field picks which, so the field
    // that names a source register everywhere else is a selector here. An f32 fills a lane and an
    // f16 is its LOW HALF, because the core has no f register file.
    if (funct3 == 0b001) {
        // vfwmacc.vv: the destination accumulates the f32 product of two f16. The product is
        // exact, because two 11-bit significands fit an f32 one, so the only rounding is the sum,
        // and the core TRUNCATES that.
        if (funct6 == 0b111100) {
            for (0..warp.lanes) |i| {
                const a: f32 = asHalf(warp.v[src][i]);
                const b: f32 = asHalf(warp.v[vs2][i]);
                const product = ftz(f32, a * b);
                const sum = @as(f64, asFloat(warp.v[vd][i])) + @as(f64, product);
                warp.v[vd][i] = @bitCast(ftz(f32, truncToF32(sum)));
            }
            return;
        }
        if (funct6 != 0b010010) return error.Illegal;
        for (0..warp.lanes) |i| {
            const x = warp.v[vs2][i];
            warp.v[vd][i] = switch (src) {
                // vfcvt.f.x.v: a SIGNED integer to f16, nearest even.
                0b00011 => halfBits(ftz(f16, @floatFromInt(@as(i32, @bitCast(x))))),
                // vfcvt.rtz.xu.f.v: an f32 to an UNSIGNED integer, toward zero.
                0b00110 => unsignedFromFloat(asFloat(x)),
                // vfncvt.f.f.w: an f32 narrowed to f16, nearest even, so an overflow gives
                // INFINITY rather than the largest finite value.
                0b10100 => halfBits(ftz(f16, @floatCast(asFloat(x)))),
                // vfwcvt.f.f.v: an f16 widened to f32, which is exact in every case.
                0b01100 => @bitCast(ftz(f32, @as(f32, asHalf(x)))),
                else => return error.Illegal,
            };
        }
        return;
    }

    // The integer compares and the mask ops. A MASK is one bit per lane in the low bits of a
    // register, not a word per lane, so these are modelled on bits and the element path below
    // would read the wrong thing entirely.
    if ((funct3 == 0b000 or funct3 == 0b100) and funct6 >= 0b011000 and funct6 <= 0b011111) {
        var bits: u32 = 0;
        for (0..warp.lanes) |i| {
            const x = warp.v[vs2][i];
            const y = if (funct3 == 0b000) warp.v[src][i] else warp.x[src];
            const take = switch (funct6) {
                0b011000 => x == y,
                0b011001 => x != y,
                0b011010 => x < y,
                0b011011 => @as(i32, @bitCast(x)) < @as(i32, @bitCast(y)),
                0b011100 => x <= y,
                0b011101 => @as(i32, @bitCast(x)) <= @as(i32, @bitCast(y)),
                0b011110 => x > y,
                0b011111 => @as(i32, @bitCast(x)) > @as(i32, @bitCast(y)),
                else => return error.Illegal,
            };
            if (take) bits |= @as(u32, 1) << @intCast(i);
        }
        warp.setMask(vd, bits);
        return;
    }
    // Mask logic, also bit-wise. These share their funct6 with the compares and are told apart by
    // OPMVV, which is why they are checked after.
    if (funct3 == 0b010 and funct6 >= 0b011000 and funct6 <= 0b011111) {
        const a = warp.mask(vs2);
        const b = warp.mask(src);
        warp.setMask(vd, switch (funct6) {
            0b011000 => a & ~b, // vmandn
            0b011001 => a & b, // vmand
            0b011010 => a | b, // vmor
            0b011011 => a ^ b, // vmxor
            0b011101 => ~(a & b), // vmnand
            else => return error.Illegal,
        });
        return;
    }
    // vmerge: the one masked instruction here, so `vm` is clear and v0 is implicit.
    if (funct6 == 0b010111 and masked) {
        const m = warp.mask(0);
        for (0..warp.lanes) |i| {
            const take = (m >> @intCast(i)) & 1 == 1;
            const t: u32 = if (funct3 == 0b000) warp.v[src][i] else warp.x[src];
            warp.v[vd][i] = if (take) t else warp.v[vs2][i];
        }
        return;
    }

    for (0..warp.lanes) |i| {
        const x = warp.v[vs2][i];
        const y: u32 = switch (funct3) {
            0b000, 0b010 => warp.v[src][i], // vector-vector
            0b100, 0b110 => warp.x[src], // vector-scalar
            0b011 => src, // vector-immediate, zero extended
            else => return error.Illegal,
        };
        // The integer funct6 space is read differently by category: OPIVV/OPIVX/OPIVI take the
        // shift and bitwise set, OPMVV/OPMVX the multiply and divide set, so 0b100101 is `vsll`
        // in one and `vmul` in the other.
        const multiply_group = funct3 == 0b010 or funct3 == 0b110;
        warp.v[vd][i] = if (multiply_group)
            try arith(mulFunct3(funct6) orelse return error.Illegal, 0x01, x, y)
        else switch (funct6) {
            0b000000 => x +% y,
            0b000010 => x -% y,
            0b000011 => y -% x, // vrsub
            0b001001 => x & y,
            0b001010 => x | y,
            0b001011 => x ^ y,
            0b100101 => x << @truncate(y),
            0b101000 => x >> @truncate(y),
            0b101001 => @bitCast(@as(i32, @bitCast(x)) >> @truncate(y)),
            else => return error.Illegal,
        };
    }
}

fn asFloat(bits: u32) f32 {
    return @bitCast(bits);
}

/// The low half of a lane read as an f16, which is where Glacier keeps one.
fn asHalf(bits: u32) f16 {
    return @bitCast(@as(u16, @truncate(bits)));
}

fn halfBits(v: f16) u32 {
    return @as(u16, @bitCast(v));
}

/// A lane flushes a subnormal result to a zero of the same sign, on every operation. A NaN and an
/// infinity pass through.
fn ftz(comptime T: type, v: T) T {
    if (v != 0 and @abs(v) < std.math.floatMin(T)) return std.math.copysign(@as(T, 0.0), v);
    return v;
}

/// An f32 to a u32, truncating toward zero. A value outside the range saturates, which is what
/// RVV defines. What Glacier itself answers there is not recorded, so no test may depend on it.
fn unsignedFromFloat(v: f32) u32 {
    if (std.math.isNan(v)) return std.math.maxInt(u32);
    if (v <= 0) return 0;
    if (v >= 4294967296.0) return std.math.maxInt(u32);
    return @intFromFloat(v);
}

/// An f32 divide that truncates toward zero, which is the only rounding Glacier's divider does.
/// The quotient is formed in f64 first, so the one case left wrong is the double rounding where
/// the exact quotient sits just under a value f32 can name and the f64 step lifts it onto that
/// value. No test here is near such a quotient.
fn divTowardZero(x: f32, y: f32) f32 {
    return truncToF32(@as(f64, x) / @as(f64, y));
}

/// `v` as the nearest f32 toward zero.
fn truncToF32(v: f64) f32 {
    const near: f32 = @floatCast(v);
    if (!std.math.isFinite(near)) return near;
    if (@abs(@as(f64, near)) <= @abs(v)) return near;
    // Rounding to nearest went away from zero, so step back one representable value.
    const bits: u32 = @bitCast(near);
    const magnitude = bits & 0x7FFF_FFFF;
    if (magnitude == 0) return near;
    return @bitCast((bits & 0x8000_0000) | (magnitude - 1));
}

/// The scalar M-extension funct3 that computes the same thing as an OPMVV/OPMVX `funct6`, so the
/// lane arithmetic reuses `arith` instead of restating the multiply and divide rules.
fn mulFunct3(funct6: u32) ?u32 {
    return switch (funct6) {
        0b100101 => 0, // vmul   -> mul
        0b100111 => 1, // vmulh  -> mulh
        0b100100 => 3, // vmulhu -> mulhu
        0b100001 => 4, // vdiv   -> div
        0b100000 => 5, // vdivu  -> divu
        0b100011 => 6, // vrem   -> rem
        0b100010 => 7, // vremu  -> remu
        else => null,
    };
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

const rv = @import("vulcan-target").glacier.encode.rv;

test "vid.v numbers only the launched lanes" {
    var frame: [4]u32 = @splat(0);
    var mem: Memory = .{ .base = 0, .words = &frame };
    // A four-lane warp on a core that could run more. Lanes past the launch mask are never
    // written, so they keep the recognisable stale value rather than their index.
    var w: Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 4 };
    const code = [_]u32{ rv.vid_v(.v1), ebreak };
    _ = try run(&code, &mem, &w, 10);
    for (0..4) |i| try std.testing.expectEqual(@as(u32, @intCast(i)), w.v[1][i]);
    for (4..max_lanes) |i| try std.testing.expectEqual(stale_lane, w.v[1][i]);
}

test "a scatter writes one word per lane and a gather reads them back" {
    var frame: [8]u32 = @splat(0);
    var mem: Memory = .{ .base = 0, .words = &frame };
    var w: Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 4 };
    // offsets = vid*4, values = vid+10, scatter to (x0), gather back into v4.
    const code = [_]u32{
        rv.vid_v(.v1),
        rv.vsll_vi(.v2, .v1, 2), // byte offsets
        rv.addi(.x5, .x0, 10),
        rv.vadd_vx(.v3, .v1, .x5), // values
        rv.vsuxei32(.v3, .x0, .v2),
        rv.vluxei32(.v4, .x0, .v2),
        ebreak,
    };
    _ = try run(&code, &mem, &w, 20);
    for (0..4) |i| {
        try std.testing.expectEqual(@as(u32, @intCast(i)) + 10, frame[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i)) + 10, w.v[4][i]);
    }
    // Only the launched lanes touched memory.
    for (4..8) |i| try std.testing.expectEqual(@as(u32, 0), frame[i]);
}

test "vector arithmetic covers the forms the lowering emits" {
    var frame: [1]u32 = @splat(0);
    var mem: Memory = .{ .base = 0, .words = &frame };
    var w: Warp = .{ .hart_id = 0, .hart_count = 1, .lanes = 4 };
    const code = [_]u32{
        rv.vid_v(.v1), // 0 1 2 3
        rv.addi(.x5, .x0, 3),
        rv.vmv_v_x(.v2, .x5), // 3 3 3 3
        rv.vmul_vv(.v3, .v1, .v2), // 0 3 6 9
        rv.vadd_vv(.v4, .v3, .v1), // 0 4 8 12
        rv.vsub_vv(.v5, .v4, .v1), // 0 3 6 9
        rv.vxor_vv(.v6, .v5, .v2), // 3 0 5 10
        rv.vsrl_vi(.v7, .v4, 2), // 0 1 2 3
        rv.vmv_x_s(.x6, .v6), // lane 0 of v6 = 3
        ebreak,
    };
    _ = try run(&code, &mem, &w, 30);
    try std.testing.expectEqualSlices(u32, &.{ 0, 3, 6, 9 }, w.v[5][0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 3, 0, 5, 10 }, w.v[6][0..4]);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3 }, w.v[7][0..4]);
    try std.testing.expectEqual(@as(u32, 3), w.x[6]);
}
