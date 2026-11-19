//! Lifting Glacier instruction words into Vulcan IR, so a host can run a kernel as native code.
//!
//! A Glacier core is a device. Words are written into its instruction aperture and a host strobes
//! a launch. There is no supervisor inside it, no traps and no calls, so one kernel is one
//! function and almost every edge between its words is direct. That is what makes this work: the
//! control flow of a region is in the words themselves, so a region lifts into one IR function
//! with real blocks rather than into a loop that dispatches on a program counter.
//!
//! A region ends where the words stop saying where they go. `ebreak` retires the warp, `ecall`
//! faults it, a barrier parks it until every other live warp arrives, and `jalr` goes wherever a
//! register points. Each of those gives the warp back with a reason and the pc to carry on from,
//! and a caller that wants to carry on lifts a region at that pc.
//!
//! The scalar registers live in stack slots rather than in the context, so `mem2reg` promotes them
//! into block parameters and every pass after it can see through them. The context is read on the
//! way in and written on the way out, which is also what makes a parked warp resumable.
//!
//! What this does not lift yet: the vector set, the floating point set, and `jalr` to an address
//! inside the same region. The first two are refused by the decoder, and the third leaves the
//! region. A kernel from the Glacier backend uses no `jalr` at all.

const std = @import("std");
const ir = @import("vulcan-ir");
const decode = @import("decode.zig");
const encode = @import("encode.zig");

const Function = ir.function.Function;
const Value = ir.function.Value;
const Block = ir.function.Block;
const Type = ir.types.Type;

pub const Error = decode.Error || std.mem.Allocator.Error || error{
    /// A branch or a jump named a word outside the program, or the program ran off its end. The
    /// hardware would fetch whatever the instruction memory holds there, so there is nothing
    /// honest to compile.
    Outside,
};

/// Why compiled code gave the warp back.
pub const Exit = enum(u32) {
    /// `ebreak`. The warp is finished.
    retired = 0,
    /// `ecall`. The code the register held is in `fault_code`.
    faulted = 1,
    /// A barrier. The warp waits for every other live warp on the core, and `pc` is the word
    /// after it.
    parked = 2,
    /// `jalr`. `pc` is where the register pointed.
    jumped = 3,
    /// The host asked the core to stop. Checked where control flows backwards, so a kernel that
    /// spins waiting for work can still be taken down.
    stopped = 4,
};

/// What a compiled kernel reads and writes. This layout is the agreement between lifted code and
/// whoever runs it, so it is `extern` and every offset in it is part of that agreement.
/// The widest warp any GC1 profile runs. The state carries room for all of them whatever the
/// core has, so one engine serves every profile.
pub const max_lanes = 16;

/// How many words one region covers at most.
///
/// A region is one function for the backend, and the backend's cost grows faster than the size of
/// what it is given: 770 words of one lane take seven seconds to compile, which no host can wait
/// for while a guest holds a kernel open. A long kernel becomes several regions instead, each
/// compiled once and kept, and a warp that steps out of one asks the host for the next. The number
/// is a balance: smaller regions compile faster, and a loop cut in half pays a host round trip
/// every turn.
///
/// A word this cannot lift is now refused when a warp reaches the region holding it rather than
/// when the kernel is first compiled, so a kernel can write memory before it is refused.
pub const region_words: u32 = 32;

pub const Warp = extern struct {
    /// The scalar registers. Register zero is here and never used, so an index needs no case of
    /// its own at run time.
    x: [32]u32 = @splat(0),
    /// Where the warp is. Written on the way out, so a caller can carry on from it.
    pc: u32 = 0,
    /// The code an `ecall` reported, which is the value of the register it named.
    fault_code: u32 = 0,
    /// The first access that went to an address nothing claims, and the pc that sent it. First
    /// wins, so the pair explains the run rather than whichever access failed last.
    fault_at: u32 = 0,
    fault_pc: u32 = 0,
    /// Set where an access went nowhere. The access retires with no writeback and the warp carries
    /// on, so without this a starved kernel reads as a correct one.
    bus_error: u32 = 0,
    /// Asked to stop, by anything other than the kernel.
    stop: u32 = 0,

    /// What the control registers answer. A kernel reads its own identity from these, which is how
    /// one image runs on every warp.
    hart_id: u32 = 0,
    hart_count: u32 = 0,
    workgroup_id: u32 = 0,
    grid_dim: u32 = 0,
    args: [encode.Csr.launch_args]u32 = @splat(0),
    /// One past the top of this warp's private scratch, which is what a kernel starts its stack
    /// at, and how much of it there is below that.
    scratch_top: u32 = 0,
    scratch_size: u32 = 0,
    /// The shared region of the core, or zero where the core was built without one.
    shared_at: u32 = 0,
    shared_size: u32 = 0,
    /// The cycle and instruction counters, which a kernel reads twice and subtracts.
    cycles: u64 = 0,
    instructions: u64 = 0,

    /// The bus, as one run of host memory and the fabric address it starts at. An access inside it
    /// is a load or a store of host memory and reaches nothing else.
    fabric: ?[*]u8 = null,
    fabric_at: u32 = 0,
    fabric_size: u32 = 0,
    /// The core private scratch, which nothing on the bus can reach.
    scratch: ?[*]u8 = null,
    scratch_at: u32 = 0,

    /// Whoever answers an address outside both runs above: a register block of the card, or
    /// nothing at all. It sets `bus_error` itself, because only it knows whether an address was
    /// claimed.
    away: ?*const fn (ctx: ?*anyopaque, at: u32, value: u32, kind: u32) callconv(.c) u64 = null,
    away_ctx: ?*anyopaque = null,

    /// The lane registers, one word per lane. A mask register is not laid out like this: its bits
    /// live in element zero, one bit per lane, which is where the core's mask unit reads them.
    v: [32][max_lanes]u32 = @splat(@splat(0)),
    /// Lanes the datapath has, and which of them this launch runs. Both are read by compiled code
    /// rather than baked into it, so one engine serves every profile, and they are two fields
    /// because a count and a mask are different things to get wrong.
    lane_count: u32 = 0,
    lane_mask: u32 = 0,
};

/// Where one lane register starts inside the warp state.
fn registerAt(which: u5) i64 {
    return @as(i64, @offsetOf(Warp, "v")) + @as(i64, which) * @sizeOf([max_lanes]u32);
}

/// The scalar arithmetic a lane word does, which is the same set the scalar side has.
fn laneArithmetic(op: decode.VectorOp) ?decode.Alu {
    return switch (op) {
        .add_vv, .add_vx, .add_vi => .add,
        .sub_vv, .sub_vx => .sub,
        .and_vv, .and_vi => .@"and",
        .or_vv, .or_vi => .@"or",
        .xor_vv => .xor,
        .sll_vv, .sll_vx, .sll_vi => .sll,
        .srl_vv, .srl_vx, .srl_vi => .srl,
        .sra_vv, .sra_vx, .sra_vi => .sra,
        .mul_vv, .mul_vx => .mul,
        .mulh_vv => .mul_high,
        .mulhu_vv => .mul_high_unsigned,
        .div_vv => .div,
        .divu_vv => .div_unsigned,
        .rem_vv => .rem,
        .remu_vv => .rem_unsigned,
        else => null,
    };
}

/// What `Warp.away` is told, and what it answers with.
///
/// One function rather than one per shape, so the lifted code holds one pointer and the host
/// switches on a word. The answer carries whether the address was claimed at all, because an
/// access that went nowhere retires with no writeback and a zero would be a wrong answer rather
/// than no answer.
pub const away = struct {
    /// How much of a word the access moves: zero a byte, one a half, two a word.
    pub const width_mask: u32 = 0b11;
    /// The access writes.
    pub const store: u32 = 1 << 2;
    /// The access reads, changes and writes back as one, with the operation in the two bits
    /// above. A read and modify is a store as well, so both bits are set.
    pub const atomic: u32 = 1 << 3;
    pub const atomic_shift = 4;
    /// Set in the answer where something claimed the address. Clear means nothing did.
    pub const answered: u64 = 1 << 32;
};

/// How a region is lifted. One of these per function, so nothing here is shared or global.
const Lifter = struct {
    allocator: std.mem.Allocator,
    func: *Function,
    program: []const u32,

    u8_t: Type,
    i8_t: Type,
    u16_t: Type,
    i16_t: Type,
    u32_t: Type,
    i32_t: Type,
    u64_t: Type,
    i64_t: Type,
    /// A lane read as a float. The core has no float register file: a lane is thirty two bits
    /// and what reads it decides whether they are a number or a float.
    f32_t: Type,
    f16_t: Type,
    /// Lanes the core this region is compiled for holds. A lane word is lowered once for every
    /// one of them, so the count is a fact about the card that the lift is given rather than a
    /// number it reads out of the warp at run time.
    lanes: u32,
    /// The first word of this region, which with `region_words` says what it covers.
    first: u32,
    bool_t: Type,
    ptr_t: Type,

    /// The context, which is the only argument a compiled region takes.
    ctx: Value,
    /// A stack slot per scalar register. Register zero has none, because it reads zero and takes
    /// no writes.
    slots: [32]Value,
    /// A value a lane operation carries across the lanes. One for the whole function, because
    /// the lanes of one word are lowered one after another and never nest.
    carry: Value,
    /// Two bytes of stack where a half is written and read back. A backend keeps a half widened
    /// in a register, so its bits are only a half where they land in memory.
    half_slot: Value,
    /// Which words start a block, and which words the region reaches at all.
    leaders: []bool,
    reached: []bool,
    blocks: []?Block,
    /// The block being written.
    at: Block,

    fn constant(self: *Lifter, value: u32) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .iconst = @as(i64, value) });
    }

    /// The same bits as a signed value, for the operations whose answer depends on the sign.
    fn signed(self: *Lifter, value: Value) Error!Value {
        return self.func.appendInst(self.at, self.i32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    }

    fn unsigned(self: *Lifter, value: Value) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    }

    fn get(self: *Lifter, which: decode.Reg) Error!Value {
        if (which == 0) return self.constant(0);
        return self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.slots[which] } });
    }

    fn set(self: *Lifter, which: decode.Reg, value: Value) Error!void {
        if (which == 0) return;
        try self.func.appendStore(self.at, value, self.slots[which]);
    }

    /// The address of a field of the context.
    fn field(self: *Lifter, comptime name: []const u8) Error!Value {
        const offset = @offsetOf(Warp, name);
        if (offset == 0) return self.ctx;
        return self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, offset);
    }

    fn read(self: *Lifter, comptime name: []const u8, ty: Type) Error!Value {
        return self.func.appendInst(self.at, ty, .{ .load = .{ .ptr = try self.field(name) } });
    }

    /// The stop a host asks for, read where control flows backwards.
    ///
    /// Volatile on purpose. Another thread writes this while the loop runs, and nothing inside the
    /// loop writes it, so a plain load is loop invariant: the optimiser lifts it out and the warp
    /// reads the flag once, before the loop it is meant to leave. A kernel that waits for work
    /// from its host then cannot be stopped at all, and whoever runs the machine waits for a warp
    /// that will never come back.
    fn readStop(self: *Lifter) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .load = .{
            .ptr = try self.field("stop"),
            .@"volatile" = true,
        } });
    }

    fn writeField(self: *Lifter, comptime name: []const u8, value: Value) Error!void {
        try self.func.appendStore(self.at, value, try self.field(name));
    }

    /// One register of the context's register array, by index.
    fn slotOf(self: *Lifter, which: usize) Error!Value {
        return self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, @as(i64, @offsetOf(Warp, "x")) + @as(i64, @intCast(which)) * 4);
    }

    fn widen(self: *Lifter, value: Value, ty: Type) Error!Value {
        return self.func.appendInst(self.at, ty, .{ .convert = .{ .value = value } });
    }

    fn compare(self: *Lifter, on: decode.Compare, a: Value, b: Value) Error!Value {
        return switch (on) {
            .eq => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = a, .rhs = b } }),
            .ne => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .ne, .lhs = a, .rhs = b } }),
            .lt_unsigned => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = a, .rhs = b } }),
            .ge_unsigned => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .ge, .lhs = a, .rhs = b } }),
            // A signed relation reads the sign off the operands, so the same bits go in with a
            // signed type on them.
            .lt => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{
                .op = .lt,
                .lhs = try self.signed(a),
                .rhs = try self.signed(b),
            } }),
            .ge => self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{
                .op = .ge,
                .lhs = try self.signed(a),
                .rhs = try self.signed(b),
            } }),
        };
    }

    /// The arithmetic of one word. The divides carry their own cases because a host divide traps
    /// where the core answers: the core gives all ones for a division by zero and wraps the most
    /// negative value over minus one, and both of those would stop the process here.
    /// A lane's bits read as a float.
    fn asFloat(self: *Lifter, value: Value) Error!Value {
        return self.func.appendInst(self.at, self.f32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
    }

    /// A float answer as the bits a lane holds, with a subnormal flushed to a zero of its own
    /// sign. The core's rounding block does that rather than carrying subnormals, so a model that
    /// kept them would answer a different low bit from the hardware on the smallest numbers.
    fn fromFloat(self: *Lifter, value: Value) Error!Value {
        const f = self.func;
        const bits = try f.appendInst(self.at, self.u32_t, .{ .unary = .{ .op = .reinterpret, .value = value } });
        const magnitude = try f.appendArithImm(self.at, self.u32_t, .bit_and, bits, 0x7fff_ffff);
        const sign = try f.appendArithImm(self.at, self.u32_t, .bit_and, bits, @as(i64, 0x8000_0000));
        // Smaller than the smallest number with a one in front of the point. A zero is smaller
        // than that too and keeps its own sign, which is the same answer.
        const smallest = try self.constant(0x0080_0000);
        const under = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = magnitude, .rhs = smallest } });
        return f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = under, .then = sign, .@"else" = bits } });
    }

    /// A half lane flushes a subnormal answer to a zero of the same sign, the way a float lane
    /// does. A half keeps its exponent from bit ten, so the smallest number with a one in front of
    /// the point is 0x0400.
    fn flushHalf(self: *Lifter, bits: Value) Error!Value {
        const f = self.func;
        const magnitude = try f.appendArithImm(self.at, self.u32_t, .bit_and, bits, 0x7fff);
        const sign = try f.appendArithImm(self.at, self.u32_t, .bit_and, bits, 0x8000);
        const smallest = try self.constant(0x0400);
        const under = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = magnitude, .rhs = smallest } });
        return f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = under, .then = sign, .@"else" = bits } });
    }
    /// One float lane word of two others.
    fn floatArithmetic(self: *Lifter, op: ir.function.BinOp, a: Value, b: Value) Error!Value {
        const answer = try self.func.appendInst(self.at, self.f32_t, .{ .arith = .{
            .op = op,
            .lhs = try self.asFloat(a),
            .rhs = try self.asFloat(b),
        } });
        return self.fromFloat(answer);
    }

    /// The smaller or the larger of two float lanes. A NaN on one side answers the other side,
    /// which is what the reference warp does and is not what a bare compare gives.
    fn floatPick(self: *Lifter, smaller: bool, a: Value, b: Value) Error!Value {
        const f = self.func;
        const x = try self.asFloat(a);
        const y = try self.asFloat(b);
        const x_is_number = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = x } });
        const y_is_number = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = y, .rhs = y } });
        const takes_x = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{
            .op = if (smaller) .lt else .gt,
            .lhs = x,
            .rhs = y,
        } });
        // Where both are numbers the relation decides; where one is not, the other side wins.
        const both = try f.appendInst(self.at, self.f32_t, .{ .select = .{ .cond = takes_x, .then = x, .@"else" = y } });
        const without_y = try f.appendInst(self.at, self.f32_t, .{ .select = .{ .cond = y_is_number, .then = both, .@"else" = x } });
        const picked = try f.appendInst(self.at, self.f32_t, .{ .select = .{ .cond = x_is_number, .then = without_y, .@"else" = y } });
        return self.fromFloat(picked);
    }
    fn arithmetic(self: *Lifter, op: decode.Alu, a: Value, b: Value) Error!Value {
        const f = self.func;
        return switch (op) {
            .add => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .add, .lhs = a, .rhs = b } }),
            .sub => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .sub, .lhs = a, .rhs = b } }),
            .xor => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = b } }),
            .@"or" => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } }),
            .@"and" => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } }),
            .sll => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .shl, .lhs = a, .rhs = try self.amount(b) } }),
            .srl => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .shr, .lhs = a, .rhs = try self.amount(b) } }),
            .sra => blk: {
                const shifted = try f.appendInst(self.at, self.i32_t, .{ .arith = .{
                    .op = .shr,
                    .lhs = try self.signed(a),
                    .rhs = try self.signed(try self.amount(b)),
                } });
                break :blk self.unsigned(shifted);
            },
            .slt => blk: {
                const less = try self.compare(.lt, a, b);
                break :blk self.widen(less, self.u32_t);
            },
            .slt_unsigned => blk: {
                const less = try self.compare(.lt_unsigned, a, b);
                break :blk self.widen(less, self.u32_t);
            },
            .mul => f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } }),
            .mul_high => self.high(try self.widen(try self.signed(a), self.i64_t), try self.widen(try self.signed(b), self.i64_t), self.i64_t),
            .mul_high_unsigned => self.high(try self.widen(a, self.u64_t), try self.widen(b, self.u64_t), self.u64_t),
            // One side signed and the other not, which is the one multiply the architecture has
            // that no language does.
            .mul_high_signed_unsigned => blk: {
                // One side signed and the other not. The unsigned side widens with zeros first and
                // then carries a signed type, because the product is taken as signed.
                const wide = try self.widen(b, self.u64_t);
                const as_signed = try self.func.appendInst(self.at, self.i64_t, .{ .unary = .{ .op = .reinterpret, .value = wide } });
                break :blk self.high(try self.widen(try self.signed(a), self.i64_t), as_signed, self.i64_t);
            },
            .div => self.divide(a, b, true, false),
            .div_unsigned => self.divide(a, b, false, false),
            .rem => self.divide(a, b, true, true),
            .rem_unsigned => self.divide(a, b, false, true),
        };
    }

    /// A shift takes the low five bits of its amount and ignores the rest. The host agrees today
    /// and the IR need not, so the masking is here rather than assumed.
    fn amount(self: *Lifter, value: Value) Error!Value {
        return self.func.appendArithImm(self.at, self.u32_t, .bit_and, value, 31);
    }

    /// The top half of the full product of two widened values.
    fn high(self: *Lifter, a: Value, b: Value, ty: Type) Error!Value {
        const whole = try self.func.appendInst(self.at, ty, .{ .arith = .{ .op = .mul, .lhs = a, .rhs = b } });
        const shifted = try self.func.appendArithImm(self.at, ty, .shr, whole, 32);
        const narrowed = try self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = shifted } });
        return narrowed;
    }

    /// A divide or a remainder, with the two answers the architecture states and a host divide
    /// would trap on.
    fn divide(self: *Lifter, a: Value, b: Value, is_signed: bool, remainder: bool) Error!Value {
        const f = self.func;
        const zero = try self.constant(0);
        const by_zero = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = b, .rhs = zero } });

        // A divisor of one wherever the real one would trap, so the division itself is always
        // safe and the answer is chosen after it.
        const one = try self.constant(1);
        var safe = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = by_zero, .then = one, .@"else" = b } });

        // The two conditions stay apart rather than being joined with a bitwise and, because an
        // and of two booleans is arithmetic on a type whose width is the backend's to choose.
        var lowest_over: ?struct { at_lowest: Value, by_minus_one: Value } = null;
        if (is_signed) {
            const lowest = try self.constant(0x8000_0000);
            const minus_one = try self.constant(0xffff_ffff);
            const at_lowest = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = a, .rhs = lowest } });
            const by_minus_one = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = b, .rhs = minus_one } });
            lowest_over = .{ .at_lowest = at_lowest, .by_minus_one = by_minus_one };
            const guarded = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = by_minus_one, .then = one, .@"else" = safe } });
            safe = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = at_lowest, .then = guarded, .@"else" = safe } });
        }

        const op: ir.function.BinOp = if (remainder) .rem else .div;
        const quotient = if (is_signed) blk: {
            const got = try f.appendInst(self.at, self.i32_t, .{ .arith = .{
                .op = op,
                .lhs = try self.signed(a),
                .rhs = try self.signed(safe),
            } });
            break :blk try self.unsigned(got);
        } else try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = op, .lhs = a, .rhs = safe } });

        // Division by zero gives all ones, and a remainder by zero gives the dividend back.
        const nothing = if (remainder) a else try self.constant(0xffff_ffff);
        var answer = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = by_zero, .then = nothing, .@"else" = quotient } });

        if (lowest_over) |both| {
            // The most negative value over minus one wraps to itself, and its remainder is zero.
            const over = if (remainder) try self.constant(0) else try self.constant(0x8000_0000);
            const guarded = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = both.by_minus_one, .then = over, .@"else" = answer } });
            answer = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = both.at_lowest, .then = guarded, .@"else" = answer } });
        }
        return answer;
    }

    /// The control register a word names, read out of the context. A register the core does not
    /// answer is refused here, because a kernel reading one is relying on hardware this cannot
    /// stand in for.
    fn control(self: *Lifter, which: u12) Error!Value {
        const csr = encode.Csr;
        const first = csr.launchArg(0).?;
        if (which >= first and which < first + csr.launch_args) {
            const index = which - first;
            const at = try self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, @as(i64, @offsetOf(Warp, "args")) + @as(i64, index) * 4);
            return self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = at } });
        }

        return switch (which) {
            csr.mhartid => self.read("hart_id", self.u32_t),
            csr.hart_count => self.read("hart_count", self.u32_t),
            csr.workgroup_id => self.read("workgroup_id", self.u32_t),
            csr.grid_dim => self.read("grid_dim", self.u32_t),
            mscratch => self.read("scratch_top", self.u32_t),
            scratchsize => self.read("scratch_size", self.u32_t),
            shmaddr => self.read("shared_at", self.u32_t),
            shmsize => self.read("shared_size", self.u32_t),
            mcycle => self.half("cycles", false),
            mcycleh => self.half("cycles", true),
            minstret => self.half("instructions", false),
            minstreth => self.half("instructions", true),
            else => error.Illegal,
        };
    }

    /// One half of a counter the context holds whole.
    fn half(self: *Lifter, comptime name: []const u8, top: bool) Error!Value {
        const whole = try self.read(name, self.u64_t);
        const shifted = if (top)
            try self.func.appendArithImm(self.at, self.u64_t, .shr, whole, 32)
        else
            whole;
        return self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = shifted } });
    }

    /// The word a jump or a branch names, as a word index into the program.
    fn target(self: *Lifter, pc: u32, offset: i32) Error!u32 {
        const to = @as(u32, @bitCast(@as(i32, @bitCast(pc)) +% offset));
        if (to % 4 != 0) return error.Outside;
        const word = to / 4;
        if (word >= self.program.len) return error.Outside;
        return word;
    }

    /// `mscratch` sits in the standard machine range rather than in the custom read only one, so it is
    /// named here beside the rest and not in `encode.zig`'s read only list.
    const mscratch: u12 = 0x340;
    const scratchsize: u12 = 0xFD2;
    const shmaddr: u12 = 0xFD0;
    const shmsize: u12 = 0xFD1;
    const mcycle: u12 = 0xB00;
    const mcycleh: u12 = 0xB80;
    /// The instruction counter is a hardware performance counter and not `minstret`, because the core
    /// counts what the issue arbiter committed across the whole core.
    const minstret: u12 = 0xB03;
    const minstreth: u12 = 0xB83;

    /// The host type one access moves, and the type the bits widen from. A byte load of a negative
    /// value has to come back sign extended, so the narrow type carries the signedness.
    fn narrow(self: *Lifter, width: decode.Width, is_signed: bool) Type {
        return switch (width) {
            .byte => if (is_signed) self.i8_t else self.u8_t,
            .half => if (is_signed) self.i16_t else self.u16_t,
            .word => self.u32_t,
        };
    }

    fn widthWord(width: decode.Width) u32 {
        return switch (width) {
            .byte => 0,
            .half => 1,
            .word => 2,
        };
    }

    /// Where in host memory a fabric address lands, and whether it lands there at all. One of these
    /// per run of memory the context holds.
    const Reach = struct {
        inside: Value,
        offset: Value,
    };

    /// Whether an address is inside one of the context's runs of memory, and how far into it. The
    /// subtraction wraps, so an address below the start of a run comes out as a huge offset and fails
    /// the length test rather than passing it.
    fn within(self: *Lifter, at: Value, comptime base: []const u8, comptime size: []const u8) Error!Reach {
        const start = try self.read(base, self.u32_t);
        const length = try self.read(size, self.u32_t);
        const offset = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .sub, .lhs = at, .rhs = start } });
        const inside = try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = offset, .rhs = length } });
        return .{ .inside = inside, .offset = offset };
    }

    /// The host address of an offset into one of the runs.
    fn hostAt(self: *Lifter, comptime base: []const u8, offset: Value) Error!Value {
        const start = try self.read(base, self.ptr_t);
        const wide = try self.widen(offset, self.u64_t);
        return self.func.appendInst(self.at, self.ptr_t, .{ .arith = .{ .op = .add, .lhs = start, .rhs = wide } });
    }

    /// What an access that reached neither run is answered with, and whether it was answered at all.
    /// The host sets the bus error itself, because only it knows which addresses the card claims.
    fn asked(self: *Lifter, at: Value, value: Value, kind: u32) Error!struct { value: Value, took: Value } {
        const handler = try self.read("away", self.ptr_t);
        const handler_ctx = try self.read("away_ctx", self.ptr_t);
        const kind_value = try self.func.appendInst(self.at, self.u32_t, .{ .iconst = @as(i64, kind) });
        const answer = try self.func.appendCallIndirect(self.at, self.u64_t, handler, &.{ handler_ctx, at, value, kind_value });

        // The flag rides above the value, so a shift brings it down rather than a mask of a bit no
        // immediate form holds.
        const flag = try self.func.appendArithImm(self.at, self.u64_t, .shr, answer, 32);
        const zero = try self.func.appendInst(self.at, self.u64_t, .{ .iconst = 0 });
        const took = try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .ne, .lhs = flag, .rhs = zero } });
        const low = try self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = answer } });
        return .{ .value = low, .took = took };
    }

    /// Bits that came back from the host, extended the way the access asks. The host answers with the
    /// bytes it read and nothing about their sign, so the extension is this side's to do.
    fn extend(self: *Lifter, bits: Value, width: decode.Width, is_signed: bool) Error!Value {
        if (width == .word) return bits;
        const cut = try self.func.appendInst(self.at, self.narrow(width, is_signed), .{ .convert = .{ .value = bits } });
        return self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = cut } });
    }

    /// A load, lowered as a run of host memory where it can be and a call where it cannot.
    fn lower(self: *Lifter, from: decode.Instruction) Error!void {
        const one = from.load;
        const old = try self.get(one.rd);
        const base = try self.get(one.rs1);
        const at = try self.func.appendArithImm(self.at, self.u32_t, .add, base, one.offset);
        const ty = self.narrow(one.width, one.signed);

        const fabric = try self.func.appendBlock();
        const other = try self.func.appendBlock();
        const scratch = try self.func.appendBlock();
        const away_block = try self.func.appendBlock();
        const merge = try self.func.appendBlock();

        const in_fabric = try self.within(at, "fabric_at", "fabric_size");
        try self.func.appendIf(self.at, in_fabric.inside, .{ .target = fabric }, .{ .target = other });

        self.at = fabric;
        const fabric_at = try self.hostAt("fabric", in_fabric.offset);
        const fabric_bits = try self.func.appendInst(self.at, ty, .{ .load = .{ .ptr = fabric_at } });
        try self.set(one.rd, try self.widen(fabric_bits, self.u32_t));
        try self.func.setJump(self.at, merge, &.{});

        self.at = other;
        const in_scratch = try self.within(at, "scratch_at", "scratch_size");
        try self.func.appendIf(self.at, in_scratch.inside, .{ .target = scratch }, .{ .target = away_block });

        self.at = scratch;
        const scratch_at = try self.hostAt("scratch", in_scratch.offset);
        const scratch_bits = try self.func.appendInst(self.at, ty, .{ .load = .{ .ptr = scratch_at } });
        try self.set(one.rd, try self.widen(scratch_bits, self.u32_t));
        try self.func.setJump(self.at, merge, &.{});

        self.at = away_block;
        const nothing = try self.constant(0);
        const answer = try self.asked(at, nothing, widthWord(one.width));
        const extended = try self.extend(answer.value, one.width, one.signed);
        // An access that reached nothing retires with no writeback, so the register keeps what it had
        // rather than taking a zero that looks like an answer.
        const kept = try self.func.appendInst(self.at, self.u32_t, .{ .select = .{
            .cond = answer.took,
            .then = extended,
            .@"else" = old,
        } });
        try self.set(one.rd, kept);
        try self.func.setJump(self.at, merge, &.{});

        self.at = merge;
    }

    /// A store, the same shape as a load with nothing to bring back.
    fn lowerStore(self: *Lifter, from: decode.Instruction) Error!void {
        const one = from.store;
        const base = try self.get(one.rs1);
        const value = try self.get(one.rs2);
        const at = try self.func.appendArithImm(self.at, self.u32_t, .add, base, one.offset);
        const ty = self.narrow(one.width, false);
        const cut = if (one.width == .word)
            value
        else
            try self.func.appendInst(self.at, ty, .{ .convert = .{ .value = value } });

        const fabric = try self.func.appendBlock();
        const other = try self.func.appendBlock();
        const scratch = try self.func.appendBlock();
        const away_block = try self.func.appendBlock();
        const merge = try self.func.appendBlock();

        const in_fabric = try self.within(at, "fabric_at", "fabric_size");
        try self.func.appendIf(self.at, in_fabric.inside, .{ .target = fabric }, .{ .target = other });

        self.at = fabric;
        try self.func.appendStore(self.at, cut, try self.hostAt("fabric", in_fabric.offset));
        try self.func.setJump(self.at, merge, &.{});

        self.at = other;
        const in_scratch = try self.within(at, "scratch_at", "scratch_size");
        try self.func.appendIf(self.at, in_scratch.inside, .{ .target = scratch }, .{ .target = away_block });

        self.at = scratch;
        try self.func.appendStore(self.at, cut, try self.hostAt("scratch", in_scratch.offset));
        try self.func.setJump(self.at, merge, &.{});

        self.at = away_block;
        _ = try self.asked(at, value, widthWord(one.width) | away.store);
        try self.func.setJump(self.at, merge, &.{});

        self.at = merge;
    }

    /// A read, change and write of one word, held together.
    ///
    /// Always the host rather than a run of memory inline, because neither host backend lowers an
    /// atomic yet. It is also the shape the hardware has: an atomic works against a register of the
    /// card as well as against memory, so one path serves both.
    fn lowerAtomic(self: *Lifter, from: decode.Instruction) Error!void {
        const one = from.atomic;
        const old = try self.get(one.rd);
        const at = try self.get(one.rs1);
        const value = try self.get(one.rs2);

        const kind = widthWord(.word) | away.store | away.atomic |
            (@as(u32, @backingInt(one.op)) << away.atomic_shift);
        const answer = try self.asked(at, value, kind);
        try self.set(one.rd, try self.func.appendInst(self.at, self.u32_t, .{ .select = .{
            .cond = answer.took,
            .then = answer.value,
            .@"else" = old,
        } }));
    }

    /// Give the warp back: write where it is, write every register it holds, and return why.
    /// One lane register, by index, as an address.
    fn laneAt(self: *Lifter, which: u5, lane: Value) Error!Value {
        const start = try self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, registerAt(which));
        const wide = try self.widen(lane, self.u64_t);
        const bytes = try self.func.appendArithImm(self.at, self.u64_t, .mul, wide, 4);
        return self.func.appendInst(self.at, self.ptr_t, .{ .arith = .{ .op = .add, .lhs = start, .rhs = bytes } });
    }

    /// Element zero of a lane register, which is also where a mask register keeps its bits.
    fn elementZero(self: *Lifter, which: u5) Error!Value {
        return self.func.appendArithImm(self.at, self.ptr_t, .add, self.ctx, registerAt(which));
    }

    fn laneLoad(self: *Lifter, which: u5, lane: Value) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = try self.laneAt(which, lane) } });
    }

    fn maskLoad(self: *Lifter, which: u5) Error!Value {
        return self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = try self.elementZero(which) } });
    }

    fn maskStore(self: *Lifter, which: u5, bits: Value) Error!void {
        try self.func.appendStore(self.at, bits, try self.elementZero(which));
    }

    /// `value != 0`, which is how a bit test becomes a condition.
    fn any(self: *Lifter, value: Value) Error!Value {
        const zero = try self.constant(0);
        return self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .ne, .lhs = value, .rhs = zero } });
    }

    /// One lane of the warp, as straight line code rather than a turn of a loop.
    ///
    /// A lane word is lowered once for every lane the core has. The core's lane count is a fact
    /// about the card and not a number a kernel chooses, so the lift is given it and every lane
    /// index is a constant here. That is what lets the optimiser keep a lane's word in a register
    /// instead of carrying it to lane memory and back four times, which is most of what a kernel
    /// of vector words used to cost.
    const Lane = struct {
        /// The lane this copy is for, and its bit.
        index: Value,
        bit: Value,
        /// Whether this lane takes the answer: the launch runs it, and v0 names it where the word
        /// is predicated.
        active: Value,
    };

    fn laneOne(self: *Lifter, masked: bool, which: u32) Error!Lane {
        const index = try self.constant(which);
        const bit = try self.constant(@as(u32, 1) << @intCast(which));
        var gate = try self.read("lane_mask", self.u32_t);
        if (masked) {
            const v0 = try self.maskLoad(0);
            gate = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = gate, .rhs = v0 } });
        }
        const taken = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = gate, .rhs = bit } });
        return .{ .index = index, .bit = bit, .active = try self.any(taken) };
    }

    /// The second operand of a lane word, which is the one thing its three forms differ by.
    fn laneOther(self: *Lifter, one: decode.Vector, index: Value) Error!Value {
        return switch (one.op) {
            .add_vx, .sub_vx, .sll_vx, .srl_vx, .sra_vx, .mul_vx, .mv_v_x, .merge_vxm, .mseq_vx, .msne_vx => self.get(one.rs),
            .add_vi, .and_vi, .or_vi, .sll_vi, .srl_vi, .sra_vi, .mv_v_i, .merge_vim, .mseq_vi => self.constant(@bitCast(one.imm)),
            // A convert has no second operand: the field that would name one picked the convert.
            .fcvt_f_x_v, .fcvt_f_xu_v, .fcvt_rtz_xu_f_v => self.constant(0),
            else => self.laneLoad(one.vs1, index),
        };
    }

    /// A lane word. Every lane runs it, and the lanes the launch left out keep what they had.
    fn lowerVector(self: *Lifter, one: decode.Vector) Error!void {
        switch (one.op) {
            .mandn_mm, .mand_mm, .mor_mm, .mxor_mm, .morn_mm, .mnand_mm, .mnor_mm, .mxnor_mm => return self.lowerMaskLogical(one),
            .mv_x_s => return self.set(one.rs, try self.maskLoad(one.vs2)),
            .mv_s_x => return self.maskStore(one.vd, try self.get(one.rs)),
            .cpop_m, .first_m => return self.lowerMaskRead(one),
            .redor_vs => return self.lowerReduceOr(one),
            .le32, .se32, .luxei32, .suxei32 => return self.lowerLaneMemory(one),
            .mseq_vv, .mseq_vx, .mseq_vi, .msne_vv, .msne_vx, .msltu_vv, .mslt_vv, .msleu_vv, .msle_vv, .mfeq_vv, .mfle_vv, .mflt_vv => return self.lowerLaneCompare(one),
            else => {},
        }

        var which: u32 = 0;
        while (which < self.lanes) : (which += 1) {
            const loop = try self.laneOne(one.masked, which);
            const old = try self.laneLoad(one.vd, loop.index);
            const other = try self.laneOther(one, loop.index);

            const answer = switch (one.op) {
                .mv_v_v, .mv_v_x, .mv_v_i => other,
                .merge_vvm, .merge_vxm, .merge_vim => blk: {
                    // v0 is an operand here rather than a predicate, so every lane the warp holds
                    // takes one side or the other.
                    const v0 = try self.maskLoad(0);
                    const picked = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = v0, .rhs = loop.bit } });
                    const from_one = try self.any(picked);
                    const from_two = try self.laneLoad(one.vs2, loop.index);
                    break :blk try self.func.appendInst(self.at, self.u32_t, .{ .select = .{
                        .cond = from_one,
                        .then = other,
                        .@"else" = from_two,
                    } });
                },
                .id_v => loop.index,
                .fadd_vv, .fsub_vv, .fmul_vv, .fdiv_vv => blk: {
                    const base = try self.laneLoad(one.vs2, loop.index);
                    break :blk try self.floatArithmetic(switch (one.op) {
                        .fadd_vv => .add,
                        .fsub_vv => .sub,
                        .fmul_vv => .mul,
                        else => .div,
                    }, base, other);
                },
                .fmin_vv, .fmax_vv => blk: {
                    const base = try self.laneLoad(one.vs2, loop.index);
                    break :blk try self.floatPick(one.op == .fmin_vv, base, other);
                },
                // A whole lane of integer to a lane of float. The field that names a second operand
                // elsewhere picked which convert this is, so there is no second operand to read.
                .fcvt_rtz_xu_f_v => blk: {
                    const f = self.func;
                    const x = try self.asFloat(try self.laneLoad(one.vs2, loop.index));
                    const raw = try f.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = x } });
                    // Toward zero, and a value the range cannot hold saturates rather than wrapping.
                    // A NaN is in no relation at all, so it is caught by its own inequality and
                    // answers the largest number, which is what the reference warp answers.
                    const zero = try f.appendInst(self.at, self.f32_t, .{ .fconst = 0.0 });
                    const over = try f.appendInst(self.at, self.f32_t, .{ .fconst = 4294967296.0 });
                    const biggest = try self.constant(0xffff_ffff);
                    const low = try self.constant(0);
                    const is_number = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = x, .rhs = x } });
                    const positive = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .gt, .lhs = x, .rhs = zero } });
                    const inside = try f.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = x, .rhs = over } });
                    const held = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = inside, .then = raw, .@"else" = biggest } });
                    const above = try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = positive, .then = held, .@"else" = low } });
                    break :blk try f.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = is_number, .then = above, .@"else" = biggest } });
                },
                // A half lives in the low sixteen bits of a lane. A backend keeps a half widened
                // in a register, so the only place its bits are really sixteen wide is memory:
                // these go through two bytes of stack rather than through a bitcast.
                .fcvt_f_h_v => blk: {
                    const cut = try self.func.appendInst(self.at, self.u16_t, .{ .convert = .{ .value = try self.laneLoad(one.vs2, loop.index) } });
                    try self.func.appendStore(self.at, cut, self.half_slot);
                    const small = try self.func.appendInst(self.at, self.f16_t, .{ .load = .{ .ptr = self.half_slot } });
                    const wide = try self.func.appendInst(self.at, self.f32_t, .{ .convert = .{ .value = small } });
                    break :blk try self.fromFloat(wide);
                },
                .fcvt_h_f_w, .fcvt_h_x_v => blk: {
                    const base = try self.laneLoad(one.vs2, loop.index);
                    const from = if (one.op == .fcvt_h_x_v)
                        try self.func.appendInst(self.at, self.f32_t, .{ .convert = .{ .value = try self.signed(base) } })
                    else
                        try self.asFloat(base);
                    const small = try self.func.appendInst(self.at, self.f16_t, .{ .convert = .{ .value = from } });
                    try self.func.appendStore(self.at, small, self.half_slot);
                    const bits = try self.func.appendInst(self.at, self.u16_t, .{ .load = .{ .ptr = self.half_slot } });
                    const wide = try self.func.appendInst(self.at, self.u32_t, .{ .convert = .{ .value = bits } });
                    break :blk try self.flushHalf(wide);
                },
                .fcvt_f_x_v, .fcvt_f_xu_v => blk: {
                    const base = try self.laneLoad(one.vs2, loop.index);
                    const from = if (one.op == .fcvt_f_x_v) try self.signed(base) else base;
                    const answer = try self.func.appendInst(self.at, self.f32_t, .{ .convert = .{ .value = from } });
                    break :blk try self.fromFloat(answer);
                },
                else => blk: {
                    const base = try self.laneLoad(one.vs2, loop.index);
                    break :blk try self.arithmetic(laneArithmetic(one.op) orelse return error.Illegal, base, other);
                },
            };

            const kept = try self.func.appendInst(self.at, self.u32_t, .{ .select = .{
                .cond = loop.active,
                .then = answer,
                .@"else" = old,
            } });
            try self.func.appendStore(self.at, kept, try self.laneAt(one.vd, loop.index));
        }
    }

    /// A compare, which writes a mask register rather than lanes. The bits of a lane the launch
    /// left out are clear, so a mask never names a lane that does not run.
    fn lowerLaneCompare(self: *Lifter, one: decode.Vector) Error!void {
        try self.func.appendStore(self.at, try self.constant(0), self.carry);
        var which: u32 = 0;
        while (which < self.lanes) : (which += 1) {
            const loop = try self.laneOne(false, which);
            const a = try self.laneLoad(one.vs2, loop.index);
            const b = try self.laneOther(one, loop.index);

            // A relation the architecture has and the lift does not: `a <= b` is `b >= a`.
            const holds = switch (one.op) {
                .mseq_vv, .mseq_vx, .mseq_vi => try self.compare(.eq, a, b),
                .msne_vv, .msne_vx => try self.compare(.ne, a, b),
                .msltu_vv => try self.compare(.lt_unsigned, a, b),
                .mslt_vv => try self.compare(.lt, a, b),
                .msleu_vv => try self.compare(.ge_unsigned, b, a),
                .msle_vv => try self.compare(.ge, b, a),
                // The float relations read the same two lanes as floats. Ordered, so a NaN answers
                // false against everything, which is what an equality against itself tests for.
                .mfeq_vv => try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = try self.asFloat(a), .rhs = try self.asFloat(b) } }),
                .mfle_vv => try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .le, .lhs = try self.asFloat(a), .rhs = try self.asFloat(b) } }),
                .mflt_vv => try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .lt, .lhs = try self.asFloat(a), .rhs = try self.asFloat(b) } }),
                else => return error.Illegal,
            };

            const both = try self.func.appendInst(self.at, self.bool_t, .{ .select = .{
                .cond = loop.active,
                .then = holds,
                .@"else" = try self.func.appendInst(self.at, self.bool_t, .{ .iconst = 0 }),
            } });
            const zero = try self.constant(0);
            const add = try self.func.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = both, .then = loop.bit, .@"else" = zero } });
            const held = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } });
            const joined = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = held, .rhs = add } });
            try self.func.appendStore(self.at, joined, self.carry);
        }

        const bits = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } });
        try self.maskStore(one.vd, bits);
    }

    /// The mask logicals, which read and write whole mask words and need no lane loop. The answer
    /// is cut to the lanes the core has, so a complement does not invent bits above them.
    fn lowerMaskLogical(self: *Lifter, one: decode.Vector) Error!void {
        const a = try self.maskLoad(one.vs2);
        const b = try self.maskLoad(one.vs1);
        const f = self.func;

        const joined = switch (one.op) {
            .mandn_mm => try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = try self.not(b) } }),
            .mand_mm => try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } }),
            .mor_mm => try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } }),
            .mxor_mm => try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = b } }),
            .morn_mm => try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = try self.not(b) } }),
            .mnand_mm => try self.not(try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b } })),
            .mnor_mm => try self.not(try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = a, .rhs = b } })),
            .mxnor_mm => try self.not(try f.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = a, .rhs = b } })),
            else => return error.Illegal,
        };
        try self.maskStore(one.vd, try self.toLanes(joined));
    }

    /// `~value`, as the exclusive or with every bit.
    fn not(self: *Lifter, value: Value) Error!Value {
        const ones = try self.constant(0xFFFF_FFFF);
        return self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_xor, .lhs = value, .rhs = ones } });
    }

    /// Cut to the lanes the core has, which is how a mask stays a mask.
    fn toLanes(self: *Lifter, bits: Value) Error!Value {
        const count = try self.constant(self.lanes);
        const one = try self.constant(1);
        const top = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .shl, .lhs = one, .rhs = count } });
        const room = try self.func.appendArithImm(self.at, self.u32_t, .sub, top, 1);
        return self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = bits, .rhs = room } });
    }

    /// `vcpop.m` and `vfirst.m`: a mask back to the scalar side, which is how a kernel with one
    /// program counter leaves a loop when no lane is left.
    fn lowerMaskRead(self: *Lifter, one: decode.Vector) Error!void {
        const start: i64 = if (one.op == .cpop_m) 0 else -1;
        try self.func.appendStore(self.at, try self.constant(@bitCast(@as(i32, @intCast(start)))), self.carry);
        const bits = try self.maskLoad(one.vs2);

        var which: u32 = 0;
        while (which < self.lanes) : (which += 1) {
            const loop = try self.laneOne(false, which);
            const named = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_and, .lhs = bits, .rhs = loop.bit } });
            const holds = try self.any(named);
            const held = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } });

            const answer = if (one.op == .cpop_m) blk: {
                const more = try self.func.appendArithImm(self.at, self.u32_t, .add, held, 1);
                break :blk try self.func.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = holds, .then = more, .@"else" = held } });
            } else blk: {
                // The lowest set bit wins, so a later lane never replaces an answer already found.
                const first = try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .eq, .lhs = held, .rhs = try self.constant(0xFFFF_FFFF) } });
                const take = try self.func.appendInst(self.at, self.bool_t, .{ .select = .{
                    .cond = holds,
                    .then = first,
                    .@"else" = try self.func.appendInst(self.at, self.bool_t, .{ .iconst = 0 }),
                } });
                break :blk try self.func.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = take, .then = loop.index, .@"else" = held } });
            };
            try self.func.appendStore(self.at, answer, self.carry);
        }

        try self.set(one.rs, try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } }));
    }

    /// `vredor.vs`: every lane of one register ored together, onto lane zero of another.
    fn lowerReduceOr(self: *Lifter, one: decode.Vector) Error!void {
        try self.func.appendStore(self.at, try self.maskLoad(one.vs1), self.carry);
        var which: u32 = 0;
        while (which < self.lanes) : (which += 1) {
            const loop = try self.laneOne(false, which);
            const value = try self.laneLoad(one.vs2, loop.index);
            const held = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } });
            const joined = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .bit_or, .lhs = held, .rhs = value } });
            const kept = try self.func.appendInst(self.at, self.u32_t, .{ .select = .{ .cond = loop.active, .then = joined, .@"else" = held } });
            try self.func.appendStore(self.at, kept, self.carry);
        }

        const answer = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.carry } });
        try self.maskStore(one.vd, answer);
    }

    /// A word per lane, from one address upwards or from an address the lane carries. A lane the
    /// launch left out makes no access at all, so it cannot fault for a lane that is not running.
    fn lowerLaneMemory(self: *Lifter, one: decode.Vector) Error!void {
        const is_store = one.op == .se32 or one.op == .suxei32;
        const indexed = one.op == .luxei32 or one.op == .suxei32;
        const base = try self.get(one.rs);

        var which: u32 = 0;
        while (which < self.lanes) : (which += 1) {
            const loop = try self.laneOne(one.masked, which);
            // A lane the launch left out jumps past its access rather than making one, which is
            // what keeps it from faulting for work it was never given.
            const work = try self.func.appendBlock();
            const after = try self.func.appendBlock();
            try self.func.appendIf(self.at, loop.active, .{ .target = work }, .{ .target = after });

            self.at = work;
            const offset = if (indexed)
                try self.laneLoad(one.vs2, loop.index)
            else
                try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .mul, .lhs = loop.index, .rhs = try self.constant(4) } });
            const sum = try self.func.appendInst(self.at, self.u32_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = offset } });
            // A word access ignores the low two bits of its address rather than trapping on them.
            const at = try self.func.appendArithImm(self.at, self.u32_t, .bit_and, sum, ~@as(i64, 3));

            if (is_store) {
                try self.wordStore(at, try self.laneLoad(one.vd, loop.index));
            } else {
                const old = try self.laneLoad(one.vd, loop.index);
                const value = try self.wordLoad(at, old);
                try self.func.appendStore(self.at, value, try self.laneAt(one.vd, loop.index));
            }

            try self.func.setJump(self.at, after, &.{});
            self.at = after;
        }
    }

    /// One word off the bus, with the two runs of host memory taken directly and anything else
    /// asked of the host. `old` is what the register keeps where nothing claimed the address.
    fn wordLoad(self: *Lifter, at: Value, old: Value) Error!Value {
        const fabric = try self.func.appendBlock();
        const other = try self.func.appendBlock();
        const scratch = try self.func.appendBlock();
        const away_block = try self.func.appendBlock();
        const merge = try self.func.appendBlock();
        const answer = try self.func.appendBlockParam(merge, self.u32_t);

        const in_fabric = try self.within(at, "fabric_at", "fabric_size");
        try self.func.appendIf(self.at, in_fabric.inside, .{ .target = fabric }, .{ .target = other });

        self.at = fabric;
        const from_fabric = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = try self.hostAt("fabric", in_fabric.offset) } });
        try self.func.setJump(self.at, merge, &.{from_fabric});

        self.at = other;
        const in_scratch = try self.within(at, "scratch_at", "scratch_size");
        try self.func.appendIf(self.at, in_scratch.inside, .{ .target = scratch }, .{ .target = away_block });

        self.at = scratch;
        const from_scratch = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = try self.hostAt("scratch", in_scratch.offset) } });
        try self.func.setJump(self.at, merge, &.{from_scratch});

        self.at = away_block;
        const answered = try self.asked(at, try self.constant(0), widthWord(.word));
        const kept = try self.func.appendInst(self.at, self.u32_t, .{ .select = .{
            .cond = answered.took,
            .then = answered.value,
            .@"else" = old,
        } });
        try self.func.setJump(self.at, merge, &.{kept});

        self.at = merge;
        return answer;
    }

    /// One word onto the bus, the same shape with nothing to bring back.
    fn wordStore(self: *Lifter, at: Value, value: Value) Error!void {
        const fabric = try self.func.appendBlock();
        const other = try self.func.appendBlock();
        const scratch = try self.func.appendBlock();
        const away_block = try self.func.appendBlock();
        const merge = try self.func.appendBlock();

        const in_fabric = try self.within(at, "fabric_at", "fabric_size");
        try self.func.appendIf(self.at, in_fabric.inside, .{ .target = fabric }, .{ .target = other });

        self.at = fabric;
        try self.func.appendStore(self.at, value, try self.hostAt("fabric", in_fabric.offset));
        try self.func.setJump(self.at, merge, &.{});

        self.at = other;
        const in_scratch = try self.within(at, "scratch_at", "scratch_size");
        try self.func.appendIf(self.at, in_scratch.inside, .{ .target = scratch }, .{ .target = away_block });

        self.at = scratch;
        try self.func.appendStore(self.at, value, try self.hostAt("scratch", in_scratch.offset));
        try self.func.setJump(self.at, merge, &.{});

        self.at = away_block;
        _ = try self.asked(at, value, widthWord(.word) | away.store);
        try self.func.setJump(self.at, merge, &.{});

        self.at = merge;
    }

    /// Whether a word is one this region covers. A word outside it is somewhere the warp leaves
    /// for, not somewhere this function goes.
    fn covers(self: *const Lifter, at: u32) bool {
        if (at < self.first or at >= self.program.len) return false;
        return at - self.first < region_words;
    }

    /// The block an edge goes to: the word's own block where this region covers it, or a block
    /// that leaves for the host to enter the region the word starts.
    fn edgeTo(self: *Lifter, at: u32) Error!Block {
        if (self.covers(at)) {
            if (self.blocks[at]) |block| return block;
        }
        const out = try self.func.appendBlock();
        const held = self.at;
        self.at = out;
        try self.leave(.jumped, try self.constant(at * 4));
        self.at = held;
        return out;
    }

    fn leave(self: *Lifter, why: Exit, pc: Value) Error!void {
        try self.writeField("pc", pc);
        for (1..self.slots.len) |which| {
            const value = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.slots[which] } });
            try self.func.appendStore(self.at, value, try self.slotOf(which));
        }
        const reason = try self.func.appendInst(self.at, self.u32_t, .{ .iconst = @as(i64, @backingInt(why)) });
        self.func.setTerminator(self.at, .{ .ret = ir.function.Ret.one(reason) });
    }

    /// A word that starts a block, where this region covers it.
    fn mark(self: *Lifter, at: u32) void {
        if (self.covers(at)) self.leaders[at] = true;
    }

    /// A word to look at next, where this region covers it. A word outside the window belongs to
    /// the region that begins there.
    fn reach(self: *Lifter, stack: *std.ArrayList(u32), at: u32) Error!void {
        if (!self.covers(at)) return;
        try stack.append(self.allocator, at);
    }

    /// Every word the region reaches from its entry, and every word that starts a block.
    fn discover(self: *Lifter, entry: u32) Error!void {
        var stack: std.ArrayList(u32) = .empty;
        defer stack.deinit(self.allocator);
        try stack.append(self.allocator, entry);

        while (stack.pop()) |at| {
            if (at >= self.program.len) return error.Outside;
            if (self.reached[at]) continue;
            self.reached[at] = true;

            const pc = at * 4;
            switch (try decode.one(self.program[at])) {
                // Nowhere else in the region. Where the warp goes next is the caller's to decide.
                .retire, .fault, .jalr => {},
                // The word after it, because a warp that was released carries on from there.
                .barrier => try self.reach(&stack, at + 1),
                .jal => |each| try self.reach(&stack, try self.target(pc, each.offset)),
                .branch => |each| {
                    try self.reach(&stack, try self.target(pc, each.offset));
                    try self.reach(&stack, at + 1);
                },
                else => try self.reach(&stack, at + 1),
            }
        }

        self.leaders[entry] = true;
        for (0..self.program.len) |at| {
            if (!self.reached[at]) continue;
            const pc: u32 = @intCast(at * 4);
            switch (try decode.one(self.program[at])) {
                .jal => |each| self.mark(try self.target(pc, each.offset)),
                .branch => |each| {
                    self.mark(try self.target(pc, each.offset));
                    self.mark(@intCast(at + 1));
                },
                .barrier => self.mark(@intCast(at + 1)),
                else => {},
            }
        }
    }

    /// The words of one block, up to a word that says where to go next or up to the next block.
    fn emit(self: *Lifter, start: u32) Error!void {
        self.at = self.blocks[start].?;
        var index = start;
        while (true) {
            const pc: u32 = @intCast(index * 4);
            const word = self.program[index];
            const one = try decode.one(word);
            switch (one) {
                .lui => |each| try self.set(each.rd, try self.constant(each.value)),
                .auipc => |each| try self.set(each.rd, try self.constant(pc +% each.value)),
                .alu_imm => |each| {
                    const a = try self.get(each.rs1);
                    const b = try self.constant(@bitCast(each.value));
                    try self.set(each.rd, try self.arithmetic(each.op, a, b));
                },
                .alu => |each| {
                    const a = try self.get(each.rs1);
                    const b = try self.get(each.rs2);
                    try self.set(each.rd, try self.arithmetic(each.op, a, b));
                },
                .csr => |each| try self.set(each.rd, try self.control(each.which)),
                // The one float the core does not do per lane. Its registers are the ordinary
                // ones, so this reads and writes them like any other arithmetic.
                .float_divide => |each| try self.set(each.rd, try self.floatArithmetic(
                    .div,
                    try self.get(each.rs1),
                    try self.get(each.rs2),
                )),
                .vector => |each| try self.lowerVector(each),
                .load => try self.lower(one),
                .store => try self.lowerStore(one),
                .atomic => try self.lowerAtomic(one),
                // One bus with one access in flight already gives the order a fence asks for.
                .fence => {},
                .retire => return self.leave(.retired, try self.constant(pc)),
                .fault => |each| {
                    try self.writeField("fault_code", try self.get(each.rs1));
                    return self.leave(.faulted, try self.constant(pc));
                },
                .barrier => return self.leave(.parked, try self.constant(pc + 4)),
                .jalr => |each| {
                    const base = try self.get(each.rs1);
                    const sum = try self.func.appendArithImm(self.at, self.u32_t, .add, base, each.offset);
                    const to = try self.func.appendArithImm(self.at, self.u32_t, .bit_and, sum, ~@as(i64, 1));
                    if (each.rd != 0) try self.set(each.rd, try self.constant(pc + 4));
                    return self.leave(.jumped, to);
                },
                .jal => |each| {
                    if (each.rd != 0) try self.set(each.rd, try self.constant(pc + 4));
                    const to = try self.target(pc, each.offset);
                    // A jump backwards is where a kernel loops, and a kernel that waits for work from
                    // its host loops forever. So the stop the host asks for is read here, which is the
                    // only place a region can be left without the kernel saying so.
                    // A jump out of the window is where this region ends and the next begins.
                    if (!self.covers(to)) return self.leave(.jumped, try self.constant(to * 4));
                    if (to <= index) return self.jumpChecked(to);
                    try self.func.setJump(self.at, self.blocks[to].?, &.{});
                    return;
                },
                .branch => |each| {
                    const a = try self.get(each.rs1);
                    const b = try self.get(each.rs2);
                    const cond = try self.compare(each.on, a, b);
                    const to = try self.target(pc, each.offset);
                    const next = try self.edgeTo(index + 1);
                    if (self.covers(to) and to <= index) {
                        const taken = try self.func.appendBlock();
                        try self.func.appendIf(self.at, cond, .{ .target = taken }, .{ .target = next });
                        self.at = taken;
                        return self.jumpChecked(to);
                    }
                    try self.func.appendIf(self.at, cond, .{ .target = try self.edgeTo(to) }, .{ .target = next });
                    return;
                },
            }

            index += 1;
            if (index >= self.program.len) return error.Outside;
            // The word after the last one this region covers is where the next region starts.
            if (!self.covers(index)) return self.leave(.jumped, try self.constant(index * 4));
            if (self.leaders[index]) {
                try self.func.setJump(self.at, self.blocks[index].?, &.{});
                return;
            }
        }
    }

    /// A jump backwards, with the host's stop read first. A kernel cannot see this and a host could
    /// not take a spinning core down without it.
    fn jumpChecked(self: *Lifter, to: u32) Error!void {
        const stop = try self.readStop();
        const zero = try self.constant(0);
        const asked_to = try self.func.appendInst(self.at, self.bool_t, .{ .icmp = .{ .op = .ne, .lhs = stop, .rhs = zero } });

        const leaving = try self.func.appendBlock();
        try self.func.appendIf(self.at, asked_to, .{ .target = leaving }, .{ .target = self.blocks[to].? });

        self.at = leaving;
        // Where the warp would have gone, so a core that is started again carries on from the loop
        // rather than from wherever it was asked to stop.
        try self.leave(.stopped, try self.constant(to * 4));
    }
};

/// Lift the region a warp reaches from `entry`, a byte offset into the program, into `func`.
///
/// The function takes one argument, the address of a `Warp`, and answers with an `Exit`.
pub fn region(allocator: std.mem.Allocator, func: *Function, program: []const u32, entry: u32, lanes: u32) Error!void {
    if (entry % 4 != 0) return error.Outside;
    if (program.len == 0 or entry / 4 >= program.len) return error.Outside;

    const leaders = try allocator.alloc(bool, program.len);
    defer allocator.free(leaders);
    const reached = try allocator.alloc(bool, program.len);
    defer allocator.free(reached);
    const blocks = try allocator.alloc(?Block, program.len);
    defer allocator.free(blocks);
    @memset(leaders, false);
    @memset(reached, false);
    @memset(blocks, null);

    const prologue = try func.appendBlock();
    var self: Lifter = .{
        .allocator = allocator,
        .func = func,
        .program = program,
        .u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } }),
        .i8_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 8 } }),
        .u16_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 16 } }),
        .i16_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 16 } }),
        .u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } }),
        .i32_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 32 } }),
        .u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } }),
        .i64_t = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } }),
        .f32_t = try func.types.intern(.{ .float = .f32 }),
        .f16_t = try func.types.intern(.{ .float = .f16 }),
        .bool_t = try func.types.intern(.bool),
        .ptr_t = try func.types.ptrGlobal(),
        .lanes = lanes,
        .first = entry / 4,
        .ctx = undefined,
        .slots = undefined,
        .carry = undefined,
        .half_slot = undefined,
        .leaders = leaders,
        .reached = reached,
        .blocks = blocks,
        .at = prologue,
    };
    self.ctx = try func.appendBlockParam(prologue, self.ptr_t);

    try self.discover(entry / 4);

    // A slot per register, filled from the context. The slots never escape, so `mem2reg` promotes
    // them and a register becomes a value rather than a memory access.
    self.slots[0] = undefined;
    for (1..self.slots.len) |which| {
        self.slots[which] = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    }
    self.carry = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u32_t } });
    self.half_slot = try func.appendInst(prologue, self.ptr_t, .{ .alloca = .{ .elem = self.u16_t } });
    for (1..self.slots.len) |which| {
        const held = try func.appendInst(prologue, self.u32_t, .{ .load = .{ .ptr = try self.slotOf(which) } });
        try func.appendStore(prologue, held, self.slots[which]);
    }

    for (0..program.len) |at| {
        if (self.leaders[at] and self.reached[at]) blocks[at] = try func.appendBlock();
    }
    try func.setJump(prologue, blocks[entry / 4].?, &.{});

    for (0..program.len) |at| {
        if (blocks[at] == null) continue;
        try self.emit(@intCast(at));
    }
}
