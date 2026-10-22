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
};

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
    bool_t: Type,
    ptr_t: Type,

    /// The context, which is the only argument a compiled region takes.
    ctx: Value,
    /// A stack slot per scalar register. Register zero has none, because it reads zero and takes
    /// no writes.
    slots: [32]Value,
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
    fn leave(self: *Lifter, why: Exit, pc: Value) Error!void {
        try self.writeField("pc", pc);
        for (1..self.slots.len) |which| {
            const value = try self.func.appendInst(self.at, self.u32_t, .{ .load = .{ .ptr = self.slots[which] } });
            try self.func.appendStore(self.at, value, try self.slotOf(which));
        }
        const reason = try self.func.appendInst(self.at, self.u32_t, .{ .iconst = @as(i64, @backingInt(why)) });
        self.func.setTerminator(self.at, .{ .ret = ir.function.Ret.one(reason) });
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
                .barrier => try stack.append(self.allocator, at + 1),
                .jal => |each| try stack.append(self.allocator, try self.target(pc, each.offset)),
                .branch => |each| {
                    try stack.append(self.allocator, try self.target(pc, each.offset));
                    try stack.append(self.allocator, at + 1);
                },
                else => try stack.append(self.allocator, at + 1),
            }
        }

        self.leaders[entry] = true;
        for (0..self.program.len) |at| {
            if (!self.reached[at]) continue;
            const pc: u32 = @intCast(at * 4);
            switch (try decode.one(self.program[at])) {
                .jal => |each| self.leaders[try self.target(pc, each.offset)] = true,
                .branch => |each| {
                    self.leaders[try self.target(pc, each.offset)] = true;
                    self.leaders[at + 1] = true;
                },
                .barrier => self.leaders[at + 1] = true,
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
                    if (to <= index) return self.jumpChecked(to);
                    try self.func.setJump(self.at, self.blocks[to].?, &.{});
                    return;
                },
                .branch => |each| {
                    const a = try self.get(each.rs1);
                    const b = try self.get(each.rs2);
                    const cond = try self.compare(each.on, a, b);
                    const to = try self.target(pc, each.offset);
                    if (to <= index) {
                        const taken = try self.func.appendBlock();
                        try self.func.appendIf(self.at, cond, .{ .target = taken }, .{ .target = self.blocks[index + 1].? });
                        self.at = taken;
                        return self.jumpChecked(to);
                    }
                    try self.func.appendIf(
                        self.at,
                        cond,
                        .{ .target = self.blocks[to].? },
                        .{ .target = self.blocks[index + 1].? },
                    );
                    return;
                },
            }

            index += 1;
            if (index >= self.program.len) return error.Outside;
            if (self.leaders[index]) {
                try self.func.setJump(self.at, self.blocks[index].?, &.{});
                return;
            }
        }
    }

    /// A jump backwards, with the host's stop read first. A kernel cannot see this and a host could
    /// not take a spinning core down without it.
    fn jumpChecked(self: *Lifter, to: u32) Error!void {
        const stop = try self.read("stop", self.u32_t);
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
pub fn region(allocator: std.mem.Allocator, func: *Function, program: []const u32, entry: u32) Error!void {
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
        .bool_t = try func.types.intern(.bool),
        .ptr_t = try func.types.ptrGlobal(),
        .ctx = undefined,
        .slots = undefined,
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
