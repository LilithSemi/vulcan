//! Lifting AArch64 instruction words into Vulcan IR, so a host can run a guest as native code.
//!
//! A block is a straight line of words that ends where control leaves it: a branch, a system
//! instruction, or an access whose outcome only the host can decide. It lifts into one IR
//! function taking a pointer to the `Cpu` and returning the pc to carry on from. Registers live in
//! the `Cpu`, so what a trap, a device access or an exception needs is already in memory when
//! compiled code returns. What `Cpu.trap` says is the reason the block stopped early.
//!
//! This only builds IR. Compiling it, keeping the code, and everything a trap means belong to
//! whoever runs the guest.
const std = @import("std");
const ir = @import("vulcan-ir");
const Cpu = @import("cpu.zig").Cpu;
const Decode = @import("decode.zig");
const Identification = @import("identification.zig");

pub const Error = std.mem.Allocator.Error || Decode.Error || error{ UnsupportedInstruction, EmptyBlock };

const Function = ir.function.Function;
const Value = ir.function.Value;

/// Lift little-endian AArch64 instructions at `guest_pc` into `func`, which must be empty. The
/// function takes a pointer to the `Cpu` and returns the pc to carry on from. Control flow, memory
/// access, or a trap terminates the block.
pub fn region(func: *Function, guest_pc: u64, bytes: []const u8) Error!void {
    if (bytes.len == 0) return error.EmptyBlock;
    if (bytes.len % 4 != 0) return error.UnsupportedInstruction;
    known = @splat(null);

    const u64_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 64 } });
    const u32_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 32 } });
    const u8_t = try func.types.intern(.{ .int = .{ .signedness = .unsigned, .bits = 8 } });
    const bool_t = try func.types.intern(.bool);
    const ptr_t = try func.types.ptrGlobal();
    const entry = try func.appendBlock();
    const cpu_ptr = try func.appendBlockParam(entry, ptr_t);
    // Advance the virtual counter once per instruction, where its actual
    // execution point is. Keeping this inside the instruction loop avoids
    // counting instructions beyond a trap and does not depend on translated
    // block boundaries.
    var pc: u64 = guest_pc;
    var terminated = false;
    var pending: u64 = 0;
    for (0..bytes.len / 4) |index| {
        if (terminated) break;
        const word = std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
        const instruction = try Decode.decode(word);
        // The counter is only observed where the block ends or a system register is
        // touched, so instructions are tallied here and added in one step there.
        pending += 1;
        if (instruction.terminates() or instruction == .system) {
            try advanceCounter(func, entry, cpu_ptr, pending, u64_t, ptr_t);
            pending = 0;
        }
        switch (instruction) {
            .movz => |wide| {
                const bits: u64 = @as(u64, wide.immediate) << wide.shift;
                const ty = if (wide.width == .x64) u64_t else u32_t;
                const value = try func.appendInst(entry, ty, .{ .iconst = @bitCast(bits) });
                if (wide.rd != 31) try storeReg(func, entry, cpu_ptr, wide.rd, value, u64_t, ptr_t);
            },
            .movn => |wide| {
                // The complement of the shifted constant, cut to the register's width so
                // that a 32-bit result has nothing above bit 31.
                const ty = if (wide.width == .x64) u64_t else u32_t;
                const width_mask: u64 = if (wide.width == .x64) std.math.maxInt(u64) else std.math.maxInt(u32);
                const bits: u64 = ~(@as(u64, wide.immediate) << wide.shift) & width_mask;
                const value = try func.appendInst(entry, ty, .{ .iconst = @bitCast(bits) });
                if (wide.rd != 31) try storeReg(func, entry, cpu_ptr, wide.rd, value, u64_t, ptr_t);
            },
            .movk => |wide| {
                if (wide.rd != 31) {
                    const bits: u64 = @as(u64, wide.immediate) << wide.shift;
                    const ty = if (wide.width == .x64) u64_t else u32_t;
                    const old = try loadReg(func, entry, cpu_ptr, wide.rd, ty, u64_t, ptr_t);
                    const clear = ~(@as(u64, 0xffff) << wide.shift);
                    const mask: u64 = if (wide.width == .x64) clear else @as(u32, @truncate(clear));
                    const mask_value = try func.appendInst(entry, ty, .{ .iconst = @bitCast(mask) });
                    const bits_value = try func.appendInst(entry, ty, .{ .iconst = @bitCast(bits) });
                    const kept = try func.appendInst(entry, ty, .{ .arith = .{ .op = .bit_and, .lhs = old, .rhs = mask_value } });
                    const result = try func.appendInst(entry, ty, .{ .arith = .{ .op = .bit_or, .lhs = kept, .rhs = bits_value } });
                    try storeReg(func, entry, cpu_ptr, wide.rd, result, u64_t, ptr_t);
                }
            },
            .arith_imm => |arith| try lowerArith(func, entry, cpu_ptr, arith, u64_t, u32_t, bool_t, ptr_t),
            .arith_reg => |arith| try lowerArithReg(func, entry, cpu_ptr, arith, u64_t, u32_t, bool_t, ptr_t),
            .arith_ext => |arith| try lowerArithExtended(func, entry, cpu_ptr, arith, u64_t, u32_t, bool_t, ptr_t),
            .logic_reg => |logic| try lowerLogicReg(func, entry, cpu_ptr, logic, u64_t, u32_t, bool_t, ptr_t),
            .logic_imm => |logic| try lowerLogicImm(func, entry, cpu_ptr, logic, u64_t, u32_t, bool_t, ptr_t),
            .variable => |variable| try lowerVariable(func, entry, cpu_ptr, variable, u64_t, u32_t, bool_t, ptr_t),
            .bitfield => |field| try lowerBitfield(func, entry, cpu_ptr, field, u64_t, u32_t, ptr_t),
            .cond_compare => |compare| try lowerCondCompare(func, entry, cpu_ptr, compare, u64_t, u32_t, bool_t, ptr_t),
            .arith_carry => |arith| try lowerArithCarry(func, entry, cpu_ptr, arith, u64_t, u32_t, bool_t, ptr_t),
            .unary => |unary| try lowerUnary(func, entry, cpu_ptr, unary, u64_t, u32_t, bool_t, ptr_t),
            .extract => |pair| try lowerExtract(func, entry, cpu_ptr, pair, u64_t, u32_t, ptr_t),
            .mul_long => |product| try lowerMultiplyLong(func, entry, cpu_ptr, product, u64_t, ptr_t),
            .mul_high => |product| try lowerMultiplyHigh(func, entry, cpu_ptr, product, u64_t, ptr_t),
            .test_branch => |branch| {
                const ty = if (branch.width == .x64) u64_t else u32_t;
                const value = try loadReg(func, entry, cpu_ptr, branch.rt, ty, u64_t, ptr_t);
                const zero = try func.appendInst(entry, ty, .{ .iconst = 0 });
                // All four of these branch on one thing being zero or not, and
                // the encoding bit says which, so the two forms differ only in
                // what they compare: the whole register, or one bit of it.
                const subject: Value = if (branch.bit_index) |position| blk: {
                    const mask = try func.appendInst(entry, ty, .{ .iconst = @as(i64, 1) << @intCast(position) });
                    break :blk try func.appendInst(entry, ty, .{ .arith = .{ .op = .bit_and, .lhs = value, .rhs = mask } });
                } else value;
                const taken = try func.appendInst(entry, bool_t, .{ .icmp = .{
                    .op = if (branch.negate) .ne else .eq,
                    .lhs = subject,
                    .rhs = zero,
                } });
                try branchOn(func, entry, pc, branch.offset, taken, u64_t);
                terminated = true;
            },
            .call => |call| {
                if (call.link) try storeReg(func, entry, cpu_ptr, 30, try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) }), u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% @as(u64, @bitCast(call.target))) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .indirect => |branch| {
                const target = try loadReg(func, entry, cpu_ptr, branch.rn, u64_t, u64_t, ptr_t);
                if (branch.link) try storeReg(func, entry, cpu_ptr, 30, try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) }), u64_t, ptr_t);
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(target) });
                terminated = true;
            },
            .b_cond => |branch| {
                const taken = try conditionValue(func, entry, cpu_ptr, branch.cond, u32_t, bool_t, ptr_t);
                try branchOn(func, entry, pc, branch.offset, taken, u64_t);
                terminated = true;
            },
            .system => |sys| {
                if (stackLevel(sys.register)) |level| {
                    // The live stack pointer and the saved one, chosen by where
                    // the guest is running. Selecting on the address rather than
                    // on the value keeps this to one store either way.
                    const live = try atLevel(func, entry, cpu_ptr, level, u64_t, u8_t, bool_t, ptr_t);
                    const address = try func.appendInst(entry, ptr_t, .{ .select = .{
                        .cond = live,
                        .then = try regAddress(func, entry, cpu_ptr, 31, u64_t, ptr_t),
                        .@"else" = try systemAddress(func, entry, cpu_ptr, @offsetOf(Cpu.System, "sp_el") + 8 * level, u64_t, ptr_t),
                    } });
                    if (sys.read) {
                        const value = try func.appendInst(entry, u64_t, .{ .load = .{ .ptr = address } });
                        try storeReg(func, entry, cpu_ptr, sys.rt, value, u64_t, ptr_t);
                    } else {
                        const from = try loadRegOrZero(func, entry, cpu_ptr, sys.rt, u64_t, u64_t, ptr_t);
                        try func.appendStore(entry, from, address);
                        // It may have landed on the live stack pointer.
                        known[31] = null;
                    }
                } else if (sys.register == .current_el) {
                    // The level in the two bits the architecture puts it in, and nothing
                    // else: EL0 reads 0, EL1 reads 4, EL2 reads 8. A kernel compares it
                    // against those, so a stray bit turns one level into another.
                    // It cannot be written, and the decoder refuses that.
                    const level = try loadField(func, entry, cpu_ptr, @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "el"), u8_t, ptr_t);
                    const encoded = try func.appendArithImm(entry, u64_t, .shl, try func.appendInst(entry, u64_t, .{ .convert = .{ .value = level } }), 2);
                    try storeReg(func, entry, cpu_ptr, sys.rt, encoded, u64_t, ptr_t);
                } else if (sys.register == .daif_set or sys.register == .daif_clear) {
                    // The two immediate forms change only the bits the mask
                    // names, so the rest of the register is left alone.
                    const at = @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "daif");
                    const before = try loadField(func, entry, cpu_ptr, at, u64_t, ptr_t);
                    const mask = try func.appendInst(entry, u64_t, .{ .iconst = sys.immediate });
                    const updated = try func.appendInst(entry, u64_t, .{ .arith = .{
                        .op = if (sys.register == .daif_set) .bit_or else .bit_and,
                        .lhs = before,
                        .rhs = if (sys.register == .daif_set)
                            mask
                        else
                            try func.appendInst(entry, u64_t, .{ .arith = .{
                                .op = .bit_xor,
                                .lhs = mask,
                                .rhs = try func.appendInst(entry, u64_t, .{ .iconst = 0xf }),
                            } }),
                    } });
                    try storeField(func, entry, cpu_ptr, at, updated, u64_t, ptr_t);
                } else if (sys.register == .nzcv) {
                    // The condition flags live in `Cpu.flags` rather than in the
                    // control block, so a read is that value widened and a write
                    // keeps only the four bits that are the flags.
                    if (sys.read) {
                        // The flags are a `u32` field, so the access is that wide
                        // and then widened; a `u64` access would read the field
                        // that follows it as well.
                        const flags = try loadField(func, entry, cpu_ptr, @offsetOf(Cpu, "flags"), u32_t, ptr_t);
                        try storeReg(func, entry, cpu_ptr, sys.rt, try func.appendInst(entry, u64_t, .{ .convert = .{ .value = flags } }), u64_t, ptr_t);
                    } else {
                        const from = try loadRegOrZero(func, entry, cpu_ptr, sys.rt, u64_t, u64_t, ptr_t);
                        const kept = try func.appendInst(entry, u32_t, .{ .arith = .{
                            .op = .bit_and,
                            .lhs = try func.appendInst(entry, u32_t, .{ .convert = .{ .value = from } }),
                            .rhs = try func.appendInst(entry, u32_t, .{ .iconst = 0xf }),
                        } });
                        try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "flags"), kept, u32_t, ptr_t);
                    }
                } else if (Identification.value(sys.register)) |constant| {
                    // Read-only, and the decoder has refused every write to it.
                    try storeReg(func, entry, cpu_ptr, sys.rt, try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(constant) }), u64_t, ptr_t);
                } else if (sys.register == .oslsr_el1) {
                    // The lock status derives from the two locks: bit 1 set while
                    // either is held, which is the reset state, and clear once
                    // both are zero. A write to the status itself is refused by
                    // the decoder, so only a read arrives here.
                    const at = @offsetOf(Cpu, "system");
                    const oslar = try loadField(func, entry, cpu_ptr, at + @offsetOf(Cpu.System, "oslar_el1"), u64_t, ptr_t);
                    const osdlr = try loadField(func, entry, cpu_ptr, at + @offsetOf(Cpu.System, "osdlr_el1"), u64_t, ptr_t);
                    const either = try func.appendInst(entry, u64_t, .{ .arith = .{ .op = .bit_or, .lhs = oslar, .rhs = osdlr } });
                    const locked = try func.appendInst(entry, bool_t, .{ .icmp = .{ .op = .ne, .lhs = either, .rhs = try func.appendInst(entry, u64_t, .{ .iconst = 0 }) } });
                    const value = try func.appendInst(entry, u64_t, .{ .select = .{ .cond = locked, .then = try func.appendInst(entry, u64_t, .{ .iconst = 0b10 }), .@"else" = try func.appendInst(entry, u64_t, .{ .iconst = 0 }) } });
                    try storeReg(func, entry, cpu_ptr, sys.rt, value, u64_t, ptr_t);
                } else if (sys.register == .daif) {
                    // The four masks are kept as bits 3:0, and the register puts them
                    // at 9:6, so the register form converts in both directions. The
                    // immediate forms and an exception use the stored layout as it is.
                    const at = @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "daif");
                    if (sys.read) {
                        const stored = try loadField(func, entry, cpu_ptr, at, u64_t, ptr_t);
                        try storeReg(func, entry, cpu_ptr, sys.rt, try func.appendArithImm(entry, u64_t, .shl, stored, 6), u64_t, ptr_t);
                    } else {
                        const from = try loadRegOrZero(func, entry, cpu_ptr, sys.rt, u64_t, u64_t, ptr_t);
                        const masks = try func.appendArithImm(entry, u64_t, .bit_and, try func.appendArithImm(entry, u64_t, .shr, from, 6), 0xf);
                        try storeField(func, entry, cpu_ptr, at, masks, u64_t, ptr_t);
                    }
                } else {
                    const offset = systemOffset(sys.register).?;
                    if (sys.read) {
                        const value = if (sys.register == .cntv_ctl_el0)
                            try timerControl(func, entry, cpu_ptr, u64_t, bool_t, ptr_t)
                        else
                            try loadField(func, entry, cpu_ptr, @offsetOf(Cpu, "system") + offset, u64_t, ptr_t);
                        try storeReg(func, entry, cpu_ptr, sys.rt, value, u64_t, ptr_t);
                    } else if (sys.register == .cntv_ctl_el0) {
                        // Only the enable and mask bits exist here. The condition is
                        // observed, not stored, so a write must not set or clear it.
                        const from = try loadRegOrZero(func, entry, cpu_ptr, sys.rt, u64_t, u64_t, ptr_t);
                        const kept = try func.appendArithImm(entry, u64_t, .bit_and, from, 0b11);
                        try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "system") + offset, kept, u64_t, ptr_t);
                    } else {
                        const from = try loadRegOrZero(func, entry, cpu_ptr, sys.rt, u64_t, u64_t, ptr_t);
                        try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "system") + offset, from, u64_t, ptr_t);
                        if (sys.register == .cntv_cval_el0) try refreshTimer(func, entry, cpu_ptr, u64_t, bool_t, ptr_t);
                    }
                }
            },
            .mul => |product| {
                const ty = if (product.width == .x64) u64_t else u32_t;
                const lhs = try loadRegOrZero(func, entry, cpu_ptr, product.rn, ty, u64_t, ptr_t);
                const rhs = try loadRegOrZero(func, entry, cpu_ptr, product.rm, ty, u64_t, ptr_t);
                const result = try func.appendInst(entry, ty, .{ .arith = .{ .op = .mul, .lhs = lhs, .rhs = rhs } });
                // Register 31 as the accumulator is zero: for an add the product stands
                // alone, and for a subtract it is negated, which is what MNEG is.
                const value = if (product.ra == 31 and product.op == .add) result else blk: {
                    const accumulator = try loadRegOrZero(func, entry, cpu_ptr, product.ra, ty, u64_t, ptr_t);
                    const op: ir.function.BinOp = if (product.op == .sub) .sub else .add;
                    break :blk try func.appendInst(entry, ty, .{ .arith = .{ .op = op, .lhs = accumulator, .rhs = result } });
                };
                if (product.rd != 31) try storeReg(func, entry, cpu_ptr, product.rd, value, u64_t, ptr_t);
            },
            .adr => |address| {
                // The displacement is measured from the instruction itself, and a
                // page form rounds that down to the page it starts in.
                // Both operands are addresses, so this is done as wrapping
                // unsigned arithmetic: a negative displacement is a large
                // unsigned one, and the sum is the address either way.
                const page_mask: u64 = 0xfffffffffffff000;
                const here: u64 = if (address.page) pc & page_mask else pc;
                const value: u64 = here +% @as(u64, @bitCast(address.offset));
                const constant = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(value) });
                try storeReg(func, entry, cpu_ptr, address.rd, constant, u64_t, ptr_t);
            },
            .csel => |select| {
                const ty = if (select.width == .x64) u64_t else u32_t;
                const lhs = try loadRegOrZero(func, entry, cpu_ptr, select.rn, ty, u64_t, ptr_t);
                const rhs = try loadRegOrZero(func, entry, cpu_ptr, select.rm, ty, u64_t, ptr_t);
                // What a failed condition leaves behind: Rm, Rm+1, ~Rm, or -Rm.
                const alternative: Value = switch (select.opc) {
                    0 => rhs,
                    1 => try func.appendArithImm(entry, ty, .add, rhs, 1),
                    2 => try func.appendInst(entry, ty, .{ .arith = .{ .op = .bit_xor, .lhs = rhs, .rhs = try func.appendInst(entry, ty, .{ .iconst = -1 }) } }),
                    3 => try func.appendInst(entry, ty, .{ .arith = .{ .op = .sub, .lhs = try func.appendInst(entry, ty, .{ .iconst = 0 }), .rhs = rhs } }),
                };
                const cond = try conditionValue(func, entry, cpu_ptr, select.cond, u32_t, bool_t, ptr_t);
                const result = try func.appendInst(entry, ty, .{ .select = .{ .cond = cond, .then = lhs, .@"else" = alternative } });
                if (select.rd != 31) try storeReg(func, entry, cpu_ptr, select.rd, result, u64_t, ptr_t);
            },
            .memory => |mem| {
                const address = try computeAddress(func, entry, cpu_ptr, mem.rn, mem.addressing, u64_t, u8_t, ptr_t);
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "address"), address, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "width"), @backingInt(mem.size), u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "dest"), mem.rt, u8_t, u64_t, ptr_t);
                if (mem.signed != .none) try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "load_signed"), @backingInt(mem.signed), u8_t, u64_t, ptr_t);
                if (mem.op == .store) {
                    const value = if (mem.rt == 31) try func.appendInst(entry, u64_t, .{ .iconst = 0 }) else try loadReg(func, entry, cpu_ptr, mem.rt, u64_t, u64_t, ptr_t);
                    try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "value"), value, u64_t, ptr_t);
                }
                const trap: Cpu.Trap = if (mem.op == .store) .store else .load;
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(trap), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .literal => |load| {
                // The address is fixed by where the instruction is, so it is a constant.
                const address = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% @as(u64, @bitCast(load.offset))) });
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "address"), address, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "width"), @backingInt(load.size), u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "dest"), load.rt, u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(Cpu.Trap.load), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .exclusive => |access| {
                // The address is the base register as it is, which may be the stack
                // pointer. What makes this exclusive is done by the run loop, which
                // owns the monitor: the block only says which kind of access it is
                // and, for a store, which register is to hear whether it happened.
                const base = try loadReg(func, entry, cpu_ptr, access.rn, u64_t, u64_t, ptr_t);
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "address"), base, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "width"), @backingInt(access.size), u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "dest"), access.rt, u8_t, u64_t, ptr_t);
                if (access.op == .store) {
                    const value = try loadRegOrZero(func, entry, cpu_ptr, access.rt, u64_t, u64_t, ptr_t);
                    try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "value"), value, u64_t, ptr_t);
                    try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "status_dest"), access.rs, u8_t, u64_t, ptr_t);
                }
                const kind: Cpu.Exclusive = if (access.op == .store) .store else .load;
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "exclusive"), @backingInt(kind), u8_t, u64_t, ptr_t);
                const trap: Cpu.Trap = if (access.op == .store) .store else .load;
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(trap), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .clrex => try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "monitor_valid"), 0, u8_t, u64_t, ptr_t),
            .pair => |mem| {
                // A pair is two accesses at consecutive addresses. Only the first
                // can be set up here, because the second must not be started
                // until the first has retired, so it is left for the run loop.
                const address = try computeAddress(func, entry, cpu_ptr, mem.rn, mem.addressing, u64_t, u8_t, ptr_t);
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "address"), address, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "width"), @backingInt(mem.size), u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "dest"), mem.rt, u8_t, u64_t, ptr_t);
                if (mem.signed != .none) try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "load_signed"), @backingInt(mem.signed), u8_t, u64_t, ptr_t);
                if (mem.op == .store) {
                    const value = if (mem.rt == 31) try func.appendInst(entry, u64_t, .{ .iconst = 0 }) else try loadReg(func, entry, cpu_ptr, mem.rt, u64_t, u64_t, ptr_t);
                    try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "value"), value, u64_t, ptr_t);
                }
                // The second access. Its value is read from the register now, but
                // the access itself waits, so that a load cannot write its
                // destination before the first half has been delivered.
                const second_address = try func.appendArithImm(entry, u64_t, .add, address, @backingInt(mem.size));
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "second_address"), second_address, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "second_width"), @backingInt(mem.size), u8_t, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "second_dest"), mem.rt2, u8_t, u64_t, ptr_t);
                if (mem.signed != .none) try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "second_signed"), @backingInt(mem.signed), u8_t, u64_t, ptr_t);
                if (mem.op == .store) {
                    const value = if (mem.rt2 == 31) try func.appendInst(entry, u64_t, .{ .iconst = 0 }) else try loadReg(func, entry, cpu_ptr, mem.rt2, u64_t, u64_t, ptr_t);
                    try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "second_value"), value, u64_t, ptr_t);
                }
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "second_pending"), 1, u8_t, u64_t, ptr_t);
                const trap: Cpu.Trap = if (mem.op == .store) .store else .load;
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(trap), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            // A hint that does not change the machine state compiles to nothing
            // at all, so the block simply continues.
            // `SEV` sets the event `WFE` waits on; `SEVL` sets it too, as a
            // local promise that the next wait falls through.
            .sev, .sevl => try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "event_set"), 1, u8_t, u64_t, ptr_t),
            .nop, .cache_op, .dsb, .dmb, .yield => {},
            .wfe => {
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(Cpu.Trap.wfe), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .trap => {
                // `BRK` is how a compiler spells a trap. Reaching one means the
                // guest did something this cannot account for, so it stops here
                // rather than running on with a state nobody can trust.
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(Cpu.Trap.brk), u8_t, u64_t, ptr_t);
                const here = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(here) });
                terminated = true;
            },
            // An `ISB` orders nothing here, but it is the architecture's signal
            // that the translation regime may have changed, so it ends the block
            // and the run loop acts on it.
            .isb, .tlbi => {
                const trap: Cpu.Trap = if (instruction == .tlbi) .tlbi else .sync;
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(trap), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            // `DC ZVA` zeroes a whole line, so it becomes a memory operation for
            // the run loop: only it knows where the address lands.
            .dc_zva => |rt| {
                const value = if (rt == 31) try func.appendInst(entry, u64_t, .{ .iconst = 0 }) else try loadReg(func, entry, cpu_ptr, rt, u64_t, u64_t, ptr_t);
                try storeField(func, entry, cpu_ptr, @offsetOf(Cpu, "address"), value, u64_t, ptr_t);
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(Cpu.Trap.dc_zva), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            // `ERET` is decided by the run loop, which restores the saved state.
            .eret => {
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(Cpu.Trap.eret), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .svc, .psci, .wfi => {
                const trap: Cpu.Trap = switch (instruction) {
                    .svc => .svc,
                    .psci => .psci,
                    .wfi => .wfi,
                    else => unreachable,
                };
                try storeFieldConst(func, entry, cpu_ptr, @offsetOf(Cpu, "trap"), @backingInt(trap), u8_t, u64_t, ptr_t);
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc +% 4) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
            .b => |offset| {
                const target = pc +% @as(u64, @bitCast(offset));
                const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(target) });
                func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
                terminated = true;
            },
        }
        pc +%= 4;
    }

    if (!terminated) {
        try advanceCounter(func, entry, cpu_ptr, pending, u64_t, ptr_t);
        const next = try func.appendInst(entry, u64_t, .{ .iconst = @bitCast(pc) });
        func.setTerminator(entry, .{ .ret = ir.function.Ret.one(next) });
    }
}

/// One `ADD`/`SUB` immediate. The `S` forms also update NZCV, which is what
/// every later condition reads, so the flags are computed in the same block
/// rather than left to the run loop.
fn lowerArith(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    arith: Decode.Arithmetic,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (arith.width == .x64) u64_t else u32_t;
    const lhs = try loadReg(func, block, cpu_ptr, arith.rn, ty, u64_t, ptr_t);
    const rhs = try func.appendInst(block, ty, .{ .iconst = arith.immediate });
    const result = try func.appendArithImm(block, ty, if (arith.op == .sub) .sub else .add, lhs, arith.immediate);
    // Register 31 is the stack pointer for the plain forms and the zero register for
    // the flag-setting ones, where naming it is how `CMP` and `CMN` throw the result
    // away. Writing it anyway would put a comparison's difference into the stack.
    if (!arith.flags or arith.rd != 31) try storeReg(func, block, cpu_ptr, arith.rd, result, u64_t, ptr_t);
    if (arith.flags) try setFlags(func, block, cpu_ptr, arith.op, arith.width, lhs, rhs, result, u32_t, bool_t, u64_t, ptr_t);
}

/// One `ADD`/`SUB` (shifted register), which is the form a compiler emits for
/// anything but a constant.
/// A branch that ends the block either way: the target when the condition holds,
/// the following instruction when it does not.
/// The address a load or store works on. A pre-index form updates the base
/// register here, because the access it describes happens at the new value. A
/// post-index form accesses the old value and updates the base afterwards, which
/// only the run loop can do, so it records what to write there.
fn computeAddress(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    rn: u5,
    addressing: Decode.Addressing,
    u64_t: ir.types.Type,
    u8_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!Value {
    const base = try loadReg(func, block, cpu_ptr, rn, u64_t, u64_t, ptr_t);
    const address: Value = switch (addressing) {
        .post_index => base,
        .register => |reg| blk: {
            // The index is a register, extended as the encoding says and then scaled.
            // Register 31 here is the zero register, not the stack pointer.
            const index = try loadRegOrZero(func, block, cpu_ptr, reg.rm, u64_t, u64_t, ptr_t);
            const extended = try extendValue(func, block, u64_t, index, reg.extend);
            const shifted = try shiftLeft(func, block, u64_t, extended, reg.amount);
            break :blk try func.appendInst(block, u64_t, .{ .arith = .{ .op = .add, .lhs = base, .rhs = shifted } });
        },
        .offset, .pre_index => |amount| try func.appendArithImm(block, u64_t, .add, base, amount),
    };
    switch (addressing) {
        .pre_index => try storeReg(func, block, cpu_ptr, rn, address, u64_t, ptr_t),
        .post_index => |amount| {
            const updated = try func.appendArithImm(block, u64_t, .add, base, amount);
            try storeField(func, block, cpu_ptr, @offsetOf(Cpu, "writeback_value"), updated, u64_t, ptr_t);
            try storeFieldConst(func, block, cpu_ptr, @offsetOf(Cpu, "writeback_dest"), rn, u8_t, u64_t, ptr_t);
            try storeFieldConst(func, block, cpu_ptr, @offsetOf(Cpu, "writeback"), 1, u8_t, u64_t, ptr_t);
        },
        .offset, .register => {},
    }
    return address;
}

fn branchOn(func: *Function, block: ir.function.Block, pc: u64, offset: i64, taken: Value, u64_t: ir.types.Type) Error!void {
    const target = try func.appendInst(block, u64_t, .{ .iconst = @bitCast(pc +% @as(u64, @bitCast(offset))) });
    const fallthrough = try func.appendInst(block, u64_t, .{ .iconst = @bitCast(pc +% 4) });
    const next = try func.appendInst(block, u64_t, .{ .select = .{ .cond = taken, .then = target, .@"else" = fallthrough } });
    func.setTerminator(block, .{ .ret = ir.function.Ret.one(next) });
}

/// One bitwise instruction with a shifted register. The bitwise instructions
/// read register 31 as the zero register rather than as the stack pointer, which
/// is what makes `MOV` and `MVN` encodings of the same instruction.
fn lowerLogicReg(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    logic: Decode.Logic,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (logic.width == .x64) u64_t else u32_t;
    const lhs = try loadRegOrZero(func, block, cpu_ptr, logic.rn, ty, u64_t, ptr_t);
    const shifted = try shiftedLogicOperand(func, block, cpu_ptr, logic.rm, logic, ty, u64_t, ptr_t);
    const rhs = if (logic.invert)
        try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_xor, .lhs = shifted, .rhs = try func.appendInst(block, ty, .{ .iconst = -1 }) } })
    else
        shifted;
    const op: ir.function.BinOp = switch (logic.op) {
        .bit_and => .bit_and,
        .bit_or => .bit_or,
        .bit_xor => .bit_xor,
    };
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    // As with the immediate form, register 31 here is a discarded result and
    // not the stack pointer.
    if (!logic.flags or logic.rd != 31) try storeReg(func, block, cpu_ptr, logic.rd, result, u64_t, ptr_t);
    if (logic.flags) try setLogicFlags(func, block, cpu_ptr, result, logic.width, u32_t, bool_t, u64_t, ptr_t);
}

fn lowerLogicImm(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    logic: Decode.LogicImm,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (logic.width == .x64) u64_t else u32_t;
    const lhs = try loadRegOrZero(func, block, cpu_ptr, logic.rn, ty, u64_t, ptr_t);
    const rhs = try func.appendInst(block, ty, .{ .iconst = @bitCast(logic.immediate) });
    const op: ir.function.BinOp = switch (logic.op) {
        .bit_and => .bit_and,
        .bit_or => .bit_or,
        .bit_xor => .bit_xor,
    };
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = op, .lhs = lhs, .rhs = rhs } });
    // Register 31 as the destination of a flag-setting instruction is not the
    // stack pointer but a request to throw the result away, which is how `TST`
    // is spelled. Without the flag bit it really is the stack pointer.
    if (!logic.flags or logic.rd != 31) try storeReg(func, block, cpu_ptr, logic.rd, result, u64_t, ptr_t);
    if (logic.flags) try setLogicFlags(func, block, cpu_ptr, result, logic.width, u32_t, bool_t, u64_t, ptr_t);
}

fn shiftedLogicOperand(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    rm: u5,
    logic: Decode.Logic,
    ty: ir.types.Type,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!Value {
    const value = try loadRegOrZero(func, block, cpu_ptr, rm, ty, u64_t, ptr_t);
    if (logic.amount == 0) return value;
    return switch (logic.shift) {
        .lsl => try func.appendArithImm(block, ty, .shl, value, logic.amount),
        .lsr => try func.appendArithImm(block, ty, .shr, value, logic.amount),
        .ror => try func.appendInst(block, ty, .{ .arith = .{
            .op = .bit_or,
            .lhs = try func.appendArithImm(block, ty, .shr, value, logic.amount),
            .rhs = try func.appendArithImm(block, ty, .shl, value, @as(i64, @backingInt(logic.width)) - logic.amount),
        } }),
        .asr => blk: {
            const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = @backingInt(logic.width) } });
            const shifted = try func.appendArithImm(block, signed, .shr, try func.appendInst(block, signed, .{ .convert = .{ .value = value } }), logic.amount);
            break :blk try func.appendInst(block, ty, .{ .convert = .{ .value = shifted } });
        },
    };
}

/// N and Z come from the result, and C and V are cleared. That is what every
/// flag-setting bitwise instruction does in this architecture, whichever form it
/// is: there is no carry out of a logical operation to report, and no overflow.
fn setLogicFlags(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    result: Value,
    width: Decode.Width,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (width == .x64) u64_t else u32_t;
    const zero = try func.appendInst(block, ty, .{ .iconst = 0 });
    const top = try func.appendInst(block, ty, .{ .iconst = @as(i64, 1) << @intCast(@backingInt(width) - 1) });
    const n = try func.appendInst(block, bool_t, .{ .icmp = .{
        .op = .ne,
        .lhs = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = result, .rhs = top } }),
        .rhs = zero,
    } });
    const z = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = result, .rhs = zero } });
    const no = try func.appendInst(block, bool_t, .{ .iconst = 0 });
    const flags = try packFlags(func, block, .{ .n = n, .z = z, .c = no, .v = no }, u32_t);
    try storeField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), flags, u64_t, ptr_t);
}

fn loadRegOrZero(func: *Function, block: ir.function.Block, cpu_ptr: Value, reg: u32, ty: ir.types.Type, u64_t: ir.types.Type, ptr_t: ir.types.Type) Error!Value {
    if (reg != 31) return loadReg(func, block, cpu_ptr, reg, ty, u64_t, ptr_t);
    return func.appendInst(block, ty, .{ .iconst = 0 });
}

fn lowerArithReg(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    arith: Decode.ArithmeticReg,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (arith.width == .x64) u64_t else u32_t;
    const rhs = try shiftedOperand(func, block, cpu_ptr, arith.rm, arith, ty, u64_t, ptr_t);
    // Register 31 is the zero register in this form, as an operand and as the
    // destination, so `NEG` reads zero and `CMP` writes nowhere.
    const lhs = try loadRegOrZero(func, block, cpu_ptr, arith.rn, ty, u64_t, ptr_t);
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = if (arith.op == .sub) .sub else .add, .lhs = lhs, .rhs = rhs } });
    if (arith.rd != 31) try storeReg(func, block, cpu_ptr, arith.rd, result, u64_t, ptr_t);
    if (arith.flags) try setFlags(func, block, cpu_ptr, arith.op, arith.width, lhs, rhs, result, u32_t, bool_t, u64_t, ptr_t);
}

/// `Rm` put through the shift the encoding names. A right shift by the full
/// width is not a shift of zero, so the decoder has already measured the amount
/// against the register size.
fn shiftedOperand(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    rm: u5,
    arith: Decode.ArithmeticReg,
    ty: ir.types.Type,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!Value {
    const value = try loadRegOrZero(func, block, cpu_ptr, rm, ty, u64_t, ptr_t);
    if (arith.amount == 0) return value;
    return switch (arith.shift) {
        .lsl => try func.appendArithImm(block, ty, .shl, value, arith.amount),
        .lsr => try func.appendArithImm(block, ty, .shr, value, arith.amount),
        .ror => unreachable, // the decoder refuses this form, so it never arrives here
        .asr => blk: { // an arithmetic shift is a signed one, so the width has to say so
            const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = @backingInt(arith.width) } });
            const shifted = try func.appendArithImm(block, signed, .shr, try func.appendInst(block, signed, .{ .convert = .{ .value = value } }), arith.amount);
            break :blk try func.appendInst(block, ty, .{ .convert = .{ .value = shifted } });
        },
    };
}

/// NZCV from an addition or a subtraction. The two carry rules are the same
/// unsigned comparison read from opposite ends, and the two overflow rules are
/// both a sign change across the operation.
fn flagsOf(
    func: *Function,
    block: ir.function.Block,
    op: Decode.ArithOp,
    width: Decode.Width,
    lhs: Value,
    rhs: Value,
    result: Value,
    /// A carry the caller has worked out, for the operations whose carry is not the
    /// plain comparison of result and operand: one that takes a carry in.
    carry_out: ?Value,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    u64_t: ir.types.Type,
) Error!Value {
    const ty = if (width == .x64) u64_t else u32_t;
    const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = @backingInt(width) } });
    const zero = try func.appendInst(block, ty, .{ .iconst = 0 });
    const top = try func.appendInst(block, ty, .{ .iconst = @as(i64, 1) << @intCast(@backingInt(width) - 1) });
    // N is the result's own sign bit, so a masked compare says it without a signed convert.
    const n = try func.appendInst(block, bool_t, .{ .icmp = .{
        .op = .ne,
        .lhs = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = result, .rhs = top } }),
        .rhs = zero,
    } });
    const z = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = result, .rhs = zero } });
    // C is a borrow for a subtract and a carry for an add, which are the same unsigned compare.
    const c = if (carry_out) |given|
        given
    else if (op == .sub)
        try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = lhs, .rhs = rhs } })
    else
        try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = result, .rhs = lhs } });
    const lhs_negative = try isNegative(func, block, lhs, signed, bool_t);
    const result_negative = try isNegative(func, block, result, signed, bool_t);
    const rhs_negative = try isNegative(func, block, rhs, signed, bool_t);
    // V is a signed overflow: the true result does not fit. The result's sign is
    // wrong, and that can only happen when the operation could have pushed past
    // the end. An add overflows when both operands share a sign and the result has
    // the other; a subtract when the operands differ in sign and the result's sign
    // is not the left operand's. `4 - 8` changes sign without overflowing, and
    // `MIN - 1` does not change sign and does: the two tests are not the same.
    const differ = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_xor, .lhs = lhs_negative, .rhs = rhs_negative } });
    const agree = try func.appendInst(block, bool_t, .{ .arith = .{
        .op = .bit_xor,
        .lhs = differ,
        .rhs = try func.appendInst(block, bool_t, .{ .iconst = 1 }),
    } });
    const flipped = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_xor, .lhs = lhs_negative, .rhs = result_negative } });
    const v = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = if (op == .sub) differ else agree, .rhs = flipped } });
    return packFlags(func, block, .{ .n = n, .z = z, .c = c, .v = v }, u32_t);
}

/// Store the NZCV that `flagsOf` computes.
fn setFlags(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    op: Decode.ArithOp,
    width: Decode.Width,
    lhs: Value,
    rhs: Value,
    result: Value,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const flags = try flagsOf(func, block, op, width, lhs, rhs, result, null, u32_t, bool_t, u64_t);
    try storeField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), flags, u64_t, ptr_t);
}

/// `CCMP` and `CCMN`. Both outcomes are computed and one is chosen, so the block
/// stays one straight run of instructions: the flags a real compare would set if
/// the condition holds, and the immediate the encoding carries if it does not.
/// The condition is read from the flags as they stand before this instruction.
fn lowerCondCompare(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    compare: Decode.CondCompare,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (compare.width == .x64) u64_t else u32_t;
    const lhs = try loadRegOrZero(func, block, cpu_ptr, compare.rn, ty, u64_t, ptr_t);
    const rhs = switch (compare.operand) {
        .register => |rm| try loadRegOrZero(func, block, cpu_ptr, rm, ty, u64_t, ptr_t),
        .immediate => |constant| try func.appendInst(block, ty, .{ .iconst = constant }),
    };
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = if (compare.op == .sub) .sub else .add, .lhs = lhs, .rhs = rhs } });
    const computed = try flagsOf(func, block, compare.op, compare.width, lhs, rhs, result, null, u32_t, bool_t, u64_t);
    const holds = try conditionValue(func, block, cpu_ptr, compare.cond, u32_t, bool_t, ptr_t);
    // N is the top bit of the four the encoding carries, where PSTATE keeps it at 31.
    const otherwise = try func.appendInst(block, u32_t, .{ .iconst = @as(i64, compare.nzcv) << 28 });
    const chosen = try func.appendInst(block, u32_t, .{ .select = .{ .cond = holds, .then = computed, .@"else" = otherwise } });
    try storeField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), chosen, u64_t, ptr_t);
}

fn isNegative(func: *Function, block: ir.function.Block, value: Value, signed: ir.types.Type, bool_t: ir.types.Type) Error!Value {
    return func.appendInst(block, bool_t, .{ .icmp = .{
        .op = .lt,
        .lhs = try func.appendInst(block, signed, .{ .convert = .{ .value = value } }),
        .rhs = try func.appendInst(block, signed, .{ .iconst = 0 }),
    } });
}

/// NZCV where PSTATE keeps it: N at bit 31, Z at 30, C at 29, V at 28.
fn packFlags(func: *Function, block: ir.function.Block, flags: struct { n: Value, z: Value, c: Value, v: Value }, u32_t: ir.types.Type) Error!Value {
    const pieces = [_]struct { flag: Value, shift: i64 }{
        .{ .flag = flags.n, .shift = 31 },
        .{ .flag = flags.z, .shift = 30 },
        .{ .flag = flags.c, .shift = 29 },
        .{ .flag = flags.v, .shift = 28 },
    };
    var nzcv = try func.appendInst(block, u32_t, .{ .iconst = 0 });
    for (pieces) |piece| {
        const bit = try func.appendInst(block, u32_t, .{ .convert = .{ .value = piece.flag } });
        nzcv = try func.appendInst(block, u32_t, .{ .arith = .{
            .op = .bit_or,
            .lhs = nzcv,
            .rhs = try func.appendArithImm(block, u32_t, .shl, bit, piece.shift),
        } });
    }
    return nzcv;
}

/// The condition codes as A64 numbers them, read out of the stored flags. The
/// composite codes are their architectural definitions rather than shorthand:
/// `hi` is `c && !z`, and the signed comparisons are `n` differing from `v`.
fn conditionValue(func: *Function, block: ir.function.Block, cpu_ptr: Value, cond: Decode.Condition, u32_t: ir.types.Type, bool_t: ir.types.Type, ptr_t: ir.types.Type) Error!Value {
    const flags = try loadField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), u32_t, ptr_t);
    const zero = try func.appendInst(block, u32_t, .{ .iconst = 0 });
    const read = struct {
        fn bit(f: *Function, b: ir.function.Block, flags_: Value, zero_: Value, ty: ir.types.Type, t: ir.types.Type, mask: i64) Error!Value {
            const masked = try f.appendInst(b, ty, .{ .arith = .{ .op = .bit_and, .lhs = flags_, .rhs = try f.appendInst(b, ty, .{ .iconst = mask }) } });
            return f.appendInst(b, t, .{ .icmp = .{ .op = .ne, .lhs = masked, .rhs = zero_ } });
        }
    };
    const n = try read.bit(func, block, flags, zero, u32_t, bool_t, 1 << 31);
    const z = try read.bit(func, block, flags, zero, u32_t, bool_t, 1 << 30);
    const c = try read.bit(func, block, flags, zero, u32_t, bool_t, 1 << 29);
    const v = try read.bit(func, block, flags, zero, u32_t, bool_t, 1 << 28);
    const logic = struct {
        fn not(f: *Function, b: ir.function.Block, value: Value, t: ir.types.Type) Error!Value {
            return f.appendInst(b, t, .{ .arith = .{ .op = .bit_xor, .lhs = value, .rhs = try f.appendInst(b, t, .{ .iconst = 1 }) } });
        }
        fn both(f: *Function, b: ir.function.Block, lhs: Value, rhs: Value, t: ir.types.Type) Error!Value {
            return f.appendInst(b, t, .{ .arith = .{ .op = .bit_and, .lhs = lhs, .rhs = rhs } });
        }
    };
    // Signed comparisons follow the definition: `lt` is a sign change between the
    // operands, which is N differing from V, and `ge` is the absence of one.
    const changed = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_xor, .lhs = n, .rhs = v } }); // n != v
    const ordered = try logic.not(func, block, changed, bool_t); // n == v
    const positive = try logic.not(func, block, z, bool_t);
    return switch (cond) {
        .eq => z,
        .ne => try logic.not(func, block, z, bool_t),
        .cs => c,
        .cc => try logic.not(func, block, c, bool_t),
        .mi => n,
        .pl => try logic.not(func, block, n, bool_t),
        .vs => v,
        .vc => try logic.not(func, block, v, bool_t),
        .hi => try logic.both(func, block, c, positive, bool_t),
        .ls => try logic.not(func, block, try logic.both(func, block, c, positive, bool_t), bool_t),
        .ge => ordered,
        .lt => changed,
        .gt => try logic.both(func, block, positive, ordered, bool_t),
        .le => try logic.not(func, block, try logic.both(func, block, positive, ordered, bool_t), bool_t),
        // Both mean "always", and a select or a conditional compare may name them.
        .al, .nv => try func.appendInst(block, bool_t, .{ .iconst = 1 }),
    };
}

/// The byte offset of a system register's storage, or null for the two that are
/// not a field of their own. `CurrentEL` and `NZCV` are read and written as
/// something other than a plain load, so they are handled by their own arms.
fn systemOffset(register: Decode.SystemRegister) ?usize {
    return switch (register) {
        .sp_el0, .sp_el1, .sp_el2 => unreachable, // handled by stackLevel before this is reached
        .current_el, .nzcv, .daif_set, .daif_clear => null,
        .spsr_el1 => @offsetOf(Cpu.System, "spsr_el1"),
        .elr_el1 => @offsetOf(Cpu.System, "elr_el1"),
        .esr_el1 => @offsetOf(Cpu.System, "esr_el1"),
        .far_el1 => @offsetOf(Cpu.System, "far_el1"),
        .vbar_el1 => @offsetOf(Cpu.System, "vbar_el1"),
        .cpacr_el1 => @offsetOf(Cpu.System, "cpacr_el1"),
        .sctlr_el1 => @offsetOf(Cpu.System, "sctlr_el1"),
        .ttbr0_el1 => @offsetOf(Cpu.System, "ttbr0_el1"),
        .ttbr1_el1 => @offsetOf(Cpu.System, "ttbr1_el1"),
        .tcr_el1 => @offsetOf(Cpu.System, "tcr_el1"),
        .mair_el1 => @offsetOf(Cpu.System, "mair_el1"),
        .tpidr_el0 => @offsetOf(Cpu.System, "tpidr_el0"),
        .tpidrro_el0 => @offsetOf(Cpu.System, "tpidrro_el0"),
        .cntvct_el0 => @offsetOf(Cpu.System, "cntvct_el0"),
        .cntv_ctl_el0 => @offsetOf(Cpu.System, "cntv_ctl_el0"),
        .cntv_cval_el0 => @offsetOf(Cpu.System, "cntv_cval_el0"),
        .tpidr_el1 => @offsetOf(Cpu.System, "tpidr_el1"),
        .mdscr_el1 => @offsetOf(Cpu.System, "mdscr_el1"),
        .cntkctl_el1 => @offsetOf(Cpu.System, "cntkctl_el1"),
        .oslar_el1 => @offsetOf(Cpu.System, "oslar_el1"),
        .osdlr_el1 => @offsetOf(Cpu.System, "osdlr_el1"),
        .oslsr_el1 => unreachable, // derived from the two locks, never stored
        .fpcr => @offsetOf(Cpu.System, "fpcr"),
        .fpsr => @offsetOf(Cpu.System, "fpsr"),
        .daif => @offsetOf(Cpu.System, "daif"),
        // Answered from `Identification`, and never stored.
        .midr_el1,
        .mpidr_el1,
        .revidr_el1,
        .ctr_el0,
        .dczid_el0,
        .cntfrq_el0,
        .clidr_el1,
        .id_aa64pfr0_el1,
        .id_aa64pfr1_el1,
        .id_aa64pfr2_el1,
        .id_aa64zfr0_el1,
        .id_aa64smfr0_el1,
        .id_aa64fpfr0_el1,
        .id_aa64isar3_el1,
        .aidr_el1,
        .id_aa64dfr0_el1,
        .id_aa64dfr1_el1,
        .id_aa64isar0_el1,
        .id_aa64isar1_el1,
        .id_aa64isar2_el1,
        .id_aa64mmfr0_el1,
        .id_aa64mmfr1_el1,
        .id_aa64mmfr2_el1,
        .id_aa64mmfr3_el1,
        .id_aa64mmfr4_el1,
        => null,
    };
}

/// Whether a system register names the current exception level's stack pointer.
/// The three of them are one encoding differing in the level, and the one
/// matching where the guest is running is the live `sp` rather than a copy, so
/// that an access through register 31 and a read of `SP_EL1` cannot disagree.
fn stackLevel(register: Decode.SystemRegister) ?u8 {
    return switch (register) {
        .sp_el0 => 0,
        .sp_el1 => 1,
        .sp_el2 => 2,
        else => null,
    };
}

/// True when a system-register access names the selected live stack pointer.
/// EL1 has two choices: EL1t uses SP_EL0; EL1h uses SP_EL1.
fn atLevel(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    level: u8,
    u64_t: ir.types.Type,
    u8_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!Value {
    const current = try func.appendInst(block, u64_t, .{ .convert = .{
        .value = try loadField(func, block, cpu_ptr, @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "el"), u8_t, ptr_t),
    } });
    const at_level = try func.appendInst(block, bool_t, .{ .icmp = .{
        .op = .eq,
        .lhs = current,
        .rhs = try func.appendInst(block, u64_t, .{ .iconst = level }),
    } });
    if (level == 0 or level == 1) {
        const at_el1 = try func.appendInst(block, bool_t, .{ .icmp = .{
            .op = .eq,
            .lhs = current,
            .rhs = try func.appendInst(block, u64_t, .{ .iconst = 1 }),
        } });
        const spsel = try loadField(func, block, cpu_ptr, @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "spsel"), u8_t, ptr_t);
        const zero = try func.appendInst(block, u8_t, .{ .iconst = 0 });
        const selected = try func.appendInst(block, bool_t, .{ .icmp = .{
            .op = if (level == 0) .eq else .ne,
            .lhs = spsel,
            .rhs = zero,
        } });
        const el1_selected = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_and, .lhs = at_el1, .rhs = selected } });
        return if (level == 0)
            func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_or, .lhs = at_level, .rhs = el1_selected } })
        else
            el1_selected;
    }
    return at_level;
}

fn systemAddress(func: *Function, block: ir.function.Block, cpu_ptr: Value, offset: usize, u64_t: ir.types.Type, ptr_t: ir.types.Type) Error!Value {
    const delta = try func.appendInst(block, u64_t, .{ .iconst = @intCast(@offsetOf(Cpu, "system") + offset) });
    return func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = cpu_ptr, .rhs = delta } });
}

fn loadField(func: *Function, block: ir.function.Block, cpu_ptr: Value, offset: usize, ty: ir.types.Type, ptr_t: ir.types.Type) Error!Value {
    const delta = try func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = cpu_ptr, .rhs = try func.appendInst(block, ty, .{ .iconst = @intCast(offset) }) } });
    return func.appendInst(block, ty, .{ .load = .{ .ptr = delta } });
}

fn advanceCounter(func: *Function, block: ir.function.Block, cpu_ptr: Value, count: u64, u64_t: ir.types.Type, ptr_t: ir.types.Type) Error!void {
    if (count == 0) return;
    const at = @offsetOf(Cpu, "system") + @offsetOf(Cpu.System, "cntvct_el0");
    const counter = try loadField(func, block, cpu_ptr, at, u64_t, ptr_t);
    try storeField(func, block, cpu_ptr, at, try func.appendArithImm(block, u64_t, .add, counter, @intCast(count)), u64_t, ptr_t);
}

fn regAddress(func: *Function, block: ir.function.Block, cpu_ptr: Value, reg: u32, u64_t: ir.types.Type, ptr_t: ir.types.Type) std.mem.Allocator.Error!Value {
    const byte_offset: i64 = if (reg == 31)
        @intCast(@offsetOf(Cpu, "sp"))
    else
        @intCast(@offsetOf(Cpu, "x") + @as(usize, reg) * @sizeOf(u64));
    const offset = try func.appendInst(block, u64_t, .{ .iconst = byte_offset });
    return func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = cpu_ptr, .rhs = offset } });
}

/// What the block being translated already holds of each register, as the value
/// loaded or stored last. A block is one straight run of IR, so a value defined
/// earlier is the register's value until something else is stored to it, and
/// reading it again is a use rather than another trip to memory. Reset at the
/// start of each translation, and one translation runs on a thread at a time.
threadlocal var known: [32]?Value = @splat(null);

fn loadReg(func: *Function, block: ir.function.Block, cpu_ptr: Value, reg: u32, ty: ir.types.Type, u64_t: ir.types.Type, ptr_t: ir.types.Type) std.mem.Allocator.Error!Value {
    const full = known[reg] orelse load: {
        const address = try regAddress(func, block, cpu_ptr, reg, u64_t, ptr_t);
        const loaded = try func.appendInst(block, u64_t, .{ .load = .{ .ptr = address } });
        known[reg] = loaded;
        break :load loaded;
    };
    if (func.valueType(full) == ty) return full;
    return func.appendInst(block, ty, .{ .convert = .{ .value = full } });
}

fn storeReg(func: *Function, block: ir.function.Block, cpu_ptr: Value, reg: u32, value: Value, u64_t: ir.types.Type, ptr_t: ir.types.Type) std.mem.Allocator.Error!void {
    const address = try regAddress(func, block, cpu_ptr, reg, u64_t, ptr_t);
    const value_ty = func.valueType(value);
    const full = if (value_ty == u64_t) value else try func.appendInst(block, u64_t, .{ .convert = .{ .value = value } });
    try func.appendStore(block, full, address);
    known[reg] = full;
}

fn storeField(func: *Function, block: ir.function.Block, cpu_ptr: Value, offset: usize, value: Value, u64_t: ir.types.Type, ptr_t: ir.types.Type) std.mem.Allocator.Error!void {
    const delta = try func.appendInst(block, u64_t, .{ .iconst = @intCast(offset) });
    const address = try func.appendInst(block, ptr_t, .{ .arith = .{ .op = .add, .lhs = cpu_ptr, .rhs = delta } });
    try func.appendStore(block, value, address);
}

fn storeFieldConst(func: *Function, block: ir.function.Block, cpu_ptr: Value, offset: usize, value: u64, ty: ir.types.Type, u64_t: ir.types.Type, ptr_t: ir.types.Type) std.mem.Allocator.Error!void {
    const constant = try func.appendInst(block, ty, .{ .iconst = @bitCast(value) });
    try storeField(func, block, cpu_ptr, offset, constant, u64_t, ptr_t);
}

/// The shifts by a register and the divides. The two families share a class and
/// nothing else, so they are two computations behind one dispatch.
///
/// A shift takes its distance modulo the register width, which x86 also does for
/// a variable shift but which is stated here rather than relied on. A divide by
/// zero gives zero on AArch64 and traps on x86, and so does the one quotient that
/// does not fit, the most negative value over minus one, which AArch64 defines as
/// the dividend itself. Both are steered around before the host divide runs.
fn lowerVariable(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    variable: Decode.Variable,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (variable.width == .x64) u64_t else u32_t;
    const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = @backingInt(variable.width) } });
    const bits: i64 = @backingInt(variable.width);
    const lhs = try loadRegOrZero(func, block, cpu_ptr, variable.rn, ty, u64_t, ptr_t);
    const rhs = try loadRegOrZero(func, block, cpu_ptr, variable.rm, ty, u64_t, ptr_t);

    const result = switch (variable.op) {
        .lsl, .lsr, .asr, .ror => blk: {
            const distance = try func.appendInst(block, ty, .{ .arith = .{
                .op = .bit_and,
                .lhs = rhs,
                .rhs = try func.appendInst(block, ty, .{ .iconst = bits - 1 }),
            } });
            break :blk switch (variable.op) {
                .lsl => try func.appendInst(block, ty, .{ .arith = .{ .op = .shl, .lhs = lhs, .rhs = distance } }),
                .lsr => try func.appendInst(block, ty, .{ .arith = .{ .op = .shr, .lhs = lhs, .rhs = distance } }),
                .asr => asr: { // an arithmetic shift is a signed one, so the width has to say so
                    const shifted = try func.appendInst(block, signed, .{ .arith = .{
                        .op = .shr,
                        .lhs = try func.appendInst(block, signed, .{ .convert = .{ .value = lhs } }),
                        .rhs = try func.appendInst(block, signed, .{ .convert = .{ .value = distance } }),
                    } });
                    break :asr try func.appendInst(block, ty, .{ .convert = .{ .value = shifted } });
                },
                .ror => ror: {
                    // The bits that leave on the right come back in on the left. The
                    // way back is masked too, so a distance of zero shifts by zero
                    // rather than by the full width.
                    const back = try func.appendInst(block, ty, .{ .arith = .{
                        .op = .bit_and,
                        .lhs = try func.appendInst(block, ty, .{ .arith = .{
                            .op = .sub,
                            .lhs = try func.appendInst(block, ty, .{ .iconst = bits }),
                            .rhs = distance,
                        } }),
                        .rhs = try func.appendInst(block, ty, .{ .iconst = bits - 1 }),
                    } });
                    break :ror try func.appendInst(block, ty, .{ .arith = .{
                        .op = .bit_or,
                        .lhs = try func.appendInst(block, ty, .{ .arith = .{ .op = .shr, .lhs = lhs, .rhs = distance } }),
                        .rhs = try func.appendInst(block, ty, .{ .arith = .{ .op = .shl, .lhs = lhs, .rhs = back } }),
                    } });
                },
                else => unreachable,
            };
        },
        .udiv, .sdiv => blk: {
            const zero = try func.appendInst(block, ty, .{ .iconst = 0 });
            const one = try func.appendInst(block, ty, .{ .iconst = 1 });
            const by_zero = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .eq, .lhs = rhs, .rhs = zero } });
            if (variable.op == .udiv) {
                const divisor = try func.appendInst(block, ty, .{ .select = .{ .cond = by_zero, .then = one, .@"else" = rhs } });
                const quotient = try func.appendInst(block, ty, .{ .arith = .{ .op = .div, .lhs = lhs, .rhs = divisor } });
                break :blk try func.appendInst(block, ty, .{ .select = .{ .cond = by_zero, .then = zero, .@"else" = quotient } });
            }
            // Minus one is every bit set, which is the same test at either width.
            const by_minus_one = try func.appendInst(block, bool_t, .{ .icmp = .{
                .op = .eq,
                .lhs = rhs,
                .rhs = try func.appendInst(block, ty, .{ .iconst = -1 }),
            } });
            const safe = try func.appendInst(block, ty, .{ .select = .{
                .cond = by_minus_one,
                .then = one,
                .@"else" = try func.appendInst(block, ty, .{ .select = .{ .cond = by_zero, .then = one, .@"else" = rhs } }),
            } });
            const quotient = try func.appendInst(block, ty, .{ .convert = .{ .value = try func.appendInst(block, signed, .{ .arith = .{
                .op = .div,
                .lhs = try func.appendInst(block, signed, .{ .convert = .{ .value = lhs } }),
                .rhs = try func.appendInst(block, signed, .{ .convert = .{ .value = safe } }),
            } }) } });
            // Over minus one the answer is the dividend negated, which wraps to
            // itself for the most negative value, as the architecture says.
            const negated = try func.appendInst(block, ty, .{ .arith = .{ .op = .sub, .lhs = zero, .rhs = lhs } });
            break :blk try func.appendInst(block, ty, .{ .select = .{
                .cond = by_zero,
                .then = zero,
                .@"else" = try func.appendInst(block, ty, .{ .select = .{ .cond = by_minus_one, .then = negated, .@"else" = quotient } }),
            } });
        },
    };
    // Register 31 as a destination is the zero register: the result is discarded,
    // not written to the stack pointer that `storeReg` would name.
    if (variable.rd != 31) try storeReg(func, block, cpu_ptr, variable.rd, result, u64_t, ptr_t);
}

/// The low `len` bits set, for `len` from 1 to 64.
fn lowMask(len: u7) u64 {
    return if (len >= 64) std.math.maxInt(u64) else (@as(u64, 1) << @intCast(len)) - 1;
}

/// One of `UBFM`, `SBFM` and `BFM`. A field is a run of `len` bits at `lsb`: in
/// the source when the encoding extracts, and in the result when it inserts.
///
/// Extracting is a left shift that puts the top of the field at the top of the
/// register followed by a right shift that brings it down, which is logical for
/// an unsigned field and arithmetic for a signed one, so the sign extension and
/// the zero extension are the same two instructions apart from the second one.
fn lowerBitfield(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    field: Decode.Bitfield,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (field.width == .x64) u64_t else u32_t;
    const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = @backingInt(field.width) } });
    const bits: i64 = @backingInt(field.width);
    const len: i64 = field.len;
    const lsb: i64 = field.lsb;
    const source = try loadRegOrZero(func, block, cpu_ptr, field.rn, ty, u64_t, ptr_t);
    // A width mask is meaningful only at the register's own width, so the constant
    // is cut to it here rather than at every use.
    const width_mask: u64 = if (field.width == .x64) std.math.maxInt(u64) else std.math.maxInt(u32);

    const result: Value = switch (field.op) {
        .unsigned, .signed => blk: {
            if (field.extract) {
                const raised = try shiftLeft(func, block, ty, source, bits - lsb - len);
                break :blk try shiftRight(func, block, ty, signed, raised, bits - len, field.op == .signed);
            }
            if (field.op == .unsigned) {
                const kept = try func.appendInst(block, ty, .{ .arith = .{
                    .op = .bit_and,
                    .lhs = source,
                    .rhs = try func.appendInst(block, ty, .{ .iconst = @bitCast(lowMask(field.len) & width_mask) }),
                } });
                break :blk try shiftLeft(func, block, ty, kept, lsb);
            }
            // Sign-extend the field where it is, and then move it up.
            const raised = try shiftLeft(func, block, ty, source, bits - len);
            const extended = try shiftRight(func, block, ty, signed, raised, bits - len, true);
            break :blk try shiftLeft(func, block, ty, extended, lsb);
        },
        .insert => blk: {
            // The bits of the field, in the place they end up.
            const value = if (field.extract) v: {
                const raised = try shiftLeft(func, block, ty, source, bits - lsb - len);
                break :v try shiftRight(func, block, ty, signed, raised, bits - len, false);
            } else v: {
                const kept = try func.appendInst(block, ty, .{ .arith = .{
                    .op = .bit_and,
                    .lhs = source,
                    .rhs = try func.appendInst(block, ty, .{ .iconst = @bitCast(lowMask(field.len) & width_mask) }),
                } });
                break :v try shiftLeft(func, block, ty, kept, lsb);
            };
            // An extract puts the field at the bottom, an insert at `lsb`.
            const place: u6 = if (field.extract) 0 else field.lsb;
            const hole = (lowMask(field.len) << place) & width_mask;
            const destination = try loadRegOrZero(func, block, cpu_ptr, field.rd, ty, u64_t, ptr_t);
            const cleared = try func.appendInst(block, ty, .{ .arith = .{
                .op = .bit_and,
                .lhs = destination,
                .rhs = try func.appendInst(block, ty, .{ .iconst = @bitCast(~hole & width_mask) }),
            } });
            break :blk try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_or, .lhs = cleared, .rhs = value } });
        },
    };
    // Register 31 as a destination is the zero register: the result is discarded.
    if (field.rd != 31) try storeReg(func, block, cpu_ptr, field.rd, result, u64_t, ptr_t);
}

/// `value << distance`, which is `value` when the distance is zero.
fn shiftLeft(func: *Function, block: ir.function.Block, ty: ir.types.Type, value: Value, distance: i64) Error!Value {
    if (distance == 0) return value;
    return func.appendArithImm(block, ty, .shl, value, distance);
}

/// `value >> distance`, arithmetic when `arithmetic` is set. An arithmetic shift is
/// a signed one, so the width has to say so.
fn shiftRight(
    func: *Function,
    block: ir.function.Block,
    ty: ir.types.Type,
    signed: ir.types.Type,
    value: Value,
    distance: i64,
    arithmetic: bool,
) Error!Value {
    if (distance == 0) return value;
    if (!arithmetic) return func.appendArithImm(block, ty, .shr, value, distance);
    const as_signed = try func.appendInst(block, signed, .{ .convert = .{ .value = value } });
    const shifted = try func.appendArithImm(block, signed, .shr, as_signed, distance);
    return func.appendInst(block, ty, .{ .convert = .{ .value = shifted } });
}

/// `(value & mask) << shift | (value >> shift) & mask`: the two halves of every
/// pair of neighbouring groups of `shift` bits change places. Applied with masks
/// and distances that double, it turns a register inside out one level at a time,
/// which is how a byte or bit reversal is built without an instruction for it.
fn swapGroups(func: *Function, block: ir.function.Block, ty: ir.types.Type, value: Value, mask: u64, shift: i64) Error!Value {
    const mask_value = try func.appendInst(block, ty, .{ .iconst = @bitCast(mask) });
    const low = try func.appendInst(block, ty, .{ .arith = .{ .op = .bit_and, .lhs = value, .rhs = mask_value } });
    const high = try func.appendInst(block, ty, .{ .arith = .{
        .op = .bit_and,
        .lhs = try func.appendArithImm(block, ty, .shr, value, shift),
        .rhs = mask_value,
    } });
    return func.appendInst(block, ty, .{ .arith = .{
        .op = .bit_or,
        .lhs = try func.appendArithImm(block, ty, .shl, low, shift),
        .rhs = high,
    } });
}

/// `pattern` repeated across a register of `bits` bits, given as the value in the
/// low `unit` bits.
fn repeated(pattern: u64, unit: u7, bits: u7) u64 {
    var result: u64 = 0;
    var at: u7 = 0;
    while (at < bits) : (at += unit) result |= pattern << @intCast(at);
    return result;
}

/// One of `RBIT`, `REV16`, `REV32`, `REV` and `CLZ`.
///
/// There is no instruction in the IR for any of them, so each is built from what
/// there is. A byte reversal swaps neighbouring bytes, then neighbouring halfwords,
/// then neighbouring words, stopping at the size the instruction reverses within.
/// A bit reversal swaps neighbouring bits, pairs and nibbles, and then reverses the
/// bytes. Counting leading zeros looks at the top half of what is left, and when it
/// is empty counts it and shifts it out, halving the size of what it looks at.
fn lowerUnary(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    unary: Decode.Unary,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (unary.width == .x64) u64_t else u32_t;
    const bits: u7 = @intCast(@backingInt(unary.width));
    var value = try loadRegOrZero(func, block, cpu_ptr, unary.rn, ty, u64_t, ptr_t);

    switch (unary.op) {
        .rev16 => value = try swapGroups(func, block, ty, value, repeated(0x00ff, 16, bits), 8),
        .rev32 => {
            value = try swapGroups(func, block, ty, value, repeated(0x00ff, 16, bits), 8);
            value = try swapGroups(func, block, ty, value, repeated(0xffff, 32, bits), 16);
        },
        .rev, .rbit => {
            if (unary.op == .rbit) {
                value = try swapGroups(func, block, ty, value, repeated(0x1, 2, bits), 1);
                value = try swapGroups(func, block, ty, value, repeated(0x3, 4, bits), 2);
                value = try swapGroups(func, block, ty, value, repeated(0xf, 8, bits), 4);
            }
            value = try swapGroups(func, block, ty, value, repeated(0x00ff, 16, bits), 8);
            value = try swapGroups(func, block, ty, value, repeated(0xffff, 32, bits), 16);
            // At 64 bits the two words change places too, and no mask is needed
            // because each half is the whole of what it moves.
            if (bits == 64) {
                value = try func.appendInst(block, ty, .{ .arith = .{
                    .op = .bit_or,
                    .lhs = try func.appendArithImm(block, ty, .shl, value, 32),
                    .rhs = try func.appendArithImm(block, ty, .shr, value, 32),
                } });
            }
        },
        .clz => {
            const original = value;
            var count = try func.appendInst(block, ty, .{ .iconst = 0 });
            var step: u7 = bits / 2;
            while (step >= 1) : (step /= 2) {
                // Are the top `step` bits of what is left all zero? If so, count them
                // and move them out of the way.
                const top = try func.appendArithImm(block, ty, .shr, value, bits - step);
                const empty = try func.appendInst(block, bool_t, .{ .icmp = .{
                    .op = .eq,
                    .lhs = top,
                    .rhs = try func.appendInst(block, ty, .{ .iconst = 0 }),
                } });
                value = try func.appendInst(block, ty, .{ .select = .{
                    .cond = empty,
                    .then = try func.appendArithImm(block, ty, .shl, value, step),
                    .@"else" = value,
                } });
                count = try func.appendInst(block, ty, .{ .select = .{
                    .cond = empty,
                    .then = try func.appendArithImm(block, ty, .add, count, step),
                    .@"else" = count,
                } });
            }
            // A register of nothing but zeros shifts out every step and still counts
            // one short, because the last bit is never looked at. It is the width.
            const nothing = try func.appendInst(block, bool_t, .{ .icmp = .{
                .op = .eq,
                .lhs = original,
                .rhs = try func.appendInst(block, ty, .{ .iconst = 0 }),
            } });
            value = try func.appendInst(block, ty, .{ .select = .{
                .cond = nothing,
                .then = try func.appendInst(block, ty, .{ .iconst = bits }),
                .@"else" = count,
            } });
        },
    }
    // Register 31 as a destination is the zero register: the result is discarded.
    if (unary.rd != 31) try storeReg(func, block, cpu_ptr, unary.rd, value, u64_t, ptr_t);
}

/// `ADD`/`SUB` with an extended second operand. The operand is extended in a full
/// register whatever the width of the operation, and cut down after: the low bits
/// of a sign extension are the register's own bits, so a 32-bit operation sees the
/// same 32 bits either way.
///
/// Register 31 is the stack pointer as the first operand, and as the destination
/// unless the flags are being set, when it is the zero register and the result is
/// discarded. As the second operand it is the zero register, which is what makes
/// this the instruction that reaches the stack pointer with `cmp sp, x1`.
fn lowerArithExtended(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    arith: Decode.ArithmeticExtended,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (arith.width == .x64) u64_t else u32_t;
    const source = try loadRegOrZero(func, block, cpu_ptr, arith.rm, u64_t, u64_t, ptr_t);
    const extended = try extendValue(func, block, u64_t, source, arith.extend);
    const shifted = try shiftLeft(func, block, u64_t, extended, arith.amount);
    const rhs = if (arith.width == .x64)
        shifted
    else
        try func.appendInst(block, u32_t, .{ .convert = .{ .value = shifted } });

    const lhs = try loadReg(func, block, cpu_ptr, arith.rn, ty, u64_t, ptr_t);
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = if (arith.op == .sub) .sub else .add, .lhs = lhs, .rhs = rhs } });
    if (!arith.flags or arith.rd != 31) try storeReg(func, block, cpu_ptr, arith.rd, result, u64_t, ptr_t);
    if (arith.flags) try setFlags(func, block, cpu_ptr, arith.op, arith.width, lhs, rhs, result, u32_t, bool_t, u64_t, ptr_t);
}

/// `source` extended to a full register as `extend` says: a mask for a zero
/// extension, and for a sign extension the field is raised to the top of the
/// register and brought back down arithmetically. The two that name the whole
/// register leave it as it is. Used by the arithmetic form that extends its second
/// operand and by the loads and stores that extend their index.
fn extendValue(func: *Function, block: ir.function.Block, u64_t: ir.types.Type, source: Value, extend: Decode.Extend) Error!Value {
    const signed = try func.types.intern(.{ .int = .{ .signedness = .signed, .bits = 64 } });
    return switch (extend) {
        .uxtb, .uxth, .uxtw => try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .bit_and,
            .lhs = source,
            .rhs = try func.appendInst(block, u64_t, .{ .iconst = @as(i64, switch (extend) {
                .uxtb => 0xff,
                .uxth => 0xffff,
                else => 0xffff_ffff,
            }) }),
        } }),
        .uxtx, .sxtx => source,
        .sxtb, .sxth, .sxtw => blk: {
            const drop: i64 = switch (extend) {
                .sxtb => 56,
                .sxth => 48,
                else => 32,
            };
            const raised = try shiftLeft(func, block, u64_t, source, drop);
            break :blk try shiftRight(func, block, u64_t, signed, raised, drop, true);
        },
    };
}

/// `SMADDL` and its relatives. Each 32-bit operand is extended to a full register,
/// which is a sign extension of the word or a mask of it, and the product is taken
/// in 64 bits. Only the low 64 bits of a product are kept, and for two operands
/// that fit in 32 bits, signed or not, those are the whole product, so one
/// multiplication serves both.
fn lowerMultiplyLong(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    product: Decode.MultiplyLong,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const extend: Decode.Extend = if (product.signed) .sxtw else .uxtw;
    const lhs = try extendValue(func, block, u64_t, try loadRegOrZero(func, block, cpu_ptr, product.rn, u64_t, u64_t, ptr_t), extend);
    const rhs = try extendValue(func, block, u64_t, try loadRegOrZero(func, block, cpu_ptr, product.rm, u64_t, u64_t, ptr_t), extend);
    const wide = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = lhs, .rhs = rhs } });
    // Register 31 as the accumulator is zero: an add leaves the product alone, and a
    // subtract negates it.
    const value = if (product.ra == 31 and product.op == .add) wide else blk: {
        const accumulator = try loadRegOrZero(func, block, cpu_ptr, product.ra, u64_t, u64_t, ptr_t);
        break :blk try func.appendInst(block, u64_t, .{ .arith = .{
            .op = if (product.op == .sub) .sub else .add,
            .lhs = accumulator,
            .rhs = wide,
        } });
    };
    if (product.rd != 31) try storeReg(func, block, cpu_ptr, product.rd, value, u64_t, ptr_t);
}

/// `SMULH` and `UMULH`: the top 64 bits of the 128-bit product.
///
/// The IR has a `mulh`, but this host backend does not lower it, so the high half
/// is built the way it is on paper. Split each operand into 32-bit halves and the
/// product is four partial products, each exact in 64 bits, shifted into place:
/// `a*b = a1*b1<<64 + (a1*b0 + a0*b1)<<32 + a0*b0`. The middle terms carry into the
/// top, and adding the low halves of each partial product and shifting is how that
/// carry is found, and it cannot overflow: three 32-bit values sum to under 2^34.
///
/// The signed product differs from the unsigned one only by treating a negative
/// operand as its value minus 2^64. That subtracts the other operand from the high
/// half once for each operand that is negative, and nothing else changes.
fn lowerMultiplyHigh(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    product: Decode.MultiplyHigh,
    u64_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const a = try loadRegOrZero(func, block, cpu_ptr, product.rn, u64_t, u64_t, ptr_t);
    const b = try loadRegOrZero(func, block, cpu_ptr, product.rm, u64_t, u64_t, ptr_t);
    const mask = try func.appendInst(block, u64_t, .{ .iconst = 0xffff_ffff });
    const a_low = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = mask } });
    const b_low = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = b, .rhs = mask } });
    const a_high = try func.appendArithImm(block, u64_t, .shr, a, 32);
    const b_high = try func.appendArithImm(block, u64_t, .shr, b, 32);

    const low_low = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = a_low, .rhs = b_low } });
    const low_high = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = a_low, .rhs = b_high } });
    const high_low = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = a_high, .rhs = b_low } });
    const high_high = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .mul, .lhs = a_high, .rhs = b_high } });

    // The carry into the top half: what the three terms that meet at bit 32 add to.
    const middle = try func.appendInst(block, u64_t, .{ .arith = .{
        .op = .add,
        .lhs = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .add,
            .lhs = try func.appendArithImm(block, u64_t, .shr, low_low, 32),
            .rhs = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = low_high, .rhs = mask } }),
        } }),
        .rhs = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = high_low, .rhs = mask } }),
    } });
    var high = try func.appendInst(block, u64_t, .{ .arith = .{
        .op = .add,
        .lhs = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .add,
            .lhs = high_high,
            .rhs = try func.appendArithImm(block, u64_t, .shr, low_high, 32),
        } }),
        .rhs = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .add,
            .lhs = try func.appendArithImm(block, u64_t, .shr, high_low, 32),
            .rhs = try func.appendArithImm(block, u64_t, .shr, middle, 32),
        } }),
    } });

    if (product.signed) {
        // `0 - (x >> 63)` is all ones when `x` is negative and zero when it is not, so
        // masking the other operand with it subtracts that operand only when it is due.
        const zero = try func.appendInst(block, u64_t, .{ .iconst = 0 });
        const a_negative = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .sub,
            .lhs = zero,
            .rhs = try func.appendArithImm(block, u64_t, .shr, a, 63),
        } });
        const b_negative = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .sub,
            .lhs = zero,
            .rhs = try func.appendArithImm(block, u64_t, .shr, b, 63),
        } });
        high = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .sub,
            .lhs = high,
            .rhs = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = b, .rhs = a_negative } }),
        } });
        high = try func.appendInst(block, u64_t, .{ .arith = .{
            .op = .sub,
            .lhs = high,
            .rhs = try func.appendInst(block, u64_t, .{ .arith = .{ .op = .bit_and, .lhs = a, .rhs = b_negative } }),
        } });
    }
    if (product.rd != 31) try storeReg(func, block, cpu_ptr, product.rd, high, u64_t, ptr_t);
}

/// `ADC`, `SBC` and their flag-setting forms. Both are one addition of three things:
/// the first operand, the second (inverted for a subtract, since `a - b` is
/// `a + ~b + 1`), and the carry flag. The carry flag means "no borrow" for a
/// subtract, so it goes in as it is for both.
///
/// The carry out cannot be read from the result alone once there is a carry in:
/// `3 + max + 1` wraps to `3` and has carried, which is indistinguishable from
/// `3 + 0 + 0`. So it is found in two steps, one for each addition, and carried if
/// either wrapped. N, Z and V do not depend on the carry-in and are computed as for
/// an ordinary add of the two operands.
fn lowerArithCarry(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    arith: Decode.AddCarry,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (arith.width == .x64) u64_t else u32_t;
    const lhs = try loadRegOrZero(func, block, cpu_ptr, arith.rn, ty, u64_t, ptr_t);
    const loaded = try loadRegOrZero(func, block, cpu_ptr, arith.rm, ty, u64_t, ptr_t);
    const rhs = if (arith.op == .sub)
        try func.appendInst(block, ty, .{ .arith = .{
            .op = .bit_xor,
            .lhs = loaded,
            .rhs = try func.appendInst(block, ty, .{ .iconst = -1 }),
        } })
    else
        loaded;

    // The carry flag is bit 29 of the flags, as a value of the operation's own width.
    const flags = try loadField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), u32_t, ptr_t);
    const carry_bit = try func.appendArithImm(block, u32_t, .bit_and, try func.appendArithImm(block, u32_t, .shr, flags, 29), 1);
    const carry_in = if (arith.width == .x64)
        try func.appendInst(block, u64_t, .{ .convert = .{ .value = carry_bit } })
    else
        carry_bit;

    const partial = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = lhs, .rhs = rhs } });
    const result = try func.appendInst(block, ty, .{ .arith = .{ .op = .add, .lhs = partial, .rhs = carry_in } });
    if (arith.rd != 31) try storeReg(func, block, cpu_ptr, arith.rd, result, u64_t, ptr_t);

    if (arith.flags) {
        const first = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = partial, .rhs = lhs } });
        const second = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .lt, .lhs = result, .rhs = partial } });
        const carry_out = try func.appendInst(block, bool_t, .{ .arith = .{ .op = .bit_or, .lhs = first, .rhs = second } });
        // The overflow rule is the addition's, of the two operands as they were added.
        const packed_flags = try flagsOf(func, block, .add, arith.width, lhs, rhs, result, carry_out, u32_t, bool_t, u64_t);
        try storeField(func, block, cpu_ptr, @offsetOf(Cpu, "flags"), packed_flags, u64_t, ptr_t);
    }
}

/// `EXTR`: `Rm` shifted right by `lsb`, with the bits that fall off the bottom of `Rn`
/// coming in at the top, that is `(Rm >> lsb) | (Rn << (width - lsb))`. A distance of
/// zero is `Rm` as it is, because shifting left by the whole width is not defined,
/// and it is not an operation this needs to perform.
fn lowerExtract(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    pair: Decode.Extract,
    u64_t: ir.types.Type,
    u32_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    const ty = if (pair.width == .x64) u64_t else u32_t;
    const low = try loadRegOrZero(func, block, cpu_ptr, pair.rm, ty, u64_t, ptr_t);
    const value = if (pair.lsb == 0) low else blk: {
        const high = try loadRegOrZero(func, block, cpu_ptr, pair.rn, ty, u64_t, ptr_t);
        const width_bits: i64 = @backingInt(pair.width);
        const rest: i64 = width_bits - @as(i64, pair.lsb);
        break :blk try func.appendInst(block, ty, .{ .arith = .{
            .op = .bit_or,
            .lhs = try func.appendArithImm(block, ty, .shr, low, pair.lsb),
            .rhs = try func.appendArithImm(block, ty, .shl, high, rest),
        } });
    };
    if (pair.rd != 31) try storeReg(func, block, cpu_ptr, pair.rd, value, u64_t, ptr_t);
}

/// The virtual timer's control with its condition recomputed: bit 2 set when the
/// counter has reached the compare, held from whenever that last happened rather
/// than only while the two are equal. A read is the one place the condition is
/// observed, so recomputing it here keeps every read current without touching
/// the stored enable and mask.
fn timerControl(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    u64_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!Value {
    const at = @offsetOf(Cpu, "system");
    const counter = try loadField(func, block, cpu_ptr, at + @offsetOf(Cpu.System, "cntvct_el0"), u64_t, ptr_t);
    const compare = try loadField(func, block, cpu_ptr, at + @offsetOf(Cpu.System, "cntv_cval_el0"), u64_t, ptr_t);
    const reached = try func.appendInst(block, bool_t, .{ .icmp = .{ .op = .ge, .lhs = counter, .rhs = compare } });
    const stored = try loadField(func, block, cpu_ptr, at + @offsetOf(Cpu.System, "cntv_ctl_el0"), u64_t, ptr_t);
    return func.appendInst(block, u64_t, .{ .arith = .{
        .op = .bit_or,
        .lhs = stored,
        .rhs = try func.appendInst(block, u64_t, .{ .select = .{ .cond = reached, .then = try func.appendInst(block, u64_t, .{ .iconst = 0b100 }), .@"else" = try func.appendInst(block, u64_t, .{ .iconst = 0 }) } }),
    } });
}

/// Recompute nothing: the condition is derived on read, so a write to the compare
/// leaves nothing stale. Kept as the one place that knows that, so a cached
/// condition later has exactly one call site to update.
fn refreshTimer(
    func: *Function,
    block: ir.function.Block,
    cpu_ptr: Value,
    u64_t: ir.types.Type,
    bool_t: ir.types.Type,
    ptr_t: ir.types.Type,
) Error!void {
    _ = func;
    _ = block;
    _ = cpu_ptr;
    _ = u64_t;
    _ = bool_t;
    _ = ptr_t;
}

test "a block lifts into one function that returns the next pc" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    // movz x0, #1 ; b .+8
    const words = [_]u32{ 0xD2800020, 0x14000002 };
    try region(&func, 0x1000, std.mem.sliceAsBytes(&words));
    try std.testing.expect(func.blockCount() >= 1);
}

test "an empty or ragged block is refused" {
    var func = Function.init(std.testing.allocator);
    defer func.deinit();
    try std.testing.expectError(error.EmptyBlock, region(&func, 0, &.{}));
    try std.testing.expectError(error.UnsupportedInstruction, region(&func, 0, &.{ 0, 0, 0 }));
}
