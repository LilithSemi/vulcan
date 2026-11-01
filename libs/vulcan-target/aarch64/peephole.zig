//! AArch64 post-allocation peepholes. `decodeMem` recognizes the bounded set of
//! offset-form ldr/str encodings the isel emits (see encode.zig), and `tryFuse` decides whether
//! two adjacent decoded memory ops may be combined into a single ldp/stp. Neither function
//! walks or mutates a code buffer. Wiring this into an actual pass over emitted code is Task 3.
//!
//! A wrong decode or a wrong fuse here would silently miscompile once Task 3's pass trusts it,
//! so `decodeMem` must be the EXACT inverse of the isel's encoders (verified against
//! strOff/ldrOff/strW/ldrW/strFp/ldrFp/strQ/ldrQ in encode.zig, see the tests), and anything
//! that is not one of those exact forms (a different addressing mode, a pre/post-index form, a
//! non-memory op) must decode to null rather than guess.

const std = @import("std");
const encode = @import("encode.zig");
const isel = @import("isel.zig");
const mm = @import("vulcan-opt").microarch;
const Reg = encode.Reg;

/// The side-table row types the fusion pass remaps. Taken directly from the isel so a fusion
/// updates the exact structures `compileFunction` resolves branches and emits debug info from.
const Fixup = isel.Fixup;
const Reloc = isel.Reloc;
const LineEntry = isel.LineEntry;

pub const MachineModel = struct {
    issue_width: u8 = 1,
    units: mm.Units = .{},
    latency: Timing = .{},
    throughput: Timing = .{},
};

pub const Timing = struct {
    alu: u8 = 1,
    mul: u8 = 1,
    div: u8 = 1,
    mem: u8 = 1,
    branch: u8 = 1,
    fpsimd: u8 = 1,
};

pub const MachineScore = struct {
    cycles: u32,
    critical_path: u32,
    instructions: u32,
    idle_slots: u32,

    pub fn accepts(candidate: MachineScore, baseline: MachineScore) bool {
        if (candidate.cycles != baseline.cycles) return candidate.cycles < baseline.cycles;
        if (candidate.critical_path != baseline.critical_path) return candidate.critical_path < baseline.critical_path;
        if (candidate.instructions != baseline.instructions) return candidate.instructions < baseline.instructions;
        return candidate.idle_slots <= baseline.idle_slots;
    }
};

const MachineClass = enum { alu, mul, div, mem, branch, fpsimd };

pub fn scoreWords(words: []const u32, model: MachineModel) MachineScore {
    var demand: [@typeInfo(MachineClass).@"enum".field_names.len]u32 = @splat(0);
    var total_demand: u32 = 0;
    // x0..x30, v0..v31, and NZCV each own a scoreboard slot. NZCV must not alias v31:
    // compare-to-branch and compare-to-select chains are real machine dependencies.
    var ready: [65]u32 = @splat(0);
    var critical_path: u32 = 0;
    for (words) |word| {
        const class = machineClass(word);
        const weight = timing(model.throughput, class);
        demand[@backingInt(class)] += weight;
        total_demand += weight;

        const registers = dependencies(word, class);
        var starts_at: u32 = 0;
        for (registers.reads) |reg| {
            if (reg) |r| starts_at = @max(starts_at, ready[r]);
        }
        const finishes_at = starts_at + timing(model.latency, class);
        for (registers.writes) |reg| {
            if (reg) |r| ready[r] = finishes_at;
        }
        critical_path = @max(critical_path, finishes_at);
    }

    const width: u32 = @max(1, model.issue_width);
    var resource_cycles = ceilDiv(total_demand, width);
    inline for (0..demand.len) |index| {
        const class: MachineClass = @fromBackingInt(@intCast(index));
        resource_cycles = @max(resource_cycles, ceilDiv(demand[index], ports(model, class)));
    }
    const cycles = @max(critical_path, resource_cycles);
    return .{
        .cycles = cycles,
        .critical_path = critical_path,
        .instructions = @intCast(words.len),
        .idle_slots = cycles * width -| total_demand,
    };
}

fn machineClass(word: u32) MachineClass {
    if (decodeMem(word) != null or isPairMemory(word) or isRegisterMemory(word)) return .mem;
    if (isDivide(word)) return .div;
    if (word & 0x7F000000 == 0x1B000000) return .mul;
    if (word & 0x7C000000 == 0x14000000 or word & 0xFF000010 == 0x54000000) return .branch;
    if (word & 0x0E000000 == 0x0E000000 or word & 0x1F000000 == 0x1E000000) return .fpsimd;
    return .alu;
}

const RegisterDeps = struct {
    reads: [4]?u7 = @splat(null),
    writes: [3]?u7 = @splat(null),
};

fn dependencies(word: u32, class: MachineClass) RegisterDeps {
    if (decodeMem(word)) |mem| {
        const data = if (mem.kind == .ldr_s or mem.kind == .str_s or mem.kind == .ldr_d or mem.kind == .str_d or mem.kind == .ldr_q or mem.kind == .str_q)
            fpReg(mem.rt)
        else
            gpReg(mem.rt);
        if (isLoad(mem.kind)) return .{ .reads = .{ gpReg(mem.rn), null, null, null }, .writes = .{ data, null, null } };
        return .{ .reads = .{ gpReg(mem.rn), data, null, null } };
    }
    if (decodePairMemory(word)) |pair| {
        const load = word & 0x00400000 != 0;
        const rt1: u5 = @truncate(word);
        const rt2: u5 = @truncate(word >> 10);
        const rn: u5 = @truncate(word >> 5);
        const data1 = if (pair.simd) fpReg(rt1) else gpReg(rt1);
        const data2 = if (pair.simd) fpReg(rt2) else gpReg(rt2);
        if (load) return .{ .reads = .{ gpReg(rn), null, null, null }, .writes = .{ data1, data2, if (pair.writeback) gpReg(rn) else null } };
        return .{ .reads = .{ gpReg(rn), data1, data2, null }, .writes = .{ if (pair.writeback) gpReg(rn) else null, null, null } };
    }
    if (isRegisterMemory(word)) {
        const rt: u5 = @truncate(word);
        const rn: u5 = @truncate(word >> 5);
        const rm: u5 = @truncate(word >> 16);
        return .{ .reads = .{ gpReg(rn), gpReg(rm), null, null }, .writes = .{ gpReg(rt), null, null } };
    }
    if (class == .mul) {
        const ra: u5 = @truncate(word >> 10);
        return .{
            .reads = .{ gpReg(@truncate(word >> 5)), gpReg(@truncate(word >> 16)), gpReg(ra), null },
            .writes = .{ gpReg(@truncate(word)), null, null },
        };
    }
    if (class == .div or isRegisterAlu(word)) {
        var result = RegisterDeps{
            .reads = .{
                gpReg(@truncate(word >> 5)),
                gpReg(@truncate(word >> 16)),
                if (readsFlags(word)) nzcvReg() else null,
                null,
            },
            .writes = .{ gpReg(@truncate(word)), null, null },
        };
        if (writesFlags(word)) result.writes[1] = nzcvReg();
        return result;
    }
    if (isImmediateAlu(word)) {
        var result = RegisterDeps{
            .reads = .{ gpReg(@truncate(word >> 5)), null, null, null },
            .writes = .{ gpReg(@truncate(word)), null, null },
        };
        if (writesFlags(word)) result.writes[1] = nzcvReg();
        return result;
    }
    if (isMoveWide(word)) return .{
        .reads = .{ gpReg(@truncate(word)), null, null, null },
        .writes = .{ gpReg(@truncate(word)), null, null },
    };
    if (vectorDependencies(word)) |result| return result;
    if (isConditionalBranch(word)) return .{ .reads = .{ nzcvReg(), null, null, null } };
    return .{};
}

fn gpReg(reg: u5) ?u7 {
    return if (reg == 31) null else reg;
}

fn fpReg(reg: u5) u7 {
    return 32 + @as(u7, reg);
}

fn nzcvReg() u7 {
    return 64;
}

fn isRegisterAlu(word: u32) bool {
    return switch (word & 0x1F000000) {
        0x0A000000, 0x0B000000, 0x1A000000 => true,
        else => false,
    };
}

fn isImmediateAlu(word: u32) bool {
    return word & 0x1F000000 == 0x11000000 or // add/sub immediate, including flag-setting forms
        word & 0x1F800000 == 0x12000000 or // logical immediate
        word & 0x1F800000 == 0x13000000; // bitfield
}

fn isConditionalBranch(word: u32) bool {
    return word & 0xFF000010 == 0x54000000;
}

fn readsFlags(word: u32) bool {
    // Integer conditional select family: CSEL, CSINC, CSINV and CSNEG (including CNEG).
    return word & 0x1FE00000 == 0x1A800000;
}

fn writesFlags(word: u32) bool {
    if ((word & 0x1F000000) == 0x0B000000 or (word & 0x1F000000) == 0x11000000)
        return word & 0x20000000 != 0; // ADDS/SUBS, register or immediate
    if ((word & 0x1F000000) == 0x0A000000 or (word & 0x1F800000) == 0x12000000)
        return word & 0x60000000 == 0x60000000; // ANDS, register or immediate
    return false;
}

fn vectorDependencies(word: u32) ?RegisterDeps {
    const form = word & 0xFFE0FC00;
    const accumulates = switch (form) {
        0x4EA09400, // MLA Vd.4S, Vn.4S, Vm.4S
        0x4E809400, // SDOT Vd.4S, Vn.16B, Vm.16B
        0x6E809400, // UDOT Vd.4S, Vn.16B, Vm.16B
        0x6E601C00, // BSL Vd.16B, Vn.16B, Vm.16B
        => true,
        else => false,
    };
    switch (form) {
        0x4EA08400, // ADD Vd.4S, Vn.4S, Vm.4S
        0x6EA08400, // SUB Vd.4S, Vn.4S, Vm.4S
        0x4EA09C00, // MUL Vd.4S, Vn.4S, Vm.4S
        0x4EA09400,
        0x4E201C00, // AND Vd.16B, Vn.16B, Vm.16B
        0x4EA01C00, // ORR/MOV Vd.16B, Vn.16B, Vm.16B
        0x6E201C00, // EOR Vd.16B, Vn.16B, Vm.16B
        0x6EA04400, // USHL Vd.4S, Vn.4S, Vm.4S
        0x4EA04400, // SSHL Vd.4S, Vn.4S, Vm.4S
        0x4E809400,
        0x6E809400,
        0x6E601C00,
        => return .{
            .reads = .{
                fpReg(@truncate(word >> 5)),
                fpReg(@truncate(word >> 16)),
                if (accumulates) fpReg(@truncate(word)) else null,
                null,
            },
            .writes = .{ fpReg(@truncate(word)), null, null },
        },
        0x4E000400, // DUP Vd.4S, Vn.S[index]
        0x5E000400, // DUP Sd, Vn.S[index]
        => return .{
            .reads = .{ fpReg(@truncate(word >> 5)), null, null, null },
            .writes = .{ fpReg(@truncate(word)), null, null },
        },
        0x6E000400, // INS Vd.S[index], Vn.S[0]
        => return .{
            .reads = .{ fpReg(@truncate(word >> 5)), fpReg(@truncate(word)), null, null },
            .writes = .{ fpReg(@truncate(word)), null, null },
        },
        else => {},
    }
    if (word & 0xFFFFFC00 == 0x4E040C00) return .{
        .reads = .{ gpReg(@truncate(word >> 5)), null, null, null },
        .writes = .{ fpReg(@truncate(word)), null, null },
    };
    if (word & 0xFF00FC00 == 0x4F005400) return .{
        .reads = .{ fpReg(@truncate(word >> 5)), null, null, null },
        .writes = .{ fpReg(@truncate(word)), null, null },
    };
    return null;
}

fn isMoveWide(word: u32) bool {
    return word & 0x1F800000 == 0x12800000;
}

fn isRegisterMemory(word: u32) bool {
    const form = word & 0xFFE0FC00;
    return form == 0xB8607800 or form == 0x38606800;
}

fn isDivide(word: u32) bool {
    const form = word & 0x7FE0FC00;
    return form == 0x1AC00C00 or form == 0x1AC00800;
}

const PairMemory = struct { simd: bool, writeback: bool = false };

fn decodePairMemory(word: u32) ?PairMemory {
    return switch (word & 0xFFC00000) {
        0xA9400000, 0xA9000000, 0x29400000, 0x29000000 => .{ .simd = false },
        0xAD400000, 0xAD000000 => .{ .simd = true },
        0xACC00000, 0xAC800000 => .{ .simd = true, .writeback = true },
        else => null,
    };
}

fn isPairMemory(word: u32) bool {
    return decodePairMemory(word) != null;
}

fn ports(model: MachineModel, class: MachineClass) u32 {
    return @max(1, switch (class) {
        .alu => model.units.alu,
        .mul, .div => model.units.muldiv,
        .mem => model.units.mem,
        .branch => model.units.branch,
        .fpsimd => model.units.fpsimd,
    });
}

fn timing(values: Timing, class: MachineClass) u32 {
    return switch (class) {
        .alu => values.alu,
        .mul => values.mul,
        .div => values.div,
        .mem => values.mem,
        .branch => values.branch,
        .fpsimd => values.fpsimd,
    };
}

fn ceilDiv(numerator: u32, denominator: u32) u32 {
    return (numerator + denominator - 1) / denominator;
}

/// The offset-form ldr/str encodings the aarch64 isel emits. Anything else is opaque to this
/// peephole and decodes to null.
pub const MemKind = enum { ldr_x, str_x, ldr_w, str_w, ldr_s, str_s, ldr_d, str_d, ldr_q, str_q };

/// A decoded memory instruction. `off` is the BYTE offset: the raw u12 scaled-imm field already
/// multiplied out by the element size, so two decoded insns can be compared directly regardless
/// of kind.
pub const MemInsn = struct { kind: MemKind, rt: u5, rn: u5, off: u32 };

/// The element size in bytes for `kind` (also the scale of its imm12 field).
fn sizeOf(kind: MemKind) u32 {
    return switch (kind) {
        .ldr_x, .str_x, .ldr_d, .str_d => 8,
        .ldr_w, .str_w, .ldr_s, .str_s => 4,
        .ldr_q, .str_q => 16,
    };
}

/// True for the `ldr_*` variants. A load writes `rt` from memory, whereas a store reads `rt`
/// into memory and never overwrites a register.
fn isLoad(kind: MemKind) bool {
    return switch (kind) {
        .ldr_x, .ldr_w, .ldr_s, .ldr_d, .ldr_q => true,
        .str_x, .str_w, .str_s, .str_d, .str_q => false,
    };
}

/// The top-10-bit class selector every recognized offset-form encoding uses (bits [31:22]);
/// the low 22 bits carry imm12/rn/rt and are zero in each base opcode below. Pre/post-index and
/// register-offset addressing modes for the same size/class live at a different bit-24 value
/// and so never collide with these classes.
const class_mask: u32 = 0xFFC00000;

/// Decode `word` if it is one of the recognized offset-form ldr/str encodings, else null.
/// Classes below are cross-checked one-for-one against encode.zig:
///   ldr_x/str_x -> ldrOff/strOff (imm12 scaled by 8)
///   ldr_w/str_w -> ldrW/strW     (imm12 scaled by 4)
///   ldr_s/str_s -> ldrFp/strFp(dbl=false) (imm12 scaled by 4)
///   ldr_d/str_d -> ldrFp/strFp(dbl=true)  (imm12 scaled by 8)
///   ldr_q/str_q -> ldrQ/strQ     (imm12 scaled by 16)
pub fn decodeMem(word: u32) ?MemInsn {
    const rt: u5 = @truncate(word & 0x1F);
    const rn: u5 = @truncate((word >> 5) & 0x1F);
    const imm12: u32 = (word >> 10) & 0xFFF;
    const kind: MemKind = switch (word & class_mask) {
        0xF9400000 => .ldr_x,
        0xF9000000 => .str_x,
        0xB9400000 => .ldr_w,
        0xB9000000 => .str_w,
        0xBD400000 => .ldr_s,
        0xBD000000 => .str_s,
        0xFD400000 => .ldr_d,
        0xFD000000 => .str_d,
        0x3DC00000 => .ldr_q,
        0x3D800000 => .str_q,
        else => return null,
    };
    return MemInsn{ .kind = kind, .rt = rt, .rn = rn, .off = imm12 * sizeOf(kind) };
}

/// Given two adjacent decoded mem insns (`a` then `b` in program order), return the fused
/// ldp/stp word if they may be paired, else null. Requires ALL of:
///  - `a.kind == b.kind` (same load/store, class, and size).
///  - `a.rn == b.rn` (same base register).
///  - `b.off == a.off + sizeOf(a.kind)` (`b` sits exactly one element above `a`).
///  - if `a`/`b` are loads: `a.rt != b.rt` (an ldp with equal destinations is UNPREDICTABLE) and
///    `a.rt != a.rn` (the first load must not overwrite the base register the second load still
///    needs). A fused ldp computes both addresses from the pre-instruction base, so this also
///    keeps the fused form execution-equivalent to the original two-instruction sequence.
///  - `a.off / sizeOf(a.kind) <= 63` (the ldp/stp imm7 is a signed 7-bit field). `a.off` is
///    derived from an unsigned u12 scaled-imm so it is always >= 0, making the range check just
///    this upper bound.
///
/// Scalar-float s/d pairs remain deferred. Q pairs use the validated SIMD register-pair encoding.
pub fn tryFuse(a: MemInsn, b: MemInsn) ?u32 {
    if (a.kind != b.kind) return null;
    if (a.rn != b.rn) return null;
    const size = sizeOf(a.kind);
    if (b.off != a.off + size) return null;
    if (isLoad(a.kind)) {
        if (a.rt == b.rt) return null;
        if ((a.kind == .ldr_x or a.kind == .ldr_w) and a.rt == a.rn) return null;
    }
    const imm7: u32 = a.off / size;
    if (imm7 > 63) return null;

    const rt1: Reg = @fromBackingInt(@intCast(a.rt));
    const rt2: Reg = @fromBackingInt(@intCast(b.rt));
    const rn: Reg = @fromBackingInt(@intCast(a.rn));
    const imm: i16 = @intCast(a.off);
    return switch (a.kind) {
        .ldr_x => encode.ldpOffX(rt1, rt2, rn, imm),
        .str_x => encode.stpOffX(rt1, rt2, rn, imm),
        .ldr_w => encode.ldpOffW(rt1, rt2, rn, imm),
        .str_w => encode.stpOffW(rt1, rt2, rn, imm),
        .ldr_q => encode.ldpOffQ(rt1, rt2, rn, imm),
        .str_q => encode.stpOffQ(rt1, rt2, rn, imm),
        .ldr_s, .str_s, .ldr_d, .str_d => null,
    };
}

fn tryPostIndexQ(pair_word: u32, add_word: u32) ?u32 {
    const pair_form = pair_word & 0xFFC00000;
    if (pair_form != 0xAD400000 and pair_form != 0xAD000000) return null;
    if ((pair_word >> 15) & 0x7F != 0) return null;
    if (add_word & 0xFFC00000 != 0x91000000) return null;
    const base: u5 = @truncate(pair_word >> 5);
    const add_source: u5 = @truncate(add_word >> 5);
    const add_dest: u5 = @truncate(add_word);
    const immediate = (add_word >> 10) & 0xFFF;
    if (add_source != base or add_dest != base or immediate != 32) return null;
    const first: Reg = @fromBackingInt(@as(u5, @truncate(pair_word)));
    const second: Reg = @fromBackingInt(@as(u5, @truncate(pair_word >> 10)));
    const base_reg: Reg = @fromBackingInt(base);
    return if (pair_form == 0xAD400000)
        encode.ldpPostQ(first, second, base_reg, 32)
    else
        encode.stpPostQ(first, second, base_reg, 32);
}

/// Fuse an adjacent integer `mul` and register `add` when the add consumes the product exactly
/// once. Physical-register matching after allocation makes this safe without extending the SSA
/// operands' live ranges: `madd` reads the multiply operands and the add's other input together.
pub fn tryFuseMulAdd(mul_word: u32, add_word: u32) ?u32 {
    const operand_mask: u32 = 0x7FE0FC00;
    if (mul_word & operand_mask != 0x1B007C00) return null;
    if (add_word & operand_mask != 0x0B000000) return null;
    const mul_wide = mul_word >> 31 != 0;
    if ((add_word >> 31 != 0) != mul_wide) return null;

    const product: u5 = @truncate(mul_word);
    const add_lhs: u5 = @truncate(add_word >> 5);
    const add_rhs: u5 = @truncate(add_word >> 16);
    if ((add_lhs == product) == (add_rhs == product)) return null;

    const rd: Reg = @fromBackingInt(@as(u5, @truncate(add_word)));
    const rn: Reg = @fromBackingInt(@as(u5, @truncate(mul_word >> 5)));
    const rm: Reg = @fromBackingInt(@as(u5, @truncate(mul_word >> 16)));
    const ra: Reg = @fromBackingInt(if (add_lhs == product) add_rhs else add_lhs);
    return encode.madd(rd, rn, rm, ra, mul_wide);
}

/// Fold `producer tmp; mov dst,tmp` into `producer dst`. The accepted producer families all write
/// only Rd and permit Rd to alias any input register. This is especially useful on CFG edges where
/// linear-scan liveness keeps an input live through mutually exclusive sibling blocks and therefore
/// cannot coalesce the result with the successor parameter during allocation.
pub fn tryCoalesceResultMove(producer: u32, move: u32) ?u32 {
    const move_class = move & 0xFFE0FFE0;
    if (move_class != 0xAA0003E0 and move_class != 0x2A0003E0) return null;

    const source: u5 = @truncate(move >> 16);
    const destination: u5 = @truncate(move);
    const producer_destination: u5 = @truncate(producer);
    if (source == 31 or destination == 31 or producer_destination != source) return null;

    const rewritable = producer & 0x1F200000 == 0x0B000000 or // add/sub shifted register
        producer & 0x1F000000 == 0x0A000000 or // logical shifted register
        producer & 0x1F000000 == 0x11000000 or // add/sub immediate
        producer & 0x1F800000 == 0x12000000 or // logical immediate
        producer & 0x1F800000 == 0x13000000 or // bitfield
        producer & 0x1FE00000 == 0x1A800000 or // conditional select
        producer & 0x1F000000 == 0x1B000000; // multiply-add family
    if (!rewritable) return null;

    return producer & ~@as(u32, 0x1F) | destination;
}

const DistantResultMove = struct {
    at: [12]usize = @splat(0),
    words: [12]u32 = @splat(0),
    len: usize = 0,
    baseline: [13]u32 = @splat(0),
    baseline_len: usize = 0,
};

fn tryCoalesceDistantResultMove(
    words: []const u32,
    removed: []const bool,
    is_block_start: []const bool,
    block_start: []const usize,
    fixups: []const Fixup,
    move_at: usize,
) ?DistantResultMove {
    if (move_at >= words.len or removed[move_at]) return null;
    const move = words[move_at];
    if (move & 0xFFE0FFE0 != 0xAA0003E0) return null;
    const source: u5 = @truncate(move >> 16);
    const destination: u5 = @truncate(move);
    if (source == 31 or destination == 31 or source == destination) return null;

    var producer_at = move_at;
    var scanned: usize = 0;
    while (producer_at > 0 and scanned < 12) {
        producer_at -= 1;
        if (removed[producer_at]) continue;
        if (is_block_start[producer_at + 1] and !hasOnlyFallthroughEntry(producer_at + 1, block_start, fixups)) return null;
        scanned += 1;

        const deps = dependencies(words[producer_at], machineClass(words[producer_at]));
        if (containsReg(deps.writes, source)) break;
        if (containsReg(deps.writes, destination) or containsReg(deps.reads, destination)) return null;
        _ = retargetReads(words[producer_at], source, destination) orelse return null;
    } else return null;

    const producer = tryCoalesceResultMove(words[producer_at], move) orelse return null;
    var result = DistantResultMove{};
    for (producer_at..move_at + 1) |at| {
        if (removed[at]) continue;
        result.baseline[result.baseline_len] = words[at];
        result.baseline_len += 1;
        if (at == move_at) continue;
        result.at[result.len] = at;
        result.words[result.len] = if (at == producer_at)
            producer
        else
            retargetReads(words[at], source, destination) orelse return null;
        result.len += 1;
    }
    return result;
}

fn retargetReads(word: u32, source: u5, destination: u5) ?u32 {
    const class = machineClass(word);
    const deps = dependencies(word, class);
    if (containsReg(deps.writes, source) or containsReg(deps.writes, destination) or containsReg(deps.reads, destination)) return null;

    var result = word;
    if (class == .mul) {
        result = replaceRegField(result, 5, source, destination);
        result = replaceRegField(result, 16, source, destination);
        return replaceRegField(result, 10, source, destination);
    }
    if (class == .div or isRegisterAlu(word)) {
        result = replaceRegField(result, 5, source, destination);
        return replaceRegField(result, 16, source, destination);
    }
    if (isImmediateAlu(word)) return replaceRegField(result, 5, source, destination);
    if (isMoveWide(word)) return result;
    return null;
}

fn replaceRegField(word: u32, shift: u5, source: u5, destination: u5) u32 {
    if (@as(u5, @truncate(word >> shift)) != source) return word;
    const mask = @as(u32, 0x1F) << shift;
    return word & ~mask | (@as(u32, destination) << shift);
}

fn containsReg(registers: anytype, reg: u5) bool {
    for (registers) |candidate| if (candidate != null and candidate.? == reg) return true;
    return false;
}

fn moveRunHasCycle(words: []const u32, move_indices: []const usize) bool {
    var reach: [31]u32 = @splat(0);
    for (move_indices) |at| {
        const move = words[at];
        if (move & 0xFFE0FFE0 != 0xAA0003E0) return true;
        const source: u5 = @truncate(move >> 16);
        const destination: u5 = @truncate(move);
        if (source == 31 or destination == 31) return true;
        reach[source] |= @as(u32, 1) << destination;
    }
    for (0..31) |through| {
        for (0..31) |source| {
            if (reach[source] & (@as(u32, 1) << @intCast(through)) != 0) reach[source] |= reach[through];
        }
    }
    for (0..31) |reg| {
        if (reach[reg] & (@as(u32, 1) << @intCast(reg)) != 0) return true;
    }
    return false;
}

/// A move that coincides with an IR block start can still be physically inside its predecessor when
/// empty blocks share a code offset. No branch or dense-switch row may enter at that offset before a
/// producer from the fallthrough path is rewritten across it.
fn hasOnlyFallthroughEntry(move_at: usize, block_start: []const usize, fixups: []const Fixup) bool {
    for (block_start, 0..) |start, block_index| {
        if (start != move_at) continue;
        for (fixups) |fixup| {
            if (fixup.adr_target == null and fixup.target == block_index) return false;
        }
    }
    return true;
}

const NarrowMagic = struct { words: [2]u32 };

fn tryNarrowUnsignedMagic(words: []const u32) ?NarrowMagic {
    if (words.len < 4) return null;
    const extend = words[0];
    if (extend & 0xFFFFFC00 != 0xD3407C00) return null;
    const product = words[1];
    if (product & 0xFFE0FC00 != 0x9B007C00) return null;

    const wide_value: u5 = @truncate(extend);
    const source: u5 = @truncate(extend >> 5);
    const product_dest: u5 = @truncate(product);
    const product_lhs: u5 = @truncate(product >> 5);
    const multiplier: u5 = @truncate(product >> 16);
    if (wide_value != product_dest or wide_value != product_lhs) return null;

    const first_shift = decodeLsr64(words[2]) orelse return null;
    const second_shift = decodeLsr64(words[3]) orelse return null;
    if (first_shift.reg != product_dest or first_shift.shift != 32) return null;
    if (second_shift.reg != product_dest or second_shift.shift == 0 or second_shift.shift > 31) return null;

    const rd: Reg = @fromBackingInt(product_dest);
    return .{ .words = .{
        encode.umull(rd, @fromBackingInt(source), @fromBackingInt(multiplier)),
        encode.lsrImm(rd, rd, 32 + second_shift.shift, true),
    } };
}

fn decodeLsr64(word: u32) ?struct { reg: u5, shift: u6 } {
    if (word & 0xFFC0FC00 != 0xD340FC00) return null;
    const rd: u5 = @truncate(word);
    const rn: u5 = @truncate(word >> 5);
    if (rd != rn) return null;
    return .{ .reg = rd, .shift = @truncate(word >> 16) };
}

fn narrowUnsignedMagic(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize,
    fixups: []Fixup,
    relocs: []Reloc,
    lines: []LineEntry,
    coalescable_result_moves: []usize,
    dead_source_result_moves: []usize,
    model: MachineModel,
) std.mem.Allocator.Error!void {
    const len = words.items.len;
    var is_block_start = try allocator.alloc(bool, len);
    defer allocator.free(is_block_start);
    @memset(is_block_start, false);
    for (block_start) |start| if (start < len) {
        is_block_start[start] = true;
    };
    var is_data = try allocator.alloc(bool, len);
    defer allocator.free(is_data);
    @memset(is_data, false);
    for (fixups) |fixup| {
        if (fixup.table_base != null) is_data[fixup.at] = true;
    }

    var removed_before = try allocator.alloc(u32, len + 1);
    defer allocator.free(removed_before);
    var write: usize = 0;
    var removed: u32 = 0;
    var i: usize = 0;
    while (i < len) {
        const rewrite = if (i + 3 < len and !is_data[i] and !is_data[i + 1] and !is_data[i + 2] and !is_data[i + 3] and
            !is_block_start[i + 1] and !is_block_start[i + 2] and !is_block_start[i + 3])
            tryNarrowUnsignedMagic(words.items[i..])
        else
            null;
        if (rewrite) |candidate| {
            if (scoreWords(&candidate.words, model).accepts(scoreWords(words.items[i .. i + 4], model))) {
                removed_before[i] = removed;
                words.items[write] = candidate.words[0];
                write += 1;
                removed_before[i + 1] = removed;
                words.items[write] = candidate.words[1];
                write += 1;
                removed += 1;
                removed_before[i + 2] = removed;
                removed += 1;
                removed_before[i + 3] = removed;
                i += 4;
                continue;
            }
        }
        removed_before[i] = removed;
        words.items[write] = words.items[i];
        write += 1;
        i += 1;
    }
    removed_before[len] = removed;

    for (block_start) |*start| start.* -= removed_before[start.*];
    for (fixups) |*fixup| {
        if (fixup.table_base) |base| fixup.table_base = base - removed_before[base];
        if (fixup.adr_target) |target| fixup.adr_target = target - removed_before[target];
        fixup.at -= removed_before[fixup.at];
    }
    for (relocs) |*reloc| reloc.offset -= removed_before[reloc.offset];
    for (coalescable_result_moves) |*move| move.* -= removed_before[move.*];
    for (dead_source_result_moves) |*move| move.* -= removed_before[move.*];
    for (lines) |*line| {
        const word = line.offset / 4;
        line.offset = @intCast((word - removed_before[word]) * 4);
    }
    words.shrinkRetainingCapacity(write);
}

/// Fuse adjacent ldr/str pairs into ldp/stp in `words`, updating the side tables to the new
/// layout. Only fuses IMMEDIATELY-ADJACENT words where both decode as mem, `tryFuse` succeeds,
/// AND the SECOND word is not a block start (control flow must not be able to enter between the
/// pair). Fusion deletes the second word (a load/store, never a branch or reloc site) and
/// replaces the first with the fused ldp/stp. Mutates `words` (shrinks it) and remaps
/// `block_start`, `fixups`, `relocs`, `lines` in place.
///
/// Miscompile surface: the fixup-resolution loop in `compileFunction` runs AFTER this pass over
/// the remapped tables, so branch displacements are recomputed from the shrunk layout. A word
/// referenced by a block start, a fixup, or a reloc is never the deleted (second) member of a
/// pair, so its remapped index is always well-defined and points at a surviving word.
pub fn pairMemory(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize, // WORD indices, remapped in place
    fixups: []Fixup, // .at WORD indices, remapped in place (.target block index unchanged)
    relocs: []Reloc, // .offset WORD indices, remapped in place
    lines: []LineEntry, // .offset BYTE offsets, remapped in place
) std.mem.Allocator.Error!void {
    return pairMemoryForModel(allocator, words, block_start, fixups, relocs, lines, .{});
}

pub fn pairMemoryForModel(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize,
    fixups: []Fixup,
    relocs: []Reloc,
    lines: []LineEntry,
    model: MachineModel,
) std.mem.Allocator.Error!void {
    return pairMemoryForModelWithMoves(allocator, words, block_start, fixups, relocs, lines, &.{}, model);
}

pub fn pairMemoryForModelWithMoves(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize,
    fixups: []Fixup,
    relocs: []Reloc,
    lines: []LineEntry,
    coalescable_result_moves: []const usize,
    model: MachineModel,
) std.mem.Allocator.Error!void {
    return pairMemoryForModelWithMoveProofs(allocator, words, block_start, fixups, relocs, lines, coalescable_result_moves, &.{}, model);
}

pub fn pairMemoryForModelWithMoveProofs(
    allocator: std.mem.Allocator,
    words: *std.ArrayList(u32),
    block_start: []usize,
    fixups: []Fixup,
    relocs: []Reloc,
    lines: []LineEntry,
    coalescable_result_moves: []const usize,
    dead_source_result_moves: []const usize,
    model: MachineModel,
) std.mem.Allocator.Error!void {
    const remapped_result_moves = try allocator.dupe(usize, coalescable_result_moves);
    defer allocator.free(remapped_result_moves);
    const remapped_dead_source_moves = try allocator.dupe(usize, dead_source_result_moves);
    defer allocator.free(remapped_dead_source_moves);
    try narrowUnsignedMagic(allocator, words, block_start, fixups, relocs, lines, remapped_result_moves, remapped_dead_source_moves, model);
    const len = words.items.len;

    // Which word indices begin a block. A word is fusable-as-second only if it is NOT a block
    // start, so that control flow can never enter between the two members of a pair.
    var is_block_start = try allocator.alloc(bool, len);
    defer allocator.free(is_block_start);
    @memset(is_block_start, false);
    for (block_start) |bs| {
        std.debug.assert(bs <= len);
        if (bs < len) is_block_start[bs] = true; // a block start == len emits nothing to fuse into
    }
    var is_data = try allocator.alloc(bool, len);
    defer allocator.free(is_data);
    @memset(is_data, false);
    for (fixups) |fixup| {
        if (fixup.table_base != null) is_data[fixup.at] = true;
    }
    var is_coalescable_result_move = try allocator.alloc(bool, len);
    defer allocator.free(is_coalescable_result_move);
    @memset(is_coalescable_result_move, false);
    for (remapped_result_moves) |at| {
        std.debug.assert(at < len);
        is_coalescable_result_move[at] = true;
    }
    var has_dead_source = try allocator.alloc(bool, len);
    defer allocator.free(has_dead_source);
    @memset(has_dead_source, false);
    for (remapped_dead_source_moves) |at| {
        std.debug.assert(at < len);
        has_dead_source[at] = true;
    }
    const removed_distant = try allocator.alloc(bool, len);
    defer allocator.free(removed_distant);
    @memset(removed_distant, false);
    const cyclic_move = try allocator.alloc(bool, len);
    defer allocator.free(cyclic_move);
    @memset(cyclic_move, false);
    var run_start: usize = 0;
    while (run_start < remapped_dead_source_moves.len) {
        var run_end = run_start + 1;
        while (run_end < remapped_dead_source_moves.len and remapped_dead_source_moves[run_end] == remapped_dead_source_moves[run_end - 1] + 1) : (run_end += 1) {}
        const run = remapped_dead_source_moves[run_start..run_end];
        if (moveRunHasCycle(words.items, run)) {
            for (run) |at| cyclic_move[at] = true;
        }
        run_start = run_end;
    }
    for (remapped_dead_source_moves) |move_at| {
        if (cyclic_move[move_at]) continue;
        const rewrite = tryCoalesceDistantResultMove(words.items, removed_distant, is_block_start, block_start, fixups, move_at) orelse continue;
        const baseline = scoreWords(rewrite.baseline[0..rewrite.baseline_len], model);
        const candidate = scoreWords(rewrite.words[0..rewrite.len], model);
        if (!candidate.accepts(baseline)) continue;
        for (rewrite.at[0..rewrite.len], rewrite.words[0..rewrite.len]) |at, word| words.items[at] = word;
        removed_distant[move_at] = true;
    }

    // Normalize two interleaved Q-load streams before looking for their pointer updates. Keeping
    // both resulting pairs in the first two slots lets the post-index scan below reason about each
    // base independently while the ordinary compactor removes the consumed trailing loads.
    if (len >= 4) for (0..len - 3) |at| {
        if (removed_distant[at] or removed_distant[at + 1] or removed_distant[at + 2] or removed_distant[at + 3] or
            is_block_start[at + 1] or is_block_start[at + 2] or is_block_start[at + 3] or
            is_data[at] or is_data[at + 1] or is_data[at + 2] or is_data[at + 3]) continue;
        const a0 = decodeMem(words.items[at]) orelse continue;
        const b0 = decodeMem(words.items[at + 1]) orelse continue;
        const a1 = decodeMem(words.items[at + 2]) orelse continue;
        const b1 = decodeMem(words.items[at + 3]) orelse continue;
        if (a0.kind != .ldr_q or b0.kind != .ldr_q or a1.kind != .ldr_q or b1.kind != .ldr_q) continue;
        const pair_a = tryFuse(a0, a1) orelse continue;
        const pair_b = tryFuse(b0, b1) orelse continue;
        const baseline = scoreWords(words.items[at .. at + 4], model);
        const candidate = scoreWords(&.{ pair_a, pair_b }, model);
        if (!candidate.accepts(baseline)) continue;
        words.items[at] = pair_a;
        words.items[at + 1] = pair_b;
        removed_distant[at + 2] = true;
        removed_distant[at + 3] = true;
    };

    // A carried vector pointer commonly reaches the machine stream as either two adjacent Q loads
    // or an already-paired `ldp q..., [base]`, followed by a few independent instructions and then
    // `add base, base, #32`. Pair and fold the update in one scored rewrite when nothing between
    // them observes or changes `base`. This is deliberately a physical-register rewrite: exact
    // register hazards and the final machine score are both known.
    for (0..len) |load_at| {
        if (removed_distant[load_at] or is_data[load_at]) continue;
        var pair_word: u32 = undefined;
        var pair_second: ?usize = null;
        if (decodePairMemory(words.items[load_at])) |pair| {
            const pair_form = words.items[load_at] & 0xFFC00000;
            if (!pair.simd or pair.writeback or (pair_form != 0xAD400000 and pair_form != 0xAD000000)) continue;
            pair_word = words.items[load_at];
        } else {
            const second_at = load_at + 1;
            if (second_at >= len or removed_distant[second_at] or is_block_start[second_at] or is_data[second_at]) continue;
            const first = decodeMem(words.items[load_at]) orelse continue;
            const second = decodeMem(words.items[second_at]) orelse continue;
            if ((first.kind != .ldr_q and first.kind != .str_q) or second.kind != first.kind) continue;
            pair_word = tryFuse(first, second) orelse continue;
            pair_second = second_at;
        }
        const base: u5 = @truncate(pair_word >> 5);
        var add_at = (pair_second orelse load_at) + 1;
        while (add_at < len and add_at - load_at <= 16) : (add_at += 1) {
            if (removed_distant[add_at]) continue;
            if (is_block_start[add_at] or is_data[add_at]) break;
            if (tryPostIndexQ(pair_word, words.items[add_at])) |post| {
                var baseline_words: [17]u32 = undefined;
                var candidate_words: [16]u32 = undefined;
                var baseline_len: usize = 0;
                var candidate_len: usize = 0;
                for (load_at..add_at + 1) |at| {
                    if (removed_distant[at]) continue;
                    baseline_words[baseline_len] = words.items[at];
                    baseline_len += 1;
                    if (at == add_at or at == pair_second) continue;
                    candidate_words[candidate_len] = if (at == load_at) post else words.items[at];
                    candidate_len += 1;
                }
                if (scoreWords(candidate_words[0..candidate_len], model).accepts(scoreWords(baseline_words[0..baseline_len], model))) {
                    words.items[load_at] = post;
                    if (pair_second) |second_at| removed_distant[second_at] = true;
                    removed_distant[add_at] = true;
                }
                break;
            }
            const class = machineClass(words.items[add_at]);
            if (class == .branch) break;
            const deps = dependencies(words.items[add_at], class);
            if (containsReg(deps.reads, base) or containsReg(deps.writes, base)) break;
        }
    }

    // `removed_before[w]` = number of words deleted with index strictly before `w`, with the twist
    // that a deleted word's own slot records the count INCLUDING its own deletion. That makes a
    // surviving word `w` map to `w - removed_before[w]` and a deleted (second-of-pair) word collapse
    // onto the fused instruction that replaced its pair's first word. Length is `len + 1` so a
    // block start sitting exactly at the end (`== len`) still has a defined slot.
    var removed_before = try allocator.alloc(u32, len + 1);
    defer allocator.free(removed_before);

    var write: usize = 0; // in-place compaction cursor (always <= the read cursor `i`)
    var removed: u32 = 0;
    var i: usize = 0;
    while (i < len) {
        if (removed_distant[i]) {
            removed += 1;
            removed_before[i] = removed;
            i += 1;
            continue;
        }
        // The scheduler may interleave two independent vector streams as A0, B0, A1, B1 to start
        // both dependency chains early. Four adjacent loads may be regrouped as pair(A0, A1),
        // pair(B0, B1): no write or observable instruction is crossed, and the strict pair
        // legality plus machine score still decide whether the rewrite commits.
        if (i + 3 < len and
            !removed_distant[i + 1] and !removed_distant[i + 2] and !removed_distant[i + 3] and
            !is_block_start[i + 1] and !is_block_start[i + 2] and !is_block_start[i + 3] and
            !is_data[i] and !is_data[i + 1] and !is_data[i + 2] and !is_data[i + 3])
        interleaved: {
            const a0 = decodeMem(words.items[i]) orelse break :interleaved;
            const b0 = decodeMem(words.items[i + 1]) orelse break :interleaved;
            const a1 = decodeMem(words.items[i + 2]) orelse break :interleaved;
            const b1 = decodeMem(words.items[i + 3]) orelse break :interleaved;
            if (!isLoad(a0.kind) or !isLoad(b0.kind) or !isLoad(a1.kind) or !isLoad(b1.kind)) break :interleaved;
            const pair_a = tryFuse(a0, a1) orelse break :interleaved;
            const pair_b = tryFuse(b0, b1) orelse break :interleaved;
            const baseline = scoreWords(words.items[i .. i + 4], model);
            const candidate = scoreWords(&.{ pair_a, pair_b }, model);
            if (!candidate.accepts(baseline)) break :interleaved;

            removed_before[i] = removed;
            removed_before[i + 1] = removed;
            removed += 1;
            removed_before[i + 2] = removed;
            removed += 1;
            removed_before[i + 3] = removed;
            words.items[write] = pair_a;
            words.items[write + 1] = pair_b;
            write += 2;
            i += 4;
            continue;
        }
        var did_fuse = false;
        // Memory and multiply-add pairs cannot cross a block start. A proven result move may cross
        // a coincident, unentered start because `hasOnlyFallthroughEntry` rules out another path.
        // Never chain a third word into the same pair (i += 2).
        if (i + 1 < len and !removed_distant[i + 1] and !is_data[i] and !is_data[i + 1]) {
            const result_entry_safe = !is_block_start[i + 1] or hasOnlyFallthroughEntry(i + 1, block_start, fixups);
            const result_proven = is_coalescable_result_move[i + 1] or has_dead_source[i + 1];
            const fused = (if (result_entry_safe and result_proven) tryCoalesceResultMove(words.items[i], words.items[i + 1]) else null) orelse
                (if (!is_block_start[i + 1]) tryFuseMulAdd(words.items[i], words.items[i + 1]) else null) orelse memory: {
                if (is_block_start[i + 1]) break :memory null;
                const a = decodeMem(words.items[i]) orelse break :memory null;
                const b = decodeMem(words.items[i + 1]) orelse break :memory null;
                break :memory tryFuse(a, b);
            };
            if (fused) |word| {
                const baseline = scoreWords(words.items[i .. i + 2], model);
                const candidate = scoreWords(&.{word}, model);
                if (candidate.accepts(baseline)) {
                    removed_before[i] = removed;
                    removed += 1;
                    removed_before[i + 1] = removed; // deleted slot collapses onto the fused word
                    words.items[write] = word;
                    write += 1;
                    i += 2;
                    did_fuse = true;
                }
            }
        }
        if (!did_fuse) {
            removed_before[i] = removed;
            words.items[write] = words.items[i];
            write += 1;
            i += 1;
        }
    }
    removed_before[len] = removed;

    // The surviving count is exactly the original minus every deleted second-of-pair word.
    std.debug.assert(write == len - removed);

    // Remap the side tables into the shrunk layout. Each keyed index is a surviving word, so its
    // new index is `w - removed_before[w]` and lands below the new word count.
    for (block_start) |*bs| {
        bs.* -= removed_before[bs.*];
        std.debug.assert(bs.* <= write);
    }
    for (fixups) |*f| {
        if (f.table_base) |base| f.table_base = base - removed_before[base];
        if (f.adr_target) |target| f.adr_target = target - removed_before[target];
        f.at -= removed_before[f.at];
        std.debug.assert(f.at < write);
    }
    for (relocs) |*r| {
        r.offset -= removed_before[r.offset];
        std.debug.assert(r.offset < write);
    }
    for (lines) |*l| {
        // Line offsets are BYTES; a line that sat on a deleted word collapses onto the fused
        // instruction, which is acceptable for debug info.
        const w = l.offset / 4;
        l.offset = @intCast((w - removed_before[w]) * 4);
    }

    words.shrinkRetainingCapacity(write);
}

test "machine score rejects lower throughput rewrites" {
    const model = MachineModel{ .issue_width = 2, .units = .{ .alu = 2, .mem = 1 } };
    const arithmetic = scoreWords(&.{ encode.add(.x0, .x1, .x2), encode.add(.x3, .x4, .x5) }, model);
    const memory = scoreWords(&.{ encode.ldrOff(.x0, .x1, 0), encode.ldrOff(.x2, .x1, 8) }, model);
    try std.testing.expect(!memory.accepts(arithmetic));
}

test "machine score tracks register dependency latency" {
    const model = MachineModel{
        .issue_width = 2,
        .units = .{ .alu = 2, .muldiv = 1 },
        .latency = .{ .mul = 4 },
    };
    const dependent = scoreWords(&.{ encode.mul(.x3, .x0, .x1), encode.add(.x4, .x3, .x2) }, model);
    const independent = scoreWords(&.{ encode.mul(.x3, .x0, .x1), encode.add(.x4, .x5, .x2) }, model);

    try std.testing.expectEqual(@as(u32, 5), dependent.critical_path);
    try std.testing.expectEqual(@as(u32, 4), independent.critical_path);
    try std.testing.expect(independent.accepts(dependent));
}

test "machine score tracks immediate producer dependencies" {
    const model = MachineModel{ .issue_width = 4, .units = .{ .alu = 3 } };
    const score = scoreWords(&.{ encode.subImm(.x1, .x0, 1), encode.csinc(.x2, .x1, .x0, .eq, false) }, model);

    try std.testing.expectEqual(@as(u32, 2), score.critical_path);
}

test "machine score tracks NZCV from compare through conditional branch" {
    const model = MachineModel{ .issue_width = 4, .latency = .{ .alu = 2, .branch = 1 } };
    const score = scoreWords(&.{ encode.cmp(.x0, .x1), encode.bcc(.eq, 0) }, model);
    try std.testing.expectEqual(@as(u32, 3), score.critical_path);
}

test "machine score tracks vector load and arithmetic chains" {
    const model = MachineModel{ .issue_width = 4, .latency = .{ .mem = 3, .fpsimd = 2 } };
    const score = scoreWords(&.{
        encode.ldrQ(.x0, .x1, 0),
        encode.addVecI32(.x2, .x0, .x3),
        encode.eorVec(.x4, .x2, .x5),
    }, model);
    try std.testing.expectEqual(@as(u32, 7), score.critical_path);
}

test "machine score tracks SIMD pair loads as memory dependencies" {
    const model = MachineModel{ .issue_width = 4, .latency = .{ .mem = 4, .fpsimd = 2 } };
    const score = scoreWords(&.{
        encode.ldpOffQ(.x0, .x2, .x1, 0),
        encode.addVecI32(.x3, .x0, .x4),
    }, model);

    try std.testing.expectEqual(@as(u32, 6), score.critical_path);
}

test "machine score keeps vector accumulate dependencies" {
    const model = MachineModel{ .issue_width = 4, .latency = .{ .fpsimd = 3 } };
    const score = scoreWords(&.{
        encode.mlaVecI32(.x0, .x1, .x2),
        encode.mlaVecI32(.x0, .x3, .x4),
    }, model);
    try std.testing.expectEqual(@as(u32, 6), score.critical_path);
}

test "machine score charges non-pipelined divide throughput" {
    const model = MachineModel{
        .issue_width = 4,
        .units = .{ .alu = 3, .muldiv = 1 },
        .latency = .{ .div = 18 },
        .throughput = .{ .div = 18 },
    };
    const score = scoreWords(&.{ encode.sdiv(.x0, .x1, .x2), encode.udiv(.x3, .x4, .x5) }, model);

    try std.testing.expectEqual(@as(u32, 36), score.cycles);
    try std.testing.expectEqual(@as(u32, 18), score.critical_path);
}

test "machine score classifies packed switch loads as memory" {
    const model = MachineModel{ .issue_width = 2, .units = .{ .alu = 2, .mem = 1 }, .latency = .{ .mem = 4 } };
    const word = scoreWords(&.{encode.ldrWReg(.x17, .x16, .x20)}, model);
    const byte = scoreWords(&.{encode.ldrbReg(.x17, .x16, .x20)}, model);

    try std.testing.expectEqual(word, byte);
}

test "rewrites widened unsigned magic division" {
    const rd: Reg = .x12;
    const source: Reg = .x7;
    const multiplier: Reg = .x9;
    const rewrite = tryNarrowUnsignedMagic(&.{
        encode.ubfm(rd, source, 0, 31),
        encode.mul(rd, rd, multiplier) | (@as(u32, 1) << 31),
        encode.lsrImm(rd, rd, 32, true),
        encode.lsrImm(rd, rd, 2, true),
    }).?;

    try std.testing.expectEqualSlices(u32, &.{
        encode.umull(rd, source, multiplier),
        encode.lsrImm(rd, rd, 34, true),
    }, &rewrite.words);
}

test "machine rewrites leave dense switch data untouched" {
    const alloc = std.testing.allocator;
    const original = [_]u32{
        encode.ubfm(.x12, .x7, 0, 31),
        encode.mul(.x12, .x12, .x9) | (@as(u32, 1) << 31),
        encode.lsrImm(.x12, .x12, 32, true),
        encode.lsrImm(.x12, .x12, 2, true),
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
    };
    var words = try wordsFrom(alloc, &original);
    defer words.deinit(alloc);
    var fixups: [original.len]Fixup = undefined;
    for (&fixups, 0..) |*fixup, at| fixup.* = .{ .at = at, .target = 0, .table_base = 0 };

    try pairMemory(alloc, &words, &.{}, &fixups, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &original, words.items);
}

test "decodeMem recognizes each ldr/str offset form and rejects a non-memory word" {
    const t = std.testing;

    try t.expectEqual(MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 1, .off = 16 }, decodeMem(encode.ldrOff(.x0, .x1, 16)).?);
    try t.expectEqual(MemInsn{ .kind = .str_x, .rt = 2, .rn = 3, .off = 8 }, decodeMem(encode.strOff(.x2, .x3, 8)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_w, .rt = 4, .rn = 5, .off = 12 }, decodeMem(encode.ldrW(.x4, .x5, 12)).?);
    try t.expectEqual(MemInsn{ .kind = .str_w, .rt = 6, .rn = 7, .off = 4 }, decodeMem(encode.strW(.x6, .x7, 4)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_s, .rt = 0, .rn = 1, .off = 8 }, decodeMem(encode.ldrFp(.x0, .x1, 8, false)).?);
    try t.expectEqual(MemInsn{ .kind = .str_s, .rt = 2, .rn = 3, .off = 4 }, decodeMem(encode.strFp(.x2, .x3, 4, false)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_d, .rt = 4, .rn = 5, .off = 16 }, decodeMem(encode.ldrFp(.x4, .x5, 16, true)).?);
    try t.expectEqual(MemInsn{ .kind = .str_d, .rt = 6, .rn = 7, .off = 8 }, decodeMem(encode.strFp(.x6, .x7, 8, true)).?);
    try t.expectEqual(MemInsn{ .kind = .ldr_q, .rt = 0, .rn = 1, .off = 32 }, decodeMem(encode.ldrQ(.x0, .x1, 32)).?);
    try t.expectEqual(MemInsn{ .kind = .str_q, .rt = 2, .rn = 3, .off = 16 }, decodeMem(encode.strQ(.x2, .x3, 16)).?);

    // Non-memory ops: rejected outright.
    try t.expectEqual(@as(?MemInsn, null), decodeMem(encode.add64(.x0, .x1, .x2)));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(encode.nop()));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0x14000000)); // b #0 (unconditional branch)

    // A different addressing mode for the same class: pre/post-index and unscaled-immediate
    // loads/stores clear bit 24, landing outside every class above (e.g. 0xF8400000 is the
    // 64-bit pre/post-index/unscaled family, distinct from ldr_x's 0xF9400000).
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0xF8400400));
    try t.expectEqual(@as(?MemInsn, null), decodeMem(0xB8400400));
}

test "tryFuse pairs two adjacent consecutive-offset x loads into ldp" {
    const a = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 0 };
    const b = MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 8 };
    const fused = tryFuse(a, b);
    try std.testing.expectEqual(encode.ldpOffX(.x0, .x1, .x2, 0), fused.?);

    const disasm = @import("disasm.zig");
    const alloc = std.testing.allocator;
    const s = try disasm.one(alloc, fused.?);
    defer alloc.free(s);
    try std.testing.expectEqualStrings("ldp x0, x1, [x2]", s);
}

test "tryFuseMulAdd combines either add operand orientation" {
    const mul_word = encode.mul(.x4, .x1, .x2);
    try std.testing.expectEqual(
        encode.madd(.x5, .x1, .x2, .x3, false),
        tryFuseMulAdd(mul_word, encode.add(.x5, .x4, .x3)).?,
    );
    try std.testing.expectEqual(
        encode.madd(.x5, .x1, .x2, .x3, false),
        tryFuseMulAdd(mul_word, encode.add(.x5, .x3, .x4)).?,
    );
}

test "tryFuseMulAdd rejects width mismatch and a doubled product" {
    const mul_word = encode.mul(.x4, .x1, .x2);
    try std.testing.expectEqual(@as(?u32, null), tryFuseMulAdd(mul_word, encode.add64(.x5, .x4, .x3)));
    try std.testing.expectEqual(@as(?u32, null), tryFuseMulAdd(mul_word, encode.add(.x5, .x4, .x4)));
}

test "tryCoalesceResultMove forwards scalar ALU results into the move destination" {
    try std.testing.expectEqual(
        encode.subShifted(.x21, .x21, .x21, 3),
        tryCoalesceResultMove(encode.subShifted(.x24, .x21, .x21, 3), encode.mov(.x21, .x24)).?,
    );
    try std.testing.expectEqual(
        encode.csel(.x21, .x20, .x25, .lt),
        tryCoalesceResultMove(encode.csel(.x24, .x20, .x25, .lt), encode.mov(.x21, .x24)).?,
    );
    try std.testing.expectEqual(@as(?u32, null), tryCoalesceResultMove(encode.ldrW(.x24, .x20, 0), encode.mov(.x21, .x24)));
    try std.testing.expectEqual(@as(?u32, null), tryCoalesceResultMove(encode.add(.x23, .x20, .x22), encode.mov(.x21, .x24)));
}

test "pairMemory coalesces an ALU result move and remaps following control flow" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.add(.x24, .x21, .x20),
        encode.mov(.x21, .x24),
        encode.b(0),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 3 };
    var fixups = [_]Fixup{.{ .at = 2, .target = 1 }};

    try pairMemoryForModelWithMoves(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{1}, .{});

    try std.testing.expectEqualSlices(u32, &.{ encode.add(.x21, .x21, .x20), encode.b(0), encode.ret() }, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 2 }, &block_start);
    try std.testing.expectEqual(@as(usize, 1), fixups[0].at);
}

test "dead result move crosses an unentered coincident block start" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.csel(.x9, .x10, .x22, .gt),
        encode.mov(.x10, .x9),
        encode.b(0),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 1, 3 };
    var fixups = [_]Fixup{.{ .at = 2, .target = 2 }};

    try pairMemoryForModelWithMoveProofs(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{}, &.{1}, .{});

    try std.testing.expectEqualSlices(u32, &.{ encode.csel(.x10, .x10, .x22, .gt), encode.b(0), encode.ret() }, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 0, 2 }, &block_start);
}

test "dead result move keeps a directly entered block start" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.b(0),
        encode.csel(.x9, .x10, .x22, .gt),
        encode.mov(.x10, .x9),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 1, 2 };
    var fixups = [_]Fixup{.{ .at = 0, .target = 2 }};

    try pairMemoryForModelWithMoveProofs(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{}, &.{2}, .{});

    try std.testing.expectEqualSlices(u32, &.{ encode.b(0), encode.csel(.x9, .x10, .x22, .gt), encode.mov(.x10, .x9), encode.ret() }, words.items);
}

test "magic rewrite remaps a later proven result move" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.ubfm(.x12, .x7, 0, 31),
        encode.mul(.x12, .x12, .x9) | (@as(u32, 1) << 31),
        encode.lsrImm(.x12, .x12, 32, true),
        encode.lsrImm(.x12, .x12, 2, true),
        encode.add(.x24, .x21, .x20),
        encode.mov(.x21, .x24),
        encode.b(0),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 7 };
    var fixups = [_]Fixup{.{ .at = 6, .target = 1 }};

    try pairMemoryForModelWithMoves(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{5}, .{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.umull(.x12, .x7, .x9),
        encode.lsrImm(.x12, .x12, 34, true),
        encode.add(.x21, .x21, .x20),
        encode.b(0),
        encode.ret(),
    }, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 4 }, &block_start);
    try std.testing.expectEqual(@as(usize, 3), fixups[0].at);
}

test "dead edge moves retarget recurrence producers across compares" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.add(.x11, .x11, .x9),
        encode.addImm(.x1, .x2, 1),
        encode.cmp(.x1, .x3),
        encode.mov(.x2, .x1),
        encode.mov(.x1, .x11),
        encode.b(0),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 1, 6 };
    var fixups = [_]Fixup{.{ .at = 5, .target = 2 }};

    try pairMemoryForModelWithMoveProofs(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{}, &.{ 3, 4 }, .{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.add(.x1, .x11, .x9),
        encode.addImm(.x2, .x2, 1),
        encode.cmp(.x2, .x3),
        encode.b(0),
        encode.ret(),
    }, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 4 }, &block_start);
    try std.testing.expectEqual(@as(usize, 3), fixups[0].at);
}

test "dead edge moves do not retarget across a directly entered block" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.add(.x11, .x11, .x9),
        encode.addImm(.x1, .x2, 1),
        encode.cmp(.x1, .x3),
        encode.mov(.x2, .x1),
        encode.mov(.x1, .x11),
        encode.b(0),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 1, 6 };
    var fixups = [_]Fixup{.{ .at = 5, .target = 1 }};

    try pairMemoryForModelWithMoveProofs(alloc, &words, &block_start, &fixups, &.{}, &.{}, &.{}, &.{ 3, 4 }, .{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.add(.x11, .x11, .x9),
        encode.addImm(.x2, .x2, 1),
        encode.cmp(.x2, .x3),
        encode.mov(.x1, .x11),
        encode.b(0),
        encode.ret(),
    }, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1, 5 }, &block_start);
    try std.testing.expectEqual(@as(usize, 4), fixups[0].at);
}

test "dead edge move cycles stay intact" {
    const words = [_]u32{
        encode.mov(.x5, .x2),
        encode.mov(.x2, .x3),
        encode.mov(.x3, .x2),
        encode.mov(.x2, .x5),
    };
    try std.testing.expect(moveRunHasCycle(&words, &.{ 0, 1, 2, 3 }));
    try std.testing.expect(!moveRunHasCycle(&words, &.{ 0, 1 }));
}

test "pairMemory also fuses multiply-add and remaps side tables" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.mul(.x4, .x1, .x2),
        encode.add(.x5, .x4, .x3),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{0};

    try pairMemory(alloc, &words, &block_start, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &.{ encode.madd(.x5, .x1, .x2, .x3, false), encode.ret() }, words.items);
}

test "tryFuse pairs two adjacent consecutive-offset w stores into stp" {
    const a = MemInsn{ .kind = .str_w, .rt = 4, .rn = 5, .off = 0 };
    const b = MemInsn{ .kind = .str_w, .rt = 6, .rn = 5, .off = 4 };
    try std.testing.expectEqual(encode.stpOffW(.x4, .x6, .x5, 0), tryFuse(a, b).?);
}

test "tryFuse allows a store pair even when rt equals rn (the load-only base-clobber rule)" {
    // Storing x2 through base x2 is fine (the store never writes a register), unlike the load
    // case where a.rt == a.rn would mean the first load overwrites the shared base.
    const a = MemInsn{ .kind = .str_x, .rt = 2, .rn = 2, .off = 0 };
    const b = MemInsn{ .kind = .str_x, .rt = 3, .rn = 2, .off = 8 };
    try std.testing.expectEqual(encode.stpOffX(.x2, .x3, .x2, 0), tryFuse(a, b).?);
}

test "tryFuse pairs q registers and leaves scalar-float pairs alone" {
    const s1 = MemInsn{ .kind = .ldr_s, .rt = 0, .rn = 1, .off = 0 };
    const s2 = MemInsn{ .kind = .ldr_s, .rt = 2, .rn = 1, .off = 4 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(s1, s2));

    const d1 = MemInsn{ .kind = .ldr_d, .rt = 0, .rn = 1, .off = 0 };
    const d2 = MemInsn{ .kind = .ldr_d, .rt = 2, .rn = 1, .off = 8 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(d1, d2));

    const q1 = MemInsn{ .kind = .ldr_q, .rt = 0, .rn = 1, .off = 0 };
    const q2 = MemInsn{ .kind = .ldr_q, .rt = 2, .rn = 1, .off = 16 };
    try std.testing.expectEqual(encode.ldpOffQ(.x0, .x2, .x1, 0), tryFuse(q1, q2).?);
}

test "pairMemory scores and fuses two interleaved q-load streams" {
    const allocator = std.testing.allocator;
    var words = try wordsFrom(allocator, &.{
        encode.ldrQ(.x12, .x23, 0),
        encode.ldrQ(.x13, .x24, 0),
        encode.ldrQ(.x14, .x23, 16),
        encode.ldrQ(.x15, .x24, 16),
    });
    defer words.deinit(allocator);

    try pairMemory(allocator, &words, &.{}, &.{}, &.{}, &.{});
    try std.testing.expectEqualSlices(u32, &.{
        encode.ldpOffQ(.x12, .x14, .x23, 0),
        encode.ldpOffQ(.x13, .x15, .x24, 0),
    }, words.items);
}

test "pairMemory folds a carried q-pair pointer update into post-index" {
    const allocator = std.testing.allocator;
    var words = try wordsFrom(allocator, &.{
        encode.ldpOffQ(.x0, .x1, .x2, 0),
        encode.add(.x3, .x4, .x5),
        encode.addImm64(.x2, .x2, 32),
    });
    defer words.deinit(allocator);

    try pairMemory(allocator, &words, &.{}, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.ldpPostQ(.x0, .x1, .x2, 32),
        encode.add(.x3, .x4, .x5),
    }, words.items);
}

test "pairMemory combines q loads with a carried pointer update" {
    const allocator = std.testing.allocator;
    var words = try wordsFrom(allocator, &.{
        encode.ldrQ(.x0, .x2, 0),
        encode.ldrQ(.x1, .x2, 16),
        encode.add(.x3, .x4, .x5),
        encode.addImm64(.x2, .x2, 32),
    });
    defer words.deinit(allocator);

    try pairMemory(allocator, &words, &.{}, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.ldpPostQ(.x0, .x1, .x2, 32),
        encode.add(.x3, .x4, .x5),
    }, words.items);
}

test "pairMemory combines q stores with a carried pointer update" {
    const allocator = std.testing.allocator;
    var words = try wordsFrom(allocator, &.{
        encode.strQ(.x0, .x2, 0),
        encode.strQ(.x1, .x2, 16),
        encode.add(.x3, .x4, .x5),
        encode.addImm64(.x2, .x2, 32),
    });
    defer words.deinit(allocator);

    try pairMemory(allocator, &words, &.{}, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.stpPostQ(.x0, .x1, .x2, 32),
        encode.add(.x3, .x4, .x5),
    }, words.items);
}

test "pairMemory post-indexes two interleaved q-load streams" {
    const allocator = std.testing.allocator;
    var words = try wordsFrom(allocator, &.{
        encode.ldrQ(.x0, .x10, 0),
        encode.ldrQ(.x2, .x11, 0),
        encode.ldrQ(.x1, .x10, 16),
        encode.ldrQ(.x3, .x11, 16),
        encode.addImm(.x4, .x4, 8),
        encode.addImm64(.x10, .x10, 32),
        encode.addImm64(.x11, .x11, 32),
    });
    defer words.deinit(allocator);

    try pairMemory(allocator, &words, &.{}, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &.{
        encode.ldpPostQ(.x0, .x1, .x10, 32),
        encode.ldpPostQ(.x2, .x3, .x11, 32),
        encode.addImm(.x4, .x4, 8),
    }, words.items);
}

/// Build an owned `std.ArrayList(u32)` from a literal slice so a test can hand `pairMemory` a
/// word buffer it will shrink in place. The caller deinits.
fn wordsFrom(allocator: std.mem.Allocator, init: []const u32) !std.ArrayList(u32) {
    var w: std.ArrayList(u32) = .empty;
    try w.appendSlice(allocator, init);
    return w;
}

test "pairMemory fuses an adjacent x-load pair and shrinks the word count by one" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.ret(),
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{0};

    try pairMemory(alloc, &words, &block_start, &.{}, &.{}, &.{});

    try std.testing.expectEqual(@as(usize, 2), words.items.len);
    try std.testing.expectEqual(encode.ldpOffX(.x0, .x1, .x2, 0), words.items[0]);
    try std.testing.expectEqual(encode.ret(), words.items[1]);
    // The fused word is an ldp, which is opaque to the single-op decoder.
    try std.testing.expectEqual(@as(?MemInsn, null), decodeMem(words.items[0]));
}

test "pairMemory does not fuse across a block boundary" {
    const alloc = std.testing.allocator;
    const original = [_]u32{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.ret(),
    };
    var words = try wordsFrom(alloc, &original);
    defer words.deinit(alloc);
    // The second load starts block 1, so control flow can enter between the pair: no fusion.
    var block_start = [_]usize{ 0, 1 };

    try pairMemory(alloc, &words, &block_start, &.{}, &.{}, &.{});

    try std.testing.expectEqualSlices(u32, &original, words.items);
    try std.testing.expectEqualSlices(usize, &.{ 0, 1 }, &block_start);
}

test "pairMemory remaps a branch fixup and block_start after a fused pair" {
    const alloc = std.testing.allocator;
    var words = try wordsFrom(alloc, &.{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 8),
        encode.b(0), // fixup site at word index 2
        encode.ret(), // block 1 starts here (word index 3)
    });
    defer words.deinit(alloc);
    var block_start = [_]usize{ 0, 3 };
    var fixups = [_]Fixup{.{ .at = 2, .target = 1 }};

    try pairMemory(alloc, &words, &block_start, &fixups, &.{}, &.{});

    try std.testing.expectEqual(@as(usize, 3), words.items.len);
    try std.testing.expectEqual(@as(usize, 1), fixups[0].at); // 2 - 1 deleted word
    try std.testing.expectEqual(@as(u32, 1), fixups[0].target); // block index unchanged
    try std.testing.expectEqual(@as(usize, 0), block_start[0]);
    try std.testing.expectEqual(@as(usize, 2), block_start[1]); // 3 - 1 deleted word
}

test "pairMemory is a no-op leaving words byte-identical when nothing is fusable" {
    const alloc = std.testing.allocator;
    // Two x-loads at NON-consecutive offsets (0 and 16): not a fusable pair.
    const original = [_]u32{
        encode.ldrOff(.x0, .x2, 0),
        encode.ldrOff(.x1, .x2, 16),
        encode.ret(),
    };
    var words = try wordsFrom(alloc, &original);
    defer words.deinit(alloc);
    var block_start = [_]usize{0};
    var fixups = [_]Fixup{.{ .at = 2, .target = 0 }};
    var relocs = [_]Reloc{.{ .offset = 2, .symbol = "x" }};
    var lines = [_]LineEntry{.{ .offset = 8, .line = 7 }};

    try pairMemory(alloc, &words, &block_start, &fixups, &relocs, &lines);

    try std.testing.expectEqualSlices(u32, &original, words.items);
    try std.testing.expectEqualSlices(usize, &.{0}, &block_start);
    try std.testing.expectEqual(@as(usize, 2), fixups[0].at);
    try std.testing.expectEqual(@as(usize, 2), relocs[0].offset);
    try std.testing.expectEqual(@as(u32, 8), lines[0].offset);
}

test "tryFuse rejects non-consecutive, different-base, rt1==rn, rt1==rt2 (load), mismatched-size, out-of-range" {
    const base = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 0 };

    // Non-consecutive offset: b sits two elements above a, not one.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 16 }));

    // Different base register.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 3, .off = 8 }));

    // a.rt == a.rn: the first load would overwrite the base the second load still needs.
    const clobbers_base = MemInsn{ .kind = .ldr_x, .rt = 2, .rn = 2, .off = 0 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(clobbers_base, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 8 }));

    // a.rt == b.rt: an ldp with equal destinations is UNPREDICTABLE.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 8 }));

    // Mismatched kind/size: an x-load cannot fuse with a w-load even at consecutive offsets.
    try std.testing.expectEqual(@as(?u32, null), tryFuse(base, MemInsn{ .kind = .ldr_w, .rt = 1, .rn = 2, .off = 8 }));

    // Out of range: imm7 = a.off / size = 512 / 8 = 64, over the 63 upper bound.
    const far = MemInsn{ .kind = .ldr_x, .rt = 0, .rn = 2, .off = 512 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(far, MemInsn{ .kind = .ldr_x, .rt = 1, .rn = 2, .off = 520 }));
}
