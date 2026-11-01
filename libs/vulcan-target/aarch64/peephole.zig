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
    var ready: [64]u32 = @splat(0);
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
    reads: [4]?u6 = @splat(null),
    writes: [2]?u6 = @splat(null),
};

fn dependencies(word: u32, class: MachineClass) RegisterDeps {
    if (decodeMem(word)) |mem| {
        const data = if (mem.kind == .ldr_s or mem.kind == .str_s or mem.kind == .ldr_d or mem.kind == .str_d or mem.kind == .ldr_q or mem.kind == .str_q)
            fpReg(mem.rt)
        else
            gpReg(mem.rt);
        if (isLoad(mem.kind)) return .{ .reads = .{ gpReg(mem.rn), null, null, null }, .writes = .{ data, null } };
        return .{ .reads = .{ gpReg(mem.rn), data, null, null } };
    }
    if (isPairMemory(word)) {
        const load = word & 0x00400000 != 0;
        const rt1: u5 = @truncate(word);
        const rt2: u5 = @truncate(word >> 10);
        const rn: u5 = @truncate(word >> 5);
        if (load) return .{ .reads = .{ gpReg(rn), null, null, null }, .writes = .{ gpReg(rt1), gpReg(rt2) } };
        return .{ .reads = .{ gpReg(rn), gpReg(rt1), gpReg(rt2), null } };
    }
    if (isRegisterMemory(word)) {
        const rt: u5 = @truncate(word);
        const rn: u5 = @truncate(word >> 5);
        const rm: u5 = @truncate(word >> 16);
        return .{ .reads = .{ gpReg(rn), gpReg(rm), null, null }, .writes = .{ gpReg(rt), null } };
    }
    if (class == .mul) {
        const ra: u5 = @truncate(word >> 10);
        return .{
            .reads = .{ gpReg(@truncate(word >> 5)), gpReg(@truncate(word >> 16)), gpReg(ra), null },
            .writes = .{ gpReg(@truncate(word)), null },
        };
    }
    if (class == .div or isRegisterAlu(word)) return .{
        .reads = .{ gpReg(@truncate(word >> 5)), gpReg(@truncate(word >> 16)), null, null },
        .writes = .{ gpReg(@truncate(word)), null },
    };
    if (isImmediateAlu(word)) return .{
        .reads = .{ gpReg(@truncate(word >> 5)), null, null, null },
        .writes = .{ gpReg(@truncate(word)), null },
    };
    return .{};
}

fn gpReg(reg: u5) ?u6 {
    return if (reg == 31) null else reg;
}

fn fpReg(reg: u5) u6 {
    return 32 + @as(u6, reg);
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

fn isRegisterMemory(word: u32) bool {
    const form = word & 0xFFE0FC00;
    return form == 0xB8607800 or form == 0x38606800;
}

fn isDivide(word: u32) bool {
    const form = word & 0x7FE0FC00;
    return form == 0x1AC00C00 or form == 0x1AC00800;
}

fn isPairMemory(word: u32) bool {
    return switch (word & 0xFFC00000) {
        0xA9400000, 0xA9000000, 0x29400000, 0x29000000 => true,
        else => false,
    };
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
/// Task 1 only added GPR (x/w) offset-form ldp/stp encoders (`ldpOffX`/`stpOffX`/`ldpOffW`/
/// `stpOffW`). The scalar-float (s/d) and 128-bit vector (q) ldp/stp forms live at different,
/// not-yet-added-or-disasm-confirmed base opcodes (the SIMD&FP register-pair family), so
/// pairing those kinds is deliberately deferred: they still decode correctly above (so a caller
/// walking code with `decodeMem` treats them as opaque single instructions, not garbage), but
/// `tryFuse` returns null for them rather than emit an unvalidated encoding.
pub fn tryFuse(a: MemInsn, b: MemInsn) ?u32 {
    if (a.kind != b.kind) return null;
    if (a.rn != b.rn) return null;
    const size = sizeOf(a.kind);
    if (b.off != a.off + size) return null;
    if (isLoad(a.kind)) {
        if (a.rt == b.rt) return null;
        if (a.rt == a.rn) return null;
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
        .ldr_s, .str_s, .ldr_d, .str_d, .ldr_q, .str_q => null,
    };
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
    model: MachineModel,
) std.mem.Allocator.Error!void {
    const len = words.items.len;
    var is_block_start = try allocator.alloc(bool, len);
    defer allocator.free(is_block_start);
    @memset(is_block_start, false);
    for (block_start) |start| if (start < len) {
        is_block_start[start] = true;
    };

    var removed_before = try allocator.alloc(u32, len + 1);
    defer allocator.free(removed_before);
    var write: usize = 0;
    var removed: u32 = 0;
    var i: usize = 0;
    while (i < len) {
        const rewrite = if (i + 3 < len and !is_block_start[i + 1] and !is_block_start[i + 2] and !is_block_start[i + 3])
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
        fixup.at -= removed_before[fixup.at];
    }
    for (relocs) |*reloc| reloc.offset -= removed_before[reloc.offset];
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
    try narrowUnsignedMagic(allocator, words, block_start, fixups, relocs, lines, model);
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
    var is_coalescable_result_move = try allocator.alloc(bool, len);
    defer allocator.free(is_coalescable_result_move);
    @memset(is_coalescable_result_move, false);
    for (coalescable_result_moves) |at| {
        std.debug.assert(at < len);
        is_coalescable_result_move[at] = true;
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
        var did_fuse = false;
        // A pair needs a following word that is not a block start. Never chain a third word into
        // the same pair (i += 2).
        if (i + 1 < len and !is_block_start[i + 1]) {
            const fused = (if (is_coalescable_result_move[i + 1]) tryCoalesceResultMove(words.items[i], words.items[i + 1]) else null) orelse
                tryFuseMulAdd(words.items[i], words.items[i + 1]) orelse memory: {
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

test "tryFuse returns null for scalar-float and vector kinds (no validated fp/q ldp/stp encoder yet)" {
    const s1 = MemInsn{ .kind = .ldr_s, .rt = 0, .rn = 1, .off = 0 };
    const s2 = MemInsn{ .kind = .ldr_s, .rt = 2, .rn = 1, .off = 4 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(s1, s2));

    const d1 = MemInsn{ .kind = .ldr_d, .rt = 0, .rn = 1, .off = 0 };
    const d2 = MemInsn{ .kind = .ldr_d, .rt = 2, .rn = 1, .off = 8 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(d1, d2));

    const q1 = MemInsn{ .kind = .ldr_q, .rt = 0, .rn = 1, .off = 0 };
    const q2 = MemInsn{ .kind = .ldr_q, .rt = 2, .rn = 1, .off = 16 };
    try std.testing.expectEqual(@as(?u32, null), tryFuse(q1, q2));
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
