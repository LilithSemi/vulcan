//! Minimal arm64 Mach-O relocatable-object writer used by the Darwin VCC driver.

const std = @import("std");
const macho = std.macho;
const isel = @import("isel.zig");
const link = @import("link.zig");

pub const Error = isel.Error;

const SectionKind = enum { text, rodata, data, bss };
const PendingReloc = struct { off: u32, name: []const u8, kind: RelocKind };
const RelocKind = enum { branch26, page21, pageoff12, got_page21, got_pageoff12, unsigned };
const Section = struct {
    kind: SectionKind,
    bytes: std.ArrayList(u8) = .empty,
    size: u64 = 0,
    addr: u64 = 0,
    fileoff: u32 = 0,
    reloff: u32 = 0,
    relocs: std.ArrayList(PendingReloc) = .empty,
};
const Symbol = struct { name: []const u8, section: ?usize, value: u64, local: bool };

fn writeStruct(w: *std.Io.Writer, value: anytype) Error!void {
    w.writeStruct(value, .little) catch return error.OutOfMemory;
}

fn writeAll(w: *std.Io.Writer, bytes: []const u8) Error!void {
    w.writeAll(bytes) catch return error.OutOfMemory;
}

fn padTo(w: *std.Io.Writer, offset: usize) Error!void {
    std.debug.assert(w.end <= offset);
    w.splatByteAll(0, offset - w.end) catch return error.OutOfMemory;
}

fn alignForward(v: usize, a: usize) usize {
    return std.mem.alignForward(usize, v, a);
}

fn sectionIndex(sections: []const Section, kind: SectionKind) ?usize {
    for (sections, 0..) |s, i| if (s.kind == kind) return i;
    return null;
}

fn symbolIndex(symbols: []const Symbol, name: []const u8) ?u32 {
    for (symbols, 0..) |s, i| if (std.mem.eql(u8, s.name, name)) return @intCast(i);
    return null;
}

fn machoName(a: std.mem.Allocator, name: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "_{s}", .{name});
}

fn relocKind(kind: isel.Kind) RelocKind {
    return switch (kind) {
        .call => .branch26,
        .adrp_pg => .page21,
        .add_pgoff => .pageoff12,
        .got_pg => .got_page21,
        .got_lo12 => .got_pageoff12,
    };
}

fn relocInfo(sym: u32, off: u32, kind: RelocKind) macho.relocation_info {
    const pcrel: u1 = switch (kind) {
        .branch26, .page21, .got_page21 => 1,
        else => 0,
    };
    const length: u2 = if (kind == .unsigned) 3 else 2;
    const rtype: macho.reloc_type_arm64 = switch (kind) {
        .unsigned => .ARM64_RELOC_UNSIGNED,
        .branch26 => .ARM64_RELOC_BRANCH26,
        .page21 => .ARM64_RELOC_PAGE21,
        .pageoff12 => .ARM64_RELOC_PAGEOFF12,
        .got_page21 => .ARM64_RELOC_GOT_LOAD_PAGE21,
        .got_pageoff12 => .ARM64_RELOC_GOT_LOAD_PAGEOFF12,
    };
    return .{
        .r_address = @intCast(off),
        .r_symbolnum = @intCast(sym),
        .r_pcrel = pcrel,
        .r_length = length,
        .r_extern = 1,
        .r_type = @backingInt(rtype),
    };
}

fn fixedName(name: []const u8) [16]u8 {
    var out: [16]u8 = @splat(0);
    @memcpy(out[0..@min(out.len, name.len)], name[0..@min(out.len, name.len)]);
    return out;
}

/// Compile `module` with the Apple arm64 calling convention and serialize an MH_OBJECT
/// accepted by Apple's system `cc`/`ld`.
pub fn writeModule(allocator: std.mem.Allocator, module: *const link.Module) Error![]u8 {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();

    var sections: std.ArrayList(Section) = .empty;
    try sections.append(a, .{ .kind = .text });
    if (for (module.data.items) |d| {
        if (d.kind == .rodata) break true;
    } else false) try sections.append(a, .{ .kind = .rodata });
    if (for (module.data.items) |d| {
        if (d.kind == .data) break true;
    } else false) try sections.append(a, .{ .kind = .data });
    if (for (module.data.items) |d| {
        if (d.kind == .bss) break true;
    } else false) try sections.append(a, .{ .kind = .bss });

    var symbols: std.ArrayList(Symbol) = .empty;
    const base_caps: isel.ModelCaps = if (module.model) |m| isel.capsForModel(m) else .{};
    var caps = base_caps;
    caps.abi = .apple;
    const text_i = sectionIndex(sections.items, .text).?;
    for (module.functions.items) |entry| {
        var compiled = try isel.compileFunction(allocator, entry.func, caps);
        defer compiled.deinit(allocator);
        const alignment = @max(@as(usize, 4), caps.fetch_align);
        const off = alignForward(sections.items[text_i].bytes.items.len, alignment);
        try sections.items[text_i].bytes.appendNTimes(a, 0, off - sections.items[text_i].bytes.items.len);
        for (compiled.code) |word| {
            var raw: [4]u8 = undefined;
            std.mem.writeInt(u32, &raw, word, .little);
            try sections.items[text_i].bytes.appendSlice(a, &raw);
        }
        try symbols.append(a, .{ .name = entry.name, .section = text_i, .value = off, .local = entry.func.is_local or std.mem.startsWith(u8, entry.name, ".") });
        for (compiled.relocs) |r| try sections.items[text_i].relocs.append(a, .{ .off = @intCast(off + r.offset * 4), .name = r.symbol, .kind = relocKind(r.kind) });
    }

    for (module.data.items) |d| {
        const kind: SectionKind = switch (d.kind) {
            .rodata => .rodata,
            .data => .data,
            .bss => .bss,
        };
        const si = sectionIndex(sections.items, kind).?;
        const off: usize = alignForward(@intCast(sections.items[si].size), 8);
        if (kind != .bss) {
            try sections.items[si].bytes.appendNTimes(a, 0, off - sections.items[si].bytes.items.len);
            try sections.items[si].bytes.appendSlice(a, d.bytes);
        }
        sections.items[si].size = off + d.size;
        try symbols.append(a, .{ .name = d.name, .section = si, .value = off, .local = std.mem.startsWith(u8, d.name, ".") });
        for (d.relocs) |r| try sections.items[si].relocs.append(a, .{ .off = @intCast(off + r.off), .name = r.symbol, .kind = .unsigned });
    }
    sections.items[text_i].size = sections.items[text_i].bytes.items.len;
    for (sections.items) |*s| {
        if (s.kind != .bss) s.size = s.bytes.items.len;
    }

    // Mach-O symbol tables conventionally put locals first, then external definitions.
    std.mem.sort(Symbol, symbols.items, {}, struct {
        fn less(_: void, x: Symbol, y: Symbol) bool {
            return @intFromBool(!x.local) < @intFromBool(!y.local);
        }
    }.less);
    for (sections.items) |s| for (s.relocs.items) |r| if (symbolIndex(symbols.items, r.name) == null)
        try symbols.append(a, .{ .name = r.name, .section = null, .value = 0, .local = false });

    const header_size = @sizeOf(macho.mach_header_64);
    const segment_size = @sizeOf(macho.segment_command_64) + @sizeOf(macho.section_64) * sections.items.len;
    const sizeofcmds = segment_size + @sizeOf(macho.symtab_command);
    var cursor = alignForward(header_size + sizeofcmds, 8);
    var vm_cursor: usize = 0;
    for (sections.items) |*s| {
        const alignment: usize = if (s.kind == .text) 4 else 8;
        vm_cursor = alignForward(vm_cursor, alignment);
        s.addr = vm_cursor;
        vm_cursor += @intCast(s.size);
        if (s.kind != .bss) {
            cursor = alignForward(cursor, alignment);
            s.fileoff = @intCast(cursor);
            cursor += s.bytes.items.len;
        }
    }
    for (sections.items) |*s| if (s.relocs.items.len != 0) {
        cursor = alignForward(cursor, 4);
        s.reloff = @intCast(cursor);
        cursor += @sizeOf(macho.relocation_info) * s.relocs.items.len;
    };
    const symoff = alignForward(cursor, 8);
    cursor = symoff + @sizeOf(macho.nlist_64) * symbols.items.len;
    const stroff = cursor;
    var strtab: std.ArrayList(u8) = .empty;
    try strtab.append(a, 0);
    var strx = try a.alloc(u32, symbols.items.len);
    for (symbols.items, 0..) |s, i| {
        strx[i] = @intCast(strtab.items.len);
        try strtab.appendSlice(a, try machoName(a, s.name));
        try strtab.append(a, 0);
    }
    const object_size = cursor + strtab.items.len;
    var out: std.ArrayList(u8) = .empty;
    var allocating = std.Io.Writer.Allocating.fromArrayList(allocator, &out);
    defer allocating.deinit();
    const w = &allocating.writer;

    try writeStruct(w, macho.mach_header_64{
        .magic = macho.MH_MAGIC_64,
        .cputype = macho.CPU_TYPE_ARM64,
        .cpusubtype = 0,
        .filetype = macho.MH_OBJECT,
        .ncmds = 2,
        .sizeofcmds = @intCast(sizeofcmds),
        .flags = macho.MH_SUBSECTIONS_VIA_SYMBOLS,
    });

    try writeStruct(w, macho.segment_command_64{
        .cmd = .SEGMENT_64,
        .cmdsize = @intCast(segment_size),
        .segname = @splat(0),
        .vmaddr = 0,
        .vmsize = vm_cursor,
        .fileoff = 0,
        .filesize = symoff,
        .maxprot = .{ .READ = true, .WRITE = true, .EXEC = true },
        .initprot = .{ .READ = true, .WRITE = true, .EXEC = true },
        .nsects = @intCast(sections.items.len),
    });
    for (sections.items) |s| {
        const names = switch (s.kind) {
            .text => .{ "__text", "__TEXT" },
            // Pointer-bearing constants need dyld rebasing before the segment
            // becomes read-only.
            .rodata => if (s.relocs.items.len == 0)
                .{ "__const", "__TEXT" }
            else
                .{ "__const", "__DATA_CONST" },
            .data => .{ "__data", "__DATA" },
            .bss => .{ "__bss", "__DATA" },
        };
        try writeStruct(w, macho.section_64{
            .sectname = fixedName(names[0]),
            .segname = fixedName(names[1]),
            .addr = s.addr,
            .size = s.size,
            .offset = s.fileoff,
            .@"align" = if (s.kind == .text) 2 else 3,
            .reloff = s.reloff,
            .nreloc = @intCast(s.relocs.items.len),
            .flags = switch (s.kind) {
                .text => macho.S_REGULAR | macho.S_ATTR_PURE_INSTRUCTIONS | macho.S_ATTR_SOME_INSTRUCTIONS,
                .bss => macho.S_ZEROFILL,
                else => macho.S_REGULAR,
            },
        });
    }
    try writeStruct(w, macho.symtab_command{
        .cmd = .SYMTAB,
        .symoff = @intCast(symoff),
        .nsyms = @intCast(symbols.items.len),
        .stroff = @intCast(stroff),
        .strsize = @intCast(strtab.items.len),
    });

    for (sections.items) |s| if (s.kind != .bss) {
        try padTo(w, s.fileoff);
        try writeAll(w, s.bytes.items);
    };
    for (sections.items) |s| if (s.relocs.items.len != 0) {
        try padTo(w, s.reloff);
        for (s.relocs.items) |r| try writeStruct(w, relocInfo(symbolIndex(symbols.items, r.name).?, r.off, r.kind));
    };

    try padTo(w, symoff);
    for (symbols.items, 0..) |s, i| {
        try writeStruct(w, macho.nlist_64{
            .n_strx = strx[i],
            .n_type = .{ .bits = .{
                .ext = !s.local,
                .type = if (s.section != null) .sect else .undf,
                .pext = false,
                .is_stab = 0,
            } },
            .n_sect = if (s.section) |si| @intCast(si + 1) else 0,
            .n_desc = .{
                .arm_thumb_def = false,
                .referenced_dynamically = false,
                .discarded_or_no_dead_strip = false,
                .weak_ref = false,
                .weak_def_or_ref_to_weak = false,
                .symbol_resolver = false,
                .alt_entry = false,
            },
            .n_value = if (s.section) |si| sections.items[si].addr + s.value else 0,
        });
    }
    try padTo(w, stroff);
    try writeAll(w, strtab.items);
    std.debug.assert(w.end == object_size);
    return allocating.toOwnedSlice();
}

test "writes an arm64 Mach-O object header" {
    var module: link.Module = .{};
    defer module.deinit(std.testing.allocator);
    const bytes = try writeModule(std.testing.allocator, &module);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqual(macho.MH_MAGIC_64, std.mem.readInt(u32, bytes[0..4], .little));
}
