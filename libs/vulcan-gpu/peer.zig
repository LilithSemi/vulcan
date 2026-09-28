//! Target-neutral peer-to-peer capability data: what a link between two accelerator devices
//! allows a transfer to do. `vulcan-gpu` already owns `Abi` and `Tensor`, the other per-target
//! capability descriptors a runtime reads before it acts, and a peer link is the same shape of
//! fact: NVLink, PCIe peer-to-peer, two runners sharing one GPU, and no link at all each answer
//! "can this transfer happen" differently, and a runtime needs one place to ask.
//!
//! This module is data plus validation only. It holds no device handle and issues no copy, so
//! it stays freestanding-clean and importable from a runtime, a scheduler, or a test, all alike.

const std = @import("std");

/// What a peer transfer can do over a link.
pub const Reach = enum {
    /// No path between the devices. A runtime must stage the transfer through host memory.
    none,
    /// A copy engine can move bytes across the link, but a kernel cannot dereference a peer
    /// pointer directly.
    copy_only,
    /// A kernel can read and write a peer pointer directly, and a copy engine can also move
    /// bytes across the link.
    addressable,
};

/// The fence a writer must issue before a peer is guaranteed to see the write.
pub const Ordering = enum {
    /// No link, so no ordering question arises.
    none,
    /// A fence local to the writing device is enough. NVLink carries store ordering to the peer
    /// without a system-wide flush.
    source_fence,
    /// The write must drain through a system-wide fence before a peer read is guaranteed to see
    /// it. PCIe peer writes can sit in a root complex buffer until one is issued.
    system_fence,
};

/// What one peer link allows. A runtime builds one per device pair and asks `check` before it
/// issues a transfer over it.
pub const Caps = struct {
    /// The access class the link offers. See `Reach`.
    reach: Reach,
    /// Whether an atomic read-modify-write over this link has defined ordering. PCIe peer
    /// atomics are not guaranteed by the specification, so a link built over PCIe leaves this
    /// false even where `reach` is `.addressable`.
    atomics: bool,
    /// The unit the importing side must map a peer allocation in, for `mappedRange`. 0 or 1
    /// means the importer maps at byte granularity.
    map_granule_bytes: u32,
    /// The offset and length granularity the copy engine accepts. 0 or 1 means any alignment.
    copy_align: u32,
    /// The most bytes one copy command can move. 0 means this descriptor states no bound; a
    /// runtime that needs one gets it from the copy engine driving the transfer, not from here.
    max_copy_bytes: u64,
    /// The fence a writer must issue before a peer read is guaranteed to see the write.
    release: Ordering,
};

/// One proposed copy: `bytes` starting at `src_offset` on the source device to `dst_offset` on
/// the destination device. The offsets and lengths are byte counts from a runtime request, not
/// values a programmer chose, so `check` treats them as untrusted.
pub const Transfer = struct { src_offset: u64, dst_offset: u64, bytes: u64 };

/// Why a link refuses a transfer. Each value names the reason a caller can act on: retry with a
/// bounce buffer, split the transfer, or repack the source.
pub const Refusal = enum {
    /// `caps.reach` is `.none`. There is no path between the devices at all.
    no_link,
    /// `src_offset` is not a multiple of `copy_align`.
    misaligned_source,
    /// `dst_offset` is not a multiple of `copy_align`.
    misaligned_destination,
    /// `bytes` is not a multiple of `copy_align`. A part-filled copy unit has no encoding on the
    /// copy engine.
    part_granule,
    /// `caps.max_copy_bytes` is nonzero and `bytes` exceeds it.
    too_large,
    /// `bytes` is 0. There is nothing to move.
    empty,
    /// `src_offset + bytes` or `dst_offset + bytes` would wrap past the top of a 64-bit address
    /// space. The request cannot name a real byte range, so it is refused rather than wrapped.
    overflow,
};

/// Whether `alignment` constrains anything. 0 and 1 both mean "any value passes".
fn constrains(alignment: u32) bool {
    return alignment > 1;
}

fn isAligned(value: u64, alignment: u32) bool {
    if (!constrains(alignment)) return true;
    return value % alignment == 0;
}

/// Why `caps` refuses `t`, or null when it allows it. Checks run in this order: `empty`,
/// `overflow`, `no_link`, `too_large`, `misaligned_source`, `misaligned_destination`,
/// `part_granule`. An empty transfer is refused before the link is even asked, and an
/// overflowing range is refused before alignment is asked, because neither is a real byte range
/// for the link to judge.
pub fn check(caps: Caps, t: Transfer) ?Refusal {
    if (t.bytes == 0) return .empty;
    _ = std.math.add(u64, t.src_offset, t.bytes) catch return .overflow;
    _ = std.math.add(u64, t.dst_offset, t.bytes) catch return .overflow;
    if (caps.reach == .none) return .no_link;
    if (caps.max_copy_bytes != 0 and t.bytes > caps.max_copy_bytes) return .too_large;
    if (!isAligned(t.src_offset, caps.copy_align)) return .misaligned_source;
    if (!isAligned(t.dst_offset, caps.copy_align)) return .misaligned_destination;
    if (!isAligned(t.bytes, caps.copy_align)) return .part_granule;
    return null;
}

/// Whether `caps` allows `t`. The negation of `check`, for a caller that only needs the answer.
pub fn accepts(caps: Caps, t: Transfer) bool {
    return check(caps, t) == null;
}

/// Round `[offset, offset + bytes)` out to whole `caps.map_granule_bytes` units, for the
/// importer that has to map it. A granule of 0 or 1 passes the range through unchanged. The
/// result never overflows: a range that reaches the top of the address space saturates at
/// `maxInt(u64)` instead of wrapping.
pub fn mappedRange(caps: Caps, offset: u64, bytes: u64) struct { base: u64, bytes: u64 } {
    const granule = caps.map_granule_bytes;
    if (!constrains(granule)) return .{ .base = offset, .bytes = bytes };

    const base = offset - (offset % granule);
    const end = std.math.add(u64, offset, bytes) catch std.math.maxInt(u64);
    const remainder = end % granule;
    const rounded_end = if (remainder == 0)
        end
    else
        std.math.add(u64, end, granule - remainder) catch std.math.maxInt(u64);
    return .{ .base = base, .bytes = rounded_end - base };
}

/// NVLink between two NVIDIA GPUs. A kernel dereferences a peer pointer directly and an atomic
/// carries defined ordering across the link, so `reach` is `.addressable` and `atomics` is true.
/// 64 KiB is the NVIDIA big page, the unit the importing GPU maps a peer allocation in. NVLink
/// keeps store ordering to the peer without a system-wide flush, so `release` is `.source_fence`.
pub const nvidia_nvlink: Caps = .{
    .reach = .addressable,
    .atomics = true,
    .map_granule_bytes = 64 * 1024,
    .copy_align = 4,
    .max_copy_bytes = 0,
    .release = .source_fence,
};

/// PCIe peer-to-peer between two NVIDIA GPUs with no NVLink between them. A kernel can still
/// dereference a peer pointer, so `reach` stays `.addressable`, but PCIe peer atomics are not
/// guaranteed by the specification, so `atomics` is false. A PCIe write can sit in a root
/// complex buffer until a system-wide fence drains it, so `release` is `.system_fence`.
pub const nvidia_pcie: Caps = .{
    .reach = .addressable,
    .atomics = false,
    .map_granule_bytes = 64 * 1024,
    .copy_align = 4,
    .max_copy_bytes = 0,
    .release = .system_fence,
};

/// Two runners on one NVIDIA GPU. This is not a peer link at all: it is a handle duplication and
/// a second mapping of memory the device already owns, so every field matches `nvidia_nvlink`.
/// It gets its own name because a caller asking "what can I do between these two runners" should
/// not have to know that the answer happens to equal NVLink's.
pub const nvidia_same_device: Caps = .{
    .reach = .addressable,
    .atomics = true,
    .map_granule_bytes = 64 * 1024,
    .copy_align = 4,
    .max_copy_bytes = 0,
    .release = .source_fence,
};

/// No link between the devices. A runtime that gets this must stage the transfer through host
/// memory: read the source to a host buffer, then write the host buffer to the destination.
pub const isolated: Caps = .{
    .reach = .none,
    .atomics = false,
    .map_granule_bytes = 1,
    .copy_align = 1,
    .max_copy_bytes = 0,
    .release = .none,
};

test "an aligned transfer within bounds is allowed" {
    const t: Transfer = .{ .src_offset = 64, .dst_offset = 128, .bytes = 256 };
    try std.testing.expectEqual(@as(?Refusal, null), check(nvidia_nvlink, t));
    try std.testing.expect(accepts(nvidia_nvlink, t));
}

test "an empty transfer is refused as empty, even with no link at all" {
    const t: Transfer = .{ .src_offset = 0, .dst_offset = 0, .bytes = 0 };
    try std.testing.expectEqual(Refusal.empty, check(isolated, t).?);
}

test "a transfer over a link with no reach is refused as no_link" {
    const t: Transfer = .{ .src_offset = 0, .dst_offset = 0, .bytes = 1 };
    try std.testing.expectEqual(Refusal.no_link, check(isolated, t).?);
}

test "a transfer above max_copy_bytes is refused as too_large before alignment is judged" {
    var caps = nvidia_nvlink;
    caps.max_copy_bytes = 8;
    // 9 bytes is also misaligned to copy_align 4, but too_large is reported first.
    const t: Transfer = .{ .src_offset = 0, .dst_offset = 0, .bytes = 9 };
    try std.testing.expectEqual(Refusal.too_large, check(caps, t).?);
}

test "a misaligned source offset is refused before the destination is judged" {
    const t: Transfer = .{ .src_offset = 1, .dst_offset = 1, .bytes = 4 };
    try std.testing.expectEqual(Refusal.misaligned_source, check(nvidia_nvlink, t).?);
}

test "a misaligned destination offset is refused when the source is aligned" {
    const t: Transfer = .{ .src_offset = 4, .dst_offset = 1, .bytes = 4 };
    try std.testing.expectEqual(Refusal.misaligned_destination, check(nvidia_nvlink, t).?);
}

test "a byte count that is not a whole number of copy units is refused as part_granule" {
    const t: Transfer = .{ .src_offset = 0, .dst_offset = 0, .bytes = 6 };
    try std.testing.expectEqual(Refusal.part_granule, check(nvidia_nvlink, t).?);
}

test "copy_align of zero allows any offset and any byte count" {
    var caps = nvidia_nvlink;
    caps.copy_align = 0;
    const t: Transfer = .{ .src_offset = 3, .dst_offset = 5, .bytes = 7 };
    try std.testing.expectEqual(@as(?Refusal, null), check(caps, t));
}

test "copy_align of one allows any offset and any byte count" {
    var caps = nvidia_nvlink;
    caps.copy_align = 1;
    const t: Transfer = .{ .src_offset = 3, .dst_offset = 5, .bytes = 7 };
    try std.testing.expectEqual(@as(?Refusal, null), check(caps, t));
}

test "a source range that would overflow the address space is refused as overflow" {
    const t: Transfer = .{
        .src_offset = std.math.maxInt(u64) - 3,
        .dst_offset = 0,
        .bytes = 8,
    };
    try std.testing.expectEqual(Refusal.overflow, check(nvidia_nvlink, t).?);
}

test "a destination range that would overflow the address space is refused as overflow" {
    const t: Transfer = .{
        .src_offset = 0,
        .dst_offset = std.math.maxInt(u64) - 3,
        .bytes = 8,
    };
    try std.testing.expectEqual(Refusal.overflow, check(nvidia_nvlink, t).?);
}

test "mappedRange rounds the base down and the end up to whole granules" {
    var caps = nvidia_nvlink;
    caps.map_granule_bytes = 4096;
    const r = mappedRange(caps, 5000, 3000);
    // 5000 rounds down to 4096, and 5000 + 3000 = 8000 rounds up to 8192.
    try std.testing.expectEqual(@as(u64, 4096), r.base);
    try std.testing.expectEqual(@as(u64, 8192 - 4096), r.bytes);
}

test "mappedRange passes an already-aligned range through with no widening" {
    var caps = nvidia_nvlink;
    caps.map_granule_bytes = 4096;
    const r = mappedRange(caps, 4096, 8192);
    try std.testing.expectEqual(@as(u64, 4096), r.base);
    try std.testing.expectEqual(@as(u64, 8192), r.bytes);
}

test "mappedRange with a granule of one leaves the range untouched" {
    var caps = nvidia_nvlink;
    caps.map_granule_bytes = 1;
    const r = mappedRange(caps, 5001, 37);
    try std.testing.expectEqual(@as(u64, 5001), r.base);
    try std.testing.expectEqual(@as(u64, 37), r.bytes);
}

test "mappedRange with a granule of zero leaves the range untouched" {
    var caps = nvidia_nvlink;
    caps.map_granule_bytes = 0;
    const r = mappedRange(caps, 5001, 37);
    try std.testing.expectEqual(@as(u64, 5001), r.base);
    try std.testing.expectEqual(@as(u64, 37), r.bytes);
}

test "mappedRange saturates instead of overflowing at the top of the address space" {
    var caps = nvidia_nvlink;
    caps.map_granule_bytes = 4096;
    // offset + bytes overflows u64, so the rounded end must saturate at maxInt rather than
    // wrap around to a small number.
    const near_top = std.math.maxInt(u64) - 10;
    const r = mappedRange(caps, near_top, 20);
    try std.testing.expectEqual(@as(u64, 0), r.base % 4096);
    try std.testing.expect(r.base <= near_top);
    try std.testing.expectEqual(std.math.maxInt(u64), r.base + r.bytes);
}

test "isolated refuses every non-empty transfer, staged or not" {
    const transfers = [_]Transfer{
        .{ .src_offset = 0, .dst_offset = 0, .bytes = 1 },
        .{ .src_offset = 17, .dst_offset = 31, .bytes = 100 },
    };
    for (transfers) |t| {
        try std.testing.expect(!accepts(isolated, t));
        try std.testing.expectEqual(Refusal.no_link, check(isolated, t).?);
    }
}

test "nvidia_pcie matches nvidia_nvlink except for atomics and its release fence" {
    try std.testing.expectEqual(nvidia_nvlink.reach, nvidia_pcie.reach);
    try std.testing.expectEqual(nvidia_nvlink.map_granule_bytes, nvidia_pcie.map_granule_bytes);
    try std.testing.expectEqual(nvidia_nvlink.copy_align, nvidia_pcie.copy_align);
    try std.testing.expect(nvidia_nvlink.atomics and !nvidia_pcie.atomics);
    try std.testing.expectEqual(Ordering.source_fence, nvidia_nvlink.release);
    try std.testing.expectEqual(Ordering.system_fence, nvidia_pcie.release);
}

test "nvidia_same_device is a mapping of nvidia_nvlink's caps, field for field" {
    try std.testing.expectEqual(nvidia_nvlink, nvidia_same_device);
}

test "isolated states no bound and no fence because there is no link to bound" {
    try std.testing.expectEqual(Reach.none, isolated.reach);
    try std.testing.expectEqual(Ordering.none, isolated.release);
    try std.testing.expectEqual(@as(u64, 0), isolated.max_copy_bytes);
}
