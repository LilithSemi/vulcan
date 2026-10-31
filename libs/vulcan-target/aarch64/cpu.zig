//! AArch64 register state and the result of one translated block.

const decode = @import("decode.zig");

/// The state of one guest cpu. `extern`, because compiled code addresses every field by offset.
pub const Cpu = extern struct {
    x: [31]u64 = @splat(0),
    sp: u64 = 0,
    /// The SIMD and floating point registers, each as its low and high 64 bits. A scalar view
    /// of one is its low bytes, and writing a scalar zeroes the rest of the register.
    v: [32][2]u64 = @splat(.{ 0, 0 }),
    pc: u64 = 0,
    /// NZCV, packed as in PSTATE: N at bit 31, Z at 30, C at 29, V at 28.
    flags: u32 = 0,
    trap: Trap = .none,
    address: u64 = 0,
    value: u64 = 0,
    /// The upper half of a 128-bit store, and of the second access of a pair of them.
    value_hi: u64 = 0,
    width: u8 = 0,
    /// The register a load fills. 0 to 30 is a general register and 31 is none. 32 and up is
    /// vector register `dest - 32`, which is how an access says which file it belongs to.
    dest: u8 = 0,
    /// Whether the load `trap` describes sign-extends what it reads, and to what width.
    /// The run loop consumes it when the load completes, which may be after a device has
    /// answered, and clears it, so a plain load after a sign-extending one is plain.
    load_signed: SignExtend = .none,
    /// Whether the access `trap` describes is an exclusive one, which the run loop
    /// treats differently from a plain load or store: a load-exclusive arms the
    /// monitor, and a store-exclusive checks it.
    exclusive: Exclusive = .none,
    /// The register a store-exclusive reports in: zero when it stored, one when it did not.
    status_dest: u8 = 0,
    /// The local exclusive monitor. A load-exclusive claims an address and a size, and a
    /// store-exclusive succeeds only if the claim is still held for the same ones. A
    /// store-exclusive clears it whether it succeeds or not, so does `CLREX`, and so does
    /// taking or returning from an exception.
    monitor_valid: bool = false,
    monitor_address: u64 = 0,
    monitor_width: u8 = 0,
    /// Whether a `SEV` arrived since the last `WFE`. `SEV` sets it, `WFE` consumes
    /// it: waiting with it set falls straight through, and waiting with it clear
    /// yields to the host until an interrupt or another `SEV` arrives.
    event_set: bool = false,
    /// A pair transfer is two accesses. The first is the one `address` names, and
    /// these hold the second, which the run loop performs once the first retires.
    second_pending: bool = false,
    second_address: u64 = 0,
    second_value: u64 = 0,
    second_value_hi: u64 = 0,
    second_width: u8 = 0,
    second_dest: u8 = 0,
    /// Whether the second half of a pair sign-extends what it reads. Only `LDPSW`
    /// sets it, and the run loop consumes and clears it with the access.
    second_signed: SignExtend = .none,
    /// A post-index access writes its base register only after the access retires,
    /// so the translated block leaves the new value here for the run loop. The flag
    /// says whether the access that just ran asked for one at all.
    writeback: bool = false,
    writeback_value: u64 = 0,
    writeback_dest: u8 = 0,
    /// The system registers, which are reached by name rather than by number.
    system: System = .{},
    /// The address and the reason of the last fault, which is what an exception
    /// handler reads out of `FAR_EL1` and `ESR_EL1`. They are recorded even though
    /// taking the exception is not built yet, so that building it is a matter of
    /// routing to a vector rather than of working out what went wrong afterwards.
    fault_address: u64 = 0,
    fault_status: u64 = 0,

    /// The system registers, kept apart from the general registers because they are
    /// reached by name rather than by number, and because a kernel reads and writes
    /// them in the first few instructions it ever runs.
    pub const System = extern struct {
        /// The exception level the guest is running at. The kernel reaches it as
        /// `CurrentEL`, and the stack pointer views below depend on it.
        el: u8 = 1,
        /// Current PSTATE.SP selection: false is SP_EL0, true is SP_EL1 at EL1.
        /// An EL1 guest enters in EL1h; exception tests can select EL1t explicitly.
        spsel: bool = true,
        /// The stack pointer banked at each exception level.
        sp_el: [4]u64 = @splat(0),
        spsr_el1: u64 = 0,
        elr_el1: u64 = 0,
        esr_el1: u64 = 0,
        far_el1: u64 = 0,
        vbar_el1: u64 = 0,
        cpacr_el1: u64 = 0,
        sctlr_el1: u64 = 0,
        ttbr0_el1: u64 = 0,
        ttbr1_el1: u64 = 0,
        tcr_el1: u64 = 0,
        mair_el1: u64 = 0,
        tpidr_el0: u64 = 0,
        tpidrro_el0: u64 = 0,
        /// The kernel's per-CPU base, which it reads on almost every function.
        tpidr_el1: u64 = 0,
        mdscr_el1: u64 = 0,
        cntkctl_el1: u64 = 0,
        /// The debug lock registers. Both read held at reset, which is what makes
        /// the status read locked until the kernel clears them. They carry no debug
        /// state here, so a write is remembered and the status derives from them.
        oslar_el1: u64 = 1,
        osdlr_el1: u64 = 1,
        fpcr: u64 = 0,
        fpsr: u64 = 0,
        /// The virtual counter. It advances while the guest runs, so a kernel that
        /// samples it twice sees time pass; a guest that only reads it needs no timer.
        cntvct_el0: u64 = 0,
        /// The virtual timer. Bit 0 of the control is the enable, bit 1 the mask,
        /// and bit 2 the condition: set when the counter has reached the compare and
        /// then held set, even if the timer is disabled or masked afterwards. The
        /// compare (`CNTV_CVAL_EL0`) is the counter value the condition waits for.
        /// Only these bits of the control are kept; the rest read zero.
        cntv_ctl_el0: u64 = 0,
        cntv_cval_el0: u64 = 0,
        /// The interrupt masks D, A, I and F, held as the four bits 3:0 that the
        /// immediate forms of `MSR` name and that an exception saves and restores.
        /// The register forms of `MSR` and `MRS` put them at 9:6, and convert. All
        /// four are set, which is what a kernel expects at entry.
        daif: u64 = 0xf,
    };

    /// Which kind of exclusive access a load or store is, if it is one.
    pub const Exclusive = enum(u8) { none, load, store };

    /// How a load extends the bytes it read into the register. The type is the
    /// decoder's, so that what a block records and what the run loop reads cannot drift.
    pub const SignExtend = decode.SignExtend;

    pub const Trap = enum(u8) {
        none,
        psci,
        wfi,
        /// `WFE`: wait for the event flag, an interrupt, or the virtual timer. The
        /// translated block clears the flag going in and the run loop decides there
        /// whether to sleep.
        wfe,
        load,
        store,
        svc,
        brk,
        /// An `ISB`, which orders nothing here but does mean the translation cache
        /// is stale. It ends a block so the run loop can act on that.
        sync,
        /// A `TLBI`, which says outright that what the translation cache holds is stale.
        tlbi,
        /// An access that address translation refused. The reason is in `esr`, and
        /// the faulting address in `far`, which is what an exception handler reads.
        translation,
        /// `DC ZVA`: zero the cache line holding `address`. The run loop does it,
        /// because it is a write to memory and the tables decide where it lands.
        dc_zva,
        /// `ERET`. The run loop restores the saved state and sets the program
        /// counter, since the destination is an address the translator never saw.
        eret,
    };
};
