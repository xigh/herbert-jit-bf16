//! The decode attention (SDPA), whatever the target: one query head against
//! the cached keys and values, and the prefill block of `rows` queries.
//!
//! ```text
//! scores[j] = (Σ_d q[d] · bf16→f32(K[j][d])) · scale     j < n
//! scores    = softmax(scores)            (with or without a sink)
//! out[d]    = Σ_j scores[j] · bf16→f32(V[j][d])
//! ```
//!
//! `K` and `V` arrive by head, in contiguous rows of `head_dim` bf16. The
//! score buffer contract: the QK pass writes `n` scores then pads up to the
//! next multiple of 16 with `-inf`; the caller sizes `scores` accordingly
//! (`sdpa.rs` checks it).
//!
//! ## ABI — no context
//!
//! ```text
//! qk       (q: *const u16, k: *const u16, scores: *mut f32, n: u64, scale: f32)
//! softmax  (scores: *mut f32, n: u64, sink: f32)
//! sv       (scores: *const f32, v: *const u16, out: *mut u16, n: u64)
//! head     (q, k, v, scores, out, n, scale: f32, sink: f32)   // Chained, Fused
//! prefill  (q, k, v, scores, out, past, scale: f32, sink: f32)
//! ```
//!
//! The emitters: `target/x86/emit_sdpa/` (AVX-512), `target/x86/emit_sdpa_avx2.rs`,
//! `target/aarch64/sdpa.rs` (NEON). The register budgets written in the
//! checks below are those of the AVX-512 kernel, the most constrained domain;
//! the other targets accept what it accepts.

use crate::jit::ir::{Decline, ExpStrategy, SdpaForm, SdpaShape};

/// Prefetch distance of the decode kernel, in K/V rows. Not swept: decode is
/// bus-bound and the prefetch is a hint that changes no output bit.
pub const SDPA_PF_DIST: usize = 4;

/// Prefetch distance of the prefill block kernel: none. The sweep
/// `pf_dist ∈ {0, 2, 4, 8, 16, 32}` of the QK pass is flat and `0` is the
/// fastest point (7 127 ns against 7 351 at 4, −3 %): preloading `K` bought
/// no stall cycle and cost an instruction. Measured at `past` 1 024, `K` in
/// L2, before the pairing of queries. `Knobs::sdpa_prefetch` imposes one
/// value on both kernels.
pub const SDPA_PF_DIST_PREFILL: usize = 0;

/// Everything the SDPA emitter needs.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SdpaKernelSpec {
    /// Shape frozen at load time.
    pub shape: SdpaShape,
    /// Work splitting (axis 2).
    pub form: SdpaForm,
    /// Exponential calculation (axis 1).
    pub exp: ExpStrategy,
    /// Prefetch distance in rows; 0 = no prefetch.
    pub pf_dist: usize,
    /// QK by groups of positions: `0` = no, otherwise **8 or 16**, like
    /// [`SdpaPrefillSpec::qk_group`] — and for the same reason.
    ///
    /// Here there is only **one** request and many positions: this is the
    /// most favorable case of the two, `q` held in registers for the entire
    /// loop instead of being reloaded per block line.
    ///
    /// **This field exists first to ensure bit-wise equality holds.** The
    /// group-based prefill block sums `n32` blocks of 32 bf16 into a
    /// single chain; as long as decode keeps four, the two paths
    /// differ by at most one bf16 step — but they differ, and a prefill
    /// block no longer matches the repeated decode bit for bit. The two
    /// ends are thus resolved **together** (`sdpa_qk_group()`), never
    /// separately.
    ///
    /// **The group size, however, changes nothing about the bits**: the
    /// accumulation chain remains unique and the horizontal reduction keeps the same
    /// pairing (cf. `qk_reduce`). 8 and 16 yield the same scores.
    pub qk_group: u8,
    /// Dump **all** output accumulators from the SV pass into the
    /// f32 scratch buffer, instead of only those that do not fit in registers.
    ///
    /// This is a **measurement arm**, not a setting to leave lying around: at
    /// `head_dim` 256 both paths are emit-able and produce the same bits, so
    /// their difference isolates exactly *what an accumulator costs in L1* —
    /// a measurement no comparison at 512 can provide, for lack of a register
    /// reference.
    ///
    /// It is in the cache key because the two arms must be able to
    /// be **alternated within the same process**: this is the project's measurement rule,
    /// and an inter-process sweep does not separate 4 %.
    pub sv_spill_all: bool,
}

impl Default for SdpaKernelSpec {
    fn default() -> Self {
        Self {
            shape: SdpaShape {
                head_dim: 256,
                sink: false,
            },
            form: SdpaForm::default(),
            exp: ExpStrategy::default(),
            pf_dist: SDPA_PF_DIST,
            qk_group: 0,
            sv_spill_all: false,
        }
    }
}

/// The three passes, when emitted as separate blobs
/// ([`SdpaForm::Split`]).
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Pass {
    /// `scores[j] = dot(q, K[j]) · scale`, then `-inf` padding.
    Qk,
    /// Softmax in-place on the padded buffer.
    Softmax,
    /// `out[d] = Σ_j scores[j] · V[j][d]`.
    Sv,
}

/// `qk` blob signature. `q` is in **bf16** (engine invariant).
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `target::generated_fn!`.
pub type SdpaQkFn = crate::jit::target::generated_fn!(*const u16, *const u16, *mut f32, u64, f32);
/// Signature of the `softmax` blob.
pub type SdpaSoftmaxFn = crate::jit::target::generated_fn!(*mut f32, u64, f32);
/// Signature of the `sv` blob. The output is in **bf16**.
pub type SdpaSvFn = crate::jit::target::generated_fn!(*const f32, *const u16, *mut u16, u64);
/// Signature of the blob "a full head" ([`SdpaForm::Chained`], [`SdpaForm::Fused`]).
///
/// `sysv64` and not `C`, like every kernel signature: under the Microsoft
/// ABI the caller would put `v` in `r8` and `scores` in `r9` and pass `out`
/// and `n` on the stack, and the kernel would run off the end of a buffer.
/// Under Linux `"C"` is System V, so only Windows tells.
pub type SdpaHeadFn = crate::jit::target::generated_fn!(
    *const u16,
    *const u16,
    *const u16,
    *mut f32,
    *mut u16,
    u64,
    f32,
    f32,
);

/// Lines that a prefill block can carry.
///
/// The limit is not decorative: the SV pass holds `rows` base line scores
/// **in integer registers** (`rcx, rdx, r8, r9, r11, r12, r13,
/// r14`), so that score diffusion costs a `vbroadcastss` and not an extra
/// `mov` reload instruction. Beyond eight lines, a memory table would be needed,
/// thus one extra instruction per query and per position: this is decided by
/// benchmark, not by anticipation.
pub const SDPA_PREFILL_MAX_ROWS: usize = 8;

/// Queries per block of the AArch64 block attention: the sixteen rows of
/// one tile of the matrix engine (`target/aarch64/sdpa_block.rs`).
pub const SDPA_BLOCK_ROWS: usize = 16;

/// Everything the SDPA **prefill** emitter needs.
///
/// A block of `rows` **consecutive** queries from the same head, against the same
/// K/V. That's the only difference with [`SdpaKernelSpec`], and that's the whole
/// deal: `K[j]` and `V[j]` are read **once for `rows` queries**, whereas the
/// decode kernel called `rows` times reads them `rows` times.
///
/// ## What this kernel does NOT change: the order of summation
///
/// Each request sees exactly the same sequence of floating-point operations as
/// in the **decode** kernel — same accumulation chains, same reduction tree,
/// same positions in the same order. The gain comes from the **residence**
/// (which loop is outside, what stays in registers), not from reassociation.
/// This is what allows the strongest thread on the construction site — the
/// **bit-to-bit** equality with repeated decode — to survive this kernel
/// instead of being traded for tolerance.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SdpaPrefillSpec {
    /// Shape fixed at load time, like for decode.
    pub shape: SdpaShape,
    /// Requests per block. **Emission constant**: register allocation
    /// depends on it. A prompt of `m` rows is processed in `m / rows` blocks, the
    /// remainder being handled by the decode kernel.
    pub rows: usize,
    /// Distance, in u16, between two lines of `q` — it's `n_heads · head_dim`
    /// of the scheduler. Fixed at load time, so an immediate offset
    /// in the emitted code rather than multiplication.
    pub row_stride: usize,
    /// Distance, in u16, between two lines of `out`.
    ///
    /// **Equal to [`Self::row_stride`] everywhere except on the output gate of
    /// Qwen3.5**: there, `q_proj` outputs `[Q₀ g₀ Q₁ g₁ …]` and the `stride`
    /// variant of the deinterleaving reads `q` with a step of `2·q_dim` while writing
    /// `attn` with a step of `q_dim`. The two steps could not be the same
    /// immediate — this is what made the "no copy" variant more expensive
    /// than §4.4 of directory 35 suggested.
    pub out_row_stride: usize,
    /// Exponential calculation, as for decode.
    pub exp: ExpStrategy,
    /// Prefetch distance in rows; 0 = no prefetch.
    pub pf_dist: usize,
    /// **16-dimensional** blocks that the SV pass accumulates together.
    ///
    /// `0` = take the default of [`SdpaPrefillSpec::widest_dim_blocks`].
    ///
    /// This is the second axis of the kernel, and it **cannot be chosen after `rows`**:
    /// `rows` amortizes the read of `V` (one row serves `rows` queries), `nb`
    /// amortizes the broadcast of the score (paid once per query/position pair,
    /// amortized over `nb` `fma`s), and the two compete for the
    /// 32 registers — `rows·nb + nb + 1 ≤ 32`. Nothing says that the largest
    /// `rows` wins: `8×2` yields 16 `fma`s over 28 instructions, `6×4` yields 24
    /// over 38.
    pub dim_blocks: usize,
    /// QK by position groups, reduction in transposed form: `0` = no,
    /// otherwise **8 or 16**.
    ///
    /// QK line-by-line costs 19.5 instructions per pair (query,
    /// position) with **4 useful compute** : the rest is zeroing the
    /// chains and horizontal reduction, both **per query**, thus
    /// insensitive to `rows`. Sixteen positions in flight amortize the
    /// reduction over sixteen scores and bring the pair down to ≈ 8.2 instructions.
    ///
    /// **Eight positions cost the same count, in half the code**:
    /// 1 zeroing + `n32` `vdpbf16ps` + 3.0 of reduction (24 instructions
    /// for 8, versus 45 for 16) + 0.25 of scale and store, i.e. ≈ 8.25
    /// versus 8.2. This is therefore **not** a second throughput tuning point:
    /// it is the lever that separates "the kernel is limited by its µops" from
    /// "the kernel is limited by the size of its unrolling" — the body of the
    /// group loop goes from ≈ 1 050 instructions to ≈ 530.
    ///
    /// At 8, the scores come out **duplicated** in the 16 lanes: the store
    /// is a masked `vcompressps` `0x3333` (8 contiguous f32) instead of a
    /// `vmovups`. This is the only cost that doesn't exist at 16, and it is counted
    /// above.
    ///
    /// **This is the only kernel setting that changes the summation order.**
    /// Sixteen accumulators forbid the four chains of the kernel line by
    /// line (16 × 4 = 64 registers): the `n32` blocks of 32 bf16 accumulate
    /// in a single chain, `((c0+c1)+c2)+c3` instead of `(c0+c1)+(c2+c3)`.
    /// The horizontal reduction, however, is reproduced **up to pairing**
    /// (see `qk_reduce`) and contributes nothing — and it remains so at 8, where
    /// the last level pairs with itself instead of a neighbor.
    ///
    /// Measured gap (300 000 draws, on a Zen 5): at
    /// `head_dim` 128, **96.8 % of scores are bit-identical**, and
    /// the maximum gap, once the scale is applied, is 3.4e−7 — three
    /// orders of magnitude below bf16 resolution. Neither order
    /// is more accurate than the other: against an f64 reference, the
    /// single chain is at +0.32 % of the mean error of the four chains.
    ///
    /// The bit-for-bit net is not lost, it is moved: it holds as soon as the
    /// decode kernel adopts the same order, and both do at the default of 8.
    pub qk_group: u8,
    /// Requests that **share each `K` load** in the group
    /// loop. `1` = the historical kernel, `K` as a memory operand of a
    /// `vdpbf16ps` per request.
    ///
    /// ## Why this field exists
    ///
    /// Grouped QK reloads `K` **from memory for each query**:
    /// `n32` loads per pair (query, position), i.e., 4 at `head_dim` 128.
    /// They touch the L1 — the `g` lines of the group are 2 KiB — but each
    /// consumes a **loader queue entry**, and that's where the kernel
    /// stalls: `PMCx0AE` bit 1 accounts for **2.82 of the 4.63 cycles per pair**,
    /// i.e., 61 %, versus 0.00 for the row-by-row kernel (measured with the
    /// PMU on a Zen 5).
    ///
    /// For `p` paired queries, the `K` segment is loaded **once** into
    /// a register and then read by `p` FMAs: loads per pair are divided
    /// by `p`, at the cost of one extra instruction for `p` FMAs (1.5 per FMA at
    /// `p = 2` instead of 1). This is an **instruction vs. queue entry** trade-off,
    /// and the latter currency is what matters.
    ///
    /// **No change in summation order**: each accumulator receives
    /// the same segments in the same order, only the source of the operand
    /// changes. Bit-wise equality with `p = 1` and with repeated decode is the
    /// recipe for this field, not an expected property.
    ///
    /// Paired queries here are `p` **consecutive lines of the block**,
    /// so `p` positions of the same head. The same mechanism will serve `p` **heads
    /// q** of the same kv head when the block carries several (GQA): only
    /// the address of `q` changes.
    ///
    /// Registers: `p·(g + n32) + 3`. At `g = 8`, `head_dim` 128, `p = 2`: 27
    /// of the 32. Beyond that the emitter refuses ([`Decline::OutOfRegisters`]).
    pub qk_pair: u8,
    /// Positions reserved per KV head: the length of the scratch of a block,
    /// and, on AArch64, the distance between two blocks of the `Vᵀ` R16 copy
    /// that the block kernel reads (`target/aarch64/sdpa_block.rs`).
    pub kv_capacity: usize,
}

/// Prefill blob signature.
///
/// `(q, k, v, scores, out, past, scale, sink)` — `past` is the number of
/// positions already in the cache **before** this block. Line `i` sees
/// `past + i + 1` keys: causality is in the geometry, not in an
/// argument.
/// `sysv64` and not `C`, like [`SdpaHeadFn`].
pub type SdpaPrefillFn = crate::jit::target::generated_fn!(
    *const u16,
    *const u16,
    *const u16,
    *mut f32,
    *mut u16,
    u64,
    f32,
    f32,
);

impl SdpaPrefillSpec {
    /// Can this target emit the block kernel? On AArch64 the block kernel is
    /// the two-GEMM attention of `target/aarch64/sdpa_block.rs`, with its own
    /// rules; the rest of this impl describes the x86 kernel.
    pub fn check(&self) -> Result<(), Decline> {
        #[cfg(target_arch = "aarch64")]
        {
            crate::jit::target::aarch64::sdpa_block::check(self)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.check_chained()
        }
    }

    /// The scratch of one block, in f32, for a cache of `kv_capacity`
    /// positions: one slot per participant (`schedule/scheduler.rs`).
    pub fn scratch_len(&self) -> usize {
        #[cfg(target_arch = "aarch64")]
        {
            crate::jit::target::aarch64::sdpa_block::scratch_f32(self)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            self.rows * self.kv_capacity.next_multiple_of(16).max(16)
        }
    }

    /// Is the chained form eligible? (the x86 block kernel)
    #[cfg_attr(target_arch = "aarch64", allow(dead_code))]
    fn check_chained(&self) -> Result<(), Decline> {
        // The chained shape is the one from which this kernel reuses the three passes.
        self.shape.check(SdpaForm::Chained)?;
        if self.rows == 0 {
            return Err(Decline::Empty);
        }
        if self.rows > SDPA_PREFILL_MAX_ROWS
            || self.row_stride < self.shape.head_dim
            || self.out_row_stride < self.shape.head_dim
        {
            return Err(Decline::PrefillRows);
        }
        // Accumulators for the SV pass: `rows · nb` + `nb` rows of V + 1
        // scattered score. If the pair doesn't fit, say it here rather than
        // letting the allocator discover it.
        let nb = self.sv_dim_blocks();
        if nb == 0 || !self.shape.vecs().is_multiple_of(nb) || self.rows * nb + nb + 1 > 32 {
            return Err(Decline::OutOfRegisters);
        }
        // QK by groups: `g` accumulators + `n32` blocks of `q` + a temporary
        // + the broadcasted scale, plus the callee's scratchpad if
        // applicable. The bound is checked here rather than discovered by the
        // allocator in the middle of an emission.
        if self.qk_group != 0 {
            if !matches!(self.qk_group, 8 | 16) {
                return Err(Decline::QkGroup);
            }
            // `p·(g + n32)` for the accumulators and `q` rows, plus the
            // reduction temporary, the broadcasted scale, and — as soon as the
            // queries are paired — the resident `K` chunk.
            let p = self.qk_pair();
            let held = p * (self.qk_group as usize + self.shape.vecs().div_ceil(2));
            if held + 2 + usize::from(p > 1) > 32 {
                return Err(Decline::OutOfRegisters);
            }
        } else if self.qk_pair > 1 {
            // Load sharing only makes sense in the group loop:
            // the row-by-row kernel takes `q` as a memory operand and re-reads
            // the **same** addresses from one position to the next. Stating this, rather
            // than accepting a button that would do nothing.
            return Err(Decline::QkGroup);
        }
        Ok(())
    }

    /// Paired requests per `K` load: the field
    /// [`SdpaPrefillSpec::qk_pair`], `1` if nobody said anything.
    pub fn qk_pair(&self) -> usize {
        (self.qk_pair as usize).max(1)
    }

    /// 16-dimensional blocks processed together by the SV pass: the field
    /// [`SdpaPrefillSpec::dim_blocks`], or the default if it is `0`.
    pub fn sv_dim_blocks(&self) -> usize {
        match self.dim_blocks {
            0 => Self::widest_dim_blocks(self.shape, self.rows),
            nb => nb,
        }
    }

    /// The flaw: the largest divisor of `head_dim/16` that fits in the
    /// 32 zmm.
    ///
    /// **A flaw, not a verdict** — it has never been swept, and it only
    /// maximizes `nb` given `rows`, which is not the same problem as maximizing
    /// the throughput of the pair. At `nb = 2`, the read of `V` makes
    /// exactly **one cache line** per position and per group, and `V` is
    /// traversed only once across all groups regardless of `nb` —
    /// that is the property that matters, and it does not depend on the choice.
    pub fn widest_dim_blocks(shape: SdpaShape, rows: usize) -> usize {
        let nv = shape.vecs();
        (1..=nv)
            .rev()
            .find(|nb| nv.is_multiple_of(*nb) && rows * nb + nb < 32)
            .unwrap_or(0)
    }

    /// Length of a **line** in the scores buffer, in f32, for `past`
    /// positions already in cache.
    ///
    /// The `rows` lines have the same stride: the block computes a rectangle of
    /// `past + rows` columns, rounded up to the next multiple of 16 because the
    /// softmax works in blocks of 16 without a mask (padding contract `-inf`,
    /// see the module header).
    pub const fn scores_row(&self, past: usize) -> usize {
        (past + self.rows).div_ceil(16) * 16
    }

    /// Total length required for the scores buffer, in f32.
    pub const fn scores_len(&self, past: usize) -> usize {
        self.rows * self.scores_row(past)
    }
}

/// Roles of the SV pass that are **not** accumulators: the rotary pool
/// of four temporaries (loading of `V`, then bf16 conversion of the
/// epilogue) and the scattered score.
///
/// This 5 fixes everything else: at `head_dim` 512 the pass requests 32
/// accumulators for 27 available registers, so **five** segments
/// overflow — not thirty-two. The formula "512 requests all registers,
/// so everything must go to memory" is wrong by a factor of six.
pub const SV_FIXED_REGS: usize = 5;

impl SdpaKernelSpec {
    /// Output accumulators that the SV pass keeps in **registers**.
    ///
    /// 27 registers remain: any `head_dim` ≤ 416 fits entirely (416
    /// is the largest multiple of 32 under 27·16). Beyond that, the surplus goes to
    /// scratch. [`SdpaKernelSpec::sv_spill_all`] forces the degenerate case where
    /// none fit — the measurement arm.
    pub fn sv_acc_in_regs(&self) -> usize {
        if self.sv_spill_all {
            return 0;
        }
        // The exp is emitted in the blob: no caller scratch to reserve.
        let reserved = SV_FIXED_REGS;
        self.shape.vecs().min(32 - reserved)
    }

    /// 16-dimension chunks accumulated in the **f32 draft** on the stack.
    ///
    /// The draft is written and reread **at each position** — so hot in L1
    /// and compliant with the reformulated invariant ("f32 never reaches DRAM") —
    /// and converted to bf16 by the epilogue before leaving the kernel.
    pub fn sv_spill(&self) -> usize {
        self.shape.vecs() - self.sv_acc_in_regs()
    }

    /// Shape and combination emittable?
    pub fn check(&self) -> Result<(), Decline> {
        self.shape.check(self.form)?;
        if self.form == SdpaForm::Fused && self.shape.sink {
            return Err(Decline::SinkInFused);
        }
        // The fused form holds `head_dim/16` live accumulators during the
        // softmax and needs every register: the sink would not fit.
        if self.qk_group != 0 {
            if !matches!(self.qk_group, 8 | 16) {
                return Err(Decline::QkGroup);
            }
            // The fused form carries its own QK, interleaved with the softmax:
            // the position group has no place in it. State it, rather than
            // silently ignoring the button — a measurement arm that does not measure
            // what it announces is worse than a refusal.
            if self.form == SdpaForm::Fused {
                return Err(Decline::Qk16InFused);
            }
            // `g` accumulators + `n32` blocks of `q` + a temporary + the scale
            // broadcast, plus the scratch of the exp callee and, in chained
            // form, the sink that traverses the three passes.
            let across = usize::from(self.form == SdpaForm::Chained);
            if self.qk_group as usize + self.shape.vecs().div_ceil(2) + 2 + across > 32
            {
                return Err(Decline::OutOfRegisters);
            }
        }
        Ok(())
    }
}


/// Generates the decode kernel of `spec`, one pass of it or the whole, for
/// the target of this build.
pub fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::target::emit_sdpa(spec, pass)
}

/// Generates the prefill block kernel of `spec` for the target of this build.
pub fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::target::emit_sdpa_prefill(spec)
}
