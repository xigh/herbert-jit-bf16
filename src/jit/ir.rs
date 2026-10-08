//! What the emitters receive, and what they refuse.
//!
//! The shapes are fixed at load time: `K` and `N` of a GEMM, `head_dim` of an
//! attention head, the tile of a micro-kernel. A kernel is emitted for one
//! shape, with immediates instead of loop bounds; `M` (the rows of a block)
//! is the only run-time dimension. A shape an emitter cannot serve is a
//! named [`Decline`], never an error: the caller falls back to the row-by-row
//! kernel or refuses the model at load.

/// Shape of the micro-**kernel**: `mr` rows of `C` by `nr` columns of `C` and per
/// **kernel** call.
///
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct TileShape {
    /// Lines of `C` (thus of `A`) processed per call.
    pub mr: usize,
    /// Columns of `C` (hence rows of `W`) processed per call.
    pub nr: usize,
}

impl TileShape {
    /// Number of R16 blocks covered by the tile.
    pub const fn n_blocks(&self) -> usize {
        self.nr / ROWS_INTERLEAVED
    }
}

impl std::fmt::Display for TileShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}x{}", self.mr, self.nr)
    }
}

/// Rows interleaved in one block of the R16 layout (`model/weight.rs`).
pub const ROWS_INTERLEAVED: usize = crate::jit::target::R16_ROWS;

/// A GEMM shape as known by the model loader.
///
/// `K` and `N` are fixed at load time; `M` is not and does not appear here.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GemmShape {
    /// Contraction dimension.
    pub k: usize,
    /// Output columns (= rows of `W`).
    pub n: usize,
}

/// Shape of a decode SDPA, as the model loader knows it.
///
/// `head_dim` is the K/N of this **kernel** — immediate, unrolled loops, no
/// tail. Only `cached_len` remains a runtime bound
/// (the "M": a loop counter). The presence of a sink changes the softmax
/// and thus is part of the shape, not the arguments.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct SdpaShape {
    /// Per-head dimension. Multiple of 16, ≤ [`SDPA_MAX_HEAD_DIM`].
    pub head_dim: usize,
    /// Softmax with attention sink (Gemma family).
    pub sink: bool,
}

/// Upper bound of `head_dim` accepted by the SDPA emitter.
///
/// The invariant is "f32 must not reach DRAM": a head wider than the
/// registers spills its accumulators to a stack scratch that is written and
/// re-read in L1, converted to bf16 before leaving the kernel.
pub const SDPA_MAX_HEAD_DIM: usize = 512;

/// How the generated code computes the softmax exponential.
///
/// Both were measured. They share the same coefficients, so the expected gap
/// is in speed,
/// not in result.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum ExpStrategy {
    /// The `INLINE_EXP16` polynomial is **re-emitted** in the blob, constants
    /// materialized in registers (`mov eax, imm` + `vpbroadcastd`) rather than
    /// read RIP-relative: the blob remains position-independent and has no
    /// symbols to resolve.
    #[default]
    Inline,
    // There is no `Call` variant: the generated code never leaves its own
    // page, and no hand-written assembly exists in the engine.
}

impl ExpStrategy {
    /// Short name, as it appears in `HERBERT_JIT_SDPA_EXP` and logs.
    pub const fn name(self) -> &'static str {
        match self {
            ExpStrategy::Inline => "inline",
        }
    }
}

/// Splitting work between generated blobs and calling Rust code.
///
/// The three emit genuinely different code shapes; the gap between them is
/// what the fusion buys, and it was measured.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub enum SdpaForm {
    /// Three blobs — QK, softmax, SV — chained by the adapter's Rust.
    /// Each pass is compliant separately; the scores go back through
    /// memory between two calls.
    Split,
    /// One blob per head: the three passes follow one another in the same code, the
    /// scores buffer never leaves the L1, and there is only one call.
    #[default]
    Chained,
    /// One blob per head, **online softmax** (decode flash-attention):
    /// blocks of 16 positions, current maximum and sum, output
    /// accumulator rescaled at each block. A single traversal of the KV cache instead
    /// of two, and no score buffer. Requires `head_dim` accumulators in
    /// registers: the emitter declines beyond what the 32 zmm can carry.
    Fused,
}

impl SdpaForm {
    /// Short name, as it appears in `HERBERT_JIT_SDPA_FORM` and the logs.
    pub const fn name(self) -> &'static str {
        match self {
            SdpaForm::Split => "split",
            SdpaForm::Chained => "chained",
            SdpaForm::Fused => "fused",
        }
    }
}

impl SdpaShape {
    /// Shapes that the SDPA emitter knows how to specialize. See [`Decline`].
    pub fn check(&self, form: SdpaForm) -> Result<(), Decline> {
        if self.head_dim == 0 {
            return Err(Decline::Empty);
        }
        if !self.head_dim.is_multiple_of(32) || self.head_dim > SDPA_MAX_HEAD_DIM {
            return Err(Decline::HeadDim);
        }
        // The fused shape keeps the output accumulator in registers throughout
        // the loop: `head_dim/16` zmm, plus the 8 exp constants, the current
        // maximum and sum, the scores block, and three temporaries. Beyond that,
        // it does not fit — and a memory accumulator would cost it exactly what
        // it is trying to gain.
        //
        // The `Split` and `Chained` forms serve 512: they dump the excess
        // segments into a stack-based f32 scratchpad,
        // re-read at the next position and converted to bf16 at the epilogue.
        if form == SdpaForm::Fused && self.head_dim / 16 > FUSED_MAX_ACC {
            return Err(Decline::HeadDim);
        }
        Ok(())
    }

    /// 16 f32 vectors per K/V row.
    pub const fn vecs(&self) -> usize {
        self.head_dim / 16
    }

    /// Distance in bytes between two K or V rows (head layout:
    /// `[cached_len, head_dim]` contiguous, see `decode.rs:872-883`).
    pub const fn kv_row_bytes(&self) -> usize {
        self.head_dim * 2
    }
}

/// Output accumulators that the fused shape can fit in registers.
///
/// 16 output zmm (`head_dim` 256), 8 exp constants, the current max and sum,
/// the block scores and three temporaries: 30 of the 32 registers. `head_dim`
/// 512 would need 32 for the output alone, so the fused form declines.
///
/// The chained form serves `head_dim` 512 by spilling the excess segments
/// (see `emit_sv`).
pub const FUSED_MAX_ACC: usize = 16;

/// Why the emitter refuses a shape. The refusal is not an error: the caller
/// falls back or refuses the model by name.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Decline {
    /// `K` is odd. The R16 layout zero-pads `B` up to `k_even`, but `A` is
    /// a contiguous `[M, K]`: reading the `K` index pair of the last row
    /// goes out of the buffer.
    OddK,
    /// `N` is not a multiple of 16: the last R16 block is partial, and
    /// a 64-byte `vmovups` would write beyond the `C` row.
    PartialRowBlock,
    /// ShapeOf empty.
    Empty,
    /// `nr` is not a multiple of 16, or the tile does not fit in the
    /// 32 vector registers.
    TileTooWide,
    /// A displacement in the generated code would exceed the 32-bit immediate.
    DisplacementOverflow,
    /// `head_dim` is not a multiple of 16, exceeds
    /// [`SDPA_MAX_HEAD_DIM`], or does not fit in the requested shape.
    HeadDim,
    /// [`SdpaForm::Fused`] with [`ExpStrategy::Call`]: the external call destroys
    /// zmm0-15 (`vzeroupper`), where the fused form has no register
    /// free. Refusal **of combination**, not of form.
    ExpCallInFused,
    /// [`SdpaForm::Fused`] does not implement the attention sink term. Emitting
    /// this combination would silently compute ordinary attention.
    SinkInFused,
    /// [`SdpaForm::Fused`] with grouped QK: the fused form
    /// interleaves its QK with the softmax and cannot hold a group of positions
    /// in flight. **Combination refusal**, as above.
    Qk16InFused,
    /// The shape or expansion strategy is outside the **scope of the AVX2 kernel**.
    ///
    /// **Refusal to port**, not a shape issue: [`SdpaForm::Fused`] already saturates
    /// the thirty-two AVX-512 registers and only has sixteen here,
    /// [`SdpaForm::Split`] is not used for decode, and [`ExpStrategy::Call`]
    /// calls an AVX-512 `.S`. Stating this is better than emitting a trimmed-down
    /// variant whose value no one would know.
    OutOfScopeAvx2,
    /// QK group size other than 8 or 16. The transposed reduction
    /// fits within four pairing levels — this is what fixes the two
    /// emittable sizes, not an interface choice.
    QkGroup,
    /// The allocator has no more registers to return in the requested area
    /// (`regs.rs`). A refusal, not a panic: it's up to the caller to fallback
    /// to a less greedy shape.
    OutOfRegisters,
    /// Out-of-domain prefill block: zero rows, more than
    /// [`SDPA_PREFILL_MAX_ROWS`](crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS), or
    /// a stride of rows shorter than `head_dim`.
    PrefillRows,
    /// The machine does not have AVX512-VNNI: `VPDPBUSD` does not exist, so no
    /// GEMM Q4 is generated. Machine **refusal**, no shape — the caller
    /// falls back to the `backend_q4` kernel, which has its AVX2/scalar dispatch.
    NoVnni,
}

impl std::fmt::Display for Decline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Decline::OddK => "Odd K (not padded)",
            Decline::PartialRowBlock => "N is not a multiple of 16",
            Decline::Empty => "empty shape",
            Decline::TileTooWide => "tile out of register budget",
            Decline::DisplacementOverflow => "displacement > 2 GiB",
            Decline::HeadDim => "head_dim out of domain (a multiple of 16, ≤ 512, holdable in registers)",
            Decline::ExpCallInFused => "External exp incompatible with fused shape",
            Decline::SinkInFused => "Attention sink incompatible with fused shape",
            Decline::OutOfScopeAvx2 => "outside the AVX2 kernel scope (shape or exp strategy)",
            Decline::Qk16InFused => "Grouped QK incompatible with fused shape",
            Decline::QkGroup => "QK group size must be 8 or 16",
            Decline::OutOfRegisters => "no more vector register in the requested region",
            Decline::PrefillRows => "out-of-domain prefill block (1..=8, stride ≥ head_dim)",
            Decline::NoVnni => "VPDPBUSD indisponible : ni AVX512-VNNI, ni AVX-VNNI 256 bits",
        };
        f.write_str(s)
    }
}

impl GemmShape {
    /// Shapes that the emitter knows how to specialize. See [`Decline`].
    ///
    /// Odd `K` and `N % 16 ≠ 0` are served: the emitter peels the last pair
    /// of `K` on a **word** load and
    /// masks the last group of 16 columns (`MicroKernelSpec::k_odd` and
    /// `n_tail`). Storage, however, never needed anything:
    /// `PackedBf16R16Weight::pack_from_row_major` already zero-pads to
    /// `k_even` and `ceil(N/16)` blocks.
    ///
    /// The two variants [`Decline::OddK`] and [`Decline::PartialRowBlock`]
    /// remain: they serve the **other** emitters (AVX2, elem, conv,
    /// recurrence, gated norm), which do not have this handling.
    pub fn check(&self) -> Result<(), Decline> {
        if self.k == 0 || self.n == 0 {
            return Err(Decline::Empty);
        }
        Ok(())
    }

    /// Pairs of `K` consumed by the kernel, **including padding**: it is
    /// `k_even / 2`, the stride of the R16 block (`b_block_bytes`).
    ///
    /// For odd `K`, the last pair has only one valid element in `A`; the
    /// zero in `B` cancels the other, and the emitter peels it to avoid
    /// over-reading (`MicroKernelSpec::k_odd`).
    pub const fn k_pairs(&self) -> usize {
        self.k.div_ceil(2)
    }

    /// Blocks of 16 lines of `W`, **padding included**: `ceil(N / 16)`.
    pub const fn n_blocks(&self) -> usize {
        self.n.div_ceil(ROWS_INTERLEAVED)
    }
}

/// Does the tile fit the registers of the target's GEMM micro-kernel, with
/// `nr` a multiple of 16? The budget is the target's
/// (`target::x86::tile_fits`: the 32 zmm; `target::aarch64::tile_fits`: the
/// 32 NEON registers).
pub fn tile_fits(tile: TileShape) -> Result<(), Decline> {
    crate::jit::target::tile_fits(tile)
}

/// The largest legal width ≤ `max_cols` for a height `mr`, in multiples of
/// 16, on the target. `None` if even 16 columns don't fit.
pub fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
    crate::jit::target::widest_tile(mr, max_cols)
}

#[cfg(target_arch = "x86_64")]
pub use crate::jit::target::x86::zmm_needed;
