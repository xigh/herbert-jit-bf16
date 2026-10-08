//! The code generator: the emitters, the generated program, the scheduler.
//!
//! What enters: the model shape and the packed weights the engine borrows.
//! What comes out: one generated program per regime (prefill, decode) that
//! every participant of the pool runs from end to end.
//!
//! Read in the order a call goes through:
//! - `ir` — the shapes the emitters accept, the tiles, the named refusals;
//! - `emit`, `emit_elem`, `emit_sdpa`, `emit_conv`, `emit_recurrence`,
//!   `emit_gated_norm` — what each kernel family is asked for, whatever the
//!   target: the specifications, their ABIs, the shape checks;
//! - `target` — the emitters of the build's architecture: `target/x86`
//!   (`iced-x86`: the AVX-512 and the AVX2 arm) or `target/aarch64` (the `a64`
//!   crate: NEON with the BF16 extension), each with the generated nodes of
//!   the execution plan;
//! - `compile`, `gemm`, `elem`, `sdpa`, `conv`, `recurrence`, `gated_norm` —
//!   one driver per kernel family: compile a shape once, keep it in a cache;
//! - `schedule` — the buffers of a layer and its lowering into nodes;
//! - `execution` — the plan of nodes and the whole-call program that runs it;
//! - `model` — the runner: the KV cache, the DeltaNet states, the RoPE
//!   tables, the two mounted regimes, the snapshot;
//! - `listing` — the disassembly of a mounted program, for reading.
//!
//! Nothing here knows the checkpoint: the weights arrive as raw slices in
//! the R16 layout, and the knobs of `knobs.rs` are the only settings.

#![deny(missing_docs)]

/// Alignment of each generated kernel within an executable mapping.
pub(crate) const KERNEL_ALIGN: usize = 64;

pub mod execution;
pub mod compile;
pub mod conv;
pub mod elem;
pub mod gemm;
pub mod ir;
pub mod knobs;
pub mod listing;
pub mod model;
pub mod gated_norm;
pub mod options;
pub mod recurrence;
pub mod schedule;
pub mod target;
// What every target emits: the specifications and their ABIs.
pub mod emit;
pub mod emit_conv;
pub mod emit_elem;
pub mod emit_gated_norm;
pub mod emit_recurrence;
pub mod emit_sdpa;
pub(crate) mod exp_constants;
// The x86-only emitters, reachable from the root.
#[cfg(target_arch = "x86_64")]
pub use target::x86::{emit_elem_avx2, emit_gemm_avx2, emit_sdpa_avx2, regs};
pub mod sdpa;

pub use compile::{compile, CompileError, CompiledGemm};
pub use emit::{AFormat, KernelFn, KernelOptions, MicroKernelSpec};
pub use emit_elem::{emit_elem, ElemOp, ElemSpec};
pub use emit_sdpa::{
    emit_sdpa, emit_sdpa_prefill, Pass, SdpaKernelSpec, SdpaPrefillSpec, SDPA_PREFILL_MAX_ROWS,
};
pub use gemm::is_supported;

/// The ISA that the engine uses on **this** machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IsaEngine {
    /// AVX-512 + BF16 : tiles zmm, `vdpbf16ps`, `vcvtneps2bf16`.
    Avx512,
    /// AVX2 + FMA: ymm tiles, the three emitters carried.
    Avx2,
    /// AArch64 NEON with the BF16 extension: `bfdot` tiles, `bfcvtn`.
    NeonBf16,
    /// RISC-V RVV 1.0 with Zvfbfmin and Zvfbfwma: `vfwmaccbf16`, `vfncvtbf16`.
    Rvv,
    /// Below the floor — nothing to emit, the caller falls back.
    BelowFloor,
}

/// The process ISA, **resolved only once**. The detection itself is memoized
/// (`crate::core::isa`), but nothing forces going through it again, and above
/// all: the question "what ISA this process uses" has **only one** answer, it
/// does not need to be recomposed from three predicates at each site.
///
/// Everything else — [`is_supported`], [`engine_supported`] and
/// [`elem::select_isa`] — reads this result.
pub fn engine_isa() -> IsaEngine {
    static ISA: std::sync::OnceLock<IsaEngine> = std::sync::OnceLock::new();
    *ISA.get_or_init(|| {
        let f = crate::core::isa::feature;
        if cfg!(target_arch = "aarch64") {
            return if f("neon") && f("bf16") && f("lse") {
                IsaEngine::NeonBf16
            } else {
                IsaEngine::BelowFloor
            };
        }
        if cfg!(target_arch = "riscv64") {
            return if crate::core::isa::below_floor().is_empty() { IsaEngine::Rvv } else { IsaEngine::BelowFloor };
        }
        if f("avx512f") && f("avx512bw") && f("avx512bf16") {
            return IsaEngine::Avx512;
        }
        if f("avx2") && f("fma") {
            return IsaEngine::Avx2;
        }
        IsaEngine::BelowFloor
    })
}

/// **Can the generated engine run on this machine?**
///
/// Distinct from [`is_supported`], which answers "does this machine have
/// AVX-512": the elementwise kernels, the bf16 GEMM and the decode SDPA also
/// have an AVX2 arm, so the setup question is whether the emitters can emit
/// here at all. `is_supported` keeps its narrow meaning. A predicate that
/// changes meaning under its callers is a trap; two names are better.
///
/// This is the predicate that `Runner::ascend` and
/// `FullJitBackend::load` consult.
pub fn engine_supported() -> bool {
    engine_isa() != IsaEngine::BelowFloor
}
pub use ir::{Decline, ExpStrategy, GemmShape, SdpaForm, SdpaShape, TileShape};
pub use model::{
    DeltaBudget, Instantaneous, ModelBudget, ModelWeights, PlanKv, Progression,
    RopeTableSet, Runner,
};
pub use schedule::{
    Activation,
    Budget, Buffers, DeltaShape, DeltaWeights, LayerWeights, Mixer, ModelShape, Scheduler,
    DEFAULT_MAX_BATCH, prefill_pass,
};
pub use sdpa::{head_kernel, prefill_kernel, sdpa_rows, CompiledSdpa, CompiledSdpaPrefill};

/// Default tile of the production path.
///
/// **8×48, decided end to end**: median of 5 interleaved runs, the best of
/// the four tiles on both machines —
/// 449.7 tok/s (+17.9 % over v8) vs 443.2 for 6×64 on the first, 274.4
/// (+8.4 %) vs 273.4 on the second.
///
/// This shape was not chosen by the micro-bench: nine tiles out of ten hit the
/// FP ceiling (1.96 `vdpbf16ps`/cycle/core), so it does not discriminate. What
/// distinguishes them is the memory hierarchy at the driver level — `A` traffic at
/// `(N/nr)·M·K·2` on one side, L2 footprint of the panel on the other
/// ([`panel_working_set`]). 8×48 is the compromise between the two.
pub const DEFAULT_TILE: TileShape = TileShape { mr: 8, nr: 48 };

/// L2 budget allocated to sweeping a column panel.
///
/// Both machines in scope are Zen 5 with **1 MiB of L2
/// per core**. We reserve three quarters of it: the rest absorbs the `C` tile,
/// the `A` lines currently being streamed, and traffic from other threads.
/// This is a step, not a tuned constant: a measurement per machine would
/// replace it.
///
/// Both machines are in **SMT**: two sibling threads share a
/// single 1 MiB L2, and the driver gives them **different** panels
/// (`parallel_for` splits the grid into contiguous ranges). The actual budget per
/// thread could therefore be 512 KiB, not 768: `Knobs::l2_budget` settles it
/// by measurement.
pub const L2_PANEL_BUDGET: usize = 768 * 1024;

/// The budget in force: [`crate::jit::knobs::Knobs::l2_budget`].
fn l2_budget() -> usize {
    knobs::knobs().l2_budget
}

/// L2 footprint of a column panel scan by the driver.
///
/// The order is N-external / M-internal: for a given
/// panel, the driver descends all row tiles. Thus remain simultaneously hot the
/// panel of `B` (`nr·K·2`), the tile of `A` (`mr·K·2`) and the tile of `C`
/// (`mr·nr·4`).
pub fn panel_working_set(shape: GemmShape, tile: TileShape) -> usize {
    tile.nr * shape.k * 2 + tile.mr * shape.k * 2 + tile.mr * tile.nr * 4
}

/// Candidates for the `auto` mode, from least to most traffic `A`-hungry.
///
/// The `A` traffic is `(N/nr)·M·K·2` : it decreases as `nr` increases. We therefore
/// take the **first** one that fits in [`L2_PANEL_BUDGET`], i.e. the widest
/// one that the shape supports.
const AUTO_CANDIDATES: &[TileShape] = &[
    TileShape { mr: 6, nr: 64 },
    TileShape { mr: 8, nr: 48 },
    TileShape { mr: 10, nr: 32 },
];

/// **Tile** chosen **by shape**: the widest whose panel fits in L2.
///
/// The compiler already works by `(K, N)` — nothing forces a global tile
/// globally. On Qwen3-1.7B this yields `6×64` everywhere except `down_proj`
/// (`K = 6144`), whose 6×64 panel weighs 842 KiB, i.e., 82% of L2.
pub fn tile_for_shape(shape: GemmShape) -> TileShape {
    tile_for_shape_within(shape, l2_budget())
}

/// [`tile_for_shape`] with explicit budget — the testable entry point.
pub fn tile_for_shape_within(shape: GemmShape, budget: usize) -> TileShape {
    AUTO_CANDIDATES
        .iter()
        .copied()
        .find(|&t| panel_working_set(shape, t) <= budget)
        .unwrap_or(*AUTO_CANDIDATES.last().expect("AUTO_CANDIDATES non empty"))
}

/// How the production path chooses its GEMM tile ([`crate::jit::knobs::Knobs::tile`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TilePolicy {
    /// The **default**, and it follows the ISA.
    ///
    /// Distinct from [`TilePolicy::Fixed`]: a pinned tile is an ablation point
    /// and must be able to produce a refusal when it does not fit the machine;
    /// the default has no reason to refuse and returns a tile this machine can
    /// emit. Without the distinction a machine without AVX-512 received
    /// [`DEFAULT_TILE`] (8×48, calibrated for thirty-two zmm, 59 registers in
    /// ymm) and every GEMM declined.
    Default_,
    /// One tile for all shapes, **hand-placed**.
    Fixed(TileShape),
    /// One tile per shape, cf. [`tile_for_shape`].
    PerShape,
}

/// The tile policy in force: [`crate::jit::knobs::Knobs::tile`].
#[cfg(target_arch = "x86_64")]
fn tile_policy() -> TilePolicy {
    knobs::knobs().tile
}

/// The production path tile **in AVX2**.
///
/// The AVX-512 production tiles (`6×64`, `8×48`) are out of reach here: the
/// AVX2 kernel consumes `mr·nr/8 + nr/8 + 3` registers and has only **sixteen**,
/// whereas the AVX-512 kernel has thirty-two and lanes twice as wide.
/// A `6×64` would require **59** registers.
///
/// `5×16` is the highest that the budget allows at `nr = 16`
/// (`5·2 + 2 + 3 = 15`), and this is what maximizes the reuse of `B`:
/// each block read serves five rows of `A`. At `nr = 32` we fall back to `mr ≤ 2`,
/// hence to a lower reuse.
///
/// No measurement underpins this choice: it is the arithmetic maximum of the
/// budget, not a measured optimum, and the first setting to sweep on AVX2.
pub const AVX2_TILE: TileShape = TileShape { mr: 5, nr: 16 };

/// Production path tile for a given shape.
///
/// **The tile depends on the ISA**, and not just on the shape: on a
/// machine without AVX-512, tiles calibrated for thirty-two zmm registers
/// would all be refused by `emit_gemm_avx2::check_avx2`, and the GEMM
/// would decline for a reason that has nothing to do with the shape. A tile
/// pinned by `TilePolicy::Fixed` keeps priority: it is an ablation point, it
/// must be able to produce a refusal.
pub fn production_tile(shape: GemmShape) -> TileShape {
    target::gemm_tile(shape)
}

/// The x86 tile policy behind [`production_tile`].
#[cfg(target_arch = "x86_64")]
pub(crate) fn x86_production_tile(shape: GemmShape) -> TileShape {
    match tile_policy() {
        // Hand-placed: rendered as-is, even if it doesn't fit
        // — that's what makes it an ablation point.
        TilePolicy::Fixed(t) => t,
        // Resolved once: `production_tile` is called at every projection of
        // every token, and the question "does this machine have AVX-512" does
        // not need to be answered there.
        TilePolicy::Default_ => {
            static T: std::sync::OnceLock<TileShape> = std::sync::OnceLock::new();
            *T.get_or_init(|| {
                if gemm::is_supported() {
                    DEFAULT_TILE
                } else {
                    AVX2_TILE
                }
            })
        }
        TilePolicy::PerShape => {
            if gemm::is_supported() {
                tile_for_shape(shape)
            } else {
                AVX2_TILE
            }
        }
    }
}
