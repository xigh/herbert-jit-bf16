//! The knobs of the emitters and of the lowering, installed once per process.
//!
//! Every choice a campaign used to make through an environment variable is
//! a field here, with the value that is the measured default today. `main`
//! builds the struct (through `Tuning`), `Engine::load` installs it before
//! any kernel is compiled, and the emitters read it with [`knobs`]. Nothing in
//! the engine reads `std::env`.
//!
//! A knob is not a verdict: each one was kept because it was measured, and
//! the doc of each field says what it changes.

use crate::jit::emit::{NeonMac, NeonOptions, RvvOptions};
use crate::jit::ir::{ExpStrategy, SdpaForm};
#[cfg(target_arch = "aarch64")]
pub use crate::jit::target::aarch64::execution::delta::L2Mode;
#[cfg(target_arch = "riscv64")]
pub use crate::jit::target::riscv64::execution::delta::L2Mode;
#[cfg(target_arch = "x86_64")]
pub use crate::jit::target::x86::execution::delta::L2Mode;
use crate::jit::TilePolicy;
use std::sync::OnceLock;

/// The knobs of the emitters and of the lowering, one field per decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Knobs {
    // ── the pass and the GEMM ──────────────────────────────────────────
    /// Rows of one prefill pass. 256 keeps the activations of a pass in L2
    /// (6 MiB at 256 on the 4B); 2048 was −0.7 % of prefill for a larger footprint.
    pub prefill_pass: usize,
    /// How the GEMM tile is chosen: the ISA default (8×48 zmm, 5×16 ymm), one
    /// tile pinned by hand, or the widest tile whose panel fits `l2_budget`.
    pub tile: TilePolicy,
    /// L2 bytes a column panel may occupy under `TilePolicy::PerShape`: three
    /// quarters of the 1 MiB Zen 5 L2.
    pub l2_budget: usize,
    /// Blocks of several rows (the prefill) run their projections on the
    /// target's matrix engine when it has one (AArch64 SME2 `bfmopa`), the
    /// activations packed once per input. No effect on x86-64.
    pub gemm_matrix: bool,
    /// Blocks of several rows run their projections on Apple's AMX
    /// (`matfp` bf16 → f32, `target/aarch64/gemm_amx.rs`), the activations
    /// packed by single `K` for thirty-two rows: the order and the arithmetic
    /// of the `bfmlal` vector kernel, hence its bits. Takes precedence over
    /// SME2. Any Apple M (M1 to M5); ignored elsewhere. Off by default.
    pub gemm_amx: bool,
    /// The variants of the AArch64 vector GEMM (the decode, and the prefill
    /// without the matrix engine): the width of the decode tile, a prefetch,
    /// `bfmlal` in place of `bfdot`, two accumulator sets, `ld1` loads, the
    /// tail of the matrix path. Measured: at decode the kernel as
    /// first written already streams at the memory ceiling on the M5 and the
    /// M3 Ultra, no variant gains there; without a matrix engine the prefill
    /// is this kernel, compute-bound, and `bfmlal` (four per cycle against one
    /// `bfdot`) takes the 1 666-token prompt of the M3 Ultra from 141 to 246
    /// tokens/s — the default on a machine without SME2, with the bits of
    /// two fused multiply-adds per pair instead of the measured `bfdot`. On
    /// the matrix path, residue rows up to six run on the `bfdot` tail inside
    /// the SME task — by `bfdot` for one residue row (a 17-token prompt goes
    /// from 121 to 137 tokens/s on the M5, the same bits as the tile; `bfmlal`
    /// gives 123), by `bfmlal` from two rows (a 20-token prompt: 167 padded,
    /// 155 by `bfdot`, 181 by `bfmlal`).
    pub neon: NeonOptions,
    /// The variants of the RISC-V vector kernels (the GEMM reads them through
    /// its `KernelOptions`, the other RISC-V emitters from here).
    pub rvv: RvvOptions,
    // ── the attention (SDPA) ───────────────────────────────────────────
    /// One kernel for QK, softmax and SV (`Chained`), three calls (`Split`), or
    /// the scores kept in registers (`Fused`).
    pub sdpa_form: SdpaForm,
    /// The exponential of the softmax: emitted inline.
    pub sdpa_exp: ExpStrategy,
    /// Queries per prefill block, 0..=8; `0` = the decode kernel row by row.
    pub sdpa_rows: usize,
    /// 16-dimension blocks the SV pass accumulates together; `0` = the widest
    /// the registers allow.
    pub sdpa_dim_blocks: usize,
    /// Prefetch distance in K/V rows, for both kernels; `None` = the emitter's
    /// defaults (0 in prefill, 4 in decode).
    pub sdpa_prefetch: Option<usize>,
    /// Positions whose QK scores share one accumulator chain: `None` = 8, the
    /// measured default, withdrawn for the `Fused` form which refuses it. A
    /// value set here is imposed and its refusal propagates.
    pub sdpa_qk_group: Option<u8>,
    /// Queries sharing one load of `K` in the prefill group loop: `None` = 2,
    /// withdrawn when the shape has no register for it. A value set here is
    /// imposed and its refusal propagates.
    pub sdpa_qk_pair: Option<u8>,
    /// Spill every SV accumulator to L1 (the yardstick of "what an accumulator
    /// costs"): never in production.
    pub sdpa_sv_spill_all: bool,
    /// AArch64: the prefill attention by blocks of sixteen queries of one
    /// head, as two GEMMs on the micro-kernels of the projections — the
    /// scores `Q·Kᵀ` in f32, the softmax of the decode kernel row by row
    /// writing `P` in bf16, then `P·V` — on the matrix engine when the
    /// machine has one (`bfmopa`), on the vector kernel otherwise. The KV
    /// append then also writes, per KV head, `K` and `Vᵀ` in the R16 layout
    /// (as much memory again as the cache). Off: the decode kernel row by row.
    /// The bits change (`P` is rounded to bf16, the sums run in the order of
    /// the GEMM); both paths are held to the bit against their
    /// transcription. Measured (interleaved, three repetitions), prefill in
    /// tokens/s: M5 (SME), 1 666 tokens 222.7 →
    /// 239.8, 6 609 tokens 172.1 → 228.4; M3 Ultra (vector `bfmlal`), 244.7 →
    /// 247.8 and 222.4 → 240.6; the same ids on both prompts. A 17-token
    /// prompt stays within the noise (M5 133.9 → 137.2 and 140.9 → 136.6,
    /// Ultra 145 → 133 on runs spread over 110–161) and its answer changes
    /// from the 9th to the 51st token. The default on AArch64.
    pub attn_block: bool,
    // ── the GatedDeltaNet mixer ────────────────────────────────────────
    /// Emit the recurrence with fused multiply-adds (measured at zero gain).
    pub deltanet_fma: bool,
    /// AArch64: the recurrence stores the state once per position (the
    /// first pass no longer writes the decayed state, the second recomputes
    /// it): the same bits, a quarter of the memory operations fewer.
    pub deltanet_store_once: bool,
    /// AArch64: the recurrence loads and stores its state by pairs of
    /// registers; the same bits.
    pub deltanet_pairs: bool,
    /// AArch64: `q · scale` computed once per position instead of once per
    /// slice of the value head; the same bits.
    pub deltanet_q_hoist: bool,
    /// AArch64: lanes per loop of the recurrence's first pass, 32 or 64; the
    /// same bits.
    pub deltanet_pass1_lanes: u8,
    /// AArch64: each value head cut into that many slices of independent
    /// columns, one task each (32 heads × split tasks); 1 = one task per head.
    /// The same bits.
    pub deltanet_vd_split: usize,
    /// AArch64 with SME2: the prefill recurrence by chunks of sixteen
    /// positions in the WY form (`deltanet_chunk.rs`), the rest of a pass on
    /// the sequential kernel; the bits change (other sums, other order).
    /// Refused without a matrix engine, where the sequential form stays.
    /// Measured on the M5 (interleaved, three repetitions),
    /// prefill in tokens/s: 1 666 tokens 241.7 → 279.3, 6 609 tokens 230.9 →
    /// 262.1, 17 tokens 139.1 → 142.7; the same ids on the long prompts, a
    /// 63-token answer diverging at its 57th. The default with SME2.
    pub deltanet_chunked: bool,
    /// The recurrence over V heads on the whole team, or on the caller alone.
    pub deltanet_recurrence_team: bool,
    /// The causal conv over channel ranges on the whole team, or on the caller.
    pub deltanet_conv_team: bool,
    /// The gated norm over row ranges on the whole team, or on the caller.
    pub deltanet_norm_team: bool,
    /// The L2 normalisation of Q and K: `None` = the best body of the ISA.
    pub deltanet_l2: Option<L2Mode>,
    // ── the team of the small nodes ────────────────────────────────────
    /// At decode (one row), the projections, the attention and the DeltaNet
    /// stages run on the whole team; `false` = one participant per node.
    pub decode_whole_team: bool,
    /// RMS norms and residual additions on the whole team, one task per row
    /// (measured: no gain at decode), instead of participant zero.
    pub elem_team: bool,
    /// The chain around the attention (head norms, RoPE, KV append, output gate)
    /// on the whole team instead of participant zero (measured: no gain).
    pub attn_chain_team: bool,
    /// Rows from which an elementwise kernel loops instead of unrolling its
    /// body: 4096 keeps the SwiGLU 9216 body out of the 94 KiB unrolled form.
    pub elem_loop_min: usize,
    /// Tasks of a node handed out dynamically, in chunks of
    /// `ceil(tasks / (participants · grain))` taken with one atomic add, so
    /// that a slower core (the E cores of an Apple M5) takes fewer; 0 = the
    /// static split of the plan. AArch64 only; the results do not change, the
    /// tasks of a node being independent. Measured on the M5, 6 workers,
    /// decode: 13.37 tokens/s static, 13.84 at 8, 13.55 at 16; prefill of 20
    /// rows 72 → 105 tokens/s.
    pub task_grain: usize,
    /// Guided chunks: the size of a chunk is recomputed at every grab as
    /// `max(task_chunk_min, remaining / (participants · grain))`, so the last
    /// chunks of a node shrink to one task and a slow core cannot hold the
    /// barrier with a large one. AArch64 only; the results do not change.
    /// Measured: M5, decode 15.09 → 15.16 tokens/s and the
    /// 1 666-token prefill 220 → 238; M3 Ultra at 16 or 20 threads, decode
    /// 27.0 → 22–26.7 and the long prefill 141 → 115–146 — the grabs of
    /// twenty threads across two dies cost more than the tail they trim.
    /// Measured again on the M5 at the later defaults (interleaved, three
    /// repetitions), prefill in tokens/s: 1 666 tokens
    /// 277.1 → 296.0, 6 609 tokens 262.1 → 277.9, 17 tokens 139.9 → 152.5,
    /// the decode within the noise. The default when the default pool mixes
    /// performance and efficiency cores (`cpu_detection::
    /// default_pool_includes_efficiency_cores`: the M5, not the M3 Ultra,
    /// whose pool is its performance cores).
    pub task_guided: bool,
    /// The smallest chunk a grab takes (static or guided).
    pub task_chunk_min: usize,
    /// The largest chunk a grab takes; 0 = no ceiling.
    pub task_chunk_max: usize,
    /// The ticket counter on its own 128-byte line instead of the line of
    /// the barrier counters, which the waiting participants poll. Measured on
    /// the M5 at decode: 15.09 → 15.21 tokens/s, the same bits. The default
    /// on AArch64.
    pub ticket_line: bool,
    /// The projections of one input in one node (gate and up; q, k and v; at
    /// decode a, b, qkv and z of DeltaNet, written straight into the a|b row):
    /// one task space, one barrier, one ramp per group. AArch64; the results
    /// do not change. Measured at decode: M5 15.09 → 15.30 tokens/s, M3 Ultra
    /// 26.4 → 27.2. The default on AArch64; ignored on x86-64,
    /// whose target has no fused projection node.
    pub fuse_projections: bool,
    /// The argmax of the logits in that many slices on the team, then a
    /// serial merge, instead of one serial scan; 0 = the serial scan.
    /// AArch64; the result does not change. Measured on the M5 at decode with
    /// 16 slices: 15.09 → 15.23 tokens/s. The default on AArch64.
    pub argmax_tasks: usize,
}

impl Default for Knobs {
    fn default() -> Self {
        // The AArch64 nodes and, without a matrix engine, the `bfmlal` vector kernel.
        let arm = cfg!(target_arch = "aarch64");
        #[cfg(target_arch = "aarch64")]
        let matrix = crate::jit::target::aarch64::gemm_sme::available();
        #[cfg(not(target_arch = "aarch64"))]
        let matrix = false;
        Knobs {
            prefill_pass: crate::jit::schedule::PREFILL_PASS_DEFAULT,
            tile: TilePolicy::Default_,
            l2_budget: crate::jit::L2_PANEL_BUDGET,
            gemm_matrix: true,
            gemm_amx: false,
            sdpa_form: SdpaForm::default(),
            sdpa_exp: ExpStrategy::default(),
            sdpa_rows: crate::jit::emit_sdpa::SDPA_PREFILL_MAX_ROWS,
            sdpa_dim_blocks: 0,
            sdpa_prefetch: None,
            sdpa_qk_group: None,
            sdpa_qk_pair: None,
            sdpa_sv_spill_all: false,
            attn_block: arm,
            deltanet_fma: false,
            deltanet_store_once: false,
            deltanet_pairs: false,
            deltanet_q_hoist: false,
            deltanet_pass1_lanes: 32,
            deltanet_vd_split: 1,
            deltanet_chunked: matrix,
            deltanet_recurrence_team: true,
            deltanet_conv_team: true,
            deltanet_norm_team: true,
            deltanet_l2: None,
            decode_whole_team: true,
            elem_team: false,
            attn_chain_team: false,
            elem_loop_min: crate::jit::emit_elem::LOOP_MIN_DIM,
            task_grain: 8,
            task_guided: arm && crate::core::cpu_detection::default_pool_includes_efficiency_cores(),
            task_chunk_min: 1,
            task_chunk_max: 0,
            ticket_line: arm,
            fuse_projections: arm,
            argmax_tasks: if arm { 16 } else { 0 },
            neon: NeonOptions {
                mac: if arm && !matrix { NeonMac::BfMlal } else { NeonMac::BfDot },
                tail_rows: 6,
                tail_mac: NeonMac::BfMlal,
                tail_bfdot_max: 1,
                ..NeonOptions::default()
            },
            rvv: RvvOptions::default(),
        }
    }
}

static KNOBS: OnceLock<Knobs> = OnceLock::new();

/// Installs the knobs for the life of the process. A second install with the
/// same values is accepted; different values are refused, because kernels
/// compiled under the first set are cached.
pub fn install(knobs: Knobs) -> Result<(), String> {
    let installed = KNOBS.get_or_init(|| knobs);
    if *installed == knobs {
        Ok(())
    } else {
        Err("the kernel knobs were already installed with other values; one Engine per process".to_string())
    }
}

/// The installed knobs, or the defaults when nothing was installed.
pub fn knobs() -> &'static Knobs {
    KNOBS.get_or_init(Knobs::default)
}
