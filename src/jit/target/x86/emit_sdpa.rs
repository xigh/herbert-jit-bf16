//! The decode attention (SDPA) emitter, AVX-512: one query head against the
//! cached keys and values.
//!
//! ```text
//! scores[j] = (Σ_d q[d] · bf16→f32(K[j][d])) · scale     j < n
//! scores    = softmax(scores)            (with or without a sink)
//! out[d]    = Σ_j scores[j] · bf16→f32(V[j][d])
//! ```
//!
//! `K` and `V` arrive by head, in contiguous rows of `head_dim` bf16: the
//! stride between two rows is `head_dim·2` bytes. The shape
//! `(head_dim, sink)` is frozen at load; the kernel is parameterised on the
//! form ([`SdpaForm`]: one blob, three blobs, or the scores kept in
//! registers) and the QK group, both knobs of `knobs.rs`.
//!
//! ## The score buffer contract: padded to a multiple of 16
//!
//! The QK pass writes `n` scores then pads up to the next multiple of 16 with
//! `-inf`. This makes the three softmax sweeps mask-free: `-inf` is neutral
//! for the maximum, and `vmaxps` brings it back to −87.34 whose exponential
//! is 1.2e−38. The SV pass stops exactly at `n`: the rows of `V` beyond the
//! context are never read. The caller provides `scores` sized to
//! `n.next_multiple_of(16)`; the driver (`sdpa.rs`) checks it.
//!
//! ## ABI — SysV, no context
//!
//! ```text
//! qk       (q: *const u16, k: *const u16, scores: *mut f32, n: u64, scale: f32)
//! softmax  (scores: *mut f32, n: u64, sink: f32)
//! sv       (scores: *const f32, v: *const u16, out: *mut u16, n: u64)
//! head     (q, k, v, scores, out, n, scale: f32, sink: f32)   // Chained, Fused
//! ```
//!
//! Common prologue: `rbp` saved, `rsp` aligned to 64, a scratch area for the
//! stashed arguments and the spilled SV accumulators when `head_dim`
//! exceeds the registers. The prefill block kernel (`x86/prefill.rs`) applies
//! the same passes to `rows` queries at once, with the QK grouped over 16
//! positions.

pub(crate) use crate::jit::emit_sdpa::{
    Pass, SdpaKernelSpec, SdpaPrefillSpec, SDPA_PREFILL_MAX_ROWS,
};

pub use x86::{emit_sdpa, emit_sdpa_prefill};

mod x86;
