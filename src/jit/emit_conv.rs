//! The causal depthwise conv + SiLU of the GatedDeltaNet mixer, whatever the
//! target: one position, `n` blocks of 16 channels, `K = 4` taps as an
//! immediate.
//!
//! Per channel `ch`, with the state `state[ch·K + j]`, for each position:
//!
//! ```text
//! state[ch·K + j] ← state[ch·K + j + 1]   (j < K−1)
//! state[ch·K + K−1] ← x[ch]
//! y[ch] ← silu(Σ_j state[ch·K + j] · w[ch·K + j])
//! ```
//!
//! ABI: `fn(data: *mut f32, w: *const u16, state: *mut f32, n_blocks: usize)`.
//! The three pointers are already offset to the first channel of the range;
//! the kernel advances by 64 / 128 / 256 bytes per block. `n_blocks ≥ 1`.
//! The emitters: `target/x86/emit_conv.rs` (AVX-512), `target/aarch64/delta.rs`.

use crate::jit::ir::Decline;

/// Shape of a conv kernel: the causal kernel length, **immediate**.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ConvSpec {
    /// `conv_kernel` — 4 across the entire Qwen3.5 family. Only emitted value.
    pub k: usize,
}

/// Channels per block: one ZMM of f32.
pub const CHANNELS_PER_BLOCK: usize = 16;

/// `fn(data, w, state, n_blocks)`.
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `target::generated_fn!`.
pub type ConvFn = crate::jit::target::generated_fn!(*mut f32, *const u16, *mut f32, usize);

impl ConvSpec {
    /// Emittable shapes: `K = 4` only — this is what the grouping of
    /// four lanes per channel assumes. Any other `K` keeps the scalar.
    pub fn check(&self) -> Result<(), Decline> {
        if self.k == 0 {
            return Err(Decline::Empty);
        }
        if self.k != 4 {
            return Err(Decline::PartialRowBlock);
        }
        Ok(())
    }
}


/// Generates the kernel of `spec` for the target of this build.
pub fn emit_conv(spec: &ConvSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::target::emit_conv(spec)
}
