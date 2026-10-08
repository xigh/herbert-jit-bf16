//! The gated RMS norm of the GatedDeltaNet mixer, whatever the target: `dim`
//! immediate (the `head_v_dim` width of a row), `n` rows in a loop.
//!
//! Per row `(position, V head)` of width `dim`:
//!
//! ```text
//! inv_rms = 1 / sqrt(Σ_d x[d]² / dim + eps)            computed ONCE per row
//! y[d]    = ((x[d] · inv_rms) · w[d]) · silu(z[d])      silu(z) = z / (1 + exp(−z))
//! ```
//!
//! ABI: `fn(out: *mut f32, z: *const f32, w: *const u16, out_bf16: *mut u16,
//! n_rows: usize, eps: f32)`. `out`, `z` and `out_bf16` advance by `dim` per
//! row; `w` is fixed. `n_rows ≥ 1`. The emitters: `target/x86/emit_gated_norm.rs`
//! (AVX-512), `target/aarch64/delta.rs`.

use crate::jit::ir::Decline;

/// Shape of a gated norm kernel: the width of a line, **immediate**.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct GatedNormSpec {
    /// `head_v_dim` — 128 across the entire Qwen3.5 family.
    pub dim: usize,
}

/// Lanes per block: one f32 ZMM.
pub const LANES: usize = 16;

/// Maximum emitted width: `w` expanded fits in registers (8 ZMM at 128, with
/// the 8 exp constants, the bound, 4 chains, and temporaries: 29 of
/// the 32). Beyond that, the caller keeps the scalar per line — parallel too.
pub const DIM_MAX: usize = 128;

/// `fn(out, z, w, out_bf16, n_lines, eps)`.
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `target::generated_fn!`.
pub type GatedNormFn = crate::jit::target::generated_fn!(*mut f32, *const f32, *const u16, *mut u16, usize, f32);

impl GatedNormSpec {
    /// Emissible shapes: `dim` multiple of 16, `0 < dim ≤` [`DIM_MAX`].
    pub fn check(&self) -> Result<(), Decline> {
        if self.dim == 0 {
            return Err(Decline::Empty);
        }
        if !self.dim.is_multiple_of(LANES) {
            return Err(Decline::PartialRowBlock);
        }
        if self.dim > DIM_MAX {
            return Err(Decline::TileTooWide);
        }
        Ok(())
    }
}


/// Generates the kernel of `spec` for the target of this build.
pub fn emit_gated_norm(spec: &GatedNormSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::target::emit_gated_norm(spec)
}
