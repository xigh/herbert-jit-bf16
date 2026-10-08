//! The eight constants of the emitted `exp` polynomial, **one copy**.
//!
//! Every emitted exponential — the AVX-512 and AVX2 elementary kernels, the
//! AVX-512 and AVX2 SDPA, and every AArch64 kernel that takes an `exp` —
//! reads these. They used to be four textual copies, each annotated
//! "copied, never re-derived", and nothing checked that the four still
//! agreed. A single definition needs no such check.
//!
//! The values are the literals as they were written; `EXP_LOG2E` and
//! `EXP_C1` are polynomial coefficients whose bits must stay exactly these,
//! which is why clippy's `approx_constant` is silenced on them rather than
//! obeyed.

pub(crate) const EXP_NEG87: f32 = -87.33654;
#[allow(clippy::approx_constant)]
pub(crate) const EXP_LOG2E: f32 = 1.442_695;
pub(crate) const EXP_C4: f32 = 0.009_672_48;
pub(crate) const EXP_C3: f32 = 0.055_504_11;
pub(crate) const EXP_C2: f32 = 0.240_226_5;
#[allow(clippy::approx_constant)]
pub(crate) const EXP_C1: f32 = 0.693_147_2;
pub(crate) const EXP_C0: f32 = 1.0;
pub(crate) const EXP_ONE_BITS: u32 = 0x3F80_0000;
