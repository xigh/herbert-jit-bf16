//! The driver of the elementwise kernels: compilation and cache.
//!
//! One blob per `(operation, width)`, no plan: an RMSNorm over 2560 values or
//! a SwiGLU over 9216 costs a few microseconds and lives in L1. The ISA of the
//! arm — zmm on AVX-512, ymm on AVX2 — is chosen once per process from what
//! the machine has, and the two arms agree to one bf16 ulp, not to the bit
//! (the gap is measured).

use crate::jit::emit_elem::{emit_elem, ElemSpec};

/// Which ISA the elementary generator uses on **this** machine.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ElemIsa {
    /// AVX-512(+BF16) : `vdpbf16ps`, `vcvtneps2bf16`, tiles zmm.
    Zmm,
    /// AVX2 + FMA : tiles ymm, `vdpbf16ps` and `vcvtneps2bf16` **emulated**.
    Ymm,
    /// AArch64 NEON with BF16: widened by `shll`, narrowed by `bfcvtn`.
    Neon,
    /// RISC-V RVV: widened by `vfwcvtbf16`, narrowed by `vfncvtbf16`.
    Rvv,
}

/// Chooses the emission ISA **based on runtime detection**, and nothing else.
///
/// No setting enters here: the JIT emits for the machine it runs on. AVX-512
/// first, because its tiles have 16 lanes instead of 8 and `vcvtneps2bf16`
/// costs one instruction instead of five.
///
/// The `Ymm` arm returns the **same result to one bf16 ulp**, not the same
/// bits. Bit-for-bit identity holds **at a fixed ISA**, not from one arm to
/// the other, for three distinct reasons, all measured and all checked:
///
/// 1. **The summation order of Σx²** (`RmsNorm` only): the two arms do not
///    partition the same elements into the same accumulator lanes — sixteen
///    against eight — and do not reduce horizontally by the same tree.
///    Floating addition is not associative: Σx² differs on 45 % of the draws,
///    and the bf16 output on 3.7 coefficients per million at `dim = 2048`.
/// 2. **The denormals** (every operation): `vcvtneps2bf16` returns `±0` as
///    soon as its f32 input has a zero exponent field, where the software
///    rounding of the `Ymm` arm, purely integer, keeps them.
/// 3. **The NaNs**: the software rounding returns `−0.0` (see `emit_elem.rs`).
///
/// This is not a degraded mode: on normal values and outside `RmsNorm` it is
/// the same computation, slower. It is not the same bits.
pub fn select_isa() -> Option<ElemIsa> {
    match crate::jit::engine_isa() {
        crate::jit::IsaEngine::Avx512 => Some(ElemIsa::Zmm),
        crate::jit::IsaEngine::Avx2 => Some(ElemIsa::Ymm),
        crate::jit::IsaEngine::NeonBf16 => Some(ElemIsa::Neon),
        crate::jit::IsaEngine::Rvv => Some(ElemIsa::Rvv),
        crate::jit::IsaEngine::BelowFloor => None,
    }
}

use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// A compiled elementary kernel.
pub struct CompiledElem {
    _exec: Mmap,
    entry: *const u8,
    code_bytes: usize,
}

// SAFETY: the pointer targets a shared, immutable RX page, whose ownership is held by `Mmap`;
// the generated code has no global state.
unsafe impl Send for CompiledElem {}
unsafe impl Sync for CompiledElem {}

impl CompiledElem {
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        unsafe { std::slice::from_raw_parts(self.entry, self.code_bytes).to_vec() }
    }
}

/// One blob on its own executable page.
fn map_blob(blob: &[u8]) -> Result<Mmap> {
    crate::jit::target::map_exec(blob.len().max(64), |map| map[..blob.len()].copy_from_slice(blob))
        .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))
}

type Cache = RwLock<HashMap<ElemSpec, Option<Arc<CompiledElem>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the already cached version.
///
/// `Ok(None)` = shape refused, the caller falls back. `Err` = generator
/// failure, which must propagate.
pub fn get_or_compile(spec: ElemSpec) -> Result<Option<Arc<CompiledElem>>> {
    if select_isa().is_none() {
        return Ok(None);
    }
    if spec.check().is_err() {
        return Ok(None);
    }
    if let Some(hit) = cache().read().unwrap().get(&spec) {
        return Ok(hit.clone());
    }
    // Shape refusals were handled by `spec.check()` above. Any error returned
    // by an emitter here is a generator failure and must reach the caller; it
    // must not be disguised as an unsupported shape.
    let blob = emit_elem(&spec)?;
    let exec = map_blob(&blob)?;
    let built = Some(Arc::new(CompiledElem {
        entry: exec.as_ptr(),
        code_bytes: blob.len(),
        _exec: exec,
    }));
    if crate::jit::options::debug_plan() {
        eprintln!(
            "DEBUG [backend_jit] elem {} dim={} bytes={}",
            spec.op.name(),
            spec.dim,
            blob.len()
        );
    }
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}
