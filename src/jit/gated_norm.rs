//! The driver of the gated RMS norm kernel of the GatedDeltaNet mixer:
//! compilation and cache. One blob per row width `dim`, compiled at
//! mount by `Scheduler::ascend`; nothing is compiled during a pass.

use crate::jit::emit_gated_norm::{emit_gated_norm, GatedNormSpec};
use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// A compiled gated norm kernel, for a `dim`.
pub struct CompiledGatedNorm {
    spec: GatedNormSpec,
    _exec: Mmap,
    entry: *const u8,
    code_bytes: usize,
}

// SAFETY: the pointer targets a shared, immutable RX page, whose ownership is held by `Mmap`;
// the generated code has no global state.
unsafe impl Send for CompiledGatedNorm {}
unsafe impl Sync for CompiledGatedNorm {}

impl CompiledGatedNorm {
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        // Exact instruction extent recorded before the RX mapping was installed.
        unsafe { std::slice::from_raw_parts(self.entry, self.code_bytes) }.to_vec()
    }

    /// The line width that this kernel serves.
    pub fn dim(&self) -> usize {
        self.spec.dim
    }
}

/// Is the kernel admissible on this machine? AVX-512 only (the
/// bf16 conversion is `vcvtneps2bf16`, as in `ElemOp::RmsNorm`).
pub fn is_supported() -> bool {
    matches!(crate::jit::engine_isa(), crate::jit::IsaEngine::Avx512 | crate::jit::IsaEngine::NeonBf16 | crate::jit::IsaEngine::Rvv)
}

type Cache = RwLock<HashMap<GatedNormSpec, Option<Arc<CompiledGatedNorm>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the cached version. `Ok(None)` = no
/// AVX-512 or out-of-budget shape, the caller keeps the scalar; `Err` =
/// generator failure.
pub fn get_or_compile(spec: GatedNormSpec) -> Result<Option<Arc<CompiledGatedNorm>>> {
    if !is_supported() || spec.check().is_err() {
        return Ok(None);
    }
    if let Some(hit) = cache().read().unwrap().get(&spec) {
        return Ok(hit.clone());
    }
    let blob = emit_gated_norm(&spec)?;
    let exec = crate::jit::target::map_exec(blob.len().max(64), |map| map[..blob.len()].copy_from_slice(&blob))
        .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))?;
    let built = Some(Arc::new(CompiledGatedNorm {
        spec,
        entry: exec.as_ptr(),
        code_bytes: blob.len(),
        _exec: exec,
    }));
    if crate::jit::options::debug_plan() {
        eprintln!("DEBUG [backend_jit] gated norm deltanet dim={} bytes={}", spec.dim, blob.len());
    }
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}

/// Row ranges on the caller thread alone ([`crate::jit::knobs::Knobs::deltanet_norm_team`] off).
pub fn single_thread() -> bool {
    !crate::jit::knobs::knobs().deltanet_norm_team
}
