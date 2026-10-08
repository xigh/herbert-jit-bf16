//! The driver of the causal conv + SiLU kernel of the GatedDeltaNet mixer:
//! compilation and cache. One blob per kernel width `K`, compiled at
//! mount by `Scheduler::ascend`; nothing is compiled during a pass.

use crate::jit::emit_conv::{emit_conv, ConvSpec};
use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// A compiled conv kernel, for a `K`.
pub struct CompiledConv {
    spec: ConvSpec,
    _exec: Mmap,
    entry: *const u8,
    code_bytes: usize,
}

// SAFETY: the pointer targets a shared, immutable RX page, whose ownership is held by `Mmap`;
// the generated code has no global state.
unsafe impl Send for CompiledConv {}
unsafe impl Sync for CompiledConv {}

impl CompiledConv {
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        // Exact instruction extent recorded before the RX mapping was installed.
        unsafe { std::slice::from_raw_parts(self.entry, self.code_bytes) }.to_vec()
    }

    /// The `K` that this kernel serves.
    pub fn k(&self) -> usize {
        self.spec.k
    }
}

/// Is the kernel emittable on this machine? AVX-512 only (no
/// AVX2 arm: it would require 8 channels per block and a different grouping).
pub fn is_supported() -> bool {
    matches!(crate::jit::engine_isa(), crate::jit::IsaEngine::Avx512 | crate::jit::IsaEngine::NeonBf16 | crate::jit::IsaEngine::Rvv)
}

type Cache = RwLock<HashMap<ConvSpec, Option<Arc<CompiledConv>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the cached version. `Ok(None)` = no
/// AVX-512 or `K ≠ 4`, the caller keeps the scalar; `Err` = generator
/// failure.
pub fn get_or_compile(spec: ConvSpec) -> Result<Option<Arc<CompiledConv>>> {
    if !is_supported() || spec.check().is_err() {
        return Ok(None);
    }
    if let Some(hit) = cache().read().unwrap().get(&spec) {
        return Ok(hit.clone());
    }
    let blob = emit_conv(&spec)?;
    let exec = crate::jit::target::map_exec(blob.len().max(64), |map| map[..blob.len()].copy_from_slice(&blob))
        .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))?;
    let built = Some(Arc::new(CompiledConv {
        spec,
        entry: exec.as_ptr(),
        code_bytes: blob.len(),
        _exec: exec,
    }));
    if crate::jit::options::debug_plan() {
        eprintln!("DEBUG [backend_jit] conv deltanet K={} bytes={}", spec.k, blob.len());
    }
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}

/// Channel ranges on the caller thread alone ([`crate::jit::knobs::Knobs::deltanet_conv_team`] off).
pub fn single_thread() -> bool {
    !crate::jit::knobs::knobs().deltanet_conv_team
}
