//! The driver of the GatedDeltaNet recurrence kernel: compilation and
//! cache. One blob per value head width `hvd`, compiled at mount by
//! `Scheduler::ascend`; nothing is compiled during a pass.
//!
//! Two forms exist: without FMA (the bits of the scalar reference, the
//! default) and with `vfmadd231ps` on the three dot products
//! (`Knobs::deltanet_fma`, measured at zero gain: the FMA removes additions
//! that were not the bottleneck).

use crate::jit::emit_recurrence::{emit_recurrence, RecurrenceSpec};
use crate::core::error::{HerbertError, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

/// A compiled recurrence kernel, for a width `hvd`.
pub struct CompiledRecurrence {
    spec: RecurrenceSpec,
    _exec: Mmap,
    entry: *const u8,
    code_bytes: usize,
}

// SAFETY : the pointer targets a shared and immutable RX page, whose
// ownership is retained by `Mmap` ; the generated code has no global
// state.
unsafe impl Send for CompiledRecurrence {}
unsafe impl Sync for CompiledRecurrence {}

impl CompiledRecurrence {
    /// The compiled specification.
    pub fn spec(&self) -> RecurrenceSpec {
        self.spec
    }
    pub(crate) fn lowering_bytes(&self) -> Vec<u8> {
        // Exact instruction extent recorded before the RX mapping was installed.
        unsafe { std::slice::from_raw_parts(self.entry, self.code_bytes) }.to_vec()
    }

    /// The V head width this kernel serves — and the only one.
    pub fn hvd(&self) -> usize {
        self.spec.hvd
    }
}

/// Is the core eligible on this machine? It only requires AVX-512F, but the
/// question "what ISA this process uses" has only one answer —
/// [`crate::jit::engine_isa`] — and an AVX2 lane is not written (16 ymm: one would
/// have to split `vd` into two passes).
pub fn is_supported() -> bool {
    matches!(crate::jit::engine_isa(), crate::jit::IsaEngine::Avx512 | crate::jit::IsaEngine::NeonBf16 | crate::jit::IsaEngine::Rvv)
}

type Cache = RwLock<HashMap<RecurrenceSpec, Option<Arc<CompiledRecurrence>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Compile `spec`, or return the already cached version.
///
/// `Ok(None)` = no AVX-512 or rejected form, the caller keeps the scalar.
/// `Err` = generator failure, which must propagate.
pub fn get_or_compile(spec: RecurrenceSpec) -> Result<Option<Arc<CompiledRecurrence>>> {
    if !is_supported() || spec.check().is_err() {
        return Ok(None);
    }
    if let Some(hit) = cache().read().unwrap().get(&spec) {
        return Ok(hit.clone());
    }
    let blob = emit_recurrence(&spec)?;
    let exec = crate::jit::target::map_exec(blob.len().max(64), |map| map[..blob.len()].copy_from_slice(&blob))
        .map_err(|e| HerbertError::Backend(format!("executable mapping: {e}")))?;
    let built = Some(Arc::new(CompiledRecurrence {
        spec,
        entry: exec.as_ptr(),
        code_bytes: blob.len(),
        _exec: exec,
    }));
    if crate::jit::options::debug_plan() {
        eprintln!(
            "DEBUG [backend_jit] recurrence deltanet hvd={} fma={} bytes={}",
            spec.hvd,
            if spec.fma { "yes" } else { "no" },
            blob.len()
        );
    }
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(spec).or_insert(built).clone())
}

/// V heads on the calling thread alone ([`crate::jit::knobs::Knobs::deltanet_recurrence_team`] off).
pub fn single_thread() -> bool {
    !crate::jit::knobs::knobs().deltanet_recurrence_team
}

/// Does the form to emit carry the FMA ([`crate::jit::knobs::Knobs::deltanet_fma`])?
/// Measured at zero gain.
pub fn fma_active() -> bool {
    crate::jit::knobs::knobs().deltanet_fma
}

/// The slices of independent value columns a head is cut into
/// ([`crate::jit::knobs::Knobs::deltanet_vd_split`]): the knob when the head
/// divides into slices of 32, one otherwise. AArch64; one on x86.
pub fn vd_split(hvd: usize) -> usize {
    let split = crate::jit::knobs::knobs().deltanet_vd_split.max(1);
    if cfg!(target_arch = "aarch64") && split > 1 && hvd.is_multiple_of(32 * split) {
        split
    } else {
        1
    }
}

/// The recurrence kernel the knobs ask for: the FMA choice on both targets,
/// the AArch64 forms on AArch64 only.
pub fn spec_of_knobs(hvd: usize) -> RecurrenceSpec {
    let k = crate::jit::knobs::knobs();
    let arm = cfg!(target_arch = "aarch64");
    let split = vd_split(hvd);
    RecurrenceSpec {
        store_once: arm && k.deltanet_store_once,
        pairs: arm && k.deltanet_pairs,
        q_hoist: arm && k.deltanet_q_hoist,
        pass1_lanes: if arm { k.deltanet_pass1_lanes } else { 32 },
        width: if split > 1 { hvd / split } else { 0 },
        ..RecurrenceSpec::new(hvd, fma_active())
    }
}
