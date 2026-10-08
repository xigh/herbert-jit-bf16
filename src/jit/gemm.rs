//! The GEMM driver: the cache of compiled shapes.
//!
//! The cache is indexed by `(K, N, tile, options, formats)`. A refusal is
//! **stored** as such: a shape the emitter declines does not go through it
//! again. In production the generated program calls the compiled kernels
//! itself, by participant; nothing here runs on the pool.

use crate::jit::compile::{compile, CompileError, CompiledGemm};
use crate::jit::emit::{AFormat, KernelOptions};
use crate::jit::ir::{GemmShape, TileShape};
use crate::core::error::Result;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct CacheKey {
    shape: GemmShape,
    tile: TileShape,
    opts: KernelOptions,
    /// Memory format of `C`. Two distinct kernels, hence two cache inputs:
    /// the epilogue differs (`vcvtneps2bf16` + 32 bytes, or 64 bytes).
    out_bf16: bool,
    /// Memory format of `A`. Same: loading differs (broadcasting
    /// directly, or converting a pair of f32 before broadcasting).
    a_fmt: AFormat,
}

type Cache = RwLock<HashMap<CacheKey, Option<Arc<CompiledGemm>>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| RwLock::new(HashMap::new()))
}

/// Can the full-width generator run on this machine — AVX-512 on x86-64, NEON
/// with BF16 on AArch64, RVV with Zvfbfwma on RISC-V?
pub fn is_supported() -> bool {
    matches!(crate::jit::engine_isa(), crate::jit::IsaEngine::Avx512 | crate::jit::IsaEngine::NeonBf16 | crate::jit::IsaEngine::Rvv)
}

/// Compile `shape` for `tile` or return the cached version.
///
/// `Ok(None)` = shape refused (the caller must fall back). `Err` = generator
/// failure, which must be propagated.
pub fn get_or_compile(
    shape: GemmShape,
    tile: TileShape,
    opts: KernelOptions,
    out_bf16: bool,
    a_fmt: AFormat,
) -> Result<Option<Arc<CompiledGemm>>> {
    let key = CacheKey {
        shape,
        tile,
        opts,
        out_bf16,
        a_fmt,
    };
    if let Some(hit) = cache().read().unwrap().get(&key) {
        return Ok(hit.clone());
    }
    let built = match compile(shape, tile, opts, out_bf16, a_fmt) {
        Ok(g) => Some(Arc::new(g)),
        Err(CompileError::Declined(_)) => None,
        Err(CompileError::Failed(e)) => return Err(e),
    };
    dump_plan(shape, tile, built.as_deref());
    let mut guard = cache().write().unwrap();
    Ok(guard.entry(key).or_insert(built).clone())
}

/// Trace, once per compiled shape, the plan actually retained (off unless
/// `options::debug_plan`).
fn dump_plan(shape: GemmShape, tile: TileShape, built: Option<&CompiledGemm>) {
    if !crate::jit::options::debug_plan() {
        return;
    }
    match built {
        Some(g) => eprintln!(
            "DEBUG [backend_jit] plan K={} N={} tile={} widths={:?}  tiles_n={} \
             kernels={} bytes={}",
            shape.k,
            shape.n,
            tile,
            g.widths(),
            g.n_tile_count(),
            g.kernel_count(),
            g.code_bytes()
        ),
        None => eprintln!(
            "DEBUG [backend_jit] plan K={} N={} tile={} DECLINED (caller fallback)",
            shape.k, shape.n, tile
        ),
    }
}
