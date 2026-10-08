//! The AArch64 target, NEON with the BF16 extension: the GEMM micro-kernel
//! (`bfdot`), the elementwise, SDPA and GatedDeltaNet emitters, and the
//! generated nodes of the execution plan — every word emitted by the `a64`
//! crate of this workspace.
//!
//! The ABI of generated code is AAPCS64 (`extern "C"`): six integer
//! arguments in `x0..x5`, eight floats in `s0..s7`. `x18` is never written
//! (the platform register of Apple systems), `x19..x28` and the low halves of
//! `v8..v15` are saved by any kernel that uses them.

pub mod deltanet;
pub mod deltanet_chunk;
pub mod elem;
pub mod execution;
pub mod gemm;
pub mod gemm_amx;
pub mod gemm_sme;
pub mod sdpa;
pub mod sdpa_block;
pub(crate) mod util;

use crate::core::error::Result;
use crate::jit::emit::MicroKernelSpec;
use crate::jit::emit_conv::ConvSpec;
use crate::jit::emit_elem::ElemSpec;
use crate::jit::emit_gated_norm::GatedNormSpec;
use crate::jit::emit_recurrence::RecurrenceSpec;
use crate::jit::emit_sdpa::{Pass, SdpaKernelSpec, SdpaPrefillSpec};
use crate::jit::execution::abi::{AttentionSpec, KvSpec, Shape, TaskGrid};
use crate::jit::execution::{ExecutionError, ExecutionPlan, GeneratedKernel};
use crate::jit::ir::{Decline, GemmShape, TileShape};
use crate::jit::target::{Assembled, KernelSet, Target};
use crate::jit::TilePolicy;
use execution::{attention, audit, bf16, delta, entry, frame, kernels, kv, transforms};

#[cfg(target_os = "macos")]
extern "C" {
    /// libSystem's instruction-cache invalidation, `<libkern/OSCacheControl.h>`.
    fn sys_icache_invalidate(start: *mut std::ffi::c_void, len: usize);
}

/// Make the instruction cache see the bytes just written at `ptr..ptr+len`.
pub(crate) fn flush_icache(ptr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    // SAFETY: the range is a live mapping of this process.
    unsafe {
        sys_icache_invalidate(ptr as *mut _, len)
    };
}

/// The AArch64 target.
pub(crate) struct Arm64;

/// The GEMM tile of a shape: the pinned one, or [`gemm::ARM_TILE`].
fn production_tile() -> TileShape {
    match crate::jit::knobs::knobs().tile {
        TilePolicy::Fixed(t) => t,
        TilePolicy::Default_ | TilePolicy::PerShape => gemm::ARM_TILE,
    }
}

/// The tile of the matrix engine: sixteen rows, the four ZA tiles.
pub const SME_TILE: TileShape = TileShape { mr: 16, nr: gemm_sme::SME_MAX_COLS };

impl Target for Arm64 {
    const R16_ROWS: usize = 16;
    fn gemm_tile(_shape: GemmShape) -> TileShape {
        production_tile()
    }
    /// Blocks of several rows run on the matrix engine when the machine has
    /// SME2 and the knob allows it ([`crate::jit::knobs::Knobs::gemm_matrix`]);
    /// one row (the decode) stays on the vector tile, bound by memory.
    ///
    /// With [`crate::jit::knobs::Knobs::gemm_amx`], those blocks run on
    /// Apple's AMX instead (`gemm_amx.rs`), the bits of the `bfmlal` vector
    /// kernel, on any Apple M (SME2 or not).
    fn gemm_plan(shape: GemmShape, rows: usize) -> crate::jit::emit::GemmPlan {
        let k8 = shape.k.is_multiple_of(8);
        let amx = rows > 1 && crate::jit::knobs::knobs().gemm_amx && gemm_amx::available() && k8;
        let matrix = rows > 1
            && crate::jit::knobs::knobs().gemm_matrix
            && gemm_sme::available()
            && k8;
        let neon = crate::jit::knobs::knobs().neon;
        if amx {
            // No vector tails: a residue of rows is a padded tile.
            let neon = crate::jit::emit::NeonOptions { tail_rows: 0, ..neon };
            let opts = crate::jit::emit::KernelOptions { matrix: true, amx: true, neon, ..Default::default() };
            crate::jit::emit::GemmPlan { tile: gemm_amx::AMX_TILE, opts }
        } else if matrix {
            let opts = crate::jit::emit::KernelOptions { matrix: true, neon, ..Default::default() };
            crate::jit::emit::GemmPlan { tile: SME_TILE, opts }
        } else {
            // One row (the decode) under the default policy: the width of the knob.
            let tile = match crate::jit::knobs::knobs().tile {
                TilePolicy::Default_ if rows == 1 => TileShape { mr: 1, nr: neon.nr },
                _ => production_tile(),
            };
            let opts = crate::jit::emit::KernelOptions { neon, ..Default::default() };
            crate::jit::emit::GemmPlan { tile, opts }
        }
    }
    /// A vector tile, or a matrix-engine tile when the machine has one.
    fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
        gemm::tile_fits(tile).or_else(|e| {
            let sme = gemm_sme::available()
                && (1..=crate::jit::emit::PACK_ROWS).contains(&tile.mr)
                && tile.nr > 0
                && tile.nr <= gemm_sme::SME_MAX_COLS
                && tile.nr.is_multiple_of(16);
            let amx = gemm_amx::available() && (1..=crate::jit::emit::AMX_ROWS).contains(&tile.mr) && tile.nr == 16;
            if sme || amx { Ok(()) } else { Err(e) }
        })
    }
    fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
        let vector = gemm::widest_tile(mr, max_cols);
        let matrix = (gemm_sme::available() && mr <= crate::jit::emit::PACK_ROWS && max_cols >= 16)
            .then(|| max_cols.min(gemm_sme::SME_MAX_COLS) / 16 * 16);
        // The AMX tile is sixteen columns, whatever the rows.
        let amx = (gemm_amx::available() && mr <= crate::jit::emit::AMX_ROWS && max_cols >= 16).then_some(16);
        vector.max(matrix).max(amx)
    }
    fn check_gemm(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
        if spec.opts.amx {
            gemm_amx::check(spec)
        } else if spec.opts.matrix {
            gemm_sme::check(spec)
        } else {
            gemm::check(spec)
        }
    }
    fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline> {
        deltanet::check_recurrence(spec)
    }
    fn emit_gemm(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
        if spec.opts.amx {
            gemm_amx::emit(spec)
        } else if spec.opts.matrix {
            gemm_sme::emit(spec)
        } else {
            gemm::emit(spec)
        }
    }
    fn emit_elem(spec: &ElemSpec) -> Result<Vec<u8>> {
        elem::emit(spec)
    }
    fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
        sdpa::emit(spec, pass)
    }
    fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
        sdpa::emit_prefill(spec)
    }
    fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
        deltanet::emit_conv(spec)
    }
    fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
        deltanet::emit_recurrence(spec)
    }
    fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
        deltanet::emit_gated_norm(spec)
    }
    fn audit_kernel(bytes: &[u8]) -> std::result::Result<(), ExecutionError> {
        audit::kernel(bytes)
    }
    fn ret_kernel() -> std::result::Result<Vec<u8>, ExecutionError> {
        audit::ret()
    }
    fn assemble(
        plan: &ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
        shared_barrier: bool,
        fuse_serial: bool,
    ) -> std::result::Result<Assembled, ExecutionError> {
        entry::assemble(plan, hybrid_spin, trace_sync, trace_pmu, shared_barrier, fuse_serial)
    }
    fn audit_calls(code: &[u8], runtime_calls: &[(usize, usize)]) -> std::result::Result<(), ExecutionError> {
        entry::audit_calls(code, runtime_calls)
    }
}

type Kernel = std::result::Result<GeneratedKernel, ExecutionError>;

impl KernelSet for Arm64 {
    fn begin_block(max_batch: usize, grids: &[TaskGrid]) -> Kernel {
        frame::begin_block(max_batch, grids)
    }
    fn end_block() -> Kernel {
        frame::end_block()
    }
    fn embedding() -> Kernel {
        kernels::embedding()
    }
    fn argmax_bf16() -> Kernel {
        kernels::argmax_bf16()
    }
    fn argmax_slices() -> Kernel {
        kernels::argmax_slices()
    }
    fn argmax_merge() -> Kernel {
        kernels::argmax_merge()
    }
    fn elementary(kernel: &crate::jit::elem::CompiledElem) -> Kernel {
        transforms::elementary(kernel)
    }
    fn offset(factor: usize, subtract: usize) -> Kernel {
        transforms::offset(factor, subtract)
    }
    fn norm_rope(
        norm: &crate::jit::elem::CompiledElem,
        rope: &crate::jit::elem::CompiledElem,
        heads: usize,
        stride: usize,
    ) -> Kernel {
        transforms::norm_rope(norm, rope, heads, stride)
    }
    fn bf16_tile_selector(gemm: &crate::jit::compile::CompiledGemm) -> Kernel {
        bf16::tile_selector(gemm)
    }
    fn bf16_multi_tile_selector(parts: &[&crate::jit::compile::CompiledGemm]) -> Kernel {
        bf16::multi_tile_selector(parts)
    }
    fn bf16_pack(k: usize) -> Kernel {
        bf16::pack(k)
    }
    fn bf16_pack_amx(k: usize) -> Kernel {
        bf16::pack_amx(k)
    }
    fn attention_tile_selector(
        sdpa: &crate::jit::sdpa::CompiledSdpa,
        prefill: Option<&crate::jit::sdpa::CompiledSdpaPrefill>,
        spec: AttentionSpec,
    ) -> Kernel {
        attention::tile_selector(sdpa, prefill, spec)
    }
    fn kv_append(spec: KvSpec) -> Kernel {
        kv::append(spec)
    }
    fn delta_convolution(shape: Shape, kernel: Option<&crate::jit::conv::CompiledConv>, fma: bool) -> Kernel {
        delta::convolution(shape, kernel, fma)
    }
    fn delta_recurrence(shape: Shape, kernel: Option<&crate::jit::recurrence::CompiledRecurrence>) -> Kernel {
        delta::recurrence(shape, kernel)
    }
    fn delta_normed_gate(
        dim: usize,
        kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
        fma: bool,
    ) -> Kernel {
        delta::normed_gate(dim, kernel, fma)
    }
    fn delta_interleave(nv: usize) -> Kernel {
        delta::interleave(nv)
    }
    fn delta_l2_gates(shape: Shape, fma: bool) -> Kernel {
        delta::l2_gates(shape, fma)
    }
}
