//! The RISC-V target, RV64GC with the vector extension 1.0 and the bf16
//! extensions Zvfbfmin and Zvfbfwma: the GEMM micro-kernel
//! (`vfwmaccbf16`), the elementwise, SDPA and GatedDeltaNet emitters, and
//! the generated nodes of the execution plan — every word emitted by the
//! `rv` crate of this workspace.
//!
//! Every kernel is independent of VLEN: it asks `vsetvli` for its vector
//! length and strip-mines, so one binary runs at VLEN 256 … 4096. Where a
//! sum spans the whole register its bits depend on how many lanes it folds;
//! the width is a knob (`--rvv-width`), and every transcription takes it.
//!
//! The ABI of generated code is LP64D (`extern "C"`): integer arguments in
//! `a0..a7`, floating-point ones in `fa0..fa7`. `gp` and `tp` are never
//! written; `s0..s11` and `fs0..fs11` are saved by any kernel that uses
//! them; the vector registers are all the caller's to lose.

pub mod deltanet;
pub mod elem;
pub mod execution;
pub mod gemm;
pub mod gemm_wide;
pub mod sdpa;
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
use execution::{attention, audit, bf16, delta, entry, frame, kernels, kv, transforms};

/// `riscv_flush_icache` of the Linux RISC-V system call table.
const SYS_RISCV_FLUSH_ICACHE: libc::c_long = 259;

/// Make the instruction cache of every hart see the bytes just written at
/// `ptr..ptr+len`: `fence.i` would only cover the hart that runs it, and the
/// pool runs the code on all of them.
pub(crate) fn flush_icache(ptr: *const u8, len: usize) {
    // SAFETY: the range is a live mapping of this process; flags 0 = all harts.
    unsafe {
        libc::syscall(SYS_RISCV_FLUSH_ICACHE, ptr, ptr.add(len), 0usize);
    }
}

/// The RISC-V target.
pub(crate) struct Rv64;

impl Target for Rv64 {
    const R16_ROWS: usize = 16;
    fn gemm_tile(_shape: GemmShape) -> TileShape {
        match crate::jit::knobs::knobs().tile {
            crate::jit::TilePolicy::Fixed(t) => t,
            _ => gemm::RV_TILE,
        }
    }
    /// The vector tile and the RISC-V knobs ([`crate::jit::knobs::Knobs::rvv`]):
    /// one row (the decode) under the default policy takes the width of the knob;
    /// the form `wide` narrows the default tiles until they fit its registers.
    fn gemm_plan(shape: GemmShape, rows: usize) -> crate::jit::emit::GemmPlan {
        let rvv = crate::jit::knobs::knobs().rvv;
        let wide = |t: TileShape| match rvv.gemm {
            crate::jit::emit::RvvGemm::Seg => t,
            crate::jit::emit::RvvGemm::Wide => gemm_wide::tile(t, rvv.width, crate::core::isa::vlenb()),
        };
        let tile = match crate::jit::knobs::knobs().tile {
            crate::jit::TilePolicy::Fixed(t) => t,
            crate::jit::TilePolicy::Default_ if rows == 1 => wide(TileShape { mr: 1, nr: usize::from(rvv.nr) }),
            _ => wide(Self::gemm_tile(shape)),
        };
        let opts = crate::jit::emit::KernelOptions { rvv, ..Default::default() };
        crate::jit::emit::GemmPlan { tile, opts }
    }
    fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
        gemm::tile_fits(tile)
    }
    fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
        gemm::widest_tile(mr, max_cols)
    }
    fn check_gemm(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
        gemm::check(spec)
    }
    fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline> {
        deltanet::check_recurrence(spec)
    }
    fn emit_gemm(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
        gemm::emit(spec)
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

impl KernelSet for Rv64 {
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
