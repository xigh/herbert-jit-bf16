//! The target boundary. Every module that emits machine code lives under its
//! target — `x86` (`iced-x86`, AVX2/AVX-512), `aarch64` (the `a64` crate,
//! NEON with the BF16 extension) or `riscv64` (the `rv` crate, RVV 1.0 with
//! the bf16 extensions) — and exactly one is compiled in. The
//! specifications (`jit/emit*.rs`), the drivers, the plan, the lowering, the
//! options and the runtime stay above it and reach the target only through
//! [`Target`] and [`KernelSet`], resolved statically on [`Native`].

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

#[cfg(target_arch = "aarch64")]
pub mod aarch64;
#[cfg(target_arch = "riscv64")]
pub mod riscv64;
#[cfg(target_arch = "x86_64")]
pub mod x86;

/// Declares a function with the calling convention of generated code on this target: SysV64
/// on x86-64, on every operating system (Windows included); C elsewhere.
#[cfg(target_arch = "x86_64")]
macro_rules! generated_abi {
    ($(#[$meta:meta])* $vis:vis unsafe fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])* $vis unsafe extern "sysv64" fn $name($($arg: $ty),*) $body
    };
}
#[cfg(not(target_arch = "x86_64"))]
macro_rules! generated_abi {
    ($(#[$meta:meta])* $vis:vis unsafe fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $body:block) => {
        $(#[$meta])* $vis unsafe extern "C" fn $name($($arg: $ty),*) $body
    };
}
/// The pointer type of a function with the calling convention of generated code;
/// `generated_fn!(A, B => R)` returns `R`.
#[cfg(target_arch = "x86_64")]
macro_rules! generated_fn {
    ($($ty:ty),* => $ret:ty) => { unsafe extern "sysv64" fn($($ty),*) -> $ret };
    ($($ty:ty),* $(,)?) => { unsafe extern "sysv64" fn($($ty),*) };
}
#[cfg(not(target_arch = "x86_64"))]
macro_rules! generated_fn {
    ($($ty:ty),* => $ret:ty) => { unsafe extern "C" fn($($ty),*) -> $ret };
    ($($ty:ty),* $(,)?) => { unsafe extern "C" fn($($ty),*) };
}
pub(crate) use generated_abi;
pub(crate) use generated_fn;

/// The target compiled into this binary.
#[cfg(target_arch = "x86_64")]
pub(crate) type Native = x86::X86;
/// The target compiled into this binary.
#[cfg(target_arch = "aarch64")]
pub(crate) type Native = aarch64::Arm64;
/// The target compiled into this binary.
#[cfg(target_arch = "riscv64")]
pub(crate) type Native = riscv64::Rv64;

/// One assembled whole-call page, before it is mapped executable.
pub(crate) struct Assembled {
    pub code: Vec<u8>,
    /// Entries without and with the progress bridge.
    pub entry_offsets: [usize; 2],
    pub kernel_offsets: Vec<usize>,
    /// Offset of every runtime call and the `RunContext` field it calls through.
    pub runtime_calls: Vec<(usize, usize)>,
}

/// Rows interleaved per block of an R16-packed bf16 weight, as the target's GEMM reads them.
pub const R16_ROWS: usize = <Native as Target>::R16_ROWS;

/// The tile the target's bf16 GEMM uses for `shape`.
pub fn gemm_tile(shape: GemmShape) -> TileShape {
    <Native as Target>::gemm_tile(shape)
}
/// How the target runs a projection of `shape` in a regime of `rows` rows per
/// block: the vector tile of [`gemm_tile`], or its matrix engine.
pub fn gemm_plan(shape: GemmShape, rows: usize) -> crate::jit::emit::GemmPlan {
    <Native as Target>::gemm_plan(shape, rows)
}
/// Does `tile` fit the target's registers?
pub fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
    <Native as Target>::tile_fits(tile)
}
/// The widest legal tile width ≤ `max_cols` at `mr` rows, on the target.
pub fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
    <Native as Target>::widest_tile(mr, max_cols)
}
/// The target side of [`MicroKernelSpec::check`].
pub(crate) fn check_gemm(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    <Native as Target>::check_gemm(spec)
}
/// The target side of [`RecurrenceSpec::check`].
pub(crate) fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline> {
    <Native as Target>::check_recurrence(spec)
}
pub(crate) fn emit_gemm(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_gemm(spec)
}
pub(crate) fn emit_elem(spec: &ElemSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_elem(spec)
}
pub(crate) fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
    <Native as Target>::emit_sdpa(spec, pass)
}
pub(crate) fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_sdpa_prefill(spec)
}
pub(crate) fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_conv(spec)
}
pub(crate) fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_recurrence(spec)
}
pub(crate) fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
    <Native as Target>::emit_gated_norm(spec)
}

/// Copy machine code into a fresh anonymous mapping and make it executable.
///
/// `fill` writes the code into the zeroed read-write buffer of `len` bytes.
/// On AArch64 and RISC-V the instruction cache does not see data writes: the
/// range is invalidated (on RISC-V, on every hart) before the mapping turns
/// read-execute, or a core could run stale words.
pub(crate) fn map_exec(len: usize, fill: impl FnOnce(&mut [u8])) -> std::io::Result<memmap2::Mmap> {
    let mut map = memmap2::MmapMut::map_anon(len.max(1))?;
    fill(&mut map[..]);
    #[cfg(target_arch = "aarch64")]
    aarch64::flush_icache(map.as_ptr(), map.len());
    #[cfg(target_arch = "riscv64")]
    riscv64::flush_icache(map.as_ptr(), map.len());
    map.make_exec()
}

/// What the plan needs from a target around its kernels.
pub(crate) trait Target {
    /// Weight packing chosen by the target: rows per R16 block.
    const R16_ROWS: usize;
    /// The bf16 GEMM tile for a shape on this machine.
    fn gemm_tile(shape: GemmShape) -> TileShape;
    /// The GEMM plan of a regime of `rows` rows per block.
    fn gemm_plan(shape: GemmShape, _rows: usize) -> crate::jit::emit::GemmPlan {
        crate::jit::emit::GemmPlan { tile: Self::gemm_tile(shape), opts: Default::default() }
    }
    /// Does the tile fit the registers of the GEMM micro-kernel?
    fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline>;
    /// The widest legal width ≤ `max_cols` for `mr` rows, or `None`.
    fn widest_tile(mr: usize, max_cols: usize) -> Option<usize>;
    /// The register and addressing budget of one micro-kernel.
    fn check_gemm(spec: &MicroKernelSpec) -> std::result::Result<(), Decline>;
    /// The register budget of the recurrence kernel.
    fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline>;
    /// The GEMM micro-kernel.
    fn emit_gemm(spec: &MicroKernelSpec) -> Result<Vec<u8>>;
    /// An elementwise kernel.
    fn emit_elem(spec: &ElemSpec) -> Result<Vec<u8>>;
    /// The decode attention of one head, whole or one pass.
    fn emit_sdpa(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>>;
    /// The prefill attention block.
    fn emit_sdpa_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>>;
    /// The DeltaNet causal conv + SiLU of one position.
    fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>>;
    /// The DeltaNet recurrence of one head and one position.
    fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>>;
    /// The DeltaNet gated RMS norm.
    fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>>;
    /// Check the control-flow boundary of an emitted kernel.
    fn audit_kernel(bytes: &[u8]) -> std::result::Result<(), ExecutionError>;
    /// The machine code of a kernel that only returns.
    fn ret_kernel() -> std::result::Result<Vec<u8>, ExecutionError>;
    /// Emit both entries of `plan`, the shared barrier when requested, and its kernels.
    fn assemble(
        plan: &ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
        shared_barrier: bool,
        fuse_serial: bool,
    ) -> std::result::Result<Assembled, ExecutionError>;
    /// Verify every call of an assembled page, runtime calls included.
    fn audit_calls(code: &[u8], runtime_calls: &[(usize, usize)]) -> std::result::Result<(), ExecutionError>;
}

type Kernel = std::result::Result<GeneratedKernel, ExecutionError>;

/// One constructor per kernel specification the lowering uses.
pub(crate) trait KernelSet {
    // Block frame.
    fn begin_block(max_batch: usize, grids: &[TaskGrid]) -> Kernel;
    fn end_block() -> Kernel;
    // Embedding and output.
    fn embedding() -> Kernel;
    fn argmax_bf16() -> Kernel;
    /// The argmax in slices then a merge (`Knobs::argmax_tasks`); AArch64.
    fn argmax_slices() -> Kernel;
    fn argmax_merge() -> Kernel;
    // Row transforms.
    fn elementary(kernel: &crate::jit::elem::CompiledElem) -> Kernel;
    fn offset(factor: usize, subtract: usize) -> Kernel;
    fn norm_rope(
        norm: &crate::jit::elem::CompiledElem,
        rope: &crate::jit::elem::CompiledElem,
        heads: usize,
        stride: usize,
    ) -> Kernel;
    // Projections.
    fn bf16_tile_selector(gemm: &crate::jit::compile::CompiledGemm) -> Kernel;
    /// Several projections of one input in one node (`Knobs::fuse_projections`); AArch64.
    fn bf16_multi_tile_selector(parts: &[&crate::jit::compile::CompiledGemm]) -> Kernel;
    /// `(call: *const PackCall, block)`: sixteen rows of `A` packed by pairs of
    /// `K` for the matrix engine ([`crate::jit::emit::packed_a_bytes`]). Only
    /// asked when [`Target::gemm_plan`] chose the matrix engine.
    fn bf16_pack(k: usize) -> Kernel {
        let _ = k;
        Err(ExecutionError("this target has no matrix engine to pack for".into()))
    }
    /// `(call: *const PackCall, block)`: thirty-two rows of `A` packed by
    /// single `K` for Apple's AMX ([`crate::jit::emit::packed_a_amx_bytes`]).
    /// Only asked when [`Target::gemm_plan`] chose the AMX.
    fn bf16_pack_amx(k: usize) -> Kernel {
        let _ = k;
        Err(ExecutionError("this target has no AMX to pack for".into()))
    }
    // Attention and KV.
    fn attention_tile_selector(
        sdpa: &crate::jit::sdpa::CompiledSdpa,
        prefill: Option<&crate::jit::sdpa::CompiledSdpaPrefill>,
        spec: AttentionSpec,
    ) -> Kernel;
    fn kv_append(spec: KvSpec) -> Kernel;
    // DeltaNet.
    fn delta_convolution(shape: Shape, kernel: Option<&crate::jit::conv::CompiledConv>, fma: bool) -> Kernel;
    fn delta_recurrence(shape: Shape, kernel: Option<&crate::jit::recurrence::CompiledRecurrence>) -> Kernel;
    fn delta_normed_gate(
        dim: usize,
        kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
        fma: bool,
    ) -> Kernel;
    fn delta_interleave(nv: usize) -> Kernel;
    fn delta_l2_gates(shape: Shape, fma: bool) -> Kernel;
}
