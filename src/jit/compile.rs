//! The GEMM compiler: a shape `(K, N)` and a tile give a set of micro-kernels
//! and the plan that chains them over the columns.
//!
//! `N` is known at load: the column plan is computed once — tiles of `nr`
//! columns, then the widest legal tile for what remains. Every kernel is
//! exact, there is no tail code, only one more kernel; the last tile masks its
//! store when `N` is not a multiple of 16. `M` varies per call: the emitter
//! produces every height `1..=mr`, and the driver stacks `⌊M/mr⌋` full tiles
//! plus one of `M mod mr`. All the kernels of a shape share one executable page.

use crate::jit::emit::{emit, AFormat, KernelFn, KernelOptions, MicroKernelSpec};
use crate::jit::ir::{widest_tile, Decline, GemmShape, TileShape, ROWS_INTERLEAVED};
use crate::jit::KERNEL_ALIGN;
use crate::core::error::HerbertError;
use memmap2::Mmap;

/// Compilation failure: an assumed refusal (the caller falls back to v2) or a
/// real generator crash.
#[derive(Debug)]
pub enum CompileError {
    /// Shape out of the specializable domain.
    Declined(Decline),
    /// Emission or mapping failed — it's a bug, not a refusal.
    Failed(HerbertError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Declined(d) => write!(f, "declines: {d}"),
            CompileError::Failed(e) => write!(f, "failure: {e}"),
        }
    }
}

impl From<Decline> for CompileError {
    fn from(d: Decline) -> Self {
        CompileError::Declined(d)
    }
}

/// The most residue rows a vector tail kernel serves: six rows of sixteen
/// columns hold 24 accumulators, two registers of `A` and four of `W`.
pub const MAX_TAIL_ROWS: usize = 6;

/// A compiled GEMM `(K, N)` for a given tile.
pub struct CompiledGemm {
    shape: GemmShape,
    tile: TileShape,
    opts: KernelOptions,
    out_bf16: bool,
    /// Memory layout of `A`. See [`MicroKernelSpec::a_fmt`].
    a_fmt: AFormat,
    /// Keeps the executable page alive: the `kernels` pointers
    /// point into it.
    _exec: Mmap,
    /// `kernels[genre][hauteur - 1]`.
    kernels: Vec<Vec<KernelFn>>,
    /// Matrix path with a tail: `tails[genre][r - 1]`, the vector kernel of
    /// `r` residue rows and sixteen columns reading packed `A`
    /// (`NeonOptions::tail_rows`), then — for a kind whose last group of
    /// sixteen is partial — the same kernels storing that partial group.
    tails: Vec<Vec<KernelFn>>,
    /// Exact instruction extents, excluding inter-kernel alignment.
    kernel_lengths: Vec<usize>,
    /// Tile kinds distinct from the plan: `(width in columns, columns
    /// valid in the last group of 16)`. Two tiles of the same width but
    /// with a different tail are **two kernels**: one masks its last
    /// store, the other does not.
    kinds: Vec<(usize, usize)>,
    /// Column plan: `(offset en columns, index de genre)`.
    n_tiles: Vec<(usize, usize)>,
    code_bytes: usize,
}

impl CompiledGemm {
    /// Residue rows served by a vector tail kernel on the matrix path.
    pub fn tail_rows(&self) -> usize {
        if self.opts.matrix { usize::from(self.opts.neon.tail_rows).min(MAX_TAIL_ROWS) } else { 0 }
    }

    /// The sets of tail kernels per kind: the `tail_mac` one, and the `bfdot`
    /// one when residues up to `tail_bfdot_max` take it and `tail_mac` is not
    /// `bfdot` already.
    pub fn tail_macs(&self) -> usize {
        let neon = self.opts.neon;
        if self.tail_rows() > 0 && neon.tail_bfdot_max > 0 && neon.tail_mac != crate::jit::emit::NeonMac::BfDot { 2 } else { 1 }
    }

    /// The mac of the tail kernel of residue `r`: the second set (`bfdot`)
    /// up to `tail_bfdot_max`, the first (`tail_mac`) above.
    pub fn tail_set_of(&self, r: usize) -> usize {
        usize::from(self.tail_macs() == 2 && r <= usize::from(self.opts.neon.tail_bfdot_max))
    }

    /// The kinds of the column plan: `(width in columns, valid columns of the
    /// last group of sixteen)`.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn kinds(&self) -> &[(usize, usize)] {
        &self.kinds
    }

    /// Owned relocation input for the whole-call emitter: per kind, the
    /// height kernels then the tail kernels, in emission order.
    pub(crate) fn lowering_parts(&self) -> (AFormat, &[(usize, usize)], Vec<Vec<u8>>) {
        let blobs = self
            .kernels
            .iter()
            .zip(&self.tails)
            .flat_map(|(heights, tails)| heights.iter().chain(tails))
            .zip(&self.kernel_lengths)
            .map(|(&function, &length)| unsafe {
                std::slice::from_raw_parts(function as *const u8, length).to_vec()
            })
            .collect();
        (self.a_fmt, &self.n_tiles, blobs)
    }

    /// Compiled shape.
    pub fn shape(&self) -> GemmShape {
        self.shape
    }

    /// Requested tile (the effective width may be smaller on the
    /// last column tile).
    pub fn tile(&self) -> TileShape {
        self.tile
    }

    /// Generation options.
    pub fn options(&self) -> KernelOptions {
        self.opts
    }

    /// Machine code bytes produced, all kernels combined.
    pub fn code_bytes(&self) -> usize {
        self.code_bytes
    }

    /// Number of distinct kernels generated.
    pub fn kernel_count(&self) -> usize {
        self.kinds.len() * self.tile.mr
    }

    /// Column plan widths, in columns.
    pub fn widths(&self) -> Vec<usize> {
        self.kinds.iter().map(|&(w, _)| w).collect()
    }

    /// Number of column tiles.
    pub fn n_tile_count(&self) -> usize {
        self.n_tiles.len()
    }

    /// Number of row tiles for `m` rows.
    pub fn m_tile_count(&self, m: usize) -> usize {
        m.div_ceil(self.tile.mr)
    }

    /// Is `C` written in bf16? See [`MicroKernelSpec::out_bf16`].
    pub fn out_bf16(&self) -> bool {
        self.out_bf16
    }

    /// Size of a `C` element, in bytes: 2 in bf16, 4 in f32.
    pub fn c_elem_bytes(&self) -> usize {
        if self.out_bf16 {
            2
        } else {
            4
        }
    }

    /// Total number of kernel calls for `m` rows.
    pub fn total_items(&self, m: usize) -> usize {
        self.n_tiles.len() * self.m_tile_count(m)
    }
}

/// One micro-kernel. On x86-64, two register budgets, two checks: `spec.check`
/// carries the AVX-512 one (32 zmm), `check_avx2` the AVX2 one (16 ymm).
/// Applying the first on a machine without AVX-512 would let through a tile
/// that the emitter would then refuse — a late refusal, and under the wrong
/// name. On AArch64 and RISC-V there is one arm.
#[cfg(target_arch = "x86_64")]
fn emit_micro(spec: &MicroKernelSpec, avx512: bool) -> Result<Vec<u8>, CompileError> {
    if avx512 {
        spec.check()?;
        emit(spec).map_err(CompileError::Failed)
    } else {
        crate::jit::emit_gemm_avx2::check_avx2(spec)?;
        crate::jit::emit_gemm_avx2::emit_gemm_avx2(spec).map_err(CompileError::Failed)
    }
}
#[cfg(any(target_arch = "aarch64", target_arch = "riscv64"))]
fn emit_micro(spec: &MicroKernelSpec, _avx512: bool) -> Result<Vec<u8>, CompileError> {
    spec.check()?;
    emit(spec).map_err(CompileError::Failed)
}

/// Compile `shape` for the `tile` tile.
pub fn compile(
    shape: GemmShape,
    tile: TileShape,
    opts: KernelOptions,
    out_bf16: bool,
    a_fmt: AFormat,
) -> Result<CompiledGemm, CompileError> {
    compile_for_isa(
        shape,
        tile,
        opts,
        out_bf16,
        a_fmt,
        crate::jit::gemm::is_supported(),
    )
}

// The ISA is explicit: the AVX2 arm can be compiled on an AVX-512 machine.
pub(crate) fn compile_for_isa(
    shape: GemmShape,
    tile: TileShape,
    opts: KernelOptions,
    out_bf16: bool,
    a_fmt: AFormat,
    avx512: bool,
) -> Result<CompiledGemm, CompileError> {
    shape.check()?;
    crate::jit::ir::tile_fits(tile)?;

    // ── Column plan ───────────────────────────────────────────────────
    //
    // The plan covers `n_padded`, not `N`: the final R16 block exists even when
    // `N` is not a multiple of 16 (the packer zero-pads it). These are the
    // **stores** that are restricted, by `n_tail`.
    let n_padded = shape.n_blocks() * ROWS_INTERLEAVED;
    let mut kinds: Vec<(usize, usize)> = Vec::new();
    let mut n_tiles: Vec<(usize, usize)> = Vec::new();
    let mut off = 0usize;
    while off < n_padded {
        let remaining = n_padded - off;
        let w = if remaining >= tile.nr {
            tile.nr
        } else {
            widest_tile(tile.mr, remaining).ok_or(Decline::TileTooWide)?
        };
        // Valid columns of the last group of 16 of this tile. Only the
        // very last tile can exceed `N`, and by at most 15 columns.
        let n_tail = if off + w > shape.n {
            shape.n - (off + w - ROWS_INTERLEAVED)
        } else {
            ROWS_INTERLEAVED
        };
        let idx = match kinds.iter().position(|&x| x == (w, n_tail)) {
            Some(i) => i,
            None => {
                kinds.push((w, n_tail));
                kinds.len() - 1
            }
        };
        n_tiles.push((off, idx));
        off += w;
    }

    // ── Emission ────────────────────────────────────────────────────────
    let k_pairs = shape.k_pairs();
    // The matrix path may add, per kind, the vector kernels of the residue
    // rows: sixteen columns each, called once per group of the tile; a kind
    // whose last group is partial gets a second set storing that group.
    let tail_rows = if opts.matrix { usize::from(opts.neon.tail_rows).min(MAX_TAIL_ROWS) } else { 0 };
    let tail_opts = |mac: crate::jit::emit::NeonMac| KernelOptions {
        matrix: false,
        neon: crate::jit::emit::NeonOptions { a_packed: true, tail_rows: 0, nr: 16, mac, ..opts.neon },
        ..opts
    };
    // The sets of tail kernels: `tail_mac`, then `bfdot` for the small residues.
    let mut tail_macs = vec![opts.neon.tail_mac];
    if tail_rows > 0 && opts.neon.tail_bfdot_max > 0 && opts.neon.tail_mac != crate::jit::emit::NeonMac::BfDot {
        tail_macs.push(crate::jit::emit::NeonMac::BfDot);
    }
    let mut blobs: Vec<Vec<u8>> = Vec::with_capacity(kinds.len() * (tile.mr + 2 * tail_rows));
    let mut tails_per_kind = Vec::with_capacity(kinds.len());
    for &(w, n_tail) in &kinds {
        let spec_of = |height: usize, nr: usize, n_tail: usize, opts: KernelOptions| MicroKernelSpec {
            mr: height,
            nr,
            k_pairs,
            k_odd: !shape.k.is_multiple_of(2),
            lda_bytes: shape.k * a_fmt.elem_bytes(),
            ldc_bytes: shape.n * if out_bf16 { 2 } else { 4 },
            n_tail,
            out_bf16,
            a_fmt,
            opts,
            k_dynamic: false,
        };
        for height in 1..=tile.mr {
            blobs.push(emit_micro(&spec_of(height, w, n_tail, opts), avx512)?);
        }
        let mut count = 0;
        for &mac in &tail_macs {
            for r in 1..=tail_rows {
                blobs.push(emit_micro(&spec_of(r, ROWS_INTERLEAVED, ROWS_INTERLEAVED, tail_opts(mac)), avx512)?);
                count += 1;
            }
            if n_tail < ROWS_INTERLEAVED {
                for r in 1..=tail_rows {
                    blobs.push(emit_micro(&spec_of(r, ROWS_INTERLEAVED, n_tail, tail_opts(mac)), avx512)?);
                    count += 1;
                }
            }
        }
        tails_per_kind.push(count);
    }

    // ── A single executable page for all kernels ──────────────────
    let mut offsets = Vec::with_capacity(blobs.len());
    let mut total = 0usize;
    for blob in &blobs {
        offsets.push(total);
        total += blob.len().div_ceil(KERNEL_ALIGN) * KERNEL_ALIGN;
    }
    let exec = crate::jit::target::map_exec(total.max(KERNEL_ALIGN), |map| {
        for (blob, &at) in blobs.iter().zip(offsets.iter()) {
            map[at..at + blob.len()].copy_from_slice(blob);
        }
    })
    .map_err(|e| CompileError::Failed(HerbertError::Backend(format!("executable mapping: {e}"))))?;

    let base = exec.as_ptr();
    let mut kernels: Vec<Vec<KernelFn>> = Vec::with_capacity(kinds.len());
    let mut tails: Vec<Vec<KernelFn>> = Vec::with_capacity(kinds.len());
    let mut cursor = 0usize;
    for &tails_of_kind in &tails_per_kind {
        let mut by_height = Vec::with_capacity(tile.mr);
        let mut by_residue = Vec::with_capacity(tails_of_kind);
        for i in 0..tile.mr + tails_of_kind {
            let at = offsets[cursor];
            // SAFETY: `at` points to the start of a kernel written above in a
            // page made executable, whose `exec` guard preserves the property.
            let f: KernelFn = unsafe { std::mem::transmute::<*const u8, KernelFn>(base.add(at)) };
            if i < tile.mr { by_height.push(f) } else { by_residue.push(f) };
            cursor += 1;
        }
        kernels.push(by_height);
        tails.push(by_residue);
    }

    let code_bytes = blobs.iter().map(|b| b.len()).sum();
    Ok(CompiledGemm {
        shape,
        tile,
        opts,
        out_bf16,
        a_fmt,
        _exec: exec,
        kernels,
        tails,
        kernel_lengths: blobs.iter().map(Vec::len).collect(),
        kinds,
        n_tiles,
        code_bytes,
    })
}
