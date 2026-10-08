//! The bf16 GEMM micro-kernel, RVV: `vfwmaccbf16.vf` on R16 weights.
//!
//! A kernel call computes an `mr × nr` tile of `C` (the specification is
//! `jit/emit.rs`). In the R16 layout one pair of `K` for sixteen rows of `W`
//! is 64 contiguous bytes, the two elements of each row side by side: one
//! `vlseg2e16` at `vl = 16` splits them into two registers — the first
//! elements of the sixteen rows, then the second ones. `A` arrives as
//! scalars (`flh` of one bf16, NaN-boxed), and each `vfwmaccbf16.vf` adds
//! one element of `A` times sixteen of `W` into sixteen f32 accumulators.
//!
//! ## The form `seg` (the first written, and the default)
//!
//! The other form, `wide` (`--rvv-gemm wide`), runs the whole vector length
//! and folds the lanes at the end of the tile: `gemm_wide.rs`.
//!
//! Every vector instruction runs at `vl = 16`: sixteen bf16 in a register
//! group of one (`e16, m1`), sixteen f32 in a group of two (`e32, m2`). It
//! needs VLEN ≥ 256 (`vlenb ≥ 32`, part of the engine's floor on RISC-V) and
//! uses no more lanes above it: the bits never depend on VLEN, and neither do
//! the counts. Each output is the sequential chain of fused multiply-adds
//! over `K`, pair by pair, the first element of a pair before the second —
//! with the measured semantics of `vfwmaccbf16` (one IEEE rounding under
//! `frm`), exactly `f32::mul_add` of the widened operands.
//!
//! ## Registers
//!
//! `v0` and `v1` stay free; the pair of block `j` lands in `v(2 + 2j)`,
//! `v(3 + 2j)`; the accumulators follow, two registers each, on an even
//! register: `2 + 2·nb + 2·mr·nb ≤ 32`. `A` takes `2·mr` floating-point
//! registers among the caller-saved ones; no frame is needed. The integer
//! registers are `a0..a2` (row 0 of `A`, block 0 of `W`, `C`), then one
//! pointer per other row and block, a loop counter and a scratch among the
//! twelve other caller-saved ones: `mr + nb ≤ 12`.
//!
//! ## Edges: `K` odd and `N % 16 ≠ 0`
//!
//! The last pair of an odd `K` loads one element of `A` and adds only its
//! product: no read beyond the row. The last group of 16 columns stores its
//! `n_tail` valid columns, at `vl = n_tail`.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec, RvvGemm};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use crate::jit::target::riscv64::util::{add_imm, backend_err};
use rv::*;

/// The production tile: four rows, two blocks.
pub const RV_TILE: TileShape = TileShape { mr: 4, nr: 32 };

/// Vector registers of a tile: `v0`, `v1`, two per block of `W`, two per accumulator.
pub const fn regs_needed(mr: usize, nr: usize) -> usize {
    let nb = nr / ROWS_INTERLEAVED;
    2 + 2 * nb + 2 * mr * nb
}

/// The integer registers after `a0..a2`: the other rows and blocks, the counter, a scratch.
pub(super) const FREE: [X; 12] = [A3, A4, A5, A6, A7, T0, T1, T2, T3, T4, T5, T6];

/// The floating-point registers `A` may use: every caller-saved one.
const A_REGS: [F; 20] = [FT0, FT1, FT2, FT3, FT4, FT5, FT6, FT7, FT8, FT9, FT10, FT11, FA0, FA1, FA2, FA3, FA4, FA5, FA6, FA7];

/// Does the tile fit the registers of the micro-kernel?
pub fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
    let nb = tile.nr / ROWS_INTERLEAVED;
    if tile.mr == 0 || tile.nr == 0 || !tile.nr.is_multiple_of(ROWS_INTERLEAVED) {
        return Err(Decline::TileTooWide);
    }
    // Vector registers; one more `f` register than the rows need, for the f32 conversion;
    // the pointers of the rows and the blocks, the counter and a scratch.
    if regs_needed(tile.mr, tile.nr) > 32 || 2 * tile.mr + 1 > A_REGS.len() || tile.mr + nb > FREE.len() {
        return Err(Decline::TileTooWide);
    }
    Ok(())
}

/// The widest legal width ≤ `max_cols` at `mr` rows.
pub fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
    (1..=max_cols / ROWS_INTERLEAVED).rev().map(|g| g * ROWS_INTERLEAVED).find(|&nr| tile_fits(TileShape { mr, nr }).is_ok())
}

/// The register and addressing budget of one micro-kernel, and the forms
/// this target does not emit.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    let tile = TileShape { mr: spec.mr, nr: spec.nr };
    match spec.opts.rvv.gemm {
        RvvGemm::Seg => tile_fits(tile)?,
        RvvGemm::Wide => super::gemm_wide::tile_fits(tile, spec.opts.rvv.width, crate::core::isa::vlenb())?,
    }
    // No matrix engine, no packed `A`, no length of `K` given at the call.
    if spec.opts.matrix || spec.opts.neon.a_packed || spec.k_dynamic {
        return Err(Decline::TileTooWide);
    }
    Ok(())
}

/// Generates the kernel described by `spec`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit riscv64: {d}")))?;
    if spec.opts.rvv.gemm == RvvGemm::Wide {
        return super::gemm_wide::emit(spec);
    }
    let mr = spec.mr;
    let nb = spec.nr / ROWS_INTERLEAVED;
    let e16 = Vtype::new(Sew::E16, Lmul::M1);
    let e32 = Vtype::new(Sew::E32, Lmul::M2);
    let wreg = |j: usize| v((2 + 2 * j) as u8);
    let acc0 = 2 + 2 * nb;
    let acc = |i: usize, j: usize| v((acc0 + 2 * (i * nb + j)) as u8);
    let af = |i: usize, e: usize| A_REGS[2 * i + e];
    let f32_tmp = A_REGS[2 * mr];

    // a0 = A row 0, a1 = W block 0, a2 = C; the other rows and blocks after them.
    let row = |i: usize| if i == 0 { A0 } else { FREE[i - 1] };
    let block = |j: usize| if j == 0 { A1 } else { FREE[mr - 1 + j - 1] };
    let counter = FREE[mr + nb - 2];
    let scratch = FREE[mr + nb - 1];

    let mut a = Asm::new();
    let bstride = b_block_bytes(spec.k_pairs) as i64;
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, scratch);
    }
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride, scratch);
    }
    a.vsetivli(ZERO, ROWS_INTERLEAVED as u32, e32);
    for i in 0..mr {
        for j in 0..nb {
            a.vmv_v_i(acc(i, j), 0);
        }
    }
    a.vsetivli(ZERO, ROWS_INTERLEAVED as u32, e16);

    let pair_bytes = spec.a_fmt.pair_bytes() as i64;
    // Element `e` (0 or 1) of the pair at byte `off` of every row, as a bf16 scalar.
    let load_a = |a: &mut Asm, off: i64, elems: usize| {
        for i in 0..mr {
            for e in 0..elems {
                let at = off + e as i64 * spec.a_fmt.elem_bytes() as i64;
                match spec.a_fmt {
                    AFormat::Bf16 => a.flh(af(i, e), row(i), at),
                    // Truncation: the high half of the f32, two bytes up.
                    AFormat::F32Trunc => a.flh(af(i, e), row(i), at + 2),
                    AFormat::F32Rne => {
                        a.flw(f32_tmp, row(i), at);
                        a.fcvt_bf16_s(af(i, e), f32_tmp, Rm::Rne);
                    }
                }
            }
        }
    };
    // One pair of every block: its load, then its products, first element first.
    let pair = |a: &mut Asm, elems: usize| {
        for j in 0..nb {
            a.vlseg(2, Sew::E16, wreg(j), block(j));
            a.addi(block(j), block(j), 64);
            for i in 0..mr {
                for e in 0..elems {
                    a.vfwmaccbf16_vf(acc(i, j), af(i, e), wreg(j).plus(e as u8));
                }
            }
        }
    };

    let full_pairs = spec.k_pairs - usize::from(spec.k_odd);
    let groups = full_pairs / 4;
    if groups > 0 {
        a.li(counter, groups as i64);
        let top = a.here();
        for p in 0..4 {
            load_a(&mut a, p * pair_bytes, 2);
            pair(&mut a, 2);
        }
        for i in 0..mr {
            a.addi(row(i), row(i), 4 * pair_bytes);
        }
        a.addi(counter, counter, -1);
        a.bnez(counter, top);
    }
    for p in 0..(full_pairs % 4) as i64 {
        load_a(&mut a, p * pair_bytes, 2);
        pair(&mut a, 2);
    }
    if spec.k_odd {
        load_a(&mut a, (full_pairs % 4) as i64 * pair_bytes, 1);
        pair(&mut a, 1);
    }

    // ── Epilogue: C rows, full groups of 16 columns, then the partial one. ──
    let elem = if spec.out_bf16 { 2 } else { 4 };
    for i in 0..mr {
        for j in 0..nb {
            let n = if j == nb - 1 { spec.n_tail } else { ROWS_INTERLEAVED };
            add_imm(&mut a, scratch, A2, (i * spec.ldc_bytes + j * ROWS_INTERLEAVED * elem) as i64, counter);
            if spec.out_bf16 {
                a.vsetivli(ZERO, n as u32, e16);
                a.vfncvtbf16_f_f_w(wreg(0), acc(i, j));
                a.vse(Sew::E16, wreg(0), scratch);
            } else {
                a.vsetivli(ZERO, n as u32, e32);
                a.vse(Sew::E32, acc(i, j), scratch);
            }
        }
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("gemm {}x{} (k_pairs={})", spec.mr, spec.nr, spec.k_pairs)))
}
