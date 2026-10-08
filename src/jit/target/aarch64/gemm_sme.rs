//! The bf16 GEMM micro-kernel on the matrix engine: SME2 `bfmopa` into ZA.
//!
//! `bfmopa za<t>.s, p/m, p/m, zn.h, zm.h` adds to every f32 element `(i, j)`
//! of a 16×16 tile (streaming vector length 64 bytes) the two products of a
//! pair of `K`: `zn.h[2i]·zm.h[2j] + zn.h[2i+1]·zm.h[2j+1]`. A pair of `K`
//! for sixteen columns of `W` is 64 contiguous bytes of the R16 layout: one
//! `ld1h` is the `zm` of one tile. The same pair for sixteen rows of `A` is
//! one `ld1h` of the packed activations (`jit/emit.rs`, [`packed_a_bytes`]):
//! the packing node of the plan has put `A` in that layout once for all the
//! projections that read it.
//!
//! A call computes up to 16 rows × 64 columns: the four 32-bit tiles of ZA
//! accumulate four blocks of sixteen columns, each pair of `K` costs one load
//! of `A`, one load per block and one outer product per block. The four tiles
//! are independent, which is what the matrix engine needs to overlap its
//! products. The epilogue reads each row of each tile — `st1w` straight from
//! ZA for an f32 output; for a bf16 output `mova` to a vector, `bfcvt` (round
//! to nearest even) and the truncating `st1h`. Only the `height` valid rows and
//! the `n_tail` valid columns of the last block are stored.
//!
//! `smstart` and `smstop` clear the vector registers: the prologue saves the
//! low halves of `v8..v15`, which belong to the caller.
//!
//! [`packed_a_bytes`]: crate::jit::emit::packed_a_bytes

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec, PACK_ROWS};
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use crate::jit::target::aarch64::util::{add_imm, backend_err, Frame};
use a64::*;

/// The widest tile: sixteen rows, the four ZA tiles of sixteen columns.
pub const SME_MAX_COLS: usize = 64;

/// Is the matrix engine usable here: SME2 with the bf16 outer product.
pub fn available() -> bool {
    let f = crate::core::isa::feature;
    f("sme2") && f("sme_b16f32") && crate::core::isa::detected().as_ref().is_ok_and(|d| d.sme_max_svl_bytes == 64)
}

/// The shapes this kernel serves.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if !available() {
        return Err(Decline::TileTooWide);
    }
    if spec.mr == 0 || spec.mr > PACK_ROWS || spec.nr == 0 || spec.nr > SME_MAX_COLS || !spec.nr.is_multiple_of(ROWS_INTERLEAVED) {
        return Err(Decline::TileTooWide);
    }
    // A packed block holds whole pairs; the loop takes them four at a time.
    if spec.k_odd || spec.a_fmt != AFormat::Bf16 || !spec.k_pairs.is_multiple_of(4) {
        return Err(Decline::OddK);
    }
    Ok(())
}

/// Generates the kernel described by `spec`, `A` packed.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit sme gemm: {d}")))?;
    let nb = spec.nr / ROWS_INTERLEAVED;
    let bstride = b_block_bytes(spec.k_pairs) as i64;
    let block = |j: usize| if j == 0 { X1 } else { x((2 + j) as u8) };
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride, X16);
    }
    a.smstart();
    a.zero_za();
    a.ptrue(p(0), Et::H);
    a.ptrue(p(2), Et::S);
    // Four pairs per turn: `A` in z0..z3, the blocks in z4.., one vector each.
    // Under `k_dynamic` the caller has put the groups in `x15`.
    let top = a.label();
    if !spec.k_dynamic {
        a.mov_imm(X15, (spec.k_pairs / 4) as u64);
    }
    a.bind(top);
    for u in 0..4i64 {
        a.ld1_z(z(u as u8), Et::H, p(0), mem(X0, u));
    }
    for j in 0..nb {
        for u in 0..4i64 {
            a.ld1_z(z(4 + (4 * j + u as usize) as u8), Et::H, p(0), mem(block(j), u));
        }
    }
    for u in 0..4u8 {
        for j in 0..nb {
            a.bfmopa(j as u8, p(0), p(0), z(u), z(4 + 4 * j as u8 + u));
        }
    }
    a.addvl(X0, X0, 4);
    for j in 0..nb {
        a.addvl(block(j), block(j), 4);
    }
    a.subs(X15, X15, 1);
    a.bne(top);

    // ── Epilogue: row by row, tile by tile. ──────────────────────────────
    // x16 = the C row, x17 = the column index of the block; w12 + i = the row.
    let masked = spec.n_tail < ROWS_INTERLEAVED;
    if masked {
        a.mov_imm(X17, spec.n_tail as u64);
        a.whilelo(p(1), Et::S, XZR, X17);
    }
    a.mov(X16, X2);
    for group in 0..spec.mr.div_ceil(4) {
        a.mov_imm(W12, (4 * group) as u64);
        for off in 0..4u8 {
            let row = 4 * group + usize::from(off);
            if row >= spec.mr {
                break;
            }
            for j in 0..nb {
                let pred = if masked && j == nb - 1 { p(1) } else { p(2) };
                let slice = ZaS { tile: j as u8, dir: Slice::H, index: W12, offset: off };
                a.mov_imm(X17, (j * 16) as u64);
                if spec.out_bf16 {
                    a.mova_from_za(z(0), p(2), slice);
                    a.bfcvt_z(z(0), p(2), z(0));
                    a.st1h_s(z(0), pred, idx_scaled(X16, X17));
                } else {
                    a.st1w_za(slice, pred, X16, X17);
                }
            }
            add_imm(&mut a, X16, X16, spec.ldc_bytes as i64, X17);
        }
    }
    a.smstop();
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("sme gemm {}x{} (k_pairs={})", spec.mr, spec.nr, spec.k_pairs)))
}
