//! The bf16 GEMM micro-kernel, RVV, form `wide` (`--rvv-gemm wide`): the
//! whole vector length on the R16 weights.
//!
//! In the R16 layout the pairs of `K` of one block of sixteen rows follow
//! each other, 64 bytes apiece: one `vle16` at `vl = 32·G` loads `G`
//! consecutive pairs of the sixteen rows. Lane `t` holds element `t % 2` of
//! pair `t / 32` of row `(t / 2) % 16`. `A` is spread to the same shape once
//! per chunk and row — its `2·G` elements loaded, then `vrgather` puts
//! element `2·(t / 32) + t % 2` in lane `t` — and one `vfwmaccbf16.vv` adds
//! `32·G` products into `32·G` f32 accumulators. At the end of the tile the
//! accumulator is folded in halves down to 32 lanes (`vslidedown` +
//! `vfadd`), then each row adds its two lanes, even plus odd (`vnsrl` of the
//! 64-bit view, then `vfadd`).
//!
//! ## The width
//!
//! `L = 32·G` lanes, from `--rvv-width` ([`lanes`]): the knob, at most the
//! lanes of the smallest register group that holds 32 (`VLEN / 16` from
//! VLEN 512, 32 at VLEN 256), rounded down to a multiple of 32 and never
//! fewer than 32. Its bits depend on `L`, not on VLEN: `--rvv-width 64` gives
//! the same bits on every machine that reaches 64 lanes.
//!
//! ## The order of the sums (what the transcription follows)
//!
//! Lane `(g, r, e)` chains, with the measured multiply-add of
//! `vfwmaccbf16` (one IEEE rounding under `frm`), the products of pairs
//! `g, g + G, g + 2G, …` of row `r`, element `e`, from +0. The fold adds lane
//! `g + G/2` to lane `g`, then `g + G/4`, … down to `g = 0`; the output is
//! `(e = 0) + (e = 1)`. An odd `K` masks the missing element of its last pair
//! (`v0`: every lane but the odd ones of that pair), so no product is added
//! there; a short last chunk runs at a shorter `vl`, tail undisturbed.
//!
//! ## Registers
//!
//! `m` registers per bf16 group of `L` lanes (1 from VLEN 512, 2 at 256), the
//! accumulators `2m` each: `v0` (the mask), the gather indices, the raw
//! chunk of `A` of each row, its spread form, one group per block of `W`,
//! then the accumulators on a multiple of `2m`. The integer registers are
//! those of the form `seg`, plus two vector lengths: `mr + nb ≤ 10`.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec, RvvWidth};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use crate::jit::target::riscv64::gemm::FREE;
use crate::jit::target::riscv64::util::{add_imm, backend_err};
use rv::*;

/// The accumulator lanes `L` of the form on a machine of `vlenb` bytes per register.
pub fn lanes(width: RvvWidth, vlenb: usize) -> usize {
    let vlen = vlenb * 8;
    let vlmax = vlen / 16 * (512 / vlen).max(1);
    (width.lanes(vlmax) / 32 * 32).max(32)
}

/// The register groups of one tile, in register numbers.
#[derive(Clone, Copy, Debug)]
struct Layout {
    /// Registers of a bf16 group of `L` lanes.
    m: usize,
    mr: usize,
    nb: usize,
    acc0: usize,
}

impl Layout {
    fn new(mr: usize, nb: usize, lanes: usize, vlenb: usize) -> Option<Layout> {
        let m = (lanes * 2).div_ceil(vlenb).next_power_of_two();
        let end = (2 + 2 * mr + nb) * m;
        let acc0 = end.next_multiple_of(2 * m);
        let fits = m <= 4 && mr > 0 && nb > 0 && acc0 + 2 * m * mr * nb <= 32 && mr + nb <= FREE.len() - 2;
        fits.then_some(Layout { m, mr, nb, acc0 })
    }
    fn idx(&self) -> V {
        v(self.m as u8)
    }
    fn raw(&self, i: usize) -> V {
        v(((2 + i) * self.m) as u8)
    }
    fn spread(&self, i: usize) -> V {
        v(((2 + self.mr + i) * self.m) as u8)
    }
    fn w(&self, j: usize) -> V {
        v(((2 + 2 * self.mr + j) * self.m) as u8)
    }
    fn acc(&self, i: usize, j: usize) -> V {
        v((self.acc0 + 2 * self.m * (i * self.nb + j)) as u8)
    }
    /// A group of `2m` free after the K loop (the raw and spread chunks).
    fn tmp(&self) -> V {
        v((2 * self.m) as u8)
    }
    /// Two groups of at most `m` inside it: the even and odd lanes of a row.
    fn even(&self) -> V {
        v((2 * self.m) as u8)
    }
    fn odd(&self) -> V {
        v((3 * self.m) as u8)
    }
}

/// Does the tile fit the registers of the form at this width and VLEN?
pub fn tile_fits(tile: TileShape, width: RvvWidth, vlenb: usize) -> std::result::Result<(), Decline> {
    if tile.mr == 0 || tile.nr == 0 || !tile.nr.is_multiple_of(ROWS_INTERLEAVED) || vlenb < 32 {
        return Err(Decline::TileTooWide);
    }
    Layout::new(tile.mr, tile.nr / ROWS_INTERLEAVED, lanes(width, vlenb), vlenb).map(|_| ()).ok_or(Decline::TileTooWide)
}

/// The tile the form runs for a wanted one: fewer rows, then fewer blocks,
/// until it fits.
pub fn tile(want: TileShape, width: RvvWidth, vlenb: usize) -> TileShape {
    let nb = (want.nr / ROWS_INTERLEAVED).max(1);
    for b in (1..=nb).rev() {
        for mr in (1..=want.mr.max(1)).rev() {
            let t = TileShape { mr, nr: b * ROWS_INTERLEAVED };
            if tile_fits(t, width, vlenb).is_ok() {
                return t;
            }
        }
    }
    TileShape { mr: 1, nr: ROWS_INTERLEAVED }
}

/// The smallest legal group holding `bits` at `sew` (ELEN 64: SEW ≤ 64·LMUL).
fn group(bits: usize, vlen: usize, sew: Sew) -> Lmul {
    let min = match sew {
        Sew::E8 => 8,
        Sew::E16 => 4,
        Sew::E32 => 2,
        Sew::E64 => 1,
    };
    if bits >= vlen {
        return regs(bits.div_ceil(vlen).next_power_of_two());
    }
    match (vlen / bits).min(min) {
        1 => Lmul::M1,
        2 => Lmul::Mf2,
        4 => Lmul::Mf4,
        _ => Lmul::Mf8,
    }
}

fn regs(n: usize) -> Lmul {
    match n {
        1 => Lmul::M1,
        2 => Lmul::M2,
        4 => Lmul::M4,
        _ => Lmul::M8,
    }
}

/// Generates the kernel described by `spec` (`spec.check()` done by the caller).
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    let vlenb = crate::core::isa::vlenb();
    let vlen = vlenb * 8;
    let lanes = lanes(spec.opts.rvv.width, vlenb);
    let (mr, nb) = (spec.mr, spec.nr / ROWS_INTERLEAVED);
    let l = Layout::new(mr, nb, lanes, vlenb)
        .ok_or_else(|| HerbertError::Backend(format!("backend_jit riscv64: wide tile {mr}x{} at {lanes} lanes, VLEN {vlen}", spec.nr)))?;
    let g = lanes / 32;
    let m = regs(l.m);
    // The products and the gathers: bf16 in `m`, f32 accumulators in `2m`, tail undisturbed.
    let e16 = Vtype::new(Sew::E16, m).tu();
    let acc_t = Vtype::new(Sew::E32, regs(2 * l.m));
    // One row of sixteen results: f32 (from the 64-bit view of 32 lanes), then bf16.
    let row_f32 = Vtype::new(Sew::E32, group(16 * 32, vlen, Sew::E32));
    let row_bf16 = Vtype::new(Sew::E16, group(16 * 16, vlen, Sew::E16));

    let row = |i: usize| if i == 0 { A0 } else { FREE[i - 1] };
    let block = |j: usize| if j == 0 { A1 } else { FREE[mr - 1 + j - 1] };
    let counter = FREE[mr + nb - 2];
    let scratch = FREE[mr + nb - 1];
    let avl_a = FREE[mr + nb];
    let avl_w = FREE[mr + nb + 1];

    let mut a = Asm::new();
    for i in 1..mr {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, scratch);
    }
    let bstride = b_block_bytes(spec.k_pairs) as i64;
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride, scratch);
    }
    a.li(scratch, lanes as i64);
    a.vsetvli(ZERO, scratch, acc_t);
    for i in 0..mr {
        for j in 0..nb {
            a.vmv_v_i(l.acc(i, j), 0);
        }
    }
    // idx[t] = 2·(t / 32) + t % 2, at e16 over the L lanes.
    a.vsetvli(ZERO, scratch, e16);
    a.vid_v(l.idx());
    a.vand_vi(l.w(0), l.idx(), 1);
    a.vsrl_vi(l.idx(), l.idx(), 5);
    a.vadd_vv(l.idx(), l.idx(), l.idx());
    a.vadd_vv(l.idx(), l.idx(), l.w(0));

    let elem = spec.a_fmt.elem_bytes() as i64;
    // One chunk: the `avl_a` elements of `A` of every row, spread over the
    // `avl_w` lanes, then every block (a masked multiply-add when `odd`).
    let chunk = |a: &mut Asm, odd: Option<usize>| {
        match spec.a_fmt {
            AFormat::Bf16 => {
                a.vsetvli(ZERO, avl_a, Vtype::new(Sew::E16, m));
                for i in 0..mr {
                    a.vle(Sew::E16, l.raw(i), row(i));
                }
            }
            AFormat::F32Trunc | AFormat::F32Rne => {
                a.vsetvli(ZERO, avl_a, Vtype::new(Sew::E32, Lmul::M1));
                for i in 0..mr {
                    a.vle(Sew::E32, l.spread(i), row(i));
                }
                a.vsetvli(ZERO, avl_a, Vtype::new(Sew::E16, Lmul::Mf2));
                for i in 0..mr {
                    if spec.a_fmt == AFormat::F32Trunc {
                        // The high half of each f32.
                        a.vnsrl_wi(l.raw(i), l.spread(i), 16);
                    } else {
                        a.vfncvtbf16_f_f_w(l.raw(i), l.spread(i));
                    }
                }
            }
        }
        a.vsetvli(ZERO, avl_w, e16);
        for i in 0..mr {
            a.vrgather_vv(l.spread(i), l.raw(i), l.idx());
        }
        if let Some(pairs) = odd {
            // v0 = the lanes before the last pair, or even: not the missing element.
            a.vid_v(l.w(0));
            a.li(scratch, 32 * (pairs as i64 - 1));
            a.vmsltu_vx(l.idx(), l.w(0), scratch);
            a.vand_vi(l.w(0), l.w(0), 1);
            a.vmseq_vi(V0, l.w(0), 0);
            a.vmor_mm(V0, V0, l.idx());
            a.vsetvli(ZERO, avl_w, e16.mu());
        }
        for j in 0..nb {
            a.vle(Sew::E16, l.w(j), block(j));
            if odd.is_none() {
                add_imm(a, block(j), block(j), 64 * g as i64, scratch);
            }
            for i in 0..mr {
                if odd.is_some() {
                    a.masked(|a| a.vfwmaccbf16_vv(l.acc(i, j), l.spread(i), l.w(j)));
                } else {
                    a.vfwmaccbf16_vv(l.acc(i, j), l.spread(i), l.w(j));
                }
            }
        }
        if odd.is_none() {
            for i in 0..mr {
                add_imm(a, row(i), row(i), 2 * g as i64 * elem, scratch);
            }
        }
    };

    // The last chunk holds 1..=G pairs, the odd one if any; the others run in the loop.
    let last = spec.k_pairs - g * ((spec.k_pairs - 1) / g);
    let full = (spec.k_pairs - last) / g;
    if full > 0 {
        a.li(avl_a, 2 * g as i64);
        a.li(avl_w, 32 * g as i64);
        a.li(counter, full as i64);
        let top = a.here();
        chunk(&mut a, None);
        a.addi(counter, counter, -1);
        a.bnez(counter, top);
    }
    a.li(avl_a, (2 * last - usize::from(spec.k_odd)) as i64);
    a.li(avl_w, 32 * last as i64);
    chunk(&mut a, spec.k_odd.then_some(last));

    // ── Epilogue: fold, add the pairs, store. ──
    let out = if spec.out_bf16 { 2 } else { 4 };
    for i in 0..mr {
        for j in 0..nb {
            let acc = l.acc(i, j);
            let mut n = lanes;
            while n > 32 {
                n /= 2;
                a.li(scratch, n as i64);
                a.vsetvli(ZERO, scratch, acc_t);
                a.vslidedown_vx(l.tmp(), acc, scratch);
                a.vfadd_vv(acc, acc, l.tmp());
            }
            a.vsetivli(ZERO, ROWS_INTERLEAVED as u32, row_f32);
            a.li(scratch, 32);
            a.vnsrl_wi(l.even(), acc, 0);
            a.vnsrl_wx(l.odd(), acc, scratch);
            a.vfadd_vv(l.even(), l.even(), l.odd());
            let n = if j == nb - 1 { spec.n_tail } else { ROWS_INTERLEAVED };
            add_imm(&mut a, scratch, A2, (i * spec.ldc_bytes + j * ROWS_INTERLEAVED * out) as i64, counter);
            if spec.out_bf16 {
                a.vsetivli(ZERO, n as u32, row_bf16);
                a.vfncvtbf16_f_f_w(l.odd(), l.even());
                a.vse(Sew::E16, l.odd(), scratch);
            } else {
                a.vsetivli(ZERO, n as u32, row_f32);
                a.vse(Sew::E32, l.even(), scratch);
            }
        }
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("gemm wide {}x{} (k_pairs={}, {lanes} lanes)", spec.mr, spec.nr, spec.k_pairs)))
}
