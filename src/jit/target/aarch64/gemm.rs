//! The bf16 GEMM micro-kernel, NEON: `bfdot` by element on R16 weights.
//!
//! A kernel call computes an `mr × nr` tile of `C` (the specification is
//! `jit/emit.rs`). In the R16 layout one pair of `K` for sixteen rows of `W`
//! is 64 contiguous bytes: four `q` registers of four rows each. One `q` load
//! of `A` brings four consecutive pairs of a row; `bfdot` by element reads
//! pair `p` of it (`a.2h[p]`) against a `q` of `W`, and accumulates, in each
//! of its four f32 lanes, the two products of that pair — the broadcast of
//! the x86 kernel is the element index here, and costs nothing.
//!
//! ## Registers — computed, not hardcoded
//!
//! `mr · nr/4` accumulators (twice that with two sets), four or eight
//! registers of `W` (one pair of one or two blocks; a wider tile streams its
//! blocks through them), `mr` of `A`, and two conversion temporaries when
//! `A` is f32: at most 32. The allocation takes `v0..v7` and `v16..v31`
//! first, then `v8..v15`, whose low halves the prologue saves (AAPCS64).
//!
//! ## The variants (`NeonOptions`, each a measurement arm)
//!
//! A `prfm` a fixed distance ahead of every block, twice per 256 bytes; a
//! decode tile of 32, 48 or 64 columns; `bfmlalb`/`bfmlalt` by element in
//! place of `bfdot` (`A` then lives in `v0..v15`); two accumulator sets over
//! the even and the odd pairs of a group; `ld1` of four registers in place of
//! two `ldp`. The default is the kernel as first written.
//!
//! ## Addresses
//!
//! One pointer per row of `A` and one per R16 block of `W`, each advanced by
//! post-indexed loads: the kernel never forms a displacement that could
//! overflow, whatever `K`. `K` is an immediate: the loop runs `⌊pairs/4⌋`
//! groups of four pairs, the remainder pairs follow in a straight line. Under `k_dynamic`
//! the count of groups is the caller's, in `x15`.
//!
//! ## Edges: `K` odd and `N % 16 ≠ 0`
//!
//! The last pair of an odd `K` loads one element of `A` (`ldr h`, the upper
//! half zeroed): no read beyond the row, no dependence on the padding of `W`.
//! The last group of 16 columns stores only its `n_tail` valid columns.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec, NeonLoad, NeonMac, NeonOptions};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use crate::jit::target::aarch64::util::{add_imm, backend_err, Frame};
use a64::*;

/// Registers of `W` held at once: one pair of one block (four `q`), or of two
/// blocks when the tile has several — a wider tile streams its blocks
/// through these two buffers.
const fn w_regs(nr: usize) -> usize {
    if nr / 4 > 8 { 8 } else { nr / 4 }
}

/// NEON registers of a tile under the default options: accumulators, the
/// buffers of `W`, one `q` of `A` per row.
pub const fn regs_needed(mr: usize, nr: usize) -> usize {
    mr * (nr / 4) + w_regs(nr) + mr
}

/// The default tile: sixteen accumulators, four registers of `W`, four of `A`.
pub const ARM_TILE: TileShape = TileShape { mr: 4, nr: 16 };

/// Does the tile fit the 32 NEON registers, with `nr` a multiple of 16?
pub fn tile_fits(tile: TileShape) -> std::result::Result<(), Decline> {
    if tile.mr == 0 || tile.nr == 0 || !tile.nr.is_multiple_of(ROWS_INTERLEAVED) {
        return Err(Decline::TileTooWide);
    }
    if regs_needed(tile.mr, tile.nr) > 32 {
        return Err(Decline::TileTooWide);
    }
    Ok(())
}

/// The widest legal width ≤ `max_cols` for `mr` rows.
pub fn widest_tile(mr: usize, max_cols: usize) -> Option<usize> {
    (1..=max_cols / ROWS_INTERLEAVED)
        .rev()
        .map(|nb| nb * ROWS_INTERLEAVED)
        .find(|&nr| regs_needed(mr, nr) <= 32)
}

/// The allocation of a kernel: the registers of `A`, of the accumulators
/// (`[row][set][quad]`), of the `W` buffers and of the f32 temporaries, in
/// the order [`ORDER`] — and whether `v8..v15` are touched.
struct Allocation {
    aq: Vec<u8>,
    acc: Vec<Vec<Vec<u8>>>,
    bq: Vec<u8>,
    temps: Vec<u8>,
    used: usize,
}

/// The allocation order: caller-saved first, then the callee-saved `v8..v15`.
const ORDER: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13,
    14, 15,
];

/// Allocates for `mr × nr` under `neon`; `None` when the budget is exceeded.
/// `bfmlal` by element reads `A` from `v0..v15`: `A` comes first then; the
/// buffers of `W` start on a multiple of four so that `ld1` of four
/// registers finds them consecutive.
fn allocate(mr: usize, nr: usize, neon: NeonOptions, f32a: bool) -> Option<Allocation> {
    let nq = nr / 4;
    let sets = usize::from(neon.acc_sets.max(1));
    let mut cursor = 0usize;
    let mut take = |n: usize, align4: bool| -> Option<Vec<u8>> {
        if align4 {
            cursor = cursor.next_multiple_of(4);
        }
        if cursor + n > 32 {
            return None;
        }
        let regs = ORDER[cursor..cursor + n].to_vec();
        cursor += n;
        Some(regs)
    };
    let take_acc = |take: &mut dyn FnMut(usize, bool) -> Option<Vec<u8>>| -> Option<Vec<Vec<Vec<u8>>>> {
        (0..mr).map(|_| (0..sets).map(|_| take(nq, false)).collect()).collect()
    };
    let temps_n = if f32a { 2 } else { 0 };
    // Packed `A`: one register per four rows holds one pair; row-major: one per row, four pairs.
    let a_regs = if neon.a_packed { mr.div_ceil(4) } else { mr };
    let (aq, acc, bq, temps) = match neon.mac {
        NeonMac::BfMlal => {
            let aq = take(a_regs, false)?;
            let bq = take(w_regs(nr), true)?;
            let acc = take_acc(&mut take)?;
            let temps = take(temps_n, false)?;
            (aq, acc, bq, temps)
        }
        NeonMac::BfDot => {
            let acc = take_acc(&mut take)?;
            let bq = take(w_regs(nr), true)?;
            let aq = take(a_regs, false)?;
            let temps = take(temps_n, false)?;
            (aq, acc, bq, temps)
        }
    };
    Some(Allocation { aq, acc, bq, temps, used: cursor })
}

/// The register budget of one kernel, its options included.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    let neon = spec.opts.neon;
    if neon.a_packed {
        // Packed `A` holds four rows per register: the row-major budget of
        // `tile_fits` does not apply, the allocation below is the budget.
        if spec.mr == 0 || spec.nr == 0 || !spec.nr.is_multiple_of(ROWS_INTERLEAVED) {
            return Err(Decline::TileTooWide);
        }
    } else {
        tile_fits(TileShape { mr: spec.mr, nr: spec.nr })?;
    }
    if !(1..=2).contains(&neon.acc_sets) {
        return Err(Decline::OutOfRegisters);
    }
    if neon.mac == NeonMac::BfMlal && spec.mr > 15 {
        return Err(Decline::OutOfRegisters);
    }
    if allocate(spec.mr, spec.nr, neon, spec.a_fmt.is_f32()).is_none() {
        return Err(Decline::OutOfRegisters);
    }
    // The packed layout holds whole groups of four pairs of bf16 (`gemm_sme::check`).
    if neon.a_packed && (spec.k_odd || !spec.k_pairs.is_multiple_of(4) || spec.a_fmt.is_f32() || spec.mr > 15) {
        return Err(Decline::OddK);
    }
    // One pointer per row (none with packed `A`) and per block, plus the
    // loop counter and the `x16`/`x17` scratch: x3..x15.
    let row_pointers = if neon.a_packed { 0 } else { spec.mr };
    if row_pointers + spec.nr / ROWS_INTERLEAVED > 14 {
        return Err(Decline::OutOfRegisters);
    }
    Ok(())
}

/// Generates the kernel described by `spec`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit: {d}")))?;
    let mr = spec.mr;
    let nq = spec.nr / 4;
    let nb = spec.nr / ROWS_INTERLEAVED;
    let f32a = spec.a_fmt.is_f32();
    let neon = spec.opts.neon;
    let sets = usize::from(neon.acc_sets.max(1));
    let Allocation { aq, acc, bq, temps, used } =
        allocate(mr, spec.nr, neon, f32a).ok_or_else(|| HerbertError::Backend("backend_jit: out of NEON registers".into()))?;
    let wbufs = bq.len() / 4;
    let save = used > 24;
    // The prefetch distance: a multiple of 8 the immediate form can address.
    let prefetch = (usize::from(neon.prefetch) / 8 * 8).min(32760);
    let prefetch_op = if neon.prefetch_l1 { Prefetch::PLDL1KEEP } else { Prefetch::PLDL2KEEP };

    // x0 = A row 0, x1 = W block 0, x2 = C; rows 1.. in x3.., blocks 1.. after them.
    let row = |i: usize| if i == 0 { X0 } else { x((2 + i) as u8) };
    let row_pointers = if neon.a_packed { 0 } else { mr };
    let block = move |j: usize| if j == 0 { X1 } else { x((2 + row_pointers + j) as u8) };
    let counter = X15;
    let bstride = b_block_bytes(spec.k_pairs);

    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], save, 0);
    for i in 1..row_pointers {
        add_imm(&mut a, row(i), row(i - 1), spec.lda_bytes as i64, X16);
    }
    for j in 1..nb {
        add_imm(&mut a, block(j), block(j - 1), bstride as i64, X16);
    }
    for regs in acc.iter().flatten() {
        for &r in regs {
            a.movi(v(r).d2(), 0);
        }
    }

    // Load `pairs` pairs (1, or 4 as a group, or the half pair) of every row of A.
    // Packed `A`: pair `p` of rows 4g..4g+3 is the `q` at `p·64 + 16·g`; the
    // group of four pairs is loaded pair by pair, just before its products.
    let load_a = |a: &mut Asm, pairs: usize, half: bool| {
        if neon.a_packed {
            return;
        }
        for i in 0..mr {
            let dst = aq[i];
            match (spec.a_fmt, pairs, half) {
                (AFormat::Bf16, 4, _) => a.ldr(q(dst), post(row(i), 16)),
                (AFormat::Bf16, 1, false) => a.ldr(s(dst), post(row(i), 4)),
                (AFormat::Bf16, _, true) => a.ldr(h(dst), at(row(i))),
                (fmt, n, half) => {
                    let (t0, t1) = (temps[0], temps[1]);
                    if n == 4 {
                        a.ldp(q(t0), q(t1), post(row(i), 32));
                    } else if half {
                        a.ldr(s(t0), at(row(i)));
                    } else {
                        a.ldr(d(t0), post(row(i), 8));
                    }
                    match fmt {
                        AFormat::F32Rne => {
                            a.bfcvtn(v(dst).h4(), v(t0).s4());
                            if n == 4 {
                                a.bfcvtn(v(dst).h8(), v(t1).s4());
                            }
                        }
                        _ => {
                            // Truncation: the high half of every f32.
                            let hi = if n == 4 { t1 } else { t0 };
                            a.uzp2(v(dst).h8(), v(t0).h8(), v(hi).h8());
                        }
                    }
                }
            }
        }
    };
    // One pair of block `j` of W into its buffer.
    let load_w = |a: &mut Asm, j: usize| {
        let base = 4 * (j % wbufs);
        match neon.load {
            NeonLoad::Ldp => {
                a.ldp(q(bq[base]), q(bq[base + 1]), post(block(j), 32));
                a.ldp(q(bq[base + 2]), q(bq[base + 3]), post(block(j), 32));
            }
            NeonLoad::Ld1x4 => a.ld1(v(bq[base]).b16(), 4, post(block(j), 64)),
        }
    };
    // The pair `p` of the group of every row of packed `A`: one `q` per four rows.
    let load_a_packed = |a: &mut Asm, p: u8| {
        for g in 0..aq.len() {
            a.ldr(q(aq[g]), mem(X0, i64::from(p) * 64 + 16 * g as i64));
        }
    };
    // The products of pair `p` (of the loaded group) of block `j` into `set`.
    let mac = |a: &mut Asm, j: usize, p: u8, set: usize| {
        let base = 4 * (j % wbufs);
        for i in 0..mr {
            // Row-major: register `i`, pair `p`; packed: register `i/4`, lane `i%4`.
            let (areg, lane) = if neon.a_packed { (aq[i / 4], (i % 4) as u8) } else { (aq[i], p) };
            for q4 in 0..4 {
                let (d, w) = (v(acc[i][set][4 * j + q4]).s4(), v(bq[base + q4]).h8());
                match neon.mac {
                    NeonMac::BfDot => a.bfdot(d, w, v(areg).h2(lane)),
                    NeonMac::BfMlal => {
                        a.bfmlalb(d, w, v(areg).h(2 * lane));
                        a.bfmlalt(d, w, v(areg).h(2 * lane + 1));
                    }
                }
            }
        }
    };
    // One pair for every block: the loads stream through the buffers, each
    // block's products issued once the next block's pair is loaded.
    let pair = |a: &mut Asm, p: u8, set: usize| {
        for j in 0..nb {
            load_w(a, j);
            if wbufs == 2 && j >= 1 {
                mac(a, j - 1, p, set);
            } else if wbufs == 1 {
                mac(a, j, p, set);
            }
        }
        if wbufs == 2 {
            mac(a, nb - 1, p, set);
        }
    };

    // With `k_dynamic` the caller has put the groups in the counter and
    // `k_pairs` (a multiple of four) is only the stride of `W`.
    let full_pairs = if spec.k_dynamic { 4 } else { spec.k_pairs - usize::from(spec.k_odd) };
    let groups = full_pairs / 4;
    let rem = full_pairs % 4;
    if groups > 0 {
        let top = a.label();
        if !spec.k_dynamic {
            a.mov_imm(counter, groups as u64);
        }
        a.bind(top);
        load_a(&mut a, 4, false);
        for p in 0..4u8 {
            if prefetch > 0 && p % 2 == 0 {
                // A 128-byte line ahead of every block, twice per 256 bytes.
                for j in 0..nb {
                    a.prfm(prefetch_op, mem(block(j), prefetch as i64));
                }
            }
            if neon.a_packed {
                load_a_packed(&mut a, p);
            }
            let set = if sets == 2 { usize::from(p % 2) } else { 0 };
            pair(&mut a, p, set);
        }
        if neon.a_packed {
            a.add(X0, X0, 256);
        }
        a.subs(counter, counter, 1);
        a.bne(top);
    }
    for _ in 0..rem {
        load_a(&mut a, 1, false);
        pair(&mut a, 0, 0);
    }
    if spec.k_odd {
        load_a(&mut a, 1, true);
        pair(&mut a, 0, 0);
    }
    if sets == 2 {
        for i in 0..mr {
            for jj in 0..nq {
                a.fadd(v(acc[i][0][jj]).s4(), v(acc[i][0][jj]).s4(), v(acc[i][1][jj]).s4());
            }
        }
    }

    // ── Epilogue: C rows, full groups of 16 columns, then the partial one. ──
    let masked = spec.n_tail < ROWS_INTERLEAVED;
    let t = bq[0];
    for i in 0..mr {
        let c = X16;
        if i == 0 {
            a.mov(c, X2);
        } else {
            add_imm(&mut a, c, X2, (i * spec.ldc_bytes) as i64, X17);
        }
        for j in 0..nb {
            let quads = &acc[i][0][4 * j..4 * j + 4];
            let partial = masked && j == nb - 1;
            if spec.out_bf16 {
                let at0 = (j * 32) as i64;
                if !partial {
                    for half in 0..2 {
                        a.bfcvtn(v(t).h4(), v(quads[2 * half]).s4());
                        a.bfcvtn(v(t).h8(), v(quads[2 * half + 1]).s4());
                        a.str(q(t), mem(c, at0 + 16 * half as i64));
                    }
                } else {
                    store_partial_bf16(&mut a, quads, t, c, at0, spec.n_tail);
                }
            } else {
                let at0 = (j * 64) as i64;
                if !partial {
                    a.stp(q(quads[0]), q(quads[1]), mem(c, at0));
                    a.stp(q(quads[2]), q(quads[3]), mem(c, at0 + 32));
                } else {
                    store_partial_f32(&mut a, quads, c, at0, spec.n_tail);
                }
            }
        }
    }
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("gemm {}x{} (k_pairs={})", spec.mr, spec.nr, spec.k_pairs)))
}

/// The first `n` f32 of four quads at `[c + offset]` (`x17` is scratch).
pub(crate) fn store_partial_f32(a: &mut Asm, quads: &[u8], c: R, offset: i64, n: usize) {
    let full = n / 4;
    for (k, &r) in quads.iter().enumerate().take(full) {
        a.str(q(r), mem(c, offset + 16 * k as i64));
    }
    let left = n % 4;
    if left == 0 {
        return;
    }
    let r = quads[full];
    let base = offset + 16 * full as i64;
    let mut lane = 0;
    if left >= 2 {
        a.str(d(r), mem(c, base));
        lane = 2;
    }
    if left % 2 == 1 {
        a.add(X17, c, (base + 4 * lane as i64) as u64);
        a.st1_lane(v(r).s(lane as u8), at(X17));
    }
}

/// The first `n` bf16 of four quads, converted, at `[c + offset]` (`x17` is scratch).
pub(crate) fn store_partial_bf16(a: &mut Asm, quads: &[u8], t: u8, c: R, offset: i64, n: usize) {
    let mut done = 0;
    for half in 0..2 {
        if done >= n {
            break;
        }
        a.bfcvtn(v(t).h4(), v(quads[2 * half]).s4());
        a.bfcvtn(v(t).h8(), v(quads[2 * half + 1]).s4());
        let here = (n - done).min(8);
        let base = offset + 2 * done as i64;
        if here == 8 {
            a.str(q(t), mem(c, base));
        } else {
            let mut lane = 0;
            if here >= 4 {
                a.str(d(t), mem(c, base));
                lane = 4;
            }
            if here - lane >= 2 {
                a.add(X17, c, (base + 2 * lane as i64) as u64);
                a.st1_lane(v(t).s((lane / 2) as u8), at(X17));
                lane += 2;
            }
            if here - lane == 1 {
                a.add(X17, c, (base + 2 * lane as i64) as u64);
                a.st1_lane(v(t).h(lane as u8), at(X17));
            }
        }
        done += here;
    }
}
