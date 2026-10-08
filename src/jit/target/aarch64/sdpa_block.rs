//! The prefill attention by blocks of sixteen queries, AArch64: two GEMMs on
//! the micro-kernels of the projections, the softmax of the decode kernel
//! between them.
//!
//! A call serves sixteen consecutive queries of one head, row `i` seeing
//! `past + i + 1` keys (`SdpaPrefillFn`, plus the R16 copies of the head's
//! cache in `x6` and `x7`). Its three phases:
//!
//! 1. **Scores** `S = Q·Kᵀ` in f32, `16 × n_pad` with `n_pad` the keys of
//!    the last row rounded up to sixteen. The weights are the keys in the
//!    R16 layout with the positions as columns, written by the KV append
//!    (`execution/kv.rs`): one call of a micro-kernel per 64 keys (or per 16
//!    on the vector path), `K = head_dim`, an immediate. The blocks of keys
//!    wholly above the diagonal are never computed.
//! 2. **Softmax**, row by row, in the order of the decode kernel: the scale,
//!    `-inf` from `n_i` to the next multiple of sixteen, then `emit_softmax`
//!    of `sdpa.rs` itself (maximum, exponentials and sum over the padded
//!    length, division). The probabilities are rounded to bf16 (`bfcvtn`,
//!    to nearest even) into `P`, zero beyond `n_i`: packed by pairs of keys
//!    for the matrix engine (`[n_pad/2][16 rows][2]`, the layout of the
//!    packing node), row-major for the vector kernel.
//! 3. **Output** `O = P·V`, bf16, straight into the attention buffer. The
//!    weights are `Vᵀ` in the R16 layout (the dimensions as columns, the
//!    keys as `K`), whose blocks are `capacity` keys long while the call
//!    reads `n_pad` of them: the micro-kernels under `k_dynamic`, the groups
//!    of four pairs in `x15`.
//!
//! On a machine with SME2 both GEMMs run on the matrix engine (`bfmopa`,
//! `gemm_sme.rs`; the queries packed first by the `trn` of the packing
//! node); otherwise on the vector kernel of `gemm.rs`, 4 × 16 tiles, with the
//! multiply-accumulate of `Knobs::neon` (`bfmlal` on the M3 Ultra).
//!
//! **The bits are not those of the decode kernel**: the scores accumulate
//! pair by pair in the GEMM's order (`bfdot`/`bfmopa` rounded to odd, or two
//! IEEE fused multiply-adds for `bfmlal`) instead of four `bfdot` lanes and a
//! pairwise reduction, and `P` is bf16 where the decode kernel keeps f32.
//! Each path is transcribed and the kernel held to the bit against it;
//! both are measured against the f32 reference.
//!
//! The scratch of a call (`scratch_f32`): the scores, `16 × capacity` f32;
//! `P`, `16 × capacity` bf16; the packed queries, `16 × head_dim` bf16.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{AFormat, KernelOptions, MicroKernelSpec, NeonMac, NeonOptions};
use crate::jit::emit_sdpa::{SdpaPrefillSpec, SDPA_BLOCK_ROWS};
use crate::jit::ir::{Decline, SdpaForm};
use crate::jit::target::aarch64::sdpa::emit_softmax;
use crate::jit::target::aarch64::util::{add_imm, backend_err, Frame};
use crate::jit::target::aarch64::{gemm, gemm_sme};
use a64::*;

const ROWS: usize = SDPA_BLOCK_ROWS;

/// Where the GEMMs run: the matrix engine, or the vector kernel with this
/// multiply-accumulate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlockOptions {
    /// Both GEMMs on the matrix engine (SME2 `bfmopa`).
    pub matrix: bool,
    /// Otherwise, the multiply-accumulate of the vector kernel.
    pub mac: NeonMac,
}

impl BlockOptions {
    /// What the projections of the prefill use in this process.
    pub fn production() -> Self {
        let k = crate::jit::knobs::knobs();
        BlockOptions { matrix: k.gemm_matrix && gemm_sme::available(), mac: k.neon.mac }
    }
}

/// The shapes of the block kernel: sixteen rows, no sink, whole blocks of
/// sixteen positions and of sixteen dimensions.
pub fn check(spec: &SdpaPrefillSpec) -> std::result::Result<(), Decline> {
    spec.shape.check(SdpaForm::Chained)?;
    if spec.shape.sink {
        return Err(Decline::SinkInFused);
    }
    if spec.rows != ROWS {
        return Err(Decline::PrefillRows);
    }
    let hd = spec.shape.head_dim;
    if spec.row_stride < hd || spec.out_row_stride < hd {
        return Err(Decline::PrefillRows);
    }
    if spec.kv_capacity < ROWS || !spec.kv_capacity.is_multiple_of(16) {
        return Err(Decline::Empty);
    }
    Ok(())
}

/// Bytes of the scores of a call.
const fn scores_bytes(capacity: usize) -> usize {
    ROWS * capacity * 4
}

/// Bytes of `P`.
const fn p_bytes(capacity: usize) -> usize {
    ROWS * capacity * 2
}

/// The scratch of one call, in f32.
pub fn scratch_f32(spec: &SdpaPrefillSpec) -> usize {
    (scores_bytes(spec.kv_capacity) + p_bytes(spec.kv_capacity) + ROWS * spec.shape.head_dim * 2) / 4
}

/// The micro-kernels a call chains, emitted once.
struct Kernels {
    /// Scores: on the matrix engine, 16, 32, 48 and 64 keys; on the vector
    /// kernel, one of 4 rows × 16 keys.
    scores: Vec<Vec<u8>>,
    /// Output: on the matrix engine, 64 dimensions then the rest of the head
    /// when it is not a multiple of 64; on the vector kernel, 4 × 16.
    out: Vec<Vec<u8>>,
}

fn kernels(spec: &SdpaPrefillSpec, opts: BlockOptions) -> Result<Kernels> {
    let hd = spec.shape.head_dim;
    let cap = spec.kv_capacity;
    let emit = |s: &MicroKernelSpec| -> Result<Vec<u8>> {
        s.check().map_err(|d| HerbertError::Backend(format!("backend_jit sdpa block: micro-kernel {}x{}: {d}", s.mr, s.nr)))?;
        crate::jit::emit::emit(s)
    };
    if opts.matrix {
        let matrix = KernelOptions { matrix: true, ..KernelOptions::default() };
        let scores = (1..=4)
            .map(|nb| {
                emit(&MicroKernelSpec {
                    mr: ROWS,
                    nr: 16 * nb,
                    k_pairs: hd / 2,
                    k_odd: false,
                    lda_bytes: hd * 2,
                    a_fmt: AFormat::Bf16,
                    ldc_bytes: cap * 4,
                    n_tail: 16,
                    out_bf16: false,
                    opts: matrix,
                    k_dynamic: false,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut widths = vec![64usize];
        if !hd.is_multiple_of(64) {
            widths.push(hd % 64);
        }
        let out = widths
            .into_iter()
            .map(|nr| {
                emit(&MicroKernelSpec {
                    mr: ROWS,
                    nr,
                    k_pairs: cap / 2,
                    k_odd: false,
                    lda_bytes: cap * 2,
                    a_fmt: AFormat::Bf16,
                    ldc_bytes: spec.out_row_stride * 2,
                    n_tail: 16,
                    out_bf16: true,
                    opts: matrix,
                    k_dynamic: true,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Kernels { scores, out })
    } else {
        let vector = KernelOptions { neon: NeonOptions { mac: opts.mac, ..NeonOptions::default() }, ..KernelOptions::default() };
        let tile = gemm::ARM_TILE;
        let scores = vec![emit(&MicroKernelSpec {
            mr: tile.mr,
            nr: tile.nr,
            k_pairs: hd / 2,
            k_odd: false,
            lda_bytes: spec.row_stride * 2,
            a_fmt: AFormat::Bf16,
            ldc_bytes: cap * 4,
            n_tail: 16,
            out_bf16: false,
            opts: vector,
            k_dynamic: false,
        })?];
        let out = vec![emit(&MicroKernelSpec {
            mr: tile.mr,
            nr: tile.nr,
            k_pairs: cap / 2,
            k_odd: false,
            lda_bytes: cap * 2,
            a_fmt: AFormat::Bf16,
            ldc_bytes: spec.out_row_stride * 2,
            n_tail: 16,
            out_bf16: true,
            opts: vector,
            k_dynamic: true,
        })?];
        Ok(Kernels { scores, out })
    }
}

/// Generates the block kernel of `spec` on the GEMMs of `opts`.
///
/// `(q, k, v, scratch, out, past, k_r16, v_r16, scale, sink)`: `q` and `out`
/// at row 0 of the block and of the head, `k`/`v` the head-major cache
/// (unused), `past` the keys before row 0, `k_r16`/`v_r16` the head's R16
/// copies. `sink` is refused by [`check`].
pub fn emit(spec: &SdpaPrefillSpec, opts: BlockOptions) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit sdpa block: {d}")))?;
    let hd = spec.shape.head_dim;
    let cap = spec.kv_capacity;
    let ks = kernels(spec, opts)?;
    // Scratch offsets, from x20.
    let off_p = scores_bytes(cap);
    let off_q = off_p + p_bytes(cap);
    let s_row = (cap * 4) as u64;
    // Bytes between two R16 blocks: of keys (sixteen positions, `head_dim`
    // dimensions), of `Vᵀ` (sixteen dimensions, `capacity` positions).
    let blk_k = (16 * hd * 2) as i64;
    let blk_v = (16 * cap * 2) as i64;

    let mut a = Asm::new();
    let score_labels: Vec<Label> = ks.scores.iter().map(|_| a.label()).collect();
    let out_labels: Vec<Label> = ks.out.iter().map(|_| a.label()).collect();
    // x19 q, x20 scratch, x21 out, x22 past, x23 keys R16, x24 values R16,
    // x25 key blocks of the call (`n_pad / 16`), x26..x28 loops; the scale at [sp].
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23, X24, X25, X26, X27, X28], false, 16);
    a.mov(X19, X0);
    a.mov(X20, X3);
    a.mov(X21, X4);
    a.mov(X22, X5);
    a.mov(X23, X6);
    a.mov(X24, X7);
    a.str(s(0), mem(SP, 0));
    // n_pad / 16 = ceil((past + 16) / 16).
    a.add(X25, X22, 31);
    a.lsr(X25, X25, 4);

    // ── 1. The scores ─────────────────────────────────────────────────────
    if opts.matrix {
        pack_queries(&mut a, spec.row_stride, hd, off_q);
        // Chunks of 64 keys, then one kernel of the 16, 32 or 48 left.
        let (chunks, done) = (a.label(), a.label());
        a.mov(X27, X23);
        a.mov(X28, X20);
        a.lsr(X26, X25, 2);
        a.bind(chunks);
        a.cbz(X26, done);
        add_imm(&mut a, X0, X20, off_q as i64, X16);
        a.mov(X1, X27);
        a.mov(X2, X28);
        a.bl(score_labels[3]);
        add_imm(&mut a, X27, X27, 4 * blk_k, X16);
        a.add(X28, X28, 256);
        a.sub(X26, X26, 1);
        a.b(chunks);
        a.bind(done);
        a.and(X26, X25, 3u64);
        let after = a.label();
        for left in 1..=3usize {
            let next = a.label();
            a.cmp(X26, left as u64);
            a.bne(next);
            add_imm(&mut a, X0, X20, off_q as i64, X16);
            a.mov(X1, X27);
            a.mov(X2, X28);
            a.bl(score_labels[left - 1]);
            a.b(after);
            a.bind(next);
        }
        a.bind(after);
    } else {
        // Four rows of queries (straight from `q`) × sixteen keys per call.
        let mr = gemm::ARM_TILE.mr;
        for g in 0..ROWS / mr {
            let top = a.label();
            a.mov(X27, X23);
            add_imm(&mut a, X28, X20, (g * mr) as i64 * s_row as i64, X16);
            a.mov(X26, X25);
            a.bind(top);
            add_imm(&mut a, X0, X19, (g * mr * spec.row_stride * 2) as i64, X16);
            a.mov(X1, X27);
            a.mov(X2, X28);
            a.bl(score_labels[0]);
            add_imm(&mut a, X27, X27, blk_k, X16);
            a.add(X28, X28, 64);
            a.subs(X26, X26, 1);
            a.bne(top);
        }
    }

    // ── 2. The softmax, row by row, into P ────────────────────────────────
    // Zero the last sixteen keys of P: a row whose keys end one block
    // earlier writes nothing there.
    a.movi(v(0).d2(), 0);
    if opts.matrix {
        // Pairs n_pad/2 − 8 .. n_pad/2: 512 contiguous bytes.
        a.sub(X9, X25, 1);
        a.add(X9, X20, X9.lsl(9));
        add_imm(&mut a, X9, X9, off_p as i64, X16);
        for c in 0..16i64 {
            a.stp(q(0), q(0), mem(X9, 32 * c));
        }
    } else {
        // Keys n_pad − 16 .. n_pad of every row: 32 bytes per row.
        a.sub(X9, X25, 1);
        a.add(X9, X20, X9.lsl(5));
        add_imm(&mut a, X9, X9, off_p as i64, X16);
        for r in 0..ROWS {
            add_imm(&mut a, X10, X9, (r * cap * 2) as i64, X16);
            a.stp(q(0), q(0), at(X10));
        }
    }
    // x27 = row i, x10 = its scores, x11 = n_i, x28 = its first element of P.
    let rows_top = a.label();
    a.mov_imm(X27, 0);
    a.mov(X10, X20);
    add_imm(&mut a, X28, X20, off_p as i64, X16);
    a.bind(rows_top);
    a.add(X11, X22, X27);
    a.add(X11, X11, 1);
    // x12 = the padded length L_i = ceil(n_i / 16)·16.
    a.add(X12, X11, 15);
    a.and(X12, X12, !15u64);
    // The scale over the L_i/4 quads (the pads are overwritten next).
    {
        let top = a.label();
        a.ldr(s(0), mem(SP, 0));
        a.dup(v(31).s4(), v(0).s(0));
        a.mov(X13, X10);
        a.lsr(X14, X12, 2);
        a.bind(top);
        a.ldr(q(0), at(X13));
        a.fmul(v(0).s4(), v(0).s4(), v(31).s4());
        a.str(q(0), post(X13, 16));
        a.subs(X14, X14, 1);
        a.bne(top);
    }
    // -inf from n_i to L_i.
    let fill = |a: &mut Asm, bits: u32| {
        let (top, done) = (a.label(), a.label());
        a.add(X13, X10, X11.lsl(2));
        a.sub(X14, X12, X11);
        a.mov_imm(W15, u64::from(bits));
        a.bind(top);
        a.cbz(X14, done);
        a.str(W15, post(X13, 4));
        a.sub(X14, X14, 1);
        a.b(top);
        a.bind(done);
    };
    fill(&mut a, 0xff80_0000);
    // The decode kernel's softmax over the padded row (x6..x9, v0..v7, v16..v29).
    emit_softmax(&mut a, X10, X11, false);
    // Zero from n_i to L_i: the pads' 2^-126 never reach P.
    fill(&mut a, 0);
    // bf16 into P, four keys per step.
    {
        let top = a.label();
        a.mov(X13, X10);
        a.mov(X14, X28);
        a.lsr(X15, X12, 2);
        a.mov_imm(X16, 64);
        a.bind(top);
        a.ldr(q(0), post(X13, 16));
        a.bfcvtn(v(0).h4(), v(0).s4());
        if opts.matrix {
            // Two pairs of keys, 64 bytes apart.
            a.st1_lane(v(0).s(0), post_reg(X14, X16));
            a.st1_lane(v(0).s(1), post_reg(X14, X16));
        } else {
            a.str(d(0), post(X14, 8));
        }
        a.subs(X15, X15, 1);
        a.bne(top);
    }
    add_imm(&mut a, X10, X10, s_row as i64, X16);
    add_imm(&mut a, X28, X28, if opts.matrix { 4 } else { (cap * 2) as i64 }, X16);
    a.add(X27, X27, 1);
    a.cmp(X27, ROWS as u64);
    a.blo(rows_top);

    // ── 3. The output, P·V, bf16 into `out` ───────────────────────────────
    if opts.matrix {
        // Chunks of 64 dimensions, then the rest; n_pad/8 groups of four pairs.
        let mut col = 0usize;
        for (kind, &label) in out_labels.iter().enumerate() {
            let width = if kind == 0 { 64 } else { hd % 64 };
            let count = if kind == 0 { hd / 64 } else { 1 };
            for _ in 0..count {
                add_imm(&mut a, X0, X20, off_p as i64, X16);
                add_imm(&mut a, X1, X24, (col / 16) as i64 * blk_v, X16);
                add_imm(&mut a, X2, X21, (col * 2) as i64, X16);
                a.lsl(X15, X25, 1);
                a.bl(label);
                col += width;
            }
        }
        debug_assert_eq!(col, hd);
    } else {
        // Four rows of P × sixteen dimensions per call.
        let mr = gemm::ARM_TILE.mr;
        for g in 0..ROWS / mr {
            let top = a.label();
            a.mov(X27, X24);
            add_imm(&mut a, X28, X21, (g * mr * spec.out_row_stride * 2) as i64, X16);
            a.mov_imm(X26, (hd / 16) as u64);
            a.bind(top);
            add_imm(&mut a, X0, X20, (off_p + g * mr * cap * 2) as i64, X16);
            a.mov(X1, X27);
            a.mov(X2, X28);
            a.lsl(X15, X25, 1);
            a.bl(out_labels[0]);
            add_imm(&mut a, X27, X27, blk_v, X16);
            a.add(X28, X28, 32);
            a.subs(X26, X26, 1);
            a.bne(top);
        }
    }
    frame.leave(&mut a);
    a.ret();
    for (blob, &label) in ks.scores.iter().zip(&score_labels).chain(ks.out.iter().zip(&out_labels)) {
        a.align(16);
        a.bind(label);
        a.bytes(blob);
    }
    a.finish().map(|c| c.bytes).map_err(backend_err(format_args!(
        "sdpa block head_dim={hd} capacity={cap} matrix={}",
        opts.matrix
    )))
}

/// The sixteen query rows at `x19` (`row_stride` elements apart) packed by
/// pairs of dimensions at `x20 + off_q`: pair `p` of the sixteen rows in 64
/// contiguous bytes, the layout of the packing node (`execution/bf16.rs`).
/// Clobbers `x4..x7`, `x12`, `x13`, `x16`, `v0..v7`, `v16..v19`.
fn pack_queries(a: &mut Asm, row_stride: usize, hd: usize, off_q: usize) {
    let row = (row_stride * 2) as u64;
    for group in 0..4u64 {
        for (i, r) in [X4, X5, X6, X7].into_iter().enumerate() {
            add_imm(a, r, X19, ((4 * group + i as u64) * row) as i64, X16);
        }
        add_imm(a, X12, X20, (off_q as u64 + 16 * group) as i64, X16);
        let top = a.label();
        a.mov_imm(X13, (hd / 8) as u64);
        a.bind(top);
        for (reg, r) in [X4, X5, X6, X7].into_iter().enumerate() {
            a.ldr(q(reg as u8), post(r, 16));
        }
        a.trn1(v(4).s4(), v(0).s4(), v(1).s4());
        a.trn2(v(5).s4(), v(0).s4(), v(1).s4());
        a.trn1(v(6).s4(), v(2).s4(), v(3).s4());
        a.trn2(v(7).s4(), v(2).s4(), v(3).s4());
        a.trn1(v(16).d2(), v(4).d2(), v(6).d2());
        a.trn1(v(17).d2(), v(5).d2(), v(7).d2());
        a.trn2(v(18).d2(), v(4).d2(), v(6).d2());
        a.trn2(v(19).d2(), v(5).d2(), v(7).d2());
        for (p, r) in [16u8, 17, 18, 19].into_iter().enumerate() {
            a.str(q(r), mem(X12, 64 * p as i64));
        }
        a.add(X12, X12, 256);
        a.subs(X13, X13, 1);
        a.bne(top);
    }
}
