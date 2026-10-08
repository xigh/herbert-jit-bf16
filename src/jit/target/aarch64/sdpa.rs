//! The decode attention (SDPA), NEON: one query head against the cached keys
//! and values (the specification is `jit/emit_sdpa.rs`).
//!
//! ## QK — four positions per load of `q`
//!
//! `bfdot vd.4s, vk.8h, vq.8h` accumulates, in each f32 lane, the two
//! products of a pair: the eight bf16 of a chunk of `q` against the same
//! chunk of a key. Four keys share each load of the chunk of `q`; after the
//! `head_dim/8` chunks, three `faddp` turn the four accumulators into the
//! four scores, `(l0+l1)+(l2+l3)` per score. A position alone (the tail)
//! reduces its accumulator with the same pairing, so a score has the same
//! bits whether its position was computed in a group or alone — which is
//! what lets a block of queries reuse this kernel's order.
//!
//! ## Softmax — padded to 16
//!
//! The QK pass pads the scores with `-inf` up to the next multiple of 16; the
//! maximum, the exponentials (the polynomial of the elementwise kernels) and
//! the sum run over the padded length without a mask. Normalisation is a
//! division without a sink, a multiplication by the inverse with one, as in
//! the scalar reference.
//!
//! ## SV — by slices of the head
//!
//! `out[d] = Σ_j p_j · V[j][d]`, one fused multiply-add per position, in
//! position order: `V` widened by `shll`, the score as a lane operand of
//! `fmla` by element. The head is cut into slices of 64 dimensions (16 f32
//! accumulators), or 32 when it is not a multiple of 64; each slice is one
//! pass over the positions, reading one 128-byte line of each row.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_sdpa::{Pass, SdpaKernelSpec, SdpaPrefillSpec};
use crate::jit::ir::{ExpStrategy, SdpaForm};
use crate::jit::target::aarch64::elem::{vexp, widen, narrow, ExpRegs};
use crate::jit::target::aarch64::util::{add_imm, backend_err, splat, Frame};
use a64::*;

/// Is this spec emittable here? The chained and split forms, the inline exp.
pub fn check(spec: &SdpaKernelSpec) -> std::result::Result<(), crate::jit::ir::Decline> {
    spec.shape.check(spec.form)?;
    if spec.form == SdpaForm::Fused || spec.exp != ExpStrategy::Inline {
        return Err(crate::jit::ir::Decline::HeadDim);
    }
    Ok(())
}

/// The QK pass: `scores[j] = (q · K[j]) · scale` for `j < n`, then `-inf` up
/// to the next multiple of 16. `scale` broadcast in `v31`. Clobbers
/// `x6..x13`, `v0..v7`.
fn emit_qk(a: &mut Asm, hd: usize, q_ptr: R, k: R, scores: R, n: R) {
    let row = (hd * 2) as i64;
    let chunks = hd / 8;
    let group = a.label();
    let singles = a.label();
    let single = a.label();
    let pad = a.label();
    let pad_loop = a.label();
    let done = a.label();
    a.mov(X6, k);
    a.mov(X7, scores);
    a.mov(X8, n);
    a.bind(group);
    a.cmp(X8, 4);
    a.blo(singles);
    for c in 0..4 {
        a.movi(v(c).d2(), 0);
    }
    add_imm(a, X10, X6, row, X9);
    add_imm(a, X11, X10, row, X9);
    add_imm(a, X12, X11, row, X9);
    for c in 0..chunks {
        let off = (c * 16) as i64;
        a.ldr(q(4), mem(q_ptr, off));
        a.ldr(q(5), mem(X6, off));
        a.ldr(q(6), mem(X10, off));
        a.ldr(q(7), mem(X11, off));
        a.bfdot(v(0).s4(), v(5).h8(), v(4).h8());
        a.ldr(q(5), mem(X12, off));
        a.bfdot(v(1).s4(), v(6).h8(), v(4).h8());
        a.bfdot(v(2).s4(), v(7).h8(), v(4).h8());
        a.bfdot(v(3).s4(), v(5).h8(), v(4).h8());
    }
    a.faddp(v(0).s4(), v(0).s4(), Some(v(1).s4()));
    a.faddp(v(2).s4(), v(2).s4(), Some(v(3).s4()));
    a.faddp(v(0).s4(), v(0).s4(), Some(v(2).s4()));
    a.fmul(v(0).s4(), v(0).s4(), v(31).s4());
    a.str(q(0), post(X7, 16));
    add_imm(a, X6, X12, row, X9);
    a.sub(X8, X8, 4);
    a.b(group);
    a.bind(singles);
    a.bind(single);
    a.cbz(X8, pad);
    a.movi(v(0).d2(), 0);
    for c in 0..chunks {
        let off = (c * 16) as i64;
        a.ldr(q(4), mem(q_ptr, off));
        a.ldr(q(5), mem(X6, off));
        a.bfdot(v(0).s4(), v(5).h8(), v(4).h8());
    }
    a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
    a.faddp(s(0), v(0).s2(), None);
    a.fmul(s(0), s(0), s(31));
    a.str(s(0), post(X7, 4));
    add_imm(a, X6, X6, row, X9);
    a.sub(X8, X8, 1);
    a.b(single);
    // `-inf` from n to the next multiple of 16.
    a.bind(pad);
    a.add(X9, n, 15);
    a.and(X9, X9, !15u64);
    a.sub(X8, X9, n);
    a.mov_imm(W9, 0xff80_0000);
    a.bind(pad_loop);
    a.cbz(X8, done);
    a.str(W9, post(X7, 4));
    a.sub(X8, X8, 1);
    a.b(pad_loop);
    a.bind(done);
}

/// The softmax over the padded scores: `L = ⌈n/16⌉·16` values at `scores`.
/// With a sink, the sink (lane 0 of `v30`) joins the maximum and the sum.
/// Clobbers `x6..x9`, `v0..v7`, `v16..v29`.
pub(crate) fn emit_softmax(a: &mut Asm, scores: R, n: R, sink: bool) {
    let k = ExpRegs::at(16);
    k.load(a, X9);
    let max_loop = a.label();
    let exp_loop = a.label();
    let norm_loop = a.label();
    // x8 = quads of the padded length.
    a.add(X8, n, 15);
    a.lsr(X8, X8, 4);
    a.lsl(X8, X8, 2);
    // Maximum.
    a.mov_imm(W9, 0xff80_0000);
    a.dup_gp(v(0).s4(), W9);
    a.mov(X6, scores);
    a.mov(X7, X8);
    a.bind(max_loop);
    a.ldr(q(1), post(X6, 16));
    a.fmax(v(0).s4(), v(0).s4(), v(1).s4());
    a.subs(X7, X7, 1);
    a.bne(max_loop);
    a.fmaxv(s(0), v(0).s4());
    if sink {
        a.fmax(s(0), s(0), s(30));
    }
    a.dup(v(29).s4(), v(0).s(0));
    // Exponentials and their sum.
    a.movi(v(28).d2(), 0);
    a.mov(X6, scores);
    a.mov(X7, X8);
    a.bind(exp_loop);
    a.ldr(q(1), at(X6));
    a.fsub(v(1).s4(), v(1).s4(), v(29).s4());
    vexp(a, &k, 1, [2, 3, 4, 5]);
    a.str(q(1), post(X6, 16));
    a.fadd(v(28).s4(), v(28).s4(), v(1).s4());
    a.subs(X7, X7, 1);
    a.bne(exp_loop);
    a.faddp(v(28).s4(), v(28).s4(), Some(v(28).s4()));
    a.faddp(s(28), v(28).s2(), None);
    if sink {
        a.dup(v(1).s4(), v(30).s(0));
        a.fsub(v(1).s4(), v(1).s4(), v(29).s4());
        vexp(a, &k, 1, [2, 3, 4, 5]);
        a.fadd(s(28), s(28), s(1));
        // Multiplication by the inverse.
        a.fmov_imm(s(1), 1.0);
        a.fdiv(s(28), s(1), s(28));
    }
    a.dup(v(28).s4(), v(28).s(0));
    a.mov(X6, scores);
    a.mov(X7, X8);
    a.bind(norm_loop);
    a.ldr(q(1), at(X6));
    if sink {
        a.fmul(v(1).s4(), v(1).s4(), v(28).s4());
    } else {
        a.fdiv(v(1).s4(), v(1).s4(), v(28).s4());
    }
    a.str(q(1), post(X6, 16));
    a.subs(X7, X7, 1);
    a.bne(norm_loop);
}

/// The SV pass: `out[d] = Σ_{j<n} scores[j] · V[j][d]`, bf16 out. Clobbers
/// `x6..x13`, `v0..v7`, `v16..v31`.
fn emit_sv(a: &mut Asm, hd: usize, scores: R, vv: R, out: R, n: R) {
    let row = (hd * 2) as i64;
    let slice = if hd.is_multiple_of(64) { 64 } else { 32 };
    let accs: Vec<u8> = [0u8, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23][..slice / 4].to_vec();
    for s0 in (0..hd).step_by(slice) {
        let quads = a.label();
        let singles = a.label();
        let single = a.label();
        let done = a.label();
        for &r in &accs {
            a.movi(v(r).d2(), 0);
        }
        add_imm(a, X6, vv, (s0 * 2) as i64, X9);
        a.mov(X7, scores);
        a.mov(X8, n);
        // One row of the slice: `slice` bf16 in `slice/8` q loads, widened
        // and multiplied by lane `lane` of v24.
        let row_body = |a: &mut Asm, base: R, lane: u8| {
            for c in 0..slice / 16 {
                a.ldp(q(25), q(26), mem(base, 32 * c as i64));
                for (k, src) in [25u8, 26].into_iter().enumerate() {
                    let quad = 4 * c + 2 * k;
                    widen(a, 29, src, false);
                    widen(a, 30, src, true);
                    a.fmla(v(accs[quad]).s4(), v(29).s4(), v(24).s(lane));
                    a.fmla(v(accs[quad + 1]).s4(), v(30).s4(), v(24).s(lane));
                }
            }
        };
        a.bind(quads);
        a.cmp(X8, 4);
        a.blo(singles);
        a.ldr(q(24), post(X7, 16));
        add_imm(a, X10, X6, row, X9);
        add_imm(a, X11, X10, row, X9);
        add_imm(a, X12, X11, row, X9);
        for (lane, base) in [X6, X10, X11, X12].into_iter().enumerate() {
            row_body(a, base, lane as u8);
        }
        add_imm(a, X6, X12, row, X9);
        a.sub(X8, X8, 4);
        a.b(quads);
        a.bind(singles);
        a.bind(single);
        a.cbz(X8, done);
        a.ldr(s(24), post(X7, 4));
        row_body(a, X6, 0);
        add_imm(a, X6, X6, row, X9);
        a.sub(X8, X8, 1);
        a.b(single);
        a.bind(done);
        add_imm(a, X10, out, (s0 * 2) as i64, X9);
        for pair in accs.chunks(2) {
            narrow(a, 31, pair[0], pair[1]);
            a.str(q(31), post(X10, 16));
        }
    }
}

/// Generates the decode kernel of `spec`: the whole head, or one pass.
pub fn emit(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit sdpa: {d}")))?;
    let hd = spec.shape.head_dim;
    let sink = spec.shape.sink;
    let mut a = Asm::new();
    // v8..v15 are not touched: no frame beyond the return.
    let frame = Frame::enter(&mut a, false, &[], false, 0);
    match pass {
        None => {
            // (q, k, v, scores, out, n, scale, sink)
            a.dup(v(31).s4(), v(0).s(0));
            a.fmov(s(30), s(1));
            emit_qk(&mut a, hd, X0, X1, X3, X5);
            emit_softmax(&mut a, X3, X5, sink);
            emit_sv(&mut a, hd, X3, X2, X4, X5);
        }
        Some(Pass::Qk) => {
            // (q, k, scores, n, scale)
            a.dup(v(31).s4(), v(0).s(0));
            emit_qk(&mut a, hd, X0, X1, X2, X3);
        }
        Some(Pass::Softmax) => {
            // (scores, n, sink)
            a.fmov(s(30), s(0));
            emit_softmax(&mut a, X0, X1, sink);
        }
        Some(Pass::Sv) => {
            // (scores, v, out, n)
            emit_sv(&mut a, hd, X0, X1, X2, X3);
        }
    }
    frame.leave(&mut a);
    a.ret();
    let _ = splat;
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("sdpa head_dim={hd} form={}", spec.form.name())))
}

/// The prefill block kernel: the two-GEMM attention of `sdpa_block.rs`, on
/// the GEMMs of the projections of this process (`Knobs::attn_block`; the
/// decode kernel serves a block row by row otherwise).
pub fn emit_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
    super::sdpa_block::emit(spec, super::sdpa_block::BlockOptions::production())
}
