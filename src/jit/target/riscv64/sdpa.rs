//! The decode attention (SDPA), RVV: one query head against the cached keys
//! and values (the specification is `jit/emit_sdpa.rs`).
//!
//! ## QK — one position at a time, folded in order
//!
//! A score is `q · K[j]` over the head: `vfwmaccbf16.vv` accumulates the
//! products of chunk `c` of the head into lanes `0..L` of an `e32, m8`
//! group (element `c·L + l` into lane `l`, a fused chain per lane), then
//! `vfredosum` adds the lanes in order from `+0`, and the scale multiplies
//! the result. `L` is the width of `--rvv-width` capped by the head and by
//! VLMAX: the bits of a score depend on `L`, never on its neighbours — a
//! position alone or in a block of queries gives the same bits.
//!
//! ## Softmax — padded to 16
//!
//! The QK pass pads the scores with `-inf` up to the next multiple of 16. The
//! maximum is `vfredmax` (exact, in any order); the exponentials (the
//! polynomial of the elementwise kernels) are written in place and summed
//! into `W` lanes then folded by `vfredosum`, as the Σx² of the norms. The
//! normalisation divides without a sink and multiplies by the inverse with
//! one, as in the scalar reference.
//!
//! ## SV — a fused chain per dimension, positions in order
//!
//! `out[d] = Σ_j p_j · V[j][d]`: each row of `V` widened by `vfwcvtbf16`,
//! the score an `f32` scalar of `vfmacc.vf`. The head is cut into slices of
//! VLMAX (`e32, m8`); each slice is one pass over the positions. No sum is
//! folded: the bits of the output do not depend on VLEN.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_sdpa::{Pass, SdpaKernelSpec, SdpaPrefillSpec};
use crate::jit::ir::{Decline, ExpStrategy, SdpaForm};
use crate::jit::target::riscv64::elem::{fconst, sum_lanes, vexp, ExpConsts};
use crate::jit::target::riscv64::util::{add_imm, backend_err};
use rv::*;

const E16M4: Vtype = Vtype::new(Sew::E16, Lmul::M4);
const E32M4: Vtype = Vtype::new(Sew::E32, Lmul::M4);
const E32M8: Vtype = Vtype::new(Sew::E32, Lmul::M8);

/// The lanes of a QK chain: the width knob, capped by VLMAX at `e32, m8`
/// (`2·vlenb`) and by the head.
pub(crate) fn qk_lanes(head_dim: usize) -> usize {
    crate::jit::knobs::knobs().rvv.width.lanes(2 * crate::core::isa::vlenb()).min(head_dim)
}

/// The dimensions of one SV slice: VLMAX at `e32, m8`, capped by the head.
pub(crate) fn sv_slice(head_dim: usize) -> usize {
    (2 * crate::core::isa::vlenb()).min(head_dim)
}

/// Is this spec emittable here? The chained and split forms, the inline exp.
pub fn check(spec: &SdpaKernelSpec) -> std::result::Result<(), Decline> {
    spec.shape.check(spec.form)?;
    if spec.form == SdpaForm::Fused || spec.exp != ExpStrategy::Inline {
        return Err(Decline::HeadDim);
    }
    Ok(())
}

/// The QK pass: `scores[j] = (q · K[j]) · scale` for `j < n`, then `-inf` up
/// to the next multiple of 16. Clobbers `t0..t6`, `ft0`, `v8..v31`.
fn emit_qk(a: &mut Asm, hd: usize, q: X, k: X, scores: X, n: X, scale: F) {
    let lanes = qk_lanes(hd);
    let row = (hd * 2) as i64;
    let (acc, vq, vk, red) = (V8, V16, V20, V24);
    let (tk, ts, tn, tq, tk2) = (T3, T4, T5, T0, T1);
    a.mv(tk, k);
    a.mv(ts, scores);
    a.mv(tn, n);
    let top = a.label();
    let pad = a.label();
    a.bind(top);
    a.beqz(tn, pad);
    a.li(T2, lanes as i64);
    a.vsetvli(ZERO, T2, E32M8);
    a.vmv_v_i(acc, 0);
    let mut c0 = 0;
    while c0 < hd {
        let len = lanes.min(hd - c0);
        let off = (c0 * 2) as i64;
        add_imm(a, tq, q, off, T6);
        add_imm(a, tk2, tk, off, T6);
        a.li(T2, len as i64);
        a.vsetvli(ZERO, T2, E16M4.tu());
        a.vle(Sew::E16, vq, tq);
        a.vle(Sew::E16, vk, tk2);
        a.vfwmaccbf16_vv(acc, vq, vk);
        c0 += len;
    }
    a.li(T2, lanes as i64);
    a.vsetvli(ZERO, T2, E32M8);
    a.vmv_s_x(red, ZERO);
    a.vfredosum_vs(red, acc, red);
    a.vfmv_f_s(FT0, red);
    a.fmul_s(FT0, FT0, scale);
    a.fsw(FT0, ts, 0);
    a.addi(ts, ts, 4);
    add_imm(a, tk, tk, row, T6);
    a.addi(tn, tn, -1);
    a.j(top);
    // `-inf` from n to the next multiple of 16.
    a.bind(pad);
    let done = a.label();
    a.addi(T2, n, 15);
    a.andi(T2, T2, -16);
    a.sub(tn, T2, n);
    a.li(T2, 0xff80_0000);
    let pad_loop = a.here();
    a.beqz(tn, done);
    a.sw(T2, ts, 0);
    a.addi(ts, ts, 4);
    a.addi(tn, tn, -1);
    a.j(pad_loop);
    a.bind(done);
}

/// The softmax over the padded scores: `⌈n/16⌉·16` values at `scores`. With a
/// sink (in `sink`), it joins the maximum and the sum. Clobbers `t0..t6`,
/// `ft0..ft10`, `v4..v31`.
pub(crate) fn emit_softmax(a: &mut Asm, scores: X, n: X, sink: Option<F>) {
    let k = ExpConsts::standard(T6);
    k.load(a, T2);
    let (x, acc, red, t1, t2, t3) = (V4, V8, V12, V16, V20, V24);
    let (fmax, fsum, fone) = (FT7, FT8, FT9);
    let lanes = sum_lanes();
    let (len, rem, p) = (T3, T4, T5);
    // len = the padded length.
    a.addi(len, n, 15);
    a.andi(len, len, -16);
    // Maximum.
    a.li(T2, 0xff80_0000);
    a.fmv_w_x(fmax, T2);
    a.vsetivli(ZERO, 1, E32M4);
    a.vfmv_s_f(red, fmax);
    a.mv(p, scores);
    a.mv(rem, len);
    let max_loop = a.here();
    a.vsetvli(T0, rem, E32M4);
    a.vle(Sew::E32, x, p);
    a.vfredmax_vs(red, x, red);
    a.slli(T1, T0, 2);
    a.add(p, p, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, max_loop);
    a.vfmv_f_s(fmax, red);
    if let Some(s) = sink {
        a.fmax_s(fmax, fmax, s);
    }
    // Exponentials, written back, summed into `lanes` lanes.
    a.li(T2, lanes as i64);
    a.minu(T2, T2, len);
    a.vsetvli(ZERO, T2, E32M4);
    a.vmv_v_i(acc, 0);
    a.li(T2, lanes as i64);
    a.mv(p, scores);
    a.mv(rem, len);
    let exp_loop = a.here();
    a.minu(T1, rem, T2);
    a.vsetvli(T0, T1, E32M4.tu());
    a.vle(Sew::E32, x, p);
    a.vfsub_vf(x, x, fmax);
    vexp(a, &k, x, [t1, t2, t3]);
    a.vse(Sew::E32, x, p);
    a.vfadd_vv(acc, acc, x);
    a.slli(T1, T0, 2);
    a.add(p, p, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, exp_loop);
    a.minu(T2, T2, len);
    a.vsetvli(ZERO, T2, E32M4);
    a.vmv_s_x(red, ZERO);
    a.vfredosum_vs(red, acc, red);
    a.vfmv_f_s(fsum, red);
    fconst(a, fone, 1.0, T2);
    if let Some(s) = sink {
        a.vsetivli(ZERO, 1, E32M4);
        a.vfmv_s_f(x, s);
        a.vfsub_vf(x, x, fmax);
        vexp(a, &k, x, [t1, t2, t3]);
        a.vfmv_f_s(FT10, x);
        a.fadd_s(fsum, fsum, FT10);
        // Multiplication by the inverse.
        a.fdiv_s(fsum, fone, fsum);
    }
    a.mv(p, scores);
    a.mv(rem, len);
    let norm_loop = a.here();
    a.vsetvli(T0, rem, E32M4);
    a.vle(Sew::E32, x, p);
    if sink.is_some() {
        a.vfmul_vf(x, x, fsum);
    } else {
        a.vfdiv_vf(x, x, fsum);
    }
    a.vse(Sew::E32, x, p);
    a.slli(T1, T0, 2);
    a.add(p, p, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, norm_loop);
}

/// The SV pass: `out[d] = Σ_{j<n} scores[j] · V[j][d]`, bf16 out. Clobbers
/// `t0..t6`, `ft0`, `v4..v23`.
fn emit_sv(a: &mut Asm, hd: usize, scores: X, vv: X, out: X, n: X) {
    let slice = sv_slice(hd);
    let row = (hd * 2) as i64;
    let (vin, acc, wide) = (V4, V8, V16);
    let (tv, ts, tn, tvl) = (T3, T4, T5, T2);
    let mut s0 = 0;
    while s0 < hd {
        let len = slice.min(hd - s0);
        a.li(tvl, len as i64);
        a.vsetvli(ZERO, tvl, E32M8);
        a.vmv_v_i(acc, 0);
        add_imm(a, tv, vv, (s0 * 2) as i64, T6);
        a.mv(ts, scores);
        a.mv(tn, n);
        let top = a.label();
        let done = a.label();
        a.bind(top);
        a.beqz(tn, done);
        a.vsetvli(ZERO, tvl, E16M4);
        a.vle(Sew::E16, vin, tv);
        a.vfwcvtbf16_f_f_v(wide, vin);
        a.vsetvli(ZERO, ZERO, E32M8);
        a.flw(FT0, ts, 0);
        a.vfmacc_vf(acc, FT0, wide);
        a.addi(ts, ts, 4);
        add_imm(a, tv, tv, row, T6);
        a.addi(tn, tn, -1);
        a.j(top);
        a.bind(done);
        a.vsetvli(ZERO, tvl, E16M4);
        a.vfncvtbf16_f_f_w(vin, acc);
        add_imm(a, T0, out, (s0 * 2) as i64, T6);
        a.vse(Sew::E16, vin, T0);
        s0 += len;
    }
}

/// Generates the decode kernel of `spec`: the whole head, or one pass.
pub fn emit(spec: &SdpaKernelSpec, pass: Option<Pass>) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit riscv64 sdpa: {d}")))?;
    let hd = spec.shape.head_dim;
    let sink = spec.shape.sink;
    let mut a = Asm::new();
    match pass {
        None => {
            // (q, k, v, scores, out, n, scale, sink)
            emit_qk(&mut a, hd, A0, A1, A3, A5, FA0);
            emit_softmax(&mut a, A3, A5, sink.then_some(FA1));
            emit_sv(&mut a, hd, A3, A2, A4, A5);
        }
        Some(Pass::Qk) => emit_qk(&mut a, hd, A0, A1, A2, A3, FA0),
        Some(Pass::Softmax) => emit_softmax(&mut a, A0, A1, sink.then_some(FA0)),
        Some(Pass::Sv) => emit_sv(&mut a, hd, A0, A1, A2, A3),
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("sdpa head_dim={hd} form={}", spec.form.name())))
}

/// The prefill block kernel: not written on this target; the decode kernel
/// serves a block row by row (`Knobs::attn_block` is off on RISC-V).
pub fn emit_prefill(spec: &SdpaPrefillSpec) -> Result<Vec<u8>> {
    Err(crate::jit::target::riscv64::util::not_yet(&format!("the prefill attention block (head_dim {})", spec.shape.head_dim)))
}
