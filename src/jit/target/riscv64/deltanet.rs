//! The GatedDeltaNet kernels, RVV: the causal conv + SiLU, the recurrence
//! and the gated RMS norm, with the ABIs of `jit/emit_conv.rs`,
//! `jit/emit_recurrence.rs` and `jit/emit_gated_norm.rs`.
//!
//! - **conv**: the vector runs over the channels. `vlseg4e32` brings the four
//!   taps of the state of `vl` channels into four register groups, one per
//!   tap; the input loaded in the group after the last tap makes the shifted
//!   state `(s1, s2, s3, x)` four consecutive groups, stored back by one
//!   `vsseg4e32`. The taps multiply their weights (`vlseg4e16`, widened) and
//!   sum as `(p0+p1)+(p2+p3)`, then the SiLU of the elementwise kernels.
//! - **recurrence**: the vector runs over the value dimensions, `e32, m8`;
//!   each cell of the state keeps the scalar reference's chain over the key
//!   dimension, in its order — fused when `fma`. No sum is folded: the bits
//!   do not depend on VLEN.
//! - **gated norm**: `Σx²` into `W` lanes then `vfredosum`, as the norms of
//!   `elem.rs`; the literal `1 / √(Σx²/dim + eps)`; then
//!   `((x · inv) · w) · silu(z)`, the f32 in place and its bf16.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_conv::ConvSpec;
use crate::jit::emit_gated_norm::GatedNormSpec;
use crate::jit::emit_recurrence::RecurrenceSpec;
use crate::jit::ir::Decline;
use crate::jit::target::riscv64::elem::{fconst, sum_lanes, vexp, ExpConsts, EXP_MAX88};
use crate::jit::target::riscv64::util::{add_imm, backend_err};
use rv::*;

const E16M1: Vtype = Vtype::new(Sew::E16, Lmul::M1);
const E16M2: Vtype = Vtype::new(Sew::E16, Lmul::M2);
const E32M2: Vtype = Vtype::new(Sew::E32, Lmul::M2);
const E32M4: Vtype = Vtype::new(Sew::E32, Lmul::M4);
const E32M8: Vtype = Vtype::new(Sew::E32, Lmul::M8);

/// `g / (exp(min(0 − g, 88)) + 1)` on the current `vl` at SEW 32: `g` in
/// `g`, the result in `g`; `t` and `u` scratch groups, `e` three more.
fn vsilu(a: &mut Asm, k: &ExpConsts, fzero: F, f88: F, fone: F, g: V, t: V, e: [V; 3]) {
    a.vfrsub_vf(t, g, fzero);
    a.vfmin_vf(t, t, f88);
    vexp(a, k, t, e);
    a.vfadd_vf(t, t, fone);
    a.vfdiv_vv(g, g, t);
}

/// The scalar constants of the SiLU: `ft7` = 0, `ft8` = 88, `ft9` = 1.
fn silu_consts(a: &mut Asm, tmp: X) -> (F, F, F) {
    a.fmv_w_x(FT7, ZERO);
    fconst(a, FT8, EXP_MAX88, tmp);
    fconst(a, FT9, 1.0, tmp);
    (FT7, FT8, FT9)
}

/// The conv of one position over `n_blocks` blocks of sixteen channels.
///
/// `a0 = data (f32, in place), a1 = w (bf16, 4 per channel), a2 = state (f32, 4 per channel), a3 = n_blocks`.
pub fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit riscv64 conv: {d}")))?;
    let mut a = Asm::new();
    let k = ExpConsts::standard(T6);
    k.load(&mut a, T2);
    let (fzero, f88, fone) = silu_consts(&mut a, T2);
    // The state taps s0..s3 in v8, v10, v12, v14; the input in v16; weights
    // in v4..v7 (e16 m1), widened into v20, v22, v24, v26.
    let rem = T4;
    a.slli(rem, A3, 4);
    let top = a.here();
    a.vsetvli(T0, rem, E32M2);
    a.vlseg(4, Sew::E32, V8, A2);
    a.vle(Sew::E32, V16, A0);
    a.vsseg(4, Sew::E32, V10, A2);
    a.vsetvli(ZERO, ZERO, E16M1);
    a.vlseg(4, Sew::E16, V4, A1);
    for (j, w) in [V20, V22, V24, V26].into_iter().enumerate() {
        a.vfwcvtbf16_f_f_v(w, V4.plus(j as u8));
    }
    a.vsetvli(ZERO, ZERO, E32M2);
    for (w, s) in [(V20, V10), (V22, V12), (V24, V14), (V26, V16)] {
        a.vfmul_vv(w, w, s);
    }
    a.vfadd_vv(V20, V20, V22);
    a.vfadd_vv(V24, V24, V26);
    a.vfadd_vv(V20, V20, V24);
    vsilu(&mut a, &k, fzero, f88, fone, V20, V28, [V22, V24, V26]);
    a.vse(Sew::E32, V20, A0);
    a.slli(T1, T0, 2);
    a.add(A0, A0, T1);
    a.slli(T1, T0, 3);
    a.add(A1, A1, T1);
    a.slli(T1, T0, 4);
    a.add(A2, A2, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, top);
    a.ret();
    a.finish().map(|c| c.bytes).map_err(backend_err(format_args!("conv K={}", spec.k)))
}

/// The forms of the recurrence this target emits: the one first written (the
/// ARM variants are refused), optionally on a slice of the value head.
pub fn check_recurrence(spec: &RecurrenceSpec) -> std::result::Result<(), Decline> {
    if spec.store_once || spec.pairs || spec.q_hoist || spec.pass1_lanes != 32 {
        return Err(Decline::OutOfRegisters);
    }
    Ok(())
}

/// The recurrence of one V head at one position, or of one slice of its value
/// columns (`spec.width`).
///
/// `a0 = s, a1 = out, a2 = q, a3 = k, a4 = v, a5 = hkd; fa0 = beta, fa1 = decay, fa2 = scale`.
///
/// Groups (`e32, m8`): `v0` the product, `v8` `kv` then `δ`, `v16` the state
/// row, `v24` `out`. Scalars: `ft0` = `k[kd]`, `ft1` = `q[kd]·scale`.
pub fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit riscv64 recurrence: {d}")))?;
    let hvd = spec.hvd;
    let lanes = if spec.width == 0 { hvd } else { spec.width };
    let line = (hvd * 4) as i64;
    let slice = (2 * crate::core::isa::vlenb()).min(lanes);
    let (prod, kv, st, out) = (V0, V8, V16, V24);
    let (ts, tk, tq, tn, tvl) = (T3, T4, T5, A6, A7);
    let mut a = Asm::new();
    let mut so = 0;
    while so < lanes {
        let width = slice.min(lanes - so);
        a.li(tvl, width as i64);
        a.vsetvli(ZERO, tvl, E32M8);
        // ── Pass 1: decay + read ──────────────────────────────────────
        a.vmv_v_i(kv, 0);
        add_imm(&mut a, ts, A0, (so * 4) as i64, T6);
        a.mv(tk, A3);
        a.mv(tn, A5);
        let pass1 = a.here();
        a.flw(FT0, tk, 0);
        a.vle(Sew::E32, st, ts);
        a.vfmul_vf(st, st, FA1);
        a.vse(Sew::E32, st, ts);
        if spec.fma {
            a.vfmacc_vf(kv, FT0, st);
        } else {
            a.vfmul_vf(prod, st, FT0);
            a.vfadd_vv(kv, kv, prod);
        }
        a.addi(tk, tk, 4);
        add_imm(&mut a, ts, ts, line, T6);
        a.addi(tn, tn, -1);
        a.bnez(tn, pass1);
        // ── δ = (v − kv) · beta ───────────────────────────────────────
        add_imm(&mut a, T0, A4, (so * 4) as i64, T6);
        a.vle(Sew::E32, prod, T0);
        a.vfsub_vv(prod, prod, kv);
        a.vfmul_vf(kv, prod, FA0);
        // ── Pass 2: write + query ─────────────────────────────────────
        a.vmv_v_i(out, 0);
        add_imm(&mut a, ts, A0, (so * 4) as i64, T6);
        a.mv(tk, A3);
        a.mv(tq, A2);
        a.mv(tn, A5);
        let pass2 = a.here();
        a.flw(FT0, tk, 0);
        a.flw(FT1, tq, 0);
        a.fmul_s(FT1, FT1, FA2);
        a.vle(Sew::E32, st, ts);
        if spec.fma {
            a.vfmacc_vf(st, FT0, kv);
        } else {
            a.vfmul_vf(prod, kv, FT0);
            a.vfadd_vv(st, st, prod);
        }
        a.vse(Sew::E32, st, ts);
        if spec.fma {
            a.vfmacc_vf(out, FT1, st);
        } else {
            a.vfmul_vf(prod, st, FT1);
            a.vfadd_vv(out, out, prod);
        }
        a.addi(tk, tk, 4);
        a.addi(tq, tq, 4);
        add_imm(&mut a, ts, ts, line, T6);
        a.addi(tn, tn, -1);
        a.bnez(tn, pass2);
        add_imm(&mut a, T0, A1, (so * 4) as i64, T6);
        a.vse(Sew::E32, out, T0);
        so += width;
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("recurrence hvd={hvd} fma={} width={}", spec.fma, spec.width)))
}

/// The gated RMS norm over `n_rows` rows of `dim`.
///
/// `a0 = out (f32, in place), a1 = z, a2 = w (bf16), a3 = out_bf16, a4 = n_rows, fa0 = eps`.
pub fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit riscv64 gated norm: {d}")))?;
    let dim = spec.dim;
    let lanes = sum_lanes().min(dim);
    let mut a = Asm::new();
    let k = ExpConsts::standard(T6);
    k.load(&mut a, T2);
    let (fzero, f88, fone) = silu_consts(&mut a, T2);
    fconst(&mut a, FT10, dim as f32, T2);
    // Groups at e32 m4: v4 x, v8 Σ lanes, v12 reduction, v16 w, v20 z, v24..v28 scratch.
    let (x, acc, red, w, z) = (V4, V8, V12, V16, V20);
    let (rem, tx, tz, tw, tb) = (T4, T3, T5, A5, A6);
    let row = a.here();
    // Pass 1: Σx² in `lanes` lanes, fused, then in order.
    a.li(T2, lanes as i64);
    a.vsetvli(ZERO, T2, E32M4);
    a.vmv_v_i(acc, 0);
    a.mv(tx, A0);
    a.li(rem, dim as i64);
    let sum = a.here();
    a.minu(T1, rem, T2);
    a.vsetvli(T0, T1, E32M4.tu());
    a.vle(Sew::E32, x, tx);
    a.vfmacc_vv(acc, x, x);
    a.slli(T1, T0, 2);
    a.add(tx, tx, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, sum);
    a.vsetvli(ZERO, T2, E32M4);
    a.vmv_s_x(red, ZERO);
    a.vfredosum_vs(red, acc, red);
    a.vfmv_f_s(FT11, red);
    // inv = 1 / sqrt(Σx²/dim + eps): the literal form.
    a.fdiv_s(FT11, FT11, FT10);
    a.fadd_s(FT11, FT11, FA0);
    a.fsqrt_s(FT11, FT11);
    a.fdiv_s(FT11, fone, FT11);
    // Pass 2: y = ((x · inv) · w) · silu(z), f32 in place and bf16.
    a.mv(tx, A0);
    a.mv(tz, A1);
    a.mv(tw, A2);
    a.mv(tb, A3);
    a.li(rem, dim as i64);
    let apply = a.here();
    a.vsetvli(T0, rem, E16M2);
    a.vle(Sew::E16, V0, tw);
    a.vfwcvtbf16_f_f_v(w, V0);
    a.vsetvli(ZERO, ZERO, E32M4);
    a.vle(Sew::E32, x, tx);
    a.vfmul_vf(x, x, FT11);
    a.vfmul_vv(x, x, w);
    a.vle(Sew::E32, z, tz);
    vsilu(&mut a, &k, fzero, f88, fone, z, V24, [V12, V16, V28]);
    a.vfmul_vv(x, x, z);
    a.vse(Sew::E32, x, tx);
    a.vsetvli(ZERO, ZERO, E16M2);
    a.vfncvtbf16_f_f_w(V0, x);
    a.vse(Sew::E16, V0, tb);
    a.slli(T1, T0, 2);
    a.add(tx, tx, T1);
    a.add(tz, tz, T1);
    a.slli(T1, T0, 1);
    a.add(tw, tw, T1);
    a.add(tb, tb, T1);
    a.sub(rem, rem, T0);
    a.bnez(rem, apply);
    add_imm(&mut a, A0, A0, (dim * 4) as i64, T2);
    add_imm(&mut a, A1, A1, (dim * 4) as i64, T2);
    add_imm(&mut a, A3, A3, (dim * 2) as i64, T2);
    a.addi(A4, A4, -1);
    a.bnez(A4, row);
    a.ret();
    a.finish().map(|c| c.bytes).map_err(backend_err(format_args!("gated norm dim={dim}")))
}
