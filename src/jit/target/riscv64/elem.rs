//! The elementwise kernels of a layer, RVV: one blob per `(op, dim)`.
//!
//! bf16 in memory, f32 in registers: the body is a strip-mined loop, a
//! `vsetvli` on the elements left at `e16, m2`, `vfwcvtbf16` widening them
//! exactly into an `e32, m4` group, IEEE single precision in between (fused
//! where the x86 kernel fuses), `vfncvtbf16` narrowing back to nearest even,
//! denormals kept (measured). The constants live in scalar
//! floating-point registers and enter through the `.vf` forms.
//!
//! The exponential is the polynomial of the x86 and ARM kernels
//! (`exp_constants.rs`, the same Horner chain and clamps): `k` rounded to
//! nearest even by `vfcvt.x.f.v` under the default `frm`.
//!
//! **The one folded sum**, Σx² of the norms, accumulates element `i` in lane
//! `i mod W` of an `e32, m4` group, then adds the lanes in order by
//! `vfredosum` — whose order the specification fixes, so the bits depend on
//! `W` and never on the machine. `W` is the knob
//! [`crate::jit::emit::RvvWidth`] (`--rvv-width`): every lane of the group
//! (VLMAX, VLEN / 8) or a fixed count.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use crate::jit::target::riscv64::util::backend_err;
use rv::*;

pub(crate) use crate::jit::exp_constants::{
    EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
};
/// Upper bound of the argument where it can be positive (see the x86 `EXP_MAX88`).
pub(crate) const EXP_MAX88: f32 = 88.0;
/// The two constants of the `tanh` form of GELU.
pub(crate) const GELU_K: f32 = 1.595_769;
pub(crate) const GELU_C: f32 = 0.044_715;

/// The loads and stores of bf16, and the f32 they widen to.
pub(crate) const E16: Vtype = Vtype::new(Sew::E16, Lmul::M2);
pub(crate) const E32: Vtype = Vtype::new(Sew::E32, Lmul::M4);

/// The lanes of the Σx² accumulator on this machine: [`crate::jit::emit::RvvWidth`]
/// capped by VLMAX at `e32, m4` (which is `vlenb`).
pub(crate) fn sum_lanes() -> usize {
    crate::jit::knobs::knobs().rvv.width.lanes(crate::core::isa::vlenb())
}

/// The scalar registers of the `exp` constants.
#[derive(Clone, Copy)]
pub(crate) struct ExpConsts {
    pub neg87: F,
    pub log2e: F,
    pub c4: F,
    pub c3: F,
    pub c2: F,
    pub c1: F,
    pub c0: F,
    /// `0x3F80_0000` in an integer register: the exponent bias of the reconstruction.
    pub one_bits: X,
}

impl ExpConsts {
    /// `ft0..ft6` and the given integer register.
    pub(crate) const fn standard(one_bits: X) -> Self {
        ExpConsts { neg87: FT0, log2e: FT1, c4: FT2, c3: FT3, c2: FT4, c1: FT5, c0: FT6, one_bits }
    }
    /// Materialise them, through `tmp`.
    pub(crate) fn load(&self, a: &mut Asm, tmp: X) {
        for (f, v) in [
            (self.neg87, EXP_NEG87),
            (self.log2e, EXP_LOG2E),
            (self.c4, EXP_C4),
            (self.c3, EXP_C3),
            (self.c2, EXP_C2),
            (self.c1, EXP_C1),
            (self.c0, EXP_C0),
        ] {
            fconst(a, f, v, tmp);
        }
        a.li(self.one_bits, i64::from(EXP_ONE_BITS));
    }
}

/// `f = value`, through the integer register `tmp`.
pub(crate) fn fconst(a: &mut Asm, f: F, value: f32, tmp: X) {
    a.li(tmp, i64::from(value.to_bits()));
    a.fmv_w_x(f, tmp);
}

/// `io = exp(io)` on the current `vl` (at SEW 32), for `io ≤ 88`: the chain of
/// the x86 `exp_inline`. Three temporary groups.
pub(crate) fn vexp(a: &mut Asm, k: &ExpConsts, io: V, t: [V; 3]) {
    let [t1, t2, t3] = t;
    a.vfmax_vf(io, io, k.neg87);
    a.vfmul_vf(t1, io, k.log2e);
    a.vfcvt_x_f_v(t2, t1);
    a.vfcvt_f_x_v(t3, t2);
    a.vfsub_vv(t1, t1, t3);
    // t3 = c4·f + c3, io = t3·f + c2, t3 = io·f + c1, io = t3·f + c0.
    a.vfmv_v_f(t3, k.c3);
    a.vfmacc_vf(t3, k.c4, t1);
    a.vfmv_v_f(io, k.c2);
    a.vfmacc_vv(io, t3, t1);
    a.vfmv_v_f(t3, k.c1);
    a.vfmacc_vv(t3, io, t1);
    a.vfmv_v_f(io, k.c0);
    a.vfmacc_vv(io, t3, t1);
    a.vsll_vi(t2, t2, 23);
    a.vadd_vx(t2, t2, k.one_bits);
    a.vfmul_vv(io, io, t2);
}

/// A strip-mined loop over `count` elements in `rem`: `vsetvli t0` on what is
/// left at `E16` (or at most `cap` elements, a register), then `body`, which
/// must leave `vtype` at `E16`; `step(a)` advances the pointers by `t0`
/// elements (`t1` = `2·t0` bytes is set for it).
pub(crate) fn strip(a: &mut Asm, count: usize, rem: X, cap: Option<X>, mut body: impl FnMut(&mut Asm), mut step: impl FnMut(&mut Asm)) {
    a.li(rem, count as i64);
    let top = a.here();
    match cap {
        Some(c) => {
            a.minu(T2, rem, c);
            a.vsetvli(T0, T2, E16.tu());
        }
        None => a.vsetvli(T0, rem, E16),
    }
    body(a);
    a.slli(T1, T0, 1);
    step(a);
    a.sub(rem, rem, T0);
    a.bnez(rem, top);
}

// The groups: two bf16 inputs at e16 m2, seven f32 groups at e32 m4.
const BA: V = V0;
const BB: V = V2;
const F1: V = V4;
const F2: V = V8;
const F3: V = V12;
const F4: V = V16;
const F5: V = V20;
const F6: V = V24;
const F7: V = V28;

/// Generates the kernel of `spec`.
pub fn emit(spec: &ElemSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit riscv64 elem: {d}")))?;
    let dim = spec.dim;
    let mut a = Asm::new();
    let rem = T4;
    match spec.op {
        // ── y = x · rsqrt(Σx²/d + ε) · w   (· (1+w) in the Gemma variant) ──
        // a0 = x, a1 = w, a2 = y, fa0 = eps.
        ElemOp::RmsNorm | ElemOp::RmsNormGemma => {
            let plus_one = spec.op == ElemOp::RmsNormGemma;
            let lanes = sum_lanes().min(dim);
            // Pass 1: Σx² in `lanes` lanes, fused, then in order.
            a.li(T3, lanes as i64);
            a.vsetvli(ZERO, T3, E32);
            a.vmv_v_i(F7, 0);
            a.mv(T5, A0);
            strip(&mut a, dim, rem, Some(T3), |a| {
                a.vle(Sew::E16, BA, T5);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vsetvli(ZERO, ZERO, E32.tu());
                a.vfmacc_vv(F7, F1, F1);
                a.vsetvli(ZERO, ZERO, E16.tu());
            }, |a| a.add(T5, T5, T1));
            a.vsetvli(ZERO, T3, E32);
            a.vmv_s_x(F6, ZERO);
            a.vfredosum_vs(F6, F7, F6);
            a.vfmv_f_s(FT0, F6);
            // inv = 1 / sqrt(Σx² · (1/dim) + eps), in scalar.
            fconst(&mut a, FT1, 1.0 / dim as f32, T3);
            a.fmul_s(FT0, FT0, FT1);
            a.fadd_s(FT0, FT0, FA0);
            a.fsqrt_s(FT0, FT0);
            fconst(&mut a, FT1, 1.0, T3);
            a.fdiv_s(FT0, FT1, FT0);
            // Pass 2: y = (x · inv) · w.
            strip(&mut a, dim, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vle(Sew::E16, BB, A1);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vfwcvtbf16_f_f_v(F2, BB);
                a.vsetvli(ZERO, ZERO, E32);
                if plus_one {
                    a.vfadd_vf(F2, F2, FT1);
                }
                a.vfmul_vf(F1, F1, FT0);
                a.vfmul_vv(F1, F1, F2);
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F1);
                a.vse(Sew::E16, BA, A2);
            }, |a| {
                a.add(A0, A0, T1);
                a.add(A1, A1, T1);
                a.add(A2, A2, T1);
            });
        }

        // ── y = gate · σ(gate) · up, and its two variants ────────────────
        // a0 = gate, a1 = up, a2 = y. SwiGLU: σ(g); geluTanh: σ(g·(K + K·C·g²));
        // SigmoidMul: y = a · σ(b), σ of the SECOND operand, no final product.
        ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul => {
            let cubic = spec.op == ElemOp::GeluTanh;
            let gate_out = spec.op == ElemOp::SigmoidMul;
            let k = ExpConsts::standard(T6);
            k.load(&mut a, T3);
            fconst(&mut a, FT7, EXP_MAX88, T3);
            fconst(&mut a, FT8, 1.0, T3);
            if cubic {
                fconst(&mut a, FT9, GELU_K, T3);
                fconst(&mut a, FT10, GELU_K * GELU_C, T3);
            }
            strip(&mut a, dim, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vle(Sew::E16, BB, A1);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vfwcvtbf16_f_f_v(F2, BB);
                a.vsetvli(ZERO, ZERO, E32);
                // F3 ← the sigmoid argument, negated.
                if gate_out {
                    a.vfsgnjn_vv(F3, F2, F2);
                } else if cubic {
                    a.vfmul_vv(F3, F1, F1);
                    a.vfmul_vf(F3, F3, FT10);
                    a.vfadd_vf(F3, F3, FT9);
                    a.vfmul_vv(F3, F3, F1);
                    a.vfsgnjn_vv(F3, F3, F3);
                } else {
                    a.vfsgnjn_vv(F3, F1, F1);
                }
                a.vfmin_vf(F3, F3, FT7);
                vexp(a, &k, F3, [F4, F5, F6]);
                a.vfadd_vf(F3, F3, FT8);
                a.vfdiv_vv(F7, F1, F3);
                if !gate_out {
                    a.vfmul_vv(F7, F7, F2);
                }
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F7);
                a.vse(Sew::E16, BA, A2);
            }, |a| {
                a.add(A0, A0, T1);
                a.add(A1, A1, T1);
                a.add(A2, A2, T1);
            });
        }

        // ── RoPE, rotate_half convention ────────────────────────────────
        // a0 = x (read and written), a1 = cos, a2 = sin; pairs (i, i + dim/2).
        ElemOp::Rope => {
            let half = dim / 2;
            a.addi(A3, A0, (half * 2) as i64);
            strip(&mut a, half, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vle(Sew::E16, BB, A3);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vfwcvtbf16_f_f_v(F2, BB);
                a.vle(Sew::E16, BA, A1);
                a.vle(Sew::E16, BB, A2);
                a.vfwcvtbf16_f_f_v(F3, BA);
                a.vfwcvtbf16_f_f_v(F4, BB);
                a.vsetvli(ZERO, ZERO, E32);
                // a' = a·c − b·s ; b' = a·s + b·c
                a.vfmul_vv(F5, F1, F3);
                a.vfnmsac_vv(F5, F2, F4);
                a.vfmul_vv(F6, F1, F4);
                a.vfmacc_vv(F6, F2, F3);
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F5);
                a.vfncvtbf16_f_f_w(BB, F6);
                a.vse(Sew::E16, BA, A0);
                a.vse(Sew::E16, BB, A3);
            }, |a| {
                for r in [A0, A3, A1, A2] {
                    a.add(r, r, T1);
                }
            });
        }

        // ── a += b ──────────────────────────────────────────────────────
        ElemOp::ResidualAdd => {
            strip(&mut a, dim, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vle(Sew::E16, BB, A1);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vfwcvtbf16_f_f_v(F2, BB);
                a.vsetvli(ZERO, ZERO, E32);
                a.vfadd_vv(F1, F1, F2);
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F1);
                a.vse(Sew::E16, BA, A0);
            }, |a| {
                a.add(A0, A0, T1);
                a.add(A1, A1, T1);
            });
        }

        // ── y = x · s ───────────────────────────────────────────────────
        ElemOp::Scale => {
            strip(&mut a, dim, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vsetvli(ZERO, ZERO, E32);
                a.vfmul_vf(F1, F1, FA0);
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F1);
                a.vse(Sew::E16, BA, A1);
            }, |a| {
                a.add(A0, A0, T1);
                a.add(A1, A1, T1);
            });
        }

        // ── y = cap · tanh(x / cap) ─────────────────────────────────────
        // `tanh(u) = (1 − e^{−2u})/(1 + e^{−2u})` on `|u|`, the sign restored.
        ElemOp::Softcap => {
            let k = ExpConsts::standard(T6);
            k.load(&mut a, T3);
            fconst(&mut a, FT7, -2.0, T3);
            a.fdiv_s(FT7, FT7, FA0);
            fconst(&mut a, FT8, 1.0, T3);
            a.li(T5, 0x8000_0000);
            strip(&mut a, dim, rem, None, |a| {
                a.vle(Sew::E16, BA, A0);
                a.vfwcvtbf16_f_f_v(F1, BA);
                a.vsetvli(ZERO, ZERO, E32);
                a.vand_vx(F2, F1, T5);
                a.vfsgnjx_vv(F1, F1, F1);
                a.vfmul_vf(F1, F1, FT7);
                vexp(a, &k, F1, [F4, F5, F6]);
                a.vfrsub_vf(F3, F1, FT8);
                a.vfadd_vf(F7, F1, FT8);
                a.vfdiv_vv(F1, F3, F7);
                a.vfmul_vf(F1, F1, FA0);
                a.vor_vv(F1, F1, F2);
                a.vsetvli(ZERO, ZERO, E16);
                a.vfncvtbf16_f_f_w(BA, F1);
                a.vse(Sew::E16, BA, A1);
            }, |a| {
                a.add(A0, A0, T1);
                a.add(A1, A1, T1);
            });
        }
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("elem {} dim={}", spec.op.name(), spec.dim)))
}
