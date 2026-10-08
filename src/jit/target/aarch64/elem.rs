//! The elementwise kernels of a layer, NEON: one blob per `(op, dim)`.
//!
//! bf16 in memory, f32 in registers: a `q` load brings eight bf16, `shll` and
//! `shll2` widen them exactly (a bf16 is the high half of its f32), `bfcvtn`
//! and `bfcvtn2` narrow eight f32 back, rounded to nearest even. The
//! arithmetic between is IEEE single precision, fused where the x86 kernel
//! fuses. `dim` is an immediate (a multiple of 32, `ElemSpec::check`): the
//! body is a loop over blocks of 32 elements, no remainder.
//!
//! The exponential is the polynomial of the x86 kernels (`exp_constants.rs`
//! values, same Horner chain, same clamps), so an activation computes the
//! same function on both targets; only the rounding of the few operations
//! the ISAs order differently can move a last bit.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_elem::{ElemOp, ElemSpec};
use crate::jit::target::aarch64::util::{backend_err, splat};
use a64::*;

pub(crate) use crate::jit::exp_constants::{
    EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
};
/// Upper bound of the argument where it can be positive (see the x86 `EXP_MAX88`).
pub(crate) const EXP_MAX88: f32 = 88.0;

/// The registers holding the polynomial constants.
#[derive(Clone, Copy)]
pub(crate) struct ExpRegs {
    pub neg87: u8,
    pub log2e: u8,
    pub c4: u8,
    pub c3: u8,
    pub c2: u8,
    pub c1: u8,
    pub c0: u8,
    /// `0x3F80_0000`: the exponent bias of the reconstruction, and 1.0.
    pub one: u8,
}

impl ExpRegs {
    /// The constants in eight consecutive registers from `first`.
    pub(crate) fn at(first: u8) -> Self {
        ExpRegs {
            neg87: first,
            log2e: first + 1,
            c4: first + 2,
            c3: first + 3,
            c2: first + 4,
            c1: first + 5,
            c0: first + 6,
            one: first + 7,
        }
    }
    /// Materialise them, through `tmp`.
    pub(crate) fn load(&self, a: &mut Asm, tmp: R) {
        splat(a, self.neg87, EXP_NEG87.to_bits(), tmp);
        splat(a, self.log2e, EXP_LOG2E.to_bits(), tmp);
        splat(a, self.c4, EXP_C4.to_bits(), tmp);
        splat(a, self.c3, EXP_C3.to_bits(), tmp);
        splat(a, self.c2, EXP_C2.to_bits(), tmp);
        splat(a, self.c1, EXP_C1.to_bits(), tmp);
        splat(a, self.c0, EXP_C0.to_bits(), tmp);
        splat(a, self.one, EXP_ONE_BITS, tmp);
    }
}

/// `io = exp(io)` on four lanes, for `io ≤ 88` (the caller bounds a positive
/// argument). The chain of the x86 `exp_inline`: clamp at −87.34, `2^k·p(f)`
/// with `k` rounded to nearest even and a degree-4 Horner in fused
/// multiply-adds. Four temporaries.
pub(crate) fn vexp(a: &mut Asm, k: &ExpRegs, io: u8, t: [u8; 4]) {
    let [t1, t2, t3, t4] = t;
    let f = |r: u8| v(r).s4();
    a.fmax(f(io), f(io), f(k.neg87));
    a.fmul(f(t1), f(io), f(k.log2e));
    a.frintn(f(t2), f(t1));
    a.fsub(f(t1), f(t1), f(t2));
    // t3 = c4·f + c3, t4 = t3·f + c2, t3 = t4·f + c1, t4 = t3·f + c0.
    a.mov_v(v(t3).b16(), v(k.c3).b16());
    a.fmla(f(t3), f(k.c4), f(t1));
    a.mov_v(v(t4).b16(), v(k.c2).b16());
    a.fmla(f(t4), f(t3), f(t1));
    a.mov_v(v(t3).b16(), v(k.c1).b16());
    a.fmla(f(t3), f(t4), f(t1));
    a.mov_v(v(t4).b16(), v(k.c0).b16());
    a.fmla(f(t4), f(t3), f(t1));
    a.fcvtzs(f(t2), f(t2));
    a.shl(f(t2), f(t2), 23);
    a.add_v(f(t2), f(t2), f(k.one));
    a.fmul(f(io), f(t4), f(t2));
}

/// Widen the low (`high = false`) or high four bf16 of `src` into f32 `dst`.
pub(crate) fn widen(a: &mut Asm, dst: u8, src: u8, high: bool) {
    if high {
        a.shll(v(dst).s4(), v(src).h8());
    } else {
        a.shll(v(dst).s4(), v(src).h4());
    }
}

/// Narrow two f32 quads into eight bf16 in `dst`.
pub(crate) fn narrow(a: &mut Asm, dst: u8, lo: u8, hi: u8) {
    a.bfcvtn(v(dst).h4(), v(lo).s4());
    a.bfcvtn(v(dst).h8(), v(hi).s4());
}

/// A loop over `blocks` blocks of 32 bf16 whose body is `body(a)`: the
/// pointers advance inside the body (post-indexed loads and stores).
fn blocks32(a: &mut Asm, blocks: usize, counter: R, mut body: impl FnMut(&mut Asm)) {
    let top = a.label();
    a.mov_imm(counter, blocks as u64);
    a.bind(top);
    body(a);
    a.subs(counter, counter, 1);
    a.bne(top);
}

/// Generates the kernel of `spec`.
pub fn emit(spec: &ElemSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit elem: {d}")))?;
    let nb = spec.blocks32();
    let mut a = Asm::new();
    match spec.op {
        // ── y = x · rsqrt(Σx²/d + ε) · w   (· (1+w) in the Gemma variant) ──
        // x0 = x, x1 = w, x2 = y, s0 = eps.
        ElemOp::RmsNorm | ElemOp::RmsNormGemma => {
            let plus_one = spec.op == ElemOp::RmsNormGemma;
            a.fmov(s(31), s(0));
            // Pass 1: Σx² in four chains of fused products of the widened x.
            for c in 0..4 {
                a.movi(v(c).d2(), 0);
            }
            a.mov(X9, X0);
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X9, 32));
                a.ldp(q(18), q(19), post(X9, 32));
                for (i, src) in [16u8, 17, 18, 19].into_iter().enumerate() {
                    widen(a, 20, src, false);
                    widen(a, 21, src, true);
                    let c = (2 * i as u8) % 4;
                    a.fmla(v(c).s4(), v(20).s4(), v(20).s4());
                    a.fmla(v(c + 1).s4(), v(21).s4(), v(21).s4());
                }
            });
            a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
            a.fadd(v(2).s4(), v(2).s4(), v(3).s4());
            a.fadd(v(0).s4(), v(0).s4(), v(2).s4());
            a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
            a.faddp(s(0), v(0).s2(), None);
            // inv = 1 / sqrt(Σx² · (1/dim) + eps), in scalar.
            a.mov_imm(W9, u64::from((1.0f32 / spec.dim as f32).to_bits()));
            a.fmov_from_gp(s(1), W9);
            a.fmul(s(0), s(0), s(1));
            a.fadd(s(0), s(0), s(31));
            a.fsqrt(s(0), s(0));
            a.fmov_imm(s(1), 1.0);
            a.fdiv(s(0), s(1), s(0));
            a.dup(v(30).s4(), v(0).s(0));
            if plus_one {
                a.fmov_imm(v(29).s4(), 1.0);
            }
            // Pass 2: y = (x · inv) · w, eight at a time.
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                a.ldp(q(20), q(21), post(X1, 32));
                a.ldp(q(22), q(23), post(X1, 32));
                for (xb, wb, out) in [(16u8, 20u8, 24u8), (17, 21, 25), (18, 22, 26), (19, 23, 27)] {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, xb, half);
                        widen(a, dst + 2, wb, half);
                        if plus_one {
                            a.fadd(v(dst + 2).s4(), v(dst + 2).s4(), v(29).s4());
                        }
                        a.fmul(v(dst).s4(), v(dst).s4(), v(30).s4());
                        a.fmul(v(dst).s4(), v(dst).s4(), v(dst + 2).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X2, 32));
                a.stp(q(26), q(27), post(X2, 32));
            });
        }

        // ── y = gate · σ(gate) · up, and its two variants ────────────────
        // x0 = gate, x1 = up, x2 = y. SwiGLU: σ(g); geluTanh: σ(g·(K + K·C·g²));
        // SigmoidMul: y = a · σ(b), σ of the SECOND operand, no final product.
        ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul => {
            let cubic = spec.op == ElemOp::GeluTanh;
            let gate_out = spec.op == ElemOp::SigmoidMul;
            let k = ExpRegs::at(16);
            k.load(&mut a, X9);
            splat(&mut a, 24, EXP_MAX88.to_bits(), X9);
            if cubic {
                const K: f32 = 1.595_769;
                const C: f32 = 0.044_715;
                splat(&mut a, 25, K.to_bits(), X9);
                splat(&mut a, 26, (K * C).to_bits(), X9);
            }
            blocks32(&mut a, nb, X10, |a| {
                for _ in 0..4 {
                    a.ldr(q(0), post(X0, 16));
                    a.ldr(q(1), post(X1, 16));
                    for (half, out) in [(false, 28u8), (true, 29u8)] {
                        widen(a, 2, 0, half);
                        widen(a, 3, 1, half);
                        // v4 ← the sigmoid argument, negated.
                        if gate_out {
                            a.fneg(v(4).s4(), v(3).s4());
                        } else if cubic {
                            a.fmul(v(4).s4(), v(2).s4(), v(2).s4());
                            a.fmul(v(4).s4(), v(4).s4(), v(26).s4());
                            a.fadd(v(4).s4(), v(4).s4(), v(25).s4());
                            a.fmul(v(4).s4(), v(4).s4(), v(2).s4());
                            a.fneg(v(4).s4(), v(4).s4());
                        } else {
                            a.fneg(v(4).s4(), v(2).s4());
                        }
                        a.fmin(v(4).s4(), v(4).s4(), v(24).s4());
                        vexp(a, &k, 4, [5, 6, 7, 27]);
                        a.fadd(v(4).s4(), v(4).s4(), v(k.one).s4());
                        a.fdiv(v(out).s4(), v(2).s4(), v(4).s4());
                        if !gate_out {
                            a.fmul(v(out).s4(), v(out).s4(), v(3).s4());
                        }
                    }
                    narrow(a, 30, 28, 29);
                    a.str(q(30), post(X2, 16));
                }
            });
        }

        // ── RoPE, rotate_half convention ────────────────────────────────
        // x0 = x (read and written), x1 = cos, x2 = sin; pairs (i, i + dim/2).
        ElemOp::Rope => {
            let half = spec.dim / 2;
            a.add(X3, X0, (half * 2) as u64);
            for _ in (0..half).step_by(8) {
                a.ldr(q(0), at(X0));
                a.ldr(q(1), at(X3));
                a.ldr(q(2), post(X1, 16));
                a.ldr(q(3), post(X2, 16));
                for (hi, (ra, rb)) in [(false, (24u8, 25u8)), (true, (26, 27))] {
                    widen(&mut a, 4, 0, hi);
                    widen(&mut a, 5, 1, hi);
                    widen(&mut a, 6, 2, hi);
                    widen(&mut a, 7, 3, hi);
                    // a' = a·c − b·s ; b' = a·s + b·c
                    a.fmul(v(ra).s4(), v(4).s4(), v(6).s4());
                    a.fmls(v(ra).s4(), v(5).s4(), v(7).s4());
                    a.fmul(v(rb).s4(), v(4).s4(), v(7).s4());
                    a.fmla(v(rb).s4(), v(5).s4(), v(6).s4());
                }
                narrow(&mut a, 28, 24, 26);
                narrow(&mut a, 29, 25, 27);
                a.str(q(28), post(X0, 16));
                a.str(q(29), post(X3, 16));
            }
        }

        // ── a += b ──────────────────────────────────────────────────────
        ElemOp::ResidualAdd => {
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), mem(X0, 0));
                a.ldp(q(18), q(19), mem(X0, 32));
                a.ldp(q(20), q(21), post(X1, 32));
                a.ldp(q(22), q(23), post(X1, 32));
                for (ab, bb, out) in [(16u8, 20u8, 24u8), (17, 21, 25), (18, 22, 26), (19, 23, 27)] {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, ab, half);
                        widen(a, dst + 2, bb, half);
                        a.fadd(v(dst).s4(), v(dst).s4(), v(dst + 2).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X0, 32));
                a.stp(q(26), q(27), post(X0, 32));
            });
        }

        // ── y = x · s ───────────────────────────────────────────────────
        ElemOp::Scale => {
            a.dup(v(31).s4(), v(0).s(0));
            blocks32(&mut a, nb, X10, |a| {
                a.ldp(q(16), q(17), post(X0, 32));
                a.ldp(q(18), q(19), post(X0, 32));
                for (xb, out) in [(16u8, 24u8), (17, 25), (18, 26), (19, 27)] {
                    for (half, dst) in [(false, 2u8), (true, 3u8)] {
                        widen(a, dst, xb, half);
                        a.fmul(v(dst).s4(), v(dst).s4(), v(31).s4());
                    }
                    narrow(a, out, 2, 3);
                }
                a.stp(q(24), q(25), post(X1, 32));
                a.stp(q(26), q(27), post(X1, 32));
            });
        }

        // ── y = cap · tanh(x / cap) ─────────────────────────────────────
        // `tanh(u) = (1 − e^{−2u})/(1 + e^{−2u})` on `|u|`, the sign restored.
        ElemOp::Softcap => {
            a.dup(v(31).s4(), v(0).s(0));
            let k = ExpRegs::at(16);
            k.load(&mut a, X9);
            splat(&mut a, 24, (-2.0f32).to_bits(), X9);
            a.fdiv(v(25).s4(), v(24).s4(), v(31).s4());
            a.movi(v(26).s4(), 0x8000_0000);
            blocks32(&mut a, nb, X10, |a| {
                for _ in 0..4 {
                    a.ldr(q(0), post(X0, 16));
                    for (half, out) in [(false, 28u8), (true, 29u8)] {
                        widen(a, 1, 0, half);
                        a.and_v(v(2).b16(), v(1).b16(), v(26).b16());
                        a.fabs(v(1).s4(), v(1).s4());
                        a.fmul(v(1).s4(), v(1).s4(), v(25).s4());
                        vexp(a, &k, 1, [3, 4, 5, 6]);
                        a.fsub(v(3).s4(), v(k.one).s4(), v(1).s4());
                        a.fadd(v(4).s4(), v(k.one).s4(), v(1).s4());
                        a.fdiv(v(1).s4(), v(3).s4(), v(4).s4());
                        a.fmul(v(1).s4(), v(1).s4(), v(31).s4());
                        a.orr_v(v(out).b16(), v(1).b16(), v(2).b16());
                    }
                    narrow(a, 30, 28, 29);
                    a.str(q(30), post(X1, 16));
                }
            });
        }
    }
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("elem {} dim={}", spec.op.name(), spec.dim)))
}
