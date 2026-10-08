//! The GatedDeltaNet kernels, NEON: the causal conv + SiLU, the recurrence
//! and the gated RMS norm, with the ABIs of `jit/emit_conv.rs`,
//! `jit/emit_recurrence.rs` and `jit/emit_gated_norm.rs`.
//!
//! - **conv**: sixteen channels per block, four at a time: the state of a
//!   channel is one `q` register (its four taps), the shift-and-insert is one
//!   `ext` against the broadcast input, the four taps sum as a tree,
//!   `(p0+p1)+(p2+p3)`, by three `faddp` over four channels. The state comes
//!   back bit for bit; the output is within the tolerance of the emitted `exp`.
//! - **recurrence**: the value head cut into slices of 32 lanes; per slice,
//!   the two passes of the specification over the key dimension. Without FMA
//!   every operation is the scalar reference's, in its order: the same bits.
//! - **gated norm**: `Σx²` in four chains, the literal
//!   `1 / √(Σx²/dim + eps)`, then `((x · inv) · w) · silu(z)`.

use crate::core::error::{HerbertError, Result};
use crate::jit::emit_conv::ConvSpec;
use crate::jit::emit_gated_norm::GatedNormSpec;
use crate::jit::emit_recurrence::RecurrenceSpec;
use crate::jit::target::aarch64::elem::{narrow, vexp, widen, ExpRegs, EXP_MAX88};
use crate::jit::target::aarch64::util::{add_imm, backend_err, splat, Frame};
use a64::*;

/// The conv of one position over `n_blocks` blocks of sixteen channels.
pub fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit conv: {d}")))?;
    let mut a = Asm::new();
    // Constants in v8..v15 (callee-saved: the prologue keeps their low halves).
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    let k = ExpRegs::at(8);
    k.load(&mut a, X9);
    splat(&mut a, 26, EXP_MAX88.to_bits(), X9);
    let top = a.label();
    a.bind(top);
    for sub in 0..4i64 {
        // x (4 channels), state (4 × 4 taps), weights (4 × 4 bf16).
        a.ldr(q(4), mem(X0, 16 * sub));
        a.ldp(q(0), q(1), mem(X2, 64 * sub));
        a.ldp(q(2), q(3), mem(X2, 64 * sub + 32));
        for c in 0..4u8 {
            a.dup(v(5).s4(), v(4).s(c));
            a.ext(v(c).b16(), v(c).b16(), v(5).b16(), 4);
        }
        a.stp(q(0), q(1), mem(X2, 64 * sub));
        a.stp(q(2), q(3), mem(X2, 64 * sub + 32));
        a.ldp(q(6), q(7), mem(X1, 32 * sub));
        widen(&mut a, 16, 6, false);
        widen(&mut a, 17, 6, true);
        widen(&mut a, 18, 7, false);
        widen(&mut a, 19, 7, true);
        for c in 0..4u8 {
            a.fmul(v(20 + c).s4(), v(c).s4(), v(16 + c).s4());
        }
        a.faddp(v(24).s4(), v(20).s4(), Some(v(21).s4()));
        a.faddp(v(25).s4(), v(22).s4(), Some(v(23).s4()));
        a.faddp(v(24).s4(), v(24).s4(), Some(v(25).s4()));
        // y / (1 + exp(min(−y, 88)))
        a.fneg(v(25).s4(), v(24).s4());
        a.fmin(v(25).s4(), v(25).s4(), v(26).s4());
        vexp(&mut a, &k, 25, [16, 17, 18, 19]);
        a.fadd(v(25).s4(), v(25).s4(), v(k.one).s4());
        a.fdiv(v(24).s4(), v(24).s4(), v(25).s4());
        a.str(q(24), mem(X0, 16 * sub));
    }
    a.add(X0, X0, 64);
    a.add(X1, X1, 128);
    a.add(X2, X2, 256);
    a.subs(X3, X3, 1);
    a.bne(top);
    frame.leave(&mut a);
    a.ret();
    a.finish().map(|c| c.bytes).map_err(backend_err(format_args!("conv K={}", spec.k)))
}

/// The recurrence register budget: any `hvd` multiple of 16 is served by slices.
pub fn check_recurrence(_spec: &RecurrenceSpec) -> std::result::Result<(), crate::jit::ir::Decline> {
    Ok(())
}

/// The recurrence of one V head at one position, or of one slice of its
/// value columns (`spec.width`).
///
/// `x0 = s, x1 = out, x2 = q, x3 = k, x4 = v, x5 = hkd; s0 = beta, s1 = decay, s2 = scale`.
///
/// Registers: `v28` decay, `v29` beta, `v30` scale; the first pass
/// accumulates `kv` in `v0..v7` (32 lanes) or `v0..v7, v20..v27` (64); the
/// second pass reads `δ` there and accumulates `out` in `v20..v27` (or in
/// `v8..v15`, saved, when the first pass took 64 lanes); `v16` is `k[kd]`,
/// `v17` is `q[kd]·scale`, `v18` (and `v19` by pairs) the state row, `v19`
/// (or `v31` by pairs) the product. With `q_hoist`, `q·scale` is written once
/// per position into 2 KiB of stack and broadcast from there.
pub fn emit_recurrence(spec: &RecurrenceSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit recurrence: {d}")))?;
    let hvd = spec.hvd;
    let lanes = if spec.width == 0 { hvd } else { spec.width };
    let line = (hvd * 4) as i64;
    let wide = spec.pass1_lanes == 64;
    let pairs = spec.pairs;
    let product = if pairs { 31u8 } else { 19 };
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], wide, if spec.q_hoist { 2048 } else { 0 });
    a.dup(v(29).s4(), v(0).s(0));
    a.dup(v(28).s4(), v(1).s(0));
    a.dup(v(30).s4(), v(2).s(0));
    if spec.q_hoist {
        // qs[kd] = q[kd] · scale, four at a time then one at a time, at [sp].
        let (quads, quad_loop, singles, single_loop, done) = (a.label(), a.label(), a.label(), a.label(), a.label());
        a.mov(X13, X2);
        a.add(X12, SP, 0);
        a.lsr(X11, X5, 2);
        a.cbz(X11, singles);
        a.bind(quads);
        a.bind(quad_loop);
        a.ldr(q(19), post(X13, 16));
        a.fmul(v(19).s4(), v(19).s4(), v(30).s4());
        a.str(q(19), post(X12, 16));
        a.subs(X11, X11, 1);
        a.bne(quad_loop);
        a.bind(singles);
        a.and(X11, X5, 3u64);
        a.cbz(X11, done);
        a.bind(single_loop);
        a.ldr(s(19), post(X13, 4));
        a.fmul(s(19), s(19), s(30));
        a.str(s(19), post(X12, 4));
        a.subs(X11, X11, 1);
        a.bne(single_loop);
        a.bind(done);
    }
    // The accumulator register of lane group `r` of the first pass.
    let acc = |r: u8| if r < 8 { r } else { 20 + r - 8 };
    let p1 = if wide { 64 } else { 32 };
    let mut so = 0;
    while so < lanes {
        let width = (lanes - so).min(p1);
        let regs = (width / 4) as u8;
        let pass1 = a.label();
        // ── Pass 1: decay + read ──────────────────────────────────────
        for r in 0..regs {
            a.movi(v(acc(r)).d2(), 0);
        }
        add_imm(&mut a, X9, X0, (so * 4) as i64, X12);
        a.mov(X10, X3);
        a.mov(X11, X5);
        a.bind(pass1);
        a.ld1r(v(16).s4(), post(X10, 4));
        let step = if pairs { 2 } else { 1 };
        for r in (0..regs).step_by(step) {
            let off = 16 * i64::from(r);
            let state: Vec<u8> = if pairs { vec![18, 19] } else { vec![18] };
            if pairs {
                a.ldp(q(18), q(19), mem(X9, off));
            } else {
                a.ldr(q(18), mem(X9, off));
            }
            for &st in &state {
                a.fmul(v(st).s4(), v(st).s4(), v(28).s4());
            }
            if !spec.store_once {
                if pairs {
                    a.stp(q(18), q(19), mem(X9, off));
                } else {
                    a.str(q(18), mem(X9, off));
                }
            }
            for (i, &st) in state.iter().enumerate() {
                let d = v(acc(r + i as u8)).s4();
                if spec.fma {
                    a.fmla(d, v(st).s4(), v(16).s4());
                } else {
                    a.fmul(v(product).s4(), v(st).s4(), v(16).s4());
                    a.fadd(d, d, v(product).s4());
                }
            }
        }
        add_imm(&mut a, X9, X9, line, X12);
        a.subs(X11, X11, 1);
        a.bne(pass1);
        // ── δ = (v − kv) · beta ───────────────────────────────────────
        for r in 0..regs {
            a.ldr(q(19), mem(X4, (so * 4) as i64 + 16 * i64::from(r)));
            a.fsub(v(19).s4(), v(19).s4(), v(acc(r)).s4());
            a.fmul(v(acc(r)).s4(), v(19).s4(), v(29).s4());
        }
        // ── Pass 2: write + query, by halves of 32 lanes ──────────────
        for half in 0..width.div_ceil(32) {
            let h0 = 32 * half;
            let hregs = ((width - h0).min(32) / 4) as u8;
            let delta = |r: u8| acc(8 * half as u8 + r);
            let out = |r: u8| if wide { 8 + r } else { 20 + r };
            let pass2 = a.label();
            for r in 0..hregs {
                a.movi(v(out(r)).d2(), 0);
            }
            add_imm(&mut a, X9, X0, ((so + h0) * 4) as i64, X12);
            a.mov(X10, X3);
            if spec.q_hoist {
                a.add(X13, SP, 0);
            } else {
                a.mov(X13, X2);
            }
            a.mov(X11, X5);
            a.bind(pass2);
            a.ld1r(v(16).s4(), post(X10, 4));
            a.ld1r(v(17).s4(), post(X13, 4));
            if !spec.q_hoist {
                a.fmul(v(17).s4(), v(17).s4(), v(30).s4());
            }
            for r in (0..hregs).step_by(step) {
                let off = 16 * i64::from(r);
                let state: Vec<u8> = if pairs { vec![18, 19] } else { vec![18] };
                if pairs {
                    a.ldp(q(18), q(19), mem(X9, off));
                } else {
                    a.ldr(q(18), mem(X9, off));
                }
                if spec.store_once {
                    // The decayed state again: the same product as in pass 1.
                    for &st in &state {
                        a.fmul(v(st).s4(), v(st).s4(), v(28).s4());
                    }
                }
                for (i, &st) in state.iter().enumerate() {
                    let d = v(delta(r + i as u8)).s4();
                    if spec.fma {
                        a.fmla(v(st).s4(), v(16).s4(), d);
                    } else {
                        a.fmul(v(product).s4(), v(16).s4(), d);
                        a.fadd(v(st).s4(), v(st).s4(), v(product).s4());
                    }
                }
                if pairs {
                    a.stp(q(18), q(19), mem(X9, off));
                } else {
                    a.str(q(18), mem(X9, off));
                }
                for (i, &st) in state.iter().enumerate() {
                    let o = v(out(r + i as u8)).s4();
                    if spec.fma {
                        a.fmla(o, v(st).s4(), v(17).s4());
                    } else {
                        a.fmul(v(product).s4(), v(st).s4(), v(17).s4());
                        a.fadd(o, o, v(product).s4());
                    }
                }
            }
            add_imm(&mut a, X9, X9, line, X12);
            a.subs(X11, X11, 1);
            a.bne(pass2);
            for r in 0..hregs {
                a.str(q(out(r)), mem(X1, ((so + h0) * 4) as i64 + 16 * i64::from(r)));
            }
        }
        so += width;
    }
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("recurrence hvd={hvd} fma={} store_once={} pairs={} q_hoist={} pass1={} width={}", spec.fma, spec.store_once, spec.pairs, spec.q_hoist, spec.pass1_lanes, spec.width)))
}

/// The gated RMS norm over `n_rows` rows of `dim`.
///
/// `x0 = out (f32, in place), x1 = z, x2 = w (bf16), x3 = out_bf16, x4 = n_rows, s0 = eps`.
pub fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit gated norm: {d}")))?;
    let dim = spec.dim;
    let quads = dim / 4;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &[], true, 0);
    let k = ExpRegs::at(8);
    k.load(&mut a, X9);
    splat(&mut a, 26, EXP_MAX88.to_bits(), X9);
    a.fmov(s(27), s(0));
    a.mov_imm(W9, u64::from((dim as f32).to_bits()));
    a.fmov_from_gp(s(28), W9);
    let row = a.label();
    a.bind(row);
    // Pass 1: Σx² in four chains, summed as a tree.
    for c in 0..4 {
        a.movi(v(c).d2(), 0);
    }
    for i in 0..quads {
        a.ldr(q(16), mem(X0, 16 * i as i64));
        let c = (i % 4) as u8;
        a.fmla(v(c).s4(), v(16).s4(), v(16).s4());
    }
    a.fadd(v(0).s4(), v(0).s4(), v(1).s4());
    a.fadd(v(2).s4(), v(2).s4(), v(3).s4());
    a.fadd(v(0).s4(), v(0).s4(), v(2).s4());
    a.faddp(v(0).s4(), v(0).s4(), Some(v(0).s4()));
    a.faddp(s(0), v(0).s2(), None);
    // inv = 1 / sqrt(Σx²/dim + eps): the literal form.
    a.fdiv(s(0), s(0), s(28));
    a.fadd(s(0), s(0), s(27));
    a.fsqrt(s(0), s(0));
    a.fmov_imm(s(1), 1.0);
    a.fdiv(s(0), s(1), s(0));
    a.dup(v(29).s4(), v(0).s(0));
    // Pass 2: y = ((x · inv) · w) · silu(z), f32 in place and bf16.
    for pair in 0..quads / 2 {
        for half in 0..2 {
            let i = 2 * pair + half;
            let off = 16 * i as i64;
            let y = 24 + half as u8;
            a.ldr(q(y), mem(X0, off));
            a.fmul(v(y).s4(), v(y).s4(), v(29).s4());
            if half == 0 {
                a.ldr(q(6), mem(X2, 16 * pair as i64));
            }
            widen(&mut a, 7, 6, half == 1);
            a.fmul(v(y).s4(), v(y).s4(), v(7).s4());
            a.ldr(q(4), mem(X1, off));
            a.fneg(v(5).s4(), v(4).s4());
            a.fmin(v(5).s4(), v(5).s4(), v(26).s4());
            vexp(&mut a, &k, 5, [16, 17, 18, 19]);
            a.fadd(v(5).s4(), v(5).s4(), v(k.one).s4());
            a.fdiv(v(4).s4(), v(4).s4(), v(5).s4());
            a.fmul(v(y).s4(), v(y).s4(), v(4).s4());
            a.str(q(y), mem(X0, off));
        }
        narrow(&mut a, 30, 24, 25);
        a.str(q(30), mem(X3, 16 * pair as i64));
    }
    add_imm(&mut a, X0, X0, (dim * 4) as i64, X9);
    add_imm(&mut a, X1, X1, (dim * 4) as i64, X9);
    add_imm(&mut a, X3, X3, (dim * 2) as i64, X9);
    a.subs(X4, X4, 1);
    a.bne(row);
    frame.leave(&mut a);
    a.ret();
    a.finish().map(|c| c.bytes).map_err(backend_err(format_args!("gated norm dim={dim}")))
}
