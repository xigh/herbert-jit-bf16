//! Scalar f32 exp/log emitted as AArch64 instructions; no libm runtime calls.
//! Algorithm/coefficients: Arm optimized-routines, MIT option (license and
//! pinned revision in `third_party/arm-optimized-routines`) — the algorithm of
//! glibc's `expf`/`logf`, evaluated in double. The operations and their order
//! are those of the x86-64 emission (`target/x86/execution/math.rs`),
//! instruction for instruction: the same IEEE double arithmetic, so the same
//! bits on both targets.
use crate::jit::execution::math_data::*;
use a64::*;

fn constant(a: &mut Asm, dst: u8, bits: u64) {
    a.mov_imm(X16, bits);
    a.fmov_from_gp(d(dst), X16);
}

/// `(tables: *const Tables, x: f32) -> f32` — tables in `x0`, `x` in `s0`, the
/// result in `s0`. Clobbers `x1..x3`, `x16`, `d1..d6`; `x0` is preserved, so a
/// caller can chain calls on the same tables. Ends with `ret`.
pub(crate) fn exp_body(a: &mut Asm, fma: bool) {
    let core = a.label();
    let special = a.label();
    let zero = a.label();
    let huge = a.label();
    let nan_inf = a.label();
    a.fmov_to_gp(W2, s(0));
    a.and(W1, W2, 0x7fff_ffffu32);
    a.mov_imm(W3, u64::from(88.0f32.to_bits()));
    a.cmp(W1, W3);
    a.bhs(special);
    a.bind(core);
    a.fcvt(d(2), s(0));
    constant(a, 1, INV);
    constant(a, 3, SHIFT);
    constant(a, 4, 1.0f64.to_bits());
    if fma {
        a.fmadd(d(0), d(1), d(2), d(3));
    } else {
        a.fmul(d(0), d(1), d(2));
        a.fadd(d(0), d(0), d(3));
    }
    a.fmov_to_gp(X1, d(0));
    a.fsub(d(0), d(0), d(3));
    a.and(X2, X1, 31);
    a.lsl(X1, X1, 47);
    if fma {
        a.fnmsub(d(1), d(1), d(2), d(0));
    } else {
        a.fmul(d(1), d(1), d(2));
        a.fsub(d(1), d(1), d(0));
    }
    a.ldr(X3, idx_scaled(X0, X2));
    a.add(X1, X1, X3);
    a.fmov_from_gp(d(5), X1);
    constant(a, 0, E0);
    constant(a, 6, E1);
    if fma {
        a.fmadd(d(0), d(0), d(1), d(6));
    } else {
        a.fmul(d(0), d(0), d(1));
        a.fadd(d(0), d(0), d(6));
    }
    a.fmul(d(2), d(1), d(1));
    constant(a, 6, E2);
    if fma {
        a.fmadd(d(1), d(1), d(6), d(4));
        a.fmadd(d(0), d(0), d(2), d(1));
    } else {
        a.fmul(d(1), d(1), d(6));
        a.fadd(d(1), d(1), d(4));
        a.fmul(d(0), d(0), d(2));
        a.fadd(d(0), d(0), d(1));
    }
    a.fmul(d(0), d(0), d(5));
    a.fcvt(s(0), d(0));
    a.ret();
    a.bind(special);
    a.mov_imm(W3, 0xff80_0000);
    a.cmp(W2, W3);
    a.beq(zero);
    a.mov_imm(W3, 0x7f80_0000);
    a.cmp(W1, W3);
    a.bhs(nan_inf);
    a.mov_imm(W3, u64::from(88.722_84_f32.to_bits()));
    a.fmov_from_gp(s(1), W3);
    a.fcmp(s(0), Some(s(1)));
    a.bgt(huge);
    a.mov_imm(W3, u64::from((-103.972_084_f32).to_bits()));
    a.fmov_from_gp(s(1), W3);
    a.fcmp(s(0), Some(s(1)));
    a.bmi(zero);
    a.b(core);
    a.bind(nan_inf);
    a.fadd(s(0), s(0), s(0));
    a.ret();
    a.bind(huge);
    a.mov_imm(W3, 0x7f80_0000);
    a.fmov_from_gp(s(0), W3);
    a.ret();
    a.bind(zero);
    a.fmov_from_gp(s(0), WZR);
    a.ret();
}

/// Same ABI as [`exp_body`]. Includes zeros, subnormals, infinities and NaNs.
/// Floating errno is not part of the inference ABI.
pub(crate) fn log_body(a: &mut Asm, fma: bool) {
    let core = a.label();
    let special = a.label();
    let zero = a.label();
    let one = a.label();
    let invalid = a.label();
    let ret = a.label();
    a.fmov_to_gp(W1, s(0));
    a.mov_imm(W3, u64::from(1.0f32.to_bits()));
    a.cmp(W1, W3);
    a.beq(one);
    a.sub(W2, W1, 0x80_0000);
    a.mov_imm(W3, 0x7eff_ffff);
    a.cmp(W2, W3);
    a.bhi(special);
    a.bind(core);
    // tmp = ix − OFF; i = (tmp >> 19) % 16; k = tmp >> 23 (arithmetic); iz = ix − (tmp & 0xff800000).
    a.mov_imm(W3, 0x3f33_0000);
    a.sub(W2, W1, W3);
    a.lsr(W3, W2, 19);
    a.and(W3, W3, 15);
    a.lsl(W3, W3, 4);
    a.and(W17, W2, 0xff80_0000u32);
    a.asr(W2, W2, 23);
    a.sub(W1, W1, W17);
    a.fmov_from_gp(s(1), W1);
    a.fcvt(d(1), s(1));
    constant(a, 5, (-1.0f64).to_bits());
    a.add(X3, X0, X3);
    a.ldr(d(6), mem(X3, 256));
    if fma {
        a.fmadd(d(1), d(1), d(6), d(5));
    } else {
        a.fmul(d(1), d(1), d(6));
        a.fadd(d(1), d(1), d(5));
    }
    a.scvtf_gp(d(2), W2);
    constant(a, 6, LN2);
    a.ldr(d(5), mem(X3, 264));
    if fma {
        a.fmadd(d(2), d(2), d(6), d(5));
    } else {
        a.fmul(d(2), d(2), d(6));
        a.fadd(d(2), d(2), d(5));
    }
    constant(a, 3, L1);
    constant(a, 6, L2);
    if fma {
        a.fmadd(d(3), d(3), d(1), d(6));
    } else {
        a.fmul(d(3), d(3), d(1));
        a.fadd(d(3), d(3), d(6));
    }
    a.fmul(d(0), d(1), d(1));
    a.fadd(d(1), d(1), d(2));
    constant(a, 6, L0);
    if fma {
        a.fmadd(d(3), d(0), d(6), d(3));
        a.fmadd(d(0), d(0), d(3), d(1));
    } else {
        a.fmul(d(6), d(6), d(0));
        a.fadd(d(3), d(3), d(6));
        a.fmul(d(0), d(0), d(3));
        a.fadd(d(0), d(0), d(1));
    }
    a.fcvt(s(0), d(0));
    a.ret();
    a.bind(special);
    a.add(W2, W1, W1);
    a.cbz(W2, zero);
    a.mov_imm(W3, 0x7f80_0000);
    a.cmp(W1, W3);
    a.beq(ret);
    a.mov_imm(W3, 0xff00_0000);
    a.cmp(W2, W3);
    a.bhs(invalid);
    a.tbnz(W1, 31, invalid);
    // Subnormal: normalise by 2^23 and take 23 off the exponent.
    a.mov_imm(W3, u64::from(8_388_608.0f32.to_bits()));
    a.fmov_from_gp(s(1), W3);
    a.fmul(s(0), s(0), s(1));
    a.fmov_to_gp(W1, s(0));
    a.mov_imm(W3, 23 << 23);
    a.sub(W1, W1, W3);
    a.b(core);
    a.bind(zero);
    a.mov_imm(W3, 0xff80_0000);
    a.fmov_from_gp(s(0), W3);
    a.ret();
    a.bind(invalid);
    a.fsub(s(0), s(0), s(0));
    a.fdiv(s(0), s(0), s(0));
    a.ret();
    a.bind(one);
    a.fmov_from_gp(s(0), WZR);
    a.bind(ret);
    a.ret();
}
