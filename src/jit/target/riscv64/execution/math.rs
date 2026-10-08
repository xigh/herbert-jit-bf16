//! Scalar f32 exp/log emitted as RISC-V instructions; no libm runtime calls.
//! Algorithm/coefficients: Arm optimized-routines, MIT option (license and
//! pinned revision in `third_party/arm-optimized-routines`) — the algorithm of
//! glibc's `expf`/`logf`, evaluated in double. The operations and their order
//! are those of the x86-64 and AArch64 emissions, instruction for
//! instruction: the same IEEE double arithmetic, so the same bits on every
//! target. The 32-bit words of the ARM version are zero-extended where it
//! compares them unsigned, sign-extended (`subw`) where it shifts them
//! arithmetically.
use crate::jit::execution::math_data::*;
use rv::*;

/// `dst = bits` as a double, through `t0` (free wherever the bodies call it:
/// `t1..t3` hold the table index and the exponent).
fn constant(a: &mut Asm, dst: F, bits: u64) {
    a.li(T0, bits as i64);
    a.fmv_d_x(dst, T0);
}

/// `(tables: *const Tables, x: f32) -> f32` — tables in `a0`, `x` in `fa0`,
/// the result in `fa0`. Clobbers `t0..t3`, `ft0..ft6`; `a0` is preserved, so
/// a caller can chain calls on the same tables. Ends with `ret`.
pub(crate) fn exp_body(a: &mut Asm, fma: bool) {
    let core = a.label();
    let special = a.label();
    let zero = a.label();
    let huge = a.label();
    let nan_inf = a.label();
    // t2 = the bits of x (zero-extended), t1 = its magnitude.
    a.fmv_x_w(T2, FA0);
    a.zext_w(T2, T2);
    a.li(T3, 0x7fff_ffff);
    a.and(T1, T2, T3);
    a.li(T3, i64::from(88.0f32.to_bits()));
    a.bgeu(T1, T3, special);
    a.bind(core);
    a.fcvt_d_s(FT2, FA0);
    constant(a, FT1, INV);
    constant(a, FT3, SHIFT);
    constant(a, FT4, 1.0f64.to_bits());
    if fma {
        a.fmadd_d(FT0, FT1, FT2, FT3);
    } else {
        a.fmul_d(FT0, FT1, FT2);
        a.fadd_d(FT0, FT0, FT3);
    }
    a.fmv_x_d(T1, FT0);
    a.fsub_d(FT0, FT0, FT3);
    a.andi(T2, T1, 31);
    a.slli(T1, T1, 47);
    if fma {
        a.fmsub_d(FT1, FT1, FT2, FT0);
    } else {
        a.fmul_d(FT1, FT1, FT2);
        a.fsub_d(FT1, FT1, FT0);
    }
    a.sh3add(T3, T2, A0);
    a.ld(T3, T3, 0);
    a.add(T1, T1, T3);
    a.fmv_d_x(FT5, T1);
    constant(a, FT0, E0);
    constant(a, FT6, E1);
    if fma {
        a.fmadd_d(FT0, FT0, FT1, FT6);
    } else {
        a.fmul_d(FT0, FT0, FT1);
        a.fadd_d(FT0, FT0, FT6);
    }
    a.fmul_d(FT2, FT1, FT1);
    constant(a, FT6, E2);
    if fma {
        a.fmadd_d(FT1, FT1, FT6, FT4);
        a.fmadd_d(FT0, FT0, FT2, FT1);
    } else {
        a.fmul_d(FT1, FT1, FT6);
        a.fadd_d(FT1, FT1, FT4);
        a.fmul_d(FT0, FT0, FT2);
        a.fadd_d(FT0, FT0, FT1);
    }
    a.fmul_d(FT0, FT0, FT5);
    a.fcvt_s_d(FA0, FT0, Rm::Dyn);
    a.ret();
    a.bind(special);
    a.li(T3, 0xff80_0000);
    a.beq(T2, T3, zero);
    a.li(T3, 0x7f80_0000);
    a.bgeu(T1, T3, nan_inf);
    a.li(T3, i64::from(88.722_84_f32.to_bits()));
    a.fmv_w_x(FT1, T3);
    a.flt_s(T3, FT1, FA0);
    a.bnez(T3, huge);
    a.li(T3, i64::from((-103.972_084_f32).to_bits()));
    a.fmv_w_x(FT1, T3);
    a.flt_s(T3, FA0, FT1);
    a.bnez(T3, zero);
    a.j(core);
    a.bind(nan_inf);
    a.fadd_s(FA0, FA0, FA0);
    a.ret();
    a.bind(huge);
    a.li(T3, 0x7f80_0000);
    a.fmv_w_x(FA0, T3);
    a.ret();
    a.bind(zero);
    a.fmv_w_x(FA0, ZERO);
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
    // t1 = ix, the bits of x (zero-extended).
    a.fmv_x_w(T1, FA0);
    a.zext_w(T1, T1);
    a.li(T3, i64::from(1.0f32.to_bits()));
    a.beq(T1, T3, one);
    a.li(T3, 0x80_0000);
    a.subw(T2, T1, T3);
    a.zext_w(T2, T2);
    a.li(T3, 0x7eff_ffff);
    a.bltu(T3, T2, special);
    a.bind(core);
    // tmp = ix − OFF (sign-extended); i = (tmp >> 19) % 16; k = tmp >> 23
    // (arithmetic); iz = ix − (tmp & 0xff800000).
    a.li(T3, 0x3f33_0000);
    a.subw(T2, T1, T3);
    a.srli(T3, T2, 19);
    a.andi(T3, T3, 15);
    a.slli(T3, T3, 4);
    a.li(T0, 0xff80_0000);
    a.and(T0, T2, T0);
    a.sraiw(T2, T2, 23);
    a.sub(T1, T1, T0);
    a.fmv_w_x(FT1, T1);
    a.fcvt_d_s(FT1, FT1);
    constant(a, FT5, (-1.0f64).to_bits());
    a.add(T3, A0, T3);
    a.fld(FT6, T3, 256);
    if fma {
        a.fmadd_d(FT1, FT1, FT6, FT5);
    } else {
        a.fmul_d(FT1, FT1, FT6);
        a.fadd_d(FT1, FT1, FT5);
    }
    a.fcvt_d_w(FT2, T2);
    constant(a, FT6, LN2);
    a.fld(FT5, T3, 264);
    if fma {
        a.fmadd_d(FT2, FT2, FT6, FT5);
    } else {
        a.fmul_d(FT2, FT2, FT6);
        a.fadd_d(FT2, FT2, FT5);
    }
    constant(a, FT3, L1);
    constant(a, FT6, L2);
    if fma {
        a.fmadd_d(FT3, FT3, FT1, FT6);
    } else {
        a.fmul_d(FT3, FT3, FT1);
        a.fadd_d(FT3, FT3, FT6);
    }
    a.fmul_d(FT0, FT1, FT1);
    a.fadd_d(FT1, FT1, FT2);
    constant(a, FT6, L0);
    if fma {
        a.fmadd_d(FT3, FT0, FT6, FT3);
        a.fmadd_d(FT0, FT0, FT3, FT1);
    } else {
        a.fmul_d(FT6, FT6, FT0);
        a.fadd_d(FT3, FT3, FT6);
        a.fmul_d(FT0, FT0, FT3);
        a.fadd_d(FT0, FT0, FT1);
    }
    a.fcvt_s_d(FA0, FT0, Rm::Dyn);
    a.ret();
    a.bind(special);
    // t2 = ix + ix (32 bits): zero for ±0.
    a.addw(T2, T1, T1);
    a.zext_w(T2, T2);
    a.beqz(T2, zero);
    a.li(T3, 0x7f80_0000);
    a.beq(T1, T3, ret);
    a.li(T3, 0xff00_0000);
    a.bgeu(T2, T3, invalid);
    a.srli(T3, T1, 31);
    a.bnez(T3, invalid);
    // Subnormal: normalise by 2^23 and take 23 off the exponent.
    a.li(T3, i64::from(8_388_608.0f32.to_bits()));
    a.fmv_w_x(FT1, T3);
    a.fmul_s(FA0, FA0, FT1);
    a.fmv_x_w(T1, FA0);
    a.zext_w(T1, T1);
    a.li(T3, 23 << 23);
    a.sub(T1, T1, T3);
    a.j(core);
    a.bind(zero);
    a.li(T3, 0xff80_0000);
    a.fmv_w_x(FA0, T3);
    a.ret();
    a.bind(invalid);
    a.fsub_s(FA0, FA0, FA0);
    a.fdiv_s(FA0, FA0, FA0);
    a.ret();
    a.bind(one);
    a.fmv_w_x(FA0, ZERO);
    a.bind(ret);
    a.ret();
}
