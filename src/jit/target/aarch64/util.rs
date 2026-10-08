//! Emission helpers shared by every AArch64 kernel: the AAPCS64 frame, wide
//! immediates in arithmetic, and the conversion of an `a64` result into the
//! engine's errors.
//!
//! The calling convention of generated code is AAPCS64: integer arguments in
//! `x0..x7`, floating-point ones in `s0..s7`; `x19..x28`, the frame pointer
//! and the low halves of `v8..v15` belong to the caller. `x18` is the
//! platform register on Apple systems and is never written. `x16`/`x17` are
//! the intra-procedure scratch registers, free in any kernel.

use crate::core::error::HerbertError;
use crate::jit::execution::ExecutionError;
use a64::*;

/// An `a64` error as an execution error.
pub(crate) fn exec_err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// An `a64` error as a backend error, prefixed by what was being emitted.
pub(crate) fn backend_err(what: impl std::fmt::Display) -> impl FnOnce(a64::Error) -> HerbertError {
    move |e| HerbertError::Backend(format!("backend_jit aarch64: {what}: {e}"))
}

/// The frame of a function: what its prologue saved, in which order.
///
/// Layout, from the entry `sp` downward: `x29, x30` (when the function calls
/// anything), the callee-saved general-purpose registers by pairs, `d8..d15`
/// when the function uses `v8..v15`, then `locals` bytes at `[sp]`. Every
/// step keeps `sp` 16-byte aligned, as AArch64 requires of any `sp`-based
/// access.
pub(crate) struct Frame {
    link: bool,
    gprs: Vec<R>,
    dregs: bool,
    locals: usize,
}

impl Frame {
    /// Emit the prologue. `link` saves `x29`/`x30` and sets up the frame
    /// pointer — required as soon as the function executes `bl` or `blr`.
    pub(crate) fn enter(a: &mut Asm, link: bool, gprs: &[R], dregs: bool, locals: usize) -> Frame {
        if link {
            a.stp(X29, X30, pre(SP, -16));
            a.add(X29, SP, 0);
        }
        for pair in gprs.chunks(2) {
            match *pair {
                [r1, r2] => a.stp(r1, r2, pre(SP, -16)),
                [r1] => a.str(r1, pre(SP, -16)),
                _ => unreachable!(),
            }
        }
        if dregs {
            for i in (8..16).step_by(2) {
                a.stp(d(i), d(i + 1), pre(SP, -16));
            }
        }
        let locals = locals.next_multiple_of(16);
        if locals > 0 {
            a.sub(SP, SP, locals as i64);
        }
        Frame { link, gprs: gprs.to_vec(), dregs, locals }
    }

    /// Emit the epilogue, without the `ret`.
    pub(crate) fn leave(&self, a: &mut Asm) {
        if self.locals > 0 {
            a.add(SP, SP, self.locals as i64);
        }
        if self.dregs {
            for i in (8..16).step_by(2).rev() {
                a.ldp(d(i), d(i + 1), post(SP, 16));
            }
        }
        for pair in self.gprs.chunks(2).rev() {
            match *pair {
                [r1, r2] => a.ldp(r1, r2, post(SP, 16)),
                [r1] => a.ldr(r1, post(SP, 16)),
                _ => unreachable!(),
            }
        }
        if self.link {
            a.ldp(X29, X30, post(SP, 16));
        }
    }
}

/// `rd = rn + imm`, through `tmp` when `imm` is wider than the 12-bit forms.
pub(crate) fn add_imm(a: &mut Asm, rd: R, rn: R, imm: i64, tmp: R) {
    let m = imm.unsigned_abs();
    if m < 4096 || (m & 0xfff == 0 && m >> 12 < 4096) {
        a.add(rd, rn, imm);
    } else {
        a.mov_imm(tmp, imm as u64);
        a.add(rd, rn, tmp);
    }
}

/// `rd = rn · imm` (unsigned), with a shift when `imm` is a power of two.
pub(crate) fn mul_imm(a: &mut Asm, rd: R, rn: R, imm: u64, tmp: R) {
    match imm {
        0 => a.mov_imm(rd, 0),
        1 => {
            if rd != rn {
                a.mov(rd, rn);
            }
        }
        m if m.is_power_of_two() => a.lsl(rd, rn, m.trailing_zeros()),
        m => {
            a.mov_imm(tmp, m);
            a.mul(rd, rn, tmp);
        }
    }
}

/// `rd = rn · imm + ra`.
pub(crate) fn madd_imm(a: &mut Asm, rd: R, rn: R, imm: u64, ra: R, tmp: R) {
    match imm {
        0 => {
            if rd != ra {
                a.mov(rd, ra);
            }
        }
        1 => a.add(rd, ra, rn),
        m if m.is_power_of_two() => a.add(rd, ra, rn.lsl(m.trailing_zeros())),
        m => {
            a.mov_imm(tmp, m);
            a.madd(rd, rn, tmp, ra);
        }
    }
}

/// Broadcast the f32 bits `bits` into every lane of `v.4s`, through `tmp`.
pub(crate) fn splat(a: &mut Asm, vreg: u8, bits: u32, tmp: R) {
    let dst = v(vreg).s4();
    let value = f32::from_bits(bits);
    if bits == 0 {
        a.movi(v(vreg).d2(), 0);
    } else if value.is_finite() && encode_fp8(f64::from(value)).is_some() {
        a.fmov_imm(dst, f64::from(value));
    } else {
        a.mov_imm(tmp.as_w(), u64::from(bits));
        a.dup_gp(dst, tmp.as_w());
    }
}
