//! Generated layer transforms around the compiled elementary kernels.

use super::{ExecutionError, GeneratedKernel};
use crate::jit::elem::CompiledElem;
use crate::jit::target::riscv64::util::{add_imm, exec_err, Frame};
use rv::*;

pub(crate) fn elementary(kernel: &CompiledElem) -> Result<GeneratedKernel, ExecutionError> {
    unsafe { GeneratedKernel::new(kernel.lowering_bytes()) }
}

/// `(value: *const usize, output: *mut u64)` -> `(value - subtract) * factor`.
/// Mounted call validation guarantees non-negative, non-overflowing offsets.
pub(crate) fn offset(factor: usize, subtract: usize) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    a.ld(T0, A0, 0);
    a.li(T1, subtract as i64);
    a.sub(T0, T0, T1);
    a.li(T1, factor as i64);
    a.mul(T0, T0, T1);
    a.sd(T0, A1, 0);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// One row: `(data, weight, cos, sin, eps)`, all heads normalized then rotated
/// in the reference per-head order. The RoPE kernel may have a partial dimension.
pub(crate) fn norm_rope(
    norm: &CompiledElem,
    rope: &CompiledElem,
    heads: usize,
    stride: usize,
) -> Result<GeneratedKernel, ExecutionError> {
    if heads == 0 || heads > i32::MAX as usize || stride > i32::MAX as usize / 2 {
        return Err(ExecutionError("invalid mounted head transform".into()));
    }
    let mut a = Asm::new();
    let norm_label = a.label();
    let rope_label = a.label();
    // s1 data, s2 weights, s3 cos, s4 sin, s5 heads left; eps in fs0.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5], &[FS0], 0);
    a.fmv_s(FS0, FA0);
    a.mv(S1, A0);
    a.mv(S2, A1);
    a.mv(S3, A2);
    a.mv(S4, A3);
    a.li(S5, heads as i64);
    let again = a.here();
    a.mv(A0, S1);
    a.mv(A1, S2);
    a.mv(A2, S1);
    a.fmv_s(FA0, FS0);
    a.call(norm_label);
    a.mv(A0, S1);
    a.mv(A1, S3);
    a.mv(A2, S4);
    a.call(rope_label);
    add_imm(&mut a, S1, S1, (stride * 2) as i64, T6);
    a.addi(S5, S5, -1);
    a.bnez(S5, again);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(norm_label);
    a.bytes(&norm.lowering_bytes());
    a.align(16);
    a.bind(rope_label);
    a.bytes(&rope.lowering_bytes());
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
