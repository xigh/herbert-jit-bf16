//! Generated layer transforms around the compiled elementary kernels.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::elem::CompiledElem;
use crate::jit::target::aarch64::util::{add_imm, exec_err, Frame};
use a64::*;

pub(crate) fn elementary(kernel: &CompiledElem) -> Result<GeneratedKernel, ExecutionError> {
    unsafe { GeneratedKernel::new(kernel.lowering_bytes()) }
}

/// `(value: *const usize, output: *mut u64)` -> `(value - subtract) * factor`.
/// Mounted call validation guarantees non-negative, non-overflowing offsets.
pub(crate) fn offset(factor: usize, subtract: usize) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    a.ldr(X8, at(X0));
    a.mov_imm(X9, subtract as u64);
    a.sub(X8, X8, X9);
    a.mov_imm(X9, factor as u64);
    a.mul(X8, X8, X9);
    a.str(X8, at(X1));
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
    let again = a.label();
    // x19 data, x20 weights, x21 cos, x22 sin, x23 heads left; eps on the stack.
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23, X24], false, 16);
    a.str(s(0), at(SP));
    a.mov(X19, X0);
    a.mov(X20, X1);
    a.mov(X21, X2);
    a.mov(X22, X3);
    a.mov_imm(X23, heads as u64);
    a.bind(again);
    a.mov(X0, X19);
    a.mov(X1, X20);
    a.mov(X2, X19);
    a.ldr(s(0), at(SP));
    a.bl(norm_label);
    a.mov(X0, X19);
    a.mov(X1, X21);
    a.mov(X2, X22);
    a.bl(rope_label);
    add_imm(&mut a, X19, X19, (stride * 2) as i64, X9);
    a.subs(X23, X23, 1);
    a.bne(again);
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
