//! Generated layer transforms around the existing elementary kernels.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::elem::CompiledElem;
use iced_x86::code_asm::*;
fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

pub(crate) fn elementary(kernel: &CompiledElem) -> Result<GeneratedKernel, ExecutionError> {
    unsafe { GeneratedKernel::new(kernel.lowering_bytes()) }
}

/// `(value: *const usize, output: *mut u64)` -> `(value - subtract) * factor`.
/// Mounted call validation guarantees non-negative, non-overflowing offsets.
pub(crate) fn offset(factor: usize, subtract: usize) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.mov(rax, qword_ptr(rdi))?;
        a.mov(rcx, subtract as u64)?;
        a.sub(rax, rcx)?;
        a.mov(rcx, factor as u64)?;
        a.imul_2(rax, rcx)?;
        a.mov(qword_ptr(rsi), rax)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
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
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut norm_label = a.create_label();
    let mut rope_label = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut again = a.create_label();
        for reg in [rbx, rbp, r12, r13, r14] {
            a.push(reg)?;
        }
        a.sub(rsp, 16)?;
        a.movd(dword_ptr(rsp), xmm0)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, heads as u64)?;
        a.set_label(&mut again)?;
        a.mov(rdi, rbx)?;
        a.mov(rsi, rbp)?;
        a.mov(rdx, rbx)?;
        a.movd(xmm0, dword_ptr(rsp))?;
        a.call(norm_label)?;
        a.mov(rdi, rbx)?;
        a.mov(rsi, r12)?;
        a.mov(rdx, r13)?;
        a.call(rope_label)?;
        a.add(rbx, (stride * 2) as i32)?;
        a.dec(r14)?;
        a.jnz(again)?;
        a.add(rsp, 16)?;
        for reg in [r14, r13, r12, rbp, rbx] {
            a.pop(reg)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut norm_label).map_err(err)?;
    a.db(&norm.lowering_bytes()).map_err(err)?;
    a.set_label(&mut rope_label).map_err(err)?;
    a.db(&rope.lowering_bytes()).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

