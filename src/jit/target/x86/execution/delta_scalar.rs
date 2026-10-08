//! Exact scalar fallbacks emitted at mount, including non-vector geometries.
use super::{delta::Shape, math, ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Same six-integer/three-float ABI as the mounted recurrence kernel.
pub(crate) fn recurrence(shape: Shape) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut zero = a.create_label();
        let mut decay_k = a.create_label();
        let mut decay_v = a.create_label();
        let mut delta = a.create_label();
        let mut update_v = a.create_label();
        let mut update_k = a.create_label();
        a.xor(r11d, r11d)?;
        a.set_label(&mut zero)?;
        a.mov(dword_ptr(rsi + r11 * 4), 0)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(zero)?;
        a.xor(r10d, r10d)?;
        a.mov(rax, rdi)?;
        a.set_label(&mut decay_k)?;
        a.movd(xmm3, dword_ptr(rcx + r10 * 4))?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut decay_v)?;
        a.movd(xmm4, dword_ptr(rax + r11 * 4))?;
        a.mulss(xmm4, xmm1)?;
        a.movd(dword_ptr(rax + r11 * 4), xmm4)?;
        a.mulss(xmm4, xmm3)?;
        a.movd(xmm5, dword_ptr(rsi + r11 * 4))?;
        a.addss(xmm5, xmm4)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm5)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(decay_v)?;
        a.add(rax, (shape.hvd * 4) as i32)?;
        a.inc(r10)?;
        a.cmp(r10, r9)?;
        a.jb(decay_k)?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut delta)?;
        a.movd(xmm3, dword_ptr(r8 + r11 * 4))?;
        a.subss(xmm3, dword_ptr(rsi + r11 * 4))?;
        a.mulss(xmm3, xmm0)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm3)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(delta)?;
        a.xor(r11d, r11d)?;
        a.set_label(&mut update_v)?;
        a.movd(xmm3, dword_ptr(rsi + r11 * 4))?;
        a.xorps(xmm4, xmm4)?;
        a.lea(rax, ptr(rdi + r11 * 4))?;
        a.xor(r10d, r10d)?;
        a.set_label(&mut update_k)?;
        a.movd(xmm5, dword_ptr(rcx + r10 * 4))?;
        a.mulss(xmm5, xmm3)?;
        a.movd(xmm6, dword_ptr(rax))?;
        a.addss(xmm6, xmm5)?;
        a.movd(dword_ptr(rax), xmm6)?;
        a.movd(xmm5, dword_ptr(rdx + r10 * 4))?;
        a.mulss(xmm5, xmm2)?;
        a.mulss(xmm6, xmm5)?;
        a.addss(xmm4, xmm6)?;
        a.add(rax, (shape.hvd * 4) as i32)?;
        a.inc(r10)?;
        a.cmp(r10, r9)?;
        a.jb(update_k)?;
        a.movd(dword_ptr(rsi + r11 * 4), xmm4)?;
        a.inc(r11)?;
        a.cmp(r11, shape.hvd as i32)?;
        a.jb(update_v)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(data, weights, state, channels, tables)` for one causal position.
pub(crate) fn convolution(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let exp = math::exp(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut channel = a.create_label();
        let mut shift = a.create_label();
        let mut sum = a.create_label();
        let mut exit = a.create_label();
        for r in [rbx, rbp, r12, r13, r14] {
            a.push(r)?;
        }
        a.sub(rsp, 16)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, r8)?;
        a.test(r13, r13)?;
        a.je(exit)?;
        a.set_label(&mut channel)?;
        a.xor(ecx, ecx)?;
        a.set_label(&mut shift)?;
        a.mov(eax, dword_ptr(r12 + rcx * 4 + 4))?;
        a.mov(dword_ptr(r12 + rcx * 4), eax)?;
        a.inc(rcx)?;
        a.cmp(rcx, (shape.ck - 1) as i32)?;
        a.jb(shift)?;
        a.mov(eax, dword_ptr(rbx))?;
        a.mov(dword_ptr(r12 + (shape.ck * 4 - 4) as i32), eax)?;
        a.xor(ecx, ecx)?;
        a.xorps(xmm0, xmm0)?;
        a.set_label(&mut sum)?;
        a.movzx(eax, word_ptr(rbp + rcx * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm1, eax)?;
        a.mulss(xmm1, dword_ptr(r12 + rcx * 4))?;
        a.addss(xmm0, xmm1)?;
        a.inc(rcx)?;
        a.cmp(rcx, shape.ck as i32)?;
        a.jb(sum)?;
        a.movd(dword_ptr(rsp), xmm0)?;
        a.movd(eax, xmm0)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mov(rdi, r14)?;
        a.call(exp_entry)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.movd(xmm1, dword_ptr(rsp))?;
        a.divss(xmm1, xmm0)?;
        a.movd(dword_ptr(rbx), xmm1)?;
        a.add(rbx, 4)?;
        a.add(rbp, (shape.ck * 2) as i32)?;
        a.add(r12, (shape.ck * 4) as i32)?;
        a.dec(r13)?;
        a.jnz(channel)?;
        a.set_label(&mut exit)?;
        a.add(rsp, 16)?;
        for r in [r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_entry).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(out, z, weight, bf16_out, 1, tables, eps)` for one V-head row.
/// Keeps the literal left-to-right RMS sum, then SiLU, then RNE narrowing.
pub(crate) fn gated_norm(dim: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gated norm width".into()));
    }
    let exp = math::exp(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut sum = a.create_label();
        let mut apply = a.create_label();
        for r in [rbx, rbp, r12, r13, r14, r15] {
            a.push(r)?;
        }
        a.sub(rsp, 24)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(r12, rdx)?;
        a.mov(r13, rcx)?;
        a.mov(r14, r9)?;
        a.xor(eax, eax)?;
        a.xorps(xmm1, xmm1)?;
        a.set_label(&mut sum)?;
        a.movd(xmm2, dword_ptr(rbx + rax * 4))?;
        a.mulss(xmm2, xmm2)?;
        a.addss(xmm1, xmm2)?;
        a.inc(rax)?;
        a.cmp(rax, dim as i32)?;
        a.jb(sum)?;
        a.mov(eax, (dim as f32).to_bits())?;
        a.movd(xmm2, eax)?;
        a.divss(xmm1, xmm2)?;
        a.addss(xmm1, xmm0)?;
        a.sqrtss(xmm1, xmm1)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm0, eax)?;
        a.divss(xmm0, xmm1)?;
        a.movd(dword_ptr(rsp), xmm0)?;
        a.xor(r15d, r15d)?;
        a.set_label(&mut apply)?;
        a.mov(eax, dword_ptr(rbp + r15 * 4))?;
        a.mov(dword_ptr(rsp + 4), eax)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mov(rdi, r14)?;
        a.call(exp_entry)?;
        a.mov(eax, 1f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.movd(xmm1, dword_ptr(rsp + 4))?;
        a.divss(xmm1, xmm0)?;
        a.movd(xmm0, dword_ptr(rbx + r15 * 4))?;
        a.mulss(xmm0, dword_ptr(rsp))?;
        a.movzx(eax, word_ptr(r12 + r15 * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm2, eax)?;
        a.mulss(xmm0, xmm2)?;
        a.mulss(xmm0, xmm1)?;
        a.movd(dword_ptr(rbx + r15 * 4), xmm0)?;
        // Same wrapping integer RNE formula as core::tensor::f32_to_bf16,
        // including its existing treatment of non-finite payloads.
        a.movd(eax, xmm0)?;
        a.mov(ecx, eax)?;
        a.shr(ecx, 16)?;
        a.and(ecx, 1)?;
        a.add(eax, 0x7fff)?;
        a.add(eax, ecx)?;
        a.shr(eax, 16)?;
        a.mov(word_ptr(r13 + r15 * 2), ax)?;
        a.inc(r15)?;
        a.cmp(r15, dim as i32)?;
        a.jb(apply)?;
        a.add(rsp, 24)?;
        for r in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_entry).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
