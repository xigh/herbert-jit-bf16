//! Small generated data-movement/output kernels for plan lowering.
use super::{ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;

fn assemble(a: &mut CodeAssembler) -> Result<GeneratedKernel, ExecutionError> {
    let bytes = a.assemble(0).map_err(|e| ExecutionError(e.to_string()))?;
    // SAFETY: these emitters obey SysV64 and only access their declared buffers.
    unsafe { GeneratedKernel::new(bytes) }
}

/// Emit a copy of `bytes` from `[source + source_offset]` to
/// `[destination + destination_offset]`, both known at assembly.
///
/// The width is taken as wide as the machine allows and the move is unrolled,
/// so a short run costs no loop and no string-move startup; past `UNROLL_MAX`
/// chunks it becomes one `rep movsb`, which clobbers rsi, rdi and rcx. The
/// vector registers used sit above the eight the task loop fills with float
/// arguments. Both sides must be disjoint.
pub(super) fn copy_constant(
    a: &mut CodeAssembler,
    destination: AsmRegister64,
    destination_offset: usize,
    source: AsmRegister64,
    source_offset: usize,
    bytes: usize,
) -> Result<(), iced_x86::IcedError> {
    const UNROLL_MAX: usize = 32;
    let widest = match crate::jit::engine_isa() {
        crate::jit::IsaEngine::Avx512 => 64,
        crate::jit::IsaEngine::Avx2 => 32,
        crate::jit::IsaEngine::NeonBf16 | crate::jit::IsaEngine::Rvv | crate::jit::IsaEngine::BelowFloor => 16,
    };
    if bytes == 0 {
        return Ok(());
    }
    let chunks = {
        let (mut left, mut count) = (bytes, 0usize);
        while left > 0 {
            left -= [64, 32, 16, 8, 4, 2, 1]
                .into_iter()
                .find(|width| *width <= widest && *width <= left)
                .unwrap_or(1);
            count += 1;
        }
        count
    };
    if chunks > UNROLL_MAX {
        // Through rax first: either side may already be rsi or rdi.
        a.lea(rax, ptr(source + source_offset as i32))?;
        a.lea(rdi, ptr(destination + destination_offset as i32))?;
        a.mov(rsi, rax)?;
        a.mov(rcx, bytes as u64)?;
        return a.rep().movsb();
    }
    let mut left = bytes;
    let mut at = 0usize;
    while left > 0 {
        let (source_at, destination_at) = (
            (source_offset + at) as i32,
            (destination_offset + at) as i32,
        );
        let width = match left {
            n if n >= 64 && widest >= 64 => {
                a.vmovdqu64(zmm16, ptr(source + source_at))?;
                a.vmovdqu64(ptr(destination + destination_at), zmm16)?;
                64
            }
            n if n >= 32 && widest >= 32 => {
                a.vmovdqu(ymm8, ptr(source + source_at))?;
                a.vmovdqu(ptr(destination + destination_at), ymm8)?;
                32
            }
            n if n >= 16 => {
                a.movups(xmm8, ptr(source + source_at))?;
                a.movups(ptr(destination + destination_at), xmm8)?;
                16
            }
            n if n >= 8 => {
                a.mov(rax, qword_ptr(source + source_at))?;
                a.mov(qword_ptr(destination + destination_at), rax)?;
                8
            }
            n if n >= 4 => {
                a.mov(eax, dword_ptr(source + source_at))?;
                a.mov(dword_ptr(destination + destination_at), eax)?;
                4
            }
            n if n >= 2 => {
                a.mov(ax, word_ptr(source + source_at))?;
                a.mov(word_ptr(destination + destination_at), ax)?;
                2
            }
            _ => {
                a.mov(al, byte_ptr(source + source_at))?;
                a.mov(byte_ptr(destination + destination_at), al)?;
                1
            }
        };
        left -= width;
        at += width;
    }
    Ok(())
}

/// Emit the backend's bf16 argmax `(logits, result_u32, vocabulary_size)`.
/// Uses strict float `>`: first tie wins, NaNs are ignored, all -inf/NaN yields
/// index zero. Vocabulary length must fit a u32 (checked by model lowering).
pub fn argmax_bf16() -> Result<GeneratedKernel, ExecutionError> {
    fn emit(a: &mut CodeAssembler) -> Result<(), iced_x86::IcedError> {
        let mut next = a.create_label();
        let mut keep = a.create_label();
        let mut done = a.create_label();
        a.mov(eax, 0xff800000u32)?;
        a.movd(xmm0, eax)?;
        a.xor(r8d, r8d)?;
        a.xor(ecx, ecx)?;
        a.test(rdx, rdx)?;
        a.je(done)?;
        a.set_label(&mut next)?;
        a.movzx(eax, word_ptr(rdi + rcx * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm1, eax)?;
        a.ucomiss(xmm1, xmm0)?;
        // Unordered sets CF/ZF, so JBE also skips NaN candidates.
        a.jbe(keep)?;
        a.movaps(xmm0, xmm1)?;
        a.mov(r8d, ecx)?;
        a.set_label(&mut keep)?;
        a.inc(rcx)?;
        a.cmp(rcx, rdx)?;
        a.jb(next)?;
        a.set_label(&mut done)?;
        a.mov(dword_ptr(rsi), r8d)?;
        a.ret()
    }
    let mut a = CodeAssembler::new(64).map_err(|e| ExecutionError(e.to_string()))?;
    emit(&mut a).map_err(|e| ExecutionError(e.to_string()))?;
    assemble(&mut a)
}

/// Emit the token embedding lookup: `(tokens: *const u32, table: *const u16, out: *mut u16, rows: *const usize)`,
/// one bf16 row of the table copied per token.
pub fn embedding() -> Result<GeneratedKernel, ExecutionError> {
    fn emit(a: &mut CodeAssembler) -> Result<(), iced_x86::IcedError> {
        // The row is contiguous on both sides, so it moves with one string copy
        // instead of a word at a time.
        // `rep movsb` reads its operands in rsi, rdi and rcx: the destination is
        // moved into rdi after the token identifier has been read from it, and the
        // call site reloads all six argument registers from the binding before each
        // task, so clobbering them here costs nothing. A row of zero values moves
        // nothing on its own, which retires the guard the loop needed, and the
        // direction flag is zero throughout the program — nothing ever sets it.
        a.mov(eax, dword_ptr(rdi))?;
        a.imul_2(rax, rcx)?;
        a.lea(rsi, ptr(rsi + rax * 2))?;
        a.mov(rdi, rdx)?;
        a.shl(rcx, 1)?;
        a.rep().movsb()?;
        a.ret()
    }
    let mut a = CodeAssembler::new(64).map_err(|e| ExecutionError(e.to_string()))?;
    emit(&mut a).map_err(|e| ExecutionError(e.to_string()))?;
    assemble(&mut a)
}
