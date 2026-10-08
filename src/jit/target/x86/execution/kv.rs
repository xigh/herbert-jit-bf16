//! The KV append: the keys and values of the current block, token-major in
//! the projection buffers, copied head-major into the layer's cache.
//!
//! One task per KV head. The cache holds every position from 0: the block
//! starting at absolute position `past` lands at `past · head_dim` in each
//! head's slice. A plan barrier follows the append before attention reads it.
use super::{ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
pub use crate::jit::execution::abi::{KvCall, KvSpec};

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// One task per KV head `(call: *const KvCall, head: usize)`. The mounted
/// reserve guarantees `past + rows <= capacity` for every valid call.
pub fn append(spec: KvSpec) -> Result<GeneratedKernel, ExecutionError> {
    spec.validate()?;
    if spec.r16 {
        return Err(ExecutionError("the R16 copies of the KV cache are an AArch64 layout".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut exit = a.create_label();
        let mut rows = a.create_label();
        a.push(rbx)?;
        a.mov(rbx, rdi)?;
        a.mov(r11, rsi)?;
        a.cmp(r11, spec.heads as i32)?;
        a.jae(exit)?;
        a.mov(rax, qword_ptr(rbx + std::mem::offset_of!(KvCall, rows) as i32))?;
        a.mov(r9, qword_ptr(rax))?;
        a.test(r9, r9)?;
        a.je(exit)?;
        a.mov(rax, qword_ptr(rbx + std::mem::offset_of!(KvCall, past) as i32))?;
        a.mov(r10, qword_ptr(rax))?;
        a.xor(r8d, r8d)?;
        a.cld()?;
        a.set_label(&mut rows)?;
        // Source: row `r8`, head `r11` of the token-major block.
        a.imul_3(rax, r8, spec.heads as i32)?;
        a.add(rax, r11)?;
        a.imul_3(rax, rax, (spec.head_dim * 2) as i32)?;
        // Target: position `past + r8` in the slice of head `r11`.
        a.lea(rdx, ptr(r10 + r8))?;
        a.imul_3(rdx, rdx, (spec.head_dim * 2) as i32)?;
        a.imul_3(rcx, r11, (spec.stride * 2) as i32)?;
        a.add(rdx, rcx)?;
        for (target, source) in [
            (std::mem::offset_of!(KvCall, keys), std::mem::offset_of!(KvCall, input_keys)),
            (std::mem::offset_of!(KvCall, values), std::mem::offset_of!(KvCall, input_values)),
        ] {
            a.mov(rdi, qword_ptr(rbx + target as i32))?;
            a.add(rdi, rdx)?;
            a.mov(rsi, qword_ptr(rbx + source as i32))?;
            a.add(rsi, rax)?;
            a.mov(rcx, (spec.head_dim * 2) as u64)?;
            a.rep().movsb()?;
        }
        a.inc(r8)?;
        a.cmp(r8, r9)?;
        a.jb(rows)?;
        a.set_label(&mut exit)?;
        a.pop(rbx)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
