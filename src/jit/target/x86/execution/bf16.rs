//! Generated traversal of the existing bf16 R16 projection kernels.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::compile::CompiledGemm;
use crate::jit::emit::b_block_bytes;
use iced_x86::code_asm::*;
pub use crate::jit::execution::abi::{Bf16Call};

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Emit `(call: *const Bf16Call, task: usize)`. The tasks run the grid in
/// N-external / M-internal order; input offsets, variable column widths and
/// M/N tails follow the column plan of `CompiledGemm`.
/// All copied kernels retain their original instruction streams.
pub fn tile_selector(gemm: &CompiledGemm) -> Result<GeneratedKernel, ExecutionError> {
    let shape = gemm.shape();
    let mr = gemm.tile().mr;
    let ce = gemm.c_elem_bytes();
    let (format, n_tiles, blobs) = gemm.lowering_parts();
    let a_row = shape.k.checked_mul(format.elem_bytes());
    let b_column = shape.k_pairs().checked_mul(4);
    if [
        a_row,
        b_column,
        Some(shape.n),
        Some(n_tiles.len()),
        Some(mr),
    ]
    .into_iter()
    .any(|v| v.is_none_or(|n| n > i32::MAX as usize))
    {
        return Err(ExecutionError(
            "bf16 projection exceeds addressing domain".into(),
        ));
    }
    debug_assert_eq!(b_column.unwrap() * 16, b_block_bytes(shape.k_pairs()));
    // Runs keep the selector proportional to distinct widths/tails, even for a
    // large language head with thousands of equal-width column tiles.
    let mut runs = Vec::new();
    let mut start = 0;
    while start < n_tiles.len() {
        let kind = n_tiles[start].1;
        let mut end = start + 1;
        while end < n_tiles.len() && n_tiles[end].1 == kind {
            end += 1;
        }
        let width = if end - start > 1 {
            n_tiles[start + 1].0 - n_tiles[start].0
        } else {
            0
        };
        runs.push((start, end, n_tiles[start].0, width, kind));
        start = end;
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut entries: Vec<_> = blobs.iter().map(|_| a.create_label()).collect();
    let mut kinds: Vec<_> = (0..blobs.len() / mr).map(|_| a.create_label()).collect();
    let emit = |a: &mut CodeAssembler,
                entries: &[CodeLabel],
                kinds: &mut [CodeLabel]|
     -> Result<(), iced_x86::IcedError> {
        let mut exit = a.create_label();
        let mut rows_ready = a.create_label();
        // Three pushes align the stack for SysV64 calls on every OS.
        for reg in [rbx, r12, r13] {
            a.push(reg)?;
        }
        a.mov(rbx, rdi)?;
        a.mov(
            r10,
            qword_ptr(rbx + std::mem::offset_of!(Bf16Call, rows) as i32),
        )?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(Bf16Call, row_count) as i32),
        )?;
        a.test(rax, rax)?;
        a.je(rows_ready)?;
        a.mov(r10, qword_ptr(rax))?;
        a.set_label(&mut rows_ready)?;
        a.test(r10, r10)?;
        a.je(exit)?;
        // Overflow-free ceil(rows / mr).
        a.mov(rax, r10)?;
        a.xor(edx, edx)?;
        a.mov(r11, mr as u64)?;
        a.div(r11)?;
        a.test(rdx, rdx)?;
        a.setne(dl)?;
        a.movzx(edx, dl)?;
        a.add(rax, rdx)?;
        a.mov(rcx, rax)?;
        a.mov(rax, rsi)?;
        a.xor(edx, edx)?;
        a.div(rcx)?;
        // RAX=column tile, R13=first row, R12=height.
        a.imul_3(r13, rdx, mr as i32)?;
        a.mov(r12, r10)?;
        a.sub(r12, r13)?;
        a.cmp(r12, r11)?;
        a.cmova(r12, r11)?;
        for &(start, end, offset, width, kind) in &runs {
            let mut next = a.create_label();
            a.cmp(rax, end as i32)?;
            a.jae(next)?;
            a.mov(r9, rax)?;
            a.sub(r9, start as i32)?;
            a.imul_3(r9, r9, width as i32)?;
            a.add(r9, offset as i32)?;
            a.jmp(kinds[kind])?;
            a.set_label(&mut next)?;
            a.nop()?;
        }
        a.jmp(exit)?;
        for (kind, label) in kinds.iter_mut().enumerate() {
            a.set_label(label)?;
            a.mov(
                rdi,
                qword_ptr(rbx + std::mem::offset_of!(Bf16Call, activations) as i32),
            )?;
            a.imul_3(r10, r13, a_row.unwrap() as i32)?;
            a.add(rdi, r10)?;
            a.mov(
                rsi,
                qword_ptr(rbx + std::mem::offset_of!(Bf16Call, weights) as i32),
            )?;
            a.imul_3(r10, r9, b_column.unwrap() as i32)?;
            a.add(rsi, r10)?;
            a.mov(
                rdx,
                qword_ptr(rbx + std::mem::offset_of!(Bf16Call, output) as i32),
            )?;
            a.imul_3(r10, r13, shape.n as i32)?;
            a.add(r10, r9)?;
            a.imul_3(r10, r10, ce as i32)?;
            a.add(rdx, r10)?;
            for height in 1..=mr {
                let mut next = a.create_label();
                a.cmp(r12, height as i32)?;
                a.jne(next)?;
                a.call(entries[kind * mr + height - 1])?;
                a.jmp(exit)?;
                a.set_label(&mut next)?;
                a.nop()?;
            }
            a.jmp(exit)?;
        }
        a.set_label(&mut exit)?;
        for reg in [r13, r12, rbx] {
            a.pop(reg)?;
        }
        a.ret()
    };
    emit(&mut a, &entries, &mut kinds).map_err(err)?;
    for (blob, label) in blobs.iter().zip(&mut entries) {
        a.set_label(label).map_err(err)?;
        a.db(blob).map_err(err)?;
    }
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
