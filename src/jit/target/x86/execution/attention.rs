//! Generated SDPA head/chunk traversal: the causal bound of every row and
//! the partial prefill chunks. Calculation never calls the Rust SDPA driver.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::sdpa::{CompiledSdpa, CompiledSdpaPrefill};
use iced_x86::code_asm::*;
pub use crate::jit::execution::abi::{AttentionSpec, AttentionCall};

fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// Compile the existing decode entry, including the three-call Split form, to
/// an owned relocatable kernel with the same `SdpaHeadFn` SysV64 signature.
pub fn head_kernel(sdpa: &CompiledSdpa) -> Result<GeneratedKernel, ExecutionError> {
    let blobs = sdpa.lowering_blobs();
    if blobs.len() == 1 {
        return unsafe { GeneratedKernel::new(blobs.into_iter().next().unwrap()) };
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut labels: Vec<_> = blobs.iter().map(|_| a.create_label()).collect();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.sub(rsp, 72)?;
        for (i, reg) in [rdi, rsi, rdx, rcx, r8, r9].into_iter().enumerate() {
            a.mov(qword_ptr(rsp + (i * 8) as i32), reg)?;
        }
        a.movd(dword_ptr(rsp + 48), xmm1)?;
        // QK(q, k, scores, n, scale).
        a.mov(rdx, rcx)?;
        a.mov(rcx, r9)?;
        a.call(labels[0])?;
        // Softmax(scores, n, sink).
        a.mov(rdi, qword_ptr(rsp + 24))?;
        a.mov(rsi, qword_ptr(rsp + 40))?;
        a.movd(xmm0, dword_ptr(rsp + 48))?;
        a.call(labels[1])?;
        // SV(scores, v, out, n).
        a.mov(rdi, qword_ptr(rsp + 24))?;
        a.mov(rsi, qword_ptr(rsp + 16))?;
        a.mov(rdx, qword_ptr(rsp + 32))?;
        a.mov(rcx, qword_ptr(rsp + 40))?;
        a.call(labels[2])?;
        a.add(rsp, 72)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    for (blob, label) in blobs.iter().zip(&mut labels) {
        a.set_label(label).map_err(err)?;
        a.db(blob).map_err(err)?;
    }
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// Emit `(call: *const AttentionCall, task: usize, worker_scores: *mut f32)`.
/// The caller reserves `ceil(rows / chunk_rows) * heads` tasks. Chunk rows are
/// those of the supplied prefill kernel, or one without it.
pub fn tile_selector(
    sdpa: &CompiledSdpa,
    prefill: Option<&CompiledSdpaPrefill>,
    spec: AttentionSpec,
) -> Result<GeneratedKernel, ExecutionError> {
    if spec.heads == 0
        || spec.gqa == 0
        || !spec.heads.is_multiple_of(spec.gqa)
        || spec.head_dim != sdpa.spec().shape.head_dim
        || spec.q_head < spec.head_dim
        || spec
            .heads
            .checked_mul(spec.q_head)
            .is_none_or(|n| n > spec.q_row)
        || spec
            .heads
            .checked_mul(spec.head_dim)
            .is_none_or(|n| n > spec.out_row)
        || [
            spec.heads,
            spec.gqa,
            spec.head_dim,
            spec.q_row,
            spec.q_head,
            spec.out_row,
        ]
        .iter()
        .any(|&n| n > i32::MAX as usize / 2)
    {
        return Err(ExecutionError(
            "invalid generated attention geometry".into(),
        ));
    }
    if let Some(p) = prefill {
        if p.spec().shape != sdpa.spec().shape
            || p.spec().row_stride != spec.q_row
            || p.spec().out_row_stride != spec.out_row
        {
            return Err(ExecutionError(
                "prefill attention kernel incompatible with layer policy".into(),
            ));
        }
    }
    let chunk_rows = prefill.map_or(1, CompiledSdpaPrefill::rows);
    let head = head_kernel(sdpa)?;
    let block = prefill.map(CompiledSdpaPrefill::lowering_bytes);
    emit_selector(spec, chunk_rows, head, block)
}

fn emit_selector(
    spec: AttentionSpec,
    chunk_rows: usize,
    head: GeneratedKernel,
    block: Option<Vec<u8>>,
) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut head_label = a.create_label();
    let mut block_label = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut exit = a.create_label();
        let mut tail = a.create_label();
        let mut row_loop = a.create_label();
        for reg in [rbx, rbp, r12, r13, r14, r15] {
            a.push(reg)?;
        }
        a.sub(rsp, 8)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rdx)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(AttentionCall, rows) as i32),
        )?;
        a.mov(r8, qword_ptr(rax))?;
        a.test(r8, r8)?;
        a.je(exit)?;
        a.mov(rax, r8)?;
        a.xor(edx, edx)?;
        a.mov(r11, chunk_rows as u64)?;
        a.div(r11)?;
        a.test(rdx, rdx)?;
        a.setne(dl)?;
        a.movzx(edx, dl)?;
        a.add(rax, rdx)?;
        a.mov(rcx, rax)?; // chunks
        a.mov(rax, rsi)?;
        a.xor(edx, edx)?;
        if spec.grouped {
            a.mov(r10, spec.gqa as u64)?;
            a.div(r10)?;
            a.mov(r12, rdx)?; // q head within KV group
            a.xor(edx, edx)?;
            a.div(rcx)?; // KV head and chunk
            a.imul_3(rax, rax, spec.gqa as i32)?;
            a.add(r12, rax)?;
        } else {
            a.div(rcx)?;
            a.mov(r12, rax)?;
        }
        a.cmp(r12, spec.heads as i32)?;
        a.jae(exit)?;
        a.imul_3(r13, rdx, chunk_rows as i32)?;
        a.lea(r14, ptr(r13 + chunk_rows as i32))?;
        a.cmp(r14, r8)?;
        a.cmova(r14, r8)?;
        a.mov(rax, r12)?;
        a.xor(edx, edx)?;
        a.mov(r10, spec.gqa as u64)?;
        a.div(r10)?;
        a.mov(r15, rax)?;
        if block.is_some() {
            a.mov(rax, r14)?;
            a.sub(rax, r13)?;
            a.cmp(rax, chunk_rows as i32)?;
            a.jne(tail)?;
            a.mov(
                r10,
                qword_ptr(rbx + std::mem::offset_of!(AttentionCall, kv_stride) as i32),
            )?;
            a.imul_2(r10, r15)?;
            a.shl(r10, 1)?;
            a.mov(
                rax,
                qword_ptr(rbx + std::mem::offset_of!(AttentionCall, past) as i32),
            )?;
            a.mov(r9, qword_ptr(rax))?;
            a.add(r9, r13)?;
            arguments(a, spec)?;
            a.call(block_label)?;
            a.jmp(exit)?;
        }
        a.set_label(&mut tail)?;
        a.nop()?;
        a.set_label(&mut row_loop)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(AttentionCall, past) as i32),
        )?;
        a.mov(r9, qword_ptr(rax))?;
        // R9 = visible keys: the causal past plus the row itself.
        a.add(r9, r13)?;
        a.inc(r9)?;
        // R10 = byte offset of this KV head: every cache starts at position 0.
        a.mov(
            r10,
            qword_ptr(rbx + std::mem::offset_of!(AttentionCall, kv_stride) as i32),
        )?;
        a.imul_2(r10, r15)?;
        a.shl(r10, 1)?;
        arguments(a, spec)?;
        a.call(head_label)?;
        a.inc(r13)?;
        a.cmp(r13, r14)?;
        a.jb(row_loop)?;
        a.set_label(&mut exit)?;
        a.add(rsp, 8)?;
        for reg in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(reg)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut head_label).map_err(err)?;
    a.db(&head.bytes).map_err(err)?;
    if let Some(blob) = block {
        a.set_label(&mut block_label).map_err(err)?;
        a.db(&blob).map_err(err)?;
    }
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

// R10 = byte offset within each KV buffer, R9 = n/past. Live state is in
// callee-saved registers; every generated head may clobber all argument registers.
fn arguments(a: &mut CodeAssembler, spec: AttentionSpec) -> Result<(), iced_x86::IcedError> {
    a.mov(
        rsi,
        qword_ptr(rbx + std::mem::offset_of!(AttentionCall, keys) as i32),
    )?;
    a.add(rsi, r10)?;
    a.mov(
        rdx,
        qword_ptr(rbx + std::mem::offset_of!(AttentionCall, values) as i32),
    )?;
    a.add(rdx, r10)?;
    a.mov(rcx, rbp)?;
    a.mov(
        rdi,
        qword_ptr(rbx + std::mem::offset_of!(AttentionCall, q) as i32),
    )?;
    a.imul_3(r10, r13, spec.q_row as i32)?;
    a.imul_3(r11, r12, spec.q_head as i32)?;
    a.add(r10, r11)?;
    a.lea(rdi, ptr(rdi + r10 * 2))?;
    a.mov(
        r8,
        qword_ptr(rbx + std::mem::offset_of!(AttentionCall, output) as i32),
    )?;
    a.imul_3(r10, r13, spec.out_row as i32)?;
    a.imul_3(r11, r12, spec.head_dim as i32)?;
    a.add(r10, r11)?;
    a.lea(r8, ptr(r8 + r10 * 2))?;
    a.movd(
        xmm0,
        dword_ptr(rbx + std::mem::offset_of!(AttentionCall, scale) as i32),
    )?;
    a.movd(
        xmm1,
        dword_ptr(rbx + std::mem::offset_of!(AttentionCall, sink) as i32),
    )?;
    Ok(())
}
