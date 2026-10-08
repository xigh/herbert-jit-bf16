//! Generated SDPA head/chunk traversal: the causal bound of every row and
//! the partial prefill chunks. Calculation never calls the Rust SDPA driver.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::sdpa::{CompiledSdpa, CompiledSdpaPrefill};
use crate::jit::target::aarch64::util::{exec_err, madd_imm, mul_imm, Frame};
use a64::*;
pub use crate::jit::execution::abi::{AttentionCall, AttentionSpec};
use std::mem::offset_of;

/// The decode entry as one relocatable kernel with the `SdpaHeadFn`
/// signature, including the three-call Split form chained here.
pub fn head_kernel(sdpa: &CompiledSdpa) -> Result<GeneratedKernel, ExecutionError> {
    let blobs = sdpa.lowering_blobs();
    if blobs.len() == 1 {
        return unsafe { GeneratedKernel::new(blobs.into_iter().next().unwrap()) };
    }
    let mut a = Asm::new();
    let labels: Vec<Label> = blobs.iter().map(|_| a.label()).collect();
    // The six integer arguments and the sink live on the stack across calls.
    let frame = Frame::enter(&mut a, true, &[], false, 64);
    a.stp(X0, X1, mem(SP, 0));
    a.stp(X2, X3, mem(SP, 16));
    a.stp(X4, X5, mem(SP, 32));
    a.str(s(1), mem(SP, 48));
    // QK(q, k, scores, n, scale).
    a.mov(X2, X3);
    a.mov(X3, X5);
    a.bl(labels[0]);
    // Softmax(scores, n, sink).
    a.ldr(X0, mem(SP, 24));
    a.ldr(X1, mem(SP, 40));
    a.ldr(s(0), mem(SP, 48));
    a.bl(labels[1]);
    // SV(scores, v, out, n).
    a.ldr(X0, mem(SP, 24));
    a.ldr(X1, mem(SP, 16));
    a.ldr(X2, mem(SP, 32));
    a.ldr(X3, mem(SP, 40));
    a.bl(labels[2]);
    frame.leave(&mut a);
    a.ret();
    for (blob, &label) in blobs.iter().zip(&labels) {
        a.align(16);
        a.bind(label);
        a.bytes(blob);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const AttentionCall, task: usize, worker_scores: *mut f32)`.
/// The caller reserves `ceil(rows / chunk_rows) * heads` tasks. Chunk rows are
/// those of the supplied prefill kernel (the sixteen of `sdpa_block.rs`,
/// which also receives the head's R16 copies in `x6`/`x7`), or one without
/// it; a partial last chunk runs the decode kernel row by row.
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
        || spec.heads.checked_mul(spec.q_head).is_none_or(|n| n > spec.q_row)
        || spec.heads.checked_mul(spec.head_dim).is_none_or(|n| n > spec.out_row)
        || [spec.heads, spec.gqa, spec.head_dim, spec.q_row, spec.q_head, spec.out_row]
            .iter()
            .any(|&n| n > i32::MAX as usize / 2)
    {
        return Err(ExecutionError("invalid generated attention geometry".into()));
    }
    if let Some(p) = prefill {
        if p.spec().shape != sdpa.spec().shape
            || p.spec().row_stride != spec.q_row
            || p.spec().out_row_stride != spec.out_row
        {
            return Err(ExecutionError("prefill attention kernel incompatible with layer policy".into()));
        }
    }
    let chunk_rows = prefill.map_or(1, CompiledSdpaPrefill::rows);
    let head = head_kernel(sdpa)?;
    let block = prefill.map(CompiledSdpaPrefill::lowering_bytes);
    emit_selector(spec, chunk_rows, head, block)
}

// x19 call, x20 query head, x21 row, x22 end row, x23 KV head, x26 worker scores.
fn emit_selector(
    spec: AttentionSpec,
    chunk_rows: usize,
    head: GeneratedKernel,
    block: Option<Vec<u8>>,
) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let head_label = a.label();
    let block_label = a.label();
    let exit = a.label();
    let tail = a.label();
    let row_loop = a.label();
    let field = |f: usize| mem(X19, f as i64);
    let frame = Frame::enter(&mut a, true, &[X19, X20, X21, X22, X23, X26], false, 0);
    a.mov(X19, X0);
    a.mov(X26, X2);
    a.ldr(X8, field(offset_of!(AttentionCall, rows)));
    a.ldr(X8, at(X8));
    a.cbz(X8, exit);
    // chunks = ceil(rows / chunk_rows)
    a.mov_imm(X11, chunk_rows as u64);
    a.add(X9, X8, X11);
    a.sub(X9, X9, 1);
    a.udiv(X9, X9, X11);
    if spec.grouped {
        a.mov_imm(X10, spec.gqa as u64);
        a.udiv(X12, X1, X10);
        a.msub(X20, X12, X10, X1); // q head within the KV group
        a.udiv(X13, X12, X9); // KV head
        a.msub(X12, X13, X9, X12); // chunk
        madd_imm(&mut a, X20, X13, spec.gqa as u64, X20, X14);
    } else {
        a.udiv(X20, X1, X9);
        a.msub(X12, X20, X9, X1);
    }
    a.mov_imm(X10, spec.heads as u64);
    a.cmp(X20, X10);
    a.bhs(exit);
    mul_imm(&mut a, X21, X12, chunk_rows as u64, X14);
    a.add(X22, X21, X11);
    a.cmp(X22, X8);
    a.csel(X22, X8, X22, Cond::Hi);
    a.mov_imm(X10, spec.gqa as u64);
    a.udiv(X23, X20, X10);
    if block.is_some() {
        a.sub(X10, X22, X21);
        a.cmp(X10, X11);
        a.bne(tail);
        kv_offset(&mut a);
        a.ldr(X5, field(offset_of!(AttentionCall, past)));
        a.ldr(X5, at(X5));
        a.add(X5, X5, X21);
        // The head's R16 copies, at the same offset as its cache.
        a.ldr(X6, field(offset_of!(AttentionCall, keys_r16)));
        a.add(X6, X6, X10);
        a.ldr(X7, field(offset_of!(AttentionCall, values_r16)));
        a.add(X7, X7, X10);
        arguments(&mut a, spec);
        a.bl(block_label);
        a.b(exit);
    }
    a.bind(tail);
    a.bind(row_loop);
    kv_offset(&mut a);
    // x5 = visible keys: the causal past plus the row itself.
    a.ldr(X5, field(offset_of!(AttentionCall, past)));
    a.ldr(X5, at(X5));
    a.add(X5, X5, X21);
    a.add(X5, X5, 1);
    arguments(&mut a, spec);
    a.bl(head_label);
    a.add(X21, X21, 1);
    a.cmp(X21, X22);
    a.blo(row_loop);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(head_label);
    a.bytes(&head.bytes);
    if let Some(blob) = block {
        a.align(16);
        a.bind(block_label);
        a.bytes(&blob);
    } else {
        a.bind(block_label);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `x10` = byte offset of the KV head in each buffer: every cache starts at position 0.
fn kv_offset(a: &mut Asm) {
    a.ldr(X10, mem(X19, offset_of!(AttentionCall, kv_stride) as i64));
    a.mul(X10, X10, X23);
    a.lsl(X10, X10, 1);
}

// x10 = byte offset in each KV buffer, x5 = n or past. The head kernels may
// clobber every argument register; the live state is callee-saved.
fn arguments(a: &mut Asm, spec: AttentionSpec) {
    let field = |f: usize| mem(X19, f as i64);
    a.ldr(X1, field(offset_of!(AttentionCall, keys)));
    a.add(X1, X1, X10);
    a.ldr(X2, field(offset_of!(AttentionCall, values)));
    a.add(X2, X2, X10);
    a.mov(X3, X26);
    a.ldr(X0, field(offset_of!(AttentionCall, q)));
    mul_imm(a, X10, X21, spec.q_row as u64, X12);
    madd_imm(a, X10, X20, spec.q_head as u64, X10, X12);
    a.add(X0, X0, X10.lsl(1));
    a.ldr(X4, field(offset_of!(AttentionCall, output)));
    mul_imm(a, X10, X21, spec.out_row as u64, X12);
    madd_imm(a, X10, X20, spec.head_dim as u64, X10, X12);
    a.add(X4, X4, X10.lsl(1));
    a.ldr(s(0), field(offset_of!(AttentionCall, scale)));
    a.ldr(s(1), field(offset_of!(AttentionCall, sink)));
}
