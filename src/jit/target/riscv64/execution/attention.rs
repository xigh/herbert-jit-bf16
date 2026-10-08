//! Generated SDPA head/chunk traversal: the causal bound of every row.
//! Calculation never calls the Rust SDPA driver. RISC-V has no block kernel:
//! every row of a chunk runs the decode kernel.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::sdpa::{CompiledSdpa, CompiledSdpaPrefill};
use crate::jit::target::riscv64::util::{exec_err, Frame};
use rv::*;
pub use crate::jit::execution::abi::{AttentionCall, AttentionSpec};
use std::mem::offset_of;

/// `rd = rs · imm` (through `tmp`).
fn mul_imm(a: &mut Asm, rd: X, rs: X, imm: u64, tmp: X) {
    a.li(tmp, imm as i64);
    a.mul(rd, rs, tmp);
}

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
    let frame = Frame::enter(&mut a, true, &[], &[], 56);
    for (i, r) in [A0, A1, A2, A3, A4, A5].into_iter().enumerate() {
        a.sd(r, SP, 8 * i as i64);
    }
    a.fsw(FA1, SP, 48);
    // QK(q, k, scores, n, scale).
    a.mv(A2, A3);
    a.mv(A3, A5);
    a.call(labels[0]);
    // Softmax(scores, n, sink).
    a.ld(A0, SP, 24);
    a.ld(A1, SP, 40);
    a.flw(FA0, SP, 48);
    a.call(labels[1]);
    // SV(scores, v, out, n).
    a.ld(A0, SP, 24);
    a.ld(A1, SP, 16);
    a.ld(A2, SP, 32);
    a.ld(A3, SP, 40);
    a.call(labels[2]);
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
/// The caller reserves `ceil(rows / chunk_rows) * heads` tasks; every row of
/// a chunk runs the decode kernel with its causal bound.
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
    if prefill.is_some() {
        return Err(ExecutionError("riscv64: no block attention kernel to traverse".into()));
    }
    let head = head_kernel(sdpa)?;
    emit_selector(spec, head)
}

// s1 call, s2 query head, s3 row, s4 end row, s5 KV head, s6 worker scores.
fn emit_selector(spec: AttentionSpec, head: GeneratedKernel) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let head_label = a.label();
    let exit = a.label();
    let f = |field: usize| field as i64;
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5, S6], &[], 0);
    a.mv(S1, A0);
    a.mv(S6, A2);
    a.ld(T0, S1, f(offset_of!(AttentionCall, rows)));
    a.ld(T0, T0, 0);
    a.beqz(T0, exit);
    // One row per chunk: chunks = rows.
    if spec.grouped {
        a.li(T2, spec.gqa as i64);
        a.divu(T3, A1, T2);
        a.remu(S2, A1, T2); // q head within the KV group
        a.divu(T4, T3, T0); // KV head
        a.remu(T3, T3, T0); // chunk
        a.mul(T4, T4, T2);
        a.add(S2, S2, T4);
    } else {
        a.divu(S2, A1, T0);
        a.remu(T3, A1, T0);
    }
    a.li(T2, spec.heads as i64);
    a.bgeu(S2, T2, exit);
    a.mv(S3, T3);
    a.addi(S4, S3, 1);
    a.minu(S4, S4, T0);
    a.li(T2, spec.gqa as i64);
    a.divu(S5, S2, T2);
    let row_loop = a.here();
    kv_offset(&mut a);
    // a5 = visible keys: the causal past plus the row itself.
    a.ld(A5, S1, f(offset_of!(AttentionCall, past)));
    a.ld(A5, A5, 0);
    a.add(A5, A5, S3);
    a.addi(A5, A5, 1);
    arguments(&mut a, spec);
    a.call(head_label);
    a.addi(S3, S3, 1);
    a.bltu(S3, S4, row_loop);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(head_label);
    a.bytes(&head.bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `t5` = byte offset of the KV head in each buffer: every cache starts at position 0.
fn kv_offset(a: &mut Asm) {
    a.ld(T5, S1, offset_of!(AttentionCall, kv_stride) as i64);
    a.mul(T5, T5, S5);
    a.slli(T5, T5, 1);
}

// t5 = byte offset in each KV buffer, a5 = n. The head kernels may clobber
// every argument register; the live state is callee-saved.
fn arguments(a: &mut Asm, spec: AttentionSpec) {
    let f = |field: usize| field as i64;
    a.ld(A1, S1, f(offset_of!(AttentionCall, keys)));
    a.add(A1, A1, T5);
    a.ld(A2, S1, f(offset_of!(AttentionCall, values)));
    a.add(A2, A2, T5);
    a.mv(A3, S6);
    a.ld(A0, S1, f(offset_of!(AttentionCall, q)));
    mul_imm(a, T5, S3, spec.q_row as u64, T6);
    mul_imm(a, T4, S2, spec.q_head as u64, T6);
    a.add(T5, T5, T4);
    a.sh1add(A0, T5, A0);
    a.ld(A4, S1, f(offset_of!(AttentionCall, output)));
    mul_imm(a, T5, S3, spec.out_row as u64, T6);
    mul_imm(a, T4, S2, spec.head_dim as u64, T6);
    a.add(T5, T5, T4);
    a.sh1add(A4, T5, A4);
    a.flw(FA0, S1, f(offset_of!(AttentionCall, scale)));
    a.flw(FA1, S1, f(offset_of!(AttentionCall, sink)));
}
