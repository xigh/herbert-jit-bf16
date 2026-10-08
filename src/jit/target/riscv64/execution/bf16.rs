//! Generated traversal of the bf16 R16 projection kernels.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::compile::CompiledGemm;
use crate::jit::emit::b_block_bytes;
use crate::jit::target::riscv64::util::{exec_err, not_yet_exec, Frame};
use rv::*;
pub use crate::jit::execution::abi::{Bf16Call, Bf16MultiCall};
use std::mem::offset_of;

/// `rd = rs · imm` (through `tmp`).
fn mul_imm(a: &mut Asm, rd: X, rs: X, imm: u64, tmp: X) {
    a.li(tmp, imm as i64);
    a.mul(rd, rs, tmp);
}

/// `rd = base + rs · imm` (through `tmp`).
fn madd_imm(a: &mut Asm, rd: X, rs: X, imm: u64, base: X, tmp: X) {
    a.li(tmp, imm as i64);
    a.mul(tmp, rs, tmp);
    a.add(rd, base, tmp);
}

/// Emit `(call: *const Bf16Call, task: usize)`. The tasks run the grid in
/// N-external / M-internal order; input offsets, variable column widths and
/// M/N tails follow the column plan of `CompiledGemm`.
/// All copied kernels retain their original instruction streams. RISC-V has
/// no matrix engine, so no residue-row tail: the vector tiles alone.
pub fn tile_selector(gemm: &CompiledGemm) -> Result<GeneratedKernel, ExecutionError> {
    if gemm.tail_rows() > 0 {
        return Err(ExecutionError("riscv64: a GEMM with a matrix-engine tail has no traversal here".into()));
    }
    let shape = gemm.shape();
    let mr = gemm.tile().mr;
    let ce = gemm.c_elem_bytes();
    let (format, n_tiles, blobs) = gemm.lowering_parts();
    let kinds_info = gemm.kinds();
    let a_row = shape.k * format.elem_bytes();
    let b_column = shape.k_pairs() * 4;
    debug_assert_eq!(b_column * 16, b_block_bytes(shape.k_pairs()));
    debug_assert_eq!(kinds_info.len() * mr, blobs.len());
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
        let width = if end - start > 1 { n_tiles[start + 1].0 - n_tiles[start].0 } else { 0 };
        runs.push((start, end, n_tiles[start].0, width, kind));
        start = end;
    }
    let mut a = Asm::new();
    let entries: Vec<Label> = blobs.iter().map(|_| a.label()).collect();
    let kinds: Vec<Label> = kinds_info.iter().map(|_| a.label()).collect();
    let exit = a.label();
    let rows_ready = a.label();
    // s1 call, s2 first row, s3 height, s4 column offset (in columns).
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4], &[], 0);
    a.mv(S1, A0);
    a.ld(T2, S1, offset_of!(Bf16Call, rows) as i64);
    a.ld(T0, S1, offset_of!(Bf16Call, row_count) as i64);
    a.beqz(T0, rows_ready);
    a.ld(T2, T0, 0);
    a.bind(rows_ready);
    a.branch_far(Cond::Eq, T2, ZERO, exit);
    a.li(T1, mr as i64);
    // m_tiles (t0) = ceil(rows / mr).
    a.add(T0, T2, T1);
    a.addi(T0, T0, -1);
    a.divu(T0, T0, T1);
    // Column tile (t3) = task / m_tiles; row tile (t4) = task % m_tiles.
    a.divu(T3, A1, T0);
    a.remu(T4, A1, T0);
    mul_imm(&mut a, S2, T4, mr as u64, T5);
    a.sub(S3, T2, S2);
    a.minu(S3, S3, T1);
    for &(start, end, offset, width, kind) in &runs {
        let next = a.label();
        a.li(T5, end as i64);
        a.bgeu(T3, T5, next);
        a.li(T5, start as i64);
        a.sub(S4, T3, T5);
        mul_imm(&mut a, S4, S4, width as u64, T6);
        a.li(T5, offset as i64);
        a.add(S4, S4, T5);
        a.j(kinds[kind]);
        a.bind(next);
    }
    a.j(exit);
    for (kind, &label) in kinds.iter().enumerate() {
        a.bind(label);
        a.ld(A0, S1, offset_of!(Bf16Call, activations) as i64);
        madd_imm(&mut a, A0, S2, a_row as u64, A0, T5);
        a.ld(A1, S1, offset_of!(Bf16Call, weights) as i64);
        madd_imm(&mut a, A1, S4, b_column as u64, A1, T5);
        a.ld(A2, S1, offset_of!(Bf16Call, output) as i64);
        mul_imm(&mut a, T6, S2, shape.n as u64, T5);
        a.add(T6, T6, S4);
        madd_imm(&mut a, A2, T6, ce as u64, A2, T5);
        for height in 1..=mr {
            let next = a.label();
            a.li(T5, height as i64);
            a.bne(S3, T5, next);
            a.call(entries[kind * mr + height - 1]);
            a.j(exit);
            a.bind(next);
        }
        a.j(exit);
    }
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    for (blob, &label) in blobs.iter().zip(&entries) {
        a.align(16);
        a.bind(label);
        a.bytes(blob);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Several projections of one input in one node (`Knobs::fuse_projections`,
/// an AArch64 knob): not emitted on this target, and never asked of it.
pub fn multi_tile_selector(_parts: &[&CompiledGemm]) -> Result<GeneratedKernel, ExecutionError> {
    Err(not_yet_exec("fused bf16 projection"))
}
