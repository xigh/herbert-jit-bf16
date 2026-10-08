//! Small generated data-movement/output kernels for plan lowering.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::aarch64::util::exec_err;
use a64::*;

fn assemble(a: Asm) -> Result<GeneratedKernel, ExecutionError> {
    let bytes = a.finish().map_err(exec_err)?.bytes;
    // SAFETY: these emitters obey AAPCS64 and only access their declared buffers.
    unsafe { GeneratedKernel::new(bytes) }
}

/// Emit a copy of `bytes` from `[source + source_offset]` to
/// `[destination + destination_offset]`, both known at assembly, unrolled in
/// 32-byte `ldp`/`stp` of `q` registers, then the tail by the widest access
/// that fits. It uses `v16`, `v17` and `x16`, `x17`; both sides must be disjoint.
pub(crate) fn copy_constant(a: &mut Asm, destination: R, destination_offset: usize, source: R, source_offset: usize, bytes: usize) {
    if bytes == 0 {
        return;
    }
    // Bases advanced once so that every offset below fits the immediate forms.
    a.mov_imm(X16, source_offset as u64);
    a.add(X16, source, X16);
    a.mov_imm(X17, destination_offset as u64);
    a.add(X17, destination, X17);
    const UNROLL_MAX: usize = 32;
    let pairs = bytes / 32;
    let mut at = 0usize;
    if pairs > UNROLL_MAX {
        let top = a.label();
        a.mov_imm(X15, pairs as u64);
        a.bind(top);
        a.ldp(q(16), q(17), post(X16, 32));
        a.stp(q(16), q(17), post(X17, 32));
        a.subs(X15, X15, 1);
        a.bne(top);
    } else {
        for _ in 0..pairs {
            a.ldp(q(16), q(17), mem(X16, at as i64));
            a.stp(q(16), q(17), mem(X17, at as i64));
            at += 32;
        }
    }
    let mut left = bytes % 32;
    for (width, reg) in [(16usize, q(16)), (8, d(16)), (4, s(16)), (2, h(16)), (1, b(16))] {
        while left >= width {
            a.ldr(reg, mem(X16, at as i64));
            a.str(reg, mem(X17, at as i64));
            at += width;
            left -= width;
        }
    }
}

/// A run-time-length copy of `x2` bytes from `x1` to `x0` (disjoint), by 64,
/// then 16, then 2 and 1 bytes. Uses `v16..v19`, `x2`, `x0`, `x1`.
pub(crate) fn copy_runtime(a: &mut Asm) {
    let big = a.label();
    let mid = a.label();
    let mid_loop = a.label();
    let small = a.label();
    let small_loop = a.label();
    let done = a.label();
    a.bind(big);
    a.cmp(X2, 64);
    a.blo(mid);
    a.ldp(q(16), q(17), mem(X1, 0));
    a.ldp(q(18), q(19), mem(X1, 32));
    a.stp(q(16), q(17), mem(X0, 0));
    a.stp(q(18), q(19), mem(X0, 32));
    a.add(X0, X0, 64);
    a.add(X1, X1, 64);
    a.sub(X2, X2, 64);
    a.b(big);
    a.bind(mid);
    a.bind(mid_loop);
    a.cmp(X2, 16);
    a.blo(small);
    a.ldr(q(16), post(X1, 16));
    a.str(q(16), post(X0, 16));
    a.sub(X2, X2, 16);
    a.b(mid_loop);
    a.bind(small);
    a.bind(small_loop);
    a.cbz(X2, done);
    a.ldrb(W16, post(X1, 1));
    a.strb(W16, post(X0, 1));
    a.sub(X2, X2, 1);
    a.b(small_loop);
    a.bind(done);
}

/// Emit the backend's bf16 argmax `(logits, result_u32, vocabulary_size)`.
/// Uses strict float `>`: first tie wins, NaNs are ignored, all -inf/NaN yields
/// index zero. Vocabulary length must fit a u32 (checked by model lowering).
pub fn argmax_bf16() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let next = a.label();
    let keep = a.label();
    let done = a.label();
    a.mov_imm(W8, 0xff80_0000);
    a.fmov_from_gp(s(0), W8);
    a.mov_imm(X8, 0);
    a.mov_imm(X9, 0);
    a.cbz(X2, done);
    a.bind(next);
    a.ldrh(W10, idx_scaled(X0, X9));
    a.lsl(W10, W10, 16);
    a.fmov_from_gp(s(1), W10);
    a.fcmp(s(1), Some(s(0)));
    // Unordered compares are not greater: NaN candidates are skipped.
    a.b_cond(Cond::Le, keep);
    a.b_cond(Cond::Vs, keep);
    a.fmov(s(0), s(1));
    a.mov(X8, X9);
    a.bind(keep);
    a.add(X9, X9, 1);
    a.cmp(X9, X2);
    a.blo(next);
    a.bind(done);
    a.str(W8, at(X1));
    a.ret();
    assemble(a)
}

/// The key of a bf16 in `w`, into `dst`: a 16-bit unsigned integer that
/// orders as the float does (the sign flipped, a negative complemented), a
/// NaN below everything. Clobbers `w16`, `w17`.
fn bf16_key(a: &mut Asm, dst: R, w: R) {
    // −0 is +0: the float compare holds them equal.
    a.and(W17, w, 0x7fffu32);
    a.cmp(W17, 0u64);
    a.csel(W16, WZR, w, Cond::Eq);
    a.lsr(dst, W16, 15);
    a.neg(dst, dst);
    a.orr(dst, dst, 0x8000u32);
    a.eor(dst, W16, dst);
    a.and(dst, dst, 0xffffu32);
    a.mov_imm(W16, 0x7f80);
    a.cmp(W17, W16);
    a.csel(dst, WZR, dst, Cond::Hi);
}

/// Emit `(logits, slots: *mut u64, vocab, task, tasks)`: the first maximum of
/// slice `task` of the logits cut into `tasks` slices of a length rounded up to
/// eight, written as `index | bits << 32`; a slice past the end writes `-inf`
/// at index 0. The keys of [`bf16_key`] order the bf16 as the float compare
/// of [`argmax_bf16`] does; eight at a time, then the first index of the
/// maximum. NaNs never win; a slice of NaNs reports one, which the merge skips.
pub fn argmax_slices() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let vector = a.label();
    let tail = a.label();
    let tail_loop = a.label();
    let scan = a.label();
    let fold = a.label();
    let found = a.label();
    let empty = a.label();
    let done = a.label();
    // x5 = slice = ceil(vocab / tasks) rounded up to eight; x6 = start; x7 = end.
    a.add(X5, X2, X4);
    a.sub(X5, X5, 1);
    a.udiv(X5, X5, X4);
    a.add(X5, X5, 7);
    a.and(X5, X5, !7u64);
    a.mul(X6, X3, X5);
    a.cmp(X6, X2);
    a.bhs(empty);
    a.add(X7, X6, X5);
    a.cmp(X7, X2);
    a.csel(X7, X2, X7, Cond::Hi);
    // v16 = 0x7fff, v17 = 0x7f80, v18 = 0x8000; v20 = the running maximum of the keys.
    a.mov_imm(W8, 0x7fff);
    a.dup_gp(v(16).h8(), W8);
    a.mov_imm(W8, 0x7f80);
    a.dup_gp(v(17).h8(), W8);
    a.mov_imm(W8, 0x8000);
    a.dup_gp(v(18).h8(), W8);
    a.movi(v(19).d2(), 0);
    a.movi(v(20).d2(), 0);
    a.mov(X9, X6);
    a.bind(vector);
    a.add(X10, X9, 8);
    a.cmp(X10, X7);
    a.bhi(tail);
    a.add(X11, X0, X9.lsl(1));
    a.ldr(q(0), at(X11));
    // −0 cleared to +0; key = w ^ ((w >> 15 signed) | 0x8000), zero where |w| > 0x7f80 (NaN).
    a.and_v(v(3).b16(), v(0).b16(), v(16).b16());
    a.cmeq(v(5).h8(), v(3).h8(), v(19).h8());
    a.bic_v(v(0).b16(), v(0).b16(), v(5).b16());
    a.sshr(v(2).h8(), v(0).h8(), 15);
    a.orr_v(v(2).b16(), v(2).b16(), v(18).b16());
    a.eor_v(v(1).b16(), v(0).b16(), v(2).b16());
    a.cmhi(v(3).h8(), v(3).h8(), v(17).h8());
    a.bic_v(v(1).b16(), v(1).b16(), v(3).b16());
    // max = key > max ? key : max.
    a.cmhi(v(4).h8(), v(1).h8(), v(20).h8());
    a.bsl(v(4).b16(), v(1).b16(), v(20).b16());
    a.mov_v(v(20).b16(), v(4).b16());
    a.mov(X9, X10);
    a.b(vector);
    // The eight lanes folded into w13, then the scalar tail.
    a.bind(tail);
    a.sub(SP, SP, 16);
    a.str(q(20), at(SP));
    a.mov_imm(W13, 0);
    a.mov_imm(X10, 0);
    a.bind(fold);
    a.ldrh(W12, idx_scaled(SP, X10));
    a.cmp(W12, W13);
    a.csel(W13, W12, W13, Cond::Hi);
    a.add(X10, X10, 1);
    a.cmp(X10, 8);
    a.blo(fold);
    a.add(SP, SP, 16);
    a.bind(tail_loop);
    a.cmp(X9, X7);
    a.bhs(scan);
    a.ldrh(W11, idx_scaled(X0, X9));
    bf16_key(&mut a, W12, W11);
    a.cmp(W12, W13);
    a.csel(W13, W12, W13, Cond::Hi);
    a.add(X9, X9, 1);
    a.b(tail_loop);
    // The first index of the maximum key in the slice.
    a.bind(scan);
    a.mov(X9, X6);
    let scan_loop = a.label();
    a.bind(scan_loop);
    a.ldrh(W11, idx_scaled(X0, X9));
    bf16_key(&mut a, W12, W11);
    a.cmp(W12, W13);
    a.beq(found);
    a.add(X9, X9, 1);
    a.cmp(X9, X7);
    a.blo(scan_loop);
    // Unreachable: the maximum came from the slice. Report its first element.
    a.mov(X9, X6);
    a.ldrh(W11, idx_scaled(X0, X9));
    a.bind(found);
    a.orr(X9, X9, X11.lsl(32));
    a.str(X9, idx_scaled(X1, X3));
    a.b(done);
    a.bind(empty);
    a.mov_imm(X9, 0xff80u64 << 32);
    a.str(X9, idx_scaled(X1, X3));
    a.bind(done);
    a.ret();
    assemble(a)
}

/// Emit `(slots, result_u32, tasks)`: the first slot whose value is strictly
/// greater than the running best, from `-inf` at index 0, NaNs skipped — the
/// rule of [`argmax_bf16`] over the maxima of the slices.
pub fn argmax_merge() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let next = a.label();
    let keep = a.label();
    let done = a.label();
    a.mov_imm(W8, 0xff80_0000);
    a.fmov_from_gp(s(0), W8);
    a.mov_imm(X8, 0);
    a.mov_imm(X9, 0);
    a.cbz(X2, done);
    a.bind(next);
    a.ldr(X10, idx_scaled(X0, X9));
    a.lsr(X11, X10, 32);
    a.lsl(W11, W11, 16);
    a.fmov_from_gp(s(1), W11);
    a.fcmp(s(1), Some(s(0)));
    a.b_cond(Cond::Le, keep);
    a.b_cond(Cond::Vs, keep);
    a.fmov(s(0), s(1));
    a.mov(W8, W10);
    a.bind(keep);
    a.add(X9, X9, 1);
    a.cmp(X9, X2);
    a.blo(next);
    a.bind(done);
    a.str(W8, at(X1));
    a.ret();
    assemble(a)
}

/// Emit the token embedding lookup: `(tokens: *const u32, table: *const u16, out: *mut u16, row: usize)`,
/// one bf16 row of the table copied per token.
pub fn embedding() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    a.ldr(W8, at(X0));
    a.mul(X8, X8, X3);
    a.add(X1, X1, X8.lsl(1));
    a.mov(X0, X2);
    a.lsl(X2, X3, 1);
    copy_runtime(&mut a);
    a.ret();
    assemble(a)
}
