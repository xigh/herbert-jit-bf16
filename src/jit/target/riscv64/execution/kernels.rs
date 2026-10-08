//! Small generated data-movement/output kernels for plan lowering.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::riscv64::util::exec_err;
use rv::*;

fn assemble(a: Asm) -> Result<GeneratedKernel, ExecutionError> {
    let bytes = a.finish().map_err(exec_err)?.bytes;
    // SAFETY: these emitters obey LP64D and only access their declared buffers.
    unsafe { GeneratedKernel::new(bytes) }
}

const E8M8: Vtype = Vtype::new(Sew::E8, Lmul::M8);

/// Emit a copy of `bytes` from `[source + source_offset]` to
/// `[destination + destination_offset]`, both known at assembly: a
/// strip-mined byte copy at `e8, m8`, whatever VLEN. It uses `v8..v15` and
/// `t0..t3`; both sides must be disjoint.
pub(crate) fn copy_constant(a: &mut Asm, destination: X, destination_offset: usize, source: X, source_offset: usize, bytes: usize) {
    if bytes == 0 {
        return;
    }
    a.li(T0, source_offset as i64);
    a.add(T0, source, T0);
    a.li(T1, destination_offset as i64);
    a.add(T1, destination, T1);
    a.li(T2, bytes as i64);
    let top = a.here();
    a.vsetvli(T3, T2, E8M8);
    a.vle(Sew::E8, V8, T0);
    a.vse(Sew::E8, V8, T1);
    a.add(T0, T0, T3);
    a.add(T1, T1, T3);
    a.sub(T2, T2, T3);
    a.bnez(T2, top);
}

/// A run-time-length copy of `a2` bytes from `a1` to `a0` (disjoint), the same
/// strip-mined loop. Uses `v8..v15`, `t3`, `a0..a2`.
pub(crate) fn copy_runtime(a: &mut Asm) {
    let done = a.label();
    a.beqz(A2, done);
    let top = a.here();
    a.vsetvli(T3, A2, E8M8);
    a.vle(Sew::E8, V8, A1);
    a.vse(Sew::E8, V8, A0);
    a.add(A0, A0, T3);
    a.add(A1, A1, T3);
    a.sub(A2, A2, T3);
    a.bnez(A2, top);
    a.bind(done);
}

/// `f = the bf16 at [base + 2·index]`, widened; its sixteen bits in `raw`.
fn load_bf16(a: &mut Asm, f: F, raw: X, base: X, index: X, tmp: X) {
    a.sh1add(tmp, index, base);
    a.lhu(raw, tmp, 0);
    a.slli(tmp, raw, 16);
    a.fmv_w_x(f, tmp);
}

/// Emit the backend's bf16 argmax `(logits, result_u32, vocabulary_size)`.
/// Uses strict float `>` (`flt.s`, false on a NaN): first tie wins, NaNs are
/// ignored, all -inf/NaN yields index zero. Vocabulary length must fit a u32
/// (checked by model lowering).
pub fn argmax_bf16() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let next = a.label();
    let keep = a.label();
    let done = a.label();
    a.li(T0, 0xff80_0000);
    a.fmv_w_x(FT0, T0);
    a.li(T0, 0);
    a.li(T1, 0);
    a.beqz(A2, done);
    a.bind(next);
    load_bf16(&mut a, FT1, T3, A0, T1, T2);
    a.flt_s(T4, FT0, FT1);
    a.beqz(T4, keep);
    a.fmv_s(FT0, FT1);
    a.mv(T0, T1);
    a.bind(keep);
    a.addi(T1, T1, 1);
    a.bltu(T1, A2, next);
    a.bind(done);
    a.sw(T0, A1, 0);
    a.ret();
    assemble(a)
}

/// Emit `(logits, slots: *mut u64, vocab, task, tasks)`: the first maximum of
/// slice `task` of the logits cut into `tasks` slices of a length rounded up to
/// eight, written as `index | bits << 32`; a slice past the end writes `-inf`
/// at index 0. The compare is the strict one of [`argmax_bf16`], from `-inf`:
/// a slice of `-inf` or NaNs reports its first element, which the merge
/// never takes over its `-inf` start.
pub fn argmax_slices() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    let scan = a.label();
    let keep = a.label();
    let empty = a.label();
    let done = a.label();
    // t0 = slice = ceil(vocab / tasks) rounded up to eight; t1 = start; t2 = end.
    a.add(T0, A2, A4);
    a.addi(T0, T0, -1);
    a.divu(T0, T0, A4);
    a.addi(T0, T0, 7);
    a.andi(T0, T0, -8);
    a.mul(T1, A3, T0);
    a.bgeu(T1, A2, empty);
    a.add(T2, T1, T0);
    a.minu(T2, T2, A2);
    // ft0 = the best value, t3 = its index, t4 = its bits: the first element.
    a.li(T5, 0xff80_0000);
    a.fmv_w_x(FT0, T5);
    a.mv(T3, T1);
    load_bf16(&mut a, FT1, T4, A0, T1, T5);
    a.mv(A5, T1);
    a.bind(scan);
    load_bf16(&mut a, FT1, T6, A0, A5, T5);
    a.flt_s(A6, FT0, FT1);
    a.beqz(A6, keep);
    a.fmv_s(FT0, FT1);
    a.mv(T3, A5);
    a.mv(T4, T6);
    a.bind(keep);
    a.addi(A5, A5, 1);
    a.bltu(A5, T2, scan);
    a.slli(T4, T4, 32);
    a.or(T3, T3, T4);
    a.sh3add(T5, A3, A1);
    a.sd(T3, T5, 0);
    a.j(done);
    a.bind(empty);
    a.li(T3, 0xff80i64 << 32);
    a.sh3add(T5, A3, A1);
    a.sd(T3, T5, 0);
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
    a.li(T0, 0xff80_0000);
    a.fmv_w_x(FT0, T0);
    a.li(T0, 0);
    a.li(T1, 0);
    a.beqz(A2, done);
    a.bind(next);
    a.sh3add(T2, T1, A0);
    a.ld(T2, T2, 0);
    a.srli(T3, T2, 32);
    a.slli(T3, T3, 16);
    a.fmv_w_x(FT1, T3);
    a.flt_s(T4, FT0, FT1);
    a.beqz(T4, keep);
    a.fmv_s(FT0, FT1);
    a.zext_w(T0, T2);
    a.bind(keep);
    a.addi(T1, T1, 1);
    a.bltu(T1, A2, next);
    a.bind(done);
    a.sw(T0, A1, 0);
    a.ret();
    assemble(a)
}

/// Emit the token embedding lookup: `(tokens: *const u32, table: *const u16, out: *mut u16, row: usize)`,
/// one bf16 row of the table copied per token.
pub fn embedding() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = Asm::new();
    a.lwu(T0, A0, 0);
    a.mul(T0, T0, A3);
    a.sh1add(A1, T0, A1);
    a.mv(A0, A2);
    a.slli(A2, A3, 1);
    copy_runtime(&mut a);
    a.ret();
    assemble(a)
}
