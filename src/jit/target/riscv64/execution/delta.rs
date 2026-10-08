//! The generated GatedDeltaNet nodes on RISC-V: L2 norms and gates, the
//! interleave of the gate projections, and the traversals of the conv, the
//! recurrence and the gated norm over rows and heads.
//!
//! The L2 norm keeps the reference order bit for bit (a left
//! to right sum of separate products and additions, a square root and a
//! division, then one product per element), one head at a time in scalar
//! registers. The gates call the emitted `exp`/`log` of `math.rs`.

use super::kernels::copy_constant;
use super::math;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::riscv64::util::{add_imm, exec_err, Frame};
use rv::*;
pub(crate) use crate::jit::execution::abi::{
    ConvCall, GateCall, InterleaveCall, L2Call, L2GatesCall, NormCall, RecurrenceCall, Shape,
};
use std::mem::offset_of;

/// L2 body selection ([`crate::jit::knobs::Knobs::deltanet_l2`]). One body
/// on this target, the scalar chain of the reference; a vector body across
/// heads is still to be written and measured.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum L2Mode {
    /// One head at a time, scalar instructions.
    Scalar,
}

fn l2_mode() -> L2Mode {
    crate::jit::knobs::knobs().deltanet_l2.unwrap_or(L2Mode::Scalar)
}

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

/// `heads` contiguous heads of `dim` f32 at `a0`, eps in `fa0`: `x ← x · 1/√(Σx² + eps)`.
/// Uses `t0..t2`, `ft1..ft3`; advances `a0` past the heads.
fn emit_l2(a: &mut Asm, dim: usize, heads: usize, mode: L2Mode) {
    let L2Mode::Scalar = mode;
    let head_bytes = dim * 4;
    for _ in 0..heads {
        a.fmv_w_x(FT1, ZERO);
        a.mv(T0, A0);
        a.li(T1, dim as i64);
        let sum = a.here();
        a.flw(FT2, T0, 0);
        a.fmul_s(FT2, FT2, FT2);
        a.fadd_s(FT1, FT1, FT2);
        a.addi(T0, T0, 4);
        a.addi(T1, T1, -1);
        a.bnez(T1, sum);
        a.fadd_s(FT1, FT1, FA0);
        a.fsqrt_s(FT1, FT1);
        a.li(T2, i64::from(1.0f32.to_bits()));
        a.fmv_w_x(FT2, T2);
        a.fdiv_s(FT2, FT2, FT1);
        a.mv(T0, A0);
        a.li(T1, dim as i64);
        let apply = a.here();
        a.flw(FT3, T0, 0);
        a.fmul_s(FT3, FT3, FT2);
        a.fsw(FT3, T0, 0);
        a.addi(T0, T0, 4);
        a.addi(T1, T1, -1);
        a.bnez(T1, apply);
        add_imm(a, A0, A0, head_bytes as i64, T2);
    }
}

/// `fa1 = 1.0`.
fn one(a: &mut Asm, f: F) {
    a.li(T0, i64::from(1.0f32.to_bits()));
    a.fmv_w_x(f, T0);
}

/// The gate of one `(row, head)` task: `beta = 1/(1 + exp(−b))`, `decay =
/// exp(−exp(a_log)·dt)` with `dt = x` above twenty, `log(1 + exp(x))` below,
/// `x = a + bias`. `(call: *const GateCall, task)` in `s1`, `s2`; the head
/// in `s3`, the `a` index in `s4`; `0(sp)` holds `dt`.
fn emit_gate_body(a: &mut Asm, nv: usize, exp_label: Label, log_label: Label) {
    let f = |field: usize| field as i64;
    let dt = a.label();
    a.li(T0, nv as i64);
    a.remu(S3, S2, T0);
    a.slli(S4, S2, 1);
    a.sub(S4, S4, S3);
    a.ld(T0, S1, f(offset_of!(GateCall, ab)));
    add_imm(a, T0, T0, (nv * 4) as i64, T1);
    a.sh2add(T0, S4, T0);
    a.flw(FA0, T0, 0);
    a.fneg_s(FA0, FA0);
    a.ld(A0, S1, f(offset_of!(GateCall, tables)));
    a.call(exp_label);
    one(a, FA1);
    a.fadd_s(FA0, FA0, FA1);
    a.fdiv_s(FA1, FA1, FA0);
    a.ld(T0, S1, f(offset_of!(GateCall, beta)));
    a.sh2add(T0, S2, T0);
    a.fsw(FA1, T0, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, ab)));
    a.sh2add(T0, S4, T0);
    a.flw(FA0, T0, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, bias)));
    a.sh1add(T0, S3, T0);
    a.lhu(T1, T0, 0);
    a.slli(T1, T1, 16);
    a.fmv_w_x(FA1, T1);
    a.fadd_s(FA0, FA0, FA1);
    a.li(T0, i64::from(20.0f32.to_bits()));
    a.fmv_w_x(FA1, T0);
    a.flt_s(T0, FA1, FA0);
    a.bnez(T0, dt);
    a.ld(A0, S1, f(offset_of!(GateCall, tables)));
    a.call(exp_label);
    one(a, FA1);
    a.fadd_s(FA0, FA0, FA1);
    a.call(log_label);
    a.bind(dt);
    a.fsw(FA0, SP, 0);
    a.ld(T0, S1, f(offset_of!(GateCall, a_log)));
    a.sh2add(T0, S3, T0);
    a.flw(FA0, T0, 0);
    a.ld(A0, S1, f(offset_of!(GateCall, tables)));
    a.call(exp_label);
    a.fneg_s(FA0, FA0);
    a.flw(FA1, SP, 0);
    a.fmul_s(FA0, FA0, FA1);
    a.call(exp_label);
    a.ld(T0, S1, f(offset_of!(GateCall, decay)));
    a.sh2add(T0, S2, T0);
    a.fsw(FA0, T0, 0);
}

/// `(context, row)` converts two `[rows, nv]` projection outputs into the
/// fixed `[rows, a|b]` gate layout before their scratch areas are reused.
pub(crate) fn interleave(nv: usize) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 8 {
        return Err(ExecutionError("invalid DeltaNet interleave width".into()));
    }
    let mut a = Asm::new();
    let f = |field: usize| field as i64;
    mul_imm(&mut a, T4, A1, (nv * 4) as u64, T5);
    a.ld(A2, A0, f(offset_of!(InterleaveCall, a)));
    a.add(A2, A2, T4);
    a.ld(A3, A0, f(offset_of!(InterleaveCall, b)));
    a.add(A3, A3, T4);
    mul_imm(&mut a, T4, A1, (nv * 8) as u64, T5);
    a.ld(A4, A0, f(offset_of!(InterleaveCall, output)));
    a.add(A4, A4, T4);
    // a and b land side by side: two contiguous runs of nv floats.
    copy_constant(&mut a, A4, 0, A2, 0, nv * 4);
    copy_constant(&mut a, A4, nv * 4, A3, 0, nv * 4);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, row·nv + head)`: the gate `head` of the row, and, on the task of
/// head zero, the L2 of the row's `2·nk` contiguous Q then K heads first — one
/// node instead of two. Rows and heads are independent, so the task order is
/// free; the order inside a head and inside a gate is the reference's.
pub(crate) fn l2_gates(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if shape.key() != shape.nk * shape.hkd {
        return Err(ExecutionError("the L2 expects contiguous Q and K heads".into()));
    }
    let mode = l2_mode();
    let mut a = Asm::new();
    let exp_label = a.label();
    let log_label = a.label();
    let gate = a.label();
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4], &[], 16);
    let l2 = offset_of!(L2GatesCall, l2);
    a.mv(S1, A0);
    a.mv(S2, A1);
    a.li(T0, shape.nv as i64);
    a.divu(T1, S2, T0);
    a.remu(T2, S2, T0);
    a.bnez(T2, gate);
    a.ld(A0, S1, (l2 + offset_of!(L2Call, qkv)) as i64);
    madd_imm(&mut a, A0, T1, (shape.conv() * 4) as u64, A0, T0);
    a.flw(FA0, S1, (l2 + offset_of!(L2Call, eps)) as i64);
    emit_l2(&mut a, shape.hkd, 2 * shape.nk, mode);
    a.bind(gate);
    add_imm(&mut a, S1, S1, offset_of!(L2GatesCall, gates) as i64, T0);
    emit_gate_body(&mut a, shape.nv, exp_label, log_label);
    frame.leave(&mut a);
    a.ret();
    a.bind(exp_label);
    math::exp_body(&mut a, fma);
    a.bind(log_label);
    math::log_body(&mut a, fma);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, V-head)` traverses all positions causally and invokes the
/// recurrence kernel. Separate heads own disjoint state/output slices.
pub(crate) fn recurrence(
    shape: Shape,
    kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let wanted = crate::jit::recurrence::spec_of_knobs(shape.hvd);
    let split = crate::jit::recurrence::vd_split(shape.hvd);
    let width = if split > 1 { shape.hvd / split } else { shape.hvd };
    let bytes = match kernel.filter(|k| k.spec() == wanted) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_recurrence::emit_recurrence(&wanted).map_err(exec_err)?,
    };
    let mut a = Asm::new();
    let entry = a.label();
    let exit = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 head, s3 q/k head offset, s4 state, s5 rows, s6 row;
    // with a split: task = head · split + slice; s7 = the slice's byte offset
    // into the value columns of the state, the output and v.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5, S6, S7], &[], 0);
    a.mv(S1, A0);
    if split > 1 {
        a.li(T0, split as i64);
        a.divu(S2, A1, T0);
        a.remu(S7, A1, T0);
        mul_imm(&mut a, S7, S7, (width * 4) as u64, T0);
    } else {
        a.mv(S2, A1);
        a.li(S7, 0);
    }
    a.li(T0, (shape.nv / shape.nk) as i64);
    a.divu(S3, S2, T0);
    mul_imm(&mut a, S3, S3, (shape.hkd * 4) as u64, T0);
    a.ld(S4, S1, f(offset_of!(RecurrenceCall, state)));
    madd_imm(&mut a, S4, S2, (shape.hkd * shape.hvd * 4) as u64, S4, T0);
    a.add(S4, S4, S7);
    a.ld(S5, S1, f(offset_of!(RecurrenceCall, rows)));
    a.ld(S5, S5, 0);
    a.li(S6, 0);
    a.beqz(S5, exit);
    let again = a.here();
    a.mv(A0, S4);
    a.ld(A1, S1, f(offset_of!(RecurrenceCall, output)));
    madd_imm(&mut a, A1, S6, (shape.value() * 4) as u64, A1, T0);
    madd_imm(&mut a, A1, S2, (shape.hvd * 4) as u64, A1, T0);
    a.add(A1, A1, S7);
    a.ld(T1, S1, f(offset_of!(RecurrenceCall, qkv)));
    madd_imm(&mut a, T1, S6, (shape.conv() * 4) as u64, T1, T0);
    a.add(A2, T1, S3);
    add_imm(&mut a, A3, A2, (shape.key() * 4) as i64, T0);
    add_imm(&mut a, A4, T1, (shape.key() * 8) as i64, T0);
    madd_imm(&mut a, A4, S2, (shape.hvd * 4) as u64, A4, T0);
    a.add(A4, A4, S7);
    a.li(A5, shape.hkd as i64);
    mul_imm(&mut a, T1, S6, shape.nv as u64, T0);
    a.add(T1, T1, S2);
    a.ld(T0, S1, f(offset_of!(RecurrenceCall, beta)));
    a.sh2add(T0, T1, T0);
    a.flw(FA0, T0, 0);
    a.ld(T0, S1, f(offset_of!(RecurrenceCall, decay)));
    a.sh2add(T0, T1, T0);
    a.flw(FA1, T0, 0);
    a.li(T0, i64::from((1.0 / (shape.hkd as f32).sqrt()).to_bits()));
    a.fmv_w_x(FA2, T0);
    a.call(entry);
    a.addi(S6, S6, 1);
    a.bltu(S6, S5, again);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, channel-block)` traverses positions in order, 16 channels per
/// independent task, through the conv kernel.
pub(crate) fn convolution(
    shape: Shape,
    kernel: Option<&crate::jit::conv::CompiledConv>,
    _fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    if !shape.conv().is_multiple_of(16) {
        return Err(ExecutionError(format!(
            "the RISC-V conv serves whole blocks of 16 channels, not {}",
            shape.conv()
        )));
    }
    let bytes = match kernel.filter(|k| k.k() == shape.ck) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_conv::emit_conv(&crate::jit::emit_conv::ConvSpec { k: shape.ck })
            .map_err(exec_err)?,
    };
    let mut a = Asm::new();
    let entry = a.label();
    let exit = a.label();
    let f = |field: usize| field as i64;
    // s1 call, s2 data cursor, s3 weights, s4 state, s5 rows left.
    let frame = Frame::enter(&mut a, true, &[S1, S2, S3, S4, S5], &[], 0);
    a.mv(S1, A0);
    a.ld(S5, S1, f(offset_of!(ConvCall, rows)));
    a.ld(S5, S5, 0);
    a.ld(S2, S1, f(offset_of!(ConvCall, data)));
    madd_imm(&mut a, S2, A1, 64, S2, T0);
    a.ld(S4, S1, f(offset_of!(ConvCall, state)));
    madd_imm(&mut a, S4, A1, (64 * shape.ck) as u64, S4, T0);
    a.ld(S3, S1, f(offset_of!(ConvCall, weights)));
    madd_imm(&mut a, S3, A1, (32 * shape.ck) as u64, S3, T0);
    a.beqz(S5, exit);
    let again = a.here();
    a.mv(A0, S2);
    a.mv(A1, S3);
    a.mv(A2, S4);
    a.li(A3, 1);
    a.call(entry);
    add_imm(&mut a, S2, S2, (shape.conv() * 4) as i64, T0);
    a.addi(S5, S5, -1);
    a.bnez(S5, again);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// `(context, row)` applies the gated norm kernel to one V-head row and writes
/// the bf16 projection input.
pub(crate) fn normed_gate(
    dim: usize,
    kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
    _fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gated norm width".into()));
    }
    let bytes = match kernel.filter(|k| k.dim() == dim) {
        Some(k) => k.lowering_bytes(),
        None => crate::jit::emit_gated_norm::emit_gated_norm(&crate::jit::emit_gated_norm::GatedNormSpec { dim })
            .map_err(exec_err)?,
    };
    let mut a = Asm::new();
    let entry = a.label();
    let f = |field: usize| field as i64;
    let frame = Frame::enter(&mut a, true, &[], &[], 0);
    a.mv(T3, A0);
    mul_imm(&mut a, T4, A1, (dim * 4) as u64, T5);
    mul_imm(&mut a, T6, A1, (dim * 2) as u64, T5);
    a.ld(A0, T3, f(offset_of!(NormCall, output)));
    a.add(A0, A0, T4);
    a.ld(A1, T3, f(offset_of!(NormCall, gate)));
    a.add(A1, A1, T4);
    a.ld(A2, T3, f(offset_of!(NormCall, weights)));
    a.ld(A3, T3, f(offset_of!(NormCall, output_bf16)));
    a.add(A3, A3, T6);
    a.li(A4, 1);
    a.flw(FA0, T3, f(offset_of!(NormCall, eps)));
    a.call(entry);
    frame.leave(&mut a);
    a.ret();
    a.align(16);
    a.bind(entry);
    a.bytes(&bytes);
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
