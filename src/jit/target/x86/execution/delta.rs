//! The generated GatedDeltaNet nodes: L2 norms and gates, the interleave of
//! the gate projections, the conv, the recurrence and the gated norm, as
//! traversals over rows and heads that call the compiled kernels.
//!
//! Every stage is emitted code on both arms; what changes is the body, never
//! the bits it must return:
//!
//! | node | AVX-512 | AVX2 |
//! |---|---|---|
//! | Q/K L2 norm and gates (`l2_gates`) | `il4`, four heads interleaved | `simd`: vector multiplies, the scalar order unrolled by four |
//! | gates β/decay | emitted scalar, `exp`/`log` of `math` | the same body |
//! | conv + SiLU | `conv.rs` vector kernel, scalar tail channels | `delta_scalar::convolution` |
//! | recurrence | `recurrence.rs` vector kernel | `delta_scalar::recurrence` |
//! | gated RMS norm | `gated_norm.rs` vector kernel | `delta_scalar::gated_norm` |
//!
//! The vector conv, recurrence and gated norm exist only in zmm; without
//! AVX-512 the traversal enters the emitted scalar body of `delta_scalar.rs`.
//! Each body is held to the scalar references with `to_bits` equality.

use super::{math, ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
pub(crate) use crate::jit::execution::abi::{GateCall, InterleaveCall, L2Call, L2GatesCall, Shape, RecurrenceCall, ConvCall, NormCall};
fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}
/// One f32 head `(data, eps)`, preserving the reference left-to-right sum and
/// distinct multiply/add, square root and division (no reassociation or FMA).
pub(crate) fn l2norm(dim: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize {
        return Err(ExecutionError("invalid DeltaNet L2 width".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut sum = a.create_label();
        let mut apply = a.create_label();
        a.xorps(xmm1, xmm1)?;
        a.xor(eax, eax)?;
        a.set_label(&mut sum)?;
        a.movd(xmm2, dword_ptr(rdi + rax * 4))?;
        a.mulss(xmm2, xmm2)?;
        a.addss(xmm1, xmm2)?;
        a.inc(rax)?;
        a.cmp(rax, dim as i32)?;
        a.jb(sum)?;
        a.addss(xmm1, xmm0)?;
        a.sqrtss(xmm1, xmm1)?;
        a.mov(ecx, 1.0f32.to_bits())?;
        a.movd(xmm0, ecx)?;
        a.divss(xmm0, xmm1)?;
        a.xor(eax, eax)?;
        a.set_label(&mut apply)?;
        a.movd(xmm2, dword_ptr(rdi + rax * 4))?;
        a.mulss(xmm2, xmm0)?;
        a.movd(dword_ptr(rdi + rax * 4), xmm2)?;
        a.inc(rax)?;
        a.cmp(rax, dim as i32)?;
        a.jb(apply)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// AVX2 application with the exact scalar reduction order.
/// Only the independent final multiplies are vectorized; the reduction is
/// unrolled four elements without reassociation or FMA.
fn l2norm_simd(dim: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize {
        return Err(ExecutionError("invalid DeltaNet L2 width".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut sum = a.create_label();
        let mut apply = a.create_label();
        a.xorps(xmm1, xmm1)?;
        a.xor(eax, eax)?;
        let four = dim / 4 * 4;
        if four != 0 {
            a.set_label(&mut sum)?;
            for lane in 0..4 {
                a.movd(xmm2, dword_ptr(rdi + rax * 4 + lane * 4))?;
                a.mulss(xmm2, xmm2)?;
                a.addss(xmm1, xmm2)?;
            }
            a.add(rax, 4)?;
            a.cmp(rax, four as i32)?;
            a.jb(sum)?;
        }
        for _ in four..dim {
            a.movd(xmm2, dword_ptr(rdi + rax * 4))?;
            a.mulss(xmm2, xmm2)?;
            a.addss(xmm1, xmm2)?;
            a.inc(rax)?;
        }
        a.addss(xmm1, xmm0)?;
        a.sqrtss(xmm1, xmm1)?;
        a.mov(ecx, 1.0f32.to_bits())?;
        a.movd(xmm0, ecx)?;
        a.divss(xmm0, xmm1)?;
        a.xor(eax, eax)?;
        let eight = dim / 8 * 8;
        if eight != 0 {
            a.vbroadcastss(ymm3, xmm0)?;
            a.set_label(&mut apply)?;
            a.vmovups(ymm2, ymmword_ptr(rdi + rax * 4))?;
            a.vmulps(ymm2, ymm2, ymm3)?;
            a.vmovups(ymmword_ptr(rdi + rax * 4), ymm2)?;
            a.add(rax, 8)?;
            a.cmp(rax, eight as i32)?;
            a.jb(apply)?;
        }
        a.vzeroupper()?;
        for _ in eight..dim {
            a.movd(xmm2, dword_ptr(rdi + rax * 4))?;
            a.mulss(xmm2, xmm0)?;
            a.movd(dword_ptr(rdi + rax * 4), xmm2)?;
            a.inc(rax)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
/// L2 body selection, resolved once at model construction from
/// `HERBERT_JIT_SPMD_L2`:
/// `scalar` (original), `simd` (AVX2), `il4` (AVX-512, four heads
/// interleaved, bit-exact) and `vec` (AVX-512 FMA vector reduction, NOT
/// bit-exact: reassociated sum, up to a few ulp). Without the variable the
/// default is `il4` when AVX-512F is present and the head width allows it
/// (`il4_width_ok`), else `simd` under AVX2, else `scalar`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum L2Mode {
    /// One head at a time, scalar SSE, the original.
    Scalar,
    /// AVX2: vector multiplies, the scalar order unrolled by four.
    Simd,
    /// AVX-512: four heads interleaved, bit-exact.
    Il4,
    /// AVX-512 FMA vector reduction, reassociated: not bit-exact.
    Vec,
}
/// Widths the `il4` body accepts (also checked by `l2norm_il4`).
pub(crate) fn il4_width_ok(dim: usize) -> bool {
    dim != 0 && dim.is_multiple_of(16) && dim <= (i32::MAX / 16) as usize
}
/// The body in force ([`crate::jit::knobs::Knobs::deltanet_l2`]): the best of
/// the ISA when none is imposed; an imposed body the ISA cannot run is refused.
/// `dim` is the head width of the model; it only steers the default.
pub(crate) fn l2_mode(dim: usize) -> Result<L2Mode, ExecutionError> {
    let avx2 = crate::core::isa::feature("avx2");
    let avx512 = crate::core::isa::feature("avx512f");
    match crate::jit::knobs::knobs().deltanet_l2 {
        None => Ok(if avx512 && il4_width_ok(dim) {
            L2Mode::Il4
        } else if avx2 {
            L2Mode::Simd
        } else {
            L2Mode::Scalar
        }),
        Some(L2Mode::Scalar) => Ok(L2Mode::Scalar),
        Some(L2Mode::Simd) if avx2 => Ok(L2Mode::Simd),
        Some(mode @ (L2Mode::Il4 | L2Mode::Vec)) if avx512 => Ok(mode),
        Some(L2Mode::Simd) => Err(ExecutionError("deltanet_l2 = simd requires AVX2".into())),
        Some(mode) => Err(ExecutionError(format!("deltanet_l2 = {mode:?} requires AVX-512F"))),
    }
}

/// `acc = 1.0 / sqrt(acc + eps)`; eps in xmm0, 1.0 in xmm15.
fn l2_finish(a: &mut CodeAssembler, acc: AsmRegisterXmm) -> Result<(), iced_x86::IcedError> {
    a.vaddss(acc, acc, xmm0)?;
    a.vsqrtss(acc, acc, acc)?;
    a.vdivss(acc, xmm15, acc)
}
/// Multiply `dim` (multiple of 16) f32 at `rdi + off` by the scalar in `acc`
/// with zmm multiplies on memory operands; `acc` is clobbered (broadcast).
fn l2_apply_zmm(
    a: &mut CodeAssembler, off: i32, dim: usize, acc: AsmRegisterXmm, accz: AsmRegisterZmm, t: [AsmRegisterZmm; 2],
) -> Result<(), iced_x86::IcedError> {
    a.vbroadcastss(accz, acc)?;
    for blk in 0..(dim / 16) as i32 {
        let z = t[blk as usize % 2];
        a.vmulps(z, accz, zmmword_ptr(rdi + off + blk * 64))?;
        a.vmovups(zmmword_ptr(rdi + off + blk * 64), z)?;
    }
    Ok(())
}
/// Sequential sum of squares of `il` contiguous heads (`dim` multiple of 16)
/// at `rdi + base`, interleaved element by element: chain `h` uses
/// accumulator xmm(1+h) and temporary xmm(1+il+h). Each chain keeps the exact
/// reference order; only the instruction stream interleaves them.
fn l2_reduce_interleaved(
    a: &mut CodeAssembler, base: i32, dim: usize, il: usize,
) -> Result<(), iced_x86::IcedError> {
    let acc = [xmm1, xmm2, xmm3, xmm4];
    let tmp = [xmm5, xmm6, xmm7, xmm8];
    for h in 0..il {
        a.vxorps(acc[h], acc[h], acc[h])?;
    }
    let mut lp = a.create_label();
    a.xor(eax, eax)?;
    a.set_label(&mut lp)?;
    for e in 0..16i32 {
        for h in 0..il {
            let off = base + (h * dim * 4) as i32 + e * 4;
            a.vmovss(tmp[h], dword_ptr(rdi + rax + off))?;
            a.vmulss(tmp[h], tmp[h], tmp[h])?;
            a.vaddss(acc[h], acc[h], tmp[h])?;
        }
    }
    a.add(rax, 64)?;
    a.cmp(rax, (dim * 4) as i32)?;
    a.jb(lp)
}
/// `heads` contiguous heads of `dim` f32 `(data, eps)`, AVX-512, bit-exact:
/// groups of four heads interleaved, remaining heads one at a time.
pub(crate) fn l2norm_il4(dim: usize, heads: usize) -> Result<GeneratedKernel, ExecutionError> {
    if !il4_width_ok(dim) || heads == 0 {
        return Err(ExecutionError("il4 L2 requires a non-zero width multiple of 16".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.mov(ecx, 1.0f32.to_bits())?;
        a.vmovd(xmm15, ecx)?;
        let groups = heads / 4;
        if groups > 0 {
            let mut gl = a.create_label();
            a.mov(edx, groups as u32)?;
            a.set_label(&mut gl)?;
            l2_reduce_interleaved(a, 0, dim, 4)?;
            for acc in [xmm1, xmm2, xmm3, xmm4] {
                l2_finish(a, acc)?;
            }
            for (h, (acc, accz)) in [(xmm1, zmm1), (xmm2, zmm2), (xmm3, zmm3), (xmm4, zmm4)].into_iter().enumerate() {
                l2_apply_zmm(a, (h * dim * 4) as i32, dim, acc, accz, [zmm5, zmm6])?;
            }
            a.add(rdi, (4 * dim * 4) as i32)?;
            a.dec(edx)?;
            a.jnz(gl)?;
        }
        for _ in 0..heads % 4 {
            l2_reduce_interleaved(a, 0, dim, 1)?;
            l2_finish(a, xmm1)?;
            l2_apply_zmm(a, 0, dim, xmm1, zmm1, [zmm5, zmm6])?;
            a.add(rdi, (dim * 4) as i32)?;
        }
        a.vzeroupper()?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
/// One head `(data, eps)`, AVX-512 FMA: sixteen partial sums in two zmm
/// accumulators, horizontal sum, then the zmm apply. NOT bit-exact with the
/// reference order.
pub(crate) fn l2norm_vec(dim: usize) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || !dim.is_multiple_of(16) || dim > (i32::MAX / 16) as usize {
        return Err(ExecutionError("vec L2 requires a non-zero width multiple of 16".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.mov(ecx, 1.0f32.to_bits())?;
        a.vmovd(xmm15, ecx)?;
        a.vxorps(zmm1, zmm1, zmm1)?;
        a.vxorps(zmm2, zmm2, zmm2)?;
        for blk in 0..(dim / 16) as i32 {
            let (acc, t) = if blk % 2 == 0 { (zmm1, zmm3) } else { (zmm2, zmm4) };
            a.vmovups(t, zmmword_ptr(rdi + blk * 64))?;
            a.vfmadd231ps(acc, t, t)?;
        }
        a.vaddps(zmm1, zmm1, zmm2)?;
        a.vextractf64x4(ymm3, zmm1, 1)?;
        a.vaddps(ymm1, ymm1, ymm3)?;
        a.vextractf128(xmm3, ymm1, 1)?;
        a.vaddps(xmm1, xmm1, xmm3)?;
        a.vshufps(xmm3, xmm1, xmm1, 0b00_00_11_10)?;
        a.vaddps(xmm1, xmm1, xmm3)?;
        a.vmovshdup(xmm3, xmm1)?;
        a.vaddss(xmm1, xmm1, xmm3)?;
        l2_finish(a, xmm1)?;
        l2_apply_zmm(a, 0, dim, xmm1, zmm1, [zmm5, zmm6])?;
        a.vzeroupper()?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// One `(row, head)` task `(context, row*nv+head)`; output order is beta then
/// decay, with the existing softplus linear branch strictly above twenty.
pub(crate) fn gates(nv: usize, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gate heads".into()));
    }
    let exp = math::exp(fma)?;
    let log = math::log(fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut exp_label = a.create_label();
    let mut log_label = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut dt = a.create_label();
        for r in [rbx, rbp, r12] {
            a.push(r)?;
        }
        a.sub(rsp, 16)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(rax, rsi)?;
        a.xor(edx, edx)?;
        a.mov(rcx, nv as u64)?;
        a.div(rcx)?;
        a.mov(r12, rdx)?;
        a.lea(rax, ptr(rbp * 2))?;
        a.sub(rax, r12)?;
        a.mov(
            rcx,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, ab) as i32),
        )?;
        a.mov(edx, dword_ptr(rcx + rax * 4 + (nv * 4) as i32))?;
        a.xor(edx, 0x80000000u32)?;
        a.movd(xmm0, edx)?;
        a.mov(
            rdi,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, tables) as i32),
        )?;
        a.call(exp_label)?;
        a.mov(eax, 1.0f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.divss(xmm1, xmm0)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, beta) as i32),
        )?;
        a.movd(dword_ptr(rax + rbp * 4), xmm1)?;
        a.lea(rax, ptr(rbp * 2))?;
        a.sub(rax, r12)?;
        a.mov(
            rcx,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, ab) as i32),
        )?;
        a.movd(xmm0, dword_ptr(rcx + rax * 4))?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, bias) as i32),
        )?;
        a.movzx(eax, word_ptr(rax + r12 * 2))?;
        a.shl(eax, 16)?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.mov(eax, 20.0f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.ucomiss(xmm0, xmm1)?;
        a.ja(dt)?;
        a.call(exp_label)?;
        a.mov(eax, 1.0f32.to_bits())?;
        a.movd(xmm1, eax)?;
        a.addss(xmm0, xmm1)?;
        a.call(log_label)?;
        a.set_label(&mut dt)?;
        a.movd(dword_ptr(rsp), xmm0)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, a_log) as i32),
        )?;
        a.movd(xmm0, dword_ptr(rax + r12 * 4))?;
        a.call(exp_label)?;
        a.movd(eax, xmm0)?;
        a.xor(eax, 0x80000000u32)?;
        a.movd(xmm0, eax)?;
        a.mulss(xmm0, dword_ptr(rsp))?;
        a.call(exp_label)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(GateCall, decay) as i32),
        )?;
        a.movd(dword_ptr(rax + rbp * 4), xmm0)?;
        a.add(rsp, 16)?;
        for r in [r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut exp_label).map_err(err)?;
    a.db(&exp.bytes).map_err(err)?;
    a.set_label(&mut log_label).map_err(err)?;
    a.db(&log.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(context, row)` converts two `[rows, nv]` projection outputs into the
/// fixed `[rows, a|b]` gate layout before their scratch areas are reused.
pub(crate) fn interleave(nv: usize) -> Result<GeneratedKernel, ExecutionError> {
    if nv == 0 || nv > i32::MAX as usize / 8 {
        return Err(ExecutionError("invalid DeltaNet interleave width".into()));
    }
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.imul_3(rax, rsi, (nv * 4) as i32)?;
        a.mov(
            r8,
            qword_ptr(rdi + std::mem::offset_of!(InterleaveCall, a) as i32),
        )?;
        a.add(r8, rax)?;
        a.mov(
            r9,
            qword_ptr(rdi + std::mem::offset_of!(InterleaveCall, b) as i32),
        )?;
        a.add(r9, rax)?;
        a.imul_3(rax, rsi, (nv * 8) as i32)?;
        a.mov(
            r10,
            qword_ptr(rdi + std::mem::offset_of!(InterleaveCall, output) as i32),
        )?;
        a.add(r10, rax)?;
        // a and b land side by side, so this is two contiguous runs of nv floats
        // and not the interleaving its name suggests: each moves in one piece
        // (a string move, not one f32 per turn).
        super::kernels::copy_constant(a, r10, 0, r8, 0, nv * 4)?;
        super::kernels::copy_constant(a, r10, nv * 4, r9, 0, nv * 4)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(context, row·nv + head)`: the gate `head` of the row, and, on the task of
/// head zero, the L2 of the row's Q then K heads first — one node instead
/// of `delta-qk-l2` then `delta-gates`, which cost a barrier between them.
/// The grid is `tasks_per_tile = nv`
/// so that, at `m = 1`, the `nv` gates of the single row still spread over
/// the team (one task per row put them all on participant zero, +0.018 ms per
/// decode step at the counter); at `m ≥ 4` the contiguous task ranges give
/// each participant whole rows. The L2 and the gates of a row are
/// independent (the L2 reads and writes `q`/`k` of `dn_qkv`, a gate reads
/// `dn_ab` and writes its `beta`/`decay`), so the task order is free. The L2
/// body is that of the L2 kernels (`il4`: the `2·nk` contiguous heads
/// in one call; the per-head bodies: `nk` Q then K pairs, Q before K as the
/// reference row traversal), the gate body is [`gates`] on the same task index.
/// The order inside a head and inside a gate is untouched: the same bits as
/// the two nodes, whatever the task ranges.
pub(crate) fn l2_gates(shape: Shape, fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let mode = l2_mode(shape.hkd)?;
    // `pairs == 0`: one interleaved call per row; else `nk` calls per Q/K half.
    let (inner, pairs) = match mode {
        L2Mode::Il4 => (l2norm_il4(shape.hkd, 2 * shape.nk)?, 0),
        L2Mode::Scalar => (l2norm(shape.hkd)?, shape.nk),
        L2Mode::Simd => (l2norm_simd(shape.hkd)?, shape.nk),
        L2Mode::Vec => (l2norm_vec(shape.hkd)?, shape.nk),
    };
    let gate = gates(shape.nv, fma)?;
    let qkv_off = (std::mem::offset_of!(L2GatesCall, l2) + std::mem::offset_of!(L2Call, qkv)) as i32;
    let eps_off = (std::mem::offset_of!(L2GatesCall, l2) + std::mem::offset_of!(L2Call, eps)) as i32;
    let gates_off = std::mem::offset_of!(L2GatesCall, gates) as i32;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut l2_entry = a.create_label();
    let mut gate_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        // rbx = context, rbp = task, r12 = the row's qkv base, r13 = head
        // pair counter; the bodies are SysV64 functions and preserve them.
        // Four pushes and eight bytes keep the stack 16-aligned at the calls.
        let mut gate = a.create_label();
        for r in [rbx, rbp, r12, r13] {
            a.push(r)?;
        }
        a.sub(rsp, 8)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(rax, rsi)?;
        a.xor(edx, edx)?;
        a.mov(rcx, shape.nv as u64)?;
        a.div(rcx)?;
        a.test(rdx, rdx)?;
        a.jnz(gate)?;
        a.imul_3(r12, rax, (shape.conv() * 4) as i32)?;
        a.add(r12, qword_ptr(rbx + qkv_off))?;
        if pairs == 0 {
            a.mov(rdi, r12)?;
            a.movd(xmm0, dword_ptr(rbx + eps_off))?;
            a.call(l2_entry)?;
        } else {
            let mut lp = a.create_label();
            a.xor(r13d, r13d)?;
            a.set_label(&mut lp)?;
            for half in [0, shape.key() * 4] {
                a.imul_3(rdi, r13, (shape.hkd * 4) as i32)?;
                a.add(rdi, r12)?;
                if half != 0 {
                    a.add(rdi, half as i32)?;
                }
                a.movd(xmm0, dword_ptr(rbx + eps_off))?;
                a.call(l2_entry)?;
            }
            a.inc(r13)?;
            a.cmp(r13, pairs as i32)?;
            a.jb(lp)?;
        }
        a.set_label(&mut gate)?;
        a.lea(rdi, ptr(rbx + gates_off))?;
        a.mov(rsi, rbp)?;
        a.call(gate_entry)?;
        a.add(rsp, 8)?;
        for r in [r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut l2_entry).map_err(err)?;
    a.db(&inner.bytes).map_err(err)?;
    a.set_label(&mut gate_entry).map_err(err)?;
    a.db(&gate.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(context, V-head)` traverses all positions causally and invokes the exact
/// mounted recurrence kernel. Separate heads own disjoint state/output slices.
pub(crate) fn recurrence(
    shape: Shape,
    kernel: Option<&crate::jit::recurrence::CompiledRecurrence>,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let bytes = match kernel.filter(|k| k.hvd() == shape.hvd) {
        Some(k) => k.lowering_bytes(),
        None => super::delta_scalar::recurrence(shape)?.bytes,
    };
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut again = a.create_label();
        let mut exit = a.create_label();
        for r in [rbx, rbp, r12, r13, r14, r15] {
            a.push(r)?;
        }
        a.sub(rsp, 8)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(rax, rsi)?;
        a.xor(edx, edx)?;
        a.mov(rcx, (shape.nv / shape.nk) as u64)?;
        a.div(rcx)?;
        a.imul_3(r12, rax, (shape.hkd * 4) as i32)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, state) as i32),
        )?;
        a.imul_3(r13, rbp, (shape.hkd * shape.hvd * 4) as i32)?;
        a.add(r13, rax)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, rows) as i32),
        )?;
        a.mov(r15, qword_ptr(rax))?;
        a.xor(r14d, r14d)?;
        a.test(r15, r15)?;
        a.je(exit)?;
        a.set_label(&mut again)?;
        a.mov(rdi, r13)?;
        a.imul_3(rax, r14, (shape.value() * 4) as i32)?;
        a.imul_3(rcx, rbp, (shape.hvd * 4) as i32)?;
        a.add(rax, rcx)?;
        a.mov(
            rsi,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, output) as i32),
        )?;
        a.add(rsi, rax)?;
        a.imul_3(rax, r14, (shape.conv() * 4) as i32)?;
        a.add(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, qkv) as i32),
        )?;
        a.lea(rdx, ptr(rax + r12))?;
        a.lea(rcx, ptr(rdx + (shape.key() * 4) as i32))?;
        a.imul_3(r8, rbp, (shape.hvd * 4) as i32)?;
        a.lea(r8, ptr(rax + r8 + (shape.key() * 8) as i32))?;
        a.mov(r9, shape.hkd as u64)?;
        a.imul_3(rax, r14, shape.nv as i32)?;
        a.add(rax, rbp)?;
        a.mov(
            r10,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, beta) as i32),
        )?;
        a.movd(xmm0, dword_ptr(r10 + rax * 4))?;
        a.mov(
            r10,
            qword_ptr(rbx + std::mem::offset_of!(RecurrenceCall, decay) as i32),
        )?;
        a.movd(xmm1, dword_ptr(r10 + rax * 4))?;
        a.mov(eax, (1.0 / (shape.hkd as f32).sqrt()).to_bits())?;
        a.movd(xmm2, eax)?;
        a.call(entry)?;
        a.inc(r14)?;
        a.cmp(r14, r15)?;
        a.jb(again)?;
        a.set_label(&mut exit)?;
        a.add(rsp, 8)?;
        for r in [r15, r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut entry).map_err(err)?;
    a.db(&bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(context, channel-block)` traverses positions in order, 16 channels per
/// independent task. The vector kernel handles each full block.
pub(crate) fn convolution(
    shape: Shape,
    kernel: Option<&crate::jit::conv::CompiledConv>,
    fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    shape.check()?;
    let kernel = kernel.filter(|k| k.k() == shape.ck);
    let scalar = super::delta_scalar::convolution(shape, fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut entry = a.create_label();
    let mut scalar_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut again = a.create_label();
        let mut exit = a.create_label();
        for r in [rbx, rbp, r12, r13, r14] {
            a.push(r)?;
        }
        a.sub(rsp, 16)?;
        a.mov(rbx, rdi)?;
        a.mov(rbp, rsi)?;
        a.mov(qword_ptr(rsp), rsi)?;
        a.mov(
            rax,
            qword_ptr(rbx + std::mem::offset_of!(ConvCall, rows) as i32),
        )?;
        a.mov(r14, qword_ptr(rax))?;
        a.imul_3(r12, rbp, 64)?;
        a.add(
            r12,
            qword_ptr(rbx + std::mem::offset_of!(ConvCall, data) as i32),
        )?;
        a.imul_3(r13, rbp, (64 * shape.ck) as i32)?;
        a.add(
            r13,
            qword_ptr(rbx + std::mem::offset_of!(ConvCall, state) as i32),
        )?;
        a.imul_3(rbp, rbp, (32 * shape.ck) as i32)?;
        a.add(
            rbp,
            qword_ptr(rbx + std::mem::offset_of!(ConvCall, weights) as i32),
        )?;
        a.test(r14, r14)?;
        a.je(exit)?;
        a.set_label(&mut again)?;
        a.mov(rdi, r12)?;
        a.mov(rsi, rbp)?;
        a.mov(rdx, r13)?;
        a.mov(
            r8,
            qword_ptr(rbx + std::mem::offset_of!(ConvCall, tables) as i32),
        )?;
        let mut done = a.create_label();
        let mut tail = a.create_label();
        if kernel.is_some() {
            a.cmp(qword_ptr(rsp), (shape.conv() / 16) as i32)?;
            a.jae(tail)?;
            a.mov(ecx, 1u32)?;
            a.call(entry)?;
            a.jmp(done)?;
        }
        a.set_label(&mut tail)?;
        a.mov(ecx, 16u32)?;
        if !shape.conv().is_multiple_of(16) {
            a.cmp(qword_ptr(rsp), (shape.conv() / 16) as i32)?;
            a.mov(eax, (shape.conv() % 16) as u32)?;
            a.cmove(ecx, eax)?;
        }
        a.call(scalar_entry)?;
        a.set_label(&mut done)?;
        a.add(r12, (shape.conv() * 4) as i32)?;
        a.dec(r14)?;
        a.jnz(again)?;
        a.set_label(&mut exit)?;
        a.add(rsp, 16)?;
        for r in [r14, r13, r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut entry).map_err(err)?;
    if let Some(kernel) = kernel {
        a.db(&kernel.lowering_bytes()).map_err(err)?;
    } else {
        // Keep the two local entry labels on distinct instructions when the
        // vector kernel is absent; iced-x86 rejects stacked labels.
        a.nop().map_err(err)?;
    }
    a.set_label(&mut scalar_entry).map_err(err)?;
    a.db(&scalar.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// `(context, row)` applies the mounted vector norm where available, otherwise
/// the generated exact scalar fallback, and writes the bf16 projection input.
pub(crate) fn normed_gate(
    dim: usize,
    kernel: Option<&crate::jit::gated_norm::CompiledGatedNorm>,
    fma: bool,
) -> Result<GeneratedKernel, ExecutionError> {
    if dim == 0 || dim > i32::MAX as usize / 4 {
        return Err(ExecutionError("invalid DeltaNet gated norm width".into()));
    }
    let kernel = kernel.filter(|k| k.dim() == dim);
    let scalar = super::delta_scalar::gated_norm(dim, fma)?;
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let mut vector_entry = a.create_label();
    let mut scalar_entry = a.create_label();
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        for r in [rbx, rbp, r12] {
            a.push(r)?;
        }
        a.mov(rbx, rdi)?;
        a.imul_3(rbp, rsi, (dim * 4) as i32)?;
        a.imul_3(r12, rsi, (dim * 2) as i32)?;
        a.mov(
            rdi,
            qword_ptr(rbx + std::mem::offset_of!(NormCall, output) as i32),
        )?;
        a.add(rdi, rbp)?;
        a.mov(
            rsi,
            qword_ptr(rbx + std::mem::offset_of!(NormCall, gate) as i32),
        )?;
        a.add(rsi, rbp)?;
        a.mov(
            rdx,
            qword_ptr(rbx + std::mem::offset_of!(NormCall, weights) as i32),
        )?;
        a.mov(
            rcx,
            qword_ptr(rbx + std::mem::offset_of!(NormCall, output_bf16) as i32),
        )?;
        a.add(rcx, r12)?;
        a.mov(r8, 1u64)?;
        a.mov(
            r9,
            qword_ptr(rbx + std::mem::offset_of!(NormCall, tables) as i32),
        )?;
        a.movd(
            xmm0,
            dword_ptr(rbx + std::mem::offset_of!(NormCall, eps) as i32),
        )?;
        if kernel.is_some() {
            a.call(vector_entry)?;
        } else {
            a.call(scalar_entry)?;
        }
        for r in [r12, rbp, rbx] {
            a.pop(r)?;
        }
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    a.set_label(&mut vector_entry).map_err(err)?;
    if let Some(kernel) = kernel {
        a.db(&kernel.lowering_bytes()).map_err(err)?;
    } else {
        a.nop().map_err(err)?;
    }
    a.set_label(&mut scalar_entry).map_err(err)?;
    a.db(&scalar.bytes).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
