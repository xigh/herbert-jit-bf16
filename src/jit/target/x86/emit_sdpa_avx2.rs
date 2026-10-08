//! The decode attention (SDPA), AVX2 arm: one head, the three passes QK,
//! softmax and SV chained in a single blob, with the ABI and semantics of
//! `emit_sdpa` in the chained form. What changes is one fact: sixteen
//! registers instead of thirty-two, eight lanes instead of sixteen.
//!
//! What the port simplifies:
//!
//! 1. **No QK by groups**, so no `vcompressps` and no opmask: the QK is done
//!    position by position — a dot product, a horizontal reduction, a
//!    `vmovss`. Slower, and the only instructions of the AVX-512 kernel
//!    without an AVX2 equivalent are exactly the ones this avoids.
//! 2. **The SV accumulator lives on the stack.** At `head_dim = 256` the
//!    output is 32 vectors of 8 f32, impossible in registers; everything
//!    spills, uniformly, and the round trip stays in L1.
//! 3. **One form, one exp.** The fused form already saturates thirty-two
//!    registers and the split form is not used by the decode driver: both
//!    decline by name here.
//!
//! What the port does not change: the summation order of the dot product
//! (four chains), the exact sequence of the softmax (maximum, `exp`, sum,
//! normalisation — a division without sink, a multiplication by the inverse
//! with one), the `-inf` padding that makes the sweeps mask-free, and the
//! coefficients of the exp polynomial (`exp_constants.rs`). Bit-for-bit
//! equality with the AVX-512 kernel is not reachable: the horizontal
//! reductions run over eight lanes instead of sixteen, so the additions
//! happen in another order; the two agree within a tolerance.

use crate::jit::emit_sdpa::SdpaKernelSpec;
use crate::jit::ir::{Decline, ExpStrategy, SdpaForm};
use crate::core::error::{HerbertError, Result};

/// Can SDPA AVX2 run here? `avx2` + `fma`, the ISA floor.
pub fn is_supported() -> bool {
    crate::jit::engine_isa() != crate::jit::IsaEngine::BelowFloor
}

/// Is this shape served by the AVX2 kernel?
///
/// Refusals are **named**: an out-of-scope form must be seen, not
/// guessed from a false output.
pub fn check_avx2(spec: &SdpaKernelSpec) -> std::result::Result<(), Decline> {
    if spec.form != SdpaForm::Chained {
        return Err(Decline::OutOfScopeAvx2);
    }
    if spec.exp != ExpStrategy::Inline {
        return Err(Decline::OutOfScopeAvx2);
    }
    // Eight f32 per register: `head_dim` must divide evenly. All
    // served heads (64, 128, 256, 512) do.
    if spec.shape.head_dim == 0 || !spec.shape.head_dim.is_multiple_of(16) {
        return Err(Decline::PartialRowBlock);
    }
    Ok(())
}

pub use x86::emit_sdpa_avx2;

mod x86 {
    use super::*;
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    /// **Sixteen entries.** `ymm16..ymm31` exist only under AVX-512.
    const YMM: [AsmRegisterYmm; 16] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15,
    ];
    const XMM: [AsmRegisterXmm; 16] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15,
    ];

    use crate::jit::exp_constants::{
        EXP_C0, EXP_C1, EXP_C2, EXP_C3, EXP_C4, EXP_LOG2E, EXP_NEG87, EXP_ONE_BITS,
    };

    /// Constant table slots, in 32-byte units.
    mod constant {
        pub const NEG87: i32 = 0;
        pub const LOG2E: i32 = 1;
        pub const C4: i32 = 2;
        pub const C3: i32 = 3;
        pub const C2: i32 = 4;
        pub const C1: i32 = 5;
        pub const C0: i32 = 6;
        pub const ONE_BITS: i32 = 7;
        /// `0xFFFF0000` — the upper half of each bf16 pair.
        pub const HI: i32 = 8;
        /// The integer `1`, for the least significant bit of the bf16 rounding.
        pub const UN: i32 = 9;
        /// `0x7FFF`, the half-ULP of bf16 rounding.
        pub const BIAS: i32 = 10;
        pub const N: i32 = 11;
    }

    /// Scaling shift in the frame.
    const OFF_SCALE: i32 = 0;
    /// Four pointers resident: `v`, `out`, `scores`, `n`.
    const OFF_HOME: i32 = 64;
    /// Base of the constant table.
    const OFF_CST: i32 = 128;
    /// Base of the accumulator `SV`.
    const OFF_SV: i32 = OFF_CST + constant::N * 32;

    fn c(slot: i32) -> AsmMemoryOperand {
        ymmword_ptr(rsp + OFF_CST + slot * 32)
    }

    /// Frame size: the table, the domiciliations, and `nv` vectors of
    /// 8 f32 for the accumulator `SV`. Rounded to 64.
    fn frame_bytes(nv: usize) -> i32 {
        let raw = OFF_SV + (nv as i32) * 32;
        (raw + 63) & !63
    }

    fn pose(asm: &mut CodeAssembler, slot: i32, bits: u32, t: usize) -> Result2 {
        asm.mov(eax, bits)?;
        asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
        asm.vpbroadcastd(YMM[t], dword_ptr(rsp + OFF_SCALE + 32))?;
        asm.vmovups(c(slot), YMM[t])?;
        Ok(())
    }

    type Result2 = std::result::Result<(), IcedError>;

    fn build_table(asm: &mut CodeAssembler, t: usize) -> Result2 {
        pose(asm, constant::NEG87, EXP_NEG87.to_bits(), t)?;
        pose(asm, constant::LOG2E, EXP_LOG2E.to_bits(), t)?;
        pose(asm, constant::C4, EXP_C4.to_bits(), t)?;
        pose(asm, constant::C3, EXP_C3.to_bits(), t)?;
        pose(asm, constant::C2, EXP_C2.to_bits(), t)?;
        pose(asm, constant::C1, EXP_C1.to_bits(), t)?;
        pose(asm, constant::C0, EXP_C0.to_bits(), t)?;
        pose(asm, constant::ONE_BITS, EXP_ONE_BITS, t)?;
        pose(asm, constant::HI, 0xFFFF_0000u32, t)?;
        pose(asm, constant::UN, 1u32, t)?;
        pose(asm, constant::BIAS, 0x7FFFu32, t)?;
        Ok(())
    }

    /// `io = exp(io)` — inline transcription of `emit_sdpa::emit_exp`,
    /// constants in memory.
    ///
    /// Domain `io ≤ 88`. The softmax is only called with arguments `≤ 0`
    /// (the maximum has been subtracted), so the upper bound is not set here —
    /// exactly as in the original.
    fn exp_inline(asm: &mut CodeAssembler, io: usize, t1: usize, t2: usize, t3: usize) -> Result2 {
        asm.vmaxps(YMM[io], YMM[io], c(constant::NEG87))?;
        asm.vmulps(YMM[t1], YMM[io], c(constant::LOG2E))?;
        asm.vroundps(YMM[t2], YMM[t1], 0u32)?;
        asm.vsubps(YMM[t1], YMM[t1], YMM[t2])?;
        asm.vmovups(YMM[t3], c(constant::C4))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C3))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C2))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C1))?;
        asm.vfmadd213ps(YMM[t3], YMM[t1], c(constant::C0))?;
        asm.vcvtps2dq(YMM[t2], YMM[t2])?;
        asm.vpslld(YMM[t2], YMM[t2], 23u32)?;
        asm.vpaddd(YMM[t2], YMM[t2], c(constant::ONE_BITS))?;
        asm.vmulps(YMM[io], YMM[t3], YMM[t2])?;
        Ok(())
    }

    /// Horizontal sum of a ymm → scalar in `xmm[dst]`.
    fn hsum(asm: &mut CodeAssembler, dst: usize, t: usize) -> Result2 {
        asm.vextractf128(XMM[t], YMM[dst], 1)?;
        asm.vaddps(XMM[dst], XMM[dst], XMM[t])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        asm.vhaddps(XMM[dst], XMM[dst], XMM[dst])?;
        Ok(())
    }

    /// Maximum horizontal of a ymm → scalar in `xmm[dst]`.
    fn hmax(asm: &mut CodeAssembler, dst: usize, t: usize) -> Result2 {
        asm.vextractf128(XMM[t], YMM[dst], 1)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        // Two permutations suffice for four lanes: (1,0,3,2) then
        // (2,3,0,1). `vhaddps` is not suitable — it adds.
        asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b10_11_00_01u32)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        asm.vshufps(XMM[t], XMM[dst], XMM[dst], 0b01_00_11_10u32)?;
        asm.vmaxps(XMM[dst], XMM[dst], XMM[t])?;
        Ok(())
    }

    /// Widens 8 bf16 (16 bytes) to 8 f32.
    fn widen(asm: &mut CodeAssembler, dst: usize, at: AsmMemoryOperand) -> Result2 {
        asm.vpmovzxwd(YMM[dst], at)?;
        asm.vpslld(YMM[dst], YMM[dst], 16u32)?;
        Ok(())
    }

    /// `acc += a·b` on bf16 pairs — emulating `vdpbf16ps`, the same
    /// as in [`crate::jit::emit_gemm_avx2`].
    fn dp_bf16(asm: &mut CodeAssembler, acc: usize, a: usize, b: AsmMemoryOperand, ta: usize, tb: usize) -> Result2 {
        asm.vpslld(YMM[ta], YMM[a], 16u32)?;
        asm.vmovups(YMM[tb], b)?;
        asm.vpslld(YMM[tb], YMM[tb], 16u32)?;
        asm.vfmadd231ps(YMM[acc], YMM[ta], YMM[tb])?;
        asm.vpand(YMM[ta], YMM[a], c(constant::HI))?;
        asm.vmovups(YMM[tb], b)?;
        asm.vpand(YMM[tb], YMM[tb], c(constant::HI))?;
        asm.vfmadd231ps(YMM[acc], YMM[ta], YMM[tb])?;
        Ok(())
    }

    /// 8 f32 → 8 bf16 (16 bytes) in `xmm[dst]`, rounded to even.
    fn cvt_bf16(asm: &mut CodeAssembler, dst: usize, src: usize, t: usize) -> Result2 {
        asm.vpsrld(YMM[t], YMM[src], 16u32)?;
        asm.vpand(YMM[t], YMM[t], c(constant::UN))?;
        asm.vpaddd(YMM[t], YMM[t], c(constant::BIAS))?;
        asm.vpaddd(YMM[t], YMM[t], YMM[src])?;
        asm.vpsrld(YMM[t], YMM[t], 16u32)?;
        asm.vextracti128(XMM[dst], YMM[t], 1)?;
        asm.vpackusdw(XMM[dst], XMM[t], XMM[dst])?;
        Ok(())
    }

    /// Generates the SDPA decode **kernel** described by `spec`, **in AVX2**.
    pub fn emit_sdpa_avx2(spec: &SdpaKernelSpec) -> Result<Vec<u8>> {
        check_avx2(spec)
            .map_err(|d| HerbertError::Backend(format!("backend_jit sdpa avx2: {d}")))?;
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: sdpa avx2 emission head_dim={} : {e}",
                spec.shape.head_dim
            ))
        })
    }

    fn assemble(spec: &SdpaKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
        // Vectors of **8** f32, and blocks of **16** bf16 (32 bytes).
        let nv = spec.shape.head_dim / 8;
        let nb16 = spec.shape.head_dim / 16;
        let row = spec.shape.kv_row_bytes() as i32;
        let sink_on = spec.shape.sink;

        let mut asm = CodeAssembler::new(64)?;
        // Without it, `iced` might choose an EVEX encoding.
        asm.set_prefer_vex(true);

        // ── prologue, ABI Chained ─────────────────────────────────────────
        // rdi=q, rsi=k, rdx=v, rcx=scores, r8=out, r9=n, xmm0=scale, xmm1=sink
        asm.push(rbp)?;
        asm.mov(rbp, rsp)?;
        asm.and(rsp, -64i32)?;
        asm.sub(rsp, frame_bytes(nv))?;

        // The scale and the sink first: building the table clobbers
        // xmm0. The sink goes on the stack too — one fewer register to pin
        // across the three passes.
        asm.vmovss(dword_ptr(rsp + OFF_SCALE), XMM[0])?;
        if sink_on {
            asm.vmovss(dword_ptr(rsp + OFF_SCALE + 4), XMM[1])?;
        }
        asm.mov(qword_ptr(rsp + OFF_HOME), rdx)?; // v
        asm.mov(qword_ptr(rsp + OFF_HOME + 8), r8)?; // out
        asm.mov(qword_ptr(rsp + OFF_HOME + 16), rcx)?; // scores
        asm.mov(qword_ptr(rsp + OFF_HOME + 24), r9)?; // n
        build_table(&mut asm, 15)?;

        // ── pass 1: QK, position by position ───────────────────────────
        //
        // rdi=q (preserved), rsi=k (advanced), rdx=scores (advanced), rcx=n.
        // Four accumulation chains, like the AVX-512 kernel: it is
        // the summation order, it does not change lightly.
        {
            const A: [usize; 4] = [0, 1, 2, 3];
            const KV: usize = 4;
            const TA: usize = 5;
            const TB: usize = 6;
            let n_acc = 4.min(nb16.max(1));
            asm.mov(rdx, rcx)?; // scores
            asm.mov(rcx, r9)?; // n
            let mut top = asm.create_label();
            let mut after = asm.create_label();
            let mut pad_top = asm.create_label();
            let mut no_pad = asm.create_label();
            asm.test(rcx, rcx)?;
            asm.jz(after)?;
            asm.set_label(&mut top)?;
            for &a in &A[..n_acc] {
                asm.vxorps(YMM[a], YMM[a], YMM[a])?;
            }
            for b in 0..nb16 {
                asm.vmovdqu(YMM[KV], ymmword_ptr(rsi + (b * 32) as i32))?;
                dp_bf16(
                    &mut asm,
                    A[b % n_acc],
                    KV,
                    ymmword_ptr(rdi + (b * 32) as i32),
                    TA,
                    TB,
                )?;
            }
            // Tree: (0+1) + (2+3) when all four chains exist — this is
            // the order of the AVX-512 kernel.
            if n_acc == 4 {
                asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[A[1]])?;
                asm.vaddps(YMM[A[2]], YMM[A[2]], YMM[A[3]])?;
                asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[A[2]])?;
            } else {
                for &a in &A[1..n_acc] {
                    asm.vaddps(YMM[A[0]], YMM[A[0]], YMM[a])?;
                }
            }
            hsum(&mut asm, A[0], TA)?;
            asm.vmovss(XMM[TA], dword_ptr(rsp + OFF_SCALE))?;
            asm.vmulss(XMM[A[0]], XMM[A[0]], XMM[TA])?;
            asm.vmovss(dword_ptr(rdx), XMM[A[0]])?;
            asm.add(rsi, row)?;
            asm.add(rdx, 4)?;
            asm.dec(rcx)?;
            asm.jnz(top)?;

            // `-inf` padding up to a multiple of **8** — this is what makes
            // the three softmax passes maskless.
            asm.set_label(&mut after)?;
            asm.mov(rax, qword_ptr(rsp + OFF_HOME + 24))?;
            asm.and(rax, 7)?;
            asm.jz(no_pad)?;
            asm.mov(r11, 8u64)?;
            asm.sub(r11, rax)?;
            asm.set_label(&mut pad_top)?;
            asm.mov(dword_ptr(rdx), 0xFF80_0000u32)?;
            asm.add(rdx, 4)?;
            asm.dec(r11)?;
            asm.jnz(pad_top)?;
            asm.set_label(&mut no_pad)?;
        }

        // ── pass 2: softmax, three sweeps over ceil(n/8) blocks ────────
        {
            const MAXV: usize = 0;
            const SUMV: usize = 1;
            const WORK: usize = 2;
            const T1: usize = 3;
            const T2: usize = 4;
            const T3: usize = 5;
            const RED: usize = 6;
            const TMP: usize = 7;
            asm.mov(rdi, qword_ptr(rsp + OFF_HOME + 16))?; // scores
            asm.mov(r11, qword_ptr(rsp + OFF_HOME + 24))?; // n
            asm.add(r11, 7i32)?;
            asm.shr(r11, 3u32)?; // nb = ceil(n/8)

            // 1. maximum
            asm.mov(eax, f32::NEG_INFINITY.to_bits())?;
            asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
            asm.vpbroadcastd(YMM[MAXV], dword_ptr(rsp + OFF_SCALE + 32))?;
            asm.mov(rsi, rdi)?;
            asm.mov(rcx, r11)?;
            let mut max_top = asm.create_label();
            asm.set_label(&mut max_top)?;
            asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
            asm.vmaxps(YMM[MAXV], YMM[MAXV], YMM[WORK])?;
            asm.add(rsi, 32)?;
            asm.dec(rcx)?;
            asm.jnz(max_top)?;
            asm.vmovups(YMM[RED], YMM[MAXV])?;
            hmax(&mut asm, RED, TMP)?;
            if sink_on {
                asm.vmovss(XMM[TMP], dword_ptr(rsp + OFF_SCALE + 4))?;
                asm.vmaxss(XMM[RED], XMM[RED], XMM[TMP])?;
            }
            asm.vbroadcastss(YMM[MAXV], XMM[RED])?;

            // 2. exp and sum
            asm.vxorps(YMM[SUMV], YMM[SUMV], YMM[SUMV])?;
            asm.mov(rsi, rdi)?;
            asm.mov(rcx, r11)?;
            let mut exp_top = asm.create_label();
            asm.set_label(&mut exp_top)?;
            asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
            asm.vsubps(YMM[WORK], YMM[WORK], YMM[MAXV])?;
            exp_inline(&mut asm, WORK, T1, T2, T3)?;
            asm.vaddps(YMM[SUMV], YMM[SUMV], YMM[WORK])?;
            asm.vmovups(ymmword_ptr(rsi), YMM[WORK])?;
            asm.add(rsi, 32)?;
            asm.dec(rcx)?;
            asm.jnz(exp_top)?;
            // `RED` still holds the scalar maximum: the sum goes into `TMP`,
            // which no longer has a role here.
            asm.vmovups(YMM[TMP], YMM[SUMV])?;
            hsum(&mut asm, TMP, T1)?;
            if sink_on {
                // The sink term enters the reference sum.
                asm.vmovss(XMM[T1], dword_ptr(rsp + OFF_SCALE + 4))?;
                asm.vsubss(XMM[T1], XMM[T1], XMM[RED])?;
                asm.vbroadcastss(YMM[WORK], XMM[T1])?;
                exp_inline(&mut asm, WORK, T1, T2, T3)?;
                asm.vaddss(XMM[TMP], XMM[TMP], XMM[WORK])?;
            }

            // 3. normalization
            if sink_on {
                asm.mov(eax, 1.0f32.to_bits())?;
                asm.mov(dword_ptr(rsp + OFF_SCALE + 32), eax)?;
                asm.vmovss(XMM[T1], dword_ptr(rsp + OFF_SCALE + 32))?;
                asm.vdivss(XMM[TMP], XMM[T1], XMM[TMP])?;
            }
            asm.vbroadcastss(YMM[SUMV], XMM[TMP])?;
            asm.mov(rsi, rdi)?;
            asm.mov(rcx, r11)?;
            let mut norm_top = asm.create_label();
            asm.set_label(&mut norm_top)?;
            asm.vmovups(YMM[WORK], ymmword_ptr(rsi))?;
            if sink_on {
                asm.vmulps(YMM[WORK], YMM[WORK], YMM[SUMV])?;
            } else {
                asm.vdivps(YMM[WORK], YMM[WORK], YMM[SUMV])?;
            }
            asm.vmovups(ymmword_ptr(rsi), YMM[WORK])?;
            asm.add(rsi, 32)?;
            asm.dec(rcx)?;
            asm.jnz(norm_top)?;
        }

        // ── pass 3: SV, accumulator ENTIRELY on stack ────────────────
        {
            const SCORE: usize = 0;
            const VLIG: usize = 1;
            const ACC: usize = 2;
            const TMP: usize = 3;
            asm.mov(rdi, qword_ptr(rsp + OFF_HOME + 16))?; // scores
            asm.mov(rsi, qword_ptr(rsp + OFF_HOME))?; // v
            asm.mov(rdx, qword_ptr(rsp + OFF_HOME + 8))?; // out
            asm.mov(rcx, qword_ptr(rsp + OFF_HOME + 24))?; // n

            // Zeroing the accumulator.
            asm.vxorps(YMM[ACC], YMM[ACC], YMM[ACC])?;
            for v in 0..nv {
                asm.vmovups(ymmword_ptr(rsp + OFF_SV + (v as i32) * 32), YMM[ACC])?;
            }
            let mut top = asm.create_label();
            let mut done = asm.create_label();
            asm.test(rcx, rcx)?;
            asm.jz(done)?;
            asm.set_label(&mut top)?;
            asm.vbroadcastss(YMM[SCORE], dword_ptr(rdi))?;
            for v in 0..nv {
                // An output YMM covers 8 dimensions, i.e., **16 bytes** of
                // `V` in bf16.
                widen(&mut asm, VLIG, xmmword_ptr(rsi + (v * 16) as i32))?;
                asm.vmovups(YMM[ACC], ymmword_ptr(rsp + OFF_SV + (v as i32) * 32))?;
                asm.vfmadd231ps(YMM[ACC], YMM[VLIG], YMM[SCORE])?;
                asm.vmovups(ymmword_ptr(rsp + OFF_SV + (v as i32) * 32), YMM[ACC])?;
            }
            asm.add(rdi, 4)?;
            asm.add(rsi, row)?;
            asm.dec(rcx)?;
            asm.jnz(top)?;
            asm.set_label(&mut done)?;

            // Epilogue: the accumulation was in f32 in the draft, which
            // goes to memory is bf16 — the engine's invariant.
            for v in 0..nv {
                asm.vmovups(YMM[ACC], ymmword_ptr(rsp + OFF_SV + (v as i32) * 32))?;
                cvt_bf16(&mut asm, VLIG, ACC, TMP)?;
                asm.vmovdqu(xmmword_ptr(rdx + (v * 16) as i32), XMM[VLIG])?;
            }
        }

        // ── epilogue ──────────────────────────────────────────────────────
        asm.mov(rsp, rbp)?;
        asm.pop(rbp)?;
        asm.vzeroupper()?;
        asm.ret()?;
        asm.assemble(0)
    }
}
