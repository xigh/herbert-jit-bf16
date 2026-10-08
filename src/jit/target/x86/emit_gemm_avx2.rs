//! The bf16 GEMM micro-kernel, AVX2 arm: the same contract as `emit.rs`
//! (`C[mr, nr] += A[mr, K] · Bᵀ`, `B` packed R16, f32 accumulation, f32 or
//! bf16 output) on a machine without AVX-512.
//!
//! # 1. `vdpbf16ps` does not exist, and bit-for-bit equality is not promised
//!
//! `VDPBF16PS` computes, per 32-bit lane, `acc += f32(a_lo)·f32(b_lo) +
//! f32(a_hi)·f32(b_hi)`. Both products are exact in f32 (a bf16 mantissa is
//! 8 bits, the product 16, f32 carries 24); what the manual does not fix is
//! the order and number of roundings in the additions. The replacement
//! performs two successive `vfmadd231ps`, two roundings in a chosen order:
//!
//! ```text
//! vpslld      ta, a, 16          ; the LOW half of each pair, as f32
//! vpslld      tb, b, 16
//! vfmadd231ps acc, ta, tb
//! vpand       ta, a, 0xFFFF0000  ; the HIGH half, already in place
//! vpand       tb, b, 0xFFFF0000
//! vfmadd231ps acc, ta, tb
//! ```
//!
//! Six instructions instead of one and two scratch registers; the mask is a
//! memory operand. This kernel is held to a tolerance, not to a bit
//! equality, because the ISA does not promise one.
//!
//! # 2. The R16 layout does not move
//!
//! A `B` block is `[k_pair, 16 rows, 2 values]`, 64 bytes per pair of `K`. A
//! `ymm` holds half of it, rows 0 to 7; rows 8 to 15 are the next 32 bytes.
//! `nr` columns need `nr/8` registers of `B`, and block `jj` sits at
//! `(jj/2)·bstride + p·64 + (jj%2)·32`.
//!
//! # 3. Sixteen registers
//!
//! `mr·nb + nb + 1 + 2 ≤ 16` with `nb = nr/8`: the accumulators, the `B`
//! registers, the broadcast of `A` and the two emulation temporaries. At
//! `nr = 16` this gives `mr ≤ 5`; at `nr = 32`, `mr ≤ 2`. [`check_avx2`]
//! refuses anything wider, never a silent overflow. `vpbroadcastd ymm, r32`
//! is AVX-512: the f32 variants of `A` go through a stack slot instead.

use crate::jit::emit::MicroKernelSpec;
use crate::jit::ir::{Decline, ROWS_INTERLEAVED};
use crate::core::error::{HerbertError, Result};

/// YMM registers consumed: `mr·num_blocks` accumulators, `num_blocks` blocks of `B`, the
/// broadcast of `A`, and the two scratch registers for the `vdpbf16ps` emulation.
pub const fn ymm_used(mr: usize, nr: usize) -> usize {
    let num_blocks = nr / 8;
    mr * num_blocks + num_blocks + 1 + 2
}

/// Does the shape fit in the AVX2 bench?
///
/// Reuses the checks from `MicroKernelSpec::check` **plus** the budget of
/// sixteen registers, which is the only constraint specific to this kernel.
pub fn check_avx2(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if spec.mr == 0 || spec.nr == 0 || spec.k_pairs == 0 {
        return Err(Decline::Empty);
    }
    // A `ymm` carries 8 columns: `nr` must divide into it without remainder. The
    // shapes served are multiples of 16, and thus of 8.
    if !spec.nr.is_multiple_of(8) {
        return Err(Decline::PartialRowBlock);
    }
    // The two edges the AVX-512 emitter handles remain out of scope here:
    // the masked store requires a mask
    // register, which AVX2 does not have, and the final half-pair is not
    // supported. Refusing them **by name** is better than emitting a kernel that
    // would write beyond the row of `C`.
    if spec.n_tail != ROWS_INTERLEAVED {
        return Err(Decline::PartialRowBlock);
    }
    if spec.k_odd {
        return Err(Decline::OddK);
    }
    if ymm_used(spec.mr, spec.nr) > 16 {
        return Err(Decline::TileTooWide);
    }
    Ok(())
}

pub use x86::emit_gemm_avx2;

mod x86 {
    use super::*;
    use crate::jit::emit::{b_block_bytes, AFormat};
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    /// **Sixteen entries.** `ymm16..ymm31` exist only under AVX-512; listing
    /// them here would make them reachable, and `iced` would encode them
    /// as EVEX — AVX-512 in a kernel that does not want it.
    const YMM: [AsmRegisterYmm; 16] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15,
    ];
    const XMM: [AsmRegisterXmm; 16] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15,
    ];

    /// Stack table: the high-halves mask, then a scratch slot
    /// for spills that go through a GPR.
    const OFF_MASK: i32 = 0;
    const OFF_SCRATCH: i32 = 32;
    const TABLE: i32 = 64;

    fn masque() -> AsmMemoryOperand {
        ymmword_ptr(rsp + OFF_MASK)
    }

    /// Generates the micro-kernel described by `spec`, **in AVX2**. Position-independent
    /// code, no relocations: the mask and the draft
    /// live in a stack table reserved at entry.
    pub fn emit_gemm_avx2(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
        check_avx2(spec).map_err(|d| HerbertError::Backend(format!("backend_jit gemm avx2: {d}")))?;
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: avx2 gemm emission {}x{} (K={}) : {e}",
                spec.mr,
                spec.nr,
                spec.k_pairs * 2
            ))
        })
    }

    fn assemble(spec: &MicroKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
        let mr = spec.mr;
        let num_blocks = spec.nr / 8;
        let lda = spec.lda_bytes;
        let ldc = spec.ldc_bytes;
        let bstride = b_block_bytes(spec.k_pairs);
        // Bytes of a pair of `K` in `A`: 4 in bf16 (2 values), 8 in f32.
        let a_pair = match spec.a_fmt {
            AFormat::Bf16 => 4usize,
            AFormat::F32Trunc | AFormat::F32Rne => 8usize,
        };

        // Allocation computed, like the AVX-512 version: accumulators, then
        // `B`, then the broadcast of `A`, then the two scratch registers.
        let acc = |i: usize, j: usize| YMM[i * num_blocks + j];
        let breg = |j: usize| YMM[mr * num_blocks + j];
        let abcast_i = mr * num_blocks + num_blocks;
        let abcast = YMM[abcast_i];
        let ta = YMM[abcast_i + 1];
        let tb = YMM[abcast_i + 2];

        let mut asm = CodeAssembler::new(64)?;
        // Without this, `iced` may choose an EVEX encoding: AVX-512 in a
        // kernel that must contain none.
        asm.set_prefer_vex(true);
        asm.sub(rsp, TABLE)?;

        // The upper-halves mask, set once. `ta` serves as a scratch
        // before the loop begins.
        asm.mov(eax, 0xFFFF_0000u32 as i32)?;
        asm.mov(dword_ptr(rsp + OFF_SCRATCH), eax)?;
        asm.vpbroadcastd(ta, dword_ptr(rsp + OFF_SCRATCH))?;
        asm.vmovups(masque(), ta)?;

        for i in 0..mr {
            for j in 0..num_blocks {
                asm.vxorps(acc(i, j), acc(i, j), acc(i, j))?;
            }
        }

        // `vdpbf16ps acc, a, b` emulated — see the header for the caveat on
        // rounding order.
        let dp = |asm: &mut CodeAssembler,
                  d: AsmRegisterYmm,
                  a: AsmRegisterYmm,
                  b: AsmRegisterYmm|
         -> std::result::Result<(), IcedError> {
            asm.vpslld(ta, a, 16u32)?;
            asm.vpslld(tb, b, 16u32)?;
            asm.vfmadd231ps(d, ta, tb)?;
            asm.vpand(ta, a, masque())?;
            asm.vpand(tb, b, masque())?;
            asm.vfmadd231ps(d, ta, tb)?;
            Ok(())
        };

        let body = |asm: &mut CodeAssembler,
                    base_pair: usize,
                    count: usize|
         -> std::result::Result<(), IcedError> {
            for u in 0..count {
                let p = base_pair + u;
                // A `ymm` takes half of an R16 block: the 16-line block
                // `jj/2`, half `jj%2`. **No byte changes place.**
                for j in 0..num_blocks {
                    let off = (j / 2) * bstride + p * 64 + (j % 2) * 32;
                    asm.vmovdqu(breg(j), ymmword_ptr(rsi + off as i32))?;
                }
                for i in 0..mr {
                    let a_off = (i * lda + p * a_pair) as i32;
                    match spec.a_fmt {
                        AFormat::Bf16 => {
                            // `vpbroadcastd ymm, m32` is AVX2 — the shape
                            // from a GPR is not.
                            asm.vpbroadcastd(abcast, dword_ptr(rdi + a_off))?;
                        }
                        AFormat::F32Trunc => {
                            // Two f32 → one truncated bf16 pair, in GPR:
                            // the AVX-512 sequence, with this difference
                            // that the broadcast goes through the stack.
                            asm.mov(r9d, dword_ptr(rdi + a_off))?;
                            asm.shr(r9d, 16u32)?;
                            asm.mov(eax, dword_ptr(rdi + (a_off + 4)))?;
                            asm.and(eax, 0xFFFF_0000u32 as i32)?;
                            asm.or(eax, r9d)?;
                            asm.mov(dword_ptr(rsp + OFF_SCRATCH), eax)?;
                            asm.vpbroadcastd(abcast, dword_ptr(rsp + OFF_SCRATCH))?;
                        }
                        AFormat::F32Rne => {
                            // Same, **round to nearest even**, done in
                            // GPRs for lack of `vcvtneps2bf16`. Same arithmetic
                            // as `emit_elem_avx2::cvt_bf16`, on a scalar:
                            // `t = bits + 0x7FFF + ((bits >> 16) & 1)`.
                            //
                            // Low value → 16 bits of low-order weights.
                            asm.mov(eax, dword_ptr(rdi + a_off))?;
                            asm.mov(r9d, eax)?;
                            asm.shr(r9d, 16u32)?;
                            asm.and(r9d, 1)?;
                            asm.add(eax, 0x7FFFu32 as i32)?;
                            asm.add(eax, r9d)?;
                            asm.shr(eax, 16u32)?;
                            // High value → 16 bits of strong weights.
                            asm.mov(r11d, dword_ptr(rdi + (a_off + 4)))?;
                            asm.mov(r9d, r11d)?;
                            asm.shr(r9d, 16u32)?;
                            asm.and(r9d, 1)?;
                            asm.add(r11d, 0x7FFFu32 as i32)?;
                            asm.add(r11d, r9d)?;
                            asm.and(r11d, 0xFFFF_0000u32 as i32)?;
                            asm.or(eax, r11d)?;
                            asm.mov(dword_ptr(rsp + OFF_SCRATCH), eax)?;
                            asm.vpbroadcastd(abcast, dword_ptr(rsp + OFF_SCRATCH))?;
                        }
                    }
                    for j in 0..num_blocks {
                        dp(asm, acc(i, j), abcast, breg(j))?;
                    }
                }
            }
            Ok(())
        };

        let unroll = spec.opts.unroll.max(1);
        let trip = spec.k_pairs / unroll;
        let rem = spec.k_pairs % unroll;
        let mut loop_label = asm.create_label();

        if trip > 0 {
            asm.mov(rcx, trip as u64)?;
            asm.set_label(&mut loop_label)?;
            body(&mut asm, 0, unroll)?;
            asm.add(rdi, (unroll * a_pair) as i32)?;
            asm.add(rsi, (unroll * 64) as i32)?;
            asm.dec(rcx)?;
            asm.jne(loop_label)?;
        }
        if rem > 0 {
            body(&mut asm, 0, rem)?;
        }

        // Epilogue. Accumulation stays in f32 in the ymm; what goes to
        // memory respects the invariant "never f32 in memory" when
        // `out_bf16` is armed.
        if spec.out_bf16 {
            // Software bf16 rounding, same sequence as `emit_elem_avx2`:
            // 8 f32 → 8 bf16 (16 bytes). `ta`/`tb` are dead here.
            asm.mov(eax, 1u32)?;
            asm.mov(dword_ptr(rsp + OFF_SCRATCH), eax)?;
            asm.vpbroadcastd(tb, dword_ptr(rsp + OFF_SCRATCH))?;
            for i in 0..mr {
                for j in 0..num_blocks {
                    let a = acc(i, j);
                    asm.vpsrld(ta, a, 16u32)?;
                    asm.vpand(ta, ta, tb)?;
                    asm.vpaddd(ta, ta, a)?;
                    // + 0x7FFF, as a broadcast immediate: we reuse `abcast`,
                    // dead since the end of the loop.
                    asm.mov(eax, 0x7FFFu32 as i32)?;
                    asm.mov(dword_ptr(rsp + OFF_SCRATCH), eax)?;
                    asm.vpbroadcastd(abcast, dword_ptr(rsp + OFF_SCRATCH))?;
                    asm.vpaddd(ta, ta, abcast)?;
                    asm.vpsrld(ta, ta, 16u32)?;
                    asm.vextracti128(XMM[abcast_i], ta, 1)?;
                    asm.vpackusdw(XMM[abcast_i], XMM[abcast_i + 1], XMM[abcast_i])?;
                    asm.vmovdqu(
                        xmmword_ptr(rdx + (i * ldc + j * 16) as i32),
                        XMM[abcast_i],
                    )?;
                }
            }
        } else {
            for i in 0..mr {
                for j in 0..num_blocks {
                    asm.vmovups(ymmword_ptr(rdx + (i * ldc + j * 32) as i32), acc(i, j))?;
                }
            }
        }

        asm.add(rsp, TABLE)?;
        asm.vzeroupper()?;
        asm.ret()?;
        asm.assemble(0)
    }
}
