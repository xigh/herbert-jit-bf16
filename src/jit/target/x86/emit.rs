//! The bf16 GEMM micro-kernel, AVX-512: `vdpbf16ps` on R16 weights.
//!
//! A kernel call computes an `mr × nr` tile of `C`:
//!
//! ```text
//! C[i, j] += Σ_k A[i, k] · W[j, k]     i < mr, j < nr
//! ```
//!
//! with `A` bf16 `[M, K]` row-major, `W` in the R16 layout
//! (`[N/16 blocks, K/2 pairs, 16 rows, 2 bf16]`, `model/weight.rs`) and `C`
//! f32 or bf16 `[M, N]` row-major. One 64-byte load of `W` brings a pair of
//! `K` values for sixteen rows; one `vpbroadcastd` brings the same pair of
//! `A`; one `vdpbf16ps` accumulates sixteen dot-product steps in f32.
//!
//! ## ABI — three pointers, no context
//!
//! ```text
//! extern "sysv64" fn(a: *const u16, b: *const u16, c: *mut f32)   // rdi, rsi, rdx
//! ```
//!
//! `K` and `N` are immediates: every address is a displacement from these
//! three bases — row `i` of `A` at `[rdi + i·2K]`, R16 block `j` of `B` at
//! `[rsi + j·64·k_pairs]`, row `i` block `j` of `C` at `[rdx + i·4N + j·64]`.
//! The prologue is empty (rdi, rsi, rdx, rcx are caller-saved).
//!
//! ## Register allocation — computed, not hardcoded
//!
//! `nb = nr/16` registers of `B`, one for the broadcast of `A`, `mr·nb`
//! accumulators: `nb·(mr+1) + 1 ≤ 32`.
//!
//! ## The loop over K — immediate, remainder peeled
//!
//! `k_pairs` is known: the loop counter is a `mov rcx, imm` and the remainder
//! `k_pairs % unroll` is emitted in a straight line, without a tail branch.
//!
//! ## Edges: `K` odd and `N % 16 ≠ 0`
//!
//! The R16 layout already pads `B` up to `k_even` and `ceil(N/16)` blocks.
//! - [`MicroKernelSpec::k_odd`]: the last pair of `K` has one valid element in
//!   `A`. It is peeled out of the loop and loaded as a **word** (`movzx`) with
//!   the high half zeroed: no read beyond the row, no dependence on the
//!   padding of `B`.
//! - [`MicroKernelSpec::n_tail`]: the last group of 16 columns may be partial.
//!   Its store, and only its store, goes under mask `k1`.
//!
//! Both are outside the hot loop; an aligned shape emits exactly the same
//! code as before.

pub(crate) use crate::jit::emit::{b_block_bytes, AFormat, MicroKernelSpec};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use crate::jit::target::x86::tile_fits;

/// The AVX-512 side of [`MicroKernelSpec::check`]: the tile fits the 32 zmm,
/// and every displacement fits a 32-bit immediate.
pub(crate) fn check_avx512(spec: &MicroKernelSpec) -> Result<(), Decline> {
    tile_fits(TileShape { mr: spec.mr, nr: spec.nr })?;
    let nb = spec.nr / ROWS_INTERLEAVED;
    let c_block = if spec.out_bf16 { 32 } else { 64 };
    let a_pair_bytes = spec.a_fmt.pair_bytes();
    let max_disp = [
        // A : last line, last peeled pair.
        (spec.mr - 1) * spec.lda_bytes + spec.k_pairs * a_pair_bytes,
        // B : last block, last pair, plus prefetch ahead.
        (nb - 1) * b_block_bytes(spec.k_pairs) + (spec.k_pairs + 2 * spec.opts.unroll) * 64,
        // C : last line, last block.
        (spec.mr - 1) * spec.ldc_bytes + (nb - 1) * c_block,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    if max_disp > i32::MAX as usize {
        return Err(Decline::DisplacementOverflow);
    }
    Ok(())
}

pub use x86::emit;

mod x86 {
    use super::{b_block_bytes, AFormat, MicroKernelSpec};
    use crate::jit::ir::ROWS_INTERLEAVED;
    use crate::core::error::{HerbertError, Result};
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    const ZMM: [AsmRegisterZmm; 32] = [
        zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13,
        zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26,
        zmm27, zmm28, zmm29, zmm30, zmm31,
    ];
    /// XMM view of the same registers — source of `vpbroadcastd zmm, xmm`.
    const XMM: [AsmRegisterXmm; 32] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15, xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23, xmm24, xmm25, xmm26,
        xmm27, xmm28, xmm29, xmm30, xmm31,
    ];
    /// View of the same registers — destination of `vcvtneps2bf16`.
    const YMM: [AsmRegisterYmm; 32] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23, ymm24, ymm25, ymm26,
        ymm27, ymm28, ymm29, ymm30, ymm31,
    ];

    /// Generates the kernel described by `spec`. The code is position-independent
    /// (no RIP-relative references, only a short jump).
    pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
        spec.check()
            .map_err(|d| HerbertError::Backend(format!("backend_jit: {d}")))?;
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!(
                "backend_jit: emission {}x{} (k_pairs={}) : {e}",
                spec.mr, spec.nr, spec.k_pairs
            ))
        })
    }

    fn assemble(spec: &MicroKernelSpec) -> std::result::Result<Vec<u8>, IcedError> {
        let mr = spec.mr;
        let nb = spec.nr / ROWS_INTERLEAVED;
        let bstride = b_block_bytes(spec.k_pairs) as i32;
        let lda = spec.lda_bytes as i32;
        let ldc = spec.ldc_bytes as i32;
        let unroll = spec.opts.unroll.max(1);
        // Bytes consumed in `A` by a pair of `K`: 4 in bf16, 8 in f32.
        let a_pair = spec.a_fmt.pair_bytes();

        // Calculated allocation: [0, mr·nb) accumulators, then nb B registers,
        // then the broadcast of A. `MicroKernelSpec::check` has guaranteed ≤ 32.
        let acc = |i: usize, j: usize| ZMM[i * nb + j];
        let breg = |j: usize| ZMM[mr * nb + j];
        let abcast_i = mr * nb + nb;
        let abcast = ZMM[abcast_i];

        let mut asm = CodeAssembler::new(64)?;
        let mut loop_label = asm.create_label();

        for i in 0..mr {
            for j in 0..nb {
                asm.vxorps(acc(i, j), acc(i, j), acc(i, j))?;
            }
        }

        // Body: `unroll` K pairs from the offset `base_pair`.
        // rdi/rsi point to pair 0 of the current group.
        let body = |asm: &mut CodeAssembler, base_pair: usize, count: usize| -> std::result::Result<(), IcedError> {
            if spec.opts.prefetch {
                for u in 0..count {
                    for j in 0..nb {
                        let off = j * bstride as usize + (base_pair + count + u) * 64;
                        asm.prefetcht0(byte_ptr(rsi + off as i32))?;
                    }
                }
            }
            for u in 0..count {
                let p = base_pair + u;
                for j in 0..nb {
                    asm.vmovdqu16(breg(j), zmmword_ptr(rsi + (j * bstride as usize + p * 64) as i32))?;
                }
                for i in 0..mr {
                    let a_off = (i * lda as usize + p * a_pair) as i32;
                    match spec.a_fmt {
                        AFormat::Bf16 => {
                            asm.vpbroadcastd(abcast, dword_ptr(rdi + a_off))?;
                        }
                        AFormat::F32Trunc => {
                            // Two f32 → a **truncated** bf16 pair, in
                            // a register. No buffer: this is the sequence of
                            // `avx512_bf16_macrokernel_nx64` byte for byte,
                            // so the same result as the replaced path.
                            // rax/r9 are caller-saved and free (empty prologue).
                            asm.mov(r9d, dword_ptr(rdi + a_off))?;
                            asm.shr(r9d, 16u32)?;
                            asm.mov(eax, dword_ptr(rdi + (a_off + 4)))?;
                            asm.and(eax, 0xFFFF_0000u32 as i32)?;
                            asm.or(eax, r9d)?;
                            asm.vpbroadcastd(abcast, eax)?;
                        }
                        AFormat::F32Rne => {
                            // Same, but **rounds to nearest even**, and
                            // without going through the GPRs. Three instructions, and
                            // `abcast` serves as a scratchpad — each reads its
                            // source entirely before writing its destination.
                            //
                            // `vpbroadcastq` instead of `vmovq`: it reads only
                            // **8 bytes** (so never beyond `A`) *and* it
                            // is encoded in EVEX, the only way to reach
                            // `zmm16+` — a tile 8×48 places `abcast` in
                            // zmm27, which `vmovq` VEX cannot encode.
                            // The pair is broadcast into the 8 qwords, so
                            // `vcvtneps2bf16` yields 8 copies of the desired dword;
                            // `vpbroadcastd` extends the first to the 16 lanes.
                            asm.vpbroadcastq(abcast, qword_ptr(rdi + a_off))?;
                            asm.vcvtneps2bf16(YMM[abcast_i], abcast)?;
                            asm.vpbroadcastd(abcast, XMM[abcast_i])?;
                        }
                    }
                    for j in 0..nb {
                        asm.vdpbf16ps(acc(i, j), abcast, breg(j))?;
                    }
                }
            }
            Ok(())
        };

        // Last pair of `K` on an odd `K`: only one element of `A` is
        // valid. We load it as a **word** — `vpbroadcastd` would read 2 bytes
        // beyond the buffer on the last line — and set the upper half
        // to zero, making the result independent of `B`'s padding.
        let half_pair = |asm: &mut CodeAssembler, p: usize| -> std::result::Result<(), IcedError> {
            for j in 0..nb {
                asm.vmovdqu16(breg(j), zmmword_ptr(rsi + (j * bstride as usize + p * 64) as i32))?;
            }
            for i in 0..mr {
                let a_off = (i * lda as usize + p * a_pair) as i32;
                match spec.a_fmt {
                    AFormat::Bf16 => {
                        asm.movzx(eax, word_ptr(rdi + a_off))?;
                        asm.vpbroadcastd(abcast, eax)?;
                    }
                    AFormat::F32Trunc => {
                        // Only one f32 read (4 bytes), truncated like the replaced
                        // path: the `shr` brings the bf16 to the bottom and leaves
                        // zero at the top, without a second `mov`.
                        asm.mov(eax, dword_ptr(rdi + a_off))?;
                        asm.shr(eax, 16u32)?;
                        asm.vpbroadcastd(abcast, eax)?;
                    }
                    AFormat::F32Rne => {
                        // `vpbroadcastd` and not `vpbroadcastq`: 4 bytes read,
                        // so never beyond `A`.
                        //
                        // `vpmovzxwd` and **not** a round-trip via GPR:
                        // `vcvtneps2bf16` leaves 16 identical words, of which
                        // broadcasting would make the pair `(a, a)`. Extending the
                        // word to dword yields `(a, 0)` — the upper half is null
                        // by construction, without reading a constant and without
                        // depending on the padding of `B`.
                        //
                        // The GPR was tried and **refused by the encoder**:
                        // `vmovd r32, xmm` is VEX, so `xmm16+` is
                        // inaccessible to it, and a 3×64 tile places `abcast`
                        // ("Register 93 is not between 77 and 92"). This is the same trap as the one
                        // already documented for the bf16 epilogue.
                        asm.vpbroadcastd(abcast, dword_ptr(rdi + a_off))?;
                        asm.vcvtneps2bf16(YMM[abcast_i], abcast)?;
                        asm.vpmovzxwd(abcast, YMM[abcast_i])?;
                    }
                }
                for j in 0..nb {
                    asm.vdpbf16ps(acc(i, j), abcast, breg(j))?;
                }
            }
            Ok(())
        };

        // The final half-pair is peeled out of the loop: it does not have the
        // same loading, and a shape with an even `K` emits none.
        let full_pairs = spec.k_pairs - usize::from(spec.k_odd);
        let trip = full_pairs / unroll;
        let rem = full_pairs % unroll;

        if trip > 0 {
            asm.mov(rcx, trip as u64)?;
            asm.set_label(&mut loop_label)?;
            body(&mut asm, 0, unroll)?;
            asm.add(rdi, (unroll * a_pair) as i32)?;
            asm.add(rsi, (unroll * 64) as i32)?;
            asm.dec(rcx)?;
            asm.jne(loop_label)?;
        }
        // Remainder peeled in a straight line: `k_pairs` is an immediate, there is
        // nothing to test at runtime.
        if rem > 0 {
            body(&mut asm, 0, rem)?;
        }
        if spec.k_odd {
            half_pair(&mut asm, rem)?;
        }

        // Epilogue. Accumulation was done in f32 in the zmm registers; what goes
        // to memory respects the invariant "never f32 in memory".
        //
        // On `N % 16 ≠ 0`, the **last** group of 16 columns is partial:
        // its store goes under mask, the others remain full. `k1` is
        // free (no mask register is callee-saved in SysV) and set
        // once per call, outside any loop.
        let masked = spec.n_tail < ROWS_INTERLEAVED;
        if masked {
            asm.mov(eax, (1u32 << spec.n_tail) - 1)?;
            asm.kmovw(k1, eax)?;
        }
        if spec.out_bf16 {
            // `vcvtneps2bf16` : 16 f32 (zmm) → 16 bf16 (ymm), round to nearest
            // even, one instruction. The block is 32 bytes instead of 64.
            //
            // The conversion targets **ymm0**, not the ymm view of the accumulator:
            // `vcvtneps2bf16` is EVEX (it accepts a high source) but the
            // `vmovups` that follows is VEX and cannot encode `ymm16+`, which
            // a 6×48 tile reaches as early as its seventeenth accumulator.
            // ymm0 is legitimate as a universal scratch register because
            // the accumulator at index 0 is written **first**: from that point on
            // zmm0 is dead, and before that the conversion reads it entirely
            // before writing to it.
            for i in 0..mr {
                for j in 0..nb {
                    let at = (i * ldc as usize + j * 32) as i32;
                    asm.vcvtneps2bf16(YMM[0], acc(i, j))?;
                    if masked && j == nb - 1 {
                        // `vmovdqu16` instead of `vmovups`: the mask must apply to
                        // **words**, not dwords — a masked `vmovups`
                        // would let through columns in pairs.
                        asm.vmovdqu16(ymmword_ptr(rdx + at).k1(), YMM[0])?;
                    } else {
                        asm.vmovups(ymmword_ptr(rdx + at), YMM[0])?;
                    }
                }
            }
        } else {
            for i in 0..mr {
                for j in 0..nb {
                    let at = (i * ldc as usize + j * 64) as i32;
                    if masked && j == nb - 1 {
                        asm.vmovups(zmmword_ptr(rdx + at).k1(), acc(i, j))?;
                    } else {
                        asm.vmovups(zmmword_ptr(rdx + at), acc(i, j))?;
                    }
                }
            }
        }

        asm.vzeroupper()?;
        asm.ret()?;
        asm.assemble(0)
    }
}
