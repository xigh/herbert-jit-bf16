//! The gated RMS norm of the GatedDeltaNet mixer, AVX-512: `dim` immediate
//! (the `head_v_dim` width of a row), `n` rows in a loop.
//!
//! Per row `(position, V head)` of width `dim`:
//!
//! ```text
//! inv_rms = 1 / sqrt(Σ_d x[d]² / dim + eps)            computed ONCE per row
//! y[d]    = ((x[d] · inv_rms) · w[d]) · silu(z[d])      silu(z) = z / (1 + exp(−z))
//! ```
//!
//! `mean(x²)`, not `Σx²` (the L2 norm of Q and K does not divide by `dim`);
//! `eps` is the config's, not the L2 norm's; norm, then gate.
//!
//! ```text
//! pass 1: acc_c ← Σ_b x_b · x_b   (vfmadd231ps, 4 chains) ;  s ← hsum(acc)
//!           inv ← 1 / sqrt(s / dim + eps)   (vdivss, vaddss, vsqrtss, vdivss — the literal form, scalar SSE)
//! pass 2, per block of 16:
//!           y ← (x · inv) · w_b ;  e ← exp(min(−z, 88)) ;  y ← y · (z / (1 + e))
//!           out ← y (f32, in place) ;  out_bf16 ← vcvtneps2bf16(y)
//! ```
//!
//! `w` is widened once, in the prologue (`dim/16` zmm, hence [`DIM_MAX`]);
//! `x` is read twice, from L1. The output is within tolerance: Σx² in four
//! chains then a tree where the scalar sums in order, and the `exp` is the
//! emitted polynomial. The associativity `((x·inv)·w)·silu` is kept, and
//! `out_bf16` is the round-to-nearest-even of `out` to the bit.
//!
//! ABI: `fn(out: *mut f32, z: *const f32, w: *const u16, out_bf16: *mut u16,
//! n_rows: usize, eps: f32)` — sysv64 `rdi, rsi, rdx, rcx, r8, xmm0`. `out`,
//! `z` and `out_bf16` advance by `dim` per row; `w` is fixed. `n_rows ≥ 1`.

pub(crate) use crate::jit::emit_gated_norm::{GatedNormSpec, LANES};
pub use x86::emit_gated_norm;

mod x86 {
    use super::{GatedNormSpec, LANES};
    use crate::jit::target::x86::emit_elem::x86::{bcast_imm, exp_inline, hsum, load_exp_constants, ExpConsts, EXP_MAX88};
    use crate::jit::ir::Decline;
    use crate::jit::regs::RegFile;
    use crate::core::error::{HerbertError, Result};
    use iced_x86::code_asm::*;
    use iced_x86::IcedError;

    const ZMM: [AsmRegisterZmm; 32] = [
        zmm0, zmm1, zmm2, zmm3, zmm4, zmm5, zmm6, zmm7, zmm8, zmm9, zmm10, zmm11, zmm12, zmm13,
        zmm14, zmm15, zmm16, zmm17, zmm18, zmm19, zmm20, zmm21, zmm22, zmm23, zmm24, zmm25, zmm26,
        zmm27, zmm28, zmm29, zmm30, zmm31,
    ];
    const YMM: [AsmRegisterYmm; 32] = [
        ymm0, ymm1, ymm2, ymm3, ymm4, ymm5, ymm6, ymm7, ymm8, ymm9, ymm10, ymm11, ymm12, ymm13,
        ymm14, ymm15, ymm16, ymm17, ymm18, ymm19, ymm20, ymm21, ymm22, ymm23, ymm24, ymm25, ymm26,
        ymm27, ymm28, ymm29, ymm30, ymm31,
    ];
    const XMM: [AsmRegisterXmm; 32] = [
        xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13,
        xmm14, xmm15, xmm16, xmm17, xmm18, xmm19, xmm20, xmm21, xmm22, xmm23, xmm24, xmm25, xmm26,
        xmm27, xmm28, xmm29, xmm30, xmm31,
    ];

    /// Emits the kernel of `spec`. The shape must have passed [`GatedNormSpec::check`].
    pub fn emit_gated_norm(spec: &GatedNormSpec) -> Result<Vec<u8>> {
        assemble(spec).map_err(|e| {
            HerbertError::Backend(format!("backend_jit: gated norm emission dim={} : {e}", spec.dim))
        })
    }

    #[derive(Debug)]
    enum EmitErr {
        Iced(IcedError),
        Alloc(Decline),
    }
    impl From<IcedError> for EmitErr {
        fn from(e: IcedError) -> Self {
            EmitErr::Iced(e)
        }
    }
    impl From<Decline> for EmitErr {
        fn from(d: Decline) -> Self {
            EmitErr::Alloc(d)
        }
    }
    impl std::fmt::Display for EmitErr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                EmitErr::Iced(e) => write!(f, "{e}"),
                EmitErr::Alloc(d) => write!(f, "register allocation: {d}"),
            }
        }
    }

    // rdi = out (f32, in place), rsi = z (f32), rdx = w (bf16, fixed),
    // rcx = out_bf16, r8 = n_lines, xmm0 = eps. `rax` is used for immediate
    // constants, `r10d` keeps `eps` out of reach of the vector registers.
    fn assemble(spec: &GatedNormSpec) -> std::result::Result<Vec<u8>, EmitErr> {
        let nb = spec.dim / LANES;
        debug_assert!(nb >= 1);
        let mut asm = CodeAssembler::new(64)?;
        let mut f = RegFile::new();
        // `eps` arrives in xmm0, which the prologue and horizontal sum will
        // clobber: an **integer** register is out of reach of everything
        // this kernel does (the lesson of `ElemOp::RmsNorm`).
        asm.vmovd(r10d, xmm0)?;
        let k = ExpConsts::alloc(&mut f)?;
        let max88 = f.high("norme_gatee:upper bound of exp")?;
        let w = f.run(nb, "norme_gatee: expanded, one ZMM per block")?;
        let n_acc = 4.min(nb);
        let acc = f.low_run(n_acc, "norme_gatee:chains of Σx²")?;
        let ld = f.low("norme_gatee:bloc lu / temporary scalar")?;
        let inv = f.any("norme_gatee:1/rms broadcast")?;
        let x = f.any("norme_gatee: x block, then y")?;
        let z = f.any("norme_gatee:z block")?;
        let e = f.any("gated_norm:−z, exp, then silu")?;
        let t = f.run(3, "norme_gatee: temporaries of the exp, then the bf16")?;

        // ── prologue: constants, `w` widened once and for all ─────────
        load_exp_constants(&mut asm, &k)?;
        bcast_imm(&mut asm, max88, EXP_MAX88.to_bits())?;
        for b in 0..nb {
            asm.vpmovzxwd(ZMM[w[b].idx()], ymmword_ptr(rdx + (b * 32) as i32))?;
            asm.vpslld(ZMM[w[b].idx()], ZMM[w[b].idx()], 16u32)?;
        }

        // ── the loop over the rows ───────────────────────────────────────
        let mut line = asm.create_label();
        asm.set_label(&mut line)?;
        // Pass 1: Σx² in `n_acc` chains, then in a tree, then horizontal.
        for a in &acc {
            asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
        }
        for b in 0..nb {
            asm.vmovups(ZMM[ld.idx()], zmmword_ptr(rdi + (b * 64) as i32))?;
            asm.vfmadd231ps(ZMM[acc[b % n_acc].idx()], ZMM[ld.idx()], ZMM[ld.idx()])?;
        }
        let a0 = acc[0];
        for a in &acc[1..] {
            asm.vaddps(ZMM[a0.idx()], ZMM[a0.idx()], ZMM[a.idx()])?;
        }
        hsum(&mut asm, a0, ld)?;
        // inv = 1 / sqrt(s / dim + eps) — the literal shape of the scalar, in
        // this order: a division by `dim` (and not a product by `1/dim`),
        // the `eps` returned from its integer register, the root, then `1/·`.
        asm.mov(eax, (spec.dim as f32).to_bits())?;
        asm.vmovd(XMM[ld.x()], eax)?;
        asm.vdivss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
        asm.vmovd(XMM[ld.x()], r10d)?;
        asm.vaddss(XMM[a0.x()], XMM[a0.x()], XMM[ld.x()])?;
        asm.vsqrtss(XMM[a0.x()], XMM[a0.x()], XMM[a0.x()])?;
        asm.mov(eax, 1.0f32.to_bits())?;
        asm.vmovd(XMM[ld.x()], eax)?;
        asm.vdivss(XMM[a0.x()], XMM[ld.x()], XMM[a0.x()])?;
        asm.vbroadcastss(ZMM[inv.idx()], XMM[a0.x()])?;
        // Pass 2: y = ((x · inv) · w) · silu(z), block by block, f32 in place
        // and bf16 alongside.
        for b in 0..nb {
            let (o32, o16) = ((b * 64) as i32, (b * 32) as i32);
            asm.vmovups(ZMM[x.idx()], zmmword_ptr(rdi + o32))?;
            asm.vmulps(ZMM[x.idx()], ZMM[x.idx()], ZMM[inv.idx()])?;
            asm.vmulps(ZMM[x.idx()], ZMM[x.idx()], ZMM[w[b].idx()])?;
            asm.vmovups(ZMM[z.idx()], zmmword_ptr(rsi + o32))?;
            // silu(z) = z / (1 + exp(min(−z, 88))) — the SwiGLU shape emitted.
            asm.vxorps(ZMM[e.idx()], ZMM[e.idx()], ZMM[e.idx()])?;
            asm.vsubps(ZMM[e.idx()], ZMM[e.idx()], ZMM[z.idx()])?;
            asm.vminps(ZMM[e.idx()], ZMM[e.idx()], ZMM[max88.idx()])?;
            exp_inline(&mut asm, &k, e, t[0], t[1], t[2])?;
            asm.vaddps(ZMM[e.idx()], ZMM[e.idx()], ZMM[k.one.idx()])?;
            asm.vdivps(ZMM[e.idx()], ZMM[z.idx()], ZMM[e.idx()])?;
            asm.vmulps(ZMM[x.idx()], ZMM[x.idx()], ZMM[e.idx()])?;
            asm.vmovups(zmmword_ptr(rdi + o32), ZMM[x.idx()])?;
            asm.vcvtneps2bf16(YMM[t[0].y_evex()], ZMM[x.idx()])?;
            asm.vmovdqu16(ymmword_ptr(rcx + o16), YMM[t[0].y_evex()])?;
        }
        // Next line.
        asm.add(rdi, (spec.dim * 4) as i32)?;
        asm.add(rsi, (spec.dim * 4) as i32)?;
        asm.add(rcx, (spec.dim * 2) as i32)?;
        asm.dec(r8)?;
        asm.jnz(line)?;
        asm.vzeroupper()?;
        asm.ret()?;
        Ok(asm.assemble(0)?)
    }
}
