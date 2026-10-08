//! The causal depthwise conv + SiLU of the GatedDeltaNet mixer, AVX-512: one
//! position, `n` blocks of 16 channels, `K = 4` taps as an immediate.
//!
//! Per channel `ch`, with the state `state[ch·K + j]`, for each position:
//!
//! ```text
//! state[ch·K + j] ← state[ch·K + j + 1]   (j < K−1)
//! state[ch·K + K−1] ← x[ch]
//! y[ch] ← silu(Σ_j state[ch·K + j] · w[ch·K + j])
//! ```
//!
//! The layout `ch·K + j` is kept: at `K = 4`, sixteen channels are 64 f32,
//! four zmm, and the shift of one f32 to the left over the 64 lanes is a
//! `valignd` per zmm; the input slides into lanes `4c+3` with a masked
//! `vpermps`. Per block of 16 channels:
//!
//! ```text
//! x ← data[0..16] ;  s_b ← state[64b..64b+16]          (b = 0..3)
//! ns_b ← valignd(s_{b+1}, s_b, 1)                       (state[i] ← state[i+1])
//! ns_b{k1=0x8888} ← vpermps(idx_x, rot(x, 4b))           (lane 4c+3 ← x[4b+c])
//! state ← ns                                            (the same bits as the scalar)
//! w_b ← widened(bf16) ;  p_b ← ns_b · w_b
//! u_b ← p_b + valignd(p_b,p_b,1) ;  u_b ← u_b + valignd(u_b,u_b,2)   (lane 4c = Σ of the 4 lanes)
//! g ← vpermt2ps(u_0,u_1)[0,4,8,12 | 16,20,24,28] ⧺ idem(u_2,u_3)    (16 sums, channels in order)
//! y ← g / (1 + exp(min(−g, 88)))                        (exp_inline, the polynomial of emit_elem)
//! data[0..16] ← y
//! ```
//!
//! The returned state is bit for bit the scalar's (copies and insertion, no
//! arithmetic). The output is within tolerance: the four taps are summed as
//! a tree, `(p0+p1)+(p2+p3)`, and the `exp` is the emitted polynomial, not
//! libm's — the regime of every activation the engine emits.
//!
//! ABI: `fn(data: *mut f32, w: *const u16, state: *mut f32, n_blocks: usize)`
//! — sysv64 `rdi, rsi, rdx, rcx`. The three pointers are already offset to
//! the first channel of the range; the kernel advances by 64 / 128 / 256
//! bytes per block. `n_blocks ≥ 1`.

pub(crate) use crate::jit::emit_conv::ConvSpec;
pub use x86::emit_conv;

mod x86 {
    use super::ConvSpec;
    use crate::jit::target::x86::emit_elem::x86::{bcast_imm, exp_inline, load_exp_constants, ExpConsts, EXP_MAX88};
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

    /// Emits the kernel of `spec`. The shape must have passed [`ConvSpec::check`].
    pub fn emit_conv(spec: &ConvSpec) -> Result<Vec<u8>> {
        assemble(spec).map_err(|e| HerbertError::Backend(format!("backend_jit: conv K={} emission: {e}", spec.k)))
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

    /// Load into `dst` a vector of 16 dwords indices, from a table
    /// placed **in the code** (after the `ret`), addressed by `lea`.
    fn load_indices(
        asm: &mut CodeAssembler,
        dst: usize,
        idx: &[u32; 16],
        tables: &mut Vec<(CodeLabel, [u32; 16])>,
    ) -> std::result::Result<(), IcedError> {
        let lbl = asm.create_label();
        asm.lea(rax, ptr(lbl))?;
        asm.vmovdqu32(ZMM[dst], zmmword_ptr(rax))?;
        tables.push((lbl, *idx));
        Ok(())
    }

    // rdi = data (f32, canal c0), rsi = w (bf16, c0·4), rdx = state (f32, c0·4),
    // rcx = n_blocks. `rax` is used for `lea`/`mov` of constants, `k1` for the mask
    // insertion.
    fn assemble(spec: &ConvSpec) -> std::result::Result<Vec<u8>, EmitErr> {
        debug_assert_eq!(spec.k, 4);
        let mut asm = CodeAssembler::new(64)?;
        let mut f = RegFile::new();
        let k = ExpConsts::alloc(&mut f)?;
        let max88 = f.high("conv:upper bound of exp")?;
        let idx_x = f.high("conv:insertion indices of x (lanes 4c+3 ← c)")?;
        let idx_g = f.high("conv:sum collection indices (0,4,8,12 | +16)")?;
        let x = f.any("conv:x, the input of 16 channels")?;
        let s = f.run(4, "conv:read state (4 ZMM = 16 channels × 4 taps), then expanded weights and products")?;
        let ns = f.run(4, "conv:shifted state")?;
        let e = f.any("conv:−g, exp, then y")?;
        let t = f.run(3, "conv: temporaries of the expr, and x flipped")?;
        let g = f.any("conv:the 16 sums")?;
        let g2 = f.any("conv:the sums of blocks 2 and 3")?;

        // ── prologue : constantes, masque ─────────────────────────────────
        load_exp_constants(&mut asm, &k)?;
        bcast_imm(&mut asm, max88, EXP_MAX88.to_bits())?;
        let mut tables: Vec<(CodeLabel, [u32; 16])> = Vec::new();
        // lane 4c+3 ← x_rotated[c] ; other lanes are masked (k1).
        let mut ix = [0u32; 16];
        for c in 0..4 {
            ix[4 * c + 3] = c as u32;
        }
        load_indices(&mut asm, idx_x.idx(), &ix, &mut tables)?;
        // vpermt2ps : indices < 16 pick from the first source, ≥ 16 from the second.
        let ig: [u32; 16] = [0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0];
        load_indices(&mut asm, idx_g.idx(), &ig, &mut tables)?;
        asm.mov(eax, 0x8888u32)?;
        asm.kmovw(k1, eax)?;

        // ── the loop over blocks ───────────────────────────────────────
        let mut bloc = asm.create_label();
        asm.set_label(&mut bloc)?;
        asm.vmovups(ZMM[x.idx()], zmmword_ptr(rdi))?;
        for b in 0..4 {
            asm.vmovups(ZMM[s[b].idx()], zmmword_ptr(rdx + (b * 64) as i32))?;
        }
        // Shift: ns_b[i] = s_b[i+1], ns_b[15] = s_{b+1}[0] (lane 15 is a
        // K−1 take, rewritten just after: the second source of b = 3 does not matter).
        for b in 0..4 {
            let hi = if b < 3 { s[b + 1] } else { s[b] };
            asm.valignd(ZMM[ns[b].idx()], ZMM[hi.idx()], ZMM[s[b].idx()], 1)?;
        }
        // Insertion of x in lanes 4c+3: x rotated by 4b lanes (t[0]), then
        // masked vpermps.
        for b in 0..4 {
            if b == 0 {
                asm.vpermps(ZMM[ns[b].idx()].k1(), ZMM[idx_x.idx()], ZMM[x.idx()])?;
            } else {
                asm.valignd(ZMM[t[0].idx()], ZMM[x.idx()], ZMM[x.idx()], (4 * b) as u32)?;
                asm.vpermps(ZMM[ns[b].idx()].k1(), ZMM[idx_x.idx()], ZMM[t[0].idx()])?;
            }
        }
        for b in 0..4 {
            asm.vmovups(zmmword_ptr(rdx + (b * 64) as i32), ZMM[ns[b].idx()])?;
        }
        // Weights: 64 bf16 = two ZMM, widened to four ZMM f32 (in s, free).
        //   s0 ← bf16[0..32], s1 ← bf16[32..64]
        //   w0 = low(s0) → s2 ; w1 = high(s0) → s3 ; w2 = low(s1) → s0 ; w3 = high(s1) → s1
        asm.vmovdqu16(ZMM[s[0].idx()], zmmword_ptr(rsi))?;
        asm.vmovdqu16(ZMM[s[1].idx()], zmmword_ptr(rsi + 64))?;
        asm.vpmovzxwd(ZMM[s[2].idx()], YMM[s[0].y_evex()])?;
        asm.vextracti64x4(YMM[s[3].y_evex()], ZMM[s[0].idx()], 1)?;
        asm.vpmovzxwd(ZMM[s[3].idx()], YMM[s[3].y_evex()])?;
        asm.vpmovzxwd(ZMM[s[0].idx()], YMM[s[1].y_evex()])?;
        asm.vextracti64x4(YMM[s[1].y_evex()], ZMM[s[1].idx()], 1)?;
        asm.vpmovzxwd(ZMM[s[1].idx()], YMM[s[1].y_evex()])?;
        // The order of expanded weights: w_b for block b = [s2, s3, s0, s1].
        let w = [s[2], s[3], s[0], s[1]];
        for b in 0..4 {
            asm.vpslld(ZMM[w[b].idx()], ZMM[w[b].idx()], 16u32)?;
        }
        // Products, then sum by groups of 4 lanes (lane 4c = Σ of channel c).
        for b in 0..4 {
            asm.vmulps(ZMM[w[b].idx()], ZMM[w[b].idx()], ZMM[ns[b].idx()])?;
            asm.valignd(ZMM[ns[b].idx()], ZMM[w[b].idx()], ZMM[w[b].idx()], 1)?;
            asm.vaddps(ZMM[w[b].idx()], ZMM[w[b].idx()], ZMM[ns[b].idx()])?;
            asm.valignd(ZMM[ns[b].idx()], ZMM[w[b].idx()], ZMM[w[b].idx()], 2)?;
            asm.vaddps(ZMM[w[b].idx()], ZMM[w[b].idx()], ZMM[ns[b].idx()])?;
        }
        // Collect: g[0..8] ← lanes 0,4,8,12 of w0 then of w1; g2 same (w2, w3);
        // g[8..16] ← g2[0..8].
        asm.vmovaps(ZMM[g.idx()], ZMM[w[0].idx()])?;
        asm.vpermt2ps(ZMM[g.idx()], ZMM[idx_g.idx()], ZMM[w[1].idx()])?;
        asm.vmovaps(ZMM[g2.idx()], ZMM[w[2].idx()])?;
        asm.vpermt2ps(ZMM[g2.idx()], ZMM[idx_g.idx()], ZMM[w[3].idx()])?;
        asm.vinserti64x4(ZMM[g.idx()], ZMM[g.idx()], YMM[g2.y_evex()], 1)?;
        // SiLU: y = g / (1 + exp(min(−g, 88))) — the SwiGLU form emitted.
        asm.vxorps(ZMM[e.idx()], ZMM[e.idx()], ZMM[e.idx()])?;
        asm.vsubps(ZMM[e.idx()], ZMM[e.idx()], ZMM[g.idx()])?;
        asm.vminps(ZMM[e.idx()], ZMM[e.idx()], ZMM[max88.idx()])?;
        exp_inline(&mut asm, &k, e, t[0], t[1], t[2])?;
        asm.vaddps(ZMM[e.idx()], ZMM[e.idx()], ZMM[k.one.idx()])?;
        asm.vdivps(ZMM[e.idx()], ZMM[g.idx()], ZMM[e.idx()])?;
        asm.vmovups(zmmword_ptr(rdi), ZMM[e.idx()])?;
        // Bloc next.
        asm.add(rdi, 64)?;
        asm.add(rsi, 128)?;
        asm.add(rdx, 256)?;
        asm.dec(rcx)?;
        asm.jnz(bloc)?;
        asm.vzeroupper()?;
        asm.ret()?;
        // The index tables, in the code page, behind the `ret`.
        for (mut lbl, idx) in tables {
            asm.set_label(&mut lbl)?;
            for v in idx {
                asm.dd(&[v])?;
            }
        }
        Ok(asm.assemble(0)?)
    }
}
