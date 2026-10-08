//! SDPA, AVX-512: the decode passes — QK, softmax, SV — and the fused form
//! that keeps the scores in registers.

#[allow(unused_imports)]
use super::*;

// ───────────────────────── passes ─────────────────────────────────────

/// QK pass. `q` **bf16** in `rdi`, `k` bf16 in `rsi` (advanced), `scores` in
/// `rdx` (advanced), counter in `rcx`, scale broadcast to
/// `xmm[scale_reg]`, `n` kept in `r10`.
///
/// **`vdpbf16ps`**: both operands being in bf16 — direct consequence
/// of the invariant "never f32 in memory" — the dot product fits
/// in **one instruction for 32 elements**, with f32 accumulation. The
/// previous path required 3 per 16 elements (`vpmovzxwd`, `vpslld`,
/// `vfmadd231ps`), i.e., **24 instructions per 32 elements versus 2**.
///
/// Four independent accumulation chains (the latency of `vdpbf16ps`
/// is 4 cycles, a single chain would pay full price) and four
/// read temporaries for `K` — all in low area: chain 0 and the
/// first temporary serve the horizontal reduction, which is VEX.
pub(super) struct QkRegs {
    /// Accumulation chains, **contiguous** (the emitted code indexes `c % n`).
    pub(super) acc: Vec<Vlow>,
    /// `K` rows in flight.
    pub(super) tmp: Vec<Vlow>,
    /// Scale `1/√head_dim`, provided by the caller.
    pub(super) scale: Vlow,
}

impl QkRegs {
    /// `chains` = number of accumulation chains requested. `4` is the
    /// historical kernel; `1` serves the **rest** (`n % 16`) of the 16-position
    /// kernel, which must sum like its full groups — same reason,
    /// and same value, as the `chains` of [`QkBlockRegs::alloc`].
    pub(super) fn alloc(
        f: &mut RegFile,
        spec: &SdpaKernelSpec,
        chains: usize,
    ) -> std::result::Result<Self, Decline> {
        let n_acc = chains.min(spec.shape.vecs().div_ceil(2));
        Ok(Self {
            acc: f.low_run(n_acc, "qk:chains")?,
            tmp: f.low_run(4, "qk:lines of K")?,
            scale: f.low("qk:scale")?,
        })
    }
    /// The chains and temporaries die with the QK pass; the scale
    /// too. Saying so allows subsequent passes to reuse the space.
    pub(super) fn free(self, f: &mut RegFile) {
        for r in self.acc.iter().chain(self.tmp.iter()) {
            f.free_low(*r);
        }
        f.free_low(self.scale);
    }
}

pub(super) fn emit_qk(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    r: &QkRegs,
) -> std::result::Result<(), IcedError> {
    let nv = spec.shape.vecs();
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    let n_acc = r.acc.len();

    let mut loop_top = asm.create_label();
    let mut after_loop = asm.create_label();
    let mut no_pad = asm.create_label();
    let mut pad_top = asm.create_label();

    asm.test(rcx, rcx)?;
    asm.jz(after_loop)?;
    asm.set_label(&mut loop_top)?;
    for a in &r.acc {
        asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
    }
    if spec.pf_dist > 0 {
        asm.prefetcht0(byte_ptr(rsi + row_bytes * spec.pf_dist as i32))?;
    }
    // A zmm holds 32 bf16. `vdpbf16ps acc, k, [q]` accumulates the 32
    // products in the 16 f32 lanes of the accumulator.
    let n32 = nv.div_ceil(2);
    for c in 0..n32 {
        let t = r.tmp[c % 4].idx();
        asm.vmovdqu16(ZMM[t], zmmword_ptr(rsi + (c * 64) as i32))?;
        asm.vdpbf16ps(
            ZMM[r.acc[c % n_acc].idx()],
            ZMM[t],
            zmmword_ptr(rdi + (c * 64) as i32),
        )?;
    }
    // Tree reduction of chains, then horizontal sum.
    let a0 = r.acc[0].idx();
    if n_acc == 4 {
        asm.vaddps(ZMM[a0], ZMM[a0], ZMM[r.acc[1].idx()])?;
        asm.vaddps(ZMM[r.acc[2].idx()], ZMM[r.acc[2].idx()], ZMM[r.acc[3].idx()])?;
        asm.vaddps(ZMM[a0], ZMM[a0], ZMM[r.acc[2].idx()])?;
    } else {
        for a in &r.acc[1..] {
            asm.vaddps(ZMM[a0], ZMM[a0], ZMM[a.idx()])?;
        }
    }
    hsum(asm, r.acc[0], r.tmp[0])?;
    asm.vmulss(XMM[a0], XMM[a0], XMM[r.scale.x()])?;
    asm.vmovss(dword_ptr(rdx), XMM[a0])?;
    asm.add(rsi, row_bytes)?;
    asm.add(rdx, 4)?;
    asm.dec(rcx)?;
    asm.jnz(loop_top)?;

    // Padding `-inf` up to a multiple of 16: it is the one that makes the three
    // passes of softmax without mask (see the module header).
    asm.set_label(&mut after_loop)?;
    asm.mov(rax, r10)?;
    asm.and(rax, 15)?;
    asm.jz(no_pad)?;
    asm.mov(r11, 16u64)?;
    asm.sub(r11, rax)?;
    asm.set_label(&mut pad_top)?;
    asm.mov(dword_ptr(rdx), 0xFF80_0000u32)?;
    asm.add(rdx, 4)?;
    asm.dec(r11)?;
    asm.jnz(pad_top)?;
    asm.set_label(&mut no_pad)?;
    Ok(())
}

/// Registers of the softmax pass. `maxv` and `sumv` are the only two states
/// that traverse the exp loop: they live out of reach of everything the
/// exponential clobbers.
pub(super) struct SoftmaxRegs {
    /// Current maximum. Vectorial, living through the loop → high zone.
    pub(super) maxv: V,
    /// Somme courante. Idem.
    pub(super) sumv: V,
    /// Block of 16 scores in progress. Input-output of the exp, so spilled by
    /// `exp_call`: does not need to survive in a register.
    pub(super) work: Vlow,
    /// Reduced maximum — **scalar**, and that's what allows it to stay in
    /// the low zone across a call: `vzeroupper` does not touch 0:127.
    pub(super) red_max: Vlow,
    /// Horizontal reduction temporary, and scratch for the `1.0`
    /// normalization sink constant. Dead at every call site.
    pub(super) red_tmp: Vlow,
    /// Reduced sum — scalar, written after the exp loop.
    pub(super) red_sum: Vlow,
    /// Polynomial temporaries, **allocated only if `Inline`** : with
    /// `Call` they do not exist, rather than existing dead.
    pub(super) exp_t: Option<[V; 3]>,
}

impl SoftmaxRegs {
    pub(super) fn alloc(f: &mut RegFile, exp: ExpStrategy) -> std::result::Result<Self, Decline> {
        Ok(Self {
            maxv: f.high("softmax:maximum courant")?,
            sumv: f.high("softmax:somme courante")?,
            work: f.low("softmax: 16-block (spilled by exp_call)")?,
            red_max: f.low("softmax:maximum reduced (scalar)")?,
            red_tmp: f.low("softmax:reduction temporary (dead at call)")?,
            red_sum: f.low("softmax:reduced sum (scalar)")?,
            exp_t: match exp {
                ExpStrategy::Inline => Some([
                    f.any("softmax:exp t1")?,
                    f.any("softmax:exp t2")?,
                    f.any("softmax:exp t3")?,
                ]),
            },
        })
    }

    /// The low roles still alive at the exp call site, and the pattern
    /// of their exemption — **written in their name**. All scalars (bits
    /// 0:127, which `vzeroupper` spares) or dead at this point. `work`
    /// exempts itself: `exp_call` spills it and reads it back.
    ///
    /// What this list does **not** contain is the essentials: neither the
    /// maximum nor the current sum, which are vectorial and live through
    /// the loop. They are in the high zone by construction, and this is
    /// the bug of the 6th session that is thus closed.
    pub(super) const EXEMPT: &'static [&'static str] = &[
        SINK_ROLE,
        "softmax:maximum reduced (scalar)",
        "softmax:reduction temporary (dead at call)",
        "softmax:reduced sum (scalar)",
    ];

    /// The triplet of temporaries, or three times `work` when the strategy
    /// is `Call`: `emit_exp` does not read them in this case.
    pub(super) fn t(&self) -> [V; 3] {
        self.exp_t.unwrap_or([self.work.into(); 3])
    }

    pub(super) fn free(self, f: &mut RegFile) {
        f.free(self.maxv);
        f.free(self.sumv);
        for r in [self.work, self.red_max, self.red_tmp, self.red_sum] {
            f.free_low(r);
        }
        for t in self.exp_t.into_iter().flatten() {
            f.free(t);
        }
    }
}

/// `sink_on` rather than the full spec: this pass is **reused as is**
/// by the prefill kernel, which re-emits it row by row.
/// It is this literal reuse — the same code, not an
/// equivalent one — that guarantees that the softmax of a prefill row produces the
/// same bits as that of decode.
pub(super) fn emit_softmax(
    asm: &mut CodeAssembler,
    sink_on: bool,
    f: &RegFile,
    r: &SoftmaxRegs,
    sink: Vlow,
    e: &Exp<'_>,
) -> std::result::Result<(), IcedError> {
    let (maxv, sumv) = (r.maxv.idx(), r.sumv.idx());
    let work = r.work.idx();
    let [t1, t2, t3] = r.t();
    let mut max_top = asm.create_label();
    let mut exp_top = asm.create_label();
    let mut norm_top = asm.create_label();

    // ── 1. maximum ────────────────────────────────────────────────────
    bcast_imm(asm, r.maxv, f32::NEG_INFINITY.to_bits())?;
    asm.mov(rsi, rdi)?;
    asm.mov(rcx, r11)?;
    asm.set_label(&mut max_top)?;
    asm.vmovups(ZMM[work], zmmword_ptr(rsi))?;
    asm.vmaxps(ZMM[maxv], ZMM[maxv], ZMM[work])?;
    asm.add(rsi, 64)?;
    asm.dec(rcx)?;
    asm.jnz(max_top)?;
    asm.vmovaps(ZMM[r.red_max.idx()], ZMM[maxv])?;
    hmax(asm, r.red_max, r.red_tmp)?;
    if sink_on {
        // `softmax_inplace_with_sink` starts with `max_val = sink`.
        asm.vmaxss(XMM[r.red_max.x()], XMM[r.red_max.x()], XMM[sink.x()])?;
    }
    asm.vbroadcastss(ZMM[maxv], XMM[r.red_max.x()])?;

    // ── 2. exp and sum ───────────────────────────────────────────────
    asm.vxorps(ZMM[sumv], ZMM[sumv], ZMM[sumv])?;
    asm.mov(rsi, rdi)?;
    asm.mov(rcx, r11)?;
    asm.set_label(&mut exp_top)?;
    asm.vmovups(ZMM[work], zmmword_ptr(rsi))?;
    asm.vsubps(ZMM[work], ZMM[work], ZMM[maxv])?;
    emit_exp(asm, e, f, r.work.into(), t1, t2, t3)?;
    asm.vaddps(ZMM[sumv], ZMM[sumv], ZMM[work])?;
    asm.vmovups(zmmword_ptr(rsi), ZMM[work])?;
    asm.add(rsi, 64)?;
    asm.dec(rcx)?;
    asm.jnz(exp_top)?;
    asm.vmovaps(ZMM[r.red_sum.idx()], ZMM[sumv])?;
    hsum(asm, r.red_sum, r.red_tmp)?;
    if sink_on {
        // The reference sum includes the sink term, which the
        // normalization does not touch afterwards. A `Call` here would destroy the
        // upper lanes of `red_sum` — but not its scalar (bits 0:127),
        // the only one that matters now. Hence its named exemption.
        asm.vsubss(XMM[r.red_tmp.x()], XMM[sink.x()], XMM[r.red_max.x()])?;
        asm.vbroadcastss(ZMM[work], XMM[r.red_tmp.x()])?;
        emit_exp(asm, e, f, r.work.into(), t1, t2, t3)?;
        asm.vaddss(XMM[r.red_sum.x()], XMM[r.red_sum.x()], XMM[work])?;
    }

    // ── 3. normalization ──────────────────────────────────────────────
    if sink_on {
        // `*score *= 1/sum` — the sink variant computes the inverse once.
        bcast_imm(asm, r.red_tmp.into(), 1.0f32.to_bits())?;
        asm.vdivss(XMM[r.red_sum.x()], XMM[r.red_tmp.x()], XMM[r.red_sum.x()])?;
    }
    asm.vbroadcastss(ZMM[sumv], XMM[r.red_sum.x()])?;
    asm.mov(rsi, rdi)?;
    asm.mov(rcx, r11)?;
    asm.set_label(&mut norm_top)?;
    asm.vmovups(ZMM[work], zmmword_ptr(rsi))?;
    if sink_on {
        asm.vmulps(ZMM[work], ZMM[work], ZMM[sumv])?;
    } else {
        // `*val /= sum` — division, like `softmax_inplace`.
        asm.vdivps(ZMM[work], ZMM[work], ZMM[sumv])?;
    }
    asm.vmovups(zmmword_ptr(rsi), ZMM[work])?;
    asm.add(rsi, 64)?;
    asm.dec(rcx)?;
    asm.jnz(norm_top)?;
    Ok(())
}

/// SV pass. `scores` in `rdi` (advanced), `v` in `rsi` (advanced), `out` in
/// `rdx`, counter in `rcx`.
///
/// Two regimes, and the choice is not cosmetic: as long as `head_dim/16`
/// accumulators fit in registers, the output does not touch memory
/// for the entire loop. At `head_dim = 512` it would require 32 — all
/// registers, without a single one to read `V` — so the output becomes
/// an accumulator in L1, read-modify-write line by line (this is what
/// `avx512_attn_sv_accum_bf16` does).
/// Registers of the SV pass.
pub(super) struct SvRegs {
    /// Output accumulators **in registers**, one per block of 16
    /// dimensions. **Contiguous** and of arbitrary scope: they are only used for
    /// EVEX forms (`vxorps`, `vfmadd231ps`, source of
    /// `vcvtneps2bf16`).
    ///
    /// There may be **fewer than `vecs()`**: beyond what the
    /// 32 zmm registers hold (`head_dim` > 416), the tail segments live in
    /// the f32 stack scratch. See [`SdpaKernelSpec::sv_acc_in_regs`].
    pub(super) acc: Vec<V>,
    /// Current score, broadcast.
    pub(super) score: V,
    /// `V` rows converted to f32, and destination of the conversion
    /// bf16 of the epilogue.
    ///
    /// **Low zone**, and it is the epilogue that imposes it:
    /// `vcvtneps2bf16` is EVEX (a high source would suit it), but the
    /// `vmovups ymm` that follows is emitted in VEX, which stops at `ymm15`.
    /// The EVEX form exists (cf. `regs.rs`) and would cost 2 extra bytes
    /// per store: we keep VEX. The constraint is in the type.
    pub(super) tmp: Vec<Vlow>,
}

impl SvRegs {
    pub(super) fn alloc(f: &mut RegFile, spec: &SdpaKernelSpec) -> std::result::Result<Self, Decline> {
        // The low zone first: it is the scarce resource, and an accumulator `run`
        // would consume it entirely.
        let tmp = f.low_run(4, "sv:V rows / bf16 conversion")?;
        Ok(Self {
            tmp,
            score: f.any("sv:score broadcast")?,
            // **Exactly** `sv_acc_in_regs()`, not "as much as we
            // can": it is the same function that sizes the stack
            // scratch (`frame_bytes`). Letting them diverge would write out of
            // bounds. If the allocator cannot serve this count, the refusal
            // propagates and the caller falls back — never an overflow.
            acc: f.run(spec.sv_acc_in_regs(), "output accumulators")?,
        })
    }
    // No `free`: the SV pass is the last of all blobs, nothing
    // reclaims its registers after it.
}

pub(super) fn emit_sv(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    r: &SvRegs,
) -> std::result::Result<(), IcedError> {
    let nv = spec.shape.vecs();
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    let n_reg = r.acc.len();
    debug_assert_eq!(n_reg, spec.sv_acc_in_regs(), "frame and allocation disjoint");
    let acc = |c: usize| r.acc[c].idx();
    let score = r.score.idx();
    // Address of the spillover tile `c` in the stack f32 scratchpad. The
    // frame was sized by `frame_bytes` for exactly these
    // `nv - n_reg` tiles.
    let slot = |c: usize| SV_SPILL + 64 * (c - n_reg) as i32;
    // Round-trip register for the overflow segment. It comes from the **same
    // rotating pool** as the `V` row, offset by 2: `tmp[c%4]` holds the `V`
    // row for segment `c`, `tmp[(c+2)%4]` its partial sum, and the two
    // are always distinct. The pool has only four entries, but each
    // use dies in the following instruction — reuse at segment
    // `c+2` comes later.
    let round = |c: usize| r.tmp[(c + 2) % 4].idx();

    let mut loop_top = asm.create_label();
    let mut done = asm.create_label();

    for c in 0..n_reg {
        asm.vxorps(ZMM[acc(c)], ZMM[acc(c)], ZMM[acc(c)])?;
    }
    if nv > n_reg {
        let z = r.tmp[0].idx();
        asm.vxorps(ZMM[z], ZMM[z], ZMM[z])?;
        for c in n_reg..nv {
            asm.vmovups(zmmword_ptr(rsp + slot(c)), ZMM[z])?;
        }
    }
    asm.test(rcx, rcx)?;
    asm.jz(done)?;
    asm.set_label(&mut loop_top)?;
    asm.vbroadcastss(ZMM[score], dword_ptr(rdi))?;
    if spec.pf_dist > 0 {
        asm.prefetcht0(byte_ptr(rsi + row_bytes * spec.pf_dist as i32))?;
    }
    for c in 0..nv {
        let t = r.tmp[c % 4];
        load_bf16_f32(asm, t.into(), rsi, (c * 32) as i32)?;
        if c < n_reg {
            asm.vfmadd231ps(ZMM[acc(c)], ZMM[t.idx()], ZMM[score])?;
        } else {
            // L1 round-trip: the partial sum is re-read at the
            // next position, so it is hot. This is what the reformulated
            // invariant allows — the f32 does not go lower than L1, and the
            // UMC counter verifies it.
            let u = round(c);
            asm.vmovups(ZMM[u], zmmword_ptr(rsp + slot(c)))?;
            asm.vfmadd231ps(ZMM[u], ZMM[t.idx()], ZMM[score])?;
            asm.vmovups(zmmword_ptr(rsp + slot(c)), ZMM[u])?;
        }
    }
    asm.add(rdi, 4)?;
    asm.add(rsi, row_bytes)?;
    asm.dec(rcx)?;
    asm.jnz(loop_top)?;
    asm.set_label(&mut done)?;
    // bf16 epilogue: accumulation was in f32 — in registers or in the
    // scratch — but what goes to memory is bf16, and the scratch
    // dies here without ever having left the kernel.
    for c in 0..nv {
        let t = r.tmp[c % 4];
        let src = if c < n_reg {
            acc(c)
        } else {
            let u = round(c);
            asm.vmovups(ZMM[u], zmmword_ptr(rsp + slot(c)))?;
            u
        };
        asm.vcvtneps2bf16(YMM[t.y()], ZMM[src])?;
        asm.vmovups(ymmword_ptr(rdx + (c * 32) as i32), YMM[t.y()])?;
    }
    Ok(())
}

/// Merged shape: softmax **online**, in blocks of 16 positions.
///
/// A single pass over the KV cache: for each block of 16 positions we
/// compute the 16 scores, update the current maximum, rescale
/// the output accumulator by the factor `exp(m_ancien − m_nouveau)`, then
/// accumulate the 16 rows of `V`. There is **no score buffer**, and
/// `K[j]` and `V[j]` are touched in the same iteration.
///
/// Registers of the fused shape — **the budget is exactly saturated**
/// at `head_dim = 256`: 16 accumulators + 8 exp constants + 8 for
/// work = 32.
///
/// There is **no** scale register, and this is a constraint, not
/// an oversight: the scale is read as a memory operand from the scratch, once
/// per position. Storing the scale in a temporary was a
/// bug — invisible at `n = 1`, where the
/// softmax of a single position equals 1 regardless of the score. Budget saturation
/// is now **detected by the allocator**: if one more role
/// were requested, `RegFile` would return [`Decline::OutOfRegisters`]
/// instead of allowing an already-taken index to be chosen.
pub(super) struct FusedRegs {
    /// Pool of four low registers, **reused phase by phase** in
    /// order: (1) the dot product chains, (2) the input-output and
    /// the temporaries of the exp polynomial, (3) the broadcast score of
    /// the `V` accumulation, (4) the bf16 conversion of the epilogue.
    ///
    /// Sharing is legitimate because the phases are **sequential** —
    /// this is the only thing to re-verify before touching it. Low zone:
    /// phase (1) reduces horizontally and phase (4) writes a
    /// `vmovups ymm` — two VEX forms, which stop at `ymm15`.
    pub(super) work: Vec<Vlow>,
    /// Block maximum reduction, then even-rank `K`/`V` row.
    /// Low zone: `hmax` is VEX.
    pub(super) blk: Vlow,
    /// Odd-rank `K`/`V` line, and third polynomial temporary.
    /// EVEX only: the area is free.
    pub(super) blk2: V,
    /// Current maximum of the online softmax. EVEX only.
    pub(super) maxv: V,
    /// Current sum. Low area: the epilogue reduces it (`hsum`) and divides
    /// it scalar (`vdivss`).
    pub(super) sumv: Vlow,
    /// Output accumulators, contiguous.
    pub(super) acc: Vec<V>,
    /// The eight polynomial constants.
    pub(super) k: ExpConsts,
}

impl FusedRegs {
    pub(super) fn alloc(f: &mut RegFile, spec: &SdpaKernelSpec) -> std::result::Result<Self, Decline> {
        // **Order matters.** The low zone is the scarce resource: a
        // `run` of accumulators requested first would eat it all and
        // leave nothing for horizontal reductions. Anything requiring
        // VEX is therefore served first.
        let work = f.low_run(4, "fused:work pool")?;
        let blk = f.low("fused: block/line pair reduction")?;
        let sumv = f.low("fused:somme courante")?;
        let k = ExpConsts::alloc(f)?;
        Ok(Self {
            work,
            blk,
            sumv,
            k,
            acc: f.run(spec.shape.vecs(), "fused:output accumulators")?,
            maxv: f.any("fused:maximum courant")?,
            blk2: f.any("fused:odd line / exp t3")?,
        })
    }
}

/// `q`=r10, `k`=rsi, `v`=r8, `out`=rdx, `n`=r9, scale in the scratchpad.
pub(super) fn emit_fused(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    r: &FusedRegs,
) -> std::result::Result<(), IcedError> {
    let nv = spec.shape.vecs();
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    let maxv = r.maxv.idx();
    let sumv = r.sumv.idx();
    let acc = |c: usize| r.acc[c].idx();
    // K/V row of rank `c`: even in `blk`, odd in `blk2`.
    let kv = |c: usize| -> V {
        if c.is_multiple_of(2) {
            r.blk.into()
        } else {
            r.blk2
        }
    };
    let w = |i: usize| r.work[i];
    let scale_mem = dword_ptr(rsp + HOME);

    let mut block_top = asm.create_label();
    let mut empty = asm.create_label();

    for c in 0..nv {
        asm.vxorps(ZMM[acc(c)], ZMM[acc(c)], ZMM[acc(c)])?;
    }
    bcast_imm(asm, r.maxv, f32::NEG_INFINITY.to_bits())?;
    asm.vxorps(ZMM[sumv], ZMM[sumv], ZMM[sumv])?;
    // Empty context: the output is null, and above all we do not divide by
    // a null sum (1/0 then 0·inf would give NaNs all over the layer).
    asm.test(r9, r9)?;
    asm.jz(empty)?;
    // rcx = number of 16-blocks, r11 = remaining positions of the current block.
    asm.mov(rcx, r9)?;
    asm.add(rcx, 15i32)?;
    asm.shr(rcx, 4u32)?;
    asm.mov(r11, r9)?;

    asm.set_label(&mut block_top)?;
    // ── 16 scores in the aligned draft, `-inf` for positions
    //    absent from the last block (same neutrality as QK padding).
    {
        let mut pos_top = asm.create_label();
        let mut pos_done = asm.create_label();
        let mut clamped = asm.create_label();
        bcast_imm(asm, w(0).into(), f32::NEG_INFINITY.to_bits())?;
        asm.vmovaps(zmmword_ptr(rsp + EXP_BUF), ZMM[w(0).idx()])?;
        asm.mov(rax, r11)?;
        asm.cmp(rax, 16i32)?;
        asm.jbe(clamped)?;
        asm.mov(rax, 16u64)?;
        asm.set_label(&mut clamped)?;
        // rax = actual positions of this block (1..=16)
        asm.lea(rdi, qword_ptr(rsp + EXP_BUF))?;
        asm.test(rax, rax)?;
        asm.jz(pos_done)?;
        asm.set_label(&mut pos_top)?;
        let n32 = nv.div_ceil(2);
        let n_acc = 4.min(n32);
        for a in 0..n_acc {
            asm.vxorps(ZMM[w(a).idx()], ZMM[w(a).idx()], ZMM[w(a).idx()])?;
        }
        if spec.pf_dist > 0 {
            asm.prefetcht0(byte_ptr(rsi + row_bytes * spec.pf_dist as i32))?;
            asm.prefetcht0(byte_ptr(r8 + row_bytes * spec.pf_dist as i32))?;
        }
        for c in 0..n32 {
            let t = kv(c).idx();
            asm.vmovdqu16(ZMM[t], zmmword_ptr(rsi + (c * 64) as i32))?;
            asm.vdpbf16ps(
                ZMM[w(c % n_acc).idx()],
                ZMM[t],
                zmmword_ptr(r10 + (c * 64) as i32),
            )?;
        }
        let w0 = w(0).idx();
        if n_acc == 4 {
            asm.vaddps(ZMM[w0], ZMM[w0], ZMM[w(1).idx()])?;
            asm.vaddps(ZMM[w(2).idx()], ZMM[w(2).idx()], ZMM[w(3).idx()])?;
            asm.vaddps(ZMM[w0], ZMM[w0], ZMM[w(2).idx()])?;
        } else {
            for a in 1..n_acc {
                asm.vaddps(ZMM[w0], ZMM[w0], ZMM[w(a).idx()])?;
            }
        }
        hsum(asm, w(0), r.blk)?;
        asm.vmulss(XMM[w(0).x()], XMM[w(0).x()], scale_mem)?;
        asm.vmovss(dword_ptr(rdi), XMM[w(0).x()])?;
        asm.add(rdi, 4)?;
        asm.add(rsi, row_bytes)?;
        asm.dec(rax)?;
        asm.jnz(pos_top)?;
        asm.set_label(&mut pos_done)?;
    }
    // ── maximum of the block, merged with the current maximum ───────────────────
    let (w0, w1) = (w(0).idx(), w(1).idx());
    let blk = r.blk.idx();
    asm.vmovaps(ZMM[w0], zmmword_ptr(rsp + EXP_BUF))?;
    asm.vmaxps(ZMM[blk], ZMM[w0], ZMM[maxv])?;
    hmax(asm, r.blk, w(1))?;
    asm.vbroadcastss(ZMM[blk], XMM[r.blk.x()])?;
    // correction = exp(m_ancien − m_nouveau) ; the first block starts from
    // −inf, so correction = exp(−inf) = 0 : the empty accumulator remains empty.
    asm.vsubps(ZMM[w1], ZMM[maxv], ZMM[blk])?;
    exp_inline(asm, &r.k, w(1).into(), w(2).into(), w(3).into(), r.blk2)?;
    asm.vmovaps(ZMM[maxv], ZMM[blk])?;
    // p = exp(scores − m_nouveau)
    asm.vsubps(ZMM[w0], ZMM[w0], ZMM[maxv])?;
    exp_inline(asm, &r.k, w(0).into(), w(2).into(), w(3).into(), r.blk.into())?;
    // current sum rescaled then increased
    asm.vmulps(ZMM[sumv], ZMM[sumv], ZMM[w1])?;
    asm.vaddps(ZMM[sumv], ZMM[sumv], ZMM[w0])?;
    // rescaled accumulators
    for c in 0..nv {
        asm.vmulps(ZMM[acc(c)], ZMM[acc(c)], ZMM[w1])?;
    }
    asm.vmovaps(zmmword_ptr(rsp + EXP_BUF), ZMM[w0])?;
    // ── accumulation of the 16 lines of V ───────────────────────────────
    {
        let mut sv_top = asm.create_label();
        let mut sv_done = asm.create_label();
        let mut small = asm.create_label();
        asm.mov(rax, r11)?;
        asm.cmp(rax, 16i32)?;
        asm.jbe(small)?;
        asm.mov(rax, 16u64)?;
        asm.set_label(&mut small)?;
        asm.lea(rdi, qword_ptr(rsp + EXP_BUF))?;
        asm.test(rax, rax)?;
        asm.jz(sv_done)?;
        asm.set_label(&mut sv_top)?;
        asm.vbroadcastss(ZMM[w(2).idx()], dword_ptr(rdi))?;
        for c in 0..nv {
            let t = kv(c);
            load_bf16_f32(asm, t, r8, (c * 32) as i32)?;
            asm.vfmadd231ps(ZMM[acc(c)], ZMM[t.idx()], ZMM[w(2).idx()])?;
        }
        asm.add(rdi, 4)?;
        asm.add(r8, row_bytes)?;
        asm.dec(rax)?;
        asm.jnz(sv_top)?;
        asm.set_label(&mut sv_done)?;
    }
    asm.sub(r11, 16i32)?;
    asm.dec(rcx)?;
    asm.jnz(block_top)?;

    // ── normalization and write ─────────────────────────────────────
    hsum(asm, r.sumv, w(1))?;
    bcast_imm(asm, w(2).into(), 1.0f32.to_bits())?;
    asm.vdivss(XMM[r.sumv.x()], XMM[w(2).x()], XMM[r.sumv.x()])?;
    asm.vbroadcastss(ZMM[sumv], XMM[r.sumv.x()])?;
    for c in 0..nv {
        asm.vmulps(ZMM[acc(c)], ZMM[acc(c)], ZMM[sumv])?;
    }

    // The empty context joins here: accumulators remained zero, no
    // division has taken place.
    asm.set_label(&mut empty)?;
    // Phase (4) of the work pool: the bf16 conversion. Low zone imposed
    // by the `vmovups ymm` emitted in VEX — it is the type that carries it.
    for c in 0..nv {
        let t = w(c % 4);
        asm.vcvtneps2bf16(YMM[t.y()], ZMM[acc(c)])?;
        asm.vmovups(ymmword_ptr(rdx + (c * 32) as i32), YMM[t.y()])?;
    }
    Ok(())
}

