//! SDPA, AVX-512: the prefill block — `rows` queries of one head against the
//! same keys and values, the QK grouped over 16 positions with its transposed
//! reduction, the softmax re-emitted row by row, the SV over the block.

#[allow(unused_imports)]
use super::*;

// ────────────── prefill: a block of `rows` requests ──────────────────

/// The row bases of the scores buffer, held in integer registers.
///
/// Eight integer registers pinned down is expensive — and that is the price of a
/// score broadcast in a single instruction (`vbroadcastss zmm, [base + rbx]`)
/// instead of two (reload the base from a table, then broadcast).
/// In the SV pass this pair executes `rows · (past + rows)` times per
/// head: that is where the cost of one extra `mov` is decided.
///
/// This is also what limits [`SDPA_PREFILL_MAX_ROWS`] to 8.
pub(super) const ROW_BASES: [AsmRegister64; SDPA_PREFILL_MAX_ROWS] =
    [rcx, rdx, r8, r9, r11, r12, r13, r14];

/// Draft area **specific to prefill**, after [`SPILL`].
///
/// The three decode zones keep their offsets: [`exp_call`] writes
/// hard-coded to `rsp + EXP_BUF` and `rsp + SPILL`, and this kernel calls it like the
/// others. We append behind, we move nothing.
pub(super) const PF_EXTRA: i32 = FRAME_BYTES;
/// Prefill blob frame: the three decode zones, plus its own.
pub(super) const PF_FRAME_BYTES: i32 = FRAME_BYTES + SCRATCH_BYTES;

// Home registration of prefill arguments, in the [`HOME`] area (64 bytes,
// eight slots) then [`PF_EXTRA`].
pub(super) const PF_Q: i32 = HOME;
pub(super) const PF_K: i32 = HOME + 8;
pub(super) const PF_V: i32 = HOME + 16;
pub(super) const PF_OUT: i32 = HOME + 24;
pub(super) const PF_PAST: i32 = HOME + 32;
/// `N = past + rows` : the positions visible by the **last** line of the
/// block, hence the width of the calculated rectangle.
pub(super) const PF_N: i32 = HOME + 40;
/// Stride of a scores line, in bytes (`align16(N) · 4`).
pub(super) const PF_ROWB: i32 = HOME + 48;
pub(super) const PF_SCALE: i32 = HOME + 56;
pub(super) const PF_SINK: i32 = HOME + 60;
pub(super) const PF_SCORES: i32 = PF_EXTRA;

/// Registers of the QK pass **in block**.
pub(super) struct QkBlockRegs {
    /// The **entire** `K[j]` row, `n32` registers. This is the only shape
    /// difference with decode, and it is the whole kernel: it is loaded once
    /// then consumed by the `rows` requests, instead of being
    /// reloaded for each one.
    pub(super) k: Vec<V>,
    /// Accumulation chains for a **single** query — same number and same reduction tree
    /// as `QkRegs`, hence the equality of bits.
    pub(super) acc: Vec<Vlow>,
    /// Horizontal reduction temporary (VEX → low zone).
    pub(super) tmp: Vlow,
    /// Scale `1/√head_dim`.
    pub(super) scale: Vlow,
}

impl QkBlockRegs {
    /// `chains` = number of accumulation chains requested. `4` reproduces
    /// the decode kernel; `1` serves the **rest** of the 16-position kernel,
    /// which has only one chain and must keep the same summation order across
    /// all its positions, full group or not (cf. `emit_qk_group`).
    pub(super) fn alloc(
        f: &mut RegFile,
        spec: &SdpaPrefillSpec,
        chains: usize,
    ) -> std::result::Result<Self, Decline> {
        let n32 = spec.shape.vecs().div_ceil(2);
        let n_acc = chains.min(n32);
        // The low area first: the K lines do not need them.
        let acc = f.low_run(n_acc, "qk block:strings")?;
        let tmp = f.low("qk block: temporary reduction")?;
        let scale = f.low("qk block:scale")?;
        let k = (0..n32)
            .map(|_| f.any("qk line block of resident K"))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(Self {
            k,
            acc,
            tmp,
            scale,
        })
    }
    pub(super) fn free(self, f: &mut RegFile) {
        for r in self.k {
            f.free(r);
        }
        for r in self.acc {
            f.free_low(r);
        }
        f.free_low(self.tmp);
        f.free_low(self.scale);
    }
}

/// `rdi` = base of `q`, `rsi` = current K row, `rbx` = current
/// column in bytes, `r10` = remaining positions, `bases[i]` = start of
/// row `i` of the scores buffer.
pub(super) fn emit_qk_block(
    asm: &mut CodeAssembler,
    spec: &SdpaPrefillSpec,
    r: &QkBlockRegs,
    bases: &[AsmRegister64],
) -> std::result::Result<(), IcedError> {
    let n32 = spec.shape.vecs().div_ceil(2);
    let n_acc = r.acc.len();
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    let q_row = (spec.row_stride * 2) as i32;
    let a0 = r.acc[0].idx();

    let mut loop_top = asm.create_label();
    // `N = past + rows ≥ 1` : there is no empty block to keep.
    asm.set_label(&mut loop_top)?;
    if spec.pf_dist > 0 {
        asm.prefetcht0(byte_ptr(rsi + row_bytes * spec.pf_dist as i32))?;
    }
    for (c, kr) in r.k.iter().enumerate() {
        asm.vmovdqu16(ZMM[kr.idx()], zmmword_ptr(rsi + (c * 64) as i32))?;
    }
    for (i, &base) in bases.iter().enumerate() {
        for a in &r.acc {
            asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
        }
        for c in 0..n32 {
            asm.vdpbf16ps(
                ZMM[r.acc[c % n_acc].idx()],
                ZMM[r.k[c].idx()],
                zmmword_ptr(rdi + (i as i32) * q_row + (c * 64) as i32),
            )?;
        }
        // **The same tree as `emit_qk`**, verbatim. Changing it here
        // — even for something "equivalent" — would break bit-for-bit equality
        // with the decode, which is the safety net of this kernel.
        if n_acc == 4 {
            asm.vaddps(ZMM[a0], ZMM[a0], ZMM[r.acc[1].idx()])?;
            asm.vaddps(ZMM[r.acc[2].idx()], ZMM[r.acc[2].idx()], ZMM[r.acc[3].idx()])?;
            asm.vaddps(ZMM[a0], ZMM[a0], ZMM[r.acc[2].idx()])?;
        } else {
            for a in &r.acc[1..] {
                asm.vaddps(ZMM[a0], ZMM[a0], ZMM[a.idx()])?;
            }
        }
        hsum(asm, r.acc[0], r.tmp)?;
        asm.vmulss(XMM[a0], XMM[a0], XMM[r.scale.x()])?;
        asm.vmovss(dword_ptr(base + rbx), XMM[a0])?;
    }
    asm.add(rsi, row_bytes)?;
    asm.add(rbx, 4i32)?;
    asm.dec(r10)?;
    asm.jnz(loop_top)?;
    Ok(())
}

// ─────────────── QK at 16 positions: the transposed reduction ────────

/// Registers for the QK pass **at 16 positions**.
///
/// What this split buys, and what it costs, per (request,
/// position) pair at `head_dim = 128` :
///
/// | | line by line | 16 positions |
/// |---|---|---|
/// | reset | 4 (one per sequence) | **1** |
/// | `vdpbf16ps` | 4 | 4 |
/// | sequence reduction | 3 | **0** |
/// | horizontal reduction | 6 (`hsum` scalar) | **2,81** (45 for 16) |
/// | scale + store | 2 | **0,125** |
/// | load of `K` / `q` | 0,5 | **0,25** |
/// | **total** | **19,5** | **8,2** |
///
/// The horizontal reduction is no longer paid per score but per group:
/// it is this, and not the dot product, that dominated the QK.
pub(super) struct QkGroupRegs {
    /// An accumulator per group position — `g` in total, 8 or 16. This
    /// prevents keeping the four kernel chains line by line (16 × 4 = 64
    /// registers) and **this is the only summation order change** of this
    /// kernel (see `SdpaPrefillSpec::qk_group`).
    pub(super) acc: Vec<V>,
    /// The `q` row of the current query, `n32` registers. The role of
    /// `K` and that of `q` are **exchanged** compared to the row-by-row
    /// kernel: here `K` is the memory operand of `vdpbf16ps`, which
    /// removes its explicit load. The 16 rows of `K` in a group
    /// make 4 KiB at `head_dim` 128: they stay in L1 for the `rows`
    /// queries.
    pub(super) q: Vec<V>,
    /// Reduced transpose temporary.
    pub(super) tmp: V,
    /// Scale `1/√head_dim`, **broadcast**: a single `vmulps` serves all 16 scores.
    pub(super) scale: V,
    /// Resident `K` segment, **present only if queries are
    /// paired** (`qk_pair > 1`). It is the one that replaces `p` loads with
    /// one: loaded once, read by the `p` `vdpbf16ps` of the same
    /// (position, segment).
    pub(super) kt: Option<V>,
    /// Paired requests per load — `1` = history kernel.
    pub(super) pair: usize,
}

impl QkGroupRegs {
    /// `n32` = blocks of 32 bf16 from a row, `head_dim / 32` rounded up
    /// to the nearest integer; `g` = positions per group; `p` = paired queries per
    /// `K` load. Parameters rather than the spec: both
    /// kernels — prefill block and decode — allocate exactly the same
    /// roles.
    ///
    /// At `p = 1` the allocation is **exactly** the same as before, `kt`
    /// included (absent): the delivered blob does not move a single byte.
    pub(super) fn alloc(
        f: &mut RegFile,
        n32: usize,
        g: usize,
        p: usize,
    ) -> std::result::Result<Self, Decline> {
        Ok(Self {
            acc: f.run(p * g, "qk:accumulators, one per (matched query, position)")?,
            q: f.run(p * n32, "qk: resident q lines")?,
            tmp: f.any("qk: reduction temporary")?,
            scale: f.any("qk: broadcast scale")?,
            kt: match p {
                1 => None,
                _ => Some(f.any("qk:shared K segment")?),
            },
            pair: p,
        })
    }
    pub(super) fn free(self, f: &mut RegFile) {
        for r in self.acc.into_iter().chain(self.q) {
            f.free(r);
        }
        f.free(self.tmp);
        f.free(self.scale);
        if let Some(kt) = self.kt {
            f.free(kt);
        }
    }
    /// The `g` accumulators of request `u` in the batch.
    pub(super) fn acc_of(&self, u: usize) -> &[V] {
        let g = self.acc.len() / self.pair;
        &self.acc[u * g..(u + 1) * g]
    }
    /// The `n32` registers of `q` of the query `u` of the packet.
    pub(super) fn q_of(&self, u: usize) -> &[V] {
        let n32 = self.q.len() / self.pair;
        &self.q[u * n32..(u + 1) * n32]
    }
}

/// Position, within the group, that the accumulator `m` accumulates.
///
/// The transpose reduction does not output scores in order. Instead
/// of correcting it with an additional `vpermps` **per group**, we distribute
/// positions at emission: it costs nothing, the displacement of `K`
/// being an immediate.
///
/// - **at 16**, the accumulator `m` score lands in lane
///   `4·(m % 4) + m / 4`, the transpose of a 4×4 matrix — its own
///   inverse, so the same formula works in both directions;
/// - **at 8**, the last level pairs with itself, the output scores are
///   **duplicated** and the `vcompressps` `0x3333` keeps only the lanes
///   `{0,1,4,5,8,9,12,13}`. They carry, in order, the accumulators
///   `0,4,1,5,2,6,3,7`: accumulator `m` must serve position
///   `2m` if it is in the first half, `2(m−4)+1` otherwise.
pub(super) const fn qk_position(m: usize, g: usize) -> usize {
    if g == 8 {
        if m < 4 {
            2 * m
        } else {
            2 * (m - 4) + 1
        }
    } else {
        4 * (m % 4) + m / 4
    }
}

/// Reduces `acc` into a single register containing the 16 scores, **in the same
/// summation order as [`hsum`]**.
///
/// This is the technical point of the kernel. `hsum` reduces a zmm by the tree
/// `(l, l+8) → (l, l+4) → (l0+l1, l2+l3) → total` ; the four levels
/// below apply **exactly the same pairing** to 16 vectors at
/// a time, two by two. Each output lane therefore only mixes
/// lanes from a single accumulator, in the same order as today: the
/// horizontal sum contributes **no** numerical drift.
///
/// The last two levels cannot be done with `vhaddps` as
/// `hsum` uses: **`HADDPS` has no 512-bit form** — it stops at
/// AVX (VEX, ymm). Two `vshufps` plus a `vaddps` reproduce it
/// identically, costing 3 instructions per pair instead of one.
///
/// **The four levels are incurred regardless of the number**
/// **of accumulators**: they reduce the 16 lanes of an accumulator, not the
/// number of registers. At 8, the first three levels exhaust the
/// registers and the fourth **pairs with itself** — same two
/// `vshufps` and same `vaddps`, so **same bits**, but the result exits
/// duplicated: 8 scores in 16 lanes. It is the store, and only it, that
/// pays this group size.
pub(super) fn qk_reduce(
    asm: &mut CodeAssembler,
    acc: &[V],
    tmp: V,
) -> std::result::Result<usize, IcedError> {
    // `lo` takes the "low" half of the pairing, `hi` the other.
    // Levels 0-1: pairing by 128-bit lanes (`vshuff64x2`).
    // Levels 2-3: within a 128-bit lane (`vshufps`).
    const IMM: [(u32, u32); 4] = [(0x44, 0xEE), (0x88, 0xDD), (0x88, 0xDD), (0x88, 0xDD)];
    let mut cur: Vec<usize> = acc.iter().map(|a| a.idx()).collect();
    for (level, &(lo, hi)) in IMM.iter().enumerate() {
        let mut next = Vec::with_capacity(cur.len().div_ceil(2));
        // `cur.len() == 1`: no more neighbor, the register pairs with
        // itself. The result is the same tree, emitted twice.
        for p in 0..cur.len().div_ceil(2) {
            let (a, b) = (cur[2 * p], cur[(2 * p + 1).min(cur.len() - 1)]);
            if level < 2 {
                asm.vshuff64x2(ZMM[tmp.idx()], ZMM[a], ZMM[b], hi)?;
                asm.vshuff64x2(ZMM[a], ZMM[a], ZMM[b], lo)?;
            } else {
                asm.vshufps(ZMM[tmp.idx()], ZMM[a], ZMM[b], hi)?;
                asm.vshufps(ZMM[a], ZMM[a], ZMM[b], lo)?;
            }
            asm.vaddps(ZMM[a], ZMM[a], ZMM[tmp.idx()])?;
            next.push(a);
        }
        cur = next;
    }
    Ok(cur[0])
}

/// Writes the `g` scores of the group to the desired address.
///
/// At 16, a `vmovups`: the scores occupy the 16 lanes, in order.
/// At 8, they exit **duplicated** (`qk_reduce`) and need to be gathered.
/// Three ways were weighed, and it's the instruction count that
/// decides, not the taste:
///
/// | | cost per group | registers |
/// |---|---|---|
/// | masked `vcompressps` | **1 instruction** | one mask register |
/// | `vpermps` + `vmovups ymm` | 2 | one zmm of indices, to be materialized |
/// | lane-by-lane extraction | 8 | none |
///
/// The mask `0x3333` is set **once** outside the loop
/// ([`set_qk_mask`]).
pub(super) fn store_scores(
    asm: &mut CodeAssembler,
    g: usize,
    s: usize,
    at: AsmMemoryOperand,
) -> std::result::Result<(), IcedError> {
    if g == 8 {
        asm.vcompressps(at.k1(), ZMM[s])
    } else {
        asm.vmovups(at, ZMM[s])
    }
}

/// `k1 = 0x3333` — the lanes that `qk_reduce` leaves distinct at `g = 8`.
///
/// Use `eax`, which carries nothing from either kernel at this point (the
/// prologue is finished, arguments are in rdi/rsi/rdx/rcx and bases
/// in callee-saved).
pub(super) fn set_qk_mask(asm: &mut CodeAssembler, g: usize) -> std::result::Result<(), IcedError> {
    if g != 8 {
        return Ok(());
    }
    asm.mov(eax, 0x3333u32)?;
    asm.kmovw(k1, eax)
}

/// The QK in groups of `g` positions.
///
/// `rdi` = base of `q`, `rsi` = first row of K of the group, `rbx` =
/// current column in bytes, `r15` = remaining groups (≥ 1 at entry),
/// `bases[i]` = start of row `i` of the scores buffer.
pub(super) fn emit_qk_group(
    asm: &mut CodeAssembler,
    spec: &SdpaPrefillSpec,
    r: &QkGroupRegs,
    bases: &[AsmRegister64],
) -> std::result::Result<(), IcedError> {
    let n32 = spec.shape.vecs().div_ceil(2);
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    let q_row = (spec.row_stride * 2) as i32;
    // `acc` carries `p · g` accumulators from the pairing: the group size
    // is what remains per request, and it is what indexes
    // `qk_position`, the mask, and the store.
    let g = r.acc.len() / r.pair;

    set_qk_mask(asm, g)?;
    let mut loop_top = asm.create_label();
    asm.set_label(&mut loop_top)?;
    // One prefetched K line per position, like the line-by-line
    // kernel — but emitted once per group and not per request.
    if spec.pf_dist > 0 {
        for m in 0..g as i32 {
            asm.prefetcht0(byte_ptr(
                rsi + row_bytes * (g as i32 * spec.pf_dist as i32 + m),
            ))?;
        }
    }
    // Requests are processed in batches of `p`: this is the batch that
    // shares each `K` load. At `p = 1` — the shipped case — we fall back
    // to the historical loop, `K` as a memory operand, byte for byte.
    let p = r.pair;
    for chunk in 0..bases.len().div_ceil(p) {
        let start = chunk * p;
        let take = p.min(bases.len() - start);
        for u in 0..take {
            for (c, qr) in r.q_of(u).iter().enumerate() {
                asm.vmovdqu16(
                    ZMM[qr.idx()],
                    zmmword_ptr(rdi + ((start + u) as i32) * q_row + (c * 64) as i32),
                )?;
            }
            for a in r.acc_of(u) {
                asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
            }
        }
        for m in 0..g {
            let j = qk_position(m, g) as i32;
            for c in 0..n32 {
                let at = zmmword_ptr(rsi + j * row_bytes + (c * 64) as i32);
                match r.kt {
                    // One load, `take` FMA : the swap that drains the load
                    // queue. The accumulation order of each request is
                    // unchanged — it is the provenance of the operand that
                    // changes, not the sequence of additions.
                    //
                    // `take > 1` is the condition, not `kt.is_some()`:
                    // when `p` does not divide `rows`, the last packet
                    // carries only one request, and loading `K` into a register for
                    // **a single** `vdpbf16ps` amortizes nothing — it adds an
                    // instruction and a dependency where the memory operand
                    // suffices. Measured: an odd `rows` paid +2.5 to
                    // +4.3 points (17th session, 8 out of 8 cases, control `p = 1`).
                    Some(kt) if take > 1 => {
                        asm.vmovdqu16(ZMM[kt.idx()], at)?;
                        for u in 0..take {
                            asm.vdpbf16ps(
                                ZMM[r.acc_of(u)[m].idx()],
                                ZMM[r.q_of(u)[c].idx()],
                                ZMM[kt.idx()],
                            )?;
                        }
                    }
                    // The query alone — unmatched packet, or kernel with
                    // `p = 1`. `acc_of(0)`/`q_of(0)` make `acc`/`q`
                    // integers when `p = 1`: the blob delivered does not move
                    // a byte. And the arithmetic is identical to the
                    // branch above, memory operand or register —
                    // no bit changes.
                    _ => {
                        asm.vdpbf16ps(
                            ZMM[r.acc_of(0)[m].idx()],
                            ZMM[r.q_of(0)[c].idx()],
                            at,
                        )?;
                    }
                }
            }
        }
        for u in 0..take {
            let s = qk_reduce(asm, r.acc_of(u), r.tmp)?;
            asm.vmulps(ZMM[s], ZMM[s], ZMM[r.scale.idx()])?;
            store_scores(asm, g, s, zmmword_ptr(bases[start + u] + rbx))?;
        }
    }
    asm.add(rsi, row_bytes * g as i32)?;
    asm.add(rbx, 4 * g as i32)?;
    asm.dec(r15)?;
    asm.jnz(loop_top)?;
    Ok(())
}

/// The same QK at 16 positions, **on the decode side**: a request, `n` positions.
///
/// This is the most favorable case of the two. The prefill block reloads `q`
/// on each row, once per group and per request; here there is only one
/// request, so `q` is loaded **once for the entire context** and
/// the group loop contains only the dot product, the
/// reduction and a store.
///
/// What this kernel must return, before any speed, is **the same
/// bits as the full prefill group**: same unique string, same
/// `qk_reduce`, same broadcast scale. The rest (`n % g`) comes back to
/// `emit_qk` with **one** string, otherwise a position falling into a
/// full group on one side and the rest on the other would suffice to
/// break the equality — by the mere effect of the splitting.
///
/// `rdi` = q, `rsi` = first row of K of the group, `rdx` = scores,
/// `r11` = remaining groups (≥ 1 at input).
pub(super) fn emit_qk_group_decode(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    r: &QkGroupRegs,
) -> std::result::Result<(), IcedError> {
    let n32 = spec.shape.vecs().div_ceil(2);
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    // The decode does not pair: `pair` is 1, but writing it this way keeps
    // the two kernels on the same rule.
    let g = r.acc.len() / r.pair;

    // Outside the loop: the query is the same for all positions.
    for (c, qr) in r.q.iter().enumerate() {
        asm.vmovdqu16(ZMM[qr.idx()], zmmword_ptr(rdi + (c * 64) as i32))?;
    }
    set_qk_mask(asm, g)?;
    let mut loop_top = asm.create_label();
    asm.set_label(&mut loop_top)?;
    if spec.pf_dist > 0 {
        for m in 0..g as i32 {
            asm.prefetcht0(byte_ptr(
                rsi + row_bytes * (g as i32 * spec.pf_dist as i32 + m),
            ))?;
        }
    }
    for a in &r.acc {
        asm.vxorps(ZMM[a.idx()], ZMM[a.idx()], ZMM[a.idx()])?;
    }
    for (m, a) in r.acc.iter().enumerate() {
        let j = qk_position(m, g) as i32;
        for c in 0..n32 {
            asm.vdpbf16ps(
                ZMM[a.idx()],
                ZMM[r.q[c].idx()],
                zmmword_ptr(rsi + j * row_bytes + (c * 64) as i32),
            )?;
        }
    }
    let s = qk_reduce(asm, &r.acc, r.tmp)?;
    asm.vmulps(ZMM[s], ZMM[s], ZMM[r.scale.idx()])?;
    store_scores(asm, g, s, zmmword_ptr(rdx))?;
    asm.add(rsi, row_bytes * g as i32)?;
    asm.add(rdx, 4 * g as i32)?;
    asm.dec(r11)?;
    asm.jnz(loop_top)?;
    Ok(())
}

/// Sort the scale of `xmm0` — **before any other use of the low area**,
/// and this is an order, not a preference: in exp inline, `RegFile::low`
/// returns `zmm0` to the sink, whose copy would overwrite the scale.
///
/// Two destinations, because the two paths do not have the same budget.
/// The row-by-row QK keeps it in its scale register — a
/// `vmovaps`, the original code, unchanged down to the byte. The one at 16
/// positions no longer has a register to give it (sixteen contiguous accumulators,
/// `n32` blocks of `q`) and **houses** it.
pub(super) fn stash_scale(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    f: &mut RegFile,
) -> std::result::Result<Option<QkRegs>, EmitErr> {
    if spec.qk_group != 0 {
        asm.vmovss(dword_ptr(rsp + QK_SCALE), XMM[0])?;
        return Ok(None);
    }
    let qk = QkRegs::alloc(f, spec, 4)?;
    asm.vmovaps(XMM[qk.scale.x()], XMM[0])?;
    Ok(Some(qk))
}

/// The entire decode QK pass, in either summation order:
/// groups of 16 then remainder, or line by line.
///
/// Input: `rdi` = q, `rsi` = k, `rdx` = scores, `rcx` = n, `r10` = n
/// (kept for padding), and the scale where [`stash_scale`] put it.
pub(super) fn emit_qk_pass(
    asm: &mut CodeAssembler,
    spec: &SdpaKernelSpec,
    f: &mut RegFile,
    stashed: Option<QkRegs>,
) -> std::result::Result<(), EmitErr> {
    if let Some(qk) = stashed {
        emit_qk(asm, spec, &qk)?;
        qk.free(f);
        return Ok(());
    }
    let g = spec.qk_group as usize;
    // Decode does not pair anything: a single request, and `q` is already
    // resident for the entire context.
    let regs = QkGroupRegs::alloc(f, spec.shape.vecs().div_ceil(2), g, 1)?;
    asm.vbroadcastss(ZMM[regs.scale.idx()], dword_ptr(rsp + QK_SCALE))?;
    asm.mov(r11, rcx)?;
    asm.shr(r11, g.trailing_zeros())?; // groups full_ones
    asm.and(rcx, g as i32 - 1)?; // remainder, served by emit_qk just after
    let mut no_group = asm.create_label();
    asm.test(r11, r11)?;
    asm.jz(no_group)?;
    emit_qk_group_decode(asm, spec, &regs)?;
    asm.set_label(&mut no_group)?;
    regs.free(f);
    // `rsi` and `rdx` have been advanced by the group loop; `rcx` holds
    // the remainder, which `emit_qk` already knows is zero (it keeps its loop), and
    // `r10` still holds `n`, from which it derives the padding `-inf`.
    let qk = QkRegs::alloc(f, spec, 1)?;
    asm.vmovss(XMM[qk.scale.x()], dword_ptr(rsp + QK_SCALE))?;
    emit_qk(asm, spec, &qk)?;
    qk.free(f);
    Ok(())
}

/// Registers for the SV pass **in blocks**.
pub(super) struct SvBlockRegs {
    /// Output accumulators: `rows × nb` blocks of 16 dimensions.
    pub(super) acc: Vec<V>,
    /// Lines of `V` from the current dimension group, and destination of the
    /// bf16 conversion epilogue — **low zone** for the same reason
    /// as in `SvRegs`: the epilogue's `vmovups ymm` is emitted in VEX.
    pub(super) tmp: Vec<Vlow>,
    /// Current score, broadcast.
    pub(super) score: V,
}

impl SvBlockRegs {
    pub(super) fn alloc(f: &mut RegFile, spec: &SdpaPrefillSpec) -> std::result::Result<Self, Decline> {
        let nb = spec.sv_dim_blocks();
        let tmp = f.low_run(nb, "sv bloc:lines of V / bf16 conversion")?;
        Ok(Self {
            tmp,
            score: f.any("sv bloc:score broadcasted")?,
            acc: f.run(spec.rows * nb, "sv block:output accumulators")?,
        })
    }
}

/// `rsi` = current row of `V`, `rbx` = column in bytes, `r10` =
/// remaining positions, `rdi` = base of `out`, `bases[i]` = row `i` of the
/// scores.
///
/// **Dimension group outside, positions inside.** This is
/// the inversion that makes everything work: for a `(query, dimension)` pair, the
/// positions are still walked in increasing order — hence the summation order
/// of decode — but each row of `V` serves `rows` queries instead
/// of one. `V` is walked **exactly once** across all the
/// groups, in slices of `nb·32` bytes, that is one cache line at
/// `nb = 2`.
pub(super) fn emit_sv_block(
    asm: &mut CodeAssembler,
    spec: &SdpaPrefillSpec,
    r: &SvBlockRegs,
    bases: &[AsmRegister64],
) -> std::result::Result<(), IcedError> {
    let nv = spec.shape.vecs();
    let nb = r.tmp.len();
    let rows = spec.rows;
    let row_bytes = spec.shape.kv_row_bytes() as i32;
    // `out_row_stride`, not `row_stride`: this is the only pass that writes,
    // and on the Qwen3.5 output gate `q` is striped twice as wide
    // as `out` (cf. `SdpaPrefillSpec::out_row_stride`).
    let out_row = (spec.out_row_stride * 2) as i32;
    let acc = |i: usize, c: usize| r.acc[i * nb + c].idx();

    for g in 0..nv / nb {
        let mut loop_top = asm.create_label();
        for i in 0..rows {
            for c in 0..nb {
                asm.vxorps(ZMM[acc(i, c)], ZMM[acc(i, c)], ZMM[acc(i, c)])?;
            }
        }
        asm.mov(rsi, qword_ptr(rsp + PF_V))?;
        asm.mov(r10, qword_ptr(rsp + PF_N))?;
        asm.xor(rbx, rbx)?;
        asm.set_label(&mut loop_top)?;
        if spec.pf_dist > 0 {
            asm.prefetcht0(byte_ptr(rsi + row_bytes * spec.pf_dist as i32))?;
        }
        for c in 0..nb {
            load_bf16_f32(asm, r.tmp[c].into(), rsi, ((g * nb + c) * 32) as i32)?;
        }
        for (i, &base) in bases.iter().enumerate() {
            asm.vbroadcastss(ZMM[r.score.idx()], dword_ptr(base + rbx))?;
            for c in 0..nb {
                asm.vfmadd231ps(ZMM[acc(i, c)], ZMM[r.tmp[c].idx()], ZMM[r.score.idx()])?;
            }
        }
        asm.add(rsi, row_bytes)?;
        asm.add(rbx, 4i32)?;
        asm.dec(r10)?;
        asm.jnz(loop_top)?;
        // Group epilogue: the accumulation was in f32 in the
        // registers, what goes to memory is bf16.
        asm.mov(rdi, qword_ptr(rsp + PF_OUT))?;
        for i in 0..rows {
            for c in 0..nb {
                let t = r.tmp[c];
                asm.vcvtneps2bf16(YMM[t.y()], ZMM[acc(i, c)])?;
                asm.vmovups(
                    ymmword_ptr(rdi + (i as i32) * out_row + ((g * nb + c) * 32) as i32),
                    YMM[t.y()],
                )?;
            }
        }
    }
    Ok(())
}

/// Prefill prologue: like [`prologue`], plus the five callee-saved integer registers
/// used by the kernel (`rbx` for the current column, `r12-r15` for the row bases and
/// the softmax loop index — which must survive the `call` of `ExpStrategy::Call`).
pub(super) fn prologue_prefill(asm: &mut CodeAssembler) -> std::result::Result<(), IcedError> {
    asm.push(rbp)?;
    asm.mov(rbp, rsp)?;
    asm.push(rbx)?;
    asm.push(r12)?;
    asm.push(r13)?;
    asm.push(r14)?;
    asm.push(r15)?;
    asm.and(rsp, -64i32)?;
    asm.sub(rsp, PF_FRAME_BYTES)?;
    Ok(())
}

pub(super) fn epilogue_prefill(asm: &mut CodeAssembler) -> std::result::Result<(), IcedError> {
    // The five `push` instructions account for 40 bytes below `rbp`.
    asm.lea(rsp, qword_ptr(rbp - 40))?;
    asm.pop(r15)?;
    asm.pop(r14)?;
    asm.pop(r13)?;
    asm.pop(r12)?;
    asm.pop(rbx)?;
    asm.pop(rbp)?;
    asm.vzeroupper()?;
    asm.ret()?;
    Ok(())
}

/// The prefill blob: QK in block, softmax line by line, SV in block.
pub(super) fn assemble_prefill(
    spec: &SdpaPrefillSpec,
) -> std::result::Result<Vec<u8>, EmitErr> {
    let mut asm = CodeAssembler::new(64)?;
    let mut f = RegFile::new();
    let rows = spec.rows;
    let bases = &ROW_BASES[..rows];
    let needs_constants = spec.exp == ExpStrategy::Inline;

    // ── prologue and homing ─────────────────────────────────────
    // rdi=q, rsi=k, rdx=v, rcx=scores, r8=out, r9=past, xmm0=scale,
    // xmm1=sink.
    prologue_prefill(&mut asm)?;
    asm.mov(qword_ptr(rsp + PF_Q), rdi)?;
    asm.mov(qword_ptr(rsp + PF_K), rsi)?;
    asm.mov(qword_ptr(rsp + PF_V), rdx)?;
    asm.mov(qword_ptr(rsp + PF_SCORES), rcx)?;
    asm.mov(qword_ptr(rsp + PF_OUT), r8)?;
    asm.mov(qword_ptr(rsp + PF_PAST), r9)?;
    asm.vmovss(dword_ptr(rsp + PF_SCALE), XMM[0])?;
    asm.vmovss(dword_ptr(rsp + PF_SINK), XMM[1])?;
    // N = past + rows ; stride of a line = align16(N) · 4 bytes.
    asm.lea(rax, qword_ptr(r9 + rows as i32))?;
    asm.mov(qword_ptr(rsp + PF_N), rax)?;
    asm.add(rax, 15i32)?;
    asm.and(rax, -16i32)?;
    asm.shl(rax, 2u32)?;
    asm.mov(qword_ptr(rsp + PF_ROWB), rax)?;

    // ── QK : the entire rectangle, `rows × N` ──────────────────────────
    // The columns `j > past + i` of row `i` are computed then
    // overwritten — `rows·(rows−1)/2` extra dot products over
    // `rows·N`, i.e. 0.3 % at `rows = 8, past = 1000`. Handling them with a
    // triangular epilogue would cost more code than they cost in
    // cycles.
    set_row_bases(&mut asm, bases)?;
    asm.mov(rdi, qword_ptr(rsp + PF_Q))?;
    asm.mov(rsi, qword_ptr(rsp + PF_K))?;
    asm.mov(r10, qword_ptr(rsp + PF_N))?;
    asm.xor(rbx, rbx)?;
    // The remainder (`N % 16`) falls back to the row-by-row kernel — but with
    // **a single chain**, just like the full groups. This is a condition
    // for bit-exact reproduction and not a style detail: decoding row
    // `i` slices at `n_i = past + i + 1`, the block at
    // `N = past + rows`, so the same position falls into a full group
    // here and into the remainder there. If the two paths did not sum
    // identically, equality would be lost solely due to the slicing.
    let chains = if spec.qk_group != 0 { 1 } else { 4 };
    if spec.qk_group != 0 {
        let g = spec.qk_group as usize;
        let regs =
            QkGroupRegs::alloc(&mut f, spec.shape.vecs().div_ceil(2), g, spec.qk_pair())?;
        asm.vbroadcastss(ZMM[regs.scale.idx()], dword_ptr(rsp + PF_SCALE))?;
        asm.mov(r15, r10)?;
        asm.shr(r15, g.trailing_zeros())?; // groups full_ones
        asm.and(r10, g as i32 - 1)?; // remainder, served lower down
        let mut no_group = asm.create_label();
        asm.test(r15, r15)?;
        asm.jz(no_group)?;
        emit_qk_group(&mut asm, spec, &regs, bases)?;
        asm.set_label(&mut no_group)?;
        regs.free(&mut f);
    }
    let qk = QkBlockRegs::alloc(&mut f, spec, chains)?;
    asm.vmovss(XMM[qk.scale.x()], dword_ptr(rsp + PF_SCALE))?;
    if spec.qk_group != 0 {
        // `r10` already holds the remainder; `rsi` and `rbx` have been advanced by
        // the groups loop. A zero remainder must skip the loop, which
        // is a do-while.
        let mut no_tail = asm.create_label();
        asm.test(r10, r10)?;
        asm.jz(no_tail)?;
        emit_qk_block(&mut asm, spec, &qk, bases)?;
        asm.set_label(&mut no_tail)?;
    } else {
        emit_qk_block(&mut asm, spec, &qk, bases)?;
    }
    qk.free(&mut f);

    // ── padding `-inf`: causality, line by line ────────────────
    // Line `i` sees `n_i = past + i + 1` keys. From `n_i` to the next multiple of
    // 16, `-inf` — **exactly** what the decode kernel writes
    // for the same line, and what makes the three softmax sweeps
    // mask-free.
    asm.mov(rax, qword_ptr(rsp + PF_PAST))?;
    for (i, &base) in bases.iter().enumerate() {
        let mut no_pad = asm.create_label();
        let mut pad_top = asm.create_label();
        asm.lea(rdi, qword_ptr(rax + (i as i32 + 1)))?;
        asm.mov(rsi, rdi)?;
        asm.and(rsi, 15i32)?;
        asm.jz(no_pad)?;
        asm.mov(r10, 16u64)?;
        asm.sub(r10, rsi)?;
        asm.lea(rsi, qword_ptr(base + rdi * 4))?;
        asm.set_label(&mut pad_top)?;
        asm.mov(dword_ptr(rsi), 0xFF80_0000u32)?;
        asm.add(rsi, 4i32)?;
        asm.dec(r10)?;
        asm.jnz(pad_top)?;
        asm.set_label(&mut no_pad)?;
    }

    // ── softmax : the decode kernel, re-emitted line by line ──────────
    // `r13` and `r14` are here the loop state, while they also carry
    // `ROW_BASES[6]` and `[7]`. This is deliberate and safe: the
    // bases are **recalculated** before the SV pass. They are no longer
    // used here.
    let sink = f.low(SINK_ROLE)?;
    let consts = if needs_constants {
        Some(ExpConsts::alloc(&mut f)?)
    } else {
        None
    };
    let sm = SoftmaxRegs::alloc(&mut f, spec.exp)?;
    let e = Exp {
        strategy: spec.exp,
        consts,
        exempt: SoftmaxRegs::EXEMPT,
    };
    asm.vmovss(XMM[sink.x()], dword_ptr(rsp + PF_SINK))?;
    if let Some(k) = consts {
        load_exp_constants(&mut asm, &k)?;
    }
    asm.mov(r13, qword_ptr(rsp + PF_SCORES))?;
    asm.mov(r14, qword_ptr(rsp + PF_ROWB))?;
    asm.xor(r15, r15)?;
    let mut row_top = asm.create_label();
    asm.set_label(&mut row_top)?;
    asm.mov(rdi, r13)?;
    asm.mov(rax, qword_ptr(rsp + PF_PAST))?;
    asm.lea(rax, qword_ptr(rax + r15 + 1))?; // n_i
    asm.add(rax, 15i32)?;
    asm.shr(rax, 4u32)?;
    asm.mov(r11, rax)?; // nb_i, exactly the decode count
    emit_softmax(&mut asm, spec.shape.sink, &f, &sm, sink, &e)?;
    asm.add(r13, r14)?;
    asm.inc(r15)?;
    asm.cmp(r15, rows as i32)?;
    asm.jb(row_top)?;
    sm.free(&mut f);
    f.free_low(sink);
    if let Some(k) = consts {
        for c in [k.neg87, k.log2e, k.c4, k.c3, k.c2, k.c1, k.c0, k.one] {
            f.free(c);
        }
    }

    // ── neutralize the masked columns: **exact zeros** ──────
    // The SV pass sweeps `N` positions for the `rows` rows at once.
    // The positions `j ≥ n_i` of row `i` do not exist for it:
    // we write `+0.0` rather than leaving the softmax residue (which
    // is ~1e−38) or, worse, the raw score never normalized beyond the
    // block of 16 of the row.
    //
    // `+0.0` is not "negligible", it is **exact**:
    // `vfmadd231ps acc, V, 0` yields `acc + 0` = `acc`, bit for bit. This is
    // what allows the SV pass to run without a mask **and** to yield
    // exactly what decode yields, which stops at `n_i`.
    //
    // The number of columns to neutralize is `N − n_i = rows − 1 − i`:
    // an emission constant, hence the full unrolling
    // (`rows·(rows−1)/2` stores, 28 at `rows = 8`).
    asm.mov(rax, qword_ptr(rsp + PF_PAST))?;
    asm.mov(rdi, qword_ptr(rsp + PF_SCORES))?;
    asm.mov(rsi, qword_ptr(rsp + PF_ROWB))?;
    for i in 0..rows {
        for t in 0..rows - 1 - i {
            asm.mov(dword_ptr(rdi + rax * 4 + ((i + 1 + t) * 4) as i32), 0u32)?;
        }
        asm.add(rdi, rsi)?;
    }

    // ── SV en bloc ────────────────────────────────────────────────────
    let sv = SvBlockRegs::alloc(&mut f, spec)?;
    set_row_bases(&mut asm, bases)?;
    emit_sv_block(&mut asm, spec, &sv, bases)?;
    epilogue_prefill(&mut asm)?;
    Ok(asm.assemble(0)?)
}

/// Loads `rows` score row bases from the scratchpad.
///
/// Twice rather than once: the softmax loop clobbers `r13`/`r14`, which
/// are also bases. Recalculating costs `rows` instructions per pass and
/// avoids making a pass depend on what the previous one left behind —
/// it's the same principle as `free()` between passes on the vector register
/// side.
pub(super) fn set_row_bases(
    asm: &mut CodeAssembler,
    bases: &[AsmRegister64],
) -> std::result::Result<(), IcedError> {
    asm.mov(rax, qword_ptr(rsp + PF_ROWB))?;
    asm.mov(bases[0], qword_ptr(rsp + PF_SCORES))?;
    for i in 1..bases.len() {
        asm.lea(bases[i], qword_ptr(bases[i - 1] + rax))?;
    }
    Ok(())
}

