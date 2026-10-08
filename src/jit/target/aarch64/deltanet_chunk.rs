//! The GatedDeltaNet recurrence by chunks of sixteen positions on the matrix
//! engine: the WY form of `chunk_gated_delta_rule` (flash-linear-attention),
//! written against the order of this engine's recurrence (`emit_recurrence.rs`:
//! the state decays before it is read).
//!
//! For one value head and sixteen positions `t` from the state `S₀`
//! (`[hkd][hvd]`), with the decays `a_t`, the gates `β_t`, the cumulative
//! decays `γ_t = a_0·…·a_t` and the ratios `r_tj = a_{j+1}·…·a_t` (`j ≤ t`):
//!
//! ```text
//! A_tj = β_t·r_tj·(k_t·k_j)  (j < t)        T = (I + A)⁻¹   (forward substitution)
//! Δ    = T·diag(β)·V − T·diag(β·γ)·(K·S₀)   the corrected values, 16 × hvd
//! O    = (diag(γ·scale)·Q)·S₀ + M·Δ          M_tj = scale·r_tj·(q_t·k_j)  (j ≤ t)
//! S₁₆  = γ₁₅·S₀ + (diag(r_15,·)·K)ᵀ·Δ
//! ```
//!
//! The row scalings of the large operands are carried by the 16 × 16 ones
//! (`T·diag(β)`, `T·diag(βγ)`) or by the transposed queries, so every large
//! product is one outer-product accumulation into ZA (`fmopa`, `fmops`), f32.
//!
//! A call runs three phases:
//!
//! 1. **Streaming**: the sixteen rows of `K` and of `Q`, sixteen dimensions
//!    at a time, loaded as the rows of two ZA tiles and read back as their
//!    columns — `Kᵀ` and `Qᵀ` into the scratch — while two other tiles
//!    accumulate `KKᵀ` and `QKᵀ`.
//! 2. **NEON**: the decays and gates, `r`, `γ`, `A`, `T` by forward
//!    substitution row by row, `T·diag(β)`, `T·diag(βγ)` and `M` transposed
//!    (their columns are the `zn` of the outer products), `Qᵀ` scaled by
//!    `γ·scale`, `K` scaled by the last row of `r`.
//! 3. **Streaming**, per block of sixteen value columns: `K·S₀` and
//!    `Q̃·S₀` over the key dimension, `Δ`, `O` into the output, then the
//!    block of the state: its rows scaled by `γ₁₅` (SVE `fmul`) into a tile,
//!    the rank-sixteen update added, the tile stored back in place.
//!
//! The bits are not those of the sequential recurrence (other sums, other
//! order); the kernel is held to the bit against a transcription of these
//! operations, and both are measured against the sequential reference.
//!
//! ABI: `(s, out, q, k, v, beta, decay)` at the first position of the chunk,
//! the head's offsets applied: `q`, `k`, `v` step by `row_bytes` (the
//! `qkv` row), `out` by `out_bytes`, `beta` and `decay` by `gate_bytes`.

use crate::core::error::{HerbertError, Result};
use crate::jit::target::aarch64::util::{add_imm, backend_err, Frame};
use a64::*;

/// Positions per chunk: the sixteen rows of a ZA tile.
pub const CHUNK: usize = 16;

/// The geometry of a chunk kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChunkSpec {
    /// Key dimension of a head, a multiple of 16.
    pub hkd: usize,
    /// Value dimension of a head, a multiple of 16.
    pub hvd: usize,
    /// Bytes between two positions of `q`, `k` and `v` (the `qkv` row).
    pub row_bytes: usize,
    /// Bytes between two positions of the output.
    pub out_bytes: usize,
    /// Bytes between two positions of `beta` and `decay`.
    pub gate_bytes: usize,
}

impl ChunkSpec {
    /// Shapes the kernel serves.
    pub fn check(&self) -> std::result::Result<(), crate::jit::ir::Decline> {
        use crate::jit::ir::Decline;
        if self.hkd == 0 || self.hvd == 0 || !self.hkd.is_multiple_of(16) || !self.hvd.is_multiple_of(16) || self.hkd > 512 || self.hvd > 512 {
            return Err(Decline::PartialRowBlock);
        }
        if !crate::jit::target::aarch64::gemm_sme::available() {
            return Err(Decline::TileTooWide);
        }
        Ok(())
    }

    /// The attention scale of the queries, `1/√hkd`.
    pub fn scale(&self) -> f32 {
        1.0 / (self.hkd as f32).sqrt()
    }
}

/// Scratch offsets from `x19` (the stack), in bytes.
struct Layout {
    kt: usize,
    qt: usize,
    ks: usize,
    gkk: usize,
    gqk: usize,
    tm: usize,
    tt: usize,
    tbt: usize,
    tbgt: usize,
    mt: usize,
    r: usize,
    pb: usize,
    /// The rows of `M` before their transpose.
    db: usize,
    /// The decays, the gates and the cumulative decays, 64 bytes apart.
    va: usize,
    total: usize,
}

impl Layout {
    fn of(spec: &ChunkSpec) -> Layout {
        let big = spec.hkd * CHUNK * 4;
        let tile = CHUNK * CHUNK * 4;
        let kt = 0;
        let qt = kt + big;
        let ks = qt + big;
        let gkk = ks + big;
        let gqk = gkk + tile;
        let tm = gqk + tile;
        let tt = tm + tile;
        let tbt = tt + tile;
        let tbgt = tbt + tile;
        let mt = tbgt + tile;
        let r = mt + tile;
        let pb = r + tile;
        let db = pb + tile;
        let va = db + tile;
        // A multiple of 4 KiB: `sub sp, sp, #n, lsl #12` in one word.
        let total = (va + 3 * 64).next_multiple_of(4096);
        Layout { kt, qt, ks, gkk, gqk, tm, tt, tbt, tbgt, mt, r, pb, db, va, total }
    }
}

/// The scratch of one call, in bytes (on the stack).
pub fn scratch_bytes(spec: &ChunkSpec) -> usize {
    Layout::of(spec).total
}

/// A 32-bit slice of tile `tile` for row or column `i` of sixteen, through `w12`.
fn slice(tile: u8, dir: Slice, i: usize) -> ZaS {
    ZaS { tile, dir, index: W12, offset: (i % 4) as u8 }
}

/// `w12` for the group of four slices of `i`, set when `i` starts a group.
fn slice_base(a: &mut Asm, i: usize) {
    if i.is_multiple_of(4) {
        a.mov_imm(W12, i as u64);
    }
}

/// Transposes the 16 × 16 f32 matrix at `x19 + from` into `x19 + to`
/// (rows of 64 bytes), by 4 × 4 blocks. Clobbers `x7`, `x8`, `v0..v7`.
fn transpose16(a: &mut Asm, from: usize, to: usize) {
    for bi in 0..4 {
        for bj in 0..4 {
            add_imm(a, X7, X19, (from + bi * 4 * 64 + bj * 16) as i64, X8);
            for r in 0..4u8 {
                a.ldr(q(r), mem(X7, i64::from(r) * 64));
            }
            a.trn1(v(4).s4(), v(0).s4(), v(1).s4());
            a.trn2(v(5).s4(), v(0).s4(), v(1).s4());
            a.trn1(v(6).s4(), v(2).s4(), v(3).s4());
            a.trn2(v(7).s4(), v(2).s4(), v(3).s4());
            a.trn1(v(0).d2(), v(4).d2(), v(6).d2());
            a.trn1(v(1).d2(), v(5).d2(), v(7).d2());
            a.trn2(v(2).d2(), v(4).d2(), v(6).d2());
            a.trn2(v(3).d2(), v(5).d2(), v(7).d2());
            add_imm(a, X7, X19, (to + bj * 4 * 64 + bi * 16) as i64, X8);
            for r in 0..4u8 {
                a.str(q(r), mem(X7, i64::from(r) * 64));
            }
        }
    }
}

/// Generates the chunk kernel of `spec`.
pub fn emit(spec: &ChunkSpec) -> Result<Vec<u8>> {
    spec.check().map_err(|d| HerbertError::Backend(format!("backend_jit deltanet chunk: {d}")))?;
    let (hkd, hvd) = (spec.hkd, spec.hvd);
    let l = Layout::of(spec);
    let row_s = (hvd * 4) as i64;
    let mut a = Asm::new();
    // x19 scratch, x20 γ₁₅ (bits), x21 blocks left, x22 block byte offset,
    // x23..x24 loop pointers; d8..d15 because smstart clears the vector registers.
    let frame = Frame::enter(&mut a, false, &[X19, X20, X21, X22, X23, X24], true, l.total);
    a.add(X19, SP, 0);

    // ── 1. Kᵀ, Qᵀ, KKᵀ, QKᵀ ─────────────────────────────────────────────────
    a.smstart();
    a.ptrue(p(0), Et::S);
    a.zero_za_s(0b1100);
    add_imm(&mut a, X7, X19, l.kt as i64, X16);
    add_imm(&mut a, X8, X19, l.qt as i64, X16);
    a.mov_imm(X9, 0);
    let dblocks = a.label();
    a.mov_imm(X10, (hkd / 16) as u64);
    a.bind(dblocks);
    a.mov(X11, X3);
    a.mov(X16, X2);
    for t in 0..CHUNK {
        slice_base(&mut a, t);
        a.ld1w_za(slice(0, Slice::H, t), p(0), X11, X9);
        a.ld1w_za(slice(1, Slice::H, t), p(0), X16, X9);
        if t + 1 < CHUNK {
            add_imm(&mut a, X11, X11, spec.row_bytes as i64, X17);
            add_imm(&mut a, X16, X16, spec.row_bytes as i64, X17);
        }
    }
    for i in 0..CHUNK {
        slice_base(&mut a, i);
        a.mova_from_za(z(0), p(0), slice(0, Slice::V, i));
        a.mova_from_za(z(1), p(0), slice(1, Slice::V, i));
        a.st1_z(z(0), Et::S, p(0), mem(X7, 0));
        a.st1_z(z(1), Et::S, p(0), mem(X8, 0));
        a.addvl(X7, X7, 1);
        a.addvl(X8, X8, 1);
        a.fmopa(2, p(0), p(0), z(0), z(0));
        a.fmopa(3, p(0), p(0), z(1), z(0));
    }
    a.add(X9, X9, 16);
    a.subs(X10, X10, 1);
    a.bne(dblocks);
    for (tile, at) in [(2u8, l.gkk), (3, l.gqk)] {
        add_imm(&mut a, X7, X19, at as i64, X16);
        for t in 0..CHUNK {
            slice_base(&mut a, t);
            a.st1w_za(slice(tile, Slice::H, t), p(0), X7, XZR);
            a.add(X7, X7, 64);
        }
    }
    a.smstop();

    // ── 2. r, γ, A, T, the scaled operands ───────────────────────────────
    // The decays and gates of the sixteen positions, then γ_t = γ_{t-1}·a_t;
    // x15 = the three vectors a, β, γ (64 bytes apart).
    add_imm(&mut a, X15, X19, l.va as i64, X16);
    a.mov(X7, X6);
    a.mov(X8, X5);
    for t in 0..CHUNK as i64 {
        a.ldr(s(0), at(X7));
        a.str(s(0), mem(X15, 4 * t));
        a.ldr(s(1), at(X8));
        a.str(s(1), mem(X15, 64 + 4 * t));
        if t == 0 {
            a.fmov(s(2), s(0));
        } else {
            a.fmul(s(2), s(2), s(0));
        }
        a.str(s(2), mem(X15, 128 + 4 * t));
        if t + 1 < CHUNK as i64 {
            add_imm(&mut a, X7, X7, spec.gate_bytes as i64, X16);
            add_imm(&mut a, X8, X8, spec.gate_bytes as i64, X16);
        }
    }
    a.fmov_to_gp(W20, s(2));
    // r rows in v0..v3: row 0 = e0; row t = row(t−1)·a_t, lane t = 1.
    a.fmov_imm(s(5), 1.0);
    for r in 0..4u8 {
        a.movi(v(r).d2(), 0);
    }
    a.ins(v(0).s(0), v(5).s(0));
    add_imm(&mut a, X7, X19, l.r as i64, X16);
    a.stp(q(0), q(1), mem(X7, 0));
    a.stp(q(2), q(3), mem(X7, 32));
    for t in 1..CHUNK {
        a.ldr(s(4), mem(X15, (4 * t) as i64));
        for r in 0..4u8 {
            a.fmul(v(r).s4(), v(r).s4(), v(4).s(0));
        }
        a.ins(v((t / 4) as u8).s((t % 4) as u8), v(5).s(0));
        a.stp(q(0), q(1), mem(X7, (t * 64) as i64));
        a.stp(q(2), q(3), mem(X7, (t * 64 + 32) as i64));
    }
    // v0..v3 = the last row of r, kept for K̃.
    // A rows (into the P scratch): (r_t · β_t) ⊙ KKᵀ_t.
    for t in 0..CHUNK {
        add_imm(&mut a, X7, X19, (l.r + 64 * t) as i64, X16);
        add_imm(&mut a, X8, X19, (l.gkk + 64 * t) as i64, X16);
        a.ldr(s(4), mem(X15, (64 + 4 * t) as i64));
        a.ldp(q(16), q(17), mem(X7, 0));
        a.ldp(q(18), q(19), mem(X7, 32));
        a.ldp(q(20), q(21), mem(X8, 0));
        a.ldp(q(22), q(23), mem(X8, 32));
        for c in 0..4u8 {
            a.fmul(v(16 + c).s4(), v(16 + c).s4(), v(4).s(0));
            a.fmul(v(16 + c).s4(), v(16 + c).s4(), v(20 + c).s4());
        }
        add_imm(&mut a, X7, X19, (l.pb + 64 * t) as i64, X16);
        a.stp(q(16), q(17), mem(X7, 0));
        a.stp(q(18), q(19), mem(X7, 32));
    }
    // T rows: T_i = e_i − Σ_{k<i} A_ik·T_k, k ascending.
    for i in 0..CHUNK {
        for c in 0..4u8 {
            a.movi(v(24 + c).d2(), 0);
        }
        a.ins(v(24 + (i / 4) as u8).s((i % 4) as u8), v(5).s(0));
        if i > 0 {
            add_imm(&mut a, X7, X19, (l.pb + 64 * i) as i64, X16);
            a.ldp(q(16), q(17), mem(X7, 0));
            a.ldp(q(18), q(19), mem(X7, 32));
            for k in 0..i {
                add_imm(&mut a, X8, X19, (l.tm + 64 * k) as i64, X16);
                a.ldp(q(20), q(21), mem(X8, 0));
                a.ldp(q(22), q(23), mem(X8, 32));
                let lane = v(16 + (k / 4) as u8).s((k % 4) as u8);
                for c in 0..4u8 {
                    a.fmls(v(24 + c).s4(), v(20 + c).s4(), lane);
                }
            }
        }
        add_imm(&mut a, X8, X19, (l.tm + 64 * i) as i64, X16);
        a.stp(q(24), q(25), mem(X8, 0));
        a.stp(q(26), q(27), mem(X8, 32));
    }
    // M rows (into the D scratch): (r_t ⊙ QKᵀ_t)·scale; zero above the diagonal with r.
    a.mov_imm(W9, u64::from(spec.scale().to_bits()));
    a.fmov_from_gp(s(6), W9);
    for t in 0..CHUNK {
        add_imm(&mut a, X7, X19, (l.r + 64 * t) as i64, X16);
        add_imm(&mut a, X8, X19, (l.gqk + 64 * t) as i64, X16);
        a.ldp(q(16), q(17), mem(X7, 0));
        a.ldp(q(18), q(19), mem(X7, 32));
        a.ldp(q(20), q(21), mem(X8, 0));
        a.ldp(q(22), q(23), mem(X8, 32));
        for c in 0..4u8 {
            a.fmul(v(16 + c).s4(), v(16 + c).s4(), v(20 + c).s4());
            a.fmul(v(16 + c).s4(), v(16 + c).s4(), v(6).s(0));
        }
        add_imm(&mut a, X7, X19, (l.db + 64 * t) as i64, X16);
        a.stp(q(16), q(17), mem(X7, 0));
        a.stp(q(18), q(19), mem(X7, 32));
    }
    // K̃ rows: K_t · r_15,t (the last row of r, still in v0..v3).
    a.mov(X7, X3);
    add_imm(&mut a, X8, X19, l.ks as i64, X16);
    for t in 0..CHUNK {
        let lane = v((t / 4) as u8).s((t % 4) as u8);
        a.mov(X9, X7);
        for _ in 0..hkd / 16 {
            a.ldp(q(16), q(17), post(X9, 32));
            a.ldp(q(18), q(19), post(X9, 32));
            for r in 0..4u8 {
                a.fmul(v(16 + r).s4(), v(16 + r).s4(), lane);
            }
            a.stp(q(16), q(17), mem(X8, 0));
            a.stp(q(18), q(19), mem(X8, 32));
            a.add(X8, X8, 64);
        }
        if t + 1 < CHUNK {
            add_imm(&mut a, X7, X7, spec.row_bytes as i64, X16);
        }
    }
    // The transposes: T → TT, M → MT.
    transpose16(&mut a, l.tm, l.tt);
    transpose16(&mut a, l.db, l.mt);
    // TbT_j = TT_j·β_j, TbgT_j = TT_j·(β_j·γ_j).
    for j in 0..CHUNK {
        add_imm(&mut a, X7, X19, (l.tt + 64 * j) as i64, X16);
        a.ldp(q(16), q(17), mem(X7, 0));
        a.ldp(q(18), q(19), mem(X7, 32));
        a.ldr(s(4), mem(X15, (64 + 4 * j) as i64));
        a.ldr(s(6), mem(X15, (128 + 4 * j) as i64));
        a.fmul(s(6), s(4), s(6));
        for c in 0..4u8 {
            a.fmul(v(20 + c).s4(), v(16 + c).s4(), v(4).s(0));
            a.fmul(v(24 + c).s4(), v(16 + c).s4(), v(6).s(0));
        }
        add_imm(&mut a, X7, X19, (l.tbt + 64 * j) as i64, X16);
        a.stp(q(20), q(21), mem(X7, 0));
        a.stp(q(22), q(23), mem(X7, 32));
        add_imm(&mut a, X7, X19, (l.tbgt + 64 * j) as i64, X16);
        a.stp(q(24), q(25), mem(X7, 0));
        a.stp(q(26), q(27), mem(X7, 32));
    }
    // Q̃ᵀ_d = Qᵀ_d ⊙ (γ·scale).
    a.mov_imm(W9, u64::from(spec.scale().to_bits()));
    a.fmov_from_gp(s(6), W9);
    add_imm(&mut a, X7, X19, (l.va + 128) as i64, X16);
    a.ldp(q(20), q(21), mem(X7, 0));
    a.ldp(q(22), q(23), mem(X7, 32));
    for c in 0..4u8 {
        a.fmul(v(20 + c).s4(), v(20 + c).s4(), v(6).s(0));
    }
    add_imm(&mut a, X7, X19, l.qt as i64, X16);
    {
        let top = a.label();
        a.mov_imm(X8, hkd as u64);
        a.bind(top);
        a.ldp(q(16), q(17), mem(X7, 0));
        a.ldp(q(18), q(19), mem(X7, 32));
        for c in 0..4u8 {
            a.fmul(v(16 + c).s4(), v(16 + c).s4(), v(20 + c).s4());
        }
        a.stp(q(16), q(17), mem(X7, 0));
        a.stp(q(18), q(19), mem(X7, 32));
        a.add(X7, X7, 64);
        a.subs(X8, X8, 1);
        a.bne(top);
    }

    // ── 3. Per block of sixteen value columns ────────────────────────────
    a.smstart();
    a.ptrue(p(0), Et::S);
    a.dup_z(z(31), Et::S, W20);
    a.mov_imm(X21, (hvd / 16) as u64);
    a.mov_imm(X22, 0);
    let blocks = a.label();
    a.bind(blocks);
    // K·S₀ (tile 0) and Q̃·S₀ (tile 1) over the key dimension.
    a.zero_za_s(0b0011);
    add_imm(&mut a, X7, X19, l.kt as i64, X16);
    add_imm(&mut a, X8, X19, l.qt as i64, X16);
    a.add(X9, X0, X22);
    {
        let top = a.label();
        a.mov_imm(X10, hkd as u64);
        a.bind(top);
        a.ld1_z(z(0), Et::S, p(0), mem(X7, 0));
        a.ld1_z(z(1), Et::S, p(0), mem(X8, 0));
        a.ld1_z(z(2), Et::S, p(0), mem(X9, 0));
        a.fmopa(0, p(0), p(0), z(0), z(2));
        a.fmopa(1, p(0), p(0), z(1), z(2));
        a.addvl(X7, X7, 1);
        a.addvl(X8, X8, 1);
        add_imm(&mut a, X9, X9, row_s, X16);
        a.subs(X10, X10, 1);
        a.bne(top);
    }
    store_tile(&mut a, 0, l.pb);
    // Δ = Σ_j Tb[:,j]⊗V_j − Tbg[:,j]⊗P_j, j ascending (tile 0).
    a.zero_za_s(0b0001);
    add_imm(&mut a, X7, X19, l.tbt as i64, X16);
    add_imm(&mut a, X9, X19, l.tbgt as i64, X16);
    add_imm(&mut a, X10, X19, l.pb as i64, X16);
    a.add(X8, X4, X22);
    for j in 0..CHUNK {
        a.ld1_z(z(0), Et::S, p(0), mem(X7, j as i64 % 8));
        a.ld1_z(z(1), Et::S, p(0), mem(X8, 0));
        a.fmopa(0, p(0), p(0), z(0), z(1));
        a.ld1_z(z(2), Et::S, p(0), mem(X9, j as i64 % 8));
        a.ld1_z(z(3), Et::S, p(0), mem(X10, j as i64 % 8));
        a.fmops(0, p(0), p(0), z(2), z(3));
        if j % 8 == 7 {
            a.addvl(X7, X7, 8);
            a.addvl(X9, X9, 8);
            a.addvl(X10, X10, 8);
        }
        if j + 1 < CHUNK {
            add_imm(&mut a, X8, X8, spec.row_bytes as i64, X16);
        }
    }
    // The sixteen rows of Δ into z8..z23, where both the output and the
    // eight key tiles of the state update read them.
    for t in 0..CHUNK as u8 {
        slice_base(&mut a, usize::from(t));
        a.mova_from_za(z(8 + t), p(0), slice(0, Slice::H, usize::from(t)));
    }
    // O = Q̃·S₀ + Σ_j M[:,j]⊗Δ_j (tile 1), then into the output rows.
    add_imm(&mut a, X7, X19, l.mt as i64, X16);
    for j in 0..CHUNK {
        a.ld1_z(z(0), Et::S, p(0), mem(X7, j as i64 % 8));
        a.fmopa(1, p(0), p(0), z(0), z(8 + j as u8));
        if j % 8 == 7 {
            a.addvl(X7, X7, 8);
        }
    }
    a.add(X8, X1, X22);
    for t in 0..CHUNK {
        slice_base(&mut a, t);
        a.st1w_za(slice(1, Slice::H, t), p(0), X8, XZR);
        if t + 1 < CHUNK {
            add_imm(&mut a, X8, X8, spec.out_bytes as i64, X16);
        }
    }
    // The state, sixteen key rows at a time: γ₁₅·S₀ into tile 2, then
    // Σ_t K̃_t⊗Δ_t, t ascending, stored back in place.
    a.add(X23, X0, X22);
    add_imm(&mut a, X24, X19, l.ks as i64, X16);
    {
        let top = a.label();
        a.mov_imm(X10, (hkd / 16) as u64);
        a.bind(top);
        a.mov(X11, X23);
        for r in 0..CHUNK {
            slice_base(&mut a, r);
            a.ld1_z(z(0), Et::S, p(0), mem(X11, 0));
            a.fmul_z(z(0), Et::S, z(0), z(31));
            a.mova_to_za(slice(2, Slice::H, r), p(0), z(0));
            add_imm(&mut a, X11, X11, row_s, X16);
        }
        a.mov(X7, X24);
        for t in 0..CHUNK {
            a.ld1_z(z(1), Et::S, p(0), mem(X7, 0));
            a.fmopa(2, p(0), p(0), z(1), z(8 + t as u8));
            add_imm(&mut a, X7, X7, (hkd * 4) as i64, X16);
        }
        a.mov(X11, X23);
        for r in 0..CHUNK {
            slice_base(&mut a, r);
            a.st1w_za(slice(2, Slice::H, r), p(0), X11, XZR);
            add_imm(&mut a, X11, X11, row_s, X16);
        }
        add_imm(&mut a, X23, X23, 16 * row_s, X16);
        a.add(X24, X24, 64);
        a.subs(X10, X10, 1);
        a.bne(top);
    }
    a.add(X22, X22, 64);
    a.subs(X21, X21, 1);
    a.bne(blocks);
    a.smstop();
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("deltanet chunk hkd={hkd} hvd={hvd}")))
}

/// The sixteen rows of `tile` into the scratch at `x19 + at`, 64 bytes apart.
/// Clobbers `x7`, `x16`, `w12`.
fn store_tile(a: &mut Asm, tile: u8, at: usize) {
    add_imm(a, X7, X19, at as i64, X16);
    for t in 0..CHUNK {
        slice_base(a, t);
        a.st1w_za(slice(tile, Slice::H, t), p(0), X7, XZR);
        a.add(X7, X7, 64);
    }
}
