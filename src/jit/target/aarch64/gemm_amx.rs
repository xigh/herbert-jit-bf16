//! The bf16 GEMM micro-kernel on Apple's AMX: `matfp` bf16 → f32 into Z.
//!
//! `matfp` with data type 1 adds to Z the outer product of the 32 bf16 of Y
//! and the 32 bf16 of X, in f32, each lane one fused multiply-add rounded to
//! nearest even, denormals kept (`crates/a64/src/amx.rs`): the arithmetic of
//! one `bfmlalb` or `bfmlalt` lane. Lane `i` of X times lane `j` of Y lands
//! in row `2j + (i & 1)` of Z, column `i >> 1`.
//!
//! X takes a pair of `K` of one R16 block of `W` — 64 contiguous bytes,
//! sixteen columns `n` × the two elements of the pair, lane `2n + t` — and Y
//! one `K` of thirty-two rows of `A`, packed by the plan's packing node
//! (`[K][32]`, `jit/emit.rs`, [`packed_a_amx_bytes`]). The even element of
//! the pair lands, for row `m` and column `n`, in row `2m` of Z, column `n`.
//! The odd element is brought to the same place by reading X two bytes
//! further (lane `2n` then holds element `2n + 1`) against the next `K` of
//! `A`. So each element of `C` is one accumulator of Z that receives its `K`
//! products one after the other, the even element of a pair before the odd,
//! the pairs in order — the order and the arithmetic of the vector kernel
//! under `bfmlal` (`gemm.rs`), hence its bits. The odd rows of Z collect
//! cross products and are never read: half of each outer product is spent,
//! the price of keeping both the R16 layout and that order.
//!
//! A call computes 32 rows × 16 columns. `set` hands a zeroed Z, as the vector
//! kernel's accumulators start at +0. Four pairs per turn: four `ldx` (one
//! pair of `W` each), eight `ldy` (one `K` of `A` each), eight `matfp`. The
//! epilogue stores the `height` useful rows of Z to the stack (`stz`), and
//! the core converts them (`bfcvtn`, nearest even, as the vector kernel) or
//! copies them, storing the `n_tail` valid columns only.
//!
//! [`packed_a_amx_bytes`]: crate::jit::emit::packed_a_amx_bytes

use crate::core::error::{HerbertError, Result};
use crate::jit::emit::{AFormat, MicroKernelSpec, AMX_ROWS};
use crate::jit::ir::{Decline, TileShape, ROWS_INTERLEAVED};
use crate::jit::target::aarch64::gemm::{store_partial_bf16, store_partial_f32};
use crate::jit::target::aarch64::util::{add_imm, backend_err, Frame};
use a64::*;

/// The tile: thirty-two rows of `A`, one R16 block of `W`.
pub const AMX_TILE: TileShape = TileShape { mr: AMX_ROWS, nr: ROWS_INTERLEAVED };

/// `matfp`, data type 1 (bits 42–45): bf16 × bf16 → f32.
const TYPE_BF16_F32: u64 = 1 << 42;

/// Is the AMX there: every Apple M so far (M1 to M5; checked on the M3 Ultra
/// and the M5). The knob `Knobs::gemm_amx` asks for it.
pub fn available() -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
}

/// The shapes this kernel serves.
pub fn check(spec: &MicroKernelSpec) -> std::result::Result<(), Decline> {
    if !available() {
        return Err(Decline::TileTooWide);
    }
    if spec.mr == 0 || spec.mr > AMX_ROWS || spec.nr != ROWS_INTERLEAVED {
        return Err(Decline::TileTooWide);
    }
    // The packed block holds whole `K`s of bf16; the loop takes four pairs a turn.
    if spec.k_odd || spec.k_dynamic || spec.a_fmt != AFormat::Bf16 || !spec.k_pairs.is_multiple_of(4) {
        return Err(Decline::OddK);
    }
    Ok(())
}

/// `matfp` operand: data type 1, X at byte `x_off`, Y at byte `y_off`.
const fn matfp(x_off: u64, y_off: u64) -> u64 {
    TYPE_BF16_F32 | x_off << 10 | y_off
}

/// Generates the kernel described by `spec`, `A` packed `[K][32]`.
pub fn emit(spec: &MicroKernelSpec) -> Result<Vec<u8>> {
    check(spec).map_err(|d| HerbertError::Backend(format!("backend_jit amx gemm: {d}")))?;
    // x3..x6: the four pairs of W of a turn, each with its X register in bits
    // 56–58; x7..x14: the eight Ks of A, each with its Y register; x15 the
    // turns; x19..x26 the eight matfp operands.
    let wp = |p: usize| x((3 + p) as u8);
    let ak = |u: usize| x((7 + u) as u8);
    let op = |i: usize| x((19 + i) as u8);
    let saved = [X19, X20, X21, X22, X23, X24, X25, X26];
    // The rows of Z stored for the core, 64 bytes each, aligned on 64.
    let scratch = AMX_ROWS * 64 + 64;
    let mut a = Asm::new();
    let frame = Frame::enter(&mut a, false, &saved, false, scratch);
    for p in 0..4u64 {
        a.mov_imm(X16, p * 64 + (p << 56));
        a.add(wp(p as usize), X1, X16);
    }
    for u in 0..8u64 {
        a.mov_imm(X16, u * 64 + (u << 56));
        a.add(ak(u as usize), X0, X16);
    }
    for p in 0..4u64 {
        // Even element: X at the pair, Y at K = 2p; odd: X two bytes on, Y at K = 2p + 1.
        a.mov_imm(op(2 * p as usize), matfp(p * 64, 2 * p * 64));
        a.mov_imm(op(2 * p as usize + 1), matfp(p * 64 + 2, (2 * p + 1) * 64));
    }
    a.amx_set();
    a.mov_imm(X15, (spec.k_pairs / 4) as u64);
    let top = a.here();
    for p in 0..4 {
        a.amx(AmxOp::Ldx, wp(p));
    }
    for u in 0..8 {
        a.amx(AmxOp::Ldy, ak(u));
    }
    for i in 0..8 {
        a.amx(AmxOp::Matfp, op(i));
    }
    for p in 0..4 {
        a.add(wp(p), wp(p), 256);
    }
    for u in 0..8 {
        a.add(ak(u), ak(u), 512);
    }
    a.subs(X15, X15, 1);
    a.bne(top);

    // ── Epilogue: rows 2m of Z to the stack, then C. ──────────────────────
    // x16 = the scratch, aligned; x9 = the stz operand; x15 walks the scratch,
    // x10 the row of C (x17 is the scratch register of the partial stores).
    a.add(X16, SP, 63);
    a.and(X16, X16, !63u64);
    for m in 0..spec.mr as u64 {
        a.mov_imm(X9, m * 64 + ((2 * m) << 56));
        a.add(X9, X16, X9);
        a.amx(AmxOp::Stz, X9);
    }
    a.amx_clr();
    a.mov(X15, X16);
    a.mov(X10, X2);
    for m in 0..spec.mr {
        a.ldp(q(0), q(1), mem(X15, 0));
        a.ldp(q(2), q(3), mem(X15, 32));
        if spec.out_bf16 {
            store_partial_bf16(&mut a, &[0, 1, 2, 3], 4, X10, 0, spec.n_tail);
        } else {
            store_partial_f32(&mut a, &[0, 1, 2, 3], X10, 0, spec.n_tail);
        }
        if m + 1 < spec.mr {
            a.add(X15, X15, 64);
            add_imm(&mut a, X10, X10, spec.ldc_bytes as i64, X11);
        }
    }
    frame.leave(&mut a);
    a.ret();
    a.finish()
        .map(|c| c.bytes)
        .map_err(backend_err(format_args!("amx gemm {}x{} (k_pairs={})", spec.mr, spec.nr, spec.k_pairs)))
}
