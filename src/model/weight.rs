//! A weight matrix packed the way the emitted GEMM reads it: the R16 layout.
//!
//! What enters: a row-major bf16 matrix `[N, K]` straight from the checkpoint.
//! What comes out: the same values, reordered as
//! `[N/16 blocks, K_even/2 pairs, 16 rows, 2 values]`, so that one 64-byte load
//! brings a pair of K values for sixteen output rows and `vdpbf16ps` (or its
//! AVX2 emulation) multiplies it against a broadcast pair of activations.
//! Rows past N in the last block and the column past an odd K are zero.
//! No value changes: bf16 in, the same bf16 out. There is no other weight
//! format in this engine.

use crate::core::error::{HerbertError, Result};
use crate::core::tensor::BF16;
use crate::model::hugepages::HugeVec;

pub const ROWS_INTERLEAVED: usize = crate::jit::target::R16_ROWS;

/// `R16` row-interleaved bf16 weight matrix: each pair is stored as
/// `[w(k0,row0), w(k1,row0), w(k0,row1), w(k1,row1), ...]`.
#[derive(Clone)]
pub struct PackedBf16R16Weight {
    /// Packed data in `[N/16 blocks, K_even/2 pairs, 16 rows, 2]` order.
    pub data_r16: HugeVec<BF16>,
    /// Number of output features (rows).
    pub n: usize,
    /// Number of input features (columns).
    pub k: usize,
    /// K rounded up to even: the kernel consumes pairs.
    pub k_even: usize,
    /// Number of row blocks (= ceil(N / 16)).
    pub n_blocks: usize,
}

impl PackedBf16R16Weight {
    /// A weight of shape `n×k` **without data** — only for a tensor the model
    /// declares and never reads (an absent optional projection).
    pub fn empty(n: usize, k: usize) -> Self {
        let k_even = (k + 1) & !1;
        Self {
            data_r16: HugeVec::from_vec(Vec::new()),
            n,
            k,
            k_even,
            n_blocks: n.div_ceil(ROWS_INTERLEAVED),
        }
    }

    /// Pack from row-major into the R16 layout the emitted GEMM consumes.
    pub fn pack_from_row_major_r16_only(data: &[BF16], n: usize, k: usize) -> Result<Self> {
        let expected = n
            .checked_mul(k)
            .ok_or_else(|| HerbertError::ModelLoad(format!("R16 pack: N*K overflows for N={n} K={k}")))?;
        if data.len() != expected {
            return Err(HerbertError::ModelLoad(format!(
                "R16 pack: {} elements for N={n} K={k}, expected {expected}",
                data.len()
            )));
        }
        let k_even = k
            .checked_add(1)
            .map(|value| value & !1)
            .ok_or_else(|| HerbertError::ModelLoad(format!("R16 pack: K={k} cannot be padded")))?;
        let n_blocks = n.div_ceil(ROWS_INTERLEAVED);
        let packed_len = n_blocks
            .checked_mul(k_even)
            .and_then(|length| length.checked_mul(ROWS_INTERLEAVED))
            .ok_or_else(|| HerbertError::ModelLoad(format!("R16 pack: packed length overflows for N={n} K={k}")))?;
        let mut data_r16 = vec![0u16; packed_len];
        for block in 0..n_blocks {
            let nr = ROWS_INTERLEAVED.min(n - block * ROWS_INTERLEAVED);
            for kp in 0..(k_even / 2) {
                let k0 = kp * 2;
                let k1 = k0 + 1;
                let base = (block * (k_even / 2) + kp) * ROWS_INTERLEAVED * 2;
                for lane in 0..nr {
                    let row = block * ROWS_INTERLEAVED + lane;
                    data_r16[base + lane * 2] = if k0 < k { data[row * k + k0] } else { 0 };
                    data_r16[base + lane * 2 + 1] = if k1 < k { data[row * k + k1] } else { 0 };
                }
            }
        }
        Ok(Self { data_r16: HugeVec::from_vec(data_r16), n, k, k_even, n_blocks })
    }
}
