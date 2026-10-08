//! The GatedDeltaNet tensors the engine must produce once at load, because
//! the loader keeps them in f32 and the generated kernels read bf16.
//!
//! This is the only place in the engine that copies weights, and it is
//! tiny: per DeltaNet layer, `Wa` and `Wb` (`[num_v_heads, hidden]`) repacked
//! R16, the conv kernel and `dt_bias` narrowed to bf16. Every value comes
//! from a bf16 checkpoint tensor widened by the loader, so the narrowing is
//! exact. `A_log`, the only true f32 of the block, is borrowed as is.

use crate::core::avec::AVec;
use crate::core::error::Result;
use crate::model::{LayerBlock, Model, PackedBf16R16Weight};

pub struct DeltaPacks {
    /// `Wa` repacked R16, `[num_v_heads, hidden]`.
    pub in_proj_a: PackedBf16R16Weight,
    /// `Wb`, idem.
    pub in_proj_b: PackedBf16R16Weight,
    /// `conv1d.weight` narrowed to bf16, flat: `w[ch · K + j]`.
    pub conv1d_weight: AVec<u16>,
    /// `dt_bias` narrowed to bf16, `[num_v_heads]`.
    pub dt_bias: AVec<u16>,
}

/// f32 → bf16, round to nearest even: the convention of the kernels' epilogue.
pub(crate) fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF).wrapping_add(lsb) >> 16) as u16
}

/// One entry per layer, `None` on the attention layers.
pub fn delta_packs(model: &Model) -> Result<Vec<Option<DeltaPacks>>> {
    let hidden = model.config.hidden_size;
    model
        .decoder_layers
        .iter()
        .map(|layer| -> Result<Option<DeltaPacks>> {
            let LayerBlock::GatedDeltaNet(dn) = &layer.block else {
                return Ok(None);
            };
            let a: Vec<u16> = dn.in_proj_a.iter().copied().map(to_bf16).collect();
            let b: Vec<u16> = dn.in_proj_b.iter().copied().map(to_bf16).collect();
            Ok(Some(DeltaPacks {
                in_proj_a: PackedBf16R16Weight::pack_from_row_major_r16_only(&a, dn.num_v_heads, hidden)?,
                in_proj_b: PackedBf16R16Weight::pack_from_row_major_r16_only(&b, dn.num_v_heads, hidden)?,
                conv1d_weight: AVec::from_vec(dn.conv1d_weight.iter().copied().map(to_bf16).collect()),
                dt_bias: AVec::from_vec(dn.dt_bias.iter().copied().map(to_bf16).collect()),
            }))
        })
        .collect()
}
