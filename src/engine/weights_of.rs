//! The borrow of every layer's weights, as the scheduler expects them.
//!
//! Nothing is allocated or repacked here: slices over what the loader and
//! [`crate::engine::packs`] already produced. On a GatedDeltaNet layer the six
//! attention fields receive empty slices the scheduler never reads; on an
//! attention layer the mixer carries no DeltaNet weights.

use crate::engine::packs::DeltaPacks;
use crate::jit::{DeltaWeights, LayerWeights, Mixer};
use crate::model::{LayerBlock, Model};

pub fn layer_weights<'a>(model: &'a Model, packs: &'a [Option<DeltaPacks>]) -> Vec<LayerWeights<'a>> {
    const EMPTY: &[u16] = &[];
    model
        .decoder_layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            let mlp = &layer.mlp;
            let (attn, mixer) = match (&layer.block, packs.get(l).and_then(Option::as_ref)) {
                (LayerBlock::GatedDeltaNet(dn), Some(p)) => (
                    None,
                    Mixer::GatedDeltaNet(DeltaWeights {
                        in_proj_qkv: &dn.in_proj_qkv.data_r16,
                        in_proj_z: &dn.in_proj_z.data_r16,
                        in_proj_a: &p.in_proj_a.data_r16,
                        in_proj_b: &p.in_proj_b.data_r16,
                        conv1d_weight: &p.conv1d_weight,
                        // The only f32 of the block: `decay = exp(−exp(a_log) · dt)`.
                        a_log: &dn.a_log,
                        dt_bias: &p.dt_bias,
                        norm_weight: &dn.norm_weight,
                        out_proj: &dn.out_proj.data_r16,
                        num_k_heads: dn.num_k_heads,
                        num_v_heads: dn.num_v_heads,
                        head_k_dim: dn.head_k_dim,
                        head_v_dim: dn.head_v_dim,
                        conv_kernel: dn.conv_kernel_size,
                    }),
                ),
                (LayerBlock::Attention(a), _) => (Some(a), Mixer::Attention),
                (LayerBlock::GatedDeltaNet(_), None) => unreachable!("delta_packs() builds one pack per DeltaNet layer"),
            };
            LayerWeights {
                input_norm: &layer.input_layernorm,
                post_attn_norm: &layer.post_attention_layernorm,
                q_norm: attn.map(|a| &a.q_norm[..]).unwrap_or(EMPTY),
                k_norm: attn.map(|a| &a.k_norm[..]).unwrap_or(EMPTY),
                wq: attn.map(|a| &a.q_proj.data_r16[..]).unwrap_or(EMPTY),
                wk: attn.map(|a| &a.k_proj.data_r16[..]).unwrap_or(EMPTY),
                wv: attn.map(|a| &a.v_proj.data_r16[..]).unwrap_or(EMPTY),
                wo: attn.map(|a| &a.o_proj.data_r16[..]).unwrap_or(EMPTY),
                activation: crate::jit::Activation::SwiGlu,
                intermediate: mlp.intermediate_size,
                n_kv_heads: attn.map(|a| a.num_kv_heads),
                head_dim: attn.map_or(model.config.head_dim, |a| a.head_dim),
                rotary_ndims: attn.map_or(model.config.rotary_ndims, |a| a.rotary_ndims),
                w_gate: &mlp.gate_proj.data_r16,
                w_up: &mlp.up_proj.data_r16,
                w_down: &mlp.down_proj.data_r16,
                mixer,
            }
        })
        .collect()
}
