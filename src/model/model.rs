//! What the loader builds: the model in memory, every matrix packed for the
//! emitted GEMM, every vector as the checkpoint stores it.
//!
//! Sizes on Qwen3.5-4B: the embedding table `[248320, 2560]` bf16 (1.2 GiB),
//! 32 decoder layers (24 with a GatedDeltaNet mixer, 8 with attention), the
//! final norm, the language head (the embedding table packed R16 when tied),
//! and the RoPE tables for `max_position_embeddings` positions.

use crate::core::config::Config;
use crate::core::tensor::BF16;
use crate::model::weight::PackedBf16R16Weight;

pub struct Model {
    /// `[vocab, hidden]`, row-major bf16: one row is copied per token.
    pub embed_tokens: crate::core::avec::AVec<BF16>,
    pub decoder_layers: Vec<DecoderLayer>,
    /// The final RMSNorm weight, `[hidden]`.
    pub norm: crate::core::avec::AVec<BF16>,
    /// `[vocab, hidden]` packed R16: the last GEMM of a token, `N = vocab`.
    pub lm_head: PackedBf16R16Weight,
    pub config: Config,
    /// `cos(pos · θ_i)`, `[max_position_embeddings, rotary_ndims / 2]` f32.
    pub cos_cache: crate::core::avec::AVec<f32>,
    /// `sin`, same layout.
    pub sin_cache: crate::core::avec::AVec<f32>,
}

/// One decoder layer: `x += mixer(norm(x))` then `x += mlp(norm(x))`.
pub struct DecoderLayer {
    pub input_layernorm: crate::core::avec::AVec<BF16>,
    pub block: LayerBlock,
    pub post_attention_layernorm: crate::core::avec::AVec<BF16>,
    pub mlp: Mlp,
}

/// The mixer of a layer.
pub enum LayerBlock {
    Attention(Attention),
    GatedDeltaNet(GatedDeltaNet),
}

/// Grouped-query attention with per-head Q/K norms, partial RoPE and the
/// fused output gate.
pub struct Attention {
    /// `[2 · num_heads · head_dim, hidden]` with the gate: Q rows first, gate
    /// rows next, once the loader has permuted them.
    pub q_proj: PackedBf16R16Weight,
    /// `[num_kv_heads · head_dim, hidden]`.
    pub k_proj: PackedBf16R16Weight,
    pub v_proj: PackedBf16R16Weight,
    /// `[hidden, num_heads · head_dim]`.
    pub o_proj: PackedBf16R16Weight,
    /// `[head_dim]`, applied to every query head before RoPE.
    pub q_norm: crate::core::avec::AVec<BF16>,
    /// `[head_dim]`, applied to every key head before RoPE.
    pub k_norm: crate::core::avec::AVec<BF16>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rotary_ndims: usize,
}

/// The GatedDeltaNet mixer of the linear-attention layers.
pub struct GatedDeltaNet {
    /// `[2 · num_k_heads · head_k_dim + num_v_heads · head_v_dim, hidden]`: Q, K, V.
    pub in_proj_qkv: PackedBf16R16Weight,
    /// `[num_v_heads · head_v_dim, hidden]`: the output gate.
    pub in_proj_z: PackedBf16R16Weight,
    /// `[num_v_heads, hidden]` f32: the decay gate `α`.
    pub in_proj_a: crate::core::avec::AVec<f32>,
    /// `[num_v_heads, hidden]` f32: the update gate `β`.
    pub in_proj_b: crate::core::avec::AVec<f32>,
    /// `[conv_dim, conv_kernel]` f32: the depthwise causal conv.
    pub conv1d_weight: crate::core::avec::AVec<f32>,
    /// `[num_v_heads]` f32: `decay = exp(−exp(A_log) · softplus(α + dt_bias))`.
    pub a_log: crate::core::avec::AVec<f32>,
    /// `[num_v_heads]` f32.
    pub dt_bias: crate::core::avec::AVec<f32>,
    /// `[head_v_dim]`: the gated RMSNorm weight.
    pub norm_weight: crate::core::avec::AVec<BF16>,
    /// `[hidden, num_v_heads · head_v_dim]`.
    pub out_proj: PackedBf16R16Weight,
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub head_k_dim: usize,
    pub head_v_dim: usize,
    pub conv_kernel_size: usize,
}

/// The SwiGLU MLP: `down(silu(gate(x)) · up(x))`.
pub struct Mlp {
    /// `[intermediate, hidden]`.
    pub gate_proj: PackedBf16R16Weight,
    pub up_proj: PackedBf16R16Weight,
    /// `[hidden, intermediate]`.
    pub down_proj: PackedBf16R16Weight,
    pub intermediate_size: usize,
}
