//! What the scheduler must know about the loaded model before it compiles
//! anything — and the refusal of everything the emitters do not serve.
//!
//! What enters: the `Config` and the `Model` the loader built.
//! What comes out: the [`Geometry`] (the model shape the kernels are compiled
//! for, the layer count, the vocabulary, the GatedDeltaNet budget), the
//! distinct attention geometries, the KV plan (which layers own a cache), the
//! RoPE table sets, and the MLP widths. Nothing here copies a weight.
//!
//! The config already refuses every other architecture; what is checked here
//! is what the *emitters* need beyond the config's word: the DeltaNet tensor
//! lengths, the multiples of 16 the R16 GEMM wants, a RoPE width the rotation
//! kernel can emit, a SiLU activation. A model that fails one of these is
//! refused at load with a sentence, never served with a plausible answer.

use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::jit::{Activation, DeltaBudget, DeltaShape, ModelShape, PlanKv, RopeTableSet};
use crate::model::{GatedDeltaNet, LayerBlock, Model};

/// Retained geometry, plus what is needed to size the runner.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Shape of a layer, as expected by the scheduler.
    pub shape: ModelShape,
    pub n_layers: usize,
    pub vocab: usize,
    /// What the GatedDeltaNet mixers reserve; `None` on a model without them.
    pub delta: Option<DeltaBudget>,
}

/// Every deviation from the served perimeter, aggregated by message.
#[derive(Default)]
struct Deviations {
    /// `(message, first layer concerned, number of layers)`, in order of appearance.
    seen: Vec<(String, Option<usize>, usize)>,
}

impl Deviations {
    fn note(&mut self, layer: Option<usize>, what: &str) {
        if let Some(e) = self.seen.iter_mut().find(|e| e.0 == what) {
            e.2 += 1;
        } else {
            self.seen.push((what.to_string(), layer, 1));
        }
    }

    /// The refusal: every deviation on stderr, the first one in the error.
    fn refusal(&self, n_layers: usize) -> HerbertError {
        eprintln!("[geometry] {} deviation(s) over {n_layers} layers:", self.seen.len());
        for (what, layer, n) in &self.seen {
            match layer {
                Some(l) => eprintln!("[geometry]   {n}/{n_layers} layers (first: {l}) — {what}"),
                None => eprintln!("[geometry]   model — {what}"),
            }
        }
        let (what, layer, _) = &self.seen[0];
        let detail = match layer {
            Some(l) => format!("layer {l}: {what}"),
            None => what.clone(),
        };
        HerbertError::Backend(format!("out of the served perimeter — {detail}"))
    }
}

/// Reads the geometry and refuses anything the emitters cannot serve.
pub fn geometry(config: &Config, model: &Model) -> Result<Geometry> {
    let mut deviations = Deviations::default();
    let shape = ModelShape {
        hidden: config.hidden_size,
        n_heads: config.num_attention_heads,
        n_kv_heads: config.num_key_value_heads,
        head_dim: config.head_dim,
        rotary_ndims: config.rotary_ndims,
        intermediate: config.intermediate_size,
        eps: config.rms_norm_eps,
        // The `1/√head_dim` convention: Qwen3.5 declares no other scale.
        attn_scale: None,
        // Qwen3.5 norms are zero-centred: `(1 + w) · rms(x)` for every norm of the model.
        norm_plus_one: config.norm_type == crate::core::config::NormType::GemmaRMSNorm,
        output_gate: config.attn_output_gate,
    };
    if !config.rotary_ndims.is_multiple_of(32) {
        deviations.note(
            None,
            &format!("partial RoPE not emittable (rotary_ndims={}: a multiple of 32 is expected)", config.rotary_ndims),
        );
    }
    if config.hidden_act != "silu" {
        deviations.note(None, &format!("hidden_act {:?} (silu expected: SwiGLU MLP, SiLU conv)", config.hidden_act));
    }
    let mut delta: Option<DeltaBudget> = None;
    for (l, layer) in model.decoder_layers.iter().enumerate() {
        match &layer.block {
            LayerBlock::Attention(attn) => {
                if attn.q_norm.len() != attn.head_dim || attn.k_norm.len() != attn.head_dim {
                    deviations.note(Some(l), "Q/K norm weights of unexpected length");
                }
            }
            LayerBlock::GatedDeltaNet(dn) => {
                deltanet_control(l, dn, config, &mut deviations);
                cumulate_delta(&mut delta, dn);
            }
        }
        if layer.mlp.intermediate_size % 16 != 0 {
            deviations.note(Some(l), "MLP width not a multiple of 16 (R16 GEMM)");
        }
    }
    if !deviations.seen.is_empty() {
        return Err(deviations.refusal(model.decoder_layers.len()));
    }
    Ok(Geometry { shape, n_layers: model.decoder_layers.len(), vocab: model.lm_head.n, delta })
}

/// What the scheduler assumes of a GatedDeltaNet mixer: a violation would be
/// a late kernel refusal or a silent wrong computation.
fn deltanet_control(l: usize, dn: &GatedDeltaNet, config: &Config, deviations: &mut Deviations) {
    if dn.num_v_heads == 0 || !dn.num_v_heads.is_multiple_of(16) {
        deviations.note(Some(l), "DeltaNet num_v_heads not a multiple of 16 (R16 GEMM on Wa/Wb)");
    }
    if dn.num_k_heads == 0 || !dn.num_v_heads.is_multiple_of(dn.num_k_heads) {
        deviations.note(Some(l), "DeltaNet GQA not integral (num_v_heads / num_k_heads)");
    }
    if dn.conv_kernel_size < 2 {
        deviations.note(Some(l), "DeltaNet conv kernel < 2");
    }
    let key_dim = dn.num_k_heads * dn.head_k_dim;
    let conv_dim = 2 * key_dim + dn.num_v_heads * dn.head_v_dim;
    if dn.conv1d_weight.len() != conv_dim * dn.conv_kernel_size
        || dn.a_log.len() != dn.num_v_heads
        || dn.dt_bias.len() != dn.num_v_heads
        || dn.norm_weight.len() != dn.head_v_dim
        || dn.in_proj_a.len() != dn.num_v_heads * config.hidden_size
        || dn.in_proj_b.len() != dn.num_v_heads * config.hidden_size
    {
        deviations.note(Some(l), "DeltaNet tensors of unexpected length");
    }
}

/// The maxima over the DeltaNet layers, which size the mixer buffers.
fn cumulate_delta(delta: &mut Option<DeltaBudget>, dn: &GatedDeltaNet) {
    let value_dim = dn.num_v_heads * dn.head_v_dim;
    let conv_dim = 2 * dn.num_k_heads * dn.head_k_dim + value_dim;
    let rec = dn.num_v_heads * dn.head_k_dim * dn.head_v_dim;
    match delta {
        None => {
            *delta = Some(DeltaBudget {
                shape: DeltaShape {
                    head_k_dim: dn.head_k_dim,
                    conv_dim,
                    value_dim,
                    num_v_heads: dn.num_v_heads,
                    conv_kernel: dn.conv_kernel_size,
                },
                n_layers: 1,
                recurrent_stride: rec,
            })
        }
        Some(d) => {
            d.shape.head_k_dim = d.shape.head_k_dim.max(dn.head_k_dim);
            d.shape.conv_dim = d.shape.conv_dim.max(conv_dim);
            d.shape.value_dim = d.shape.value_dim.max(value_dim);
            d.shape.num_v_heads = d.shape.num_v_heads.max(dn.num_v_heads);
            d.shape.conv_kernel = d.shape.conv_kernel.max(dn.conv_kernel_size);
            d.recurrent_stride = d.recurrent_stride.max(rec);
            d.n_layers += 1;
        }
    }
}

/// The model's distinct MLP widths with their activation, in order of
/// appearance: the scheduler compiles one activation kernel per width.
pub fn mlp_widths(model: &Model) -> Vec<(Activation, usize)> {
    let mut v: Vec<(Activation, usize)> = Vec::new();
    for layer in &model.decoder_layers {
        let key = (Activation::SwiGlu, layer.mlp.intermediate_size);
        if !v.contains(&key) {
            v.push(key);
        }
    }
    v
}

/// The distinct attention geometries `(head_dim, rotary_ndims)` of the model:
/// one set of kernels per `head_dim`, one RoPE kernel per `rotary_ndims`.
pub fn attention_geometries(config: &Config, model: &Model) -> Vec<(usize, usize)> {
    let mut v: Vec<(usize, usize)> = Vec::new();
    for layer in &model.decoder_layers {
        if let LayerBlock::Attention(attn) = &layer.block {
            let key = (attn.head_dim, attn.rotary_ndims);
            if !v.contains(&key) {
                v.push(key);
            }
        }
    }
    if v.is_empty() {
        v.push((config.head_dim, config.rotary_ndims));
    }
    v
}

/// The KV plan, one entry per layer: an attention layer owns a cache for its
/// `head_dim`, a GatedDeltaNet layer has none (its state is recurrent).
pub fn plan_kv(model: &Model) -> Vec<PlanKv> {
    model
        .decoder_layers
        .iter()
        .map(|layer| match &layer.block {
            LayerBlock::Attention(attn) => PlanKv::Own(attn.head_dim),
            LayerBlock::GatedDeltaNet(_) => PlanKv::Absent,
        })
        .collect()
}

/// One set of `cos`/`sin` tables per declared `rotary_ndims`: the loader's
/// tables, which serve every attention layer.
pub fn tables_rope<'a>(model: &'a Model, geometries: &[(usize, usize)]) -> Result<Vec<RopeTableSet<'a>>> {
    let mut v: Vec<RopeTableSet<'a>> = Vec::new();
    for &(_, rn) in geometries {
        if v.iter().any(|t| t.rotary_ndims == rn) {
            continue;
        }
        if rn != model.config.rotary_ndims {
            return Err(HerbertError::Backend(format!(
                "no RoPE tables for rotary_ndims={rn}: the loader built them for {}",
                model.config.rotary_ndims
            )));
        }
        v.push(RopeTableSet { rotary_ndims: rn, cos: &model.cos_cache, sin: &model.sin_cache });
    }
    Ok(v)
}
