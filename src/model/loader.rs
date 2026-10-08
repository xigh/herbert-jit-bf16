//! The checkpoint read into the model structures — Qwen3.5, text only.
//!
//! What enters: the model directory (`config.json`, one or more
//! `*.safetensors` shards, their index when sharded) and a packer: the function
//! that turns a row-major bf16 matrix into the weight the engine reads.
//! What comes out: the `Config` and a `Model` — the embedding table, the 32
//! decoder layers (attention or GatedDeltaNet mixer, then a SwiGLU MLP), the
//! final norm, the language head (the embedding table again when it is tied)
//! and the RoPE tables.
//!
//! Every matrix stays in bf16: it is mapped from the shard, packed, and that
//! is all. The small vectors (norms, the DeltaNet gates, decay and bias)
//! are read as f32 or bf16 exactly as the reference implementation keeps
//! them. Tensors the engine does not use — the multi-token-prediction head
//! `mtp.*`, the vision tower — are skipped and counted.

use crate::core::avec::AVec;
use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::core::tensor::{bf16_to_f32, f32_to_bf16, BF16};
use crate::model::model::{Attention, DecoderLayer, GatedDeltaNet, LayerBlock, Mlp, Model};
use crate::model::weight::PackedBf16R16Weight;
use memmap2::Mmap;
use safetensors::tensor::TensorView;
use safetensors::SafeTensors;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// A Hugging Face cache directory points at a snapshot; a plain directory is
/// used as is.
pub fn resolve_model_dir(model_dir: &Path) -> Result<PathBuf> {
    if model_dir.join("config.json").exists() {
        return Ok(model_dir.to_path_buf());
    }
    let snapshots = model_dir.join("snapshots");
    if !snapshots.is_dir() {
        return Ok(model_dir.to_path_buf());
    }
    if let Ok(reference) = std::fs::read_to_string(model_dir.join("refs").join("main")) {
        let candidate = snapshots.join(reference.trim());
        if candidate.join("config.json").exists() {
            return Ok(candidate);
        }
    }
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&snapshots)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.join("config.json").exists())
        .collect();
    candidates.sort();
    candidates.pop().ok_or_else(|| {
        HerbertError::ModelLoad(format!("no snapshot with a config.json under {}", snapshots.display()))
    })
}

/// A bf16 vector from a tensor view; f32 and f16 are rounded to nearest even.
pub fn load_bf16_from_view(view: &TensorView<'_>, name: &str) -> Result<Vec<BF16>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::BF16 => Ok(bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes([b[0], b[1]])).collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .as_chunks::<2>().0.iter()
            .map(|b| f32_to_bf16(half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32()))
            .collect()),
        safetensors::Dtype::F32 => Ok(bytes
            .as_chunks::<4>().0.iter()
            .map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])))
            .collect()),
        dtype => Err(HerbertError::ModelLoad(format!("unsupported dtype {dtype:?} for {name}"))),
    }
}

/// An f32 vector from a tensor view; bf16 and f16 are widened exactly.
pub fn load_f32_from_view(view: &TensorView<'_>, name: &str) -> Result<Vec<f32>> {
    let bytes = view.data();
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()),
        safetensors::Dtype::F16 => Ok(bytes
            .as_chunks::<2>().0.iter()
            .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
            .collect()),
        safetensors::Dtype::BF16 => Ok(bytes.as_chunks::<2>().0.iter().map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()),
        dtype => Err(HerbertError::ModelLoad(format!("unsupported dtype {dtype:?} for {name}"))),
    }
}

/// The RoPE tables, `[max_position_embeddings, rotary_ndims / 2]` each:
/// `cos(pos · θ^(−2i/rotary_ndims))` and its sine. Qwen3.5 declares an MRoPE
/// (three position axes for images); on text the three axes carry the same
/// position and the interleaved pattern is inert, so the tables are the
/// ordinary ones.
pub fn compute_rope_cache(config: &Config) -> (Vec<f32>, Vec<f32>) {
    let rotary_ndims = config.rotary_ndims;
    let half_dim = rotary_ndims / 2;
    let positions = config.max_position_embeddings;
    let inv_freq: Vec<f32> = (0..half_dim)
        .map(|i| 1.0 / config.rope_theta.powf((2 * i) as f32 / rotary_ndims as f32))
        .collect();
    let mut cos = vec![0.0f32; positions * half_dim];
    let mut sin = vec![0.0f32; positions * half_dim];
    for pos in 0..positions {
        for i in 0..half_dim {
            let angle = pos as f32 * inv_freq[i];
            cos[pos * half_dim + i] = angle.cos();
            sin[pos * half_dim + i] = angle.sin();
        }
    }
    (cos, sin)
}

/// What a tensor name means to this model.
#[derive(Debug)]
enum Kind {
    Embedding,
    FinalNorm,
    LmHead,
    /// A layer's norm or small vector, kept bf16.
    Bf16 { layer: usize, field: &'static str },
    /// A layer's small vector, kept f32 (the DeltaNet gates, conv, decay, bias).
    F32 { layer: usize, field: &'static str },
    /// A layer's matrix, packed for the GEMM.
    Matrix { layer: usize, field: &'static str },
    Skipped,
}

/// The tensor names of a Qwen3.5 text checkpoint. The prefix varies with the
/// packaging (`model.`, `model.language_model.`, `language_model.model.`).
fn classify(name: &str) -> Kind {
    let name = name
        .strip_prefix("language_model.model.")
        .or_else(|| name.strip_prefix("model.language_model."))
        .or_else(|| name.strip_prefix("model."))
        .map(|rest| format!("model.{rest}"))
        .unwrap_or_else(|| name.to_string());
    match name.as_str() {
        "model.embed_tokens.weight" => return Kind::Embedding,
        "model.norm.weight" => return Kind::FinalNorm,
        "lm_head.weight" => return Kind::LmHead,
        _ => {}
    }
    let Some(rest) = name.strip_prefix("model.layers.") else {
        return Kind::Skipped;
    };
    let Some((index, field)) = rest.split_once('.') else {
        return Kind::Skipped;
    };
    let Ok(layer) = index.parse::<usize>() else {
        return Kind::Skipped;
    };
    match field {
        "input_layernorm.weight" => Kind::Bf16 { layer, field: "input_layernorm" },
        "post_attention_layernorm.weight" => Kind::Bf16 { layer, field: "post_attention_layernorm" },
        "self_attn.q_proj.weight" => Kind::Matrix { layer, field: "q_proj" },
        "self_attn.k_proj.weight" => Kind::Matrix { layer, field: "k_proj" },
        "self_attn.v_proj.weight" => Kind::Matrix { layer, field: "v_proj" },
        "self_attn.o_proj.weight" => Kind::Matrix { layer, field: "o_proj" },
        "self_attn.q_norm.weight" => Kind::Bf16 { layer, field: "q_norm" },
        "self_attn.k_norm.weight" => Kind::Bf16 { layer, field: "k_norm" },
        "mlp.gate_proj.weight" => Kind::Matrix { layer, field: "gate_proj" },
        "mlp.up_proj.weight" => Kind::Matrix { layer, field: "up_proj" },
        "mlp.down_proj.weight" => Kind::Matrix { layer, field: "down_proj" },
        "linear_attn.in_proj_qkv.weight" => Kind::Matrix { layer, field: "in_proj_qkv" },
        "linear_attn.in_proj_z.weight" => Kind::Matrix { layer, field: "in_proj_z" },
        "linear_attn.out_proj.weight" => Kind::Matrix { layer, field: "out_proj" },
        "linear_attn.in_proj_a.weight" => Kind::F32 { layer, field: "in_proj_a" },
        "linear_attn.in_proj_b.weight" => Kind::F32 { layer, field: "in_proj_b" },
        "linear_attn.conv1d.weight" => Kind::F32 { layer, field: "conv1d" },
        "linear_attn.A_log" => Kind::F32 { layer, field: "a_log" },
        "linear_attn.dt_bias" => Kind::F32 { layer, field: "dt_bias" },
        "linear_attn.norm.weight" => Kind::Bf16 { layer, field: "norm" },
        _ => Kind::Skipped,
    }
}

/// One tensor, read and converted, before it joins its layer.
enum Loaded {
    Embedding(Vec<BF16>),
    FinalNorm(Vec<BF16>),
    LmHead(Vec<BF16>),
    Bf16 { layer: usize, field: &'static str, data: Vec<BF16> },
    F32 { layer: usize, field: &'static str, data: Vec<f32> },
    Matrix { layer: usize, field: &'static str, weight: PackedBf16R16Weight },
    Skipped(String),
}

/// The tensors of one layer, collected across shards, then assembled.
#[derive(Default)]
struct PartialLayer {
    bf16: HashMap<&'static str, Vec<BF16>>,
    f32: HashMap<&'static str, Vec<f32>>,
    matrices: HashMap<&'static str, PackedBf16R16Weight>,
}

impl PartialLayer {
    /// The vectors of a layer (norms, DeltaNet parameters), each on a cache line.
    fn take_bf16(&mut self, layer: usize, field: &str) -> Result<AVec<BF16>> {
        self.bf16.remove(field).map(AVec::from_vec).ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }
    fn take_f32(&mut self, layer: usize, field: &str) -> Result<AVec<f32>> {
        self.f32.remove(field).map(AVec::from_vec).ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }
    fn take_matrix(&mut self, layer: usize, field: &str) -> Result<PackedBf16R16Weight> {
        self.matrices.remove(field).ok_or_else(|| HerbertError::ModelLoad(format!("layer {layer}: missing {field}")))
    }

    /// The decoder layer: its mixer (attention or GatedDeltaNet, decided by the
    /// config's layer types) and its dense SwiGLU MLP.
    fn finalize(mut self, layer: usize, config: &Config) -> Result<DecoderLayer> {
        let block = if config.is_deltanet_layer(layer) {
            LayerBlock::GatedDeltaNet(GatedDeltaNet {
                in_proj_qkv: self.take_matrix(layer, "in_proj_qkv")?,
                in_proj_z: self.take_matrix(layer, "in_proj_z")?,
                in_proj_a: self.take_f32(layer, "in_proj_a")?,
                in_proj_b: self.take_f32(layer, "in_proj_b")?,
                conv1d_weight: self.take_f32(layer, "conv1d")?,
                a_log: self.take_f32(layer, "a_log")?,
                dt_bias: self.take_f32(layer, "dt_bias")?,
                norm_weight: self.take_bf16(layer, "norm")?,
                out_proj: self.take_matrix(layer, "out_proj")?,
                num_k_heads: config.linear_num_key_heads,
                num_v_heads: config.linear_num_value_heads,
                head_k_dim: config.linear_key_head_dim,
                head_v_dim: config.linear_value_head_dim,
                conv_kernel_size: config.linear_conv_kernel_dim,
            })
        } else {
            LayerBlock::Attention(Attention {
                q_proj: self.take_matrix(layer, "q_proj")?,
                k_proj: self.take_matrix(layer, "k_proj")?,
                v_proj: self.take_matrix(layer, "v_proj")?,
                o_proj: self.take_matrix(layer, "o_proj")?,
                q_norm: self.take_bf16(layer, "q_norm")?,
                k_norm: self.take_bf16(layer, "k_norm")?,
                num_heads: config.num_attention_heads,
                num_kv_heads: config.num_key_value_heads,
                head_dim: config.head_dim,
                rotary_ndims: config.rotary_ndims,
            })
        };
        Ok(DecoderLayer {
            input_layernorm: self.take_bf16(layer, "input_layernorm")?,
            block,
            post_attention_layernorm: self.take_bf16(layer, "post_attention_layernorm")?,
            mlp: Mlp {
                gate_proj: self.take_matrix(layer, "gate_proj")?,
                up_proj: self.take_matrix(layer, "up_proj")?,
                down_proj: self.take_matrix(layer, "down_proj")?,
                intermediate_size: config.intermediate_size,
            },
        })
    }
}

/// Read the checkpoint. `pack(matrix, rows, columns, name)` turns each
/// row-major bf16 matrix into the engine's weight; it runs on several threads,
/// one slice of the shard's tensors each.
pub fn load_model(
    model_dir: &Path,
    pack: impl Fn(&[BF16], usize, usize, &str) -> Result<PackedBf16R16Weight> + Send + Sync,
    show_progress: bool,
) -> Result<(Config, Model)> {
    let model_dir = resolve_model_dir(model_dir)?;
    let config = Config::from_file(&model_dir.join("config.json"))?;

    // The shards, and which tensor lives in which one.
    let index_path = model_dir.join("model.safetensors.index.json");
    let (shards, tensor_shard): (Vec<String>, Option<HashMap<String, String>>) = if index_path.exists() {
        #[derive(serde::Deserialize)]
        struct Index {
            weight_map: HashMap<String, String>,
        }
        let index: Index = serde_json::from_slice(&std::fs::read(&index_path)?)?;
        let names: HashSet<String> = index.weight_map.values().map(|s| shard_file_name(s).to_string()).collect();
        let mut shards: Vec<String> = names.into_iter().collect();
        shards.sort();
        let map = index.weight_map.iter().map(|(t, s)| (t.clone(), shard_file_name(s).to_string())).collect();
        (shards, Some(map))
    } else {
        (vec!["model.safetensors".to_string()], None)
    };

    let mut embed_tokens: Option<Vec<BF16>> = None;
    let mut final_norm: Option<Vec<BF16>> = None;
    let mut lm_head_raw: Option<Vec<BF16>> = None;
    let mut layers: Vec<PartialLayer> = (0..config.num_layers).map(|_| PartialLayer::default()).collect();
    let mut skipped: BTreeMap<String, usize> = BTreeMap::new();
    let started = Instant::now();

    for (shard_index, shard) in shards.iter().enumerate() {
        let shard_started = Instant::now();
        let file = std::fs::File::open(model_dir.join(shard))
            .map_err(|e| HerbertError::ModelLoad(format!("cannot open shard {shard}: {e}")))?;
        // SAFETY: the file is read-only and not modified while mapped.
        let mapped = unsafe { Mmap::map(&file) }.map_err(|e| HerbertError::ModelLoad(format!("cannot map shard {shard}: {e}")))?;
        let tensors = SafeTensors::deserialize(&mapped[..])
            .map_err(|e| HerbertError::ModelLoad(format!("cannot parse shard {shard}: {e}")))?;
        let names: Vec<String> = match &tensor_shard {
            Some(map) => map.iter().filter(|(_, s)| *s == shard).map(|(t, _)| t.clone()).collect(),
            None => tensors.names().into_iter().map(|s| s.to_string()).collect(),
        };

        // Packing is CPU-bound: the shard's tensors are split across threads.
        let workers = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(names.len().max(1));
        let chunk = names.len().div_ceil(workers);
        let read_one = |name: &str| -> Result<Loaded> {
            let view = tensors
                .tensor(name)
                .map_err(|e| HerbertError::ModelLoad(format!("tensor {name} missing from {shard}: {e}")))?;
            Ok(match classify(name) {
                Kind::Embedding => Loaded::Embedding(load_bf16_from_view(&view, name)?),
                Kind::FinalNorm => Loaded::FinalNorm(load_bf16_from_view(&view, name)?),
                Kind::LmHead => Loaded::LmHead(load_bf16_from_view(&view, name)?),
                Kind::Bf16 { layer, field } => Loaded::Bf16 { layer, field, data: load_bf16_from_view(&view, name)? },
                Kind::F32 { layer, field } => Loaded::F32 { layer, field, data: load_f32_from_view(&view, name)? },
                Kind::Matrix { layer, field } => {
                    let shape = view.shape();
                    if shape.len() != 2 {
                        return Err(HerbertError::ModelLoad(format!("{name}: a matrix has 2 dimensions, not {shape:?}")));
                    }
                    let data = load_bf16_from_view(&view, name)?;
                    Loaded::Matrix { layer, field, weight: pack(&data, shape[0], shape[1], name)? }
                }
                Kind::Skipped => Loaded::Skipped(name.split('.').take(2).collect::<Vec<_>>().join(".")),
            })
        };
        let loaded: Vec<Loaded> = std::thread::scope(|scope| -> Result<Vec<Loaded>> {
            let handles: Vec<_> = names
                .chunks(chunk.max(1))
                .map(|slice| scope.spawn(move || slice.iter().map(|n| read_one(n)).collect::<Result<Vec<_>>>()))
                .collect();
            let mut all = Vec::with_capacity(names.len());
            for handle in handles {
                all.extend(handle.join().map_err(|_| HerbertError::ModelLoad("a loader thread panicked".into()))??);
            }
            Ok(all)
        })?;

        for item in loaded {
            match item {
                Loaded::Embedding(v) => embed_tokens = Some(v),
                Loaded::FinalNorm(v) => final_norm = Some(v),
                Loaded::LmHead(v) => lm_head_raw = Some(v),
                Loaded::Bf16 { layer, field, data } => {
                    layers.get_mut(layer).ok_or_else(|| out_of_range(layer, &config))?.bf16.insert(field, data);
                }
                Loaded::F32 { layer, field, data } => {
                    layers.get_mut(layer).ok_or_else(|| out_of_range(layer, &config))?.f32.insert(field, data);
                }
                Loaded::Matrix { layer, field, weight } => {
                    layers.get_mut(layer).ok_or_else(|| out_of_range(layer, &config))?.matrices.insert(field, weight);
                }
                Loaded::Skipped(prefix) => *skipped.entry(prefix).or_default() += 1,
            }
        }
        if show_progress {
            eprintln!(
                "[loader] shard {}/{}: {} tensors, {:.2} GiB mapped, {:.1} s",
                shard_index + 1,
                shards.len(),
                names.len(),
                mapped.len() as f64 / (1u64 << 30) as f64,
                shard_started.elapsed().as_secs_f64()
            );
        }
    }
    if !skipped.is_empty() {
        let list: Vec<String> = skipped.iter().map(|(p, n)| format!("{p}.* ×{n}")).collect();
        eprintln!("[loader] skipped, not used by the engine: {}", list.join(", "));
    }

    let embed_tokens = embed_tokens.ok_or_else(|| HerbertError::ModelLoad("missing model.embed_tokens.weight".into()))?;
    let norm = final_norm.ok_or_else(|| HerbertError::ModelLoad("missing model.norm.weight".into()))?;
    // The language head: its own matrix, or the embedding table when tied.
    let lm_head = match lm_head_raw {
        Some(raw) => pack(&raw, config.vocab_size, config.hidden_size, "lm_head.weight")?,
        None if config.tie_word_embeddings => {
            pack(&embed_tokens, config.vocab_size, config.hidden_size, "lm_head.weight (tied)")?
        }
        None => return Err(HerbertError::ModelLoad("missing lm_head.weight and tie_word_embeddings is false".into())),
    };
    let decoder_layers: Vec<DecoderLayer> = layers
        .into_iter()
        .enumerate()
        .map(|(l, partial)| partial.finalize(l, &config))
        .collect::<Result<_>>()?;
    let (cos_cache, sin_cache) = compute_rope_cache(&config);
    if show_progress {
        eprintln!("[loader] {} layers assembled, RoPE tables built, {:.1} s in all", decoder_layers.len(), started.elapsed().as_secs_f64());
    }
    let model = Model {
        embed_tokens: AVec::from_vec(embed_tokens),
        decoder_layers,
        norm: AVec::from_vec(norm),
        lm_head,
        config: config.clone(),
        cos_cache: AVec::from_vec(cos_cache),
        sin_cache: AVec::from_vec(sin_cache),
    };
    Ok((config, model))
}

fn shard_file_name(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn out_of_range(layer: usize, config: &Config) -> HerbertError {
    HerbertError::ModelLoad(format!("tensor of layer {layer} but the config declares {} layers", config.num_layers))
}
