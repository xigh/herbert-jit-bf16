//! The checkpoint, read into bf16 weights and nothing else.
//!
//! What enters: the model directory (`config.json`, `*.safetensors`).
//! What comes out: the `Config` and a `Model` whose every matrix is
//! packed in the R16 layout the emitted GEMM reads (`model/weight.rs`), in
//! bf16, exactly the bytes of the checkpoint. No quantization, no cache on
//! disk: the shards are mapped and repacked at every start.
//!
//! One matrix is reordered on the way, not converted: on a model with the fused
//! attention gate (`attn_output_gate`, Qwen3.5), `q_proj` holds `[Q; gate]`
//! rows interleaved per head, and the scheduler expects Q compact then gate
//! compact. The permutation is `crate::jit::schedule::permute_wq_rows`, applied
//! here and nowhere else, to every gated `q_proj`.

use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::model::loader;
use crate::model::{Model, PackedBf16R16Weight};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Mutex;

/// What the loader did, counted as it goes and printed once: the proof, in
/// the log, that nothing was quantized.
#[derive(Default)]
struct Census {
    tensors: usize,
    bytes: usize,
    permuted_wq: usize,
    /// Tensor kinds, layer index stripped, with their count.
    kinds: BTreeMap<String, usize>,
}

impl Census {
    fn kind(name: &str) -> String {
        match name.split(".layers.").nth(1) {
            Some(rest) => match rest.split_once('.') {
                Some((_, tail)) => tail.to_string(),
                None => rest.to_string(),
            },
            None => name.to_string(),
        }
    }

    fn report(&self) {
        let kinds: Vec<String> = self.kinds.iter().map(|(k, n)| format!("{k}×{n}")).collect();
        eprintln!(
            "[loader] bf16 everywhere: {} matrices packed R16 ({:.2} GiB), {} gated q_proj permuted, no Q4, no cache — {}",
            self.tensors,
            self.bytes as f64 / (1u64 << 30) as f64,
            self.permuted_wq,
            kinds.join(", ")
        );
    }
}

/// Read the model: every matrix becomes a packed bf16 R16 weight.
pub fn load_bf16_model(model_dir: &Path, show_progress: bool) -> Result<(Config, Model)> {
    let config = Config::from_file(&model_dir.join("config.json"))?;
    eprintln!(
        "[loader] {} — hidden {}, {} layers, {} heads / {} kv heads × {}, intermediate {}, vocab {}, rotary {} dims, rope_theta {}, tied head {}",
        config.model_type,
        config.hidden_size,
        config.num_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
        config.rotary_ndims,
        config.rope_theta,
        config.tie_word_embeddings
    );
    let permute_wq = config.attn_output_gate;
    let (n_heads, head_dim) = (config.num_attention_heads, config.head_dim);
    let census = Mutex::new(Census::default());
    let (config, model) = loader::load_model(
        model_dir,
        |bf16_data, out_f, in_f, name| {
            let permuted;
            let data = if permute_wq && name.contains(".layers.") && name.ends_with("q_proj.weight") {
                permuted = crate::jit::schedule::permute_wq_rows(bf16_data, n_heads, head_dim, in_f)
                    .ok_or_else(|| {
                        HerbertError::ModelLoad(format!(
                            "{name}: {out_f}×{in_f} is not the 2·{n_heads}·{head_dim} rows of a gated q_proj"
                        ))
                    })?;
                census.lock().unwrap().permuted_wq += 1;
                permuted.as_slice()
            } else {
                bf16_data
            };
            {
                let mut c = census.lock().unwrap();
                c.tensors += 1;
                c.bytes += 2 * data.len();
                *c.kinds.entry(Census::kind(name)).or_default() += 1;
            }
            PackedBf16R16Weight::pack_from_row_major_r16_only(data, out_f, in_f)
        },
        show_progress,
    )?;
    census.lock().unwrap().report();
    Ok((config, model))
}
