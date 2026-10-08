//! The engine: the model mounted into the two generated regimes and driven
//! step by step.
//!
//! What enters: a model directory, the thread count, the number of positions
//! to reserve, the tuning and measurement settings.
//! What it does at `load`: reads the checkpoint into packed bf16 weights,
//! checks the geometry, builds the `Runner` (KV cache, DeltaNet states, RoPE
//! tables, compiled kernels) and mounts both regimes — prefill and decode —
//! as two generated programs. Nothing is allocated after that.
//! What the chat calls afterwards: `prefill` (append rows to the live state),
//! `generate` (decode in one launch, one callback per token), `snapshot` and
//! `restore` (the whole state, for the disk), `write_trace` (the measurements).

use crate::core::config::Config;
use crate::core::error::{HerbertError, Result};
use crate::engine::geometry::{self, Geometry};
use crate::engine::packs::DeltaPacks;
use crate::engine::tuning::{Measure, Tuning};
use crate::jit::execution::MountOptions;
use crate::jit::{
    engine_supported, prefill_pass, Instantaneous, LayerWeights, ModelBudget,
    ModelWeights, Runner, DEFAULT_MAX_BATCH,
};
use crate::model::Model;
use std::path::Path;
use std::time::Instant;

/// Rows one prefill pass hands to the generated program: the pass width, capped
/// by the widest block the frame can hold.
pub fn causal_batch_width() -> usize {
    prefill_pass().min(DEFAULT_MAX_BATCH)
}

/// The loaded, mounted model. Field order matters: `layers` borrows into
/// `model` and `delta_packs`, so it is declared first and dropped first.
pub struct Engine {
    layers: Vec<LayerWeights<'static>>,
    model: Box<Model>,
    #[allow(dead_code)]
    delta_packs: Vec<Option<DeltaPacks>>,
    config: Config,
    geo: Geometry,
    runner: Runner,
    /// Positions the runner reserved: the ceiling of a conversation.
    reserved: usize,
    /// Wall times of the two phases, for the log.
    pub load_seconds: f64,
    pub mount_seconds: f64,
}

impl Engine {
    /// Read, check, reserve and mount. `reserve_positions` bounds the whole
    /// conversation (prompt and every generated token); it is capped by the
    /// model's RoPE tables. `tuning` and `measure` are fixed for the life of
    /// the engine: both regimes are lowered with them.
    pub fn load(model_dir: &Path, threads: usize, reserve_positions: usize, tuning: &Tuning, measure: &Measure) -> Result<Engine> {
        if !engine_supported() {
            return Err(HerbertError::Backend(
                "the ISA floor is not met: AVX-512 (+BF16) or AVX2+FMA is required".to_string(),
            ));
        }
        // The knobs of the emitters and of the pool, before either exists.
        tuning.install()?;
        if threads > 0 {
            crate::pool::thread_pool::configure_global_pool(threads)
                .map_err(|e| HerbertError::Backend(format!("thread pool: {e}")))?;
        }
        // Refuse before streaming what the machine cannot hold: the checkpoint is
        // bf16 and stays bf16, so its size on disk is the floor of the weights.
        crate::core::memory::refuse_if_it_cannot_fit("weights", crate::core::memory::checkpoint_bytes(model_dir))?;
        let started = Instant::now();
        let (config, model) = crate::engine::loader::load_bf16_model(model_dir, true)?;
        let model = Box::new(model);
        let geo = geometry::geometry(&config, &model)?;
        let delta_packs = crate::engine::packs::delta_packs(&model)?;
        // The borrowed slices point into heap buffers that are never moved or
        // written again; `layers` is declared before them and dropped first.
        let layers: Vec<LayerWeights<'static>> = unsafe {
            std::mem::transmute(crate::engine::weights_of::layer_weights(&model, &delta_packs))
        };
        let load_seconds = started.elapsed().as_secs_f64();

        let mounted = Instant::now();
        let geometries = geometry::attention_geometries(&model.config, &model);
        let plan_kv = geometry::plan_kv(&model);
        let tables = geometry::tables_rope(&model, &geometries)?;
        // The RoPE tables bound the context: never reserve beyond them.
        let positions = tables
            .iter()
            .map(|t| {
                let half = (t.rotary_ndims / 2).max(1);
                (t.cos.len() / half).min(t.sin.len() / half)
            })
            .min()
            .unwrap_or(0);
        let reserved = reserve_positions.min(positions);
        let budget = ModelBudget { kv_reserve_tokens: reserved, vocab: geo.vocab, max_batch: causal_batch_width() };
        let widths = geometry::mlp_widths(&model);
        // Keys and values, bf16, on every layer that owns a cache, at the model's KV head count.
        let kv_bytes: u64 = plan_kv
            .iter()
            .map(|p| match p {
                crate::jit::PlanKv::Own(head_dim) => (4 * geo.shape.n_kv_heads * reserved * head_dim) as u64,
                crate::jit::PlanKv::Absent => 0,
            })
            .sum();
        crate::core::memory::refuse_if_it_cannot_fit("KV cache", kv_bytes)?;
        let mut runner = Runner::new_with_geometries(
            geo.shape,
            geo.n_layers,
            budget,
            &tables,
            &widths,
            &geometries,
            &plan_kv,
            geo.delta,
            tuning.executor_options(),
        )?;
        {
            let weights = Self::borrow_weights(&model, &layers);
            runner.mount_execution(
                &weights,
                MountOptions {
                    trace_sync: measure.sync_trace,
                    pmu: measure.pmu,
                    shared_barrier: tuning.shared_barrier,
                    fuse_serial: tuning.fuse_serial,
                    trace_blocks: measure.trace_blocks,
                },
            )?;
        }
        let mount_seconds = mounted.elapsed().as_secs_f64();
        let faulted = Instant::now();
        let touched = Self::fault_in(&model, &layers, threads.max(1));
        eprintln!("[engine] {:.2} GiB of weights touched in {:.3} s", touched as f64 / (1u64 << 30) as f64, faulted.elapsed().as_secs_f64());
        eprintln!(
            "[engine] loaded in {load_seconds:.1} s, mounted in {mount_seconds:.2} s — {reserved} positions reserved, \
             prefill program {:?}, decode program {:?}",
            runner.execution_code_bytes(true),
            runner.execution_code_bytes(false)
        );
        Ok(Engine {
            layers,
            model,
            delta_packs,
            config,
            geo,
            runner,
            reserved,
            load_seconds,
            mount_seconds,
        })
    }

    /// Reads one word per 4 KiB of every matrix, `threads` at a time: a page
    /// the system compressed or never mapped is brought back here rather than
    /// during the first prompt. On the M5, with the swap nearly full, macOS
    /// compresses the packed weights as soon as they are written; the first
    /// prefill then spent 0.9 s decompressing them one fault at a time (1.1 s
    /// for 20 tokens, 0.19 s once touched here, in 0.4 s). Returns the bytes
    /// covered.
    fn fault_in(model: &Model, layers: &[LayerWeights<'_>], threads: usize) -> usize {
        let mut slices: Vec<&[u16]> = vec![&model.lm_head.data_r16];
        for l in layers {
            slices.extend([l.wq, l.wk, l.wv, l.wo, l.w_gate, l.w_up, l.w_down]);
            if let crate::jit::Mixer::GatedDeltaNet(d) = &l.mixer {
                slices.extend([d.in_proj_qkv, d.in_proj_z, d.in_proj_a, d.in_proj_b, d.out_proj]);
            }
        }
        let bytes = slices.iter().map(|s| s.len() * 2).sum();
        // The workers of the pool, configured just before, each taking the
        // next matrix: no threads of its own.
        let next = std::sync::atomic::AtomicUsize::new(0);
        let pool = crate::pool::thread_pool::global_pool();
        let touch = |_: usize| loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let Some(slice) = slices.get(i) else { break };
            let mut sum = 0u16;
            for j in (0..slice.len()).step_by(2048) {
                sum = sum.wrapping_add(unsafe { std::ptr::read_volatile(&slice[j]) });
            }
            std::hint::black_box(sum);
        };
        if pool.collective(threads.min(pool.num_workers()).max(1), touch, || {}).is_err() {
            // A pool that cannot run a collective leaves the pages to the first prompt.
            return 0;
        }
        bytes
    }

    fn borrow_weights<'a>(
        model: &'a Model,
        layers: &'a [LayerWeights<'a>],
    ) -> ModelWeights<'a> {
        ModelWeights {
            embed: &model.embed_tokens,
            layers,
            final_norm: &model.norm,
            lm_head: &model.lm_head.data_r16,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geo
    }

    pub fn runner(&self) -> &Runner {
        &self.runner
    }

    /// Positions already in the state: prompt rows and fed tokens.
    pub fn pos(&self) -> usize {
        self.runner.pos()
    }

    /// The ceiling of a conversation, in positions.
    pub fn reserved(&self) -> usize {
        self.reserved
    }

    /// Back to position zero: KV cache, DeltaNet states and position cleared.
    pub fn reset(&mut self) {
        self.runner.reset();
    }

    /// Append `tokens` to the live state from the current position, and return
    /// the greedy token of the last row. The second turn of a conversation calls
    /// this with only its new rows.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        if tokens.is_empty() {
            return Err(HerbertError::Backend("prefill of zero tokens".to_string()));
        }
        if self.runner.pos() + tokens.len() > self.reserved {
            return Err(HerbertError::Backend(format!(
                "the conversation exceeds the {} reserved positions ({} in the state, {} to add)",
                self.reserved,
                self.runner.pos(),
                tokens.len()
            )));
        }
        let weights = Self::borrow_weights(&self.model, &self.layers);
        self.runner.prefill_mounted(&weights, tokens, None)?;
        Ok(self.argmax())
    }

    /// Decode in one launch of the generated program: `first` is fed to block 0,
    /// then `step(argmax, logits_bf16)` is called at the end of every block and
    /// returns the token to feed next, or `None` to stop. Returns the number of
    /// blocks run.
    pub fn generate(
        &mut self,
        first: u32,
        max_steps: usize,
        step: &mut dyn FnMut(u32, &[u16]) -> Result<Option<u32>>,
    ) -> Result<usize> {
        let room = self.reserved.saturating_sub(self.runner.pos());
        let max_steps = max_steps.min(room);
        if max_steps == 0 {
            return Ok(0);
        }
        let weights = Self::borrow_weights(&self.model, &self.layers);
        self.runner.generate_mounted(&weights, first, max_steps, step)
    }

    /// The last logits row, bf16 bits, `vocab_size` long.
    pub fn logits(&self) -> &[u16] {
        &self.runner.logits()[..self.config.vocab_size]
    }

    /// The greedy token of the last logits row.
    pub fn argmax(&self) -> u32 {
        self.runner.execution_argmax().unwrap_or_else(|| argmax_bf16(self.logits()))
    }

    /// The whole state at the current position, frozen.
    pub fn snapshot(&self) -> Result<Instantaneous> {
        let mut slot = self.runner.snapshot_reserve(self.reserved);
        self.runner.snapshot_take(&mut slot)?;
        Ok(slot)
    }

    /// Replace the state by `slot`, rewound to position `p`.
    pub fn restore(&mut self, slot: &Instantaneous, p: usize) -> Result<()> {
        self.runner.reset();
        self.runner.restore_from(slot, p)
    }

    /// The trace of the latest call of one regime, when it was mounted with
    /// measurements; nothing otherwise.
    pub fn write_trace(&self, prefill: bool, path: &Path) -> Result<()> {
        crate::engine::trace::write_execution_trace(&self.runner, prefill, path)
            .map_err(|e| HerbertError::Backend(format!("trace {}: {e}", path.display())))
    }
}

/// Argmax on bf16 logits, comparing the reconstructed f32 values.
pub fn argmax_bf16(logits: &[u16]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &b) in logits.iter().enumerate() {
        let v = f32::from_bits((b as u32) << 16);
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}
