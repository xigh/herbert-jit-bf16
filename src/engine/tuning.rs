//! What `main` decides about the engine, as named fields.
//!
//! Nothing in the engine reads the environment: every setting is a field of
//! one of these two structs, built in `main` with its default here, and a
//! sentence that says what it changes. `Engine::load` installs `Tuning`
//! once per process before the pool is created and before any kernel is
//! compiled, so the emitters and the pool read the same values.
//!
//! `Tuning` is about speed and costs nothing to read; `Measure` is about
//! looking inside a call and costs a little (stamps and counters per node).

use crate::core::error::{HerbertError, Result};
use crate::jit::execution::{Executor, ExecutorOptions, PmuEvent};
pub use crate::jit::knobs::Knobs;
pub use crate::pool::settings::PoolSettings;

/// The knobs of the generated program, of the emitters and of the pool.
#[derive(Clone, Debug, PartialEq)]
pub struct Tuning {
    /// The executor of each regime: spin barriers (every participant busy-waits
    /// at a barrier) or hybrid (spin, then sleep after `hybrid_spin` pauses).
    pub prefill_executor: Executor,
    pub decode_executor: Executor,
    /// Participants of the prefill; `None` = every worker of the pool.
    pub prefill_workers: Option<usize>,
    /// Participants of the decode; `None` = every worker of the pool. By
    /// default sixteen on a Mac with more than sixteen performance cores,
    /// the whole pool elsewhere (`cpu_detection::default_decode_workers`:
    /// M3 Ultra, pool of 20, 27.37 → 27.69 tokens/s).
    pub decode_workers: Option<usize>,
    /// Pauses before a hybrid participant sleeps.
    pub hybrid_spin: u32,
    /// One barrier word shared by every node instead of one per node.
    pub shared_barrier: bool,
    /// Fuse the barriers around serial nodes (review 035: no gain measured).
    pub fuse_serial: bool,
    /// The emitters and the lowering: pass width, tile, SDPA, DeltaNet, teams.
    pub kernels: Knobs,
    /// The worker pool: spin, pinning, who computes.
    pub pool: PoolSettings,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            prefill_executor: Executor::SpmdSpin,
            decode_executor: Executor::SpmdSpin,
            prefill_workers: None,
            decode_workers: crate::core::cpu_detection::default_decode_workers(),
            hybrid_spin: 512,
            shared_barrier: false,
            fuse_serial: false,
            kernels: Knobs::default(),
            pool: PoolSettings::default(),
        }
    }
}

impl Tuning {
    pub fn executor_options(&self) -> ExecutorOptions {
        ExecutorOptions {
            prefill: self.prefill_executor,
            decode: self.decode_executor,
            prefill_workers: self.prefill_workers,
            decode_workers: self.decode_workers,
            hybrid_spin: self.hybrid_spin,
        }
    }

    /// Installs the kernel knobs and the pool settings for the process.
    pub fn install(&self) -> Result<()> {
        crate::jit::knobs::install(self.kernels).map_err(HerbertError::Backend)?;
        crate::pool::settings::install(self.pool.clone()).map_err(HerbertError::Backend)
    }
}

/// What to record inside a call.
#[derive(Clone, Copy, Debug, Default)]
pub struct Measure {
    /// `rdtscp` stamps per node and participant: work, progress, barrier, sleep.
    pub sync_trace: bool,
    /// One hardware counter read by `rdpmc` around every node (Linux, user RDPMC).
    pub pmu: Option<PmuEvent>,
    /// Blocks kept in the trace buffers: a prefill pass is one block, a
    /// decoded token is one block.
    pub trace_blocks: usize,
}

impl Measure {
    pub fn none() -> Self {
        Measure::default()
    }

    pub fn is_on(&self) -> bool {
        self.sync_trace || self.pmu.is_some()
    }

    /// Time stamps only.
    pub fn sync(trace_blocks: usize) -> Self {
        Measure { sync_trace: true, pmu: None, trace_blocks }
    }

    /// One PMU event, with the stamps.
    pub fn pmu(event: PmuEvent, trace_blocks: usize) -> Self {
        Measure { sync_trace: true, pmu: Some(event), trace_blocks }
    }
}
