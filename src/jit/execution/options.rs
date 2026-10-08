use std::fmt;

/// Independently selectable prefill/decode executor: the generated program,
/// with one of two barrier policies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Executor {
    /// Generated driver with generation-counted spin barriers.
    #[default]
    SpmdSpin,
    /// Generated driver with bounded spin then runtime waiting.
    SpmdHybrid,
}

impl std::str::FromStr for Executor {
    type Err = OptionError;
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "spmd-spin" => Ok(Self::SpmdSpin),
            "spmd-hybrid" => Ok(Self::SpmdHybrid),
            _ => Err(OptionError(format!("unknown JIT executor {value:?}; expected spmd-spin or spmd-hybrid"))),
        }
    }
}

/// Invalid executor configuration, rejected before a call changes model state.
#[derive(Debug, PartialEq, Eq)]
pub struct OptionError(pub String);
impl fmt::Display for OptionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for OptionError {}

/// Settings captured at mount. Team size is resolved against the existing pool.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutorOptions {
    /// Prefill executor.
    pub prefill: Executor,
    /// Decode executor.
    pub decode: Executor,
    /// Requested prefill team, or pool capacity when absent.
    pub prefill_workers: Option<usize>,
    /// Requested decode team, or pool capacity when absent.
    pub decode_workers: Option<usize>,
    /// Number of PAUSE instructions before sleeping in hybrid mode.
    pub hybrid_spin: u32,
}

impl ExecutorOptions {
    /// Resolve a fixed team for one call. Oversized budgets are capped at the
    /// pool capacity; a zero-capacity pool is an error.
    pub fn participants(&self, prefill: bool, capacity: usize) -> Result<usize, OptionError> {
        if capacity == 0 {
            return Err(OptionError("SPMD pool has no participants".into()));
        }
        let requested = if prefill {
            self.prefill_workers
        } else {
            self.decode_workers
        };
        if requested == Some(0) {
            return Err(OptionError("SPMD budget must be positive".into()));
        }
        Ok(requested.unwrap_or(capacity).min(capacity))
    }
}

impl Default for ExecutorOptions {
    /// The generated program with spin barriers, every worker, in both regimes.
    fn default() -> Self {
        ExecutorOptions { prefill: Executor::SpmdSpin, decode: Executor::SpmdSpin, prefill_workers: None, decode_workers: None, hybrid_spin: 512 }
    }
}

/// What a mount records and how its barriers are laid out; the engine's
/// `Tuning` and `Measure` map onto it.
#[derive(Clone, Copy, Debug, Default)]
pub struct MountOptions {
    /// `rdtscp` stamps per node and participant.
    pub trace_sync: bool,
    /// One hardware counter read around every node.
    pub pmu: Option<super::PmuEvent>,
    /// One barrier word shared by every node.
    pub shared_barrier: bool,
    /// Barriers fused around serial nodes.
    pub fuse_serial: bool,
    /// Blocks the trace buffers keep.
    pub trace_blocks: usize,
}
