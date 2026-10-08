//! What the worker pool is told before it is created, installed once per
//! process by `Engine::load` (through `Tuning`). Nothing here reads `std::env`.

use std::sync::OnceLock;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolSettings {
    /// Bounded busy-wait of workers and coordinator before they sleep, in
    /// microseconds; `0` = sleep at once. 500 µs measured best.
    pub spin_us: u64,
    /// The coordinator busy-waits like the workers; `false` = it sleeps at once.
    pub coordinator_spins: bool,
    /// The calling thread is worker 0 and computes; `false` = the old pool, T
    /// children plus a coordinator that only dispatches.
    pub caller_is_worker: bool,
    /// The logical CPUs of the workers, in order (worker `i` on the `i`-th);
    /// `None` = one CPU per physical core, CCDs interleaved.
    pub cpus: Option<Vec<usize>>,
    /// Pin the thread that creates the pool on this CPU; `None` = on the first
    /// worker core when the caller is worker 0.
    pub coordinator_cpu: Option<usize>,
    /// Sample `/proc/stat` once and warn when a targeted CPU is already busy
    /// (an interrupt storm skews every measurement without failing anything).
    pub verify_cpus: bool,
}

impl Default for PoolSettings {
    fn default() -> Self {
        PoolSettings {
            spin_us: crate::pool::thread_pool::SPIN_DEFAULT_US,
            coordinator_spins: true,
            caller_is_worker: true,
            cpus: None,
            coordinator_cpu: None,
            verify_cpus: true,
        }
    }
}

static SETTINGS: OnceLock<PoolSettings> = OnceLock::new();

/// Installs the settings for the life of the process; the same values twice
/// are accepted, different ones refused (the pool is already built).
pub fn install(settings: PoolSettings) -> Result<(), String> {
    let installed = SETTINGS.get_or_init(|| settings.clone());
    if *installed == settings {
        Ok(())
    } else {
        Err("the pool settings were already installed with other values; one Engine per process".to_string())
    }
}

/// The installed settings, or the defaults when nothing was installed.
pub fn settings() -> &'static PoolSettings {
    SETTINGS.get_or_init(PoolSettings::default)
}
