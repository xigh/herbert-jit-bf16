//! The worker pool the generated program runs on.
//!
//! What it offers is one operation, the **collective**: `participants`
//! threads enter the same function once, each with its identifier, and the
//! caller returns when every one of them has left. The generated program is
//! such a function — every participant runs the whole program, node after
//! node, and the barriers inside the program keep them in step. The pool
//! never splits work itself: the program does, by participant identifier.
//!
//! The threads are persistent and pinned, one per physical core. Between two
//! collectives a worker busy-waits for `spin_us` before it sleeps on a
//! condition variable, so the next call finds its core awake (a core that
//! slept costs 20 to 40 µs of wake-up at the first barrier). The calling
//! thread is participant 0 by default: it computes on its own core instead
//! of waiting, and the pool creates only T−1 threads.

use crate::core::cpu_detection::get_performance_cpu_count;
use std::any::Any;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::Instant;

thread_local! {
    // A participant must never submit another collective: some members may
    // already be waiting for it at a barrier.
    static IN_COLLECTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct CollectiveScope;
impl CollectiveScope {
    fn enter() -> Self {
        IN_COLLECTIVE.with(|flag| flag.set(true));
        Self
    }
}
impl Drop for CollectiveScope {
    fn drop(&mut self) {
        IN_COLLECTIVE.with(|flag| flag.set(false));
    }
}

type ParticipantFn = unsafe fn(*const (), usize);

/// One collective, published to the workers under the mutex.
#[derive(Clone, Copy)]
struct Team {
    /// Identifiers `0..participants` enter; the other workers skip this generation.
    participants: usize,
    /// Threads the coordinator waits for: all participants, or all but itself.
    expected: usize,
    ctx: *const (),
    func: ParticipantFn,
}

// SAFETY: `ctx` points to the closure on the submitter's stack, which stays
// blocked in `collective` until every participant has left.
unsafe impl Send for Team {}
unsafe impl Sync for Team {}

#[derive(Default)]
struct SharedState {
    coordinator_wake: u64,
    shutdown: bool,
    generation: u64,
    team: Option<Team>,
    completed: usize,
    panic: Option<(usize, String)>,
}

struct Shared {
    state: Mutex<SharedState>,
    /// Wakes the workers on a new generation.
    worker_cv: Condvar,
    /// Wakes the coordinator on a completion, or on `notify_coordinator`.
    done_cv: Condvar,
    /// Mirror of `state.generation`, for the workers' busy-wait.
    generation_atomic: AtomicU64,
    /// Participants done with the current generation, for the coordinator's busy-wait.
    done_atomic: AtomicUsize,
    /// Busy-wait budget of a worker between two collectives.
    spin: std::time::Duration,
    /// Busy-wait budget of the coordinator while it waits for the team.
    spin_coord: std::time::Duration,
    /// The calling thread is participant 0.
    caller_worker: bool,
}

/// Busy-wait until `ready()` or the budget is spent; the clock is read every
/// 64 iterations only.
#[inline]
fn wait_actively(budget: std::time::Duration, mut ready: impl FnMut() -> bool) -> bool {
    if budget.is_zero() {
        return false;
    }
    let start_at = Instant::now();
    let mut tours: u32 = 0;
    loop {
        if ready() {
            return true;
        }
        tours = tours.wrapping_add(1);
        if tours.is_multiple_of(64) && start_at.elapsed() >= budget {
            return ready();
        }
        std::hint::spin_loop();
    }
}

/// The measured busy-wait: 500 µs (3 ms does not do better).
pub const SPIN_DEFAULT_US: u64 = 500;

#[derive(Debug)]
pub enum ThreadPoolError {
    WorkerPanic { worker_id: usize, message: String },
    PoolAlreadyInitialized { existing: usize, requested: usize },
    /// A team must fit the pool: `1..=num_workers` participants.
    InvalidParticipants { requested: usize, capacity: usize },
    /// A participant tried to submit a collective of its own.
    NestedCollectiveDispatch,
}

impl std::fmt::Display for ThreadPoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThreadPoolError::InvalidParticipants { requested, capacity } => {
                write!(f, "collective participants {requested} outside 1..={capacity}")
            }
            ThreadPoolError::NestedCollectiveDispatch => write!(f, "nested pool dispatch from a collective participant"),
            ThreadPoolError::WorkerPanic { worker_id, message } => write!(f, "worker {worker_id} panicked: {message}"),
            ThreadPoolError::PoolAlreadyInitialized { existing, requested } => {
                write!(f, "thread pool already initialized with {existing} workers (requested {requested})")
            }
        }
    }
}

impl std::error::Error for ThreadPoolError {}

pub struct ThreadPool {
    num_workers: usize,
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// One collective at a time.
    dispatch_lock: Mutex<()>,
    collective_submissions: AtomicU64,
}

impl ThreadPool {
    fn new_with_affinity_map(worker_cpu_ids: Vec<Option<usize>>) -> Self {
        let settings = crate::pool::settings::settings();
        let spin = std::time::Duration::from_micros(settings.spin_us);
        let spin_coord = if settings.coordinator_spins { spin } else { std::time::Duration::ZERO };
        Self::new_with_options(worker_cpu_ids, spin, spin_coord, settings.caller_is_worker)
    }

    /// The full constructor: the modes passed explicitly.
    fn new_with_options(
        worker_cpu_ids: Vec<Option<usize>>,
        spin: std::time::Duration,
        spin_coord: std::time::Duration,
        caller_worker: bool,
    ) -> Self {
        let num_workers = worker_cpu_ids.len().max(1);
        let shared = Arc::new(Shared {
            state: Mutex::new(SharedState::default()),
            worker_cv: Condvar::new(),
            done_cv: Condvar::new(),
            generation_atomic: AtomicU64::new(0),
            done_atomic: AtomicUsize::new(0),
            spin,
            spin_coord,
            caller_worker,
        });
        // Said once in the log, so a measurement can be read against its pool.
        eprintln!(
            "[pool] {} workers on cpus [{}] · caller = worker 0: {} · spin {} µs · coordinator spins: {}",
            num_workers,
            worker_cpu_ids.iter().map(|c| c.map_or("-".to_string(), |c| c.to_string())).collect::<Vec<_>>().join(","),
            if caller_worker { "yes" } else { "no" },
            spin.as_micros(),
            if spin_coord.is_zero() { "no" } else { "yes" },
        );
        let mut workers = Vec::with_capacity(num_workers);
        for (worker_id, pin_to) in worker_cpu_ids.into_iter().enumerate() {
            if shared.caller_worker && worker_id == 0 {
                continue; // the caller thread holds this role
            }
            let shared_clone = Arc::clone(&shared);
            workers.push(thread::spawn(move || {
                if let Some(cpu_id) = pin_to {
                    set_thread_affinity(cpu_id);
                }
                worker_loop(worker_id, shared_clone);
            }));
        }
        ThreadPool { num_workers, shared, workers, dispatch_lock: Mutex::new(()), collective_submissions: AtomicU64::new(0) }
    }

    /// `num_workers` workers, pinned one per physical core (SMT siblings skipped:
    /// each thread keeps a whole L2).
    pub fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let core_ids = crate::pool::topology::physical_core_ids();
        Self::new_with_affinity_map((0..num_workers).map(|w| core_ids.get(w).copied()).collect())
    }

    /// One worker per listed logical CPU, pinned to it.
    pub fn new_on_core_ids(core_ids: Vec<usize>) -> Self {
        let worker_cpu_ids = if core_ids.is_empty() { vec![None] } else { core_ids.into_iter().map(Some).collect() };
        Self::new_with_affinity_map(worker_cpu_ids)
    }

    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Does participant 0 run on the submitting thread?
    pub fn caller_participates(&self) -> bool {
        self.shared.caller_worker
    }

    /// Wake the coordinator's hook (see [`Self::collective_with_coordinator`]).
    /// The generation under the mutex prevents a lost wake between hook and sleep.
    pub fn notify_coordinator(&self) {
        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.coordinator_wake = state.coordinator_wake.wrapping_add(1);
        self.shared.done_cv.notify_all();
    }

    /// Collectives accepted so far.
    pub fn collective_submissions(&self) -> u64 {
        self.collective_submissions.load(Ordering::Relaxed)
    }

    /// Exactly `participants` threads run `f(id)` once, `id` in `0..participants`;
    /// returns when all have left. `cancel` runs if a participant panics,
    /// before the pool joins the others: it must release every barrier the
    /// program may be waiting at, and must not panic.
    pub fn collective<F, C>(&self, participants: usize, f: F, cancel: C) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize) + Sync,
        C: Fn() + Sync,
    {
        self.collective_impl(participants, &f, &cancel, None)
    }

    /// A collective with a hook run on the calling thread while it waits for
    /// the team: the progress bridge of the generated program, woken by
    /// [`Self::notify_coordinator`]. The hook may borrow non-`Send` state and
    /// must not dispatch pool work; a hook panic cancels the team.
    pub fn collective_with_coordinator<F, C, P>(&self, participants: usize, f: F, cancel: C, mut pump: P) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize) + Sync,
        C: Fn() + Sync,
        P: FnMut(),
    {
        self.collective_impl(participants, &f, &cancel, Some(&mut pump))
    }

    fn collective_impl<F, C>(&self, participants: usize, f: &F, cancel: &C, mut pump: Option<&mut dyn FnMut()>) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize) + Sync,
        C: Fn() + Sync,
    {
        if IN_COLLECTIVE.with(|flag| flag.get()) {
            return Err(ThreadPoolError::NestedCollectiveDispatch);
        }
        if participants == 0 || participants > self.num_workers {
            return Err(ThreadPoolError::InvalidParticipants { requested: participants, capacity: self.num_workers });
        }
        self.collective_submissions.fetch_add(1, Ordering::Relaxed);
        let has_pump = pump.is_some();
        let mut guarded_pump = || {
            let _scope = CollectiveScope::enter();
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                if let Some(pump) = pump.as_deref_mut() {
                    pump();
                }
            })) {
                cancel();
                std::panic::resume_unwind(payload);
            }
        };
        let entry = |id: usize| {
            let _scope = CollectiveScope::enter();
            if let Err(payload) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(id))) {
                cancel();
                std::panic::resume_unwind(payload);
            }
        };
        self.dispatch(participants, entry, if has_pump { Some(&mut guarded_pump) } else { None })
    }

    /// Publishes the team, runs participant 0 on this thread when it is a
    /// worker, waits for `expected` completions (busy, then on the condition
    /// variable), running the hook in between.
    fn dispatch<F>(&self, participants: usize, f: F, mut pump: Option<&mut dyn FnMut()>) -> Result<(), ThreadPoolError>
    where
        F: Fn(usize) + Sync,
    {
        let _dispatch_guard = self.dispatch_lock.lock().unwrap_or_else(|error| error.into_inner());

        unsafe fn call_f<F>(ctx: *const (), id: usize)
        where
            F: Fn(usize) + Sync,
        {
            // SAFETY: `ctx` points to `f` on the caller's stack, which stays
            // blocked in `dispatch` until every participant has completed.
            let f = unsafe { &*(ctx as *const F) };
            (f)(id);
        }

        let expected = if self.shared.caller_worker { participants - 1 } else { participants };
        let team = Team { participants, expected, ctx: &f as *const F as *const (), func: call_f::<F> };

        let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        state.completed = 0;
        state.panic = None;
        state.team = Some(team);
        state.generation = state.generation.wrapping_add(1);
        self.shared.done_atomic.store(0, Ordering::Release);
        self.shared.generation_atomic.store(state.generation, Ordering::Release);
        self.shared.worker_cv.notify_all();

        if self.shared.caller_worker {
            // Participant 0 is this thread: it works while the others work.
            drop(state);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { call_f::<F>(team.ctx, 0) }));
            state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            if let Err(payload) = result {
                if state.panic.is_none() {
                    state.panic = Some((0, panic_payload_to_string(payload)));
                }
            }
        }

        if pump.is_none() && !self.shared.spin_coord.is_zero() {
            // Busy-wait without the mutex, so the workers can count themselves
            // done; the loop on `done_cv` below remains the truth.
            drop(state);
            let done = &self.shared.done_atomic;
            wait_actively(self.shared.spin_coord, || done.load(Ordering::Acquire) >= expected);
            state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
        }

        while state.completed < expected {
            if let Some(hook) = pump.as_deref_mut() {
                let wake = state.coordinator_wake;
                drop(state);
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(hook));
                state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
                if let Err(payload) = result {
                    if state.panic.is_none() {
                        state.panic = Some((0, panic_payload_to_string(payload)));
                    }
                    pump = None;
                }
                if state.completed >= expected {
                    break;
                }
                if wake != state.coordinator_wake {
                    continue;
                }
            }
            state = self.shared.done_cv.wait(state).unwrap_or_else(|e| e.into_inner());
        }

        if let Some((worker_id, message)) = state.panic.take() {
            return Err(ThreadPoolError::WorkerPanic { worker_id, message });
        }
        Ok(())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        {
            let mut state = self.shared.state.lock().unwrap_or_else(|e| e.into_inner());
            state.shutdown = true;
            self.shared.worker_cv.notify_all();
            self.shared.done_cv.notify_all();
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// A worker: wait for a new generation (busy, then asleep), enter the team if
/// its identifier is a participant, count itself done.
fn worker_loop(worker_id: usize, shared: Arc<Shared>) {
    let mut seen_generation = 0u64;
    loop {
        if !shared.spin.is_zero() {
            let generation = &shared.generation_atomic;
            wait_actively(shared.spin, || generation.load(Ordering::Acquire) != seen_generation);
        }
        let team = {
            let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
            while !state.shutdown && state.generation == seen_generation {
                state = shared.worker_cv.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            if state.shutdown {
                return;
            }
            seen_generation = state.generation;
            state.team
        };
        let Some(team) = team else { continue };
        if worker_id >= team.participants {
            continue;
        }
        // SAFETY: the team is published under the mutex and stays valid until
        // the coordinator has seen every completion of this generation.
        let result = std::panic::catch_unwind(|| unsafe { (team.func)(team.ctx, worker_id) });
        let mut state = shared.state.lock().unwrap_or_else(|e| e.into_inner());
        if let Err(payload) = result {
            if state.panic.is_none() {
                state.panic = Some((worker_id, panic_payload_to_string(payload)));
            }
        }
        state.completed += 1;
        // Under the mutex, after the count: when the coordinator sees the
        // atomic full, the count under the mutex already is, and no late
        // increment can leak into the next generation.
        shared.done_atomic.fetch_add(1, Ordering::AcqRel);
        if state.completed >= team.expected {
            shared.done_cv.notify_one();
        }
    }
}

/// Pin the current thread to one logical CPU; a no-op off Linux.
#[cfg(target_os = "linux")]
fn set_thread_affinity(cpu_id: usize) {
    unsafe {
        let mut cpuset: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut cpuset);
        libc::CPU_SET(cpu_id, &mut cpuset);
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &cpuset) != 0 {
            eprintln!("[pool] sched_setaffinity({cpu_id}) failed: {}", std::io::Error::last_os_error());
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn set_thread_affinity(_cpu_id: usize) {}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

static GLOBAL_POOL: OnceLock<ThreadPool> = OnceLock::new();
static GLOBAL_POOL_THREAD_OVERRIDE: OnceLock<usize> = OnceLock::new();

/// Sets the worker count of the global pool before its first use.
pub fn configure_global_pool(num_threads: usize) -> Result<(), ThreadPoolError> {
    let requested = num_threads.max(1);
    if let Some(pool) = GLOBAL_POOL.get() {
        let existing = pool.num_workers();
        if existing != requested {
            return Err(ThreadPoolError::PoolAlreadyInitialized { existing, requested });
        }
        return Ok(());
    }
    if let Some(existing) = GLOBAL_POOL_THREAD_OVERRIDE.get() {
        if *existing != requested {
            return Err(ThreadPoolError::PoolAlreadyInitialized { existing: *existing, requested });
        }
        return Ok(());
    }
    let _ = GLOBAL_POOL_THREAD_OVERRIDE.set(requested);
    Ok(())
}

/// The global pool, created on first use with the configured count or one
/// worker per performance core.
pub fn global_pool() -> &'static ThreadPool {
    GLOBAL_POOL.get_or_init(|| {
        let num_threads = GLOBAL_POOL_THREAD_OVERRIDE.get().copied().unwrap_or_else(get_performance_cpu_count).max(1);
        warn_cpus_busy(num_threads);
        let settings = crate::pool::settings::settings();
        // The thread that creates the pool is the coordinator: pinned where the
        // settings say, or on the first worker core when it is participant 0
        // (unpinned, it lands on a worker's core during the GEMM: 18.7 vs
        // 28.5 tok/s measured).
        if let Some(cpu) = settings.coordinator_cpu {
            set_thread_affinity(cpu);
        } else if settings.caller_is_worker {
            if let Some(&cpu) = crate::pool::topology::physical_core_ids().first() {
                set_thread_affinity(cpu);
            }
        }
        ThreadPool::new(num_threads)
    })
}

/// Before pinning the global pool: are the targeted CPUs free?
///
/// A CPU already taken (another process, an interrupt storm, `steal` from a
/// VM) skews everything the pool measures there without ever failing a run —
/// a laptop's touchscreen controller at 13 000 to 42 000 IRQ/s on one CPU
/// cost 2 tok/s at decode and nobody noticed. `/proc/stat` is sampled over a short window,
/// once per process, and the run continues with a warning in the log.
fn warn_cpus_busy(num_threads: usize) {
    if !crate::pool::settings::settings().verify_cpus {
        return;
    }
    let core_ids = crate::pool::topology::physical_core_ids();
    let targeted = &core_ids[..num_threads.min(core_ids.len())];
    if targeted.is_empty() {
        return;
    }
    const THRESHOLD: f32 = 0.10;
    #[cfg(target_os = "linux")]
    let self_before = crate::pool::topology::cpu_time_self_jiffies();
    let window_of = std::time::Duration::from_millis(300);
    let occupations = crate::pool::topology::occupation_cpus(targeted, window_of);
    // Our own share during the window: a model still loading shows here.
    #[cfg(target_os = "linux")]
    let own = match (self_before, crate::pool::topology::cpu_time_self_jiffies()) {
        (Some(a), Some(b)) => {
            let user_hz = unsafe { libc::sysconf(libc::_SC_CLK_TCK) }.max(1) as f32;
            (b.saturating_sub(a)) as f32 / (window_of.as_secs_f32() * user_hz)
        }
        _ => 0.0,
    };
    #[cfg(not(target_os = "linux"))]
    let own = 0.0f32;
    for o in occupations {
        if o.occupied >= THRESHOLD {
            let mut detail = Vec::new();
            if o.lost >= 0.02 {
                detail.push(format!("{:.0} % off accounting = interrupts", o.lost * 100.0));
            }
            if o.irq >= 0.02 {
                detail.push(format!("irq {:.0} %", o.irq * 100.0));
            }
            if o.stolen >= 0.02 {
                detail.push(format!("stolen {:.0} %", o.stolen * 100.0));
            }
            let detail = if detail.is_empty() { String::new() } else { format!(" (of which {})", detail.join(", ")) };
            let self_txt = if own >= 0.02 {
                format!("; this process itself consumed {:.0} % of a CPU during the window", own * 100.0)
            } else {
                String::new()
            };
            eprintln!(
                "[pool] warning: cpu{} busy at {:.0} %{} for {} ms before the first work — another process or an interrupt storm; what is measured there is suspicious (see /proc/interrupts; PoolSettings::cpus picks other cores){}",
                o.cpu,
                o.occupied * 100.0,
                detail,
                window_of.as_millis(),
                self_txt
            );
        }
    }
}
