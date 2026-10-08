use super::*;
use std::any::Any;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};

use super::pmu::{Participants as PmuParticipants, PmuEvent};

pub(crate) fn rdtscp_available() -> bool {
    crate::core::isa::feature("rdtscp")
}

/// Timestamps emitted by the SPMD entry around one plan node and its barrier.
/// Zero means that the corresponding path was not taken (for example, no
/// hybrid sleep or no progress bridge). `RDTSCP` is used so each sample also
/// orders the preceding instructions before the timestamp is published.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SyncTraceRecord {
    /// Entry into the node, before task-range selection.
    pub work_start: u64,
    /// Exit from the node, after all assigned tasks or progress work.
    pub work_end: u64,
    /// Participant-zero entry into the optional progress bridge.
    pub progress_start: u64,
    /// Participant-zero return from the optional progress bridge.
    pub progress_end: u64,
    /// Arrival at the generation barrier.
    pub barrier_start: u64,
    /// Entry into the hybrid condition-variable wait after the spin budget.
    pub sleep_start: u64,
    /// Return from the hybrid condition-variable wait.
    pub sleep_end: u64,
    /// Departure from the generation barrier.
    pub barrier_end: u64,
    /// `IA32_TSC_AUX` returned with `work_start`.
    pub work_start_aux: u32,
    /// `IA32_TSC_AUX` returned with `work_end`.
    pub work_end_aux: u32,
    /// `IA32_TSC_AUX` returned with `progress_start`.
    pub progress_start_aux: u32,
    /// `IA32_TSC_AUX` returned with `progress_end`.
    pub progress_end_aux: u32,
    /// `IA32_TSC_AUX` returned with `barrier_start`.
    pub barrier_start_aux: u32,
    /// `IA32_TSC_AUX` returned with `sleep_start`.
    pub sleep_start_aux: u32,
    /// `IA32_TSC_AUX` returned with `sleep_end`.
    pub sleep_end_aux: u32,
    /// `IA32_TSC_AUX` returned with `barrier_end`.
    pub barrier_end_aux: u32,
}

/// Owned copy of the most recent optional synchronization trace. Records are
/// indexed `[block][node][participant]`; the final node of each block is the
/// block-progress barrier and is zero when no progress callback was supplied.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyncTraceSnapshot {
    /// Blocks present in the captured call.
    pub blocks: usize,
    /// Plan nodes plus the optional block-progress slot.
    pub nodes_per_block: usize,
    /// Fixed collective team size of the captured call.
    pub participants: usize,
    /// Row-major `[block][node][participant]` records.
    pub records: Vec<SyncTraceRecord>,
}

/// One hardware-counter sample at each synchronization trace marker. A value
/// of `u64::MAX` means Linux temporarily reported no scheduled PMU index.
#[repr(C, align(64))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PmuTraceRecord {
    /// Counter at entry into the node.
    pub work_start: u64,
    /// Counter after the node's assigned work.
    pub work_end: u64,
    /// Counter before participant zero calls the progress bridge.
    pub progress_start: u64,
    /// Counter after participant zero returns from the progress bridge.
    pub progress_end: u64,
    /// Counter on arrival at the generation barrier.
    pub barrier_start: u64,
    /// Counter immediately before an optional hybrid wait.
    pub sleep_start: u64,
    /// Counter immediately after an optional hybrid wait.
    pub sleep_end: u64,
    /// Counter on departure from the generation barrier.
    pub barrier_end: u64,
}

/// Owned hardware-counter trace from the most recent generated call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PmuTraceSnapshot {
    /// Hardware event counted by every sample.
    pub event: PmuEvent,
    /// Blocks present in the captured call.
    pub blocks: usize,
    /// Plan nodes plus the optional block-progress slot.
    pub nodes_per_block: usize,
    /// Fixed collective team size of the captured call.
    pub participants: usize,
    /// Row-major `[block][node][participant]` records.
    pub records: Vec<PmuTraceRecord>,
}

#[derive(Default)]
struct SyncTraceStorage {
    records: Vec<SyncTraceRecord>,
    blocks: usize,
    nodes_per_block: usize,
    participants: usize,
}

#[derive(Default)]
struct PmuTraceStorage {
    event: Option<PmuEvent>,
    participants: Option<PmuParticipants>,
    records: Vec<PmuTraceRecord>,
    blocks: usize,
    nodes_per_block: usize,
    team: usize,
}

/// Barrier counters accessed by generated x86 instructions. Each counter is
/// naturally aligned. Locked arrival RMWs and generation publication provide
/// the full inter-participant memory fence.
#[repr(C, align(128))]
pub(crate) struct Synchronization {
    pub remaining: AtomicU64,
    pub generation: AtomicU64,
    pub status: AtomicU64,
    /// The next task of the current node when the target hands tasks out
    /// dynamically (AArch64, [`crate::jit::knobs::Knobs::task_grain`]): each
    /// participant takes a chunk with one atomic add, the last arrival at the
    /// barrier sets it back to zero before publishing the generation.
    pub ticket: AtomicU64,
    _pad: [u64; 12],
    /// The same ticket on the next 128-byte line, away from the counters the
    /// waiting participants poll ([`crate::jit::knobs::Knobs::ticket_line`]).
    pub ticket_far: AtomicU64,
}

/// Preallocated synchronization state for successive calls. An execution error
/// poisons this state until the owner has reset/restored the model.
pub struct ExecutionSession {
    pub(crate) sync: Synchronization,
    lock: Mutex<Option<ExecutionProgress>>,
    cv: Condvar,
    pub(crate) entries: AtomicU64,
    pub(crate) poisoned: bool,
    trace: SyncTraceStorage,
    pmu: PmuTraceStorage,
}
impl Default for ExecutionSession {
    fn default() -> Self {
        Self::new()
    }
}
impl ExecutionSession {
    /// Reserve synchronization resources before inference starts.
    pub fn new() -> Self {
        Self {
            sync: Synchronization {
                remaining: AtomicU64::new(0),
                generation: AtomicU64::new(0),
                status: AtomicU64::new(0),
                ticket: AtomicU64::new(0),
                _pad: [0; 12],
                ticket_far: AtomicU64::new(0),
            },
            lock: Mutex::new(None),
            cv: Condvar::new(),
            entries: AtomicU64::new(0),
            poisoned: false,
            trace: SyncTraceStorage::default(),
            pmu: PmuTraceStorage::default(),
        }
    }
    /// Total participant entries, excluding the synchronous reference driver.
    pub fn jit_entries(&self) -> u64 {
        self.entries.load(Ordering::Relaxed)
    }
    /// Whether an earlier call failed after entering execution.
    pub fn needs_reset(&self) -> bool {
        self.poisoned
    }
    /// Reserve trace storage at mount. The generated call never grows it.
    pub fn reserve_sync_trace(&mut self, records: usize) -> Result<(), ExecutionError> {
        if self.trace.records.len() < records {
            self.trace
                .records
                .try_reserve_exact(records - self.trace.records.len())
                .map_err(|e| {
                    ExecutionError(format!("cannot reserve SPMD synchronization trace: {e}"))
                })?;
            self.trace
                .records
                .resize(records, SyncTraceRecord::default());
        }
        Ok(())
    }
    /// Copy the most recent trace after all participants have joined.
    pub fn sync_trace(&self) -> Option<SyncTraceSnapshot> {
        let n = self
            .trace
            .blocks
            .checked_mul(self.trace.nodes_per_block)?
            .checked_mul(self.trace.participants)?;
        (n > 0).then(|| SyncTraceSnapshot {
            blocks: self.trace.blocks,
            nodes_per_block: self.trace.nodes_per_block,
            participants: self.trace.participants,
            records: self.trace.records[..n].to_vec(),
        })
    }
    /// Copy the most recent PMU trace after all generated participants joined.
    pub fn pmu_trace(&self) -> Option<PmuTraceSnapshot> {
        let event = self.pmu.event?;
        let n = self
            .pmu
            .blocks
            .checked_mul(self.pmu.nodes_per_block)?
            .checked_mul(self.pmu.team)?;
        (n > 0).then(|| PmuTraceSnapshot {
            event,
            blocks: self.pmu.blocks,
            nodes_per_block: self.pmu.nodes_per_block,
            participants: self.pmu.team,
            records: self.pmu.records[..n].to_vec(),
        })
    }
    /// Open one pinned event on each persistent participant before inference.
    pub fn configure_pmu(
        &mut self,
        pool: &crate::pool::thread_pool::ThreadPool,
        event: Option<PmuEvent>,
        records: usize,
    ) -> Result<(), ExecutionError> {
        self.pmu = PmuTraceStorage::default();
        let Some(event) = event else { return Ok(()) };
        self.pmu
            .records
            .try_reserve_exact(records)
            .map_err(|e| ExecutionError(format!("cannot reserve SPMD PMU trace: {e}")))?;
        self.pmu.records.resize(records, PmuTraceRecord::default());
        self.pmu.participants = Some(PmuParticipants::prepare(pool, event)?);
        self.pmu.event = Some(event);
        Ok(())
    }

    pub(crate) fn prepare_pmu_trace(
        &mut self,
        pool: &crate::pool::thread_pool::ThreadPool,
        blocks: usize,
        nodes_per_block: usize,
        participants: usize,
    ) -> Result<(*const *const u8, *mut PmuTraceRecord), ExecutionError> {
        if nodes_per_block == 0 {
            self.pmu.blocks = 0;
            self.pmu.nodes_per_block = 0;
            self.pmu.team = 0;
            return Ok((std::ptr::null(), std::ptr::null_mut()));
        }
        let n = blocks
            .checked_mul(nodes_per_block)
            .and_then(|n| n.checked_mul(participants))
            .ok_or_else(|| ExecutionError("SPMD PMU trace size overflow".into()))?;
        if n > self.pmu.records.len() {
            return Err(ExecutionError(
                "SPMD PMU trace was not reserved at mount".into(),
            ));
        }
        let counters = self.pmu.participants.as_ref().ok_or_else(|| {
            ExecutionError("SPMD PMU participants were not prepared at mount".into())
        })?;
        counters.validate_caller(pool)?;
        self.pmu.records[..n].fill(PmuTraceRecord::default());
        self.pmu.blocks = blocks;
        self.pmu.nodes_per_block = nodes_per_block;
        self.pmu.team = participants;
        Ok((counters.pages(), self.pmu.records.as_mut_ptr()))
    }
    pub(crate) fn prepare_sync_trace(
        &mut self,
        blocks: usize,
        nodes_per_block: usize,
        participants: usize,
    ) -> Result<*mut SyncTraceRecord, ExecutionError> {
        if nodes_per_block == 0 {
            self.trace.blocks = 0;
            self.trace.nodes_per_block = 0;
            self.trace.participants = 0;
            return Ok(std::ptr::null_mut());
        }
        let n = blocks
            .checked_mul(nodes_per_block)
            .and_then(|n| n.checked_mul(participants))
            .ok_or_else(|| ExecutionError("SPMD synchronization trace size overflow".into()))?;
        if n > self.trace.records.len() {
            return Err(ExecutionError(
                "SPMD synchronization trace was not reserved at mount".into(),
            ));
        }
        self.trace.records[..n].fill(SyncTraceRecord::default());
        self.trace.blocks = blocks;
        self.trace.nodes_per_block = nodes_per_block;
        self.trace.participants = participants;
        Ok(self.trace.records.as_mut_ptr())
    }
    /// Mark the session usable after its owner reset or restored all model state.
    ///
    /// # Safety
    /// Every mutable buffer affected by the failed call must already have been
    /// reset or restored. No participant may still be using this session.
    pub unsafe fn after_model_reset(&mut self) {
        self.poisoned = false;
        self.sync.status.store(0, Ordering::Relaxed);
    }
    pub(crate) fn prepare(&mut self, participants: usize) -> Result<(), ExecutionError> {
        if self.poisoned {
            return Err(ExecutionError(
                "SPMD state needs reset or restoration after a failed call".into(),
            ));
        }
        self.sync
            .remaining
            .store(participants as u64, Ordering::Relaxed);
        self.sync.status.store(0, Ordering::Relaxed);
        self.sync.ticket.store(0, Ordering::Relaxed);
        self.sync.ticket_far.store(0, Ordering::Relaxed);
        *self.lock.get_mut().unwrap_or_else(|e| e.into_inner()) = None;
        self.trace.blocks = 0;
        self.trace.nodes_per_block = 0;
        self.trace.participants = 0;
        self.pmu.blocks = 0;
        self.pmu.nodes_per_block = 0;
        self.pmu.team = 0;
        Ok(())
    }
    pub(crate) fn cancel(&self, status: u64) {
        let _ = self
            .sync
            .status
            .compare_exchange(0, status, Ordering::AcqRel, Ordering::Acquire);
        self.notify();
    }
    fn notify(&self) {
        // Match the waiter's check/sleep mutex to prevent a lost notification.
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.cv.notify_all();
    }
}

pub(crate) struct ProgressSlot<'a> {
    pub callback: Option<&'a mut dyn FnMut(ExecutionProgress) -> Result<(), ExecutionError>>,
    pub error: Option<ExecutionError>,
    pub panic: Option<Box<dyn Any + Send>>,
    /// Set by the callback to end the call after the current block (generation);
    /// null when the call cannot stop. The cell outlives the call: the caller joins every
    /// participant before its frame is left.
    pub stop: *const std::cell::Cell<bool>,
    /// The call ended on `stop`, not on an error.
    pub stopped: bool,
}

/// Shared status of a call its callback stopped at a block end: every participant leaves
/// at the next barrier, and the caller treats it as a normal end.
pub(crate) const STOP_STATUS: u64 = 5;

/// Stack context stays fixed until the collective has joined. Participant zero
/// alone accesses progress; tables and code are immutable throughout the call.
#[repr(C)]
pub(crate) struct RunContext<'s, 'p> {
    pub table: *const TaskBinding,
    pub blocks: usize,
    pub block_stride: usize,
    pub participants: usize,
    pub session: &'s ExecutionSession,
    pub progress: UnsafeCell<ProgressSlot<'p>>,
    pub pool: &'s crate::pool::thread_pool::ThreadPool,
    pub handoff: bool,
    pub trace: *mut SyncTraceRecord,
    pub trace_nodes_per_block: usize,
    pub pmu_pages: *const *const u8,
    pub pmu_trace: *mut PmuTraceRecord,
    /// The runtime entries the generated program calls, read through this context
    /// rather than written into the code as absolute addresses.
    pub call_wait: crate::jit::target::generated_fn!(*const RunContext<'_, '_>, u64),
    pub call_notify: crate::jit::target::generated_fn!(*const RunContext<'_, '_>),
    pub call_progress: crate::jit::target::generated_fn!(*const RunContext<'_, '_>, usize, usize),
}

/// The `RunContext` fields a generated runtime call may read its destination from.
pub(crate) fn runtime_call_fields() -> [usize; 3] {
    [
        std::mem::offset_of!(RunContext<'_, '_>, call_wait),
        std::mem::offset_of!(RunContext<'_, '_>, call_notify),
        std::mem::offset_of!(RunContext<'_, '_>, call_progress),
    ]
}
// SAFETY: the callback always stays on its originating thread. When participant
// zero is the caller, only its progress bridge accesses the slot. Otherwise the
// bridge publishes a mailbox event and waits, and only the caller's coordinator
// hook accesses the slot. `handoff` is fixed from the pool policy before launch.
// All other writes use atomics or the session mutex; the launch joins every role.
unsafe impl Sync for RunContext<'_, '_> {}

crate::jit::target::generated_abi! {
    pub(crate) unsafe fn wait(context: *const RunContext<'_, '_>, generation: u64) {
        let c = unsafe { &*context };
        let s = c.session;
        let mut guard = s.lock.lock().unwrap_or_else(|e| e.into_inner());
        while s.sync.generation.load(Ordering::Acquire) == generation
            && s.sync.status.load(Ordering::Acquire) == 0
        {
            guard = s.cv.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
    }
}

crate::jit::target::generated_abi! {
    pub(crate) unsafe fn notify(context: *const RunContext<'_, '_>) {
        unsafe { &*context }.session.notify();
    }
}

crate::jit::target::generated_abi! {
    pub(crate) unsafe fn progress(
        context: *const RunContext<'_, '_>,
        block: usize,
        layer: usize,
    ) {
        let c = unsafe { &*context };
        let event = ExecutionProgress {
            block,
            blocks: c.blocks,
            layer: (layer != usize::MAX).then_some(layer),
        };
        if c.handoff {
            let mut pending = c.session.lock.lock().unwrap_or_else(|e| e.into_inner());
            *pending = Some(event);
            drop(pending);
            c.pool.notify_coordinator();
            let mut pending = c.session.lock.lock().unwrap_or_else(|e| e.into_inner());
            while pending.is_some() && c.session.sync.status.load(Ordering::Acquire) == 0 {
                pending = c
                    .session
                    .cv
                    .wait(pending)
                    .unwrap_or_else(|e| e.into_inner());
            }
        } else {
            unsafe { invoke_progress(c, event) };
        }
    }
}

/// Called only on the submitting thread, never by a worker.
/// Safety: no other access to the callback slot may overlap this call.
pub(crate) unsafe fn pump_progress(c: &RunContext<'_, '_>) {
    let event = *c.session.lock.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(event) = event {
        unsafe { invoke_progress(c, event) };
        let mut pending = c.session.lock.lock().unwrap_or_else(|e| e.into_inner());
        *pending = None;
        c.session.cv.notify_all();
    }
}

unsafe fn invoke_progress(c: &RunContext<'_, '_>, event: ExecutionProgress) {
    let slot = unsafe { &mut *c.progress.get() };
    let Some(callback) = slot.callback.as_deref_mut() else {
        return;
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| callback(event))) {
        Ok(Ok(())) => {
            // Only a block end may stop: every node of the block, `end_block` included, has run.
            // SAFETY: a non-null `stop` points to the caller's cell, alive for the whole call.
            if event.layer.is_none() && !slot.stop.is_null() && unsafe { (*slot.stop).get() } {
                slot.stopped = true;
                c.session.cancel(STOP_STATUS);
            }
        }
        Ok(Err(error)) => {
            slot.error = Some(error);
            c.session.cancel(1);
        }
        Err(payload) => {
            slot.panic = Some(payload);
            c.session.cancel(2);
        }
    }
}
