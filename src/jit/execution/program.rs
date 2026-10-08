use super::runtime::{self, RunContext};
use super::*;
use crate::pool::thread_pool::ThreadPool;
use crate::jit::target::{Native, Target};
use memmap2::Mmap;
use std::sync::atomic::Ordering;

/// A whole-call program, its immutable common plan, and its owned RX kernels.
pub struct CompiledExecution {
    plan: ExecutionPlan,
    code: Mmap,
    kernel_offsets: Vec<usize>,
    // Two entries: callbacks absent/present. No progress call instructions exist
    // at all in the absent variant.
    entry_offsets: [usize; 2],
    runtime_calls: Vec<(usize, usize)>,
    trace_sync: bool,
    trace_pmu: bool,
    fused_serial_barriers: usize,
}

fn error(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

type Entry = crate::jit::target::generated_fn!(*const RunContext<'_, '_>, usize);

impl CompiledExecution {
    /// Emit loops and barriers at mount. `hybrid_spin=None` selects spin-only;
    /// `Some(0)` sleeps immediately. This setting applies to this program only.
    pub fn compile(plan: ExecutionPlan, hybrid_spin: Option<u32>) -> Result<Self, ExecutionError> {
        Self::compile_with_sync_trace(plan, hybrid_spin, false)
    }

    /// Compile an entry with optional in-JIT timestamps around every operation,
    /// barrier, hybrid sleep and progress bridge.
    pub fn compile_with_sync_trace(
        plan: ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
    ) -> Result<Self, ExecutionError> {
        Self::compile_with_instrumentation(plan, hybrid_spin, trace_sync, false)
    }

    /// Compile optional TSC and Linux PMU sampling directly into every
    /// operation, progress and barrier boundary.
    pub fn compile_with_instrumentation(
        plan: ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
    ) -> Result<Self, ExecutionError> {
        Self::compile_configured(plan, hybrid_spin, trace_sync, trace_pmu, false, false)
    }

    pub(crate) fn compile_configured(
        plan: ExecutionPlan,
        hybrid_spin: Option<u32>,
        trace_sync: bool,
        trace_pmu: bool,
        shared_barrier: bool,
        fuse_serial: bool,
    ) -> Result<Self, ExecutionError> {
        if trace_sync && !runtime::rdtscp_available() {
            return Err(ExecutionError(
                "SPMD synchronization trace requires RDTSCP".into(),
            ));
        }
        let assembled = Native::assemble(
            &plan,
            hybrid_spin,
            trace_sync,
            trace_pmu,
            shared_barrier,
            fuse_serial,
        )?;
        let code = crate::jit::target::map_exec(assembled.code.len(), |rw| {
            rw.copy_from_slice(&assembled.code)
        })
        .map_err(error)?;
        let fused_serial_barriers = serial_barriers_fused(&plan, fuse_serial);
        let program = Self {
            plan,
            code,
            kernel_offsets: assembled.kernel_offsets,
            entry_offsets: assembled.entry_offsets,
            runtime_calls: assembled.runtime_calls,
            trace_sync,
            trace_pmu,
            fused_serial_barriers,
        };
        program.audit_calls()?;
        Ok(program)
    }

    /// The plan of this program.
    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
    /// Immutable machine code, for archiving and independent disassembly.
    pub fn code(&self) -> &[u8] {
        &self.code
    }
    /// Machine code of exactly one whole-call entry, excluding its kernels.
    pub fn entry_code(&self, with_progress: bool) -> &[u8] {
        let start = self.entry_offsets[usize::from(with_progress)];
        let end = if with_progress {
            self.kernel_offsets
                .first()
                .copied()
                .unwrap_or(self.code.len())
        } else {
            self.entry_offsets[1]
        };
        &self.code[start..end]
    }

    /// Start of each plan kernel in [`Self::code`], in plan kernel order: kernel `k`
    /// spans up to the next offset, the last one up to the end of the image.
    pub fn kernel_offsets(&self) -> &[usize] {
        &self.kernel_offsets
    }
    /// Start of the two whole-call entries in [`Self::code`]: without, then with progress.
    pub fn entry_offsets(&self) -> [usize; 2] {
        self.entry_offsets
    }

    /// Offsets of the runtime CALL instructions and the `RunContext` field each calls through.
    pub fn runtime_calls(&self) -> &[(usize, usize)] {
        &self.runtime_calls
    }

    /// Number of per-block barriers removed between consecutive operations
    /// whose maximum active team is participant zero alone.
    pub fn fused_serial_barriers(&self) -> usize {
        self.fused_serial_barriers
    }

    /// Number of generated synchronization points executed by a block when no
    /// block-progress callback is installed.
    pub fn barriers_per_block(&self) -> usize {
        self.plan.nodes.len() - self.fused_serial_barriers
    }

    /// Trace slots needed for the largest call accepted by this program.
    pub fn sync_trace_capacity(
        &self,
        blocks: usize,
        participants: usize,
    ) -> Result<usize, ExecutionError> {
        if self.trace_sync {
            blocks
                .checked_mul(self.plan.nodes.len() + 1)
                .and_then(|records| records.checked_mul(participants))
                .ok_or_else(|| ExecutionError("SPMD synchronization trace size overflow".into()))
        } else {
            Ok(0)
        }
    }

    /// PMU slots needed for the largest accepted generated call.
    pub fn pmu_trace_capacity(
        &self,
        blocks: usize,
        participants: usize,
    ) -> Result<usize, ExecutionError> {
        if self.trace_pmu {
            blocks
                .checked_mul(self.plan.nodes.len() + 1)
                .and_then(|records| records.checked_mul(participants))
                .ok_or_else(|| ExecutionError("SPMD PMU trace size overflow".into()))
        } else {
            Ok(0)
        }
    }

    /// Decode the final RX image and verify every CALL destination. Generated
    /// kernels use relative calls; runtime bridges use a checked MOV/CALL pair.
    pub fn audit_calls(&self) -> Result<(), ExecutionError> {
        Native::audit_calls(&self.code, &self.runtime_calls)
    }

    /// Enter the generated program once per participant in one pool submission.
    /// No allocation, compilation, code mutation or per-operation dispatch occurs.
    /// Progress executes on the submitting thread, including when the caller is
    /// excluded from the team. The callback therefore need not implement Send.
    ///
    /// # Safety
    /// All bound addresses must remain valid, with adequate capacities, proper
    /// alignment and correct kernel argument types for the complete call.
    /// Different tasks in every operation must have disjoint writes, or
    /// generated synchronization.
    /// Dependencies must cover every cross-task read-after-write hazard.
    /// Dynamic counts and bases must be aligned, readable, and immutable during
    /// their consuming node; preceding generated nodes may update them behind
    /// a barrier. No Rust reference may alias those writes during execution.
    pub unsafe fn run_spmd(
        &self,
        pool: &ThreadPool,
        session: &mut ExecutionSession,
        bindings: &Bindings,
        blocks: usize,
        participants: usize,
        progress: Option<&mut dyn FnMut(ExecutionProgress) -> Result<(), ExecutionError>>,
    ) -> Result<(), ExecutionError> {
        unsafe { self.run_spmd_until(pool, session, bindings, blocks, participants, progress, None) }
    }

    /// [`Self::run_spmd`], ending normally after a block whose end callback set `stop`: the
    /// callback runs before the block-end barrier, where every participant then leaves.
    ///
    /// # Safety
    /// The contract of [`Self::run_spmd`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_spmd_until(
        &self,
        pool: &ThreadPool,
        session: &mut ExecutionSession,
        bindings: &Bindings,
        blocks: usize,
        participants: usize,
        progress: Option<&mut dyn FnMut(ExecutionProgress) -> Result<(), ExecutionError>>,
        stop: Option<&std::cell::Cell<bool>>,
    ) -> Result<(), ExecutionError> {
        bindings.validate(&self.plan, blocks, participants)?;
        if participants == 0 || participants > pool.num_workers() {
            return Err(ExecutionError(
                "SPMD participants exceed pool capacity or are zero".into(),
            ));
        }
        session.prepare(participants)?;
        let trace_nodes = if self.trace_sync {
            self.plan.nodes.len() + 1
        } else {
            0
        };
        let trace = session.prepare_sync_trace(blocks, trace_nodes, participants)?;
        let pmu_nodes = if self.trace_pmu {
            self.plan.nodes.len() + 1
        } else {
            0
        };
        let (pmu_pages, pmu_trace) =
            session.prepare_pmu_trace(pool, blocks, pmu_nodes, participants)?;
        let offset = self.entry_offsets[usize::from(progress.is_some())];
        let function: Entry = unsafe { std::mem::transmute(self.code.as_ptr().add(offset)) };
        let context = RunContext {
            table: bindings.table.as_ptr(),
            blocks,
            block_stride: bindings.block_stride(),
            participants,
            session,
            pool,
            handoff: !pool.caller_participates(),
            trace,
            trace_nodes_per_block: trace_nodes.max(pmu_nodes),
            pmu_pages,
            pmu_trace,
            call_wait: runtime::wait,
            call_notify: runtime::notify,
            call_progress: runtime::progress,
            progress: std::cell::UnsafeCell::new(runtime::ProgressSlot {
                callback: progress,
                error: None,
                panic: None,
                stop: stop.map_or(std::ptr::null(), |s| s as *const std::cell::Cell<bool>),
                stopped: false,
            }),
        };
        let enter = |id| {
            context.session.entries.fetch_add(1, Ordering::Relaxed);
            unsafe { function(&context, id) };
        };
        let cancel = || context.session.cancel(3);
        let result = if context.handoff && offset == self.entry_offsets[1] {
            pool.collective_with_coordinator(participants, enter, cancel, || {
                unsafe { runtime::pump_progress(&context) };
            })
        } else {
            pool.collective(participants, enter, cancel)
        };
        let status = context.session.sync.status.load(Ordering::Acquire);
        let slot = context.progress.into_inner();
        let stopped = slot.stopped
            && status == runtime::STOP_STATUS
            && slot.error.is_none()
            && slot.panic.is_none()
            && result.is_ok();
        session.poisoned = !stopped && (status != 0 || result.is_err());
        if stopped {
            return Ok(());
        }
        if let Some(e) = slot.error {
            return Err(e);
        }
        if slot.panic.is_some() {
            return Err(ExecutionError(
                "execution progress callback panicked".into(),
            ));
        }
        result.map_err(error)?;
        if status != 0 {
            return Err(ExecutionError(format!("SPMD execution aborted ({status})")));
        }
        Ok(())
    }
}
