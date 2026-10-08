//! The RISC-V whole-call program: the entry loop, the barriers, the progress
//! bridge and the trace stamps around the kernels of a plan, assembled into
//! one page, and the decoder audit of its runtime calls.
//!
//! The AArch64 program transposed (`target/aarch64/execution/entry.rs`),
//! with the memory model of RVWMO: the arrival at a barrier is an
//! `amoadd.d.aqrl` (release of this participant's writes, acquire of the
//! others'), the last arrival re-arms `remaining` and publishes the next
//! generation with another `amoadd.d.aqrl`, and the waiters read the
//! generation with a load followed by `fence r, rw` (an acquire), spinning
//! on `pause`. Tasks are handed out by ticket with a relaxed `amoadd.d`. The
//! kernels are called through `auipc` + `jalr` pairs (the page can exceed
//! the reach of `jal`), which the decoder reports as direct calls. The trace
//! reads `time` behind a `fence`; RISC-V gives user space no core number,
//! so the auxiliary word is 0.

use super::*;
use crate::jit::execution::runtime::{self, RunContext};
use crate::jit::target::riscv64::util::{add_imm, exec_err, Frame};
use crate::jit::target::Assembled;
use rv::*;
use std::mem::{offset_of, size_of};

// The registers the entry keeps across every kernel call (callee-saved).
const CTX: X = S1;
const PART: X = S2;
const BLOCK: X = S3;
const BIND: X = S4;
const TASK: X = S5;
const END: X = S6;
const STEP: X = S7;
/// The address of the ticket.
const TICKET: X = S8;
/// Participants · grain, the divisor of the guided chunks.
const DIVISOR: X = S9;
/// The register a runtime call goes through.
const CALL: X = T6;

/// How the tasks of a node are handed out: the knobs of the process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Dispatch {
    /// Chunks of about `tasks / (participants · grain)`; 0 = the static split.
    pub grain: usize,
    /// The chunk recomputed from the remaining tasks at every grab.
    pub guided: bool,
    /// The smallest and (when nonzero) largest chunk.
    pub chunk_min: usize,
    pub chunk_max: usize,
    /// The ticket on its own line.
    pub ticket_line: bool,
}

impl Dispatch {
    fn of_knobs() -> Dispatch {
        let k = crate::jit::knobs::knobs();
        Dispatch { grain: k.task_grain, guided: k.task_guided, chunk_min: k.task_chunk_min, chunk_max: k.task_chunk_max, ticket_line: k.ticket_line }
    }
    /// The byte offset of the ticket in the session.
    fn ticket_offset(&self) -> usize {
        let sync = offset_of!(ExecutionSession, sync);
        if self.ticket_line {
            sync + offset_of!(runtime::Synchronization, ticket_far)
        } else {
            sync + offset_of!(runtime::Synchronization, ticket)
        }
    }
}

fn dispatch() -> Dispatch {
    Dispatch::of_knobs()
}

fn sync_field(field: usize) -> usize {
    offset_of!(ExecutionSession, sync) + field
}

/// `rd = the session field at `off`` (through `tmp` for a wide offset).
fn session_addr(a: &mut Asm, rd: X, off: usize) {
    a.ld(rd, CTX, offset_of!(RunContext<'_, '_>, session) as i64);
    add_imm(a, rd, rd, off as i64, T6);
}

/// Emit both entries, the shared barrier when requested, and the kernels of `plan`.
pub(crate) fn assemble(
    plan: &ExecutionPlan,
    hybrid_spin: Option<u32>,
    trace_sync: bool,
    trace_pmu: bool,
    shared_barrier: bool,
    fuse_serial: bool,
) -> Result<Assembled, ExecutionError> {
    if trace_pmu {
        return Err(ExecutionError(
            "the generated PMU trace needs user access to a hardware counter, which this target does not offer".into(),
        ));
    }
    let mut a = Asm::new();
    let kernels: Vec<Label> = plan.kernels.iter().map(|_| a.label()).collect();
    let entries = [a.label(), a.label()];
    let dispatch = dispatch();
    let barrier = a.label();
    let mut runtime_labels = Vec::new();
    for (i, &entry) in entries.iter().enumerate() {
        a.bind(entry);
        emit_entry(
            &mut a,
            plan,
            &kernels,
            hybrid_spin,
            i == 1,
            trace_sync,
            shared_barrier.then_some(barrier),
            fuse_serial,
            dispatch,
            &mut runtime_labels,
        );
    }
    if shared_barrier {
        a.bind(barrier);
        emit_barrier_function(&mut a, hybrid_spin, trace_sync, &mut runtime_labels);
    }
    for (kernel, &label) in plan.kernels.iter().zip(&kernels) {
        a.align(16);
        a.bind(label);
        a.bytes(&kernel.bytes);
    }
    let code = a.finish().map_err(exec_err)?;
    let at = |l: Label| code.label_offset(l).ok_or_else(|| ExecutionError("unbound program label".into()));
    Ok(Assembled {
        entry_offsets: [at(entries[0])?, at(entries[1])?],
        kernel_offsets: kernels.iter().map(|&l| at(l)).collect::<Result<_, _>>()?,
        runtime_calls: runtime_labels
            .iter()
            .map(|&(l, field)| at(l).map(|p| (p, field)))
            .collect::<Result<_, _>>()?,
        code: code.bytes,
    })
}

/// The word of `ld t6, field(s1)`: the load that must precede every runtime call.
fn runtime_load_word(field: usize) -> u32 {
    ((field as u32) << 20) | (u32::from(CTX.n()) << 15) | (3 << 12) | (u32::from(CALL.n()) << 7) | 0x03
}

/// Decode the final RX image and verify every call: a direct call (a `jal`,
/// or an `auipc` + `jalr` pair) stays inside the image, and every other
/// `jalr` that links is a registered runtime call through `t6`, loaded by
/// the instruction just before it from one of the allowed `RunContext`
/// fields.
pub(crate) fn audit_calls(code: &[u8], runtime_calls: &[(usize, usize)]) -> Result<(), ExecutionError> {
    let fields = runtime::runtime_call_fields();
    let insns = decode_all(code);
    let mut found = 0;
    for (i, (offset, insn)) in insns.iter().enumerate() {
        let Some(insn) = insn else { continue };
        match insn.flow {
            Flow::Call(rel) => {
                let target = *offset as i64 + rel;
                if target < 0 || target as usize >= code.len() {
                    return Err(ExecutionError("emitted call leaves the RX image".into()));
                }
            }
            Flow::IndirectCall => {
                let Some((_, field)) = runtime_calls.iter().find(|(ip, _)| ip == offset) else {
                    return Err(ExecutionError("unregistered indirect call".into()));
                };
                let load = i.checked_sub(1).and_then(|p| insns[p].1.as_ref()).map(|p| p.word);
                if !fields.contains(field) || insn.target_reg != Some(CALL.n()) || load != Some(runtime_load_word(*field)) || *field % 8 != 0 || *field >= 2048 {
                    return Err(ExecutionError("runtime call target mismatch".into()));
                }
                found += 1;
            }
            _ => {}
        }
    }
    if found != runtime_calls.len() {
        return Err(ExecutionError("runtime call audit count mismatch".into()));
    }
    Ok(())
}

/// Call the runtime entry stored in the context field at `field`.
fn runtime_call(a: &mut Asm, field: usize, calls: &mut Vec<(Label, usize)>) {
    a.ld(CALL, CTX, field as i64);
    let at = a.here();
    a.jalr(RA, CALL, 0);
    calls.push((at, field));
}

/// The barrier, inline. Uses `t0..t6`; `exit` is taken on a nonzero status.
/// `trace_node = None` means the node index is in `BIND` (the shared barrier
/// function, which receives it as its fourth argument).
fn emit_inline_barrier(
    a: &mut Asm,
    spin: Option<u32>,
    exit: Label,
    trace_sync: bool,
    trace_node: Option<usize>,
    calls: &mut Vec<(Label, usize)>,
) {
    let remaining = sync_field(offset_of!(runtime::Synchronization, remaining));
    let generation = sync_field(offset_of!(runtime::Synchronization, generation));
    let status = sync_field(offset_of!(runtime::Synchronization, status));
    let (t_status, t_rem, t_gen, t_seen, t_old, t_budget) = (T0, T1, T2, T3, T4, T5);
    let last = a.label();
    let wait_loop = a.label();
    let done = a.label();
    session_addr(a, t_status, status);
    session_addr(a, t_rem, remaining);
    session_addr(a, t_gen, generation);
    a.ld(t_old, t_status, 0);
    a.branch_far(Cond::Ne, t_old, ZERO, exit);
    // The generation is read before the arrival is published.
    a.ld(t_seen, t_gen, 0);
    a.li(t_old, -1);
    a.amoadd_d(t_old, t_old, t_rem, Order::AcqRel);
    a.li(T6, 1);
    a.beq(t_old, T6, last);
    if let Some(budget) = spin {
        a.li(t_budget, i64::from(budget));
    }
    a.bind(wait_loop);
    a.ld(t_old, t_status, 0);
    a.branch_far(Cond::Ne, t_old, ZERO, exit);
    a.ld(t_old, t_gen, 0);
    a.fence(Fence::R, Fence::RW);
    a.bne(t_old, t_seen, done);
    if spin.is_some() {
        let pause = a.label();
        a.bnez(t_budget, pause);
        if trace_sync {
            emit_trace_stamp(a, trace_node, offset_of!(runtime::SyncTraceRecord, sleep_start));
        }
        a.mv(A0, CTX);
        a.mv(A1, t_seen);
        runtime_call(a, offset_of!(RunContext<'_, '_>, call_wait), calls);
        if trace_sync {
            emit_trace_stamp(a, trace_node, offset_of!(runtime::SyncTraceRecord, sleep_end));
        }
        session_addr(a, t_status, status);
        a.ld(t_old, t_status, 0);
        a.branch_far(Cond::Ne, t_old, ZERO, exit);
        a.j(done);
        a.bind(pause);
        a.addi(t_budget, t_budget, -1);
    }
    a.pause();
    a.j(wait_loop);
    a.bind(last);
    // Re-arm, then publish the next generation: the release of this
    // `amoadd.d.aqrl` orders the re-arming, the ticket back to zero and every
    // write of the team before it. Every participant has taken its last
    // ticket of this node before arriving.
    a.ld(t_old, CTX, offset_of!(RunContext<'_, '_>, participants) as i64);
    a.sd(t_old, t_rem, 0);
    session_addr(a, t_old, dispatch().ticket_offset());
    a.sd(ZERO, t_old, 0);
    a.li(t_old, 1);
    a.amoadd_d(ZERO, t_old, t_gen, Order::AcqRel);
    if spin.is_some() {
        a.mv(A0, CTX);
        runtime_call(a, offset_of!(RunContext<'_, '_>, call_notify), calls);
    }
    a.bind(done);
}

#[allow(clippy::too_many_arguments)]
fn emit_barrier_call(
    a: &mut Asm,
    spin: Option<u32>,
    exit: Label,
    trace_sync: bool,
    trace_node: usize,
    shared: Option<Label>,
    calls: &mut Vec<(Label, usize)>,
) {
    if let Some(barrier) = shared {
        a.mv(A0, CTX);
        a.mv(A1, PART);
        a.mv(A2, BLOCK);
        a.li(A3, trace_node as i64);
        a.call(barrier);
        a.branch_far(Cond::Ne, A0, ZERO, exit);
    } else {
        emit_inline_barrier(a, spin, exit, trace_sync, Some(trace_node), calls);
    }
}

/// One barrier function shared by both whole-call entries: `(context,
/// participant, block, trace_node) -> status`. Generated callers leave the
/// whole-call loop on nonzero.
fn emit_barrier_function(a: &mut Asm, spin: Option<u32>, trace_sync: bool, calls: &mut Vec<(Label, usize)>) {
    let abort = a.label();
    let epilogue = a.label();
    let frame = Frame::enter(a, true, &[CTX, PART, BLOCK, BIND], &[], 0);
    a.mv(CTX, A0);
    a.mv(PART, A1);
    a.mv(BLOCK, A2);
    a.mv(BIND, A3);
    emit_inline_barrier(a, spin, abort, trace_sync, None, calls);
    a.li(A0, 0);
    a.j(epilogue);
    a.bind(abort);
    session_addr(a, T0, sync_field(offset_of!(runtime::Synchronization, status)));
    a.ld(A0, T0, 0);
    a.bind(epilogue);
    frame.leave(a);
    a.ret();
}

/// `t0` = the task count of the current node: `*task_count`, or the static count.
fn task_count(a: &mut Asm) {
    let ready = a.label();
    a.ld(T0, BIND, offset_of!(TaskBinding, tasks) as i64);
    a.ld(T1, BIND, offset_of!(TaskBinding, task_count) as i64);
    a.beqz(T1, ready);
    a.ld(T0, T1, 0);
    a.bind(ready);
}

/// The value of one integer argument of a task, into `dst`:
/// `base + stride·task + participant_stride·participant + dynamic base`,
/// every addition checked for unsigned overflow.
fn task_argument(a: &mut Asm, i: usize, dst: X, invalid: Label) {
    let argument = offset_of!(TaskBinding, arguments) + i * size_of::<Argument>();
    a.ld(dst, BIND, (argument + 8) as i64);
    a.mul(dst, dst, TASK);
    a.ld(T2, BIND, argument as i64);
    a.add(dst, dst, T2);
    a.branch_far(Cond::Ltu, dst, T2, invalid);
    a.ld(T2, BIND, (offset_of!(TaskBinding, participant_strides) + i * 8) as i64);
    a.mul(T2, T2, PART);
    a.add(dst, dst, T2);
    a.branch_far(Cond::Ltu, dst, T2, invalid);
    let ready = a.label();
    a.ld(T2, BIND, (offset_of!(TaskBinding, argument_bases) + i * 8) as i64);
    a.beqz(T2, ready);
    a.ld(T2, T2, 0);
    a.add(dst, dst, T2);
    a.branch_far(Cond::Ltu, dst, T2, invalid);
    a.bind(ready);
}

#[allow(clippy::too_many_arguments)]
fn emit_entry(
    a: &mut Asm,
    plan: &ExecutionPlan,
    kernels: &[Label],
    spin: Option<u32>,
    with_progress: bool,
    trace_sync: bool,
    shared_barrier: Option<Label>,
    fuse_serial: bool,
    dispatch: Dispatch,
    calls: &mut Vec<(Label, usize)>,
) {
    let exit = a.label();
    let block_loop = a.label();
    let invalid = a.label();
    let frame = Frame::enter(a, true, &[CTX, PART, BLOCK, BIND, TASK, END, STEP, TICKET, DIVISOR], &[], 0);
    // The bounds of a chunk, applied to STEP.
    let clamp = |a: &mut Asm| {
        if dispatch.chunk_min > 1 {
            a.li(T2, dispatch.chunk_min as i64);
            a.maxu(STEP, STEP, T2);
        }
        if dispatch.chunk_max > 0 {
            a.li(T2, dispatch.chunk_max as i64);
            a.minu(STEP, STEP, T2);
        }
    };
    // STEP = max(STEP, 1).
    let at_least_one = |a: &mut Asm| {
        a.seqz(T2, STEP);
        a.add(STEP, STEP, T2);
    };
    a.mv(CTX, A0);
    a.mv(PART, A1);
    a.li(BLOCK, 0);
    a.bind(block_loop);
    a.ld(BIND, CTX, offset_of!(RunContext<'_, '_>, table) as i64);
    a.ld(T2, CTX, offset_of!(RunContext<'_, '_>, block_stride) as i64);
    a.mul(T2, T2, BLOCK);
    a.add(BIND, BIND, T2);
    for (node_index, node) in plan.nodes.iter().enumerate() {
        let keep_barrier = !fuse_serial || !serial_barrier_can_be_fused(plan, node_index);
        if trace_sync {
            emit_trace_stamp(a, Some(node_index), offset_of!(runtime::SyncTraceRecord, work_start));
        }
        match node.operation {
            Operation::Kernel { kernel, worker_limit, distribution, max_tasks } => {
                let empty = a.label();
                let tasks = a.label();
                task_count(a);
                a.li(T2, max_tasks as i64);
                a.branch_far(Cond::Ltu, T2, T0, invalid);
                a.branch_far(Cond::Eq, T0, ZERO, empty);
                // active (t3) = min(team, worker_limit (M = 1 cap), tasks). Zero
                // tasks skip the work, but every participant joins the barrier.
                a.ld(T3, CTX, offset_of!(RunContext<'_, '_>, participants) as i64);
                a.li(T2, worker_limit as i64);
                let workers_ready = a.label();
                a.ld(T4, BIND, offset_of!(TaskBinding, worker_rows) as i64);
                a.beqz(T4, workers_ready);
                a.ld(T4, T4, 0);
                a.li(T5, 1);
                a.bne(T4, T5, workers_ready);
                a.ld(T2, BIND, offset_of!(TaskBinding, worker_limit_m1) as i64);
                a.bind(workers_ready);
                a.minu(T3, T3, T2);
                a.minu(T3, T3, T0);
                a.branch_far(Cond::Geu, PART, T3, empty);
                // Dynamic: the ticket is reset by the barrier that closes
                // this node, so only a node that keeps its barrier takes it.
                let dynamic = (dispatch.grain > 0 && keep_barrier).then(|| a.label());
                match distribution {
                    _ if dynamic.is_some() => {
                        // chunk = max(1, tasks / (active · grain)); TICKET = its
                        // address, DIVISOR = active · grain for the guided chunks.
                        a.li(DIVISOR, dispatch.grain as i64);
                        a.mul(DIVISOR, DIVISOR, T3);
                        a.divu(STEP, T0, DIVISOR);
                        at_least_one(a);
                        clamp(a);
                        session_addr(a, TICKET, dispatch.ticket_offset());
                    }
                    Distribution::Contiguous => {
                        // chunk = ceil(tasks / active), without addition overflow.
                        a.divu(T4, T0, T3);
                        a.remu(T5, T0, T3);
                        a.snez(T5, T5);
                        a.add(T4, T4, T5);
                        a.mul(TASK, PART, T4);
                        a.add(END, TASK, T4);
                        a.minu(END, END, T0);
                        a.li(STEP, 1);
                    }
                    Distribution::Cyclic => {
                        a.mv(TASK, PART);
                        a.mv(END, T0);
                        a.mv(STEP, T3);
                    }
                }
                if let Some(grab) = dynamic {
                    a.bind(grab);
                    if dispatch.guided {
                        // chunk = max(min, remaining / (active · grain)): the read of
                        // the ticket is a hint, the grab below is what counts.
                        task_count(a);
                        a.ld(T2, TICKET, 0);
                        a.branch_far(Cond::Geu, T2, T0, empty);
                        a.sub(T2, T0, T2);
                        a.divu(STEP, T2, DIVISOR);
                        at_least_one(a);
                        clamp(a);
                    }
                    a.amoadd_d(TASK, STEP, TICKET, Order::Relaxed);
                    task_count(a);
                    a.branch_far(Cond::Geu, TASK, T0, empty);
                    a.add(END, TASK, STEP);
                    a.minu(END, END, T0);
                } else {
                    a.branch_far(Cond::Geu, TASK, END, empty);
                }
                a.bind(tasks);
                for (i, reg) in [A0, A1, A2, A3, A4, A5].into_iter().enumerate() {
                    task_argument(a, i, reg, invalid);
                }
                for i in 0..8u8 {
                    a.flw(f(10 + i), BIND, (offset_of!(TaskBinding, floats) + usize::from(i) * 4) as i64);
                }
                a.call(kernels[kernel]);
                if let Some(grab) = dynamic {
                    a.addi(TASK, TASK, 1);
                    a.branch_far(Cond::Ltu, TASK, END, tasks);
                    a.j(grab);
                } else {
                    a.add(TASK, TASK, STEP);
                    a.branch_far(Cond::Ltu, TASK, END, tasks);
                }
                a.bind(empty);
            }
            Operation::LayerEnd { layer } => {
                if with_progress {
                    emit_progress(a, layer, trace_sync, node_index, calls);
                }
            }
        }
        if trace_sync {
            emit_trace_stamp(a, Some(node_index), offset_of!(runtime::SyncTraceRecord, work_end));
            if keep_barrier {
                emit_trace_stamp(a, Some(node_index), offset_of!(runtime::SyncTraceRecord, barrier_start));
            }
        }
        if keep_barrier {
            emit_barrier_call(a, spin, exit, trace_sync, node_index, shared_barrier, calls);
            if trace_sync {
                emit_trace_stamp(a, Some(node_index), offset_of!(runtime::SyncTraceRecord, barrier_end));
            }
        }
        add_imm(a, BIND, BIND, size_of::<TaskBinding>() as i64, T6);
    }
    if with_progress {
        let trace_node = plan.nodes.len();
        emit_progress(a, usize::MAX, trace_sync, trace_node, calls);
        if trace_sync {
            emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, barrier_start));
        }
        emit_barrier_call(a, spin, exit, trace_sync, trace_node, shared_barrier, calls);
        if trace_sync {
            emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, barrier_end));
        }
    }
    a.addi(BLOCK, BLOCK, 1);
    a.ld(T2, CTX, offset_of!(RunContext<'_, '_>, blocks) as i64);
    a.branch_far(Cond::Ltu, BLOCK, T2, block_loop);
    a.j(exit);
    a.bind(invalid);
    // status: 0 → 4, first cause wins.
    session_addr(a, T3, sync_field(offset_of!(runtime::Synchronization, status)));
    let retry = a.here();
    let set = a.label();
    a.lr_d(T0, T3, Order::AcqRel);
    a.bnez(T0, set);
    a.li(T1, 4);
    a.sc_d(T2, T1, T3, Order::Release);
    a.bnez(T2, retry);
    a.bind(set);
    if spin.is_some() {
        a.mv(A0, CTX);
        runtime_call(a, offset_of!(RunContext<'_, '_>, call_notify), calls);
    }
    a.bind(exit);
    frame.leave(a);
    a.ret();
}

/// Store the `time` counter in the calling participant's preallocated
/// record, with 0 as the auxiliary word (no core number in user space). Uses
/// `a4..a7`, which no barrier or node holds live.
fn emit_trace_stamp(a: &mut Asm, node: Option<usize>, field: usize) {
    use runtime::SyncTraceRecord as T;
    let aux_field = match field {
        f if f == offset_of!(T, work_start) => offset_of!(T, work_start_aux),
        f if f == offset_of!(T, work_end) => offset_of!(T, work_end_aux),
        f if f == offset_of!(T, progress_start) => offset_of!(T, progress_start_aux),
        f if f == offset_of!(T, progress_end) => offset_of!(T, progress_end_aux),
        f if f == offset_of!(T, barrier_start) => offset_of!(T, barrier_start_aux),
        f if f == offset_of!(T, sleep_start) => offset_of!(T, sleep_start_aux),
        f if f == offset_of!(T, sleep_end) => offset_of!(T, sleep_end_aux),
        f if f == offset_of!(T, barrier_end) => offset_of!(T, barrier_end_aux),
        _ => unreachable!("unknown synchronization trace field"),
    };
    let done = a.label();
    a.ld(A4, CTX, offset_of!(RunContext<'_, '_>, trace) as i64);
    a.beqz(A4, done);
    a.ld(A5, CTX, offset_of!(RunContext<'_, '_>, trace_nodes_per_block) as i64);
    a.mul(A5, BLOCK, A5);
    match node {
        Some(n) => {
            a.li(A6, n as i64);
            a.add(A5, A5, A6);
        }
        None => a.add(A5, A5, BIND),
    }
    a.ld(A6, CTX, offset_of!(RunContext<'_, '_>, participants) as i64);
    a.mul(A5, A5, A6);
    a.add(A5, A5, PART);
    a.li(A6, size_of::<T>() as i64);
    a.mul(A5, A5, A6);
    a.add(A4, A4, A5);
    a.fence(Fence::RW, Fence::RW);
    a.rdtime(A7);
    a.sd(A7, A4, field as i64);
    a.sw(ZERO, A4, aux_field as i64);
    a.bind(done);
}

fn emit_progress(a: &mut Asm, layer: usize, trace_sync: bool, trace_node: usize, calls: &mut Vec<(Label, usize)>) {
    let done = a.label();
    a.bnez(PART, done);
    if trace_sync {
        emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, progress_start));
    }
    a.mv(A0, CTX);
    a.mv(A1, BLOCK);
    a.li(A2, layer as i64);
    runtime_call(a, offset_of!(RunContext<'_, '_>, call_progress), calls);
    if trace_sync {
        emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, progress_end));
    }
    a.bind(done);
}
