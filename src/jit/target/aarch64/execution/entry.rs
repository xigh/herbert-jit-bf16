//! The AArch64 whole-call program: the entry loop, the barriers, the progress
//! bridge and the trace stamps around the kernels of a plan, assembled into
//! one page, and the decoder audit of its runtime calls.
//!
//! The same program as the x86-64 one, instruction for instruction where the
//! ISAs allow it. What changes is the memory model: the arrival at a barrier
//! is an `ldaddal` (release of this participant's writes, acquire of the
//! others'), the last arrival re-arms `remaining` and publishes the next
//! generation with another `ldaddal`, and the waiters read the generation
//! with `ldar`. The trace reads `cntvct_el0` behind an `isb`, the core
//! number from `tpidrro_el0`; there is no user PMU on macOS.

use super::*;
use crate::jit::execution::runtime::{self, RunContext};
use crate::jit::target::aarch64::util::{exec_err, Frame};
use crate::jit::target::Assembled;
use a64::*;
use std::mem::{offset_of, size_of};

// The registers the entry keeps across every kernel call (callee-saved).
const CTX: R = X19;
const PART: R = X20;
const BLOCK: R = X21;
const BIND: R = X22;
const TASK: R = X23;
const END: R = X24;
const STEP: R = X25;

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

fn ctx_field(field: usize) -> Mem {
    mem(CTX, field as i64)
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

/// Decode the final RX image and verify every call: `bl` stays inside the
/// image, and every `blr` is a registered runtime call through `x16`, loaded
/// by the instruction just before it from one of the allowed `RunContext`
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
                // `ldr x16, [x19, #field]`: unsigned offset form, 8-byte scale.
                let expected = 0xf940_0000 | (((*field / 8) as u32) << 10) | (19 << 5) | 16;
                if !fields.contains(field) || insn.target_reg != Some(16) || load != Some(expected) || *field % 8 != 0 {
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
    a.ldr(X16, ctx_field(field));
    let at = a.here();
    a.blr(X16);
    calls.push((at, field));
}

/// The barrier, inline. Uses `x9..x17`; `exit` is taken on a nonzero status.
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
    let sync = offset_of!(ExecutionSession, sync);
    let remaining = sync + offset_of!(runtime::Synchronization, remaining);
    let generation = sync + offset_of!(runtime::Synchronization, generation);
    let status = sync + offset_of!(runtime::Synchronization, status);
    let last = a.label();
    let wait_loop = a.label();
    let done = a.label();
    a.ldr(X10, ctx_field(offset_of!(RunContext<'_, '_>, session)));
    a.ldr(X9, mem(X10, status as i64));
    a.cbnz(X9, exit);
    a.add(X13, X10, remaining as i64);
    a.add(X14, X10, generation as i64);
    // The generation is read before the arrival is published.
    a.ldr(X11, at(X14));
    a.movn(X9, 0, 0);
    a.ldadd(Order::AcqRel, X9, X12, X13);
    a.cmp(X12, 1);
    a.beq(last);
    if let Some(budget) = spin {
        a.mov_imm(W15, u64::from(budget));
    }
    a.bind(wait_loop);
    a.ldr(X9, mem(X10, status as i64));
    a.cbnz(X9, exit);
    a.ldar(X9, X14);
    a.cmp(X9, X11);
    a.bne(done);
    if spin.is_some() {
        let pause = a.label();
        a.cbnz(W15, pause);
        if trace_sync {
            emit_trace_stamp(a, trace_node, offset_of!(runtime::SyncTraceRecord, sleep_start));
        }
        a.mov(X0, CTX);
        a.mov(X1, X11);
        runtime_call(a, offset_of!(RunContext<'_, '_>, call_wait), calls);
        if trace_sync {
            emit_trace_stamp(a, trace_node, offset_of!(runtime::SyncTraceRecord, sleep_end));
        }
        a.ldr(X10, ctx_field(offset_of!(RunContext<'_, '_>, session)));
        a.ldr(X9, mem(X10, status as i64));
        a.cbnz(X9, exit);
        a.b(done);
        a.bind(pause);
        a.sub(W15, W15, 1);
    }
    a.isb();
    a.b(wait_loop);
    a.bind(last);
    // Re-arm, then publish the next generation: the release of this
    // `ldaddal` orders the re-arming, the ticket back to zero and every write
    // of the team before it. Every participant has taken its last ticket of
    // this node before arriving.
    a.ldr(X9, ctx_field(offset_of!(RunContext<'_, '_>, participants)));
    a.str(X9, at(X13));
    a.str(XZR, mem(X10, dispatch().ticket_offset() as i64));
    a.mov_imm(X9, 1);
    a.ldadd(Order::AcqRel, X9, X12, X14);
    if spin.is_some() {
        a.mov(X0, CTX);
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
        a.mov(X0, CTX);
        a.mov(X1, PART);
        a.mov(X2, BLOCK);
        a.mov_imm(X3, trace_node as u64);
        a.bl(barrier);
        a.cbnz(X0, exit);
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
    let frame = Frame::enter(a, true, &[CTX, PART, BLOCK, BIND], false, 0);
    a.mov(CTX, X0);
    a.mov(PART, X1);
    a.mov(BLOCK, X2);
    a.mov(BIND, X3);
    emit_inline_barrier(a, spin, abort, trace_sync, None, calls);
    a.mov_imm(X0, 0);
    a.b(epilogue);
    a.bind(abort);
    a.ldr(X10, ctx_field(offset_of!(RunContext<'_, '_>, session)));
    a.ldr(
        X0,
        mem(X10, (offset_of!(ExecutionSession, sync) + offset_of!(runtime::Synchronization, status)) as i64),
    );
    a.bind(epilogue);
    frame.leave(a);
    a.ret();
}

/// `x8` = the task count of the current node: `*task_count`, or the static count.
fn task_count(a: &mut Asm) {
    let ready = a.label();
    a.ldr(X8, mem(BIND, offset_of!(TaskBinding, tasks) as i64));
    a.ldr(X9, mem(BIND, offset_of!(TaskBinding, task_count) as i64));
    a.cbz(X9, ready);
    a.ldr(X8, at(X9));
    a.bind(ready);
}

/// The value of one integer argument of a task, into `dst`:
/// `base + stride·task + participant_stride·participant + dynamic base`,
/// every addition checked for overflow.
fn task_argument(a: &mut Asm, i: usize, dst: R, invalid: Label) {
    let argument = offset_of!(TaskBinding, arguments) + i * size_of::<Argument>();
    a.ldr(dst, mem(BIND, (argument + 8) as i64));
    a.mul(dst, dst, TASK);
    a.ldr(X9, mem(BIND, argument as i64));
    a.adds(dst, dst, X9);
    a.b_cond(Cond::Hs, invalid);
    a.ldr(X9, mem(BIND, (offset_of!(TaskBinding, participant_strides) + i * 8) as i64));
    a.mul(X9, X9, PART);
    a.adds(dst, dst, X9);
    a.b_cond(Cond::Hs, invalid);
    let ready = a.label();
    a.ldr(X9, mem(BIND, (offset_of!(TaskBinding, argument_bases) + i * 8) as i64));
    a.cbz(X9, ready);
    a.ldr(X9, at(X9));
    a.adds(dst, dst, X9);
    a.b_cond(Cond::Hs, invalid);
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
    // x26 = the ticket, x27 = participants · grain (the guided divisor).
    let frame = Frame::enter(a, true, &[CTX, PART, BLOCK, BIND, TASK, END, STEP, X26, X27, X28], false, 0);
    // The bounds of a chunk, applied to STEP.
    let clamp = |a: &mut Asm| {
        if dispatch.chunk_min > 1 {
            a.mov_imm(X9, dispatch.chunk_min as u64);
            a.cmp(STEP, X9);
            a.csel(STEP, X9, STEP, Cond::Lo);
        }
        if dispatch.chunk_max > 0 {
            a.mov_imm(X9, dispatch.chunk_max as u64);
            a.cmp(STEP, X9);
            a.csel(STEP, X9, STEP, Cond::Hi);
        }
    };
    a.mov(CTX, X0);
    a.mov(PART, X1);
    a.mov_imm(BLOCK, 0);
    a.bind(block_loop);
    a.ldr(BIND, ctx_field(offset_of!(RunContext<'_, '_>, table)));
    a.ldr(X9, ctx_field(offset_of!(RunContext<'_, '_>, block_stride)));
    a.madd(BIND, X9, BLOCK, BIND);
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
                a.mov_imm(X9, max_tasks as u64);
                a.cmp(X8, X9);
                a.b_cond(Cond::Hi, invalid);
                a.cbz(X8, empty);
                // active = min(team, worker_limit (M = 1 cap), tasks). Zero
                // tasks skip the work, but every participant joins the barrier.
                a.ldr(X10, ctx_field(offset_of!(RunContext<'_, '_>, participants)));
                a.mov_imm(X9, worker_limit as u64);
                let workers_ready = a.label();
                a.ldr(X11, mem(BIND, offset_of!(TaskBinding, worker_rows) as i64));
                a.cbz(X11, workers_ready);
                a.ldr(X11, at(X11));
                a.cmp(X11, 1);
                a.bne(workers_ready);
                a.ldr(X9, mem(BIND, offset_of!(TaskBinding, worker_limit_m1) as i64));
                a.bind(workers_ready);
                a.cmp(X10, X9);
                a.csel(X10, X9, X10, Cond::Hi);
                a.cmp(X10, X8);
                a.csel(X10, X8, X10, Cond::Hi);
                a.cmp(PART, X10);
                a.bhs(empty);
                // Dynamic: the ticket is reset by the barrier that closes
                // this node, so only a node that keeps its barrier takes it.
                let dynamic = (dispatch.grain > 0 && keep_barrier).then(|| a.label());
                match distribution {
                    _ if dynamic.is_some() => {
                        // chunk = max(1, tasks / (active · grain)); x26 = the ticket,
                        // x27 = active · grain for the guided recomputation.
                        a.mov_imm(X27, dispatch.grain as u64);
                        a.mul(X27, X27, X10);
                        a.udiv(STEP, X8, X27);
                        a.cmp(STEP, 0);
                        a.cinc(STEP, STEP, Cond::Eq);
                        clamp(a);
                        a.ldr(X26, ctx_field(offset_of!(RunContext<'_, '_>, session)));
                        a.add(X26, X26, dispatch.ticket_offset() as i64);
                    }
                    Distribution::Contiguous => {
                        // chunk = ceil(tasks / active), without addition overflow.
                        a.udiv(X11, X8, X10);
                        a.msub(X12, X11, X10, X8);
                        a.cmp(X12, 0);
                        a.cinc(X11, X11, Cond::Ne);
                        a.mul(TASK, PART, X11);
                        a.add(END, TASK, X11);
                        a.cmp(END, X8);
                        a.csel(END, X8, END, Cond::Hi);
                        a.mov_imm(STEP, 1);
                    }
                    Distribution::Cyclic => {
                        a.mov(TASK, PART);
                        a.mov(END, X8);
                        a.mov(STEP, X10);
                    }
                }
                if let Some(grab) = dynamic {
                    a.bind(grab);
                    if dispatch.guided {
                        // chunk = max(min, remaining / (active · grain)): the read of
                        // the ticket is a hint, the grab below is what counts.
                        task_count(a);
                        a.ldr(X9, at(X26));
                        a.subs(X9, X8, X9);
                        a.b_cond(Cond::Ls, empty);
                        a.udiv(STEP, X9, X27);
                        a.cmp(STEP, 0);
                        a.cinc(STEP, STEP, Cond::Eq);
                        clamp(a);
                    }
                    a.ldadd(Order::Relaxed, STEP, TASK, X26);
                    task_count(a);
                    a.cmp(TASK, X8);
                    a.bhs(empty);
                    a.add(END, TASK, STEP);
                    a.cmp(END, X8);
                    a.csel(END, X8, END, Cond::Hi);
                } else {
                    a.cmp(TASK, END);
                    a.bhs(empty);
                }
                a.bind(tasks);
                for (i, reg) in [X0, X1, X2, X3, X4, X5].into_iter().enumerate() {
                    task_argument(a, i, reg, invalid);
                }
                for i in 0..8u8 {
                    a.ldr(s(i), mem(BIND, (offset_of!(TaskBinding, floats) + usize::from(i) * 4) as i64));
                }
                a.bl(kernels[kernel]);
                if let Some(grab) = dynamic {
                    a.add(TASK, TASK, 1);
                    a.cmp(TASK, END);
                    a.blo(tasks);
                    a.b(grab);
                } else {
                    a.add(TASK, TASK, STEP);
                    a.cmp(TASK, END);
                    a.blo(tasks);
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
        a.add(BIND, BIND, size_of::<TaskBinding>() as i64);
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
    a.add(BLOCK, BLOCK, 1);
    a.ldr(X9, ctx_field(offset_of!(RunContext<'_, '_>, blocks)));
    a.cmp(BLOCK, X9);
    a.blo(block_loop);
    a.b(exit);
    a.bind(invalid);
    // status: 0 → 4, first cause wins.
    a.ldr(X10, ctx_field(offset_of!(RunContext<'_, '_>, session)));
    a.add(X13, X10, (offset_of!(ExecutionSession, sync) + offset_of!(runtime::Synchronization, status)) as i64);
    a.mov_imm(X9, 0);
    a.mov_imm(X12, 4);
    a.cas(Order::AcqRel, X9, X12, X13);
    if spin.is_some() {
        a.mov(X0, CTX);
        runtime_call(a, offset_of!(RunContext<'_, '_>, call_notify), calls);
    }
    a.bind(exit);
    frame.leave(a);
    a.ret();
}

/// Store the virtual counter in the calling participant's preallocated record,
/// with the core number (`tpidrro_el0`, low byte on Darwin) as the auxiliary
/// word. Every register it uses is restored.
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
    a.stp(X0, X1, pre(SP, -32));
    a.stp(X2, X3, mem(SP, 16));
    a.ldr(X0, ctx_field(offset_of!(RunContext<'_, '_>, trace)));
    a.cbz(X0, done);
    a.ldr(X1, ctx_field(offset_of!(RunContext<'_, '_>, trace_nodes_per_block)));
    a.mul(X1, BLOCK, X1);
    match node {
        Some(n) => {
            a.mov_imm(X2, n as u64);
            a.add(X1, X1, X2);
        }
        None => a.add(X1, X1, BIND),
    }
    a.ldr(X2, ctx_field(offset_of!(RunContext<'_, '_>, participants)));
    a.madd(X1, X1, X2, PART);
    a.mov_imm(X2, size_of::<T>() as u64);
    a.madd(X0, X1, X2, X0);
    a.isb();
    a.mrs(X3, SysReg::CNTVCT_EL0);
    a.str(X3, mem(X0, field as i64));
    a.mrs(X3, SysReg::TPIDRRO_EL0);
    a.and(W3, W3, 0xff);
    a.str(W3, mem(X0, aux_field as i64));
    a.bind(done);
    a.ldp(X2, X3, mem(SP, 16));
    a.ldp(X0, X1, post(SP, 32));
}

fn emit_progress(a: &mut Asm, layer: usize, trace_sync: bool, trace_node: usize, calls: &mut Vec<(Label, usize)>) {
    let done = a.label();
    a.cbnz(PART, done);
    if trace_sync {
        emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, progress_start));
    }
    a.mov(X0, CTX);
    a.mov(X1, BLOCK);
    a.mov_imm(X2, layer as u64);
    runtime_call(a, offset_of!(RunContext<'_, '_>, call_progress), calls);
    if trace_sync {
        emit_trace_stamp(a, Some(trace_node), offset_of!(runtime::SyncTraceRecord, progress_end));
    }
    a.bind(done);
}
