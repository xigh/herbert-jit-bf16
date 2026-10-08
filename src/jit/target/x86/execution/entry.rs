//! The x86-64 whole-call program: the entry loop, the barriers, the progress
//! bridge and the instrumentation stamps around the kernels of a plan,
//! assembled into one page, and the decoder audit of its runtime calls.

use super::*;
use crate::jit::execution::runtime::{self, RunContext};
use crate::jit::target::Assembled;
use iced_x86::code_asm::*;

fn error(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
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
    let mut a = CodeAssembler::new(64).map_err(error)?;
    let mut kernels: Vec<_> = plan.kernels.iter().map(|_| a.create_label()).collect();
    let mut entries = [a.create_label(), a.create_label()];
    let mut barrier = a.create_label();
    // Absolute runtime calls are annotated; the final decoder verifies that
    // every indirect CALL uses one of these immediate targets.
    let mut runtime_labels = Vec::new();
    for (i, entry) in entries.iter_mut().enumerate() {
        a.set_label(entry).map_err(error)?;
        emit_entry(
            &mut a,
            plan,
            &kernels,
            hybrid_spin,
            i == 1,
            trace_sync,
            trace_pmu,
            shared_barrier.then_some(barrier),
            fuse_serial,
            &mut runtime_labels,
        )
        .map_err(error)?;
    }
    if shared_barrier {
        a.set_label(&mut barrier).map_err(error)?;
        emit_barrier_function(
            &mut a,
            hybrid_spin,
            trace_sync,
            trace_pmu,
            &mut runtime_labels,
        )
        .map_err(error)?;
    }
    for (kernel, label) in plan.kernels.iter().zip(&mut kernels) {
        a.set_label(label).map_err(error)?;
        a.db(&kernel.bytes).map_err(error)?;
    }
    let assembled = a
        .assemble_options(
            0,
            iced_x86::BlockEncoderOptions::RETURN_NEW_INSTRUCTION_OFFSETS,
        )
        .map_err(error)?;
    let entry_offsets = [
        assembled.label_ip(&entries[0]).map_err(error)? as usize,
        assembled.label_ip(&entries[1]).map_err(error)? as usize,
    ];
    let kernel_offsets = kernels
        .iter()
        .map(|l| assembled.label_ip(l).map(|p| p as usize).map_err(error))
        .collect::<Result<Vec<_>, _>>()?;
    let runtime_calls = runtime_labels
        .iter()
        .map(|(l, target)| {
            assembled
                .label_ip(l)
                .map(|p| (p as usize, *target))
                .map_err(error)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Assembled {
        code: assembled.inner.code_buffer,
        entry_offsets,
        kernel_offsets,
        runtime_calls,
    })
}

/// Decode the final RX image and verify every CALL destination. Generated
/// kernels use relative calls; runtime bridges use a checked MOV/CALL pair.
pub(crate) fn audit_calls(code: &[u8], runtime_calls: &[(usize, usize)]) -> Result<(), ExecutionError> {
    use iced_x86::{Decoder, DecoderOptions, FlowControl, OpKind, Register};
    let fields = runtime::runtime_call_fields();
    let mut d = Decoder::new(64, code, DecoderOptions::NONE);
    let mut found = 0;
    while d.can_decode() {
        let i = d.decode();
        match i.flow_control() {
            FlowControl::Call => {
                if i.near_branch_target() >= code.len() as u64 {
                    return Err(ExecutionError("emitted call leaves the RX image".into()));
                }
            }
            FlowControl::IndirectCall => {
                let Some((_, field)) = runtime_calls.iter().find(|(ip, _)| *ip == i.ip() as usize)
                else {
                    return Err(ExecutionError("unregistered indirect call".into()));
                };
                if !fields.contains(field)
                    || i.op0_kind() != OpKind::Memory
                    || i.memory_base() != Register::RBX
                    || i.memory_index() != Register::None
                    || i.memory_displacement64() != *field as u64
                {
                    return Err(ExecutionError("runtime call target mismatch".into()));
                }
                found += 1;
            }
            _ => (),
        }
    }
    if found != runtime_calls.len() {
        return Err(ExecutionError("runtime call audit count mismatch".into()));
    }
    Ok(())
}

/// Call the runtime entry stored in the context field at `field` (rbx holds the context).
fn runtime_call(
    a: &mut CodeAssembler,
    field: usize,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    let mut label = a.create_label();
    a.set_label(&mut label)?;
    a.call(qword_ptr(rbx + field as i32))?;
    calls.push((label, field));
    Ok(())
}

fn emit_inline_barrier(
    a: &mut CodeAssembler,
    spin: Option<u32>,
    exit: CodeLabel,
    trace_sync: bool,
    trace_pmu: bool,
    trace_node: Option<usize>,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    // rbx context; r12 participant; r13 block; r14 binding; r15 task.
    // Session starts with sync? Use offset_of rather than assuming Rust layout.
    let sync = std::mem::offset_of!(ExecutionSession, sync) as i32;
    let remaining = sync + std::mem::offset_of!(runtime::Synchronization, remaining) as i32;
    let generation = sync + std::mem::offset_of!(runtime::Synchronization, generation) as i32;
    let status = sync + std::mem::offset_of!(runtime::Synchronization, status) as i32;
    let mut last = a.create_label();
    let mut wait_loop = a.create_label();
    let mut done = a.create_label();
    a.mov(
        r10,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, session) as i32),
    )?;
    a.cmp(qword_ptr(r10 + status), 0)?;
    a.jne(exit)?;
    a.mov(r11, qword_ptr(r10 + generation))?;
    a.lock().dec(qword_ptr(r10 + remaining))?;
    a.je(last)?;
    if let Some(budget) = spin {
        a.mov(ecx, budget)?;
    }
    a.set_label(&mut wait_loop)?;
    a.cmp(qword_ptr(r10 + status), 0)?;
    a.jne(exit)?;
    a.cmp(qword_ptr(r10 + generation), r11)?;
    a.jne(done)?;
    if spin.is_some() {
        let mut pause = a.create_label();
        a.test(ecx, ecx)?;
        a.jne(pause)?;
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                trace_node,
                std::mem::offset_of!(runtime::SyncTraceRecord, sleep_start),
                trace_sync,
                trace_pmu,
            )?;
        }
        a.mov(rdi, rbx)?;
        a.mov(rsi, r11)?;
        runtime_call(a, std::mem::offset_of!(RunContext<'_, '_>, call_wait), calls)?;
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                trace_node,
                std::mem::offset_of!(runtime::SyncTraceRecord, sleep_end),
                trace_sync,
                trace_pmu,
            )?;
        }
        a.mov(
            r10,
            qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, session) as i32),
        )?;
        a.cmp(qword_ptr(r10 + status), 0)?;
        a.jne(exit)?;
        a.jmp(done)?;
        a.set_label(&mut pause)?;
        a.dec(ecx)?;
    }
    a.pause()?;
    a.jmp(wait_loop)?;
    a.set_label(&mut last)?;
    a.mov(
        rax,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, participants) as i32),
    )?;
    a.mov(qword_ptr(r10 + remaining), rax)?;
    a.lock().inc(qword_ptr(r10 + generation))?;
    if spin.is_some() {
        a.mov(rdi, rbx)?;
        runtime_call(a, std::mem::offset_of!(RunContext<'_, '_>, call_notify), calls)?;
    }
    a.set_label(&mut done)?;
    // A label must attach to an instruction before another label is defined.
    a.nop()?;
    Ok(())
}

fn emit_barrier_call(
    a: &mut CodeAssembler,
    spin: Option<u32>,
    exit: CodeLabel,
    trace_sync: bool,
    trace_pmu: bool,
    trace_node: usize,
    shared: Option<CodeLabel>,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    if let Some(barrier) = shared {
        a.mov(rdi, rbx)?;
        a.mov(rsi, r12)?;
        a.mov(rdx, r13)?;
        a.mov(rcx, trace_node as u64)?;
        a.call(barrier)?;
        a.test(rax, rax)?;
        a.jne(exit)?;
        Ok(())
    } else {
        emit_inline_barrier(
            a,
            spin,
            exit,
            trace_sync,
            trace_pmu,
            Some(trace_node),
            calls,
        )
    }
}

/// Emit one SysV64 barrier function shared by both whole-call entries. Its
/// arguments are `(context, participant, block, trace_node)` and its return is
/// the shared status. Generated callers leave the whole-call loop on nonzero.
fn emit_barrier_function(
    a: &mut CodeAssembler,
    spin: Option<u32>,
    trace_sync: bool,
    trace_pmu: bool,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    let mut abort = a.create_label();
    let mut epilogue = a.create_label();
    for reg in [rbx, r12, r13, r14] {
        a.push(reg)?;
    }
    // Entry RSP is 8 mod 16; four pushes retain it. Reserve one word so calls
    // to the explicit sysv64 runtime bridges see a 16-byte-aligned caller RSP.
    a.sub(rsp, 8)?;
    a.mov(rbx, rdi)?;
    a.mov(r12, rsi)?;
    a.mov(r13, rdx)?;
    a.mov(r14, rcx)?;
    emit_inline_barrier(a, spin, abort, trace_sync, trace_pmu, None, calls)?;
    a.xor(eax, eax)?;
    a.jmp(epilogue)?;
    a.set_label(&mut abort)?;
    a.mov(
        r10,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, session) as i32),
    )?;
    a.mov(
        rax,
        qword_ptr(
            r10 + (std::mem::offset_of!(ExecutionSession, sync)
                + std::mem::offset_of!(runtime::Synchronization, status)) as i32,
        ),
    )?;
    a.set_label(&mut epilogue)?;
    a.add(rsp, 8)?;
    for reg in [r14, r13, r12, rbx] {
        a.pop(reg)?;
    }
    a.ret()?;
    Ok(())
}

fn emit_entry(
    a: &mut CodeAssembler,
    plan: &ExecutionPlan,
    kernels: &[CodeLabel],
    spin: Option<u32>,
    with_progress: bool,
    trace_sync: bool,
    trace_pmu: bool,
    shared_barrier: Option<CodeLabel>,
    fuse_serial: bool,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    let mut exit = a.create_label();
    let mut block_loop = a.create_label();
    let mut invalid = a.create_label();
    for reg in [rbx, rbp, r12, r13, r14, r15] {
        a.push(reg)?;
    }
    // SysV64 on EVERY OS. Six saved registers leave RSP % 16 == 8;
    // reserve 24 bytes for end/step and alignment before generated/runtime calls.
    a.sub(rsp, 24)?;
    a.mov(rbx, rdi)?;
    a.mov(r12, rsi)?;
    a.xor(r13d, r13d)?;
    a.set_label(&mut block_loop)?;
    a.mov(
        r14,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, table) as i32),
    )?;
    a.mov(
        rax,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, block_stride) as i32),
    )?;
    a.imul_2(rax, r13)?;
    a.add(r14, rax)?;
    for (node_index, node) in plan.nodes.iter().enumerate() {
        let keep_barrier = !fuse_serial || !serial_barrier_can_be_fused(plan, node_index);
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                Some(node_index),
                std::mem::offset_of!(runtime::SyncTraceRecord, work_start),
                trace_sync,
                trace_pmu,
            )?;
        }
        match node.operation {
            Operation::Kernel {
                kernel,
                worker_limit,
                distribution,
                max_tasks,
            } => {
                let mut empty = a.create_label();
                let mut tasks = a.create_label();
                // active = min(team, worker_limit, tasks). Zero tasks skip work,
                // but every participant still executes the barrier below.
                let mut count_ready = a.create_label();
                a.mov(
                    r8,
                    qword_ptr(r14 + std::mem::offset_of!(TaskBinding, tasks) as i32),
                )?;
                a.mov(
                    rax,
                    qword_ptr(r14 + std::mem::offset_of!(TaskBinding, task_count) as i32),
                )?;
                a.test(rax, rax)?;
                a.je(count_ready)?;
                a.mov(r8, qword_ptr(rax))?;
                a.set_label(&mut count_ready)?;
                a.mov(rax, max_tasks as u64)?;
                a.cmp(r8, rax)?;
                a.ja(invalid)?;
                a.test(r8, r8)?;
                a.je(empty)?;
                a.mov(
                    rcx,
                    qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, participants) as i32),
                )?;
                a.mov(rax, worker_limit as u64)?;
                let mut workers_ready = a.create_label();
                a.mov(
                    r10,
                    qword_ptr(r14 + std::mem::offset_of!(TaskBinding, worker_rows) as i32),
                )?;
                a.test(r10, r10)?;
                a.je(workers_ready)?;
                a.cmp(qword_ptr(r10), 1)?;
                a.jne(workers_ready)?;
                a.mov(
                    rax,
                    qword_ptr(r14 + std::mem::offset_of!(TaskBinding, worker_limit_m1) as i32),
                )?;
                a.set_label(&mut workers_ready)?;
                a.cmp(rcx, rax)?;
                a.cmova(rcx, rax)?;
                a.cmp(rcx, r8)?;
                a.cmova(rcx, r8)?;
                a.cmp(r12, rcx)?;
                a.jae(empty)?;
                match distribution {
                    Distribution::Contiguous => {
                        // ceil(tasks / active), without addition overflow.
                        a.mov(rax, r8)?;
                        a.xor(edx, edx)?;
                        a.div(rcx)?;
                        a.test(rdx, rdx)?;
                        a.setne(dl)?;
                        a.movzx(edx, dl)?;
                        a.add(rax, rdx)?;
                        a.mov(r15, r12)?;
                        a.imul_2(r15, rax)?;
                        a.add(rax, r15)?;
                        a.cmp(rax, r8)?;
                        a.cmova(rax, r8)?;
                        a.mov(qword_ptr(rsp), rax)?;
                        a.mov(qword_ptr(rsp + 8), 1)?;
                    }
                    Distribution::Cyclic => {
                        a.mov(r15, r12)?;
                        a.mov(qword_ptr(rsp), r8)?;
                        a.mov(qword_ptr(rsp + 8), rcx)?;
                    }
                }
                a.cmp(r15, qword_ptr(rsp))?;
                a.jae(empty)?;
                a.set_label(&mut tasks)?;
                for (i, reg) in [rdi, rsi, rdx, rcx, r8, r9].into_iter().enumerate() {
                    let offset = (std::mem::offset_of!(TaskBinding, arguments)
                        + i * std::mem::size_of::<Argument>())
                        as i32;
                    a.mov(rax, qword_ptr(r14 + offset + 8))?;
                    a.imul_2(rax, r15)?;
                    a.add(rax, qword_ptr(r14 + offset))?;
                    a.jc(invalid)?;
                    a.mov(
                        r10,
                        qword_ptr(
                            r14 + (std::mem::offset_of!(TaskBinding, participant_strides) + i * 8)
                                as i32,
                        ),
                    )?;
                    a.imul_2(r10, r12)?;
                    a.add(rax, r10)?;
                    a.jc(invalid)?;
                    let mut base_ready = a.create_label();
                    a.mov(
                        r10,
                        qword_ptr(
                            r14 + (std::mem::offset_of!(TaskBinding, argument_bases) + i * 8)
                                as i32,
                        ),
                    )?;
                    a.test(r10, r10)?;
                    a.je(base_ready)?;
                    a.add(rax, qword_ptr(r10))?;
                    a.jc(invalid)?;
                    a.set_label(&mut base_ready)?;
                    a.mov(reg, rax)?;
                }
                for (i, reg) in [xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7]
                    .into_iter()
                    .enumerate()
                {
                    a.movd(
                        reg,
                        dword_ptr(r14 + (std::mem::offset_of!(TaskBinding, floats) + i * 4) as i32),
                    )?;
                }
                a.call(kernels[kernel])?;
                a.add(r15, qword_ptr(rsp + 8))?;
                a.cmp(r15, qword_ptr(rsp))?;
                a.jb(tasks)?;
                a.set_label(&mut empty)?;
                a.nop()?;
            }
            Operation::LayerEnd { layer } => {
                if with_progress {
                    emit_progress(a, layer, trace_sync, trace_pmu, node_index, calls)?;
                }
            }
        }
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                Some(node_index),
                std::mem::offset_of!(runtime::SyncTraceRecord, work_end),
                trace_sync,
                trace_pmu,
            )?;
            if keep_barrier {
                emit_instrument_stamp(
                    a,
                    Some(node_index),
                    std::mem::offset_of!(runtime::SyncTraceRecord, barrier_start),
                    trace_sync,
                    trace_pmu,
                )?;
            }
        }
        if keep_barrier {
            emit_barrier_call(
                a,
                spin,
                exit,
                trace_sync,
                trace_pmu,
                node_index,
                shared_barrier,
                calls,
            )?;
            if trace_sync || trace_pmu {
                emit_instrument_stamp(
                    a,
                    Some(node_index),
                    std::mem::offset_of!(runtime::SyncTraceRecord, barrier_end),
                    trace_sync,
                    trace_pmu,
                )?;
            }
        }
        a.add(r14, std::mem::size_of::<TaskBinding>() as i32)?;
    }
    if with_progress {
        let trace_node = plan.nodes.len();
        emit_progress(a, usize::MAX, trace_sync, trace_pmu, trace_node, calls)?;
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                Some(trace_node),
                std::mem::offset_of!(runtime::SyncTraceRecord, barrier_start),
                trace_sync,
                trace_pmu,
            )?;
        }
        emit_barrier_call(
            a,
            spin,
            exit,
            trace_sync,
            trace_pmu,
            trace_node,
            shared_barrier,
            calls,
        )?;
        if trace_sync || trace_pmu {
            emit_instrument_stamp(
                a,
                Some(trace_node),
                std::mem::offset_of!(runtime::SyncTraceRecord, barrier_end),
                trace_sync,
                trace_pmu,
            )?;
        }
    }
    a.inc(r13)?;
    a.cmp(
        r13,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, blocks) as i32),
    )?;
    a.jb(block_loop)?;
    a.jmp(exit)?;
    a.set_label(&mut invalid)?;
    a.mov(
        r10,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, session) as i32),
    )?;
    a.xor(eax, eax)?;
    a.mov(r11d, 4)?;
    a.lock().cmpxchg(
        qword_ptr(
            r10 + (std::mem::offset_of!(ExecutionSession, sync)
                + std::mem::offset_of!(runtime::Synchronization, status)) as i32,
        ),
        r11,
    )?;
    if spin.is_some() {
        a.mov(rdi, rbx)?;
        runtime_call(a, std::mem::offset_of!(RunContext<'_, '_>, call_notify), calls)?;
    }
    a.set_label(&mut exit)?;
    a.add(rsp, 24)?;
    for reg in [r15, r14, r13, r12, rbp, rbx] {
        a.pop(reg)?;
    }
    a.ret()?;
    Ok(())
}

fn emit_instrument_stamp(
    a: &mut CodeAssembler,
    node: Option<usize>,
    field: usize,
    trace_sync: bool,
    trace_pmu: bool,
) -> Result<(), iced_x86::IcedError> {
    if trace_sync {
        emit_trace_stamp(a, node, field)?;
    }
    if trace_pmu {
        emit_pmu_stamp(a, node, field)?;
    }
    Ok(())
}

/// Read one Linux perf counter directly from its mmap metadata page. The
/// emitted seqlock protocol follows perf_event_open(2): the current hardware
/// index and signed counter width are sampled on the participant that owns the
/// event. No runtime bridge is involved.
fn emit_pmu_stamp(
    a: &mut CodeAssembler,
    node: Option<usize>,
    field: usize,
) -> Result<(), iced_x86::IcedError> {
    let mut retry = a.create_label();
    let mut unavailable = a.create_label();
    let mut store = a.create_label();
    let mut done = a.create_label();
    for reg in [rax, rcx, rdx, rbp, r8, r9, r10, r11] {
        a.push(reg)?;
    }
    a.mov(
        r10,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, pmu_pages) as i32),
    )?;
    a.test(r10, r10)?;
    a.je(done)?;
    a.mov(r10, qword_ptr(r10 + r12 * 8))?;
    a.mov(
        r11,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, pmu_trace) as i32),
    )?;
    a.mov(rax, r13)?;
    a.mov(
        rcx,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, trace_nodes_per_block) as i32),
    )?;
    a.imul_2(rax, rcx)?;
    if let Some(node) = node {
        a.add(rax, node as i32)?;
    } else {
        a.add(rax, r14)?;
    }
    a.mov(
        rcx,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, participants) as i32),
    )?;
    a.imul_2(rax, rcx)?;
    a.add(rax, r12)?;
    a.imul_3(
        rax,
        rax,
        std::mem::size_of::<runtime::PmuTraceRecord>() as i32,
    )?;
    a.add(r11, rax)?;
    a.set_label(&mut retry)?;
    a.mov(r8d, dword_ptr(r10 + 8))?;
    a.test(r8b, 1)?;
    a.jne(retry)?;
    a.lfence()?;
    a.mov(ecx, dword_ptr(r10 + 12))?;
    a.test(ecx, ecx)?;
    a.je(unavailable)?;
    a.mov(rbp, qword_ptr(r10 + 16))?;
    a.movzx(r9d, word_ptr(r10 + 48))?;
    a.dec(ecx)?;
    a.lfence()?;
    a.rdpmc()?;
    a.shl(rdx, 32)?;
    a.or(rax, rdx)?;
    // Sign extend the hardware counter according to pmc_width.
    a.mov(ecx, 64)?;
    a.sub(ecx, r9d)?;
    a.shl(rax, cl)?;
    a.sar(rax, cl)?;
    a.add(rax, rbp)?;
    a.lfence()?;
    a.cmp(r8d, dword_ptr(r10 + 8))?;
    a.jne(retry)?;
    a.jmp(store)?;
    a.set_label(&mut unavailable)?;
    a.mov(rax, u64::MAX)?;
    a.set_label(&mut store)?;
    a.mov(qword_ptr(r11 + field as i32), rax)?;
    a.set_label(&mut done)?;
    for reg in [r11, r10, r9, r8, rbp, rdx, rcx, rax] {
        a.pop(reg)?;
    }
    Ok(())
}

/// Store one serialized timestamp in the calling participant's preallocated
/// record. The generated entry keeps all live registers intact. `RDTSCP` also
/// prevents the measured work from moving after its end marker.
fn emit_trace_stamp(
    a: &mut CodeAssembler,
    node: Option<usize>,
    field: usize,
) -> Result<(), iced_x86::IcedError> {
    let aux_field = match field {
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, work_start) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, work_start_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, work_end) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, work_end_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, progress_start) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, progress_start_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, progress_end) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, progress_end_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, barrier_start) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, barrier_start_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, sleep_start) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, sleep_start_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, sleep_end) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, sleep_end_aux)
        }
        f if f == std::mem::offset_of!(runtime::SyncTraceRecord, barrier_end) => {
            std::mem::offset_of!(runtime::SyncTraceRecord, barrier_end_aux)
        }
        _ => unreachable!("unknown synchronization trace field"),
    };
    let mut done = a.create_label();
    for reg in [rax, rcx, rdx, r10, r11] {
        a.push(reg)?;
    }
    a.mov(
        r10,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, trace) as i32),
    )?;
    a.test(r10, r10)?;
    a.je(done)?;
    a.mov(r11, r13)?;
    a.mov(
        rax,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, trace_nodes_per_block) as i32),
    )?;
    a.imul_2(r11, rax)?;
    if let Some(node) = node {
        a.add(r11, node as i32)?;
    } else {
        a.add(r11, r14)?;
    }
    a.mov(
        rax,
        qword_ptr(rbx + std::mem::offset_of!(RunContext<'_, '_>, participants) as i32),
    )?;
    a.imul_2(r11, rax)?;
    a.add(r11, r12)?;
    a.imul_3(
        r11,
        r11,
        std::mem::size_of::<runtime::SyncTraceRecord>() as i32,
    )?;
    a.add(r10, r11)?;
    a.rdtscp()?;
    a.shl(rdx, 32)?;
    a.or(rax, rdx)?;
    a.mov(qword_ptr(r10 + field as i32), rax)?;
    a.mov(dword_ptr(r10 + aux_field as i32), ecx)?;
    a.lfence()?;
    a.set_label(&mut done)?;
    for reg in [r11, r10, rdx, rcx, rax] {
        a.pop(reg)?;
    }
    Ok(())
}

fn emit_progress(
    a: &mut CodeAssembler,
    layer: usize,
    trace_sync: bool,
    trace_pmu: bool,
    trace_node: usize,
    calls: &mut Vec<(CodeLabel, usize)>,
) -> Result<(), iced_x86::IcedError> {
    let mut done = a.create_label();
    a.test(r12, r12)?;
    a.jne(done)?;
    if trace_sync || trace_pmu {
        emit_instrument_stamp(
            a,
            Some(trace_node),
            std::mem::offset_of!(runtime::SyncTraceRecord, progress_start),
            trace_sync,
            trace_pmu,
        )?;
    }
    a.mov(rdi, rbx)?;
    a.mov(rsi, r13)?;
    a.mov(rdx, layer as u64)?;
    runtime_call(a, std::mem::offset_of!(RunContext<'_, '_>, call_progress), calls)?;
    if trace_sync || trace_pmu {
        emit_instrument_stamp(
            a,
            Some(trace_node),
            std::mem::offset_of!(runtime::SyncTraceRecord, progress_end),
            trace_sync,
            trace_pmu,
        )?;
    }
    a.set_label(&mut done)?;
    a.nop()?;
    Ok(())
}
