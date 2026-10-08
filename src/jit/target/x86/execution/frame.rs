//! Generated block preparation shared by prefill and decode. The host fills one
//! frame before launch; every block's row count, token cursor, task grids and
//! position advancement are computed by the generated program.
use super::{ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
pub use crate::jit::execution::abi::{CallFrame, TaskGrid};

/// `(multiplier, shift)` giving `u / divisor` as `(u * multiplier) >> shift` for every
/// `u` the block can produce, or `None` when no pair of that form is exact.
///
/// **The bound is what makes this safe.** A block never carries more than
/// `max_batch` rows, so the dividend lives in `0..=max_batch + divisor - 1`: the pair is
/// checked against `u / divisor` on **every** value of that range at assembly time, and
/// the caller keeps the division when the check fails. No reciprocal is trusted on the
/// strength of its derivation.
fn reciprocal(divisor: usize, max_batch: usize) -> Option<(u32, u32)> {
    let last = max_batch.checked_add(divisor - 1)?;
    (1..=31u32).find_map(|shift| {
        let multiplier = u32::try_from((1usize << shift) / divisor + 1).ok()?;
        let exact = (0..=last).all(|u| {
            u.checked_mul(multiplier as usize)
                .is_some_and(|p| (p >> shift) == u / divisor)
        });
        exact.then_some((multiplier, shift))
    })
}

/// Tasks of one grid for the `r8` rows of this block, into `rax`:
/// `ceil(rows / rows_per_task) * tasks_per_tile`.
///
/// **Both factors are known when the program is assembled**, so the general form —
/// a 64-bit `div` by an immediate, plus the rounding on its remainder, a dozen
/// instructions repeated once per grid — is only emitted when nothing simpler holds.
/// One row per task (the majority of the grids: the count *is* the row count), a power
/// of two, or a checked reciprocal each collapse to one or two instructions, and a
/// single task per tile drops the multiplication.
fn count_of_rows(
    a: &mut CodeAssembler,
    grid: TaskGrid,
    max_batch: usize,
) -> Result<(), iced_x86::IcedError> {
    let rows = grid.rows_per_task;
    if rows == 1 {
        a.mov(rax, r8)?;
    } else if rows.is_power_of_two() {
        a.lea(rax, ptr(r8 + (rows as i32 - 1)))?;
        a.shr(rax, rows.trailing_zeros())?;
    } else if let Some((multiplier, shift)) = reciprocal(rows, max_batch) {
        a.lea(rax, ptr(r8 + (rows as i32 - 1)))?;
        a.imul_3(rax, rax, multiplier as i32)?;
        a.shr(rax, shift)?;
    } else {
        a.mov(rax, r8)?;
        a.xor(edx, edx)?;
        a.mov(rcx, rows as u64)?;
        a.div(rcx)?;
        a.test(rdx, rdx)?;
        a.setne(dl)?;
        a.movzx(edx, dl)?;
        a.add(rax, rdx)?;
    }
    match grid.tasks_per_tile {
        1 => {}
        tile if i32::try_from(tile).is_ok() => a.imul_3(rax, rax, tile as i32)?,
        tile => {
            a.mov(rcx, tile as u64)?;
            a.imul_2(rax, rcx)?;
        }
    }
    Ok(())
}

/// Emit one block's setup. ABI `(frame: *mut CallFrame)`, one task executed by
/// participant zero, followed by the normal plan barrier. The frame's token
/// range, starting position and counter allocation must be validated before
/// launch. The generated code also handles zero remaining tokens without reads.
pub fn begin_block(
    max_batch: usize,
    grids: &[TaskGrid],
) -> Result<GeneratedKernel, ExecutionError> {
    if max_batch == 0 || grids.len() > i32::MAX as usize / 8 {
        return Err(ExecutionError(
            "generated block geometry exceeds addressing domain".into(),
        ));
    }
    for grid in grids {
        grid.capacity(max_batch)?;
    }
    fn emit(
        a: &mut CodeAssembler,
        max_batch: usize,
        grids: &[TaskGrid],
    ) -> Result<(), iced_x86::IcedError> {
        a.mov(
            r8,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, total_tokens) as i32),
        )?;
        a.mov(
            r9,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, processed) as i32),
        )?;
        // Saturate the remaining tokens at zero, then bound them by the block width:
        // both clamps are conditional moves, in the same form.
        a.cmp(r8, r9)?;
        a.cmovb(r8, r9)?;
        a.sub(r8, r9)?;
        a.mov(rax, max_batch as u64)?;
        a.cmp(r8, rax)?;
        a.cmova(r8, rax)?;
        a.mov(
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, rows) as i32),
            r8,
        )?;
        a.mov(
            rax,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, tokens) as i32),
        )?;
        a.lea(rax, ptr(rax + r9 * 4))?;
        a.mov(
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, token_cursor) as i32),
            rax,
        )?;
        a.add(r9, r8)?;
        a.cmp(
            r9,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, total_tokens) as i32),
        )?;
        a.setae(r11b)?;
        a.movzx(r11d, r11b)?;
        a.mov(
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, last_block) as i32),
            r11,
        )?;
        a.mov(
            rsi,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, counts) as i32),
        )?;
        for (index, grid) in grids.iter().enumerate() {
            let mut ready = a.create_label();
            if grid.last_block_only {
                a.xor(eax, eax)?;
                a.test(r11, r11)?;
                a.je(ready)?;
            }
            count_of_rows(a, *grid, max_batch)?;
            a.set_label(&mut ready)?;
            a.mov(qword_ptr(rsi + (index * 8) as i32), rax)?;
        }
        a.ret()
    }
    let mut a = CodeAssembler::new(64).map_err(|e| ExecutionError(e.to_string()))?;
    emit(&mut a, max_batch, grids).map_err(|e| ExecutionError(e.to_string()))?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(|e| ExecutionError(e.to_string()))?) }
}

/// Emit advancement after all layers and the optional language head. ABI
/// `(frame: *mut CallFrame)`, one task. Existing plan synchronization keeps every
/// participant on the same block before the next `begin_block` runs.
pub fn end_block() -> Result<GeneratedKernel, ExecutionError> {
    let mut a = CodeAssembler::new(64).map_err(|e| ExecutionError(e.to_string()))?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        a.mov(
            rax,
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, rows) as i32),
        )?;
        a.add(
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, processed) as i32),
            rax,
        )?;
        a.add(
            qword_ptr(rdi + std::mem::offset_of!(CallFrame, position) as i32),
            rax,
        )?;
        a.ret()
    };
    emit(&mut a).map_err(|e| ExecutionError(e.to_string()))?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(|e| ExecutionError(e.to_string()))?) }
}
