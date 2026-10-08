//! Generated block preparation shared by prefill and decode. The host fills one
//! frame before launch; every block's row count, token cursor, task grids and
//! position advancement are computed by the generated program.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::aarch64::util::{exec_err, mul_imm};
use a64::*;
pub use crate::jit::execution::abi::{CallFrame, TaskGrid};
use std::mem::offset_of;

/// Tasks of one grid for the `x8` rows of this block, into `x13`:
/// `ceil(rows / rows_per_task) * tasks_per_tile`. Both factors are known when
/// the program is assembled: one row per task is a move, a power of two a
/// shift, anything else one `udiv`.
fn count_of_rows(a: &mut Asm, grid: TaskGrid) {
    let rows = grid.rows_per_task as u64;
    if rows == 1 {
        a.mov(X13, X8);
    } else {
        a.mov_imm(X14, rows - 1);
        a.add(X13, X8, X14);
        if rows.is_power_of_two() {
            a.lsr(X13, X13, rows.trailing_zeros());
        } else {
            a.mov_imm(X14, rows);
            a.udiv(X13, X13, X14);
        }
    }
    mul_imm(a, X13, X13, grid.tasks_per_tile as u64, X14);
}

/// Emit one block's setup. ABI `(frame: *mut CallFrame)`, one task executed by
/// participant zero, followed by the normal plan barrier. The frame's token
/// range, starting position and counter allocation must be validated before
/// launch. The generated code also handles zero remaining tokens without reads.
pub fn begin_block(max_batch: usize, grids: &[TaskGrid]) -> Result<GeneratedKernel, ExecutionError> {
    if max_batch == 0 || grids.len() > (1 << 12) {
        return Err(ExecutionError("generated block geometry exceeds addressing domain".into()));
    }
    for grid in grids {
        grid.capacity(max_batch)?;
    }
    let f = |field: usize| mem(X0, field as i64);
    let mut a = Asm::new();
    a.ldr(X8, f(offset_of!(CallFrame, total_tokens)));
    a.ldr(X9, f(offset_of!(CallFrame, processed)));
    // Saturate the remaining tokens at zero, then bound them by the block width.
    a.cmp(X8, X9);
    a.csel(X8, X9, X8, Cond::Lo);
    a.sub(X8, X8, X9);
    a.mov_imm(X10, max_batch as u64);
    a.cmp(X8, X10);
    a.csel(X8, X10, X8, Cond::Hi);
    a.str(X8, f(offset_of!(CallFrame, rows)));
    a.ldr(X10, f(offset_of!(CallFrame, tokens)));
    a.add(X10, X10, X9.lsl(2));
    a.str(X10, f(offset_of!(CallFrame, token_cursor)));
    a.add(X9, X9, X8);
    a.ldr(X11, f(offset_of!(CallFrame, total_tokens)));
    a.cmp(X9, X11);
    a.cset(X11, Cond::Hs);
    a.str(X11, f(offset_of!(CallFrame, last_block)));
    a.ldr(X12, f(offset_of!(CallFrame, counts)));
    for (index, grid) in grids.iter().enumerate() {
        let ready = a.label();
        if grid.last_block_only {
            a.mov_imm(X13, 0);
            a.cbz(X11, ready);
        }
        count_of_rows(&mut a, *grid);
        a.bind(ready);
        a.str(X13, mem(X12, (index * 8) as i64));
    }
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit advancement after all layers and the optional language head. ABI
/// `(frame: *mut CallFrame)`, one task. Existing plan synchronization keeps every
/// participant on the same block before the next `begin_block` runs.
pub fn end_block() -> Result<GeneratedKernel, ExecutionError> {
    let f = |field: usize| mem(X0, field as i64);
    let mut a = Asm::new();
    a.ldr(X8, f(offset_of!(CallFrame, rows)));
    a.ldr(X9, f(offset_of!(CallFrame, processed)));
    a.add(X9, X9, X8);
    a.str(X9, f(offset_of!(CallFrame, processed)));
    a.ldr(X9, f(offset_of!(CallFrame, position)));
    a.add(X9, X9, X8);
    a.str(X9, f(offset_of!(CallFrame, position)));
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
