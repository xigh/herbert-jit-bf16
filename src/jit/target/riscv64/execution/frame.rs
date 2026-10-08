//! Generated block preparation shared by prefill and decode. The host fills one
//! frame before launch; every block's row count, token cursor, task grids and
//! position advancement are computed by the generated program.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::riscv64::util::exec_err;
use rv::*;
pub use crate::jit::execution::abi::{CallFrame, TaskGrid};
use std::mem::offset_of;

/// Tasks of one grid for the `t0` rows of this block, into `t5`:
/// `ceil(rows / rows_per_task) * tasks_per_tile`. Both factors are known when
/// the program is assembled: one row per task is a move, a power of two a
/// shift, anything else one `divu`.
fn count_of_rows(a: &mut Asm, grid: TaskGrid) {
    let rows = grid.rows_per_task as i64;
    if rows == 1 {
        a.mv(T5, T0);
    } else {
        a.addi(T5, T0, 0);
        a.li(T6, rows - 1);
        a.add(T5, T5, T6);
        if (rows as u64).is_power_of_two() {
            a.srli(T5, T5, rows.trailing_zeros());
        } else {
            a.li(T6, rows);
            a.divu(T5, T5, T6);
        }
    }
    let tiles = grid.tasks_per_tile as i64;
    if tiles != 1 {
        a.li(T6, tiles);
        a.mul(T5, T5, T6);
    }
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
    let f = |field: usize| field as i64;
    let mut a = Asm::new();
    a.ld(T0, A0, f(offset_of!(CallFrame, total_tokens)));
    a.ld(T1, A0, f(offset_of!(CallFrame, processed)));
    // Saturate the remaining tokens at zero, then bound them by the block width.
    a.maxu(T0, T0, T1);
    a.sub(T0, T0, T1);
    a.li(T2, max_batch as i64);
    a.minu(T0, T0, T2);
    a.sd(T0, A0, f(offset_of!(CallFrame, rows)));
    a.ld(T2, A0, f(offset_of!(CallFrame, tokens)));
    a.sh2add(T2, T1, T2);
    a.sd(T2, A0, f(offset_of!(CallFrame, token_cursor)));
    a.add(T1, T1, T0);
    a.ld(T3, A0, f(offset_of!(CallFrame, total_tokens)));
    a.sltu(T3, T1, T3);
    a.xori(T3, T3, 1);
    a.sd(T3, A0, f(offset_of!(CallFrame, last_block)));
    a.ld(T4, A0, f(offset_of!(CallFrame, counts)));
    // One count per grid, the pointer advanced by each (any number of grids).
    for grid in grids {
        let ready = a.label();
        if grid.last_block_only {
            a.li(T5, 0);
            a.beqz(T3, ready);
        }
        count_of_rows(&mut a, *grid);
        a.bind(ready);
        a.sd(T5, T4, 0);
        a.addi(T4, T4, 8);
    }
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit advancement after all layers and the optional language head. ABI
/// `(frame: *mut CallFrame)`, one task. Existing plan synchronization keeps every
/// participant on the same block before the next `begin_block` runs.
pub fn end_block() -> Result<GeneratedKernel, ExecutionError> {
    let f = |field: usize| field as i64;
    let mut a = Asm::new();
    a.ld(T0, A0, f(offset_of!(CallFrame, rows)));
    a.ld(T1, A0, f(offset_of!(CallFrame, processed)));
    a.add(T1, T1, T0);
    a.sd(T1, A0, f(offset_of!(CallFrame, processed)));
    a.ld(T1, A0, f(offset_of!(CallFrame, position)));
    a.add(T1, T1, T0);
    a.sd(T1, A0, f(offset_of!(CallFrame, position)));
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
