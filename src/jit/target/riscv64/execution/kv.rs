//! The KV append: the keys and values of the current block, token-major in
//! the projection buffers, copied head-major into the layer's cache.
//!
//! One task per KV head. The cache holds every position from 0: the block
//! starting at absolute position `past` lands at `past · head_dim` in each
//! head's slice. A plan barrier follows the append before attention reads it.
//! The R16 copies of `KvSpec::r16` serve the block attention of AArch64
//! only: refused here.
use super::kernels::copy_constant;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::riscv64::util::{add_imm, exec_err};
use rv::*;
pub use crate::jit::execution::abi::{KvCall, KvSpec};
use std::mem::offset_of;

/// One task per KV head `(call: *const KvCall, head: usize)`. The mounted
/// reserve guarantees `past + rows <= capacity` for every valid call.
pub fn append(spec: KvSpec) -> Result<GeneratedKernel, ExecutionError> {
    spec.validate()?;
    if spec.r16 {
        return Err(ExecutionError("riscv64: the R16 copies of the KV cache serve a block attention this target does not have".into()));
    }
    let row_bytes = spec.head_dim * 2;
    let mut a = Asm::new();
    let exit = a.label();
    let f = |field: usize| field as i64;
    a.li(T0, spec.heads as i64);
    a.bgeu(A1, T0, exit);
    // a2 = rows left, a3 = past.
    a.ld(A2, A0, f(offset_of!(KvCall, rows)));
    a.ld(A2, A2, 0);
    a.beqz(A2, exit);
    a.ld(A3, A0, f(offset_of!(KvCall, past)));
    a.ld(A3, A3, 0);
    // a4 = source offset of row 0, head a1: token-major, `heads · head_dim` per row.
    a.li(T0, row_bytes as i64);
    a.mul(A4, A1, T0);
    // a5 = target offset of position `past`: head slice, then position.
    a.li(T0, (spec.stride * 2) as i64);
    a.mul(A5, A1, T0);
    a.li(T0, row_bytes as i64);
    a.mul(T1, A3, T0);
    a.add(A5, A5, T1);
    // a6/a7 = key/value source, t4/t5 = key/value target.
    a.ld(A6, A0, f(offset_of!(KvCall, input_keys)));
    a.add(A6, A6, A4);
    a.ld(A7, A0, f(offset_of!(KvCall, input_values)));
    a.add(A7, A7, A4);
    a.ld(T4, A0, f(offset_of!(KvCall, keys)));
    a.add(T4, T4, A5);
    a.ld(T5, A0, f(offset_of!(KvCall, values)));
    a.add(T5, T5, A5);
    a.li(A3, (spec.heads * row_bytes) as i64);
    let rows = a.here();
    copy_constant(&mut a, T4, 0, A6, 0, row_bytes);
    copy_constant(&mut a, T5, 0, A7, 0, row_bytes);
    a.add(A6, A6, A3);
    a.add(A7, A7, A3);
    add_imm(&mut a, T4, T4, row_bytes as i64, T6);
    add_imm(&mut a, T5, T5, row_bytes as i64, T6);
    a.addi(A2, A2, -1);
    a.bnez(A2, rows);
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
