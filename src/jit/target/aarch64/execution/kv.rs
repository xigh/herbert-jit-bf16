//! The KV append: the keys and values of the current block, token-major in
//! the projection buffers, copied head-major into the layer's cache.
//!
//! One task per KV head. The cache holds every position from 0: the block
//! starting at absolute position `past` lands at `past · head_dim` in each
//! head's slice. A plan barrier follows the append before attention reads it.
//!
//! Under `KvSpec::r16` (the block attention) each position also goes to the
//! R16 copies of the head: the key's pairs of dimensions, one 32-bit lane
//! every 64 bytes, in the block of its sixteen positions; the value's
//! dimensions, sixteen per block of `Vᵀ`, interleaved with the other
//! position of its pair — the 64 bytes of the pair are read, the other
//! position's half kept by `uzp`, the two halves put back together by `zip`.
use super::kernels::copy_constant;
use super::{ExecutionError, GeneratedKernel};
use crate::jit::target::aarch64::util::{exec_err, mul_imm, Frame};
use a64::*;
pub use crate::jit::execution::abi::{KvCall, KvSpec};
use std::mem::offset_of;

/// One task per KV head `(call: *const KvCall, head: usize)`. The mounted
/// reserve guarantees `past + rows <= capacity` for every valid call.
pub fn append(spec: KvSpec) -> Result<GeneratedKernel, ExecutionError> {
    spec.validate()?;
    let row_bytes = spec.head_dim * 2;
    let mut a = Asm::new();
    let exit = a.label();
    let rows = a.label();
    let field = |f: usize| mem(X0, f as i64);
    // x19 = keys R16 of the head, x20 = values R16 of the head, x21 = the position.
    let frame = Frame::enter(&mut a, false, if spec.r16 { &[X19, X20, X21] } else { &[] }, false, 0);
    a.mov_imm(X8, spec.heads as u64);
    a.cmp(X1, X8);
    a.bhs(exit);
    a.ldr(X9, field(offset_of!(KvCall, rows)));
    a.ldr(X9, at(X9));
    a.cbz(X9, exit);
    a.ldr(X10, field(offset_of!(KvCall, past)));
    a.ldr(X10, at(X10));
    // x2 = source of row 0, head x1: token-major, `heads · head_dim` per row.
    mul_imm(&mut a, X2, X1, row_bytes as u64, X8);
    // x3 = target of position `past`: head slice, then position.
    mul_imm(&mut a, X3, X1, (spec.stride * 2) as u64, X8);
    mul_imm(&mut a, X8, X10, row_bytes as u64, X11);
    a.add(X3, X3, X8);
    a.ldr(X4, field(offset_of!(KvCall, input_keys)));
    a.add(X4, X4, X2);
    a.ldr(X5, field(offset_of!(KvCall, input_values)));
    a.add(X5, X5, X2);
    a.ldr(X6, field(offset_of!(KvCall, keys)));
    a.add(X6, X6, X3);
    a.ldr(X7, field(offset_of!(KvCall, values)));
    a.add(X7, X7, X3);
    a.mov_imm(X12, (spec.heads * row_bytes) as u64);
    if spec.r16 {
        // The head's R16 copies are `stride` elements apart, like its cache.
        mul_imm(&mut a, X8, X1, (spec.stride * 2) as u64, X11);
        a.ldr(X19, field(offset_of!(KvCall, keys_r16)));
        a.add(X19, X19, X8);
        a.ldr(X20, field(offset_of!(KvCall, values_r16)));
        a.add(X20, X20, X8);
        a.mov(X21, X10);
    }
    a.bind(rows);
    copy_constant(&mut a, X6, 0, X4, 0, row_bytes);
    copy_constant(&mut a, X7, 0, X5, 0, row_bytes);
    if spec.r16 {
        r16_row(&mut a, spec);
        a.add(X21, X21, 1);
    }
    a.add(X4, X4, X12);
    a.add(X5, X5, X12);
    a.add_imm_wide(X6, row_bytes);
    a.add_imm_wide(X7, row_bytes);
    a.subs(X9, X9, 1);
    a.bne(rows);
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// The R16 copies of one position `x21`, its key row at `x4` and its value
/// row at `x5` (neither advanced). Clobbers `x2`, `x3`, `x8`, `x11`, `x13`,
/// `x14`, `v0..v7`, `v16`, `v17`.
fn r16_row(a: &mut Asm, spec: KvSpec) {
    let hd = spec.head_dim;
    // Keys: block `pos / 16` (`16·hd` elements), lane `pos % 16` of every
    // pair of dimensions, the pairs 64 bytes apart.
    a.lsr(X2, X21, 4);
    mul_imm(a, X2, X2, (hd * 32) as u64, X3);
    a.and(X3, X21, 15u64);
    a.add(X2, X2, X3.lsl(2));
    a.add(X2, X2, X19);
    a.mov_imm(X3, 64);
    for c in 0..hd / 8 {
        a.ldr(q(0), mem(X4, 16 * c as i64));
        for lane in 0..4u8 {
            a.st1_lane(v(0).s(lane), post_reg(X2, X3));
        }
    }
    // Values: the pair `pos / 2` of every block of sixteen dimensions, the
    // blocks `32·capacity` bytes apart; the position is the even or the odd
    // half of the pair.
    let (odd, done) = (a.label(), a.label());
    a.lsr(X8, X21, 1);
    a.add(X8, X20, X8.lsl(6));
    a.mov_imm(X11, (32 * spec.capacity) as u64);
    a.mov(X14, X5);
    a.mov_imm(X13, (hd / 16) as u64);
    a.tbnz(X21, 0, odd);
    for is_odd in [false, true] {
        if is_odd {
            a.bind(odd);
        }
        let top = a.label();
        a.bind(top);
        a.ldp(q(1), q(2), mem(X8, 0));
        a.ldp(q(3), q(4), mem(X8, 32));
        a.ldp(q(5), q(6), post(X14, 32));
        // The other position's half of the pair.
        if is_odd {
            a.uzp1(v(16).h8(), v(1).h8(), v(2).h8());
            a.uzp1(v(17).h8(), v(3).h8(), v(4).h8());
        } else {
            a.uzp2(v(16).h8(), v(1).h8(), v(2).h8());
            a.uzp2(v(17).h8(), v(3).h8(), v(4).h8());
        }
        let (lo, hi) = if is_odd { ((16u8, 5u8), (17u8, 6u8)) } else { ((5u8, 16u8), (6u8, 17u8)) };
        a.zip1(v(1).h8(), v(lo.0).h8(), v(lo.1).h8());
        a.zip2(v(2).h8(), v(lo.0).h8(), v(lo.1).h8());
        a.zip1(v(3).h8(), v(hi.0).h8(), v(hi.1).h8());
        a.zip2(v(4).h8(), v(hi.0).h8(), v(hi.1).h8());
        a.stp(q(1), q(2), mem(X8, 0));
        a.stp(q(3), q(4), mem(X8, 32));
        a.add(X8, X8, X11);
        a.subs(X13, X13, 1);
        a.bne(top);
        if !is_odd {
            a.b(done);
        }
    }
    a.bind(done);
}

trait AddWide {
    fn add_imm_wide(&mut self, r: R, n: usize);
}
impl AddWide for Asm {
    fn add_imm_wide(&mut self, r: R, n: usize) {
        crate::jit::target::aarch64::util::add_imm(self, r, r, n as i64, X13);
    }
}
