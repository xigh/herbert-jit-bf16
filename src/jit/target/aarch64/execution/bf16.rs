//! Generated traversal of the bf16 R16 projection kernels.
use super::{ExecutionError, GeneratedKernel};
use crate::jit::compile::CompiledGemm;
use crate::jit::emit::b_block_bytes;
use crate::jit::target::aarch64::util::{exec_err, madd_imm, mul_imm, Frame};
use a64::*;
pub use crate::jit::execution::abi::{Bf16Call, Bf16MultiCall};
use std::mem::offset_of;

/// Emit `(call: *const Bf16Call, task: usize)`. The tasks run the grid in
/// N-external / M-internal order; input offsets, variable column widths and
/// M/N tails follow the column plan of `CompiledGemm`.
/// All copied kernels retain their original instruction streams.
///
/// With a tail (`CompiledGemm::tail_rows`, the matrix path): residue rows
/// `r = rows mod 16` with `1 ≤ r ≤ tail` are not a padded tile but a vector
/// kernel of `r` rows run by the task of the last row tile after its full
/// tile, once per group of sixteen columns, on the weights that tile has
/// just streamed.
pub fn tile_selector(gemm: &CompiledGemm) -> Result<GeneratedKernel, ExecutionError> {
    let shape = gemm.shape();
    let mr = gemm.tile().mr;
    let ce = gemm.c_elem_bytes();
    let (format, n_tiles, blobs) = gemm.lowering_parts();
    let tail = gemm.tail_rows();
    let kinds_info = gemm.kinds();
    let a_row = shape.k * format.elem_bytes();
    let b_column = shape.k_pairs() * 4;
    debug_assert_eq!(b_column * 16, b_block_bytes(shape.k_pairs()));
    // Per kind: `mr` height kernels, then per mac set `tail` inner tail
    // kernels and `tail` masked ones when the kind's last group of sixteen is partial.
    let macs = gemm.tail_macs();
    let per_kind: Vec<usize> = kinds_info.iter().map(|&(_, n_tail)| mr + macs * tail * if n_tail < 16 { 2 } else { 1 }).collect();
    let kind_base: Vec<usize> = per_kind.iter().scan(0, |acc, &n| { let b = *acc; *acc += n; Some(b) }).collect();
    debug_assert_eq!(kind_base.last().map_or(0, |b| b + per_kind.last().unwrap()), blobs.len());
    // Runs keep the selector proportional to distinct widths/tails, even for a
    // large language head with thousands of equal-width column tiles.
    let mut runs = Vec::new();
    let mut start = 0;
    while start < n_tiles.len() {
        let kind = n_tiles[start].1;
        let mut end = start + 1;
        while end < n_tiles.len() && n_tiles[end].1 == kind {
            end += 1;
        }
        let width = if end - start > 1 { n_tiles[start + 1].0 - n_tiles[start].0 } else { 0 };
        runs.push((start, end, n_tiles[start].0, width, kind));
        start = end;
    }
    let mut a = Asm::new();
    let entries: Vec<Label> = blobs.iter().map(|_| a.label()).collect();
    let kinds: Vec<Label> = kinds_info.iter().map(|_| a.label()).collect();
    let exit = a.label();
    let rows_ready = a.label();
    // x19 call, x20 first row, x21 height, x22 column offset (in columns);
    // with a tail: x23 full row tiles, x24 the tail is used, x25 residue
    // rows, x26 this task is the last row tile.
    let saved: &[R] = if tail > 0 { &[X19, X20, X21, X22, X23, X24, X25, X26] } else { &[X19, X20, X21, X22] };
    let frame = Frame::enter(&mut a, true, saved, false, 0);
    a.mov(X19, X0);
    a.ldr(X10, mem(X19, offset_of!(Bf16Call, rows) as i64));
    a.ldr(X8, mem(X19, offset_of!(Bf16Call, row_count) as i64));
    a.cbz(X8, rows_ready);
    a.ldr(X10, at(X8));
    a.bind(rows_ready);
    a.cbz(X10, exit);
    a.mov_imm(X9, mr as u64);
    if tail > 0 {
        // full = rows / 16, r = rows mod 16. Tail used when 1 ≤ r ≤ tail: then
        // m_tiles = max(full, 1), the last row tile also doing the residue;
        // otherwise m_tiles = full + (r != 0), the residue a padded tile.
        let (from_full, padded, m_done) = (a.label(), a.label(), a.label());
        debug_assert_eq!(mr, 16);
        a.lsr(X23, X10, 4);
        a.and(X25, X10, 15u64);
        a.mov_imm(X24, 0);
        a.cbz(X25, from_full);
        a.mov_imm(X13, tail as u64);
        a.cmp(X25, X13);
        a.bhi(padded);
        a.mov_imm(X24, 1);
        a.cmp(X23, 0u64);
        a.cset(X13, Cond::Eq);
        a.add(X8, X23, X13);
        a.b(m_done);
        a.bind(padded);
        a.add(X8, X23, 1);
        a.b(m_done);
        a.bind(from_full);
        a.mov(X8, X23);
        a.bind(m_done);
    } else {
        // m_tiles = ceil(rows / mr).
        a.add(X8, X10, X9);
        a.sub(X8, X8, 1);
        a.udiv(X8, X8, X9);
    }
    // Column tile = task / m_tiles; row tile = task % m_tiles.
    a.udiv(X11, X1, X8);
    a.msub(X12, X11, X8, X1);
    if tail > 0 {
        a.add(X13, X12, 1);
        a.cmp(X13, X8);
        a.cset(X26, Cond::Eq);
    }
    mul_imm(&mut a, X20, X12, mr as u64, X13);
    a.sub(X21, X10, X20);
    a.cmp(X21, X9);
    a.csel(X21, X9, X21, Cond::Hi);
    for &(start, end, offset, width, kind) in &runs {
        let next = a.label();
        a.mov_imm(X13, end as u64);
        a.cmp(X11, X13);
        a.bhs(next);
        a.mov_imm(X13, start as u64);
        a.sub(X22, X11, X13);
        mul_imm(&mut a, X22, X22, width as u64, X14);
        a.mov_imm(X13, offset as u64);
        a.add(X22, X22, X13);
        a.b(kinds[kind]);
        a.bind(next);
    }
    a.b(exit);
    for (kind, &label) in kinds.iter().enumerate() {
        a.bind(label);
        let (sme, after, tail_label) = (a.label(), a.label(), a.label());
        if tail > 0 {
            // The tail runs after the full tile of the last row tile, or alone
            // when there is no full tile.
            a.cbz(X24, sme);
            a.cbz(X26, sme);
            a.cbz(X23, tail_label);
        }
        a.bind(sme);
        a.ldr(X0, mem(X19, offset_of!(Bf16Call, activations) as i64));
        madd_imm(&mut a, X0, X20, a_row as u64, X0, X13);
        a.ldr(X1, mem(X19, offset_of!(Bf16Call, weights) as i64));
        madd_imm(&mut a, X1, X22, b_column as u64, X1, X13);
        a.ldr(X2, mem(X19, offset_of!(Bf16Call, output) as i64));
        mul_imm(&mut a, X14, X20, shape.n as u64, X13);
        a.add(X14, X14, X22);
        madd_imm(&mut a, X2, X14, ce as u64, X2, X13);
        for height in 1..=mr {
            let next = a.label();
            a.cmp(X21, height as u64);
            a.bne(next);
            a.bl(entries[kind_base[kind] + height - 1]);
            a.b(if tail > 0 { after } else { exit });
            a.bind(next);
        }
        a.b(exit);
        if tail > 0 {
            a.bind(after);
            a.cbz(X24, exit);
            a.cbz(X26, exit);
            a.bind(tail_label);
            // Rows 16·full.. of packed A (block `full`), the same weights, C at row 16·full;
            // one call per group of sixteen columns, the masked kernels on a partial last group.
            let (width, n_tail) = kinds_info[kind];
            let groups = width / 16;
            let masked = n_tail < 16;
            for r in 1..=tail {
                let next = a.label();
                a.cmp(X25, r as u64);
                a.bne(next);
                for g in 0..groups {
                    a.ldr(X0, mem(X19, offset_of!(Bf16Call, activations) as i64));
                    madd_imm(&mut a, X0, X23, (16 * a_row) as u64, X0, X13);
                    a.ldr(X1, mem(X19, offset_of!(Bf16Call, weights) as i64));
                    a.add(X14, X22, (16 * g) as i64);
                    madd_imm(&mut a, X1, X14, b_column as u64, X1, X13);
                    a.ldr(X2, mem(X19, offset_of!(Bf16Call, output) as i64));
                    mul_imm(&mut a, X15, X23, (16 * shape.n) as u64, X13);
                    a.add(X14, X15, X14);
                    madd_imm(&mut a, X2, X14, ce as u64, X2, X13);
                    let per_mac = tail * if masked { 2 } else { 1 };
                    let set = if masked && g == groups - 1 { 1 } else { 0 };
                    a.bl(entries[kind_base[kind] + mr + gemm.tail_set_of(r) * per_mac + set * tail + r - 1]);
                }
                a.b(exit);
                a.bind(next);
            }
            a.b(exit);
        }
    }
    a.bind(exit);
    frame.leave(&mut a);
    a.ret();
    for (blob, &label) in blobs.iter().zip(&entries) {
        a.align(16);
        a.bind(label);
        a.bytes(blob);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const PackCall, block)`: the sixteen rows of block `block` of
/// `A` (row-major, `k` bf16 per row) packed by pairs of `K` — pair `p` of the
/// sixteen rows in 64 contiguous bytes (`jit/emit.rs`, `packed_a_bytes`). Rows
/// past the current count repeat the last valid row: the matrix engine
/// computes them, the GEMM never stores them. `k` is even and its pairs a
/// multiple of four (`gemm_sme::check`).
pub fn pack(k: usize) -> Result<GeneratedKernel, ExecutionError> {
    use crate::jit::execution::abi::PackCall;
    let pairs = k / 2;
    if k == 0 || !k.is_multiple_of(2) || !pairs.is_multiple_of(4) {
        return Err(ExecutionError(format!("no packing of rows of {k}: pairs by groups of four")));
    }
    let row = (k * 2) as u64;
    let mut a = Asm::new();
    let exit = a.label();
    let field = |f: usize| mem(X0, f as i64);
    a.ldr(X8, field(offset_of!(PackCall, rows)));
    a.ldr(X8, at(X8));
    a.lsl(X9, X1, 4);
    a.subs(X10, X8, X9);
    a.b_cond(Cond::Ls, exit);
    // x10 = valid rows − 1 in this block, at most 15.
    a.mov_imm(X11, 16);
    a.cmp(X10, X11);
    a.csel(X10, X11, X10, Cond::Hi);
    a.sub(X10, X10, 1);
    // x2 = first row of the block, x3 = its packed block.
    a.ldr(X2, field(offset_of!(PackCall, source)));
    madd_imm(&mut a, X2, X9, row, X2, X11);
    a.ldr(X3, field(offset_of!(PackCall, dest)));
    madd_imm(&mut a, X3, X9, row, X3, X11);
    for group in 0..4u64 {
        // Rows 4g..4g+3, each clamped to the last valid row.
        for (i, r) in [X4, X5, X6, X7].into_iter().enumerate() {
            a.mov_imm(X11, 4 * group + i as u64);
            a.cmp(X11, X10);
            a.csel(X11, X10, X11, Cond::Hi);
            mul_imm(&mut a, X11, X11, row, X12);
            a.add(r, X2, X11);
        }
        a.add(X12, X3, 16 * group);
        let top = a.label();
        a.mov_imm(X13, (pairs / 4) as u64);
        a.bind(top);
        for (reg, r) in [X4, X5, X6, X7].into_iter().enumerate() {
            a.ldr(q(reg as u8), post(r, 16));
        }
        a.trn1(v(4).s4(), v(0).s4(), v(1).s4());
        a.trn2(v(5).s4(), v(0).s4(), v(1).s4());
        a.trn1(v(6).s4(), v(2).s4(), v(3).s4());
        a.trn2(v(7).s4(), v(2).s4(), v(3).s4());
        a.trn1(v(16).d2(), v(4).d2(), v(6).d2());
        a.trn1(v(17).d2(), v(5).d2(), v(7).d2());
        a.trn2(v(18).d2(), v(4).d2(), v(6).d2());
        a.trn2(v(19).d2(), v(5).d2(), v(7).d2());
        for (p, r) in [16u8, 17, 18, 19].into_iter().enumerate() {
            a.str(q(r), mem(X12, 64 * p as i64));
        }
        a.add(X12, X12, 256);
        a.subs(X13, X13, 1);
        a.bne(top);
    }
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const PackCall, block)`: the thirty-two rows of block
/// `block` of `A` (row-major, `k` bf16 per row) packed by single `K` for the
/// AMX — `K` number `k` of the thirty-two rows in 64 contiguous bytes
/// (`jit/emit.rs`, `packed_a_amx_bytes`). Rows past the current count repeat
/// the last valid row, as [`pack`]. Eight rows at a time, eight `K`s per
/// turn: an 8 × 8 transposition of bf16 (`trn1`/`trn2` on halves, words,
/// then doublewords). `k` is a multiple of 8 (`gemm_amx::check`).
pub fn pack_amx(k: usize) -> Result<GeneratedKernel, ExecutionError> {
    use crate::jit::execution::abi::PackCall;
    if k == 0 || !k.is_multiple_of(8) {
        return Err(ExecutionError(format!("no AMX packing of rows of {k}: K by groups of eight")));
    }
    let row = (k * 2) as u64;
    let rows_per_block = crate::jit::emit::AMX_ROWS as u64;
    let mut a = Asm::new();
    let exit = a.label();
    let field = |f: usize| mem(X0, f as i64);
    a.ldr(X8, field(offset_of!(PackCall, rows)));
    a.ldr(X8, at(X8));
    a.lsl(X9, X1, 5);
    a.subs(X10, X8, X9);
    a.b_cond(Cond::Ls, exit);
    // x10 = valid rows − 1 in this block, at most 31.
    a.mov_imm(X11, rows_per_block);
    a.cmp(X10, X11);
    a.csel(X10, X11, X10, Cond::Hi);
    a.sub(X10, X10, 1);
    // x2 = first row of the block, x3 = its packed block (the size of 32 rows).
    a.ldr(X2, field(offset_of!(PackCall, source)));
    madd_imm(&mut a, X2, X9, row, X2, X11);
    a.ldr(X3, field(offset_of!(PackCall, dest)));
    madd_imm(&mut a, X3, X9, row, X3, X11);
    let ptrs = [X4, X5, X6, X7, X12, X13, X14, X15];
    for group in 0..4u64 {
        // Rows 8g..8g+7, each clamped to the last valid row.
        for (i, &r) in ptrs.iter().enumerate() {
            a.mov_imm(X11, 8 * group + i as u64);
            a.cmp(X11, X10);
            a.csel(X11, X10, X11, Cond::Hi);
            mul_imm(&mut a, X11, X11, row, X16);
            a.add(r, X2, X11);
        }
        // x16 = the 16 bytes of this group in K 0 of the packed block.
        a.add(X16, X3, 16 * group);
        let top = a.label();
        a.mov_imm(X17, (k / 8) as u64);
        a.bind(top);
        for (i, &r) in ptrs.iter().enumerate() {
            a.ldr(q(i as u8), post(r, 16));
        }
        // Halves: t0..t7 in v16..v23.
        for pair in 0..4u8 {
            a.trn1(v(16 + 2 * pair).h8(), v(2 * pair).h8(), v(2 * pair + 1).h8());
            a.trn2(v(17 + 2 * pair).h8(), v(2 * pair).h8(), v(2 * pair + 1).h8());
        }
        // Words: u0..u7 in v24..v31 (u0 = K 0|4 rows 0-3, u1 = K 1|5, u2 = K 2|6,
        // u3 = K 3|7; u4..u7 the same for rows 4-7).
        for half in 0..2u8 {
            let (t, u) = (16 + 4 * half, 24 + 4 * half);
            a.trn1(v(u).s4(), v(t).s4(), v(t + 2).s4());
            a.trn1(v(u + 1).s4(), v(t + 1).s4(), v(t + 3).s4());
            a.trn2(v(u + 2).s4(), v(t).s4(), v(t + 2).s4());
            a.trn2(v(u + 3).s4(), v(t + 1).s4(), v(t + 3).s4());
        }
        // Doublewords: K c (0..7) of rows 0-7 in v(c).
        for c in 0..4u8 {
            a.trn1(v(c).d2(), v(24 + c).d2(), v(28 + c).d2());
            a.trn2(v(c + 4).d2(), v(24 + c).d2(), v(28 + c).d2());
        }
        for c in 0..8u8 {
            a.str(q(c), mem(X16, 64 * i64::from(c)));
        }
        a.add(X16, X16, 512);
        a.subs(X17, X17, 1);
        a.bne(top);
    }
    a.bind(exit);
    a.ret();
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}

/// Emit `(call: *const Bf16MultiCall, task: usize)`: the tasks of `parts`
/// projections laid end to end, part by part in the order given, each part
/// served by its own tile selector reached by a tail call with the task
/// renumbered from its first column tile. The parts share the tile (the same
/// plan of the same regime) and read the rows of the first call.
pub fn multi_tile_selector(parts: &[&CompiledGemm]) -> Result<GeneratedKernel, ExecutionError> {
    if parts.is_empty() || parts.len() > 4 {
        return Err(ExecutionError("a fused projection node takes one to four parts".into()));
    }
    let mr = parts[0].tile().mr;
    if parts.iter().any(|p| p.tile().mr != mr) {
        return Err(ExecutionError("the parts of a fused projection node must share the tile height".into()));
    }
    let selectors: Vec<GeneratedKernel> = parts.iter().map(|p| tile_selector(p)).collect::<Result<_, _>>()?;
    let mut bounds = Vec::with_capacity(parts.len());
    let mut start = 0usize;
    for p in parts {
        bounds.push((start, start + p.n_tile_count()));
        start += p.n_tile_count();
    }
    let mut a = Asm::new();
    let entries: Vec<Label> = selectors.iter().map(|_| a.label()).collect();
    let exit = a.label();
    let rows_ready = a.label();
    let first = offset_of!(Bf16MultiCall, calls);
    a.ldr(X10, mem(X0, (first + offset_of!(Bf16Call, rows)) as i64));
    a.ldr(X8, mem(X0, (first + offset_of!(Bf16Call, row_count)) as i64));
    a.cbz(X8, rows_ready);
    a.ldr(X10, at(X8));
    a.bind(rows_ready);
    a.cbz(X10, exit);
    // m_tiles = ceil(rows / mr); column tile over all parts = task / m_tiles; row tile = task % m_tiles.
    a.mov_imm(X9, mr as u64);
    a.add(X8, X10, X9);
    a.sub(X8, X8, 1);
    a.udiv(X8, X8, X9);
    a.udiv(X11, X1, X8);
    a.msub(X12, X11, X8, X1);
    for (i, &(s, e)) in bounds.iter().enumerate() {
        let next = a.label();
        a.mov_imm(X13, e as u64);
        a.cmp(X11, X13);
        a.bhs(next);
        // The part's task: (column tile − its first) · m_tiles + row tile; its call.
        a.mov_imm(X13, s as u64);
        a.sub(X9, X11, X13);
        a.madd(X1, X9, X8, X12);
        a.add(X0, X0, (first + i * std::mem::size_of::<Bf16Call>()) as i64);
        a.b(entries[i]);
        a.bind(next);
    }
    a.bind(exit);
    a.ret();
    for (selector, &label) in selectors.iter().zip(&entries) {
        a.align(16);
        a.bind(label);
        a.bytes(&selector.bytes);
    }
    unsafe { GeneratedKernel::new(a.finish().map_err(exec_err)?.bytes) }
}
