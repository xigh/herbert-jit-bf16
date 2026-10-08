//! Emission helpers shared by every RISC-V kernel, and the conversion of an
//! `rv` result into the engine's errors.
//!
//! The calling convention of generated code is LP64D: integer arguments in
//! `a0..a7`, floating-point ones in `fa0..fa7`; `s0..s11` and `fs0..fs11`
//! belong to the caller; `gp` and `tp` are never written; `t0..t6` are free
//! in any kernel.

use crate::core::error::HerbertError;
use crate::jit::execution::ExecutionError;

/// An `rv` error as an execution error.
pub(crate) fn exec_err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}

/// An `rv` error as a backend error, prefixed by what was being emitted.
pub(crate) fn backend_err(what: impl std::fmt::Display) -> impl FnOnce(rv::Error) -> HerbertError {
    move |e| HerbertError::Backend(format!("backend_jit riscv64: {what}: {e}"))
}

/// What is not emitted yet on this target, as a backend error: a refusal
/// that names itself, never a fallback.
pub(crate) fn not_yet(what: &str) -> HerbertError {
    HerbertError::Backend(format!("backend_jit riscv64: {what} is not emitted on this target yet"))
}

/// The same, as an execution error.
pub(crate) fn not_yet_exec(what: &str) -> ExecutionError {
    ExecutionError(format!("riscv64: the {what} node is not emitted on this target yet"))
}

/// `rd = rs + imm`: one `addi` when the immediate fits twelve bits, else `li`
/// into `tmp` and an `add` (`tmp` must differ from `rs`).
pub(crate) fn add_imm(a: &mut rv::Asm, rd: rv::X, rs: rv::X, imm: i64, tmp: rv::X) {
    if (-2048..2048).contains(&imm) {
        a.addi(rd, rs, imm);
    } else {
        a.li(tmp, imm);
        a.add(rd, rs, tmp);
    }
}

/// The frame of a function: what its prologue saved, where.
///
/// One adjustment of `sp` by a multiple of 16; `locals` bytes at `[sp]`, then
/// `ra` (when the function calls anything), the integer registers and the
/// floating-point ones, eight bytes each.
pub(crate) struct Frame {
    saved: Vec<rv::X>,
    fsaved: Vec<rv::F>,
    locals: usize,
    size: usize,
}

impl Frame {
    /// Emit the prologue.
    pub(crate) fn enter(a: &mut rv::Asm, link: bool, gprs: &[rv::X], fprs: &[rv::F], locals: usize) -> Frame {
        let locals = locals.next_multiple_of(8);
        let mut saved = Vec::new();
        if link {
            saved.push(rv::RA);
        }
        saved.extend_from_slice(gprs);
        let size = (locals + 8 * (saved.len() + fprs.len())).next_multiple_of(16);
        if size > 0 {
            add_imm(a, rv::SP, rv::SP, -(size as i64), rv::T6);
        }
        let frame = Frame { saved, fsaved: fprs.to_vec(), locals, size };
        for (i, &r) in frame.saved.iter().enumerate() {
            a.sd(r, rv::SP, (locals + 8 * i) as i64);
        }
        for (i, &f) in frame.fsaved.iter().enumerate() {
            a.fsd(f, rv::SP, (locals + 8 * (frame.saved.len() + i)) as i64);
        }
        frame
    }

    /// Emit the epilogue, without the `ret`.
    pub(crate) fn leave(&self, a: &mut rv::Asm) {
        for (i, &r) in self.saved.iter().enumerate() {
            a.ld(r, rv::SP, (self.locals + 8 * i) as i64);
        }
        for (i, &f) in self.fsaved.iter().enumerate() {
            a.fld(f, rv::SP, (self.locals + 8 * (self.saved.len() + i)) as i64);
        }
        if self.size > 0 {
            add_imm(a, rv::SP, rv::SP, self.size as i64, rv::T6);
        }
    }
}
