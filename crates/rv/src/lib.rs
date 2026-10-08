//! # rv — a RISC-V run-time assembler and decoder
//!
//! What `a64` is to the ARM target of herbert, this crate is to its RISC-V
//! target: kernels are *emitted* word by word at run time, never written as
//! intrinsics, inline assembly or C. It covers RV64GC and RVV 1.0 as far as
//! the engine needs them, and nothing is guessed: every encoding is checked
//! against `llvm-mc`, both ways.
//!
//! - [`Asm`]: the buffer, [`Label`]s and their fixups, one method per
//!   instruction, named like the assembler syntax (`add`, `ld`, `vle`,
//!   `vfwmaccbf16_vf`…; dots become underscores); operands are typed ([`X`]
//!   for the integer file, [`F`] for the floating-point file, [`V`] for the
//!   vector file) and an operand an encoding cannot carry is an error at
//!   [`Asm::finish`], never a wrong word;
//! - [`decode()`]: every word the assembler can produce, back to its control
//!   flow ([`Flow`]) and to text `llvm-mc` accepts — the audit of generated
//!   code and the listings are built on it.
//!
//! The instruction groups: the scalar set (`base.rs`: RV64IM, A, Zba, Zbb,
//! Zbs, fences, Zihintpause, CSRs, F and D with the half and bf16 moves of
//! Zfbfmin), and the vector set (`vector.rs`: RVV 1.0 with Zvfbfmin and
//! Zvfbfwma). The compressed extension is never emitted.
//!
//! ```
//! use rv::*;
//! let mut a = Asm::new();
//! let lp = a.here();
//! a.vsetvli(T0, A2, Vtype::new(Sew::E16, Lmul::M1));
//! a.vle(Sew::E16, V4, A1);
//! a.vfwmaccbf16_vf(V8, FA0, V4);
//! a.sub(A2, A2, T0);
//! a.bnez(A2, lp);
//! a.ret();
//! let code = a.finish().unwrap();
//! assert_eq!(code.bytes.len(), 24);
//! assert_eq!(decode(0x0000_8067).unwrap().flow, Flow::Return);
//! ```

#![deny(missing_docs)]

mod asm;
mod base;
mod decode;
mod imm;
mod reg;
mod vector;

pub use asm::{Asm, Code, Error, Label};
pub use base::{Cond, Csr, Fence, Order};
pub use decode::{decode, decode_all, Flow, Insn};
pub use imm::{li_sequence, LiStep};
pub use reg::*;

/// The `nop` word, `addi zero, zero, 0`.
pub const NOP: u32 = 0x0000_0013;
