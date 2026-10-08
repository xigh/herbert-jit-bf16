//! # a64 — an AArch64 run-time assembler and decoder
//!
//! What `iced-x86` is to the x86-64 target of herbert, this crate is to its
//! ARM target: kernels are *emitted* word by word at run time, never written
//! as intrinsics, inline assembly or C. It covers ARMv8/ARMv9 as far as the
//! engine needs it, and nothing is guessed: every encoding is checked against
//! the system assembler.
//!
//! - [`Asm`]: the buffer, [`Label`]s and their fixups, one method per
//!   instruction, named like the assembler syntax (`add`, `ldr`, `fmla`,
//!   `bfdot`…); operands are typed ([`R`] for the general-purpose file, [`Vr`]
//!   for the SIMD&FP file, [`Mem`] for addressing) and an operand an encoding
//!   cannot carry is an error at [`Asm::finish`], never a wrong word;
//! - [`decode()`]: every word the assembler can produce, back to its control
//!   flow ([`Flow`]) and to text the system assembler accepts — the audit of
//!   generated code and the listings are built on it.
//!
//! The instruction groups: the base set (`base.rs`: arithmetic, logic,
//! moves, bitfields, branches, loads and stores of every register file, LSE
//! atomics, barriers, system registers), Advanced SIMD and floating point
//! with the BF16 extension (`simd.rs`), and the SME/SVE subset of the
//! matrix engine (`sme.rs`). Apart from Arm: the AMX coprocessor of Apple's
//! M1 to M5 (`amx.rs`), raw words no assembler checks, whose encoding comes
//! from probes run on the machines.
//!
//! ```
//! use a64::*;
//! let mut a = Asm::new();
//! let done = a.label();
//! a.cbz(X2, done);
//! a.ldr(q(0), post(X0, 16));
//! a.bfdot(v(1).s4(), v(0).h8(), v(2).h2(3));
//! a.bind(done);
//! a.ret();
//! let code = a.finish().unwrap();
//! assert_eq!(code.bytes.len(), 16);
//! assert_eq!(decode(0xd65f_03c0).unwrap().flow, Flow::Return);
//! ```

#![deny(missing_docs)]

mod amx;
mod asm;
mod base;
mod decode;
mod decode_simd;
mod imm;
mod operand;
mod reg;
mod simd;
mod sme;

pub use amx::AmxOp;
pub use asm::{Asm, Code, Error, Label};
pub use base::{Order, Rt};
pub use decode::{decode, decode_all, Flow, Insn};
pub use imm::{decode_fp8, decode_logical, encode_fp8, encode_logical};
pub use sme::{Et, Slice, ZaS};
pub use operand::{at, idx, idx_scaled, mem, post, post_reg, pre, Barrier, Cond, Extend, Mem, Op2, Prefetch, Shift, SysReg};
pub use reg::*;

/// The `nop` word.
pub const NOP: u32 = 0xd503_201f;
