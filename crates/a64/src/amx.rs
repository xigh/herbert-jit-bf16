//! Apple's AMX, the matrix coprocessor of the M1 to M5: not an Arm
//! instruction set, and no assembler knows it.
//!
//! Every instruction is one word `0x0020_1000 | op << 5 | operand`, in a
//! space Armv8 leaves unallocated: `op` (0 to 22) names the operation and
//! `operand` the general register whose 64-bit value carries its fields
//! (address, register index, offsets, mode) — except op 17, `set` and `clr`,
//! whose operand is an immediate 0 or 1, preceded by three `nop` as in the
//! public reverse engineering of the unit (corsix/amx), from which this
//! encoding comes. `set` enables the unit for the thread with its state
//! zeroed; `clr` releases it.
//!
//! Nothing here is checked against a system assembler. What is checked, on
//! the M3 Ultra and the M5, by probes emitted with this encoding: every op
//! executes; `ldx`/`ldy` put 64 bytes in the
//! register their bits 56–58 name, `ldz`/`stz` a row of Z named by bits
//! 56–61; `matfp` with data type 1 (bits 42–45) adds to Z the 32 × 32 outer
//! product of the bf16 of Y (offset in bytes at bits 0–8) and of X (bits
//! 10–18) in f32, each lane one fused multiply-add rounded to nearest even,
//! denormals kept; `set` hands zeroed X, Y and Z; a load by the core right
//! after an `stz` sees the stored bytes.

use crate::asm::Asm;
use crate::reg::R;

/// The operations of the unit, by their number in the instruction word.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum AmxOp {
    /// 64 bytes from memory into an X register.
    Ldx = 0,
    /// 64 bytes from memory into a Y register.
    Ldy = 1,
    /// An X register to memory.
    Stx = 2,
    /// A Y register to memory.
    Sty = 3,
    /// 64 bytes from memory into a row of Z.
    Ldz = 4,
    /// A row of Z to memory.
    Stz = 5,
    /// Interleaved load of Z.
    Ldzi = 6,
    /// Interleaved store of Z.
    Stzi = 7,
    /// Z (or Y) into X.
    Extrx = 8,
    /// Z (or X) into Y.
    Extry = 9,
    /// f64 multiply-add.
    Fma64 = 10,
    /// f64 multiply-subtract.
    Fms64 = 11,
    /// f32 multiply-add.
    Fma32 = 12,
    /// f32 multiply-subtract.
    Fms32 = 13,
    /// Integer multiply-accumulate, 16 bits.
    Mac16 = 14,
    /// f16 multiply-add.
    Fma16 = 15,
    /// f16 multiply-subtract.
    Fms16 = 16,
    /// Integer vector operations.
    Vecint = 18,
    /// Floating-point vector operations.
    Vecfp = 19,
    /// Integer matrix operations.
    Matint = 20,
    /// Floating-point matrix operations, bf16 among their data types.
    Matfp = 21,
    /// Table lookup.
    Genlut = 22,
}

const BASE: u32 = 0x0020_1000;
const NAMES: [&str; 23] = [
    "ldx", "ldy", "stx", "sty", "ldz", "stz", "ldzi", "stzi", "extrx", "extry", "fma64", "fms64", "fma32", "fms32",
    "mac16", "fma16", "fms16", "set/clr", "vecint", "vecfp", "matint", "matfp", "genlut",
];

impl Asm {
    /// The AMX operation `op`, its operand the value of `rn` (an `X` register
    /// other than `xzr` and `sp`).
    pub fn amx(&mut self, op: AmxOp, rn: R) {
        if !rn.x || rn.n > 30 {
            self.fail(format!("amx {}: the operand is an X register x0..x30", NAMES[op as usize]));
            return;
        }
        self.word(BASE | (op as u32) << 5 | u32::from(rn.n));
    }

    /// `set`: the unit enabled for this thread, its state zeroed.
    pub fn amx_set(&mut self) {
        for _ in 0..3 {
            self.nop();
        }
        self.word(BASE | 17 << 5);
    }

    /// `clr`: the unit released.
    pub fn amx_clr(&mut self) {
        for _ in 0..3 {
            self.nop();
        }
        self.word(BASE | 17 << 5 | 1);
    }
}

/// The text of an AMX word, `None` outside the space: a raw `.inst` the
/// system assembler takes back, the operation in the comment.
pub(crate) fn decode(w: u32) -> Option<String> {
    if w & 0xffff_fc00 != BASE {
        return None;
    }
    let (op, operand) = ((w >> 5) & 31, w & 31);
    match op {
        17 if operand <= 1 => Some(format!(".inst {w:#010x} // amx {}", if operand == 0 { "set" } else { "clr" })),
        17 => None,
        0..=22 if operand <= 30 => Some(format!(".inst {w:#010x} // amx {} x{operand}", NAMES[op as usize])),
        _ => None,
    }
}
