//! Registers and vector types: the integer file ([`X`]), the floating-point
//! file ([`F`]), the vector file ([`V`]), and what a `vsetvli` configures
//! ([`Sew`], [`Lmul`], [`Vtype`]).

use std::fmt;

/// An integer register, `x0..x31`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct X(pub(crate) u8);
/// A floating-point register, `f0..f31`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct F(pub(crate) u8);
/// A vector register, `v0..v31`; a group of LMUL registers is named by its first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct V(pub(crate) u8);

/// Integer register `n`.
pub const fn x(n: u8) -> X {
    assert!(n < 32, "x0..x31");
    X(n)
}
/// Floating-point register `n`.
pub const fn f(n: u8) -> F {
    assert!(n < 32, "f0..f31");
    F(n)
}
/// Vector register `n`.
pub const fn v(n: u8) -> V {
    assert!(n < 32, "v0..v31");
    V(n)
}

impl X {
    /// The register number.
    pub const fn n(self) -> u8 {
        self.0
    }
}
impl F {
    /// The register number.
    pub const fn n(self) -> u8 {
        self.0
    }
}
impl V {
    /// The register number.
    pub const fn n(self) -> u8 {
        self.0
    }
    /// The register `k` places further (`v(8).plus(2)` is `v10`), for the
    /// members of a group or of a segment.
    pub const fn plus(self, k: u8) -> V {
        v(self.0 + k)
    }
}

const X_NAMES: [&str; 32] = [
    "zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", "s0", "s1", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "s2", "s3",
    "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6",
];
const F_NAMES: [&str; 32] = [
    "ft0", "ft1", "ft2", "ft3", "ft4", "ft5", "ft6", "ft7", "fs0", "fs1", "fa0", "fa1", "fa2", "fa3", "fa4", "fa5", "fa6", "fa7",
    "fs2", "fs3", "fs4", "fs5", "fs6", "fs7", "fs8", "fs9", "fs10", "fs11", "ft8", "ft9", "ft10", "ft11",
];

/// The ABI name of integer register `n`.
pub(crate) fn x_name(n: u32) -> &'static str {
    X_NAMES[n as usize & 31]
}
/// The ABI name of floating-point register `n`.
pub(crate) fn f_name(n: u32) -> &'static str {
    F_NAMES[n as usize & 31]
}

impl fmt::Display for X {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(x_name(u32::from(self.0)))
    }
}
impl fmt::Display for F {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(f_name(u32::from(self.0)))
    }
}
impl fmt::Display for V {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

macro_rules! regs {
    ($ty:ident, $ctor:ident: $($name:ident = $n:expr),* $(,)?) => {
        $( #[doc = concat!("`", stringify!($name), "`, register ", stringify!($n), ".")] pub const $name: $ty = $ctor($n); )*
    };
}
regs!(X, x: ZERO = 0, RA = 1, SP = 2, GP = 3, TP = 4, T0 = 5, T1 = 6, T2 = 7, S0 = 8, S1 = 9, A0 = 10, A1 = 11, A2 = 12,
    A3 = 13, A4 = 14, A5 = 15, A6 = 16, A7 = 17, S2 = 18, S3 = 19, S4 = 20, S5 = 21, S6 = 22, S7 = 23, S8 = 24, S9 = 25,
    S10 = 26, S11 = 27, T3 = 28, T4 = 29, T5 = 30, T6 = 31);
regs!(F, f: FT0 = 0, FT1 = 1, FT2 = 2, FT3 = 3, FT4 = 4, FT5 = 5, FT6 = 6, FT7 = 7, FS0 = 8, FS1 = 9, FA0 = 10, FA1 = 11,
    FA2 = 12, FA3 = 13, FA4 = 14, FA5 = 15, FA6 = 16, FA7 = 17, FS2 = 18, FS3 = 19, FS4 = 20, FS5 = 21, FS6 = 22, FS7 = 23,
    FS8 = 24, FS9 = 25, FS10 = 26, FS11 = 27, FT8 = 28, FT9 = 29, FT10 = 30, FT11 = 31);
regs!(V, v: V0 = 0, V1 = 1, V2 = 2, V3 = 3, V4 = 4, V5 = 5, V6 = 6, V7 = 7, V8 = 8, V9 = 9, V10 = 10, V11 = 11, V12 = 12,
    V13 = 13, V14 = 14, V15 = 15, V16 = 16, V17 = 17, V18 = 18, V19 = 19, V20 = 20, V21 = 21, V22 = 22, V23 = 23, V24 = 24,
    V25 = 25, V26 = 26, V27 = 27, V28 = 28, V29 = 29, V30 = 30, V31 = 31);

/// Selected element width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Sew {
    /// 8 bits.
    E8,
    /// 16 bits.
    E16,
    /// 32 bits.
    E32,
    /// 64 bits.
    E64,
}

impl Sew {
    /// The `vsew` field.
    pub const fn bits(self) -> u32 {
        self as u32
    }
    /// Width in bits.
    pub const fn width(self) -> u32 {
        8 << self as u32
    }
    /// Width in bytes.
    pub const fn bytes(self) -> usize {
        1 << self as usize
    }
    /// The `width` field of a vector load or store of this element width.
    pub(crate) const fn mem_width(self) -> u32 {
        match self {
            Sew::E8 => 0,
            Sew::E16 => 5,
            Sew::E32 => 6,
            Sew::E64 => 7,
        }
    }
    pub(crate) fn from_bits(b: u32) -> Sew {
        [Sew::E8, Sew::E16, Sew::E32, Sew::E64][b as usize & 3]
    }
    pub(crate) fn from_mem_width(w: u32) -> Option<Sew> {
        match w {
            0 => Some(Sew::E8),
            5 => Some(Sew::E16),
            6 => Some(Sew::E32),
            7 => Some(Sew::E64),
            _ => None,
        }
    }
}

impl fmt::Display for Sew {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.width())
    }
}

/// Register-group multiplier: a fraction of a register, or a group of 1 to 8.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Lmul {
    /// One eighth.
    Mf8,
    /// A quarter.
    Mf4,
    /// A half.
    Mf2,
    /// One register.
    M1,
    /// Two registers.
    M2,
    /// Four registers.
    M4,
    /// Eight registers.
    M8,
}

impl Lmul {
    /// The `vlmul` field.
    pub const fn bits(self) -> u32 {
        match self {
            Lmul::Mf8 => 5,
            Lmul::Mf4 => 6,
            Lmul::Mf2 => 7,
            Lmul::M1 => 0,
            Lmul::M2 => 1,
            Lmul::M4 => 2,
            Lmul::M8 => 3,
        }
    }
    /// Registers per group (1 for the fractions).
    pub const fn regs(self) -> u8 {
        match self {
            Lmul::M2 => 2,
            Lmul::M4 => 4,
            Lmul::M8 => 8,
            _ => 1,
        }
    }
    /// LMUL as a power of two: −3 for mf8 … 3 for m8.
    pub const fn log2(self) -> i32 {
        match self {
            Lmul::Mf8 => -3,
            Lmul::Mf4 => -2,
            Lmul::Mf2 => -1,
            Lmul::M1 => 0,
            Lmul::M2 => 1,
            Lmul::M4 => 2,
            Lmul::M8 => 3,
        }
    }
    /// The multiplier of `2^log2`, if one exists.
    pub const fn from_log2(log2: i32) -> Option<Lmul> {
        Some(match log2 {
            -3 => Lmul::Mf8,
            -2 => Lmul::Mf4,
            -1 => Lmul::Mf2,
            0 => Lmul::M1,
            1 => Lmul::M2,
            2 => Lmul::M4,
            3 => Lmul::M8,
            _ => return None,
        })
    }
    /// The group of `regs` whole registers (1, 2, 4 or 8).
    pub const fn of_regs(regs: u8) -> Option<Lmul> {
        Some(match regs {
            1 => Lmul::M1,
            2 => Lmul::M2,
            4 => Lmul::M4,
            8 => Lmul::M8,
            _ => return None,
        })
    }
    /// Twice this multiplier (the destination of a widening operation).
    pub const fn double(self) -> Option<Lmul> {
        Lmul::from_log2(self.log2() + 1)
    }
    pub(crate) fn from_bits(b: u32) -> Option<Lmul> {
        Some(match b & 7 {
            0 => Lmul::M1,
            1 => Lmul::M2,
            2 => Lmul::M4,
            3 => Lmul::M8,
            5 => Lmul::Mf8,
            6 => Lmul::Mf4,
            7 => Lmul::Mf2,
            _ => return None,
        })
    }
}

impl fmt::Display for Lmul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Lmul::Mf8 => "mf8",
            Lmul::Mf4 => "mf4",
            Lmul::Mf2 => "mf2",
            Lmul::M1 => "m1",
            Lmul::M2 => "m2",
            Lmul::M4 => "m4",
            Lmul::M8 => "m8",
        })
    }
}

/// What a `vsetvli` sets: element width, group, the tail and mask policies
/// (agnostic when `true`), and the alternate format of Zvfbfa (`e16alt`: the
/// floating-point instructions read and write bf16 instead of IEEE half).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Vtype {
    /// Element width.
    pub sew: Sew,
    /// Register group.
    pub lmul: Lmul,
    /// Tail agnostic (`ta`) rather than undisturbed (`tu`).
    pub ta: bool,
    /// Mask agnostic (`ma`) rather than undisturbed (`mu`).
    pub ma: bool,
    /// `altfmt`, bit 8 (Zvfbfa): at SEW 16 the floating-point format is bf16.
    pub alt: bool,
}

impl Vtype {
    /// `sew`, `lmul`, tail and mask agnostic: the usual configuration.
    pub const fn new(sew: Sew, lmul: Lmul) -> Vtype {
        Vtype { sew, lmul, ta: true, ma: true, alt: false }
    }
    /// The same with tail undisturbed (an accumulator kept across a short strip).
    pub const fn tu(self) -> Vtype {
        Vtype { ta: false, ..self }
    }
    /// The same with mask undisturbed.
    pub const fn mu(self) -> Vtype {
        Vtype { ma: false, ..self }
    }
    /// The same in the alternate format (`e8alt`, `e16alt`; Zvfbfa).
    pub const fn alt(self) -> Vtype {
        Vtype { alt: true, ..self }
    }
    /// The immediate of `vsetvli`.
    pub const fn bits(self) -> u32 {
        (self.alt as u32) << 8 | (self.ma as u32) << 7 | (self.ta as u32) << 6 | self.sew.bits() << 3 | self.lmul.bits()
    }
    /// The configuration an immediate names, if it is one (`vill` and
    /// reserved bits refused).
    pub fn from_bits(b: u32) -> Option<Vtype> {
        if b >> 9 != 0 || (b >> 3) & 7 > 3 {
            return None;
        }
        let sew = Sew::from_bits((b >> 3) & 7);
        let alt = (b >> 8) & 1 == 1;
        if alt && !matches!(sew, Sew::E8 | Sew::E16) {
            return None;
        }
        Some(Vtype { sew, lmul: Lmul::from_bits(b)?, ta: (b >> 6) & 1 == 1, ma: (b >> 7) & 1 == 1, alt })
    }
}

impl fmt::Display for Vtype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}, {}, {}, {}",
            self.sew,
            if self.alt { "alt" } else { "" },
            self.lmul,
            if self.ta { "ta" } else { "tu" },
            if self.ma { "ma" } else { "mu" }
        )
    }
}

/// A rounding mode of a scalar floating-point instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Rm {
    /// To nearest, ties to even.
    Rne,
    /// Toward zero.
    Rtz,
    /// Down.
    Rdn,
    /// Up.
    Rup,
    /// To nearest, ties to max magnitude.
    Rmm,
    /// The mode of `frm`.
    Dyn,
}

impl Rm {
    pub(crate) const fn bits(self) -> u32 {
        match self {
            Rm::Rne => 0,
            Rm::Rtz => 1,
            Rm::Rdn => 2,
            Rm::Rup => 3,
            Rm::Rmm => 4,
            Rm::Dyn => 7,
        }
    }
    pub(crate) fn from_bits(b: u32) -> Option<Rm> {
        Some(match b {
            0 => Rm::Rne,
            1 => Rm::Rtz,
            2 => Rm::Rdn,
            3 => Rm::Rup,
            4 => Rm::Rmm,
            7 => Rm::Dyn,
            _ => return None,
        })
    }
    pub(crate) fn name(self) -> &'static str {
        match self {
            Rm::Rne => "rne",
            Rm::Rtz => "rtz",
            Rm::Rdn => "rdn",
            Rm::Rup => "rup",
            Rm::Rmm => "rmm",
            Rm::Dyn => "dyn",
        }
    }
}
