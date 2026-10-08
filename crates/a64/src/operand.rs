//! Operands other than registers: condition codes, shifts and extensions of a
//! second operand, addressing modes, barrier domains, system registers and
//! prefetch operations.

use crate::reg::R;

/// A condition code, as `b.cond`, `csel` and `ccmp` read it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Cond {
    /// Equal (Z).
    Eq = 0,
    /// Not equal.
    Ne = 1,
    /// Carry set / unsigned higher or same.
    Hs = 2,
    /// Carry clear / unsigned lower.
    Lo = 3,
    /// Negative.
    Mi = 4,
    /// Positive or zero.
    Pl = 5,
    /// Overflow.
    Vs = 6,
    /// No overflow.
    Vc = 7,
    /// Unsigned higher.
    Hi = 8,
    /// Unsigned lower or same.
    Ls = 9,
    /// Signed greater or equal.
    Ge = 10,
    /// Signed less than.
    Lt = 11,
    /// Signed greater than.
    Gt = 12,
    /// Signed less or equal.
    Le = 13,
    /// Always.
    Al = 14,
    /// Always (the second encoding).
    Nv = 15,
}

impl Cond {
    /// The condition that holds exactly when `self` does not.
    pub const fn invert(self) -> Cond {
        Cond::from_bits(self as u32 ^ 1)
    }
    /// The condition of a 4-bit field.
    pub const fn from_bits(b: u32) -> Cond {
        match b & 15 {
            0 => Cond::Eq,
            1 => Cond::Ne,
            2 => Cond::Hs,
            3 => Cond::Lo,
            4 => Cond::Mi,
            5 => Cond::Pl,
            6 => Cond::Vs,
            7 => Cond::Vc,
            8 => Cond::Hi,
            9 => Cond::Ls,
            10 => Cond::Ge,
            11 => Cond::Lt,
            12 => Cond::Gt,
            13 => Cond::Le,
            14 => Cond::Al,
            _ => Cond::Nv,
        }
    }
    /// The assembler name.
    pub const fn name(self) -> &'static str {
        match self {
            Cond::Eq => "eq",
            Cond::Ne => "ne",
            Cond::Hs => "hs",
            Cond::Lo => "lo",
            Cond::Mi => "mi",
            Cond::Pl => "pl",
            Cond::Vs => "vs",
            Cond::Vc => "vc",
            Cond::Hi => "hi",
            Cond::Ls => "ls",
            Cond::Ge => "ge",
            Cond::Lt => "lt",
            Cond::Gt => "gt",
            Cond::Le => "le",
            Cond::Al => "al",
            Cond::Nv => "nv",
        }
    }
}

/// The shift of a shifted-register operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Shift {
    /// Logical left.
    Lsl = 0,
    /// Logical right.
    Lsr = 1,
    /// Arithmetic right.
    Asr = 2,
    /// Rotate right (logical instructions only).
    Ror = 3,
}

/// The extension of an extended-register operand, or of a register offset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Extend {
    /// Zero-extend a byte.
    Uxtb = 0,
    /// Zero-extend a half.
    Uxth = 1,
    /// Zero-extend a word.
    Uxtw = 2,
    /// The whole register (`lsl` in a register offset).
    Uxtx = 3,
    /// Sign-extend a byte.
    Sxtb = 4,
    /// Sign-extend a half.
    Sxth = 5,
    /// Sign-extend a word.
    Sxtw = 6,
    /// The whole register, signed.
    Sxtx = 7,
}

/// The second operand of an arithmetic or logical instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Op2 {
    /// An immediate. Negative values turn `add` into `sub` and back.
    Imm(i64),
    /// A register, unshifted.
    Reg(R),
    /// A shifted register.
    Shifted(R, Shift, u32),
    /// An extended register, shifted left by 0..=4.
    Extended(R, Extend, u32),
}

impl From<R> for Op2 {
    fn from(r: R) -> Self {
        Op2::Reg(r)
    }
}
macro_rules! op2_from_int {
    ($($t:ty),*) => {$(
        impl From<$t> for Op2 {
            fn from(v: $t) -> Self {
                Op2::Imm(v as i64)
            }
        }
    )*};
}
op2_from_int!(i32, i64, u32, u64, usize, isize, u8, u16);

impl R {
    /// `reg, lsl #n`
    pub const fn lsl(self, n: u32) -> Op2 {
        Op2::Shifted(self, Shift::Lsl, n)
    }
    /// `reg, lsr #n`
    pub const fn lsr(self, n: u32) -> Op2 {
        Op2::Shifted(self, Shift::Lsr, n)
    }
    /// `reg, asr #n`
    pub const fn asr(self, n: u32) -> Op2 {
        Op2::Shifted(self, Shift::Asr, n)
    }
    /// `reg, <extend> #n`
    pub const fn ext(self, e: Extend, n: u32) -> Op2 {
        Op2::Extended(self, e, n)
    }
}

/// An addressing mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mem {
    /// `[base, #offset]`: scaled unsigned offset when it fits, else the
    /// unscaled signed 9-bit form.
    Off(R, i64),
    /// `[base, #offset]!`
    Pre(R, i64),
    /// `[base], #offset`
    Post(R, i64),
    /// `[base, index, <extend> #shift]`: the shift is 0 or the access size.
    Idx(R, R, Extend, bool),
    /// `[base], index` — structure loads and stores only.
    PostReg(R, R),
}

/// `[base, #offset]`
pub const fn mem(base: R, offset: i64) -> Mem {
    Mem::Off(base, offset)
}
/// `[base]`
pub const fn at(base: R) -> Mem {
    Mem::Off(base, 0)
}
/// `[base, #offset]!`
pub const fn pre(base: R, offset: i64) -> Mem {
    Mem::Pre(base, offset)
}
/// `[base], #offset`
pub const fn post(base: R, offset: i64) -> Mem {
    Mem::Post(base, offset)
}
/// `[base, index]`
pub const fn idx(base: R, index: R) -> Mem {
    Mem::Idx(base, index, Extend::Uxtx, false)
}
/// `[base, index, lsl #size]`, the index counted in elements.
pub const fn idx_scaled(base: R, index: R) -> Mem {
    Mem::Idx(base, index, Extend::Uxtx, true)
}
/// `[base], index`
pub const fn post_reg(base: R, index: R) -> Mem {
    Mem::PostReg(base, index)
}

/// The shareability domain of a barrier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Barrier {
    /// Full system.
    Sy = 0b1111,
    /// Full system, loads.
    Ld = 0b1101,
    /// Full system, stores.
    St = 0b1110,
    /// Inner shareable.
    Ish = 0b1011,
    /// Inner shareable, loads.
    IshLd = 0b1001,
    /// Inner shareable, stores.
    IshSt = 0b1010,
}

impl Barrier {
    pub(crate) const fn name(self) -> &'static str {
        match self {
            Barrier::Sy => "sy",
            Barrier::Ld => "ld",
            Barrier::St => "st",
            Barrier::Ish => "ish",
            Barrier::IshLd => "ishld",
            Barrier::IshSt => "ishst",
        }
    }
    pub(crate) const fn from_bits(b: u32) -> Option<Barrier> {
        Some(match b {
            0b1111 => Barrier::Sy,
            0b1101 => Barrier::Ld,
            0b1110 => Barrier::St,
            0b1011 => Barrier::Ish,
            0b1001 => Barrier::IshLd,
            0b1010 => Barrier::IshSt,
            _ => return None,
        })
    }
}

/// A system register read with `mrs`: `op0:op1:CRn:CRm:op2`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SysReg(pub u32);

impl SysReg {
    /// The virtual counter, readable from user space.
    pub const CNTVCT_EL0: SysReg = SysReg::new(3, 3, 14, 0, 2);
    /// The counter frequency.
    pub const CNTFRQ_EL0: SysReg = SysReg::new(3, 3, 14, 0, 0);
    /// The self-synchronised virtual counter (FEAT_ECV).
    pub const CNTVCTSS_EL0: SysReg = SysReg::new(3, 3, 14, 0, 6);
    /// The floating-point control register.
    pub const FPCR: SysReg = SysReg::new(3, 3, 4, 4, 0);
    /// The thread pointer (read-only thread id register of user space).
    pub const TPIDRRO_EL0: SysReg = SysReg::new(3, 3, 13, 0, 3);
    /// The streaming vector length query of SME, `svcr`.
    pub const SVCR: SysReg = SysReg::new(3, 3, 4, 2, 2);

    /// The register `S<op0>_<op1>_C<crn>_C<crm>_<op2>`.
    pub const fn new(op0: u32, op1: u32, crn: u32, crm: u32, op2: u32) -> SysReg {
        SysReg(((op0 & 3) << 14) | ((op1 & 7) << 11) | ((crn & 15) << 7) | ((crm & 15) << 3) | (op2 & 7))
    }
    pub(crate) fn name(self) -> String {
        match self {
            SysReg::CNTVCT_EL0 => "cntvct_el0".into(),
            SysReg::CNTFRQ_EL0 => "cntfrq_el0".into(),
            SysReg::CNTVCTSS_EL0 => "cntvctss_el0".into(),
            SysReg::FPCR => "fpcr".into(),
            SysReg::TPIDRRO_EL0 => "tpidrro_el0".into(),
            SysReg::SVCR => "svcr".into(),
            SysReg(v) => format!(
                "s{}_{}_c{}_c{}_{}",
                (v >> 14) & 3,
                (v >> 11) & 7,
                (v >> 7) & 15,
                (v >> 3) & 15,
                v & 7
            ),
        }
    }
}

/// A prefetch operation of `prfm`: `<type><target><policy>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Prefetch(pub u32);

impl Prefetch {
    /// Load, into L1, keep.
    pub const PLDL1KEEP: Prefetch = Prefetch(0b00000);
    /// Load, into L1, streaming.
    pub const PLDL1STRM: Prefetch = Prefetch(0b00001);
    /// Load, into L2, keep.
    pub const PLDL2KEEP: Prefetch = Prefetch(0b00010);
    /// Load, into L2, streaming.
    pub const PLDL2STRM: Prefetch = Prefetch(0b00011);
    /// Load, into L3, keep.
    pub const PLDL3KEEP: Prefetch = Prefetch(0b00100);
    /// Store, into L1, keep.
    pub const PSTL1KEEP: Prefetch = Prefetch(0b10000);

    pub(crate) fn name(self) -> String {
        let kind = match self.0 >> 3 {
            0 => "pld",
            1 => "pli",
            2 => "pst",
            _ => return format!("#{}", self.0),
        };
        let level = ((self.0 >> 1) & 3) + 1;
        if level > 3 {
            return format!("#{}", self.0);
        }
        let policy = if self.0 & 1 == 0 { "keep" } else { "strm" };
        format!("{kind}l{level}{policy}")
    }
}
