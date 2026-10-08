//! Registers: the general-purpose file (`X`/`W` views, the zero register and
//! the stack pointer) and the SIMD&FP file (vector arrangements, indexed
//! elements, scalar views).
//!
//! Register 31 is the zero register in some encodings and the stack pointer
//! in others. The two are distinct values here ([`XZR`] and [`SP`]), and every
//! encoder states which one its field accepts: passing `SP` where the
//! encoding would read `XZR` is an error, never a silent substitution.

/// A general-purpose register operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct R {
    /// 0..=30, 31 = the zero register, 32 = the stack pointer.
    pub(crate) n: u8,
    /// The 64-bit `X` view, or the 32-bit `W` view.
    pub(crate) x: bool,
}

/// `Xn`, 0..=30.
pub const fn x(n: u8) -> R {
    assert!(n <= 30, "x(n): n must be 0..=30, use XZR or SP");
    R { n, x: true }
}
/// `Wn`, 0..=30.
pub const fn w(n: u8) -> R {
    assert!(n <= 30, "w(n): n must be 0..=30, use WZR or WSP");
    R { n, x: false }
}
/// The 64-bit zero register.
pub const XZR: R = R { n: 31, x: true };
/// The 32-bit zero register.
pub const WZR: R = R { n: 31, x: false };
/// The stack pointer.
pub const SP: R = R { n: 32, x: true };
/// The 32-bit view of the stack pointer.
pub const WSP: R = R { n: 32, x: false };

macro_rules! gp_consts {
    ($($xn:ident $wn:ident $i:literal),* $(,)?) => {
        $(
            #[doc = concat!("`x", stringify!($i), "`")]
            pub const $xn: R = x($i);
            #[doc = concat!("`w", stringify!($i), "`")]
            pub const $wn: R = w($i);
        )*
    };
}
gp_consts!(
    X0 W0 0, X1 W1 1, X2 W2 2, X3 W3 3, X4 W4 4, X5 W5 5, X6 W6 6, X7 W7 7,
    X8 W8 8, X9 W9 9, X10 W10 10, X11 W11 11, X12 W12 12, X13 W13 13, X14 W14 14,
    X15 W15 15, X16 W16 16, X17 W17 17, X18 W18 18, X19 W19 19, X20 W20 20,
    X21 W21 21, X22 W22 22, X23 W23 23, X24 W24 24, X25 W25 25, X26 W26 26,
    X27 W27 27, X28 W28 28, X29 W29 29, X30 W30 30,
);
/// The frame pointer, `x29`.
pub const FP: R = X29;
/// The link register, `x30`.
pub const LR: R = X30;

impl R {
    /// Is this the 64-bit view?
    pub const fn is_x(self) -> bool {
        self.x
    }
    /// The same register, 64-bit view.
    pub const fn as_x(self) -> R {
        R { n: self.n, x: true }
    }
    /// The same register, 32-bit view.
    pub const fn as_w(self) -> R {
        R { n: self.n, x: false }
    }
    /// Is this the zero register?
    pub const fn is_zr(self) -> bool {
        self.n == 31
    }
    /// Is this the stack pointer?
    pub const fn is_sp(self) -> bool {
        self.n == 32
    }
    /// The 5-bit field of an encoding that reads 31 as the zero register.
    pub(crate) fn zr(self) -> Result<u32, String> {
        if self.is_sp() {
            Err(format!("{self} is not accepted here (register 31 reads as the zero register)"))
        } else {
            Ok(u32::from(self.n))
        }
    }
    /// The 5-bit field of an encoding that reads 31 as the stack pointer.
    pub(crate) fn sp(self) -> Result<u32, String> {
        if self.is_zr() {
            Err(format!("{self} is not accepted here (register 31 reads as the stack pointer)"))
        } else if self.is_sp() {
            Ok(31)
        } else {
            Ok(u32::from(self.n))
        }
    }
}

impl std::fmt::Display for R {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match (self.n, self.x) {
            (31, true) => f.write_str("xzr"),
            (31, false) => f.write_str("wzr"),
            (32, true) => f.write_str("sp"),
            (32, false) => f.write_str("wsp"),
            (n, true) => write!(f, "x{n}"),
            (n, false) => write!(f, "w{n}"),
        }
    }
}

/// A vector arrangement: lanes × lane width, in a 64- or 128-bit register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Arr {
    /// 8 bytes, 64 bits.
    B8,
    /// 16 bytes.
    B16,
    /// 4 halves, 64 bits.
    H4,
    /// 8 halves.
    H8,
    /// 2 singles, 64 bits.
    S2,
    /// 4 singles.
    S4,
    /// 1 double, 64 bits.
    D1,
    /// 2 doubles.
    D2,
}

impl Arr {
    /// log2 of the lane width in bytes: 0 = B, 1 = H, 2 = S, 3 = D.
    pub const fn size(self) -> u32 {
        match self {
            Arr::B8 | Arr::B16 => 0,
            Arr::H4 | Arr::H8 => 1,
            Arr::S2 | Arr::S4 => 2,
            Arr::D1 | Arr::D2 => 3,
        }
    }
    /// Is the register used in full (128 bits)?
    pub const fn q(self) -> bool {
        matches!(self, Arr::B16 | Arr::H8 | Arr::S4 | Arr::D2)
    }
    /// Number of lanes.
    pub const fn lanes(self) -> u32 {
        (if self.q() { 16 } else { 8 }) >> self.size()
    }
    /// The arrangement of `size` lanes in a `q`-wide register.
    pub const fn of(size: u32, q: bool) -> Arr {
        match (size, q) {
            (0, false) => Arr::B8,
            (0, true) => Arr::B16,
            (1, false) => Arr::H4,
            (1, true) => Arr::H8,
            (2, false) => Arr::S2,
            (2, true) => Arr::S4,
            (3, false) => Arr::D1,
            _ => Arr::D2,
        }
    }
    /// The assembler suffix, `4s`, `8h`…
    pub const fn suffix(self) -> &'static str {
        match self {
            Arr::B8 => "8b",
            Arr::B16 => "16b",
            Arr::H4 => "4h",
            Arr::H8 => "8h",
            Arr::S2 => "2s",
            Arr::S4 => "4s",
            Arr::D1 => "1d",
            Arr::D2 => "2d",
        }
    }
}

/// The width of an indexed element or of a scalar view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Esz {
    /// Byte.
    B,
    /// Half (16 bits).
    H,
    /// Single (32 bits).
    S,
    /// Double (64 bits).
    D,
    /// Quad (128 bits), scalar view only.
    Q,
    /// A pair of halves addressed as one 32-bit element: the `2h[i]` operand
    /// of `bfdot` by element.
    H2,
}

impl Esz {
    /// log2 of the width in bytes (`H2` counts as 32 bits).
    pub const fn size(self) -> u32 {
        match self {
            Esz::B => 0,
            Esz::H => 1,
            Esz::S | Esz::H2 => 2,
            Esz::D => 3,
            Esz::Q => 4,
        }
    }
    const fn letter(self) -> &'static str {
        match self {
            Esz::B => "b",
            Esz::H => "h",
            Esz::S => "s",
            Esz::D => "d",
            Esz::Q => "q",
            Esz::H2 => "2h",
        }
    }
}

/// A SIMD&FP register operand: a whole vector with its arrangement, one
/// indexed element, or a scalar view.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Vr {
    /// `vN.<arrangement>`.
    V(u8, Arr),
    /// `vN.<width>[index]`.
    E(u8, Esz, u8),
    /// `bN`, `hN`, `sN`, `dN`, `qN`.
    S(u8, Esz),
}

/// Vector register `n`, to be given an arrangement or an index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VReg(pub u8);

/// Vector register `n`, 0..=31.
pub const fn v(n: u8) -> VReg {
    assert!(n <= 31, "v(n): n must be 0..=31");
    VReg(n)
}

impl VReg {
    /// `.8b`
    pub const fn b8(self) -> Vr {
        Vr::V(self.0, Arr::B8)
    }
    /// `.16b`
    pub const fn b16(self) -> Vr {
        Vr::V(self.0, Arr::B16)
    }
    /// `.4h`
    pub const fn h4(self) -> Vr {
        Vr::V(self.0, Arr::H4)
    }
    /// `.8h`
    pub const fn h8(self) -> Vr {
        Vr::V(self.0, Arr::H8)
    }
    /// `.2s`
    pub const fn s2(self) -> Vr {
        Vr::V(self.0, Arr::S2)
    }
    /// `.4s`
    pub const fn s4(self) -> Vr {
        Vr::V(self.0, Arr::S4)
    }
    /// `.1d`
    pub const fn d1(self) -> Vr {
        Vr::V(self.0, Arr::D1)
    }
    /// `.2d`
    pub const fn d2(self) -> Vr {
        Vr::V(self.0, Arr::D2)
    }
    /// `.<arr>`
    pub const fn arr(self, a: Arr) -> Vr {
        Vr::V(self.0, a)
    }
    /// `.b[i]`
    pub const fn b(self, i: u8) -> Vr {
        Vr::E(self.0, Esz::B, i)
    }
    /// `.h[i]`
    pub const fn h(self, i: u8) -> Vr {
        Vr::E(self.0, Esz::H, i)
    }
    /// `.s[i]`
    pub const fn s(self, i: u8) -> Vr {
        Vr::E(self.0, Esz::S, i)
    }
    /// `.d[i]`
    pub const fn d(self, i: u8) -> Vr {
        Vr::E(self.0, Esz::D, i)
    }
    /// `.2h[i]`, the pair operand of `bfdot` by element.
    pub const fn h2(self, i: u8) -> Vr {
        Vr::E(self.0, Esz::H2, i)
    }
}

/// Scalar `bN`.
pub const fn b(n: u8) -> Vr {
    Vr::S(n, Esz::B)
}
/// Scalar `hN`.
pub const fn h(n: u8) -> Vr {
    Vr::S(n, Esz::H)
}
/// Scalar `sN`.
pub const fn s(n: u8) -> Vr {
    Vr::S(n, Esz::S)
}
/// Scalar `dN`.
pub const fn d(n: u8) -> Vr {
    Vr::S(n, Esz::D)
}
/// Scalar `qN`.
pub const fn q(n: u8) -> Vr {
    Vr::S(n, Esz::Q)
}

impl Vr {
    /// The register number.
    pub const fn n(self) -> u8 {
        match self {
            Vr::V(n, _) | Vr::E(n, _, _) | Vr::S(n, _) => n,
        }
    }
    pub(crate) fn vec(self) -> Result<(u32, Arr), String> {
        match self {
            Vr::V(n, a) => Ok((u32::from(n), a)),
            other => Err(format!("{other} is not a vector with an arrangement")),
        }
    }
    pub(crate) fn scalar(self) -> Result<(u32, Esz), String> {
        match self {
            Vr::S(n, e) => Ok((u32::from(n), e)),
            other => Err(format!("{other} is not a scalar register")),
        }
    }
    pub(crate) fn elem(self) -> Result<(u32, Esz, u32), String> {
        match self {
            Vr::E(n, e, i) => Ok((u32::from(n), e, u32::from(i))),
            other => Err(format!("{other} is not an indexed element")),
        }
    }
}

impl std::fmt::Display for Vr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Vr::V(n, a) => write!(f, "v{n}.{}", a.suffix()),
            Vr::E(n, e, i) => write!(f, "v{n}.{}[{i}]", e.letter()),
            Vr::S(n, e) => write!(f, "{}{n}", e.letter()),
        }
    }
}

/// A scalable vector register of SVE/SME, `zN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Z(pub u8);
/// A predicate register of SVE/SME, `pN`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct P(pub u8);

/// Scalable vector register `n`, 0..=31.
pub const fn z(n: u8) -> Z {
    assert!(n <= 31, "z(n): n must be 0..=31");
    Z(n)
}
/// Predicate register `n`, 0..=15.
pub const fn p(n: u8) -> P {
    assert!(n <= 15, "p(n): n must be 0..=15");
    P(n)
}
