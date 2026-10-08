//! The base instruction set: integer arithmetic and logic, moves, bitfields,
//! multiplications and divisions, conditional selects, branches, loads and
//! stores of every register file, LSE atomics, barriers and system registers.

use crate::asm::{check, Asm, Fix, Label};
use crate::imm::encode_logical;
use crate::operand::{Barrier, Cond, Extend, Mem, Op2, Prefetch, Shift, SysReg};
use crate::reg::{Esz, Vr, R, WZR, XZR};

/// A load/store target: a general-purpose register or a SIMD&FP scalar view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rt {
    /// `x`/`w`.
    R(R),
    /// `b`/`h`/`s`/`d`/`q`.
    V(Vr),
}
impl From<R> for Rt {
    fn from(r: R) -> Self {
        Rt::R(r)
    }
}
impl From<Vr> for Rt {
    fn from(v: Vr) -> Self {
        Rt::V(v)
    }
}

/// Memory ordering of an atomic.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// No ordering.
    Relaxed,
    /// Acquire.
    Acquire,
    /// Release.
    Release,
    /// Acquire and release.
    AcqRel,
}
impl Order {
    const fn bits(self) -> (u32, u32) {
        match self {
            Order::Relaxed => (0, 0),
            Order::Acquire => (1, 0),
            Order::Release => (0, 1),
            Order::AcqRel => (1, 1),
        }
    }
}

fn sf(r: R) -> u32 {
    u32::from(r.is_x()) << 31
}

fn same_width(regs: &[R]) -> Result<(), String> {
    check(regs.windows(2).all(|p| p[0].is_x() == p[1].is_x()), || {
        format!(
            "mixed register widths: {}",
            regs.iter().map(|r| r.to_string()).collect::<Vec<_>>().join(", ")
        )
    })
}

/// The access of a load/store: log2 size, SIMD bit, opc for load and store.
#[derive(Clone, Copy)]
struct Access {
    size: u32,
    v: bool,
    ld: u32,
    st: u32,
    /// log2 of the bytes moved (4 for `q`).
    scale: u32,
}

fn access(rt: Rt) -> Result<(u32, Access), String> {
    Ok(match rt {
        Rt::R(r) => (r.zr()?, Access { size: if r.is_x() { 3 } else { 2 }, v: false, ld: 1, st: 0, scale: if r.is_x() { 3 } else { 2 } }),
        Rt::V(v) => {
            let (n, e) = v.scalar()?;
            let a = match e {
                Esz::B => Access { size: 0, v: true, ld: 1, st: 0, scale: 0 },
                Esz::H => Access { size: 1, v: true, ld: 1, st: 0, scale: 1 },
                Esz::S => Access { size: 2, v: true, ld: 1, st: 0, scale: 2 },
                Esz::D => Access { size: 3, v: true, ld: 1, st: 0, scale: 3 },
                Esz::Q => Access { size: 0, v: true, ld: 3, st: 2, scale: 4 },
                Esz::H2 => return Err("2h is not a load/store width".into()),
            };
            (n, a)
        }
    })
}

/// A single-register load or store word for `opc` of `a`.
fn ldst_word(rt: u32, a: Access, opc: u32, m: Mem, writeback_guard: Option<u32>) -> Result<u32, String> {
    let base = 0x3800_0000 | (a.size << 30) | (u32::from(a.v) << 26) | (opc << 22) | rt;
    match m {
        Mem::Off(rn, off) => {
            let rn = rn.sp()?;
            let unit = 1i64 << a.scale;
            if off >= 0 && off % unit == 0 && off / unit < 4096 {
                Ok(base | 0x0100_0000 | (((off / unit) as u32) << 10) | (rn << 5))
            } else if (-256..256).contains(&off) {
                Ok(base | (((off as u32) & 0x1ff) << 12) | (rn << 5))
            } else {
                Err(format!("offset {off} fits neither the scaled nor the unscaled form"))
            }
        }
        Mem::Pre(rn, off) | Mem::Post(rn, off) => {
            let n = rn.sp()?;
            if let Some(t) = writeback_guard {
                check(t != n || n == 31, || "writeback into the transferred register".into())?;
            }
            check((-256..256).contains(&off), || format!("pre/post offset {off} outside -256..255"))?;
            let mode = if matches!(m, Mem::Pre(..)) { 0b11 } else { 0b01 };
            Ok(base | (((off as u32) & 0x1ff) << 12) | (mode << 10) | (n << 5))
        }
        Mem::Idx(rn, rm, ext, scaled) => {
            let option = match ext {
                Extend::Uxtw | Extend::Sxtw => {
                    check(!rm.is_x(), || format!("{ext:?} takes a w index"))?;
                    ext as u32
                }
                Extend::Uxtx | Extend::Sxtx => {
                    check(rm.is_x(), || format!("{ext:?} takes an x index"))?;
                    ext as u32
                }
                other => return Err(format!("{other:?} is not a register-offset extension")),
            };
            Ok(base | 0x0020_0800 | (rm.zr()? << 16) | (option << 13) | (u32::from(scaled) << 12) | (rn.sp()? << 5))
        }
        Mem::PostReg(..) => Err("register post-index exists only for structure loads".into()),
    }
}

impl Asm {
    // ── moves ───────────────────────────────────────────────────────────

    /// `mov rd, rn` — `orr` from the zero register, or `add #0` when either is the stack pointer.
    pub fn mov(&mut self, rd: R, rn: R) {
        if rd.is_sp() || rn.is_sp() {
            self.add(rd, rn, 0);
        } else {
            self.orr(rd, if rd.is_x() { XZR } else { WZR }, rn);
        }
    }

    /// `movz rd, #imm16, lsl #shift`
    pub fn movz(&mut self, rd: R, imm16: u32, shift: u32) {
        self.emit(movw(rd, 0b10, imm16, shift));
    }
    /// `movn rd, #imm16, lsl #shift`
    pub fn movn(&mut self, rd: R, imm16: u32, shift: u32) {
        self.emit(movw(rd, 0b00, imm16, shift));
    }
    /// `movk rd, #imm16, lsl #shift`
    pub fn movk(&mut self, rd: R, imm16: u32, shift: u32) {
        self.emit(movw(rd, 0b11, imm16, shift));
    }

    /// Materialise any constant in the fewest instructions: one `movz`, one
    /// `movn`, one `orr` of a bitmask immediate, or `movz`/`movn` then `movk`s.
    pub fn mov_imm(&mut self, rd: R, value: u64) {
        let bits = if rd.is_x() { 64 } else { 32 };
        if bits == 32 && value > u64::from(u32::MAX) {
            self.fail(format!("{value:#x} does not fit {rd}"));
            self.word(0);
            return;
        }
        let halves = bits / 16;
        let hw = |i: u32| ((value >> (16 * i)) & 0xffff) as u32;
        let zeros = (0..halves).filter(|&i| hw(i) == 0).count() as u32;
        let ones = (0..halves).filter(|&i| hw(i) == 0xffff).count() as u32;
        if zeros >= halves - 1 {
            let i = (0..halves).find(|&i| hw(i) != 0).unwrap_or(0);
            self.movz(rd, hw(i), 16 * i);
            return;
        }
        if ones >= halves - 1 {
            let i = (0..halves).find(|&i| hw(i) != 0xffff).unwrap_or(0);
            self.movn(rd, !hw(i) & 0xffff, 16 * i);
            return;
        }
        if !rd.is_sp() && encode_logical(value, bits).is_some() {
            self.orr(rd, if rd.is_x() { XZR } else { WZR }, value);
            return;
        }
        if ones > zeros {
            let first = (0..halves).find(|&i| hw(i) != 0xffff).unwrap_or(0);
            self.movn(rd, !hw(first) & 0xffff, 16 * first);
            for i in first + 1..halves {
                if hw(i) != 0xffff {
                    self.movk(rd, hw(i), 16 * i);
                }
            }
        } else {
            let first = (0..halves).find(|&i| hw(i) != 0).unwrap_or(0);
            self.movz(rd, hw(first), 16 * first);
            for i in first + 1..halves {
                if hw(i) != 0 {
                    self.movk(rd, hw(i), 16 * i);
                }
            }
        }
    }

    // ── arithmetic ──────────────────────────────────────────────────────

    /// `add rd, rn, op2`
    pub fn add(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = addsub(false, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `adds rd, rn, op2`
    pub fn adds(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = addsub(false, true, rd, rn, op.into());
        self.emit(w);
    }
    /// `sub rd, rn, op2`
    pub fn sub(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = addsub(true, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `subs rd, rn, op2`
    pub fn subs(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = addsub(true, true, rd, rn, op.into());
        self.emit(w);
    }
    /// `cmp rn, op2`
    pub fn cmp(&mut self, rn: R, op: impl Into<Op2>) {
        let zr = if rn.is_x() { XZR } else { WZR };
        self.subs(zr, rn, op);
    }
    /// `cmn rn, op2`
    pub fn cmn(&mut self, rn: R, op: impl Into<Op2>) {
        let zr = if rn.is_x() { XZR } else { WZR };
        self.adds(zr, rn, op);
    }
    /// `neg rd, rm`
    pub fn neg(&mut self, rd: R, rm: R) {
        let zr = if rd.is_x() { XZR } else { WZR };
        self.sub(rd, zr, rm);
    }

    // ── logic ───────────────────────────────────────────────────────────

    /// `and rd, rn, op2`
    pub fn and(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(0, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `orr rd, rn, op2`
    pub fn orr(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(1, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `eor rd, rn, op2`
    pub fn eor(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(2, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `ands rd, rn, op2`
    pub fn ands(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(3, false, rd, rn, op.into());
        self.emit(w);
    }
    /// `bic rd, rn, op2` (`and` with the complement).
    pub fn bic(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(0, true, rd, rn, op.into());
        self.emit(w);
    }
    /// `orn rd, rn, op2`
    pub fn orn(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        let w = logical(1, true, rd, rn, op.into());
        self.emit(w);
    }
    /// `tst rn, op2`
    pub fn tst(&mut self, rn: R, op: impl Into<Op2>) {
        let zr = if rn.is_x() { XZR } else { WZR };
        self.ands(zr, rn, op);
    }
    /// `mvn rd, rm`
    pub fn mvn(&mut self, rd: R, rm: R) {
        let zr = if rd.is_x() { XZR } else { WZR };
        self.orn(rd, zr, rm);
    }

    // ── shifts and bitfields ────────────────────────────────────────────

    /// `lsl rd, rn, #n` or `lsl rd, rn, rm`
    pub fn lsl(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        self.shift(rd, rn, op.into(), 0);
    }
    /// `lsr rd, rn, #n` or `lsr rd, rn, rm`
    pub fn lsr(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        self.shift(rd, rn, op.into(), 1);
    }
    /// `asr rd, rn, #n` or `asr rd, rn, rm`
    pub fn asr(&mut self, rd: R, rn: R, op: impl Into<Op2>) {
        self.shift(rd, rn, op.into(), 2);
    }
    fn shift(&mut self, rd: R, rn: R, op: Op2, kind: u32) {
        let bits = if rd.is_x() { 64 } else { 32 };
        match op {
            Op2::Imm(n) if (0..bits as i64).contains(&n) => {
                let n = n as u32;
                match kind {
                    0 => self.ubfm(rd, rn, (bits - n) % bits, bits - 1 - n),
                    1 => self.ubfm(rd, rn, n, bits - 1),
                    _ => self.sbfm(rd, rn, n, bits - 1),
                }
            }
            Op2::Imm(n) => {
                self.fail(format!("shift {n} outside 0..{bits}"));
                self.word(0);
            }
            Op2::Reg(rm) => {
                let w = (|| {
                    same_width(&[rd, rn, rm])?;
                    Ok(sf(rd) | 0x1ac0_2000 | (rm.zr()? << 16) | (kind << 10) | (rn.zr()? << 5) | rd.zr()?)
                })();
                self.emit(w);
            }
            other => {
                self.fail(format!("{other:?} is not a shift amount"));
                self.word(0);
            }
        }
    }
    /// `ubfm rd, rn, #immr, #imms`
    pub fn ubfm(&mut self, rd: R, rn: R, immr: u32, imms: u32) {
        self.emit(bitfield(0b10, rd, rn, immr, imms));
    }
    /// `sbfm rd, rn, #immr, #imms`
    pub fn sbfm(&mut self, rd: R, rn: R, immr: u32, imms: u32) {
        self.emit(bitfield(0b00, rd, rn, immr, imms));
    }
    /// `bfm rd, rn, #immr, #imms`
    pub fn bfm(&mut self, rd: R, rn: R, immr: u32, imms: u32) {
        self.emit(bitfield(0b01, rd, rn, immr, imms));
    }
    /// `ubfx rd, rn, #lsb, #width`
    pub fn ubfx(&mut self, rd: R, rn: R, lsb: u32, width: u32) {
        self.ubfm(rd, rn, lsb, lsb + width - 1);
    }
    /// `sbfx rd, rn, #lsb, #width`
    pub fn sbfx(&mut self, rd: R, rn: R, lsb: u32, width: u32) {
        self.sbfm(rd, rn, lsb, lsb + width - 1);
    }
    /// `bfi rd, rn, #lsb, #width`
    pub fn bfi(&mut self, rd: R, rn: R, lsb: u32, width: u32) {
        let bits = if rd.is_x() { 64 } else { 32 };
        self.bfm(rd, rn, (bits - lsb) % bits, width - 1);
    }
    /// `uxth wd, wn`
    pub fn uxth(&mut self, rd: R, rn: R) {
        self.ubfm(rd.as_w(), rn.as_w(), 0, 15);
    }
    /// `uxtb wd, wn`
    pub fn uxtb(&mut self, rd: R, rn: R) {
        self.ubfm(rd.as_w(), rn.as_w(), 0, 7);
    }
    /// `sxtw xd, wn`
    pub fn sxtw(&mut self, rd: R, rn: R) {
        self.sbfm(rd.as_x(), rn.as_x(), 0, 31);
    }

    // ── multiplication and division ─────────────────────────────────────

    /// `madd rd, rn, rm, ra` — `rd = ra + rn·rm`
    pub fn madd(&mut self, rd: R, rn: R, rm: R, ra: R) {
        self.emit(muladd(false, rd, rn, rm, ra));
    }
    /// `msub rd, rn, rm, ra` — `rd = ra − rn·rm`
    pub fn msub(&mut self, rd: R, rn: R, rm: R, ra: R) {
        self.emit(muladd(true, rd, rn, rm, ra));
    }
    /// `mul rd, rn, rm`
    pub fn mul(&mut self, rd: R, rn: R, rm: R) {
        let zr = if rd.is_x() { XZR } else { WZR };
        self.madd(rd, rn, rm, zr);
    }
    /// `umulh xd, xn, xm`
    pub fn umulh(&mut self, rd: R, rn: R, rm: R) {
        let w = (|| {
            check(rd.is_x() && rn.is_x() && rm.is_x(), || "umulh is 64-bit only".into())?;
            Ok(0x9bc0_7c00 | (rm.zr()? << 16) | (rn.zr()? << 5) | rd.zr()?)
        })();
        self.emit(w);
    }
    /// `udiv rd, rn, rm`
    pub fn udiv(&mut self, rd: R, rn: R, rm: R) {
        self.emit(div(false, rd, rn, rm));
    }
    /// `sdiv rd, rn, rm`
    pub fn sdiv(&mut self, rd: R, rn: R, rm: R) {
        self.emit(div(true, rd, rn, rm));
    }

    // ── conditional selects ─────────────────────────────────────────────

    /// `csel rd, rn, rm, cond`
    pub fn csel(&mut self, rd: R, rn: R, rm: R, cond: Cond) {
        self.emit(condsel(0, 0, rd, rn, rm, cond));
    }
    /// `csinc rd, rn, rm, cond`
    pub fn csinc(&mut self, rd: R, rn: R, rm: R, cond: Cond) {
        self.emit(condsel(0, 1, rd, rn, rm, cond));
    }
    /// `csinv rd, rn, rm, cond`
    pub fn csinv(&mut self, rd: R, rn: R, rm: R, cond: Cond) {
        self.emit(condsel(1, 0, rd, rn, rm, cond));
    }
    /// `csneg rd, rn, rm, cond`
    pub fn csneg(&mut self, rd: R, rn: R, rm: R, cond: Cond) {
        self.emit(condsel(1, 1, rd, rn, rm, cond));
    }
    /// `cset rd, cond` — 1 if `cond`, else 0.
    pub fn cset(&mut self, rd: R, cond: Cond) {
        let zr = if rd.is_x() { XZR } else { WZR };
        self.csinc(rd, zr, zr, cond.invert());
    }
    /// `csetm rd, cond` — all ones if `cond`, else 0.
    pub fn csetm(&mut self, rd: R, cond: Cond) {
        let zr = if rd.is_x() { XZR } else { WZR };
        self.csinv(rd, zr, zr, cond.invert());
    }
    /// `cinc rd, rn, cond` — `rn + 1` if `cond`, else `rn`.
    pub fn cinc(&mut self, rd: R, rn: R, cond: Cond) {
        self.csinc(rd, rn, rn, cond.invert());
    }

    /// `adr rd, label`
    pub fn adr(&mut self, rd: R, label: Label) {
        let w = rd.zr().map(|d| 0x1000_0000 | d);
        self.emit_fix(w, label, Fix::Adr21);
    }

    // ── branches ────────────────────────────────────────────────────────

    /// `b label`
    pub fn b(&mut self, label: Label) {
        self.emit_fix(Ok(0x1400_0000), label, Fix::Imm26);
    }
    /// `bl label`
    pub fn bl(&mut self, label: Label) {
        self.emit_fix(Ok(0x9400_0000), label, Fix::Imm26);
    }
    /// `b.cond label`
    pub fn b_cond(&mut self, cond: Cond, label: Label) {
        self.emit_fix(Ok(0x5400_0000 | cond as u32), label, Fix::Imm19);
    }
    /// `b.eq label`
    pub fn beq(&mut self, l: Label) {
        self.b_cond(Cond::Eq, l);
    }
    /// `b.ne label`
    pub fn bne(&mut self, l: Label) {
        self.b_cond(Cond::Ne, l);
    }
    /// `b.hs label` (unsigned ≥)
    pub fn bhs(&mut self, l: Label) {
        self.b_cond(Cond::Hs, l);
    }
    /// `b.lo label` (unsigned <)
    pub fn blo(&mut self, l: Label) {
        self.b_cond(Cond::Lo, l);
    }
    /// `b.hi label` (unsigned >)
    pub fn bhi(&mut self, l: Label) {
        self.b_cond(Cond::Hi, l);
    }
    /// `b.ls label` (unsigned ≤)
    pub fn bls(&mut self, l: Label) {
        self.b_cond(Cond::Ls, l);
    }
    /// `b.lt label`
    pub fn blt(&mut self, l: Label) {
        self.b_cond(Cond::Lt, l);
    }
    /// `b.ge label`
    pub fn bge(&mut self, l: Label) {
        self.b_cond(Cond::Ge, l);
    }
    /// `b.gt label`
    pub fn bgt(&mut self, l: Label) {
        self.b_cond(Cond::Gt, l);
    }
    /// `b.le label`
    pub fn ble(&mut self, l: Label) {
        self.b_cond(Cond::Le, l);
    }
    /// `b.mi label`
    pub fn bmi(&mut self, l: Label) {
        self.b_cond(Cond::Mi, l);
    }
    /// `b.pl label`
    pub fn bpl(&mut self, l: Label) {
        self.b_cond(Cond::Pl, l);
    }
    /// `cbz rt, label`
    pub fn cbz(&mut self, rt: R, label: Label) {
        let w = rt.zr().map(|t| sf(rt) | 0x3400_0000 | t);
        self.emit_fix(w, label, Fix::Imm19);
    }
    /// `cbnz rt, label`
    pub fn cbnz(&mut self, rt: R, label: Label) {
        let w = rt.zr().map(|t| sf(rt) | 0x3500_0000 | t);
        self.emit_fix(w, label, Fix::Imm19);
    }
    /// `tbz rt, #bit, label`
    pub fn tbz(&mut self, rt: R, bit: u32, label: Label) {
        let w = tb(0x3600_0000, rt, bit);
        self.emit_fix(w, label, Fix::Imm14);
    }
    /// `tbnz rt, #bit, label`
    pub fn tbnz(&mut self, rt: R, bit: u32, label: Label) {
        let w = tb(0x3700_0000, rt, bit);
        self.emit_fix(w, label, Fix::Imm14);
    }
    /// `br xn`
    pub fn br(&mut self, rn: R) {
        let w = (|| {
            check(rn.is_x(), || "br takes an x register".into())?;
            Ok(0xd61f_0000 | (rn.zr()? << 5))
        })();
        self.emit(w);
    }
    /// `blr xn`
    pub fn blr(&mut self, rn: R) {
        let w = (|| {
            check(rn.is_x(), || "blr takes an x register".into())?;
            Ok(0xd63f_0000 | (rn.zr()? << 5))
        })();
        self.emit(w);
    }
    /// `ret` (through `x30`)
    pub fn ret(&mut self) {
        self.word(0xd65f_03c0);
    }

    // ── loads and stores ────────────────────────────────────────────────

    /// `ldr rt, mem` — any general-purpose or SIMD&FP width.
    pub fn ldr(&mut self, rt: impl Into<Rt>, m: Mem) {
        let rt = rt.into();
        let w = access(rt).and_then(|(t, a)| ldst_word(t, a, a.ld, m, matches!(rt, Rt::R(_)).then_some(t)));
        self.emit(w);
    }
    /// `str rt, mem`
    pub fn str(&mut self, rt: impl Into<Rt>, m: Mem) {
        let rt = rt.into();
        let w = access(rt).and_then(|(t, a)| ldst_word(t, a, a.st, m, None));
        self.emit(w);
    }
    /// `ldrb wt, mem`
    pub fn ldrb(&mut self, rt: R, m: Mem) {
        self.emit(narrow(rt, 0, 1, m, true));
    }
    /// `strb wt, mem`
    pub fn strb(&mut self, rt: R, m: Mem) {
        self.emit(narrow(rt, 0, 0, m, false));
    }
    /// `ldrh wt, mem`
    pub fn ldrh(&mut self, rt: R, m: Mem) {
        self.emit(narrow(rt, 1, 1, m, true));
    }
    /// `strh wt, mem`
    pub fn strh(&mut self, rt: R, m: Mem) {
        self.emit(narrow(rt, 1, 0, m, false));
    }
    /// `ldrsw xt, mem`
    pub fn ldrsw(&mut self, rt: R, m: Mem) {
        let w = (|| {
            check(rt.is_x(), || "ldrsw loads into an x register".into())?;
            let t = rt.zr()?;
            ldst_word(t, Access { size: 2, v: false, ld: 2, st: 2, scale: 2 }, 2, m, Some(t))
        })();
        self.emit(w);
    }
    /// `ldp rt1, rt2, mem`
    pub fn ldp(&mut self, rt1: impl Into<Rt>, rt2: impl Into<Rt>, m: Mem) {
        self.emit(pair(true, rt1.into(), rt2.into(), m));
    }
    /// `stp rt1, rt2, mem`
    pub fn stp(&mut self, rt1: impl Into<Rt>, rt2: impl Into<Rt>, m: Mem) {
        self.emit(pair(false, rt1.into(), rt2.into(), m));
    }
    /// `prfm op, mem`
    pub fn prfm(&mut self, op: Prefetch, m: Mem) {
        let w = (|| {
            check(op.0 < 32, || "prefetch operation outside 0..31".into())?;
            match m {
                Mem::Off(rn, off) if off >= 0 && off % 8 == 0 && off / 8 < 4096 => {
                    Ok(0xf980_0000 | (((off / 8) as u32) << 10) | (rn.sp()? << 5) | op.0)
                }
                Mem::Off(rn, off) if (-256..256).contains(&off) => {
                    Ok(0xf880_0000 | (((off as u32) & 0x1ff) << 12) | (rn.sp()? << 5) | op.0)
                }
                Mem::Idx(..) => ldst_word(op.0, Access { size: 3, v: false, ld: 2, st: 2, scale: 3 }, 2, m, None),
                other => Err(format!("prfm cannot address {other:?}")),
            }
        })();
        self.emit(w);
    }

    // ── atomics and exclusives ──────────────────────────────────────────

    /// `ldadd{a,l,al} rs, rt, [rn]` — `rt ← [rn]; [rn] ← [rn] + rs`
    pub fn ldadd(&mut self, order: Order, rs: R, rt: R, rn: R) {
        self.emit(lse(0x3820_0000, order, rs, rt, rn));
    }
    /// `swp{a,l,al} rs, rt, [rn]`
    pub fn swp(&mut self, order: Order, rs: R, rt: R, rn: R) {
        self.emit(lse(0x3820_8000, order, rs, rt, rn));
    }
    /// `ldset{a,l,al} rs, rt, [rn]` — `[rn] |= rs`
    pub fn ldset(&mut self, order: Order, rs: R, rt: R, rn: R) {
        self.emit(lse(0x3820_3000, order, rs, rt, rn));
    }
    /// `cas{a,l,al} rs, rt, [rn]` — if `[rn] == rs` then `[rn] ← rt`; `rs ← old`.
    pub fn cas(&mut self, order: Order, rs: R, rt: R, rn: R) {
        let w = (|| {
            same_width(&[rs, rt])?;
            check(rn.is_x(), || "the address is an x register".into())?;
            let (l, o0) = order.bits();
            Ok(0x88a0_7c00 | (u32::from(rt.is_x()) << 30) | (l << 22) | (rs.zr()? << 16) | (o0 << 15) | (rn.sp()? << 5) | rt.zr()?)
        })();
        self.emit(w);
    }
    /// `ldar rt, [rn]`
    pub fn ldar(&mut self, rt: R, rn: R) {
        self.emit(exclusive(0x88df_fc00, None, rt, rn));
    }
    /// `stlr rt, [rn]`
    pub fn stlr(&mut self, rt: R, rn: R) {
        self.emit(exclusive(0x889f_fc00, None, rt, rn));
    }
    /// `ldaxr rt, [rn]`
    pub fn ldaxr(&mut self, rt: R, rn: R) {
        self.emit(exclusive(0x885f_fc00, None, rt, rn));
    }
    /// `ldxr rt, [rn]`
    pub fn ldxr(&mut self, rt: R, rn: R) {
        self.emit(exclusive(0x885f_7c00, None, rt, rn));
    }
    /// `stlxr ws, rt, [rn]`
    pub fn stlxr(&mut self, ws: R, rt: R, rn: R) {
        self.emit(exclusive(0x8800_fc00, Some(ws), rt, rn));
    }
    /// `stxr ws, rt, [rn]`
    pub fn stxr(&mut self, ws: R, rt: R, rn: R) {
        self.emit(exclusive(0x8800_7c00, Some(ws), rt, rn));
    }
    /// `ldapr rt, [rn]` (RCpc acquire)
    pub fn ldapr(&mut self, rt: R, rn: R) {
        let w = (|| {
            check(rn.is_x(), || "the address is an x register".into())?;
            Ok(0xb8bf_c000 | (u32::from(rt.is_x()) << 30) | (rn.sp()? << 5) | rt.zr()?)
        })();
        self.emit(w);
    }

    // ── system ──────────────────────────────────────────────────────────

    /// `nop`
    pub fn nop(&mut self) {
        self.word(crate::NOP);
    }
    /// `yield`
    pub fn yield_(&mut self) {
        self.word(0xd503_203f);
    }
    /// `wfe`
    pub fn wfe(&mut self) {
        self.word(0xd503_205f);
    }
    /// `wfi`
    pub fn wfi(&mut self) {
        self.word(0xd503_207f);
    }
    /// `sev`
    pub fn sev(&mut self) {
        self.word(0xd503_209f);
    }
    /// `sevl`
    pub fn sevl(&mut self) {
        self.word(0xd503_20bf);
    }
    /// `dmb <domain>`
    pub fn dmb(&mut self, b: Barrier) {
        self.word(0xd503_30bf | ((b as u32) << 8));
    }
    /// `dsb <domain>`
    pub fn dsb(&mut self, b: Barrier) {
        self.word(0xd503_309f | ((b as u32) << 8));
    }
    /// `isb`
    pub fn isb(&mut self) {
        self.word(0xd503_3fdf);
    }
    /// `mrs xt, <sysreg>`
    pub fn mrs(&mut self, rt: R, reg: SysReg) {
        let w = (|| {
            check(rt.is_x(), || "mrs reads into an x register".into())?;
            Ok(0xd520_0000 | (reg.0 << 5) | rt.zr()?)
        })();
        self.emit(w);
    }
    /// `msr <sysreg>, xt`
    pub fn msr(&mut self, reg: SysReg, rt: R) {
        let w = (|| {
            check(rt.is_x(), || "msr writes from an x register".into())?;
            Ok(0xd500_0000 | (reg.0 << 5) | rt.zr()?)
        })();
        self.emit(w);
    }
    /// `brk #imm16`
    pub fn brk(&mut self, imm: u32) {
        let w = check(imm < 0x10000, || "brk takes a 16-bit immediate".into()).map(|_| 0xd420_0000 | (imm << 5));
        self.emit(w);
    }
    /// `udf #imm16` — a guaranteed undefined instruction.
    pub fn udf(&mut self, imm: u32) {
        let w = check(imm < 0x10000, || "udf takes a 16-bit immediate".into()).map(|_| imm);
        self.emit(w);
    }
}

fn movw(rd: R, opc: u32, imm16: u32, shift: u32) -> Result<u32, String> {
    let bits = if rd.is_x() { 64 } else { 32 };
    check(imm16 < 0x10000, || format!("{imm16:#x} is not a 16-bit immediate"))?;
    check(shift.is_multiple_of(16) && shift < bits, || format!("shift {shift} is not a multiple of 16 below {bits}"))?;
    Ok(sf(rd) | (opc << 29) | 0x1280_0000 | ((shift / 16) << 21) | (imm16 << 5) | rd.zr()?)
}

fn addsub(sub: bool, setflags: bool, rd: R, rn: R, op: Op2) -> Result<u32, String> {
    let rd_field = |r: R| if setflags { r.zr() } else { r.sp() };
    let top = |sub: bool| sf(rd) | (u32::from(sub) << 30) | (u32::from(setflags) << 29);
    match op {
        Op2::Imm(v) => {
            same_width(&[rd, rn])?;
            let (sub, v) = if v < 0 { (!sub, v.unsigned_abs()) } else { (sub, v as u64) };
            let (sh, imm) = if v < 4096 {
                (0, v)
            } else if v & 0xfff == 0 && v >> 12 < 4096 {
                (1, v >> 12)
            } else {
                return Err(format!("{v:#x} is not a 12-bit immediate (optionally shifted by 12)"));
            };
            Ok(top(sub) | 0x1100_0000 | (sh << 22) | ((imm as u32) << 10) | (rn.sp()? << 5) | rd_field(rd)?)
        }
        Op2::Reg(rm) if rd.is_sp() || rn.is_sp() => {
            addsub(sub, setflags, rd, rn, Op2::Extended(rm, if rd.is_x() { Extend::Uxtx } else { Extend::Uxtw }, 0))
        }
        Op2::Reg(rm) => addsub(sub, setflags, rd, rn, Op2::Shifted(rm, Shift::Lsl, 0)),
        Op2::Shifted(rm, shift, amount) => {
            same_width(&[rd, rn, rm])?;
            check(shift != Shift::Ror, || "add/sub cannot rotate".into())?;
            check(amount < if rd.is_x() { 64 } else { 32 }, || format!("shift {amount} too large"))?;
            Ok(top(sub) | 0x0b00_0000 | ((shift as u32) << 22) | (rm.zr()? << 16) | (amount << 10) | (rn.zr()? << 5) | rd.zr()?)
        }
        Op2::Extended(rm, ext, amount) => {
            same_width(&[rd, rn])?;
            check(amount <= 4, || format!("extend shift {amount} above 4"))?;
            let wide = matches!(ext, Extend::Uxtx | Extend::Sxtx);
            check(rm.is_x() == wide, || format!("{ext:?} takes a {} register", if wide { "x" } else { "w" }))?;
            Ok(top(sub) | 0x0b20_0000 | (rm.zr()? << 16) | ((ext as u32) << 13) | (amount << 10) | (rn.sp()? << 5) | rd_field(rd)?)
        }
    }
}

fn logical(opc: u32, invert: bool, rd: R, rn: R, op: Op2) -> Result<u32, String> {
    let bits = if rd.is_x() { 64 } else { 32 };
    let top = sf(rd) | (opc << 29);
    match op {
        Op2::Imm(v) => {
            same_width(&[rd, rn])?;
            let mask = if bits == 64 { u64::MAX } else { 0xffff_ffff };
            let v = if invert { !(v as u64) } else { v as u64 } & mask;
            let (n, immr, imms) = encode_logical(v, bits).ok_or_else(|| format!("{v:#x} is not a bitmask immediate"))?;
            let d = if opc == 3 { rd.zr()? } else { rd.sp()? };
            Ok(top | 0x1200_0000 | (n << 22) | (immr << 16) | (imms << 10) | (rn.zr()? << 5) | d)
        }
        Op2::Reg(rm) => logical(opc, invert, rd, rn, Op2::Shifted(rm, Shift::Lsl, 0)),
        Op2::Shifted(rm, shift, amount) => {
            same_width(&[rd, rn, rm])?;
            check(amount < bits, || format!("shift {amount} too large"))?;
            Ok(top | 0x0a00_0000 | ((shift as u32) << 22) | (u32::from(invert) << 21) | (rm.zr()? << 16) | (amount << 10) | (rn.zr()? << 5) | rd.zr()?)
        }
        Op2::Extended(..) => Err("logical instructions take no extended register".into()),
    }
}

fn bitfield(opc: u32, rd: R, rn: R, immr: u32, imms: u32) -> Result<u32, String> {
    same_width(&[rd, rn])?;
    let bits = if rd.is_x() { 64 } else { 32 };
    check(immr < bits && imms < bits, || format!("bitfield ({immr}, {imms}) outside 0..{bits}"))?;
    let n = u32::from(rd.is_x());
    Ok(sf(rd) | (opc << 29) | 0x1300_0000 | (n << 22) | (immr << 16) | (imms << 10) | (rn.zr()? << 5) | rd.zr()?)
}

fn muladd(sub: bool, rd: R, rn: R, rm: R, ra: R) -> Result<u32, String> {
    same_width(&[rd, rn, rm, ra])?;
    Ok(sf(rd) | 0x1b00_0000 | (rm.zr()? << 16) | (u32::from(sub) << 15) | (ra.zr()? << 10) | (rn.zr()? << 5) | rd.zr()?)
}

fn div(signed: bool, rd: R, rn: R, rm: R) -> Result<u32, String> {
    same_width(&[rd, rn, rm])?;
    Ok(sf(rd) | 0x1ac0_0800 | (u32::from(signed) << 10) | (rm.zr()? << 16) | (rn.zr()? << 5) | rd.zr()?)
}

fn condsel(op: u32, op2: u32, rd: R, rn: R, rm: R, cond: Cond) -> Result<u32, String> {
    same_width(&[rd, rn, rm])?;
    Ok(sf(rd) | (op << 30) | 0x1a80_0000 | (rm.zr()? << 16) | ((cond as u32) << 12) | (op2 << 10) | (rn.zr()? << 5) | rd.zr()?)
}

fn tb(base: u32, rt: R, bit: u32) -> Result<u32, String> {
    check(bit < if rt.is_x() { 64 } else { 32 }, || format!("bit {bit} beyond {rt}"))?;
    Ok(base | ((bit >> 5) << 31) | ((bit & 31) << 19) | rt.zr()?)
}

fn narrow(rt: R, size: u32, opc: u32, m: Mem, load: bool) -> Result<u32, String> {
    check(!rt.is_x(), || "byte and half accesses name a w register".into())?;
    let t = rt.zr()?;
    ldst_word(t, Access { size, v: false, ld: 1, st: 0, scale: size }, opc, m, load.then_some(t))
}

fn pair(load: bool, rt1: Rt, rt2: Rt, m: Mem) -> Result<u32, String> {
    let (t1, a1) = access(rt1)?;
    let (t2, a2) = access(rt2)?;
    check(a1.scale == a2.scale && a1.v == a2.v, || "a pair moves two registers of one width".into())?;
    let (opc, scale) = match (a1.v, a1.scale) {
        (false, 2) => (0b00, 2),
        (false, 3) => (0b10, 3),
        (true, 2) => (0b00, 2),
        (true, 3) => (0b01, 3),
        (true, 4) => (0b10, 4),
        _ => return Err("pairs move w, x, s, d or q registers".into()),
    };
    if load {
        check(t1 != t2, || "a pair load into one register".into())?;
    }
    let (rn, off, mode) = match m {
        Mem::Off(rn, off) => (rn, off, 0b10),
        Mem::Pre(rn, off) => (rn, off, 0b11),
        Mem::Post(rn, off) => (rn, off, 0b01),
        other => return Err(format!("pairs cannot address {other:?}")),
    };
    let unit = 1i64 << scale;
    check(off % unit == 0 && (-64..64).contains(&(off / unit)), || format!("pair offset {off} is not a 7-bit multiple of {unit}"))?;
    let n = rn.sp()?;
    if load && mode != 0b10 && !a1.v {
        check(n != t1 && n != t2, || "writeback into a transferred register".into())?;
    }
    Ok((opc << 30) | 0x2800_0000 | (u32::from(a1.v) << 26) | (mode << 23) | (u32::from(load) << 22) | ((((off / unit) as u32) & 0x7f) << 15) | (t2 << 10) | (n << 5) | t1)
}

fn lse(base: u32, order: Order, rs: R, rt: R, rn: R) -> Result<u32, String> {
    same_width(&[rs, rt])?;
    check(rn.is_x(), || "the address is an x register".into())?;
    let (a, r) = order.bits();
    Ok(base | ((if rt.is_x() { 3 } else { 2 }) << 30) | (a << 23) | (r << 22) | (rs.zr()? << 16) | (rn.sp()? << 5) | rt.zr()?)
}

fn exclusive(base: u32, ws: Option<R>, rt: R, rn: R) -> Result<u32, String> {
    check(rn.is_x(), || "the address is an x register".into())?;
    let s = match ws {
        Some(ws) => {
            check(!ws.is_x(), || "the status register is a w register".into())?;
            let s = ws.zr()?;
            check(s != rt.zr()? && s != rn.sp()?, || "the status register overlaps the transfer".into())?;
            s << 16
        }
        None => 0,
    };
    Ok(base | (u32::from(rt.is_x()) << 30) | s | (rn.sp()? << 5) | rt.zr()?)
}
