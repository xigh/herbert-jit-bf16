//! Advanced SIMD and floating point: vector and scalar arithmetic, by-element
//! forms, conversions, moves between register files, permutations, shifts,
//! lane reductions, and the BF16 extension (`bfdot`, `bfmmla`, `bfmlal[bt]`,
//! `bfcvt`, `bfcvtn[2]`).
//!
//! A method takes the operands the assembler syntax names: `fadd` over three
//! vectors of one arrangement is the vector form, over three scalars the
//! scalar form; `fmla` with an indexed third operand is the by-element form.
//! Any other combination is refused.

use crate::asm::{check, Asm};
use crate::imm::encode_fp8;
use crate::operand::{Cond, Mem};
use crate::reg::{Arr, Esz, Vr, R};

fn q(a: Arr) -> u32 {
    u32::from(a.q()) << 30
}

/// Three vectors of one arrangement.
fn v3(d: Vr, n: Vr, m: Vr) -> Result<(u32, u32, u32, Arr), String> {
    let (d, a) = d.vec()?;
    let (n, an) = n.vec()?;
    let (m, am) = m.vec()?;
    check(a == an && a == am, || format!("arrangements differ: {a:?} {an:?} {am:?}"))?;
    Ok((d, n, m, a))
}
/// Two vectors of one arrangement.
fn v2(d: Vr, n: Vr) -> Result<(u32, u32, Arr), String> {
    let (d, a) = d.vec()?;
    let (n, an) = n.vec()?;
    check(a == an, || format!("arrangements differ: {a:?} {an:?}"))?;
    Ok((d, n, a))
}
/// A floating-point vector arrangement: its `sz` bit.
fn fp_arr(a: Arr) -> Result<u32, String> {
    match a {
        Arr::S2 | Arr::S4 => Ok(0),
        Arr::D2 => Ok(1),
        other => Err(format!("{other:?} is not a single or double arrangement")),
    }
}
/// The `type` field of a scalar floating-point register.
fn fp_type(e: Esz) -> Result<u32, String> {
    match e {
        Esz::S => Ok(0),
        Esz::D => Ok(1),
        Esz::H => Ok(3),
        other => Err(format!("{other:?} is not a floating-point scalar")),
    }
}
/// The `sz` bit of an Advanced SIMD scalar form (single or double only).
fn fp_sz(e: Esz) -> Result<u32, String> {
    match e {
        Esz::S => Ok(0),
        Esz::D => Ok(1),
        other => Err(format!("{other:?} has no Advanced SIMD scalar form here")),
    }
}
/// Three scalars of one width.
fn s3(d: Vr, n: Vr, m: Vr) -> Result<(u32, u32, u32, Esz), String> {
    let (d, e) = d.scalar()?;
    let (n, en) = n.scalar()?;
    let (m, em) = m.scalar()?;
    check(e == en && e == em, || "scalar widths differ".into())?;
    Ok((d, n, m, e))
}
fn s2(d: Vr, n: Vr) -> Result<(u32, u32, Esz), String> {
    let (d, e) = d.scalar()?;
    let (n, en) = n.scalar()?;
    check(e == en, || "scalar widths differ".into())?;
    Ok((d, n, e))
}

/// The `imm5` of an element operand, from its width and index.
fn imm5(e: Esz, i: u32) -> Result<u32, String> {
    let (size, lanes) = match e {
        Esz::B => (0, 16),
        Esz::H => (1, 8),
        Esz::S => (2, 4),
        Esz::D => (3, 2),
        other => return Err(format!("{other:?} is not an element width")),
    };
    check(i < lanes, || format!("index {i} beyond {lanes} lanes"))?;
    Ok(((i << 1) | 1) << size)
}

/// Floating-point binary operation encodings: vector, and scalar (with
/// whether the scalar form is the Advanced SIMD scalar class).
struct Fp2 {
    vector: u32,
    scalar: Option<(u32, bool)>,
}

const FADD: Fp2 = Fp2 { vector: 0x0e20_d400, scalar: Some((0x1e20_2800, false)) };
const FSUB: Fp2 = Fp2 { vector: 0x0ea0_d400, scalar: Some((0x1e20_3800, false)) };
const FMUL: Fp2 = Fp2 { vector: 0x2e20_dc00, scalar: Some((0x1e20_0800, false)) };
const FDIV: Fp2 = Fp2 { vector: 0x2e20_fc00, scalar: Some((0x1e20_1800, false)) };
const FMAX: Fp2 = Fp2 { vector: 0x0e20_f400, scalar: Some((0x1e20_4800, false)) };
const FMIN: Fp2 = Fp2 { vector: 0x0ea0_f400, scalar: Some((0x1e20_5800, false)) };
const FMAXNM: Fp2 = Fp2 { vector: 0x0e20_c400, scalar: Some((0x1e20_6800, false)) };
const FMINNM: Fp2 = Fp2 { vector: 0x0ea0_c400, scalar: Some((0x1e20_7800, false)) };
const FNMUL: Fp2 = Fp2 { vector: 0, scalar: Some((0x1e20_8800, false)) };
const FABD: Fp2 = Fp2 { vector: 0x2ea0_d400, scalar: Some((0x7ea0_d400, true)) };
const FRECPS: Fp2 = Fp2 { vector: 0x0e20_fc00, scalar: Some((0x5e20_fc00, true)) };
const FRSQRTS: Fp2 = Fp2 { vector: 0x0ea0_fc00, scalar: Some((0x5ea0_fc00, true)) };
const FCMEQ: Fp2 = Fp2 { vector: 0x0e20_e400, scalar: Some((0x5e20_e400, true)) };
const FCMGE: Fp2 = Fp2 { vector: 0x2e20_e400, scalar: Some((0x7e20_e400, true)) };
const FCMGT: Fp2 = Fp2 { vector: 0x2ea0_e400, scalar: Some((0x7ea0_e400, true)) };
const FADDP: Fp2 = Fp2 { vector: 0x2e20_d400, scalar: None };
const FMAXP: Fp2 = Fp2 { vector: 0x2e20_f400, scalar: None };
const FMINP: Fp2 = Fp2 { vector: 0x2ea0_f400, scalar: None };
const FMLA: Fp2 = Fp2 { vector: 0x0e20_cc00, scalar: None };
const FMLS: Fp2 = Fp2 { vector: 0x0ea0_cc00, scalar: None };

fn fp_binary(op: &Fp2, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    match d {
        Vr::V(..) => {
            check(op.vector != 0, || "no vector form".into())?;
            let (d, n, m, a) = v3(d, n, m)?;
            Ok(op.vector | q(a) | (fp_arr(a)? << 22) | (m << 16) | (n << 5) | d)
        }
        Vr::S(..) => {
            let (base, simd) = op.scalar.ok_or("no scalar form")?;
            let (d, n, m, e) = s3(d, n, m)?;
            let t = if simd { fp_sz(e)? } else { fp_type(e)? };
            Ok(base | (t << 22) | (m << 16) | (n << 5) | d)
        }
        Vr::E(..) => Err(format!("{d} cannot be a destination here")),
    }
}

/// Floating-point unary encodings: vector two-register misc, scalar (and
/// whether the scalar is the Advanced SIMD scalar class).
struct Fp1 {
    vector: u32,
    scalar: Option<(u32, bool)>,
}
const FABS: Fp1 = Fp1 { vector: 0x0ea0_f800, scalar: Some((0x1e20_c000, false)) };
const FNEG: Fp1 = Fp1 { vector: 0x2ea0_f800, scalar: Some((0x1e21_4000, false)) };
const FSQRT: Fp1 = Fp1 { vector: 0x2ea1_f800, scalar: Some((0x1e21_c000, false)) };
const FRECPE: Fp1 = Fp1 { vector: 0x0ea1_d800, scalar: Some((0x5ea1_d800, true)) };
const FRSQRTE: Fp1 = Fp1 { vector: 0x2ea1_d800, scalar: Some((0x7ea1_d800, true)) };
const FRINTN: Fp1 = Fp1 { vector: 0x0e21_8800, scalar: Some((0x1e24_4000, false)) };
const FRINTM: Fp1 = Fp1 { vector: 0x0e21_9800, scalar: Some((0x1e25_4000, false)) };
const FRINTP: Fp1 = Fp1 { vector: 0x0ea1_8800, scalar: Some((0x1e24_c000, false)) };
const FRINTZ: Fp1 = Fp1 { vector: 0x0ea1_9800, scalar: Some((0x1e25_c000, false)) };
const FCVTZS: Fp1 = Fp1 { vector: 0x0ea1_b800, scalar: Some((0x5ea1_b800, true)) };
const FCVTNS: Fp1 = Fp1 { vector: 0x0e21_a800, scalar: Some((0x5e21_a800, true)) };
const FCVTMS: Fp1 = Fp1 { vector: 0x0e21_b800, scalar: Some((0x5e21_b800, true)) };
const SCVTF: Fp1 = Fp1 { vector: 0x0e21_d800, scalar: Some((0x5e21_d800, true)) };
const UCVTF: Fp1 = Fp1 { vector: 0x2e21_d800, scalar: Some((0x7e21_d800, true)) };
const FCMEQ0: Fp1 = Fp1 { vector: 0x0ea0_d800, scalar: None };
const FCMGT0: Fp1 = Fp1 { vector: 0x0ea0_c800, scalar: None };
const FCMGE0: Fp1 = Fp1 { vector: 0x2ea0_c800, scalar: None };
const FCMLT0: Fp1 = Fp1 { vector: 0x0ea0_e800, scalar: None };

fn fp_unary(op: &Fp1, d: Vr, n: Vr) -> Result<u32, String> {
    match d {
        Vr::V(..) => {
            let (d, n, a) = v2(d, n)?;
            Ok(op.vector | q(a) | (fp_arr(a)? << 22) | (n << 5) | d)
        }
        Vr::S(..) => {
            let (base, simd) = op.scalar.ok_or("no scalar form")?;
            let (d, n, e) = s2(d, n)?;
            let t = if simd { fp_sz(e)? } else { fp_type(e)? };
            Ok(base | (t << 22) | (n << 5) | d)
        }
        Vr::E(..) => Err(format!("{d} cannot be a destination here")),
    }
}

/// The by-element form of a floating-point multiply: vector or scalar.
fn fp_elem(vector_base: u32, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    let (mn, me, i) = m.elem()?;
    let (top, dd, nn, sz) = match d {
        Vr::V(..) => {
            let (dd, nn, a) = v2(d, n)?;
            (vector_base | q(a), dd, nn, fp_arr(a)?)
        }
        Vr::S(..) => {
            let (dd, nn, e) = s2(d, n)?;
            (vector_base | 0x5000_0000, dd, nn, fp_sz(e)?)
        }
        Vr::E(..) => return Err(format!("{d} cannot be a destination here")),
    };
    let (hl, want) = match (sz, me) {
        (0, Esz::S) => {
            check(i < 4, || format!("index {i} beyond 4 singles"))?;
            (((i >> 1) << 11) | ((i & 1) << 21), 0)
        }
        (1, Esz::D) => {
            check(i < 2, || format!("index {i} beyond 2 doubles"))?;
            (i << 11, 1)
        }
        _ => return Err(format!("element {m} does not match the arrangement")),
    };
    Ok(top | (want << 22) | hl | (mn << 16) | (nn << 5) | dd)
}

/// Integer three-same: `mask` lists the lane sizes the operation accepts.
fn int3(base: u32, sizes: u32, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    let (d, n, m, a) = v3(d, n, m)?;
    check(sizes & (1 << a.size()) != 0 && a != Arr::D1, || format!("{a:?} not accepted"))?;
    Ok(base | q(a) | (a.size() << 22) | (m << 16) | (n << 5) | d)
}
/// Bytewise logic: the size field belongs to the opcode.
fn logic3(base: u32, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    let (d, n, m, a) = v3(d, n, m)?;
    check(matches!(a, Arr::B8 | Arr::B16), || "bitwise operations take 8b or 16b".into())?;
    Ok(base | q(a) | (m << 16) | (n << 5) | d)
}

/// The `immh:immb` of a shift by `amount` on lanes of `size`.
fn shift_imm(size: u32, amount: u32, left: bool) -> Result<u32, String> {
    let esize = 8u32 << size;
    if left {
        check(amount < esize, || format!("left shift {amount} ≥ {esize}"))?;
        Ok(esize + amount)
    } else {
        check((1..=esize).contains(&amount), || format!("right shift {amount} outside 1..={esize}"))?;
        Ok(2 * esize - amount)
    }
}

impl Asm {
    // ── floating-point arithmetic ───────────────────────────────────────

    /// `fadd`
    pub fn fadd(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FADD, d, n, m));
    }
    /// `fsub`
    pub fn fsub(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FSUB, d, n, m));
    }
    /// `fmul`, vector, scalar, or by element.
    pub fn fmul(&mut self, d: Vr, n: Vr, m: Vr) {
        if matches!(m, Vr::E(..)) {
            self.emit(fp_elem(0x0f80_9000, d, n, m));
        } else {
            self.emit(fp_binary(&FMUL, d, n, m));
        }
    }
    /// `fdiv`
    pub fn fdiv(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FDIV, d, n, m));
    }
    /// `fmax`
    pub fn fmax(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FMAX, d, n, m));
    }
    /// `fmin`
    pub fn fmin(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FMIN, d, n, m));
    }
    /// `fmaxnm`
    pub fn fmaxnm(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FMAXNM, d, n, m));
    }
    /// `fminnm`
    pub fn fminnm(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FMINNM, d, n, m));
    }
    /// `fnmul`, scalar.
    pub fn fnmul(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FNMUL, d, n, m));
    }
    /// `fabd`
    pub fn fabd(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FABD, d, n, m));
    }
    /// `frecps`
    pub fn frecps(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FRECPS, d, n, m));
    }
    /// `frsqrts`
    pub fn frsqrts(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FRSQRTS, d, n, m));
    }
    /// `fcmeq`
    pub fn fcmeq(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FCMEQ, d, n, m));
    }
    /// `fcmge`
    pub fn fcmge(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FCMGE, d, n, m));
    }
    /// `fcmgt`
    pub fn fcmgt(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(fp_binary(&FCMGT, d, n, m));
    }
    /// `fcmeq vd, vn, #0.0`
    pub fn fcmeq_zero(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCMEQ0, d, n));
    }
    /// `fcmgt vd, vn, #0.0`
    pub fn fcmgt_zero(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCMGT0, d, n));
    }
    /// `fcmge vd, vn, #0.0`
    pub fn fcmge_zero(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCMGE0, d, n));
    }
    /// `fcmlt vd, vn, #0.0`
    pub fn fcmlt_zero(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCMLT0, d, n));
    }
    /// `faddp`: vector pairwise, or the scalar sum of the two lanes of a `2s`/`2d`.
    pub fn faddp(&mut self, d: Vr, n: Vr, m: Option<Vr>) {
        self.emit(pairwise(&FADDP, 0x7e30_d800, d, n, m));
    }
    /// `fmaxp`: vector pairwise, or the scalar maximum of two lanes.
    pub fn fmaxp(&mut self, d: Vr, n: Vr, m: Option<Vr>) {
        self.emit(pairwise(&FMAXP, 0x7e30_f800, d, n, m));
    }
    /// `fminp`: vector pairwise, or the scalar minimum of two lanes.
    pub fn fminp(&mut self, d: Vr, n: Vr, m: Option<Vr>) {
        self.emit(pairwise(&FMINP, 0x7eb0_f800, d, n, m));
    }
    /// `fmla`, vector or by element.
    pub fn fmla(&mut self, d: Vr, n: Vr, m: Vr) {
        if matches!(m, Vr::E(..)) {
            self.emit(fp_elem(0x0f80_1000, d, n, m));
        } else {
            self.emit(fp_binary(&FMLA, d, n, m));
        }
    }
    /// `fmls`, vector or by element.
    pub fn fmls(&mut self, d: Vr, n: Vr, m: Vr) {
        if matches!(m, Vr::E(..)) {
            self.emit(fp_elem(0x0f80_5000, d, n, m));
        } else {
            self.emit(fp_binary(&FMLS, d, n, m));
        }
    }
    /// `fmadd d, n, m, a` — `a + n·m`, fused, scalar.
    pub fn fmadd(&mut self, d: Vr, n: Vr, m: Vr, a: Vr) {
        self.emit(fma4(0x1f00_0000, d, n, m, a));
    }
    /// `fmsub d, n, m, a` — `a − n·m`, fused, scalar.
    pub fn fmsub(&mut self, d: Vr, n: Vr, m: Vr, a: Vr) {
        self.emit(fma4(0x1f00_8000, d, n, m, a));
    }
    /// `fnmadd d, n, m, a` — `−a − n·m`, fused, scalar.
    pub fn fnmadd(&mut self, d: Vr, n: Vr, m: Vr, a: Vr) {
        self.emit(fma4(0x1f20_0000, d, n, m, a));
    }
    /// `fnmsub d, n, m, a` — `−a + n·m`, fused, scalar.
    pub fn fnmsub(&mut self, d: Vr, n: Vr, m: Vr, a: Vr) {
        self.emit(fma4(0x1f20_8000, d, n, m, a));
    }
    /// `fabs`
    pub fn fabs(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FABS, d, n));
    }
    /// `fneg`
    pub fn fneg(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FNEG, d, n));
    }
    /// `fsqrt`
    pub fn fsqrt(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FSQRT, d, n));
    }
    /// `frecpe`
    pub fn frecpe(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRECPE, d, n));
    }
    /// `frsqrte`
    pub fn frsqrte(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRSQRTE, d, n));
    }
    /// `frintn` (to nearest, ties to even)
    pub fn frintn(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRINTN, d, n));
    }
    /// `frintm` (toward −∞)
    pub fn frintm(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRINTM, d, n));
    }
    /// `frintp` (toward +∞)
    pub fn frintp(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRINTP, d, n));
    }
    /// `frintz` (toward zero)
    pub fn frintz(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FRINTZ, d, n));
    }
    /// `fcvtzs` within the SIMD file (same width), vector or scalar.
    pub fn fcvtzs(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCVTZS, d, n));
    }
    /// `fcvtns` within the SIMD file.
    pub fn fcvtns(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCVTNS, d, n));
    }
    /// `fcvtms` within the SIMD file.
    pub fn fcvtms(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&FCVTMS, d, n));
    }
    /// `scvtf` within the SIMD file.
    pub fn scvtf(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&SCVTF, d, n));
    }
    /// `ucvtf` within the SIMD file.
    pub fn ucvtf(&mut self, d: Vr, n: Vr) {
        self.emit(fp_unary(&UCVTF, d, n));
    }

    /// `fcvt d, n` between scalar half, single and double.
    pub fn fcvt(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, de) = d.scalar()?;
            let (nn, ne) = n.scalar()?;
            check(de != ne, || "fcvt converts between two widths".into())?;
            Ok(0x1e22_4000 | (fp_type(ne)? << 22) | (fp_type(de)? << 15) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `fcvtzs rd, sn|dn` — toward zero, into a general-purpose register.
    pub fn fcvtzs_gp(&mut self, rd: R, n: Vr) {
        self.emit(fp_to_gp(0x1e38_0000, rd, n));
    }
    /// `fcvtms rd, sn|dn` — toward −∞, into a general-purpose register.
    pub fn fcvtms_gp(&mut self, rd: R, n: Vr) {
        self.emit(fp_to_gp(0x1e30_0000, rd, n));
    }
    /// `fcvtns rd, sn|dn` — to nearest even, into a general-purpose register.
    pub fn fcvtns_gp(&mut self, rd: R, n: Vr) {
        self.emit(fp_to_gp(0x1e20_0000, rd, n));
    }
    /// `scvtf sd|dd, rn` — from a signed general-purpose register.
    pub fn scvtf_gp(&mut self, d: Vr, rn: R) {
        let w = (|| {
            let (dd, e) = d.scalar()?;
            Ok((u32::from(rn.is_x()) << 31) | 0x1e22_0000 | (fp_type(e)? << 22) | (rn.zr()? << 5) | dd)
        })();
        self.emit(w);
    }
    /// `fmov d, n` between two scalars of one width.
    pub fn fmov(&mut self, d: Vr, n: Vr) {
        let w = s2(d, n).and_then(|(dd, nn, e)| Ok(0x1e20_4000 | (fp_type(e)? << 22) | (nn << 5) | dd));
        self.emit(w);
    }
    /// `fmov wd, sn` / `fmov xd, dn`
    pub fn fmov_to_gp(&mut self, rd: R, n: Vr) {
        let w = (|| {
            let (nn, e) = n.scalar()?;
            match (rd.is_x(), e) {
                (false, Esz::S) => Ok(0x1e26_0000 | (nn << 5) | rd.zr()?),
                (true, Esz::D) => Ok(0x9e66_0000 | (nn << 5) | rd.zr()?),
                _ => Err(format!("fmov {rd}, {n}: widths differ")),
            }
        })();
        self.emit(w);
    }
    /// `fmov sd, wn` / `fmov dd, xn`
    pub fn fmov_from_gp(&mut self, d: Vr, rn: R) {
        let w = (|| {
            let (dd, e) = d.scalar()?;
            match (rn.is_x(), e) {
                (false, Esz::S) => Ok(0x1e27_0000 | (rn.zr()? << 5) | dd),
                (true, Esz::D) => Ok(0x9e67_0000 | (rn.zr()? << 5) | dd),
                _ => Err(format!("fmov {d}, {rn}: widths differ")),
            }
        })();
        self.emit(w);
    }
    /// `fmov d, #imm` — scalar or vector (`2s`, `4s`, `2d`), when `imm` has an 8-bit encoding.
    pub fn fmov_imm(&mut self, d: Vr, imm: f64) {
        let w = (|| {
            let imm8 = encode_fp8(imm).ok_or_else(|| format!("{imm} has no 8-bit floating-point encoding"))?;
            match d {
                Vr::S(dd, e) => Ok(0x1e20_1000 | (fp_type(e)? << 22) | (imm8 << 13) | u32::from(dd)),
                Vr::V(dd, a) => {
                    let op = fp_arr(a)?;
                    Ok(0x0f00_f400 | q(a) | (op << 29) | ((imm8 >> 5) << 16) | ((imm8 & 31) << 5) | u32::from(dd))
                }
                Vr::E(..) => Err(format!("fmov {d}, #imm: not a register")),
            }
        })();
        self.emit(w);
    }
    /// `fcmp n, m` (scalar), or `fcmp n, #0.0` when `m` is `None`.
    pub fn fcmp(&mut self, n: Vr, m: Option<Vr>) {
        let w = (|| {
            let (nn, e) = n.scalar()?;
            match m {
                Some(m) => {
                    let (mm, em) = m.scalar()?;
                    check(e == em, || "fcmp widths differ".into())?;
                    Ok(0x1e20_2000 | (fp_type(e)? << 22) | (mm << 16) | (nn << 5))
                }
                None => Ok(0x1e20_2008 | (fp_type(e)? << 22) | (nn << 5)),
            }
        })();
        self.emit(w);
    }
    /// `fcsel d, n, m, cond`
    pub fn fcsel(&mut self, d: Vr, n: Vr, m: Vr, cond: Cond) {
        let w = s3(d, n, m).and_then(|(dd, nn, mm, e)| Ok(0x1e20_0c00 | (fp_type(e)? << 22) | (mm << 16) | ((cond as u32) << 12) | (nn << 5) | dd));
        self.emit(w);
    }
    /// `fcvtn vd.4h, vn.4s` (half) or `fcvtn vd.2s, vn.2d`; `fcvtn2` into the upper half.
    pub fn fcvtn(&mut self, d: Vr, n: Vr) {
        self.emit(narrow_fp(0x0e21_6800, d, n));
    }
    /// `fcvtl vd.4s, vn.4h` or `fcvtl vd.2d, vn.2s`; `fcvtl2` from the upper half.
    pub fn fcvtl(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            let sz = match (da, na) {
                (Arr::S4, Arr::H4 | Arr::H8) => 0,
                (Arr::D2, Arr::S2 | Arr::S4) => 1,
                _ => return Err(format!("fcvtl {d}, {n}")),
            };
            Ok(0x0e21_7800 | q(na) | (sz << 22) | (nn << 5) | dd)
        })();
        self.emit(w);
    }

    // ── integer vectors ─────────────────────────────────────────────────

    /// `add` (vector)
    pub fn add_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x0e20_8400, 0b1111, d, n, m));
    }
    /// `sub` (vector)
    pub fn sub_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_8400, 0b1111, d, n, m));
    }
    /// `mul` (vector)
    pub fn mul_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x0e20_9c00, 0b0111, d, n, m));
    }
    /// `cmeq` (vector)
    pub fn cmeq(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_8c00, 0b1111, d, n, m));
    }
    /// `cmgt` (vector, signed)
    pub fn cmgt(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x0e20_3400, 0b1111, d, n, m));
    }
    /// `cmhi` (vector, unsigned)
    pub fn cmhi(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_3400, 0b1111, d, n, m));
    }
    /// `cmhs` (vector, unsigned ≥)
    pub fn cmhs(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_3c00, 0b1111, d, n, m));
    }
    /// `ushl` (vector)
    pub fn ushl(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_4400, 0b1111, d, n, m));
    }
    /// `sshl` (vector)
    pub fn sshl(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x0e20_4400, 0b1111, d, n, m));
    }
    /// `umax` (vector)
    pub fn umax_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(int3(0x2e20_6400, 0b0111, d, n, m));
    }
    /// `and` (vector)
    pub fn and_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x0e20_1c00, d, n, m));
    }
    /// `bic` (vector)
    pub fn bic_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x0e60_1c00, d, n, m));
    }
    /// `orr` (vector)
    pub fn orr_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x0ea0_1c00, d, n, m));
    }
    /// `orn` (vector)
    pub fn orn_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x0ee0_1c00, d, n, m));
    }
    /// `eor` (vector)
    pub fn eor_v(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x2e20_1c00, d, n, m));
    }
    /// `bsl` — `d = (d & n) | (!d & m)`
    pub fn bsl(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x2e60_1c00, d, n, m));
    }
    /// `bit` — insert `n` where `m` is set
    pub fn bit(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x2ea0_1c00, d, n, m));
    }
    /// `bif` — insert `n` where `m` is clear
    pub fn bif(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(logic3(0x2ee0_1c00, d, n, m));
    }
    /// `mov vd.<16b|8b>, vn` — `orr` of a register with itself.
    pub fn mov_v(&mut self, d: Vr, n: Vr) {
        self.orr_v(d, n, n);
    }
    /// `not` / `mvn` (vector)
    pub fn not_v(&mut self, d: Vr, n: Vr) {
        let w = v2(d, n).and_then(|(dd, nn, a)| {
            check(matches!(a, Arr::B8 | Arr::B16), || "not takes 8b or 16b".into())?;
            Ok(0x2e20_5800 | q(a) | (nn << 5) | dd)
        });
        self.emit(w);
    }
    /// `neg` (vector)
    pub fn neg_v(&mut self, d: Vr, n: Vr) {
        let w = v2(d, n).map(|(dd, nn, a)| 0x2e20_b800 | q(a) | (a.size() << 22) | (nn << 5) | dd);
        self.emit(w);
    }
    /// `xtn vd.<narrow>, vn.<wide>`; `xtn2` into the upper half.
    pub fn xtn(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            check(na.q() && na.size() == da.size() + 1, || format!("xtn {d}, {n}"))?;
            Ok(0x0e21_2800 | q(da) | (da.size() << 22) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `shll vd.<wide>, vn.<narrow>, #esize` (`shll2` from the upper half):
    /// the widening shift by the lane width — bf16 to f32 in one instruction.
    pub fn shll(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            check(da.q() && da.size() == na.size() + 1 && na.size() < 3, || format!("shll {d}, {n}"))?;
            Ok(0x2e21_3800 | q(na) | (na.size() << 22) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `ushll vd.<wide>, vn.<narrow>, #shift` (`ushll2` from the upper half).
    pub fn ushll(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(widen(0x2f00_a400, d, n, shift));
    }
    /// `sshll vd.<wide>, vn.<narrow>, #shift` (`sshll2` from the upper half).
    pub fn sshll(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(widen(0x0f00_a400, d, n, shift));
    }
    /// `shl vd, vn, #shift`
    pub fn shl(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x0f00_5400, d, n, shift, true));
    }
    /// `sli vd, vn, #shift` — shift left and insert.
    pub fn sli(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x2f00_5400, d, n, shift, true));
    }
    /// `ushr vd, vn, #shift`
    pub fn ushr(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x2f00_0400, d, n, shift, false));
    }
    /// `sshr vd, vn, #shift`
    pub fn sshr(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x0f00_0400, d, n, shift, false));
    }
    /// `sri vd, vn, #shift` — shift right and insert.
    pub fn sri(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x2f00_4400, d, n, shift, false));
    }
    /// `usra vd, vn, #shift` — unsigned shift right and accumulate.
    pub fn usra(&mut self, d: Vr, n: Vr, shift: u32) {
        self.emit(shift_same(0x2f00_1400, d, n, shift, false));
    }
    /// `shrn vd.<narrow>, vn.<wide>, #shift` (`shrn2` into the upper half).
    pub fn shrn(&mut self, d: Vr, n: Vr, shift: u32) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            check(na.q() && na.size() == da.size() + 1, || format!("shrn {d}, {n}"))?;
            let imm = shift_imm(da.size(), shift, false)?;
            Ok(0x0f00_8400 | q(da) | (imm << 16) | (nn << 5) | dd)
        })();
        self.emit(w);
    }

    // ── moves between lanes and files ───────────────────────────────────

    /// `dup vd.<arr>, vn.<e>[i]` or `dup vd.<arr>, rn`.
    pub fn dup(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, a) = d.vec()?;
            let (nn, e, i) = n.elem()?;
            check(e.size() == a.size() && e != Esz::H2, || format!("dup {d}, {n}: widths differ"))?;
            check(a != Arr::D1, || "dup into 1d".into())?;
            Ok(0x0e00_0400 | q(a) | (imm5(e, i)? << 16) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `dup vd.<arr>, rn`
    pub fn dup_gp(&mut self, d: Vr, rn: R) {
        let w = (|| {
            let (dd, a) = d.vec()?;
            check(a != Arr::D1 && (a.size() == 3) == rn.is_x(), || format!("dup {d}, {rn}"))?;
            let e = [Esz::B, Esz::H, Esz::S, Esz::D][a.size() as usize];
            Ok(0x0e00_0c00 | q(a) | (imm5(e, 0)? << 16) | (rn.zr()? << 5) | dd)
        })();
        self.emit(w);
    }
    /// `mov <scalar>, vn.<e>[i]` — one lane into a scalar view.
    pub fn dup_scalar(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, de) = d.scalar()?;
            let (nn, e, i) = n.elem()?;
            check(de == e, || format!("mov {d}, {n}: widths differ"))?;
            Ok(0x5e00_0400 | (imm5(e, i)? << 16) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `ins vd.<e>[i], vn.<e>[j]` (`mov` of one lane)
    pub fn ins(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, de, di) = d.elem()?;
            let (nn, ne, ni) = n.elem()?;
            check(de == ne, || format!("ins {d}, {n}: widths differ"))?;
            let size = de.size();
            Ok(0x6e00_0400 | (imm5(de, di)? << 16) | ((ni << size) << 11) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `ins vd.<e>[i], rn` (`mov` from a general-purpose register)
    pub fn ins_gp(&mut self, d: Vr, rn: R) {
        let w = (|| {
            let (dd, e, i) = d.elem()?;
            check((e == Esz::D) == rn.is_x(), || format!("ins {d}, {rn}: widths differ"))?;
            Ok(0x4e00_1c00 | (imm5(e, i)? << 16) | (rn.zr()? << 5) | dd)
        })();
        self.emit(w);
    }
    /// `umov rd, vn.<e>[i]` (`mov` for `s` and `d` lanes)
    pub fn umov(&mut self, rd: R, n: Vr) {
        let w = (|| {
            let (nn, e, i) = n.elem()?;
            check((e == Esz::D) == rd.is_x(), || format!("umov {rd}, {n}: widths differ"))?;
            Ok(0x0e00_3c00 | (u32::from(rd.is_x()) << 30) | (imm5(e, i)? << 16) | (nn << 5) | rd.zr()?)
        })();
        self.emit(w);
    }
    /// `movi`: a vector (or `d` scalar) of a repeated constant. Accepts the
    /// byte, shifted 16-bit, shifted 32-bit (and their `mvni` complements) and
    /// 64-bit byte-mask forms; `value` is one lane.
    pub fn movi(&mut self, d: Vr, value: u64) {
        let w = (|| {
            let (dd, size, qbit) = match d {
                Vr::V(n, a) => (u32::from(n), a.size(), q(a)),
                Vr::S(n, Esz::D) => (u32::from(n), 3, 0),
                other => return Err(format!("movi {other}")),
            };
            let enc = |op: u32, cmode: u32, imm8: u32| {
                0x0f00_0400 | qbit | (op << 29) | ((imm8 >> 5) << 16) | (cmode << 12) | ((imm8 & 31) << 5) | dd
            };
            match size {
                0 => {
                    check(value < 256, || format!("{value:#x} is not a byte"))?;
                    Ok(enc(0, 0b1110, value as u32))
                }
                1 => {
                    let v = value as u32 & 0xffff;
                    check(value < 0x10000, || format!("{value:#x} is not a half"))?;
                    for (shift, cmode) in [(0, 0b1000), (8, 0b1010)] {
                        if v & !(0xff << shift) == 0 {
                            return Ok(enc(0, cmode, v >> shift));
                        }
                        let inv = !v & 0xffff;
                        if inv & !(0xff << shift) == 0 {
                            return Ok(enc(1, cmode, inv >> shift));
                        }
                    }
                    Err(format!("{value:#x} has no 16-bit movi form"))
                }
                2 => {
                    check(value <= u64::from(u32::MAX), || format!("{value:#x} is not a word"))?;
                    let v = value as u32;
                    for (shift, cmode) in [(0, 0b0000), (8, 0b0010), (16, 0b0100), (24, 0b0110)] {
                        if v & !(0xff << shift) == 0 {
                            return Ok(enc(0, cmode, v >> shift));
                        }
                        let inv = !v;
                        if inv & !(0xff << shift) == 0 {
                            return Ok(enc(1, cmode, inv >> shift));
                        }
                    }
                    Err(format!("{value:#x} has no 32-bit movi form"))
                }
                _ => {
                    let mut imm8 = 0;
                    for byte in 0..8 {
                        match (value >> (8 * byte)) & 0xff {
                            0 => {}
                            0xff => imm8 |= 1 << byte,
                            _ => return Err(format!("{value:#x} is not a byte mask")),
                        }
                    }
                    Ok(enc(1, 0b1110, imm8))
                }
            }
        })();
        self.emit(w);
    }
    /// `ext vd, vn, vm, #bytes`
    pub fn ext(&mut self, d: Vr, n: Vr, m: Vr, bytes: u32) {
        let w = v3(d, n, m).and_then(|(dd, nn, mm, a)| {
            check(matches!(a, Arr::B8 | Arr::B16) && bytes < a.lanes(), || format!("ext {d}: index {bytes}"))?;
            Ok(0x2e00_0000 | q(a) | (mm << 16) | (bytes << 11) | (nn << 5) | dd)
        });
        self.emit(w);
    }
    /// `zip1`
    pub fn zip1(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_3800, d, n, m));
    }
    /// `zip2`
    pub fn zip2(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_7800, d, n, m));
    }
    /// `uzp1`
    pub fn uzp1(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_1800, d, n, m));
    }
    /// `uzp2`
    pub fn uzp2(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_5800, d, n, m));
    }
    /// `trn1`
    pub fn trn1(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_2800, d, n, m));
    }
    /// `trn2`
    pub fn trn2(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(permute(0x0e00_6800, d, n, m));
    }
    /// `tbl vd, {vn..vn+count-1}.16b, vm`
    pub fn tbl(&mut self, d: Vr, n: Vr, count: u32, m: Vr) {
        let w = (|| {
            let (dd, a) = d.vec()?;
            let (nn, na) = n.vec()?;
            let (mm, ma) = m.vec()?;
            check(matches!(a, Arr::B8 | Arr::B16) && na == Arr::B16 && ma == a && (1..=4).contains(&count), || "tbl operands".into())?;
            Ok(0x0e00_0000 | q(a) | (mm << 16) | ((count - 1) << 13) | (nn << 5) | dd)
        })();
        self.emit(w);
    }

    // ── reductions ──────────────────────────────────────────────────────

    /// `fmaxv sd, vn.4s`
    pub fn fmaxv(&mut self, d: Vr, n: Vr) {
        self.emit(across_fp(0x2e30_f800, d, n));
    }
    /// `fminv sd, vn.4s`
    pub fn fminv(&mut self, d: Vr, n: Vr) {
        self.emit(across_fp(0x2eb0_f800, d, n));
    }
    /// `fmaxnmv sd, vn.4s`
    pub fn fmaxnmv(&mut self, d: Vr, n: Vr) {
        self.emit(across_fp(0x2e30_c800, d, n));
    }
    /// `addv <scalar>, vn.<arr>`
    pub fn addv(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, e) = d.scalar()?;
            let (nn, a) = n.vec()?;
            check(e.size() == a.size() && a.size() < 3 && a != Arr::S2, || format!("addv {d}, {n}"))?;
            Ok(0x0e31_b800 | q(a) | (a.size() << 22) | (nn << 5) | dd)
        })();
        self.emit(w);
    }

    // ── BF16 ────────────────────────────────────────────────────────────

    /// `bfdot vd.<2s|4s>, vn.<4h|8h>, vm.<4h|8h>` or, by element,
    /// `bfdot vd.4s, vn.8h, vm.2h[i]`: each f32 lane accumulates the dot
    /// product of a pair of bf16.
    pub fn bfdot(&mut self, d: Vr, n: Vr, m: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            let pair = match (da, na) {
                (Arr::S2, Arr::H4) | (Arr::S4, Arr::H8) => q(da),
                _ => return Err(format!("bfdot {d}, {n}")),
            };
            match m {
                Vr::V(mm, ma) => {
                    check(ma == na, || format!("bfdot {n}, {m}"))?;
                    Ok(0x2e40_fc00 | pair | (u32::from(mm) << 16) | (nn << 5) | dd)
                }
                Vr::E(mm, Esz::H2, i) => {
                    check(i < 4, || format!("bfdot index {i} beyond 3"))?;
                    let i = u32::from(i);
                    Ok(0x0f40_f000 | pair | ((i & 1) << 21) | (u32::from(mm) << 16) | ((i >> 1) << 11) | (nn << 5) | dd)
                }
                other => Err(format!("bfdot third operand {other}")),
            }
        })();
        self.emit(w);
    }
    /// `bfmmla vd.4s, vn.8h, vm.8h`: a 2×4 by 4×2 bf16 product accumulated
    /// into the 2×2 f32 matrix of `vd`.
    pub fn bfmmla(&mut self, d: Vr, n: Vr, m: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            let (mm, ma) = m.vec()?;
            check(da == Arr::S4 && na == Arr::H8 && ma == Arr::H8, || "bfmmla v.4s, v.8h, v.8h".into())?;
            Ok(0x6e40_ec00 | (mm << 16) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `bfmlalb vd.4s, vn.8h, vm.8h|vm.h[i]`: the even bf16 lanes widened and
    /// fused-multiply-added into f32.
    pub fn bfmlalb(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(bfmlal(false, d, n, m));
    }
    /// `bfmlalt`: the odd bf16 lanes.
    pub fn bfmlalt(&mut self, d: Vr, n: Vr, m: Vr) {
        self.emit(bfmlal(true, d, n, m));
    }
    /// `bfcvtn vd.4h, vn.4s` or `bfcvtn2 vd.8h, vn.4s`: f32 to bf16, rounded per FPCR.
    pub fn bfcvtn(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, da) = d.vec()?;
            let (nn, na) = n.vec()?;
            check(na == Arr::S4 && matches!(da, Arr::H4 | Arr::H8), || format!("bfcvtn {d}, {n}"))?;
            Ok(0x0ea1_6800 | q(da) | (nn << 5) | dd)
        })();
        self.emit(w);
    }
    /// `bfcvt hd, sn`: one f32 to bf16.
    pub fn bfcvt(&mut self, d: Vr, n: Vr) {
        let w = (|| {
            let (dd, de) = d.scalar()?;
            let (nn, ne) = n.scalar()?;
            check(de == Esz::H && ne == Esz::S, || "bfcvt hd, sn".into())?;
            Ok(0x1e63_4000 | (nn << 5) | dd)
        })();
        self.emit(w);
    }

    // ── structure loads and stores ──────────────────────────────────────

    /// `ld1 {vt..vt+count-1}.<arr>, mem`: `[xn]`, `[xn], #bytes` or `[xn], xm`.
    pub fn ld1(&mut self, t: Vr, count: u32, m: Mem) {
        self.emit(multiple(true, t, count, m));
    }
    /// `st1 {vt..vt+count-1}.<arr>, mem`
    pub fn st1(&mut self, t: Vr, count: u32, m: Mem) {
        self.emit(multiple(false, t, count, m));
    }
    /// `ld1r {vt.<arr>}, mem`: one element, replicated.
    pub fn ld1r(&mut self, t: Vr, m: Mem) {
        let w = (|| {
            let (tt, a) = t.vec()?;
            let bytes = 1i64 << a.size();
            let (rn, post) = match m {
                Mem::Off(rn, 0) => (rn, None),
                Mem::Post(rn, off) => {
                    check(off == bytes, || format!("ld1r post-index must be {bytes}"))?;
                    (rn, Some(31))
                }
                Mem::PostReg(rn, rm) => (rn, Some(rm.zr()?)),
                other => return Err(format!("ld1r cannot address {other:?}")),
            };
            let base = match post {
                None => 0x0d40_c000,
                Some(r) => 0x0dc0_c000 | (r << 16),
            };
            Ok(base | q(a) | (a.size() << 10) | (rn.sp()? << 5) | tt)
        })();
        self.emit(w);
    }
    /// `ld1 {vt.<e>}[i], mem` — one lane.
    pub fn ld1_lane(&mut self, t: Vr, m: Mem) {
        self.emit(lane(true, t, m));
    }
    /// `st1 {vt.<e>}[i], mem` — one lane.
    pub fn st1_lane(&mut self, t: Vr, m: Mem) {
        self.emit(lane(false, t, m));
    }
}

fn pairwise(op: &Fp2, scalar: u32, d: Vr, n: Vr, m: Option<Vr>) -> Result<u32, String> {
    match m {
        Some(m) => fp_binary(op, d, n, m),
        None => {
            let (dd, e) = d.scalar()?;
            let (nn, a) = n.vec()?;
            let sz = match (e, a) {
                (Esz::S, Arr::S2) => 0,
                (Esz::D, Arr::D2) => 1,
                _ => return Err(format!("pairwise {d}, {n}")),
            };
            Ok(scalar | (sz << 22) | (nn << 5) | dd)
        }
    }
}

fn fma4(base: u32, d: Vr, n: Vr, m: Vr, a: Vr) -> Result<u32, String> {
    let (dd, nn, mm, e) = s3(d, n, m)?;
    let (aa, ea) = a.scalar()?;
    check(e == ea, || "fused operand widths differ".into())?;
    Ok(base | (fp_type(e)? << 22) | (mm << 16) | (aa << 10) | (nn << 5) | dd)
}

fn fp_to_gp(base: u32, rd: R, n: Vr) -> Result<u32, String> {
    let (nn, e) = n.scalar()?;
    Ok((u32::from(rd.is_x()) << 31) | base | (fp_type(e)? << 22) | (nn << 5) | rd.zr()?)
}

fn narrow_fp(base: u32, d: Vr, n: Vr) -> Result<u32, String> {
    let (dd, da) = d.vec()?;
    let (nn, na) = n.vec()?;
    let sz = match (da, na) {
        (Arr::H4 | Arr::H8, Arr::S4) => 0,
        (Arr::S2 | Arr::S4, Arr::D2) => 1,
        _ => return Err(format!("narrowing {d}, {n}")),
    };
    Ok(base | q(da) | (sz << 22) | (nn << 5) | dd)
}

fn widen(base: u32, d: Vr, n: Vr, shift: u32) -> Result<u32, String> {
    let (dd, da) = d.vec()?;
    let (nn, na) = n.vec()?;
    check(da.q() && da.size() == na.size() + 1, || format!("widening {d}, {n}"))?;
    let imm = shift_imm(na.size(), shift, true)?;
    Ok(base | q(na) | (imm << 16) | (nn << 5) | dd)
}

fn shift_same(base: u32, d: Vr, n: Vr, shift: u32, left: bool) -> Result<u32, String> {
    let (dd, nn, a) = v2(d, n)?;
    check(a != Arr::D1, || "1d shifts use the scalar form".into())?;
    let imm = shift_imm(a.size(), shift, left)?;
    Ok(base | q(a) | (imm << 16) | (nn << 5) | dd)
}

fn permute(base: u32, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    let (dd, nn, mm, a) = v3(d, n, m)?;
    check(a != Arr::D1, || "permutations of 1d".into())?;
    Ok(base | q(a) | (a.size() << 22) | (mm << 16) | (nn << 5) | dd)
}

fn across_fp(base: u32, d: Vr, n: Vr) -> Result<u32, String> {
    let (dd, e) = d.scalar()?;
    let (nn, a) = n.vec()?;
    check(e == Esz::S && a == Arr::S4, || "reductions over 4s into s".into())?;
    Ok(base | q(a) | (nn << 5) | dd)
}

fn bfmlal(top: bool, d: Vr, n: Vr, m: Vr) -> Result<u32, String> {
    let (dd, da) = d.vec()?;
    let (nn, na) = n.vec()?;
    check(da == Arr::S4 && na == Arr::H8, || "bfmlal[bt] v.4s, v.8h, …".into())?;
    let t = u32::from(top) << 30;
    match m {
        Vr::V(mm, Arr::H8) => Ok(0x2ec0_fc00 | t | (u32::from(mm) << 16) | (nn << 5) | dd),
        Vr::E(mm, Esz::H, i) => {
            check(mm < 16 && i < 8, || "bfmlal by element: v0..v15, index 0..7".into())?;
            let i = u32::from(i);
            Ok(0x0fc0_f000 | t | (((i >> 1) & 1) << 21) | ((i & 1) << 20) | (u32::from(mm) << 16) | ((i >> 2) << 11) | (nn << 5) | dd)
        }
        other => Err(format!("bfmlal third operand {other}")),
    }
}

fn multiple(load: bool, t: Vr, count: u32, m: Mem) -> Result<u32, String> {
    let (tt, a) = t.vec()?;
    let opcode = match count {
        1 => 0b0111,
        2 => 0b1010,
        3 => 0b0110,
        4 => 0b0010,
        _ => return Err(format!("ld1/st1 of {count} registers")),
    };
    let bytes = i64::from(count) * if a.q() { 16 } else { 8 };
    let (rn, post) = match m {
        Mem::Off(rn, 0) => (rn, None),
        Mem::Post(rn, off) => {
            check(off == bytes, || format!("post-index must be {bytes}"))?;
            (rn, Some(31))
        }
        Mem::PostReg(rn, rm) => (rn, Some(rm.zr()?)),
        other => return Err(format!("ld1/st1 cannot address {other:?}")),
    };
    let base = match post {
        None => 0x0c00_0000,
        Some(r) => 0x0c80_0000 | (r << 16),
    };
    Ok(base | q(a) | (u32::from(load) << 22) | (opcode << 12) | (a.size() << 10) | (rn.sp()? << 5) | tt)
}

fn lane(load: bool, t: Vr, m: Mem) -> Result<u32, String> {
    let (tt, e, i) = t.elem()?;
    let (opcode, s, size, qb, bytes) = match e {
        Esz::B => (0b000, (i >> 3) & 1, i & 3, i >> 4, 1),
        Esz::H => (0b010, (i >> 1) & 1, (i & 1) << 1, i >> 2, 2),
        Esz::S => (0b100, i & 1, 0, i >> 1, 4),
        Esz::D => (0b100, 0, 1, i, 8),
        other => return Err(format!("{other:?} lanes")),
    };
    let (qb, s, size) = match e {
        Esz::B => (i >> 3, (i >> 2) & 1, i & 3),
        _ => (qb, s, size),
    };
    check(qb < 2, || format!("lane index {i} out of range"))?;
    let (rn, post) = match m {
        Mem::Off(rn, 0) => (rn, None),
        Mem::Post(rn, off) => {
            check(off == bytes, || format!("post-index must be {bytes}"))?;
            (rn, Some(31))
        }
        Mem::PostReg(rn, rm) => (rn, Some(rm.zr()?)),
        other => return Err(format!("lane access cannot address {other:?}")),
    };
    let base = match post {
        None => 0x0d00_0000,
        Some(r) => 0x0d80_0000 | (r << 16),
    };
    Ok(base | (qb << 30) | (u32::from(load) << 22) | (opcode << 13) | (s << 12) | (size << 10) | (rn.sp()? << 5) | tt)
}
