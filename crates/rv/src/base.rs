//! The scalar set: RV64I and M, the atomics (A), the bit-manipulation
//! extensions Zba, Zbb and Zbs, fences and the pause hint, the CSR
//! instructions, and the floating-point files F and D with the half and
//! bf16 moves and conversions (Zfh loads, Zfbfmin).

use crate::asm::{check, Asm, Fix, Label};
use crate::reg::{Rm, F, X, RA, ZERO};

pub(crate) const OP_LOAD: u32 = 0x03;
pub(crate) const OP_LOAD_FP: u32 = 0x07;
pub(crate) const OP_MISC_MEM: u32 = 0x0f;
pub(crate) const OP_IMM: u32 = 0x13;
pub(crate) const OP_AUIPC: u32 = 0x17;
pub(crate) const OP_IMM_32: u32 = 0x1b;
pub(crate) const OP_STORE: u32 = 0x23;
pub(crate) const OP_STORE_FP: u32 = 0x27;
pub(crate) const OP_AMO: u32 = 0x2f;
pub(crate) const OP_OP: u32 = 0x33;
pub(crate) const OP_LUI: u32 = 0x37;
pub(crate) const OP_OP_32: u32 = 0x3b;
pub(crate) const OP_MADD: u32 = 0x43;
pub(crate) const OP_MSUB: u32 = 0x47;
pub(crate) const OP_NMSUB: u32 = 0x4b;
pub(crate) const OP_NMADD: u32 = 0x4f;
pub(crate) const OP_FP: u32 = 0x53;
pub(crate) const OP_V: u32 = 0x57;
pub(crate) const OP_BRANCH: u32 = 0x63;
pub(crate) const OP_JALR: u32 = 0x67;
pub(crate) const OP_JAL: u32 = 0x6f;
pub(crate) const OP_SYSTEM: u32 = 0x73;

/// The memory order of an atomic: its `aq` and `rl` bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Order {
    /// Neither.
    Relaxed,
    /// `.aq`.
    Acquire,
    /// `.rl`.
    Release,
    /// `.aqrl`: sequentially consistent.
    AcqRel,
}

impl Order {
    pub(crate) fn bits(self) -> u32 {
        match self {
            Order::Relaxed => 0,
            Order::Acquire => 2,
            Order::Release => 1,
            Order::AcqRel => 3,
        }
    }
    pub(crate) fn suffix(bits: u32) -> &'static str {
        ["", ".rl", ".aq", ".aqrl"][bits as usize & 3]
    }
}

/// The comparison of a conditional branch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Cond {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Signed less than.
    Lt,
    /// Signed greater or equal.
    Ge,
    /// Unsigned less than.
    Ltu,
    /// Unsigned greater or equal.
    Geu,
}

impl Cond {
    pub(crate) fn funct3(self) -> u32 {
        match self {
            Cond::Eq => 0,
            Cond::Ne => 1,
            Cond::Lt => 4,
            Cond::Ge => 5,
            Cond::Ltu => 6,
            Cond::Geu => 7,
        }
    }
    /// The opposite comparison.
    pub fn invert(self) -> Cond {
        match self {
            Cond::Eq => Cond::Ne,
            Cond::Ne => Cond::Eq,
            Cond::Lt => Cond::Ge,
            Cond::Ge => Cond::Lt,
            Cond::Ltu => Cond::Geu,
            Cond::Geu => Cond::Ltu,
        }
    }
}

/// The predecessor or successor set of a `fence`: device input and output,
/// memory reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Fence(pub u32);

impl Fence {
    /// Memory reads.
    pub const R: Fence = Fence(2);
    /// Memory writes.
    pub const W: Fence = Fence(1);
    /// Memory reads and writes.
    pub const RW: Fence = Fence(3);
    /// Everything.
    pub const IORW: Fence = Fence(15);
    pub(crate) fn name(bits: u32) -> String {
        let mut s = String::new();
        for (bit, c) in [(8, 'i'), (4, 'o'), (2, 'r'), (1, 'w')] {
            if bits & bit != 0 {
                s.push(c);
            }
        }
        if s.is_empty() {
            "0".into()
        } else {
            s
        }
    }
}

/// A control and status register.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Csr(pub u16);

impl Csr {
    /// Accrued floating-point exceptions.
    pub const FFLAGS: Csr = Csr(0x001);
    /// Dynamic rounding mode.
    pub const FRM: Csr = Csr(0x002);
    /// `frm` and `fflags` together.
    pub const FCSR: Csr = Csr(0x003);
    /// The first element a vector instruction starts from.
    pub const VSTART: Csr = Csr(0x008);
    /// Fixed-point rounding mode.
    pub const VXRM: Csr = Csr(0x00a);
    /// Cycle counter.
    pub const CYCLE: Csr = Csr(0xc00);
    /// Wall-clock timer.
    pub const TIME: Csr = Csr(0xc01);
    /// Retired instructions.
    pub const INSTRET: Csr = Csr(0xc02);
    /// Vector length.
    pub const VL: Csr = Csr(0xc20);
    /// Vector type.
    pub const VTYPE: Csr = Csr(0xc21);
    /// Bytes per vector register (VLEN / 8).
    pub const VLENB: Csr = Csr(0xc22);
    pub(crate) fn name(n: u32) -> String {
        match n {
            0x001 => "fflags".into(),
            0x002 => "frm".into(),
            0x003 => "fcsr".into(),
            0x008 => "vstart".into(),
            0x00a => "vxrm".into(),
            0xc00 => "cycle".into(),
            0xc01 => "time".into(),
            0xc02 => "instret".into(),
            0xc20 => "vl".into(),
            0xc21 => "vtype".into(),
            0xc22 => "vlenb".into(),
            n => format!("{n:#x}"),
        }
    }
}

// ── formats ─────────────────────────────────────────────────────────────

pub(crate) fn r(f7: u32, rs2: u32, rs1: u32, f3: u32, rd: u32, op: u32) -> u32 {
    f7 << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op
}

pub(crate) fn i(imm: i64, rs1: u32, f3: u32, rd: u32, op: u32) -> Result<u32, String> {
    check((-2048..2048).contains(&imm), || format!("immediate {imm} is not a signed 12-bit value"))?;
    Ok(((imm as u32) & 0xfff) << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op)
}

fn s(imm: i64, rs2: u32, rs1: u32, f3: u32, op: u32) -> Result<u32, String> {
    check((-2048..2048).contains(&imm), || format!("offset {imm} is not a signed 12-bit value"))?;
    let imm = imm as u32;
    Ok((imm >> 5 & 0x7f) << 25 | rs2 << 20 | rs1 << 15 | f3 << 12 | (imm & 0x1f) << 7 | op)
}

fn shift(sh: u32, top6: u32, rs1: u32, f3: u32, rd: u32, op: u32, max: u32) -> Result<u32, String> {
    check(sh < max, || format!("shift {sh} is not below {max}"))?;
    Ok(top6 << 26 | sh << 20 | rs1 << 15 | f3 << 12 | rd << 7 | op)
}

fn xn(r: X) -> u32 {
    u32::from(r.0)
}
fn fnum(r: F) -> u32 {
    u32::from(r.0)
}

macro_rules! op_r {
    ($($(#[$doc:meta])* $name:ident = ($f7:expr, $f3:expr, $op:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, rs1: X, rs2: X) {
                self.emit(Ok(r($f7, xn(rs2), xn(rs1), $f3, xn(rd), $op)));
            } )*
        }
    };
}

op_r! {
    /// `add rd, rs1, rs2`.
    add = (0, 0, OP_OP);
    /// `sub rd, rs1, rs2`.
    sub = (0x20, 0, OP_OP);
    /// `sll rd, rs1, rs2`.
    sll = (0, 1, OP_OP);
    /// `slt rd, rs1, rs2`.
    slt = (0, 2, OP_OP);
    /// `sltu rd, rs1, rs2`.
    sltu = (0, 3, OP_OP);
    /// `xor rd, rs1, rs2`.
    xor = (0, 4, OP_OP);
    /// `srl rd, rs1, rs2`.
    srl = (0, 5, OP_OP);
    /// `sra rd, rs1, rs2`.
    sra = (0x20, 5, OP_OP);
    /// `or rd, rs1, rs2`.
    or = (0, 6, OP_OP);
    /// `and rd, rs1, rs2`.
    and = (0, 7, OP_OP);
    /// `addw rd, rs1, rs2`.
    addw = (0, 0, OP_OP_32);
    /// `subw rd, rs1, rs2`.
    subw = (0x20, 0, OP_OP_32);
    /// `sllw rd, rs1, rs2`.
    sllw = (0, 1, OP_OP_32);
    /// `srlw rd, rs1, rs2`.
    srlw = (0, 5, OP_OP_32);
    /// `sraw rd, rs1, rs2`.
    sraw = (0x20, 5, OP_OP_32);
    /// `mul rd, rs1, rs2`.
    mul = (1, 0, OP_OP);
    /// `mulh rd, rs1, rs2`.
    mulh = (1, 1, OP_OP);
    /// `mulhsu rd, rs1, rs2`.
    mulhsu = (1, 2, OP_OP);
    /// `mulhu rd, rs1, rs2`.
    mulhu = (1, 3, OP_OP);
    /// `div rd, rs1, rs2`.
    div = (1, 4, OP_OP);
    /// `divu rd, rs1, rs2`.
    divu = (1, 5, OP_OP);
    /// `rem rd, rs1, rs2`.
    rem = (1, 6, OP_OP);
    /// `remu rd, rs1, rs2`.
    remu = (1, 7, OP_OP);
    /// `mulw rd, rs1, rs2`.
    mulw = (1, 0, OP_OP_32);
    /// `divw rd, rs1, rs2`.
    divw = (1, 4, OP_OP_32);
    /// `divuw rd, rs1, rs2`.
    divuw = (1, 5, OP_OP_32);
    /// `remw rd, rs1, rs2`.
    remw = (1, 6, OP_OP_32);
    /// `remuw rd, rs1, rs2`.
    remuw = (1, 7, OP_OP_32);
    /// `sh1add rd, rs1, rs2`: `rs2 + (rs1 << 1)` (Zba).
    sh1add = (0x10, 2, OP_OP);
    /// `sh2add rd, rs1, rs2`: `rs2 + (rs1 << 2)` (Zba).
    sh2add = (0x10, 4, OP_OP);
    /// `sh3add rd, rs1, rs2`: `rs2 + (rs1 << 3)` (Zba).
    sh3add = (0x10, 6, OP_OP);
    /// `add.uw rd, rs1, rs2`: `rs2 + zext32(rs1)` (Zba).
    add_uw = (0x04, 0, OP_OP_32);
    /// `sh1add.uw rd, rs1, rs2` (Zba).
    sh1add_uw = (0x10, 2, OP_OP_32);
    /// `sh2add.uw rd, rs1, rs2` (Zba).
    sh2add_uw = (0x10, 4, OP_OP_32);
    /// `sh3add.uw rd, rs1, rs2` (Zba).
    sh3add_uw = (0x10, 6, OP_OP_32);
    /// `andn rd, rs1, rs2` (Zbb).
    andn = (0x20, 7, OP_OP);
    /// `orn rd, rs1, rs2` (Zbb).
    orn = (0x20, 6, OP_OP);
    /// `xnor rd, rs1, rs2` (Zbb).
    xnor = (0x20, 4, OP_OP);
    /// `min rd, rs1, rs2` (Zbb).
    min = (0x05, 4, OP_OP);
    /// `minu rd, rs1, rs2` (Zbb).
    minu = (0x05, 5, OP_OP);
    /// `max rd, rs1, rs2` (Zbb).
    max = (0x05, 6, OP_OP);
    /// `maxu rd, rs1, rs2` (Zbb).
    maxu = (0x05, 7, OP_OP);
    /// `rol rd, rs1, rs2` (Zbb).
    rol = (0x30, 1, OP_OP);
    /// `ror rd, rs1, rs2` (Zbb).
    ror = (0x30, 5, OP_OP);
    /// `rolw rd, rs1, rs2` (Zbb).
    rolw = (0x30, 1, OP_OP_32);
    /// `rorw rd, rs1, rs2` (Zbb).
    rorw = (0x30, 5, OP_OP_32);
    /// `bset rd, rs1, rs2` (Zbs).
    bset = (0x14, 1, OP_OP);
    /// `bclr rd, rs1, rs2` (Zbs).
    bclr = (0x24, 1, OP_OP);
    /// `binv rd, rs1, rs2` (Zbs).
    binv = (0x34, 1, OP_OP);
    /// `bext rd, rs1, rs2` (Zbs).
    bext = (0x24, 5, OP_OP);
}

macro_rules! op_i {
    ($($(#[$doc:meta])* $name:ident = ($f3:expr, $op:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, rs1: X, imm: i64) {
                self.emit(i(imm, xn(rs1), $f3, xn(rd), $op));
            } )*
        }
    };
}

op_i! {
    /// `addi rd, rs1, imm`.
    addi = (0, OP_IMM);
    /// `slti rd, rs1, imm`.
    slti = (2, OP_IMM);
    /// `sltiu rd, rs1, imm`.
    sltiu = (3, OP_IMM);
    /// `xori rd, rs1, imm`.
    xori = (4, OP_IMM);
    /// `ori rd, rs1, imm`.
    ori = (6, OP_IMM);
    /// `andi rd, rs1, imm`.
    andi = (7, OP_IMM);
    /// `addiw rd, rs1, imm`.
    addiw = (0, OP_IMM_32);
}

macro_rules! op_shift {
    ($($(#[$doc:meta])* $name:ident = ($top6:expr, $f3:expr, $op:expr, $max:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, rs1: X, sh: u32) {
                self.emit(shift(sh, $top6, xn(rs1), $f3, xn(rd), $op, $max));
            } )*
        }
    };
}

op_shift! {
    /// `slli rd, rs1, sh`.
    slli = (0, 1, OP_IMM, 64);
    /// `srli rd, rs1, sh`.
    srli = (0, 5, OP_IMM, 64);
    /// `srai rd, rs1, sh`.
    srai = (0b010000, 5, OP_IMM, 64);
    /// `slliw rd, rs1, sh`.
    slliw = (0, 1, OP_IMM_32, 32);
    /// `srliw rd, rs1, sh`.
    srliw = (0, 5, OP_IMM_32, 32);
    /// `sraiw rd, rs1, sh`.
    sraiw = (0b010000, 5, OP_IMM_32, 32);
    /// `slli.uw rd, rs1, sh`: `zext32(rs1) << sh` (Zba).
    slli_uw = (0b000010, 1, OP_IMM_32, 64);
    /// `rori rd, rs1, sh` (Zbb).
    rori = (0b011000, 5, OP_IMM, 64);
    /// `roriw rd, rs1, sh` (Zbb).
    roriw = (0b011000, 5, OP_IMM_32, 32);
    /// `bseti rd, rs1, sh` (Zbs).
    bseti = (0b001010, 1, OP_IMM, 64);
    /// `bclri rd, rs1, sh` (Zbs).
    bclri = (0b010010, 1, OP_IMM, 64);
    /// `binvi rd, rs1, sh` (Zbs).
    binvi = (0b011010, 1, OP_IMM, 64);
    /// `bexti rd, rs1, sh` (Zbs).
    bexti = (0b010010, 5, OP_IMM, 64);
}

macro_rules! op_unary {
    ($($(#[$doc:meta])* $name:ident = ($imm12:expr, $f3:expr, $op:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, rs1: X) {
                self.emit(Ok(($imm12 as u32) << 20 | xn(rs1) << 15 | $f3 << 12 | xn(rd) << 7 | $op));
            } )*
        }
    };
}

op_unary! {
    /// `clz rd, rs1` (Zbb).
    clz = (0x600, 1, OP_IMM);
    /// `ctz rd, rs1` (Zbb).
    ctz = (0x601, 1, OP_IMM);
    /// `cpop rd, rs1` (Zbb).
    cpop = (0x602, 1, OP_IMM);
    /// `clzw rd, rs1` (Zbb).
    clzw = (0x600, 1, OP_IMM_32);
    /// `ctzw rd, rs1` (Zbb).
    ctzw = (0x601, 1, OP_IMM_32);
    /// `cpopw rd, rs1` (Zbb).
    cpopw = (0x602, 1, OP_IMM_32);
    /// `sext.b rd, rs1` (Zbb).
    sext_b = (0x604, 1, OP_IMM);
    /// `sext.h rd, rs1` (Zbb).
    sext_h = (0x605, 1, OP_IMM);
    /// `zext.h rd, rs1` (Zbb, RV64).
    zext_h = (0x080, 4, OP_OP_32);
    /// `rev8 rd, rs1` (Zbb, RV64).
    rev8 = (0x6b8, 5, OP_IMM);
    /// `orc.b rd, rs1` (Zbb).
    orc_b = (0x287, 5, OP_IMM);
}

macro_rules! loads {
    ($($(#[$doc:meta])* $name:ident = $f3:expr;)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, base: X, off: i64) {
                self.emit(i(off, xn(base), $f3, xn(rd), OP_LOAD));
            } )*
        }
    };
}
loads! {
    /// `lb rd, off(base)`.
    lb = 0;
    /// `lh rd, off(base)`.
    lh = 1;
    /// `lw rd, off(base)`.
    lw = 2;
    /// `ld rd, off(base)`.
    ld = 3;
    /// `lbu rd, off(base)`.
    lbu = 4;
    /// `lhu rd, off(base)`.
    lhu = 5;
    /// `lwu rd, off(base)`.
    lwu = 6;
}

macro_rules! stores {
    ($($(#[$doc:meta])* $name:ident = $f3:expr;)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rs2: X, base: X, off: i64) {
                self.emit(s(off, xn(rs2), xn(base), $f3, OP_STORE));
            } )*
        }
    };
}
stores! {
    /// `sb rs2, off(base)`.
    sb = 0;
    /// `sh rs2, off(base)`.
    sh = 1;
    /// `sw rs2, off(base)`.
    sw = 2;
    /// `sd rs2, off(base)`.
    sd = 3;
}

macro_rules! amos {
    ($($(#[$doc:meta])* $name:ident = ($f5:expr, $f3:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, rs2: X, addr: X, order: Order) {
                self.emit(Ok(r($f5 << 2 | order.bits(), xn(rs2), xn(addr), $f3, xn(rd), OP_AMO)));
            } )*
        }
    };
}
amos! {
    /// `amoswap.w rd, rs2, (addr)`.
    amoswap_w = (0b00001, 2);
    /// `amoadd.w rd, rs2, (addr)`.
    amoadd_w = (0b00000, 2);
    /// `amoxor.w rd, rs2, (addr)`.
    amoxor_w = (0b00100, 2);
    /// `amoand.w rd, rs2, (addr)`.
    amoand_w = (0b01100, 2);
    /// `amoor.w rd, rs2, (addr)`.
    amoor_w = (0b01000, 2);
    /// `amomin.w rd, rs2, (addr)`.
    amomin_w = (0b10000, 2);
    /// `amomax.w rd, rs2, (addr)`.
    amomax_w = (0b10100, 2);
    /// `amominu.w rd, rs2, (addr)`.
    amominu_w = (0b11000, 2);
    /// `amomaxu.w rd, rs2, (addr)`.
    amomaxu_w = (0b11100, 2);
    /// `amoswap.d rd, rs2, (addr)`.
    amoswap_d = (0b00001, 3);
    /// `amoadd.d rd, rs2, (addr)`.
    amoadd_d = (0b00000, 3);
    /// `amoxor.d rd, rs2, (addr)`.
    amoxor_d = (0b00100, 3);
    /// `amoand.d rd, rs2, (addr)`.
    amoand_d = (0b01100, 3);
    /// `amoor.d rd, rs2, (addr)`.
    amoor_d = (0b01000, 3);
    /// `amomin.d rd, rs2, (addr)`.
    amomin_d = (0b10000, 3);
    /// `amomax.d rd, rs2, (addr)`.
    amomax_d = (0b10100, 3);
    /// `amominu.d rd, rs2, (addr)`.
    amominu_d = (0b11000, 3);
    /// `amomaxu.d rd, rs2, (addr)`.
    amomaxu_d = (0b11100, 3);
    /// `sc.w rd, rs2, (addr)`: 0 in `rd` when the store happened.
    sc_w = (0b00011, 2);
    /// `sc.d rd, rs2, (addr)`: 0 in `rd` when the store happened.
    sc_d = (0b00011, 3);
}

impl Asm {
    // ── upper immediates, jumps, branches ──────────────────────────────

    /// `lui rd, imm20`: `rd = sext(imm20 << 12)`.
    pub fn lui(&mut self, rd: X, imm20: u32) {
        let w = check(imm20 < 1 << 20, || format!("lui immediate {imm20:#x} exceeds 20 bits")).map(|_| imm20 << 12 | xn(rd) << 7 | OP_LUI);
        self.emit(w);
    }
    /// `auipc rd, imm20`.
    pub fn auipc(&mut self, rd: X, imm20: u32) {
        let w = check(imm20 < 1 << 20, || format!("auipc immediate {imm20:#x} exceeds 20 bits")).map(|_| imm20 << 12 | xn(rd) << 7 | OP_AUIPC);
        self.emit(w);
    }
    /// `jal rd, label` (±1 MiB).
    pub fn jal(&mut self, rd: X, label: Label) {
        self.emit_fix(Ok(vec![xn(rd) << 7 | OP_JAL]), label, Fix::J20);
    }
    /// `j label`: `jal zero, label`.
    pub fn j(&mut self, label: Label) {
        self.jal(ZERO, label);
    }
    /// `jalr rd, off(rs1)`.
    pub fn jalr(&mut self, rd: X, rs1: X, off: i64) {
        self.emit(i(off, xn(rs1), 0, xn(rd), OP_JALR));
    }
    /// `jr rs1`: `jalr zero, 0(rs1)`.
    pub fn jr(&mut self, rs1: X) {
        self.jalr(ZERO, rs1, 0);
    }
    /// `ret`: `jalr zero, 0(ra)`.
    pub fn ret(&mut self) {
        self.jalr(ZERO, RA, 0);
    }
    /// `call label`: `auipc ra` then `jalr ra` (±2 GiB).
    pub fn call(&mut self, label: Label) {
        self.emit_fix(Ok(vec![xn(RA) << 7 | OP_AUIPC, xn(RA) << 15 | xn(RA) << 7 | OP_JALR]), label, Fix::Pair);
    }
    /// `la rd, label`: `auipc rd` then `addi rd, rd` — the address of a label (±2 GiB).
    pub fn la(&mut self, rd: X, label: Label) {
        self.emit_fix(Ok(vec![xn(rd) << 7 | OP_AUIPC, xn(rd) << 15 | xn(rd) << 7 | OP_IMM]), label, Fix::Pair);
    }
    /// A conditional branch to `label` (±4 KiB).
    pub fn branch(&mut self, cond: Cond, rs1: X, rs2: X, label: Label) {
        self.emit_fix(Ok(vec![r(0, xn(rs2), xn(rs1), cond.funct3(), 0, OP_BRANCH)]), label, Fix::B12);
    }
    /// A conditional branch that reaches ±1 MiB: the opposite branch over a `j`.
    pub fn branch_far(&mut self, cond: Cond, rs1: X, rs2: X, label: Label) {
        let skip = crate::asm::b_field(8);
        self.emit(Ok(r(0, xn(rs2), xn(rs1), cond.invert().funct3(), 0, OP_BRANCH) | skip));
        self.j(label);
    }
    /// `beq rs1, rs2, label`.
    pub fn beq(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Eq, rs1, rs2, label);
    }
    /// `bne rs1, rs2, label`.
    pub fn bne(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Ne, rs1, rs2, label);
    }
    /// `blt rs1, rs2, label`.
    pub fn blt(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Lt, rs1, rs2, label);
    }
    /// `bge rs1, rs2, label`.
    pub fn bge(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Ge, rs1, rs2, label);
    }
    /// `bltu rs1, rs2, label`.
    pub fn bltu(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Ltu, rs1, rs2, label);
    }
    /// `bgeu rs1, rs2, label`.
    pub fn bgeu(&mut self, rs1: X, rs2: X, label: Label) {
        self.branch(Cond::Geu, rs1, rs2, label);
    }
    /// `bgt rs1, rs2, label`: `blt rs2, rs1`.
    pub fn bgt(&mut self, rs1: X, rs2: X, label: Label) {
        self.blt(rs2, rs1, label);
    }
    /// `ble rs1, rs2, label`: `bge rs2, rs1`.
    pub fn ble(&mut self, rs1: X, rs2: X, label: Label) {
        self.bge(rs2, rs1, label);
    }
    /// `bgtu rs1, rs2, label`: `bltu rs2, rs1`.
    pub fn bgtu(&mut self, rs1: X, rs2: X, label: Label) {
        self.bltu(rs2, rs1, label);
    }
    /// `bleu rs1, rs2, label`: `bgeu rs2, rs1`.
    pub fn bleu(&mut self, rs1: X, rs2: X, label: Label) {
        self.bgeu(rs2, rs1, label);
    }
    /// `beqz rs1, label`.
    pub fn beqz(&mut self, rs1: X, label: Label) {
        self.beq(rs1, ZERO, label);
    }
    /// `bnez rs1, label`.
    pub fn bnez(&mut self, rs1: X, label: Label) {
        self.bne(rs1, ZERO, label);
    }

    // ── pseudo-instructions ─────────────────────────────────────────────

    /// `nop`: `addi zero, zero, 0`.
    pub fn nop(&mut self) {
        self.word(crate::NOP);
    }
    /// `mv rd, rs1`: `addi rd, rs1, 0`.
    pub fn mv(&mut self, rd: X, rs1: X) {
        self.addi(rd, rs1, 0);
    }
    /// `not rd, rs1`: `xori rd, rs1, -1`.
    pub fn not(&mut self, rd: X, rs1: X) {
        self.xori(rd, rs1, -1);
    }
    /// `neg rd, rs1`: `sub rd, zero, rs1`.
    pub fn neg(&mut self, rd: X, rs1: X) {
        self.sub(rd, ZERO, rs1);
    }
    /// `seqz rd, rs1`: `sltiu rd, rs1, 1`.
    pub fn seqz(&mut self, rd: X, rs1: X) {
        self.sltiu(rd, rs1, 1);
    }
    /// `snez rd, rs1`: `sltu rd, zero, rs1`.
    pub fn snez(&mut self, rd: X, rs1: X) {
        self.sltu(rd, ZERO, rs1);
    }
    /// `sext.w rd, rs1`: `addiw rd, rs1, 0`.
    pub fn sext_w(&mut self, rd: X, rs1: X) {
        self.addiw(rd, rs1, 0);
    }
    /// `zext.w rd, rs1`: `add.uw rd, rs1, zero` (Zba).
    pub fn zext_w(&mut self, rd: X, rs1: X) {
        self.add_uw(rd, rs1, ZERO);
    }
    /// `li rd, value`: the shortest `lui`/`addi(w)`/`slli` sequence
    /// ([`crate::li_sequence`]).
    pub fn li(&mut self, rd: X, value: i64) {
        for step in crate::imm::li_sequence(value) {
            match step {
                crate::imm::LiStep::Lui(hi) => self.lui(rd, hi),
                crate::imm::LiStep::Addi(lo, from_zero) => self.addi(rd, if from_zero { ZERO } else { rd }, lo),
                crate::imm::LiStep::Addiw(lo) => self.addiw(rd, rd, lo),
                crate::imm::LiStep::Slli(sh) => self.slli(rd, rd, sh),
            }
        }
    }
    // ── atomics, fences, system ─────────────────────────────────────────

    /// `lr.w rd, (addr)`.
    pub fn lr_w(&mut self, rd: X, addr: X, order: Order) {
        self.emit(Ok(r(0b00010 << 2 | order.bits(), 0, xn(addr), 2, xn(rd), OP_AMO)));
    }
    /// `lr.d rd, (addr)`.
    pub fn lr_d(&mut self, rd: X, addr: X, order: Order) {
        self.emit(Ok(r(0b00010 << 2 | order.bits(), 0, xn(addr), 3, xn(rd), OP_AMO)));
    }
    /// `fence pred, succ`.
    pub fn fence(&mut self, pred: Fence, succ: Fence) {
        let w = check(pred.0 != 0 && succ.0 != 0, || "a fence orders nothing with an empty set (use pause)".into())
            .map(|_| pred.0 << 24 | succ.0 << 20 | OP_MISC_MEM);
        self.emit(w);
    }
    /// `fence.tso`.
    pub fn fence_tso(&mut self) {
        self.word(0b1000 << 28 | 3 << 24 | 3 << 20 | OP_MISC_MEM);
    }
    /// `fence.i`: this hart's instruction fetches see its earlier stores.
    pub fn fence_i(&mut self) {
        self.word(1 << 12 | OP_MISC_MEM);
    }
    /// `pause` (Zihintpause): a spin-wait hint.
    pub fn pause(&mut self) {
        self.word(1 << 24 | OP_MISC_MEM);
    }
    /// `ecall`.
    pub fn ecall(&mut self) {
        self.word(OP_SYSTEM);
    }
    /// `ebreak`.
    pub fn ebreak(&mut self) {
        self.word(1 << 20 | OP_SYSTEM);
    }
    /// `csrrw rd, csr, rs1`.
    pub fn csrrw(&mut self, rd: X, csr: Csr, rs1: X) {
        self.emit(Ok(u32::from(csr.0) << 20 | xn(rs1) << 15 | 1 << 12 | xn(rd) << 7 | OP_SYSTEM));
    }
    /// `csrrs rd, csr, rs1`.
    pub fn csrrs(&mut self, rd: X, csr: Csr, rs1: X) {
        self.emit(Ok(u32::from(csr.0) << 20 | xn(rs1) << 15 | 2 << 12 | xn(rd) << 7 | OP_SYSTEM));
    }
    /// `csrrc rd, csr, rs1`.
    pub fn csrrc(&mut self, rd: X, csr: Csr, rs1: X) {
        self.emit(Ok(u32::from(csr.0) << 20 | xn(rs1) << 15 | 3 << 12 | xn(rd) << 7 | OP_SYSTEM));
    }
    /// `csrrwi rd, csr, uimm5`.
    pub fn csrrwi(&mut self, rd: X, csr: Csr, uimm: u32) {
        let w = check(uimm < 32, || format!("CSR immediate {uimm} exceeds 5 bits"))
            .map(|_| u32::from(csr.0) << 20 | uimm << 15 | 5 << 12 | xn(rd) << 7 | OP_SYSTEM);
        self.emit(w);
    }
    /// `csrr rd, csr`: `csrrs rd, csr, zero`.
    pub fn csrr(&mut self, rd: X, csr: Csr) {
        self.csrrs(rd, csr, ZERO);
    }
    /// `csrw csr, rs1`: `csrrw zero, csr, rs1`.
    pub fn csrw(&mut self, csr: Csr, rs1: X) {
        self.csrrw(ZERO, csr, rs1);
    }
    /// `rdtime rd`: the wall-clock timer.
    pub fn rdtime(&mut self, rd: X) {
        self.csrr(rd, Csr::TIME);
    }
    /// `rdcycle rd` (the kernel may forbid it to user space).
    pub fn rdcycle(&mut self, rd: X) {
        self.csrr(rd, Csr::CYCLE);
    }
}

// ── floating point ──────────────────────────────────────────────────────

/// Scalar floating-point formats, the `fmt` field.
pub(crate) const FMT_S: u32 = 0;
pub(crate) const FMT_D: u32 = 1;
pub(crate) const FMT_H: u32 = 2;

fn fp_r(f5: u32, fmt: u32, rs2: u32, rs1: u32, rm: u32, rd: u32) -> u32 {
    r(f5 << 2 | fmt, rs2, rs1, rm, rd, OP_FP)
}

macro_rules! fp_arith {
    ($($(#[$doc:meta])* $name:ident = ($f5:expr, $fmt:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, fd: F, fs1: F, fs2: F) {
                self.emit(Ok(fp_r($f5, $fmt, fnum(fs2), fnum(fs1), Rm::Dyn.bits(), fnum(fd))));
            } )*
        }
    };
}
fp_arith! {
    /// `fadd.s fd, fs1, fs2`.
    fadd_s = (0b00000, FMT_S);
    /// `fsub.s fd, fs1, fs2`.
    fsub_s = (0b00001, FMT_S);
    /// `fmul.s fd, fs1, fs2`.
    fmul_s = (0b00010, FMT_S);
    /// `fdiv.s fd, fs1, fs2`.
    fdiv_s = (0b00011, FMT_S);
    /// `fadd.d fd, fs1, fs2`.
    fadd_d = (0b00000, FMT_D);
    /// `fsub.d fd, fs1, fs2`.
    fsub_d = (0b00001, FMT_D);
    /// `fmul.d fd, fs1, fs2`.
    fmul_d = (0b00010, FMT_D);
    /// `fdiv.d fd, fs1, fs2`.
    fdiv_d = (0b00011, FMT_D);
}

macro_rules! fp_fixed {
    ($($(#[$doc:meta])* $name:ident = ($f5:expr, $fmt:expr, $f3:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, fd: F, fs1: F, fs2: F) {
                self.emit(Ok(fp_r($f5, $fmt, fnum(fs2), fnum(fs1), $f3, fnum(fd))));
            } )*
        }
    };
}
fp_fixed! {
    /// `fsgnj.s fd, fs1, fs2`.
    fsgnj_s = (0b00100, FMT_S, 0);
    /// `fsgnjn.s fd, fs1, fs2`.
    fsgnjn_s = (0b00100, FMT_S, 1);
    /// `fsgnjx.s fd, fs1, fs2`.
    fsgnjx_s = (0b00100, FMT_S, 2);
    /// `fmin.s fd, fs1, fs2`.
    fmin_s = (0b00101, FMT_S, 0);
    /// `fmax.s fd, fs1, fs2`.
    fmax_s = (0b00101, FMT_S, 1);
    /// `fsgnj.d fd, fs1, fs2`.
    fsgnj_d = (0b00100, FMT_D, 0);
    /// `fsgnjn.d fd, fs1, fs2`.
    fsgnjn_d = (0b00100, FMT_D, 1);
    /// `fsgnjx.d fd, fs1, fs2`.
    fsgnjx_d = (0b00100, FMT_D, 2);
    /// `fmin.d fd, fs1, fs2`.
    fmin_d = (0b00101, FMT_D, 0);
    /// `fmax.d fd, fs1, fs2`.
    fmax_d = (0b00101, FMT_D, 1);
}

macro_rules! fp_cmp {
    ($($(#[$doc:meta])* $name:ident = ($fmt:expr, $f3:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, fs1: F, fs2: F) {
                self.emit(Ok(fp_r(0b10100, $fmt, fnum(fs2), fnum(fs1), $f3, xn(rd))));
            } )*
        }
    };
}
fp_cmp! {
    /// `feq.s rd, fs1, fs2`.
    feq_s = (FMT_S, 2);
    /// `flt.s rd, fs1, fs2`.
    flt_s = (FMT_S, 1);
    /// `fle.s rd, fs1, fs2`.
    fle_s = (FMT_S, 0);
    /// `feq.d rd, fs1, fs2`.
    feq_d = (FMT_D, 2);
    /// `flt.d rd, fs1, fs2`.
    flt_d = (FMT_D, 1);
    /// `fle.d rd, fs1, fs2`.
    fle_d = (FMT_D, 0);
}

macro_rules! fp_fma {
    ($($(#[$doc:meta])* $name:ident = ($op:expr, $fmt:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, fd: F, fs1: F, fs2: F, fs3: F) {
                self.emit(Ok(fnum(fs3) << 27 | $fmt << 25 | fnum(fs2) << 20 | fnum(fs1) << 15 | Rm::Dyn.bits() << 12 | fnum(fd) << 7 | $op));
            } )*
        }
    };
}
fp_fma! {
    /// `fmadd.s fd, fs1, fs2, fs3`: `fs1·fs2 + fs3`, one rounding.
    fmadd_s = (OP_MADD, FMT_S);
    /// `fmsub.s fd, fs1, fs2, fs3`: `fs1·fs2 − fs3`.
    fmsub_s = (OP_MSUB, FMT_S);
    /// `fnmsub.s fd, fs1, fs2, fs3`: `−fs1·fs2 + fs3`.
    fnmsub_s = (OP_NMSUB, FMT_S);
    /// `fnmadd.s fd, fs1, fs2, fs3`: `−fs1·fs2 − fs3`.
    fnmadd_s = (OP_NMADD, FMT_S);
    /// `fmadd.d fd, fs1, fs2, fs3`.
    fmadd_d = (OP_MADD, FMT_D);
    /// `fmsub.d fd, fs1, fs2, fs3`.
    fmsub_d = (OP_MSUB, FMT_D);
    /// `fnmsub.d fd, fs1, fs2, fs3`.
    fnmsub_d = (OP_NMSUB, FMT_D);
    /// `fnmadd.d fd, fs1, fs2, fs3`.
    fnmadd_d = (OP_NMADD, FMT_D);
}

/// Integer operand widths of the conversions, the `rs2` field.
pub(crate) const CVT_W: u32 = 0;
pub(crate) const CVT_WU: u32 = 1;
pub(crate) const CVT_L: u32 = 2;
pub(crate) const CVT_LU: u32 = 3;

macro_rules! fp_to_int {
    ($($(#[$doc:meta])* $name:ident = ($fmt:expr, $ty:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, rd: X, fs1: F, rm: Rm) {
                self.emit(Ok(fp_r(0b11000, $fmt, $ty, fnum(fs1), rm.bits(), xn(rd))));
            } )*
        }
    };
}
fp_to_int! {
    /// `fcvt.w.s rd, fs1, rm`.
    fcvt_w_s = (FMT_S, CVT_W);
    /// `fcvt.wu.s rd, fs1, rm`.
    fcvt_wu_s = (FMT_S, CVT_WU);
    /// `fcvt.l.s rd, fs1, rm`.
    fcvt_l_s = (FMT_S, CVT_L);
    /// `fcvt.lu.s rd, fs1, rm`.
    fcvt_lu_s = (FMT_S, CVT_LU);
    /// `fcvt.w.d rd, fs1, rm`.
    fcvt_w_d = (FMT_D, CVT_W);
    /// `fcvt.wu.d rd, fs1, rm`.
    fcvt_wu_d = (FMT_D, CVT_WU);
    /// `fcvt.l.d rd, fs1, rm`.
    fcvt_l_d = (FMT_D, CVT_L);
    /// `fcvt.lu.d rd, fs1, rm`.
    fcvt_lu_d = (FMT_D, CVT_LU);
}

macro_rules! int_to_fp {
    ($($(#[$doc:meta])* $name:ident = ($fmt:expr, $ty:expr);)*) => {
        impl Asm {
            $( $(#[$doc])* pub fn $name(&mut self, fd: F, rs1: X, rm: Rm) {
                self.emit(Ok(fp_r(0b11010, $fmt, $ty, xn(rs1), rm.bits(), fnum(fd))));
            } )*
        }
    };
}
int_to_fp! {
    /// `fcvt.s.w fd, rs1, rm`.
    fcvt_s_w = (FMT_S, CVT_W);
    /// `fcvt.s.wu fd, rs1, rm`.
    fcvt_s_wu = (FMT_S, CVT_WU);
    /// `fcvt.s.l fd, rs1, rm`.
    fcvt_s_l = (FMT_S, CVT_L);
    /// `fcvt.s.lu fd, rs1, rm`.
    fcvt_s_lu = (FMT_S, CVT_LU);
    /// `fcvt.d.l fd, rs1, rm`.
    fcvt_d_l = (FMT_D, CVT_L);
    /// `fcvt.d.lu fd, rs1, rm`.
    fcvt_d_lu = (FMT_D, CVT_LU);
}

impl Asm {
    /// `flw fd, off(base)`.
    pub fn flw(&mut self, fd: F, base: X, off: i64) {
        self.emit(i(off, xn(base), 2, fnum(fd), OP_LOAD_FP));
    }
    /// `fld fd, off(base)`.
    pub fn fld(&mut self, fd: F, base: X, off: i64) {
        self.emit(i(off, xn(base), 3, fnum(fd), OP_LOAD_FP));
    }
    /// `flh fd, off(base)` (Zfh or Zfbfmin): a half or bf16, NaN-boxed.
    pub fn flh(&mut self, fd: F, base: X, off: i64) {
        self.emit(i(off, xn(base), 1, fnum(fd), OP_LOAD_FP));
    }
    /// `fsw fs2, off(base)`.
    pub fn fsw(&mut self, fs2: F, base: X, off: i64) {
        self.emit(s(off, fnum(fs2), xn(base), 2, OP_STORE_FP));
    }
    /// `fsd fs2, off(base)`.
    pub fn fsd(&mut self, fs2: F, base: X, off: i64) {
        self.emit(s(off, fnum(fs2), xn(base), 3, OP_STORE_FP));
    }
    /// `fsh fs2, off(base)` (Zfh or Zfbfmin).
    pub fn fsh(&mut self, fs2: F, base: X, off: i64) {
        self.emit(s(off, fnum(fs2), xn(base), 1, OP_STORE_FP));
    }
    /// `fsqrt.s fd, fs1`.
    pub fn fsqrt_s(&mut self, fd: F, fs1: F) {
        self.emit(Ok(fp_r(0b01011, FMT_S, 0, fnum(fs1), Rm::Dyn.bits(), fnum(fd))));
    }
    /// `fsqrt.d fd, fs1`.
    pub fn fsqrt_d(&mut self, fd: F, fs1: F) {
        self.emit(Ok(fp_r(0b01011, FMT_D, 0, fnum(fs1), Rm::Dyn.bits(), fnum(fd))));
    }
    /// `fmv.s fd, fs1`: `fsgnj.s fd, fs1, fs1`.
    pub fn fmv_s(&mut self, fd: F, fs1: F) {
        self.fsgnj_s(fd, fs1, fs1);
    }
    /// `fneg.s fd, fs1`: `fsgnjn.s fd, fs1, fs1`.
    pub fn fneg_s(&mut self, fd: F, fs1: F) {
        self.fsgnjn_s(fd, fs1, fs1);
    }
    /// `fabs.s fd, fs1`: `fsgnjx.s fd, fs1, fs1`.
    pub fn fabs_s(&mut self, fd: F, fs1: F) {
        self.fsgnjx_s(fd, fs1, fs1);
    }
    /// `fmv.d fd, fs1`: `fsgnj.d fd, fs1, fs1`.
    pub fn fmv_d(&mut self, fd: F, fs1: F) {
        self.fsgnj_d(fd, fs1, fs1);
    }
    /// `fclass.s rd, fs1`.
    pub fn fclass_s(&mut self, rd: X, fs1: F) {
        self.emit(Ok(fp_r(0b11100, FMT_S, 0, fnum(fs1), 1, xn(rd))));
    }
    /// `fmv.x.w rd, fs1`: the bits of a single.
    pub fn fmv_x_w(&mut self, rd: X, fs1: F) {
        self.emit(Ok(fp_r(0b11100, FMT_S, 0, fnum(fs1), 0, xn(rd))));
    }
    /// `fmv.w.x fd, rs1`: a single from its bits.
    pub fn fmv_w_x(&mut self, fd: F, rs1: X) {
        self.emit(Ok(fp_r(0b11110, FMT_S, 0, xn(rs1), 0, fnum(fd))));
    }
    /// `fmv.x.d rd, fs1`.
    pub fn fmv_x_d(&mut self, rd: X, fs1: F) {
        self.emit(Ok(fp_r(0b11100, FMT_D, 0, fnum(fs1), 0, xn(rd))));
    }
    /// `fmv.d.x fd, rs1`.
    pub fn fmv_d_x(&mut self, fd: F, rs1: X) {
        self.emit(Ok(fp_r(0b11110, FMT_D, 0, xn(rs1), 0, fnum(fd))));
    }
    /// `fmv.x.h rd, fs1` (Zfh or Zfbfmin): the sign-extended sixteen bits.
    pub fn fmv_x_h(&mut self, rd: X, fs1: F) {
        self.emit(Ok(fp_r(0b11100, FMT_H, 0, fnum(fs1), 0, xn(rd))));
    }
    /// `fmv.h.x fd, rs1` (Zfh or Zfbfmin): sixteen bits, NaN-boxed — the
    /// scalar operand of `vfwmaccbf16.vf`.
    pub fn fmv_h_x(&mut self, fd: F, rs1: X) {
        self.emit(Ok(fp_r(0b11110, FMT_H, 0, xn(rs1), 0, fnum(fd))));
    }
    /// `fcvt.s.d fd, fs1, rm`.
    pub fn fcvt_s_d(&mut self, fd: F, fs1: F, rm: Rm) {
        self.emit(Ok(fp_r(0b01000, FMT_S, 1, fnum(fs1), rm.bits(), fnum(fd))));
    }
    /// `fcvt.d.s fd, fs1`: exact (encoded with `rne`, as the assemblers do).
    pub fn fcvt_d_s(&mut self, fd: F, fs1: F) {
        self.emit(Ok(fp_r(0b01000, FMT_D, 0, fnum(fs1), Rm::Rne.bits(), fnum(fd))));
    }
    /// `fcvt.d.w fd, rs1`: exact.
    pub fn fcvt_d_w(&mut self, fd: F, rs1: X) {
        self.emit(Ok(fp_r(0b11010, FMT_D, CVT_W, xn(rs1), Rm::Rne.bits(), fnum(fd))));
    }
    /// `fcvt.d.wu fd, rs1`: exact.
    pub fn fcvt_d_wu(&mut self, fd: F, rs1: X) {
        self.emit(Ok(fp_r(0b11010, FMT_D, CVT_WU, xn(rs1), Rm::Rne.bits(), fnum(fd))));
    }
    /// `fcvt.bf16.s fd, fs1, rm` (Zfbfmin).
    pub fn fcvt_bf16_s(&mut self, fd: F, fs1: F, rm: Rm) {
        self.emit(Ok(fp_r(0b01000, FMT_H, 0b01000, fnum(fs1), rm.bits(), fnum(fd))));
    }
    /// `fcvt.s.bf16 fd, fs1` (Zfbfmin): exact.
    pub fn fcvt_s_bf16(&mut self, fd: F, fs1: F) {
        self.emit(Ok(fp_r(0b01000, FMT_S, 0b00110, fnum(fs1), Rm::Rne.bits(), fnum(fd))));
    }
}
