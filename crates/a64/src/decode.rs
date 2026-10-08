//! The decoder: every word this crate can emit, back to its control flow and
//! to assembler text.
//!
//! Two consumers. The **audit** of generated code needs, for every word, to
//! know whether it is an instruction at all and where it can transfer
//! control ([`Flow`]); a word outside the decoded set is reported as unknown
//! rather than guessed. The **listing** needs text: it is written in the
//! syntax the system assembler accepts, canonical forms rather than aliases
//! (`orr x0, xzr, x1` rather than `mov`), so that a listing can be
//! reassembled (by `clang`).

use crate::imm::{decode_fp8, decode_logical};
use crate::operand::{Barrier, Cond, Prefetch, SysReg};

/// Where an instruction can send control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// To the next instruction only.
    Next,
    /// Unconditionally to `offset` bytes from this instruction.
    Branch(i64),
    /// To `offset` bytes from this instruction, or to the next one.
    CondBranch(i64),
    /// A call to `offset` bytes from this instruction (`bl`), returning next.
    Call(i64),
    /// A return through `x30`, or another register (`ret xn`).
    Return,
    /// A branch through a register (`br`).
    IndirectBranch,
    /// A call through a register (`blr`).
    IndirectCall,
    /// An exception: `brk`, `udf`, `svc`, `hvc`, `smc`.
    Trap,
}

/// One decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    /// The instruction word.
    pub word: u32,
    /// Assembler text.
    pub text: String,
    /// Control flow.
    pub flow: Flow,
    /// The register a `blr`/`br`/`ret` goes through.
    pub target_reg: Option<u8>,
}

/// Decode one word; `None` when it is outside the set this crate knows.
pub fn decode(word: u32) -> Option<Insn> {
    let (text, flow) = decode_inner(word)?;
    let target_reg = matches!(flow, Flow::IndirectBranch | Flow::IndirectCall | Flow::Return).then_some(((word >> 5) & 31) as u8);
    Some(Insn { word, text, flow, target_reg })
}

/// Decode a whole buffer: `(offset, instruction or None)` per word.
pub fn decode_all(bytes: &[u8]) -> Vec<(usize, Option<Insn>)> {
    bytes
        .chunks(4)
        .enumerate()
        .map(|(i, c)| {
            if c.len() < 4 {
                (i * 4, None)
            } else {
                (i * 4, decode(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            }
        })
        .collect()
}

fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}
fn sext(v: u32, width: u32) -> i64 {
    let shift = 64 - width;
    ((u64::from(v) << shift) as i64) >> shift
}

/// A general-purpose register name: `zr` or `sp` for 31.
fn gp(n: u32, x: bool, sp: bool) -> String {
    match (n, x, sp) {
        (31, true, true) => "sp".into(),
        (31, false, true) => "wsp".into(),
        (31, true, false) => "xzr".into(),
        (31, false, false) => "wzr".into(),
        (n, true, _) => format!("x{n}"),
        (n, false, _) => format!("w{n}"),
    }
}
fn xr(n: u32) -> String {
    gp(n, true, false)
}
fn xsp(n: u32) -> String {
    gp(n, true, true)
}

const ARR: [[&str; 2]; 4] = [["8b", "16b"], ["4h", "8h"], ["2s", "4s"], ["1d", "2d"]];
fn arr(size: u32, q: u32) -> &'static str {
    ARR[size as usize][q as usize]
}
fn vreg(n: u32, size: u32, q: u32) -> String {
    format!("v{n}.{}", arr(size, q))
}
fn fp_letter(t: u32) -> Option<char> {
    match t {
        0 => Some('s'),
        1 => Some('d'),
        3 => Some('h'),
        _ => None,
    }
}
fn fp_imm(v: f64) -> String {
    format!("#{v:?}")
}

fn decode_inner(w: u32) -> Option<(String, Flow)> {
    let next = |t: String| Some((t, Flow::Next));
    let rd = bits(w, 4, 0);
    let rn = bits(w, 9, 5);
    let rm = bits(w, 20, 16);
    let sf = bits(w, 31, 31) == 1;

    // ── Apple's AMX, outside the Arm encodings (`amx.rs`) ───────────────
    if let Some(text) = crate::amx::decode(w) {
        return next(text);
    }

    // ── exact words ────────────────────────────────────────────────────
    match w {
        0xd503_201f => return next("nop".into()),
        0xd503_203f => return next("yield".into()),
        0xd503_205f => return next("wfe".into()),
        0xd503_207f => return next("wfi".into()),
        0xd503_209f => return next("sev".into()),
        0xd503_20bf => return next("sevl".into()),
        0xd503_3fdf => return next("isb".into()),
        0xd503_477f => return next("smstart".into()),
        0xd503_467f => return next("smstop".into()),
        0xd503_437f => return next("smstart sm".into()),
        0xd503_427f => return next("smstop sm".into()),
        0xd503_457f => return next("smstart za".into()),
        0xd503_447f => return next("smstop za".into()),
        0xc008_00ff => return next("zero {za}".into()),
        _ => {}
    }
    if w & 0xffff_f0ff == 0xd503_30bf || w & 0xffff_f0ff == 0xd503_309f {
        let b = Barrier::from_bits(bits(w, 11, 8))?;
        let op = if bits(w, 7, 5) == 0b101 { "dmb" } else { "dsb" };
        return next(format!("{op} {}", b.name()));
    }
    if w & 0xfff0_0000 == 0xd530_0000 {
        return next(format!("mrs {}, {}", xr(rd), SysReg(bits(w, 20, 5)).name()));
    }
    if w & 0xfff0_0000 == 0xd510_0000 {
        return next(format!("msr {}, {}", SysReg(bits(w, 20, 5)).name(), xr(rd)));
    }
    if w & 0xffe0_001f == 0xd420_0000 {
        return Some((format!("brk #{:#x}", bits(w, 20, 5)), Flow::Trap));
    }
    if w & 0xffe0_001c == 0xd400_0000 && bits(w, 1, 0) != 0 {
        let op = ["", "svc", "hvc", "smc"][bits(w, 1, 0) as usize];
        return Some((format!("{op} #{:#x}", bits(w, 20, 5)), Flow::Trap));
    }
    if w & 0xffff_0000 == 0 {
        return Some((format!("udf #{}", w & 0xffff), Flow::Trap));
    }

    // ── branches ───────────────────────────────────────────────────────
    if w & 0x7c00_0000 == 0x1400_0000 {
        let off = sext(bits(w, 25, 0), 26) * 4;
        return Some(if sf { (format!("bl #{off}"), Flow::Call(off)) } else { (format!("b #{off}"), Flow::Branch(off)) });
    }
    if w & 0xff00_0010 == 0x5400_0000 {
        let off = sext(bits(w, 23, 5), 19) * 4;
        return Some((format!("b.{} #{off}", Cond::from_bits(bits(w, 3, 0)).name()), Flow::CondBranch(off)));
    }
    if w & 0x7e00_0000 == 0x3400_0000 {
        let off = sext(bits(w, 23, 5), 19) * 4;
        let op = if bits(w, 24, 24) == 1 { "cbnz" } else { "cbz" };
        return Some((format!("{op} {}, #{off}", gp(rd, sf, false)), Flow::CondBranch(off)));
    }
    if w & 0x7e00_0000 == 0x3600_0000 {
        let off = sext(bits(w, 18, 5), 14) * 4;
        let bit = (bits(w, 31, 31) << 5) | bits(w, 23, 19);
        let op = if bits(w, 24, 24) == 1 { "tbnz" } else { "tbz" };
        return Some((format!("{op} {}, #{bit}, #{off}", gp(rd, bit >= 32, false)), Flow::CondBranch(off)));
    }
    if w & 0xffff_fc1f == 0xd61f_0000 {
        return Some((format!("br {}", xr(rn)), Flow::IndirectBranch));
    }
    if w & 0xffff_fc1f == 0xd63f_0000 {
        return Some((format!("blr {}", xr(rn)), Flow::IndirectCall));
    }
    if w & 0xffff_fc1f == 0xd65f_0000 {
        return Some((if rn == 30 { "ret".into() } else { format!("ret {}", xr(rn)) }, Flow::Return));
    }

    // ── data processing, immediate ─────────────────────────────────────
    if w & 0x9f00_0000 == 0x1000_0000 {
        let imm = (bits(w, 23, 5) << 2) | bits(w, 30, 29);
        return next(format!("adr {}, #{}", xr(rd), sext(imm, 21)));
    }
    if w & 0x1f80_0000 == 0x1280_0000 {
        let op = match bits(w, 30, 29) {
            0 => "movn",
            2 => "movz",
            3 => "movk",
            _ => return None,
        };
        let hw = bits(w, 22, 21);
        if !sf && hw > 1 {
            return None;
        }
        return next(format!("{op} {}, #{:#x}, lsl #{}", gp(rd, sf, false), bits(w, 20, 5), hw * 16));
    }
    if w & 0x1f80_0000 == 0x1100_0000 {
        let op = ["add", "adds", "sub", "subs"][bits(w, 30, 29) as usize];
        let s = bits(w, 29, 29) == 1;
        let shift = if bits(w, 22, 22) == 1 { ", lsl #12" } else { "" };
        return next(format!("{op} {}, {}, #{}{shift}", gp(rd, sf, !s), gp(rn, sf, true), bits(w, 21, 10)));
    }
    if w & 0x1f80_0000 == 0x1200_0000 {
        let opc = bits(w, 30, 29);
        let n = bits(w, 22, 22);
        let value = decode_logical(n, bits(w, 21, 16), bits(w, 15, 10), if sf { 64 } else { 32 })?;
        let op = ["and", "orr", "eor", "ands"][opc as usize];
        return next(format!("{op} {}, {}, #{value:#x}", gp(rd, sf, opc != 3), gp(rn, sf, false)));
    }
    if w & 0x1f80_0000 == 0x1300_0000 {
        let op = match bits(w, 30, 29) {
            0 => "sbfm",
            1 => "bfm",
            2 => "ubfm",
            _ => return None,
        };
        if bits(w, 22, 22) != u32::from(sf) {
            return None;
        }
        return next(format!("{op} {}, {}, #{}, #{}", gp(rd, sf, false), gp(rn, sf, false), bits(w, 21, 16), bits(w, 15, 10)));
    }

    // ── data processing, register ──────────────────────────────────────
    if w & 0x1f20_0000 == 0x0b00_0000 {
        let op = ["add", "adds", "sub", "subs"][bits(w, 30, 29) as usize];
        let shift = ["lsl", "lsr", "asr", "ror"][bits(w, 23, 22) as usize];
        if shift == "ror" {
            return None;
        }
        return next(format!("{op} {}, {}, {}, {shift} #{}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false), bits(w, 15, 10)));
    }
    if w & 0x1fe0_0000 == 0x0b20_0000 {
        let op = ["add", "adds", "sub", "subs"][bits(w, 30, 29) as usize];
        let s = bits(w, 29, 29) == 1;
        let option = bits(w, 15, 13);
        let ext = ["uxtb", "uxth", "uxtw", "uxtx", "sxtb", "sxth", "sxtw", "sxtx"][option as usize];
        let wide = option & 3 == 3;
        return next(format!("{op} {}, {}, {}, {ext} #{}", gp(rd, sf, !s), gp(rn, sf, true), gp(rm, wide && sf, false), bits(w, 12, 10)));
    }
    if w & 0x1f00_0000 == 0x0a00_0000 {
        let n = bits(w, 21, 21);
        let op = [["and", "bic"], ["orr", "orn"], ["eor", "eon"], ["ands", "bics"]][bits(w, 30, 29) as usize][n as usize];
        let shift = ["lsl", "lsr", "asr", "ror"][bits(w, 23, 22) as usize];
        return next(format!("{op} {}, {}, {}, {shift} #{}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false), bits(w, 15, 10)));
    }
    if w & 0x7fe0_f000 == 0x1ac0_2000 {
        let op = ["lslv", "lsrv", "asrv", "rorv"][bits(w, 11, 10) as usize];
        return next(format!("{op} {}, {}, {}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false)));
    }
    if w & 0x7fe0_f800 == 0x1ac0_0800 {
        let op = if bits(w, 10, 10) == 1 { "sdiv" } else { "udiv" };
        return next(format!("{op} {}, {}, {}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false)));
    }
    if w & 0x7fe0_0000 == 0x1b00_0000 {
        let op = if bits(w, 15, 15) == 1 { "msub" } else { "madd" };
        return next(format!("{op} {}, {}, {}, {}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false), gp(bits(w, 14, 10), sf, false)));
    }
    if w & 0xffe0_fc00 == 0x9bc0_7c00 || w & 0xffe0_fc00 == 0x9b40_7c00 {
        let op = if w & 0x0080_0000 != 0 { "umulh" } else { "smulh" };
        return next(format!("{op} {}, {}, {}", xr(rd), xr(rn), xr(rm)));
    }
    if w & 0x1fe0_0800 == 0x1a80_0000 && bits(w, 29, 29) == 0 {
        let op = [["csel", "csinc"], ["csinv", "csneg"]][bits(w, 30, 30) as usize][bits(w, 10, 10) as usize];
        return next(format!("{op} {}, {}, {}, {}", gp(rd, sf, false), gp(rn, sf, false), gp(rm, sf, false), Cond::from_bits(bits(w, 15, 12)).name()));
    }

    // ── loads and stores ───────────────────────────────────────────────
    if let Some(t) = decode_ldst(w) {
        return next(t);
    }

    // ── SIMD & FP ──────────────────────────────────────────────────────
    if let Some(t) = crate::decode_simd::decode_simd(w) {
        return next(t);
    }
    if let Some(t) = decode_sme(w) {
        return next(t);
    }
    None
}

/// The SME and streaming SVE subset of `sme.rs`.
fn decode_sme(w: u32) -> Option<String> {
    let t = ["b", "h", "s", "d"];
    let zd = bits(w, 4, 0);
    let zn = bits(w, 9, 5);
    let zm = bits(w, 20, 16);
    let slice = |f_hi: u32, tile_off: u32| {
        let dir = if f_hi & 0x8000 != 0 { 'v' } else { 'h' };
        let rs = 12 + ((f_hi >> 13) & 3);
        format!("za{}{dir}.s[w{rs}, {}]", tile_off >> 2, tile_off & 3)
    };
    if w & 0xffff_ff00 == 0xc008_0000 {
        let mask = w & 0xff;
        if mask == 0xff {
            return Some("zero {za}".into());
        }
        let tiles: Vec<String> = (0..4).filter(|t| mask & (0x11 << t) == 0x11 << t).map(|t| format!("za{t}.s")).collect();
        return (tiles.len() * 2 == mask.count_ones() as usize && !tiles.is_empty()).then(|| format!("zero {{{}}}", tiles.join(", ")));
    }
    if w & 0xff3f_ffe0 == 0x2518_e3e0 && bits(w, 4, 4) == 0 {
        return Some(format!("ptrue p{}.{}", bits(w, 3, 0), t[bits(w, 23, 22) as usize]));
    }
    if w & 0xff20_fc10 == 0x2520_1c00 {
        return Some(format!("whilelo p{}.{}, x{}, x{}", bits(w, 3, 0), t[bits(w, 23, 22) as usize], zn, zm));
    }
    if w & 0xffff_f800 == 0x04bf_5800 {
        return Some(format!("rdsvl {}, #{}", gp(zd, true, false), crate::decode::sext6(bits(w, 10, 5))));
    }
    if w & 0xffe0_f800 == 0x0420_5000 {
        return Some(format!("addvl {}, {}, #{}", gp(zd, true, true), gp(zm, true, true), crate::decode::sext6(bits(w, 10, 5))));
    }
    let contiguous = [
        (0xa4a0_a000u32, 0xa4a0_4000u32, "ld1h", "h", true),
        (0xa540_a000, 0xa540_4000, "ld1w", "s", true),
        (0xe4a0_e000, 0xe4a0_4000, "st1h", "h", false),
        (0xe540_e000, 0xe540_4000, "st1w", "s", false),
        (0xa4c0_a000, 0xa4c0_4000, "ld1h", "s", true),
        (0xe4c0_e000, 0xe4c0_4000, "st1h", "s", false),
    ];
    for (imm_base, reg_base, op, et, load) in contiguous {
        let pg = if load { format!("p{}/z", bits(w, 12, 10)) } else { format!("p{}", bits(w, 12, 10)) };
        if w & 0xfff0_e000 == imm_base {
            let vl = ((bits(w, 19, 16) as i32) << 28) >> 28;
            return Some(format!("{op} {{ z{zd}.{et} }}, {pg}, [{}, #{vl}, mul vl]", gp(zn, true, true)));
        }
        if w & 0xffe0_e000 == reg_base {
            let shift = if et == "h" || op.ends_with('h') { 1 } else { 2 };
            return Some(format!("{op} {{ z{zd}.{et} }}, {pg}, [{}, x{zm}, lsl #{shift}]", gp(zn, true, true)));
        }
    }
    if w & 0xffe0_001c == 0x8080_0010 {
        return Some(format!("fmops za{}.s, p{}/m, p{}/m, z{zn}.s, z{zm}.s", bits(w, 1, 0), bits(w, 12, 10), bits(w, 15, 13)));
    }
    if w & 0xff20_fc00 == 0x6500_0800 && matches!(bits(w, 23, 22), 1 | 2) {
        let et = t[bits(w, 23, 22) as usize];
        return Some(format!("fmul z{zd}.{et}, z{zn}.{et}, z{zm}.{et}"));
    }
    if w & 0xff3f_fc00 == 0x0520_3800 {
        let size = bits(w, 23, 22);
        return Some(format!("mov z{zd}.{}, {}", t[size as usize], gp(zn, size == 3, true)));
    }
    if w & 0xffe0_001c == 0x8180_0000 || w & 0xffe0_001c == 0x8080_0000 {
        let (op, et) = if w & 0x0100_0000 != 0 { ("bfmopa", "h") } else { ("fmopa", "s") };
        return Some(format!("{op} za{}.s, p{}/m, p{}/m, z{zn}.{et}, z{zm}.{et}", bits(w, 1, 0), bits(w, 12, 10), bits(w, 15, 13)));
    }
    if w & 0xffff_0200 == 0xc082_0000 {
        return Some(format!("mova z{zd}.s, p{}/m, {}", bits(w, 12, 10), slice(w, bits(w, 8, 5))));
    }
    if w & 0xffff_0010 == 0xc080_0000 {
        return Some(format!("mova {}, p{}/m, z{zn}.s", slice(w, bits(w, 3, 0)), bits(w, 12, 10)));
    }
    if w & 0xffe0_0010 == 0xe080_0000 {
        return Some(format!("ld1w {{{}}}, p{}/z, [{}, x{zm}, lsl #2]", slice(w, bits(w, 3, 0)), bits(w, 12, 10), gp(zn, true, true)));
    }
    if w & 0xffe0_0010 == 0xe0a0_0000 {
        return Some(format!("st1w {{{}}}, p{}, [{}, x{zm}, lsl #2]", slice(w, bits(w, 3, 0)), bits(w, 12, 10), gp(zn, true, true)));
    }
    if w & 0xffff_e000 == 0x658a_a000 {
        return Some(format!("bfcvt z{zd}.h, p{}/m, z{zn}.s", bits(w, 12, 10)));
    }
    None
}

pub(crate) fn sext6(v: u32) -> i32 {
    ((v as i32) << 26) >> 26
}

/// The register a load/store transfers, by `(V, size, opc)`.
fn ldst_reg(v: u32, size: u32, opc: u32, t: u32) -> Option<(&'static str, String)> {
    Some(if v == 0 {
        match (size, opc) {
            (0, 0) => ("strb", gp(t, false, false)),
            (0, 1) => ("ldrb", gp(t, false, false)),
            (0, 2) => ("ldrsb", gp(t, true, false)),
            (0, 3) => ("ldrsb", gp(t, false, false)),
            (1, 0) => ("strh", gp(t, false, false)),
            (1, 1) => ("ldrh", gp(t, false, false)),
            (1, 2) => ("ldrsh", gp(t, true, false)),
            (1, 3) => ("ldrsh", gp(t, false, false)),
            (2, 0) => ("str", gp(t, false, false)),
            (2, 1) => ("ldr", gp(t, false, false)),
            (2, 2) => ("ldrsw", gp(t, true, false)),
            (3, 0) => ("str", gp(t, true, false)),
            (3, 1) => ("ldr", gp(t, true, false)),
            (3, 2) => ("prfm", Prefetch(t).name()),
            _ => return None,
        }
    } else {
        let (op, letter) = match (size, opc) {
            (0, 0) => ("str", 'b'),
            (0, 1) => ("ldr", 'b'),
            (0, 2) => ("str", 'q'),
            (0, 3) => ("ldr", 'q'),
            (1, 0) => ("str", 'h'),
            (1, 1) => ("ldr", 'h'),
            (2, 0) => ("str", 's'),
            (2, 1) => ("ldr", 's'),
            (3, 0) => ("str", 'd'),
            (3, 1) => ("ldr", 'd'),
            _ => return None,
        };
        (op, format!("{letter}{t}"))
    })
}

fn decode_ldst(w: u32) -> Option<String> {
    let rt = bits(w, 4, 0);
    let rn = bits(w, 9, 5);
    let size = bits(w, 31, 30);
    let v = bits(w, 26, 26);
    let opc = bits(w, 23, 22);
    // Single register, unsigned scaled offset.
    if w & 0x3b00_0000 == 0x3900_0000 {
        let (op, r) = ldst_reg(v, size, opc, rt)?;
        let scale = if v == 1 && opc >= 2 { 4 } else { size };
        return Some(format!("{op} {r}, [{}, #{}]", xsp(rn), bits(w, 21, 10) << scale));
    }
    if w & 0x3b20_0000 == 0x3800_0000 {
        let (op, r) = ldst_reg(v, size, opc, rt)?;
        let imm = sext(bits(w, 20, 12), 9);
        return match bits(w, 11, 10) {
            0 => {
                let op = match op {
                    "ldr" => "ldur",
                    "str" => "stur",
                    "ldrb" => "ldurb",
                    "strb" => "sturb",
                    "ldrh" => "ldurh",
                    "strh" => "sturh",
                    "ldrsw" => "ldursw",
                    "ldrsb" => "ldursb",
                    "ldrsh" => "ldursh",
                    "prfm" => "prfum",
                    _ => return None,
                };
                Some(format!("{op} {r}, [{}, #{imm}]", xsp(rn)))
            }
            1 if op != "prfm" => Some(format!("{op} {r}, [{}], #{imm}", xsp(rn))),
            3 if op != "prfm" => Some(format!("{op} {r}, [{}, #{imm}]!", xsp(rn))),
            _ => None,
        };
    }
    if w & 0x3b20_0c00 == 0x3820_0800 {
        let (op, r) = ldst_reg(v, size, opc, rt)?;
        let option = bits(w, 15, 13);
        let s = bits(w, 12, 12);
        let scale = if v == 1 && opc >= 2 { 4 } else { size };
        let index = gp(bits(w, 20, 16), option & 1 == 1, false);
        let ext = match option {
            2 => "uxtw",
            3 => "lsl",
            6 => "sxtw",
            7 => "sxtx",
            _ => return None,
        };
        let amount = if s == 1 { format!(" #{scale}") } else if ext == "lsl" { " #0".into() } else { String::new() };
        return Some(format!("{op} {r}, [{}, {index}, {ext}{amount}]", xsp(rn)));
    }
    if w & 0xbfff_fc00 == 0xb8bf_c000 {
        return Some(format!("ldapr {}, [{}]", gp(rt, size == 3, false), xsp(rn)));
    }
    // LSE atomics.
    if w & 0x3f20_0c00 == 0x3820_0000 {
        let a = bits(w, 23, 23);
        let r = bits(w, 22, 22);
        let o3 = bits(w, 15, 15);
        let op = match (o3, bits(w, 14, 12)) {
            (0, 0) => "ldadd",
            (0, 3) => "ldset",
            (1, 0) => "swp",
            _ => return None,
        };
        let order = ["", "l", "a", "al"][((a << 1) | r) as usize];
        let x = size == 3;
        if size < 2 {
            return None;
        }
        return Some(format!("{op}{order} {}, {}, [{}]", gp(bits(w, 20, 16), x, false), gp(rt, x, false), xsp(rn)));
    }
    // Load/store pairs.
    if w & 0x3a00_0000 == 0x2800_0000 {
        let opc = bits(w, 31, 30);
        let load = bits(w, 22, 22) == 1;
        let mode = bits(w, 24, 23);
        let (letter, scale) = match (v, opc) {
            (0, 0) => ('w', 2),
            (0, 2) => ('x', 3),
            (1, 0) => ('s', 2),
            (1, 1) => ('d', 3),
            (1, 2) => ('q', 4),
            _ => return None,
        };
        let reg = |n: u32| match letter {
            'w' => gp(n, false, false),
            'x' => gp(n, true, false),
            c => format!("{c}{n}"),
        };
        let imm = sext(bits(w, 21, 15), 7) << scale;
        let op = match (mode, load) {
            (0, true) => "ldnp",
            (0, false) => "stnp",
            (_, true) => "ldp",
            (_, false) => "stp",
        };
        let (a, b) = (reg(rt), reg(bits(w, 14, 10)));
        return Some(match mode {
            0 | 2 => format!("{op} {a}, {b}, [{}, #{imm}]", xsp(rn)),
            1 => format!("{op} {a}, {b}, [{}], #{imm}", xsp(rn)),
            _ => format!("{op} {a}, {b}, [{}, #{imm}]!", xsp(rn)),
        });
    }
    // Exclusives, acquire/release, compare-and-swap.
    if w & 0x3f00_0000 == 0x0800_0000 && size >= 2 {
        let x = size == 3;
        let o2 = bits(w, 23, 23);
        let l = bits(w, 22, 22);
        let o1 = bits(w, 21, 21);
        let o0 = bits(w, 15, 15);
        let rs = bits(w, 20, 16);
        let rt2 = bits(w, 14, 10);
        let t = gp(rt, x, false);
        return match (o2, l, o1) {
            (0, 1, 0) if rs == 31 && rt2 == 31 => Some(format!("{} {t}, [{}]", if o0 == 1 { "ldaxr" } else { "ldxr" }, xsp(rn))),
            (0, 0, 0) if rt2 == 31 => Some(format!("{} {}, {t}, [{}]", if o0 == 1 { "stlxr" } else { "stxr" }, gp(rs, false, false), xsp(rn))),
            (1, 1, 0) if rs == 31 && rt2 == 31 && o0 == 1 => Some(format!("ldar {t}, [{}]", xsp(rn))),
            (1, 0, 0) if rs == 31 && rt2 == 31 && o0 == 1 => Some(format!("stlr {t}, [{}]", xsp(rn))),
            (1, _, 1) if rt2 == 31 => {
                let order = ["", "l", "a", "al"][((l << 1) | o0) as usize];
                Some(format!("cas{order} {}, {t}, [{}]", gp(rs, x, false), xsp(rn)))
            }
            _ => None,
        };
    }
    None
}

pub(crate) fn fp_scalar_name(t: u32, n: u32) -> Option<String> {
    Some(format!("{}{n}", fp_letter(t)?))
}
pub(crate) fn fmt_fp_imm(imm8: u32) -> String {
    fp_imm(decode_fp8(imm8))
}
pub(crate) fn v_name(n: u32, size: u32, q: u32) -> String {
    vreg(n, size, q)
}
pub(crate) fn arr_name(size: u32, q: u32) -> &'static str {
    arr(size, q)
}
pub(crate) fn gp_name(n: u32, x: bool, sp: bool) -> String {
    gp(n, x, sp)
}
pub(crate) fn field(w: u32, hi: u32, lo: u32) -> u32 {
    bits(w, hi, lo)
}
