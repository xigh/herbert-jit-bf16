//! The decoder: every word this crate can emit, back to its control flow and
//! to assembler text.
//!
//! Two consumers. The **audit** of generated code needs, for every word, to
//! know whether it is an instruction at all and where it can transfer
//! control ([`Flow`]); a word outside the decoded set is reported as unknown
//! rather than guessed. The **listing** needs text: it is written in the
//! syntax `llvm-mc` accepts, base forms rather than aliases (`addi a0, a1, 0`
//! rather than `mv`, numeric branch offsets), so that a listing can be
//! reassembled.
//!
//! A compressed (16-bit) instruction is recognised by its low bits and
//! reported as unknown: this crate never emits one, and the audit refuses
//! what it cannot account for.

use crate::base::*;
use crate::reg::{f_name, x_name, Rm, Sew, Vtype};
use crate::vector::*;

/// Where an instruction can send control.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    /// To the next instruction only.
    Next,
    /// Unconditionally to `offset` bytes from this instruction.
    Branch(i64),
    /// To `offset` bytes from this instruction, or to the next one.
    CondBranch(i64),
    /// A call to `offset` bytes from this instruction, returning next.
    Call(i64),
    /// `jalr zero, 0(ra)`.
    Return,
    /// A jump through a register (`jalr zero, off(rs1)`).
    IndirectBranch,
    /// A call through a register (`jalr rd, off(rs1)`, `rd ≠ zero`).
    IndirectCall,
    /// An exception: `ecall`, `ebreak`.
    Trap,
}

/// One decoded instruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Insn {
    /// The instruction word.
    pub word: u32,
    /// Its length in bytes (always 4 for what this crate decodes).
    pub len: usize,
    /// Assembler text.
    pub text: String,
    /// Control flow.
    pub flow: Flow,
    /// The register a `jalr` goes through.
    pub target_reg: Option<u8>,
}

/// Decode one word; `None` when it is outside the set this crate knows.
pub fn decode(word: u32) -> Option<Insn> {
    if word & 3 != 3 {
        return None;
    }
    let (text, flow) = decode_inner(word)?;
    let target_reg = (word & 0x7f == OP_JALR).then_some(((word >> 15) & 31) as u8);
    Some(Insn { word, len: 4, text, flow, target_reg })
}

/// Decode a whole buffer: `(offset, instruction or None)` per word.
///
/// An `auipc rX` followed by a `jalr` through `rX` is the pc-relative call
/// (or jump) of [`crate::Asm::call`]: the `jalr` is reported as a
/// [`Flow::Call`] (or [`Flow::Branch`]) to the target the pair computes,
/// relative to the `jalr` itself.
pub fn decode_all(bytes: &[u8]) -> Vec<(usize, Option<Insn>)> {
    let mut out: Vec<(usize, Option<Insn>)> = bytes
        .chunks(4)
        .enumerate()
        .map(|(i, c)| {
            if c.len() < 4 {
                (i * 4, None)
            } else {
                (i * 4, decode(u32::from_le_bytes([c[0], c[1], c[2], c[3]])))
            }
        })
        .collect();
    for i in 1..out.len() {
        let (Some(first), Some(second)) = (&out[i - 1].1, &out[i].1) else { continue };
        let (a, j) = (first.word, second.word);
        let rd = (a >> 7) & 31;
        if a & 0x7f == OP_AUIPC && rd != 0 && j & 0x7f == OP_JALR && (j >> 12) & 7 == 0 && (j >> 15) & 31 == rd {
            let hi = i64::from((a & 0xffff_f000) as i32);
            let lo = i64::from((j as i32) >> 20);
            let offset = hi + lo - 4;
            let link = (j >> 7) & 31;
            if let Some(insn) = &mut out[i].1 {
                insn.flow = if link == 0 { Flow::Branch(offset) } else { Flow::Call(offset) };
            }
        }
    }
    out
}

fn bits(w: u32, hi: u32, lo: u32) -> u32 {
    (w >> lo) & ((1u32 << (hi - lo + 1)) - 1)
}
fn sext(v: u32, width: u32) -> i64 {
    let shift = 64 - width;
    ((u64::from(v) << shift) as i64) >> shift
}
fn xr(n: u32) -> &'static str {
    x_name(n)
}
fn fr(n: u32) -> &'static str {
    f_name(n)
}

fn i_imm(w: u32) -> i64 {
    sext(w >> 20, 12)
}
fn s_imm(w: u32) -> i64 {
    sext(bits(w, 31, 25) << 5 | bits(w, 11, 7), 12)
}
fn b_imm(w: u32) -> i64 {
    sext(bits(w, 31, 31) << 12 | bits(w, 7, 7) << 11 | bits(w, 30, 25) << 5 | bits(w, 11, 8) << 1, 13)
}
fn j_imm(w: u32) -> i64 {
    sext(bits(w, 31, 31) << 20 | bits(w, 19, 12) << 12 | bits(w, 20, 20) << 11 | bits(w, 30, 21) << 1, 21)
}

fn decode_inner(w: u32) -> Option<(String, Flow)> {
    let next = |t: String| Some((t, Flow::Next));
    let rd = bits(w, 11, 7);
    let rs1 = bits(w, 19, 15);
    let rs2 = bits(w, 24, 20);
    let f3 = bits(w, 14, 12);
    let f7 = bits(w, 31, 25);
    match w & 0x7f {
        OP_LUI => next(format!("lui {}, {}", xr(rd), w >> 12)),
        OP_AUIPC => next(format!("auipc {}, {}", xr(rd), w >> 12)),
        OP_JAL => {
            let off = j_imm(w);
            let flow = if rd == 0 { Flow::Branch(off) } else { Flow::Call(off) };
            Some((format!("jal {}, {off}", xr(rd)), flow))
        }
        OP_JALR if f3 == 0 => {
            let off = i_imm(w);
            let flow = match (rd, rs1, off) {
                (0, 1, 0) => Flow::Return,
                (0, _, _) => Flow::IndirectBranch,
                _ => Flow::IndirectCall,
            };
            Some((format!("jalr {}, {off}({})", xr(rd), xr(rs1)), flow))
        }
        OP_BRANCH => {
            let name = match f3 {
                0 => "beq",
                1 => "bne",
                4 => "blt",
                5 => "bge",
                6 => "bltu",
                7 => "bgeu",
                _ => return None,
            };
            let off = b_imm(w);
            Some((format!("{name} {}, {}, {off}", xr(rs1), xr(rs2)), Flow::CondBranch(off)))
        }
        OP_LOAD => {
            let name = ["lb", "lh", "lw", "ld", "lbu", "lhu", "lwu"].get(f3 as usize)?;
            next(format!("{name} {}, {}({})", xr(rd), i_imm(w), xr(rs1)))
        }
        OP_STORE => {
            let name = ["sb", "sh", "sw", "sd"].get(f3 as usize)?;
            next(format!("{name} {}, {}({})", xr(rs2), s_imm(w), xr(rs1)))
        }
        OP_IMM => op_imm(w, rd, rs1, f3),
        OP_IMM_32 => op_imm_32(w, rd, rs1, f3),
        OP_OP => op_reg(rd, rs1, rs2, f3, f7, false),
        OP_OP_32 => op_reg(rd, rs1, rs2, f3, f7, true),
        OP_AMO => amo(w, rd, rs1, rs2, f3),
        OP_MISC_MEM => misc_mem(w, rd, rs1, f3),
        OP_SYSTEM => system(w, rd, rs1, f3),
        OP_LOAD_FP => match f3 {
            1..=3 => next(format!("{} {}, {}({})", ["", "flh", "flw", "fld"][f3 as usize], fr(rd), i_imm(w), xr(rs1))),
            _ => vmem(w, false).and_then(next),
        },
        OP_STORE_FP => match f3 {
            1..=3 => next(format!("{} {}, {}({})", ["", "fsh", "fsw", "fsd"][f3 as usize], fr(rs2), s_imm(w), xr(rs1))),
            _ => vmem(w, true).and_then(next),
        },
        OP_MADD | OP_MSUB | OP_NMSUB | OP_NMADD => fma(w, rd, rs1, rs2, f3),
        OP_FP => op_fp(rd, rs1, rs2, f3, f7),
        OP_V => opv(w, rd, rs1, rs2, f3).and_then(next),
        _ => None,
    }
}

fn op_imm(w: u32, rd: u32, rs1: u32, f3: u32) -> Option<(String, Flow)> {
    let imm = i_imm(w);
    let top6 = bits(w, 31, 26);
    let sh = bits(w, 25, 20);
    let text = match f3 {
        0 => format!("addi {}, {}, {imm}", xr(rd), xr(rs1)),
        2 => format!("slti {}, {}, {imm}", xr(rd), xr(rs1)),
        3 => format!("sltiu {}, {}, {imm}", xr(rd), xr(rs1)),
        4 => format!("xori {}, {}, {imm}", xr(rd), xr(rs1)),
        6 => format!("ori {}, {}, {imm}", xr(rd), xr(rs1)),
        7 => format!("andi {}, {}, {imm}", xr(rd), xr(rs1)),
        1 => {
            let name = match (w >> 20, top6) {
                (0x600, _) => "clz",
                (0x601, _) => "ctz",
                (0x602, _) => "cpop",
                (0x604, _) => "sext.b",
                (0x605, _) => "sext.h",
                _ => "",
            };
            if !name.is_empty() {
                format!("{name} {}, {}", xr(rd), xr(rs1))
            } else {
                let name = match top6 {
                    0 => "slli",
                    0b001010 => "bseti",
                    0b010010 => "bclri",
                    0b011010 => "binvi",
                    _ => return None,
                };
                format!("{name} {}, {}, {sh}", xr(rd), xr(rs1))
            }
        }
        5 => match w >> 20 {
            0x6b8 => format!("rev8 {}, {}", xr(rd), xr(rs1)),
            0x287 => format!("orc.b {}, {}", xr(rd), xr(rs1)),
            _ => {
                let name = match top6 {
                    0 => "srli",
                    0b010000 => "srai",
                    0b011000 => "rori",
                    0b010010 => "bexti",
                    _ => return None,
                };
                format!("{name} {}, {}, {sh}", xr(rd), xr(rs1))
            }
        },
        _ => return None,
    };
    Some((text, Flow::Next))
}

fn op_imm_32(w: u32, rd: u32, rs1: u32, f3: u32) -> Option<(String, Flow)> {
    let top7 = bits(w, 31, 25);
    let top6 = bits(w, 31, 26);
    let sh5 = bits(w, 24, 20);
    let sh6 = bits(w, 25, 20);
    let text = match f3 {
        0 => format!("addiw {}, {}, {}", xr(rd), xr(rs1), i_imm(w)),
        1 => match (w >> 20, top7, top6) {
            (0x600, _, _) => format!("clzw {}, {}", xr(rd), xr(rs1)),
            (0x601, _, _) => format!("ctzw {}, {}", xr(rd), xr(rs1)),
            (0x602, _, _) => format!("cpopw {}, {}", xr(rd), xr(rs1)),
            (_, 0, _) => format!("slliw {}, {}, {sh5}", xr(rd), xr(rs1)),
            (_, _, 0b000010) => format!("slli.uw {}, {}, {sh6}", xr(rd), xr(rs1)),
            _ => return None,
        },
        5 => {
            let name = match top7 {
                0 => "srliw",
                0b0100000 => "sraiw",
                0b0110000 => "roriw",
                _ => return None,
            };
            format!("{name} {}, {}, {sh5}", xr(rd), xr(rs1))
        }
        _ => return None,
    };
    Some((text, Flow::Next))
}

fn op_reg(rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32, w32: bool) -> Option<(String, Flow)> {
    if w32 && f7 == 0x04 && f3 == 4 && rs2 == 0 {
        return Some((format!("zext.h {}, {}", xr(rd), xr(rs1)), Flow::Next));
    }
    let name = if !w32 {
        match (f7, f3) {
            (0, 0) => "add",
            (0x20, 0) => "sub",
            (0, 1) => "sll",
            (0, 2) => "slt",
            (0, 3) => "sltu",
            (0, 4) => "xor",
            (0, 5) => "srl",
            (0x20, 5) => "sra",
            (0, 6) => "or",
            (0, 7) => "and",
            (1, 0) => "mul",
            (1, 1) => "mulh",
            (1, 2) => "mulhsu",
            (1, 3) => "mulhu",
            (1, 4) => "div",
            (1, 5) => "divu",
            (1, 6) => "rem",
            (1, 7) => "remu",
            (0x10, 2) => "sh1add",
            (0x10, 4) => "sh2add",
            (0x10, 6) => "sh3add",
            (0x20, 7) => "andn",
            (0x20, 6) => "orn",
            (0x20, 4) => "xnor",
            (0x05, 4) => "min",
            (0x05, 5) => "minu",
            (0x05, 6) => "max",
            (0x05, 7) => "maxu",
            (0x30, 1) => "rol",
            (0x30, 5) => "ror",
            (0x14, 1) => "bset",
            (0x24, 1) => "bclr",
            (0x34, 1) => "binv",
            (0x24, 5) => "bext",
            _ => return None,
        }
    } else {
        match (f7, f3) {
            (0, 0) => "addw",
            (0x20, 0) => "subw",
            (0, 1) => "sllw",
            (0, 5) => "srlw",
            (0x20, 5) => "sraw",
            (1, 0) => "mulw",
            (1, 4) => "divw",
            (1, 5) => "divuw",
            (1, 6) => "remw",
            (1, 7) => "remuw",
            (0x04, 0) => "add.uw",
            (0x10, 2) => "sh1add.uw",
            (0x10, 4) => "sh2add.uw",
            (0x10, 6) => "sh3add.uw",
            (0x30, 1) => "rolw",
            (0x30, 5) => "rorw",
            _ => return None,
        }
    };
    Some((format!("{name} {}, {}, {}", xr(rd), xr(rs1), xr(rs2)), Flow::Next))
}

fn amo(w: u32, rd: u32, rs1: u32, rs2: u32, f3: u32) -> Option<(String, Flow)> {
    let size = match f3 {
        2 => "w",
        3 => "d",
        _ => return None,
    };
    let order = Order::suffix(bits(w, 26, 25));
    let text = match bits(w, 31, 27) {
        0b00010 if rs2 == 0 => format!("lr.{size}{order} {}, ({})", xr(rd), xr(rs1)),
        0b00010 => return None,
        f5 => {
            let name = match f5 {
                0b00011 => "sc",
                0b00001 => "amoswap",
                0b00000 => "amoadd",
                0b00100 => "amoxor",
                0b01100 => "amoand",
                0b01000 => "amoor",
                0b10000 => "amomin",
                0b10100 => "amomax",
                0b11000 => "amominu",
                0b11100 => "amomaxu",
                _ => return None,
            };
            format!("{name}.{size}{order} {}, {}, ({})", xr(rd), xr(rs2), xr(rs1))
        }
    };
    Some((text, Flow::Next))
}

fn misc_mem(w: u32, rd: u32, rs1: u32, f3: u32) -> Option<(String, Flow)> {
    if rd != 0 || rs1 != 0 {
        return None;
    }
    let text = match (f3, w >> 20) {
        (1, 0) => "fence.i".to_string(),
        (0, 0x010) => "pause".to_string(),
        (0, 0x833) => "fence.tso".to_string(),
        (0, imm) if imm >> 8 == 0 && imm >> 4 != 0 && imm & 15 != 0 => format!("fence {}, {}", Fence::name(imm >> 4), Fence::name(imm & 15)),
        _ => return None,
    };
    Some((text, Flow::Next))
}

fn system(w: u32, rd: u32, rs1: u32, f3: u32) -> Option<(String, Flow)> {
    match (f3, w) {
        (0, 0x0000_0073) => return Some(("ecall".into(), Flow::Trap)),
        (0, 0x0010_0073) => return Some(("ebreak".into(), Flow::Trap)),
        _ => {}
    }
    let csr = Csr::name(w >> 20);
    let text = match f3 {
        1 => format!("csrrw {}, {csr}, {}", xr(rd), xr(rs1)),
        2 => format!("csrrs {}, {csr}, {}", xr(rd), xr(rs1)),
        3 => format!("csrrc {}, {csr}, {}", xr(rd), xr(rs1)),
        5 => format!("csrrwi {}, {csr}, {rs1}", xr(rd)),
        6 => format!("csrrsi {}, {csr}, {rs1}", xr(rd)),
        7 => format!("csrrci {}, {csr}, {rs1}", xr(rd)),
        _ => return None,
    };
    Some((text, Flow::Next))
}

fn fmt_letter(fmt: u32) -> Option<&'static str> {
    match fmt {
        FMT_S => Some("s"),
        FMT_D => Some("d"),
        FMT_H => Some("h"),
        _ => None,
    }
}

/// `, rm` after a rounding operation, nothing for the dynamic mode.
fn rm_suffix(f3: u32) -> Option<String> {
    let rm = Rm::from_bits(f3)?;
    Some(if rm == Rm::Dyn { String::new() } else { format!(", {}", rm.name()) })
}

/// The rounding field of an exact conversion: `rne` is how the assemblers
/// write it, and prints as nothing.
fn exact_suffix(f3: u32) -> Option<String> {
    let rm = Rm::from_bits(f3)?;
    Some(if rm == Rm::Rne { String::new() } else { format!(", {}", rm.name()) })
}

fn fma(w: u32, rd: u32, rs1: u32, rs2: u32, f3: u32) -> Option<(String, Flow)> {
    let fmt = fmt_letter(bits(w, 26, 25)).filter(|&f| f != "h")?;
    let name = match w & 0x7f {
        OP_MADD => "fmadd",
        OP_MSUB => "fmsub",
        OP_NMSUB => "fnmsub",
        _ => "fnmadd",
    };
    let rs3 = bits(w, 31, 27);
    Some((format!("{name}.{fmt} {}, {}, {}, {}{}", fr(rd), fr(rs1), fr(rs2), fr(rs3), rm_suffix(f3)?), Flow::Next))
}

fn op_fp(rd: u32, rs1: u32, rs2: u32, f3: u32, f7: u32) -> Option<(String, Flow)> {
    let f5 = f7 >> 2;
    let fmt_bits = f7 & 3;
    let fmt = fmt_letter(fmt_bits)?;
    let sd = fmt != "h";
    let int = |t: u32| ["w", "wu", "l", "lu"].get(t as usize).copied();
    let text = match f5 {
        0b00000..=0b00011 if sd => {
            let name = ["fadd", "fsub", "fmul", "fdiv"][f5 as usize];
            format!("{name}.{fmt} {}, {}, {}{}", fr(rd), fr(rs1), fr(rs2), rm_suffix(f3)?)
        }
        0b01011 if sd && rs2 == 0 => format!("fsqrt.{fmt} {}, {}{}", fr(rd), fr(rs1), rm_suffix(f3)?),
        0b00100 if sd => {
            let name = ["fsgnj", "fsgnjn", "fsgnjx"].get(f3 as usize)?;
            format!("{name}.{fmt} {}, {}, {}", fr(rd), fr(rs1), fr(rs2))
        }
        0b00101 if sd => {
            let name = ["fmin", "fmax"].get(f3 as usize)?;
            format!("{name}.{fmt} {}, {}, {}", fr(rd), fr(rs1), fr(rs2))
        }
        0b10100 if sd => {
            let name = ["fle", "flt", "feq"].get(f3 as usize)?;
            format!("{name}.{fmt} {}, {}, {}", xr(rd), fr(rs1), fr(rs2))
        }
        0b11100 if rs2 == 0 && f3 == 1 && sd => format!("fclass.{fmt} {}, {}", xr(rd), fr(rs1)),
        0b11100 if rs2 == 0 && f3 == 0 => {
            let x = match fmt {
                "s" => "w",
                d => d,
            };
            format!("fmv.x.{x} {}, {}", xr(rd), fr(rs1))
        }
        0b11110 if rs2 == 0 && f3 == 0 => {
            let x = match fmt {
                "s" => "w",
                d => d,
            };
            format!("fmv.{x}.x {}, {}", fr(rd), xr(rs1))
        }
        0b11000 if sd => format!("fcvt.{}.{fmt} {}, {}{}", int(rs2)?, xr(rd), fr(rs1), rm_suffix(f3)?),
        0b11010 if sd => {
            let exact = fmt == "d" && rs2 <= CVT_WU;
            let rm = if exact { exact_suffix(f3)? } else { rm_suffix(f3)? };
            format!("fcvt.{fmt}.{} {}, {}{rm}", int(rs2)?, fr(rd), xr(rs1))
        }
        0b01000 => match (fmt_bits, rs2) {
            (FMT_S, 1) => format!("fcvt.s.d {}, {}{}", fr(rd), fr(rs1), rm_suffix(f3)?),
            (FMT_D, 0) => format!("fcvt.d.s {}, {}{}", fr(rd), fr(rs1), exact_suffix(f3)?),
            (FMT_H, 0b01000) => format!("fcvt.bf16.s {}, {}{}", fr(rd), fr(rs1), rm_suffix(f3)?),
            (FMT_S, 0b00110) => format!("fcvt.s.bf16 {}, {}{}", fr(rd), fr(rs1), exact_suffix(f3)?),
            _ => return None,
        },
        _ => return None,
    };
    Some((text, Flow::Next))
}

// ── vector ──────────────────────────────────────────────────────────────

fn vr(n: u32) -> String {
    format!("v{n}")
}

fn vmem(w: u32, store: bool) -> Option<String> {
    let nf = bits(w, 31, 29) + 1;
    if bits(w, 28, 28) != 0 {
        return None;
    }
    let mop = bits(w, 27, 26);
    let vm = bits(w, 25, 25);
    let field = bits(w, 24, 20);
    let rs1 = bits(w, 19, 15);
    let eew = Sew::from_mem_width(bits(w, 14, 12))?;
    let vd = bits(w, 11, 7);
    let e = eew.width();
    let mask = if vm == 0 { ", v0.t" } else { "" };
    let dir = if store { "s" } else { "l" };
    Some(match mop {
        MOP_UNIT => match field {
            0 => {
                let name = if nf == 1 { format!("v{dir}e{e}.v") } else { format!("v{dir}seg{nf}e{e}.v") };
                format!("{name} {}, ({}){mask}", vr(vd), xr(rs1))
            }
            UMOP_FF if !store => {
                let name = if nf == 1 { format!("vle{e}ff.v") } else { format!("vlseg{nf}e{e}ff.v") };
                format!("{name} {}, ({}){mask}", vr(vd), xr(rs1))
            }
            UMOP_WHOLE if vm == 1 && matches!(nf, 1 | 2 | 4 | 8) => {
                if store {
                    if eew != Sew::E8 {
                        return None;
                    }
                    format!("vs{nf}r.v {}, ({})", vr(vd), xr(rs1))
                } else {
                    format!("vl{nf}re{e}.v {}, ({})", vr(vd), xr(rs1))
                }
            }
            UMOP_MASK if vm == 1 && nf == 1 && eew == Sew::E8 => format!("v{dir}m.v {}, ({})", vr(vd), xr(rs1)),
            _ => return None,
        },
        MOP_STRIDED => {
            let name = if nf == 1 { format!("v{dir}se{e}.v") } else { format!("v{dir}sseg{nf}e{e}.v") };
            format!("{name} {}, ({}), {}{mask}", vr(vd), xr(rs1), xr(field))
        }
        _ => {
            let order = if mop == MOP_ORDERED { "o" } else { "u" };
            let name = if nf == 1 { format!("v{dir}{order}xei{e}.v") } else { format!("v{dir}{order}xseg{nf}ei{e}.v") };
            format!("{name} {}, ({}), {}{mask}", vr(vd), xr(rs1), vr(field))
        }
    })
}

fn opv(w: u32, rd: u32, rs1: u32, rs2: u32, f3: u32) -> Option<String> {
    if f3 == 7 {
        return if w >> 31 == 0 {
            let vt = Vtype::from_bits(bits(w, 30, 20))?;
            Some(format!("vsetvli {}, {}, {vt}", xr(rd), xr(rs1)))
        } else if w >> 30 == 3 {
            let vt = Vtype::from_bits(bits(w, 29, 20))?;
            Some(format!("vsetivli {}, {rs1}, {vt}", xr(rd)))
        } else if w >> 25 == 0b1000000 {
            Some(format!("vsetvl {}, {}, {}", xr(rd), xr(rs1), xr(rs2)))
        } else {
            None
        };
    }
    let funct6 = bits(w, 31, 26);
    let vm = bits(w, 25, 25);
    let op = VOPS.iter().find(|op| {
        op.funct6 == funct6
            && op.funct3 == f3
            && match op.form {
                Form::Unary(code) | Form::XUnary(code) => rs1 == code,
                Form::XS | Form::FS => rs1 == 0 && vm == 1,
                Form::SX | Form::SF | Form::MvV | Form::MvI => rs2 == 0 && vm == 1,
                Form::Id => rs2 == 0 && rs1 == 0b10001,
                Form::Whole(code) => rs1 == code && vm == 1,
                Form::MergeVv | Form::MergeVx | Form::MergeVi | Form::MergeVf => vm == 0,
                Form::Mm => vm == 1,
                _ => true,
            }
    })?;
    let n = op.name;
    let mask = if op.form.maskable() && vm == 0 { ", v0.t" } else { "" };
    let simm = sext(rs1, 5);
    let (vd, vs2) = (vr(rd), vr(rs2));
    Some(match op.form {
        Form::Vv | Form::Mm => format!("{n} {vd}, {vs2}, {}{mask}", vr(rs1)),
        Form::Vx => format!("{n} {vd}, {vs2}, {}{mask}", xr(rs1)),
        Form::Vf => format!("{n} {vd}, {vs2}, {}{mask}", fr(rs1)),
        Form::Vi => format!("{n} {vd}, {vs2}, {simm}{mask}"),
        Form::Vu => format!("{n} {vd}, {vs2}, {rs1}{mask}"),
        Form::MacVv => format!("{n} {vd}, {}, {vs2}{mask}", vr(rs1)),
        Form::MacVx => format!("{n} {vd}, {}, {vs2}{mask}", xr(rs1)),
        Form::MacVf => format!("{n} {vd}, {}, {vs2}{mask}", fr(rs1)),
        Form::Unary(_) | Form::Whole(_) => format!("{n} {vd}, {vs2}{mask}"),
        Form::XUnary(_) | Form::XS => format!("{n} {}, {vs2}{mask}", xr(rd)),
        Form::FS => format!("{n} {}, {vs2}", fr(rd)),
        Form::SX => format!("{n} {vd}, {}", xr(rs1)),
        Form::SF => format!("{n} {vd}, {}", fr(rs1)),
        Form::MvV => format!("{n} {vd}, {}", vr(rs1)),
        Form::MvI => format!("{n} {vd}, {simm}"),
        Form::Id => format!("{n} {vd}{mask}"),
        Form::MergeVv => format!("{n} {vd}, {vs2}, {}, v0", vr(rs1)),
        Form::MergeVx => format!("{n} {vd}, {vs2}, {}, v0", xr(rs1)),
        Form::MergeVi => format!("{n} {vd}, {vs2}, {simm}, v0"),
        Form::MergeVf => format!("{n} {vd}, {vs2}, {}, v0", fr(rs1)),
    })
}
