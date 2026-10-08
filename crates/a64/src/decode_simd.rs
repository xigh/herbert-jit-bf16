//! The Advanced SIMD and floating-point half of the decoder.

use crate::decode::{arr_name, field as bits, fmt_fp_imm, fp_scalar_name, gp_name, v_name};
use crate::operand::Cond;

fn fp_vec(n: u32, sz: u32, q: u32) -> Option<String> {
    if sz == 1 && q == 0 {
        return None;
    }
    Some(v_name(n, 2 + sz, q))
}

fn list(first: u32, count: u32, suffix: &str) -> String {
    let regs: Vec<String> = (0..count).map(|i| format!("v{}.{suffix}", (first + i) % 32)).collect();
    format!("{{ {} }}", regs.join(", "))
}

fn elem_letter(size: u32) -> char {
    ['b', 'h', 's', 'd'][size as usize]
}

pub(crate) fn decode_simd(w: u32) -> Option<String> {
    let d = bits(w, 4, 0);
    let n = bits(w, 9, 5);
    let m = bits(w, 20, 16);
    let q = bits(w, 30, 30);
    let u = bits(w, 29, 29);
    let size = bits(w, 23, 22);

    // ── structure loads and stores ─────────────────────────────────────
    if w & 0xbfbf_0000 == 0x0c00_0000 || w & 0xbfa0_0000 == 0x0c80_0000 {
        let count = match bits(w, 15, 12) {
            0b0111 => 1,
            0b1010 => 2,
            0b0110 => 3,
            0b0010 => 4,
            _ => return None,
        };
        let op = if bits(w, 22, 22) == 1 { "ld1" } else { "st1" };
        let regs = list(d, count, arr_name(bits(w, 11, 10), q));
        let base = gp_name(n, true, true);
        return Some(if bits(w, 23, 23) == 0 {
            format!("{op} {regs}, [{base}]")
        } else if m == 31 {
            format!("{op} {regs}, [{base}], #{}", count * if q == 1 { 16 } else { 8 })
        } else {
            format!("{op} {regs}, [{base}], {}", gp_name(m, true, false))
        });
    }
    if w & 0xbfff_f000 == 0x0d40_c000 || w & 0xbfe0_f000 == 0x0dc0_c000 {
        let s = bits(w, 11, 10);
        let regs = list(d, 1, arr_name(s, q));
        let base = gp_name(n, true, true);
        return Some(if bits(w, 23, 23) == 0 {
            format!("ld1r {regs}, [{base}]")
        } else if m == 31 {
            format!("ld1r {regs}, [{base}], #{}", 1 << s)
        } else {
            format!("ld1r {regs}, [{base}], {}", gp_name(m, true, false))
        });
    }
    if (w & 0xbf9f_0000 == 0x0d00_0000 || w & 0xbfa0_0000 == 0x0d80_0000) && matches!(bits(w, 15, 13), 0b000 | 0b010 | 0b100) {
        let s = bits(w, 12, 12);
        let sz = bits(w, 11, 10);
        let (letter, index, bytes) = match bits(w, 15, 13) {
            0b000 => ('b', (q << 3) | (s << 2) | sz, 1),
            0b010 if sz & 1 == 0 => ('h', (q << 2) | (s << 1) | (sz >> 1), 2),
            0b100 if sz == 0 => ('s', (q << 1) | s, 4),
            0b100 if sz == 1 && s == 0 => ('d', q, 8),
            _ => return None,
        };
        let op = if bits(w, 22, 22) == 1 { "ld1" } else { "st1" };
        let base = gp_name(n, true, true);
        let t = format!("{{ v{d}.{letter} }}[{index}]");
        return Some(if bits(w, 23, 23) == 0 {
            format!("{op} {t}, [{base}]")
        } else if m == 31 {
            format!("{op} {t}, [{base}], #{bytes}")
        } else {
            format!("{op} {t}, [{base}], {}", gp_name(m, true, false))
        });
    }

    // ── BF16 ───────────────────────────────────────────────────────────
    if w & 0xbfe0_fc00 == 0x2e40_fc00 {
        return Some(format!("bfdot {}, {}, {}", v_name(d, 2, q), v_name(n, 1, q), v_name(m, 1, q)));
    }
    if w & 0xffe0_fc00 == 0x6e40_ec00 {
        return Some(format!("bfmmla v{d}.4s, v{n}.8h, v{m}.8h"));
    }
    if w & 0xbfe0_fc00 == 0x2ec0_fc00 {
        let op = if q == 1 { "bfmlalt" } else { "bfmlalb" };
        return Some(format!("{op} v{d}.4s, v{n}.8h, v{m}.8h"));
    }
    if w & 0xbfc0_f400 == 0x0f40_f000 {
        let i = (bits(w, 11, 11) << 1) | bits(w, 21, 21);
        return Some(format!("bfdot {}, {}, v{m}.2h[{i}]", v_name(d, 2, q), v_name(n, 1, q)));
    }
    if w & 0xbfc0_f400 == 0x0fc0_f000 {
        let i = (bits(w, 11, 11) << 2) | (bits(w, 21, 21) << 1) | bits(w, 20, 20);
        let op = if q == 1 { "bfmlalt" } else { "bfmlalb" };
        return Some(format!("{op} v{d}.4s, v{n}.8h, v{}.h[{i}]", bits(w, 19, 16)));
    }
    if w & 0xbfff_fc00 == 0x0ea1_6800 {
        return Some(format!("bfcvtn{} v{d}.{}, v{n}.4s", if q == 1 { "2" } else { "" }, if q == 1 { "8h" } else { "4h" }));
    }

    // ── three same ─────────────────────────────────────────────────────
    if w & 0x9f20_0400 == 0x0e20_0400 {
        let opcode = bits(w, 15, 11);
        if opcode >= 0b11000 {
            let a = bits(w, 23, 23);
            let sz = bits(w, 22, 22);
            let op = match (u, a, opcode) {
                (0, 0, 0b11000) => "fmaxnm",
                (0, 0, 0b11001) => "fmla",
                (0, 0, 0b11010) => "fadd",
                (0, 0, 0b11011) => "fmulx",
                (0, 0, 0b11100) => "fcmeq",
                (0, 0, 0b11110) => "fmax",
                (0, 0, 0b11111) => "frecps",
                (0, 1, 0b11000) => "fminnm",
                (0, 1, 0b11001) => "fmls",
                (0, 1, 0b11010) => "fsub",
                (0, 1, 0b11110) => "fmin",
                (0, 1, 0b11111) => "frsqrts",
                (1, 0, 0b11000) => "fmaxnmp",
                (1, 0, 0b11010) => "faddp",
                (1, 0, 0b11011) => "fmul",
                (1, 0, 0b11100) => "fcmge",
                (1, 0, 0b11101) => "facge",
                (1, 0, 0b11110) => "fmaxp",
                (1, 0, 0b11111) => "fdiv",
                (1, 1, 0b11010) => "fabd",
                (1, 1, 0b11100) => "fcmgt",
                (1, 1, 0b11101) => "facgt",
                (1, 1, 0b11110) => "fminp",
                _ => return None,
            };
            return Some(format!("{op} {}, {}, {}", fp_vec(d, sz, q)?, fp_vec(n, sz, q)?, fp_vec(m, sz, q)?));
        }
        if opcode == 0b00011 {
            let op = [["and", "bic", "orr", "orn"], ["eor", "bsl", "bit", "bif"]][u as usize][size as usize];
            let a = if q == 1 { "16b" } else { "8b" };
            return Some(format!("{op} v{d}.{a}, v{n}.{a}, v{m}.{a}"));
        }
        let op = match (u, opcode) {
            (0, 0b10000) => "add",
            (1, 0b10000) => "sub",
            (0, 0b10011) => "mul",
            (1, 0b10001) => "cmeq",
            (0, 0b00110) => "cmgt",
            (1, 0b00110) => "cmhi",
            (1, 0b00111) => "cmhs",
            (0, 0b01000) => "sshl",
            (1, 0b01000) => "ushl",
            (1, 0b01100) => "umax",
            _ => return None,
        };
        if size == 3 && q == 0 {
            return None;
        }
        return Some(format!("{op} {}, {}, {}", v_name(d, size, q), v_name(n, size, q), v_name(m, size, q)));
    }

    // ── two-register miscellaneous ─────────────────────────────────────
    if w & 0x9f3e_0c00 == 0x0e20_0800 {
        let opcode = bits(w, 16, 12);
        let a = bits(w, 23, 23);
        let sz = bits(w, 22, 22);
        let fp = match (u, a, opcode) {
            (0, 0, 0b11000) => Some("frintn"),
            (0, 0, 0b11001) => Some("frintm"),
            (0, 1, 0b11000) => Some("frintp"),
            (0, 1, 0b11001) => Some("frintz"),
            (1, 0, 0b11000) => Some("frinta"),
            (1, 0, 0b11001) => Some("frintx"),
            (0, 0, 0b11010) => Some("fcvtns"),
            (0, 0, 0b11011) => Some("fcvtms"),
            (0, 1, 0b11010) => Some("fcvtps"),
            (0, 1, 0b11011) => Some("fcvtzs"),
            (1, 1, 0b11011) => Some("fcvtzu"),
            (0, 0, 0b11100) => Some("fcvtas"),
            (0, 0, 0b11101) => Some("scvtf"),
            (1, 0, 0b11101) => Some("ucvtf"),
            (0, 1, 0b11101) => Some("frecpe"),
            (1, 1, 0b11101) => Some("frsqrte"),
            (0, 1, 0b01111) => Some("fabs"),
            (1, 1, 0b01111) => Some("fneg"),
            (1, 1, 0b11111) => Some("fsqrt"),
            _ => None,
        };
        if let Some(op) = fp {
            return Some(format!("{op} {}, {}", fp_vec(d, sz, q)?, fp_vec(n, sz, q)?));
        }
        let zero = match (u, a, opcode) {
            (0, 1, 0b01100) => Some("fcmgt"),
            (0, 1, 0b01101) => Some("fcmeq"),
            (0, 1, 0b01110) => Some("fcmlt"),
            (1, 1, 0b01100) => Some("fcmge"),
            (1, 1, 0b01101) => Some("fcmle"),
            _ => None,
        };
        if let Some(op) = zero {
            return Some(format!("{op} {}, {}, #0.0", fp_vec(d, sz, q)?, fp_vec(n, sz, q)?));
        }
        let two = if q == 1 { "2" } else { "" };
        return match (u, opcode) {
            (0, 0b10110) if size < 2 => Some(format!("fcvtn{two} {}, {}", v_name(d, 1 + size, q), v_name(n, 2 + size, 1))),
            (0, 0b10111) if size < 2 => Some(format!("fcvtl{two} {}, {}", v_name(d, 2 + size, 1), v_name(n, 1 + size, q))),
            (0, 0b10010) if size < 3 => Some(format!("xtn{two} {}, {}", v_name(d, size, q), v_name(n, size + 1, 1))),
            (1, 0b10011) if size < 3 => Some(format!("shll{two} {}, {}, #{}", v_name(d, size + 1, 1), v_name(n, size, q), 8 << size)),
            (1, 0b00101) if size == 0 => Some(format!("not {}, {}", v_name(d, 0, q), v_name(n, 0, q))),
            (1, 0b01011) if !(size == 3 && q == 0) => Some(format!("neg {}, {}", v_name(d, size, q), v_name(n, size, q))),
            _ => None,
        };
    }

    // ── across lanes ───────────────────────────────────────────────────
    if w & 0x9f3e_0c00 == 0x0e30_0800 {
        let opcode = bits(w, 16, 12);
        return match (u, bits(w, 23, 23), opcode) {
            (1, 0, 0b01111) if q == 1 && size == 0 => Some(format!("fmaxv s{d}, v{n}.4s")),
            (1, 1, 0b01111) if q == 1 && size == 2 => Some(format!("fminv s{d}, v{n}.4s")),
            (1, 0, 0b01100) if q == 1 && size == 0 => Some(format!("fmaxnmv s{d}, v{n}.4s")),
            (0, _, 0b11011) if size < 3 => Some(format!("addv {}{d}, {}", elem_letter(size), v_name(n, size, q))),
            _ => None,
        };
    }

    // ── copy ───────────────────────────────────────────────────────────
    if w & 0x9fe0_8400 == 0x0e00_0400 {
        let imm5 = bits(w, 20, 16);
        let imm4 = bits(w, 14, 11);
        let size = imm5.trailing_zeros();
        if size > 3 {
            return None;
        }
        let index = imm5 >> (size + 1);
        let letter = elem_letter(size);
        return match (u, imm4) {
            (0, 0b0000) => Some(format!("dup {}, v{n}.{letter}[{index}]", v_name(d, size, q))),
            (0, 0b0001) => Some(format!("dup {}, {}", v_name(d, size, q), gp_name(n, size == 3, false))),
            (0, 0b0011) if q == 1 => Some(format!("ins v{d}.{letter}[{index}], {}", gp_name(n, size == 3, false))),
            (0, 0b0111) if (size == 3) == (q == 1) => Some(format!("umov {}, v{n}.{letter}[{index}]", gp_name(d, q == 1, false))),
            (1, _) if q == 1 => Some(format!("ins v{d}.{letter}[{index}], v{n}.{letter}[{}]", imm4 >> size)),
            _ => None,
        };
    }
    if w & 0xbf20_8c00 == 0x0e00_0800 {
        let op = match bits(w, 14, 12) {
            0b001 => "uzp1",
            0b010 => "trn1",
            0b011 => "zip1",
            0b101 => "uzp2",
            0b110 => "trn2",
            0b111 => "zip2",
            _ => return None,
        };
        return Some(format!("{op} {}, {}, {}", v_name(d, size, q), v_name(n, size, q), v_name(m, size, q)));
    }
    if w & 0xbfe0_8400 == 0x2e00_0000 {
        let a = if q == 1 { "16b" } else { "8b" };
        return Some(format!("ext v{d}.{a}, v{n}.{a}, v{m}.{a}, #{}", bits(w, 14, 11)));
    }
    if w & 0xbfe0_9c00 == 0x0e00_0000 {
        let a = if q == 1 { "16b" } else { "8b" };
        return Some(format!("tbl v{d}.{a}, {}, v{m}.{a}", list(n, bits(w, 14, 13) + 1, "16b")));
    }

    // ── modified immediate and shifts by immediate ─────────────────────
    if w & 0x9f80_0400 == 0x0f00_0400 {
        let immh = bits(w, 22, 19);
        if immh == 0 {
            let cmode = bits(w, 15, 12);
            let imm8 = (bits(w, 18, 16) << 5) | bits(w, 9, 5);
            if bits(w, 11, 11) != 0 {
                return None;
            }
            return match (u, cmode) {
                (op, c) if c & 0b1001 == 0 => {
                    let name = if op == 0 { "movi" } else { "mvni" };
                    Some(format!("{name} v{d}.{}, #{imm8:#x}, lsl #{}", if q == 1 { "4s" } else { "2s" }, (c >> 1) * 8))
                }
                (op, c) if c & 0b1101 == 0b1000 => {
                    let name = if op == 0 { "movi" } else { "mvni" };
                    Some(format!("{name} v{d}.{}, #{imm8:#x}, lsl #{}", if q == 1 { "8h" } else { "4h" }, ((c >> 1) & 1) * 8))
                }
                (0, 0b1110) => Some(format!("movi v{d}.{}, #{imm8:#x}", if q == 1 { "16b" } else { "8b" })),
                (1, 0b1110) => {
                    let mut v = 0u64;
                    for byte in 0..8 {
                        if imm8 & (1 << byte) != 0 {
                            v |= 0xff << (8 * byte);
                        }
                    }
                    Some(if q == 1 { format!("movi v{d}.2d, #{v:#x}") } else { format!("movi d{d}, #{v:#x}") })
                }
                (0, 0b1111) => Some(format!("fmov v{d}.{}, {}", if q == 1 { "4s" } else { "2s" }, fmt_fp_imm(imm8))),
                (1, 0b1111) if q == 1 => Some(format!("fmov v{d}.2d, {}", fmt_fp_imm(imm8))),
                _ => None,
            };
        }
        let size = 31 - immh.leading_zeros();
        let esize = 8u32 << size;
        let immhb = bits(w, 22, 16);
        let right = 2 * esize - immhb;
        let left = immhb - esize;
        let opcode = bits(w, 15, 11);
        if size == 3 && q == 0 && !matches!(opcode, 0b10000 | 0b10100) {
            return None;
        }
        let two = if q == 1 { "2" } else { "" };
        return match (u, opcode) {
            (0, 0b00000) => Some(format!("sshr {}, {}, #{right}", v_name(d, size, q), v_name(n, size, q))),
            (1, 0b00000) => Some(format!("ushr {}, {}, #{right}", v_name(d, size, q), v_name(n, size, q))),
            (1, 0b00010) => Some(format!("usra {}, {}, #{right}", v_name(d, size, q), v_name(n, size, q))),
            (0, 0b01010) => Some(format!("shl {}, {}, #{left}", v_name(d, size, q), v_name(n, size, q))),
            (1, 0b01010) => Some(format!("sli {}, {}, #{left}", v_name(d, size, q), v_name(n, size, q))),
            (1, 0b01000) => Some(format!("sri {}, {}, #{right}", v_name(d, size, q), v_name(n, size, q))),
            (0, 0b10000) if size < 3 => Some(format!("shrn{two} {}, {}, #{right}", v_name(d, size, q), v_name(n, size + 1, 1))),
            (0, 0b10100) if size < 3 => Some(format!("sshll{two} {}, {}, #{left}", v_name(d, size + 1, 1), v_name(n, size, q))),
            (1, 0b10100) if size < 3 => Some(format!("ushll{two} {}, {}, #{left}", v_name(d, size + 1, 1), v_name(n, size, q))),
            _ => None,
        };
    }

    // ── floating point by element ──────────────────────────────────────
    if w & 0x9f00_0400 == 0x0f00_0000 && bits(w, 23, 23) == 1 {
        let sz = bits(w, 22, 22);
        let op = match (u, bits(w, 15, 12)) {
            (0, 0b0001) => "fmla",
            (0, 0b0101) => "fmls",
            (0, 0b1001) => "fmul",
            (1, 0b1001) => "fmulx",
            _ => return None,
        };
        let (letter, index) = if sz == 0 {
            ('s', (bits(w, 11, 11) << 1) | bits(w, 21, 21))
        } else if bits(w, 21, 21) == 0 {
            ('d', bits(w, 11, 11))
        } else {
            return None;
        };
        return Some(format!("{op} {}, {}, v{m}.{letter}[{index}]", fp_vec(d, sz, q)?, fp_vec(n, sz, q)?));
    }
    if w & 0xdf00_0400 == 0x5f00_0000 && bits(w, 23, 23) == 1 {
        let sz = bits(w, 22, 22);
        let op = match (u, bits(w, 15, 12)) {
            (0, 0b0001) => "fmla",
            (0, 0b0101) => "fmls",
            (0, 0b1001) => "fmul",
            _ => return None,
        };
        let (letter, index) = if sz == 0 {
            ('s', (bits(w, 11, 11) << 1) | bits(w, 21, 21))
        } else if bits(w, 21, 21) == 0 {
            ('d', bits(w, 11, 11))
        } else {
            return None;
        };
        return Some(format!("{op} {letter}{d}, {letter}{n}, v{m}.{letter}[{index}]"));
    }

    // ── Advanced SIMD scalar ───────────────────────────────────────────
    if w & 0xdf20_0400 == 0x5e20_0400 {
        let a = bits(w, 23, 23);
        let letter = if bits(w, 22, 22) == 1 { 'd' } else { 's' };
        let op = match (u, a, bits(w, 15, 11)) {
            (0, 0, 0b11111) => "frecps",
            (0, 1, 0b11111) => "frsqrts",
            (1, 1, 0b11010) => "fabd",
            (0, 0, 0b11100) => "fcmeq",
            (1, 0, 0b11100) => "fcmge",
            (1, 1, 0b11100) => "fcmgt",
            _ => return None,
        };
        return Some(format!("{op} {letter}{d}, {letter}{n}, {letter}{m}"));
    }
    if w & 0xdf3e_0c00 == 0x5e20_0800 {
        let a = bits(w, 23, 23);
        let letter = if bits(w, 22, 22) == 1 { 'd' } else { 's' };
        let op = match (u, a, bits(w, 16, 12)) {
            (0, 1, 0b11101) => "frecpe",
            (1, 1, 0b11101) => "frsqrte",
            (0, 1, 0b11011) => "fcvtzs",
            (0, 0, 0b11010) => "fcvtns",
            (0, 0, 0b11011) => "fcvtms",
            (0, 0, 0b11101) => "scvtf",
            (1, 0, 0b11101) => "ucvtf",
            _ => return None,
        };
        return Some(format!("{op} {letter}{d}, {letter}{n}"));
    }
    if w & 0xdf3e_0c00 == 0x5e30_0800 {
        let sz = bits(w, 22, 22);
        let (letter, a) = if sz == 1 { ('d', "2d") } else { ('s', "2s") };
        let op = match (u, bits(w, 23, 23), bits(w, 16, 12)) {
            (1, 0, 0b01101) => "faddp",
            (1, 0, 0b01111) => "fmaxp",
            (1, 1, 0b01111) => "fminp",
            _ => return None,
        };
        return Some(format!("{op} {letter}{d}, v{n}.{a}"));
    }
    if w & 0xffe0_fc00 == 0x5e00_0400 {
        let imm5 = bits(w, 20, 16);
        let size = imm5.trailing_zeros();
        if size > 3 {
            return None;
        }
        let letter = elem_letter(size);
        return Some(format!("dup {letter}{d}, v{n}.{letter}[{}]", imm5 >> (size + 1)));
    }

    // ── scalar floating point ──────────────────────────────────────────
    let t = bits(w, 23, 22);
    if w & 0x7f20_fc00 == 0x1e20_0000 {
        let sf = bits(w, 31, 31) == 1;
        let fp = fp_scalar_name(t, n);
        return match (bits(w, 20, 19), bits(w, 18, 16)) {
            (0, 0b000) => Some(format!("fcvtns {}, {}", gp_name(d, sf, false), fp?)),
            (1, 0b000) => Some(format!("fcvtps {}, {}", gp_name(d, sf, false), fp?)),
            (2, 0b000) => Some(format!("fcvtms {}, {}", gp_name(d, sf, false), fp?)),
            (3, 0b000) => Some(format!("fcvtzs {}, {}", gp_name(d, sf, false), fp?)),
            (3, 0b001) => Some(format!("fcvtzu {}, {}", gp_name(d, sf, false), fp?)),
            (0, 0b010) => Some(format!("scvtf {}, {}", fp_scalar_name(t, d)?, gp_name(n, sf, false))),
            (0, 0b011) => Some(format!("ucvtf {}, {}", fp_scalar_name(t, d)?, gp_name(n, sf, false))),
            (0, 0b110) if (sf && t == 1) || (!sf && t == 0) => Some(format!("fmov {}, {}", gp_name(d, sf, false), fp?)),
            (0, 0b111) if (sf && t == 1) || (!sf && t == 0) => Some(format!("fmov {}, {}", fp_scalar_name(t, d)?, gp_name(n, sf, false))),
            _ => None,
        };
    }
    if w & 0xff20_3c07 == 0x1e20_2000 && bits(w, 15, 14) == 0 {
        let fp = fp_scalar_name(t, n)?;
        return match bits(w, 4, 3) {
            0b00 => Some(format!("fcmp {fp}, {}", fp_scalar_name(t, m)?)),
            0b01 if m == 0 => Some(format!("fcmp {fp}, #0.0")),
            0b10 => Some(format!("fcmpe {fp}, {}", fp_scalar_name(t, m)?)),
            0b11 if m == 0 => Some(format!("fcmpe {fp}, #0.0")),
            _ => None,
        };
    }
    if w & 0xff20_7c00 == 0x1e20_4000 {
        let opcode = bits(w, 20, 15);
        if t == 1 && opcode == 0b000110 {
            return Some(format!("bfcvt h{d}, s{n}"));
        }
        let op = match opcode {
            0b000000 => "fmov",
            0b000001 => "fabs",
            0b000010 => "fneg",
            0b000011 => "fsqrt",
            0b001000 => "frintn",
            0b001001 => "frintp",
            0b001010 => "frintm",
            0b001011 => "frintz",
            0b001100 => "frinta",
            0b001110 => "frintx",
            o if o >> 2 == 0b0001 => {
                let to = o & 3;
                if to == t || to == 2 {
                    return None;
                }
                return Some(format!("fcvt {}, {}", fp_scalar_name(to, d)?, fp_scalar_name(t, n)?));
            }
            _ => return None,
        };
        return Some(format!("{op} {}, {}", fp_scalar_name(t, d)?, fp_scalar_name(t, n)?));
    }
    if w & 0xff20_1fe0 == 0x1e20_1000 {
        return Some(format!("fmov {}, {}", fp_scalar_name(t, d)?, fmt_fp_imm(bits(w, 20, 13))));
    }
    if w & 0xff20_0c00 == 0x1e20_0800 {
        let op = match bits(w, 15, 12) {
            0b0000 => "fmul",
            0b0001 => "fdiv",
            0b0010 => "fadd",
            0b0011 => "fsub",
            0b0100 => "fmax",
            0b0101 => "fmin",
            0b0110 => "fmaxnm",
            0b0111 => "fminnm",
            0b1000 => "fnmul",
            _ => return None,
        };
        return Some(format!("{op} {}, {}, {}", fp_scalar_name(t, d)?, fp_scalar_name(t, n)?, fp_scalar_name(t, m)?));
    }
    if w & 0xff20_0c00 == 0x1e20_0c00 {
        return Some(format!(
            "fcsel {}, {}, {}, {}",
            fp_scalar_name(t, d)?,
            fp_scalar_name(t, n)?,
            fp_scalar_name(t, m)?,
            Cond::from_bits(bits(w, 15, 12)).name()
        ));
    }
    if w & 0xff00_0000 == 0x1f00_0000 {
        let op = [["fmadd", "fmsub"], ["fnmadd", "fnmsub"]][bits(w, 21, 21) as usize][bits(w, 15, 15) as usize];
        return Some(format!(
            "{op} {}, {}, {}, {}",
            fp_scalar_name(t, d)?,
            fp_scalar_name(t, n)?,
            fp_scalar_name(t, m)?,
            fp_scalar_name(t, bits(w, 14, 10))?
        ));
    }
    None
}
