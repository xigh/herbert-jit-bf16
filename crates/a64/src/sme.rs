//! The Scalable Matrix Extension and the streaming SVE subset it runs with:
//! entering and leaving streaming mode, predicates, contiguous scalable
//! loads and stores, the outer products into ZA, and the moves between ZA
//! slices and vectors.
//!
//! Scalable registers are [`Z`] and predicates [`P`]; the element width of
//! an operand is named by [`Et`]. A ZA tile of 32-bit elements is
//! `za0.s..za3.s`; a slice is a tile, a direction, a base register among
//! `w12..w15` and an immediate offset.

use crate::asm::{check, Asm};
use crate::operand::Mem;
use crate::reg::{Z, P, R};

/// The element width of a scalable operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Et {
    /// Bytes.
    B = 0,
    /// Halves.
    H = 1,
    /// Singles.
    S = 2,
    /// Doubles.
    D = 3,
}

/// The direction of a ZA slice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slice {
    /// A row of the tile.
    H,
    /// A column of the tile.
    V,
}

/// A 32-bit ZA slice: tile `za0..za3`, direction, index register `w12..w15`, offset `0..3`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ZaS {
    /// Tile 0..=3.
    pub tile: u8,
    /// Row or column.
    pub dir: Slice,
    /// `w12..w15`.
    pub index: R,
    /// Immediate offset 0..=3.
    pub offset: u8,
}

fn slice_fields(za: ZaS) -> Result<u32, String> {
    let rs = za.index.zr()?;
    check((12..=15).contains(&rs) && !za.index.is_x(), || format!("{} is not w12..w15", za.index))?;
    check(za.tile < 4 && za.offset < 4, || "a 32-bit slice is za0..za3 with an offset 0..3".into())?;
    Ok((u32::from(za.dir == Slice::V) << 15) | ((rs - 12) << 13) | (u32::from(za.tile) << 2) | u32::from(za.offset))
}

fn zp(z: Z, p: P) -> Result<(u32, u32), String> {
    check(z.0 < 32 && p.0 < 16, || "register out of range".into())?;
    Ok((u32::from(z.0), u32::from(p.0)))
}

impl Asm {
    /// `smstart`: streaming SVE mode and ZA on.
    pub fn smstart(&mut self) {
        self.word(0xd503_477f);
    }
    /// `smstop`: streaming SVE mode and ZA off.
    pub fn smstop(&mut self) {
        self.word(0xd503_467f);
    }
    /// `smstart sm`: streaming mode only.
    pub fn smstart_sm(&mut self) {
        self.word(0xd503_437f);
    }
    /// `smstop sm`
    pub fn smstop_sm(&mut self) {
        self.word(0xd503_427f);
    }
    /// `smstart za`: ZA only.
    pub fn smstart_za(&mut self) {
        self.word(0xd503_457f);
    }
    /// `smstop za`
    pub fn smstop_za(&mut self) {
        self.word(0xd503_447f);
    }
    /// `zero {za}`: all of ZA to zero.
    pub fn zero_za(&mut self) {
        self.word(0xc008_00ff);
    }
    /// `zero {za<t>.s, …}`: the 32-bit tiles whose bit is set in `tiles` (bit t = za<t>.s).
    pub fn zero_za_s(&mut self, tiles: u8) {
        // A 32-bit tile t covers the 64-bit tiles t and t+4: mask bits 0x11 << t.
        let mask: u32 = (0..4).filter(|t| tiles & (1 << t) != 0).map(|t| 0x11u32 << t).sum();
        let w = check(tiles != 0 && tiles < 16, || "zero of no tile".into()).map(|_| 0xc008_0000 | mask);
        self.emit(w);
    }
    /// `ptrue pd.<t>` (pattern `all`).
    pub fn ptrue(&mut self, pd: P, t: Et) {
        let w = check(pd.0 < 16, || "predicate out of range".into()).map(|_| 0x2518_e3e0 | ((t as u32) << 22) | u32::from(pd.0));
        self.emit(w);
    }
    /// `whilelo pd.<t>, xn, xm`: lanes while `xn + i < xm`.
    pub fn whilelo(&mut self, pd: P, t: Et, rn: R, rm: R) {
        let w = (|| {
            check(pd.0 < 16 && rn.is_x() && rm.is_x(), || "whilelo pd, xn, xm".into())?;
            Ok(0x2520_1c00 | ((t as u32) << 22) | (rm.zr()? << 16) | (1 << 12) | (rn.zr()? << 5) | u32::from(pd.0))
        })();
        self.emit(w);
    }
    /// `rdsvl xd, #imm` — the streaming vector length in bytes, times `imm`.
    pub fn rdsvl(&mut self, rd: R, imm: i32) {
        let w = (|| {
            check((-32..32).contains(&imm) && rd.is_x(), || "rdsvl xd, #-32..31".into())?;
            Ok(0x04bf_5800 | (((imm as u32) & 63) << 5) | rd.zr()?)
        })();
        self.emit(w);
    }
    /// `addvl xd, xn, #imm` — `xn + imm · VL` (bytes).
    pub fn addvl(&mut self, rd: R, rn: R, imm: i32) {
        let w = (|| {
            check((-32..32).contains(&imm), || "addvl #-32..31".into())?;
            Ok(0x0420_5000 | (rn.sp()? << 16) | (((imm as u32) & 63) << 5) | rd.sp()?)
        })();
        self.emit(w);
    }
    /// `ld1<t> {zt.<t>}, pg/z, mem` for halves or singles: `[xn]`, `[xn, #imm, mul vl]`
    /// (the offset in vectors, `-8..7`), or `[xn, xm, lsl #size]`.
    pub fn ld1_z(&mut self, zt: Z, t: Et, pg: P, m: Mem) {
        self.emit(sve_ldst(true, zt, t, pg, m));
    }
    /// `st1<t> {zt.<t>}, pg, mem`, the addressing of [`Asm::ld1_z`].
    pub fn st1_z(&mut self, zt: Z, t: Et, pg: P, m: Mem) {
        self.emit(sve_ldst(false, zt, t, pg, m));
    }
    /// `ld1h {zt.s}, pg/z, mem` — halves zero-extended into 32-bit lanes.
    pub fn ld1h_s(&mut self, zt: Z, pg: P, m: Mem) {
        self.emit(sve_ldst_bases(zt, pg, m, 0xa4c0_a000, 0xa4c0_4000));
    }
    /// `st1h {zt.s}, pg, mem` — the low half of each 32-bit lane, a truncating store.
    pub fn st1h_s(&mut self, zt: Z, pg: P, m: Mem) {
        self.emit(sve_ldst_bases(zt, pg, m, 0xe4c0_e000, 0xe4c0_4000));
    }
    /// `bfmopa za<t>.s, pn/m, pm/m, zn.h, zm.h`: `za[i][j] += zn[2i]·zm[2j] + zn[2i+1]·zm[2j+1]`.
    pub fn bfmopa(&mut self, tile: u8, pn: P, pm: P, zn: Z, zm: Z) {
        self.emit(mopa(0x8180_0000, tile, pn, pm, zn, zm));
    }
    /// `fmopa za<t>.s, pn/m, pm/m, zn.s, zm.s`: `za[i][j] += zn[i]·zm[j]`.
    pub fn fmopa(&mut self, tile: u8, pn: P, pm: P, zn: Z, zm: Z) {
        self.emit(mopa(0x8080_0000, tile, pn, pm, zn, zm));
    }
    /// `fmops za<t>.s, pn/m, pm/m, zn.s, zm.s`: `za[i][j] -= zn[i]·zm[j]`.
    pub fn fmops(&mut self, tile: u8, pn: P, pm: P, zn: Z, zm: Z) {
        self.emit(mopa(0x8080_0010, tile, pn, pm, zn, zm));
    }
    /// `fmul zd.<t>, zn.<t>, zm.<t>` (SVE, unpredicated, singles or halves):
    /// lane by lane, IEEE under `FPCR`; legal in streaming mode.
    pub fn fmul_z(&mut self, zd: Z, t: Et, zn: Z, zm: Z) {
        let w = (|| {
            check(zd.0 < 32 && zn.0 < 32 && zm.0 < 32 && matches!(t, Et::H | Et::S), || "fmul z.<h|s>, z, z".into())?;
            Ok(0x6500_0800 | ((t as u32) << 22) | (u32::from(zm.0) << 16) | (u32::from(zn.0) << 5) | u32::from(zd.0))
        })();
        self.emit(w);
    }
    /// `dup zd.<t>, wn|xn` (SVE, the `mov` alias): a general register in every lane.
    pub fn dup_z(&mut self, zd: Z, t: Et, rn: R) {
        let w = (|| {
            check(zd.0 < 32 && (rn.is_x() == (t == Et::D)), || "dup z.<b|h|s>, wn or z.d, xn".into())?;
            Ok(0x0520_3800 | ((t as u32) << 22) | (rn.sp()? << 5) | u32::from(zd.0))
        })();
        self.emit(w);
    }
    /// `mova zd.s, pg/m, za<t><h|v>.s[ws, #off]` — one 32-bit slice into a vector.
    pub fn mova_from_za(&mut self, zd: Z, pg: P, za: ZaS) {
        let w = (|| {
            let (d, g) = zp(zd, pg)?;
            check(g < 8, || "mova takes p0..p7".into())?;
            // The slice fields sit at bits 8..5 here, the tile index shifted by one.
            let f = slice_fields(za)?;
            let tile_off = (f & 15) << 5;
            Ok(0xc082_0000 | (f & 0xe000) | (g << 10) | tile_off | d)
        })();
        self.emit(w);
    }
    /// `mova za<t><h|v>.s[ws, #off], pg/m, zn.s` — one vector into a 32-bit slice.
    pub fn mova_to_za(&mut self, za: ZaS, pg: P, zn: Z) {
        let w = (|| {
            let (n, g) = zp(zn, pg)?;
            check(g < 8, || "mova takes p0..p7".into())?;
            let f = slice_fields(za)?;
            Ok(0xc080_0000 | (f & 0xe000) | (g << 10) | (n << 5) | (f & 15))
        })();
        self.emit(w);
    }
    /// `ld1w {za<t><h|v>.s[ws, #off]}, pg/z, [xn, xm, lsl #2]` — one row or column of a tile from memory.
    pub fn ld1w_za(&mut self, za: ZaS, pg: P, rn: R, rm: R) {
        self.emit(za_ldst(0xe080_0000, za, pg, rn, rm));
    }
    /// `st1w {za<t><h|v>.s[ws, #off]}, pg, [xn, xm, lsl #2]`
    pub fn st1w_za(&mut self, za: ZaS, pg: P, rn: R, rm: R) {
        self.emit(za_ldst(0xe0a0_0000, za, pg, rn, rm));
    }
    /// `bfcvt zd.h, pg/m, zn.s`: singles to bf16 in the even halves.
    pub fn bfcvt_z(&mut self, zd: Z, pg: P, zn: Z) {
        let w = zp(zd, pg).and_then(|(d, g)| {
            check(g < 8 && zn.0 < 32, || "bfcvt z, p0..p7/m, z".into())?;
            Ok(0x658a_a000 | (g << 10) | (u32::from(zn.0) << 5) | d)
        });
        self.emit(w);
    }
}

fn mopa(base: u32, tile: u8, pn: P, pm: P, zn: Z, zm: Z) -> Result<u32, String> {
    check(tile < 4 && pn.0 < 8 && pm.0 < 8 && zn.0 < 32 && zm.0 < 32, || "mopa za0..3.s, p0..7, p0..7, z, z".into())?;
    Ok(base | (u32::from(zm.0) << 16) | (u32::from(pm.0) << 13) | (u32::from(pn.0) << 10) | (u32::from(zn.0) << 5) | u32::from(tile))
}

fn sve_ldst(load: bool, zt: Z, t: Et, pg: P, m: Mem) -> Result<u32, String> {
    let (imm_base, reg_base) = match (load, t) {
        (true, Et::H) => (0xa4a0_a000, 0xa4a0_4000),
        (true, Et::S) => (0xa540_a000, 0xa540_4000),
        (false, Et::H) => (0xe4a0_e000, 0xe4a0_4000),
        (false, Et::S) => (0xe540_e000, 0xe540_4000),
        _ => return Err(format!("{t:?} contiguous access not encoded")),
    };
    sve_ldst_bases(zt, pg, m, imm_base, reg_base)
}

/// A contiguous access of the given encodings, `[xn, #vl, mul vl]` or `[xn, xm, lsl #s]`.
fn sve_ldst_bases(zt: Z, pg: P, m: Mem, imm_base: u32, reg_base: u32) -> Result<u32, String> {
    let (tt, g) = zp(zt, pg)?;
    check(g < 8, || "contiguous loads and stores take p0..p7".into())?;
    match m {
        Mem::Off(rn, vl) => {
            check((-8..8).contains(&vl), || format!("offset {vl} vectors outside -8..7"))?;
            Ok(imm_base | ((((vl as i32) as u32) & 15) << 16) | (g << 10) | (rn.sp()? << 5) | tt)
        }
        Mem::Idx(rn, rm, crate::operand::Extend::Uxtx, true) => {
            check(!rm.is_zr(), || "the index register cannot be xzr".into())?;
            Ok(reg_base | (rm.zr()? << 16) | (g << 10) | (rn.sp()? << 5) | tt)
        }
        other => Err(format!("scalable access cannot address {other:?}")),
    }
}

fn za_ldst(base: u32, za: ZaS, pg: P, rn: R, rm: R) -> Result<u32, String> {
    check(pg.0 < 8 && rn.is_x() && rm.is_x(), || "ld1w/st1w za: p0..p7, [xn, xm, lsl #2]".into())?;
    let f = slice_fields(za)?;
    Ok(base | (rm.zr()? << 16) | (f & 0xe000) | (u32::from(pg.0) << 10) | (rn.sp()? << 5) | (f & 15))
}
