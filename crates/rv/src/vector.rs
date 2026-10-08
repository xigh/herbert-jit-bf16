//! RVV 1.0: configuration (`vsetvli`…), loads and stores of every
//! addressing mode, and the arithmetic of `OP-V` — integer, fixed widths,
//! floating point, widening and narrowing, reductions, permutations, masks —
//! with the bf16 extensions Zvfbfmin (`vfwcvtbf16`, `vfncvtbf16`) and
//! Zvfbfwma (`vfwmaccbf16`).
//!
//! The arithmetic is one table ([`VOPS`]) from which both the encoder
//! methods and the decoder are generated, so the two cannot disagree on a
//! `funct6`. Every method emits the unmasked form; [`Asm::masked`] wraps one
//! call to give it `v0.t`. Operands follow the assembler's order: `vd, vs2,
//! vs1` for most, `vd, vs1, vs2` for the multiply-adds (`vfmacc.vv vd, vs1,
//! vs2` is `vd += vs1·vs2`).

use crate::asm::{check, Asm};
use crate::base::{OP_LOAD_FP, OP_STORE_FP, OP_V};
use crate::reg::{Sew, Vtype, F, V, X};

pub(crate) const OPIVV: u32 = 0;
pub(crate) const OPFVV: u32 = 1;
pub(crate) const OPMVV: u32 = 2;
pub(crate) const OPIVI: u32 = 3;
pub(crate) const OPIVX: u32 = 4;
pub(crate) const OPFVF: u32 = 5;
pub(crate) const OPMVX: u32 = 6;

/// The operand shape of an `OP-V` instruction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Form {
    /// `vd, vs2, vs1`.
    Vv,
    /// `vd, vs2, rs1`.
    Vx,
    /// `vd, vs2, fs1`.
    Vf,
    /// `vd, vs2, simm5`.
    Vi,
    /// `vd, vs2, uimm5`.
    Vu,
    /// `vd, vs1, vs2` (multiply-add).
    MacVv,
    /// `vd, rs1, vs2`.
    MacVx,
    /// `vd, fs1, vs2`.
    MacVf,
    /// `vd, vs2`, the `vs1` field a sub-opcode.
    Unary(u32),
    /// `rd, vs2` into an integer register, maskable (`vcpop.m`, `vfirst.m`).
    XUnary(u32),
    /// `vmv.x.s rd, vs2`.
    XS,
    /// `vfmv.f.s fd, vs2`.
    FS,
    /// `vd, rs1` with `vs2 = 0`, unmasked (`vmv.s.x`, `vmv.v.x`).
    SX,
    /// `vd, fs1` with `vs2 = 0`, unmasked (`vfmv.s.f`, `vfmv.v.f`).
    SF,
    /// `vmv.v.v vd, vs1`.
    MvV,
    /// `vmv.v.i vd, simm5`.
    MvI,
    /// `vid.v vd`.
    Id,
    /// `vmv<n>r.v vd, vs2`: the immediate is n − 1.
    Whole(u32),
    /// `vd, vs2, vs1, v0` (always under `v0`).
    MergeVv,
    /// `vd, vs2, rs1, v0`.
    MergeVx,
    /// `vd, vs2, simm5, v0`.
    MergeVi,
    /// `vd, vs2, fs1, v0`.
    MergeVf,
    /// `vd, vs2, vs1`, never masked (mask logic, `vcompress.vm`).
    Mm,
}

impl Form {
    /// Does the instruction take `v0.t`?
    pub(crate) fn maskable(self) -> bool {
        matches!(
            self,
            Form::Vv | Form::Vx | Form::Vf | Form::Vi | Form::Vu | Form::MacVv | Form::MacVx | Form::MacVf | Form::Unary(_) | Form::XUnary(_) | Form::Id
        )
    }
}

/// One `OP-V` instruction.
#[derive(Clone, Copy, Debug)]
pub(crate) struct VOp {
    pub name: &'static str,
    pub funct6: u32,
    pub funct3: u32,
    pub form: Form,
}

fn vn(r: V) -> u32 {
    u32::from(r.0)
}
fn xn(r: X) -> u32 {
    u32::from(r.0)
}
fn fnum(r: F) -> u32 {
    u32::from(r.0)
}

fn opv(funct6: u32, vm: u32, vs2: u32, vs1: u32, funct3: u32, vd: u32) -> u32 {
    funct6 << 26 | vm << 25 | vs2 << 20 | vs1 << 15 | funct3 << 12 | vd << 7 | OP_V
}

fn simm5(imm: i64) -> Result<u32, String> {
    check((-16..16).contains(&imm), || format!("immediate {imm} is not a signed 5-bit value"))?;
    Ok(imm as u32 & 31)
}
fn uimm5(imm: u32) -> Result<u32, String> {
    check(imm < 32, || format!("immediate {imm} exceeds 5 bits"))?;
    Ok(imm)
}

impl Asm {
    /// An `OP-V` word with the mask the form allows.
    #[allow(clippy::too_many_arguments)]
    fn vop(&mut self, name: &str, maskable: bool, fixed_vm: Option<u32>, funct6: u32, funct3: u32, vs2: u32, vs1: Result<u32, String>, vd: u32) {
        let vm = match fixed_vm {
            Some(vm) => self.no_mask(name).map(|_| vm),
            None if maskable => Ok(self.take_vm()),
            None => self.no_mask(name).map(|_| 1),
        };
        let w = vm.and_then(|vm| vs1.map(|vs1| opv(funct6, vm, vs2, vs1, funct3, vd)));
        self.emit(w);
    }
}

macro_rules! vop_fn {
    (Vv, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, vs1`.")]
        pub fn $m(&mut self, vd: V, vs2: V, vs1: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(vn(vs1)), vn(vd)); }
    };
    (Vx, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, rs1`.")]
        pub fn $m(&mut self, vd: V, vs2: V, rs1: X) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(xn(rs1)), vn(vd)); }
    };
    (Vf, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, fs1`.")]
        pub fn $m(&mut self, vd: V, vs2: V, fs1: F) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(fnum(fs1)), vn(vd)); }
    };
    (Vi, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, simm5`.")]
        pub fn $m(&mut self, vd: V, vs2: V, imm: i64) { self.vop($n, true, None, $f6, $f3, vn(vs2), simm5(imm), vn(vd)); }
    };
    (Vu, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, uimm5`.")]
        pub fn $m(&mut self, vd: V, vs2: V, imm: u32) { self.vop($n, true, None, $f6, $f3, vn(vs2), uimm5(imm), vn(vd)); }
    };
    (MacVv, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs1, vs2`.")]
        pub fn $m(&mut self, vd: V, vs1: V, vs2: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(vn(vs1)), vn(vd)); }
    };
    (MacVx, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, rs1, vs2`.")]
        pub fn $m(&mut self, vd: V, rs1: X, vs2: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(xn(rs1)), vn(vd)); }
    };
    (MacVf, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, fs1, vs2`.")]
        pub fn $m(&mut self, vd: V, fs1: F, vs2: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok(fnum(fs1)), vn(vd)); }
    };
    (Unary, $m:ident, $n:expr, $f6:expr, $f3:expr, $code:expr) => {
        #[doc = concat!("`", $n, " vd, vs2`.")]
        pub fn $m(&mut self, vd: V, vs2: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok($code), vn(vd)); }
    };
    (XUnary, $m:ident, $n:expr, $f6:expr, $f3:expr, $code:expr) => {
        #[doc = concat!("`", $n, " rd, vs2`.")]
        pub fn $m(&mut self, rd: X, vs2: V) { self.vop($n, true, None, $f6, $f3, vn(vs2), Ok($code), xn(rd)); }
    };
    (XS, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " rd, vs2`: element 0.")]
        pub fn $m(&mut self, rd: X, vs2: V) { self.vop($n, false, None, $f6, $f3, vn(vs2), Ok(0), xn(rd)); }
    };
    (FS, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " fd, vs2`: element 0.")]
        pub fn $m(&mut self, fd: F, vs2: V) { self.vop($n, false, None, $f6, $f3, vn(vs2), Ok(0), fnum(fd)); }
    };
    (SX, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, rs1`.")]
        pub fn $m(&mut self, vd: V, rs1: X) { self.vop($n, false, None, $f6, $f3, 0, Ok(xn(rs1)), vn(vd)); }
    };
    (SF, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, fs1`.")]
        pub fn $m(&mut self, vd: V, fs1: F) { self.vop($n, false, None, $f6, $f3, 0, Ok(fnum(fs1)), vn(vd)); }
    };
    (MvV, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs1`.")]
        pub fn $m(&mut self, vd: V, vs1: V) { self.vop($n, false, None, $f6, $f3, 0, Ok(vn(vs1)), vn(vd)); }
    };
    (MvI, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, simm5`.")]
        pub fn $m(&mut self, vd: V, imm: i64) { self.vop($n, false, None, $f6, $f3, 0, simm5(imm), vn(vd)); }
    };
    (Id, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd`: each element its index.")]
        pub fn $m(&mut self, vd: V) { self.vop($n, true, None, $f6, $f3, 0, Ok(0b10001), vn(vd)); }
    };
    (Whole, $m:ident, $n:expr, $f6:expr, $f3:expr, $code:expr) => {
        #[doc = concat!("`", $n, " vd, vs2`: whole registers, whatever `vl`.")]
        pub fn $m(&mut self, vd: V, vs2: V) { self.vop($n, false, None, $f6, $f3, vn(vs2), Ok($code), vn(vd)); }
    };
    (MergeVv, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, vs1, v0`: `v0 ? vs1 : vs2`.")]
        pub fn $m(&mut self, vd: V, vs2: V, vs1: V) { self.vop($n, false, Some(0), $f6, $f3, vn(vs2), Ok(vn(vs1)), vn(vd)); }
    };
    (MergeVx, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, rs1, v0`.")]
        pub fn $m(&mut self, vd: V, vs2: V, rs1: X) { self.vop($n, false, Some(0), $f6, $f3, vn(vs2), Ok(xn(rs1)), vn(vd)); }
    };
    (MergeVi, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, simm5, v0`.")]
        pub fn $m(&mut self, vd: V, vs2: V, imm: i64) { self.vop($n, false, Some(0), $f6, $f3, vn(vs2), simm5(imm), vn(vd)); }
    };
    (MergeVf, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, fs1, v0`.")]
        pub fn $m(&mut self, vd: V, vs2: V, fs1: F) { self.vop($n, false, Some(0), $f6, $f3, vn(vs2), Ok(fnum(fs1)), vn(vd)); }
    };
    (Mm, $m:ident, $n:expr, $f6:expr, $f3:expr) => {
        #[doc = concat!("`", $n, " vd, vs2, vs1`.")]
        pub fn $m(&mut self, vd: V, vs2: V, vs1: V) { self.vop($n, false, None, $f6, $f3, vn(vs2), Ok(vn(vs1)), vn(vd)); }
    };
}

macro_rules! vops {
    ($($m:ident = ($n:literal, $f6:expr, $f3:ident, $form:ident $(($code:expr))?);)*) => {
        /// Every `OP-V` instruction of the crate.
        pub(crate) static VOPS: &[VOp] = &[
            $( VOp { name: $n, funct6: $f6, funct3: $f3, form: Form::$form $(($code))? }, )*
        ];
        impl Asm {
            $( vop_fn!($form, $m, $n, $f6, $f3 $(, $code)?); )*
        }
    };
}

vops! {
    // ── integer ───────────────────────────────────────────────────────
    vadd_vv = ("vadd.vv", 0b000000, OPIVV, Vv);
    vadd_vx = ("vadd.vx", 0b000000, OPIVX, Vx);
    vadd_vi = ("vadd.vi", 0b000000, OPIVI, Vi);
    vsub_vv = ("vsub.vv", 0b000010, OPIVV, Vv);
    vsub_vx = ("vsub.vx", 0b000010, OPIVX, Vx);
    vrsub_vx = ("vrsub.vx", 0b000011, OPIVX, Vx);
    vrsub_vi = ("vrsub.vi", 0b000011, OPIVI, Vi);
    vminu_vv = ("vminu.vv", 0b000100, OPIVV, Vv);
    vminu_vx = ("vminu.vx", 0b000100, OPIVX, Vx);
    vmin_vv = ("vmin.vv", 0b000101, OPIVV, Vv);
    vmin_vx = ("vmin.vx", 0b000101, OPIVX, Vx);
    vmaxu_vv = ("vmaxu.vv", 0b000110, OPIVV, Vv);
    vmaxu_vx = ("vmaxu.vx", 0b000110, OPIVX, Vx);
    vmax_vv = ("vmax.vv", 0b000111, OPIVV, Vv);
    vmax_vx = ("vmax.vx", 0b000111, OPIVX, Vx);
    vand_vv = ("vand.vv", 0b001001, OPIVV, Vv);
    vand_vx = ("vand.vx", 0b001001, OPIVX, Vx);
    vand_vi = ("vand.vi", 0b001001, OPIVI, Vi);
    vor_vv = ("vor.vv", 0b001010, OPIVV, Vv);
    vor_vx = ("vor.vx", 0b001010, OPIVX, Vx);
    vor_vi = ("vor.vi", 0b001010, OPIVI, Vi);
    vxor_vv = ("vxor.vv", 0b001011, OPIVV, Vv);
    vxor_vx = ("vxor.vx", 0b001011, OPIVX, Vx);
    vxor_vi = ("vxor.vi", 0b001011, OPIVI, Vi);
    vsll_vv = ("vsll.vv", 0b100101, OPIVV, Vv);
    vsll_vx = ("vsll.vx", 0b100101, OPIVX, Vx);
    vsll_vi = ("vsll.vi", 0b100101, OPIVI, Vu);
    vsrl_vv = ("vsrl.vv", 0b101000, OPIVV, Vv);
    vsrl_vx = ("vsrl.vx", 0b101000, OPIVX, Vx);
    vsrl_vi = ("vsrl.vi", 0b101000, OPIVI, Vu);
    vsra_vv = ("vsra.vv", 0b101001, OPIVV, Vv);
    vsra_vx = ("vsra.vx", 0b101001, OPIVX, Vx);
    vsra_vi = ("vsra.vi", 0b101001, OPIVI, Vu);
    vnsrl_wv = ("vnsrl.wv", 0b101100, OPIVV, Vv);
    vnsrl_wx = ("vnsrl.wx", 0b101100, OPIVX, Vx);
    vnsrl_wi = ("vnsrl.wi", 0b101100, OPIVI, Vu);
    vnsra_wi = ("vnsra.wi", 0b101101, OPIVI, Vu);
    vmseq_vv = ("vmseq.vv", 0b011000, OPIVV, Vv);
    vmseq_vx = ("vmseq.vx", 0b011000, OPIVX, Vx);
    vmseq_vi = ("vmseq.vi", 0b011000, OPIVI, Vi);
    vmsne_vv = ("vmsne.vv", 0b011001, OPIVV, Vv);
    vmsne_vx = ("vmsne.vx", 0b011001, OPIVX, Vx);
    vmsne_vi = ("vmsne.vi", 0b011001, OPIVI, Vi);
    vmsltu_vv = ("vmsltu.vv", 0b011010, OPIVV, Vv);
    vmsltu_vx = ("vmsltu.vx", 0b011010, OPIVX, Vx);
    vmslt_vv = ("vmslt.vv", 0b011011, OPIVV, Vv);
    vmslt_vx = ("vmslt.vx", 0b011011, OPIVX, Vx);
    vmsleu_vv = ("vmsleu.vv", 0b011100, OPIVV, Vv);
    vmsleu_vx = ("vmsleu.vx", 0b011100, OPIVX, Vx);
    vmsleu_vi = ("vmsleu.vi", 0b011100, OPIVI, Vi);
    vmsle_vv = ("vmsle.vv", 0b011101, OPIVV, Vv);
    vmsle_vx = ("vmsle.vx", 0b011101, OPIVX, Vx);
    vmsle_vi = ("vmsle.vi", 0b011101, OPIVI, Vi);
    vmsgtu_vx = ("vmsgtu.vx", 0b011110, OPIVX, Vx);
    vmsgtu_vi = ("vmsgtu.vi", 0b011110, OPIVI, Vi);
    vmsgt_vx = ("vmsgt.vx", 0b011111, OPIVX, Vx);
    vmsgt_vi = ("vmsgt.vi", 0b011111, OPIVI, Vi);
    vmerge_vvm = ("vmerge.vvm", 0b010111, OPIVV, MergeVv);
    vmerge_vxm = ("vmerge.vxm", 0b010111, OPIVX, MergeVx);
    vmerge_vim = ("vmerge.vim", 0b010111, OPIVI, MergeVi);
    vmv_v_v = ("vmv.v.v", 0b010111, OPIVV, MvV);
    vmv_v_x = ("vmv.v.x", 0b010111, OPIVX, SX);
    vmv_v_i = ("vmv.v.i", 0b010111, OPIVI, MvI);
    vmv1r_v = ("vmv1r.v", 0b100111, OPIVI, Whole(0));
    vmv2r_v = ("vmv2r.v", 0b100111, OPIVI, Whole(1));
    vmv4r_v = ("vmv4r.v", 0b100111, OPIVI, Whole(3));
    vmv8r_v = ("vmv8r.v", 0b100111, OPIVI, Whole(7));
    vmul_vv = ("vmul.vv", 0b100101, OPMVV, Vv);
    vmul_vx = ("vmul.vx", 0b100101, OPMVX, Vx);
    vmulhu_vv = ("vmulhu.vv", 0b100100, OPMVV, Vv);
    vmulh_vv = ("vmulh.vv", 0b100111, OPMVV, Vv);
    vdivu_vv = ("vdivu.vv", 0b100000, OPMVV, Vv);
    vdivu_vx = ("vdivu.vx", 0b100000, OPMVX, Vx);
    vremu_vx = ("vremu.vx", 0b100010, OPMVX, Vx);
    vmacc_vv = ("vmacc.vv", 0b101101, OPMVV, MacVv);
    vmacc_vx = ("vmacc.vx", 0b101101, OPMVX, MacVx);
    vmadd_vx = ("vmadd.vx", 0b101001, OPMVX, MacVx);
    vwaddu_vv = ("vwaddu.vv", 0b110000, OPMVV, Vv);
    vwmulu_vv = ("vwmulu.vv", 0b111000, OPMVV, Vv);
    vwmulu_vx = ("vwmulu.vx", 0b111000, OPMVX, Vx);
    vwmaccu_vx = ("vwmaccu.vx", 0b111100, OPMVX, MacVx);
    vzext_vf2 = ("vzext.vf2", 0b010010, OPMVV, Unary(0b00110));
    vsext_vf2 = ("vsext.vf2", 0b010010, OPMVV, Unary(0b00111));
    vzext_vf4 = ("vzext.vf4", 0b010010, OPMVV, Unary(0b00100));
    vredsum_vs = ("vredsum.vs", 0b000000, OPMVV, Vv);
    vredmaxu_vs = ("vredmaxu.vs", 0b000110, OPMVV, Vv);
    vredmax_vs = ("vredmax.vs", 0b000111, OPMVV, Vv);
    vredminu_vs = ("vredminu.vs", 0b000100, OPMVV, Vv);
    vmv_x_s = ("vmv.x.s", 0b010000, OPMVV, XS);
    vmv_s_x = ("vmv.s.x", 0b010000, OPMVX, SX);
    vcpop_m = ("vcpop.m", 0b010000, OPMVV, XUnary(0b10000));
    vfirst_m = ("vfirst.m", 0b010000, OPMVV, XUnary(0b10001));
    vid_v = ("vid.v", 0b010100, OPMVV, Id);
    viota_m = ("viota.m", 0b010100, OPMVV, Unary(0b10000));
    vmand_mm = ("vmand.mm", 0b011001, OPMVV, Mm);
    vmnand_mm = ("vmnand.mm", 0b011101, OPMVV, Mm);
    vmandn_mm = ("vmandn.mm", 0b011000, OPMVV, Mm);
    vmor_mm = ("vmor.mm", 0b011010, OPMVV, Mm);
    vmxor_mm = ("vmxor.mm", 0b011011, OPMVV, Mm);
    vmorn_mm = ("vmorn.mm", 0b011100, OPMVV, Mm);
    vmnor_mm = ("vmnor.mm", 0b011110, OPMVV, Mm);
    vmxnor_mm = ("vmxnor.mm", 0b011111, OPMVV, Mm);
    vcompress_vm = ("vcompress.vm", 0b010111, OPMVV, Mm);
    // ── permutations ──────────────────────────────────────────────────
    vrgather_vv = ("vrgather.vv", 0b001100, OPIVV, Vv);
    vrgather_vx = ("vrgather.vx", 0b001100, OPIVX, Vx);
    vrgather_vi = ("vrgather.vi", 0b001100, OPIVI, Vu);
    vrgatherei16_vv = ("vrgatherei16.vv", 0b001110, OPIVV, Vv);
    vslideup_vx = ("vslideup.vx", 0b001110, OPIVX, Vx);
    vslideup_vi = ("vslideup.vi", 0b001110, OPIVI, Vu);
    vslidedown_vx = ("vslidedown.vx", 0b001111, OPIVX, Vx);
    vslidedown_vi = ("vslidedown.vi", 0b001111, OPIVI, Vu);
    vslide1up_vx = ("vslide1up.vx", 0b001110, OPMVX, Vx);
    vslide1down_vx = ("vslide1down.vx", 0b001111, OPMVX, Vx);
    vfslide1up_vf = ("vfslide1up.vf", 0b001110, OPFVF, Vf);
    vfslide1down_vf = ("vfslide1down.vf", 0b001111, OPFVF, Vf);
    // ── floating point ────────────────────────────────────────────────
    vfadd_vv = ("vfadd.vv", 0b000000, OPFVV, Vv);
    vfadd_vf = ("vfadd.vf", 0b000000, OPFVF, Vf);
    vfsub_vv = ("vfsub.vv", 0b000010, OPFVV, Vv);
    vfsub_vf = ("vfsub.vf", 0b000010, OPFVF, Vf);
    vfrsub_vf = ("vfrsub.vf", 0b100111, OPFVF, Vf);
    vfmul_vv = ("vfmul.vv", 0b100100, OPFVV, Vv);
    vfmul_vf = ("vfmul.vf", 0b100100, OPFVF, Vf);
    vfdiv_vv = ("vfdiv.vv", 0b100000, OPFVV, Vv);
    vfdiv_vf = ("vfdiv.vf", 0b100000, OPFVF, Vf);
    vfrdiv_vf = ("vfrdiv.vf", 0b100001, OPFVF, Vf);
    vfmin_vv = ("vfmin.vv", 0b000100, OPFVV, Vv);
    vfmin_vf = ("vfmin.vf", 0b000100, OPFVF, Vf);
    vfmax_vv = ("vfmax.vv", 0b000110, OPFVV, Vv);
    vfmax_vf = ("vfmax.vf", 0b000110, OPFVF, Vf);
    vfsgnj_vv = ("vfsgnj.vv", 0b001000, OPFVV, Vv);
    vfsgnj_vf = ("vfsgnj.vf", 0b001000, OPFVF, Vf);
    vfsgnjn_vv = ("vfsgnjn.vv", 0b001001, OPFVV, Vv);
    vfsgnjx_vv = ("vfsgnjx.vv", 0b001010, OPFVV, Vv);
    vfmacc_vv = ("vfmacc.vv", 0b101100, OPFVV, MacVv);
    vfmacc_vf = ("vfmacc.vf", 0b101100, OPFVF, MacVf);
    vfnmacc_vv = ("vfnmacc.vv", 0b101101, OPFVV, MacVv);
    vfnmacc_vf = ("vfnmacc.vf", 0b101101, OPFVF, MacVf);
    vfmsac_vv = ("vfmsac.vv", 0b101110, OPFVV, MacVv);
    vfmsac_vf = ("vfmsac.vf", 0b101110, OPFVF, MacVf);
    vfnmsac_vv = ("vfnmsac.vv", 0b101111, OPFVV, MacVv);
    vfnmsac_vf = ("vfnmsac.vf", 0b101111, OPFVF, MacVf);
    vfmadd_vv = ("vfmadd.vv", 0b101000, OPFVV, MacVv);
    vfmadd_vf = ("vfmadd.vf", 0b101000, OPFVF, MacVf);
    vfnmadd_vv = ("vfnmadd.vv", 0b101001, OPFVV, MacVv);
    vfmsub_vv = ("vfmsub.vv", 0b101010, OPFVV, MacVv);
    vfmsub_vf = ("vfmsub.vf", 0b101010, OPFVF, MacVf);
    vfnmsub_vv = ("vfnmsub.vv", 0b101011, OPFVV, MacVv);
    vfnmsub_vf = ("vfnmsub.vf", 0b101011, OPFVF, MacVf);
    vfwadd_vv = ("vfwadd.vv", 0b110000, OPFVV, Vv);
    vfwadd_wv = ("vfwadd.wv", 0b110100, OPFVV, Vv);
    vfwmul_vv = ("vfwmul.vv", 0b111000, OPFVV, Vv);
    vfwmacc_vv = ("vfwmacc.vv", 0b111100, OPFVV, MacVv);
    vfwmacc_vf = ("vfwmacc.vf", 0b111100, OPFVF, MacVf);
    vfwmaccbf16_vv = ("vfwmaccbf16.vv", 0b111011, OPFVV, MacVv);
    vfwmaccbf16_vf = ("vfwmaccbf16.vf", 0b111011, OPFVF, MacVf);
    vfredusum_vs = ("vfredusum.vs", 0b000001, OPFVV, Vv);
    vfredosum_vs = ("vfredosum.vs", 0b000011, OPFVV, Vv);
    vfredmin_vs = ("vfredmin.vs", 0b000101, OPFVV, Vv);
    vfredmax_vs = ("vfredmax.vs", 0b000111, OPFVV, Vv);
    vfwredusum_vs = ("vfwredusum.vs", 0b110001, OPFVV, Vv);
    vfwredosum_vs = ("vfwredosum.vs", 0b110011, OPFVV, Vv);
    vmfeq_vv = ("vmfeq.vv", 0b011000, OPFVV, Vv);
    vmfeq_vf = ("vmfeq.vf", 0b011000, OPFVF, Vf);
    vmfle_vv = ("vmfle.vv", 0b011001, OPFVV, Vv);
    vmfle_vf = ("vmfle.vf", 0b011001, OPFVF, Vf);
    vmflt_vv = ("vmflt.vv", 0b011011, OPFVV, Vv);
    vmflt_vf = ("vmflt.vf", 0b011011, OPFVF, Vf);
    vmfne_vv = ("vmfne.vv", 0b011100, OPFVV, Vv);
    vmfne_vf = ("vmfne.vf", 0b011100, OPFVF, Vf);
    vmfgt_vf = ("vmfgt.vf", 0b011101, OPFVF, Vf);
    vmfge_vf = ("vmfge.vf", 0b011111, OPFVF, Vf);
    vfmerge_vfm = ("vfmerge.vfm", 0b010111, OPFVF, MergeVf);
    vfmv_v_f = ("vfmv.v.f", 0b010111, OPFVF, SF);
    vfmv_f_s = ("vfmv.f.s", 0b010000, OPFVV, FS);
    vfmv_s_f = ("vfmv.s.f", 0b010000, OPFVF, SF);
    vfsqrt_v = ("vfsqrt.v", 0b010011, OPFVV, Unary(0b00000));
    vfrsqrt7_v = ("vfrsqrt7.v", 0b010011, OPFVV, Unary(0b00100));
    vfrec7_v = ("vfrec7.v", 0b010011, OPFVV, Unary(0b00101));
    vfclass_v = ("vfclass.v", 0b010011, OPFVV, Unary(0b10000));
    vfcvt_xu_f_v = ("vfcvt.xu.f.v", 0b010010, OPFVV, Unary(0b00000));
    vfcvt_x_f_v = ("vfcvt.x.f.v", 0b010010, OPFVV, Unary(0b00001));
    vfcvt_f_xu_v = ("vfcvt.f.xu.v", 0b010010, OPFVV, Unary(0b00010));
    vfcvt_f_x_v = ("vfcvt.f.x.v", 0b010010, OPFVV, Unary(0b00011));
    vfcvt_rtz_xu_f_v = ("vfcvt.rtz.xu.f.v", 0b010010, OPFVV, Unary(0b00110));
    vfcvt_rtz_x_f_v = ("vfcvt.rtz.x.f.v", 0b010010, OPFVV, Unary(0b00111));
    vfwcvt_f_f_v = ("vfwcvt.f.f.v", 0b010010, OPFVV, Unary(0b01100));
    vfwcvt_f_x_v = ("vfwcvt.f.x.v", 0b010010, OPFVV, Unary(0b01011));
    vfwcvtbf16_f_f_v = ("vfwcvtbf16.f.f.v", 0b010010, OPFVV, Unary(0b01101));
    vfncvt_f_f_w = ("vfncvt.f.f.w", 0b010010, OPFVV, Unary(0b10100));
    vfncvt_rod_f_f_w = ("vfncvt.rod.f.f.w", 0b010010, OPFVV, Unary(0b10101));
    vfncvt_x_f_w = ("vfncvt.x.f.w", 0b010010, OPFVV, Unary(0b10001));
    vfncvtbf16_f_f_w = ("vfncvtbf16.f.f.w", 0b010010, OPFVV, Unary(0b11101));
}

// ── configuration ───────────────────────────────────────────────────────

/// The alternate format exists at SEW 8 and 16 only.
fn alt_ok(vtype: Vtype) -> Result<(), String> {
    check(!vtype.alt || matches!(vtype.sew, Sew::E8 | Sew::E16), || format!("no alternate format at {}", vtype.sew))
}

impl Asm {
    /// `vsetvli rd, rs1, vtype`: `vl = min(rs1, VLMAX)` into `rd`; `rs1 = zero`
    /// with `rd ≠ zero` asks for VLMAX, both `zero` keeps `vl`.
    pub fn vsetvli(&mut self, rd: X, avl: X, vtype: Vtype) {
        let w = alt_ok(vtype).map(|_| vtype.bits() << 20 | xn(avl) << 15 | 7 << 12 | xn(rd) << 7 | OP_V);
        self.emit(w);
    }
    /// `vsetivli rd, uimm5, vtype`: an immediate application vector length.
    pub fn vsetivli(&mut self, rd: X, avl: u32, vtype: Vtype) {
        let w = alt_ok(vtype).and_then(|_| uimm5(avl)).map(|avl| 3 << 30 | vtype.bits() << 20 | avl << 15 | 7 << 12 | xn(rd) << 7 | OP_V);
        self.emit(w);
    }
    /// `vsetvl rd, rs1, rs2`: the type from a register.
    pub fn vsetvl(&mut self, rd: X, avl: X, vtype: X) {
        self.emit(Ok(1 << 31 | xn(vtype) << 20 | xn(avl) << 15 | 7 << 12 | xn(rd) << 7 | OP_V));
    }
}

// ── loads and stores ────────────────────────────────────────────────────

/// Addressing modes of a vector memory access, the `mop` field.
pub(crate) const MOP_UNIT: u32 = 0;
pub(crate) const MOP_UNORDERED: u32 = 1;
pub(crate) const MOP_STRIDED: u32 = 2;
pub(crate) const MOP_ORDERED: u32 = 3;
/// Unit-stride sub-modes, the `lumop`/`sumop` field.
pub(crate) const UMOP_WHOLE: u32 = 0b01000;
pub(crate) const UMOP_MASK: u32 = 0b01011;
pub(crate) const UMOP_FF: u32 = 0b10000;

impl Asm {
    #[allow(clippy::too_many_arguments)]
    fn vmem(&mut self, name: &str, store: bool, nf: u32, mop: u32, maskable: bool, field: u32, rs1: X, width: u32, vd: V) {
        let vm = if maskable { Ok(self.take_vm()) } else { self.no_mask(name).map(|_| 1) };
        let w = vm.and_then(|vm| {
            check((1..=8).contains(&nf), || format!("{nf} fields: a segment has 1 to 8"))?;
            Ok((nf - 1) << 29 | mop << 26 | vm << 25 | field << 20 | xn(rs1) << 15 | width << 12 | vn(vd) << 7 | if store { OP_STORE_FP } else { OP_LOAD_FP })
        });
        self.emit(w);
    }
    /// `vle<eew>.v vd, (rs1)`.
    pub fn vle(&mut self, eew: Sew, vd: V, rs1: X) {
        self.vmem("vle", false, 1, MOP_UNIT, true, 0, rs1, eew.mem_width(), vd);
    }
    /// `vle<eew>ff.v vd, (rs1)`: fault-only-first.
    pub fn vleff(&mut self, eew: Sew, vd: V, rs1: X) {
        self.vmem("vleff", false, 1, MOP_UNIT, true, UMOP_FF, rs1, eew.mem_width(), vd);
    }
    /// `vse<eew>.v vs3, (rs1)`.
    pub fn vse(&mut self, eew: Sew, vs3: V, rs1: X) {
        self.vmem("vse", true, 1, MOP_UNIT, true, 0, rs1, eew.mem_width(), vs3);
    }
    /// `vlseg<nf>e<eew>.v vd, (rs1)`: `nf` interleaved fields into `vd`, `vd+1`…
    pub fn vlseg(&mut self, nf: u32, eew: Sew, vd: V, rs1: X) {
        self.vmem("vlseg", false, nf, MOP_UNIT, true, 0, rs1, eew.mem_width(), vd);
    }
    /// `vsseg<nf>e<eew>.v vs3, (rs1)`.
    pub fn vsseg(&mut self, nf: u32, eew: Sew, vs3: V, rs1: X) {
        self.vmem("vsseg", true, nf, MOP_UNIT, true, 0, rs1, eew.mem_width(), vs3);
    }
    /// `vlse<eew>.v vd, (rs1), rs2`: a byte stride in `rs2`.
    pub fn vlse(&mut self, eew: Sew, vd: V, rs1: X, stride: X) {
        self.vmem("vlse", false, 1, MOP_STRIDED, true, xn(stride), rs1, eew.mem_width(), vd);
    }
    /// `vsse<eew>.v vs3, (rs1), rs2`.
    pub fn vsse(&mut self, eew: Sew, vs3: V, rs1: X, stride: X) {
        self.vmem("vsse", true, 1, MOP_STRIDED, true, xn(stride), rs1, eew.mem_width(), vs3);
    }
    /// `vlsseg<nf>e<eew>.v vd, (rs1), rs2`.
    pub fn vlsseg(&mut self, nf: u32, eew: Sew, vd: V, rs1: X, stride: X) {
        self.vmem("vlsseg", false, nf, MOP_STRIDED, true, xn(stride), rs1, eew.mem_width(), vd);
    }
    /// `vssseg<nf>e<eew>.v vs3, (rs1), rs2`.
    pub fn vssseg(&mut self, nf: u32, eew: Sew, vs3: V, rs1: X, stride: X) {
        self.vmem("vssseg", true, nf, MOP_STRIDED, true, xn(stride), rs1, eew.mem_width(), vs3);
    }
    /// `vluxei<eew>.v vd, (rs1), vs2`: byte offsets of width `eew` in `vs2`.
    pub fn vluxei(&mut self, eew: Sew, vd: V, rs1: X, index: V) {
        self.vmem("vluxei", false, 1, MOP_UNORDERED, true, vn(index), rs1, eew.mem_width(), vd);
    }
    /// `vloxei<eew>.v vd, (rs1), vs2`: ordered.
    pub fn vloxei(&mut self, eew: Sew, vd: V, rs1: X, index: V) {
        self.vmem("vloxei", false, 1, MOP_ORDERED, true, vn(index), rs1, eew.mem_width(), vd);
    }
    /// `vsuxei<eew>.v vs3, (rs1), vs2`.
    pub fn vsuxei(&mut self, eew: Sew, vs3: V, rs1: X, index: V) {
        self.vmem("vsuxei", true, 1, MOP_UNORDERED, true, vn(index), rs1, eew.mem_width(), vs3);
    }
    /// `vsoxei<eew>.v vs3, (rs1), vs2`.
    pub fn vsoxei(&mut self, eew: Sew, vs3: V, rs1: X, index: V) {
        self.vmem("vsoxei", true, 1, MOP_ORDERED, true, vn(index), rs1, eew.mem_width(), vs3);
    }
    /// `vl<n>re<eew>.v vd, (rs1)`: `n` whole registers, whatever `vl`.
    pub fn vl_r(&mut self, n: u32, eew: Sew, vd: V, rs1: X) {
        if !matches!(n, 1 | 2 | 4 | 8) {
            self.fail(format!("vl{n}r: 1, 2, 4 or 8 registers"));
        }
        self.vmem("vl<n>r", false, n, MOP_UNIT, false, UMOP_WHOLE, rs1, eew.mem_width(), vd);
    }
    /// `vs<n>r.v vs3, (rs1)`.
    pub fn vs_r(&mut self, n: u32, vs3: V, rs1: X) {
        if !matches!(n, 1 | 2 | 4 | 8) {
            self.fail(format!("vs{n}r: 1, 2, 4 or 8 registers"));
        }
        self.vmem("vs<n>r", true, n, MOP_UNIT, false, UMOP_WHOLE, rs1, Sew::E8.mem_width(), vs3);
    }
    /// `vlm.v vd, (rs1)`: a mask, ⌈vl/8⌉ bytes.
    pub fn vlm(&mut self, vd: V, rs1: X) {
        self.vmem("vlm.v", false, 1, MOP_UNIT, false, UMOP_MASK, rs1, Sew::E8.mem_width(), vd);
    }
    /// `vsm.v vs3, (rs1)`.
    pub fn vsm(&mut self, vs3: V, rs1: X) {
        self.vmem("vsm.v", true, 1, MOP_UNIT, false, UMOP_MASK, rs1, Sew::E8.mem_width(), vs3);
    }
}
