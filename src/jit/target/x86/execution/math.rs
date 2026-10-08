//! Scalar f32 exp/log emitted as x86-64 instructions; no libm runtime calls.
//! Algorithm/coefficients: Arm optimized-routines, MIT option (license and
//! pinned revision in `third_party/arm-optimized-routines`). The FMA form keeps
//! the evaluation order used by the native GNU f32 math implementation.
use super::{math_data::*, ExecutionError, GeneratedKernel};
use iced_x86::code_asm::*;
fn err(e: impl std::fmt::Display) -> ExecutionError {
    ExecutionError(e.to_string())
}
fn constant(
    a: &mut CodeAssembler,
    dst: AsmRegisterXmm,
    bits: u64,
) -> Result<(), iced_x86::IcedError> {
    a.mov(r11, bits)?;
    a.movq(dst, r11)
}

/// `(tables: *const Tables, x: f32) -> f32`. All scratch registers are volatile
/// in sysv64. Tables are owned by the mounted plan and immutable during a call.
/// The instructions of [`exp`], written into any assembler, so a larger kernel can
/// hold the routine and `call` it locally. Same ABI: tables in `rdi`, `x` in `xmm0`, result in `xmm0`; it clobbers `rax`, `rcx`, `rdx`, `r11` and `xmm1`-`xmm6`, and ends with `ret`.
pub(crate) fn exp_body(a: &mut CodeAssembler, fma: bool) -> Result<(), iced_x86::IcedError> {
    let mut core = a.create_label();
    let mut special = a.create_label();
    let mut zero = a.create_label();
    let mut huge = a.create_label();
    let mut nan_inf = a.create_label();
    a.movd(edx, xmm0)?;
    a.mov(eax, edx)?;
    a.and(eax, 0x7fffffffu32)?;
    a.cmp(eax, 88.0f32.to_bits())?;
    a.jae(special)?;
    a.set_label(&mut core)?;
    a.cvtss2sd(xmm2, xmm0)?;
    constant(a, xmm1, INV)?;
    constant(a, xmm3, SHIFT)?;
    constant(a, xmm4, 1.0f64.to_bits())?;
    a.movapd(xmm0, xmm1)?;
    if fma {
        a.vfmadd132sd(xmm0, xmm3, xmm2)?;
    } else {
        a.mulsd(xmm0, xmm2)?;
        a.addsd(xmm0, xmm3)?;
    }
    a.movq(rax, xmm0)?;
    a.subsd(xmm0, xmm3)?;
    a.mov(rcx, rax)?;
    a.shl(rax, 47)?;
    a.and(ecx, 31)?;
    if fma {
        a.vfmsub132sd(xmm1, xmm0, xmm2)?;
    } else {
        a.mulsd(xmm1, xmm2)?;
        a.subsd(xmm1, xmm0)?;
    }
    a.add(rax, qword_ptr(rdi + rcx * 8))?;
    a.movq(xmm5, rax)?;
    constant(a, xmm0, E0)?;
    constant(a, xmm6, E1)?;
    if fma {
        a.vfmadd213sd(xmm0, xmm1, xmm6)?;
    } else {
        a.mulsd(xmm0, xmm1)?;
        a.addsd(xmm0, xmm6)?;
    }
    a.movapd(xmm2, xmm1)?;
    a.mulsd(xmm2, xmm1)?;
    constant(a, xmm6, E2)?;
    if fma {
        a.vfmadd132sd(xmm1, xmm4, xmm6)?;
        a.vfmadd132sd(xmm0, xmm1, xmm2)?;
    } else {
        a.mulsd(xmm1, xmm6)?;
        a.addsd(xmm1, xmm4)?;
        a.mulsd(xmm0, xmm2)?;
        a.addsd(xmm0, xmm1)?;
    }
    a.mulsd(xmm0, xmm5)?;
    a.cvtsd2ss(xmm0, xmm0)?;
    a.ret()?;
    a.set_label(&mut special)?;
    a.cmp(edx, 0xff800000u32)?;
    a.je(zero)?;
    a.cmp(eax, 0x7f800000u32)?;
    a.jae(nan_inf)?;
    constant(a, xmm1, 88.722_84_f32.to_bits() as u64)?;
    a.ucomiss(xmm0, xmm1)?;
    a.ja(huge)?;
    constant(a, xmm1, (-103.972_084_f32).to_bits() as u64)?;
    a.ucomiss(xmm0, xmm1)?;
    a.jb(zero)?;
    a.jmp(core)?;
    a.set_label(&mut nan_inf)?;
    a.addss(xmm0, xmm0)?;
    a.ret()?;
    a.set_label(&mut huge)?;
    a.mov(eax, 0x7f800000u32)?;
    a.movd(xmm0, eax)?;
    a.ret()?;
    a.set_label(&mut zero)?;
    a.xorps(xmm0, xmm0)?;
    a.ret()
}

pub(crate) fn exp(fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = CodeAssembler::new(64).map_err(err)?;
    exp_body(&mut a, fma).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}

/// Same ABI as exp. Includes zeros, subnormals, infinities and NaNs. Floating
/// errno is not part of the inference ABI; no foreign exception helper is called.
pub(crate) fn log(fma: bool) -> Result<GeneratedKernel, ExecutionError> {
    let mut a = CodeAssembler::new(64).map_err(err)?;
    let emit = |a: &mut CodeAssembler| -> Result<(), iced_x86::IcedError> {
        let mut core = a.create_label();
        let mut special = a.create_label();
        let mut zero = a.create_label();
        let mut one = a.create_label();
        let mut invalid = a.create_label();
        let mut ret = a.create_label();
        a.movd(eax, xmm0)?;
        a.cmp(eax, 1.0f32.to_bits())?;
        a.je(one)?;
        a.lea(edx, ptr(rax - 0x800000))?;
        a.cmp(edx, 0x7effffffu32)?;
        a.ja(special)?;
        a.set_label(&mut core)?;
        a.lea(edx, ptr(rax - 0x3f330000))?;
        a.mov(ecx, edx)?;
        a.shr(ecx, 19)?;
        a.and(ecx, 15)?;
        a.shl(ecx, 4)?;
        a.mov(esi, edx)?;
        a.sar(edx, 23)?;
        a.and(esi, 0xff800000u32)?;
        a.sub(eax, esi)?;
        a.movd(xmm1, eax)?;
        a.cvtss2sd(xmm1, xmm1)?;
        constant(a, xmm5, (-1.0f64).to_bits())?;
        a.movq(xmm6, qword_ptr(rdi + rcx + 256))?;
        if fma {
            a.vfmadd132sd(xmm1, xmm5, xmm6)?;
        } else {
            a.mulsd(xmm1, xmm6)?;
            a.addsd(xmm1, xmm5)?;
        }
        a.cvtsi2sd(xmm2, edx)?;
        constant(a, xmm6, LN2)?;
        a.movq(xmm5, qword_ptr(rdi + rcx + 264))?;
        if fma {
            a.vfmadd132sd(xmm2, xmm5, xmm6)?;
        } else {
            a.mulsd(xmm2, xmm6)?;
            a.addsd(xmm2, xmm5)?;
        }
        constant(a, xmm3, L1)?;
        constant(a, xmm6, L2)?;
        if fma {
            a.vfmadd213sd(xmm3, xmm1, xmm6)?;
        } else {
            a.mulsd(xmm3, xmm1)?;
            a.addsd(xmm3, xmm6)?;
        }
        a.movapd(xmm0, xmm1)?;
        a.mulsd(xmm0, xmm1)?;
        a.addsd(xmm1, xmm2)?;
        constant(a, xmm6, L0)?;
        if fma {
            a.vfmadd231sd(xmm3, xmm0, xmm6)?;
            a.vfmadd132sd(xmm0, xmm1, xmm3)?;
        } else {
            a.mulsd(xmm6, xmm0)?;
            a.addsd(xmm3, xmm6)?;
            a.mulsd(xmm0, xmm3)?;
            a.addsd(xmm0, xmm1)?;
        }
        a.cvtsd2ss(xmm0, xmm0)?;
        a.ret()?;
        a.set_label(&mut special)?;
        a.mov(edx, eax)?;
        a.add(edx, edx)?;
        a.je(zero)?;
        a.cmp(eax, 0x7f800000u32)?;
        a.je(ret)?;
        a.cmp(edx, 0xff000000u32)?;
        a.jae(invalid)?;
        a.test(eax, eax)?;
        a.js(invalid)?;
        constant(a, xmm1, (8388608.0f32).to_bits() as u64)?;
        a.mulss(xmm0, xmm1)?;
        a.movd(eax, xmm0)?;
        a.sub(eax, 23 << 23)?;
        a.jmp(core)?;
        a.set_label(&mut zero)?;
        a.mov(eax, 0xff800000u32)?;
        a.movd(xmm0, eax)?;
        a.ret()?;
        a.set_label(&mut invalid)?;
        a.subss(xmm0, xmm0)?;
        a.divss(xmm0, xmm0)?;
        a.ret()?;
        a.set_label(&mut one)?;
        a.xorps(xmm0, xmm0)?;
        a.set_label(&mut ret)?;
        a.ret()
    };
    emit(&mut a).map_err(err)?;
    unsafe { GeneratedKernel::new(a.assemble(0).map_err(err)?) }
}
