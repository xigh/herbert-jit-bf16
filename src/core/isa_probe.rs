//! CPUID and XCR0, read through two assembled functions rather than a compiler
//! intrinsic or a runtime-detection macro.
//!
//! The probe is two small SysV64 functions assembled with `iced-x86` and mapped
//! executable for the duration of one reading. The operating-system rules are
//! those of the standard library's detection: AVX-class features need OSXSAVE
//! and XCR0 bits 1-2 (SSE and AVX state), AVX-512 features also need XCR0 bits
//! 5-7 (opmask, ZMM_Hi256, Hi16_ZMM).

use iced_x86::code_asm::*;

/// The CPU features this repository decides on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avxvnni: bool,
    pub avxneconvert: bool,
    pub avx512f: bool,
    pub avx512dq: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub rdtscp: bool,
}

fn assemble(
    emit: impl FnOnce(&mut CodeAssembler) -> std::result::Result<(), IcedError>,
) -> std::result::Result<memmap2::Mmap, String> {
    let mut a = CodeAssembler::new(64).map_err(|e| format!("probe assembler: {e}"))?;
    emit(&mut a).map_err(|e| format!("probe emission: {e}"))?;
    let code = a.assemble(0).map_err(|e| format!("probe assembly: {e}"))?;
    let mut map = memmap2::MmapMut::map_anon(code.len()).map_err(|e| format!("probe mapping: {e}"))?;
    map.copy_from_slice(&code);
    map.make_exec().map_err(|e| format!("probe protection: {e}"))
}

/// Read the features of the machine running this code.
pub fn detect() -> std::result::Result<Features, String> {
    // fn(leaf: u32, subleaf: u32, out: *mut [u32; 4]) — eax, ebx, ecx, edx.
    let cpuid_page = assemble(|a| {
        a.push(rbx)?;
        a.mov(r8, rdx)?;
        a.mov(eax, edi)?;
        a.mov(ecx, esi)?;
        a.cpuid()?;
        a.mov(dword_ptr(r8), eax)?;
        a.mov(dword_ptr(r8 + 4), ebx)?;
        a.mov(dword_ptr(r8 + 8), ecx)?;
        a.mov(dword_ptr(r8 + 12), edx)?;
        a.pop(rbx)?;
        a.ret()
    })?;
    // fn(xcr: u32) -> u64.
    let xgetbv_page = assemble(|a| {
        a.mov(ecx, edi)?;
        a.xgetbv()?;
        a.shl(rdx, 32u32)?;
        a.or(rax, rdx)?;
        a.ret()
    })?;
    // SAFETY: both pages hold one complete SysV64 function (rbx restored, no
    // stack use beyond the push) and outlive every call below.
    let cpuid_fn: unsafe extern "sysv64" fn(u32, u32, *mut u32) =
        unsafe { std::mem::transmute(cpuid_page.as_ptr()) };
    let xgetbv_fn: unsafe extern "sysv64" fn(u32) -> u64 =
        unsafe { std::mem::transmute(xgetbv_page.as_ptr()) };
    let cpuid = |leaf: u32, subleaf: u32| {
        let mut r = [0u32; 4];
        // SAFETY: `r` holds the four words the function writes.
        unsafe { cpuid_fn(leaf, subleaf, r.as_mut_ptr()) };
        r
    };
    let bit = |word: u32, n: u32| (word >> n) & 1 == 1;
    let max_leaf = cpuid(0, 0)[0];
    let leaf1 = if max_leaf >= 1 { cpuid(1, 0) } else { [0; 4] };
    let leaf7 = if max_leaf >= 7 { cpuid(7, 0) } else { [0; 4] };
    let leaf7_1 = if max_leaf >= 7 && leaf7[0] >= 1 { cpuid(7, 1) } else { [0; 4] };
    let max_extended = cpuid(0x8000_0000, 0)[0];
    let extended1 = if max_extended >= 0x8000_0001 { cpuid(0x8000_0001, 0) } else { [0; 4] };
    let osxsave = bit(leaf1[2], 27);
    // SAFETY: XGETBV is only executed when OSXSAVE says the OS enabled it.
    let xcr0 = if osxsave { unsafe { xgetbv_fn(0) } } else { 0 };
    let os_avx = osxsave && xcr0 & 0b110 == 0b110;
    let os_avx512 = os_avx && xcr0 & 0xe0 == 0xe0;
    Ok(Features {
        sse42: bit(leaf1[2], 20),
        avx: os_avx && bit(leaf1[2], 28),
        fma: os_avx && bit(leaf1[2], 12),
        f16c: os_avx && bit(leaf1[2], 29),
        avx2: os_avx && bit(leaf7[1], 5),
        avxvnni: os_avx && bit(leaf7_1[0], 4),
        avxneconvert: os_avx && bit(leaf7_1[3], 5),
        avx512f: os_avx512 && bit(leaf7[1], 16),
        avx512dq: os_avx512 && bit(leaf7[1], 17),
        avx512bw: os_avx512 && bit(leaf7[1], 30),
        avx512vl: os_avx512 && bit(leaf7[1], 31),
        avx512vnni: os_avx512 && bit(leaf7[2], 11),
        avx512bf16: os_avx512 && bit(leaf7_1[0], 5),
        rdtscp: bit(extended1[3], 27),
    })
}
