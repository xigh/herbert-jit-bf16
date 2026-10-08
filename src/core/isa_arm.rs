//! The AArch64 features, read from the kernel.
//!
//! macOS does not let user space read the `ID_AA64*` registers (an `mrs` of
//! them traps); it publishes the same answers as `hw.optional.arm.*` sysctls,
//! which is what this probe reads, once. Every ISA decision of the ARM target
//! goes through these fields.

/// The CPU features the AArch64 target decides on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// Advanced SIMD (NEON).
    pub neon: bool,
    /// Half-precision arithmetic (FEAT_FP16).
    pub fp16: bool,
    /// Integer dot products (FEAT_DotProd).
    pub dotprod: bool,
    /// `fmlal` half to single (FEAT_FHM).
    pub fhm: bool,
    /// The BF16 extension: `bfdot`, `bfmmla`, `bfmlal[bt]`, `bfcvt[n]` (FEAT_BF16).
    pub bf16: bool,
    /// The IEEE-conformant BF16 dot products under `FPCR.EBF` (FEAT_EBF16).
    pub ebf16: bool,
    /// 8-bit integer matrix multiplies (FEAT_I8MM).
    pub i8mm: bool,
    /// Large System Extensions: `ldadd`, `cas`, `swp` (FEAT_LSE).
    pub lse: bool,
    /// Scalable Matrix Extension (FEAT_SME).
    pub sme: bool,
    /// SME2 (FEAT_SME2).
    pub sme2: bool,
    /// `bfmopa` into single-precision tiles (SME_B16F32).
    pub sme_b16f32: bool,
    /// `fmopa` single into single (SME_F32F32).
    pub sme_f32f32: bool,
    /// The largest streaming vector length, in bytes (0 without SME).
    pub sme_max_svl_bytes: u32,
    /// Bytes per cache line.
    pub cacheline: u32,
}

fn sysctl_u64(name: &str) -> Option<u64> {
    let key = std::ffi::CString::new(name).ok()?;
    let mut value = [0u8; 8];
    let mut len = value.len();
    // SAFETY: `value` holds `len` writable bytes, the key is NUL-terminated.
    let rc = unsafe {
        libc::sysctlbyname(key.as_ptr(), value.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0)
    };
    if rc != 0 {
        return None;
    }
    Some(match len {
        4 => u64::from(u32::from_ne_bytes(value[..4].try_into().ok()?)),
        8 => u64::from_ne_bytes(value),
        _ => return None,
    })
}

fn flag(name: &str) -> bool {
    sysctl_u64(name).is_some_and(|v| v != 0)
}

/// Read the features of the machine running this code.
pub fn detect() -> std::result::Result<Features, String> {
    if sysctl_u64("hw.optional.arm64").is_none() {
        return Err("the kernel publishes no hw.optional.arm64: not an AArch64 macOS".into());
    }
    Ok(Features {
        neon: flag("hw.optional.arm.AdvSIMD") || flag("hw.optional.neon"),
        fp16: flag("hw.optional.arm.FEAT_FP16"),
        dotprod: flag("hw.optional.arm.FEAT_DotProd"),
        fhm: flag("hw.optional.arm.FEAT_FHM"),
        bf16: flag("hw.optional.arm.FEAT_BF16"),
        ebf16: flag("hw.optional.arm.FEAT_EBF16"),
        i8mm: flag("hw.optional.arm.FEAT_I8MM"),
        lse: flag("hw.optional.arm.FEAT_LSE"),
        sme: flag("hw.optional.arm.FEAT_SME"),
        sme2: flag("hw.optional.arm.FEAT_SME2"),
        sme_b16f32: flag("hw.optional.arm.SME_B16F32"),
        sme_f32f32: flag("hw.optional.arm.SME_F32F32"),
        sme_max_svl_bytes: sysctl_u64("hw.optional.arm.sme_max_svl_b").unwrap_or(0) as u32,
        cacheline: sysctl_u64("hw.cachelinesize").unwrap_or(64) as u32,
    })
}

/// The number of cores of the performance level `level` (0 = the fastest),
/// or `None` when the kernel publishes no levels.
pub fn cores_of_level(level: u32) -> Option<usize> {
    sysctl_u64(&format!("hw.perflevel{level}.physicalcpu")).map(|n| n as usize)
}
