//! The ISA floor, checked at startup before anything is allocated.
//!
//! The engine emits its own kernels. On x86-64 everything it produces assumes
//! at least **AVX2 + FMA**; above the floor, two independent axes decide *how
//! fast*, never *whether*: AVX-512 (with BF16: `vdpbf16ps`) and the 256-bit
//! `VPDPBUSD` (`avxvnni`). On AArch64 the floor is **NEON + BF16 + LSE**: the
//! bf16 dot products (`bfdot`), the bf16 narrowing (`bfcvtn`) and the atomics
//! of the generated barriers; SME2 is reported, not required. Below the floor
//! there is no arm to fall back to, only a machine that cannot run this
//! binary, and it is told so on the first line of output.
//!
//! The check runs at run time, not build time: a build script only sees the
//! machine it runs on, and a binary is a file that gets copied. On x86-64,
//! CPUID and XCR0 are read through a probe assembled and mapped once
//! (`isa_probe.rs`), with the operating-system rules of the standard
//! library's detection; on AArch64 macOS the kernel's `hw.optional.arm.*`
//! (`isa_arm.rs`), user space having no access to the ID registers. On
//! riscv64 Linux the floor is **RV64GC + V 1.0 at VLEN ≥ 256 + Zba + Zbb +
//! Zvfbfmin + Zvfbfwma** — the vector unit, the bf16 conversions and the
//! widening bf16 multiply-add of every GEMM — read from `riscv_hwprobe` and the `isa` line
//! of `/proc/cpuinfo`, the vector length from `vlenb` (`isa_rv.rs`).

#[cfg(target_arch = "x86_64")]
#[path = "isa_probe.rs"]
mod probe;
#[cfg(target_arch = "aarch64")]
#[path = "isa_arm.rs"]
mod probe;
#[cfg(target_arch = "riscv64")]
#[path = "isa_rv.rs"]
mod probe;
pub use probe::Features;
#[cfg(target_arch = "aarch64")]
pub use probe::cores_of_level;

/// The features of this machine, read once by the probe, or why they could not be.
pub fn detected() -> &'static Result<Features, String> {
    static DETECTED: std::sync::OnceLock<Result<Features, String>> = std::sync::OnceLock::new();
    DETECTED.get_or_init(|| {
        probe::detect()
    })
}

/// One line naming what this machine offers, for the startup log.
#[cfg(target_arch = "aarch64")]
pub fn summary() -> String {
    let f = |n: &str| feature(n);
    let svl = detected().as_ref().map_or(0, |f| f.sme_max_svl_bytes);
    format!(
        "neon={} bf16={} ebf16={} i8mm={} dotprod={} fp16={} lse={} sme={} sme2={} sme_b16f32={} svl={}B → GEMM bfdot",
        yes(f("neon")),
        yes(f("bf16")),
        yes(f("ebf16")),
        yes(f("i8mm")),
        yes(f("dotprod")),
        yes(f("fp16")),
        yes(f("lse")),
        yes(f("sme")),
        yes(f("sme2")),
        yes(f("sme_b16f32")),
        svl,
    )
}

/// One line naming what this machine offers, for the startup log.
#[cfg(target_arch = "x86_64")]
pub fn summary() -> String {
    let f = |n: &str| feature(n);
    let lanes = if f("avx512f") && f("avx512bw") && f("avx512vl") && f("avx512vnni") {
        "zmm"
    } else if f("avxvnni") {
        "ymm"
    } else {
        "ymm-no-vnni"
    };
    format!(
        "avx2={} fma={} avxvnni={} avx512f={} avx512bf16={} → GEMM lanes {lanes}",
        yes(f("avx2")),
        yes(f("fma")),
        yes(f("avxvnni")),
        yes(f("avx512f")),
        yes(f("avx512bf16")),
        lanes = lanes,
    )
}

/// One line naming what this machine offers, for the startup log.
#[cfg(target_arch = "riscv64")]
pub fn summary() -> String {
    let f = |n: &str| feature(n);
    let vlen = detected().as_ref().map_or(0, |f| f.vlenb * 8);
    format!(
        "v={} vlen={vlen} zvfbfmin={} zvfbfwma={} zvfbfa={} zvfh={} zfbfmin={} zba={} zbb={} zbs={} zihintpause={} zacas={} → GEMM vfwmaccbf16",
        yes(f("v")),
        yes(f("zvfbfmin")),
        yes(f("zvfbfwma")),
        yes(f("zvfbfa")),
        yes(f("zvfh")),
        yes(f("zfbfmin")),
        yes(f("zba")),
        yes(f("zbb")),
        yes(f("zbs")),
        yes(f("zihintpause")),
        yes(f("zacas")),
    )
}

/// Bytes per vector register on this machine (VLEN / 8), 0 when unknown.
#[cfg(target_arch = "riscv64")]
pub fn vlenb() -> usize {
    match detected() {
        Ok(f) => f.vlenb as usize,
        Err(_) => 0,
    }
}

fn yes(b: bool) -> &'static str {
    if b {
        "yes"
    } else {
        "no"
    }
}

/// Is this CPU feature present? CPUID and XCR0 as read by the probe, the single
/// source of every ISA decision in the engine — never `/proc/cpuinfo`, which
/// describes the CPU rather than what the detection sees. `false` when the probe
/// could not run ([`refuse_if_below_floor`] says why).
#[cfg(target_arch = "aarch64")]
pub fn feature(name: &str) -> bool {
    let Ok(f) = detected() else {
        return false;
    };
    match name {
        "neon" => f.neon,
        "fp16" => f.fp16,
        "dotprod" => f.dotprod,
        "fhm" => f.fhm,
        "bf16" => f.bf16,
        "ebf16" => f.ebf16,
        "i8mm" => f.i8mm,
        "lse" => f.lse,
        "sme" => f.sme,
        "sme2" => f.sme2,
        "sme_b16f32" => f.sme_b16f32,
        "sme_f32f32" => f.sme_f32f32,
        // The generated trace reads the virtual counter, always there on AArch64.
        "rdtscp" | "cntvct" => true,
        _ => false,
    }
}

/// Is this CPU feature present? What the kernel says it enabled for user
/// space (hwprobe and the `isa` string), the single source of every ISA
/// decision of the RISC-V target. `false` when the probe could not run
/// ([`refuse_if_below_floor`] says why).
#[cfg(target_arch = "riscv64")]
pub fn feature(name: &str) -> bool {
    let Ok(f) = detected() else {
        return false;
    };
    match name {
        "ima" => f.ima,
        "fd" => f.fd,
        "c" => f.c,
        "v" => f.v,
        "zba" => f.zba,
        "zbb" => f.zbb,
        "zbs" => f.zbs,
        "zfh" => f.zfh,
        "zfhmin" => f.zfhmin,
        "zfbfmin" => f.zfbfmin,
        "zvfh" => f.zvfh,
        "zvfhmin" => f.zvfhmin,
        "zvfbfmin" => f.zvfbfmin,
        "zvfbfwma" => f.zvfbfwma,
        "zvfbfa" => f.zvfbfa,
        "zihintpause" => f.zihintpause,
        "zacas" => f.zacas,
        "zicond" => f.zicond,
        // The GEMM holds sixteen bf16 in one register: VLEN ≥ 256.
        "vlen256" => f.vlenb >= 32,
        // The generated trace reads the `time` CSR, which Linux lets user space read.
        "rdtscp" | "rdtime" => true,
        _ => false,
    }
}

/// Is this CPU feature present? CPUID and XCR0 as read by the probe, the single
/// source of every ISA decision in the engine — never `/proc/cpuinfo`, which
/// describes the CPU rather than what the detection sees. `false` when the probe
/// could not run ([`refuse_if_below_floor`] says why).
#[cfg(target_arch = "x86_64")]
pub fn feature(name: &str) -> bool {
    let Ok(f) = detected() else {
        return false;
    };
    match name {
        "sse4.2" => f.sse42,
        "avx" => f.avx,
        "avx2" => f.avx2,
        "fma" => f.fma,
        "f16c" => f.f16c,
        "avxvnni" => f.avxvnni,
        "avxneconvert" => f.avxneconvert,
        "avx512f" => f.avx512f,
        "avx512dq" => f.avx512dq,
        "avx512bw" => f.avx512bw,
        "avx512vl" => f.avx512vl,
        "avx512vnni" => f.avx512vnni,
        "avx512bf16" => f.avx512bf16,
        "rdtscp" => f.rdtscp,
        _ => false,
    }
}

/// **The startup check.** Call it first, before parsing anything, before
/// touching a model.
///
/// Returns the missing features, in reading order. An empty vector means the
/// machine is above the floor.
pub fn below_floor() -> Vec<&'static str> {
    #[cfg(target_arch = "x86_64")]
    let floor = ["avx2", "fma"];
    #[cfg(target_arch = "aarch64")]
    let floor = ["neon", "bf16", "lse"];
    #[cfg(target_arch = "riscv64")]
    let floor = ["ima", "fd", "v", "vlen256", "zba", "zbb", "zvfbfmin", "zvfbfwma"];
    floor.into_iter().filter(|f| !feature(f)).collect()
}

/// Prints the refusal and returns the process exit code, or `None` when the
/// machine is fine.
///
/// It writes rather than returning an error on purpose: this runs before the
/// logger and before anything that could carry an error nicely, and the whole
/// point is that the message is the first thing on the terminal.
pub fn refuse_if_below_floor() -> Option<i32> {
    if let Err(e) = detected() {
        eprintln!("herbert: the CPU features could not be read ({e}); the engine cannot run here.");
        return Some(2);
    }
    let missing = below_floor();
    if missing.is_empty() {
        return None;
    }
    eprintln!(
        "herbert: this CPU is below the ISA floor — {} missing.\n\
         \n\
         The engine generates its own kernels and every one of them assumes \
         the floor ({}); there is no arm below that, so the binary would fail \
         later, inside a generated kernel, with a SIGILL that names nothing.\n\
         \n\
         Detected: {}",
        missing.join(" and "),
        if cfg!(target_arch = "x86_64") {
            "AVX2 and FMA"
        } else if cfg!(target_arch = "riscv64") {
            "RV64GC, V with VLEN ≥ 256, Zba, Zbb, Zvfbfmin and Zvfbfwma"
        } else {
            "NEON, BF16 and LSE"
        },
        summary(),
    );
    Some(2)
}
