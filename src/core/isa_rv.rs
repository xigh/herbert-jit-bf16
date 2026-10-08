//! The RISC-V features, read from the kernel.
//!
//! User space cannot read `misa` or the extension CSRs; Linux publishes what
//! it enabled for user space in two places, both read here, once: the
//! `riscv_hwprobe` system call (keys `BASE_BEHAVIOR` and `IMA_EXT_0`) and the
//! `isa` line of `/proc/cpuinfo`. A feature is taken as present when either
//! says so — a kernel may name an extension in the string before it has a
//! hwprobe bit for it. The vector length, which no file names, is read by a
//! one-word kernel emitted with `rv` (`csrr a0, vlenb`), and only once the
//! kernel has said V is there. Every ISA decision of the RISC-V target goes
//! through these fields.

/// The CPU features the RISC-V target decides on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Features {
    /// The base behaviour I, M and A (`RISCV_HWPROBE_BASE_BEHAVIOR_IMA`).
    pub ima: bool,
    /// Single and double precision (F and D).
    pub fd: bool,
    /// Compressed instructions (C): the engine emits none, the rest of the binary may.
    pub c: bool,
    /// The vector extension V, 1.0.
    pub v: bool,
    /// Address generation (`sh1add`…).
    pub zba: bool,
    /// Basic bit manipulation (`min`, `clz`, `rev8`…).
    pub zbb: bool,
    /// Single-bit instructions.
    pub zbs: bool,
    /// Half precision.
    pub zfh: bool,
    /// Half-precision moves and conversions.
    pub zfhmin: bool,
    /// Scalar bf16 moves and conversions (`fcvt.bf16.s`, `flh`).
    pub zfbfmin: bool,
    /// Vector half precision.
    pub zvfh: bool,
    /// Vector half-precision conversions.
    pub zvfhmin: bool,
    /// Vector bf16 conversions (`vfwcvtbf16`, `vfncvtbf16`).
    pub zvfbfmin: bool,
    /// The widening bf16 multiply-add `vfwmaccbf16`.
    pub zvfbfwma: bool,
    /// The alternate bf16 format of `vtype` (`e16alt`), named by the `isa`
    /// string only: no hwprobe bit is known for it here.
    pub zvfbfa: bool,
    /// The `pause` hint (without it, the same word is a harmless fence).
    pub zihintpause: bool,
    /// Compare-and-swap (`amocas`).
    pub zacas: bool,
    /// Integer conditional operations.
    pub zicond: bool,
    /// Bytes per vector register, VLEN / 8 (0 without V).
    pub vlenb: u32,
    /// Bytes per cache line.
    pub cacheline: u32,
}

/// `SYS_riscv_hwprobe` of the Linux RISC-V system call table.
const SYS_RISCV_HWPROBE: libc::c_long = 258;
const KEY_BASE_BEHAVIOR: i64 = 3;
const KEY_IMA_EXT_0: i64 = 4;
const BASE_BEHAVIOR_IMA: u64 = 1;

/// `(name, bit of IMA_EXT_0)`, as `<asm/hwprobe.h>` numbers them.
const IMA_EXT_0: [(&str, u32); 16] = [
    ("fd", 0),
    ("c", 1),
    ("v", 2),
    ("zba", 3),
    ("zbb", 4),
    ("zbs", 5),
    ("zfh", 27),
    ("zfhmin", 28),
    ("zvfh", 30),
    ("zvfhmin", 31),
    ("zacas", 34),
    ("zicond", 35),
    ("zihintpause", 36),
    ("zfbfmin", 52),
    ("zvfbfmin", 53),
    ("zvfbfwma", 54),
];

#[repr(C)]
struct Pair {
    key: i64,
    value: u64,
}

/// The two hwprobe words, or `None` when the kernel has no such call.
pub(crate) fn hwprobe() -> Option<(u64, u64)> {
    let mut pairs = [Pair { key: KEY_BASE_BEHAVIOR, value: 0 }, Pair { key: KEY_IMA_EXT_0, value: 0 }];
    // SAFETY: `pairs` holds two writable key/value pairs; no CPU set is passed.
    let rc = unsafe { libc::syscall(SYS_RISCV_HWPROBE, pairs.as_mut_ptr(), pairs.len(), 0usize, std::ptr::null_mut::<u8>(), 0u32) };
    if rc != 0 || pairs.iter().any(|p| p.key < 0) {
        return None;
    }
    Some((pairs[0].value, pairs[1].value))
}

/// The extensions of the `isa` line of `/proc/cpuinfo`, lower case: the
/// single letters after `rv64`, then each `_`-separated name.
pub(crate) fn isa_string_extensions() -> Option<Vec<String>> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let line = text.lines().find(|l| l.split(':').next().is_some_and(|k| k.trim() == "isa"))?;
    let isa = line.split(':').nth(1)?.trim().to_ascii_lowercase();
    let rest = isa.strip_prefix("rv64")?;
    let mut parts = rest.split('_');
    let mut out: Vec<String> = parts.next()?.chars().map(String::from).collect();
    out.extend(parts.filter(|p| !p.is_empty()).map(String::from));
    Some(out)
}

/// The value of `csrr vlenb`, run through a two-word kernel. Only called once
/// the kernel has said V is there: without it the read traps.
fn read_vlenb() -> std::result::Result<u32, String> {
    let mut a = rv::Asm::new();
    a.csrr(rv::A0, rv::Csr::VLENB);
    a.ret();
    let code = a.finish().map_err(|e| e.to_string())?;
    let map = crate::jit::target::map_exec(code.bytes.len(), |m| m.copy_from_slice(&code.bytes)).map_err(|e| e.to_string())?;
    let f: crate::jit::target::generated_fn!(=> u64) = unsafe { std::mem::transmute(map.as_ptr()) };
    // SAFETY: two words just assembled: read a CSR V makes readable, return it.
    Ok(unsafe { f() } as u32)
}

fn cacheline() -> u32 {
    std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(64)
}

/// Read the features of the machine running this code.
pub fn detect() -> std::result::Result<Features, String> {
    let probe = hwprobe();
    let string = isa_string_extensions();
    if probe.is_none() && string.is_none() {
        return Err("neither riscv_hwprobe nor the isa line of /proc/cpuinfo answered".into());
    }
    let (base, ext) = probe.unwrap_or((0, 0));
    let listed = |name: &str| -> bool {
        let Some(list) = &string else { return false };
        match name {
            "fd" => list.iter().any(|e| e == "f") && list.iter().any(|e| e == "d"),
            n => list.iter().any(|e| e == n),
        }
    };
    let has = |name: &str| -> bool {
        let bit = IMA_EXT_0.iter().find(|(n, _)| *n == name).map(|&(_, b)| b);
        bit.is_some_and(|b| ext >> b & 1 == 1) || listed(name)
    };
    let ima = base & BASE_BEHAVIOR_IMA != 0 || (listed("i") && listed("m") && listed("a"));
    let v = has("v");
    Ok(Features {
        ima,
        fd: has("fd"),
        c: has("c"),
        v,
        zba: has("zba"),
        zbb: has("zbb"),
        zbs: has("zbs"),
        zfh: has("zfh"),
        zfhmin: has("zfhmin"),
        zfbfmin: has("zfbfmin"),
        zvfh: has("zvfh"),
        zvfhmin: has("zvfhmin"),
        zvfbfmin: has("zvfbfmin"),
        zvfbfwma: has("zvfbfwma"),
        zvfbfa: listed("zvfbfa"),
        zihintpause: has("zihintpause"),
        zacas: has("zacas"),
        zicond: has("zicond"),
        vlenb: if v { read_vlenb()? } else { 0 },
        cacheline: cacheline(),
    })
}
