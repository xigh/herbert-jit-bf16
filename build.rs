//! One decision taken at build time: the target must be one the engine can
//! emit for — x86_64 Linux or Windows (AVX2/AVX-512 kernels entered through
//! the `sysv64` ABI), aarch64 macOS (NEON/BF16 kernels entered through the
//! AAPCS64 `C` ABI), or riscv64 Linux (RVV 1.0 kernels entered through the
//! LP64D `C` ABI). Nothing else can run it.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let x86 = arch == "x86_64" && matches!(os.as_str(), "linux" | "windows");
    let arm = arch == "aarch64" && os == "macos";
    let riscv = arch == "riscv64" && os == "linux";
    if !(x86 || arm || riscv) {
        panic!(
            "herbert-bf16 builds for x86_64 Linux and Windows, aarch64 macOS and riscv64 Linux — \
             asked for {arch}-{os}. The engine JIT-emits its own kernels (AVX2/AVX-512 through \
             `sysv64`, NEON/BF16 through AAPCS64, RVV through LP64D); there is nothing behind \
             them elsewhere."
        );
    }
}
