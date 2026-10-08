//! herbert-bf16 — Qwen3.5-4B in bf16, no quantization, every kernel emitted at
//! run time, driven by one `main` that runs a multi-turn chat.
//!
//! The modules follow the order in which `main` uses them:
//!
//! | module   | what it does |
//! |----------|--------------|
//! | `core`   | the model's `config.json`, bf16 conversions, the ISA check, the memory check, the sampler |
//! | `pool`   | the worker threads the generated program runs on, pinned to physical cores |
//! | `model`  | the checkpoint read from safetensors into packed bf16 weights (R16 layout) |
//! | `jit`    | the emitters (GEMM, norms, attention, GatedDeltaNet), the generated program and its scheduler |
//! | `engine` | the model mounted into the two generated regimes (prefill, decode) and driven step by step |
//! | `chat`   | the Qwen3.5 chat template, tokens, streaming detokenizer, tools, and the session |
//!
//! Numerical contract: weights bf16, activations bf16, accumulators f32, KV
//! cache bf16. Nothing is quantized.
//!
//! Three targets emit the kernels: x86-64 (AVX2/AVX-512, `iced-x86`), AArch64
//! (NEON with the BF16 extension, the `a64` crate of this workspace) and
//! RISC-V (RVV 1.0 with the bf16 extensions, the `rv` crate).

// Index loops over emitted registers, wide kernel signatures and the
// `engine::engine` / `model::model` files read better than what these lints ask for.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments, clippy::type_complexity, clippy::module_inception)]

#[cfg(not(any(
    all(target_arch = "x86_64", any(target_os = "linux", target_os = "windows")),
    all(target_arch = "aarch64", target_os = "macos"),
    all(target_arch = "riscv64", target_os = "linux")
)))]
compile_error!(
    "herbert-bf16 builds for x86_64 Linux and Windows, aarch64 macOS and riscv64 Linux: the \
     engine emits its own AVX2/AVX-512, NEON/BF16 or RVV kernels and calls them through \
     `sysv64`, AAPCS64 or LP64D."
);

pub mod core;
pub mod pool;
pub mod model;
pub mod jit;
pub mod engine;
pub mod chat;
