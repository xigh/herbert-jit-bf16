//! What every other module needs and nothing else: the model's `config.json`,
//! the error type, bf16 conversions, the ISA and memory checks, the CPU count
//! and the sampler.
//!
//! The crate builds for x86_64 Linux and Windows only (`lib.rs`, `build.rs`):
//! the engine emits AVX2/AVX-512 kernels and enters them through the `sysv64`
//! ABI, which exists nowhere else. The machine that runs the binary is checked
//! again at start (`isa::refuse_if_below_floor`).

pub mod avec;
pub mod config;
pub mod cpu_detection;
pub mod error;
pub mod isa;
pub mod memory;
pub mod sampler;
pub mod tensor;

pub use config::Config;
pub use error::HerbertError;
