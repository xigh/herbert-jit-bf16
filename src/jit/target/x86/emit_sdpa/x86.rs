//! The AVX-512 SDPA emitter, in four files: the toolbox (`base.rs`), the
//! decode passes (`decode.rs`), the prefill block (`prefill.rs`), the
//! prologue, epilogue and final assembly (`assembly.rs`).

use super::{Pass, SdpaKernelSpec, SdpaPrefillSpec, SDPA_PREFILL_MAX_ROWS};
use crate::jit::ir::{Decline, ExpStrategy, SdpaForm};
use crate::jit::regs::{RegFile, Vlow, V};
use crate::core::error::{HerbertError, Result};
use iced_x86::code_asm::*;
use iced_x86::IcedError;

mod assembly;
mod base;
mod decode;
mod prefill;

// The TWO entry points of the x86 tier, re-published as-is: this is the only
// surface that the rest of the generator knows.
pub use base::{emit_sdpa, emit_sdpa_prefill};

// The rest only flows BETWEEN the four sub-modules, which reach it through
// `use super::*`. Plain `use` and not `pub(super) use`: only `base` has
// anything public enough to re-export, so on the other three the re-export was
// a no-op that the compiler rightly complained about.
use assembly::*;
pub(super) use base::*;
use decode::*;
use prefill::*;
