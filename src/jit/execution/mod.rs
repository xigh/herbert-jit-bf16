//! The execution plan and the whole-call program.
//!
//! A plan is a list of nodes — one kernel each, with its task count, its
//! participant limit and its dependencies; a program is the plan assembled
//! into one page of machine code that every participant runs from the first
//! block to the last, with the barriers between nodes emitted inline. The
//! bindings own the addresses and task counts each node reads.

mod plan;
pub use plan::*;
mod options;
pub use options::*;
pub(crate) mod runtime;
pub use runtime::{
    ExecutionSession, PmuTraceRecord, PmuTraceSnapshot, SyncTraceRecord, SyncTraceSnapshot,
};
mod pmu;
pub use pmu::PmuEvent;
mod program;
pub use program::CompiledExecution;

pub mod abi;
pub(crate) mod build;
pub(crate) mod math_data;

// The kernels of the plan, reachable from it: those of the build's target.
#[cfg(target_arch = "x86_64")]
pub use crate::jit::target::x86::execution::{attention, bf16, frame, kernels, kv};
#[cfg(target_arch = "aarch64")]
pub use crate::jit::target::aarch64::execution::{attention, bf16, frame, kernels, kv};
#[cfg(target_arch = "riscv64")]
pub use crate::jit::target::riscv64::execution::{attention, bf16, frame, kernels, kv};
