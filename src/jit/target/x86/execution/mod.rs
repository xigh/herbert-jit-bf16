//! The x86-64 kernels of the execution plan: frame preparation, data
//! movement and output, the projection and attention traversals, the KV
//! append, the DeltaNet stages and the emitted math.

#[allow(unused_imports)]
pub(crate) use crate::jit::execution::*;

/// Generated memory movement, lookup and output kernels.
pub mod kernels;
/// Lowering of Q4 projection preparation and tile traversal.
/// Generated preparation and completion of a token block.
pub mod frame;
/// Lowering of bf16 R16 projection tile traversal.
pub mod bf16;
/// Lowering of SDPA head/chunk traversal and split-pass orchestration.
pub mod attention;
/// Lowering of KV compaction and append.
pub mod kv;
pub(crate) mod transforms;
pub(crate) mod math;
pub(crate) mod delta;
pub(crate) mod delta_scalar;
pub(crate) mod audit;
pub(crate) mod entry;
