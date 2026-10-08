//! The checkpoint in memory: the model structures, the packed weight layout
//! the emitted GEMM reads, the loader that fills them, and the huge-page
//! buffer the weights live in.

pub mod hugepages;
pub mod loader;
pub mod model;
pub mod weight;

pub use model::{Attention, DecoderLayer, GatedDeltaNet, LayerBlock, Mlp, Model};
pub use weight::PackedBf16R16Weight;
