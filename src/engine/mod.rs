//! The model mounted into the generated engine: the bf16 loader with its
//! census, the geometry checks, the DeltaNet packs, the weight borrowing, the
//! `Engine` the chat drives, the trace writer and the settings `main` builds.

pub mod engine;
pub mod geometry;
pub mod loader;
pub mod packs;
pub mod trace;
pub mod tuning;
pub mod weights_of;

pub use engine::Engine;
pub use tuning::{Measure, Tuning};
