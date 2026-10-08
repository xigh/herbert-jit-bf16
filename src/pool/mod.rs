//! The worker pool the generated program runs on: the collective, the CPU
//! topology it pins to, and the settings installed before it is created.
//! Nothing here knows a model or a kernel.

pub mod settings;
pub mod thread_pool;
pub mod topology;
