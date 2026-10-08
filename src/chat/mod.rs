//! The chat layer: the tokenizer and its special tokens, the Qwen3.5 template,
//! the streaming detokenizer, the tools, and the session that ties them to
//! the engine.

pub mod detok;
pub mod session;
pub mod template;
pub mod tokens;
pub mod tools;

pub use session::{Block, Chat, TurnStats};
pub use tokens::Tokens;
pub use tools::{current_date_time, Tool};
