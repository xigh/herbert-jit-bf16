//! The one decode-time decision the lowering asks about, read from the
//! installed knobs: whether a node at `m = 1` takes the whole team.

/// The pool at `m = 1`: projections, SDPA, DeltaNet.
#[derive(Clone, Copy, Debug)]
pub struct OptionsDecode {
    /// Whole team at `m = 1` ([`crate::jit::knobs::Knobs::decode_whole_team`]).
    pub pool_m1: bool,
}

/// The decode options in force.
pub fn decode() -> OptionsDecode {
    OptionsDecode { pool_m1: crate::jit::knobs::knobs().decode_whole_team }
}

/// Print every compiled GEMM plan once: off.
pub fn debug_plan() -> bool {
    false
}
