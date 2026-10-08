//! What happens during a pass once the layer is decided: today only the
//! order in which SDPA items are enumerated.

/// SDPA item enumeration order: `false` = by head (the historical order),
/// `true` = the `gqa` query heads of a KV head adjacent, so the second
/// traversal of `K` is served by the cache. Measured neutral on Qwen3.5-4B;
/// kept as the historical order.
pub(crate) fn sdpa_item_order_grouped() -> bool {
    false
}
