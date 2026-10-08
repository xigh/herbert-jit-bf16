//! The call structures of the execution plan: the frame, the task grids, and
//! the `#[repr(C)]` records every kernel reads through its one pointer
//! argument. The lowering fills them; the x86 kernels consume them.

use super::math_data::Tables;
use super::ExecutionError;

// ── bf16 ──────────────────────────────────────────────────────────────────

/// Stable projection arguments. Input format and output width belong to the
/// compiled kernel; this context supplies only addresses and the current rows.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bf16Call {
    /// bf16 or f32 activations, matching the compiled input format.
    pub activations: *const u8,
    /// Existing R16 packed bf16 weights.
    pub weights: *const u16,
    /// bf16 or f32 output, matching the compiled output format.
    pub output: *mut u8,
    /// Inline rows when `row_count` is null.
    pub rows: usize,
    /// Optional rows produced by a preceding generated block node.
    pub row_count: *const usize,
}

/// Up to four projections of one input in one node: their tasks form one
/// index space, the column tiles of every part in order
/// ([`crate::jit::knobs::Knobs::fuse_projections`]).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Bf16MultiCall {
    /// The calls of the parts; `count` of them are live.
    pub calls: [Bf16Call; 4],
    /// How many parts the node has.
    pub count: usize,
}

/// The packing of `A` for the matrix engine: `source` row-major `[rows, K]`
/// bf16, `dest` in blocks of sixteen rows by pairs of `K`. Rows past `rows` in
/// the last block repeat the last row: computed, never stored.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PackCall {
    /// Row-major bf16 activations.
    pub source: *const u16,
    /// The packed blocks.
    pub dest: *mut u16,
    /// Current rows, produced by block preparation.
    pub rows: *const usize,
}

// ── kv ────────────────────────────────────────────────────────────────────

/// Cache geometry fixed at mount. The calling model validates the full call's
/// absolute position range and owns the buffers for this geometry.
#[derive(Clone, Copy, Debug)]
pub struct KvSpec {
    /// KV heads.
    pub heads: usize,
    /// bf16 elements per head.
    pub head_dim: usize,
    /// bf16 elements reserved per KV head.
    pub stride: usize,
    /// Positions reserved per KV head.
    pub capacity: usize,
    /// Maximum rows in one block.
    pub max_batch: usize,
    /// AArch64: the append also writes each position into the R16 copies of
    /// `KvCall::keys_r16` and `KvCall::values_r16` (the block attention,
    /// `Knobs::attn_block`). The x86 append refuses it.
    pub r16: bool,
}
impl KvSpec {
    pub(crate) fn validate(self) -> Result<(), ExecutionError> {
        let valid = self.heads > 0
            && self.head_dim > 0
            && self.max_batch > 0
            && self.capacity >= self.max_batch
            && self
                .capacity
                .checked_mul(self.head_dim)
                .is_some_and(|n| n <= self.stride)
            && self
                .heads
                .checked_mul(self.stride)
                .and_then(|n| n.checked_mul(2))
                .is_some_and(|n| n <= isize::MAX as usize)
            && self
                .heads
                .checked_mul(self.head_dim)
                .and_then(|n| n.checked_mul(2))
                .is_some_and(|n| n <= i32::MAX as usize)
            && self.stride <= i32::MAX as usize / 2
            // The R16 copies: whole blocks of sixteen positions and of sixteen dimensions.
            && (!self.r16 || (self.head_dim.is_multiple_of(16) && self.capacity.is_multiple_of(16)));
        if !valid {
            return Err(ExecutionError(
                "invalid generated KV reserve geometry".into(),
            ));
        }
        Ok(())
    }
}
/// Stable cache addresses. Rows and past are produced by block preparation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KvCall {
    /// Head-major keys.
    pub keys: *mut u16,
    /// Head-major values.
    pub values: *mut u16,
    /// Current block's token-major projected/normalized keys.
    pub input_keys: *const u16,
    /// Current block's token-major values.
    pub input_values: *const u16,
    /// Current rows.
    pub rows: *const usize,
    /// Absolute first position of this block.
    pub past: *const usize,
    /// Under `KvSpec::r16`, per KV head (`stride` elements apart): the keys
    /// in the R16 layout with the positions as columns
    /// (`[capacity/16 blocks][head_dim/2 pairs][16][2]`), and the values
    /// transposed in the R16 layout with the dimensions as columns
    /// (`[head_dim/16 blocks][capacity/2 pairs][16][2]`) — the weights of
    /// the two GEMMs of the block attention. Null otherwise.
    pub keys_r16: *mut u16,
    /// The values' R16 copy (`Vᵀ`), as above.
    pub values_r16: *mut u16,
}

// ── attention ─────────────────────────────────────────────────────────────

/// Immutable per-layer geometry, resolved at mount. All strides here are in bf16 elements.
#[derive(Clone, Copy, Debug)]
pub struct AttentionSpec {
    /// Query heads.
    pub heads: usize,
    /// Query heads per KV head.
    pub gqa: usize,
    /// Elements per head.
    pub head_dim: usize,
    /// Elements between query rows (can include the output gate).
    pub q_row: usize,
    /// Elements between query heads (can include the output gate).
    pub q_head: usize,
    /// Elements between output rows.
    pub out_row: usize,
    /// Enumerate the query heads of a KV head adjacently.
    pub grouped: bool,
}
/// Stable addresses and generated state for one attention operation. The caller
/// validates all buffer capacities before entry. Count/position pointers remain immutable during the operation.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AttentionCall {
    /// Query activations.
    pub q: *const u16,
    /// Head-major KV buffers.
    pub keys: *const u16,
    /// Head-major values.
    pub values: *const u16,
    /// Token-major output.
    pub output: *mut u16,
    /// Current block rows, generated by block preparation.
    pub rows: *const usize,
    /// Absolute position of the first query in this block.
    pub past: *const usize,
    /// bf16 elements per KV head.
    pub kv_stride: usize,
    /// Attention scale.
    pub scale: f32,
    /// Softmax sink (ignored when absent from the compiled shape).
    pub sink: f32,
    /// The R16 copies of `KvCall` (`kv_stride` elements per KV head), read
    /// by the AArch64 block attention; null without it.
    pub keys_r16: *const u16,
    /// The values' R16 copy (`Vᵀ`), as above.
    pub values_r16: *const u16,
}

// ── frame ─────────────────────────────────────────────────────────────────

/// Stable storage owned by a mounted model. No Rust references to fields may
/// remain live while generated code writes the frame.
#[repr(C)]
#[derive(Debug)]
pub struct CallFrame {
    /// Pointer to the caller's token IDs for this call.
    pub tokens: *const u32,
    /// Number of token IDs in this call.
    pub total_tokens: usize,
    /// Absolute model position at the current block; initialized before launch.
    pub position: usize,
    /// Tokens completed in this call; initialized to zero before launch.
    pub processed: usize,
    /// Current block length, written by `begin_block`.
    pub rows: usize,
    /// First token of the current block, written by `begin_block`.
    pub token_cursor: *const u32,
    /// One if this is the last block, zero otherwise.
    pub last_block: usize,
    /// Preallocated task counters, one for each mounted `TaskGrid`.
    pub counts: *mut usize,
}
/// `ceil(block_rows / rows_per_task) * tasks_per_tile`, optionally only on the
/// final block. A grid with `rows_per_task == max_batch` describes one task per
/// block (or one head operation only on the final block).
#[derive(Clone, Copy, Debug)]
pub struct TaskGrid {
    /// Rows covered by one tile, e.g. GEMM MR or SDPA prefill rows.
    pub rows_per_task: usize,
    /// Tasks per row tile, e.g. number of column tiles or attention heads.
    pub tasks_per_tile: usize,
    /// Skip this grid before the last block, e.g. final norm and language head.
    pub last_block_only: bool,
}
impl TaskGrid {
    /// Validate the grid and compute its maximum task count at mount.
    pub fn capacity(self, max_batch: usize) -> Result<usize, ExecutionError> {
        if self.rows_per_task == 0 || max_batch == 0 {
            return Err(ExecutionError("empty generated block/grid geometry".into()));
        }
        max_batch
            .div_ceil(self.rows_per_task)
            .checked_mul(self.tasks_per_tile)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| ExecutionError("generated task grid capacity overflow".into()))
    }
}

// ── delta ─────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GateCall {
    pub ab: *const f32,
    pub a_log: *const f32,
    pub bias: *const u16,
    pub beta: *mut f32,
    pub decay: *mut f32,
    pub tables: *const Tables,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct InterleaveCall {
    pub a: *const f32,
    pub b: *const f32,
    pub output: *mut f32,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct L2Call {
    pub qkv: *mut f32,
    pub eps: f32,
}
/// The context of [`l2_gates`]: the [`L2Call`] of the row's Q and K heads,
/// then the [`GateCall`] of its gates, in one word-aligned block.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct L2GatesCall {
    pub l2: L2Call,
    pub gates: GateCall,
}
#[derive(Clone, Copy)]
pub(crate) struct Shape {
    pub nk: usize,
    pub nv: usize,
    pub hkd: usize,
    pub hvd: usize,
    pub ck: usize,
}
impl Shape {
    pub fn key(self) -> usize {
        self.nk * self.hkd
    }
    pub fn value(self) -> usize {
        self.nv * self.hvd
    }
    pub fn conv(self) -> usize {
        2 * self.key() + self.value()
    }
    pub fn check(self) -> Result<(), ExecutionError> {
        if self.nk == 0
            || self.nv == 0
            || !self.nv.is_multiple_of(self.nk)
            || self.hkd == 0
            || self.hvd == 0
            || self.ck < 2
            || [self.nk, self.nv, self.hkd, self.hvd, self.ck]
                .iter()
                .any(|&n| n > 1 << 16)
            || self
                .conv()
                .checked_mul(self.ck)
                .is_none_or(|n| n > i32::MAX as usize / 4)
            || self
                .nv
                .checked_mul(self.hkd)
                .and_then(|n| n.checked_mul(self.hvd))
                .is_none_or(|n| n > i32::MAX as usize / 4)
        {
            return Err(ExecutionError("invalid generated DeltaNet geometry".into()));
        }
        Ok(())
    }
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RecurrenceCall {
    pub state: *mut f32,
    pub output: *mut f32,
    pub qkv: *const f32,
    pub beta: *const f32,
    pub decay: *const f32,
    pub rows: *const usize,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ConvCall {
    pub data: *mut f32,
    pub weights: *const u16,
    pub state: *mut f32,
    pub rows: *const usize,
    pub tables: *const Tables,
}
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct NormCall {
    pub output: *mut f32,
    pub gate: *const f32,
    pub weights: *const u16,
    pub output_bf16: *mut u16,
    pub tables: *const Tables,
    pub eps: f32,
}

