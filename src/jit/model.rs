//! The runner: the model as a whole, driven token by token.
//!
//! What it holds: the KV cache of every attention layer (bf16, one slice per
//! layer at its own `head_dim`), the two GatedDeltaNet states (conv and
//! recurrent, f32), the RoPE tables in bf16, the logits of the last row, the
//! position. What it does: `mount_execution` lowers both regimes into
//! generated programs (`mounted.rs`); `prefill_mounted` appends rows to the
//! state, `generate_mounted` decodes in one launch with a callback per token.
//! [`Instantaneous`] is the whole state frozen, restorable to an earlier
//! position on a pure-attention model and to its exact position once a
//! DeltaNet state is part of it; it is what the chat saves to disk.

use crate::jit::schedule::{Activation, Budget, DeltaShape, LayerWeights, Mixer, ModelShape, Scheduler};
use crate::core::error::{HerbertError, Result};

/// What a layer does with the KV cache — **one entry per layer**, in order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanKv {
    /// The layer owns its cache and reserves it for its `head_dim`.
    Own(usize),
    /// No cache — linear mixer.
    Absent,
}

/// The weights of the entire model, **borrowed** from the engine: everything
/// arrives in raw slices. Projections are in R16 format, norms and the embedding
/// table are bf16 vectors.
pub struct ModelWeights<'a> {
    /// Embedding table, `[vocab, hidden]` row-major, bf16.
    pub embed: &'a [u16],
    /// One input per layer, in order.
    pub layers: &'a [LayerWeights<'a>],
    /// Final RMSNorm weights.
    pub final_norm: &'a [u16],
    /// Language head, R16 format, `N = vocab`, `K = hidden`.
    ///
    /// With `tie_word_embeddings`, it's the R16 packing of `embed` — the
    /// base loader already produces it, there's nothing to decide here.
    pub lm_head: &'a [u16],
}

/// What is reserved once, at prompt launch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ModelBudget {
    /// Reserved positions in the KV. The budget in bytes follows
    /// exactly (see [`Runner::reserved_bytes`]).
    pub kv_reserve_tokens: usize,
    /// Size of the vocabulary, hence of the logits buffer.
    pub vocab: usize,
    /// Positions processed in a single prefill pass (see [`Budget::max_batch`]).
    /// `1` = repeated decode prefill; the default cut-off is
    /// [`crate::jit::DEFAULT_MAX_BATCH`].
    pub max_batch: usize,
}

/// A set of `cos`/`sin` tables from the loader, for one value of `rotary_ndims`.
///
/// The tables arrive as f32, arranged `[positions, rotary_ndims/2]`; the
/// [`Runner`] rounds them to bf16 once, during setup. One per distinct
/// `rotary_ndims`: the frequencies depend on the width, so a narrow set is
/// not a prefix of a wide one.
pub struct RopeTableSet<'a> {
    /// `rotary_ndims/2` inputs per position.
    pub rotary_ndims: usize,
    /// `cos`, f32, `[positions, rotary_ndims/2]`.
    pub cos: &'a [f32],
    /// `sin`, same layout.
    pub sin: &'a [f32],
}

/// What to reserve for the model's GatedDeltaNet mixers.
///
/// Sibling of [`ModelBudget`]: it describes **only the reserve**, never the
/// geometry of a layer, which is read from its
/// [`DeltaWeights`](crate::jit::schedule::DeltaWeights).
///
/// `DeltaShape` alone is not enough: the recurrent state size is
/// `num_v_heads · head_k_dim · head_v_dim`, and `head_k_dim` cannot be derived from it
/// (`conv_dim` mixes `num_k_heads` and `head_k_dim` into a single product). Hence
/// [`Self::recurrent_stride`], read by the adapter on the layers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeltaBudget {
    /// The maxima **over the layers** that size the scheduler's
    /// buffers (see [`Scheduler::with_geometries_delta`]).
    pub shape: DeltaShape,
    /// Number of `linear_attention` layers — the only ones that carry state.
    pub n_layers: usize,
    /// `num_v_heads · head_k_dim · head_v_dim`, **maximum across layers**:
    /// the length of a recurrent state slot.
    pub recurrent_stride: usize,
}

/// The size of the logits buffer: the vocabulary rounded up to the next
/// multiple of 16. The packed head already carries its zero tail block; the
/// logits buffer needs the same room so the last tile can be stored whole.
pub const fn logits_buffer_size(vocab: usize) -> usize {
    vocab.next_multiple_of(crate::jit::ir::ROWS_INTERLEAVED)
}

/// Serialized snapshot format version. **Increment it as soon as the
/// structure changes**: KV state bytes have no redundancy that would allow
/// detecting that they are being read out of order.
const SNAPSHOT_FORMAT: u32 = 1;

/// 64 GiB — larger than any real snapshot (10.2 GB measured at 128k
/// positions on the Gemma 4 26B), small enough that a corrupted length
/// is refused instead of being allocated.
const MAX_BLOCK_SNAPSHOT: usize = 64 << 30;

fn length(r: &mut impl std::io::Read, element_size: usize) -> std::io::Result<usize> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    let n = u64::from_le_bytes(b) as usize;
    if n.saturating_mul(element_size) > MAX_BLOCK_SNAPSHOT {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("block of {} elements: beyond the ceiling", n),
        ));
    }
    Ok(n)
}

fn write_usizes(w: &mut impl std::io::Write, v: &[usize]) -> std::io::Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    for x in v {
        w.write_all(&(*x as u64).to_le_bytes())?;
    }
    Ok(())
}

fn read_usizes(r: &mut impl std::io::Read) -> std::io::Result<Vec<usize>> {
    let n = length(r, 8)?;
    let mut v = Vec::with_capacity(n);
    let mut b = [0u8; 8];
    for _ in 0..n {
        r.read_exact(&mut b)?;
        v.push(u64::from_le_bytes(b) as usize);
    }
    Ok(v)
}

fn write_u16s(w: &mut impl std::io::Write, v: &[u16]) -> std::io::Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    // A block, not u16 by u16: `kv` is in gigabytes.
    let mut buffer = Vec::with_capacity(v.len() * 2);
    for x in v {
        buffer.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buffer)
}

fn read_u16s(r: &mut impl std::io::Read) -> std::io::Result<Vec<u16>> {
    let n = length(r, 2)?;
    let mut raw = vec![0u8; n * 2];
    r.read_exact(&mut raw)?;
    Ok(raw
        .as_chunks::<2>().0.iter()
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

fn write_f32s(w: &mut impl std::io::Write, v: &[f32]) -> std::io::Result<()> {
    w.write_all(&(v.len() as u64).to_le_bytes())?;
    let mut buffer = Vec::with_capacity(v.len() * 4);
    for x in v {
        buffer.extend_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buffer)
}

fn read_f32s(r: &mut impl std::io::Read) -> std::io::Result<Vec<f32>> {
    let n = length(r, 4)?;
    let mut raw = vec![0u8; n * 4];
    r.read_exact(&mut raw)?;
    Ok(raw
        .as_chunks::<4>().0.iter()
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// `Debug` **hand-written**: a `derive` would dump gigabytes of `kv`
/// as soon as an `Instantaneous` appears in an `unwrap_err()` or an error message.
impl std::fmt::Debug for Instantaneous {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Instantane(pos={} rempli={} {} o)",
            self.pos,
            self.is_filled,
            self.bytes()
        )
    }
}

/// A frozen [`Runner`] state at a position: the grain of the **prefix cache**
/// of the server (agentic prompt: only re-prefill the delta).
///
/// Reserved once per [`Runner::snapshot_reserve`] (setup), filled by
/// [`Runner::snapshot_take`], replayed by [`Runner::restore_from`] —
/// these two do not allocate. It carries everything that varies between passes:
/// the resident KV per layer, the two DeltaNet states, the logits of the
/// last token (an exact hit makes them free without any recalculation) and `pos`.
/// The **tokens** that identify this state live at the caller: here there is
/// only state, no key.
pub struct Instantaneous {
    /// The original geometry — the footprint that invalidates the snapshot when the
    /// `Runner` is rebuilt (different buffer, different pass, different model).
    kv_offsets: Vec<usize>,
    /// Offsets in [`Self::kv`], `n_layers + 1` entries, same convention as
    /// `Runner::kv_offsets` but at the snapshot's capacities.
    snap_offsets: Vec<usize>,
    /// Positions the slot can hold, per layer:
    /// `min(kv_positions[l], max_positions)`, `0` on a layer without cache.
    snap_positions: Vec<usize>,
    kv: Vec<u16>,
    dn_conv: Vec<f32>,
    dn_recurrent: Vec<f32>,
    logits: Vec<u16>,
    pos: usize,
    is_filled: bool,
}

impl Instantaneous {
    /// Bytes held by the slot: KV, DeltaNet conv and recurrent states, logits.
    pub fn byte_size(&self) -> usize {
        self.kv.len() * 2 + (self.dn_conv.len() + self.dn_recurrent.len()) * 4 + self.logits.len() * 2
    }

    /// Frozen state positions (`0` until nothing has been taken).
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Has a state been taken?
    pub fn is_filled(&self) -> bool {
        self.is_filled
    }

    /// Becomes an empty slot again (buffers remain reserved).
    pub fn clear(&mut self) {
        self.is_filled = false;
        self.pos = 0;
    }

    /// The bf16 logits of the last token of the frozen state — what an **exact** hit
    /// renders without recomputing anything. Wider than the vocabulary
    /// (rounded ROWS_INTERLEAVED): the caller slices at its `vocab`.
    pub fn logits(&self) -> &[u16] {
        &self.logits
    }

    /// Bytes reserved by this slot.
    pub fn bytes(&self) -> usize {
        2 * (self.kv.len() + self.logits.len())
            + 4 * (self.dn_conv.len() + self.dn_recurrent.len())
            + std::mem::size_of::<usize>()
                * (self.kv_offsets.len() + self.snap_offsets.len() + self.snap_positions.len())
    }

    // ── Serialization: the snapshot on disk ─────────────────────────
    //
    // The fields are private and must remain so: a snapshot whose offsets
    // were reconstructed elsewhere would be a misaligned KV state, hence
    // plausible but false text. Serialization lives with the structure.
    //
    // The format writes EVERY length (never inferred from a dimension) and ends
    // with the geometry **re-read then compared**: a snapshot taken under a
    // different KV guard is not "almost right", it is unreadable, and it is a
    // refusal that is required, not a conversion.

    /// Writes the snapshot. `is_filled = false` is refused: an empty slot has nothing
    /// to say, and writing it would produce a file we would read for nothing.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        if !self.is_filled {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "empty snapshot: nothing to write",
            ));
        }
        w.write_all(&SNAPSHOT_FORMAT.to_le_bytes())?;
        write_usizes(w, &self.kv_offsets)?;
        write_usizes(w, &self.snap_offsets)?;
        write_usizes(w, &self.snap_positions)?;
        write_u16s(w, &self.kv)?;
        write_u16s(w, &self.logits)?;
        write_f32s(w, &self.dn_conv)?;
        write_f32s(w, &self.dn_recurrent)?;
        w.write_all(&(self.pos as u64).to_le_bytes())
    }

    /// Reads an instant. Every length is bounded: a corrupted byte must not
    /// request a 2^64 allocation.
    pub fn read(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut v = [0u8; 4];
        r.read_exact(&mut v)?;
        if u32::from_le_bytes(v) != SNAPSHOT_FORMAT {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "snapshot of another version of format",
            ));
        }
        Ok(Self {
            kv_offsets: read_usizes(r)?,
            snap_offsets: read_usizes(r)?,
            snap_positions: read_usizes(r)?,
            kv: read_u16s(r)?,
            logits: read_u16s(r)?,
            dn_conv: read_f32s(r)?,
            dn_recurrent: read_f32s(r)?,
            pos: {
                let mut b = [0u8; 8];
                r.read_exact(&mut b)?;
                u64::from_le_bytes(b) as usize
            },
            is_filled: true,
        })
    }

    /// Does the frozen state return to `p`? The predicate of [`Runner::restore_from`],
    /// without executing it: a filled slot rewinds to any earlier position on a
    /// pure-attention model, and only to its exact position once a DeltaNet
    /// state is part of it (the recurrence does not go backwards).
    pub fn restorable_at(&self, p: usize) -> bool {
        self.is_filled && p <= self.pos && (p == self.pos || self.dn_recurrent.is_empty())
    }
}

/// The model scheduler: it holds the KV, the RoPE tables, and the logits.
pub struct Runner {
    execution_options: crate::jit::execution::ExecutorOptions,
    execution: Option<mounted::ModelExecution>,
    execution_session: crate::jit::execution::ExecutionSession,
    generated_argmax: Box<u32>,
    /// Tokens fed by a one-launch generation, `[reserve]`.
    generation_tokens: crate::core::avec::AVec<u32>,
    argmax_valid: bool,
    sched: Scheduler,
    shape: ModelShape,
    n_layers: usize,
    /// KV of all layers, **exact per layer**:
    /// `Σ_l 2 · n_kv_heads · kv_strides[l]` bf16. The stride is no longer constant,
    /// so the offset of a layer is no longer a product — it is read from
    /// [`Self::kv_offsets`].
    kv: crate::core::avec::AVec<u16>,
    /// Cumulative offsets in [`Self::kv`], in bf16: `n_layers + 1` inputs.
    /// Layer `l` occupies `kv_offsets[l]..kv_offsets[l+1]`, keys first
    /// (`n_kv_heads · kv_strides[l]`), values next. A layer without cache
    /// occupies an **empty** interval.
    ///
    /// Computed **once**, at setup: nothing allocates afterwards.
    kv_offsets: Vec<usize>,
    /// Reserved length per **head and per layer**, in bf16 —
    /// `kv_reserve_tokens · head_dim_l`, with the `head_dim` **of the layer**.
    /// Equals `0` on a layer that has no cache (linear mixer): it then does
    /// not reserve a single byte.
    ///
    kv_strides: Vec<usize>,
    /// Positions reserved per KV head, on every layer that has a cache. This
    /// ceiling bounds a pass, and it is explicitly guarded in `advance`.
    kv_reserve_tokens: usize,
    /// AArch64 block attention (`Knobs::attn_block`): each layer with a cache
    /// holds, after its keys and values, their R16 copies — the keys with
    /// the positions as columns, the values transposed with the dimensions
    /// as columns (`KvCall::keys_r16`), `n_kv_heads · kv_strides[l]` each.
    /// The KV append keeps them; a restoration rebuilds them from the cache.
    kv_r16: bool,
    /// One set of tables per **distinct** `rotary_ndims`:
    /// `(rotary_ndims, cos, sin)`, in bf16, `[positions, rotary_ndims/2]`.
    /// A linear sweep, like the activation kernels: one or two sets, and
    /// nothing must allocate during a pass.
    ropes: Vec<(usize, crate::core::avec::AVec<u16>, crate::core::avec::AVec<u16>)>,
    /// Logits of the last pass, **in bf16**.
    ///
    /// **The buffer is wider than the vocabulary**: rounded up to the next
    /// multiple of 16 (`ir::ROWS_INTERLEAVED`), the room of the head's last
    /// tile ([`logits_buffer_size`]).
    logits: crate::core::avec::AVec<u16>,
    /// The **actual** vocabulary, shorter than `logits` when it is not a
    /// multiple of 16. Everything returned to the caller stops there.
    vocab: usize,
    /// Positions already in the KV.
    pos: usize,
    /// Rolling states of causal conv1d, **one slice per layer
    /// `linear_attention`**: `n_delta · dn_conv_stride` f32. Empty on a
    /// model without a linear mixer.
    ///
    /// Next to the KV, and for the same reason: it is a persistent state that
    /// the scheduler writes and never allocates.
    dn_conv: crate::core::avec::AVec<f32>,
    /// `conv_dim · conv_kernel` — the length of a conv slot.
    dn_conv_stride: usize,
    /// Recurrent states: `n_delta · dn_rec_stride` f32.
    dn_recurrent: crate::core::avec::AVec<f32>,
    /// `num_v_heads · head_k_dim · head_v_dim` — a state location.
    dn_rec_stride: usize,
}

/// What a prefill pass returns to the caller — **nothing but numbers**.
///
/// `Copy`, three scalars, no ownership: the structure is built
/// in the register at the edge of the hot loop and passed by value. No
/// `String`, no `format!`: it's the **CALLER** who formats, outside the loop.
///
/// `traites` and `total` count **tokens of the current prompt**, not
/// passes: that is what a progress bar wants. `seconds` is the time
/// elapsed since the start of this `prefill` (not since the pass), which makes
/// [`Progression::tokens_per_second`] naturally smoothed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Progression {
    /// Tokens of the prompt already absorbed, including the current pass.
    pub traites: usize,
    /// Total prompt tokens to absorb for this call.
    pub total: usize,
    /// Seconds elapsed since the start of this prefill call.
    pub seconds: f32,
}

impl Progression {
    /// The average throughput since the start of the call, in tokens/s.
    ///
    /// A division, no allocation. `0.0` as long as the clock hasn't moved —
    /// never display infinity.
    #[inline]
    pub fn tokens_per_second(&self) -> f32 {
        if self.seconds > 0.0 {
            self.traites as f32 / self.seconds
        } else {
            0.0
        }
    }

    /// The absorbed fraction, in `[0, 1]`.
    #[inline]
    pub fn fraction(&self) -> f32 {
        if self.total > 0 {
            self.traites as f32 / self.total as f32
        } else {
            0.0
        }
    }
}

impl Runner {
    /// Compile the kernels and reserve all model state. The attention geometry
    /// may vary by layer:
    /// `head_dim` and `rotary_ndims` carried by the layer.
    ///
    /// `geometries` = the **distinct** `(head_dim, rotary_ndims)` pairs of the
    /// model, as `widths` is for the MLP. `tables` = a set of `cos`/`sin`
    /// per distinct `rotary_ndims`, each laid out as `[positions, rotary_ndims/2]`.
    ///
    /// `shape.head_dim` must be the **maximum** across layers: it is this one that
    /// sizes the attention buffers. The **KV stride**, however, is no longer its
    /// own since G3: it comes from `plan_kv`.
    ///
    /// `plan_kv` = **one input per layer, in order**:
    /// `Some(head_dim)` if the layer has its own KV cache, `None` if it
    /// does not (linear mixer). An empty
    /// slice requests a uniform `shape.head_dim` reserve on all layers.
    ///
    /// `plan_kv` must match the weights: a `Mixer::Attention` layer
    /// declared `None`, or declared narrower than its `LayerWeights::head_dim`,
    /// is refused **explicitly** in the first pass by the scheduler's saturation guard,
    /// never by an out-of-bounds indexing.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_geometries(
        shape: ModelShape,
        n_layers: usize,
        budget: ModelBudget,
        tables: &[RopeTableSet<'_>],
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        plan_kv: &[PlanKv],
        delta: Option<DeltaBudget>,
        execution_options: crate::jit::execution::ExecutorOptions,
    ) -> Result<Self> {
        Self::ascend(shape, n_layers, budget, tables, widths, geometries, plan_kv, delta, execution_options)
    }

    /// The constructor body.
    #[allow(clippy::too_many_arguments)]
    fn ascend(
        shape: ModelShape,
        n_layers: usize,
        budget: ModelBudget,
        tables: &[RopeTableSet<'_>],
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        plan_kv: &[PlanKv],
        delta: Option<DeltaBudget>,
        execution_options: crate::jit::execution::ExecutorOptions,
    ) -> Result<Self> {
        // Each declared `rotary_ndims` must have its set of tables, and each
        // set must cover the reserve. Stating this here, once: a table too
        // short is caught at the layer, but a **missing** table would cause
        // a refusal during the pass for something visible at setup.
        for &(_, rn) in geometries {
            let half = rn / 2;
            let want = budget.kv_reserve_tokens * half;
            if !tables.iter().any(|t| t.rotary_ndims == rn) {
                return Err(HerbertError::Backend(format!(
                    "backend_jit: no RoPE lookup tables for rotary_ndims={rn}"
                )));
            }
            for t in tables.iter().filter(|t| t.rotary_ndims == rn) {
                if t.cos.len() < want || t.sin.len() < want {
                    return Err(HerbertError::Backend(format!(
                        "backend_jit : tables RoPE too short for rotary_ndims={rn} \
                         ({} reserved positions × {half} = {want}, received \
                         cos={} sin={})",
                        budget.kv_reserve_tokens,
                        t.cos.len(),
                        t.sin.len()
                    )));
                }
            }
        }
        let b = Budget {
            kv_reserve_tokens: budget.kv_reserve_tokens,
            max_batch: budget.max_batch.max(1),
        };
        let sched = match delta {
            Some(d) => Scheduler::with_geometries_delta(shape, b, widths, geometries, d.shape)?,
            None => Scheduler::with_geometries(shape, b, widths, geometries)?,
        };
        // ── Exact per-layer KV reserve ──────────────────────────────────
        //
        // One step per layer and **cumulative** offsets, calculated here, once:
        // since the step is no longer constant, a layer's offset is no longer a
        // product. None of this is recalculated during a pass.
        //
        if !plan_kv.is_empty() && plan_kv.len() != n_layers {
            return Err(HerbertError::Backend(format!(
                "backend_jit : plan KV for {} layers expected for {n_layers}",
                plan_kv.len()
            )));
        }
        // No plan = uniform reservation at the shape's `head_dim`: the
        // pre-G3 behavior, for single-geometry constructors.
        let plan: Vec<PlanKv> = if plan_kv.is_empty() {
            vec![PlanKv::Own(shape.head_dim); n_layers]
        } else {
            plan_kv.to_vec()
        };
        let kv_strides: Vec<usize> = plan
            .iter()
            .map(|p| match *p {
                PlanKv::Own(h) => budget.kv_reserve_tokens * h,
                PlanKv::Absent => 0,
            })
            .collect();
        let kv_r16 = cfg!(target_arch = "aarch64") && crate::jit::knobs::knobs().attn_block;
        let mut kv_offsets = Vec::with_capacity(n_layers + 1);
        let mut kv_total = 0usize;
        kv_offsets.push(0);
        for &st in &kv_strides {
            kv_total += if kv_r16 { 4 } else { 2 } * shape.n_kv_heads * st;
            kv_offsets.push(kv_total);
        }
        // The two persistent states of the linear mixer. `AVec::zeroed` already
        // makes them blank: a new `Runner` does not need `reset`.
        let (dn_conv_stride, dn_rec_stride, n_delta) = match delta {
            Some(d) => (
                d.shape.conv_dim * d.shape.conv_kernel,
                d.recurrent_stride,
                d.n_layers,
            ),
            None => (0, 0, 0),
        };
        let runner = Self {
            execution_options,
            execution: None,
            execution_session: crate::jit::execution::ExecutionSession::new(),
            generated_argmax: Box::new(0),
            generation_tokens: crate::core::avec::AVec::new(),
            argmax_valid: false,
            sched,
            shape,
            n_layers,
            kv: crate::core::avec::AVec::zeroed(kv_total),
            kv_offsets,
            kv_strides,
            kv_reserve_tokens: budget.kv_reserve_tokens,
            kv_r16,
            // One set per distinct `rotary_ndims`, converted to bf16 **once**.
            // Sets not declared by `geometries` are not retained: a
            // layer requesting one would be refused by name.
            ropes: {
                let mut v: Vec<(usize, crate::core::avec::AVec<u16>, crate::core::avec::AVec<u16>)> = Vec::new();
                for t in tables {
                    let rn = t.rotary_ndims;
                    if !geometries.iter().any(|&(_, r)| r == rn) || v.iter().any(|(r, _, _)| *r == rn) {
                        continue;
                    }
                    let want = budget.kv_reserve_tokens * (rn / 2);
                    v.push((
                        rn,
                        crate::core::avec::AVec::from_vec(t.cos[..want].iter().copied().map(to_bf16).collect()),
                        crate::core::avec::AVec::from_vec(t.sin[..want].iter().copied().map(to_bf16).collect()),
                    ));
                }
                v
            },
            logits: crate::core::avec::AVec::zeroed(logits_buffer_size(budget.vocab)),
            vocab: budget.vocab,
            pos: 0,
            dn_conv: crate::core::avec::AVec::zeroed(n_delta * dn_conv_stride),
            dn_conv_stride,
            dn_recurrent: crate::core::avec::AVec::zeroed(n_delta * dn_rec_stride),
            dn_rec_stride,
        };
        Ok(runner)
    }

    /// Reserved bytes — layer buffers, KV, RoPE tables, logits, and the
    /// persistent states of the GatedDeltaNet mixers.
    pub fn reserved_bytes(&self) -> usize {
        self.sched.reserved_bytes()
            + 2 * (self.kv.len()
                + self
                    .ropes
                    .iter()
                    .map(|(_, c, s)| c.len() + s.len())
                    .sum::<usize>()
                + self.logits.len())
            + 4 * (self.dn_conv.len() + self.dn_recurrent.len())
    }

    /// Number of lines that the buffers of this `Runner` can carry in one pass.
    ///
    /// Exposed so the caller knows whether it can **reuse** this `Runner`
    /// for a wider pass — the bidirectional embedding needs it, which
    /// requires a single pass.
    pub fn max_batch(&self) -> usize {
        self.sched.max_batch()
    }

    /// Positions already consumed.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Starts from an empty KV **and pristine DeltaNet states**, without
    /// reallocating anything. This is the **start of a prompt**.
    ///
    /// Resetting both states is not a precaution: it is the
    /// only thing that plays for them the role that `cached_len` plays for the KV.
    /// Stale KV is never re-read — the read stops at `cached_len`.
    /// The state of a causal conv and the recurrent state, however, are **independent
    /// of the context** and **always read, in full, from the first position**.
    /// A second prompt launched without this `fill(0.0)` would continue the recurrence of
    /// the first: the text would remain plausible and be wrong, with no
    /// signal — no error, no NaN, no bound exceeded.
    ///
    /// `fill` does not allocate: the two `Vec` keep their length. And the call
    /// is at **launch** of the prompt, never during prefill nor decode.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.argmax_valid = false;
        // Logical reset invalidates every KV row; scratch buffers are overwritten
        // before use. No participant remains after a failed collective returns.
        unsafe { self.execution_session.after_model_reset(); }
        self.dn_conv.fill(0.0);
        self.dn_recurrent.fill(0.0);
    }

    // ── The state snapshot ──────────────────────────────────────────

    /// Reserves an [`Instantaneous`] for this `Runner`, sized to
    /// `max_positions` positions.
    ///
    /// **This is the only allocation point for the prefix cache**, to be called at
    /// setup (the adapter reserves its N slots once): neither capture nor
    /// restoration allocate afterwards.
    pub fn snapshot_reserve(&self, max_positions: usize) -> Instantaneous {
        let mut snap_offsets = Vec::with_capacity(self.n_layers + 1);
        let mut snap_positions = Vec::with_capacity(self.n_layers);
        let mut total = 0usize;
        snap_offsets.push(0);
        for l in 0..self.n_layers {
            let cap = if self.kv_strides[l] == 0 { 0 } else { self.kv_reserve_tokens.min(max_positions) };
            snap_positions.push(cap);
            let hd = self.layer_head_dim(l);
            total += 2 * self.shape.n_kv_heads * cap * hd;
            snap_offsets.push(total);
        }
        Instantaneous {
            kv_offsets: self.kv_offsets.clone(),
            snap_offsets,
            snap_positions,
            kv: vec![0; total],
            dn_conv: vec![0.0; self.dn_conv.len()],
            dn_recurrent: vec![0.0; self.dn_recurrent.len()],
            logits: vec![0; self.logits.len()],
            pos: 0,
            is_filled: false,
        }
    }

    /// `head_dim` reserved for layer `l` — `0` on a layer without cache.
    fn layer_head_dim(&self, l: usize) -> usize {
        if self.kv_strides[l] == 0 { 0 } else { self.kv_strides[l] / self.kv_reserve_tokens }
    }

    /// The snapshot is valid only for the geometry that reserved it: the same
    /// KV layout (thus the same reservation and the same pass) and the same
    /// DeltaNet state lengths. A `Runner` rebuilt larger has
    /// other offsets — the snapshot taken before is expired, specifically.
    fn compatible_snapshot(&self, s: &Instantaneous) -> Result<()> {
        if s.kv_offsets != self.kv_offsets
            || s.dn_conv.len() != self.dn_conv.len()
            || s.dn_recurrent.len() != self.dn_recurrent.len()
            || s.logits.len() != self.logits.len()
        {
            return Err(HerbertError::Backend(
                "backend_jit : the snapshot has been reserved for a different Runner \
                 geometry (KV cache, pass, or model differ) — reclaim it after any \
                 reconstruction"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Copies the current state into `s`: the `pos` first positions of every
    /// owning layer's KV, the DeltaNet states, the last token's logits, `pos`.
    /// No allocation.
    ///
    /// Named refusal if a layer holds more positions than the snapshot's
    /// reserve (`max_positions` too short for this state) — never any
    /// silent truncation.
    pub fn snapshot_take(&self, s: &mut Instantaneous) -> Result<()> {
        if self.execution_session.needs_reset() {
            return Err(HerbertError::Backend("cannot snapshot failed execution before reset or restoration".into()));
        }
        self.compatible_snapshot(s)?;
        // Invalidated BEFORE the first copy: a grab that fails
        // halfway (slot too short, layer N) would otherwise leave a slot
        // half old half new, still marked valid.
        s.is_filled = false;
        for l in 0..self.n_layers {
            let st = self.kv_strides[l];
            if st == 0 {
                continue;
            }
            let hd = self.layer_head_dim(l);
            let used = self.pos;
            if used > s.snap_positions[l] {
                return Err(HerbertError::Backend(format!(
                    "backend_jit : snapshot rejected — layer {l} holds {used} \
                     positions, the slot buffer holds {} (increase positions_max)",
                    s.snap_positions[l]
                )));
            }
            let snap_st = s.snap_positions[l] * hd;
            let (src_at, dst_at) = (self.kv_offsets[l], s.snap_offsets[l]);
            // Keys then values: both halves have the same layout.
            for half in 0..2 {
                let src0 = src_at + half * self.shape.n_kv_heads * st;
                let dst0 = dst_at + half * self.shape.n_kv_heads * snap_st;
                for h in 0..self.shape.n_kv_heads {
                    let n = used * hd;
                    s.kv[dst0 + h * snap_st..dst0 + h * snap_st + n]
                        .copy_from_slice(&self.kv[src0 + h * st..src0 + h * st + n]);
                }
            }
        }
        s.dn_conv.copy_from_slice(&self.dn_conv);
        s.dn_recurrent.copy_from_slice(&self.dn_recurrent);
        s.logits.copy_from_slice(&self.logits);
        s.pos = self.pos;
        s.is_filled = true;
        Ok(())
    }

    /// Restores the state of `s` **truncated to `p` positions** (`p ≤ s.pos()`), in
    /// place of the current state. No allocation. After this return, the `Runner`
    /// is exactly "a prompt of `p` tokens has just been prefilled":
    /// `continue_prefill`/`step` chain.
    ///
    /// Named refusals — never a silent partial restoration:
    /// - empty snapshot or of a different shape;
    /// - `p > s.pos()` (the state does not go that far);
    /// - `p < s.pos()` on a recurrent state model (DeltaNet does not go
    ///   backwards: the state is that of `s.pos()`, exactly).
    pub fn restore_from(&mut self, s: &Instantaneous, p: usize) -> Result<()> {
        self.compatible_snapshot(s)?;
        if !s.is_filled {
            return Err(HerbertError::Backend(
                "backend_jit : restoration refused — snapshot is empty".to_string(),
            ));
        }
        if p > s.pos {
            return Err(HerbertError::Backend(format!(
                "backend_jit : restoration refused — p={p} beyond snapshot  ({} \
                 positions)",
                s.pos
            )));
        }
        if p < s.pos && !self.dn_recurrent.is_empty() {
            return Err(HerbertError::Backend(format!(
                "backend_jit : restoration refused — the recurrent state (DeltaNet) is \
                 at position {}, it does not revert to p={p}; only an EXACT prefix is \
                 restorable on this model",
                s.pos
            )));
        }
        for l in 0..self.n_layers {
            let st = self.kv_strides[l];
            if st == 0 {
                continue;
            }
            let hd = self.layer_head_dim(l);
            // Only the first `p` positions count: beyond `p`, nothing will ever be read again.
            let used = p;
            let snap_st = s.snap_positions[l] * hd;
            let (dst_at, src_at) = (self.kv_offsets[l], s.snap_offsets[l]);
            let nkv = self.shape.n_kv_heads;
            for half in 0..2 {
                let dst0 = dst_at + half * nkv * st;
                let src0 = src_at + half * nkv * snap_st;
                for h in 0..nkv {
                    let n = used * hd;
                    let snap = &s.kv[src0 + h * snap_st..src0 + h * snap_st + n];
                    let live = &mut self.kv[dst0 + h * st..dst0 + h * st + n];
                    // Only from the first position that differs: the R16 copies
                    // always hold the bytes of the cache (the KV append writes
                    // both, a restoration rebuilds both), so a position whose
                    // bytes are the snapshot's needs neither copy nor transpose —
                    // the common case of `--rewrite-history`, which restores
                    // every turn to a prefix the cache still holds.
                    let from = first_different_row(live, snap, hd);
                    if from == used {
                        continue;
                    }
                    live[from * hd..].copy_from_slice(&snap[from * hd..]);
                    if self.kv_r16 {
                        let (cache, copies) = self.kv.split_at_mut(dst_at + 2 * nkv * st);
                        let rows = &cache[dst0 + h * st..dst0 + h * st + n];
                        let copy = &mut copies[half * nkv * st + h * st..half * nkv * st + (h + 1) * st];
                        if half == 0 {
                            transpose_keys_r16(rows, copy, hd, from, used);
                        } else {
                            transpose_values_r16(rows, copy, hd, self.kv_reserve_tokens, from, used);
                        }
                    }
                }
            }
        }
        self.dn_conv.copy_from_slice(&s.dn_conv);
        self.dn_recurrent.copy_from_slice(&s.dn_recurrent);
        self.logits.copy_from_slice(&s.logits);
        self.pos = p;
        self.argmax_valid = false;
        unsafe { self.execution_session.after_model_reset(); }
        Ok(())
    }

    /// Logits of the last pass, in bf16.
    pub fn logits(&self) -> &[u16] {
        &self.logits
    }
}

/// The first position (row of `hd` elements) where `live` and `snap` differ,
/// or their length in rows when they are equal.
fn first_different_row(live: &[u16], snap: &[u16], hd: usize) -> usize {
    live.chunks_exact(hd).zip(snap.chunks_exact(hd)).position(|(a, b)| a != b).unwrap_or(live.len() / hd)
}

/// The R16 copy of one head's keys for positions `from..to`, from its
/// head-major rows: what the KV append writes (`target/aarch64/execution/kv.rs`).
fn transpose_keys_r16(keys: &[u16], copy: &mut [u16], hd: usize, from: usize, to: usize) {
    for j in from..to {
        for d in 0..hd {
            copy[r16_key(j, d, hd)] = keys[j * hd + d];
        }
    }
}

/// The R16 copy of one head's values (`Vᵀ`) for positions `from..to`.
fn transpose_values_r16(values: &[u16], copy: &mut [u16], hd: usize, capacity: usize, from: usize, to: usize) {
    for j in from..to {
        for d in 0..hd {
            copy[r16_value(j, d, capacity)] = values[j * hd + d];
        }
    }
}

/// Where key `j`, dimension `d` lives in the R16 copy of the keys of one KV
/// head: the positions are the columns, blocks of sixteen of them,
/// `[capacity/16][head_dim/2 pairs][16][2]`.
pub(crate) const fn r16_key(j: usize, d: usize, head_dim: usize) -> usize {
    (j / 16) * 16 * head_dim + (d / 2) * 32 + (j % 16) * 2 + d % 2
}

/// Where key `j`, dimension `d` lives in the R16 copy of the values of one KV
/// head: `Vᵀ`, the dimensions as columns, `[head_dim/16][capacity/2 pairs][16][2]`.
pub(crate) const fn r16_value(j: usize, d: usize, capacity: usize) -> usize {
    (d / 16) * 16 * capacity + (j / 2) * 32 + (d % 16) * 2 + j % 2
}

/// Round to even, like the epilogue `vcvtneps2bf16` of the kernels.
fn to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7FFF).wrapping_add(lsb) >> 16) as u16
}

mod mounted;

impl Runner {
    /// Number of layers of the mounted model.
    pub fn layer_count(&self) -> usize {
        self.n_layers
    }
}
