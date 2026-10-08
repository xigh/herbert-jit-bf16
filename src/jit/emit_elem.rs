//! The elementwise kernels of a layer, whatever the target: one blob per
//! `(op, dim)`.
//!
//! | kernel | site in the layer | formula |
//! |---|---|---|
//! | [`ElemOp::RmsNorm`], [`ElemOp::RmsNormGemma`] | the layer norms, the final norm | `y = x · rsqrt(Σx²/d + ε) · w` (Gemma: `· (1 + w)`) |
//! | [`ElemOp::SwiGlu`] | the MLP activation | `y = gate · σ(gate) · up` |
//! | [`ElemOp::SigmoidMul`] | the attention output gate | `y = x · σ(gate)` |
//! | [`ElemOp::ResidualAdd`] | the two residuals | `a += b` |
//! | [`ElemOp::Rope`] | the rotation of Q and K | `(x, x') ← (x·c − x'·s, x·s + x'·c)` |
//!
//! Everything is bf16 in memory and accumulates in f32 in registers; no
//! kernel converts at its input or output — the producer writes what the
//! consumer reads. `dim` is an immediate: one shape, one kernel, no remainder
//! check at run time. The emitters: `target/x86/emit_elem.rs` (AVX-512),
//! `target/x86/emit_elem_avx2.rs`, `target/aarch64/elem.rs` (NEON).

use crate::jit::ir::Decline;

/// The elementary kernels that this emitter can produce.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum ElemOp {
    /// `y[i] = x[i] · rsqrt(Σx²/d + ε) · w[i]`, all in bf16.
    ///
    /// ABI : `fn(x: *const u16, w: *const u16, y: *mut u16, eps: f32)`.
    RmsNorm,
    /// `y[i] = x[i] · rsqrt(Σx²/d + ε) · (1 + w[i])` — the *Gemma* convention,
    /// which **Qwen3.5** applies to all its norms (`NormType::GemmaRMSNorm`).
    ///
    /// Same ABI as [`ElemOp::RmsNorm`], and **the same emitter**: the only difference
    /// is a `vaddps` of a broadcast constant on widened `w`, after the
    /// `vpslld` that brings it to f32. One instruction per half-block.
    ///
    /// **Do not assume we can get away with folding `w' = w + 1` at
    /// load time.** The identity holds over the reals, but is false in the format:
    /// `w'` would be stored in bf16, whose quantum near 1.0 is
    /// `2⁻⁸ ≈ 3,9e-3`, whereas these weights are small deviations around 0.
    /// A `w = 1,2e-2` would fold to within 3,9e-3 — **~30 % relative error**.
    /// The scalar reference adds in **f32**, after widening; this kernel does
    /// the same, in the same place.
    RmsNormGemma,
    /// `y[i] = gate[i] · σ(gate[i]) · up[i]`, with `σ(t) = 1/(1+e^{-t})`.
    ///
    /// ABI : `fn(gate: *const u16, up: *const u16, y: *mut u16)`.
    SwiGlu,
    /// `y[i] = geluTanh(gate[i]) · up[i]`, the Gemma4 activation.
    ///
    /// `geluTanh(g) = ½·g·(1 + tanh(z))`, `z = √(2/π)·(g + 0,044715·g³)`.
    ///
    /// **Emitted as a SwiGLU with cubic argument**, via the identity
    /// `½(1 + tanh z) = σ(2z)` :
    ///
    /// ```text
    /// geluTanh(g) = g · σ(t),   t = 2z = K·g + K·C·g³ = g·(K + K·C·g²)
    /// K = 2√(2/π) = 1,5957691   C = 0,044715
    /// ```
    ///
    /// All the exponential machinery is therefore shared with
    /// [`ElemOp::SwiGlu`]: four more instructions, not a second kernel.
    /// And the argument to `exp` remains bounded by [`EXP_MAX88`] as there.
    ///
    /// ABI : `fn(gate: *const u16, up: *const u16, y: *mut u16)`.
    GeluTanh,
    /// `y[i] = a[i] · σ(b[i])` — the **attention output gate** of Qwen3.5.
    ///
    /// ABI: `fn(a: *const u16, b: *const u16, y: *mut u16)`, that of
    /// [`SwiGluFn`]. `a` is the SDPA output, `b` the gate.
    ///
    /// **Emitted by the [`ElemOp::SwiGlu`] arm**, which is a variant with
    /// two differences, and not a second emitter:
    ///
    /// 1. the sigmoid argument comes from the **second** operand (`b`, `rsi`)
    ///    instead of the first;
    /// 2. the final `vmulps` **disappears** — the `vdivps` of SwiGLU already divides
    ///    by `1 + exp(−·)`, it suffices to give it `a` in the numerator:
    ///    `a / (1 + exp(−b)) = a · σ(b)`.
    ///
    /// No extra register, no extra constant. In particular the
    /// bound [`EXP_MAX88`] is **reused**: as with SwiGLU the argument to
    /// `exp` is `−b`, positive as soon as `b` is negative, and without the bound a
    /// `b < −88` would yield an **arbitrary** number instead of ~0.
    SigmoidMul,
    /// `a[i] += b[i]`, en place.
    ///
    /// ABI : `fn(a: *mut u16, b: *const u16)`.
    ResidualAdd,
    /// `y[i] = x[i] · s`, all in bf16, `s` scalar f32.
    ///
    /// ABI: `fn(x: *const u16, y: *mut u16, s: f32)` — that of [`RmsNormFn`]
    /// **without the weights**, the scalar taking the place of the `eps`.
    ///
    /// The three scales of the Gemma4 input branch
    /// (`G-branch-entree-A.md` §1): the end-of-layer `layer_scalar`, the
    /// `√256` from the `embed_tokens_per_layer` lookup, and the `× 1/√2` from the
    /// upstream combine. This is the only product by **broadcast constant** in the catalog —
    /// none of the other six kernels does this.
    ///
    /// Body of [`ElemOp::ResidualAdd`] with one difference: the second operand
    /// is not read from memory but broadcast once from `xmm0`, and the
    /// output goes into `rsi` instead of overwriting the input.
    Scale,
    /// RoPE on one head, in place, `rotate_half` convention (Llama/Qwen):
    /// for `i < dim/2`, the pair `(x[i], x[i+dim/2])` rotates by the angle of the
    /// pair `(cos[i], sin[i])`.
    ///
    /// ABI : `fn(x: *mut u16, cos: *const u16, sin: *const u16)`, `dim` being
    /// the `head_dim` and the tables having `dim/2` inputs.
    ///
    /// The `cos`/`sin` tables are in **bf16**, like everything in
    /// memory. They are model constants just like the **weights** — but their
    /// precision directly affects the rotation angle, unlike a **weight**.
    /// The error is **measured** instead of assumed negligible.
    Rope,
    /// `y[i] = cap · tanh(x[i] / cap)` — the softcapping of logits of the Gemma
    /// models; not used by Qwen3.5, kept as an emitted operation.
    ///
    /// ABI: that of [`ElemOp::Scale`], `fn(x: *const u16, y: *mut u16,
    /// cap: f32)` — a broadcast scalar, one input, one output.
    ///
    /// **In greedy mode, this gap is not visible**: `tanh` is monotone
    /// increasing, so the argmax is invariant. It is however false as soon as we
    /// sample (temperature, top-p): without the cap, the gap between two
    /// logits is not compressed and the distribution after softmax is not the
    /// same. A kernel that no non-regression greedy can claim.
    ///
    /// **Emitted with the exponential of [`ElemOp::SwiGlu`]**, not a second
    /// transcendental: via `tanh(u) = (1 − e^{−2u}) / (1 + e^{−2u})` applied to
    /// `|u|`, with the sign restored at the end (`vpord`). Taking `|u|` is not
    /// cosmetic — the argument of `exp` remains **negative or zero**, thus within the
    /// domain that [`EXP_MAX88`] guards, and the saturation `|x| ≫ cap` outputs
    /// exactly `±cap` instead of an `∞/∞`.
    Softcap,
}

impl ElemOp {
    /// Short name, for logs and cache.
    pub const fn name(self) -> &'static str {
        match self {
            ElemOp::RmsNorm => "rmsnorm",
            ElemOp::RmsNormGemma => "rmsnorm_gemma",
            ElemOp::SwiGlu => "swiglu",
            ElemOp::GeluTanh => "gelutanh",
            ElemOp::SigmoidMul => "sigmoid_mul",
            ElemOp::ResidualAdd => "residual",
            ElemOp::Scale => "scale",
            ElemOp::Rope => "rope",
            ElemOp::Softcap => "softcap",
        }
    }
}

/// Shape of a kernel element: the operation and the dimension, both
/// fixed at load time.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ElemSpec {
    /// The operation.
    pub op: ElemOp,
    /// Vector length, **immediate**.
    pub dim: usize,
}

/// `fn(x, w, y, eps)` — [`ElemOp::RmsNorm`].
// The calling convention of generated code (`sysv64` on x86-64 on every
// operating system, AAPCS64 on AArch64): see `target::generated_fn!`.
pub type RmsNormFn = crate::jit::target::generated_fn!(*const u16, *const u16, *mut u16, f32);
/// `fn(gate, up, y)` — [`ElemOp::SwiGlu`] and [`ElemOp::GeluTanh`], same ABI.
pub type SwiGluFn = crate::jit::target::generated_fn!(*const u16, *const u16, *mut u16);
/// `fn(a, b)` — [`ElemOp::ResidualAdd`].
pub type ResidualAddFn = crate::jit::target::generated_fn!(*mut u16, *const u16);
/// `fn(x, y, s)` — [`ElemOp::Scale`].
pub type ScaleFn = crate::jit::target::generated_fn!(*const u16, *mut u16, f32);
/// `fn(x, cos, sin)` — [`ElemOp::Rope`].
pub type RopeFn = crate::jit::target::generated_fn!(*mut u16, *const u16, *const u16);

impl ElemSpec {
    /// Emissible shapes.
    ///
    /// `dim` multiple of 32: this is what a zmm holds in bf16, and what makes
    /// the loops remainder-free. The two dimensions of Qwen3-0.6B (1024, 3072)
    /// and those of Qwen3.5 (1024, 3584) are.
    pub fn check(&self) -> Result<(), Decline> {
        if self.dim == 0 {
            return Err(Decline::Empty);
        }
        if !self.dim.is_multiple_of(32) {
            return Err(Decline::PartialRowBlock);
        }
        Ok(())
    }

    /// 32 bf16 vectors covered by `dim`.
    pub const fn blocks32(&self) -> usize {
        self.dim / 32
    }
}

/// The two emission arms, as seen by the loop policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoopArm {
    /// AVX-512: 32 bf16 per block.
    Zmm,
    /// AVX2: 16 bf16 per block.
    Ymm,
}

/// Width from which the MLP activations and the output gate are emitted as
/// a loop: on Qwen3.5-4B the `SwiGlu` 9 216 is called once per layer in
/// decode and its unrolled body (94 KiB) arrives cold in L1i every time;
/// looping it takes 1,0 % off the decode step and the generated program
/// shrinks from 7,75 to 4,76 MB. `Knobs::elem_loop_min` overrides it.
pub const LOOP_MIN_DIM: usize = 4_096;

/// The threshold applied ([`crate::jit::knobs::Knobs::elem_loop_min`]),
/// constant for the life of the kernel cache.
pub fn loop_min_dim() -> usize {
    crate::jit::knobs::knobs().elem_loop_min
}

/// Loop or unroll: the policy, as a table.
///
/// Every elementary kernel is emitted fully unrolled: one copy of the block
/// body per 32 (or 16) elements. On the MLP width of a 27B model that is a
/// 183 KiB blob in AVX-512 and 443 KiB in AVX2, several times the L1i. The
/// looped form keeps `U` block bodies and runs `dim / (U · lanes)` turns over
/// moving cursors; the body itself is byte-for-byte the same, so the output
/// bits are the same.
///
/// The table below is a measurement, not a rule. Ten widths × eight
/// operations × `U ∈ {4, 8, 16, 32}` × five machines × three repetitions,
/// synthetic data, zero output divergence. Gain against the unrolled
/// witness **at its fastest repetition** (the unrolled blob moves by 14 to
/// 18 % between runs depending on where its pages land; the loop by less
/// than 1 %):
///
/// | form | Zen 4 | Zen 5 | Lunar Lake | Zen 3 (a) | Zen 3 (b) |
/// |---|---:|---:|---:|---:|---:|
/// | `swiglu` 17 408 | +2.8 | +2.4 | +2.8 | +1.0 | −0.1 |
/// | `swiglu` 12 288 | −0.3 | +0.3 | +1.9 | −4.2 | +1.9 |
/// | `swiglu` 6 144 | 0.0 | +0.3 | −0.1 | −3.1 | −0.8 |
///
/// The norms, the residual add, the scale and RoPE only gain above
/// 8 192 elements, which no served `hidden` or `head_dim` reaches, and RoPE
/// loses up to 31 % at small widths on Zen 3: they stay unrolled. The
/// "code bytes per tick" predictor was tested and does not hold: it is a
/// constant per (machine, op) and cannot select a width.
///
/// **The cache key does not change.** `ElemSpec { op, dim }` is the whole
/// key and this function is pure in `(op, dim, arm)`, the arm being fixed
/// for the process: two callers asking the same `(op, dim)` always get the
/// same blob. Do not add a field for this.
///
/// Returns the number of blocks per loop turn, or `None` to unroll.
pub fn loop_unroll(op: ElemOp, dim: usize, arm: LoopArm) -> Option<usize> {
    if !matches!(op, ElemOp::SwiGlu | ElemOp::GeluTanh | ElemOp::SigmoidMul) {
        return None;
    }
    if dim < loop_min_dim() {
        return None;
    }
    let (lanes, unroll) = match arm {
        LoopArm::Zmm => (32, 16),
        LoopArm::Ymm => (16, 8),
    };
    let blocks = dim / lanes;
    blocks.is_multiple_of(unroll).then_some(unroll)
}


/// Generates the kernel of `spec` for the target of this build (the arm it
/// selects: [`crate::jit::elem::select_isa`]).
pub fn emit_elem(spec: &ElemSpec) -> crate::core::error::Result<Vec<u8>> {
    crate::jit::target::emit_elem(spec)
}
