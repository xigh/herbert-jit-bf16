//! The kernels of the GatedDeltaNet stages compiled at mount (the
//! recurrence, the causal conv, the gated norm), the `eps` of the per-head
//! L2 norm, and the two switches that put the elementwise kernels and the
//! chain around the attention on the whole team.

use super::*;
use crate::jit::conv::CompiledConv;
use crate::jit::emit_conv::ConvSpec;
use crate::jit::emit_gated_norm::GatedNormSpec;
use crate::jit::gated_norm::CompiledGatedNorm;

/// The RMS norms and residual additions of every layer on the whole team,
/// one task per row, instead of participant zero ([`crate::jit::knobs::Knobs::elem_team`]).
/// Rows are independent: the split changes no bit.
pub fn spmd_elem_team() -> bool {
    crate::jit::knobs::knobs().elem_team
}

/// The chain around the attention (head norms and RoPE, KV append by head,
/// output gate) on the whole team instead of participant zero
/// ([`crate::jit::knobs::Knobs::attn_chain_team`]). Rows and heads are
/// independent: the split changes no bit.
pub fn spmd_attn_chain_team() -> bool {
    crate::jit::knobs::knobs().attn_chain_team
}

/// The recurrence **kernel** to compile **at setup** for a `DeltaShape`.
///
/// `value_dim / num_v_heads` is the width `head_v_dim` of the layers — the
/// maxima of `DeltaShape` are those of a single-geometry model, and the arm
/// rechecks the width of **its** layer before employing the kernel.
pub(crate) fn recurrence_at_compile_time(delta: &DeltaShape) -> Result<Option<Arc<CompiledRecurrence>>> {
    let plan = crate::jit::options::debug_plan();
    if delta.num_v_heads == 0 {
        return Ok(None);
    }
    let hvd = delta.value_dim / delta.num_v_heads;
    let kernel_of = crate::jit::recurrence::get_or_compile(crate::jit::recurrence::spec_of_knobs(hvd))?;
    if plan && kernel_of.is_none() {
        eprintln!(
            "DEBUG [backend_jit] deltanet recurrence hvd={hvd} : not eligible ({}) → scalar",
            if crate::jit::recurrence::is_supported() { "width out of budget" } else { "no AVX-512" }
        );
    }
    if plan && crate::jit::recurrence::fma_active() {
        eprintln!("DEBUG [jit] recurrence deltanet hvd={hvd}: fma form (Knobs::deltanet_fma)");
    }
    Ok(kernel_of)
}

/// The conv kernel to compile **on the fly** for a `DeltaShape`
/// (`conv_kernel`, 4 on Qwen3.5).
pub(crate) fn conv_at_compile_time(delta: &DeltaShape) -> Result<Option<Arc<CompiledConv>>> {
    let plan = crate::jit::options::debug_plan();
    let k = delta.conv_kernel;
    let kernel_of = crate::jit::conv::get_or_compile(ConvSpec { k })?;
    if plan && kernel_of.is_none() {
        eprintln!(
            "DEBUG [backend_jit] conv deltanet K={k} : not eligible ({}) → scalar",
            if crate::jit::conv::is_supported() { "K ≠ 4" } else { "no AVX-512" }
        );
    }
    Ok(kernel_of)
}

/// The `eps` of the per-head L2 norm of the GatedDeltaNet mixer. This is
/// **not** the `eps` of the gated norm, which is the config's: two
/// neighbouring norms, two constants.
pub(crate) const DN_L2_EPS: f32 = 1e-6;

/// The gated norm kernel to compile **at mount time** for a `DeltaShape`
/// (`head_v_dim = value_dim / num_v_heads`, 128 on Qwen3.5). The branch
/// re-verifies the width of **its** layer before employing the kernel.
pub(crate) fn gated_norm_at_compile_time(delta: &DeltaShape) -> Result<Option<Arc<CompiledGatedNorm>>> {
    let plan = crate::jit::options::debug_plan();
    if delta.num_v_heads == 0 {
        return Ok(None);
    }
    let dim = delta.value_dim / delta.num_v_heads;
    let kernel_of = crate::jit::gated_norm::get_or_compile(GatedNormSpec { dim })?;
    if plan && kernel_of.is_none() {
        eprintln!(
            "DEBUG [backend_jit] gated norm deltanet dim={dim} : not emittable ({}) → scalar",
            if crate::jit::gated_norm::is_supported() { "width out of shape" } else { "no AVX-512" }
        );
    }
    Ok(kernel_of)
}
