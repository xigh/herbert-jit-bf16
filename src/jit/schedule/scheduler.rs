//! `impl Scheduler`: the setup. Everything that compiles a kernel happens
//! here, and it happens once, at mount; a pass only calls compiled kernels.

use super::*;

impl Scheduler {
    /// Compile all kernels and reserve all buffers. **Nothing will be
    /// allocated afterwards** — this is the contract of this constructor.
    ///
    /// `geometries` = the distinct pairs `(head_dim, rotary_ndims)` in any
    /// order. A homogeneous model passes only one; Gemma4 passes two
    /// (256/256 on sliding layers, 512/512 on full layers). The `head_dim`
    /// must all be ≤ `shape.head_dim`, which sizes the buffers.
    ///
    /// It produces **one set of kernels per `head_dim`** ([`HeadKernels`]) and
    /// **one RoPE kernel per `rotary_ndims`**: these are two axes, not one.
    pub fn with_geometries(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
    ) -> Result<Self> {
        Self::ascend(shape, budget, widths, geometries, None)
    }

    /// Compile a hybrid model with variable attention geometry and
    /// GatedDeltaNet mixer buffers.
    pub fn with_geometries_delta(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        delta: DeltaShape,
    ) -> Result<Self> {
        Self::ascend(shape, budget, widths, geometries, Some(delta))
    }

    /// The common body of the four constructors.
    fn ascend(
        shape: ModelShape,
        budget: Budget,
        widths: &[(Activation, usize)],
        geometries: &[(usize, usize)],
        delta: Option<DeltaShape>,
    ) -> Result<Self> {
        if geometries.is_empty() {
            return Err(HerbertError::Backend(
                "backend_jit scheduler : no attention geometry declared  (head_dim, \
                 rotary_ndims pairs)"
                    .to_string(),
            ));
        }
        for &(hd, rn) in geometries {
            // A larger `rotary_ndims` than the head would cause the RoPE kernel
            // to overflow **onto the next head**, without error or NaN. Stating
            // this here, once, rather than assuming it on every table read.
            if rn == 0 || rn > hd {
                return Err(HerbertError::Backend(format!(
                    "backend_jit scheduler: rotary_ndims={rn} out of [1, head_dim={hd}]"
                )));
            }
            // The `q`/`k`/`v`/`attn` buffers and the KV stride are sized based
            // on `shape.head_dim`: a wider layer would write out of bounds.
            // Same control as on `intermediate`, same message.
            if hd > shape.head_dim {
                return Err(HerbertError::Backend(format!(
                    "backend_jit scheduler: head_dim={hd} exceeds the reserved maximum \
                     ({}).",
                    shape.head_dim
                )));
            }
        }
        if let Some(&(_, too_many)) = widths.iter().find(|&&(_, d)| d > shape.intermediate) {
            return Err(HerbertError::Backend(format!(
                "backend_jit scheduler : intermediate={too_many} exceeds the reserved \
                 maximum ({})",
                shape.intermediate
            )));
        }
        let need = |op: ElemOp, dim: usize| -> Result<Arc<CompiledElem>> {
            elem_kernel(ElemSpec { op, dim })?.ok_or_else(|| {
                HerbertError::Backend(format!(
                    "backend_jit scheduler: {} dim={dim} not schedulable",
                    op.name()
                ))
            })
        };
        // A single norm convention choice for **all** norms in the model:
        // this is what `norm_plus_one` guarantees, and this is what prevents
        // processing only a part of them.
        let norm_op = if shape.norm_plus_one {
            ElemOp::RmsNormGemma
        } else {
            ElemOp::RmsNorm
        };
        // ── one set of kernels per distinct `head_dim` ────────────────────────
        //
        // Compiled **at mount time**, never during a pass: lazy compilation
        // would allocate in the middle of prefill.
        let mut heads: Vec<HeadKernels> = Vec::with_capacity(geometries.len());
        for &(hd, rn) in geometries {
            if heads.iter().any(|k| k.head_dim == hd) {
                continue;
            }
            // The shape **of this geometry** : `q_dim` and `q_row_stride`
            // derive from it, and they are the ones that parameterize the
            // block kernel.
            let sl = shape.for_layer(hd, rn, shape.n_kv_heads);
            let sdpa = head_kernel(SdpaShape {
                head_dim: hd,
                sink: false,
            })?
            .ok_or_else(|| {
                HerbertError::Backend(format!(
                    "backend_jit scheduler: sdpa head_dim={hd} not schedulable"
                ))
            })?;
            // The prefill kernel only makes sense beyond a single row. At `m = 1`
            // it would do exactly what the decode kernel does, plus extra code.
            // `sdpa_rows() == 0` disables it: it's the witness of the
            // measurement.
            let sdpa_prefill = if budget.max_batch > 1 && sdpa_rows() > 0 {
                prefill_kernel(
                    SdpaShape {
                        head_dim: hd,
                        sink: false,
                    },
                    sdpa_rows(),
                    // Two distinct steps: `q` is striped twice as wide as
                    // `attn` on a gated model (the gate after `Q` on each row).
                    sl.q_row_stride(),
                    sl.q_dim(),
                    budget.kv_reserve_tokens,
                )?
            } else {
                None
            };
            heads.push(HeadKernels {
                head_dim: hd,
                norm_head: need(norm_op, hd)?,
                // Compiled only if the model requires it.
                gate_row: if shape.output_gate {
                    Some(need(ElemOp::SigmoidMul, sl.q_dim())?)
                } else {
                    None
                },
                sdpa,
                sdpa_prefill,
            });
        }
        // ── a RoPE kernel per distinct `rotary_ndims` ───────────────────────
        //
        // `rotary_ndims`, not `head_dim`: the kernel only writes its first
        // `spec.dim` u16 values, so **the partial RoPE is this very kernel**,
        // compiled shorter. The NoPE tail is not "preserved", it is never
        // reached. The two axes often overlap but do not coincide: Qwen3.5
        // rotates 64 out of 256, Gemma4 rotates 512 out of 512.
        let mut ropes: Vec<(usize, Arc<CompiledElem>)> = Vec::with_capacity(geometries.len());
        for &(_, rn) in geometries {
            if ropes.iter().any(|(r, _)| *r == rn) {
                continue;
            }
            ropes.push((rn, need(ElemOp::Rope, rn)?));
        }
        // The SDPA kernel pads scores to the next multiple of 16. During prefill,
        // the blocks (head, chunk) run in parallel: one slot per worker,
        // otherwise two blocks would write to the same buffer. One slot holds the
        // `rows` lines of the block — the **maximum** across the datasets, since
        // a `head_dim` may have a block kernel and another may not.
        //
        // A slot is the scratch of the block kernel (`CompiledSdpaPrefill::
        // scratch_len`: its rows of scores on x86; on AArch64 the scores, `P`
        // and the packed queries of its two GEMMs), or one row of scores for
        // the decode kernel.
        let scores_stride = heads
            .iter()
            .map(|k| {
                k.sdpa_prefill
                    .as_ref()
                    .map_or(budget.kv_reserve_tokens.next_multiple_of(16).max(16), |p| p.scratch_len())
            })
            .max()
            .unwrap_or(16);
        let slots = if budget.max_batch > 1 {
            global_pool().num_workers().max(1)
        } else {
            1
        };
        let mut s = Self {
            heads,
            ropes,
            norm_hidden: need(norm_op, shape.hidden)?,
            swiglu: {
                let mut v = Vec::with_capacity(widths.len());
                for &(act, dim) in widths {
                    let op = match act {
                        Activation::SwiGlu => ElemOp::SwiGlu,
                        Activation::GeluTanh => ElemOp::GeluTanh,
                    };
                    v.push((act, dim, need(op, dim)?));
                }
                v
            },
            residual: need(ElemOp::ResidualAdd, shape.hidden)?,
            recurrence: None,
            conv: None,
            gated_norm: None,
            buffers: Buffers::new(shape, budget, scores_stride, slots),
            max_batch: budget.max_batch.max(1),
            scores_stride,
            shape,
        };
        if let Some(delta) = delta {
            let m = budget.max_batch.max(1);
            // The only allocation of the mixer, and it happens **here**: the
            // contract of these constructors is that nothing is allocated
            // afterwards.
            s.buffers.dn_qkv = crate::core::avec::AVec::zeroed(m * delta.conv_dim);
            s.buffers.dn_z = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            s.buffers.dn_ab = crate::core::avec::AVec::zeroed(m * 2 * delta.num_v_heads);
            s.buffers.dn_out = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            // The f32 → bf16 boundary before `Wo`, the only u16 buffer of the
            // mixer.
            s.buffers.dn_out_bf16 = crate::core::avec::AVec::zeroed(m * delta.value_dim);
            // The two gates, `max_batch · num_v_heads` each — the
            // discrepancy assumed in directory 36 §6, which required them to be
            // recalculated per token.
            s.buffers.dn_beta = crate::core::avec::AVec::zeroed(m * delta.num_v_heads);
            s.buffers.dn_decay = crate::core::avec::AVec::zeroed(m * delta.num_v_heads);
            // The emitted recurrence kernel, compiled **here** like all the
            // others — `None` = the scalar body (no AVX-512, or a width out of
            // budget).
            s.recurrence = recurrence_at_compile_time(&delta)?;
            // E3: the conv + SiLU kernel emitted, compiled **here** too.
            s.conv = conv_at_compile_time(&delta)?;
            // E6: the gated norm kernel emitted, same.
            s.gated_norm = gated_norm_at_compile_time(&delta)?;
        }
        // The packing buffer of the matrix engine, when the target's plan for
        // a block of `max_batch` rows asks for it: sized for the widest input
        // of a projection.
        let m = budget.max_batch.max(1);
        let widest_k = [shape.hidden, shape.q_dim(), shape.intermediate, delta.map_or(0, |d| d.value_dim)]
            .into_iter()
            .max()
            .unwrap_or(0);
        let probe = GemmShape { k: shape.hidden, n: shape.hidden };
        let plan = crate::jit::target::gemm_plan(probe, m);
        if plan.opts.matrix {
            s.buffers.packed = crate::core::avec::AVec::zeroed(crate::jit::emit::packed_bytes(plan.opts, m, widest_k) / 2);
        }
        Ok(s)
    }

    /// ShapeOf served.
    pub fn shape(&self) -> ModelShape {
        self.shape
    }

    /// Number of tokens the buffers can carry in a single prefill pass.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Buffers — the caller writes the residual input stream to them and
    /// reads the output from them.
    pub fn buffers(&mut self) -> &mut Buffers {
        &mut self.buffers
    }

    /// Bytes reserved by buffers.
    pub fn reserved_bytes(&self) -> usize {
        self.buffers.bytes()
    }

}
