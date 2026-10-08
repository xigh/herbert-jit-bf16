//! SDPA, AVX-512: the prologue, the epilogue, the emission errors and the
//! final assembly of the passes into one blob.

#[allow(unused_imports)]
use super::*;

// ───────────────────────── assembly ─────────────────────────────────

/// Common prologue: `rbp` saved, `rsp` aligned to 64, scratch reserved.
/// The alignment makes a `call` legal and serves as a buffer for `ExpStrategy::Call`.
pub(super) fn prologue(asm: &mut CodeAssembler, frame: i32) -> std::result::Result<(), IcedError> {
    asm.push(rbp)?;
    asm.mov(rbp, rsp)?;
    asm.and(rsp, -64i32)?;
    asm.sub(rsp, frame)?;
    Ok(())
}

pub(super) fn epilogue(asm: &mut CodeAssembler) -> std::result::Result<(), IcedError> {
    asm.mov(rsp, rbp)?;
    asm.pop(rbp)?;
    asm.vzeroupper()?;
    asm.ret()?;
    Ok(())
}

/// The two ways an emission can fail.
///
/// [`EmitErr::Alloc`] is new: until now, running out of registers could not
/// *fail* — since indices were written by hand, one extra register
/// resulted in an already-taken index, thus producing incorrect code.
/// It is now a refusal, and it propagates.
#[derive(Debug)]
pub(super) enum EmitErr {
    Iced(IcedError),
    Alloc(Decline),
}

impl From<IcedError> for EmitErr {
    fn from(e: IcedError) -> Self {
        EmitErr::Iced(e)
    }
}
impl From<Decline> for EmitErr {
    fn from(d: Decline) -> Self {
        EmitErr::Alloc(d)
    }
}
impl std::fmt::Display for EmitErr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmitErr::Iced(e) => write!(f, "{e}"),
            EmitErr::Alloc(d) => write!(f, "register allocation: {d}"),
        }
    }
}

pub(super) fn assemble(
    spec: &SdpaKernelSpec,
    pass: Option<Pass>,
) -> std::result::Result<Vec<u8>, EmitErr> {
    let mut asm = CodeAssembler::new(64)?;
    let needs_constants = spec.exp == ExpStrategy::Inline;
    // The role table lives for the duration of emitting a blob. Each pass
    // takes what it needs and **returns** what dies with it: this is
    // what allows the three chained passes to fit within 32 registers
    // without any of them assuming what the other has left behind.
    let mut f = RegFile::new();

    match (spec.form, pass) {
        // ── separated blobs ─────────────────────────────────────────────
        (SdpaForm::Split, Some(Pass::Qk)) => {
            // rdi=q, rsi=k, rdx=scores, rcx=n, xmm0=scale
            prologue(&mut asm, frame_bytes(spec, pass))?;
            asm.mov(r10, rcx)?;
            let stashed = stash_scale(&mut asm, spec, &mut f)?;
            emit_qk_pass(&mut asm, spec, &mut f, stashed)?;
            epilogue(&mut asm)?;
        }
        (SdpaForm::Split, Some(Pass::Softmax)) => {
            // rdi=scores, rsi=n, xmm0=sink
            //
            let sink = f.low(SINK_ROLE)?;
            let consts = if needs_constants {
                Some(ExpConsts::alloc(&mut f)?)
            } else {
                None
            };
            let sm = SoftmaxRegs::alloc(&mut f, spec.exp)?;
            let e = Exp {
                strategy: spec.exp,
                consts,
                exempt: SoftmaxRegs::EXEMPT,
            };
            prologue(&mut asm, frame_bytes(spec, pass))?;
            asm.vmovaps(XMM[sink.x()], XMM[0])?;
            asm.mov(r11, rsi)?;
            asm.add(r11, 15i32)?;
            asm.shr(r11, 4u32)?;
            if let Some(k) = consts {
                load_exp_constants(&mut asm, &k)?;
            }
            emit_softmax(&mut asm, spec.shape.sink, &f, &sm, sink, &e)?;
            epilogue(&mut asm)?;
        }
        (SdpaForm::Split, Some(Pass::Sv)) => {
            // rdi=scores, rsi=v, rdx=out, rcx=n
            let sv = SvRegs::alloc(&mut f, spec)?;
            prologue(&mut asm, frame_bytes(spec, pass))?;
            emit_sv(&mut asm, spec, &sv)?;
            epilogue(&mut asm)?;
        }
        // `emit_sdpa` has already ruled out this case.
        (SdpaForm::Split, None) => unreachable!("split shape without pass"),
        // ── a blob per head, three chained passes ─────────────────
        (SdpaForm::Chained, _) => {
            // rdi=q, rsi=k, rdx=v, rcx=scores, r8=out, r9=n,
            // xmm0=scale, xmm1=sink
            //
            // The sink traverses **the three passes** (it is read by the
            // softmax): it is allocated before them and freed after. The scale
            // dies with the QK, and `QkRegs::free` says so.
            //
            let sink = f.low(SINK_ROLE)?;
            prologue(&mut asm, frame_bytes(spec, pass))?;
            // Relocation: the passes destroy rdi/rsi/rdx/rcx.
            asm.mov(r10, r9)?; // n
            // Scale first, then sink: `low()` may have returned
            // `zmm0` to the sink (exp inline), and its copy would clobber `xmm0`.
            let stashed = stash_scale(&mut asm, spec, &mut f)?;
            asm.vmovaps(XMM[sink.x()], XMM[1])?; // sink
            asm.mov(qword_ptr(rsp + HOME + 8), rdx)?; // v
            asm.mov(qword_ptr(rsp + HOME + 16), r8)?; // out
            asm.mov(qword_ptr(rsp + HOME + 24), rcx)?; // scores
            // QK: rdi=q already, rsi=k already, rdx=scores, rcx=n
            asm.mov(rdx, rcx)?;
            asm.mov(rcx, r10)?;
            emit_qk_pass(&mut asm, spec, &mut f, stashed)?;
            // Softmax : rdi=scores, r11=nb
            let consts = if needs_constants {
                Some(ExpConsts::alloc(&mut f)?)
            } else {
                None
            };
            let sm = SoftmaxRegs::alloc(&mut f, spec.exp)?;
            let e = Exp {
                strategy: spec.exp,
                consts,
                exempt: SoftmaxRegs::EXEMPT,
            };
            asm.mov(rdi, qword_ptr(rsp + HOME + 24))?;
            asm.mov(r11, r10)?;
            asm.add(r11, 15i32)?;
            asm.shr(r11, 4u32)?;
            if let Some(k) = consts {
                load_exp_constants(&mut asm, &k)?;
            }
            emit_softmax(&mut asm, spec.shape.sink, &f, &sm, sink, &e)?;
            sm.free(&mut f);
            f.free_low(sink);
            if let Some(k) = consts {
                for c in [k.neg87, k.log2e, k.c4, k.c3, k.c2, k.c1, k.c0, k.one] {
                    f.free(c);
                }
            }
            // SV : rdi=scores, rsi=v, rdx=out, rcx=n
            let sv = SvRegs::alloc(&mut f, spec)?;
            asm.mov(rdi, qword_ptr(rsp + HOME + 24))?;
            asm.mov(rsi, qword_ptr(rsp + HOME + 8))?;
            asm.mov(rdx, qword_ptr(rsp + HOME + 16))?;
            asm.mov(rcx, r10)?;
            emit_sv(&mut asm, spec, &sv)?;
            epilogue(&mut asm)?;
        }
        // ── one blob per head, online softmax ────────────────────────
        (SdpaForm::Fused, _) => {
            // rdi=q, rsi=k, rdx=v, rcx=scores(ignored), r8=out, r9=n,
            // xmm0=scale, xmm1=sink(not supported)
            let fr = FusedRegs::alloc(&mut f, spec)?;
            prologue(&mut asm, frame_bytes(spec, pass))?;
            asm.mov(r10, rdi)?; // q
            let v = rdx;
            asm.mov(rax, r8)?; // out
            asm.mov(r8, v)?; // v
            asm.mov(rdx, rax)?; // out
            asm.vmovss(dword_ptr(rsp + HOME), XMM[0])?;
            load_exp_constants(&mut asm, &fr.k)?;
            emit_fused(&mut asm, spec, &fr)?;
            epilogue(&mut asm)?;
        }
    }
    Ok(asm.assemble(0)?)
}
