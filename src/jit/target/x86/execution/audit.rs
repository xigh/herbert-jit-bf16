//! The x86-64 boundary audit of an emitted kernel, and the assembled no-op kernel.

use crate::jit::execution::ExecutionError;
use iced_x86::{Decoder, DecoderOptions, FlowControl, OpKind};

/// A single `ret`.
pub(crate) fn ret() -> Result<Vec<u8>, ExecutionError> {
    let err = |e: iced_x86::IcedError| ExecutionError(e.to_string());
    let mut a = iced_x86::code_asm::CodeAssembler::new(64).map_err(err)?;
    a.ret().map_err(err)?;
    a.assemble(0).map_err(err)
}

/// Check the control-flow boundary of an already emitted kernel: every instruction
/// decodes, and every call or branch is a near transfer to an instruction boundary of
/// the same kernel.
pub(crate) fn kernel(bytes: &[u8]) -> Result<(), ExecutionError> {
    if bytes.is_empty() {
        return Err(ExecutionError("empty generated kernel".into()));
    }
    let mut decoder = Decoder::new(64, bytes, DecoderOptions::NONE);
    let mut boundaries = Vec::new();
    let mut targets = Vec::new();
    while decoder.can_decode() {
        let i = decoder.decode();
        if i.is_invalid() {
            return Err(ExecutionError("invalid generated instruction".into()));
        }
        boundaries.push(i.ip());
        match i.flow_control() {
            FlowControl::Call
            | FlowControl::ConditionalBranch
            | FlowControl::UnconditionalBranch => {
                // SYSCALL/SYSENTER also classify as Call in iced. They have
                // no near-branch operand and are runtime transfers, never
                // generated-kernel calls.
                if i.op0_kind() != OpKind::NearBranch64 {
                    return Err(ExecutionError(
                        "kernel contains a non-local control transfer".into(),
                    ));
                }
                targets.push(i.near_branch_target());
            }
            FlowControl::IndirectCall
            | FlowControl::IndirectBranch
            | FlowControl::Interrupt
            | FlowControl::Exception
            | FlowControl::XbeginXabortXend => {
                return Err(ExecutionError(
                    "kernel contains an unauditable or runtime control transfer".into(),
                ));
            }
            _ => (),
        }
    }
    if targets
        .iter()
        .any(|target| boundaries.binary_search(target).is_err())
    {
        return Err(ExecutionError(
            "kernel control transfer leaves its instruction boundaries".into(),
        ));
    }
    Ok(())
}
