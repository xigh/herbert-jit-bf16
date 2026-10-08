//! The AArch64 boundary audit of an emitted kernel, and the assembled no-op kernel.

use crate::jit::execution::ExecutionError;
use crate::jit::target::aarch64::util::exec_err;
use a64::{decode_all, Asm, Flow};

/// A single `ret`.
pub(crate) fn ret() -> Result<Vec<u8>, ExecutionError> {
    let mut a = Asm::new();
    a.ret();
    Ok(a.finish().map_err(exec_err)?.bytes)
}

/// Check the control-flow boundary of an already emitted kernel: every word
/// decodes to an instruction the `a64` decoder knows, and every call or
/// branch is a direct transfer to a word of the same kernel. Indirect
/// transfers and exceptions (`br`, `blr`, `svc`, `brk`, `udf`) are refused.
pub(crate) fn kernel(bytes: &[u8]) -> Result<(), ExecutionError> {
    if bytes.is_empty() {
        return Err(ExecutionError("empty generated kernel".into()));
    }
    if !bytes.len().is_multiple_of(4) {
        return Err(ExecutionError("generated kernel is not a whole number of instructions".into()));
    }
    for (offset, insn) in decode_all(bytes) {
        let Some(insn) = insn else {
            return Err(ExecutionError(format!("invalid or unknown generated instruction at +{offset:#x}")));
        };
        let target = match insn.flow {
            Flow::Next | Flow::Return => continue,
            Flow::Branch(rel) | Flow::CondBranch(rel) | Flow::Call(rel) => offset as i64 + rel,
            Flow::IndirectBranch | Flow::IndirectCall | Flow::Trap => {
                return Err(ExecutionError(
                    "kernel contains an unauditable or runtime control transfer".into(),
                ));
            }
        };
        if target < 0 || target as usize >= bytes.len() {
            return Err(ExecutionError(
                "kernel control transfer leaves its instruction boundaries".into(),
            ));
        }
    }
    Ok(())
}
