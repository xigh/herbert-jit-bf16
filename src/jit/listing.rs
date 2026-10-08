//! The disassembly of a mounted regime: what the generated program looks
//! like once emitted, node by node, for reading.
//!
//! What enters: one mounted program (`Runner::execution_program`) — its plan
//! (nodes, labels) and its executable image (entries, kernels at known
//! offsets). What comes out: text files, one per selected node, in the
//! target's syntax (NASM for x86-64, the system assembler's for AArch64,
//! llvm-mc's for RISC-V), plus the whole-call entry and a table of the nodes.
//! Nothing here runs: it decodes the bytes the emitters wrote.

use crate::core::error::{HerbertError, Result};
use crate::jit::execution::{CompiledExecution, Operation};
#[cfg(target_arch = "x86_64")]
use iced_x86::{Decoder, DecoderOptions, Formatter, NasmFormatter};
use std::fmt::Write as _;
use std::path::Path;

/// `bytes` decoded as AArch64 from address `base`, one instruction per line:
/// `offset  word  mnemonic operands`, branch targets as absolute offsets.
#[cfg(target_arch = "aarch64")]
pub fn disassemble(bytes: &[u8], base: u64) -> String {
    let mut text = String::new();
    for (offset, insn) in a64::decode_all(bytes) {
        let address = base + offset as u64;
        let word = bytes.get(offset..offset + 4).map_or(0, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let line = match insn {
            Some(i) => match i.flow {
                a64::Flow::Branch(rel) | a64::Flow::CondBranch(rel) | a64::Flow::Call(rel) => {
                    let target = address as i64 + rel;
                    format!("{}   // → {target:06x}", i.text)
                }
                _ => i.text,
            },
            None => format!(".inst {word:#010x}"),
        };
        let _ = writeln!(text, "{address:06x}  {word:08x}  {line}");
    }
    text
}

/// `bytes` decoded as RISC-V from address `base`, one instruction per line:
/// `offset  word  mnemonic operands`, branch targets as absolute offsets.
#[cfg(target_arch = "riscv64")]
pub fn disassemble(bytes: &[u8], base: u64) -> String {
    let mut text = String::new();
    for (offset, insn) in rv::decode_all(bytes) {
        let address = base + offset as u64;
        let word = bytes.get(offset..offset + 4).map_or(0, |c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
        let line = match insn {
            Some(i) => match i.flow {
                rv::Flow::Branch(rel) | rv::Flow::CondBranch(rel) | rv::Flow::Call(rel) => {
                    let target = address as i64 + rel;
                    format!("{}   # → {target:06x}", i.text)
                }
                _ => i.text,
            },
            None => format!(".word {word:#010x}"),
        };
        let _ = writeln!(text, "{address:06x}  {word:08x}  {line}");
    }
    text
}

/// `bytes` decoded as x86-64 from address `base`, one instruction per line:
/// `offset  bytes  mnemonic operands`.
#[cfg(target_arch = "x86_64")]
pub fn disassemble(bytes: &[u8], base: u64) -> String {
    let mut decoder = Decoder::with_ip(64, bytes, base, DecoderOptions::NONE);
    let mut formatter = NasmFormatter::new();
    formatter.options_mut().set_first_operand_char_index(10);
    formatter.options_mut().set_hex_prefix("0x");
    formatter.options_mut().set_hex_suffix("");
    formatter.options_mut().set_uppercase_hex(false);
    let mut text = String::new();
    let mut line = String::new();
    while decoder.can_decode() {
        let instruction = decoder.decode();
        line.clear();
        formatter.format(&instruction, &mut line);
        let start = (instruction.ip() - base) as usize;
        let end = start + instruction.len();
        let hex: Vec<String> = bytes[start..end].iter().map(|b| format!("{b:02x}")).collect();
        let _ = writeln!(text, "{:06x}  {:<24} {line}", instruction.ip() - base, hex.join(" "));
    }
    text
}

/// Writes the listings of `program` into `dir`: `plan.txt` (every node: index,
/// operation, layer, kernel, tasks, participants, size), `entry.asm` (the
/// whole-call entry, the loop over blocks and nodes with its barriers) and
/// one `node-NNN-<operation>.asm` per node whose layer is in `layers` (or
/// without a layer: the embedding and the head). Two nodes that share a
/// kernel share a file.
pub fn write_listings(program: &CompiledExecution, dir: &Path, layers: &[usize]) -> Result<usize> {
    std::fs::create_dir_all(dir)?;
    let plan = program.plan();
    let code = program.code();
    let offsets = program.kernel_offsets();
    let kernel_span = |k: usize| -> (usize, usize) {
        let start = offsets[k];
        let end = offsets.get(k + 1).copied().unwrap_or(code.len());
        (start, end)
    };
    let mut table = String::new();
    let _ = writeln!(table, "{:>4}  {:<28} {:>5}  {:>6}  {:>7}  {:>7}  {:>8}", "node", "operation", "layer", "kernel", "tasks", "workers", "bytes");
    let mut written = 0usize;
    let mut done: Vec<usize> = Vec::new();
    for (index, (node, label)) in plan.nodes().iter().zip(plan.labels()).enumerate() {
        let layer = label.layer.map_or("-".to_string(), |l| l.to_string());
        match node.operation {
            Operation::Kernel { kernel, max_tasks, worker_limit, .. } => {
                let (start, end) = kernel_span(kernel);
                let _ = writeln!(table, "{index:>4}  {:<28} {layer:>5}  {kernel:>6}  {max_tasks:>7}  {worker_limit:>7}  {:>8}", label.operation, end - start);
                let wanted = match label.layer {
                    Some(l) => layers.contains(&l),
                    None => true,
                };
                if wanted && !done.contains(&kernel) {
                    let name = format!("node-{index:03}-{}.asm", label.operation.replace(' ', "-"));
                    let mut text = format!(
                        "; node {index}: {} (layer {layer}), kernel {kernel} at image offset 0x{start:x}, {} bytes, {max_tasks} tasks, up to {worker_limit} participants\n\n",
                        label.operation,
                        end - start
                    );
                    text.push_str(&disassemble(&code[start..end], 0));
                    std::fs::write(dir.join(name), text)?;
                    done.push(kernel);
                    written += 1;
                }
            }
            Operation::LayerEnd { layer: l } => {
                let _ = writeln!(table, "{index:>4}  {:<28} {l:>5}  {:>6}  {:>7}  {:>7}  {:>8}", "layer-end (progress)", "-", "-", "-", "-");
            }
        }
    }
    std::fs::write(dir.join("plan.txt"), table)?;
    let entry = program.entry_code(false);
    let [start, _] = program.entry_offsets();
    let mut text = format!("; the whole-call entry without the progress bridge, {} bytes at image offset 0x{start:x}\n\n", entry.len());
    text.push_str(&disassemble(entry, 0));
    std::fs::write(dir.join("entry.asm"), text).map_err(HerbertError::Io)?;
    Ok(written + 2)
}
