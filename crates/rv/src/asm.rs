//! The assembler: a buffer of 32-bit instruction words, labels, and the
//! fixups that bind branches, calls and address computations to them.
//!
//! Every instruction method appends one word (or a few, for the synthesised
//! `li`, `call` and `la`). An operand an encoding cannot carry — an
//! out-of-range immediate, a branch beyond its reach, a mask on an
//! instruction that has none — never produces a wrong word: the first such
//! error is kept, the all-zero word (defined illegal on RISC-V) holds its
//! place so later offsets stay right, and [`Asm::finish`] returns it.
//! Emission code therefore reads like assembly, without a `?` per line, and
//! still cannot mis-encode silently.
//!
//! The compressed extension is never emitted: every instruction is four
//! bytes, and a label is a word index.

use std::fmt;

/// A position in the code, created unbound and bound once.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Label(pub(crate) u32);

/// An emission error: what was refused, and at which byte offset.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Byte offset of the refused instruction.
    pub at: usize,
    /// What the encoding could not carry.
    pub message: String,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "rv at +{:#x}: {}", self.at, self.message)
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Fix {
    /// Conditional branches: a 13-bit signed byte offset, ±4 KiB.
    B12,
    /// `jal`: a 21-bit signed byte offset, ±1 MiB.
    J20,
    /// `auipc` then `jalr` (or `addi`, `ld`…) with the low twelve bits: ±2 GiB.
    Pair,
}

/// The assembled code and where its labels landed.
#[derive(Clone, Debug)]
pub struct Code {
    /// The machine code, little-endian words.
    pub bytes: Vec<u8>,
    labels: Vec<Option<u32>>,
}

impl Code {
    /// Byte offset of a bound label.
    pub fn label_offset(&self, label: Label) -> Option<usize> {
        self.labels
            .get(label.0 as usize)
            .copied()
            .flatten()
            .map(|word| word as usize * 4)
    }
}

/// The assembler.
#[derive(Clone, Debug, Default)]
pub struct Asm {
    pub(crate) code: Vec<u32>,
    labels: Vec<Option<u32>>,
    fixups: Vec<(usize, Label, Fix)>,
    error: Option<Error>,
    /// Set by [`Asm::masked`]: the next vector instruction takes `v0.t`.
    mask: bool,
}

impl Asm {
    /// An empty buffer.
    pub fn new() -> Self {
        Self::default()
    }

    /// A new, unbound label.
    pub fn label(&mut self) -> Label {
        self.labels.push(None);
        Label(self.labels.len() as u32 - 1)
    }

    /// Bind `label` to the current position. Several labels may share one.
    pub fn bind(&mut self, label: Label) {
        let here = self.code.len() as u32;
        match self.labels.get_mut(label.0 as usize) {
            Some(slot @ None) => *slot = Some(here),
            Some(Some(_)) => self.fail(format!("label {} bound twice", label.0)),
            None => self.fail(format!("label {} belongs to another assembler", label.0)),
        }
    }

    /// A new label bound here.
    pub fn here(&mut self) -> Label {
        let l = self.label();
        self.bind(l);
        l
    }

    /// Current byte offset.
    pub fn offset(&self) -> usize {
        self.code.len() * 4
    }

    /// The first error recorded so far.
    pub fn error(&self) -> Option<&Error> {
        self.error.as_ref()
    }

    /// Append one raw instruction word.
    pub fn word(&mut self, w: u32) {
        self.code.push(w);
    }

    /// Append already assembled code (a whole number of words).
    pub fn bytes(&mut self, bytes: &[u8]) {
        if !bytes.len().is_multiple_of(4) {
            self.fail(format!("{} bytes are not a whole number of instructions", bytes.len()));
            return;
        }
        for c in bytes.as_chunks::<4>().0 {
            self.code.push(u32::from_le_bytes(*c));
        }
    }

    /// Pad with `nop` up to a multiple of `bytes` (a power of two, ≥ 4).
    pub fn align(&mut self, bytes: usize) {
        if !bytes.is_power_of_two() || bytes < 4 {
            self.fail(format!("alignment {bytes} is not a power of two ≥ 4"));
            return;
        }
        while !self.offset().is_multiple_of(bytes) {
            self.code.push(crate::NOP);
        }
    }

    /// Emit, through `body`, exactly one vector instruction under the mask
    /// `v0.t`: `a.masked(|a| a.vle(E16, V4, A1))` is `vle16.v v4, (a1), v0.t`.
    /// Anything else — no instruction, several, a scalar one, a vector one
    /// without a mask field — is an error.
    pub fn masked(&mut self, body: impl FnOnce(&mut Asm)) {
        let start = self.code.len();
        self.mask = true;
        body(self);
        if self.mask {
            self.mask = false;
            self.fail("masked: no maskable vector instruction was emitted".into());
        } else if self.code.len() != start + 1 {
            self.fail(format!("masked: {} instructions emitted, one expected", self.code.len() - start));
        }
    }

    /// Resolve every branch and return the code, or the first error.
    pub fn finish(mut self) -> Result<Code, Error> {
        if self.mask {
            self.fail("a mask was requested and never consumed".into());
        }
        for &(at, label, fix) in &self.fixups {
            let Some(target) = self.labels.get(label.0 as usize).copied().flatten() else {
                let e = Error { at: at * 4, message: format!("label {} is never bound", label.0) };
                return Err(self.error.take().unwrap_or(e));
            };
            let delta = (i64::from(target) - at as i64) * 4;
            let reach = match fix {
                Fix::B12 => 1i64 << 12,
                Fix::J20 => 1i64 << 20,
                Fix::Pair => 1i64 << 31,
            };
            let reach_ok = match fix {
                Fix::Pair => (-reach - 0x800..reach - 0x800).contains(&delta),
                _ => (-reach..reach).contains(&delta),
            };
            if !reach_ok {
                let e = Error { at: at * 4, message: format!("branch to label {} out of range ({delta} bytes)", label.0) };
                return Err(self.error.take().unwrap_or(e));
            }
            match fix {
                Fix::B12 => self.code[at] |= b_field(delta as i32),
                Fix::J20 => self.code[at] |= j_field(delta as i32),
                Fix::Pair => {
                    let (hi, lo) = split_pcrel(delta);
                    self.code[at] |= (hi as u32) << 12;
                    self.code[at + 1] |= ((lo as u32) & 0xfff) << 20;
                }
            }
        }
        if let Some(e) = self.error {
            return Err(e);
        }
        Ok(Code {
            bytes: self.code.iter().flat_map(|w| w.to_le_bytes()).collect(),
            labels: self.labels,
        })
    }

    /// Record `message` as an error of the instruction about to be emitted.
    pub(crate) fn fail(&mut self, message: String) {
        if self.error.is_none() {
            self.error = Some(Error { at: self.offset(), message });
        }
    }

    /// Append a word, or the illegal all-zero word and the error.
    pub(crate) fn emit(&mut self, word: Result<u32, String>) {
        match word {
            Ok(w) => self.code.push(w),
            Err(m) => {
                self.fail(m);
                self.code.push(0);
            }
        }
    }

    /// Append a word whose label field is filled at [`Asm::finish`]; for
    /// [`Fix::Pair`], `words` is the `auipc` and its low-part partner.
    pub(crate) fn emit_fix(&mut self, words: Result<Vec<u32>, String>, label: Label, fix: Fix) {
        if label.0 as usize >= self.labels.len() {
            self.fail(format!("label {} belongs to another assembler", label.0));
        }
        match words {
            Ok(ws) => {
                self.fixups.push((self.code.len(), label, fix));
                self.code.extend(ws);
            }
            Err(m) => {
                let n = if matches!(fix, Fix::Pair) { 2 } else { 1 };
                self.fail(m);
                self.code.extend(std::iter::repeat_n(0, n));
            }
        }
    }

    /// The `vm` bit of the vector instruction being emitted: 0 under
    /// [`Asm::masked`], which it consumes, else 1.
    pub(crate) fn take_vm(&mut self) -> u32 {
        if std::mem::take(&mut self.mask) {
            0
        } else {
            1
        }
    }

    /// A vector instruction that has no mask field: refused under [`Asm::masked`].
    pub(crate) fn no_mask(&mut self, what: &str) -> Result<(), String> {
        if std::mem::take(&mut self.mask) {
            Err(format!("{what} takes no mask"))
        } else {
            Ok(())
        }
    }
}

/// `Err` unless `cond`.
pub(crate) fn check(cond: bool, message: impl FnOnce() -> String) -> Result<(), String> {
    if cond {
        Ok(())
    } else {
        Err(message())
    }
}

/// The offset field of a B-type word (bits 31, 30..25, 11..8, 7).
pub(crate) fn b_field(off: i32) -> u32 {
    let o = off as u32;
    (o >> 12 & 1) << 31 | (o >> 5 & 0x3f) << 25 | (o >> 1 & 0xf) << 8 | (o >> 11 & 1) << 7
}

/// The offset field of a J-type word (bits 31, 30..21, 20, 19..12).
pub(crate) fn j_field(off: i32) -> u32 {
    let o = off as u32;
    (o >> 20 & 1) << 31 | (o >> 1 & 0x3ff) << 21 | (o >> 11 & 1) << 20 | (o >> 12 & 0xff) << 12
}

/// A pc-relative offset as the `auipc` upper twenty bits and the signed low
/// twelve its partner adds.
pub(crate) fn split_pcrel(delta: i64) -> (i64, i64) {
    let hi = (delta + 0x800) >> 12;
    let lo = delta - (hi << 12);
    (hi & 0xf_ffff, lo)
}
