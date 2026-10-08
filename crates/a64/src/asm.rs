//! The assembler: a buffer of instruction words, labels, and the fixups that
//! bind branches to them.
//!
//! Every instruction method appends one word (or a few, for the synthesised
//! `mov_imm`). An operand an encoding cannot carry — an out-of-range
//! immediate, the stack pointer where register 31 reads as zero, a branch
//! beyond its reach — never produces a wrong word: the first such error is
//! kept, an `udf #0` holds its place so later offsets stay right, and
//! [`Asm::finish`] returns it. Emission code therefore reads like assembly,
//! without a `?` per line, and still cannot mis-encode silently.

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
        write!(f, "a64 at +{:#x}: {}", self.at, self.message)
    }
}
impl std::error::Error for Error {}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Fix {
    /// `b`, `bl`: imm26 at bits 25..0.
    Imm26,
    /// `b.cond`, `cbz`, `cbnz`, `ldr` literal: imm19 at bits 23..5.
    Imm19,
    /// `tbz`, `tbnz`: imm14 at bits 18..5.
    Imm14,
    /// `adr`: imm21 split in immhi (23..5) and immlo (30..29).
    Adr21,
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
            self.code.push(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
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

    /// Resolve every branch and return the code, or the first error.
    pub fn finish(mut self) -> Result<Code, Error> {
        for &(at, label, fix) in &self.fixups {
            let Some(target) = self.labels.get(label.0 as usize).copied().flatten() else {
                let e = Error { at: at * 4, message: format!("label {} is never bound", label.0) };
                return Err(self.error.take().unwrap_or(e));
            };
            let delta = i64::from(target) - at as i64;
            let (bits, word_delta) = match fix {
                Fix::Imm26 => (26, true),
                Fix::Imm19 => (19, true),
                Fix::Imm14 => (14, true),
                Fix::Adr21 => (21, false),
            };
            let value = if word_delta { delta } else { delta * 4 };
            let limit = 1i64 << (bits - 1);
            if value < -limit || value >= limit {
                let e = Error { at: at * 4, message: format!("branch to label {} out of range ({} bytes)", label.0, delta * 4) };
                return Err(self.error.take().unwrap_or(e));
            }
            let field = (value as u32) & ((1u32 << bits) - 1);
            let w = &mut self.code[at];
            *w |= match fix {
                Fix::Imm26 => field,
                Fix::Imm19 => field << 5,
                Fix::Imm14 => field << 5,
                Fix::Adr21 => ((field & 3) << 29) | ((field >> 2) << 5),
            };
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

    /// Append a word, or `udf #0` and the error.
    pub(crate) fn emit(&mut self, word: Result<u32, String>) {
        match word {
            Ok(w) => self.code.push(w),
            Err(m) => {
                self.fail(m);
                self.code.push(0);
            }
        }
    }

    /// Append a word whose label field is filled at [`Asm::finish`].
    pub(crate) fn emit_fix(&mut self, word: Result<u32, String>, label: Label, fix: Fix) {
        if label.0 as usize >= self.labels.len() {
            self.fail(format!("label {} belongs to another assembler", label.0));
        }
        if word.is_ok() {
            self.fixups.push((self.code.len(), label, fix));
        }
        self.emit(word);
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
