//! Materialising a 64-bit constant: the `lui`/`addi(w)`/`slli` sequence of
//! `li`, the recursive split the LLVM backend uses without its extension
//! shortcuts.

/// One step of [`li_sequence`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiStep {
    /// `lui rd, hi20`.
    Lui(u32),
    /// `addi rd, rd, lo12`, or `addi rd, zero, lo12` when the flag is set.
    Addi(i64, bool),
    /// `addiw rd, rd, lo12`.
    Addiw(i64),
    /// `slli rd, rd, sh`.
    Slli(u32),
}

fn sext12(v: i64) -> i64 {
    (v << 52) >> 52
}

/// The instructions that leave `value` in a register, first to last.
pub fn li_sequence(value: i64) -> Vec<LiStep> {
    let mut out = Vec::new();
    generate(value, &mut out);
    out
}

fn generate(value: i64, out: &mut Vec<LiStep>) {
    if i64::from(value as i32) == value {
        let lo = sext12(value);
        let hi = ((value - lo) >> 12) as u32 & 0xf_ffff;
        if hi != 0 {
            out.push(LiStep::Lui(hi));
            if lo != 0 {
                out.push(LiStep::Addiw(lo));
            }
        } else {
            out.push(LiStep::Addi(lo, true));
        }
        return;
    }
    let lo = sext12(value);
    let rest = value.wrapping_sub(lo);
    let shift = 12 + ((rest as u64) >> 12).trailing_zeros();
    generate(rest >> shift, out);
    out.push(LiStep::Slli(shift));
    if lo != 0 {
        out.push(LiStep::Addi(lo, false));
    }
}
