//! The bf16 element type and its two conversions: widening is exact, narrowing
//! rounds to nearest even like the `vcvtneps2bf16` epilogue of the kernels.

pub type BF16 = u16;

/// BF16 → f32: zero-extend the upper 16 bits.
#[inline(always)]
pub fn bf16_to_f32(bf16: BF16) -> f32 {
    f32::from_bits((bf16 as u32) << 16)
}

/// Convert f32 to BF16 with round-to-nearest-even (IEEE 754 RNE).
#[inline(always)]
pub fn f32_to_bf16(f32_val: f32) -> BF16 {
    let bits = f32_val.to_bits();
    let lsb = (bits >> 16) & 1;
    let round_bit = (bits >> 15) & 1;
    let sticky = bits & 0x7FFF;
    let mut result = bits >> 16;
    if round_bit == 1 && (sticky != 0 || lsb == 1) {
        result += 1;
    }
    result as u16
}
