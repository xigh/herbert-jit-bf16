//! The two compressed immediate forms of A64: the bitmask immediate of the
//! logical instructions and the 8-bit floating-point immediate of `fmov`.

/// The `(N, immr, imms)` fields encoding `value` as a bitmask immediate of a
/// `bits`-wide register, or `None` when it has no such encoding (0, all ones,
/// or not a rotated run of ones replicated over a power-of-two element).
pub fn encode_logical(value: u64, bits: u32) -> Option<(u32, u32, u32)> {
    debug_assert!(bits == 32 || bits == 64);
    let full = if bits == 64 { u64::MAX } else { (1u64 << bits) - 1 };
    if value == 0 || value & full == full || value & !full != 0 {
        return None;
    }
    // The smallest element whose repetition makes the register.
    let mut size = bits;
    loop {
        size /= 2;
        let mask = (1u64 << size) - 1;
        if (value & mask) != ((value >> size) & mask) {
            size *= 2;
            break;
        }
        if size <= 2 {
            break;
        }
    }
    let mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
    let mut imm = value & mask;
    let (rotation, ones);
    if is_shifted_mask(imm) {
        rotation = imm.trailing_zeros();
        ones = (imm >> rotation).trailing_ones();
    } else {
        imm |= !mask;
        if !is_shifted_mask(!imm) {
            return None;
        }
        let leading = imm.leading_ones();
        rotation = 64 - leading;
        ones = leading + imm.trailing_ones() - (64 - size);
    }
    let immr = (size - rotation) & (size - 1);
    let nimms = ((!(size as u64 - 1)) << 1) | u64::from(ones - 1);
    let n = (((nimms >> 6) & 1) ^ 1) as u32;
    Some((n, immr, (nimms & 0x3f) as u32))
}

fn is_shifted_mask(v: u64) -> bool {
    v != 0 && {
        let filled = (v - 1) | v;
        filled.wrapping_add(1) & filled == 0
    }
}

/// The value of a bitmask immediate, `None` for a reserved encoding.
pub fn decode_logical(n: u32, immr: u32, imms: u32, bits: u32) -> Option<u64> {
    let combined = (n << 6) | (!imms & 0x3f);
    if combined == 0 {
        return None;
    }
    let len = 31 - combined.leading_zeros();
    if len < 1 || (bits == 32 && n == 1) {
        return None;
    }
    let size = 1u32 << len;
    let r = immr & (size - 1);
    let s = imms & (size - 1);
    if s == size - 1 {
        return None;
    }
    let pattern = (1u64 << (s + 1)) - 1;
    let elem_mask = if size == 64 { u64::MAX } else { (1u64 << size) - 1 };
    let rotated = if r == 0 {
        pattern
    } else {
        ((pattern >> r) | (pattern << (size - r))) & elem_mask
    };
    let mut value = 0u64;
    let mut at = 0;
    while at < bits {
        value |= rotated << at;
        at += size;
    }
    Some(if bits == 64 { value } else { value & 0xffff_ffff })
}

/// The 8-bit immediate of `fmov` for `value`, or `None` when it has no exact
/// encoding (±(16..31)/16 · 2^(−3..4)).
pub fn encode_fp8(value: f64) -> Option<u32> {
    let bits = value.to_bits();
    if bits & ((1u64 << 48) - 1) != 0 {
        return None;
    }
    let a = bits >> 63;
    let not_b = (bits >> 62) & 1;
    let b = not_b ^ 1;
    let replicated = (bits >> 54) & 0xff;
    if replicated != if b == 1 { 0xff } else { 0 } {
        return None;
    }
    let cdefgh = (bits >> 48) & 0x3f;
    Some(((a << 7) | (b << 6) | cdefgh) as u32)
}

/// The value of an 8-bit `fmov` immediate.
pub fn decode_fp8(imm8: u32) -> f64 {
    let a = u64::from((imm8 >> 7) & 1);
    let b = u64::from((imm8 >> 6) & 1);
    let cdefgh = u64::from(imm8 & 0x3f);
    f64::from_bits((a << 63) | ((b ^ 1) << 62) | (if b == 1 { 0xff } else { 0 } << 54) | (cdefgh << 48))
}
