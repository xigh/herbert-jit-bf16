//! Coefficients transcribed from Arm optimized-routines (MIT option).
//! Copyright (c) 2017-2025, Arm Limited. See `third_party/arm-optimized-routines/LICENSE`
//! at the root of the repository; original files and revision in `SOURCE.json` beside it.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct Tables {
    pub exp: [u64; 32],
    pub log: [[u64; 2]; 16],
}
pub(crate) const TABLES: Tables = Tables {
    exp: [
        0x3ff0000000000000,
        0x3fefd9b0d3158574,
        0x3fefb5586cf9890f,
        0x3fef9301d0125b51,
        0x3fef72b83c7d517b,
        0x3fef54873168b9aa,
        0x3fef387a6e756238,
        0x3fef1e9df51fdee1,
        0x3fef06fe0a31b715,
        0x3feef1a7373aa9cb,
        0x3feedea64c123422,
        0x3feece086061892d,
        0x3feebfdad5362a27,
        0x3feeb42b569d4f82,
        0x3feeab07dd485429,
        0x3feea47eb03a5585,
        0x3feea09e667f3bcd,
        0x3fee9f75e8ec5f74,
        0x3feea11473eb0187,
        0x3feea589994cce13,
        0x3feeace5422aa0db,
        0x3feeb737b0cdc5e5,
        0x3feec49182a3f090,
        0x3feed503b23e255d,
        0x3feee89f995ad3ad,
        0x3feeff76f2fb5e47,
        0x3fef199bdd85529c,
        0x3fef3720dcef9069,
        0x3fef5818dcfba487,
        0x3fef7c97337b9b5f,
        0x3fefa4afa2a490da,
        0x3fefd0765b6e4540,
    ],
    log: [
        [0x3ff661ec79f8f3be, 0xbfd57bf7808caade],
        [0x3ff571ed4aaf883d, 0xbfd2bef0a7c06ddb],
        [0x3ff49539f0f010b0, 0xbfd01eae7f513a67],
        [0x3ff3c995b0b80385, 0xbfcb31d8a68224e9],
        [0x3ff30d190c8864a5, 0xbfc6574f0ac07758],
        [0x3ff25e227b0b8ea0, 0xbfc1aa2bc79c8100],
        [0x3ff1bb4a4a1a343f, 0xbfba4e76ce8c0e5e],
        [0x3ff12358f08ae5ba, 0xbfb1973c5a611ccc],
        [0x3ff0953f419900a7, 0xbfa252f438e10c1e],
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3fee608cfd9a47ac, 0x3faaa5aa5df25984],
        [0x3feca4b31f026aa0, 0x3fbc5e53aa362eb4],
        [0x3feb2036576afce6, 0x3fc526e57720db08],
        [0x3fe9c2d163a1aa2d, 0x3fcbc2860d224770],
        [0x3fe886e6037841ed, 0x3fd1058bc8a07ee1],
        [0x3fe767dcf5534862, 0x3fd4043057b6ee09],
    ],
};
pub(crate) const INV: u64 = 0x40471547652b82fe;
pub(crate) const SHIFT: u64 = 0x4338000000000000;
pub(crate) const E0: u64 = 0x3ebc6af84b912394;
pub(crate) const E1: u64 = 0x3f2ebfce50fac4f3;
pub(crate) const E2: u64 = 0x3f962e42ff0c52d6;
pub(crate) const LN2: u64 = 0x3fe62e42fefa39ef;
pub(crate) const L0: u64 = 0xbfd00ea348b88334;
pub(crate) const L1: u64 = 0x3fd5575b0be00b6a;
pub(crate) const L2: u64 = 0xbfdffffef20a4123;

