//! The vector register file of the emitters: registers handed out by role,
//! once each until freed, and split into the two domains the encodings need.
//!
//! [`RegFile`] hands an index out only once until it is explicitly freed.
//! [`Vlow`] additionally restricts an allocation to `zmm0-15`, the registers
//! reachable by the VEX forms the emitters select at the sites that need a
//! `ymm` or `xmm` view.
//!
//! ## A refusal from `iced` is not a CPU constraint
//!
//! `iced` refuses `vmovups [rdi], ymm31` in its default overload, and it is
//! tempting to conclude that the ISA cannot encode it. It can: `VMOVUPS` has
//! both a VEX form (`ymm0-15`) and an EVEX form (`ymm0-31`, the `R'` bit of
//! EVEX extends the register field), and `asm.evex().vmovups(…)` emits it.
//! What stops at `ymm15` is the default encoding selection of the overload,
//! not the silicon. Before writing here that a register is forbidden
//! somewhere: check the manual, check that its revision covers AVX-512, and
//! look for the API escape hatch (`evex()`, a neighbouring mnemonic).
//!
//! The low zone is kept by choice, not by constraint: the VEX encoding takes 4
//! bytes where EVEX takes 6, and none of these stores needs a mask. [`Vlow`]
//! means "reachable by the forms the emitter chooses", not "allowed by the
//! hardware". The two mnemonics that only have an EVEX form (`vmovdqu16`,
//! `vcvtneps2bf16`) go up to `ymm31` without asking — hence [`V::y_evex`].
//!
//! Two traps, both of the same shape — a true
//! constraint written in a comment that a second use ignored: a softmax
//! maximum kept in a register the exponential clobbers, and a scale kept in
//! a temporary the fused form overwrote, invisible at `n = 1`. Explicit
//! allocation is the answer to both. And `ymm16..ymm31` do not exist outside
//! AVX-512: using them makes `iced` encode EVEX silently, and the binary
//! faults on the target machine.

use crate::jit::ir::Decline;

/// Number of AVX-512 vector registers.
pub const N_REGS: usize = 32;
/// First register of the high zone. Registers below it are reachable by the
/// VEX forms selected by the emitter.
pub const HIGH: usize = 16;

/// An allocated vector register, of any zone.
///
/// Usable in any **EVEX** instruction — that is, anything the generator emits
/// in 512 bits (`vaddps`, `vmulps`, `vfmadd231ps`, `vdpbf16ps`,
/// `vcvtneps2bf16`, `vbroadcastss`…). What it does **not** offer, and that is
/// the point: no `ymm`/`xmm` view, because the instructions that consume them
/// in this generator are VEX forms.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct V(u8);

/// A register from the **low zone** (`zmm0-15`).
///
/// Unlike [`V`], it is reachable by the **VEX** forms chosen by the emitter —
/// horizontal reductions (`vextractf128`, `vhaddps`, `vshufps`) and
/// `vmovups ymm`. This is an encoding choice, not an ISA limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Vlow(u8);

impl V {
    /// Raw index, used to index the emitter's `ZMM` table.
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    /// `ymm` view **only in an EVEX instruction**.
    ///
    /// `vcvtneps2bf16`, `vinserti64x4`, `vextracti64x4` and `vpmovzxwd` have
    /// **only** EVEX forms: `iced` encodes them on `ymm16-31` without us
    /// having to ask for anything. `vmovups`, on the other hand, has both forms, and `iced`
    /// chooses the VEX — hence [`Vlow::y`] at its call sites. The name of this
    /// method is there so that the distinction can be read in place rather than
    /// rediscovered in the debugger.
    pub const fn y_evex(self) -> usize {
        self.0 as usize
    }
}

impl Vlow {
    /// Raw index, to index `ZMM`.
    pub const fn idx(self) -> usize {
        self.0 as usize
    }
    /// Vue `ymm` — exists only in the low region, because the `vmovups ymm`
    /// emitted by default is the VEX form, which stops at `ymm15`. For an
    /// EVEX instruction, use [`V::y_evex`], which goes up to `ymm31`.
    pub const fn y(self) -> usize {
        self.0 as usize
    }
    /// View `xmm` — same reason: `vhaddps`, `vextractf128`, `vmovss` are VEX.
    pub const fn x(self) -> usize {
        self.0 as usize
    }
}

impl From<Vlow> for V {
    fn from(r: Vlow) -> V {
        V(r.0)
    }
}

/// Occupation table of the 32 vector registers, by role.
///
/// A `RegFile` lives for the duration of an emission. It performs **no**
/// allocation during the execution of the generated code: it is a generation
/// object, not an inference object.
#[derive(Clone, Debug)]
pub struct RegFile {
    /// Role occupied by each index, `None` if free.
    role: [Option<&'static str>; N_REGS],
}

impl Default for RegFile {
    fn default() -> Self {
        Self::new()
    }
}

impl RegFile {
    /// An empty table: the 32 free registers, no role assigned.
    pub fn new() -> Self {
        Self {
            role: [None; N_REGS],
        }
    }

    fn take_at(&mut self, i: usize, role: &'static str) -> V {
        debug_assert!(self.role[i].is_none());
        self.role[i] = Some(role);
        V(i as u8)
    }

    /// Reserves a register from the low zone (`zmm0-15`) for `role`.
    ///
    /// Ask when the role needs a `ymm`/`xmm` view — therefore a VEX
    /// encoding. Otherwise, prefer [`Self::high`]: the high region is the
    /// only one that survives an external call.
    pub fn low(&mut self, role: &'static str) -> Result<Vlow, Decline> {
        for i in 0..HIGH {
            if self.role[i].is_none() {
                return Ok(Vlow(self.take_at(i, role).0));
            }
        }
        Err(Decline::OutOfRegisters)
    }

    /// Reserves a register from the high zone (`zmm16-31`) for `role`.
    pub fn high(&mut self, role: &'static str) -> Result<V, Decline> {
        for i in (HIGH..N_REGS).rev() {
            if self.role[i].is_none() {
                return Ok(self.take_at(i, role));
            }
        }
        Err(Decline::OutOfRegisters)
    }

    /// Reserve a register wherever it is — high zone first, because it is
    /// the scarce resource for anything that must survive a call, but also
    /// the one that no one else can use for VEX.
    pub fn any(&mut self, role: &'static str) -> Result<V, Decline> {
        for i in (0..N_REGS).rev() {
            if self.role[i].is_none() {
                return Ok(self.take_at(i, role));
            }
        }
        Err(Decline::OutOfRegisters)
    }

    /// Reserves `n` **consecutive** registers for `role`.
    ///
    /// The accumulators (SV output, dot product chains) are indexed
    /// by a `|c| base + c` in the emitted code: they must be contiguous.
    /// This is the only reason for this variant — do not use it for
    /// independent roles, which have no need to be neighbors.
    pub fn run(&mut self, n: usize, role: &'static str) -> Result<Vec<V>, Decline> {
        if n == 0 {
            return Ok(Vec::new());
        }
        for base in 0..=N_REGS.saturating_sub(n) {
            if (base..base + n).all(|i| self.role[i].is_none()) {
                return Ok((base..base + n).map(|i| self.take_at(i, role)).collect());
            }
        }
        Err(Decline::OutOfRegisters)
    }

    /// Reserve `n` consecutive registers **in low area**.
    pub fn low_run(&mut self, n: usize, role: &'static str) -> Result<Vec<Vlow>, Decline> {
        if n == 0 {
            return Ok(Vec::new());
        }
        for base in 0..=HIGH.saturating_sub(n) {
            if (base..base + n).all(|i| self.role[i].is_none()) {
                return Ok((base..base + n)
                    .map(|i| Vlow(self.take_at(i, role).0))
                    .collect());
            }
        }
        Err(Decline::OutOfRegisters)
    }

    /// Frees a register: its role is dead.
    pub fn free(&mut self, r: V) {
        self.role[r.idx()] = None;
    }

    /// Frees a register from the low area.
    pub fn free_low(&mut self, r: Vlow) {
        self.role[r.idx()] = None;
    }

}
