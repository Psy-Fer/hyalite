//! Which compute backend a [`Database`](crate::Database) resolved to.
//!
//! Reporting the resolved backend is load-bearing, not cosmetic: a benchmark number is
//! meaningless without knowing which kernel ran, and a downstream tool (rustar) will log it so a
//! reproducibility or performance question can be answered from the log rather than by guessing
//! at the user's CPU. See `handover.md` §4 and §7.
//!
//! M0 ships only [`Backend::Scalar`]. SIMD variants (SSE4.1, AVX2, NEON) and the runtime
//! dispatch + override hook arrive in later milestones; the enum is `#[non_exhaustive]` so adding
//! them is not a breaking change.

use core::fmt;

/// The alignment backend actually used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The portable scalar reference kernel. Always available; the correctness oracle.
    Scalar,
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Backend::Scalar => f.write_str("scalar"),
        }
    }
}
