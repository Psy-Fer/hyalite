//! Typed construction errors.
//!
//! Every invariant the hot kernel relies on is validated once, at construction, and reported
//! as a typed [`Error`] here — so the kernel can assume the invariant and stay branch-free and
//! infallible (see the static width proof in [`crate::width`]).

use core::fmt;

/// Result alias for fallible `hyalite` construction paths.
pub type Result<T> = core::result::Result<T, Error>;

/// A construction-time error. These arise only while building a scoring scheme or database —
/// never in the alignment hot path, which is infallible by design.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The alphabet length was zero. There is nothing to score against.
    EmptyAlphabet,

    /// The substitution matrix length does not equal `alphabet_len * alphabet_len`.
    MatrixShape {
        /// The declared alphabet length.
        alphabet_len: usize,
        /// `alphabet_len * alphabet_len`, the number of entries expected.
        expected: usize,
        /// The number of entries actually supplied.
        got: usize,
    },

    /// A gap penalty was negative. Penalties are non-negative magnitudes that are subtracted;
    /// a negative penalty would reward gaps.
    NegativeGapPenalty {
        /// The gap-open penalty as supplied.
        gap_open: i32,
        /// The gap-extend penalty as supplied.
        gap_ext: i32,
    },

    /// `gap_open < gap_ext`. Opal issue #28: the kernel misbehaves in this regime, so it is
    /// rejected up front and the kernel may assume `gap_open >= gap_ext`.
    GapOpenLessThanExtend {
        /// The gap-open penalty.
        gap_open: i32,
        /// The gap-extend penalty.
        gap_ext: i32,
    },

    /// The provably-reachable score range exceeds what the widest supported integer (`i32`)
    /// can hold without overflow. Reduce sequence lengths or penalty magnitudes.
    ScoreRangeTooWide {
        /// The conservative bound on `|score|` computed by the width proof.
        bound: i64,
    },

    /// A sequence contained an encoded symbol `>= alphabet_len`. Inputs must be pre-encoded
    /// alphabet indices in `0..alphabet_len`.
    SymbolOutOfRange {
        /// The offending symbol value.
        symbol: usize,
        /// The alphabet length it must be below.
        alphabet_len: usize,
    },

    /// A required builder field was not set before `build()`.
    IncompleteBuilder {
        /// The name of the missing field.
        field: &'static str,
    },

    /// A database was built with no sequences to search against.
    EmptyDatabase,

    /// A backend was forced (via the builder or `HYALITE_BACKEND`) that is not available on this
    /// build/CPU. In M0 only the scalar backend is available.
    BackendUnavailable {
        /// The backend that was requested but is unavailable.
        backend: crate::backend::Backend,
    },

    /// A backend name (from the builder or `HYALITE_BACKEND`) could not be parsed.
    InvalidBackendName {
        /// The unrecognised name.
        name: String,
    },

    /// A traceback ([`align`](crate::align)) needed more working memory than the caller's
    /// `max_bytes` budget allowed. The full-matrix path needs `3 * (m+1) * (n+1) * 4` bytes; raise
    /// the budget, or await the linear-space path that serves larger pairs within a bounded
    /// footprint.
    TracebackBudgetExceeded {
        /// Bytes the full-matrix traceback would require.
        needed_bytes: u64,
        /// The caller's `max_bytes` budget.
        budget_bytes: usize,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::EmptyAlphabet => write!(f, "alphabet length must be greater than zero"),
            Error::MatrixShape {
                alphabet_len,
                expected,
                got,
            } => write!(
                f,
                "substitution matrix must have {expected} entries \
                 (alphabet_len {alphabet_len} squared), but got {got}"
            ),
            Error::NegativeGapPenalty { gap_open, gap_ext } => write!(
                f,
                "gap penalties must be non-negative magnitudes, \
                 but got gap_open={gap_open}, gap_ext={gap_ext}"
            ),
            Error::GapOpenLessThanExtend { gap_open, gap_ext } => write!(
                f,
                "gap_open ({gap_open}) must be >= gap_ext ({gap_ext}) (Opal issue #28)"
            ),
            Error::ScoreRangeTooWide { bound } => write!(
                f,
                "reachable score magnitude bound ({bound}) exceeds the i32 range; \
                 reduce sequence lengths or penalty magnitudes"
            ),
            Error::SymbolOutOfRange {
                symbol,
                alphabet_len,
            } => write!(
                f,
                "encoded symbol {symbol} is out of range for alphabet_len {alphabet_len} \
                 (must be in 0..{alphabet_len})"
            ),
            Error::IncompleteBuilder { field } => {
                write!(f, "database builder is missing required field: {field}")
            }
            Error::EmptyDatabase => {
                write!(f, "database must contain at least one sequence")
            }
            Error::BackendUnavailable { backend } => {
                write!(f, "backend {backend} is not available on this build/CPU")
            }
            Error::InvalidBackendName { name } => write!(
                f,
                "unrecognised backend name {name:?}; expected one of: \
                 auto, scalar, sse4.1, avx2, neon"
            ),
            Error::TracebackBudgetExceeded {
                needed_bytes,
                budget_bytes,
            } => write!(
                f,
                "traceback needs {needed_bytes} bytes but the budget is {budget_bytes}; \
                 raise max_bytes (the linear-space path is not yet available)"
            ),
        }
    }
}

impl core::error::Error for Error {}
