//! Scoring scheme: substitution matrix + affine gap penalties, validated at construction.
//!
//! # Gap penalty convention
//!
//! `hyalite` follows Opal: a gap of length `n` costs
//!
//! ```text
//! gap_open + (n - 1) * gap_ext
//! ```
//!
//! i.e. the first gap base is charged `gap_open` and each subsequent base `gap_ext`. This
//! differs from the ksw2/parasail `gap_open + n * gap_ext` convention by one `gap_ext` per gap.
//! Following Opal lets test vectors be lifted directly from Opal and STAR's `ClipMate`.
//!
//! Penalties are supplied as **non-negative magnitudes** that are subtracted during alignment.
//!
//! # Gap direction
//!
//! A gap runs in one of two directions, and [`Scoring::new_asymmetric`] can charge them
//! differently:
//!
//! | Direction | DP matrix | Consumes | Meaning when the query is a read and the target a reference |
//! |---|---|---|---|
//! | **query-gap** | `E` | target only | a deletion from the read |
//! | **target-gap** | `F` | query only | an insertion into the read |
//!
//! [`Scoring::new`] charges both directions the same, which is what every scheme lifted from
//! Opal assumes.

use crate::error::{Error, Result};
use crate::mode::Mode;
use crate::width::{self, ScoreWidth};

/// The four affine-gap penalty magnitudes, split by [gap direction](self#gap-direction).
///
/// Threaded through every kernel in place of a single `(gap_open, gap_ext)` pair so that the
/// scalar, striped, and inter-sequence DPs all read the same four values, and a symmetric
/// scheme is exactly the case where both halves agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gaps {
    /// Open/extend for a gap in the query (`E`; consumes target only).
    pub(crate) query_open: i32,
    pub(crate) query_ext: i32,
    /// Open/extend for a gap in the target (`F`; consumes query only).
    pub(crate) target_open: i32,
    pub(crate) target_ext: i32,
}

impl Gaps {
    /// Both directions charged alike.
    pub(crate) fn symmetric(open: i32, ext: i32) -> Self {
        Gaps {
            query_open: open,
            query_ext: ext,
            target_open: open,
            target_ext: ext,
        }
    }

    /// `(open, ext)` for a gap in the query (`E`).
    #[inline]
    pub(crate) fn query(self) -> (i32, i32) {
        (self.query_open, self.query_ext)
    }

    /// `(open, ext)` for a gap in the target (`F`).
    #[inline]
    pub(crate) fn target(self) -> (i32, i32) {
        (self.target_open, self.target_ext)
    }

    /// Whether both directions are charged alike.
    #[inline]
    pub(crate) fn is_symmetric(self) -> bool {
        self.query() == self.target()
    }

    /// The larger open and the larger extend across both directions. The width proof bounds the
    /// reachable score magnitude with these, which is exact for a symmetric scheme and a safe
    /// over-estimate otherwise (a path's gaps are charged at most this much per base).
    #[inline]
    pub(crate) fn worst(self) -> (i32, i32) {
        (
            self.query_open.max(self.target_open),
            self.query_ext.max(self.target_ext),
        )
    }
}

/// A validated substitution-matrix-plus-affine-gap scoring scheme.
///
/// Construct with [`Scoring::new`] (one penalty pair for both gap directions) or
/// [`Scoring::new_asymmetric`] (a pair per direction). Either enforces every invariant the
/// kernel relies on: non-empty alphabet, correctly shaped matrix, non-negative penalties, and
/// `gap_open >= gap_ext` per direction (Opal issue #28). Once built, a `Scoring` is guaranteed
/// valid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Scoring {
    alphabet_len: usize,
    matrix: Vec<i32>,
    gaps: Gaps,
    min_entry: i32,
    max_entry: i32,
}

impl Scoring {
    /// Build a scoring scheme from a row-major `alphabet_len × alphabet_len` substitution
    /// matrix and affine gap penalties.
    ///
    /// `matrix[q * alphabet_len + t]` is the score for aligning query symbol `q` with target
    /// symbol `t` (both are pre-encoded alphabet indices in `0..alphabet_len`). `gap_open` and
    /// `gap_ext` are non-negative penalty magnitudes; a gap of length `n` costs
    /// `gap_open + (n - 1) * gap_ext` (Opal's convention).
    ///
    /// # Errors
    ///
    /// - [`Error::EmptyAlphabet`] if `alphabet_len == 0`.
    /// - [`Error::MatrixShape`] if `matrix.len() != alphabet_len * alphabet_len`.
    /// - [`Error::NegativeGapPenalty`] if either penalty is negative.
    /// - [`Error::GapOpenLessThanExtend`] if `gap_open < gap_ext`.
    pub fn new(alphabet_len: usize, matrix: Vec<i32>, gap_open: i32, gap_ext: i32) -> Result<Self> {
        Self::build(alphabet_len, matrix, Gaps::symmetric(gap_open, gap_ext))
    }

    /// Build a scoring scheme that charges the two [gap directions](self#gap-direction)
    /// differently.
    ///
    /// `query_gap_*` penalise a gap in the query (the `E` matrix; consumes target only, a
    /// *deletion* when the query is a read) and `target_gap_*` a gap in the target (the `F`
    /// matrix; consumes query only, an *insertion*). Each direction is validated exactly as in
    /// [`Scoring::new`], and passing the same pair twice is equivalent to calling it.
    ///
    /// # Translating from bwa
    ///
    /// bwa's `-O o_del,o_ins -E e_del,e_ins` uses the ksw2 convention (`open + n * ext` for a
    /// gap of length `n`) and orders the pair deletion-first, so
    ///
    /// ```text
    /// Scoring::new_asymmetric(al, matrix, o_del + e_del, e_del, o_ins + e_ins, e_ins)
    /// ```
    ///
    /// reproduces `bwa mem -O o_del,o_ins -E e_del,e_ins` under Opal's convention. For the
    /// defaults `-O 6,7 -E 1,2` that is `(7, 1, 9, 2)`.
    ///
    /// # Errors
    ///
    /// As [`Scoring::new`], with each penalty pair checked in turn: the query-gap pair first,
    /// then the target-gap pair. The reported [`Error::NegativeGapPenalty`] /
    /// [`Error::GapOpenLessThanExtend`] carries the offending direction's pair.
    pub fn new_asymmetric(
        alphabet_len: usize,
        matrix: Vec<i32>,
        query_gap_open: i32,
        query_gap_ext: i32,
        target_gap_open: i32,
        target_gap_ext: i32,
    ) -> Result<Self> {
        Self::build(
            alphabet_len,
            matrix,
            Gaps {
                query_open: query_gap_open,
                query_ext: query_gap_ext,
                target_open: target_gap_open,
                target_ext: target_gap_ext,
            },
        )
    }

    /// The shared constructor behind [`Scoring::new`] and [`Scoring::new_asymmetric`].
    fn build(alphabet_len: usize, matrix: Vec<i32>, gaps: Gaps) -> Result<Self> {
        if alphabet_len == 0 {
            return Err(Error::EmptyAlphabet);
        }

        // `checked_mul` guards absurd alphabet sizes whose square overflows `usize`; such a
        // matrix could never be supplied, so it is reported as a shape mismatch.
        let expected = alphabet_len.checked_mul(alphabet_len);
        if expected != Some(matrix.len()) {
            return Err(Error::MatrixShape {
                alphabet_len,
                expected: expected.unwrap_or(usize::MAX),
                got: matrix.len(),
            });
        }

        // Each direction is checked on its own, so an asymmetric scheme cannot smuggle an
        // invalid pair past the guard that a symmetric one would trip.
        for (gap_open, gap_ext) in [gaps.query(), gaps.target()] {
            if gap_open < 0 || gap_ext < 0 {
                return Err(Error::NegativeGapPenalty { gap_open, gap_ext });
            }
            if gap_open < gap_ext {
                return Err(Error::GapOpenLessThanExtend { gap_open, gap_ext });
            }
        }

        // Safe: alphabet_len >= 1 implies the matrix is non-empty.
        let min_entry = *matrix.iter().min().expect("non-empty matrix");
        let max_entry = *matrix.iter().max().expect("non-empty matrix");

        Ok(Self {
            alphabet_len,
            matrix,
            gaps,
            min_entry,
            max_entry,
        })
    }

    /// The alphabet length (number of distinct symbols).
    #[must_use]
    pub fn alphabet_len(&self) -> usize {
        self.alphabet_len
    }

    /// The gap-open penalty magnitude.
    ///
    /// Under an asymmetric scheme there is no single such value, and this returns the
    /// **query-gap** one; use [`Scoring::query_gap_open`] / [`Scoring::target_gap_open`] to say
    /// which you mean.
    #[must_use]
    #[deprecated(
        since = "0.3.0",
        note = "ambiguous under asymmetric gaps: use query_gap_open() or target_gap_open()"
    )]
    pub fn gap_open(&self) -> i32 {
        self.gaps.query_open
    }

    /// The gap-extend penalty magnitude.
    ///
    /// Under an asymmetric scheme there is no single such value, and this returns the
    /// **query-gap** one; use [`Scoring::query_gap_ext`] / [`Scoring::target_gap_ext`] to say
    /// which you mean.
    #[must_use]
    #[deprecated(
        since = "0.3.0",
        note = "ambiguous under asymmetric gaps: use query_gap_ext() or target_gap_ext()"
    )]
    pub fn gap_ext(&self) -> i32 {
        self.gaps.query_ext
    }

    /// The open penalty for a gap in the **query** (the `E` matrix; consumes target only, a
    /// deletion when the query is a read).
    #[must_use]
    pub fn query_gap_open(&self) -> i32 {
        self.gaps.query_open
    }

    /// The extend penalty for a gap in the **query**. See [`Scoring::query_gap_open`].
    #[must_use]
    pub fn query_gap_ext(&self) -> i32 {
        self.gaps.query_ext
    }

    /// The open penalty for a gap in the **target** (the `F` matrix; consumes query only, an
    /// insertion when the query is a read).
    #[must_use]
    pub fn target_gap_open(&self) -> i32 {
        self.gaps.target_open
    }

    /// The extend penalty for a gap in the **target**. See [`Scoring::target_gap_open`].
    #[must_use]
    pub fn target_gap_ext(&self) -> i32 {
        self.gaps.target_ext
    }

    /// Whether both [gap directions](self#gap-direction) are charged alike, i.e. whether this
    /// scheme is one [`Scoring::new`] could have built.
    #[must_use]
    pub fn has_symmetric_gaps(&self) -> bool {
        self.gaps.is_symmetric()
    }

    /// The four penalties as one value, for threading through the kernels.
    #[inline]
    pub(crate) fn gaps(&self) -> Gaps {
        self.gaps
    }

    /// The most negative and most positive substitution-matrix entries, respectively.
    #[must_use]
    pub fn entry_bounds(&self) -> (i32, i32) {
        (self.min_entry, self.max_entry)
    }

    /// The substitution score for query symbol `q` against target symbol `t` (both encoded
    /// indices in `0..alphabet_len`).
    ///
    /// # Panics
    ///
    /// Panics if `q` or `t` is `>= alphabet_len`. Callers in the hot path pass pre-validated
    /// encoded indices, so this bound is a debug guard rather than a runtime cost there.
    #[inline]
    #[must_use]
    pub fn score(&self, q: usize, t: usize) -> i32 {
        assert!(
            q < self.alphabet_len && t < self.alphabet_len,
            "symbol index out of range: q={q}, t={t}, alphabet_len={}",
            self.alphabet_len
        );
        self.matrix[q * self.alphabet_len + t]
    }

    /// The substitution-score row for query symbol `q`: `self.matrix[q * al .. (q + 1) * al]`,
    /// indexable by target symbol. Hoisting this out of a DP's inner column loop moves the query
    /// bound check from per-cell to per-row (the target symbol is still bound-checked per access).
    ///
    /// # Panics
    ///
    /// Panics if `q >= alphabet_len`.
    #[inline]
    #[must_use]
    pub(crate) fn score_row(&self, q: usize) -> &[i32] {
        let al = self.alphabet_len;
        &self.matrix[q * al..q * al + al]
    }

    /// Prove the narrowest [`ScoreWidth`] whose range cannot overflow for `mode` over sequences
    /// bounded by `max_query_len` and `max_target_len`. See [`ScoreWidth`].
    ///
    /// # Errors
    ///
    /// [`Error::ScoreRangeTooWide`] if the reachable score magnitude exceeds the `i32` range.
    pub fn required_width(
        &self,
        mode: Mode,
        max_query_len: usize,
        max_target_len: usize,
    ) -> Result<ScoreWidth> {
        width::required_width(
            mode,
            self.min_entry,
            self.max_entry,
            self.gaps,
            max_query_len,
            max_target_len,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A simple DNA-ish match/mismatch matrix over an `n`-symbol alphabet.
    fn match_mismatch(n: usize, m: i32, x: i32) -> Vec<i32> {
        let mut v = vec![x; n * n];
        for i in 0..n {
            v[i * n + i] = m;
        }
        v
    }

    #[test]
    fn valid_scheme_round_trips_all_accessors() {
        let s = Scoring::new(4, match_mismatch(4, 2, -1), 3, 1).unwrap();
        assert_eq!(s.alphabet_len(), 4);
        assert_eq!(s.query_gap_open(), 3);
        assert_eq!(s.query_gap_ext(), 1);
        assert_eq!(s.target_gap_open(), 3);
        assert_eq!(s.target_gap_ext(), 1);
        assert!(s.has_symmetric_gaps());
        assert_eq!(s.entry_bounds(), (-1, 2));
        assert_eq!(s.score(0, 0), 2);
        assert_eq!(s.score(0, 1), -1);
        assert_eq!(s.score(3, 3), 2);
    }

    #[test]
    fn accepts_gap_open_equal_to_gap_ext_boundary() {
        // The #28 guard is `>=`, so equal penalties (linear gaps) must be accepted.
        assert!(Scoring::new(2, match_mismatch(2, 1, -1), 5, 5).is_ok());
    }

    #[test]
    fn rejects_gap_open_below_gap_ext_by_one() {
        // Just across the boundary in the other direction.
        let err = Scoring::new(2, match_mismatch(2, 1, -1), 4, 5).unwrap_err();
        assert_eq!(
            err,
            Error::GapOpenLessThanExtend {
                gap_open: 4,
                gap_ext: 5
            }
        );
    }

    #[test]
    fn rejects_empty_alphabet() {
        assert_eq!(
            Scoring::new(0, vec![], 1, 1).unwrap_err(),
            Error::EmptyAlphabet
        );
        // Even if a caller passes a stray matrix, alphabet emptiness is reported first.
        assert_eq!(
            Scoring::new(0, vec![1, 2, 3], 1, 1).unwrap_err(),
            Error::EmptyAlphabet
        );
    }

    #[test]
    fn rejects_every_wrong_matrix_shape() {
        // Too short, too long, and off-by-one on both sides of the exact size.
        for (n, len) in [(3usize, 8usize), (3, 10), (2, 3), (2, 5), (4, 0)] {
            let err = Scoring::new(n, vec![0; len], 1, 1).unwrap_err();
            assert_eq!(
                err,
                Error::MatrixShape {
                    alphabet_len: n,
                    expected: n * n,
                    got: len
                },
                "n={n}, len={len}"
            );
        }
        // Exactly right sizes must pass for a range of alphabet lengths.
        for n in 1..=6 {
            assert!(
                Scoring::new(n, vec![0; n * n], 1, 1).is_ok(),
                "n={n} exact size"
            );
        }
    }

    #[test]
    fn rejects_negative_penalties_from_either_field() {
        for (go, ge) in [(-1, 0), (0, -1), (-5, -5), (2, -3)] {
            let err = Scoring::new(2, match_mismatch(2, 1, -1), go, ge).unwrap_err();
            assert_eq!(
                err,
                Error::NegativeGapPenalty {
                    gap_open: go,
                    gap_ext: ge
                },
                "go={go}, ge={ge}"
            );
        }
    }

    #[test]
    fn negative_penalty_is_checked_before_the_ordering_invariant() {
        // gap_open=-1 < gap_ext=0 would also trip #28, but the negative check must win so the
        // caller sees the more specific error.
        let err = Scoring::new(2, match_mismatch(2, 1, -1), -1, 0).unwrap_err();
        assert_eq!(
            err,
            Error::NegativeGapPenalty {
                gap_open: -1,
                gap_ext: 0
            }
        );
    }

    #[test]
    fn score_lookup_is_asymmetric_when_matrix_is() {
        // Guard against a q/t transposition bug: use a deliberately asymmetric matrix.
        let matrix = vec![
            0, 1, 2, //
            3, 4, 5, //
            6, 7, 8,
        ];
        let s = Scoring::new(3, matrix, 1, 1).unwrap();
        assert_eq!(s.score(0, 2), 2, "row 0, col 2");
        assert_eq!(s.score(2, 0), 6, "row 2, col 0");
        assert_eq!(s.score(1, 2), 5);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn score_out_of_range_panics() {
        let s = Scoring::new(2, match_mismatch(2, 1, -1), 1, 1).unwrap();
        let _ = s.score(2, 0);
    }

    #[test]
    fn entry_bounds_reflect_extremes_for_varied_matrices() {
        for (n, m, x) in [(2, 1, -1), (4, 5, -4), (5, 100, -100), (3, 0, 0)] {
            let s = Scoring::new(n, match_mismatch(n, m, x), 1, 1).unwrap();
            assert_eq!(
                s.entry_bounds(),
                (m.min(x), m.max(x)),
                "n={n}, m={m}, x={x}"
            );
        }
    }

    #[test]
    fn required_width_delegates_and_reflects_penalties() {
        let s = Scoring::new(4, match_mismatch(4, 2, -1), 3, 1).unwrap();
        // Small local search fits i8.
        assert_eq!(s.required_width(Mode::Sw, 20, 20).unwrap(), ScoreWidth::I8);
        // Global over long sequences escalates.
        assert_eq!(
            s.required_width(Mode::Nw, 40_000, 40_000).unwrap(),
            ScoreWidth::I32
        );
    }
}
