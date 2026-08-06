//! The result of an alignment.

use core::fmt;

/// The best-scoring alignment found, carrying exactly the four fields the determinism contract
/// pins: score, database index, query end position, and target end position.
///
/// `query_end` / `target_end` are `0`-based indices of the last aligned position in each
/// sequence, or `None` when that sequence contributes no aligned position — either because the
/// sequence is empty / nothing aligned to it, or because the search was
/// [`SearchType::Score`](crate::SearchType::Score), which does not track end positions.
///
/// `db_index` is the index of the winning sequence in a database scan; for a single pair
/// alignment ([`align_pair`](crate::align_pair)) it is always `0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BestHit {
    /// The optimal alignment score.
    pub score: i32,
    /// Index of the winning database sequence (`0` for a single pair).
    pub db_index: usize,
    /// `0`-based end position in the query, or `None` (see [`BestHit`]).
    pub query_end: Option<usize>,
    /// `0`-based end position in the target, or `None` (see [`BestHit`]).
    pub target_end: Option<usize>,
}

/// A local (`SW`) alignment's score and its aligned **span** in each sequence — the start and end
/// coordinates, without the column-by-column operations.
///
/// The spans are **half-open** `[start, end)`, matching [`Alignment`](crate::Alignment):
/// `query[query_start..query_end]` and `target[target_start..target_end]` are the aligned regions
/// (so `query_end` is one past the last aligned symbol — `= BestHit.query_end + 1` when non-empty).
/// An alignment that found nothing scoring above `0` is **empty**: `score == 0` and every coordinate
/// is `0` (`start == end`).
///
/// Returned by [`align_pair_span`](crate::align_pair_span). Unlike [`Alignment`](crate::Alignment)
/// it needs no traceback matrix or CIGAR — a single forward pass with start tracking recovers it in
/// `O(target)` working memory — so it is the cheap way to get the aligned region's coordinates (e.g.
/// bwa-style mate rescue) when the operations themselves are not needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalSpan {
    /// The optimal local alignment score (`>= 0`; `0` for the empty alignment).
    pub score: i32,
    /// Start of the aligned query span (inclusive, `0`-based).
    pub query_start: usize,
    /// End of the aligned query span (exclusive).
    pub query_end: usize,
    /// Start of the aligned target span (inclusive, `0`-based).
    pub target_start: usize,
    /// End of the aligned target span (exclusive).
    pub target_end: usize,
}

impl LocalSpan {
    /// The empty span: nothing aligned above the local-alignment floor of `0`.
    pub(crate) const EMPTY: LocalSpan = LocalSpan {
        score: 0,
        query_start: 0,
        query_end: 0,
        target_start: 0,
        target_end: 0,
    };
}

impl fmt::Display for LocalSpan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "score={} query=[{}, {}) target=[{}, {})",
            self.score, self.query_start, self.query_end, self.target_start, self.target_end
        )
    }
}

impl fmt::Display for BestHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={} db={}", self.score, self.db_index)?;
        match (self.query_end, self.target_end) {
            (Some(q), Some(t)) => write!(f, " query_end={q} target_end={t}"),
            _ => Ok(()),
        }
    }
}
