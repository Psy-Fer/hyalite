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

impl fmt::Display for BestHit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "score={} db={}", self.score, self.db_index)?;
        match (self.query_end, self.target_end) {
            (Some(q), Some(t)) => write!(f, " query_end={q} target_end={t}"),
            _ => Ok(()),
        }
    }
}
