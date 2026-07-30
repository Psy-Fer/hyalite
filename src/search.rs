//! Search types: how much of the alignment the caller wants computed.

use core::fmt;

/// What the search computes and returns.
///
/// The variants are ordered by cost. `Score` and `ScoreEnd` run entirely in the SIMD score pass
/// and allocate no traceback. `Alignment` additionally recovers the full traceback (operations
/// and aligned span) with a scalar per-target pass, bounded to `max_bytes` working memory (see
/// [`align`](crate::align) and [`Database::scan_aligned`](crate::Database::scan_aligned)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SearchType {
    /// Best score only. Cheapest; no end positions are tracked.
    Score,
    /// Best score plus the query and target end positions of the optimal alignment. Still no
    /// traceback allocation.
    ScoreEnd,
    /// The full alignment (score, span, and operations), recovered by scalar traceback. The
    /// `max_bytes` budget bounds the traceback working memory; a database built with this search
    /// type validates at construction that the budget suffices for its declared maximum problem
    /// size, so the alignment scans are infallible.
    Alignment {
        /// Traceback working-memory budget, in bytes (see [`align`](crate::align)).
        max_bytes: usize,
    },
}

impl SearchType {
    /// Whether this search reports the alignment end positions (query/target). True for
    /// [`SearchType::ScoreEnd`] and [`SearchType::Alignment`].
    #[must_use]
    pub const fn tracks_end(self) -> bool {
        matches!(self, SearchType::ScoreEnd | SearchType::Alignment { .. })
    }

    /// The traceback budget for [`SearchType::Alignment`], or `None` for the score-only types.
    #[must_use]
    pub const fn max_bytes(self) -> Option<usize> {
        match self {
            SearchType::Alignment { max_bytes } => Some(max_bytes),
            _ => None,
        }
    }
}

impl fmt::Display for SearchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchType::Score => f.write_str("Score"),
            SearchType::ScoreEnd => f.write_str("ScoreEnd"),
            SearchType::Alignment { max_bytes } => {
                write!(f, "Alignment {{ max_bytes: {max_bytes} }}")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_and_end_tracking() {
        assert!(!SearchType::Score.tracks_end());
        assert!(SearchType::ScoreEnd.tracks_end());
        assert!(SearchType::Alignment { max_bytes: 4096 }.tracks_end());
    }

    #[test]
    fn max_bytes_only_on_alignment() {
        assert_eq!(SearchType::Score.max_bytes(), None);
        assert_eq!(SearchType::ScoreEnd.max_bytes(), None);
        assert_eq!(
            SearchType::Alignment { max_bytes: 4096 }.max_bytes(),
            Some(4096)
        );
    }

    #[test]
    fn display_round_trips_names() {
        assert_eq!(SearchType::Score.to_string(), "Score");
        assert_eq!(SearchType::ScoreEnd.to_string(), "ScoreEnd");
        assert_eq!(
            SearchType::Alignment { max_bytes: 64 }.to_string(),
            "Alignment { max_bytes: 64 }"
        );
    }
}
