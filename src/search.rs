//! Search types: how much of the alignment the caller wants computed.

use core::fmt;

/// What the search computes and returns.
///
/// The variants are ordered by cost. `Score` and `ScoreEnd` run entirely in the SIMD score
/// pass and allocate no traceback. `Alignment` (with linear-space Hirschberg traceback and a
/// caller memory budget) is planned for a later milestone and is intentionally absent here so
/// the M0 surface stays minimal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SearchType {
    /// Best score only. Cheapest; no end positions are tracked.
    Score,
    /// Best score plus the query and target end positions of the optimal alignment. Still no
    /// traceback allocation.
    ScoreEnd,
}

impl SearchType {
    /// Whether this search reports the alignment end positions (query/target). True for
    /// [`SearchType::ScoreEnd`].
    #[must_use]
    pub const fn tracks_end(self) -> bool {
        matches!(self, SearchType::ScoreEnd)
    }
}

impl fmt::Display for SearchType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchType::Score => f.write_str("Score"),
            SearchType::ScoreEnd => f.write_str("ScoreEnd"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_score_end_tracks_end() {
        assert!(!SearchType::Score.tracks_end());
        assert!(SearchType::ScoreEnd.tracks_end());
    }

    #[test]
    fn display_round_trips_names() {
        assert_eq!(SearchType::Score.to_string(), "Score");
        assert_eq!(SearchType::ScoreEnd.to_string(), "ScoreEnd");
    }
}
