//! The database-scan API: an immutable, shareable [`Database`] and per-thread [`Scratch`].
//!
//! This is the core split the whole design turns on. A [`Database`] is built once, is immutable,
//! and is `Send + Sync` so it can sit behind an `Arc` and be shared across a thread pool. Each
//! worker holds its own mutable [`Scratch`]; [`Database::scan`] borrows it, allocates nothing,
//! and is infallible — every fallible check (symbol range, score-width proof) happens once at
//! [`build`](DatabaseBuilder::build).
//!
//! ```
//! use hyalite::{Database, Mode, Scoring, Scratch, SearchType};
//!
//! let scoring = Scoring::new(4, vec![
//!     2, -1, -1, -1,
//!     -1, 2, -1, -1,
//!     -1, -1, 2, -1,
//!     -1, -1, -1, 2,
//! ], 2, 1).unwrap();
//!
//! let db = Database::builder()
//!     .sequences(&[vec![0u8, 1, 2, 3], vec![2u8, 2, 2]])
//!     .scoring(scoring)
//!     .mode(Mode::Sw)
//!     .search_type(SearchType::ScoreEnd)
//!     .max_query_len(8)
//!     .build()
//!     .unwrap();
//!
//! let mut scratch = Scratch::new(&db);
//! let hit = db.scan(&mut scratch, &[0u8, 1, 2, 3]);
//! assert_eq!(hit.db_index, 0);
//! ```

use crate::backend::{self, Backend, BackendChoice};
use crate::error::{Error, Result};
use crate::hit::BestHit;
use crate::kernel::{DpBuffers, align_core};
use crate::mode::Mode;
use crate::scoring::Scoring;
use crate::search::SearchType;
use crate::width::ScoreWidth;

/// An immutable, thread-safe set of target sequences plus the resolved scoring, mode, search
/// type, and score width to scan a query against. Build one with [`Database::builder`].
#[derive(Debug, Clone)]
pub struct Database {
    sequences: Vec<Vec<u8>>,
    scoring: Scoring,
    mode: Mode,
    search_type: SearchType,
    max_query_len: usize,
    max_target_len: usize,
    width: ScoreWidth,
    backend: Backend,
}

impl Database {
    /// Start building a database.
    #[must_use]
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder::new()
    }

    /// Scan `query` against every sequence and return the single best [`BestHit`].
    ///
    /// Infallible and allocation-free: it reuses `scratch`. The winner is the highest score; ties
    /// are broken by the **smallest database index**, resolved by a scalar reduction over a
    /// per-sequence comparison — never a lane-order-dependent horizontal max. Together with the
    /// per-alignment end tie-break, this is what makes the result identical across every backend.
    ///
    /// `query` must be pre-encoded indices in `0..alphabet_len` and no longer than the
    /// `max_query_len` declared at build time. Passing out-of-range symbols or an over-long query
    /// is a caller-contract violation and will panic rather than return silently wrong results.
    #[must_use]
    pub fn scan(&self, scratch: &mut Scratch, query: &[u8]) -> BestHit {
        debug_assert!(
            query.len() <= self.max_query_len,
            "query length {} exceeds declared max_query_len {}",
            query.len(),
            self.max_query_len
        );

        // SIMD backends run the inter-sequence kernel. The resolver only selects one for a
        // SIMD-eligible database, so this dispatch is safe to take unconditionally here.
        if self.backend != Backend::Scalar {
            return crate::inter::scan_dispatch(
                self.backend,
                &self.sequences,
                &self.scoring,
                self.mode,
                self.search_type,
                query,
            );
        }

        // Reduce to the best (score, end) per sequence, then take a scalar argmax over the
        // database index. Iterating ascending and replacing only on a strictly greater score
        // keeps the smallest index on a tie.
        let mut best_score = i32::MIN;
        let mut best_index = 0usize;
        let mut best_query_end = None;
        let mut best_target_end = None;
        for (index, seq) in self.sequences.iter().enumerate() {
            let (score, query_end, target_end) =
                align_core(query, seq, &self.scoring, self.mode, &mut scratch.buf);
            if score > best_score {
                best_score = score;
                best_index = index;
                best_query_end = query_end;
                best_target_end = target_end;
            }
        }

        let (query_end, target_end) = if self.search_type.tracks_end() {
            (best_query_end, best_target_end)
        } else {
            (None, None)
        };

        BestHit {
            score: best_score,
            db_index: best_index,
            query_end,
            target_end,
        }
    }

    /// The number of sequences in the database (always `>= 1`).
    #[must_use]
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// The alignment mode this database scans in.
    #[must_use]
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// The search type (whether end positions are reported).
    #[must_use]
    pub fn search_type(&self) -> SearchType {
        self.search_type
    }

    /// The scoring scheme.
    #[must_use]
    pub fn scoring(&self) -> &Scoring {
        &self.scoring
    }

    /// The integer width proven sufficient for scores in this database. Informational in M0
    /// (the scalar kernel computes in `i32`); it selects the lane width for SIMD backends later.
    #[must_use]
    pub fn score_width(&self) -> ScoreWidth {
        self.width
    }

    /// The resolved compute backend. Always [`Backend::Scalar`] in M0, but reflects the
    /// [`BackendChoice`] and `HYALITE_BACKEND` override once SIMD backends exist.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The maximum query length this database was built for.
    #[must_use]
    pub fn max_query_len(&self) -> usize {
        self.max_query_len
    }

    /// The length of the longest sequence in the database.
    #[must_use]
    pub fn max_target_len(&self) -> usize {
        self.max_target_len
    }
}

/// A builder for [`Database`]. Required: [`sequences`](Self::sequences),
/// [`scoring`](Self::scoring), [`mode`](Self::mode), [`max_query_len`](Self::max_query_len).
/// [`search_type`](Self::search_type) defaults to [`SearchType::Score`].
#[derive(Debug, Clone, Default)]
pub struct DatabaseBuilder {
    sequences: Option<Vec<Vec<u8>>>,
    scoring: Option<Scoring>,
    mode: Option<Mode>,
    search_type: Option<SearchType>,
    max_query_len: Option<usize>,
    backend_choice: Option<BackendChoice>,
}

impl DatabaseBuilder {
    fn new() -> Self {
        DatabaseBuilder::default()
    }

    /// Set the target sequences (pre-encoded alphabet indices). Copied into the builder.
    #[must_use]
    pub fn sequences<S: AsRef<[u8]>>(mut self, sequences: &[S]) -> Self {
        self.sequences = Some(sequences.iter().map(|s| s.as_ref().to_vec()).collect());
        self
    }

    /// Set the scoring scheme.
    #[must_use]
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = Some(scoring);
        self
    }

    /// Set the alignment mode.
    #[must_use]
    pub fn mode(mut self, mode: Mode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Set the search type. Defaults to [`SearchType::Score`] if unset.
    #[must_use]
    pub fn search_type(mut self, search_type: SearchType) -> Self {
        self.search_type = Some(search_type);
        self
    }

    /// Declare the maximum query length that will be scanned. Required so the score-width proof
    /// can be completed at build time, keeping [`Database::scan`] infallible.
    #[must_use]
    pub fn max_query_len(mut self, max_query_len: usize) -> Self {
        self.max_query_len = Some(max_query_len);
        self
    }

    /// Override backend selection. Takes precedence over the `HYALITE_BACKEND` environment
    /// variable, which in turn takes precedence over automatic detection. Forcing a backend that
    /// is not available makes [`build`](Self::build) fail with [`Error::BackendUnavailable`].
    #[must_use]
    pub fn backend(mut self, choice: BackendChoice) -> Self {
        self.backend_choice = Some(choice);
        self
    }

    /// Validate everything and build the [`Database`].
    ///
    /// # Errors
    ///
    /// - [`Error::IncompleteBuilder`] if a required field is unset.
    /// - [`Error::EmptyDatabase`] if no sequences were provided.
    /// - [`Error::SymbolOutOfRange`] if any sequence contains a symbol `>= alphabet_len`.
    /// - [`Error::ScoreRangeTooWide`] if scores could overflow `i32` for the declared lengths.
    /// - [`Error::InvalidBackendName`] if `HYALITE_BACKEND` is set to an unrecognised value.
    /// - [`Error::BackendUnavailable`] if a forced backend is not available.
    pub fn build(self) -> Result<Database> {
        let sequences = self
            .sequences
            .ok_or(Error::IncompleteBuilder { field: "sequences" })?;
        let scoring = self
            .scoring
            .ok_or(Error::IncompleteBuilder { field: "scoring" })?;
        let mode = self
            .mode
            .ok_or(Error::IncompleteBuilder { field: "mode" })?;
        let max_query_len = self.max_query_len.ok_or(Error::IncompleteBuilder {
            field: "max_query_len",
        })?;
        let search_type = self.search_type.unwrap_or(SearchType::Score);

        if sequences.is_empty() {
            return Err(Error::EmptyDatabase);
        }

        let alphabet_len = scoring.alphabet_len();
        for seq in &sequences {
            for &sym in seq {
                if sym as usize >= alphabet_len {
                    return Err(Error::SymbolOutOfRange {
                        symbol: sym as usize,
                        alphabet_len,
                    });
                }
            }
        }

        let max_target_len = sequences.iter().map(Vec::len).max().unwrap_or(0);

        // Prove i32 suffices for any query up to max_query_len against these targets.
        let width = scoring.required_width(mode, max_query_len, max_target_len)?;

        // Resolve the backend: an explicit builder choice wins, else HYALITE_BACKEND, else auto.
        let choice = match self.backend_choice {
            Some(choice) => choice,
            None => backend::choice_from_env()?.unwrap_or(BackendChoice::Auto),
        };
        let resolved = backend::resolve(choice)?;

        // A SIMD backend is only *usable* for a SIMD-eligible database (i8 width, small alphabet).
        // If one was auto-selected but the database is ineligible, fall back to scalar; if it was
        // explicitly forced, that is an error rather than a silent fallback.
        let applicable = crate::inter::kernel_applies(width, scoring.alphabet_len());
        let backend = if resolved == Backend::Scalar || applicable {
            resolved
        } else {
            match choice {
                BackendChoice::Force(_) => {
                    return Err(Error::BackendUnavailable { backend: resolved });
                }
                BackendChoice::Auto => Backend::Scalar,
            }
        };

        Ok(Database {
            sequences,
            scoring,
            mode,
            search_type,
            max_query_len,
            max_target_len,
            width,
            backend,
        })
    }
}

/// Per-thread mutable working memory for [`Database::scan`]. Create one per worker thread and
/// reuse it across scans; it holds no reference to the database, so it can outlive individual
/// scan calls but should match the database it was sized for.
#[derive(Debug)]
pub struct Scratch {
    buf: DpBuffers,
}

impl Scratch {
    /// Allocate scratch pre-sized for `db`, so no scan reallocates.
    #[must_use]
    pub fn new(db: &Database) -> Self {
        Scratch {
            buf: DpBuffers::with_capacity(db.max_query_len(), db.max_target_len()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dna_scoring() -> Scoring {
        Scoring::new(
            4,
            vec![
                2, -1, -1, -1, //
                -1, 2, -1, -1, //
                -1, -1, 2, -1, //
                -1, -1, -1, 2,
            ],
            2,
            1,
        )
        .unwrap()
    }

    fn db_with(seqs: &[Vec<u8>], mode: Mode, st: SearchType) -> Database {
        Database::builder()
            .sequences(seqs)
            .scoring(dna_scoring())
            .mode(mode)
            .search_type(st)
            .max_query_len(16)
            .build()
            .unwrap()
    }

    #[test]
    fn database_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Database>();
    }

    #[test]
    fn build_reports_every_missing_required_field() {
        let base = || Database::builder();
        assert_eq!(
            base().build().unwrap_err(),
            Error::IncompleteBuilder { field: "sequences" }
        );
        assert_eq!(
            base().sequences(&[vec![0u8]]).build().unwrap_err(),
            Error::IncompleteBuilder { field: "scoring" }
        );
        assert_eq!(
            base()
                .sequences(&[vec![0u8]])
                .scoring(dna_scoring())
                .build()
                .unwrap_err(),
            Error::IncompleteBuilder { field: "mode" }
        );
        assert_eq!(
            base()
                .sequences(&[vec![0u8]])
                .scoring(dna_scoring())
                .mode(Mode::Sw)
                .build()
                .unwrap_err(),
            Error::IncompleteBuilder {
                field: "max_query_len"
            }
        );
    }

    #[test]
    fn empty_database_is_rejected() {
        let empty: [Vec<u8>; 0] = [];
        let err = Database::builder()
            .sequences(&empty)
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .build()
            .unwrap_err();
        assert_eq!(err, Error::EmptyDatabase);
    }

    #[test]
    fn out_of_range_symbol_in_a_sequence_is_rejected() {
        let err = Database::builder()
            .sequences(&[vec![0u8, 1], vec![2u8, 9]])
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .build()
            .unwrap_err();
        assert_eq!(
            err,
            Error::SymbolOutOfRange {
                symbol: 9,
                alphabet_len: 4
            }
        );
    }

    #[test]
    fn search_type_defaults_to_score() {
        let db = Database::builder()
            .sequences(&[vec![0u8]])
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .build()
            .unwrap();
        assert_eq!(db.search_type(), SearchType::Score);
    }

    #[test]
    fn accessors_reflect_construction() {
        let db = db_with(
            &[vec![0u8, 1, 2, 3], vec![2u8, 2]],
            Mode::Hw,
            SearchType::ScoreEnd,
        );
        assert_eq!(db.sequence_count(), 2);
        assert_eq!(db.mode(), Mode::Hw);
        assert_eq!(db.search_type(), SearchType::ScoreEnd);
        assert_eq!(db.max_query_len(), 16);
        assert_eq!(db.max_target_len(), 4);
        // Under Auto the backend is CPU-dependent (SSE4.1 where available); just require it valid.
        assert!(db.backend().is_available());
        assert_eq!(db.score_width(), ScoreWidth::I8);
    }

    #[test]
    fn forcing_scalar_backend_builds_and_reports_scalar() {
        let db = Database::builder()
            .sequences(&[vec![0u8]])
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .backend(BackendChoice::Force(Backend::Scalar))
            .build()
            .unwrap();
        assert_eq!(db.backend(), Backend::Scalar);
    }

    #[test]
    fn forcing_a_backend_tracks_availability() {
        // The database (i8 width, alphabet 4) is SIMD-eligible, so forcing a backend succeeds iff
        // that backend is available on this CPU; otherwise it is a clean BackendUnavailable.
        for b in [Backend::Sse41, Backend::Avx2, Backend::Neon] {
            let result = Database::builder()
                .sequences(&[vec![0u8]])
                .scoring(dna_scoring())
                .mode(Mode::Sw)
                .max_query_len(8)
                .backend(BackendChoice::Force(b))
                .build();
            if b.is_available() {
                assert_eq!(result.unwrap().backend(), b);
            } else {
                assert_eq!(
                    result.unwrap_err(),
                    Error::BackendUnavailable { backend: b }
                );
            }
        }
    }

    #[test]
    fn forcing_simd_on_an_ineligible_database_errors() {
        // A wide-score database (long global alignment) needs i16, so no SIMD kernel applies.
        // Forcing one must fail loudly rather than silently falling back.
        let scoring = Scoring::new(2, vec![100, -100, -100, 100], 2, 1).unwrap();
        let long = vec![0u8; 200];
        let result = Database::builder()
            .sequences(&[long])
            .scoring(scoring)
            .mode(Mode::Nw)
            .max_query_len(200)
            .backend(BackendChoice::Force(Backend::Sse41))
            .build();
        // On a CPU without SSE4.1 the resolver rejects it before the eligibility check; either way
        // the outcome is a build error, never a silent scalar fallback.
        assert!(matches!(result, Err(Error::BackendUnavailable { .. })));
    }

    #[test]
    fn explicit_scalar_choice_overrides_auto_detection() {
        // Forcing scalar yields scalar even on a machine where Auto would pick a SIMD backend —
        // the explicit builder choice wins over detection (and, in turn, over the env var).
        let db = Database::builder()
            .sequences(&[vec![0u8]])
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .backend(BackendChoice::Force(Backend::Scalar))
            .build()
            .unwrap();
        assert_eq!(db.backend(), Backend::Scalar);
    }

    #[test]
    fn scan_picks_the_best_scoring_sequence() {
        // Query ACGT: sequence 1 is a perfect match, sequence 0 a poor one.
        let db = db_with(
            &[vec![2u8, 2, 2, 2], vec![0u8, 1, 2, 3]],
            Mode::Sw,
            SearchType::ScoreEnd,
        );
        let mut scratch = Scratch::new(&db);
        let hit = db.scan(&mut scratch, &[0u8, 1, 2, 3]);
        assert_eq!(hit.db_index, 1);
        assert_eq!(hit.score, 8);
        assert_eq!((hit.query_end, hit.target_end), (Some(3), Some(3)));
    }

    #[test]
    fn tie_break_prefers_smallest_database_index() {
        // Two identical best-scoring sequences: index 0 must win.
        let db = db_with(
            &[vec![0u8, 1, 2, 3], vec![0u8, 1, 2, 3], vec![3u8]],
            Mode::Sw,
            SearchType::ScoreEnd,
        );
        let mut scratch = Scratch::new(&db);
        let hit = db.scan(&mut scratch, &[0u8, 1, 2, 3]);
        assert_eq!(hit.db_index, 0);
        assert_eq!(hit.score, 8);
    }

    #[test]
    fn score_search_type_suppresses_ends_in_scan() {
        let db = db_with(&[vec![0u8, 1, 2, 3]], Mode::Sw, SearchType::Score);
        let mut scratch = Scratch::new(&db);
        let hit = db.scan(&mut scratch, &[0u8, 1, 2, 3]);
        assert_eq!(hit.score, 8);
        assert_eq!((hit.query_end, hit.target_end), (None, None));
    }

    #[test]
    fn scratch_reuse_across_many_scans_is_consistent() {
        // The same scratch, reused across scans of differing target/query sizes, must produce the
        // same answer each time an input repeats — catches stale-buffer bugs from capacity reuse.
        let db = db_with(
            &[vec![0u8, 1], vec![0u8, 1, 2, 3, 3, 2, 1, 0], vec![2u8]],
            Mode::Nw,
            SearchType::ScoreEnd,
        );
        let mut scratch = Scratch::new(&db);
        let queries: [&[u8]; 4] = [
            &[0, 1, 2, 3],
            &[2],
            &[0, 1, 2, 3, 3, 2, 1, 0],
            &[0, 1, 2, 3],
        ];
        let mut first_repeat = None;
        for q in queries {
            let hit = db.scan(&mut scratch, q);
            // Re-run the same query with a fresh scratch and require identical results.
            let mut fresh = Scratch::new(&db);
            let hit_fresh = db.scan(&mut fresh, q);
            assert_eq!(hit, hit_fresh, "reused vs fresh scratch differ for {q:?}");
            if q == [0u8, 1, 2, 3].as_slice() {
                match first_repeat {
                    None => first_repeat = Some(hit),
                    Some(prev) => {
                        assert_eq!(prev, hit, "same query gave different result on reuse")
                    }
                }
            }
        }
    }
}
