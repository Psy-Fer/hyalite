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

use crate::align::{AlignedHit, Alignment};
use crate::backend::{self, Backend, BackendChoice};
use crate::error::{Error, Result};
use crate::hit::BestHit;
use crate::inter::{self, Layout, LayoutChoice, Packed, SimdScratch};
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
    /// The database packed for the inter-sequence kernel; `Some` only for a SIMD backend.
    packed: Option<Packed>,
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

        // SIMD backends run the inter-sequence kernel over the prebuilt packing, reusing the
        // scratch buffers. The resolver only selects a SIMD backend for an eligible database, so
        // `packed` is `Some` whenever the backend is non-scalar.
        if let Some(packed) = &self.packed {
            return inter::scan_dispatch(
                self.backend,
                packed,
                &self.sequences,
                &self.scoring,
                self.mode,
                self.search_type,
                query,
                &mut scratch.simd,
                &mut scratch.buf,
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

    /// Scan `query` against every sequence and write **one [`BestHit`] per database sequence**
    /// into `out`, in database order (`out[i].db_index == i`). `out` is cleared first and reused,
    /// so repeated calls allocate nothing once it has grown.
    ///
    /// This is the per-target counterpart to [`scan`](Self::scan): where `scan` returns only the
    /// single best hit, `scan_all` returns them all. Same determinism guarantee — each entry is
    /// bit-identical across backends.
    ///
    /// For [`SearchType::Score`] the per-sequence scores are computed by the resolved backend
    /// (SIMD-accelerated for an eligible database) and end positions are `None`. For
    /// [`SearchType::ScoreEnd`] the SIMD backends track end positions in-vector, so scores *and*
    /// ends are SIMD-accelerated; the scalar backend (and inputs whose positions exceed the `i16`
    /// position domain) recover ends via the scalar kernel per sequence. Every entry is
    /// bit-identical across paths.
    ///
    /// Same caller contract as [`scan`](Self::scan): `query` is pre-encoded, `<= max_query_len`.
    pub fn scan_all(&self, scratch: &mut Scratch, query: &[u8], out: &mut Vec<BestHit>) {
        debug_assert!(
            query.len() <= self.max_query_len,
            "query length {} exceeds declared max_query_len {}",
            query.len(),
            self.max_query_len
        );

        out.clear();
        out.reserve(self.sequences.len());

        // `ScoreEnd` needs per-target end positions. SIMD backends track them in-vector; otherwise
        // (scalar backend, or positions that would not fit the `i16` position domain) we recover
        // them with the scalar DP per sequence. `Score` uses the SIMD per-target kernel when a SIMD
        // backend is resolved.
        if self.search_type.tracks_end() {
            let ends_fit_i16 =
                self.max_query_len <= i16::MAX as usize && self.max_target_len <= i16::MAX as usize;
            if let Some(packed) = self.packed.as_ref().filter(|p| {
                inter::backend_tracks_ends(self.backend)
                    && ends_fit_i16
                    && inter::fill_ends_available(p)
            }) {
                inter::fill_ends(
                    self.backend,
                    packed,
                    self.mode,
                    self.scoring.gap_open(),
                    self.scoring.gap_ext(),
                    query,
                    &mut scratch.simd,
                );
                let (cols, rows) = scratch.simd.ends();
                // Scores land in the database's resolved width; positions are always `i16`. Read
                // per index (no allocation) from the matching width's buffer.
                let i8_width = self.width == ScoreWidth::I8;
                for index in 0..self.sequences.len() {
                    let score = if i8_width {
                        scratch.simd.scores()[index] as i32
                    } else {
                        scratch.simd.scores16()[index] as i32
                    };
                    // Position domain stores DP grid indices; `end = grid - 1`, `None` at grid 0
                    // (nothing aligned), matching the scalar oracle's `checked_sub(1)`.
                    out.push(BestHit {
                        score,
                        db_index: index,
                        query_end: (rows[index] as usize).checked_sub(1),
                        target_end: (cols[index] as usize).checked_sub(1),
                    });
                }
            } else {
                for (index, seq) in self.sequences.iter().enumerate() {
                    let (score, query_end, target_end) =
                        align_core(query, seq, &self.scoring, self.mode, &mut scratch.buf);
                    out.push(BestHit {
                        score,
                        db_index: index,
                        query_end,
                        target_end,
                    });
                }
            }
        } else if let Some(packed) = &self.packed {
            inter::fill_scores(
                self.backend,
                packed,
                self.mode,
                self.scoring.gap_open(),
                self.scoring.gap_ext(),
                query,
                &mut scratch.simd,
            );
            // The per-target scores land in the width the database resolved to.
            let push = |out: &mut Vec<BestHit>, index: usize, score: i32| {
                out.push(BestHit {
                    score,
                    db_index: index,
                    query_end: None,
                    target_end: None,
                });
            };
            match self.width {
                ScoreWidth::I8 => {
                    for (index, &score) in scratch.simd.scores().iter().enumerate() {
                        push(out, index, score as i32);
                    }
                }
                ScoreWidth::I16 => {
                    for (index, &score) in scratch.simd.scores16().iter().enumerate() {
                        push(out, index, score as i32);
                    }
                }
                ScoreWidth::I32 => {
                    for (index, &score) in scratch.simd.scores32().iter().enumerate() {
                        push(out, index, score);
                    }
                }
            }
        } else {
            for (index, seq) in self.sequences.iter().enumerate() {
                let (score, _, _) =
                    align_core(query, seq, &self.scoring, self.mode, &mut scratch.buf);
                out.push(BestHit {
                    score,
                    db_index: index,
                    query_end: None,
                    target_end: None,
                });
            }
        }
    }

    /// Scan `query` against every sequence and write each sequence's best **score** (in `db_index`
    /// order) into `out` (cleared and reused). This is the lightweight counterpart to
    /// [`scan_all`](Self::scan_all): the same per-sequence score array the single-best
    /// [`scan`](Self::scan) computes and then discards, but returned in full and without building a
    /// [`BestHit`] per sequence.
    ///
    /// Useful when a caller needs *all* the scores rather than just the winner — e.g. a demultiplexer
    /// choosing the best-matching barcode and gating on the margin to the second-best, or any
    /// tie/ambiguity handling. The array is directly sortable; find the best and second-best with a
    /// single pass. Scores are bit-identical across every backend (the same per-target kernel
    /// [`scan_all`](Self::scan_all) uses), independent of the database's [`SearchType`] (end
    /// positions, if any, are not computed here).
    ///
    /// Allocation-free apart from growing `out`. Same caller contract as [`scan`](Self::scan):
    /// `query` is pre-encoded and `<= max_query_len`.
    pub fn scan_scores(&self, scratch: &mut Scratch, query: &[u8], out: &mut Vec<i32>) {
        debug_assert!(
            query.len() <= self.max_query_len,
            "query length {} exceeds declared max_query_len {}",
            query.len(),
            self.max_query_len
        );

        out.clear();
        out.reserve(self.sequences.len());

        if let Some(packed) = &self.packed {
            inter::fill_scores(
                self.backend,
                packed,
                self.mode,
                self.scoring.gap_open(),
                self.scoring.gap_ext(),
                query,
                &mut scratch.simd,
            );
            // Scores land in the width the database resolved to.
            match self.width {
                ScoreWidth::I8 => out.extend(scratch.simd.scores().iter().map(|&s| s as i32)),
                ScoreWidth::I16 => out.extend(scratch.simd.scores16().iter().map(|&s| s as i32)),
                ScoreWidth::I32 => out.extend(scratch.simd.scores32().iter().copied()),
            }
        } else {
            for seq in &self.sequences {
                out.push(align_core(query, seq, &self.scoring, self.mode, &mut scratch.buf).0);
            }
        }
    }

    /// Scan `query` against every sequence and return the full [`Alignment`] of the single best
    /// hit, tagged with its database index.
    ///
    /// The best target is found by the fast (SIMD, where applicable) score pass — same winner and
    /// tie-break as [`scan`](Self::scan) — and only that one target is traced back, so the scalar
    /// traceback runs once rather than per sequence. The database must be built with
    /// [`SearchType::Alignment`]; its `max_bytes` budget was proven sufficient at construction, so
    /// this is infallible.
    ///
    /// Same caller contract as [`scan`](Self::scan): `query` is pre-encoded and `<= max_query_len`.
    #[must_use]
    pub fn scan_aligned(&self, scratch: &mut Scratch, query: &[u8]) -> AlignedHit {
        let max_bytes = self.alignment_budget();
        let best = self.scan(scratch, query);
        let alignment = self.traceback_for(best.db_index, query, max_bytes);
        AlignedHit {
            db_index: best.db_index,
            alignment,
        }
    }

    /// Scan `query` against every sequence and write **one [`Alignment`] per database sequence**
    /// into `out`, in database order (`out[i]` is the alignment against sequence `i`). `out` is
    /// cleared first and reused.
    ///
    /// This is the per-target counterpart to [`scan_aligned`](Self::scan_aligned): it traces back
    /// every sequence. The database must be built with [`SearchType::Alignment`]; the budget was
    /// proven sufficient at construction, so this is infallible.
    ///
    /// Same caller contract as [`scan`](Self::scan): `query` is pre-encoded and `<= max_query_len`.
    pub fn scan_all_aligned(&self, _scratch: &mut Scratch, query: &[u8], out: &mut Vec<Alignment>) {
        let max_bytes = self.alignment_budget();
        out.clear();
        out.reserve(self.sequences.len());
        for index in 0..self.sequences.len() {
            out.push(self.traceback_for(index, query, max_bytes));
        }
    }

    /// The traceback budget, asserting the database was built for alignment.
    fn alignment_budget(&self) -> usize {
        self.search_type.max_bytes().expect(
            "scan_aligned/scan_all_aligned require a database built with SearchType::Alignment",
        )
    }

    /// Trace back `query` against sequence `db_index`. The budget was proven sufficient for the
    /// declared maximum problem size at construction, so this cannot exceed it.
    fn traceback_for(&self, db_index: usize, query: &[u8], max_bytes: usize) -> Alignment {
        debug_assert!(
            query.len() <= self.max_query_len,
            "query length {} exceeds declared max_query_len {}",
            query.len(),
            self.max_query_len
        );
        crate::align::traceback(
            query,
            &self.sequences[db_index],
            &self.scoring,
            self.mode,
            max_bytes,
        )
        .expect("traceback budget proven sufficient at construction")
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

    /// The resolved compute backend. Reflects the [`BackendChoice`] and `HYALITE_BACKEND` override,
    /// falling back to the fastest one this CPU supports.
    #[must_use]
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// The kernel data layout, or `None` for the scalar backend (which does not pack the database).
    #[must_use]
    pub fn layout(&self) -> Option<Layout> {
        self.packed.as_ref().map(Packed::layout)
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
    layout_choice: Option<LayoutChoice>,
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

    /// Override the SIMD kernel [`Layout`]. Defaults to [`LayoutChoice::Auto`], which picks
    /// [`Layout::Precomputed`] for a database small enough to keep its score table cache-resident
    /// and [`Layout::Gathered`] otherwise. Ignored by the scalar backend. Layout affects
    /// performance only, never results.
    #[must_use]
    pub fn layout(mut self, choice: LayoutChoice) -> Self {
        self.layout_choice = Some(choice);
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

        // For an `Alignment` search, prove the traceback budget covers every problem this database
        // can pose — the whole `[0, max_query_len] x [0, max_target_len]` box, including empty
        // queries and empty target sequences. With that proven once, every
        // `scan_aligned`/`scan_all_aligned` call is infallible and within budget — no check in the
        // loop.
        if let Some(max_bytes) = search_type.max_bytes() {
            let needed =
                crate::align::traceback_min_bytes_for_database(max_query_len, max_target_len);
            if needed > max_bytes as u64 {
                return Err(Error::TracebackBudgetExceeded {
                    needed_bytes: needed,
                    budget_bytes: max_bytes,
                });
            }
        }

        // Resolve the backend: an explicit builder choice wins, else HYALITE_BACKEND, else auto.
        let choice = match self.backend_choice {
            Some(choice) => choice,
            None => backend::choice_from_env()?.unwrap_or(BackendChoice::Auto),
        };
        let resolved = backend::resolve(choice)?;

        // A SIMD backend is only *usable* for a SIMD-eligible database (an `i8`/`i16` width the
        // inter-sequence kernel supports, at a layout that fits). If one was auto-selected but the
        // database is ineligible, fall back to scalar; if it was explicitly forced, that is an error.
        let layout_choice = self.layout_choice.unwrap_or_default();
        let plan = resolved.simd_lanes(width).and_then(|lanes| {
            inter::simd_plan(width, alphabet_len, &sequences, lanes, layout_choice)
                .map(|layout| (lanes, layout))
        });
        let backend = if resolved == Backend::Scalar || plan.is_some() {
            resolved
        } else {
            match choice {
                BackendChoice::Force(_) => {
                    return Err(Error::BackendUnavailable { backend: resolved });
                }
                BackendChoice::Auto => Backend::Scalar,
            }
        };

        // Pack the database once (query-independent) at the proven width, in the planned layout.
        let packed = if backend == Backend::Scalar {
            None
        } else {
            let (lanes, layout) = plan.expect("a SIMD backend implies a SIMD plan");
            Some(match width {
                ScoreWidth::I8 => Packed::I8(inter::PackedDb::<i8>::build(
                    &sequences, lanes, layout, &scoring,
                )),
                ScoreWidth::I16 => Packed::I16(inter::PackedDb::<i16>::build(
                    &sequences, lanes, layout, &scoring,
                )),
                ScoreWidth::I32 => Packed::I32(inter::PackedDb::<i32>::build(
                    &sequences, lanes, layout, &scoring,
                )),
            })
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
            packed,
        })
    }
}

/// Per-thread mutable working memory for [`Database::scan`]. Create one per worker thread and
/// reuse it across scans; it holds no reference to the database, so it can outlive individual
/// scan calls but should match the database it was sized for.
#[derive(Debug)]
pub struct Scratch {
    /// Full-matrix scalar DP buffers: used by the scalar scan, and by the SIMD scan to recover the
    /// winner's end positions.
    buf: DpBuffers,
    /// SIMD inter-sequence working memory (empty for a scalar-backend database).
    simd: SimdScratch,
}

impl Scratch {
    /// Allocate scratch pre-sized for `db`, so no scan reallocates.
    #[must_use]
    pub fn new(db: &Database) -> Self {
        let simd = match db.backend().simd_lanes(db.score_width()) {
            Some(lanes) => SimdScratch::new(
                db.sequence_count(),
                db.max_target_len(),
                lanes,
                db.score_width(),
            ),
            None => SimdScratch::empty(),
        };
        Scratch {
            buf: DpBuffers::with_capacity(db.max_query_len(), db.max_target_len()),
            simd,
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
        // A large alphabet (20 > 16) rules out the byte-shuffle Gathered gather, and no other SIMD
        // layout serves an i8 database with alphabet_len > 16, so no SIMD kernel applies. Forcing
        // one must fail loudly rather than silently falling back. (Score widths i8/i16/i32 are all
        // SIMD-eligible now, so width alone no longer makes a database ineligible.)
        let al = 20usize;
        let mut matrix = vec![-1i32; al * al];
        for d in 0..al {
            matrix[d * al + d] = 1; // small entries → proves i8
        }
        let scoring = Scoring::new(al, matrix, 2, 1).unwrap();
        let seq: Vec<u8> = (0..al as u8).collect();
        let result = Database::builder()
            .sequences(&[seq])
            .scoring(scoring)
            .mode(Mode::Nw)
            .max_query_len(al)
            .backend(BackendChoice::Force(Backend::Sse41))
            .build();
        // On a CPU without SSE4.1 the resolver rejects it before the eligibility check; either way
        // the outcome is a build error, never a silent scalar fallback.
        assert!(matches!(result, Err(Error::BackendUnavailable { .. })));
    }

    #[test]
    fn layout_is_reported_and_overridable() {
        use crate::LayoutChoice;

        let scalar = Database::builder()
            .sequences(&[vec![0u8, 1, 2, 3]])
            .scoring(dna_scoring())
            .mode(Mode::Sw)
            .max_query_len(8)
            .backend(BackendChoice::Force(Backend::Scalar))
            .build()
            .unwrap();
        assert_eq!(
            scalar.layout(),
            None,
            "scalar backend does not pack the database"
        );

        let build = |b: Backend, choice: Option<LayoutChoice>| {
            let mut builder = Database::builder()
                .sequences(&[vec![0u8, 1, 2, 3], vec![2u8, 2]])
                .scoring(dna_scoring())
                .mode(Mode::Ov)
                .max_query_len(8)
                .backend(BackendChoice::Force(b));
            if let Some(c) = choice {
                builder = builder.layout(c);
            }
            builder.build().unwrap()
        };

        for b in [Backend::Sse41, Backend::Avx2, Backend::Neon] {
            if !b.is_available() {
                continue;
            }
            // A tiny database auto-selects Precomputed (its score table is cache-resident).
            assert_eq!(
                build(b, None).layout(),
                Some(Layout::Precomputed),
                "{b} auto"
            );
            // Both layouts are forceable and reported back.
            for layout in [Layout::Gathered, Layout::Precomputed] {
                let db = build(b, Some(LayoutChoice::Force(layout)));
                assert_eq!(db.layout(), Some(layout), "{b} forced {layout}");
            }
        }
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
    fn scan_scores_returns_the_per_sequence_array() {
        // Three "barcodes"; a query matching barcode 1 best. scan_scores gives every score in
        // db_index order — the demultiplexer / margin use case.
        let db = db_with(
            &[
                vec![2u8, 2, 2, 2], // GGGG  — poor
                vec![0u8, 1, 2, 3], // ACGT  — exact match to the query
                vec![0u8, 1, 2, 2], // ACGG  — near match
            ],
            Mode::Sw,
            SearchType::Score,
        );
        let mut scratch = Scratch::new(&db);
        let mut scores = vec![999; 7]; // pre-filled: must be cleared, not appended to
        db.scan_scores(&mut scratch, &[0u8, 1, 2, 3], &mut scores);

        // One score per sequence, in order; matches the best hit and align_pair per sequence.
        assert_eq!(scores.len(), 3);
        assert_eq!(scores[1], 8); // exact 4-match
        assert!(scores[1] > scores[2] && scores[2] > scores[0]);
        assert_eq!(db.scan(&mut scratch, &[0u8, 1, 2, 3]).score, scores[1]);

        // Margin between best and second-best (what a demux would gate on).
        let mut sorted = scores.clone();
        sorted.sort_unstable_by(|a, b| b.cmp(a));
        let margin = sorted[0] - sorted[1];
        assert_eq!(sorted[0], 8);
        assert!(margin > 0);

        // Reused across a different-length query with no stale entries.
        db.scan_scores(&mut scratch, &[0u8, 1], &mut scores);
        assert_eq!(scores.len(), 3);
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
