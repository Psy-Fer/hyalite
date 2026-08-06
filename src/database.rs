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
use crate::inter::{self, Layout, LayoutChoice, Packed, PackedGroup, PackedGroups, SimdScratch};
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
    /// The database packed at one uniform width for the inter-sequence kernel; `Some` only for a
    /// SIMD backend that is *not* using per-sequence width escalation (see `groups`).
    packed: Option<Packed>,
    /// Per-sequence-width-escalated packing: `Some` when a SIMD `Score`/`ScoreEnd` database has
    /// sequences spanning more than one proven width, so short sequences run at a narrow width.
    /// Mutually exclusive with `packed`.
    groups: Option<PackedGroups>,
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

        // Per-sequence-width-escalated database: run each width group and take the scalar argmax
        // over the global database index (smallest index on a tie).
        if let Some(groups) = &self.groups {
            return self.scan_grouped(groups, scratch, query);
        }

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

    /// The per-target score of group-local sequence `local`, read from the width-matched buffer that
    /// [`inter::fill_scores`]/[`inter::fill_ends`] just filled for `group`.
    #[inline]
    fn group_score(&self, group: &PackedGroup, simd: &SimdScratch, local: usize) -> i32 {
        match group.packed.width() {
            ScoreWidth::I8 => simd.scores()[local] as i32,
            ScoreWidth::I16 => simd.scores16()[local] as i32,
            ScoreWidth::I32 => simd.scores32()[local],
        }
    }

    /// Single-best scan over a per-sequence-width-escalated database. Runs each width group's
    /// per-target kernel, then a **scalar** argmax over the global `db_index` (smallest index on a
    /// tie) — never a lane-order-dependent reduction, matching every other path.
    fn scan_grouped(&self, groups: &PackedGroups, scratch: &mut Scratch, query: &[u8]) -> BestHit {
        let mut best_score = i32::MIN;
        let mut best_index = 0usize;
        for group in &groups.groups {
            inter::fill_scores(
                self.backend,
                &group.packed,
                self.mode,
                self.scoring.gap_open(),
                self.scoring.gap_ext(),
                query,
                &mut scratch.simd,
            );
            for (local, &global) in group.indices.iter().enumerate() {
                let s = self.group_score(group, &scratch.simd, local);
                // Strictly-greater score, or equal score with a smaller global index, wins.
                if s > best_score || (s == best_score && global < best_index) {
                    best_score = s;
                    best_index = global;
                }
            }
        }

        let (query_end, target_end) = if self.search_type.tracks_end() {
            // Recover the winner's ends with one scalar alignment (bit-identical to the oracle),
            // exactly as the uniform single-best path does.
            let (score, qe, te) = align_core(
                query,
                &self.sequences[best_index],
                &self.scoring,
                self.mode,
                &mut scratch.buf,
            );
            debug_assert_eq!(
                score, best_score,
                "grouped scan disagrees with re-alignment"
            );
            (qe, te)
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

    /// `scan_all` over a per-sequence-width-escalated database: run each width group's per-target
    /// kernel and **scatter** its results into `out` by global `db_index`. Every sequence belongs to
    /// exactly one group, so every `out[i]` is written exactly once.
    fn scan_all_grouped(
        &self,
        groups: &PackedGroups,
        scratch: &mut Scratch,
        query: &[u8],
        out: &mut Vec<BestHit>,
    ) {
        let placeholder = BestHit {
            score: i32::MIN,
            db_index: 0,
            query_end: None,
            target_end: None,
        };
        out.clear();
        out.resize(self.sequences.len(), placeholder);

        let (go, ge) = (self.scoring.gap_open(), self.scoring.gap_ext());
        let tracks_end = self.search_type.tracks_end();
        let ends_fit_i16 =
            self.max_query_len <= i16::MAX as usize && self.max_target_len <= i16::MAX as usize;

        for group in &groups.groups {
            let n = group.indices.len();
            let simd_ends = tracks_end
                && inter::backend_tracks_ends(self.backend)
                && ends_fit_i16
                && inter::fill_ends_available(&group.packed);
            if simd_ends {
                inter::fill_ends(
                    self.backend,
                    &group.packed,
                    self.mode,
                    go,
                    ge,
                    query,
                    &mut scratch.simd,
                );
                for local in 0..n {
                    let global = group.indices[local];
                    let score = self.group_score(group, &scratch.simd, local);
                    let (cols, rows) = scratch.simd.ends();
                    out[global] = BestHit {
                        score,
                        db_index: global,
                        query_end: (rows[local] as usize).checked_sub(1),
                        target_end: (cols[local] as usize).checked_sub(1),
                    };
                }
            } else if tracks_end {
                // No in-vector ends for this group (e.g. positions exceed the `i16` domain): recover
                // ends with the scalar DP per sequence.
                for &global in &group.indices {
                    let (score, qe, te) = align_core(
                        query,
                        &self.sequences[global],
                        &self.scoring,
                        self.mode,
                        &mut scratch.buf,
                    );
                    out[global] = BestHit {
                        score,
                        db_index: global,
                        query_end: qe,
                        target_end: te,
                    };
                }
            } else {
                inter::fill_scores(
                    self.backend,
                    &group.packed,
                    self.mode,
                    go,
                    ge,
                    query,
                    &mut scratch.simd,
                );
                for local in 0..n {
                    let global = group.indices[local];
                    let score = self.group_score(group, &scratch.simd, local);
                    out[global] = BestHit {
                        score,
                        db_index: global,
                        query_end: None,
                        target_end: None,
                    };
                }
            }
        }
    }

    /// `scan_scores` over a per-sequence-width-escalated database: run each group's score kernel and
    /// scatter the per-sequence scores into `out` by global `db_index`.
    fn scan_scores_grouped(
        &self,
        groups: &PackedGroups,
        scratch: &mut Scratch,
        query: &[u8],
        out: &mut Vec<i32>,
    ) {
        out.clear();
        out.resize(self.sequences.len(), 0);
        let (go, ge) = (self.scoring.gap_open(), self.scoring.gap_ext());
        for group in &groups.groups {
            inter::fill_scores(
                self.backend,
                &group.packed,
                self.mode,
                go,
                ge,
                query,
                &mut scratch.simd,
            );
            for (local, &global) in group.indices.iter().enumerate() {
                out[global] = self.group_score(group, &scratch.simd, local);
            }
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

        // Per-sequence-width-escalated database: run each width group and scatter its results into
        // `out` by global `db_index`.
        if let Some(groups) = &self.groups {
            self.scan_all_grouped(groups, scratch, query, out);
            return;
        }

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
                for index in 0..self.sequences.len() {
                    let score = match self.width {
                        ScoreWidth::I8 => scratch.simd.scores()[index] as i32,
                        ScoreWidth::I16 => scratch.simd.scores16()[index] as i32,
                        ScoreWidth::I32 => scratch.simd.scores32()[index],
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

        if let Some(groups) = &self.groups {
            self.scan_scores_grouped(groups, scratch, query, out);
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

    /// The integer width proven sufficient for scores in this database — the **widest** width used.
    ///
    /// A SIMD `Score`/`ScoreEnd` database whose sequences span more than one proven width runs each
    /// sequence at its *own* narrower width (per-sequence escalation; see `DETERMINISM.md` §4), but
    /// this reports the widest of them (the width the longest sequence needs). Results are identical
    /// regardless — every sufficient width yields the same score (§6), so escalation is a performance
    /// choice only.
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
    /// For a per-sequence-width-escalated database this reports the layout of its widest-width group
    /// (all groups share the same layout choice).
    #[must_use]
    pub fn layout(&self) -> Option<Layout> {
        self.packed.as_ref().map(Packed::layout).or_else(|| {
            self.groups
                .as_ref()
                .and_then(|g| g.groups.last())
                .map(|g| g.packed.layout())
        })
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
        let uniform_plan = resolved.simd_lanes(width).and_then(|lanes| {
            inter::simd_plan(width, alphabet_len, &sequences, lanes, layout_choice)
                .map(|layout| (lanes, layout))
        });

        // Per-sequence width escalation (`Score`/`ScoreEnd`): partition the sequences by their *own*
        // proven width so a mixed-length database runs its short sequences at a narrow width (more
        // lanes) instead of the whole database at the single widest width. This is an *alternative*
        // eligibility path: a mixed database whose one uniform widest-width packing is too big for the
        // SIMD layout can still be SIMD-eligible when each narrower group's packing fits. `Some` only
        // for a resolved SIMD backend with sequences spanning >1 width, all groups eligible.
        // `Alignment` keeps the uniform path (its traceback budget is proven once for the whole box).
        let groups = if resolved != Backend::Scalar
            && matches!(search_type, SearchType::Score | SearchType::ScoreEnd)
        {
            build_width_groups(
                &sequences,
                &scoring,
                mode,
                max_query_len,
                alphabet_len,
                resolved,
                layout_choice,
            )
        } else {
            None
        };

        // A SIMD backend is usable if the uniform packing *or* the grouped packing is eligible. If
        // one was auto-selected but neither fits, fall back to scalar; if it was forced, that errors.
        let backend = if resolved == Backend::Scalar || uniform_plan.is_some() || groups.is_some() {
            resolved
        } else {
            match choice {
                BackendChoice::Force(_) => {
                    return Err(Error::BackendUnavailable { backend: resolved });
                }
                BackendChoice::Auto => Backend::Scalar,
            }
        };
        // Grouping needs a SIMD backend; drop it if we fell back to scalar.
        let groups = if backend == Backend::Scalar {
            None
        } else {
            groups
        };

        // Otherwise pack the database once (query-independent) at the single proven width.
        let packed = if backend == Backend::Scalar || groups.is_some() {
            None
        } else {
            let (lanes, layout) = uniform_plan.expect("a SIMD backend implies a SIMD plan");
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
            groups,
        })
    }
}

/// Try to build a per-sequence-width-escalated packing (see [`PackedGroups`]). Returns `Some` only
/// when the sequences span more than one proven width **and** every resulting group is
/// SIMD-eligible; otherwise `None`, and the caller uses the uniform-width packing. `backend` must be
/// a SIMD backend.
fn build_width_groups(
    sequences: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    max_query_len: usize,
    alphabet_len: usize,
    backend: Backend,
    layout_choice: LayoutChoice,
) -> Option<PackedGroups> {
    // Each sequence's OWN proven width. `required_width` is monotone in target length, so a
    // per-sequence width can never exceed the whole-database proof (at `max_target_len`) that already
    // succeeded — hence this is infallible.
    let widths: Vec<ScoreWidth> = sequences
        .iter()
        .map(|s| {
            scoring
                .required_width(mode, max_query_len, s.len())
                .expect("per-sequence width proof cannot exceed the whole-database proof")
        })
        .collect();

    // Only escalate when the sequences genuinely span more than one width.
    let mut present = widths.clone();
    present.sort();
    present.dedup();
    if present.len() < 2 {
        return None;
    }

    let mut groups = Vec::with_capacity(present.len());
    for w in [ScoreWidth::I8, ScoreWidth::I16, ScoreWidth::I32] {
        let indices: Vec<usize> = (0..sequences.len()).filter(|&i| widths[i] == w).collect();
        if indices.is_empty() {
            continue;
        }
        // Group-local order preserves original database order, so `indices` is ascending.
        let group_seqs: Vec<Vec<u8>> = indices.iter().map(|&i| sequences[i].clone()).collect();
        let lanes = backend.simd_lanes(w)?;
        // If any group is SIMD-ineligible (e.g. `i8` with `alphabet_len > 16`), abandon escalation
        // and let the caller use the uniform packing — never a silent correctness change.
        let layout = inter::simd_plan(w, alphabet_len, &group_seqs, lanes, layout_choice)?;
        let packed = match w {
            ScoreWidth::I8 => Packed::I8(inter::PackedDb::<i8>::build(
                &group_seqs,
                lanes,
                layout,
                scoring,
            )),
            ScoreWidth::I16 => Packed::I16(inter::PackedDb::<i16>::build(
                &group_seqs,
                lanes,
                layout,
                scoring,
            )),
            ScoreWidth::I32 => Packed::I32(inter::PackedDb::<i32>::build(
                &group_seqs,
                lanes,
                layout,
                scoring,
            )),
        };
        groups.push(PackedGroup { packed, indices });
    }
    Some(PackedGroups { groups })
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
        let simd = if let Some(groups) = &db.groups {
            // Escalated: size working buffers for every width present across the groups.
            let lanes_for = |w: ScoreWidth| {
                groups
                    .groups
                    .iter()
                    .any(|g| g.packed.width() == w)
                    .then(|| db.backend.simd_lanes(w))
                    .flatten()
            };
            SimdScratch::new_grouped(
                db.sequence_count(),
                db.max_target_len(),
                lanes_for(ScoreWidth::I8),
                lanes_for(ScoreWidth::I16),
                lanes_for(ScoreWidth::I32),
            )
        } else {
            match db.backend().simd_lanes(db.score_width()) {
                Some(lanes) => SimdScratch::new(
                    db.sequence_count(),
                    db.max_target_len(),
                    lanes,
                    db.score_width(),
                ),
                None => SimdScratch::empty(),
            }
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
        // A large alphabet (25 > 16) rules out the byte-shuffle Gathered gather; only the Precomputed
        // layout could serve it, and here its score table is far too big for the cache budget (one
        // 200k-long sequence), so no SIMD kernel applies. Forcing one must fail loudly rather than
        // silently falling back. (Small alphabets, and large alphabets whose Precomputed table fits,
        // are all SIMD-eligible now — width and alphabet size alone no longer make a database
        // ineligible.)
        let al = 25usize;
        let mut matrix = vec![-1i32; al * al];
        for d in 0..al {
            matrix[d * al + d] = 1; // match +1, so short queries keep the score in i8
        }
        let scoring = Scoring::new(al, matrix, 2, 1).unwrap();
        // A very long target: the Precomputed table (alphabet_len × width × lanes) dwarfs the budget.
        let seq: Vec<u8> = (0..200_000u32).map(|i| (i % al as u32) as u8).collect();
        let result = Database::builder()
            .sequences(&[seq])
            .scoring(scoring)
            .mode(Mode::Sw)
            .max_query_len(100) // keeps SW width at i8 (min(100, len) · 1 = 100 < 128)
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

    // --- Per-sequence width escalation --------------------------------------

    /// A `+20` match makes SW prove `i8` for a ~4 nt target (80), `i16` for ~10 nt (200), and `i32`
    /// for ~2000 nt (40000) — so a database mixing those lengths spans three widths.
    fn wide_dna_scoring() -> Scoring {
        let mut mat = vec![-5i32; 16];
        for i in 0..4 {
            mat[i * 4 + i] = 20;
        }
        Scoring::new(4, mat, 8, 2).unwrap()
    }

    /// Two `i8`, two `i16`, one `i32` sequence (by SW width proof at these lengths).
    fn mixed_width_seqs() -> Vec<Vec<u8>> {
        let s = |len: usize| -> Vec<u8> { (0..len).map(|i| (i % 4) as u8).collect() };
        vec![s(4), s(6), s(10), s(24), s(2000)]
    }

    fn available_simd() -> Vec<Backend> {
        [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect()
    }

    #[test]
    fn escalation_partitions_a_mixed_width_database() {
        let scoring = wide_dna_scoring();
        let seqs = mixed_width_seqs();
        // Sanity: the sequences really do span i8/i16/i32 under SW.
        let widths: Vec<ScoreWidth> = seqs
            .iter()
            .map(|s| scoring.required_width(Mode::Sw, 2000, s.len()).unwrap())
            .collect();
        assert_eq!(
            widths,
            vec![
                ScoreWidth::I8,
                ScoreWidth::I8,
                ScoreWidth::I16,
                ScoreWidth::I16,
                ScoreWidth::I32
            ]
        );

        for b in available_simd() {
            let db = Database::builder()
                .sequences(&seqs)
                .scoring(scoring.clone())
                .mode(Mode::Sw)
                .search_type(SearchType::Score)
                .max_query_len(2000)
                .backend(BackendChoice::Force(b))
                .build()
                .unwrap();
            let groups = db
                .groups
                .as_ref()
                .expect("a mixed-width SIMD database must escalate");
            assert!(
                db.packed.is_none(),
                "grouped and uniform are mutually exclusive"
            );
            assert_eq!(groups.groups.len(), 3, "one group per distinct width");
            // Groups are ascending width; indices ascending and covering every sequence exactly once.
            let mut seen: Vec<usize> = groups
                .groups
                .iter()
                .flat_map(|g| g.indices.iter().copied())
                .collect();
            seen.sort();
            assert_eq!(seen, (0..seqs.len()).collect::<Vec<_>>());
            assert_eq!(db.score_width(), ScoreWidth::I32, "widest group reported");
        }
    }

    #[test]
    fn escalated_results_match_the_scalar_oracle() {
        let scoring = wide_dna_scoring();
        let seqs = mixed_width_seqs();
        let simd = available_simd();
        if simd.is_empty() {
            return;
        }
        // Queries of assorted lengths over the alphabet. Kept short: the width proof (hence the
        // grouping) is fixed at build time by `max_query_len`, so a long query adds no coverage but
        // would make the scalar oracle's full-matrix DP against the 2000 nt target very slow.
        let queries: Vec<Vec<u8>> = [3usize, 8, 30, 50]
            .into_iter()
            .map(|len| (0..len).map(|i| ((i * 3 + 1) % 4) as u8).collect())
            .collect();

        for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let build = |b: BackendChoice| {
                    Database::builder()
                        .sequences(&seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(2000)
                        .backend(b)
                        .build()
                        .unwrap()
                };
                let oracle = build(BackendChoice::Force(Backend::Scalar));
                let mut os = Scratch::new(&oracle);
                for &b in &simd {
                    let db = build(BackendChoice::Force(b));
                    let mut gs = Scratch::new(&db);
                    for q in &queries {
                        // single-best
                        assert_eq!(
                            db.scan(&mut gs, q),
                            oracle.scan(&mut os, q),
                            "{b} scan {mode} {st} len={}",
                            q.len()
                        );
                        // per-target
                        let (mut ga, mut oa) = (Vec::new(), Vec::new());
                        db.scan_all(&mut gs, q, &mut ga);
                        oracle.scan_all(&mut os, q, &mut oa);
                        assert_eq!(ga, oa, "{b} scan_all {mode} {st} len={}", q.len());
                        // scores array
                        let (mut gsc, mut osc) = (Vec::new(), Vec::new());
                        db.scan_scores(&mut gs, q, &mut gsc);
                        oracle.scan_scores(&mut os, q, &mut osc);
                        assert_eq!(gsc, osc, "{b} scan_scores {mode} {st} len={}", q.len());
                    }
                }
            }
        }
    }

    use proptest::prelude::*;

    /// Random mixed-width databases: a large match value (`+3000`) makes short and long sequences
    /// prove different widths (`i16` for the short, `i32` for the long), with randomized sequence
    /// contents, queries, mismatch, and gaps. This is the coverage the integration proptest cannot
    /// give: being in-crate it asserts `groups.is_some()` (so grouping is genuinely exercised — the
    /// HIGH-severity non-vacuity gap the review flagged), including an **i32 group**, and checks every
    /// grouped scan (`scan`/`scan_all`/`scan_scores`, `Score`+`ScoreEnd`, all modes) against the
    /// forced-scalar oracle. Complements the single deterministic `escalated_results_match_the_scalar_oracle`.
    fn escalation_case() -> impl Strategy<Value = (i32, i32, i32, Vec<Vec<u8>>, Vec<u8>)> {
        let dna = |len: usize| prop::collection::vec(0u8..4, len);
        // Fixed lengths spanning the i16/i32 boundary at match +3000, mql 16:
        // len·3000 -> 12000/24000 (i16), 36000/48000 (i32).
        let seqs = (dna(4), dna(8), dna(12), dna(16)).prop_map(|(a, b, c, d)| vec![a, b, c, d]);
        let gaps = (0i32..=20).prop_flat_map(|go| (Just(go), 0i32..=go));
        ((-3000i32..=-500), gaps, seqs, dna_query())
            .prop_map(|(mism, (go, ge), seqs, q)| (mism, go, ge, seqs, q))
    }

    fn dna_query() -> impl Strategy<Value = Vec<u8>> {
        prop::collection::vec(0u8..4, 1..=16)
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(120))]
        #[test]
        fn escalation_groups_and_matches_scalar((mism, go, ge, seqs, q) in escalation_case()) {
            let simd = available_simd();
            if simd.is_empty() {
                return Ok(());
            }
            let mut matrix = vec![mism; 16];
            for d in 0..4 {
                matrix[d * 4 + d] = 3000;
            }
            let scoring = Scoring::new(4, matrix, go, ge).unwrap();

            for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    let build = |b: Backend| {
                        Database::builder()
                            .sequences(&seqs)
                            .scoring(scoring.clone())
                            .mode(mode)
                            .search_type(st)
                            .max_query_len(16)
                            .backend(BackendChoice::Force(b))
                            .build()
                            .unwrap()
                    };
                    let oracle = build(Backend::Scalar);
                    let mut os = Scratch::new(&oracle);
                    let want = oracle.scan(&mut os, &q);
                    let (mut wa, mut ws) = (Vec::new(), Vec::new());
                    oracle.scan_all(&mut os, &q, &mut wa);
                    oracle.scan_scores(&mut os, &q, &mut ws);

                    for &b in &simd {
                        let db = build(b);
                        // Non-vacuity: under SW the width is `min(mql,len)·3000`, which spans i16
                        // (len 4/8) and i32 (len 12/16), so grouping MUST trigger with an i32 group.
                        if mode == Mode::Sw {
                            let groups = db
                                .groups
                                .as_ref()
                                .expect("SW mixed-width database must escalate");
                            prop_assert!(
                                groups
                                    .groups
                                    .iter()
                                    .any(|g| g.packed.width() == ScoreWidth::I32),
                                "an i32 group must be present"
                            );
                        }
                        let mut gs = Scratch::new(&db);
                        prop_assert_eq!(db.scan(&mut gs, &q), want, "{} scan {} {}", b, mode, st);
                        let (mut ga, mut gsc) = (Vec::new(), Vec::new());
                        db.scan_all(&mut gs, &q, &mut ga);
                        db.scan_scores(&mut gs, &q, &mut gsc);
                        prop_assert_eq!(&ga, &wa, "{} scan_all {} {}", b, mode, st);
                        prop_assert_eq!(&gsc, &ws, "{} scan_scores {} {}", b, mode, st);
                    }
                }
            }
        }
    }
}
