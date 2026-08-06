//! Property-based / differential test harness (M1).
//!
//! `proptest` generates random alphabets, substitution matrices, gap parameters, sequences, and
//! databases and asserts invariants that must hold for *every* input — the class of test that
//! catches an algorithm that passes hand-picked cases but is fragile on the general case.
//!
//! Today the only alignment backend is scalar, so the "differential" axis is scalar-vs-brute
//! (an independent oracle) plus internal-consistency invariants. When the SIMD backends land in
//! M2, the cross-backend comparison drops into [`assert_all_backends_agree`] and every property
//! below immediately covers them too.

mod common;

use common::{ALL_MODES, brute, reference_scan};
use hyalite::{
    Backend, BackendChoice, BestHit, Database, Layout, LayoutChoice, Mode, ScoreWidth, Scoring,
    Scratch, SearchType, align, align_pair, align_pair_span,
};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// Cross-backend hook (scaffold for M2)
// ---------------------------------------------------------------------------

/// Run every currently-available backend and assert they return an identical [`BestHit`],
/// returning that agreed result. In M0/M1 there is only the scalar backend, so this is a single
/// call; the determinism contract is enforced here once SSE4.1/AVX2/NEON are added.
fn assert_all_backends_agree(
    q: &[u8],
    t: &[u8],
    scoring: &Scoring,
    mode: Mode,
    st: SearchType,
) -> BestHit {
    let scalar = align_pair(q, t, scoring, mode, st).unwrap();
    // M2: for backend in forced_backends() { prop_assert_eq!(run(backend, ...), scalar) }
    scalar
}

// ---------------------------------------------------------------------------
// Strategies
// ---------------------------------------------------------------------------

/// `(alphabet_len, matrix, gap_open, gap_ext)` with `gap_open >= gap_ext >= 0` so
/// `Scoring::new` always succeeds. `matrix` entries span both signs.
fn scheme(al_max: usize, entry: std::ops::RangeInclusive<i32>) -> impl Strategy<Value = Scheme> {
    (2usize..=al_max)
        .prop_flat_map(move |al| {
            let mat = prop::collection::vec(entry.clone(), al * al);
            let gaps = (0i32..=10).prop_flat_map(|go| (Just(go), 0i32..=go));
            (Just(al), mat, gaps)
        })
        .prop_map(|(al, matrix, (gap_open, gap_ext))| Scheme {
            al,
            matrix,
            gap_open,
            gap_ext,
        })
}

#[derive(Clone, Debug)]
struct Scheme {
    al: usize,
    matrix: Vec<i32>,
    gap_open: i32,
    gap_ext: i32,
}

impl Scheme {
    fn scoring(&self) -> Scoring {
        Scoring::new(self.al, self.matrix.clone(), self.gap_open, self.gap_ext).unwrap()
    }

    /// Mirror the matrix into a symmetric one (upper triangle wins).
    fn symmetric(mut self) -> Self {
        let al = self.al;
        for i in 0..al {
            for j in (i + 1)..al {
                self.matrix[j * al + i] = self.matrix[i * al + j];
            }
        }
        self
    }
}

fn seq(al: usize, max_len: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec(0u8..al as u8, 0..=max_len)
}

/// A scheme plus a query and target over its alphabet.
fn scheme_and_pair(
    al_max: usize,
    entry: std::ops::RangeInclusive<i32>,
    max_len: usize,
) -> impl Strategy<Value = (Scheme, Vec<u8>, Vec<u8>)> {
    scheme(al_max, entry).prop_flat_map(move |s| {
        let al = s.al;
        (Just(s), seq(al, max_len), seq(al, max_len))
    })
}

// ---------------------------------------------------------------------------
// Differential vs the brute-force oracle (small inputs)
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// Every backend agrees with exhaustive brute-force enumeration, in every mode.
    #[test]
    fn scalar_matches_brute_force((s, q, t) in scheme_and_pair(3, -6..=6, 3)) {
        let scoring = s.scoring();
        for mode in ALL_MODES {
            let hit = assert_all_backends_agree(&q, &t, &scoring, mode, SearchType::Score);
            let expected = brute(mode, &q, &t, &s.matrix, s.al, s.gap_open, s.gap_ext);
            prop_assert_eq!(hit.score, expected, "mode {}, q={:?}, t={:?}", mode, q, t);
        }
    }
}

// ---------------------------------------------------------------------------
// Internal-consistency invariants (larger inputs; no brute force needed)
// ---------------------------------------------------------------------------

proptest! {
    /// `Score` and `ScoreEnd` must agree on the score itself.
    #[test]
    fn score_and_score_end_agree((s, q, t) in scheme_and_pair(4, -8..=8, 25)) {
        let scoring = s.scoring();
        for mode in ALL_MODES {
            let a = align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap().score;
            let b = align_pair(&q, &t, &scoring, mode, SearchType::ScoreEnd).unwrap().score;
            prop_assert_eq!(a, b, "mode {}", mode);
        }
    }

    /// Freeing more end gaps can only raise the score, and local clamps at zero:
    /// `SW >= OV >= HW >= NW` and `SW >= 0`.
    #[test]
    fn mode_score_ordering((s, q, t) in scheme_and_pair(4, -8..=8, 25)) {
        let scoring = s.scoring();
        let score = |mode| align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap().score;
        let (sw, ov, hw, nw) = (score(Mode::Sw), score(Mode::Ov), score(Mode::Hw), score(Mode::Nw));
        prop_assert!(sw >= 0, "SW negative: {}", sw);
        prop_assert!(sw >= ov, "SW {} < OV {}", sw, ov);
        prop_assert!(ov >= hw, "OV {} < HW {}", ov, hw);
        prop_assert!(hw >= nw, "HW {} < NW {}", hw, nw);
    }

    /// With a symmetric substitution matrix, swapping query and target does not change the score
    /// for the symmetric modes (SW, NW, OV). HW is intentionally asymmetric.
    #[test]
    fn symmetric_modes_are_symmetric((s, q, t) in scheme_and_pair(4, -8..=8, 20)) {
        let scoring = s.symmetric().scoring();
        for mode in [Mode::Sw, Mode::Nw, Mode::Ov] {
            let a = align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap().score;
            let b = align_pair(&t, &q, &scoring, mode, SearchType::Score).unwrap().score;
            prop_assert_eq!(a, b, "mode {} not symmetric", mode);
        }
    }
}

// ---------------------------------------------------------------------------
// The width proof, validated against real scores
// ---------------------------------------------------------------------------

proptest! {
    /// The static width proof must never under-provision: the actual score magnitude produced by
    /// the DP must fit inside the range of the width the proof selected.
    #[test]
    fn width_proof_contains_actual_score((s, q, t) in scheme_and_pair(4, -8..=8, 30)) {
        let scoring = s.scoring();
        for mode in ALL_MODES {
            let width = match scoring.required_width(mode, q.len(), t.len()) {
                Ok(w) => w,
                Err(_) => continue, // score range too wide even for i32 (won't happen at these sizes)
            };
            let hit = align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap();
            prop_assert!(
                (hit.score as i64).abs() <= width.max_abs(),
                "mode {}: |score {}| exceeds {} range ({})",
                mode, hit.score, width, width.max_abs()
            );
            // Sanity that the widths are ordered as claimed.
            prop_assert!(matches!(width, ScoreWidth::I8 | ScoreWidth::I16 | ScoreWidth::I32));
        }
    }
}

// ---------------------------------------------------------------------------
// Database scan differential
// ---------------------------------------------------------------------------

/// A scheme plus a small database and a query.
fn scheme_db_query() -> impl Strategy<Value = (Scheme, Vec<Vec<u8>>, Vec<u8>)> {
    scheme(4, -8..=8).prop_flat_map(|s| {
        let al = s.al;
        let db = prop::collection::vec(seq(al, 12), 1..=6);
        (Just(s), db, seq(al, 12))
    })
}

/// Like [`scheme_db_query`] but with wide matrix entries, so many schemes prove to `i16` width and
/// exercise the `i16` inter-sequence kernel. `12`-long perfect matches at `80`/cell reach `960`,
/// well past the `i8` range but inside `i16`; the tests filter to the modes that actually prove
/// `i16` (`i8` is covered by [`scan_identical_across_simd_backends`], `i32` stays scalar).
fn scheme_db_query_wide() -> impl Strategy<Value = (Scheme, Vec<Vec<u8>>, Vec<u8>)> {
    scheme(4, -80..=80).prop_flat_map(|s| {
        let al = s.al;
        let db = prop::collection::vec(seq(al, 12), 1..=6);
        (Just(s), db, seq(al, 12))
    })
}

/// Like [`scheme_db_query_wide`] but with much larger matrix entries, so many schemes prove to
/// `i32` width and exercise the `i32` inter-sequence kernel. `12`-long perfect matches at up to
/// `5000`/cell reach `60000`, past the `i16` range; the tests filter to the modes that actually
/// prove `i32` (`i8`/`i16` are covered above).
fn scheme_db_query_verywide() -> impl Strategy<Value = (Scheme, Vec<Vec<u8>>, Vec<u8>)> {
    scheme(4, -5000..=5000).prop_flat_map(|s| {
        let al = s.al;
        let db = prop::collection::vec(seq(al, 12), 1..=6);
        (Just(s), db, seq(al, 12))
    })
}

/// Moderately wide entries and a spread of **sequence lengths** (1..=40) so a database's sequences
/// often span more than one proven width — exercising per-sequence width escalation (the SIMD
/// backend partitions into width groups; the scalar reference does not).
fn scheme_mixed_length_db() -> impl Strategy<Value = (Scheme, Vec<Vec<u8>>, Vec<u8>)> {
    scheme(4, -40..=70).prop_flat_map(|s| {
        let al = s.al;
        let db = prop::collection::vec(seq(al, 40), 2..=8);
        (Just(s), db, seq(al, 20))
    })
}

/// A **large alphabet** (17..=25, e.g. proteins) that the byte-shuffle Gathered gather cannot serve
/// (`> 16` entries), so the SIMD kernel must use the Precomputed layout. Small entries keep short
/// sequences inside `i8`, so this exercises the large-alphabet path at every width the proof picks.
fn scheme_db_query_large_alphabet() -> impl Strategy<Value = (Scheme, Vec<Vec<u8>>, Vec<u8>)> {
    (17usize..=25)
        .prop_flat_map(|al| {
            let mat = prop::collection::vec(-6i32..=6, al * al);
            let gaps = (0i32..=6).prop_flat_map(|go| (Just(go), 0i32..=go));
            (Just(al), mat, gaps)
        })
        .prop_map(|(al, matrix, (gap_open, gap_ext))| Scheme {
            al,
            matrix,
            gap_open,
            gap_ext,
        })
        .prop_flat_map(|s| {
            let al = s.al;
            let db = prop::collection::vec(seq(al, 14), 1..=6);
            (Just(s), db, seq(al, 14))
        })
}

proptest! {
    /// `Database::scan` equals the best `align_pair` over the database with the smallest-index
    /// tie-break, for random databases, queries, modes, and search types.
    #[test]
    fn scan_matches_reference((s, db_seqs, q) in scheme_db_query()) {
        let scoring = s.scoring();
        for mode in ALL_MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let db = Database::builder()
                    .sequences(&db_seqs)
                    .scoring(scoring.clone())
                    .mode(mode)
                    .search_type(st)
                    .max_query_len(12)
                    .build()
                    .unwrap();
                let mut scratch = Scratch::new(&db);
                let got = db.scan(&mut scratch, &q);
                let want = reference_scan(&db_seqs, &scoring, mode, st, &q);
                prop_assert_eq!(got, want, "mode {}, {}", mode, st);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-backend determinism: forced Scalar vs forced SSE4.1 must be bit-identical
// ---------------------------------------------------------------------------

fn scan_forced(
    backend: Backend,
    layout: LayoutChoice,
    seqs: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    st: SearchType,
    query: &[u8],
) -> BestHit {
    let db = Database::builder()
        .sequences(seqs)
        .scoring(scoring.clone())
        .mode(mode)
        .search_type(st)
        .max_query_len(12)
        .backend(BackendChoice::Force(backend))
        .layout(layout)
        .build()
        .unwrap();
    let mut scratch = Scratch::new(&db);
    db.scan(&mut scratch, query)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// The determinism contract, now testable: for every SIMD-eligible database, each available
    /// SIMD backend (SSE4.1, AVX2) in **either layout** (Gathered, Precomputed) yields the exact
    /// same `BestHit` as scalar — across all modes and both search types. Skipped on CPUs with no
    /// SIMD backend.
    #[test]
    fn scan_identical_across_simd_backends((s, db_seqs, q) in scheme_db_query()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(()); // no SIMD backend on this CPU; nothing to compare
        }
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            // Only i8-width, small-alphabet databases route through the SIMD kernel.
            let width = scoring.required_width(mode, 12, max_t).unwrap();
            if width != ScoreWidth::I8 || scoring.alphabet_len() > 16 {
                continue;
            }
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let scalar = scan_forced(
                    Backend::Scalar, LayoutChoice::Auto, &db_seqs, &scoring, mode, st, &q,
                );
                for &b in &simd {
                    for layout in [Layout::Gathered, Layout::Precomputed] {
                        let got = scan_forced(
                            b, LayoutChoice::Force(layout), &db_seqs, &scoring, mode, st, &q,
                        );
                        prop_assert_eq!(
                            got, scalar, "{}/{} disagrees with scalar for {} {}", b, layout, mode, st
                        );
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// The determinism contract at `i16` width: for every database whose proof selects `i16`, each
    /// available SIMD backend (SSE4.1 → 8 lanes, AVX2 → 16 lanes) yields the exact same `BestHit`
    /// as scalar — across all modes and both search types. Only the Precomputed layout applies at
    /// `i16` (the byte-shuffle Gathered gather is `i8`-specific). This is the regression guard for
    /// the `i16` inter-sequence kernel; its absence let a lane-count/width mismatch reach a build.
    /// Skipped on CPUs with no SIMD backend.
    #[test]
    fn scan_identical_across_simd_backends_i16((s, db_seqs, q) in scheme_db_query_wide()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(()); // no SIMD backend on this CPU; nothing to compare
        }
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            // Target the `i16` kernel specifically; `i8` and `i32` are covered elsewhere.
            if scoring.required_width(mode, 12, max_t).unwrap() != ScoreWidth::I16 {
                continue;
            }
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let scalar = scan_forced(
                    Backend::Scalar, LayoutChoice::Auto, &db_seqs, &scoring, mode, st, &q,
                );
                for &b in &simd {
                    let got = scan_forced(
                        b, LayoutChoice::Force(Layout::Precomputed), &db_seqs, &scoring, mode, st, &q,
                    );
                    prop_assert_eq!(
                        got, scalar, "i16 {}/Precomputed disagrees with scalar for {} {}", b, mode, st
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    /// The determinism contract at `i32` width: for every database whose proof selects `i32`, each
    /// available SIMD backend (SSE4.1 → 4 lanes, AVX2 → 8 lanes) yields the exact same `BestHit` as
    /// scalar — across all modes and both search types. Only the Precomputed layout applies at `i32`
    /// (the byte-shuffle Gathered gather is `i8`-specific). At `i32` the SIMD kernel is
    /// arithmetically identical to the scalar oracle (same `i32::MIN/4` sentinel, same plain
    /// non-saturating ops), so this guards that the new `i32` lane backends, the width-aware packing
    /// stride, and the dispatch/scratch plumbing all line up. Single-best `scan` recovers the
    /// winner's `ScoreEnd` positions via one scalar re-alignment; the in-vector `i32` end kernel is
    /// exercised by the `scan_all` test below. Skipped on CPUs with no SIMD backend.
    #[test]
    fn scan_identical_across_simd_backends_i32((s, db_seqs, q) in scheme_db_query_verywide()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(()); // no SIMD backend on this CPU; nothing to compare
        }
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            // Target the `i32` kernel specifically; `i8` and `i16` are covered above.
            if scoring.required_width(mode, 12, max_t).unwrap() != ScoreWidth::I32 {
                continue;
            }
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let scalar = scan_forced(
                    Backend::Scalar, LayoutChoice::Auto, &db_seqs, &scoring, mode, st, &q,
                );
                for &b in &simd {
                    let got = scan_forced(
                        b, LayoutChoice::Force(Layout::Precomputed), &db_seqs, &scoring, mode, st, &q,
                    );
                    prop_assert_eq!(
                        got, scalar, "i32 {}/Precomputed disagrees with scalar for {} {}", b, mode, st
                    );
                }
            }
        }
    }
}

/// Deterministic guard that the i32 inter-sequence cross-backend path is genuinely exercised — the
/// `scheme_db_query_verywide` proptests reach i32 in ~all cases today, but that is a property of the
/// generator's distribution; if a future tweak stopped reaching i32 they would pass **vacuously**.
/// This pins a known-i32 small-alphabet database (entries ±5000, length 12 → magnitude 60000 > i16)
/// and checks every SIMD backend against scalar for `scan`/`scan_all`, both search types, all modes.
#[test]
fn i32_inter_cross_backend_is_exercised() {
    let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
        .into_iter()
        .filter(|b| b.is_available())
        .collect();
    if simd.is_empty() {
        return;
    }
    let scoring = Scoring::new(4, common::identity_matrix(4, 5000, -3000), 600, 100).unwrap();
    let db_seqs: Vec<Vec<u8>> = (0..6u32)
        .map(|k| (0..12u32).map(|i| ((i + k) % 4) as u8).collect())
        .collect();
    let q: Vec<u8> = (0..12u32).map(|i| ((i * 3 + 1) % 4) as u8).collect();

    for mode in ALL_MODES {
        // Confirm this really is an i32-width database (non-vacuity).
        assert_eq!(
            scoring.required_width(mode, 12, 12),
            Ok(ScoreWidth::I32),
            "{mode} must prove i32 for this guard to be meaningful"
        );
        for st in [SearchType::Score, SearchType::ScoreEnd] {
            let build = |b: Backend| {
                Database::builder()
                    .sequences(&db_seqs)
                    .scoring(scoring.clone())
                    .mode(mode)
                    .search_type(st)
                    .max_query_len(12)
                    .backend(BackendChoice::Force(b))
                    .build()
                    .unwrap()
            };
            let oracle = build(Backend::Scalar);
            let mut os = Scratch::new(&oracle);
            let want = oracle.scan(&mut os, &q);
            let mut wa = Vec::new();
            oracle.scan_all(&mut os, &q, &mut wa);
            for &b in &simd {
                let db = build(b);
                let mut gs = Scratch::new(&db);
                assert_eq!(db.scan(&mut gs, &q), want, "{b} scan {mode} {st}");
                let mut ga = Vec::new();
                db.scan_all(&mut gs, &q, &mut ga);
                assert_eq!(ga, wa, "{b} scan_all {mode} {st}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-target scan_all
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// `scan_all` returns one hit per database sequence (in order), each bit-identical to
    /// `align_pair` for that sequence, on every backend × layout and both search types. And
    /// `scan` equals the smallest-index best of `scan_all`.
    #[test]
    fn scan_all_matches_per_sequence_and_scan((s, db_seqs, q) in scheme_db_query()) {
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                // Independent per-sequence reference and its smallest-index best.
                let want: Vec<BestHit> = db_seqs
                    .iter()
                    .enumerate()
                    .map(|(i, seq)| BestHit {
                        db_index: i,
                        ..align_pair(&q, seq, &scoring, mode, st).unwrap()
                    })
                    .collect();
                let want_best = want
                    .iter()
                    .copied()
                    .reduce(|a, c| if c.score > a.score { c } else { a })
                    .unwrap();

                let eligible = matches!(scoring.required_width(mode, 12, max_t), Ok(ScoreWidth::I8))
                    && scoring.alphabet_len() <= 16;
                let mut configs = vec![(Backend::Scalar, LayoutChoice::Auto)];
                if eligible {
                    for b in [Backend::Sse41, Backend::Avx2] {
                        if b.is_available() {
                            configs.push((b, LayoutChoice::Force(Layout::Gathered)));
                            configs.push((b, LayoutChoice::Force(Layout::Precomputed)));
                        }
                    }
                }

                for (b, layout) in configs {
                    let db = Database::builder()
                        .sequences(&db_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(12)
                        .backend(BackendChoice::Force(b))
                        .layout(layout)
                        .build()
                        .unwrap();
                    let mut scratch = Scratch::new(&db);
                    let mut out = Vec::new();
                    db.scan_all(&mut scratch, &q, &mut out);
                    prop_assert_eq!(&out, &want, "scan_all {} {} {}", b, mode, st);
                    let hit = db.scan(&mut scratch, &q);
                    prop_assert_eq!(hit, want_best, "scan vs best-of-scan_all {} {} {}", b, mode, st);
                    // `scan_scores` returns exactly the per-sequence score array of `scan_all`.
                    let mut scores = Vec::new();
                    db.scan_scores(&mut scratch, &q, &mut scores);
                    let want_scores: Vec<i32> = out.iter().map(|h| h.score).collect();
                    prop_assert_eq!(&scores, &want_scores, "scan_scores {} {} {}", b, mode, st);
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// `scan_all` at `i16` width: one hit per database sequence (in order), each bit-identical to
    /// `align_pair`, on every available SIMD backend (Precomputed layout only) and both search
    /// types — plus `scan` equal to the smallest-index best. The `i16` per-target `fill_scores`
    /// path, distinct from the single-best `scan` path above.
    #[test]
    fn scan_all_matches_per_sequence_and_scan_i16((s, db_seqs, q) in scheme_db_query_wide()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(());
        }
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            if scoring.required_width(mode, 12, max_t).unwrap() != ScoreWidth::I16 {
                continue;
            }
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let want: Vec<BestHit> = db_seqs
                    .iter()
                    .enumerate()
                    .map(|(i, seq)| BestHit {
                        db_index: i,
                        ..align_pair(&q, seq, &scoring, mode, st).unwrap()
                    })
                    .collect();
                let want_best = want
                    .iter()
                    .copied()
                    .reduce(|a, c| if c.score > a.score { c } else { a })
                    .unwrap();

                for &b in &simd {
                    let db = Database::builder()
                        .sequences(&db_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(12)
                        .backend(BackendChoice::Force(b))
                        .layout(LayoutChoice::Force(Layout::Precomputed))
                        .build()
                        .unwrap();
                    let mut scratch = Scratch::new(&db);
                    let mut out = Vec::new();
                    db.scan_all(&mut scratch, &q, &mut out);
                    prop_assert_eq!(&out, &want, "i16 scan_all {} {} {}", b, mode, st);
                    let hit = db.scan(&mut scratch, &q);
                    prop_assert_eq!(hit, want_best, "i16 scan vs best {} {} {}", b, mode, st);
                    let mut scores = Vec::new();
                    db.scan_scores(&mut scratch, &q, &mut scores);
                    let want_scores: Vec<i32> = out.iter().map(|h| h.score).collect();
                    prop_assert_eq!(&scores, &want_scores, "i16 scan_scores {} {} {}", b, mode, st);
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// `scan_all` at `i32` width: one hit per database sequence (in order), each bit-identical to
    /// `align_pair`, on every available SIMD backend (Precomputed layout only) and both search
    /// types — plus `scan` equal to the smallest-index best and `scan_scores` equal to the score
    /// array. Exercises the `i32` per-target `fill_scores` path (distinct from the single-best
    /// `scan` path above) and its `scores32` reduction, and — for `ScoreEnd` — the in-vector `i32`
    /// end kernel (`fill_ends` at `i32`, positions carried as `i32` and narrowed to `i16` on store),
    /// verified against the per-sequence `align_pair` ends. Proven non-vacuous by sabotaging the
    /// `i32` `pos_store`.
    #[test]
    fn scan_all_matches_per_sequence_and_scan_i32((s, db_seqs, q) in scheme_db_query_verywide()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(());
        }
        let scoring = s.scoring();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            if scoring.required_width(mode, 12, max_t).unwrap() != ScoreWidth::I32 {
                continue;
            }
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let want: Vec<BestHit> = db_seqs
                    .iter()
                    .enumerate()
                    .map(|(i, seq)| BestHit {
                        db_index: i,
                        ..align_pair(&q, seq, &scoring, mode, st).unwrap()
                    })
                    .collect();
                let want_best = want
                    .iter()
                    .copied()
                    .reduce(|a, c| if c.score > a.score { c } else { a })
                    .unwrap();

                for &b in &simd {
                    let db = Database::builder()
                        .sequences(&db_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(12)
                        .backend(BackendChoice::Force(b))
                        .layout(LayoutChoice::Force(Layout::Precomputed))
                        .build()
                        .unwrap();
                    let mut scratch = Scratch::new(&db);
                    let mut out = Vec::new();
                    db.scan_all(&mut scratch, &q, &mut out);
                    prop_assert_eq!(&out, &want, "i32 scan_all {} {} {}", b, mode, st);
                    let hit = db.scan(&mut scratch, &q);
                    prop_assert_eq!(hit, want_best, "i32 scan vs best {} {} {}", b, mode, st);
                    let mut scores = Vec::new();
                    db.scan_scores(&mut scratch, &q, &mut scores);
                    let want_scores: Vec<i32> = out.iter().map(|h| h.score).collect();
                    prop_assert_eq!(&scores, &want_scores, "i32 scan_scores {} {} {}", b, mode, st);
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Per-sequence width escalation: on a database whose sequences span multiple proven widths, a
    /// SIMD backend partitions into width groups while forced-scalar does not — and the two must
    /// agree exactly, for `scan`, `scan_all`, and `scan_scores`, in every mode and search type. This
    /// is the cross-backend guard for the grouped scatter/argmax paths (the exact-width and grouping
    /// non-vacuity are pinned by the unit tests in `database.rs`).
    #[test]
    fn escalated_scan_matches_scalar((s, db_seqs, q) in scheme_mixed_length_db()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(());
        }
        let scoring = s.scoring();
        for mode in ALL_MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let build = |b: Backend| {
                    Database::builder()
                        .sequences(&db_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(40)
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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    /// Large-alphabet (protein-scale, 17..=25) databases: the byte-shuffle Gathered gather cannot
    /// index `> 16` entries, so the SIMD kernel uses the Precomputed layout — which serves any
    /// alphabet. Every available SIMD backend must agree with the scalar oracle for `scan`,
    /// `scan_all`, and `scan_scores`, across all modes and both search types. This is the guard that
    /// opened the protein door (previously these databases fell back to scalar). Skipped on CPUs with
    /// no SIMD backend.
    #[test]
    fn large_alphabet_scan_matches_scalar((s, db_seqs, q) in scheme_db_query_large_alphabet()) {
        let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
            .into_iter()
            .filter(|b| b.is_available())
            .collect();
        if simd.is_empty() {
            return Ok(());
        }
        prop_assert!(s.al > 16, "generator must produce a large alphabet");
        let scoring = s.scoring();
        for mode in ALL_MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let build = |b: Backend| {
                    Database::builder()
                        .sequences(&db_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(14)
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
                let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);
                for &b in &simd {
                    let db = build(b);
                    // Large alphabet ⇒ Precomputed (never Gathered), at the proven width (pin the
                    // width so "large-alphabet at i16" doesn't rest on luck of the random scoring).
                    prop_assert_eq!(db.layout(), Some(Layout::Precomputed), "{} {} {}", b, mode, st);
                    prop_assert_eq!(
                        db.score_width(),
                        scoring.required_width(mode, 14, max_t).unwrap(),
                        "{} width {} {}", b, mode, st
                    );
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

/// Large alphabet (20 symbols) at **i32** width: the audit found the large-alphabet SIMD path was
/// covered at i8/i16 but never i32. A big diagonal (`+3000`) over 14-long sequences proves i32
/// (`14·3000 = 42000 > 32767`); the alphabet (> 16) forces Precomputed. Every SIMD backend must
/// match the scalar oracle for `scan`/`scan_all`/`scan_scores`, both search types, all modes.
#[test]
fn large_alphabet_at_i32_matches_scalar() {
    let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
        .into_iter()
        .filter(|b| b.is_available())
        .collect();
    if simd.is_empty() {
        return;
    }
    let al = 20usize;
    let mut matrix = vec![-1000i32; al * al];
    for d in 0..al {
        matrix[d * al + d] = 3000;
    }
    let scoring = Scoring::new(al, matrix, 500, 100).unwrap();
    // Uniform length 14 → uniform i32 (not escalated), so this exercises the i32 large-alphabet
    // Precomputed kernel directly.
    let db_seqs: Vec<Vec<u8>> = (0..5u32)
        .map(|k| (0..14u32).map(|i| ((i + k) % al as u32) as u8).collect())
        .collect();
    let q: Vec<u8> = (0..14u32)
        .map(|i| ((i * 3 + 1) % al as u32) as u8)
        .collect();

    for mode in ALL_MODES {
        for st in [SearchType::Score, SearchType::ScoreEnd] {
            let build = |b: Backend| {
                Database::builder()
                    .sequences(&db_seqs)
                    .scoring(scoring.clone())
                    .mode(mode)
                    .search_type(st)
                    .max_query_len(14)
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
                assert_eq!(db.score_width(), ScoreWidth::I32, "{b} {mode} {st}");
                assert_eq!(db.layout(), Some(Layout::Precomputed), "{b} {mode} {st}");
                let mut gs = Scratch::new(&db);
                assert_eq!(db.scan(&mut gs, &q), want, "{b} scan {mode} {st}");
                let (mut ga, mut gsc) = (Vec::new(), Vec::new());
                db.scan_all(&mut gs, &q, &mut ga);
                db.scan_scores(&mut gs, &q, &mut gsc);
                assert_eq!(ga, wa, "{b} scan_all {mode} {st}");
                assert_eq!(gsc, ws, "{b} scan_scores {mode} {st}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// `align_pair_span` (starts-only local alignment): its score and half-open ends must equal the
    /// full traceback's, and its span must be **self-consistent** — a global (`NW`) alignment of the
    /// reported aligned substrings scores exactly `span.score`. That consistency oracle is what
    /// guarantees the returned `(start, end)` bound a single genuine optimal local alignment (not two
    /// halves of different equal-scoring ones). Starts also match the traceback here; on ties the
    /// contract only promises consistency, so if that ever diverges this asserts the weaker invariant.
    #[test]
    fn align_pair_span_is_consistent_and_matches_traceback((s, q, t) in scheme_and_pair(4, -8..=8, 30)) {
        let scoring = s.scoring();
        let span = align_pair_span(&q, &t, &scoring).unwrap();
        let full = align(&q, &t, &scoring, Mode::Sw, usize::MAX).unwrap();

        // Score always matches the traceback (and hence align_pair SW).
        prop_assert_eq!(span.score, full.score, "score");

        if span.score == 0 {
            // Empty alignment: start == end on both axes.
            prop_assert_eq!(span.query_start, span.query_end);
            prop_assert_eq!(span.target_start, span.target_end);
            return Ok(());
        }

        // Ends match the traceback (both half-open).
        prop_assert_eq!(span.query_end, full.query_end, "query_end");
        prop_assert_eq!(span.target_end, full.target_end, "target_end");

        // The GUARANTEED contract on the start: the span bounds a real optimal alignment — a global
        // (NW) alignment of the aligned substrings recovers the score exactly. This is the
        // independent oracle for `(start, end)` consistency and holds even on ties (where the *exact*
        // start `align_pair_span` picks may legitimately differ from the traceback's). Exact
        // start-equality with the traceback is verified separately, and only where it is a theorem —
        // over *every* short pair — by `align_pair_span_equals_traceback_exhaustive`.
        let qsub = &q[span.query_start..span.query_end];
        let tsub = &t[span.target_start..span.target_end];
        let nw = align_pair(qsub, tsub, &scoring, Mode::Nw, SearchType::Score)
            .unwrap()
            .score;
        prop_assert_eq!(nw, span.score, "span is not a consistent optimal alignment");
    }
}

#[test]
fn align_pair_span_hand_cases() {
    // match +2, mismatch -1, gap 2/1 over {A,C,G,T}.
    let scoring = Scoring::new(4, common::identity_matrix(4, 2, -1), 2, 1).unwrap();
    // Query aligns to an interior window of the target: target = X X q q q X X.
    let q = vec![0u8, 1, 2];
    let t = vec![3u8, 3, 0, 1, 2, 3, 3];
    let span = align_pair_span(&q, &t, &scoring).unwrap();
    assert_eq!(span.score, 6, "3 matches at +2");
    assert_eq!(
        (span.query_start, span.query_end),
        (0, 3),
        "whole query, half-open"
    );
    assert_eq!(
        (span.target_start, span.target_end),
        (2, 5),
        "interior window [2, 5)"
    );

    // Nothing aligns above zero (all-mismatch, steep gaps) ⇒ the empty span.
    let neg = Scoring::new(4, common::identity_matrix(4, 1, -9), 9, 9).unwrap();
    let empty = align_pair_span(&[0u8, 0], &[1u8, 1], &neg).unwrap();
    assert_eq!(empty.score, 0);
    assert_eq!(empty.query_start, empty.query_end);
    assert_eq!(empty.target_start, empty.target_end);

    // Empty inputs are always the empty span.
    assert_eq!(align_pair_span(&[], &t, &scoring).unwrap().score, 0);
    assert_eq!(align_pair_span(&q, &[], &scoring).unwrap().score, 0);

    // i32-width scores: the span DP computes in i32, so a large-magnitude case must still produce
    // the correct score and a self-consistent span. match +3000 over a 5-long exact match proves
    // i32 (5·3000 = 15000... i16; use a longer run). A 20-long exact match reaches 60000 > i16.
    let big = Scoring::new(4, common::identity_matrix(4, 3000, -1000), 500, 100).unwrap();
    let r: Vec<u8> = (0..20u8).map(|i| i % 4).collect();
    assert_eq!(
        big.required_width(Mode::Sw, r.len(), r.len()),
        Ok(ScoreWidth::I32)
    );
    let span = align_pair_span(&r, &r, &big).unwrap();
    assert_eq!(span.score, 20 * 3000, "20 matches at +3000");
    assert_eq!((span.query_start, span.query_end), (0, 20));
    assert_eq!((span.target_start, span.target_end), (0, 20));
    // Self-consistency at i32: NW of the aligned substrings recovers the score.
    let nw = align_pair(
        &r[span.query_start..span.query_end],
        &r[span.target_start..span.target_end],
        &big,
        Mode::Nw,
        SearchType::Score,
    )
    .unwrap()
    .score;
    assert_eq!(nw, span.score);
    // And it agrees with the full traceback at i32 width.
    let full = align(&r, &r, &big, Mode::Sw, usize::MAX).unwrap();
    assert_eq!(
        (
            span.score,
            span.query_start,
            span.query_end,
            span.target_start,
            span.target_end
        ),
        (
            full.score,
            full.query_start,
            full.query_end,
            full.target_start,
            full.target_end
        )
    );
}

/// Exhaustive guard for `align_pair_span`: over **every** pair up to length 4 (three-letter
/// alphabet) and several scorings — including a free-extension (`gap_ext = 0`) scheme — the returned
/// span equals the full traceback's exactly (score, both ends, both starts) and is self-consistent.
/// The random property test above widens the length/entry range; this pins the tie-break behaviour
/// on the full short-input space where equal-scoring alignments are most common.
#[test]
fn align_pair_span_equals_traceback_exhaustive() {
    use common::{all_sequences, identity_matrix};
    let scorings = [
        Scoring::new(3, identity_matrix(3, 2, -1), 2, 1).unwrap(),
        Scoring::new(3, identity_matrix(3, 1, -1), 1, 1).unwrap(),
        Scoring::new(3, identity_matrix(3, 3, -2), 4, 1).unwrap(),
        Scoring::new(3, identity_matrix(3, 2, -3), 1, 0).unwrap(), // free gap extension
    ];
    let seqs = all_sequences(3, 4);
    for sc in &scorings {
        for q in &seqs {
            for t in &seqs {
                let span = align_pair_span(q, t, sc).unwrap();
                let full = align(q, t, sc, Mode::Sw, usize::MAX).unwrap();
                assert_eq!(span.score, full.score, "score q={q:?} t={t:?}");
                if span.score == 0 {
                    assert_eq!(span.query_start, span.query_end);
                    assert_eq!(span.target_start, span.target_end);
                    continue;
                }
                assert_eq!(
                    (
                        span.query_start,
                        span.query_end,
                        span.target_start,
                        span.target_end
                    ),
                    (
                        full.query_start,
                        full.query_end,
                        full.target_start,
                        full.target_end
                    ),
                    "span vs traceback q={q:?} t={t:?}"
                );
                // Self-consistency: NW of the aligned substrings recovers the score.
                let nw = align_pair(
                    &q[span.query_start..span.query_end],
                    &t[span.target_start..span.target_end],
                    sc,
                    Mode::Nw,
                    SearchType::Score,
                )
                .unwrap()
                .score;
                assert_eq!(nw, span.score, "inconsistent span q={q:?} t={t:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Adversarial robustness (stable stand-in for cargo-fuzz)
// ---------------------------------------------------------------------------

/// Adversarial scheme: large gap-open penalties, `gap_ext` down to 0 (free extension), and
/// strongly negative mismatches — the gap regimes around Opal's open bugs (#28/#33). `gap_open
/// >= gap_ext` still holds, so construction always succeeds.
fn adversarial_scheme() -> impl Strategy<Value = Scheme> {
    (2usize..=4)
        .prop_flat_map(|al| {
            let mat = prop::collection::vec(-30i32..=5, al * al);
            let gaps = (0i32..=80).prop_flat_map(|go| (Just(go), 0i32..=go));
            (Just(al), mat, gaps)
        })
        .prop_map(|(al, matrix, (gap_open, gap_ext))| Scheme {
            al,
            matrix,
            gap_open,
            gap_ext,
        })
}

/// Low-complexity / homopolymer-biased sequences: runs of a single symbol interleaved with the
/// occasional other symbol stress gap-run handling the way real adapter/polyA data does.
fn low_complexity_seq(al: usize, max_len: usize) -> impl Strategy<Value = Vec<u8>> {
    prop::collection::vec((0u8..al as u8, 1usize..=6), 0..=max_len).prop_map(|runs| {
        runs.into_iter()
            .flat_map(|(sym, run)| std::iter::repeat_n(sym, run))
            .take(40)
            .collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Extreme gap configs and low-complexity inputs must never panic, and every invariant that
    /// held on tame inputs must still hold. A panic here fails the test with a shrunk witness.
    #[test]
    fn adversarial_inputs_hold_all_invariants(
        s in adversarial_scheme(),
        (q, t) in (2usize..=4).prop_flat_map(|al| {
            (low_complexity_seq(al, 40), low_complexity_seq(al, 40))
        })
    ) {
        // Clamp symbols to the scheme's alphabet (the seq strategy used its own al).
        let q: Vec<u8> = q.iter().map(|&x| x % s.al as u8).collect();
        let t: Vec<u8> = t.iter().map(|&x| x % s.al as u8).collect();
        let scoring = s.scoring();

        let score = |mode| align_pair(&q, &t, &scoring, mode, SearchType::ScoreEnd).unwrap();

        // No panic getting here; now the mode ordering must still hold.
        let (sw, ov, hw, nw) = (
            score(Mode::Sw).score,
            score(Mode::Ov).score,
            score(Mode::Hw).score,
            score(Mode::Nw).score,
        );
        prop_assert!(sw >= 0 && sw >= ov && ov >= hw && hw >= nw,
            "ordering broke: SW={} OV={} HW={} NW={}", sw, ov, hw, nw);

        // End positions, when reported, must be in range.
        for mode in ALL_MODES {
            let hit = score(mode);
            if let Some(qe) = hit.query_end { prop_assert!(qe < q.len()); }
            if let Some(te) = hit.target_end { prop_assert!(te < t.len()); }
        }
    }
}
