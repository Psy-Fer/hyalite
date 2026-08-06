#![no_main]
//! Differential fuzz target: the determinism contract, coverage-guided.
//!
//! Each input is decoded into a scoring scheme + a small database + a query, then:
//!   * every available SIMD backend's `scan` / `scan_all` / `scan_scores` is asserted **bit-identical**
//!     to the scalar oracle (the inter-sequence kernel, all widths + per-sequence escalation);
//!   * `align_pair` (the striped intra-sequence SIMD kernel) is asserted equal to a forced-scalar
//!     single-target database scan (the striped-vs-scalar differential);
//!   * `align_pair_span` is asserted self-consistent (NW of the aligned substrings recovers the score).
//!
//! The fuzzing profile runs with `overflow-checks = true`, so any wrapping arithmetic in the
//! non-saturating `i32` model crashes here rather than silently diverging. Magnitude *regimes*
//! deliberately reach the `i32` sentinel boundary, where the reviewed overflow bugs lived.
//!
//! Run: `cargo +nightly fuzz run differential`.

use arbitrary::{Result, Unstructured};
use hyalite::{
    Backend, BackendChoice, Database, Mode, Scoring, Scratch, SearchType, align_pair,
    align_pair_span,
};
use libfuzzer_sys::fuzz_target;

const MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];

struct Case {
    scoring: Scoring,
    mode: Mode,
    search_type: SearchType,
    targets: Vec<Vec<u8>>,
    query: Vec<u8>,
}

/// Decode a structured case from raw bytes. Returns `Err` (⇒ the target simply skips the input) when
/// bytes run out or the scoring is rejected — never a panic.
fn decode(u: &mut Unstructured) -> Result<Case> {
    let al = u.int_in_range(2u8..=6)? as usize;

    // Magnitude regime: bias exploration across tame → i32-boundary scores so the fuzzer spends real
    // time near the sentinel ceiling instead of overwhelmingly hitting `ScoreRangeTooWide`.
    let (lo, hi): (i32, i32) = match u.int_in_range(0u8..=3)? {
        0 => (-8, 8),
        1 => (-1_000, 1_000),
        2 => (-100_000, 100_000),
        _ => (-600_000_000, 600_000_000), // straddles the i32 sentinel ceiling (|i32::MIN/4|-1)
    };

    let mut matrix = vec![0i32; al * al];
    for cell in &mut matrix {
        *cell = u.int_in_range(lo..=hi)?;
    }
    // `gap_open >= gap_ext >= 0` (Scoring::new requires it). Reuse the regime's upper bound.
    let go_hi = hi.max(0);
    let gap_open = u.int_in_range(0..=go_hi)?;
    let gap_ext = u.int_in_range(0..=gap_open)?;

    let scoring = Scoring::new(al, matrix, gap_open, gap_ext).map_err(|_| arbitrary::Error::IncorrectFormat)?;

    let mode = MODES[u.int_in_range(0u8..=4)? as usize];
    let search_type = if u.arbitrary()? {
        SearchType::Score
    } else {
        SearchType::ScoreEnd
    };

    let sym = |u: &mut Unstructured| -> Result<u8> { Ok(u.int_in_range(0u8..=(al as u8 - 1))?) };
    let seq = |u: &mut Unstructured| -> Result<Vec<u8>> {
        let len = u.int_in_range(0usize..=24)?;
        (0..len).map(|_| sym(u)).collect()
    };

    let n_targets = u.int_in_range(1usize..=5)?;
    let targets = (0..n_targets).map(|_| seq(u)).collect::<Result<Vec<_>>>()?;
    let query = seq(u)?;

    Ok(Case {
        scoring,
        mode,
        search_type,
        targets,
        query,
    })
}

fn available_simd() -> Vec<Backend> {
    [Backend::Sse41, Backend::Avx2, Backend::Neon]
        .into_iter()
        .filter(|b| b.is_available())
        .collect()
}

fn build(
    backend: Backend,
    targets: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    st: SearchType,
    max_query_len: usize,
) -> Option<Database> {
    Database::builder()
        .sequences(targets)
        .scoring(scoring.clone())
        .mode(mode)
        .search_type(st)
        .max_query_len(max_query_len)
        .backend(BackendChoice::Force(backend))
        .build()
        .ok()
}

fn run(case: Case) {
    let Case {
        scoring,
        mode,
        search_type,
        targets,
        query,
    } = case;
    let mql = query.len().max(1);

    // --- Inter-sequence: every SIMD backend must equal the scalar oracle -------------------------
    // The width proof runs before backend selection, so if the scalar build is rejected
    // (ScoreRangeTooWide) every backend is too — nothing to compare.
    if let Some(scalar_db) = build(Backend::Scalar, &targets, &scoring, mode, search_type, mql) {
        let mut ss = Scratch::new(&scalar_db);
        let want_scan = scalar_db.scan(&mut ss, &query);
        let mut want_all = Vec::new();
        scalar_db.scan_all(&mut ss, &query, &mut want_all);
        let mut want_scores = Vec::new();
        scalar_db.scan_scores(&mut ss, &query, &mut want_scores);

        for b in available_simd() {
            // A forced SIMD backend that is ineligible for this database errors at build; that is
            // allowed (not a determinism violation), so skip it rather than assert.
            let Some(db) = build(b, &targets, &scoring, mode, search_type, mql) else {
                continue;
            };
            let mut s = Scratch::new(&db);
            assert_eq!(db.scan(&mut s, &query), want_scan, "{b} scan {mode} {search_type}");
            let mut all = Vec::new();
            db.scan_all(&mut s, &query, &mut all);
            assert_eq!(all, want_all, "{b} scan_all {mode} {search_type}");
            let mut scores = Vec::new();
            db.scan_scores(&mut s, &query, &mut scores);
            assert_eq!(scores, want_scores, "{b} scan_scores {mode} {search_type}");
        }
    }

    // --- Pairwise: striped `align_pair` vs a forced-scalar single-target database ----------------
    for t in &targets {
        let got = align_pair(&query, t, &scoring, mode, search_type);
        // Oracle: the same pair as a one-target scalar database (uses the scalar DP, not striped).
        let oracle = build(Backend::Scalar, std::slice::from_ref(t), &scoring, mode, search_type, mql)
            .map(|db| db.scan(&mut Scratch::new(&db), &query));
        // Only compare when both entry points built. They prove width from the same scoring but not
        // always the same lengths: `align_pair` uses the *exact* query length, while the oracle DB
        // uses `max_query_len = query.len().max(1)` — so for an empty query the DB proves a slightly
        // *wider* bound and may reject (`ScoreRangeTooWide`) a case `align_pair` accepts. That
        // asymmetry is a property of the harness, not a determinism violation, so any non-(Ok, Ok)
        // combination is simply skipped.
        if let (Ok(a), Some(b)) = (got, oracle) {
            assert_eq!(a, b, "align_pair vs scalar-db {mode} {search_type}");
        }

        // `align_pair_span` (SW local span): always self-consistent — a global (NW) alignment of the
        // reported substrings recovers the score.
        if let Ok(span) = align_pair_span(&query, t, &scoring) {
            if span.score > 0 {
                let qsub = &query[span.query_start..span.query_end];
                let tsub = &t[span.target_start..span.target_end];
                if let Ok(nw) = align_pair(qsub, tsub, &scoring, Mode::Nw, SearchType::Score) {
                    assert_eq!(nw.score, span.score, "align_pair_span not self-consistent");
                }
            }
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(case) = decode(&mut u) {
        run(case);
    }
});
