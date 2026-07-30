//! `Database` alignment API (`SearchType::Alignment`, `scan_aligned`, `scan_all_aligned`).
//!
//! The traceback itself is exercised exhaustively in the crate's unit tests; here the concern is
//! the database wiring: that `scan_all_aligned` returns exactly the per-target `align()` result
//! for every sequence, that `scan_aligned` picks the same winner as `scan` (its db-index
//! tie-break) and returns that target's alignment, and that all of this is identical on every
//! backend. Plus the build-time budget validation that makes the scans infallible.

mod common;

use common::{ALL_MODES, dna};
use hyalite::{
    Backend, BackendChoice, Database, Error, Mode, ScoreWidth, Scoring, Scratch, SearchType, align,
};
use proptest::prelude::*;

#[allow(clippy::type_complexity)]
fn scheme_db_query() -> impl Strategy<Value = (usize, Vec<i32>, i32, i32, Vec<Vec<u8>>, Vec<u8>)> {
    (2usize..=4).prop_flat_map(|al| {
        let matrix = prop::collection::vec(-5i32..=5, al * al);
        let gaps = (0i32..=8).prop_flat_map(|go| (Just(go), 0i32..=go));
        let db = prop::collection::vec(prop::collection::vec(0u8..al as u8, 0..8), 1..6);
        let q = prop::collection::vec(0u8..al as u8, 0..8);
        (Just(al), matrix, gaps, db, q).prop_map(|(al, m, (go, ge), db, q)| (al, m, go, ge, db, q))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2500))]

    #[test]
    fn database_alignment_matches_free_align(
        (al, matrix, go, ge, db_seqs, q) in scheme_db_query(),
    ) {
        let scoring = Scoring::new(al, matrix, go, ge).unwrap();
        let max_t = db_seqs.iter().map(Vec::len).max().unwrap_or(0);

        for mode in ALL_MODES {
            // Independent per-target reference (scalar free-function align) and its best.
            let want: Vec<_> = db_seqs
                .iter()
                .map(|seq| align(&q, seq, &scoring, mode, usize::MAX).unwrap())
                .collect();
            let best_score = want.iter().map(|a| a.score).max().unwrap();
            let want_idx = want.iter().position(|a| a.score == best_score).unwrap();

            // Scalar always; SIMD backends too when the database is eligible for them.
            let eligible = matches!(scoring.required_width(mode, 12, max_t), Ok(ScoreWidth::I8))
                && scoring.alphabet_len() <= 16;
            let mut backends = vec![Backend::Scalar];
            if eligible {
                for b in [Backend::Sse41, Backend::Avx2] {
                    if b.is_available() {
                        backends.push(b);
                    }
                }
            }

            for b in backends {
                let db = Database::builder()
                    .sequences(&db_seqs)
                    .scoring(scoring.clone())
                    .mode(mode)
                    .search_type(SearchType::Alignment { max_bytes: usize::MAX })
                    .max_query_len(12)
                    .backend(BackendChoice::Force(b))
                    .build()
                    .unwrap();
                let mut scratch = Scratch::new(&db);

                let mut out = Vec::new();
                db.scan_all_aligned(&mut scratch, &q, &mut out);
                prop_assert_eq!(&out, &want, "scan_all_aligned {} {}", b, mode);

                let hit = db.scan_aligned(&mut scratch, &q);
                prop_assert_eq!(hit.db_index, want_idx, "winner {} {}", b, mode);
                prop_assert_eq!(&hit.alignment, &want[want_idx], "scan_aligned {} {}", b, mode);
            }
        }
    }
}

#[test]
fn alignment_budget_is_validated_at_build() {
    let seqs = vec![vec![0u8, 1, 2, 3, 0, 1, 2, 3]];

    // A one-byte budget cannot cover an 8x8 traceback: construction is rejected up front.
    let err = Database::builder()
        .sequences(&seqs)
        .scoring(dna())
        .mode(Mode::Nw)
        .search_type(SearchType::Alignment { max_bytes: 1 })
        .max_query_len(8)
        .build()
        .unwrap_err();
    assert!(
        matches!(err, Error::TracebackBudgetExceeded { .. }),
        "got {err:?}"
    );

    // A generous budget builds and the (infallible) scan produces the alignment.
    let db = Database::builder()
        .sequences(&seqs)
        .scoring(dna())
        .mode(Mode::Nw)
        .search_type(SearchType::Alignment {
            max_bytes: usize::MAX,
        })
        .max_query_len(8)
        .build()
        .unwrap();
    let mut scratch = Scratch::new(&db);
    let hit = db.scan_aligned(&mut scratch, &[0, 1, 2, 3, 0, 1, 2, 3]);
    assert_eq!(hit.db_index, 0);
    assert_eq!(hit.alignment.score, 16); // eight matches at +2
    assert_eq!(hit.alignment.cigar(), "8M");
}

#[test]
fn scan_all_aligned_is_per_target_in_order() {
    // Three targets; the query matches the middle one best.
    let seqs = vec![vec![0u8, 0, 0, 0], vec![0u8, 1, 2, 3], vec![3u8, 3, 3, 3]];
    let db = Database::builder()
        .sequences(&seqs)
        .scoring(dna())
        .mode(Mode::Sw)
        .search_type(SearchType::Alignment {
            max_bytes: usize::MAX,
        })
        .max_query_len(4)
        .build()
        .unwrap();
    let mut scratch = Scratch::new(&db);

    let mut out = Vec::new();
    db.scan_all_aligned(&mut scratch, &[0, 1, 2, 3], &mut out);
    assert_eq!(out.len(), 3);
    assert_eq!(out[1].score, 8); // exact match against target 1
    assert_eq!(out[1].cigar(), "4M");

    let hit = db.scan_aligned(&mut scratch, &[0, 1, 2, 3]);
    assert_eq!(hit.db_index, 1);
    assert_eq!(hit.alignment, out[1]);
}
