//! Regression tests derived from known Opal / pyopal / STAR bugs. Each guards a bug class that
//! affected the upstream C++ implementation, confirming hyalite is not susceptible. See the
//! project's Opal/STAR issue analysis for the mapping.

mod common;

use common::{ALL_MODES, all_sequences, brute, brute_nw, identity_matrix, reference_scan};
use hyalite::{
    Backend, BackendChoice, Database, Error, Layout, LayoutChoice, Mode, ScoreWidth, Scoring,
    Scratch, SearchType, align_pair,
};

/// Scan `query` against `seqs` on `backend`+`layout` (skips build errors, e.g. an ineligible
/// database for a SIMD backend).
fn scan_on(
    backend: Backend,
    layout: LayoutChoice,
    seqs: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    q: &[u8],
) -> Option<hyalite::BestHit> {
    let db = Database::builder()
        .sequences(seqs)
        .scoring(scoring.clone())
        .mode(mode)
        .search_type(SearchType::ScoreEnd)
        .max_query_len(q.len().max(1))
        .backend(BackendChoice::Force(backend))
        .layout(layout)
        .build()
        .ok()?;
    let mut scratch = Scratch::new(&db);
    Some(db.scan(&mut scratch, q))
}

/// Assert every available backend × layout agrees with the scalar oracle on this single pair.
fn all_backends_agree(seqs: &[Vec<u8>], scoring: &Scoring, mode: Mode, q: &[u8]) {
    let scalar = scan_on(Backend::Scalar, LayoutChoice::Auto, seqs, scoring, mode, q).unwrap();
    for b in [Backend::Sse41, Backend::Avx2, Backend::Neon] {
        if !b.is_available() {
            continue;
        }
        for layout in [Layout::Gathered, Layout::Precomputed] {
            if let Some(got) = scan_on(b, LayoutChoice::Force(layout), seqs, scoring, mode, q) {
                assert_eq!(got, scalar, "{b}/{layout} disagrees for {mode}, q={q:?}");
            }
        }
    }
}

/// **Opal #33** — `opal -o 1 -e 1 -x 2 -a NW` on `AABB` vs `AABBC` segfaults from an E-matrix
/// underflow with small *equal* gap penalties (not rejected by the #28 `open >= ext` guard). Must
/// produce the correct score, no panic.
#[test]
fn opal_33_small_equal_gap_penalties_nw() {
    let matrix = identity_matrix(3, 2, -2); // A,B,C; match +2, mismatch -2
    let scoring = Scoring::new(3, matrix.clone(), 1, 1).unwrap();
    let q = [0u8, 0, 1, 1]; // AABB
    let t = [0u8, 0, 1, 1, 2]; // AABBC

    let hit = align_pair(&q, &t, &scoring, Mode::Nw, SearchType::ScoreEnd).unwrap();
    assert_eq!(hit.score, brute_nw(&q, &t, &matrix, 3, 1, 1));
    // NW of these lengths proves to i8, so this also exercises the SIMD kernel.
    all_backends_agree(&[t.to_vec()], &scoring, Mode::Nw, &q);
}

/// **Opal #28** — inverted penalties (`gap_open < gap_ext`) are the root of the affine-init
/// breakage; hyalite rejects them at construction rather than misbehaving.
#[test]
fn opal_28_inverted_gap_penalties_rejected() {
    let matrix = identity_matrix(2, 1, -1);
    assert_eq!(
        Scoring::new(2, matrix, 1, 2).unwrap_err(),
        Error::GapOpenLessThanExtend {
            gap_open: 1,
            gap_ext: 2
        }
    );
}

/// **Opal #28 / #33 class** — small and equal gap penalties in every mode. Exhaustively check the
/// affine boundary/init against the independent brute-force oracle: no panic, exact scores.
#[test]
fn small_penalty_affine_init_matches_brute() {
    // All modes over all short pairs (len ≤ 2), every `gap_open >= gap_ext` in {0,1,2}.
    let seqs = all_sequences(2, 2);
    for (go, ge) in [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)] {
        let matrix = identity_matrix(2, 2, -2);
        let scoring = Scoring::new(2, matrix.clone(), go, ge).unwrap();
        for q in &seqs {
            for t in &seqs {
                for mode in ALL_MODES {
                    let got = align_pair(q, t, &scoring, mode, SearchType::Score)
                        .unwrap()
                        .score;
                    let want = brute(mode, q, t, &matrix, 2, go, ge);
                    assert_eq!(got, want, "gap ({go},{ge}) {mode} q={q:?} t={t:?}");
                }
            }
        }
    }
    // Deeper coverage in NW specifically (the mode Opal #33 crashed in), where the oracle is cheap.
    let deep = all_sequences(2, 4);
    for (go, ge) in [(1, 1), (2, 1), (0, 0)] {
        let matrix = identity_matrix(2, 3, -2);
        let scoring = Scoring::new(2, matrix.clone(), go, ge).unwrap();
        for q in &deep {
            for t in &deep {
                let got = align_pair(q, t, &scoring, Mode::Nw, SearchType::Score)
                    .unwrap()
                    .score;
                assert_eq!(
                    got,
                    brute_nw(q, t, &matrix, 2, go, ge),
                    "NW gap ({go},{ge})"
                );
            }
        }
    }
}

/// **pyopal #3 / Opal #27** — non-`SW` modes overflow the score integer far more easily on long
/// sequences. hyalite proves the width up front: it either widens or returns a typed error, never
/// a silently-wrong score.
#[test]
fn non_local_overflow_is_a_typed_error_not_wrong_scores() {
    let scoring = Scoring::new(2, vec![1, -1, -1, 1], 1, 1).unwrap();
    // A global alignment of two ~2-billion-long sequences blows past i32.
    let err = scoring
        .required_width(Mode::Nw, 2_000_000_000, 2_000_000_000)
        .unwrap_err();
    assert!(matches!(err, Error::ScoreRangeTooWide { .. }));
    // Realistic long global alignment escalates to a wider width rather than overflowing i8.
    assert_eq!(
        scoring.required_width(Mode::Nw, 500, 500).unwrap(),
        ScoreWidth::I16
    );
}

/// **pyopal #6** — Opal crashes aligning against an empty database. hyalite rejects it.
#[test]
fn pyopal_6_empty_database_rejected() {
    let scoring = Scoring::new(2, vec![1, -1, -1, 1], 1, 1).unwrap();
    let empty: [Vec<u8>; 0] = [];
    let err = Database::builder()
        .sequences(&empty)
        .scoring(scoring)
        .mode(Mode::Sw)
        .max_query_len(8)
        .build()
        .unwrap_err();
    assert_eq!(err, Error::EmptyDatabase);
}

/// **Opal #10 shape** — Opal returned wrong Smith-Waterman scores under BLOSUM-style protein
/// scoring (large alphabet, gap_open 13 / gap_ext 2). Exercise a protein-scale alphabet (24 > 16,
/// so the scalar path) against the independent oracle; also covers the i16/i32 width escalation.
#[test]
fn protein_scale_scoring_matches_reference() {
    const AL: usize = 24;
    // A varied, deterministic BLOSUM-ish matrix: diagonal positive, off-diagonal mostly negative.
    let mut matrix = vec![0i32; AL * AL];
    for i in 0..AL {
        for j in 0..AL {
            matrix[i * AL + j] = if i == j {
                4 + (i % 3) as i32
            } else {
                -1 - ((i + j) % 4) as i32
            };
        }
    }
    let scoring = Scoring::new(AL, matrix, 13, 2).unwrap();

    let mut state = 0xB105u64;
    let next = |s: &mut u64| {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *s >> 33
    };
    let protein = |len: usize, s: &mut u64| -> Vec<u8> {
        (0..len).map(|_| (next(s) % AL as u64) as u8).collect()
    };

    let seqs: Vec<Vec<u8>> = (0..12)
        .map(|_| protein(30 + (next(&mut state) % 40) as usize, &mut state))
        .collect();
    let queries: Vec<Vec<u8>> = (0..4)
        .map(|_| protein(40 + (next(&mut state) % 30) as usize, &mut state))
        .collect();
    let max_q = queries.iter().map(Vec::len).max().unwrap();

    for mode in ALL_MODES {
        // Large alphabet (24 > 16) rules out the byte-shuffle Gathered layout, but a proven-`i16`
        // width still runs on SIMD via the Precomputed layout; a proven-`i32` width falls back to
        // scalar. Either way the result must equal best-of-`align_pair`.
        let db = Database::builder()
            .sequences(&seqs)
            .scoring(scoring.clone())
            .mode(mode)
            .search_type(SearchType::ScoreEnd)
            .max_query_len(max_q)
            .build()
            .unwrap();
        if db.score_width() == ScoreWidth::I32 {
            assert_eq!(db.backend(), Backend::Scalar, "i32 width must be scalar");
        }
        let mut scratch = Scratch::new(&db);
        for q in &queries {
            let got = db.scan(&mut scratch, q);
            let want = reference_scan(&seqs, &scoring, mode, SearchType::ScoreEnd, q);
            assert_eq!(got, want, "protein {mode}");
        }
    }
}
