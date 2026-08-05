//! Asymmetric gap penalties: a gap in the query (`E`, a deletion) and a gap in the target (`F`,
//! an insertion) charged independently.
//!
//! The properties worth pinning down are:
//!
//! 1. **Compatibility** — a scheme whose two directions agree is indistinguishable from the one
//!    `Scoring::new` builds, on every mode, search type, and backend.
//! 2. **Correctness** — the DP agrees with the brute-force path-enumeration oracle in
//!    `tests/common`, which charges the two directions independently by construction.
//! 3. **Transposition** — swapping query and target, the substitution matrix, *and* the two
//!    penalty pairs must leave the score unchanged. This is the invariant a direction mix-up
//!    breaks and a symmetric test can never see.
//! 4. **Determinism** — every backend still returns bit-identical results.

mod common;

use common::{ALL_MODES, Gaps, brute_asym, identity_matrix, reference_scan};
use hyalite::{
    AlignOp, Alignment, Backend, BackendChoice, Database, Layout, LayoutChoice, Mode, ScoreWidth,
    Scoring, Scratch, SearchType, align, align_pair, align_pair_position_max, align_pairs,
};
use proptest::prelude::*;

/// bwa's default mate-rescue scheme translated into hyalite's convention: `-O 6,7 -E 1,2` with
/// deletions (query-gaps) cheaper to open and extend than insertions (target-gaps).
fn bwa_defaults() -> Gaps {
    Gaps {
        query_open: 7,  // o_del + e_del
        query_ext: 1,   // e_del
        target_open: 9, // o_ins + e_ins
        target_ext: 2,  // e_ins
    }
}

fn dna_matrix() -> Vec<i32> {
    identity_matrix(4, 2, -3)
}

/// Re-score an alignment from its ops, charging each maximal gap run in its own direction.
/// Structurally unlike the DP, so agreement is meaningful.
fn rescore(a: &Alignment, q: &[u8], t: &[u8], s: &Scoring) -> i32 {
    let gap = |open: i32, ext: i32, len: i32| open + (len - 1) * ext;
    let mut qi = a.query_start;
    let mut tj = a.target_start;
    let mut total = 0i32;
    let mut k = 0;
    while k < a.ops.len() {
        match a.ops[k] {
            AlignOp::Match | AlignOp::Mismatch => {
                total += s.score(q[qi] as usize, t[tj] as usize);
                qi += 1;
                tj += 1;
                k += 1;
            }
            // `Ins` consumes query only: a gap in the target. `Del` consumes target only: a gap
            // in the query.
            AlignOp::Ins => {
                let mut len = 0;
                while k < a.ops.len() && matches!(a.ops[k], AlignOp::Ins) {
                    len += 1;
                    qi += 1;
                    k += 1;
                }
                total -= gap(s.target_gap_open(), s.target_gap_ext(), len);
            }
            AlignOp::Del => {
                let mut len = 0;
                while k < a.ops.len() && matches!(a.ops[k], AlignOp::Del) {
                    len += 1;
                    tj += 1;
                    k += 1;
                }
                total -= gap(s.query_gap_open(), s.query_gap_ext(), len);
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------
// Construction and validation
// ---------------------------------------------------------------------------

#[test]
fn equal_pairs_build_exactly_what_scoring_new_builds() {
    let sym = Scoring::new(4, dna_matrix(), 5, 2).unwrap();
    let asym = Scoring::new_asymmetric(4, dna_matrix(), 5, 2, 5, 2).unwrap();
    assert_eq!(sym, asym, "equal directions must be the symmetric scheme");
    assert!(asym.has_symmetric_gaps());
}

#[test]
fn accessors_report_each_direction() {
    let s = Scoring::new_asymmetric(4, dna_matrix(), 7, 1, 9, 2).unwrap();
    assert_eq!((s.query_gap_open(), s.query_gap_ext()), (7, 1));
    assert_eq!((s.target_gap_open(), s.target_gap_ext()), (9, 2));
    assert!(!s.has_symmetric_gaps());
}

#[test]
fn each_direction_is_validated_on_its_own() {
    use hyalite::Error;
    // A negative penalty in either direction is rejected.
    for (qo, qe, to, te) in [(-1, 0, 2, 1), (2, 1, 3, -1)] {
        assert!(matches!(
            Scoring::new_asymmetric(4, dna_matrix(), qo, qe, to, te),
            Err(Error::NegativeGapPenalty { .. })
        ));
    }
    // So is `open < ext`, whichever direction carries it. The reported pair is the offending one.
    assert_eq!(
        Scoring::new_asymmetric(4, dna_matrix(), 1, 4, 5, 2).unwrap_err(),
        Error::GapOpenLessThanExtend {
            gap_open: 1,
            gap_ext: 4
        }
    );
    assert_eq!(
        Scoring::new_asymmetric(4, dna_matrix(), 5, 2, 1, 4).unwrap_err(),
        Error::GapOpenLessThanExtend {
            gap_open: 1,
            gap_ext: 4
        }
    );
    // Linear gaps (open == ext) remain legal per direction.
    assert!(Scoring::new_asymmetric(4, dna_matrix(), 3, 3, 5, 1).is_ok());
}

// ---------------------------------------------------------------------------
// Hand-checked cases
// ---------------------------------------------------------------------------

#[test]
fn direction_prices_pick_the_cheaper_gap() {
    // Query and target differ by one base, in opposite directions: aligning ACGT to ACT needs a
    // gap in the target's copy of `G`... in one orientation, and the reverse in the other.
    let q = [0u8, 1, 2, 3]; // ACGT
    let t = [0u8, 1, 3]; // ACT  -> the extra query base is an insertion (target-gap)
    let matrix = dna_matrix();

    // Cheap insertions, dear deletions.
    let cheap_ins = Scoring::new_asymmetric(4, matrix.clone(), 20, 20, 1, 1).unwrap();
    // Cheap deletions, dear insertions.
    let cheap_del = Scoring::new_asymmetric(4, matrix.clone(), 1, 1, 20, 20).unwrap();

    let a = align_pair(&q, &t, &cheap_ins, Mode::Nw, SearchType::Score)
        .unwrap()
        .score;
    let b = align_pair(&q, &t, &cheap_del, Mode::Nw, SearchType::Score)
        .unwrap()
        .score;
    // The only way to align these globally spends exactly one target-gap, so the scheme that
    // prices target-gaps low must score higher.
    assert!(
        a > b,
        "cheap target-gaps should win here: cheap_ins={a}, cheap_del={b}"
    );
    assert_eq!(a, 3 * 2 - 1, "3 matches at +2, one target-gap at 1");
    assert_eq!(b, 3 * 2 - 20, "3 matches at +2, one target-gap at 20");

    // The traceback agrees with the score-only kernel and spends the gap it says it does.
    let al = align(&q, &t, &cheap_ins, Mode::Nw, usize::MAX).unwrap();
    assert_eq!(al.score, a);
    assert_eq!(al.ops.iter().filter(|o| **o == AlignOp::Ins).count(), 1);
    assert_eq!(al.ops.iter().filter(|o| **o == AlignOp::Del).count(), 0);
}

#[test]
fn one_direction_can_be_free_while_the_other_is_not() {
    // A zero-cost query-gap makes HW's leading/trailing target overhang free even in NW.
    let q = [0u8, 1];
    let t = [0u8, 1, 2, 3, 3];
    let s = Scoring::new_asymmetric(4, dna_matrix(), 0, 0, 30, 30).unwrap();
    let nw = align_pair(&q, &t, &s, Mode::Nw, SearchType::Score)
        .unwrap()
        .score;
    assert_eq!(
        nw, 4,
        "two matches at +2; the three extra target bases cost 0"
    );
}

// ---------------------------------------------------------------------------
// Compatibility, correctness, transposition
// ---------------------------------------------------------------------------

/// `(alphabet_len, matrix, gaps)` with each direction independently valid.
fn scheme() -> impl Strategy<Value = (usize, Vec<i32>, Gaps)> {
    (2usize..=4).prop_flat_map(|al| {
        let mat = prop::collection::vec(-6i32..=6, al * al);
        let pair = || (0i32..=8).prop_flat_map(|open| (Just(open), 0i32..=open));
        (Just(al), mat, pair(), pair()).prop_map(|(al, mat, q, t)| {
            (
                al,
                mat,
                Gaps {
                    query_open: q.0,
                    query_ext: q.1,
                    target_open: t.0,
                    target_ext: t.1,
                },
            )
        })
    })
}

fn scheme_and_pair() -> impl Strategy<Value = (usize, Vec<i32>, Gaps, Vec<u8>, Vec<u8>)> {
    scheme().prop_flat_map(|(al, mat, gaps)| {
        let q = prop::collection::vec(0u8..al as u8, 0..=6);
        let t = prop::collection::vec(0u8..al as u8, 0..=6);
        (Just(al), Just(mat), Just(gaps), q, t)
    })
}

/// The transpose of a row-major `al × al` matrix: `m[q * al + t]` becomes `m[t * al + q]`.
fn transpose(matrix: &[i32], al: usize) -> Vec<i32> {
    let mut out = vec![0; al * al];
    for q in 0..al {
        for t in 0..al {
            out[t * al + q] = matrix[q * al + t];
        }
    }
    out
}

/// The mode that plays the same role once query and target are exchanged.
fn transposed_mode(mode: Mode) -> Mode {
    match mode {
        Mode::Hw => Mode::Shw,
        Mode::Shw => Mode::Hw,
        other => other, // SW, NW and OV are symmetric in their two sequences
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(80))]

    /// (2) The DP agrees with the independent brute-force oracle, which charges `Right` moves
    /// (query-gaps) and `Down` moves (target-gaps) from separate counters.
    #[test]
    fn brute_force_oracle_agrees((al, mat, gaps, q, t) in scheme_and_pair()) {
        let scoring = gaps.scoring(al, mat.clone());
        for mode in ALL_MODES {
            let want = brute_asym(mode, &q, &t, &mat, al, gaps);
            let got = align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap().score;
            prop_assert_eq!(got, want, "{} q={:?} t={:?} gaps={:?}", mode, q, t, gaps);
        }
    }

    /// (1) Both directions charged alike is exactly the symmetric scheme, on every mode and
    /// search type.
    #[test]
    fn equal_directions_match_the_symmetric_scheme(
        (al, mat, gaps, q, t) in scheme_and_pair()
    ) {
        let directions = [
            (gaps.query_open, gaps.query_ext),
            (gaps.target_open, gaps.target_ext),
        ];
        for (open, ext) in directions {
            let sym = Scoring::new(al, mat.clone(), open, ext).unwrap();
            let asym = Gaps::symmetric(open, ext).scoring(al, mat.clone());
            for mode in ALL_MODES {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    prop_assert_eq!(
                        align_pair(&q, &t, &asym, mode, st).unwrap(),
                        align_pair(&q, &t, &sym, mode, st).unwrap(),
                        "{} {}", mode, st
                    );
                }
            }
        }
    }

    /// (3) Transposing the pair, the matrix, and the two penalty pairs leaves the score
    /// unchanged. A kernel that charged `E` with the target pair (or vice versa) fails here.
    #[test]
    fn transposition_preserves_the_score((al, mat, gaps, q, t) in scheme_and_pair()) {
        let forward = gaps.scoring(al, mat.clone());
        let backward = gaps.transposed().scoring(al, transpose(&mat, al));
        for mode in ALL_MODES {
            let a = align_pair(&q, &t, &forward, mode, SearchType::Score).unwrap().score;
            let b = align_pair(&t, &q, &backward, transposed_mode(mode), SearchType::Score)
                .unwrap()
                .score;
            prop_assert_eq!(a, b, "{} vs {}", mode, transposed_mode(mode));
        }
    }

    /// The traceback's ops re-score to the reported score, with each gap run charged in its own
    /// direction, and that score is the score-only kernel's.
    #[test]
    fn traceback_ops_rescore((al, mat, gaps, q, t) in scheme_and_pair()) {
        let scoring = gaps.scoring(al, mat);
        for mode in ALL_MODES {
            let a = align(&q, &t, &scoring, mode, usize::MAX).unwrap();
            let want = align_pair(&q, &t, &scoring, mode, SearchType::Score).unwrap().score;
            prop_assert_eq!(a.score, want, "{} traceback vs score kernel", mode);
            prop_assert_eq!(rescore(&a, &q, &t, &scoring), a.score, "{} rescore", mode);
        }
    }

    /// `align_pairs` (one pair per entry) matches `align_pair` per pair under asymmetric gaps.
    #[test]
    fn batched_pairs_match_single_pairs((al, mat, gaps, q, t) in scheme_and_pair()) {
        let scoring = gaps.scoring(al, mat);
        let pairs = [(q.clone(), t.clone()), (t.clone(), q.clone())];
        let mut out = Vec::new();
        for mode in ALL_MODES {
            align_pairs(&pairs, &scoring, mode, SearchType::Score, &mut out).unwrap();
            for (i, (pq, pt)) in pairs.iter().enumerate() {
                let want = align_pair(pq, pt, &scoring, mode, SearchType::Score).unwrap();
                prop_assert_eq!(out[i].score, want.score, "{} pair {}", mode, i);
                prop_assert_eq!(out[i].db_index, i);
            }
        }
    }

    /// The per-target-position maxima (the striped SIMD path when the width allows) match a
    /// naive SW DP that charges the two directions separately.
    #[test]
    fn position_maxima_match_a_naive_dp((al, mat, gaps, q, t) in scheme_and_pair()) {
        let scoring = gaps.scoring(al, mat.clone());
        let mut out = Vec::new();
        align_pair_position_max(&q, &t, &scoring, &mut out).unwrap();

        let (m, n) = (q.len(), t.len());
        let ninf = i32::MIN / 4;
        let mut h = vec![vec![0i32; n + 1]; m + 1];
        let mut e = vec![vec![ninf; n + 1]; m + 1];
        let mut f = vec![vec![ninf; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                e[i][j] = (h[i][j - 1] - gaps.query_open).max(e[i][j - 1] - gaps.query_ext);
                f[i][j] = (h[i - 1][j] - gaps.target_open).max(f[i - 1][j] - gaps.target_ext);
                let sub = mat[q[i - 1] as usize * al + t[j - 1] as usize];
                h[i][j] = (h[i - 1][j - 1] + sub).max(e[i][j]).max(f[i][j]).max(0);
            }
        }
        let want: Vec<i32> = (0..n)
            .map(|j| (0..=m).map(|i| h[i][j + 1]).max().unwrap_or(0).max(0))
            .collect();
        prop_assert_eq!(out, want);
    }
}

// ---------------------------------------------------------------------------
// Determinism across backends and layouts
// ---------------------------------------------------------------------------

fn available_backends() -> Vec<Backend> {
    [
        Backend::Scalar,
        Backend::Sse41,
        Backend::Avx2,
        Backend::Neon,
    ]
    .into_iter()
    .filter(|b| b.is_available())
    .collect()
}

/// Every backend and layout returns the same result as the scalar reference scan, for a database
/// scored with asymmetric gaps, at each score width.
#[test]
fn every_backend_agrees_under_asymmetric_gaps() {
    let scoring = bwa_defaults().scoring(4, dna_matrix());
    // Short sequences prove to i8; the long, highly-divergent ones push the proof to i16/i32.
    let short: Vec<Vec<u8>> = vec![
        vec![0, 1, 2, 3, 0, 1, 2, 3],
        vec![0, 1, 2, 3, 3, 2, 1, 0],
        vec![2, 2, 2, 2],
        vec![],
        vec![0, 0, 1, 1, 2, 2, 3, 3, 0, 1],
    ];
    let long: Vec<Vec<u8>> = (0..9)
        .map(|k| {
            (0..220u32)
                .map(|i| ((i * (k + 1) + i / 7) % 4) as u8)
                .collect()
        })
        .collect();

    // Which backends actually ran, so a build that quietly declined every SIMD combination
    // cannot pass this test as a scalar-only run.
    let mut exercised: Vec<Backend> = Vec::new();

    for seqs in [&short, &long] {
        let query: Vec<u8> = seqs[0].iter().copied().chain([0, 1, 2]).collect();
        for mode in ALL_MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                let want = reference_scan(seqs, &scoring, mode, st, &query);
                for backend in available_backends() {
                    for layout in [LayoutChoice::Auto, LayoutChoice::Force(Layout::Gathered)] {
                        let built = Database::builder()
                            .sequences(seqs)
                            .scoring(scoring.clone())
                            .mode(mode)
                            .search_type(st)
                            .max_query_len(query.len())
                            .backend(BackendChoice::Force(backend))
                            .layout(layout)
                            .build();
                        // A forced SIMD backend that has no eligible packing for this database
                        // declines to build; that is the pre-existing rule, unrelated to gaps.
                        let db = match built {
                            Ok(db) => db,
                            Err(hyalite::Error::BackendUnavailable { .. }) => continue,
                            Err(e) => panic!("{mode} {st} {backend} {layout:?}: {e}"),
                        };
                        if !exercised.contains(&backend) {
                            exercised.push(backend);
                        }
                        let mut scratch = Scratch::new(&db);
                        let got = db.scan(&mut scratch, &query);
                        assert_eq!(got, want, "{mode} {st} {backend} {layout:?}");

                        // Per-target scans must agree with `align_pair` sequence by sequence.
                        let mut all = Vec::new();
                        db.scan_all(&mut scratch, &query, &mut all);
                        for (i, seq) in seqs.iter().enumerate() {
                            let pair = align_pair(&query, seq, &scoring, mode, st).unwrap();
                            assert_eq!(
                                all[i].score, pair.score,
                                "{mode} {st} {backend} {layout:?} seq {i}"
                            );
                        }
                    }
                }
            }
        }
    }

    assert_eq!(
        exercised,
        available_backends(),
        "every available backend must have run at least one asymmetric scan"
    );
}

/// The width proof takes the worse of the two directions, so an expensive direction escalates the
/// width even when the other is cheap — and the result still matches the scalar oracle.
#[test]
fn width_escalates_on_the_expensive_direction() {
    let matrix = identity_matrix(4, 1, -1);
    let q: Vec<u8> = (0..40u32).map(|i| (i % 4) as u8).collect();
    let t: Vec<u8> = (0..40u32).map(|i| ((i + 1) % 4) as u8).collect();

    let cheap = Scoring::new_asymmetric(4, matrix.clone(), 2, 1, 2, 1).unwrap();
    let dear = Scoring::new_asymmetric(4, matrix.clone(), 2, 1, 200, 1).unwrap();
    assert_eq!(
        cheap.required_width(Mode::Sw, q.len(), t.len()).unwrap(),
        ScoreWidth::I8
    );
    assert_eq!(
        dear.required_width(Mode::Sw, q.len(), t.len()).unwrap(),
        ScoreWidth::I16,
        "a 200-point target-gap open drives E/F past the i8 range"
    );

    // And the two schemes really do score differently, so the escalation is not vacuous.
    let a = align_pair(&q, &t, &cheap, Mode::Nw, SearchType::Score)
        .unwrap()
        .score;
    let b = align_pair(&q, &t, &dear, Mode::Nw, SearchType::Score)
        .unwrap()
        .score;
    assert!(a > b, "cheap={a} dear={b}");
}
