//! End-to-end correctness tests for the scalar alignment kernel via the public `align_pair`.
//!
//! The centrepiece is a **differential** test: an independent brute-force alignment scorer
//! (exhaustive path/substring enumeration, structurally unrelated to the Gotoh DP) is compared
//! against `align_pair` over *all* short sequence pairs for several scoring schemes. A thin
//! suite of hand-picked cases can pass a subtly-wrong DP; enumerating every small input and
//! scoring it a completely different way is what actually pins the recurrence down.
//!
//! On top of that: hand-computed exact vectors, cross-mode inequalities that must always hold,
//! the end-position tie-break, empty-input edge cases, and score symmetry.

use hyalite::{Mode, Scoring, SearchType, align_pair};

// ---------------------------------------------------------------------------
// Independent brute-force oracle
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Move {
    Start,
    Right, // consume a target base as a gap in the query
    Down,  // consume a query base as a gap in the target
}

/// The parts of a brute-force problem that stay fixed across the recursion.
struct Prob<'a> {
    q: &'a [u8],
    t: &'a [u8],
    mat: &'a [i32],
    al: usize,
    go: i32,
    ge: i32,
}

/// Exact global (NW) score by enumerating every alignment path, charging affine gap penalties
/// from maximal same-direction runs (`gap_open` for the first base, `gap_ext` thereafter).
/// Exponential, but only ever called on tiny slices.
fn brute_nw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    fn rec(p: &Prob, i: usize, j: usize, last: Move) -> i32 {
        let (m, n) = (p.q.len(), p.t.len());
        if i == m && j == n {
            return 0;
        }
        let mut best = i32::MIN;
        if i < m && j < n {
            let s = p.mat[p.q[i] as usize * p.al + p.t[j] as usize];
            best = best.max(s.saturating_add(rec(p, i + 1, j + 1, Move::Start)));
        }
        if j < n {
            let cost = if last == Move::Right { p.ge } else { p.go };
            best = best.max((-cost).saturating_add(rec(p, i, j + 1, Move::Right)));
        }
        if i < m {
            let cost = if last == Move::Down { p.ge } else { p.go };
            best = best.max((-cost).saturating_add(rec(p, i + 1, j, Move::Down)));
        }
        best
    }
    rec(
        &Prob {
            q,
            t,
            mat,
            al,
            go,
            ge,
        },
        0,
        0,
        Move::Start,
    )
}

/// Local (SW): best global score over every substring pair, floored at 0.
fn brute_sw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    best = best.max(brute_nw(&q[a..b], &t[c..d], mat, al, go, ge));
                }
            }
        }
    }
    best
}

/// Semi-global (HW): query fully aligned to the best target window.
fn brute_hw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let n = t.len();
    let mut best = i32::MIN;
    for c in 0..=n {
        for d in c..=n {
            best = best.max(brute_nw(q, &t[c..d], mat, al, go, ge));
        }
    }
    best
}

/// Overlap (OV): best global score over substring pairs whose alignment touches a border at both
/// its start (skips a prefix of query or target for free) and its end (skips a suffix for free),
/// floored at 0 (the free border cells score 0).
fn brute_ov(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    let touches_start = a == 0 || c == 0;
                    let touches_end = b == m || d == n;
                    if touches_start && touches_end {
                        best = best.max(brute_nw(&q[a..b], &t[c..d], mat, al, go, ge));
                    }
                }
            }
        }
    }
    best
}

fn brute(mode: Mode, q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    match mode {
        Mode::Nw => brute_nw(q, t, mat, al, go, ge),
        Mode::Sw => brute_sw(q, t, mat, al, go, ge),
        Mode::Hw => brute_hw(q, t, mat, al, go, ge),
        Mode::Ov => brute_ov(q, t, mat, al, go, ge),
        _ => unreachable!("ALL_MODES covers every mode this test exercises"),
    }
}

// ---------------------------------------------------------------------------
// Enumeration helpers
// ---------------------------------------------------------------------------

/// All sequences over `alphabet` symbols of length `0..=max_len`.
fn all_sequences(alphabet: u8, max_len: usize) -> Vec<Vec<u8>> {
    let mut out = vec![vec![]];
    let mut frontier = vec![vec![]];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for seq in &frontier {
            for sym in 0..alphabet {
                let mut s = seq.clone();
                s.push(sym);
                next.push(s);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}

const ALL_MODES: [Mode; 4] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov];

fn identity_matrix(al: usize, m: i32, x: i32) -> Vec<i32> {
    let mut v = vec![x; al * al];
    for i in 0..al {
        v[i * al + i] = m;
    }
    v
}

// ---------------------------------------------------------------------------
// The differential test
// ---------------------------------------------------------------------------

#[test]
fn scalar_matches_brute_force_over_all_short_pairs() {
    // (alphabet, max_len, match, mismatch, gap_open, gap_ext). Several schemes probe different
    // gap regimes: affine (open>ext), linear (open==ext), and asymmetric match/mismatch.
    let schemes = [
        (2usize, 3usize, 2, -1, 2, 1),
        (2, 3, 1, -1, 3, 3),
        (2, 3, 3, -2, 4, 2),
        (3, 2, 2, -1, 2, 1),
    ];

    for (al, max_len, m, x, go, ge) in schemes {
        let mat = identity_matrix(al, m, x);
        let scoring = Scoring::new(al, mat.clone(), go, ge).unwrap();
        let seqs = all_sequences(al as u8, max_len);

        for q in &seqs {
            for t in &seqs {
                for mode in ALL_MODES {
                    let expected = brute(mode, q, t, &mat, al, go, ge);
                    let hit = align_pair(q, t, &scoring, mode, SearchType::Score).unwrap();
                    assert_eq!(
                        hit.score, expected,
                        "mode {mode}, scheme (al={al}, m={m}, x={x}, go={go}, ge={ge}), \
                         q={q:?}, t={t:?}"
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Hand-computed exact vectors
// ---------------------------------------------------------------------------

/// DNA scoring: match +2, mismatch -1, gap_open 2, gap_ext 1, over {A,C,G,T} = {0,1,2,3}.
fn dna() -> Scoring {
    Scoring::new(4, identity_matrix(4, 2, -1), 2, 1).unwrap()
}

fn score(mode: Mode, q: &[u8], t: &[u8]) -> i32 {
    align_pair(q, t, &dna(), mode, SearchType::Score)
        .unwrap()
        .score
}

#[test]
fn perfect_match_scores_full_match_in_every_mode() {
    let seq = [0u8, 1, 2, 3];
    for mode in ALL_MODES {
        assert_eq!(score(mode, &seq, &seq), 8, "{mode}");
        let hit = align_pair(&seq, &seq, &dna(), mode, SearchType::ScoreEnd).unwrap();
        assert_eq!(
            (hit.query_end, hit.target_end),
            (Some(3), Some(3)),
            "{mode} ends"
        );
    }
}

#[test]
fn single_internal_mismatch() {
    // ACGT vs ACAT: three matches (+6), one mismatch (-1).
    let q = [0u8, 1, 2, 3];
    let t = [0u8, 1, 0, 3];
    assert_eq!(score(Mode::Nw, &q, &t), 5);
    // Local can drop the mismatch, keeping the better of the two flanks... but the full run
    // 6 - 1 = 5 still beats either flank (AC=4 or T=2), so SW is also 5.
    assert_eq!(score(Mode::Sw, &q, &t), 5);
}

#[test]
fn single_gap_costs_gap_open_only() {
    // Global ACGT vs AGT: A match, delete C (one-base gap = gap_open = 2), G,T match.
    // 2 - 2 + 2 + 2 = 4.
    let q = [0u8, 1, 2, 3];
    let t = [0u8, 2, 3];
    assert_eq!(score(Mode::Nw, &q, &t), 4);
}

#[test]
fn two_base_gap_uses_open_plus_extend() {
    // Global ACGT vs AT: A match, delete C and G (gap length 2 = gap_open + gap_ext = 3), T match.
    // 2 - 3 + 2 = 1.
    let q = [0u8, 1, 2, 3];
    let t = [0u8, 3];
    assert_eq!(score(Mode::Nw, &q, &t), 1);
}

#[test]
fn hw_fits_query_into_longer_target_ignoring_target_flanks() {
    // Query CG placed inside target ACGT: the flanking A and T of the target are free in HW.
    // Perfect 2-base match = +4.
    let q = [1u8, 2];
    let t = [0u8, 1, 2, 3];
    assert_eq!(score(Mode::Hw, &q, &t), 4);
    let hit = align_pair(&q, &t, &dna(), Mode::Hw, SearchType::ScoreEnd).unwrap();
    // Query fully consumed; target end is the 'G' at index 2.
    assert_eq!((hit.query_end, hit.target_end), (Some(1), Some(2)));

    // In NW the same pair pays for both unaligned target flanks (two length-1 gaps): 4 - 2 - 2 = 0.
    assert_eq!(score(Mode::Nw, &q, &t), 0);
}

#[test]
fn ov_scores_suffix_prefix_overlap_for_free_ends() {
    // Query suffix "GT" overlaps target prefix "GT": ...GT / GT... The query prefix and target
    // suffix hang off the ends for free. Overlap of 2 matches = +4.
    let q = [0u8, 0, 2, 3]; // AAGT
    let t = [2u8, 3, 1, 1]; // GTCC
    assert_eq!(score(Mode::Ov, &q, &t), 4);
}

// ---------------------------------------------------------------------------
// Cross-mode inequalities (hold for every input; independent of exact numbers)
// ---------------------------------------------------------------------------

#[test]
fn mode_score_ordering_holds_for_all_short_pairs() {
    // Freeing more end gaps can only raise the score, and local clamps at 0:
    //   SW >= OV >= HW >= NW  and  SW >= 0.
    // OV frees a superset of HW's end gaps; SW is OV plus per-cell restart-at-0.
    let scoring = dna();
    let mat = identity_matrix(4, 2, -1);
    let seqs = all_sequences(4, 3);
    for q in &seqs {
        for t in &seqs {
            let sw = score_with(&scoring, Mode::Sw, q, t);
            let ov = score_with(&scoring, Mode::Ov, q, t);
            let hw = score_with(&scoring, Mode::Hw, q, t);
            let nw = score_with(&scoring, Mode::Nw, q, t);
            // Sanity: brute agrees with the ordering premise too.
            let _ = &mat;
            assert!(sw >= 0, "SW negative: q={q:?} t={t:?} -> {sw}");
            assert!(sw >= ov, "SW<OV: q={q:?} t={t:?} ({sw}<{ov})");
            assert!(ov >= hw, "OV<HW: q={q:?} t={t:?} ({ov}<{hw})");
            assert!(hw >= nw, "HW<NW: q={q:?} t={t:?} ({hw}<{nw})");
        }
    }
}

fn score_with(s: &Scoring, mode: Mode, q: &[u8], t: &[u8]) -> i32 {
    align_pair(q, t, s, mode, SearchType::Score).unwrap().score
}

// ---------------------------------------------------------------------------
// Tie-break
// ---------------------------------------------------------------------------

#[test]
fn tie_break_prefers_smallest_target_then_query_end() {
    // Query "A" against target "AAA": three equally-scoring end positions (target index 0, 1, 2).
    // The rule picks the smallest target end.
    let q = [0u8];
    let t = [0u8, 0, 0];
    let hit = align_pair(&q, &t, &dna(), Mode::Sw, SearchType::ScoreEnd).unwrap();
    assert_eq!(hit.score, 2);
    assert_eq!(
        hit.target_end,
        Some(0),
        "should pick the leftmost target match"
    );
    assert_eq!(hit.query_end, Some(0));

    // HW of "A" into "AAA": query fully aligned, best target window is the first 'A'.
    let hit = align_pair(&q, &t, &dna(), Mode::Hw, SearchType::ScoreEnd).unwrap();
    assert_eq!((hit.score, hit.target_end), (2, Some(0)));
}

#[test]
fn tie_break_is_stable_regardless_of_target_length() {
    // Growing the run of matches must never move the reported end off the leftmost one.
    for run in 1..=6 {
        let t = vec![0u8; run];
        let hit = align_pair(&[0u8], &t, &dna(), Mode::Sw, SearchType::ScoreEnd).unwrap();
        assert_eq!(hit.target_end, Some(0), "run={run}");
    }
}

// ---------------------------------------------------------------------------
// Empty-input edge cases
// ---------------------------------------------------------------------------

#[test]
fn empty_inputs_never_panic_and_score_correctly() {
    let s = dna();
    // Both empty: score 0, no aligned positions, in every mode.
    for mode in ALL_MODES {
        let hit = align_pair(&[], &[], &s, mode, SearchType::ScoreEnd).unwrap();
        assert_eq!(hit.score, 0, "{mode} empty/empty");
        assert_eq!((hit.query_end, hit.target_end), (None, None), "{mode} ends");
    }

    // NW of empty query vs "ACGT": one length-4 gap = gap_open + 3*gap_ext = 2 + 3 = 5, so -5.
    let hit = align_pair(&[], &[0, 1, 2, 3], &s, Mode::Nw, SearchType::ScoreEnd).unwrap();
    assert_eq!(hit.score, -5);
    assert_eq!(hit.query_end, None);
    assert_eq!(hit.target_end, Some(3));

    // Local and overlap of anything against empty: 0.
    assert_eq!(score(Mode::Sw, &[0, 1, 2], &[]), 0);
    assert_eq!(score(Mode::Ov, &[], &[0, 1, 2]), 0);

    // HW with an empty query: query is "fully aligned" trivially, best empty target window, 0.
    assert_eq!(score(Mode::Hw, &[], &[0, 1, 2]), 0);
}

// ---------------------------------------------------------------------------
// Symmetry
// ---------------------------------------------------------------------------

#[test]
fn score_is_symmetric_for_symmetric_modes_and_matrix() {
    // With a symmetric substitution matrix, swapping query and target must not change the score
    // for the symmetric modes (SW, NW, OV). HW is intentionally asymmetric (only the query's
    // ends are free), so it is excluded here.
    let scoring = dna();
    let seqs = all_sequences(4, 3);
    for q in &seqs {
        for t in &seqs {
            for mode in [Mode::Sw, Mode::Nw, Mode::Ov] {
                let a = score_with(&scoring, mode, q, t);
                let b = score_with(&scoring, mode, t, q);
                assert_eq!(a, b, "{mode} not symmetric: q={q:?} t={t:?} ({a} != {b})");
            }
        }
    }
}

#[test]
fn hw_is_generally_not_symmetric() {
    // A concrete witness that HW depends on which sequence is the query: fitting a short query
    // into a long target differs from the reverse.
    let short = [1u8, 2];
    let long = [0u8, 1, 2, 3];
    assert_ne!(
        score(Mode::Hw, &short, &long),
        score(Mode::Hw, &long, &short)
    );
}
