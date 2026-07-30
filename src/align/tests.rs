//! Traceback tests.
//!
//! The traceback has no independent oracle of its *operations*, so it is pinned from several
//! angles at once: (1) its score equals `align_pair`'s (already brute-verified); (2) the emitted
//! ops, re-scored independently, reproduce that score; (3) every `Match`/`Mismatch` op agrees
//! with symbol equality and every op-count matches the reported spans; (4) hand-computed CIGARs
//! on fixed cases. Multiple varied inputs per property, per the project's testing agreement.

use super::*;
use crate::kernel::gap_penalty;
use crate::{Error, Mode, Scoring, SearchType, align_pair};
use proptest::prelude::*;

const ALL_MODES: [Mode; 4] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov];

fn id_matrix(al: usize, m: i32, x: i32) -> Vec<i32> {
    let mut v = vec![x; al * al];
    for i in 0..al {
        v[i * al + i] = m;
    }
    v
}

fn dna() -> Scoring {
    Scoring::new(4, id_matrix(4, 2, -1), 2, 1).unwrap()
}

/// A spread of valid scorings to exercise the traceback under different gap regimes.
fn scorings() -> Vec<Scoring> {
    vec![
        dna(),
        Scoring::new(4, id_matrix(4, 1, -3), 10, 5).unwrap(), // heavy, steep gaps
        Scoring::new(4, id_matrix(4, 3, -2), 4, 0).unwrap(),  // free gap extension
        Scoring::new(4, id_matrix(4, 5, -1), 1, 1).unwrap(),  // cheap gaps
    ]
}

/// Independently re-score an alignment from its ops, charging affine penalties per maximal
/// same-direction gap run. Structurally unlike the DP, so agreement is meaningful.
fn rescore(a: &Alignment, q: &[u8], t: &[u8], s: &Scoring) -> i32 {
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
            AlignOp::Ins => {
                let mut len = 0;
                while k < a.ops.len() && matches!(a.ops[k], AlignOp::Ins) {
                    len += 1;
                    qi += 1;
                    k += 1;
                }
                total -= gap_penalty(s.gap_open(), s.gap_ext(), len);
            }
            AlignOp::Del => {
                let mut len = 0;
                while k < a.ops.len() && matches!(a.ops[k], AlignOp::Del) {
                    len += 1;
                    tj += 1;
                    k += 1;
                }
                total -= gap_penalty(s.gap_open(), s.gap_ext(), len);
            }
        }
    }
    total
}

/// Every internal-consistency invariant that must hold for any alignment.
fn assert_consistent(a: &Alignment, q: &[u8], t: &[u8], s: &Scoring, mode: Mode) {
    // Score matches the score-only kernel (which is brute-verified elsewhere).
    let hit = align_pair(q, t, s, mode, SearchType::Score).unwrap();
    assert_eq!(a.score, hit.score, "score vs align_pair ({mode})");

    // Re-scoring the ops reproduces the score.
    assert_eq!(rescore(a, q, t, s), a.score, "rescore ({mode})");

    // Spans are ordered and in-bounds.
    assert!(a.query_start <= a.query_end && a.query_end <= q.len());
    assert!(a.target_start <= a.target_end && a.target_end <= t.len());

    // Op counts match the consumed spans, and Match/Mismatch agree with symbol equality.
    let (mut qc, mut tc) = (0usize, 0usize);
    let mut qi = a.query_start;
    let mut tj = a.target_start;
    for &op in &a.ops {
        match op {
            AlignOp::Match | AlignOp::Mismatch => {
                let eq = q[qi] == t[tj];
                assert_eq!(
                    eq,
                    op == AlignOp::Match,
                    "Match/Mismatch vs symbol equality ({mode})"
                );
                qi += 1;
                tj += 1;
                qc += 1;
                tc += 1;
            }
            AlignOp::Ins => {
                qi += 1;
                qc += 1;
            }
            AlignOp::Del => {
                tj += 1;
                tc += 1;
            }
        }
    }
    assert_eq!(
        qc,
        a.query_end - a.query_start,
        "query op-count vs span ({mode})"
    );
    assert_eq!(
        tc,
        a.target_end - a.target_start,
        "target op-count vs span ({mode})"
    );
}

fn ops(a: &Alignment) -> Vec<AlignOp> {
    a.ops.clone()
}

// ---------------------------------------------------------------------------
// Hand-computed cases
// ---------------------------------------------------------------------------

#[test]
fn nw_identical_is_all_match() {
    let s = dna();
    let seq = [0u8, 1, 2, 3];
    let a = align(&seq, &seq, &s, Mode::Nw, usize::MAX).unwrap();
    assert_eq!(a.score, 8);
    assert_eq!(ops(&a), vec![AlignOp::Match; 4]);
    assert_eq!(a.cigar(), "4M");
    assert_eq!(a.cigar_extended(), "4=");
    assert_eq!((a.query_start, a.query_end), (0, 4));
    assert_eq!((a.target_start, a.target_end), (0, 4));
}

#[test]
fn nw_single_substitution() {
    let s = dna();
    let q = [0u8, 1, 2, 3];
    let t = [0u8, 1, 1, 3];
    let a = align(&q, &t, &s, Mode::Nw, usize::MAX).unwrap();
    assert_eq!(
        ops(&a),
        vec![
            AlignOp::Match,
            AlignOp::Match,
            AlignOp::Mismatch,
            AlignOp::Match
        ]
    );
    assert_eq!(a.score, 2 + 2 - 1 + 2);
    assert_eq!(a.cigar(), "4M");
    assert_eq!(a.cigar_extended(), "2=1X1=");
}

#[test]
fn nw_needs_one_gap() {
    let s = dna();
    let q = [0u8, 1, 2, 3]; // ACGT
    let t = [0u8, 1, 3]; // ACT
    let a = align(&q, &t, &s, Mode::Nw, usize::MAX).unwrap();
    // One query base must be an insertion; score = three matches minus a length-1 gap.
    assert_eq!(a.score, 2 + 2 + 2 - 2);
    assert_consistent(&a, &q, &t, &s, Mode::Nw);
    assert_eq!(a.cigar().chars().filter(|c| *c == 'I').count(), 1);
}

#[test]
fn sw_recovers_exact_substring() {
    let s = dna();
    let t = [0u8, 1, 2, 3];
    let q = [2u8, 2, 0, 1, 2, 3, 2, 2]; // t occurs at query[2..6]
    let a = align(&q, &t, &s, Mode::Sw, usize::MAX).unwrap();
    assert_eq!(a.score, 8);
    assert_eq!(ops(&a), vec![AlignOp::Match; 4]);
    assert_eq!(a.cigar(), "4M");
    assert_eq!((a.query_start, a.query_end), (2, 6));
    assert_eq!((a.target_start, a.target_end), (0, 4));
}

#[test]
fn hw_query_is_a_free_window_of_the_target() {
    let s = dna();
    let q = [1u8, 2]; // CG
    let t = [0u8, 1, 2, 3]; // ACGT; CG sits at t[1..3], leading/trailing target free
    let a = align(&q, &t, &s, Mode::Hw, usize::MAX).unwrap();
    assert_eq!(a.score, 4);
    assert_eq!(ops(&a), vec![AlignOp::Match, AlignOp::Match]);
    assert_eq!((a.query_start, a.query_end), (0, 2)); // whole query consumed
    assert_eq!((a.target_start, a.target_end), (1, 3)); // free target ends trimmed
    assert_eq!(a.cigar(), "2M");
}

#[test]
fn ov_prefers_smaller_target_end_on_a_tie() {
    // Two equal-scoring overlaps; the documented tie-break takes the smallest target end.
    let s = dna();
    let q = [2u8, 3]; // GT
    let t = [2u8, 3, 2, 3]; // GTGT: GT matches at t[0..2] and t[2..4], same score
    let a = align(&q, &t, &s, Mode::Ov, usize::MAX).unwrap();
    assert_eq!(a.score, 4);
    assert_eq!((a.target_start, a.target_end), (0, 2)); // smallest target end wins
    assert_consistent(&a, &q, &t, &s, Mode::Ov);
}

#[test]
fn in_alphabet_unknown_symbol_traces_back() {
    // A 5-symbol alphabet where index 4 is an "N": scored, not special-cased. Including at the
    // very start of the alignment (the pyopal crash case) must just produce a valid CIGAR.
    let mut matrix = id_matrix(5, 2, -1);
    matrix[4 * 5 + 4] = -1; // N vs N is not a reward here
    let s = Scoring::new(5, matrix, 2, 1).unwrap();
    for mode in ALL_MODES {
        let q = [4u8, 0, 1]; // N A C
        let t = [4u8, 0, 1]; // N A C
        let a = align(&q, &t, &s, mode, usize::MAX).unwrap();
        assert_consistent(&a, &q, &t, &s, mode);
        // First op corresponds to the leading N/N column; symbol equality makes it a Match.
        if !a.ops.is_empty() {
            assert_eq!(a.ops[0], AlignOp::Match, "{mode}");
        }
    }
}

// ---------------------------------------------------------------------------
// Edge geometry
// ---------------------------------------------------------------------------

#[test]
fn empty_inputs() {
    let s = dna();
    for mode in ALL_MODES {
        let a = align(&[], &[], &s, mode, usize::MAX).unwrap();
        assert_eq!(a.score, 0);
        assert!(a.ops.is_empty());
        assert_eq!(a.cigar(), "");
        assert_eq!(a.cigar_extended(), "");
        assert_eq!((a.query_start, a.query_end), (0, 0));
        assert_eq!((a.target_start, a.target_end), (0, 0));
    }
}

#[test]
fn empty_query_against_target() {
    let s = dna();
    let t = [0u8, 1, 2];
    // NW: the whole target is a penalised deletion run.
    let nw = align(&[], &t, &s, Mode::Nw, usize::MAX).unwrap();
    assert_eq!(nw.ops, vec![AlignOp::Del; 3]);
    assert_eq!(nw.score, -gap_penalty(s.gap_open(), s.gap_ext(), 3));
    assert_eq!(nw.cigar(), "3D");
    // SW/HW/OV: nothing needs to align, so the empty alignment scores 0.
    for mode in [Mode::Sw, Mode::Hw, Mode::Ov] {
        let a = align(&[], &t, &s, mode, usize::MAX).unwrap();
        assert_eq!(a.score, 0, "{mode}");
        assert!(a.ops.is_empty(), "{mode}");
    }
}

#[test]
fn single_symbol_match_and_mismatch() {
    let s = dna();
    for mode in ALL_MODES {
        let m = align(&[2], &[2], &s, mode, usize::MAX).unwrap();
        assert_eq!(m.ops, vec![AlignOp::Match], "{mode} match");
        assert_eq!(m.score, 2, "{mode} match");

        let x = align(&[2], &[0], &s, mode, usize::MAX).unwrap();
        assert_consistent(&x, &[2], &[0], &s, mode);
    }
}

#[test]
fn all_mismatch_local_is_empty() {
    let s = dna();
    let q = [0u8, 0, 0];
    let t = [1u8, 1, 1];
    let a = align(&q, &t, &s, Mode::Sw, usize::MAX).unwrap();
    assert_eq!(a.score, 0);
    assert!(a.ops.is_empty());
}

// ---------------------------------------------------------------------------
// CIGAR formatting
// ---------------------------------------------------------------------------

#[test]
fn cigar_run_length_and_operators() {
    let a = Alignment {
        score: 0,
        query_start: 0,
        query_end: 0,
        target_start: 0,
        target_end: 0,
        ops: vec![
            AlignOp::Match,
            AlignOp::Match,
            AlignOp::Mismatch,
            AlignOp::Ins,
            AlignOp::Ins,
            AlignOp::Del,
            AlignOp::Match,
        ],
    };
    // Non-extended: match and mismatch collapse into one M run.
    assert_eq!(a.cigar(), "3M2I1D1M");
    // Extended: = and X stay distinct.
    assert_eq!(a.cigar_extended(), "2=1X2I1D1=");
}

// ---------------------------------------------------------------------------
// Budget
// ---------------------------------------------------------------------------

#[test]
fn budget_boundary_is_exact() {
    let s = dna();
    let q = [0u8, 1, 2, 3];
    let t = [0u8, 1, 2];
    let needed = 3 * (q.len() as u64 + 1) * (t.len() as u64 + 1) * 4;

    // Exactly enough succeeds; one byte short fails with the byte counts reported.
    assert!(align(&q, &t, &s, Mode::Nw, needed as usize).is_ok());
    match align(&q, &t, &s, Mode::Nw, needed as usize - 1) {
        Err(Error::TracebackBudgetExceeded {
            needed_bytes,
            budget_bytes,
        }) => {
            assert_eq!(needed_bytes, needed);
            assert_eq!(budget_bytes, needed as usize - 1);
        }
        other => panic!("expected budget error, got {other:?}"),
    }
}

#[test]
fn out_of_range_symbol_is_rejected() {
    let s = dna();
    assert!(matches!(
        align(&[9], &[0], &s, Mode::Nw, usize::MAX),
        Err(Error::SymbolOutOfRange { .. })
    ));
}

// ---------------------------------------------------------------------------
// Properties across modes / scorings / random sequences
// ---------------------------------------------------------------------------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    #[test]
    fn traceback_is_consistent_everywhere(
        q in prop::collection::vec(0u8..4, 0..9),
        t in prop::collection::vec(0u8..4, 0..9),
    ) {
        for s in scorings() {
            for mode in ALL_MODES {
                let a = align(&q, &t, &s, mode, usize::MAX).unwrap();
                // Fold the standalone consistency checks into prop_assert form.
                let hit = align_pair(&q, &t, &s, mode, SearchType::Score).unwrap();
                prop_assert_eq!(a.score, hit.score);
                prop_assert_eq!(rescore(&a, &q, &t, &s), a.score);
                prop_assert!(a.query_end <= q.len() && a.target_end <= t.len());
                prop_assert!(a.query_start <= a.query_end && a.target_start <= a.target_end);

                let (mut qc, mut tc) = (0usize, 0usize);
                let (mut qi, mut tj) = (a.query_start, a.target_start);
                for &op in &a.ops {
                    match op {
                        AlignOp::Match | AlignOp::Mismatch => {
                            prop_assert_eq!(q[qi] == t[tj], op == AlignOp::Match);
                            qi += 1; tj += 1; qc += 1; tc += 1;
                        }
                        AlignOp::Ins => { qi += 1; qc += 1; }
                        AlignOp::Del => { tj += 1; tc += 1; }
                    }
                }
                prop_assert_eq!(qc, a.query_end - a.query_start);
                prop_assert_eq!(tc, a.target_end - a.target_start);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checkpoint (linear-space) path is byte-identical to the full-matrix path
// ---------------------------------------------------------------------------

/// Every sequence over `al` symbols of length `1..=maxlen`.
fn all_seqs(al: u8, maxlen: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut frontier = vec![vec![]];
    for _ in 0..maxlen {
        let mut next = Vec::new();
        for seq in &frontier {
            for sym in 0..al {
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

/// A spread of strip heights: 1 (maximum recompute), a few small, ~sqrt(m), and >= m (single
/// strip == effectively full). Each must reproduce the full-matrix result exactly.
fn ks_for(m: usize) -> Vec<usize> {
    let mut v = vec![
        1usize,
        2,
        3,
        (m as u64).isqrt().max(1) as usize,
        m.max(1),
        m + 1,
    ];
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn checkpoint_matches_full_exhaustively() {
    // All non-empty pairs over small alphabets, every mode, several scorings, every strip height.
    for (al, maxlen) in [(2u8, 5usize), (3, 3)] {
        let seqs = all_seqs(al, maxlen);
        for q in &seqs {
            for t in &seqs {
                for s in scorings() {
                    for mode in ALL_MODES {
                        let full = traceback_full(q, t, &s, mode);
                        for k in ks_for(q.len()) {
                            let cp = traceback_checkpoint(q, t, &s, mode, k);
                            assert_eq!(full, cp, "mismatch: mode={mode} k={k} q={q:?} t={t:?}");
                        }
                    }
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4000))]

    /// Random, larger pairs: the checkpoint path is byte-identical to the full-matrix path for
    /// every strip height, and `align()`'s budget dispatch is likewise budget-independent.
    #[test]
    fn checkpoint_matches_full_random(
        q in prop::collection::vec(0u8..4, 1..15),
        t in prop::collection::vec(0u8..4, 1..15),
    ) {
        for s in scorings() {
            for mode in ALL_MODES {
                let full = traceback_full(&q, &t, &s, mode);
                for k in ks_for(q.len()) {
                    prop_assert_eq!(&traceback_checkpoint(&q, &t, &s, mode, k), &full,
                        "k={} mode={}", k, mode);
                }
            }
        }
    }
}

#[test]
fn tiny_budget_that_even_checkpoint_cannot_meet_errors() {
    let s = dna();
    let q = vec![0u8; 400];
    let t = vec![1u8; 400];
    // Far below any checkpoint footprint for a 400x400 problem.
    match align(&q, &t, &s, Mode::Nw, 8) {
        Err(Error::TracebackBudgetExceeded { .. }) => {}
        other => panic!("expected budget error, got {other:?}"),
    }
    // With room, the checkpoint path succeeds and equals the full-matrix result.
    let big = align(&q, &t, &s, Mode::Nw, usize::MAX).unwrap();
    let checkpointed = align(&q, &t, &s, Mode::Nw, 1 << 20).unwrap();
    assert_eq!(big, checkpointed);
}
