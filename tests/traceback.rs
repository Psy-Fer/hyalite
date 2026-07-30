//! Traceback differential tests against the independent brute-force oracle.
//!
//! `align` reports a full alignment; `common::brute` computes the optimal *score* a completely
//! different way (exhaustive path / substring enumeration). This asserts, over random scorings
//! and sequences across every mode, that (1) the reported score is optimal (equals the oracle),
//! and (2) the reported ops re-score to exactly that value. Together these pin both halves of the
//! result: the score is right, and the operations are a valid optimal path achieving it.

mod common;

use common::{ALL_MODES, brute};
use hyalite::{AlignOp, Alignment, Mode, Scoring, align};
use proptest::prelude::*;

/// Re-score an alignment from its ops alone, charging affine penalties per maximal gap run.
fn rescore(a: &Alignment, q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let gap = |len: usize| {
        if len == 0 {
            0
        } else {
            go + (len as i32 - 1) * ge
        }
    };
    let sub = |x: u8, y: u8| mat[x as usize * al + y as usize];
    let (mut qi, mut tj) = (a.query_start, a.target_start);
    let mut total = 0i32;
    let mut k = 0;
    while k < a.ops.len() {
        match a.ops[k] {
            AlignOp::Match | AlignOp::Mismatch => {
                total += sub(q[qi], t[tj]);
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
                total -= gap(len);
            }
            AlignOp::Del => {
                let mut len = 0;
                while k < a.ops.len() && matches!(a.ops[k], AlignOp::Del) {
                    len += 1;
                    tj += 1;
                    k += 1;
                }
                total -= gap(len);
            }
        }
    }
    total
}

/// A random valid scoring (`gap_open >= gap_ext >= 0`) plus two short sequences.
fn scheme_and_pair() -> impl Strategy<Value = (usize, Vec<i32>, i32, i32, Vec<u8>, Vec<u8>)> {
    (2usize..=4).prop_flat_map(|al| {
        let matrix = prop::collection::vec(-5i32..=5, al * al);
        let gaps = (0i32..=8).prop_flat_map(|go| (Just(go), 0i32..=go));
        let q = prop::collection::vec(0u8..al as u8, 0..8);
        let t = prop::collection::vec(0u8..al as u8, 0..8);
        (Just(al), matrix, gaps, q, t).prop_map(|(al, m, (go, ge), q, t)| (al, m, go, ge, q, t))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6000))]

    #[test]
    fn align_is_optimal_and_ops_rescore(
        (al, matrix, go, ge, q, t) in scheme_and_pair(),
    ) {
        let scoring = Scoring::new(al, matrix.clone(), go, ge).unwrap();
        for mode in ALL_MODES {
            let a = align(&q, &t, &scoring, mode, usize::MAX).unwrap();
            prop_assert_eq!(a.score, brute(mode, &q, &t, &matrix, al, go, ge), "{}", mode);
            prop_assert_eq!(rescore(&a, &q, &t, &matrix, al, go, ge), a.score, "{}", mode);
        }
    }
}

#[test]
fn known_global_alignment_with_a_deletion() {
    // A G T  vs  A C G T: the target's C has no query partner (a deletion, target-consuming).
    let matrix = vec![
        2, -1, -1, -1, //
        -1, 2, -1, -1, //
        -1, -1, 2, -1, //
        -1, -1, -1, 2, //
    ];
    let scoring = Scoring::new(4, matrix.clone(), 2, 1).unwrap();
    let q = [0u8, 2, 3];
    let t = [0u8, 1, 2, 3];
    let a = align(&q, &t, &scoring, Mode::Nw, usize::MAX).unwrap();
    assert_eq!(a.score, 2 + 2 + 2 - 2); // three matches minus a length-1 gap
    assert_eq!(a.cigar().chars().filter(|c| *c == 'D').count(), 1);
    assert_eq!(rescore(&a, &q, &t, &matrix, 4, 2, 1), a.score);
    assert_eq!((a.query_start, a.query_end), (0, 3));
    assert_eq!((a.target_start, a.target_end), (0, 4));
}
