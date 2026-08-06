#![no_main]
//! Traceback / `Alignment` fuzz target.
//!
//! The traceback is scalar-only, so the axis here is not cross-backend but internal-consistency,
//! with three independent oracles per input:
//!   * **Ops re-score to the reported score.** Walking the returned `ops` from
//!     `(query_start, target_start)` — summing matrix entries for `Match`/`Mismatch` and the affine
//!     `gap_open + (n-1)·gap_ext` for each maximal `Ins`/`Del` run — must reproduce `alignment.score`
//!     exactly, and must consume exactly the reported query/target spans. This catches any
//!     traceback-walk or CIGAR bug independently of the DP that produced the alignment.
//!   * **Score agrees with the score-only pass** (`align_pair(.., Score)`).
//!   * **Budget-independence:** the full-matrix path (`max_bytes = usize::MAX`) and the linear-space
//!     checkpoint path (tiny budgets) must return **byte-identical** alignments; an over-tight budget
//!     returns `TracebackBudgetExceeded`, never a panic or a different result.
//!   * The `Database` traceback (`scan_aligned` / `scan_all_aligned`) matches `align()` for the pair.
//!
//! `overflow-checks = true` (fuzz profile) turns any i32 wrap in the scalar DP into a crash.
//!
//! Run: `cargo +nightly fuzz run traceback`.

use arbitrary::{Result, Unstructured};
use hyalite::{
    AlignOp, Database, Error, Mode, Scoring, Scratch, SearchType, align, align_pair,
};
use libfuzzer_sys::fuzz_target;

const MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];

struct Case {
    al: usize,
    matrix: Vec<i32>,
    gap_open: i32,
    gap_ext: i32,
    mode: Mode,
    query: Vec<u8>,
    target: Vec<u8>,
}

fn decode(u: &mut Unstructured) -> Result<Case> {
    let al = u.int_in_range(2u8..=6)? as usize;
    let (lo, hi): (i32, i32) = match u.int_in_range(0u8..=3)? {
        0 => (-8, 8),
        1 => (-1_000, 1_000),
        2 => (-100_000, 100_000),
        _ => (-600_000_000, 600_000_000),
    };
    let mut matrix = vec![0i32; al * al];
    for cell in &mut matrix {
        *cell = u.int_in_range(lo..=hi)?;
    }
    let gap_open = u.int_in_range(0..=hi.max(0))?;
    let gap_ext = u.int_in_range(0..=gap_open)?;
    let mode = MODES[u.int_in_range(0u8..=4)? as usize];

    let sym = |u: &mut Unstructured| -> Result<u8> { Ok(u.int_in_range(0u8..=(al as u8 - 1))?) };
    // Modest lengths: the checkpoint path recomputes strips, so keep the DP cheap for fuzz throughput.
    let seq = |u: &mut Unstructured| -> Result<Vec<u8>> {
        let len = u.int_in_range(0usize..=40)?;
        (0..len).map(|_| sym(u)).collect()
    };
    let query = seq(u)?;
    let target = seq(u)?;

    Ok(Case {
        al,
        matrix,
        gap_open,
        gap_ext,
        mode,
        query,
        target,
    })
}

/// The affine cost of a gap of length `n` under the crate's convention.
fn gap_cost(n: usize, go: i64, ge: i64) -> i64 {
    if n == 0 {
        0
    } else {
        go + (n as i64 - 1) * ge
    }
}

/// Independently re-score an alignment from its ops, returning `(score, query_consumed,
/// target_consumed)`. `Match`/`Mismatch` add the matrix entry and advance both; a maximal run of
/// `Ins` (query-only) or `Del` (target-only) subtracts one affine gap penalty.
fn rescore(
    ops: &[AlignOp],
    query: &[u8],
    target: &[u8],
    al: usize,
    matrix: &[i32],
    go: i64,
    ge: i64,
    qs: usize,
    ts: usize,
) -> (i64, usize, usize) {
    let mut score = 0i64;
    let (mut qi, mut ti) = (qs, ts);
    // Current gap run: (is_ins, length). Flushed (penalty applied) when the op kind changes.
    let mut run: Option<(bool, usize)> = None;
    let flush = |run: &mut Option<(bool, usize)>, score: &mut i64| {
        if let Some((_, n)) = run.take() {
            *score -= gap_cost(n, go, ge);
        }
    };
    for &op in ops {
        match op {
            AlignOp::Ins => match run {
                Some((true, ref mut n)) => *n += 1,
                _ => {
                    flush(&mut run, &mut score);
                    run = Some((true, 1));
                }
            },
            AlignOp::Del => match run {
                Some((false, ref mut n)) => *n += 1,
                _ => {
                    flush(&mut run, &mut score);
                    run = Some((false, 1));
                }
            },
            AlignOp::Match | AlignOp::Mismatch => {
                flush(&mut run, &mut score);
                score += matrix[query[qi] as usize * al + target[ti] as usize] as i64;
                qi += 1;
                ti += 1;
                continue;
            }
        }
        // Advance the consumed axis for the gap op just recorded.
        match op {
            AlignOp::Ins => qi += 1,
            AlignOp::Del => ti += 1,
            _ => {}
        }
    }
    flush(&mut run, &mut score);
    (score, qi - qs, ti - ts)
}

fn run(case: Case) {
    let Case {
        al,
        matrix,
        gap_open,
        gap_ext,
        mode,
        query,
        target,
    } = case;
    let Ok(scoring) = Scoring::new(al, matrix.clone(), gap_open, gap_ext) else {
        return;
    };

    // Full-matrix traceback is the reference. A pathological over-range input is rejected by the
    // width proof (ScoreRangeTooWide) — nothing to check.
    let Ok(full) = align(&query, &target, &scoring, mode, usize::MAX) else {
        return;
    };

    // Oracle 1: ops re-score to the reported score and consume exactly the reported spans.
    let (rescored, qc, tc) = rescore(
        &full.ops,
        &query,
        &target,
        al,
        &matrix,
        gap_open as i64,
        gap_ext as i64,
        full.query_start,
        full.target_start,
    );
    assert_eq!(
        qc,
        full.query_end - full.query_start,
        "ops consume wrong query span, {mode}"
    );
    assert_eq!(
        tc,
        full.target_end - full.target_start,
        "ops consume wrong target span, {mode}"
    );
    assert_eq!(rescored, full.score as i64, "ops re-score != reported score, {mode}");

    // Oracle 2: the score matches the score-only pass.
    if let Ok(hit) = align_pair(&query, &target, &scoring, mode, SearchType::Score) {
        assert_eq!(hit.score, full.score, "traceback score != score-pass, {mode}");
    }

    // Oracle 3: budget-independence — the checkpoint path must be byte-identical, or cleanly refuse.
    for &budget in &[0usize, 1, 32, 256, 4096] {
        match align(&query, &target, &scoring, mode, budget) {
            Ok(a) => assert_eq!(a, full, "budget {budget} diverges from full-matrix, {mode}"),
            Err(Error::TracebackBudgetExceeded { .. }) => {}
            Err(e) => panic!("unexpected traceback error at budget {budget}: {e:?}"),
        }
    }

    // Oracle 4: the Database traceback matches align() for this pair (single-target DB, generous
    // budget so construction proves the budget sufficient).
    if let Ok(db) = Database::builder()
        .sequences(std::slice::from_ref(&target))
        .scoring(scoring.clone())
        .mode(mode)
        .search_type(SearchType::Alignment { max_bytes: usize::MAX })
        .max_query_len(query.len().max(1))
        .build()
    {
        let mut scratch = Scratch::new(&db);
        let hit = db.scan_aligned(&mut scratch, &query);
        assert_eq!(hit.db_index, 0);
        assert_eq!(hit.alignment, full, "scan_aligned != align(), {mode}");

        let mut all = Vec::new();
        db.scan_all_aligned(&mut scratch, &query, &mut all);
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], full, "scan_all_aligned != align(), {mode}");
    }
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    if let Ok(case) = decode(&mut u) {
        run(case);
    }
});
