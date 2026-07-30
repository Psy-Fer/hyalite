//! Traceback: the full alignment (operations + coordinates), not just the score.
//!
//! [`align`] runs the scalar Gotoh affine DP and then walks the `H`/`E`/`F` matrices back from
//! the optimal end cell to recover the sequence of [`AlignOp`]s and the aligned span in each
//! sequence. Traceback is scalar and per-pair regardless of backend (SIMD accelerates only the
//! score pass), so there is a single implementation and cross-backend determinism is automatic.
//!
//! # Memory budget
//!
//! This chunk stores the full `H`/`E`/`F` matrices, so it needs `3 * (m+1) * (n+1) * 4` bytes.
//! [`align`] takes a `max_bytes` budget and returns [`Error::TracebackBudgetExceeded`] when the
//! full matrix would exceed it. The linear-space Hirschberg path that serves larger pairs within
//! a bounded footprint is a following chunk; until it lands, pass a generous budget (or
//! `usize::MAX`) for unconditional full-matrix traceback.
//!
//! # Canonical traceback
//!
//! Among equally-scoring optimal alignments the walk is deterministic and documented: the end
//! cell is chosen by the same tie-break as the score kernels (smallest target end, then smallest
//! query end); at each step a **diagonal** (substitution) predecessor is preferred over a gap,
//! and a target-consuming gap ([`AlignOp::Del`]) over a query-consuming one ([`AlignOp::Ins`]);
//! within a gap, closing the gap is preferred over extending it. For local (`SW`) alignments the
//! walk stops at the first `0` cell, yielding a maximal-scoring local alignment.

use crate::error::{Error, Result};
use crate::kernel::{Flags, gap_penalty};
use crate::mode::Mode;
use crate::scoring::Scoring;
use core::fmt;

/// Sentinel for unreachable `E`/`F` cells, matching the scalar kernel. Divided down from
/// `i32::MIN` so the repeated `- gap` subtractions during the DP cannot underflow.
const NEG: i32 = i32::MIN / 4;

/// One column of an alignment: how a single query and/or target position was consumed.
///
/// Orientation follows SAM/CIGAR with the **target as the reference**: [`Ins`](AlignOp::Ins)
/// consumes a query symbol against a gap in the target, and [`Del`](AlignOp::Del) consumes a
/// target symbol against a gap in the query. [`Match`](AlignOp::Match) vs
/// [`Mismatch`](AlignOp::Mismatch) is decided by **symbol equality**, not the sign of the
/// substitution score, so a caller matrix that scores unequal symbols positively still reports a
/// `Mismatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlignOp {
    /// Aligned pair of identical symbols.
    Match,
    /// Aligned pair of differing symbols (a substitution).
    Mismatch,
    /// A query symbol against a gap in the target (insertion relative to the target).
    Ins,
    /// A target symbol against a gap in the query (deletion relative to the target).
    Del,
}

/// A full alignment: score, the aligned span in each sequence, and the column-by-column ops.
///
/// The spans are **half-open** `[start, end)` in each sequence: `query[query_start..query_end]`
/// and `target[target_start..target_end]` are the aligned regions. (This differs from
/// [`BestHit`](crate::BestHit), whose `query_end`/`target_end` are *inclusive* last positions;
/// here `end` is one past the last aligned symbol, so `end == BestHit.end + 1` when non-empty.)
/// An empty alignment (a local search that found nothing scoring above `0`) has `ops` empty and
/// `start == end`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Alignment {
    /// The optimal alignment score.
    pub score: i32,
    /// Start of the aligned query span (inclusive, `0`-based).
    pub query_start: usize,
    /// End of the aligned query span (exclusive).
    pub query_end: usize,
    /// Start of the aligned target span (inclusive, `0`-based).
    pub target_start: usize,
    /// End of the aligned target span (exclusive).
    pub target_end: usize,
    /// The alignment operations, in order from `query_start`/`target_start` onward.
    pub ops: Vec<AlignOp>,
}

impl Alignment {
    /// The CIGAR string with match and mismatch collapsed to `M` (the standard operator set):
    /// e.g. `"5M1I3M"`. An empty alignment yields `""`.
    #[must_use]
    pub fn cigar(&self) -> String {
        self.encode(false)
    }

    /// The extended CIGAR string distinguishing `=` (match) from `X` (mismatch): e.g.
    /// `"5=1I2=1X"`. Carries the same information as [`ops`](Self::ops); an empty alignment
    /// yields `""`.
    #[must_use]
    pub fn cigar_extended(&self) -> String {
        self.encode(true)
    }

    fn encode(&self, extended: bool) -> String {
        let letter = |op: AlignOp| -> char {
            match op {
                AlignOp::Match => {
                    if extended {
                        '='
                    } else {
                        'M'
                    }
                }
                AlignOp::Mismatch => {
                    if extended {
                        'X'
                    } else {
                        'M'
                    }
                }
                AlignOp::Ins => 'I',
                AlignOp::Del => 'D',
            }
        };

        let mut out = String::new();
        let mut run: Option<(char, usize)> = None;
        for &op in &self.ops {
            let c = letter(op);
            match run {
                Some((rc, len)) if rc == c => run = Some((rc, len + 1)),
                Some((rc, len)) => {
                    // `write!` to a String is infallible.
                    use fmt::Write;
                    let _ = write!(out, "{len}{rc}");
                    run = Some((c, 1));
                }
                None => run = Some((c, 1)),
            }
        }
        if let Some((rc, len)) = run {
            use fmt::Write;
            let _ = write!(out, "{len}{rc}");
        }
        out
    }
}

/// Align a single query against a single target and recover the full alignment.
///
/// `query` and `target` are pre-encoded alphabet indices (`0..scoring.alphabet_len()`), as with
/// [`align_pair`](crate::align_pair). The score equals what `align_pair` returns for the same
/// inputs; this additionally reports the operations and the aligned span (see [`Alignment`]).
///
/// `max_bytes` bounds the traceback working memory. The full-matrix path used here needs
/// `3 * (query.len()+1) * (target.len()+1) * 4` bytes; exceeding `max_bytes` is a typed error
/// (the linear-space path for larger pairs is a following chunk). Pass `usize::MAX` for an
/// unconditional full-matrix traceback.
///
/// # Errors
///
/// - [`Error::SymbolOutOfRange`] if any symbol is `>= scoring.alphabet_len()`.
/// - [`Error::ScoreRangeTooWide`] if the reachable score could overflow `i32` for these lengths.
/// - [`Error::TracebackBudgetExceeded`] if the full matrix would exceed `max_bytes`.
pub fn align(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    max_bytes: usize,
) -> Result<Alignment> {
    let alphabet_len = scoring.alphabet_len();
    for &sym in query.iter().chain(target.iter()) {
        if sym as usize >= alphabet_len {
            return Err(Error::SymbolOutOfRange {
                symbol: sym as usize,
                alphabet_len,
            });
        }
    }
    scoring.required_width(mode, query.len(), target.len())?;

    let m = query.len();
    let n = target.len();

    // Full-matrix footprint: three `i32` matrices of `(m+1)*(n+1)` cells. Computed in `u64` so the
    // product cannot overflow before it is compared against the budget (the pyopal #5/#8 lesson:
    // size the traceback buffer from an exact bound, never a truncated one).
    let cells = (m as u64 + 1).saturating_mul(n as u64 + 1);
    let needed = cells
        .saturating_mul(3)
        .saturating_mul(core::mem::size_of::<i32>() as u64);
    if needed > max_bytes as u64 {
        return Err(Error::TracebackBudgetExceeded {
            needed_bytes: needed,
            budget_bytes: max_bytes,
        });
    }

    Ok(traceback_full(query, target, scoring, mode))
}

/// Which affine state the backward walk is currently in.
enum State {
    /// A substitution/match matrix cell.
    H,
    /// Inside a horizontal (target-consuming) gap.
    E,
    /// Inside a vertical (query-consuming) gap.
    F,
}

/// Full `H`/`E`/`F` DP with backward traceback. Caller has validated symbols, the width bound,
/// and the memory budget.
fn traceback_full(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> Alignment {
    let m = query.len();
    let n = target.len();
    let flags = Flags::for_mode(mode);
    let (go, ge) = (scoring.gap_open(), scoring.gap_ext());
    let cols = n + 1;
    let idx = |i: usize, j: usize| i * cols + j;

    let mut h = vec![0i32; (m + 1) * cols];
    let mut e = vec![NEG; (m + 1) * cols];
    let mut f = vec![NEG; (m + 1) * cols];

    // Borders. `H` borders are free (0) or a penalised gap run; the matching `E`/`F` border is set
    // so the generic backward walk can trace a penalised border gap, and left `NEG` when the border
    // is free (the walk stops there instead of consulting it).
    for j in 1..=n {
        h[idx(0, j)] = if flags.top_row_free {
            0
        } else {
            -gap_penalty(go, ge, j)
        };
        e[idx(0, j)] = if flags.top_row_free {
            NEG
        } else {
            h[idx(0, j)]
        };
    }
    for i in 1..=m {
        h[idx(i, 0)] = if flags.left_col_free {
            0
        } else {
            -gap_penalty(go, ge, i)
        };
        f[idx(i, 0)] = if flags.left_col_free {
            NEG
        } else {
            h[idx(i, 0)]
        };
    }

    for i in 1..=m {
        for j in 1..=n {
            e[idx(i, j)] = (h[idx(i, j - 1)] - go).max(e[idx(i, j - 1)] - ge);
            f[idx(i, j)] = (h[idx(i - 1, j)] - go).max(f[idx(i - 1, j)] - ge);
            let sub = scoring.score(query[i - 1] as usize, target[j - 1] as usize);
            let diag = h[idx(i - 1, j - 1)] + sub;
            let mut cell = diag.max(e[idx(i, j)]).max(f[idx(i, j)]);
            if flags.local {
                cell = cell.max(0);
            }
            h[idx(i, j)] = cell;
        }
    }

    // Optimal end cell, by the determinism-contract tie-break (max score, then smallest
    // `(grid_col, grid_row)` = smallest target end, then smallest query end) over the mode's
    // answer region — identical to the score kernels.
    let mut best_score = NEG;
    let mut gr = 0usize;
    let mut gc = 0usize;
    let mut consider = |score: i32, i: usize, j: usize| {
        if score > best_score || (score == best_score && (j, i) < (gc, gr)) {
            best_score = score;
            gr = i;
            gc = j;
        }
    };
    if flags.local {
        for i in 0..=m {
            for j in 0..=n {
                consider(h[idx(i, j)], i, j);
            }
        }
    } else {
        consider(h[idx(m, n)], m, n);
        if flags.answer_last_row {
            for j in 0..=n {
                consider(h[idx(m, j)], m, j);
            }
        }
        if flags.answer_last_col {
            for i in 0..=m {
                consider(h[idx(i, n)], i, n);
            }
        }
    }

    // Backward walk from the end cell.
    let mut i = gr;
    let mut j = gc;
    let mut state = State::H;
    let mut ops_rev: Vec<AlignOp> = Vec::new();
    loop {
        match state {
            State::H => {
                if flags.local && h[idx(i, j)] == 0 {
                    break;
                }
                if i == 0 && j == 0 {
                    break;
                }
                if i == 0 {
                    if flags.top_row_free {
                        break;
                    }
                    state = State::E; // trace the penalised top-row gap
                    continue;
                }
                if j == 0 {
                    if flags.left_col_free {
                        break;
                    }
                    state = State::F; // trace the penalised left-column gap
                    continue;
                }
                let v = h[idx(i, j)];
                let sub = scoring.score(query[i - 1] as usize, target[j - 1] as usize);
                if v == h[idx(i - 1, j - 1)] + sub {
                    ops_rev.push(if query[i - 1] == target[j - 1] {
                        AlignOp::Match
                    } else {
                        AlignOp::Mismatch
                    });
                    i -= 1;
                    j -= 1;
                } else if v == e[idx(i, j)] {
                    state = State::E;
                } else {
                    state = State::F;
                }
            }
            State::E => {
                ops_rev.push(AlignOp::Del);
                let ev = e[idx(i, j)];
                j -= 1;
                if ev == h[idx(i, j)] - go {
                    state = State::H;
                } // else stays in E (extend)
            }
            State::F => {
                ops_rev.push(AlignOp::Ins);
                let fv = f[idx(i, j)];
                i -= 1;
                if fv == h[idx(i, j)] - go {
                    state = State::H;
                } // else stays in F (extend)
            }
        }
    }

    ops_rev.reverse();
    Alignment {
        score: best_score,
        query_start: i,
        query_end: gr,
        target_start: j,
        target_end: gc,
        ops: ops_rev,
    }
}

#[cfg(test)]
mod tests;
