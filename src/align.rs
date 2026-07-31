//! Traceback: the full alignment (operations + coordinates), not just the score.
//!
//! [`align`] runs the scalar Gotoh affine DP and then walks the `H`/`E`/`F` matrices back from
//! the optimal end cell to recover the sequence of [`AlignOp`]s and the aligned span in each
//! sequence. Traceback is scalar and per-pair regardless of backend (SIMD accelerates only the
//! score pass), so there is a single implementation and cross-backend determinism is automatic.
//!
//! # Memory budget
//!
//! [`align`] takes a `max_bytes` budget. The full-matrix path stores the `H`/`E`/`F` matrices
//! (`3 * (m+1) * (n+1) * 4` bytes) and is used whenever it fits. Above the budget, a
//! **checkpoint** path bounds memory to `O(n * sqrt(m))`: it stores only every `sqrt(m)`-th DP
//! row and recomputes each row-strip on demand during the walk. Because the walk logic is shared
//! and the recomputed cells are bit-for-bit the full-matrix cells, the checkpoint path returns a
//! **byte-identical** `Alignment` to the full-matrix path — the budget affects memory and time
//! (~2x), never the result. If even the checkpoint footprint exceeds `max_bytes`,
//! [`Error::TracebackBudgetExceeded`] is returned.
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

/// A database sequence (identified by its index) paired with the alignment against it. Returned
/// by [`Database::scan_aligned`](crate::Database::scan_aligned).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AlignedHit {
    /// Index of the aligned database sequence.
    pub db_index: usize,
    /// The alignment against that sequence.
    pub alignment: Alignment,
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
/// `max_bytes` bounds the traceback working memory. The full-matrix path (needing
/// `3 * (query.len()+1) * (target.len()+1) * 4` bytes) is used when it fits; above it a
/// checkpoint path bounds memory to `O(n * sqrt(m))` at ~2x time and returns a **byte-identical**
/// result. Pass `usize::MAX` to force the full-matrix path.
///
/// # Errors
///
/// - [`Error::SymbolOutOfRange`] if any symbol is `>= scoring.alphabet_len()`.
/// - [`Error::ScoreRangeTooWide`] if the reachable score could overflow `i32` for these lengths.
/// - [`Error::TracebackBudgetExceeded`] if even the checkpoint footprint exceeds `max_bytes`.
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
    traceback(query, target, scoring, mode, max_bytes)
}

/// The traceback dispatch (full-matrix vs checkpoint) without the input revalidation done by
/// [`align`]. Callers that have already validated symbols and the width bound — notably the
/// [`Database`](crate::Database) scan paths — use this directly.
pub(crate) fn traceback(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    max_bytes: usize,
) -> Result<Alignment> {
    let m = query.len();
    let n = target.len();

    // Full-matrix footprint: three `i32` matrices of `(m+1)*(n+1)` cells. Computed in `u64` so the
    // product cannot overflow before it is compared against the budget (the pyopal #5/#8 lesson:
    // size the traceback buffer from an exact bound, never a truncated one).
    let full = full_matrix_bytes(m, n);
    if full <= max_bytes as u64 || m == 0 || n == 0 {
        return Ok(traceback_full(query, target, scoring, mode));
    }

    // Over budget: fall to the checkpoint path. Balance the strip height so the peak footprint is
    // near-minimal (`~sqrt(m)`); if even that exceeds the budget, report it.
    let k = (m as u64).isqrt().max(1) as usize;
    let peak = checkpoint_bytes(m, n, k);
    if peak > max_bytes as u64 {
        return Err(Error::TracebackBudgetExceeded {
            needed_bytes: peak,
            budget_bytes: max_bytes,
        });
    }
    Ok(traceback_checkpoint(query, target, scoring, mode, k))
}

/// The smallest `max_bytes` for which [`traceback`] succeeds on an `m`-by-`n` problem: the cheaper
/// of the full-matrix and checkpoint footprints. Monotonic in `m` and `n` **on the positive
/// domain**; the degenerate `m == 0` / `n == 0` edges take the unconditional full-matrix path (a
/// single row/column) and can exceed the interior value for a tiny opposite dimension — see
/// [`traceback_min_bytes_for_database`], which a database uses so every sub-problem is covered.
pub(crate) fn traceback_min_bytes(m: usize, n: usize) -> u64 {
    let full = full_matrix_bytes(m, n);
    if m == 0 || n == 0 {
        return full;
    }
    let k = (m as u64).isqrt().max(1) as usize;
    full.min(checkpoint_bytes(m, n, k))
}

/// The smallest `max_bytes` that makes **every** sub-problem of a database whose largest scan is
/// `max_m` by `max_n` traceable: the maximum of [`traceback_min_bytes`] over the whole
/// `[0, max_m] x [0, max_n]` box. `traceback_min_bytes` is monotonic on the positive domain, so the
/// only points that can exceed `traceback_min_bytes(max_m, max_n)` are the degenerate edges — an
/// empty query (`full_matrix_bytes(0, max_n)`) or an empty target sequence
/// (`full_matrix_bytes(max_m, 0)`), a single DP row/column. Folding those in makes the value an
/// exact upper bound and monotonic non-decreasing in both arguments, so a database validates its
/// declared maximum once and every shorter scan is then infallible *and* within budget.
pub(crate) fn traceback_min_bytes_for_database(max_m: usize, max_n: usize) -> u64 {
    traceback_min_bytes(max_m, max_n)
        .max(full_matrix_bytes(max_m, 0))
        .max(full_matrix_bytes(0, max_n))
}

/// Bytes the full-matrix path allocates: three `i32` matrices of `(m+1) * (n+1)` cells.
fn full_matrix_bytes(m: usize, n: usize) -> u64 {
    (m as u64 + 1)
        .saturating_mul(n as u64 + 1)
        .saturating_mul(3)
        .saturating_mul(4)
}

/// Peak bytes the checkpoint path allocates for strip height `k`: the checkpoint rows (`H` and
/// `F` at every `k`-th row) plus one recomputed strip of up to `k + 1` rows (`H`/`E`/`F`).
fn checkpoint_bytes(m: usize, n: usize, k: usize) -> u64 {
    let num_ckpt = m as u64 / k as u64 + 1;
    let strip_rows = k as u64 + 1;
    (2 * num_ckpt + 3 * strip_rows)
        .saturating_mul(n as u64 + 1)
        .saturating_mul(4)
}

/// Which affine state the backward walk is currently in.
#[derive(Clone, Copy)]
enum State {
    /// A substitution/match matrix cell.
    H,
    /// Inside a horizontal (target-consuming) gap.
    E,
    /// Inside a vertical (query-consuming) gap.
    F,
}

/// Running best cell under the determinism-contract tie-break: maximise score, then minimise
/// `(grid_col, grid_row)` (smallest target end, then smallest query end). Identical to the score
/// kernels' selection, so every path agrees on where the alignment ends.
struct Best {
    score: i32,
    row: usize,
    col: usize,
}

impl Best {
    fn new() -> Self {
        Best {
            score: NEG,
            row: 0,
            col: 0,
        }
    }

    fn consider(&mut self, score: i32, i: usize, j: usize) {
        if score > self.score || (score == self.score && (j, i) < (self.col, self.row)) {
            self.score = score;
            self.row = i;
            self.col = j;
        }
    }
}

/// Feed one DP row into [`Best`] over the mode's answer region. Called with each row (`0..=m`) by
/// both traceback paths, so end selection is bit-identical between them.
fn consider_row(best: &mut Best, flags: &Flags, i: usize, m: usize, n: usize, hrow: &[i32]) {
    if flags.local {
        for (j, &s) in hrow.iter().enumerate().take(n + 1) {
            best.consider(s, i, j);
        }
    } else {
        if flags.answer_last_col {
            best.consider(hrow[n], i, n);
        }
        if i == m {
            best.consider(hrow[n], m, n); // corner, always in the answer region
            if flags.answer_last_row {
                for (j, &s) in hrow.iter().enumerate().take(n + 1) {
                    best.consider(s, m, j);
                }
            }
        }
    }
}

/// Random access to the DP cells the backward walk needs. The full-matrix impl indexes stored
/// matrices; the checkpoint impl recomputes row-strips on demand. Both hand back the exact same
/// values, so the shared [`walk`] produces byte-identical output whichever backs it.
///
/// `prepare(i)` must be called before accessing cells at row `i`; it guarantees rows `i` and
/// `i - 1` are available. The walk only ever reads rows `i` and `i - 1` for a non-increasing `i`.
trait CellSource {
    fn prepare(&mut self, i: usize);
    fn h(&self, i: usize, j: usize) -> i32;
    fn e(&self, i: usize, j: usize) -> i32;
    fn f(&self, i: usize, j: usize) -> i32;
}

/// The shared backward walk. Given the optimal end cell `(gr, gc)`, it retraces the canonical
/// alignment using only [`CellSource`] reads, returning the start cell and the ops (forward
/// order). This is the single source of truth for the traceback's tie-break and stop rules.
fn walk<C: CellSource>(
    cells: &mut C,
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    flags: &Flags,
    gr: usize,
    gc: usize,
) -> (usize, usize, Vec<AlignOp>) {
    let go = scoring.gap_open();
    let mut i = gr;
    let mut j = gc;
    let mut state = State::H;
    // An alignment path has at most `m + n` ops (each step consumes a query and/or target base).
    let mut ops_rev: Vec<AlignOp> = Vec::with_capacity(query.len() + target.len());
    loop {
        cells.prepare(i);
        match state {
            State::H => {
                let hij = cells.h(i, j);
                if flags.local && hij == 0 {
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
                let sub = scoring.score(query[i - 1] as usize, target[j - 1] as usize);
                if hij == cells.h(i - 1, j - 1) + sub {
                    ops_rev.push(if query[i - 1] == target[j - 1] {
                        AlignOp::Match
                    } else {
                        AlignOp::Mismatch
                    });
                    i -= 1;
                    j -= 1;
                } else if hij == cells.e(i, j) {
                    state = State::E;
                } else {
                    state = State::F;
                }
            }
            State::E => {
                ops_rev.push(AlignOp::Del);
                let ev = cells.e(i, j);
                j -= 1;
                if ev == cells.h(i, j) - go {
                    state = State::H;
                } // else stays in E (extend)
            }
            State::F => {
                ops_rev.push(AlignOp::Ins);
                let fv = cells.f(i, j);
                i -= 1;
                if fv == cells.h(i, j) - go {
                    state = State::H;
                } // else stays in F (extend)
            }
        }
    }
    ops_rev.reverse();
    (i, j, ops_rev)
}

/// [`CellSource`] over fully-materialised `H`/`E`/`F` matrices.
struct FullCells {
    h: Vec<i32>,
    e: Vec<i32>,
    f: Vec<i32>,
    cols: usize,
}

impl CellSource for FullCells {
    fn prepare(&mut self, _i: usize) {}
    fn h(&self, i: usize, j: usize) -> i32 {
        self.h[i * self.cols + j]
    }
    fn e(&self, i: usize, j: usize) -> i32 {
        self.e[i * self.cols + j]
    }
    fn f(&self, i: usize, j: usize) -> i32 {
        self.f[i * self.cols + j]
    }
}

/// One border/recurrence step over a strip, writing cell `(local_r, j)` from row `local_r - 1`.
/// Shared by the full DP and strip recompute so their cell values are identical by construction.
#[allow(clippy::too_many_arguments)]
fn fill_row(
    h: &mut [i32],
    e: &mut [i32],
    f: &mut [i32],
    prev: usize,
    cur: usize,
    cols: usize,
    i: usize,
    n: usize,
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    flags: &Flags,
) {
    let (go, ge) = (scoring.gap_open(), scoring.gap_ext());
    let base = cur * cols;
    let pbase = prev * cols;
    h[base] = if flags.left_col_free {
        0
    } else {
        -gap_penalty(go, ge, i)
    };
    e[base] = NEG;
    // Matching the top-row `E` border: set `F[i][0]` so the walk can trace and *close* a penalised
    // left-column gap; leave `NEG` when the column is free (the walk stops there instead).
    f[base] = if flags.left_col_free { NEG } else { h[base] };
    for j in 1..=n {
        e[base + j] = (h[base + j - 1] - go).max(e[base + j - 1] - ge);
        f[base + j] = (h[pbase + j] - go).max(f[pbase + j] - ge);
        let sub = scoring.score(query[i - 1] as usize, target[j - 1] as usize);
        let diag = h[pbase + j - 1] + sub;
        let mut cell = diag.max(e[base + j]).max(f[base + j]);
        if flags.local {
            cell = cell.max(0);
        }
        h[base + j] = cell;
    }
}

/// Full `H`/`E`/`F` DP with backward traceback. Caller has validated symbols, the width bound,
/// and the memory budget.
fn traceback_full(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> Alignment {
    let m = query.len();
    let n = target.len();
    let flags = Flags::for_mode(mode);
    let (go, ge) = (scoring.gap_open(), scoring.gap_ext());
    let cols = n + 1;

    let mut h = vec![0i32; (m + 1) * cols];
    let mut e = vec![NEG; (m + 1) * cols];
    let mut f = vec![NEG; (m + 1) * cols];

    // Top row (row 0) borders: free (0) or a penalised gap run. The matching `E[0][j]` is set so
    // the walk can trace a penalised top-row gap, and left `NEG` when the row is free.
    for j in 1..=n {
        h[j] = if flags.top_row_free {
            0
        } else {
            -gap_penalty(go, ge, j)
        };
        e[j] = if flags.top_row_free { NEG } else { h[j] };
    }
    for i in 1..=m {
        fill_row(
            &mut h,
            &mut e,
            &mut f,
            i - 1,
            i,
            cols,
            i,
            n,
            query,
            target,
            scoring,
            &flags,
        );
    }

    let mut best = Best::new();
    for i in 0..=m {
        consider_row(&mut best, &flags, i, m, n, &h[i * cols..i * cols + cols]);
    }

    let mut cells = FullCells { h, e, f, cols };
    let (qs, ts, ops) = walk(
        &mut cells, query, target, scoring, &flags, best.row, best.col,
    );
    Alignment {
        score: best.score,
        query_start: qs,
        query_end: best.row,
        target_start: ts,
        target_end: best.col,
        ops,
    }
}

/// [`CellSource`] backed by row checkpoints and a single recomputed strip, bounding memory to
/// `O(n * sqrt(m))`. Checkpoint `c` holds `H` and `F` at row `c * k`; a strip covers rows
/// `[base ..= base + k]` (row `base` from its checkpoint, the rest recomputed) and is reloaded as
/// the walk crosses below it. Recomputed cells equal the full-matrix cells bit-for-bit.
struct CheckpointCells<'a> {
    query: &'a [u8],
    target: &'a [u8],
    scoring: &'a Scoring,
    flags: Flags,
    m: usize,
    n: usize,
    k: usize,
    cols: usize,
    ckpt_h: Vec<i32>,
    ckpt_f: Vec<i32>,
    strip_h: Vec<i32>,
    strip_e: Vec<i32>,
    strip_f: Vec<i32>,
    strip_base: usize,
    strip_top: usize,
    loaded: bool,
}

impl CheckpointCells<'_> {
    /// Strip base (a checkpoint row) whose recomputed rows include row `i`.
    fn base_for(&self, i: usize) -> usize {
        if i == 0 { 0 } else { (i - 1) / self.k * self.k }
    }

    /// Recompute the strip rooted at checkpoint row `base` into `strip_*`.
    fn load_strip(&mut self, base: usize) {
        let cols = self.cols;
        let top = (base + self.k).min(self.m);
        let rows = top - base + 1;
        self.strip_h.clear();
        self.strip_h.resize(rows * cols, 0);
        self.strip_e.clear();
        self.strip_e.resize(rows * cols, NEG);
        self.strip_f.clear();
        self.strip_f.resize(rows * cols, NEG);

        // Local row 0 == checkpoint row `base` (its `H`/`F`; `E` only matters for the true top row).
        let c = base / self.k;
        self.strip_h[0..cols].copy_from_slice(&self.ckpt_h[c * cols..c * cols + cols]);
        self.strip_f[0..cols].copy_from_slice(&self.ckpt_f[c * cols..c * cols + cols]);
        if base == 0 {
            for j in 1..=self.n {
                self.strip_e[j] = if self.flags.top_row_free {
                    NEG
                } else {
                    self.strip_h[j]
                };
            }
        }

        for r in 1..rows {
            let i = base + r;
            fill_row(
                &mut self.strip_h,
                &mut self.strip_e,
                &mut self.strip_f,
                r - 1,
                r,
                cols,
                i,
                self.n,
                self.query,
                self.target,
                self.scoring,
                &self.flags,
            );
        }
        self.strip_base = base;
        self.strip_top = top;
        self.loaded = true;
    }

    fn local(&self, i: usize, j: usize) -> usize {
        (i - self.strip_base) * self.cols + j
    }
}

impl CellSource for CheckpointCells<'_> {
    fn prepare(&mut self, i: usize) {
        let base = self.base_for(i);
        if !self.loaded || base != self.strip_base {
            self.load_strip(base);
        }
    }
    fn h(&self, i: usize, j: usize) -> i32 {
        self.strip_h[self.local(i, j)]
    }
    fn e(&self, i: usize, j: usize) -> i32 {
        self.strip_e[self.local(i, j)]
    }
    fn f(&self, i: usize, j: usize) -> i32 {
        self.strip_f[self.local(i, j)]
    }
}

/// Checkpoint (linear-space) DP + backward traceback. Byte-identical to [`traceback_full`]: the
/// forward pass finds the same end cell and stores every `k`-th `(H, F)` row, then the shared
/// [`walk`] retraces over strips recomputed to the exact full-matrix cell values.
fn traceback_checkpoint(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    k: usize,
) -> Alignment {
    let m = query.len();
    let n = target.len();
    let flags = Flags::for_mode(mode);
    let (go, ge) = (scoring.gap_open(), scoring.gap_ext());
    let cols = n + 1;
    let num_ckpt = m / k + 1;

    let mut ckpt_h = vec![0i32; num_ckpt * cols];
    let mut ckpt_f = vec![NEG; num_ckpt * cols];

    // Forward pass in two rolling rows (a 2-row ping-pong buffer), storing every k-th `(H, F)`
    // row and selecting the end cell with the shared tie-break.
    let mut hh = vec![0i32; 2 * cols];
    let mut ee = vec![NEG; 2 * cols];
    let mut ff = vec![NEG; 2 * cols];
    for (j, cell) in hh.iter_mut().enumerate().take(n + 1).skip(1) {
        *cell = if flags.top_row_free {
            0
        } else {
            -gap_penalty(go, ge, j)
        };
    }
    let mut best = Best::new();
    consider_row(&mut best, &flags, 0, m, n, &hh[0..cols]);
    // Checkpoint 0 is row 0 (H border already in slot 0; F stays NEG).
    ckpt_h[0..cols].copy_from_slice(&hh[0..cols]);

    let (mut prev, mut cur) = (0usize, 1usize);
    for i in 1..=m {
        fill_row(
            &mut hh, &mut ee, &mut ff, prev, cur, cols, i, n, query, target, scoring, &flags,
        );
        consider_row(
            &mut best,
            &flags,
            i,
            m,
            n,
            &hh[cur * cols..cur * cols + cols],
        );
        if i % k == 0 {
            let c = i / k;
            ckpt_h[c * cols..c * cols + cols].copy_from_slice(&hh[cur * cols..cur * cols + cols]);
            ckpt_f[c * cols..c * cols + cols].copy_from_slice(&ff[cur * cols..cur * cols + cols]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    // The forward-pass rolling buffers are dead once the checkpoints are captured. Free them before
    // the walk so the peak footprint matches `checkpoint_bytes` exactly: the walk then holds only
    // the checkpoints (`2 * num_ckpt` rows) plus one recomputed strip (`3 * (k + 1)` rows). Left
    // alive they would live through the walk and push the real peak `6 * (n + 1) * 4` bytes above
    // the budgeted bound (`3 * (k + 1) >= 6` for `k >= 1`, so the forward pass itself stays under
    // the walk's footprint).
    drop(hh);
    drop(ee);
    drop(ff);

    let mut cells = CheckpointCells {
        query,
        target,
        scoring,
        flags: Flags::for_mode(mode),
        m,
        n,
        k,
        cols,
        ckpt_h,
        ckpt_f,
        strip_h: Vec::new(),
        strip_e: Vec::new(),
        strip_f: Vec::new(),
        strip_base: 0,
        strip_top: 0,
        loaded: false,
    };
    let (qs, ts, ops) = walk(
        &mut cells, query, target, scoring, &flags, best.row, best.col,
    );
    Alignment {
        score: best.score,
        query_start: qs,
        query_end: best.row,
        target_start: ts,
        target_end: best.col,
        ops,
    }
}

#[cfg(test)]
mod tests;
