//! Scalar reference alignment kernel — the correctness oracle.
//!
//! This is the plain, clarity-first Gotoh affine-gap dynamic program that every SIMD backend
//! must reproduce bit-for-bit. It computes in `i32`; the [static width proof](crate::width)
//! guarantees that width is sufficient for any inputs `align_pair` accepts, so no saturation
//! occurs and the result equals what a correctly-saturating narrower backend would produce.
//!
//! # Mode semantics
//!
//! All four modes share one recurrence and differ only in which border cells start free and
//! which cells are candidates for the answer:
//!
//! | Mode | Query start/end gaps | Target start/end gaps | Answer over |
//! |------|----------------------|-----------------------|-------------|
//! | `NW`  | penalised           | penalised             | corner `(m, n)` |
//! | `HW`  | **free**            | penalised             | last row (query fully aligned; target window free) |
//! | `SHW` | penalised           | **free**              | last column (target fully aligned; query window free) |
//! | `OV`  | **free**            | **free**              | last row ∪ last column (overlap) |
//! | `SW`  | free (local)        | free (local)          | every cell, clamped at 0 |
//!
//! These are standard textbook semantics chosen for a well-defined oracle. Exact byte-parity
//! with Opal's / STAR's end-gap conventions is a separate concern verified against Opal test
//! vectors during the rustar integration milestone (see `handover.md` §8).
//!
//! # Tie-break
//!
//! Among cells achieving the best score, the reported end is the one with the **smallest target
//! position, then the smallest query position**. This is resolved by explicit scalar comparison
//! so it is independent of any lane order — a load-bearing part of the determinism contract.

use crate::error::{Error, Result};
use crate::hit::{BestHit, LocalSpan};
use crate::mode::Mode;
use crate::scoring::Scoring;
use crate::search::SearchType;

/// Sentinel for "unreachable" cells in the `E`/`F` gap matrices. Divided down from `i32::MIN`
/// so repeated `- gap` subtractions cannot underflow.
const NEG: i32 = i32::MIN / 4;

/// Per-mode control flags derived from [`Mode`]. Shared with the inter-sequence kernel
/// ([`crate::inter`]) so both kernels derive the mode's border/answer geometry identically.
pub(crate) struct Flags {
    /// Top row starts at 0 (leading query gap is free).
    pub(crate) top_row_free: bool,
    /// Left column starts at 0 (leading target gap is free).
    pub(crate) left_col_free: bool,
    /// The last row is part of the answer region (trailing query gap is free).
    pub(crate) answer_last_row: bool,
    /// The last column is part of the answer region (trailing target gap is free).
    pub(crate) answer_last_col: bool,
    /// Local mode: clamp every cell at 0 and take the answer over the whole matrix.
    pub(crate) local: bool,
}

impl Flags {
    pub(crate) fn for_mode(mode: Mode) -> Self {
        match mode {
            Mode::Nw => Flags {
                top_row_free: false,
                left_col_free: false,
                answer_last_row: false,
                answer_last_col: false,
                local: false,
            },
            Mode::Hw => Flags {
                top_row_free: true,
                left_col_free: false,
                answer_last_row: true,
                answer_last_col: false,
                local: false,
            },
            Mode::Ov => Flags {
                top_row_free: true,
                left_col_free: true,
                answer_last_row: true,
                answer_last_col: true,
                local: false,
            },
            Mode::Sw => Flags {
                top_row_free: true,
                left_col_free: true,
                answer_last_row: false,
                answer_last_col: false,
                local: true,
            },
            // The transpose of `Hw`: the query's ends are free (query prefix/suffix unaligned),
            // the target is aligned end to end. Answer over the last column `H[*][n]`.
            Mode::Shw => Flags {
                top_row_free: false,
                left_col_free: true,
                answer_last_row: false,
                answer_last_col: true,
                local: false,
            },
        }
    }
}

/// The penalty for a gap of length `len` under Opal's convention: `gap_open + (len - 1) *
/// gap_ext`, or 0 for an empty gap. Computed in `i64` so the intermediate cannot overflow; the
/// width proof guarantees the result fits `i32`.
pub(crate) fn gap_penalty(gap_open: i32, gap_ext: i32, len: usize) -> i32 {
    if len == 0 {
        0
    } else {
        (gap_open as i64 + (len as i64 - 1) * gap_ext as i64) as i32
    }
}

/// Running best cell under the tie-break: maximise score, then minimise `(grid_col, grid_row)`
/// which is `(target_end + 1, query_end + 1)`.
struct Best {
    score: i32,
    grid_row: usize,
    grid_col: usize,
}

impl Best {
    fn consider(&mut self, score: i32, grid_row: usize, grid_col: usize) {
        let better = score > self.score
            || (score == self.score && (grid_col, grid_row) < (self.grid_col, self.grid_row));
        if better {
            self.score = score;
            self.grid_row = grid_row;
            self.grid_col = grid_col;
        }
    }
}

/// Reusable working memory for the scalar DP. Held per worker thread inside
/// [`Scratch`](crate::Scratch) and reused across every target in a scan, so the hot path
/// allocates nothing once the buffers are sized. Sizing to a smaller problem than the capacity
/// never reallocates.
#[derive(Debug)]
pub(crate) struct DpBuffers {
    /// The full `(m + 1) * (n + 1)` `H` matrix, row-major.
    h: Vec<i32>,
    /// One column of the `F` (target-gap) matrix, carried down the rows.
    f: Vec<i32>,
}

impl DpBuffers {
    /// Empty buffers, grown on first use. Suitable for one-shot [`align_pair`].
    pub(crate) fn new() -> Self {
        DpBuffers {
            h: Vec::new(),
            f: Vec::new(),
        }
    }

    /// Buffers pre-sized for queries up to `max_query_len` against targets up to
    /// `max_target_len`, so no scan reallocates.
    pub(crate) fn with_capacity(max_query_len: usize, max_target_len: usize) -> Self {
        let cols = max_target_len + 1;
        DpBuffers {
            h: Vec::with_capacity((max_query_len + 1) * cols),
            f: Vec::with_capacity(cols),
        }
    }
}

/// Fill `buf.h` with the full `(m + 1) * (n + 1)` `H` matrix (row-major) for one query/target pair
/// under `flags`. `buf.f` is used as the carried `F` column. Callers must have validated that every
/// symbol is `< scoring.alphabet_len()`; this routine indexes the scoring matrix directly. `buf` is
/// resized (reusing capacity) and fully overwritten, so its prior contents are irrelevant.
#[inline]
fn fill_dp(query: &[u8], target: &[u8], scoring: &Scoring, flags: &Flags, buf: &mut DpBuffers) {
    let m = query.len();
    let n = target.len();
    let (gap_open, gap_ext) = (scoring.gap_open(), scoring.gap_ext());
    let cols = n + 1;
    let idx = |i: usize, j: usize| i * cols + j;

    // H[i][j]: best score aligning query[..i] with target[..j]. Reset to a clean zero matrix,
    // reusing the existing allocation whenever capacity allows.
    let h = &mut buf.h;
    h.clear();
    h.resize((m + 1) * cols, 0);
    // Border initialisation. Only H borders, E[i][0], and F[0][j] are ever read by the
    // recurrence; E/F border sentinels are folded in below.
    for j in 1..=n {
        h[idx(0, j)] = if flags.top_row_free {
            0
        } else {
            -gap_penalty(gap_open, gap_ext, j)
        };
    }
    for i in 1..=m {
        h[idx(i, 0)] = if flags.left_col_free {
            0
        } else {
            -gap_penalty(gap_open, gap_ext, i)
        };
    }

    // F[i][j] carried down the columns; F[0][j] = NEG (no target-gap can end at row 0).
    let f = &mut buf.f;
    f.clear();
    f.resize(cols, NEG);

    for i in 1..=m {
        let mut e = NEG; // E[i][0]: no query-gap can end at column 0.
        let qrow = scoring.score_row(query[i - 1] as usize); // bound-check once per row, not per cell
        for j in 1..=n {
            e = (h[idx(i, j - 1)] - gap_open).max(e - gap_ext);
            f[j] = (h[idx(i - 1, j)] - gap_open).max(f[j] - gap_ext);
            let sub = qrow[target[j - 1] as usize];
            let diag = h[idx(i - 1, j - 1)] + sub;
            let mut cell = diag.max(e).max(f[j]);
            if flags.local {
                cell = cell.max(0);
            }
            h[idx(i, j)] = cell;
        }
    }
}

/// Reduce the filled `H` matrix to `(score, query_end, target_end)` over the mode's answer region,
/// applying the lane-order-independent tie-break (maximise score, then minimise `(target_end,
/// query_end)`). `h` must be the `(m + 1) * (n + 1)` matrix from [`fill_dp`].
#[inline]
fn reduce_answer(
    h: &[i32],
    flags: &Flags,
    m: usize,
    n: usize,
) -> (i32, Option<usize>, Option<usize>) {
    let cols = n + 1;
    let idx = |i: usize, j: usize| i * cols + j;
    let mut best = Best {
        score: NEG,
        grid_row: 0,
        grid_col: 0,
    };
    if flags.local {
        for i in 0..=m {
            for j in 0..=n {
                best.consider(h[idx(i, j)], i, j);
            }
        }
    } else {
        best.consider(h[idx(m, n)], m, n);
        if flags.answer_last_row {
            for j in 0..=n {
                best.consider(h[idx(m, j)], m, j);
            }
        }
        if flags.answer_last_col {
            for i in 0..=m {
                best.consider(h[idx(i, n)], i, n);
            }
        }
    }
    (
        best.score,
        best.grid_row.checked_sub(1),
        best.grid_col.checked_sub(1),
    )
}

/// Run the scalar DP for one query/target pair, returning `(score, query_end, target_end)`.
///
/// Callers must have validated that every symbol is `< scoring.alphabet_len()`; this routine
/// indexes the scoring matrix directly. `buf` is resized (reusing capacity) and fully
/// overwritten, so its prior contents are irrelevant.
pub(crate) fn align_core(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    buf: &mut DpBuffers,
) -> (i32, Option<usize>, Option<usize>) {
    let flags = Flags::for_mode(mode);
    fill_dp(query, target, scoring, &flags, buf);
    reduce_answer(&buf.h, &flags, query.len(), target.len())
}

/// Reusable working memory for the pairwise entry points ([`align_pair_with`],
/// [`align_pair_position_max_with`]). Create one per worker thread and reuse it across calls so the
/// striped SIMD kernel and the scalar DP allocate nothing per call — the same contract the database
/// scan path gets from [`Scratch`](crate::Scratch). The one-shot [`align_pair`] /
/// [`align_pair_position_max`] wrap a throwaway `PairScratch`, so prefer the `_with` variants in a
/// hot loop (e.g. many reads against expected windows).
pub struct PairScratch {
    /// Striped kernel buffers, one set per element width (only the resolved width's set is filled).
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    s8: crate::striped::StripedBufs<i8>,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    s16: crate::striped::StripedBufs<i16>,
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    s32: crate::striped::StripedBufs<i32>,
    /// Scalar DP buffers (fallback path, and the `ScoreEnd`/`Alignment` cases).
    buf: DpBuffers,
}

impl PairScratch {
    /// A fresh scratch with empty buffers; they grow to the largest problem seen and are then
    /// reused without reallocating.
    #[must_use]
    pub fn new() -> Self {
        PairScratch {
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            s8: crate::striped::StripedBufs::new(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            s16: crate::striped::StripedBufs::new(),
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            s32: crate::striped::StripedBufs::new(),
            buf: DpBuffers::new(),
        }
    }
}

impl Default for PairScratch {
    fn default() -> Self {
        Self::new()
    }
}

/// Align a single query against a single target and return the best-scoring [`BestHit`].
///
/// `query` and `target` are pre-encoded alphabet indices (`0..scoring.alphabet_len()`), matching
/// Opal's convention. Results are, by the determinism contract, exactly what every backend returns.
/// This allocates its working memory per call; use [`align_pair_with`] with a reused [`PairScratch`]
/// in a hot loop.
///
/// # Errors
///
/// - [`Error::SymbolOutOfRange`] if any symbol is `>= scoring.alphabet_len()`.
/// - [`Error::ScoreRangeTooWide`] if the reachable score could overflow `i32` for these lengths.
pub fn align_pair(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
) -> Result<BestHit> {
    align_pair_with(
        &mut PairScratch::new(),
        query,
        target,
        scoring,
        mode,
        search_type,
    )
}

/// [`align_pair`] reusing a caller-provided [`PairScratch`], so a hot loop of pair alignments
/// allocates nothing per call. Same result and errors as [`align_pair`].
///
/// # Errors
///
/// See [`align_pair`].
pub fn align_pair_with(
    scratch: &mut PairScratch,
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
) -> Result<BestHit> {
    let alphabet_len = scoring.alphabet_len();
    for &sym in query.iter().chain(target.iter()) {
        if sym as usize >= alphabet_len {
            return Err(Error::SymbolOutOfRange {
                symbol: sym as usize,
                alphabet_len,
            });
        }
    }

    // Prove i32 suffices for these lengths before running the DP.
    let width = scoring.required_width(mode, query.len(), target.len())?;

    // Fast path: striped (Farrar) SIMD for a score-only search, at whichever of `i8`/`i16`/`i32` the
    // width proof selected. Bit-identical to the scalar kernel below (the same DP, narrower lanes),
    // so this only affects speed. `ScoreEnd`/`Alignment` use the scalar path.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    if search_type == SearchType::Score && !query.is_empty() && !target.is_empty() {
        if let Some(score) = crate::striped::farrar_score_simd(
            query,
            target,
            scoring,
            mode,
            width,
            &mut scratch.s8,
            &mut scratch.s16,
            &mut scratch.s32,
        ) {
            return Ok(BestHit {
                score,
                db_index: 0,
                query_end: None,
                target_end: None,
            });
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    let _ = width;

    let (score, query_end, target_end) = align_core(query, target, scoring, mode, &mut scratch.buf);

    let (query_end, target_end) = if search_type.tracks_end() {
        (query_end, target_end)
    } else {
        (None, None)
    };

    Ok(BestHit {
        score,
        db_index: 0,
        query_end,
        target_end,
    })
}

/// Align a batch of independent `(query, target)` pairs, writing one [`BestHit`] per pair — in input
/// order, with `db_index == pair index` — into `out` (cleared and reused). Distinct from
/// [`Database::scan`](crate::Database::scan) (one query against many targets) and from
/// [`align_pair`] (a single pair): here every pair has its **own** query and target.
///
/// Reuses one internal [`PairScratch`] across the whole batch, so — beyond growing `out` — it
/// allocates nothing per pair. Each pair's result is exactly what [`align_pair`] would return for it.
///
/// # Errors
///
/// Returns the first pair's error: [`Error::SymbolOutOfRange`] for an out-of-range symbol, or
/// [`Error::ScoreRangeTooWide`] if a pair's reachable score could overflow `i32`.
pub fn align_pairs<Q: AsRef<[u8]>, T: AsRef<[u8]>>(
    pairs: &[(Q, T)],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
    out: &mut Vec<BestHit>,
) -> Result<()> {
    out.clear();
    out.reserve(pairs.len());
    let mut scratch = PairScratch::new();
    for (index, (q, t)) in pairs.iter().enumerate() {
        let mut hit = align_pair_with(
            &mut scratch,
            q.as_ref(),
            t.as_ref(),
            scoring,
            mode,
            search_type,
        )?;
        hit.db_index = index;
        out.push(hit);
    }
    Ok(())
}

/// Align a single query against a single target in **local (Smith-Waterman)** mode and fill `out`
/// with the *per-target-position maxima*: `out[t]` is the best local alignment score **ending at
/// target position `t`** (0-based), i.e. `max_i H[i][t]` over the SW DP. Returns the best score and
/// its target end as a [`BestHit`] (`query_end` is always `None` — this search does not track the
/// query axis; `target_end` is `None` only when the best score is `0`, i.e. no positive alignment).
///
/// This is the primitive a bwa-mem-style **mate-rescue** consumer needs for a second-best-score
/// (`score2`) / mapping-quality computation: run it once for the read against its expected mate
/// window, then feed `out` (plus the returned best `score`/`target_end` and the scoring matrix's
/// maximum entry) to [`score2`] to obtain the competing peak. `out` is cleared and refilled.
///
/// The array is a pure per-column maximum: every value is `>= 0` (the SW floor), it needs no
/// tie-break, and it is deterministic by construction — bit-identical across every backend (a
/// striped SIMD kernel fills it at `i8`/`i16` width, the scalar DP otherwise), unlike an end
/// *position*. Positions are 0-based; a consumer wanting half-open coordinates adds 1.
///
/// # Errors
///
/// - [`Error::SymbolOutOfRange`] if any symbol is `>= scoring.alphabet_len()`.
/// - [`Error::ScoreRangeTooWide`] if the reachable score could overflow `i32` for these lengths.
pub fn align_pair_position_max(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    out: &mut Vec<i32>,
) -> Result<BestHit> {
    align_pair_position_max_with(&mut PairScratch::new(), query, target, scoring, out)
}

/// [`align_pair_position_max`] reusing a caller-provided [`PairScratch`], so a hot loop (e.g. a
/// mate-rescue consumer aligning many reads against their expected windows) allocates nothing per
/// call. Same result and errors as [`align_pair_position_max`].
///
/// # Errors
///
/// See [`align_pair_position_max`].
pub fn align_pair_position_max_with(
    scratch: &mut PairScratch,
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    out: &mut Vec<i32>,
) -> Result<BestHit> {
    let alphabet_len = scoring.alphabet_len();
    for &sym in query.iter().chain(target.iter()) {
        if sym as usize >= alphabet_len {
            return Err(Error::SymbolOutOfRange {
                symbol: sym as usize,
                alphabet_len,
            });
        }
    }

    // Per-position maxima / `score2` is a local-alignment concept (see `score2`); this entry is
    // SW-only. Prove i32 suffices for these lengths before running the DP.
    let mode = Mode::Sw;
    let width = scoring.required_width(mode, query.len(), target.len())?;
    out.clear();
    out.reserve(target.len()); // one entry per target column; avoids growth reallocs on either path

    // Fill `out` with the per-target-position maxima. The striped SIMD kernel does it at whichever
    // of `i8`/`i16`/`i32` the width proof selected; otherwise the scalar full-matrix DP. Both produce
    // the identical array (width proof), and the score / target end are derived from it below, so the
    // result is backend-independent.
    let filled = {
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        {
            !query.is_empty()
                && !target.is_empty()
                && crate::striped::farrar_position_max_simd(
                    query,
                    target,
                    scoring,
                    width,
                    out,
                    &mut scratch.s8,
                    &mut scratch.s16,
                    &mut scratch.s32,
                )
                .is_some()
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            let _ = width;
            false
        }
    };
    if !filled {
        let flags = Flags::for_mode(mode);
        fill_dp(query, target, scoring, &flags, &mut scratch.buf);
        let buf = &scratch.buf;
        let (m, n) = (query.len(), target.len());
        let cols = n + 1;
        for t in 0..n {
            // Best SW score ending at target position `t` = max over the query axis of H's column
            // `t + 1`. The `0` seed is the SW floor (an empty alignment); H[0][*] is already 0.
            let mut mx = 0i32;
            for i in 0..=m {
                let v = buf.h[i * cols + (t + 1)];
                if v > mx {
                    mx = v;
                }
            }
            out.push(mx);
        }
    }

    // Best score = the array maximum (>= 0). Target end = the smallest position achieving it, or
    // `None` when the best is 0 (the empty alignment ends nowhere) — matching the scalar oracle.
    let score = out.iter().copied().max().unwrap_or(0);
    let target_end = if score > 0 {
        out.iter().position(|&v| v == score)
    } else {
        None
    };
    Ok(BestHit {
        score,
        db_index: 0,
        query_end: None,
        target_end,
    })
}

/// The local (`SW`) alignment **span** — score plus the half-open aligned region in each sequence —
/// of `query` against `target`, without the alignment operations (see [`LocalSpan`]).
///
/// This is the cheap way to get *both* ends **and** both starts of the optimal local alignment: a
/// single forward Smith-Waterman pass that tracks, alongside the score, the start coordinate of the
/// best alignment ending at each cell. It uses `O(target)` working memory — no traceback matrix, no
/// linear-space checkpointing, and no CIGAR — where [`align`](crate::align) would. The reported span
/// is always **self-consistent**: `query[query_start..query_end]` aligned globally against
/// `target[target_start..target_end]` scores exactly `score`. `SW`/local only (starts are only
/// well-defined for a local alignment; the other modes' start coordinates are fixed by the mode).
///
/// The score and end positions equal those of `align_pair(.., Sw, ScoreEnd)`, and the full span
/// matches what [`align`](crate::align)'s traceback reports (verified exhaustively for all short
/// pairs and by property tests on larger ones). The *guaranteed* contract on ties is the
/// self-consistency above (the start belongs to the same optimal alignment as the end); the
/// tie-break is fixed and deterministic.
///
/// # Errors
///
/// - [`Error::SymbolOutOfRange`] if any symbol is `>= scoring.alphabet_len()`.
/// - [`Error::ScoreRangeTooWide`] if the reachable score could overflow `i32` for these lengths.
pub fn align_pair_span(query: &[u8], target: &[u8], scoring: &Scoring) -> Result<LocalSpan> {
    let alphabet_len = scoring.alphabet_len();
    for &sym in query.iter().chain(target.iter()) {
        if sym as usize >= alphabet_len {
            return Err(Error::SymbolOutOfRange {
                symbol: sym as usize,
                alphabet_len,
            });
        }
    }
    scoring.required_width(Mode::Sw, query.len(), target.len())?;

    let (m, n) = (query.len(), target.len());
    if m == 0 || n == 0 {
        return Ok(LocalSpan::EMPTY);
    }
    let (go, ge) = (scoring.gap_open(), scoring.gap_ext());
    let cols = n + 1;

    // Two ping-ponged H rows plus, in lock-step, the 0-based `(query_start, target_start)` of the
    // best local alignment ending at each cell. `F` is carried down each column (with its start);
    // `E` is carried across each row (a scalar). Start values are only ever *read* for cells whose
    // score is positive — a real alignment — so the `(0, 0)` fill for empty (score-0) cells is inert.
    let mut h_prev = vec![0i32; cols];
    let mut h_cur = vec![0i32; cols];
    let mut hs_prev = vec![(0usize, 0usize); cols];
    let mut hs_cur = vec![(0usize, 0usize); cols];
    let mut f = vec![NEG; cols];
    let mut fs = vec![(0usize, 0usize); cols];

    // Best cell under the SW tie-break: maximise score, then minimise `(target_end, query_end)` —
    // identical to `reduce_answer`/`BestHit`, so the reported ends match `align_pair(.., ScoreEnd)`.
    // The floor is `0` (the empty alignment).
    let mut best_score = 0i32;
    let mut best_row = 0usize; // grid row = exclusive query end
    let mut best_col = 0usize; // grid col = exclusive target end
    let mut best_start = (0usize, 0usize);

    for i in 1..=m {
        let qrow = scoring.score_row(query[i - 1] as usize);
        h_cur[0] = 0;
        hs_cur[0] = (0, 0);
        let mut e = NEG;
        let mut es = (0usize, 0usize);
        for j in 1..=n {
            // E: gap in the query (horizontal, consumes target). Open from H[i][j-1] or extend E.
            let e_open = h_cur[j - 1] - go;
            let e_ext = e - ge;
            if e_open >= e_ext {
                e = e_open;
                es = hs_cur[j - 1];
            } else {
                e = e_ext;
            }
            // F: gap in the target (vertical, consumes query). Open from H[i-1][j] or extend F.
            let f_open = h_prev[j] - go;
            let f_ext = f[j] - ge;
            if f_open >= f_ext {
                f[j] = f_open;
                fs[j] = hs_prev[j];
            } else {
                f[j] = f_ext;
            }
            // H = max(diag, E, F, 0), with the start of whichever term wins. Priority on ties:
            // diag > E > F (a fixed, deterministic order); the `0` floor means "empty" (fresh start).
            let diag = h_prev[j - 1] + qrow[target[j - 1] as usize];
            let (mut cell, mut start) = (0i32, (i - 1, j - 1)); // 0 => empty; start unused unless > 0
            if diag > cell {
                cell = diag;
                // A diagonal step onto a `0` (empty) predecessor *starts* the alignment at this pair;
                // otherwise it inherits the predecessor's start.
                start = if h_prev[j - 1] > 0 {
                    hs_prev[j - 1]
                } else {
                    (i - 1, j - 1)
                };
            }
            if e > cell {
                cell = e;
                start = es;
            }
            if f[j] > cell {
                cell = f[j];
                start = fs[j];
            }
            h_cur[j] = cell;
            hs_cur[j] = start;

            // Global best: same tie-break as `reduce_answer` (max score, then min (col, row)).
            let better = cell > best_score || (cell == best_score && (j, i) < (best_col, best_row));
            if cell > 0 && better {
                best_score = cell;
                best_row = i;
                best_col = j;
                best_start = start;
            }
        }
        std::mem::swap(&mut h_prev, &mut h_cur);
        std::mem::swap(&mut hs_prev, &mut hs_cur);
    }

    if best_score == 0 {
        return Ok(LocalSpan::EMPTY);
    }
    Ok(LocalSpan {
        score: best_score,
        query_start: best_start.0,
        query_end: best_row, // grid row is one past the last aligned query symbol
        target_start: best_start.1,
        target_end: best_col,
    })
}

/// bwa-mem-compatible **second-best score (`score2`)** from a per-target-position maxima array
/// (as produced by [`align_pair_position_max`]).
///
/// Returns the highest-scoring secondary *peak* whose target end lies **outside** the exclusion
/// window `[best_target_end - w, best_target_end + w]`, where `w = ceil(best_score / matrix_max)` —
/// `w` being a lower bound on the primary alignment's target span, so anything inside the window is
/// treated as overlapping the best hit. This reproduces bwa's `ksw.c` recipe exactly:
///
/// - only columns scoring `>= min_score` are considered (bwa's `minsc` threshold);
/// - contiguous above-threshold columns are **collapsed into a single peak**, keeping the run's
///   maximum and its position (bwa checks contiguity against the last kept peak *position*, so a
///   run whose maximum is not at its end can split — replicated here);
/// - among qualifying peaks, the highest score wins; ties go to the smallest target position.
///
/// Returns `(score2, target_pos)` (0-based `target_pos`) or `None` if no secondary peak qualifies
/// (equivalently bwa's `score2 = 0`). Returns `None` if `matrix_max <= 0` (no positive substitution
/// score, so the window is undefined). `matrix_max` is the **maximum substitution-matrix entry**
/// (`scoring.entry_bounds().1`), not the maximum observed score.
#[must_use]
pub fn score2(
    colmax: &[i32],
    best_score: i32,
    best_target_end: usize,
    matrix_max: i32,
    min_score: i32,
) -> Option<(i32, usize)> {
    if matrix_max <= 0 {
        return None;
    }
    // Half-width = ceil(best_score / matrix_max), computed in i64 (best_score >= 0 for SW).
    let w = ((best_score.max(0) as i64 + matrix_max as i64 - 1) / matrix_max as i64).max(0);
    let te = best_target_end as i64;
    let (low, high) = (te - w, te + w);

    let mut best2: Option<(i32, usize)> = None;
    // Fold a finished peak into `best2` if it falls outside the exclusion window.
    let consider = |best2: &mut Option<(i32, usize)>, score: i32, pos: usize| {
        let p = pos as i64;
        if (p < low || p > high) && best2.is_none_or(|(bs, _)| score > bs) {
            *best2 = Some((score, pos));
        }
    };

    // Current open peak: (running max score, position of that max).
    let mut cur: Option<(i32, usize)> = None;
    for (i, &v) in colmax.iter().enumerate() {
        if v < min_score {
            continue;
        }
        match cur {
            // Contiguous with the current peak's max position: extend, keeping the higher score.
            Some((cs, cp)) if cp + 1 == i => {
                if cs < v {
                    cur = Some((v, i));
                }
            }
            // A gap (sub-threshold column) or improvement past the stored position starts a new
            // peak; finish the previous one first.
            _ => {
                if let Some((cs, cp)) = cur {
                    consider(&mut best2, cs, cp);
                }
                cur = Some((v, i));
            }
        }
    }
    if let Some((cs, cp)) = cur {
        consider(&mut best2, cs, cp);
    }
    best2
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matrix(alphabet_len: usize, m: i32, x: i32) -> Vec<i32> {
        let mut v = vec![x; alphabet_len * alphabet_len];
        for i in 0..alphabet_len {
            v[i * alphabet_len + i] = m;
        }
        v
    }

    #[test]
    fn symbol_out_of_range_is_reported_not_panicked() {
        let s = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let err = align_pair(&[0, 4], &[0], &s, Mode::Sw, SearchType::ScoreEnd).unwrap_err();
        assert_eq!(
            err,
            Error::SymbolOutOfRange {
                symbol: 4,
                alphabet_len: 4
            }
        );
        // Out-of-range in the target is caught too.
        let err = align_pair(&[0], &[9], &s, Mode::Nw, SearchType::Score).unwrap_err();
        assert_eq!(
            err,
            Error::SymbolOutOfRange {
                symbol: 9,
                alphabet_len: 4
            }
        );
    }

    #[test]
    fn score_search_type_suppresses_end_positions() {
        let s = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let hit = align_pair(
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &s,
            Mode::Sw,
            SearchType::Score,
        )
        .unwrap();
        assert_eq!(hit.score, 8);
        assert_eq!((hit.query_end, hit.target_end), (None, None));
        // ...but the identical alignment under ScoreEnd reports them.
        let hit = align_pair(
            &[0, 1, 2, 3],
            &[0, 1, 2, 3],
            &s,
            Mode::Sw,
            SearchType::ScoreEnd,
        )
        .unwrap();
        assert_eq!((hit.query_end, hit.target_end), (Some(3), Some(3)));
    }

    // --- G1: the width proof bounds *every* cell, not just the final score (see DETERMINISM.md) ---

    /// Re-run the exact scalar recurrence and return the largest magnitude over all **real**
    /// (non-sentinel) `H`/`E`/`F` cells. Cells at or below `NEG/2` are −∞ sentinels and are
    /// excluded — in a narrow backend they map to the width's reserved `MIN` sentinel and never
    /// carry a real score. Deliberately in lock-step with `align_core`: this measures the very
    /// cells the kernel produces.
    fn max_real_cell_magnitude(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> i64 {
        fn note(max_mag: &mut i64, v: i32, sentinel_floor: i64) {
            let v = v as i64;
            if v > sentinel_floor {
                *max_mag = (*max_mag).max(v.abs());
            }
        }

        let m = query.len();
        let n = target.len();
        let flags = Flags::for_mode(mode);
        let (gap_open, gap_ext) = (scoring.gap_open(), scoring.gap_ext());
        let cols = n + 1;
        let idx = |i: usize, j: usize| i * cols + j;
        let sentinel_floor = (NEG / 2) as i64;
        let mut max_mag = 0i64;

        let mut h = vec![0i32; (m + 1) * cols];
        for j in 1..=n {
            h[idx(0, j)] = if flags.top_row_free {
                0
            } else {
                -gap_penalty(gap_open, gap_ext, j)
            };
        }
        for i in 1..=m {
            h[idx(i, 0)] = if flags.left_col_free {
                0
            } else {
                -gap_penalty(gap_open, gap_ext, i)
            };
        }
        for i in 0..=m {
            note(&mut max_mag, h[idx(i, 0)], sentinel_floor);
        }
        for j in 0..=n {
            note(&mut max_mag, h[idx(0, j)], sentinel_floor);
        }

        let mut f = vec![NEG; cols];
        for i in 1..=m {
            let mut e = NEG;
            for j in 1..=n {
                e = (h[idx(i, j - 1)] - gap_open).max(e - gap_ext);
                f[j] = (h[idx(i - 1, j)] - gap_open).max(f[j] - gap_ext);
                let sub = scoring.score(query[i - 1] as usize, target[j - 1] as usize);
                let diag = h[idx(i - 1, j - 1)] + sub;
                let mut cell = diag.max(e).max(f[j]);
                if flags.local {
                    cell = cell.max(0);
                }
                h[idx(i, j)] = cell;
                note(&mut max_mag, e, sentinel_floor);
                note(&mut max_mag, f[j], sentinel_floor);
                note(&mut max_mag, cell, sentinel_floor);
            }
        }
        max_mag
    }

    #[test]
    fn overlap_intermediate_cells_fit_i8_at_cr4_scale() {
        // With the tightened overlap width bound, a 40 nt vs 30 nt CR4-scored overlap proves to i8.
        // Verify against the real DP that every H/E/F cell actually fits i8 — otherwise a
        // saturating SIMD backend would diverge from the scalar oracle.
        let scoring = Scoring::new(2, vec![1, -2, -2, 1], 2, 2).unwrap();
        let q = [0u8, 1].repeat(20); // 40
        let t = [1u8, 0].repeat(15); // 30, phase-shifted for many mismatches
        let width = scoring.required_width(Mode::Ov, q.len(), t.len()).unwrap();
        assert_eq!(
            width,
            crate::ScoreWidth::I8,
            "CR4-scale overlap should prove to i8"
        );
        let mag = max_real_cell_magnitude(&q, &t, &scoring, Mode::Ov);
        assert!(
            mag <= width.max_abs(),
            "overlap cell magnitude {mag} exceeds i8 range"
        );
    }

    #[test]
    fn intermediate_cell_bound_holds_for_gap_dominated_global_alignment() {
        // A concrete case where E/F (not the diagonal) carry the extreme values: a global
        // alignment of two dissimilar sequences with a heavy mismatch and gap regime. The most
        // negative cell must still fit the proven width.
        let scoring = Scoring::new(2, vec![1, -8, -8, 1], 6, 2).unwrap();
        let q = vec![0u8; 20];
        let t = vec![1u8; 20];
        let width = scoring.required_width(Mode::Nw, q.len(), t.len()).unwrap();
        let max_mag = max_real_cell_magnitude(&q, &t, &scoring, Mode::Nw);
        assert!(
            max_mag <= width.max_abs(),
            "max cell magnitude {max_mag} exceeds {width} range ({})",
            width.max_abs()
        );
    }

    use proptest::prelude::*;

    fn scheme_and_pair() -> impl Strategy<Value = (usize, Vec<i32>, i32, i32, Vec<u8>, Vec<u8>)> {
        (2usize..=4)
            .prop_flat_map(|al| {
                let mat = prop::collection::vec(-8i32..=8, al * al);
                let gaps = (0i32..=10).prop_flat_map(|go| (Just(go), 0i32..=go));
                let q = prop::collection::vec(0u8..al as u8, 0..=30);
                let t = prop::collection::vec(0u8..al as u8, 0..=30);
                (Just(al), mat, gaps, q, t)
            })
            .prop_map(|(al, mat, (go, ge), q, t)| (al, mat, go, ge, q, t))
    }

    proptest! {
        /// G1: for every mode, the largest magnitude among *all real `H`/`E`/`F` cells* fits the
        /// width the proof selected — not merely the final score. This is exactly what lets a
        /// saturating i8/i16 backend be bit-identical to the wide scalar oracle. Input ranges are
        /// bounded so the reachable magnitude stays far above `NEG/2`, keeping real cells cleanly
        /// separable from sentinels.
        #[test]
        fn intermediate_cells_fit_the_proven_width(
            (al, mat, go, ge, q, t) in scheme_and_pair()
        ) {
            let scoring = Scoring::new(al, mat, go, ge).unwrap();
            for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
                let width = scoring.required_width(mode, q.len(), t.len()).unwrap();
                let max_mag = max_real_cell_magnitude(&q, &t, &scoring, mode);
                prop_assert!(
                    max_mag <= width.max_abs(),
                    "mode {}: max intermediate cell magnitude {} exceeds {} range ({})",
                    mode, max_mag, width, width.max_abs()
                );
            }
        }
    }

    // ---- Per-position maxima (`align_pair_position_max`) + `score2` ---------------------------

    /// Independent naive full-matrix SW oracle: returns the per-target-column maxima and the global
    /// best. Deliberately a separate implementation from `fill_dp` (Vec-of-Vec, explicit E/F).
    fn naive_sw_colmax(q: &[u8], t: &[u8], s: &Scoring) -> (Vec<i32>, i32) {
        let (m, n) = (q.len(), t.len());
        let (go, ge) = (s.gap_open(), s.gap_ext());
        let ninf = i32::MIN / 2;
        let mut h = vec![vec![0i32; n + 1]; m + 1];
        let mut e = vec![vec![ninf; n + 1]; m + 1];
        let mut f = vec![vec![ninf; n + 1]; m + 1];
        for i in 1..=m {
            for j in 1..=n {
                e[i][j] = (h[i][j - 1] - go).max(e[i][j - 1] - ge);
                f[i][j] = (h[i - 1][j] - go).max(f[i - 1][j] - ge);
                let sub = s.score(q[i - 1] as usize, t[j - 1] as usize);
                h[i][j] = (h[i - 1][j - 1] + sub).max(e[i][j]).max(f[i][j]).max(0);
            }
        }
        let mut colmax = vec![0i32; n];
        let mut best = 0i32;
        for (jt, cm) in colmax.iter_mut().enumerate() {
            let mut mx = 0i32;
            for row in h.iter() {
                mx = mx.max(row[jt + 1]);
            }
            *cm = mx;
            best = best.max(mx);
        }
        (colmax, best)
    }

    #[test]
    fn position_max_hand_computed_cases() {
        let s = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let mut out = Vec::new();

        // Perfect diagonal: best ending at target pos t is the t+1-long match run. Query axis is
        // not tracked, so query_end is None; target_end is the best column.
        let hit = align_pair_position_max(&[0, 1, 2, 3], &[0, 1, 2, 3], &s, &mut out).unwrap();
        assert_eq!(out, vec![2, 4, 6, 8]);
        assert_eq!(hit.score, 8);
        assert_eq!((hit.query_end, hit.target_end), (None, Some(3)));

        // Match only at the last target position.
        let hit = align_pair_position_max(&[0], &[1, 1, 0], &s, &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 2]);
        assert_eq!(hit.score, 2);
        assert_eq!((hit.query_end, hit.target_end), (None, Some(2)));

        // All-mismatch clamps to the SW floor everywhere; best 0 => no target end.
        let hit = align_pair_position_max(&[0, 0], &[1, 1], &s, &mut out).unwrap();
        assert_eq!(out, vec![0, 0]);
        assert_eq!(hit.score, 0);
        assert_eq!(hit.target_end, None);

        // Empty target => empty array; empty query => all-zero array. Both score 0, no target end.
        let hit = align_pair_position_max(&[0, 1], &[], &s, &mut out).unwrap();
        assert!(out.is_empty());
        assert_eq!(hit.score, 0);
        assert_eq!(hit.target_end, None);
        let hit = align_pair_position_max(&[], &[0, 1, 2], &s, &mut out).unwrap();
        assert_eq!(out, vec![0, 0, 0]);
        assert_eq!(hit.target_end, None);
    }

    #[test]
    fn position_max_out_of_range_symbol_is_reported() {
        let s = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let mut out = vec![1, 2, 3]; // must be left cleared/refilled, never read stale
        let err = align_pair_position_max(&[0, 5], &[0], &s, &mut out).unwrap_err();
        assert_eq!(
            err,
            Error::SymbolOutOfRange {
                symbol: 5,
                alphabet_len: 4
            }
        );
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        /// The per-position maxima array equals the independent naive full-matrix column maxima,
        /// its length is the target length, the returned hit equals the scalar SW oracle, and the
        /// global best equals the array maximum (all >= 0). Many varied alphabets/matrices/gaps.
        #[test]
        fn position_max_matches_naive_and_oracle((al, mat, go, ge, q, t) in scheme_and_pair()) {
            let s = Scoring::new(al, mat, go, ge).unwrap();
            let mut out = Vec::new();
            let hit = align_pair_position_max(&q, &t, &s, &mut out).unwrap();

            prop_assert_eq!(out.len(), t.len());
            let (want_col, want_best) = naive_sw_colmax(&q, &t, &s);
            prop_assert_eq!(&out, &want_col);

            // Score and target end match the scalar SW oracle; query end is not tracked here.
            let oracle = align_pair(&q, &t, &s, Mode::Sw, SearchType::ScoreEnd).unwrap();
            prop_assert_eq!(hit.score, oracle.score);
            prop_assert_eq!(hit.target_end, oracle.target_end);
            prop_assert_eq!(hit.query_end, None);

            let arr_best = out.iter().copied().max().unwrap_or(0);
            prop_assert_eq!(hit.score, arr_best);
            prop_assert_eq!(hit.score, want_best);
            prop_assert!(out.iter().all(|&v| v >= 0));
        }
    }

    #[test]
    fn align_pairs_matches_per_pair_and_reuses_out() {
        // Each batched result equals the independent one-shot `align_pair` for that pair (with
        // `db_index` = pair index), across modes/search types/widths and varied/empty geometries;
        // and the `out` buffer is cleared and reused between batches.
        let s_i8 = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let s_i16 = Scoring::new(4, matrix(4, 20, -5), 8, 2).unwrap();
        // match +3000 pushes the longer pairs to i32 (20·3000 = 60000 > i16), so the batch drives
        // the striped i32 path and the reused `PairScratch::s32` buffers too.
        let s_i32 = Scoring::new(4, matrix(4, 3000, -1000), 500, 100).unwrap();
        let mk = |seed: u64, len: usize| -> Vec<u8> {
            (0..len)
                .map(|i| ((seed.wrapping_mul(i as u64 + 1) >> 3) % 4) as u8)
                .collect()
        };
        let lens = [0usize, 1, 4, 9, 20];
        // A batch that mixes lengths (incl. empties) within one call.
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = lens
            .iter()
            .enumerate()
            .flat_map(|(a, &ql)| {
                lens.iter()
                    .enumerate()
                    .map(move |(b, &tl)| (mk(0x1000 + a as u64, ql), mk(0x2000 + b as u64, tl)))
            })
            .collect();

        let mut out = Vec::new();
        for s in [&s_i8, &s_i16, &s_i32] {
            for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    align_pairs(&pairs, s, mode, st, &mut out).unwrap();
                    assert_eq!(out.len(), pairs.len());
                    for (i, (q, t)) in pairs.iter().enumerate() {
                        let mut want = align_pair(q, t, s, mode, st).unwrap();
                        want.db_index = i;
                        assert_eq!(out[i], want, "{mode} {st} pair {i}");
                    }
                }
            }
        }

        // Reused `out`: a shorter second batch leaves no stale entries.
        align_pairs(&pairs[..3], &s_i8, Mode::Sw, SearchType::Score, &mut out).unwrap();
        assert_eq!(out.len(), 3);

        // Error propagation: an out-of-range symbol in any pair surfaces.
        let bad: Vec<(Vec<u8>, Vec<u8>)> = vec![(vec![0u8, 1], vec![0u8]), (vec![9u8], vec![0u8])];
        assert!(matches!(
            align_pairs(&bad, &s_i8, Mode::Nw, SearchType::Score, &mut out),
            Err(Error::SymbolOutOfRange { .. })
        ));
    }

    #[test]
    fn pair_scratch_reuse_matches_one_shot() {
        // A single reused `PairScratch` must give bit-identical results to the allocating one-shot
        // entry across a sequence of varying sizes and modes — the resize-grow-then-shrink pattern
        // is exactly where a stale-buffer bug would show. Widths span i8 and i16.
        let s_i8 = Scoring::new(4, matrix(4, 2, -1), 2, 1).unwrap();
        let s_i16 = Scoring::new(4, matrix(4, 20, -5), 8, 2).unwrap(); // crosses into i16
        let mk = |seed: u64, len: usize| -> Vec<u8> {
            (0..len)
                .map(|i| ((seed.wrapping_mul(i as u64 + 1) >> 3) % 4) as u8)
                .collect()
        };
        // Deliberately non-monotonic lengths: big, small, big, empty, ... to force grow + shrink.
        let lens = [40usize, 3, 55, 1, 30, 0, 12, 48, 2];

        let mut scratch = PairScratch::new();
        let mut out_reused = Vec::new();
        let mut out_oneshot = Vec::new();
        for (k, &ql) in lens.iter().enumerate() {
            let tl = lens[(k + 3) % lens.len()];
            for s in [&s_i8, &s_i16] {
                let q = mk(0x9e37_79b1 + k as u64, ql);
                let t = mk(0x1234_5678 + k as u64, tl);
                for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
                    for st in [SearchType::Score, SearchType::ScoreEnd] {
                        let reused = align_pair_with(&mut scratch, &q, &t, s, mode, st).unwrap();
                        let oneshot = align_pair(&q, &t, s, mode, st).unwrap();
                        assert_eq!(
                            reused, oneshot,
                            "align_pair_with {mode} {st} ql={ql} tl={tl}"
                        );
                    }
                }
                // Position-max entry too (SW-only).
                let reused =
                    align_pair_position_max_with(&mut scratch, &q, &t, s, &mut out_reused).unwrap();
                let oneshot = align_pair_position_max(&q, &t, s, &mut out_oneshot).unwrap();
                assert_eq!(reused, oneshot, "position_max_with hit ql={ql} tl={tl}");
                assert_eq!(
                    out_reused, out_oneshot,
                    "position_max_with array ql={ql} tl={tl}"
                );
            }
        }
    }

    /// Reference `score2`: a structurally different transcription of bwa's `ksw.c` recipe — build
    /// the explicit peak list (`b[]`), then filter by the exclusion window. Cross-checks the
    /// streaming `score2` implementation.
    fn score2_reference(
        colmax: &[i32],
        best_score: i32,
        te: usize,
        matrix_max: i32,
        min_score: i32,
    ) -> Option<(i32, usize)> {
        if matrix_max <= 0 {
            return None;
        }
        let mut peaks: Vec<(i32, usize)> = Vec::new();
        for (i, &v) in colmax.iter().enumerate() {
            if v < min_score {
                continue;
            }
            match peaks.last().copied() {
                Some((_, p)) if p + 1 == i => {
                    let last = peaks.last_mut().unwrap();
                    if last.0 < v {
                        *last = (v, i);
                    }
                }
                _ => peaks.push((v, i)),
            }
        }
        let w = ((best_score.max(0) as i64 + matrix_max as i64 - 1) / matrix_max as i64).max(0);
        let (low, high) = (te as i64 - w, te as i64 + w);
        let mut best2: Option<(i32, usize)> = None;
        for (score, pos) in peaks {
            let p = pos as i64;
            if (p < low || p > high) && best2.is_none_or(|(bs, _)| score > bs) {
                best2 = Some((score, pos));
            }
        }
        best2
    }

    #[test]
    fn score2_hand_computed_cases() {
        // Two distinct peaks; the primary (pos 1) is inside the window, the far one survives.
        let colmax = [0, 5, 0, 0, 0, 0, 0, 0, 4, 0];
        assert_eq!(score2(&colmax, 5, 1, 1, 1), Some((4, 8))); // w=5, window [-4,6]; 8 outside

        // The literally-second-highest cell (the 4 adjacent to the max 5) is COLLAPSED into the
        // primary peak, not returned as score2 — the whole point of the window+collapse.
        let colmax = [0, 3, 5, 4, 0];
        assert_eq!(score2(&colmax, 5, 2, 5, 1), None); // w=1, window [1,3]; only peak is inside

        // The collapse-split quirk: a contiguous run whose max is not at its end splits into two
        // peaks (bwa checks contiguity against the last stored max position).
        let colmax = [3, 5, 4, 6];
        assert_eq!(score2(&colmax, 1, 20, 1, 1), Some((6, 3))); // peaks (5,1),(6,3); best is (6,3)

        // Ties go to the smallest target position (strict `>` on replacement).
        let colmax = [7, 0, 7];
        assert_eq!(score2(&colmax, 1, 10, 1, 1), Some((7, 0)));

        // No qualifying peak => None; and a non-positive matrix max => None.
        assert_eq!(score2(&[0, 0, 0], 0, 0, 1, 1), None);
        assert_eq!(score2(&[5, 5], 5, 0, 0, 1), None);

        // Threshold gates out sub-minsc columns. matrix_max=9 => w=1, window [5,7].
        let colmax = [2, 0, 0, 0, 0, 0, 9];
        assert_eq!(score2(&colmax, 9, 6, 9, 5), None); // pos6 (best) in window; pos0=2 < minsc 5
        assert_eq!(score2(&colmax, 9, 6, 9, 1), Some((2, 0))); // minsc 1: pos0 a peak, outside
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(600))]

        /// The streaming `score2` matches the explicit-peak-list reference on random arrays,
        /// windows, thresholds, and matrix maxima — including the degenerate cases.
        #[test]
        fn score2_matches_reference(
            colmax in prop::collection::vec(0i32..=20, 0..=40),
            best_score in 0i32..=40,
            te in 0usize..=40,
            matrix_max in -2i32..=8,
            min_score in 1i32..=6,
        ) {
            prop_assert_eq!(
                score2(&colmax, best_score, te, matrix_max, min_score),
                score2_reference(&colmax, best_score, te, matrix_max, min_score)
            );
        }
    }
}
