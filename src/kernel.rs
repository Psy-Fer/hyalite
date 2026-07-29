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
//! | `NW` | penalised            | penalised             | corner `(m, n)` |
//! | `HW` | **free**             | penalised             | last row (query fully aligned; target window free) |
//! | `OV` | **free**             | **free**              | last row ∪ last column (overlap) |
//! | `SW` | free (local)         | free (local)          | every cell, clamped at 0 |
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
use crate::hit::BestHit;
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
    let m = query.len();
    let n = target.len();
    let flags = Flags::for_mode(mode);
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
        }
    }

    // Answer region.
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

    let query_end = best.grid_row.checked_sub(1);
    let target_end = best.grid_col.checked_sub(1);
    (best.score, query_end, target_end)
}

/// Align a single query against a single target and return the best-scoring [`BestHit`].
///
/// `query` and `target` are pre-encoded alphabet indices (`0..scoring.alphabet_len()`), matching
/// Opal's convention. This is the day-one public pair-alignment entry point; it is currently
/// scalar-backed (a SIMD striped backend lands in a later milestone) and its results are, by the
/// determinism contract, exactly what every future backend will return.
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
    scoring.required_width(mode, query.len(), target.len())?;

    let mut buf = DpBuffers::new();
    let (score, query_end, target_end) = align_core(query, target, scoring, mode, &mut buf);

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
            for mode in [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov] {
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
}
