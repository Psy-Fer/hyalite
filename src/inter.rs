//! Inter-sequence (Rognes / SWIPE) alignment kernel.
//!
//! One query is aligned against a whole batch of database sequences at once, with database
//! sequence `k` occupying SIMD lane `k`. The DP recurrence is identical to the scalar kernel
//! ([`crate::kernel`]); only the data layout changes — each column step advances all lanes
//! together. The arithmetic runs at **i8 saturating** width (`DETERMINISM.md` §1), so this path
//! is used only for databases whose proven [`ScoreWidth`](crate::ScoreWidth) is `I8` and whose
//! alphabet fits a byte shuffle (`alphabet_len <= 16`); everything else stays on the scalar path.
//!
//! The kernel computes **scores only** — no in-vector position tracking. For `ScoreEnd` the end
//! positions of the single winning sequence are recovered with one scalar re-alignment
//! ([`crate::kernel::align_core`]), which is bit-identical to the oracle by construction (see
//! `DETERMINISM.md`, "end positions").
//!
//! The kernel is generic over a [`Lanes`] backend. This module provides the safe scalar reference
//! impl [`ScalarLanes`]; the SIMD impls (SSE4.1, AVX2) are added in later milestones and must pass
//! the same differential tests.

// M2a: the kernel is exercised only by this module's differential tests; it is wired into
// `Database::scan` dispatch in M2b, at which point this allow is removed.
#![allow(dead_code)]

use crate::hit::BestHit;
use crate::kernel::{self, Flags, gap_penalty};
use crate::mode::Mode;
use crate::scoring::Scoring;
use crate::search::SearchType;

/// The −∞ sentinel at i8 width: unreachable cells. Real scores are provably in `[-127, 127]` for
/// an `I8`-width database, so they never collide with it.
const NEG8: i8 = i8::MIN;

/// The maximum alphabet length the byte-shuffle substitution lookup supports.
pub(crate) const MAX_SHUFFLE_ALPHABET: usize = 16;

/// A SIMD lane backend over `i8` elements with saturating arithmetic. All ops are element-wise
/// across `LANES` lanes. The scalar reference ([`ScalarLanes`]) implements this without `unsafe`;
/// SIMD impls specialise each method to intrinsics.
pub(crate) trait Lanes {
    /// Number of lanes processed at once.
    const LANES: usize;
    /// The vector type holding `LANES` `i8` values.
    type V: Copy;

    /// Broadcast one value to every lane.
    fn splat(v: i8) -> Self::V;
    /// Element-wise saturating add.
    fn add_sat(a: Self::V, b: Self::V) -> Self::V;
    /// Element-wise saturating subtract.
    fn sub_sat(a: Self::V, b: Self::V) -> Self::V;
    /// Element-wise signed max.
    fn max(a: Self::V, b: Self::V) -> Self::V;
    /// Per lane: `mask` lane non-zero selects `a`, else `b`.
    fn select(mask: Self::V, a: Self::V, b: Self::V) -> Self::V;
    /// Load `LANES` values from the start of `src`.
    fn load(src: &[i8]) -> Self::V;
    /// Store `LANES` values to the start of `dst`.
    fn store(v: Self::V, dst: &mut [i8]);
    /// Per lane `k`: `table[indices[k]]`. `table` has `<= 16` entries; `indices[k] < table.len()`.
    fn shuffle_lookup(table: &[i8], indices: &[u8]) -> Self::V;
}

/// The safe scalar reference lane backend, `N` lanes wide, backed by `[i8; N]`. Exercised at
/// several widths in tests to confirm lane-count independence before any SIMD exists.
#[derive(Clone, Copy)]
pub(crate) struct ScalarLanes<const N: usize>;

impl<const N: usize> Lanes for ScalarLanes<N> {
    const LANES: usize = N;
    type V = [i8; N];

    fn splat(v: i8) -> [i8; N] {
        [v; N]
    }
    fn add_sat(a: [i8; N], b: [i8; N]) -> [i8; N] {
        let mut o = [0i8; N];
        for k in 0..N {
            o[k] = a[k].saturating_add(b[k]);
        }
        o
    }
    fn sub_sat(a: [i8; N], b: [i8; N]) -> [i8; N] {
        let mut o = [0i8; N];
        for k in 0..N {
            o[k] = a[k].saturating_sub(b[k]);
        }
        o
    }
    fn max(a: [i8; N], b: [i8; N]) -> [i8; N] {
        let mut o = [0i8; N];
        for k in 0..N {
            o[k] = a[k].max(b[k]);
        }
        o
    }
    fn select(mask: [i8; N], a: [i8; N], b: [i8; N]) -> [i8; N] {
        let mut o = [0i8; N];
        for k in 0..N {
            o[k] = if mask[k] != 0 { a[k] } else { b[k] };
        }
        o
    }
    fn load(src: &[i8]) -> [i8; N] {
        let mut v = [0i8; N];
        v.copy_from_slice(&src[..N]);
        v
    }
    fn store(v: [i8; N], dst: &mut [i8]) {
        dst[..N].copy_from_slice(&v);
    }
    fn shuffle_lookup(table: &[i8], indices: &[u8]) -> [i8; N] {
        let mut o = [0i8; N];
        for k in 0..N {
            o[k] = table[indices[k] as usize];
        }
        o
    }
}

/// Whether the inter-sequence i8 kernel can be used for a database with this width/alphabet.
pub(crate) fn kernel_applies(width: crate::ScoreWidth, alphabet_len: usize) -> bool {
    width == crate::ScoreWidth::I8 && alphabet_len <= MAX_SHUFFLE_ALPHABET
}

/// Build the per-query-position substitution profile: `profile[i * al + s] = score(query[i], s)`
/// as `i8`. Valid only for an `I8`-width database, where every entry fits.
fn build_profile(query: &[u8], scoring: &Scoring) -> Vec<i8> {
    let al = scoring.alphabet_len();
    let mut profile = vec![0i8; query.len() * al];
    for (i, &q) in query.iter().enumerate() {
        for s in 0..al {
            profile[i * al + s] = scoring.score(q as usize, s) as i8;
        }
    }
    profile
}

/// Compute the per-lane best score for one batch of `batch.len()` (`<= L::LANES`) targets.
/// Writes `batch.len()` scores into `out`. Score-only; end positions are recovered elsewhere.
#[allow(clippy::too_many_arguments)] // an inter-sequence DP inherently takes many parameters
fn scan_batch<L: Lanes>(
    query: &[u8],
    profile: &[i8],
    al: usize,
    batch: &[&[u8]],
    go: i8,
    ge: i8,
    flags: &Flags,
    out: &mut [i8],
) {
    let lanes = L::LANES;
    let qlen = query.len();

    // Per-lane target lengths (0 for unused lanes) and the batch's padded column count `w`.
    let mut lens = vec![0usize; lanes];
    for (k, t) in batch.iter().enumerate() {
        lens[k] = t.len();
    }
    let w = lens.iter().copied().max().unwrap_or(0);

    // Per-column target residues, `residues[(j-1) * lanes + k]` = target k's residue at column j
    // (1-based), padded with 0 where the lane is shorter.
    let mut residues = vec![0u8; w * lanes];
    for (k, t) in batch.iter().enumerate() {
        for (j0, &r) in t.iter().enumerate() {
            residues[j0 * lanes + k] = r;
        }
    }

    // Column masks: `mask_le[j]` lane = all-ones iff `j <= len_k`; `mask_eq[j]` iff `j == len_k`.
    let mut scratch = vec![0i8; lanes];
    let make_mask = |pred: &dyn Fn(usize) -> bool, scratch: &mut [i8]| {
        for (k, slot) in scratch.iter_mut().enumerate() {
            *slot = if pred(k) { -1 } else { 0 };
        }
        L::load(scratch)
    };
    let mask_le: Vec<L::V> = (0..=w)
        .map(|j| make_mask(&|k| j <= lens[k], &mut scratch))
        .collect();
    let mask_eq: Vec<L::V> = (0..=w)
        .map(|j| make_mask(&|k| j == lens[k], &mut scratch))
        .collect();

    let go_v = L::splat(go);
    let ge_v = L::splat(ge);
    let zero = L::splat(0);
    let neg = L::splat(NEG8);

    // Row 0 (H[0][*]) and the down-carried F column (F[0][*] = −∞).
    let mut h_prev: Vec<L::V> = (0..=w)
        .map(|j| {
            if flags.top_row_free {
                zero
            } else {
                L::splat(-gap_penalty(go as i32, ge as i32, j) as i8)
            }
        })
        .collect();
    let mut h_cur: Vec<L::V> = vec![zero; w + 1];
    let mut f: Vec<L::V> = vec![neg; w + 1];

    // Answer accumulators updated during the sweep (need every row): SW's running max and OV's
    // best last-column cell. Both are floored at 0 by their modes.
    let mut sw_ans = zero;
    let mut ov_lastcol = zero;

    for i in 1..=qlen {
        h_cur[0] = if flags.left_col_free {
            zero
        } else {
            L::splat(-gap_penalty(go as i32, ge as i32, i) as i8)
        };
        let profile_row = &profile[(i - 1) * al..(i - 1) * al + al];
        let mut e = neg; // E[i][0]
        for j in 1..=w {
            e = L::max(L::sub_sat(h_cur[j - 1], go_v), L::sub_sat(e, ge_v));
            f[j] = L::max(L::sub_sat(h_prev[j], go_v), L::sub_sat(f[j], ge_v));
            let sub = L::shuffle_lookup(profile_row, &residues[(j - 1) * lanes..j * lanes]);
            let diag = L::add_sat(h_prev[j - 1], sub);
            let mut cell = L::max(diag, L::max(e, f[j]));
            if flags.local {
                cell = L::max(cell, zero);
                sw_ans = L::select(mask_le[j], L::max(sw_ans, cell), sw_ans);
            }
            if flags.answer_last_col {
                ov_lastcol = L::select(mask_eq[j], L::max(ov_lastcol, cell), ov_lastcol);
            }
            h_cur[j] = cell;
        }
        std::mem::swap(&mut h_prev, &mut h_cur);
    }

    // After the sweep `h_prev` holds the last query row (row 0 if the query was empty).
    let last_row = &h_prev;

    let ans = if flags.local {
        sw_ans
    } else if flags.answer_last_row && flags.answer_last_col {
        // OV: best of the last row (j <= len_k) and the accumulated last column.
        let mut lastrow = last_row[0];
        for j in 1..=w {
            lastrow = L::select(mask_le[j], L::max(lastrow, last_row[j]), lastrow);
        }
        L::max(lastrow, ov_lastcol)
    } else if flags.answer_last_row {
        // HW: best of the last row over j <= len_k (including the free-or-penalised j = 0 border).
        let mut hw = last_row[0];
        for j in 1..=w {
            hw = L::select(mask_le[j], L::max(hw, last_row[j]), hw);
        }
        hw
    } else {
        // NW: exactly H[qlen][len_k] per lane.
        let mut nw = last_row[0]; // covers len_k == 0
        for j in 1..=w {
            nw = L::select(mask_eq[j], last_row[j], nw);
        }
        nw
    };

    let mut ans_arr = vec![0i8; lanes];
    L::store(ans, &mut ans_arr);
    out.copy_from_slice(&ans_arr[..batch.len()]);
}

/// Scan `query` against every sequence using the inter-sequence kernel with lane backend `L`.
///
/// Requires an `I8`-width, `alphabet_len <= 16` database (see [`kernel_applies`]); the caller is
/// responsible for that gate. Returns the same [`BestHit`] the scalar path would: highest score,
/// smallest `db_index` on a tie, and — for `ScoreEnd` — the winner's end positions via a single
/// scalar re-alignment.
pub(crate) fn scan_batched<L: Lanes>(
    sequences: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
    query: &[u8],
) -> BestHit {
    let al = scoring.alphabet_len();
    let flags = Flags::for_mode(mode);
    let profile = build_profile(query, scoring);
    let (go, ge) = (scoring.gap_open() as i8, scoring.gap_ext() as i8);

    // Per-sequence scores, then a scalar argmax over the database index (smallest index on tie).
    let mut best_score = i8::MIN;
    let mut best_index = 0usize;
    let mut lane_scores = vec![0i8; L::LANES];
    let refs: Vec<&[u8]> = sequences.iter().map(Vec::as_slice).collect();

    for (batch_start, batch) in refs
        .chunks(L::LANES)
        .enumerate()
        .map(|(b, c)| (b * L::LANES, c))
    {
        scan_batch::<L>(
            query,
            &profile,
            al,
            batch,
            go,
            ge,
            &flags,
            &mut lane_scores[..batch.len()],
        );
        for (k, &s) in lane_scores[..batch.len()].iter().enumerate() {
            if s > best_score {
                best_score = s;
                best_index = batch_start + k;
            }
        }
    }

    let (query_end, target_end) = if search_type.tracks_end() {
        // Recover the winner's ends with one scalar alignment — bit-identical to the oracle.
        let mut buf = kernel::DpBuffers::new();
        let (score, qe, te) =
            kernel::align_core(query, &sequences[best_index], scoring, mode, &mut buf);
        debug_assert_eq!(
            score, best_score as i32,
            "inter-sequence score disagrees with scalar re-alignment for the winner"
        );
        (qe, te)
    } else {
        (None, None)
    };

    BestHit {
        score: best_score as i32,
        db_index: best_index,
        query_end,
        target_end,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, Scratch};
    use proptest::prelude::*;

    /// Reference scan via the public scalar path.
    fn scalar_scan(
        seqs: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        st: SearchType,
        query: &[u8],
    ) -> BestHit {
        let db = Database::builder()
            .sequences(seqs)
            .scoring(scoring.clone())
            .mode(mode)
            .search_type(st)
            .max_query_len(query.len().max(1))
            .build()
            .unwrap();
        let mut scratch = Scratch::new(&db);
        db.scan(&mut scratch, query)
    }

    const MODES: [Mode; 4] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov];

    /// (alphabet_len<=4, small match/mismatch matrix, small gaps, seqs, query) — kept in ranges
    /// where the score width is provably `I8`.
    fn scenario() -> impl Strategy<Value = (Scoring, Vec<Vec<u8>>, Vec<u8>)> {
        (2usize..=4)
            .prop_flat_map(|al| {
                let mat = prop::collection::vec(-4i32..=4, al * al);
                let gaps = (0i32..=4).prop_flat_map(|go| (Just(go), 0i32..=go));
                let seqs =
                    prop::collection::vec(prop::collection::vec(0u8..al as u8, 0..=10), 1..=9);
                let q = prop::collection::vec(0u8..al as u8, 0..=10);
                (Just(al), mat, gaps, seqs, q)
            })
            .prop_map(|(al, mat, (go, ge), seqs, q)| {
                (Scoring::new(al, mat, go, ge).unwrap(), seqs, q)
            })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(400))]

        /// The inter-sequence kernel is bit-identical to the scalar oracle across all modes, both
        /// search types, and several lane counts — the lane-count independence the determinism
        /// contract demands, checked before any real SIMD exists.
        #[test]
        fn inter_sequence_matches_scalar((scoring, seqs, q) in scenario()) {
            // Restrict to the kernel's domain: I8 width for every mode under test.
            let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
            for mode in MODES {
                prop_assume!(
                    kernel_applies(
                        scoring.required_width(mode, q.len(), max_t).unwrap(),
                        scoring.alphabet_len()
                    )
                );
            }

            for mode in MODES {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    let want = scalar_scan(&seqs, &scoring, mode, st, &q);
                    let got1 = scan_batched::<ScalarLanes<1>>(&seqs, &scoring, mode, st, &q);
                    let got4 = scan_batched::<ScalarLanes<4>>(&seqs, &scoring, mode, st, &q);
                    let got8 = scan_batched::<ScalarLanes<8>>(&seqs, &scoring, mode, st, &q);
                    prop_assert_eq!(got1, want, "1 lane, {} {}", mode, st);
                    prop_assert_eq!(got4, want, "4 lanes, {} {}", mode, st);
                    prop_assert_eq!(got8, want, "8 lanes, {} {}", mode, st);
                }
            }
        }
    }
}
