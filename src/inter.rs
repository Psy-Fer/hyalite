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
//! The kernel is generic over a [`Lanes`] backend: [`sse41::Sse41`] and [`avx2::Avx2`] on x86-64,
//! [`neon::Neon`] on aarch64, and a test-only `ScalarLanes` reference that the differential tests
//! check every SIMD backend against. Query-independent packing lives in [`PackedDb`] (built once);
//! per-scan working memory lives in [`SimdScratch`], so the hot path allocates nothing.

// The generic kernel is live in production via SSE4.1/AVX2 (x86-64) and NEON (aarch64). On an
// exotic target with no SIMD backend it would be unused; CI only builds x86-64 and aarch64, so we
// only silence dead code there. `ScalarLanes` is a test-only reference lane (gated `cfg(test)`).
#![cfg_attr(
    not(any(target_arch = "x86_64", target_arch = "aarch64")),
    allow(dead_code)
)]

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

/// Answer-position tracking for `ScoreEnd`, split from [`Lanes`] so a backend can add it separately
/// (the score-only path needs only [`Lanes`]). Positions live in a parallel `i16` domain because
/// answer-cell coordinates can exceed the `i8` score range.
pub(crate) trait LanesEnds: Lanes {
    /// A vector of `LANES` `i16` values (answer-cell coordinates).
    type PosV: Copy;

    /// Broadcast one `i16` to every position lane.
    fn pos_splat(v: i16) -> Self::PosV;
    /// Store `LANES` `i16` values to the start of `dst`.
    fn pos_store(v: Self::PosV, dst: &mut [i16]);

    /// Lexicographic answer update, per lane, for lanes where `active` (an `0x00`/`0xFF` mask) is
    /// set: replace `(best_score, best_col, best_row)` with `(cell, col, row)` when
    /// `cell > best_score`, or when `cell == best_score` and `(col, row)` is lexicographically less
    /// than `(best_col, best_row)`. `col`/`row` are broadcast scalars. Returns the updated triple.
    ///
    /// This is exactly the scalar tie-break (smallest target end, then query end) reproduced across
    /// lanes, so the reported end is independent of lane order.
    fn update_answer(
        active: Self::V,
        best_score: Self::V,
        best_col: Self::PosV,
        best_row: Self::PosV,
        cell: Self::V,
        col: i16,
        row: i16,
    ) -> (Self::V, Self::PosV, Self::PosV);
}

/// The safe scalar reference lane backend, `N` lanes wide, backed by `[i8; N]`. Test-only: it is
/// the differential oracle for the SIMD lanes and exercises lane-count independence at several
/// widths. Production always uses a real SIMD backend or the pairwise scalar kernel.
#[cfg(test)]
#[derive(Clone, Copy)]
pub(crate) struct ScalarLanes<const N: usize>;

#[cfg(test)]
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

#[cfg(test)]
impl<const N: usize> LanesEnds for ScalarLanes<N> {
    type PosV = [i16; N];

    fn pos_splat(v: i16) -> [i16; N] {
        [v; N]
    }
    fn pos_store(v: [i16; N], dst: &mut [i16]) {
        dst[..N].copy_from_slice(&v);
    }
    fn update_answer(
        active: [i8; N],
        best_score: [i8; N],
        best_col: [i16; N],
        best_row: [i16; N],
        cell: [i8; N],
        col: i16,
        row: i16,
    ) -> ([i8; N], [i16; N], [i16; N]) {
        let mut s = best_score;
        let mut c = best_col;
        let mut r = best_row;
        for k in 0..N {
            if active[k] == 0 {
                continue;
            }
            let cv = cell[k];
            let better = cv > s[k] || (cv == s[k] && (col < c[k] || (col == c[k] && row < r[k])));
            if better {
                s[k] = cv;
                c[k] = col;
                r[k] = row;
            }
        }
        (s, c, r)
    }
}

/// Whether the inter-sequence i8 kernel can be used for a database with this width/alphabet.
pub(crate) fn kernel_applies(width: crate::ScoreWidth, alphabet_len: usize) -> bool {
    width == crate::ScoreWidth::I8 && alphabet_len <= MAX_SHUFFLE_ALPHABET
}

/// The kernel data layout for the substitution scores, reported by
/// [`Database::layout`](crate::Database::layout). Like the backend and score width, the layout is
/// a **performance choice only** — it never changes results (see `DETERMINISM.md`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Layout {
    /// General layout: a per-column byte-shuffle gather of substitution scores from a per-letter
    /// query profile. The standard Rognes inner loop; works for any database.
    Gathered,
    /// Small-fixed-database layout: a precomputed query-letter × database-column score table, so
    /// each cell's substitution vector is a direct load with no gather. Chosen automatically when
    /// the table is small enough to stay cache-resident (see [`Database::builder`](crate::Database::builder)).
    Precomputed,
}

impl core::fmt::Display for Layout {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Layout::Gathered => f.write_str("gathered"),
            Layout::Precomputed => f.write_str("precomputed"),
        }
    }
}

/// How a [`Database`](crate::Database) chooses its kernel [`Layout`]: automatically from the
/// database size, or forced (for benchmarking or pinning behaviour).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LayoutChoice {
    /// Pick [`Layout::Precomputed`] when the score table stays cache-resident, else
    /// [`Layout::Gathered`].
    #[default]
    Auto,
    /// Force a specific layout.
    Force(Layout),
}

/// Precomputed-table budget: `Auto` picks [`Layout::Precomputed`] when the estimated table size
/// (`sum over batches of alphabet_len × width × lanes` bytes) is at most this. 256 KiB stays
/// comfortably within L2; the CR4 adapter table is ~10 KiB. Overridable via the builder.
const PRECOMPUTED_MAX_BYTES: usize = 256 * 1024;

/// Estimated `Precomputed` table size in bytes for this database and lane count.
fn precomputed_table_bytes(sequences: &[Vec<u8>], lanes: usize, alphabet_len: usize) -> usize {
    sequences
        .chunks(lanes)
        .map(|chunk| {
            let w = chunk.iter().map(Vec::len).max().unwrap_or(0);
            alphabet_len.saturating_mul(w).saturating_mul(lanes)
        })
        .sum()
}

/// Resolve the layout for a database, honouring an explicit choice or auto-selecting by size.
pub(crate) fn choose_layout(
    sequences: &[Vec<u8>],
    lanes: usize,
    alphabet_len: usize,
    choice: LayoutChoice,
) -> Layout {
    match choice {
        LayoutChoice::Force(layout) => layout,
        LayoutChoice::Auto => {
            if precomputed_table_bytes(sequences, lanes, alphabet_len) <= PRECOMPUTED_MAX_BYTES {
                Layout::Precomputed
            } else {
                Layout::Gathered
            }
        }
    }
}

/// Per-batch substitution-score data, in the chosen [`Layout`].
#[derive(Debug, Clone)]
enum SubScores {
    /// `residues[(j-1) * lanes + k]` = target `k`'s residue at column `j`; the kernel shuffles the
    /// per-letter query profile by it.
    Gathered { residues: Vec<u8> },
    /// `table[(q * w + (j-1)) * lanes + k]` = `score(q, target_k[j])`, baked at build time; the
    /// kernel loads it directly, no gather.
    Precomputed { table: Vec<i8> },
}

/// One batch of up to `lanes` database sequences, packed for the inter-sequence kernel. Built
/// once at [`Database`](crate::Database) construction — query-independent, so no scan rebuilds it.
#[derive(Debug, Clone)]
struct PackedBatch {
    /// Number of real sequences in this batch (`<= lanes`); trailing lanes are padding.
    real: usize,
    /// Padded column count (the longest sequence in the batch).
    w: usize,
    /// `mask_le[j * lanes + k]` = `0xFF` iff `j <= len_k`, else `0`. `(w + 1) * lanes` bytes.
    mask_le: Vec<i8>,
    /// `mask_eq[j * lanes + k]` = `0xFF` iff `j == len_k`, else `0`. `(w + 1) * lanes` bytes.
    mask_eq: Vec<i8>,
    /// Substitution scores in the chosen layout.
    sub: SubScores,
}

/// The whole database packed for a fixed lane count and layout. Immutable and query-independent.
#[derive(Debug, Clone)]
pub(crate) struct PackedDb {
    lanes: usize,
    layout: Layout,
    alphabet_len: usize,
    /// Per-letter query profile `letter_profile[q * al + s] = score(q, s)` — used only by the
    /// `Gathered` layout (empty for `Precomputed`, which bakes scores into the table).
    letter_profile: Vec<i8>,
    batches: Vec<PackedBatch>,
}

impl PackedDb {
    /// Pack `sequences` for `lanes`-wide SIMD in the given `layout`. `scoring` must be valid for an
    /// `I8`-width database, so every entry fits `i8`.
    pub(crate) fn build(
        sequences: &[Vec<u8>],
        lanes: usize,
        layout: Layout,
        scoring: &Scoring,
    ) -> Self {
        let al = scoring.alphabet_len();
        let mut batches = Vec::new();

        for chunk in sequences.chunks(lanes) {
            let real = chunk.len();
            let w = chunk.iter().map(Vec::len).max().unwrap_or(0);

            let mut lens = vec![0usize; lanes];
            let mut residues = vec![0u8; w * lanes];
            for (k, s) in chunk.iter().enumerate() {
                lens[k] = s.len();
                for (j0, &r) in s.iter().enumerate() {
                    residues[j0 * lanes + k] = r;
                }
            }

            let mut mask_le = vec![0i8; (w + 1) * lanes];
            let mut mask_eq = vec![0i8; (w + 1) * lanes];
            for j in 0..=w {
                for k in 0..lanes {
                    mask_le[j * lanes + k] = if j <= lens[k] { -1 } else { 0 };
                    mask_eq[j * lanes + k] = if j == lens[k] { -1 } else { 0 };
                }
            }

            let sub = match layout {
                Layout::Gathered => SubScores::Gathered { residues },
                Layout::Precomputed => {
                    // table[(q * w + j0) * lanes + k] = score(q, residues[j0 * lanes + k]).
                    let mut table = vec![0i8; al * w * lanes];
                    for q in 0..al {
                        for j0 in 0..w {
                            for k in 0..lanes {
                                let t = residues[j0 * lanes + k] as usize;
                                table[(q * w + j0) * lanes + k] = scoring.score(q, t) as i8;
                            }
                        }
                    }
                    SubScores::Precomputed { table }
                }
            };

            batches.push(PackedBatch {
                real,
                w,
                mask_le,
                mask_eq,
                sub,
            });
        }

        let letter_profile = match layout {
            Layout::Gathered => {
                let mut lp = vec![0i8; al * al];
                for q in 0..al {
                    for s in 0..al {
                        lp[q * al + s] = scoring.score(q, s) as i8;
                    }
                }
                lp
            }
            Layout::Precomputed => Vec::new(),
        };

        PackedDb {
            lanes,
            layout,
            alphabet_len: al,
            letter_profile,
            batches,
        }
    }

    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }
}

/// Reusable, query-independent-sized working memory for the SIMD scan. Held in
/// [`Scratch`](crate::Scratch) so the hot path allocates nothing.
#[derive(Debug)]
pub(crate) struct SimdScratch {
    /// Two `H` rows, ping-ponged: `2 * (max_target_len + 1) * lanes` bytes.
    h: Vec<i8>,
    /// The down-carried `F` column: `(max_target_len + 1) * lanes` bytes.
    f: Vec<i8>,
    /// One batch of per-lane scores (used by the single-best `scan` reduction).
    lane_scores: Vec<i8>,
    /// Per-database-sequence scores (used by the per-target `scan_all`): `sequence_count` bytes.
    scores: Vec<i8>,
    /// Per-database-sequence answer columns (`target_end + 1`), filled by `fill_ends`.
    cols: Vec<i16>,
    /// Per-database-sequence answer rows (`query_end + 1`), filled by `fill_ends`.
    rows: Vec<i16>,
}

impl SimdScratch {
    pub(crate) fn new(sequence_count: usize, max_target_len: usize, lanes: usize) -> Self {
        let row = (max_target_len + 1) * lanes;
        SimdScratch {
            h: vec![0i8; 2 * row],
            f: vec![0i8; row],
            lane_scores: vec![0i8; lanes],
            scores: vec![0i8; sequence_count],
            cols: vec![0i16; sequence_count],
            rows: vec![0i16; sequence_count],
        }
    }

    /// Empty buffers for a scalar-backend database, which never runs the SIMD kernel.
    pub(crate) fn empty() -> Self {
        SimdScratch {
            h: Vec::new(),
            f: Vec::new(),
            lane_scores: Vec::new(),
            scores: Vec::new(),
            cols: Vec::new(),
            rows: Vec::new(),
        }
    }

    /// The per-target scores filled by [`fill_scores`], one per database sequence.
    pub(crate) fn scores(&self) -> &[i8] {
        &self.scores
    }

    /// The per-target end positions filled by [`fill_ends`]: `(target_end_col, query_end_row)`
    /// slices, one entry per database sequence in `db_index` order.
    pub(crate) fn ends(&self) -> (&[i16], &[i16]) {
        (&self.cols, &self.rows)
    }
}

/// Per-batch column loop over prebuilt packing and reused byte buffers. Writes `batch.real`
/// scores into `out`. Score-only; end positions are recovered elsewhere.
// `#[inline(always)]` so this hot loop — and `L`'s intrinsic ops — folds into the
// `#[target_feature]` SIMD shim (dispatch shape §5 of `handover.md`).
#[inline(always)]
#[allow(clippy::too_many_arguments)] // an inter-sequence DP inherently takes many parameters
fn scan_batch<L: Lanes>(
    query: &[u8],
    letter_profile: &[i8],
    al: usize,
    batch: &PackedBatch,
    go: i8,
    ge: i8,
    flags: &Flags,
    h: &mut [i8],
    f: &mut [i8],
    out: &mut [i8],
) {
    let lanes = L::LANES;
    let qlen = query.len();
    let w = batch.w;
    let cols = (w + 1) * lanes;

    let go_v = L::splat(go);
    let ge_v = L::splat(ge);
    let zero = L::splat(0);
    let neg = L::splat(NEG8);

    // Two H rows ping-ponged via mutable-reference swap (no copy). `prev` holds row i-1, `cur`
    // is written for row i.
    let half = h.len() / 2;
    let (row_a, row_b) = h.split_at_mut(half);
    let mut prev: &mut [i8] = &mut row_a[..cols];
    let mut cur: &mut [i8] = &mut row_b[..cols];

    // Row 0 into `prev`; the F column to −∞.
    for j in 0..=w {
        let border = if flags.top_row_free {
            zero
        } else {
            L::splat(-gap_penalty(go as i32, ge as i32, j) as i8)
        };
        L::store(border, &mut prev[j * lanes..]);
        L::store(neg, &mut f[j * lanes..]);
    }

    // SW running max and the best last-column cell, accumulated over every row. The last column is
    // seeded with the row-0 border `H[0][len_k]` so that a *penalised* top border (`SHW`) is
    // counted; for a free top row (`OV`) that border is `0`, matching the previous `zero` seed.
    let mut sw_ans = zero;
    let mut ov_lastcol = zero;
    if flags.answer_last_col {
        ov_lastcol = neg;
        for j in 0..=w {
            ov_lastcol = L::select(
                L::load(&batch.mask_eq[j * lanes..]),
                L::max(ov_lastcol, L::load(&prev[j * lanes..])),
                ov_lastcol,
            );
        }
    }

    for i in 1..=qlen {
        let border = if flags.left_col_free {
            zero
        } else {
            L::splat(-gap_penalty(go as i32, ge as i32, i) as i8)
        };
        L::store(border, &mut cur[0..lanes]);
        let qi = query[i - 1] as usize;
        let mut e = neg; // E[i][0]
        for j in 1..=w {
            let col = j * lanes;
            let prevcol = (j - 1) * lanes;
            e = L::max(
                L::sub_sat(L::load(&cur[prevcol..]), go_v),
                L::sub_sat(e, ge_v),
            );
            let f_j = L::max(
                L::sub_sat(L::load(&prev[col..]), go_v),
                L::sub_sat(L::load(&f[col..]), ge_v),
            );
            L::store(f_j, &mut f[col..]);
            // Substitution vector = score(query[i], target_k[j]) per lane, from the chosen layout.
            let sub = match &batch.sub {
                SubScores::Gathered { residues } => L::shuffle_lookup(
                    &letter_profile[qi * al..qi * al + al],
                    &residues[prevcol..col],
                ),
                SubScores::Precomputed { table } => L::load(&table[(qi * w + (j - 1)) * lanes..]),
            };
            let diag = L::add_sat(L::load(&prev[prevcol..]), sub);
            let mut cell = L::max(diag, L::max(e, f_j));
            if flags.local {
                cell = L::max(cell, zero);
                sw_ans = L::select(L::load(&batch.mask_le[col..]), L::max(sw_ans, cell), sw_ans);
            }
            if flags.answer_last_col {
                ov_lastcol = L::select(
                    L::load(&batch.mask_eq[col..]),
                    L::max(ov_lastcol, cell),
                    ov_lastcol,
                );
            }
            L::store(cell, &mut cur[col..]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    // `prev` now holds the last query row (row 0 if the query was empty).
    let last_row = &*prev;
    let load_row = |j: usize| L::load(&last_row[j * lanes..]);

    let ans = if flags.local {
        sw_ans
    } else if flags.answer_last_row && flags.answer_last_col {
        // OV: best of the last row (j <= len_k) and the accumulated last column.
        let mut lastrow = load_row(0);
        for j in 1..=w {
            lastrow = L::select(
                L::load(&batch.mask_le[j * lanes..]),
                L::max(lastrow, load_row(j)),
                lastrow,
            );
        }
        L::max(lastrow, ov_lastcol)
    } else if flags.answer_last_row {
        // HW: best of the last row over j <= len_k (including the j = 0 border).
        let mut hw = load_row(0);
        for j in 1..=w {
            hw = L::select(
                L::load(&batch.mask_le[j * lanes..]),
                L::max(hw, load_row(j)),
                hw,
            );
        }
        hw
    } else if flags.answer_last_col {
        // SHW: best of the last column H[i][len_k], i = 0..=qlen — seeded with row 0, accumulated
        // over the sweep, and with the corner as the i = qlen term.
        ov_lastcol
    } else {
        // NW: exactly H[qlen][len_k] per lane.
        let mut nw = load_row(0); // covers len_k == 0
        for j in 1..=w {
            nw = L::select(L::load(&batch.mask_eq[j * lanes..]), load_row(j), nw);
        }
        nw
    };

    // Store the answer vector, then copy out the real lanes (`32` covers the widest backend).
    let mut ans_arr = [0i8; 32];
    L::store(ans, &mut ans_arr);
    out.copy_from_slice(&ans_arr[..batch.real]);
}

/// Like [`scan_batch`], but also tracks each lane's answer-cell coordinates for `ScoreEnd`.
/// Writes per-lane `(score, col, row)` where `col = target_end + 1` and `row = query_end + 1` (both
/// grid coordinates; `0` means "no aligned position"). The tie-break — smallest target end, then
/// query end — is applied in-vector by [`Lanes::update_answer`], matching the scalar oracle.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn scan_batch_ends<L: LanesEnds>(
    query: &[u8],
    letter_profile: &[i8],
    al: usize,
    batch: &PackedBatch,
    go: i8,
    ge: i8,
    flags: &Flags,
    h: &mut [i8],
    f: &mut [i8],
    out_score: &mut [i8],
    out_col: &mut [i16],
    out_row: &mut [i16],
) {
    let lanes = L::LANES;
    let qlen = query.len();
    let w = batch.w;
    let cols = (w + 1) * lanes;

    let go_v = L::splat(go);
    let ge_v = L::splat(ge);
    let zero = L::splat(0);
    let neg = L::splat(NEG8);
    let all = L::splat(-1); // every lane active

    let half = h.len() / 2;
    let (row_a, row_b) = h.split_at_mut(half);
    let mut prev: &mut [i8] = &mut row_a[..cols];
    let mut cur: &mut [i8] = &mut row_b[..cols];

    for j in 0..=w {
        let border = if flags.top_row_free {
            zero
        } else {
            L::splat(-gap_penalty(go as i32, ge as i32, j) as i8)
        };
        L::store(border, &mut prev[j * lanes..]);
        L::store(neg, &mut f[j * lanes..]);
    }

    // Answer accumulator (score, col, row), seeded to −∞ at the origin.
    let mut a_s = neg;
    let mut a_c = L::pos_splat(0);
    let mut a_r = L::pos_splat(0);
    // Local mode's empty alignment: a `0` at grid (0, 0) — the lexicographically smallest cell.
    if flags.local {
        (a_s, a_c, a_r) = L::update_answer(all, a_s, a_c, a_r, zero, 0, 0);
    }
    // OV also considers the last column at row 0 (grid row 0 = smallest query end): `H[0][len_k]`,
    // read from the seeded top row before the sweep overwrites it.
    if flags.answer_last_col {
        for j in 0..=w {
            (a_s, a_c, a_r) = L::update_answer(
                L::load(&batch.mask_eq[j * lanes..]),
                a_s,
                a_c,
                a_r,
                L::load(&prev[j * lanes..]),
                j as i16,
                0,
            );
        }
    }

    for i in 1..=qlen {
        let border = if flags.left_col_free {
            zero
        } else {
            L::splat(-gap_penalty(go as i32, ge as i32, i) as i8)
        };
        L::store(border, &mut cur[0..lanes]);
        let qi = query[i - 1] as usize;
        let mut e = neg;
        for j in 1..=w {
            let col = j * lanes;
            let prevcol = (j - 1) * lanes;
            e = L::max(
                L::sub_sat(L::load(&cur[prevcol..]), go_v),
                L::sub_sat(e, ge_v),
            );
            let f_j = L::max(
                L::sub_sat(L::load(&prev[col..]), go_v),
                L::sub_sat(L::load(&f[col..]), ge_v),
            );
            L::store(f_j, &mut f[col..]);
            let sub = match &batch.sub {
                SubScores::Gathered { residues } => L::shuffle_lookup(
                    &letter_profile[qi * al..qi * al + al],
                    &residues[prevcol..col],
                ),
                SubScores::Precomputed { table } => L::load(&table[(qi * w + (j - 1)) * lanes..]),
            };
            let diag = L::add_sat(L::load(&prev[prevcol..]), sub);
            let mut cell = L::max(diag, L::max(e, f_j));
            if flags.local {
                cell = L::max(cell, zero);
                // SW: every cell (within each lane's real columns) is a candidate.
                (a_s, a_c, a_r) = L::update_answer(
                    L::load(&batch.mask_le[col..]),
                    a_s,
                    a_c,
                    a_r,
                    cell,
                    j as i16,
                    i as i16,
                );
            }
            if flags.answer_last_col {
                // OV last column: the cell at each lane's own final column `len_k`.
                (a_s, a_c, a_r) = L::update_answer(
                    L::load(&batch.mask_eq[col..]),
                    a_s,
                    a_c,
                    a_r,
                    cell,
                    j as i16,
                    i as i16,
                );
            }
            L::store(cell, &mut cur[col..]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    // Post-sweep border cells at the final query row `qlen` (`prev` holds it; row 0 if empty).
    // HW/OV consider the whole last row (`j <= len_k`); NW considers only the corner (`j == len_k`).
    if !flags.local {
        let last_mask = |j: usize| -> &[i8] {
            let col = j * lanes;
            if flags.answer_last_row {
                &batch.mask_le[col..]
            } else {
                &batch.mask_eq[col..]
            }
        };
        for j in 0..=w {
            let cell = L::load(&prev[j * lanes..]);
            (a_s, a_c, a_r) = L::update_answer(
                L::load(last_mask(j)),
                a_s,
                a_c,
                a_r,
                cell,
                j as i16,
                qlen as i16,
            );
        }
    }

    let mut score_arr = [0i8; 32];
    let mut col_arr = [0i16; 32];
    let mut row_arr = [0i16; 32];
    L::store(a_s, &mut score_arr);
    L::pos_store(a_c, &mut col_arr);
    L::pos_store(a_r, &mut row_arr);
    out_score.copy_from_slice(&score_arr[..batch.real]);
    out_col.copy_from_slice(&col_arr[..batch.real]);
    out_row.copy_from_slice(&row_arr[..batch.real]);
}

/// Scan `query` against every sequence using the inter-sequence kernel with lane backend `L`.
///
/// Requires an `I8`-width, `alphabet_len <= 16` database (see [`kernel_applies`]); the caller
/// gates that. Returns the same [`BestHit`] the scalar path would: highest score, smallest
/// `db_index` on a tie, and — for `ScoreEnd` — the winner's ends via one scalar re-alignment.
#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_batched<L: Lanes>(
    packed: &PackedDb,
    sequences: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
    query: &[u8],
    sc: &mut SimdScratch,
    dp: &mut kernel::DpBuffers,
) -> BestHit {
    let flags = Flags::for_mode(mode);
    let (go, ge) = (scoring.gap_open() as i8, scoring.gap_ext() as i8);

    let mut best_score = i8::MIN;
    let mut best_index = 0usize;
    for (b, batch) in packed.batches.iter().enumerate() {
        scan_batch::<L>(
            query,
            &packed.letter_profile,
            packed.alphabet_len,
            batch,
            go,
            ge,
            &flags,
            &mut sc.h,
            &mut sc.f,
            &mut sc.lane_scores[..batch.real],
        );
        let start = b * packed.lanes;
        for k in 0..batch.real {
            let s = sc.lane_scores[k];
            if s > best_score {
                best_score = s;
                best_index = start + k;
            }
        }
    }

    let (query_end, target_end) = if search_type.tracks_end() {
        // Recover the winner's ends with one scalar alignment — bit-identical to the oracle.
        let (score, qe, te) = kernel::align_core(query, &sequences[best_index], scoring, mode, dp);
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

/// Run the inter-sequence scan on the resolved SIMD `backend`. The resolver only selects one for a
/// CPU that supports it and a database it applies to, so the arms are exhaustive in practice.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_dispatch(
    backend: crate::Backend,
    packed: &PackedDb,
    sequences: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    search_type: SearchType,
    query: &[u8],
    sc: &mut SimdScratch,
    dp: &mut kernel::DpBuffers,
) -> BestHit {
    match backend {
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Avx2 => {
            avx2::run(packed, sequences, scoring, mode, search_type, query, sc, dp)
        }
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Sse41 => {
            sse41::run(packed, sequences, scoring, mode, search_type, query, sc, dp)
        }
        #[cfg(target_arch = "aarch64")]
        crate::Backend::Neon => {
            neon::run(packed, sequences, scoring, mode, search_type, query, sc, dp)
        }
        other => unreachable!("no inter-sequence kernel for backend {other}"),
    }
}

/// Fill `sc.scores[0..sequence_count]` with the per-target score of every database sequence, in
/// `db_index` order. This is the per-target primitive behind `Database::scan_all`: unlike
/// [`scan_dispatch`], it does not reduce to a single best hit. Scores only; end positions are
/// recovered by the caller (currently via scalar re-alignment).
#[inline(always)]
fn fill_scores_lanes<L: Lanes>(
    packed: &PackedDb,
    query: &[u8],
    go: i8,
    ge: i8,
    flags: &Flags,
    sc: &mut SimdScratch,
) {
    for (b, batch) in packed.batches.iter().enumerate() {
        let start = b * packed.lanes;
        let end = start + batch.real;
        scan_batch::<L>(
            query,
            &packed.letter_profile,
            packed.alphabet_len,
            batch,
            go,
            ge,
            flags,
            &mut sc.h,
            &mut sc.f,
            &mut sc.scores[start..end],
        );
    }
}

/// Fill per-target scores on the resolved SIMD `backend` (see [`fill_scores_lanes`]).
pub(crate) fn fill_scores(
    backend: crate::Backend,
    packed: &PackedDb,
    mode: Mode,
    gap_open: i32,
    gap_ext: i32,
    query: &[u8],
    sc: &mut SimdScratch,
) {
    let flags = Flags::for_mode(mode);
    let (go, ge) = (gap_open as i8, gap_ext as i8);
    match backend {
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Avx2 => avx2::run_scores(packed, query, go, ge, &flags, sc),
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Sse41 => sse41::run_scores(packed, query, go, ge, &flags, sc),
        #[cfg(target_arch = "aarch64")]
        crate::Backend::Neon => neon::run_scores(packed, query, go, ge, &flags, sc),
        other => unreachable!("no inter-sequence kernel for backend {other}"),
    }
}

/// Whether the resolved `backend` has an in-vector `ScoreEnd` (end-position) kernel. All the SIMD
/// backends do; the scalar backend has no packing and recovers ends directly.
pub(crate) fn backend_tracks_ends(backend: crate::Backend) -> bool {
    matches!(
        backend,
        crate::Backend::Sse41 | crate::Backend::Avx2 | crate::Backend::Neon
    )
}

/// Fill per-target scores AND end positions on the resolved SIMD `backend` (see
/// [`fill_ends_lanes`]). Only valid when [`backend_tracks_ends`] and positions fit `i16`.
pub(crate) fn fill_ends(
    backend: crate::Backend,
    packed: &PackedDb,
    mode: Mode,
    gap_open: i32,
    gap_ext: i32,
    query: &[u8],
    sc: &mut SimdScratch,
) {
    let flags = Flags::for_mode(mode);
    let (go, ge) = (gap_open as i8, gap_ext as i8);
    match backend {
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Avx2 => avx2::run_ends(packed, query, go, ge, &flags, sc),
        #[cfg(target_arch = "x86_64")]
        crate::Backend::Sse41 => sse41::run_ends(packed, query, go, ge, &flags, sc),
        #[cfg(target_arch = "aarch64")]
        crate::Backend::Neon => neon::run_ends(packed, query, go, ge, &flags, sc),
        other => unreachable!("no inter-sequence end kernel for backend {other}"),
    }
}

/// Fill `sc.scores`/`sc.cols`/`sc.rows` with each database sequence's score and answer-cell
/// coordinates (per-target `ScoreEnd`). The end tie-break is applied in-vector. Stage 1 exercises
/// each SIMD backend via its `run_ends` shim, and by the `ScalarLanes` reference in tests.
#[inline(always)]
fn fill_ends_lanes<L: LanesEnds>(
    packed: &PackedDb,
    query: &[u8],
    go: i8,
    ge: i8,
    flags: &Flags,
    sc: &mut SimdScratch,
) {
    for (b, batch) in packed.batches.iter().enumerate() {
        let start = b * packed.lanes;
        let end = start + batch.real;
        scan_batch_ends::<L>(
            query,
            &packed.letter_profile,
            packed.alphabet_len,
            batch,
            go,
            ge,
            flags,
            &mut sc.h,
            &mut sc.f,
            &mut sc.scores[start..end],
            &mut sc.cols[start..end],
            &mut sc.rows[start..end],
        );
    }
}

/// SSE4.1 lane backend: 16 `i8` lanes per `__m128i`.
#[cfg(target_arch = "x86_64")]
pub(crate) mod sse41 {
    // Intrinsics require `unsafe`; the crate is otherwise `deny(unsafe_code)`.
    #![allow(unsafe_code)]

    use super::{
        Flags, Lanes, LanesEnds, PackedDb, SimdScratch, fill_ends_lanes, fill_scores_lanes,
        scan_batched,
    };
    use crate::hit::BestHit;
    use crate::kernel::DpBuffers;
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use crate::search::SearchType;
    use core::arch::x86_64::*;

    /// 16-lane SSE4.1 backend. Ops are `#[inline(always)]` so they fold into the
    /// `#[target_feature]` shim below and get SSE4.1 codegen with no per-op call.
    #[derive(Clone, Copy)]
    pub(crate) struct Sse41;

    impl Lanes for Sse41 {
        const LANES: usize = 16;
        type V = __m128i;

        #[inline(always)]
        fn splat(v: i8) -> __m128i {
            unsafe { _mm_set1_epi8(v) }
        }
        #[inline(always)]
        fn add_sat(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_adds_epi8(a, b) }
        }
        #[inline(always)]
        fn sub_sat(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_subs_epi8(a, b) }
        }
        #[inline(always)]
        fn max(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_max_epi8(a, b) } // SSE4.1
        }
        #[inline(always)]
        fn select(mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
            // blendv picks `a` where mask's high bit is set (our masks are 0x00 / 0xFF). SSE4.1.
            unsafe { _mm_blendv_epi8(b, a, mask) }
        }
        #[inline(always)]
        fn load(src: &[i8]) -> __m128i {
            debug_assert!(src.len() >= 16);
            unsafe { _mm_loadu_si128(src.as_ptr().cast()) }
        }
        #[inline(always)]
        fn store(v: __m128i, dst: &mut [i8]) {
            debug_assert!(dst.len() >= 16);
            unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast(), v) }
        }
        #[inline(always)]
        fn shuffle_lookup(table: &[i8], indices: &[u8]) -> __m128i {
            debug_assert!(table.len() <= 16 && indices.len() >= 16);
            unsafe {
                // Pad the substitution table to 16 bytes, then byte-shuffle by the lane residues.
                // Residues are `< alphabet_len <= 16`, so their high bit is clear and PSHUFB does a
                // plain table lookup (no lane zeroing). PSHUFB is SSSE3, implied by SSE4.1.
                let mut padded = [0i8; 16];
                padded[..table.len()].copy_from_slice(table);
                let t = _mm_loadu_si128(padded.as_ptr().cast());
                let idx = _mm_loadu_si128(indices.as_ptr().cast());
                _mm_shuffle_epi8(t, idx)
            }
        }
    }

    impl LanesEnds for Sse41 {
        // 16 `i16` positions across two registers: lanes 0..8 and 8..16.
        type PosV = (__m128i, __m128i);

        #[inline(always)]
        fn pos_splat(v: i16) -> (__m128i, __m128i) {
            unsafe {
                let s = _mm_set1_epi16(v);
                (s, s)
            }
        }
        #[inline(always)]
        fn pos_store(v: (__m128i, __m128i), dst: &mut [i16]) {
            debug_assert!(dst.len() >= 16);
            unsafe {
                _mm_storeu_si128(dst.as_mut_ptr().cast(), v.0);
                _mm_storeu_si128(dst[8..].as_mut_ptr().cast(), v.1);
            }
        }
        #[inline(always)]
        fn update_answer(
            active: __m128i,
            best_score: __m128i,
            best_col: (__m128i, __m128i),
            best_row: (__m128i, __m128i),
            cell: __m128i,
            col: i16,
            row: i16,
        ) -> (__m128i, (__m128i, __m128i), (__m128i, __m128i)) {
            unsafe {
                let gt = _mm_cmpgt_epi8(cell, best_score);
                let eq = _mm_cmpeq_epi8(cell, best_score);
                let colv = _mm_set1_epi16(col);
                let rowv = _mm_set1_epi16(row);
                // Per i16 half: (col < best_col) | (col == best_col & row < best_row).
                let lex = |bc: __m128i, br: __m128i| {
                    let col_lt = _mm_cmpgt_epi16(bc, colv); // bc > col  <=>  col < bc
                    let col_eq = _mm_cmpeq_epi16(colv, bc);
                    let row_lt = _mm_cmpgt_epi16(br, rowv);
                    _mm_or_si128(col_lt, _mm_and_si128(col_eq, row_lt))
                };
                let lex_lo = lex(best_col.0, best_row.0);
                let lex_hi = lex(best_col.1, best_row.1);
                // Narrow the two i16 lex masks to one i8 mask (SSE pack keeps natural order).
                let lex8 = _mm_packs_epi16(lex_lo, lex_hi);
                let take = _mm_and_si128(active, _mm_or_si128(gt, _mm_and_si128(eq, lex8)));
                let new_score = _mm_blendv_epi8(best_score, cell, take);
                // Widen the take mask back to i16 halves (sign-extend keeps natural order).
                let take_lo = _mm_cvtepi8_epi16(take);
                let take_hi = _mm_cvtepi8_epi16(_mm_srli_si128(take, 8));
                let new_col = (
                    _mm_blendv_epi8(best_col.0, colv, take_lo),
                    _mm_blendv_epi8(best_col.1, colv, take_hi),
                );
                let new_row = (
                    _mm_blendv_epi8(best_row.0, rowv, take_lo),
                    _mm_blendv_epi8(best_row.1, rowv, take_hi),
                );
                (new_score, new_col, new_row)
            }
        }
    }

    /// Feature-enabled shim: everything inlined here is compiled with SSE4.1.
    #[target_feature(enable = "sse4.1")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn scan_ff(
        packed: &PackedDb,
        sequences: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        search_type: SearchType,
        query: &[u8],
        sc: &mut SimdScratch,
        dp: &mut DpBuffers,
    ) -> BestHit {
        scan_batched::<Sse41>(packed, sequences, scoring, mode, search_type, query, sc, dp)
    }

    /// Safe entry point. Only called for a resolved `Sse41` backend, which the resolver returns
    /// exclusively when `sse4.1` is detected — so the feature precondition of [`scan_ff`] holds.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        packed: &PackedDb,
        sequences: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        search_type: SearchType,
        query: &[u8],
        sc: &mut SimdScratch,
        dp: &mut DpBuffers,
    ) -> BestHit {
        debug_assert!(std::is_x86_feature_detected!("sse4.1"));
        unsafe { scan_ff(packed, sequences, scoring, mode, search_type, query, sc, dp) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn scores_ff(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_scores_lanes::<Sse41>(packed, query, go, ge, flags, sc);
    }

    /// Per-target scores (see [`super::fill_scores`]). Same feature precondition as [`run`].
    pub(crate) fn run_scores(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        debug_assert!(std::is_x86_feature_detected!("sse4.1"));
        unsafe { scores_ff(packed, query, go, ge, flags, sc) }
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn ends_ff(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_ends_lanes::<Sse41>(packed, query, go, ge, flags, sc);
    }

    /// Per-target scores and end positions (see [`super::fill_ends`]). Same precondition as [`run`].
    pub(crate) fn run_ends(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        debug_assert!(std::is_x86_feature_detected!("sse4.1"));
        unsafe { ends_ff(packed, query, go, ge, flags, sc) }
    }
}

/// AVX2 lane backend: 32 `i8` lanes per `__m256i`.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    #![allow(unsafe_code)]

    use super::{
        Flags, Lanes, LanesEnds, PackedDb, SimdScratch, fill_ends_lanes, fill_scores_lanes,
        scan_batched,
    };
    use crate::hit::BestHit;
    use crate::kernel::DpBuffers;
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use crate::search::SearchType;
    use core::arch::x86_64::*;

    /// 32-lane AVX2 backend. Ops are `#[inline(always)]` so they fold into the `#[target_feature]`
    /// shim below.
    #[derive(Clone, Copy)]
    pub(crate) struct Avx2;

    impl Lanes for Avx2 {
        const LANES: usize = 32;
        type V = __m256i;

        #[inline(always)]
        fn splat(v: i8) -> __m256i {
            unsafe { _mm256_set1_epi8(v) }
        }
        #[inline(always)]
        fn add_sat(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_adds_epi8(a, b) }
        }
        #[inline(always)]
        fn sub_sat(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_subs_epi8(a, b) }
        }
        #[inline(always)]
        fn max(a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_max_epi8(a, b) }
        }
        #[inline(always)]
        fn select(mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
            unsafe { _mm256_blendv_epi8(b, a, mask) }
        }
        #[inline(always)]
        fn load(src: &[i8]) -> __m256i {
            debug_assert!(src.len() >= 32);
            unsafe { _mm256_loadu_si256(src.as_ptr().cast()) }
        }
        #[inline(always)]
        fn store(v: __m256i, dst: &mut [i8]) {
            debug_assert!(dst.len() >= 32);
            unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), v) }
        }
        #[inline(always)]
        fn shuffle_lookup(table: &[i8], indices: &[u8]) -> __m256i {
            debug_assert!(table.len() <= 16 && indices.len() >= 32);
            unsafe {
                // `_mm256_shuffle_epi8` shuffles *within each 128-bit half* independently, so the
                // 16-byte table is broadcast to both halves; residues `< 16` then index the right
                // copy in either half. Same PSHUFB lookup as SSE4.1, twice.
                let mut padded = [0i8; 16];
                padded[..table.len()].copy_from_slice(table);
                let t128 = _mm_loadu_si128(padded.as_ptr().cast());
                let t = _mm256_broadcastsi128_si256(t128);
                let idx = _mm256_loadu_si256(indices.as_ptr().cast());
                _mm256_shuffle_epi8(t, idx)
            }
        }
    }

    impl LanesEnds for Avx2 {
        // 32 `i16` positions across two registers: lanes 0..16 and 16..32.
        type PosV = (__m256i, __m256i);

        #[inline(always)]
        fn pos_splat(v: i16) -> (__m256i, __m256i) {
            unsafe {
                let s = _mm256_set1_epi16(v);
                (s, s)
            }
        }
        #[inline(always)]
        fn pos_store(v: (__m256i, __m256i), dst: &mut [i16]) {
            debug_assert!(dst.len() >= 32);
            unsafe {
                _mm256_storeu_si256(dst.as_mut_ptr().cast(), v.0);
                _mm256_storeu_si256(dst[16..].as_mut_ptr().cast(), v.1);
            }
        }
        #[inline(always)]
        fn update_answer(
            active: __m256i,
            best_score: __m256i,
            best_col: (__m256i, __m256i),
            best_row: (__m256i, __m256i),
            cell: __m256i,
            col: i16,
            row: i16,
        ) -> (__m256i, (__m256i, __m256i), (__m256i, __m256i)) {
            unsafe {
                let gt = _mm256_cmpgt_epi8(cell, best_score);
                let eq = _mm256_cmpeq_epi8(cell, best_score);
                let colv = _mm256_set1_epi16(col);
                let rowv = _mm256_set1_epi16(row);
                let lex = |bc: __m256i, br: __m256i| {
                    let col_lt = _mm256_cmpgt_epi16(bc, colv); // col < bc
                    let col_eq = _mm256_cmpeq_epi16(colv, bc);
                    let row_lt = _mm256_cmpgt_epi16(br, rowv);
                    _mm256_or_si256(col_lt, _mm256_and_si256(col_eq, row_lt))
                };
                let lex_lo = lex(best_col.0, best_row.0);
                let lex_hi = lex(best_col.1, best_row.1);
                // Narrow to i8. `packs` interleaves the two 128-bit halves, so permute the 64-bit
                // chunks (0,2,1,3) back to natural lane order.
                let packed = _mm256_packs_epi16(lex_lo, lex_hi);
                let lex8 = _mm256_permute4x64_epi64(packed, 0b11_01_10_00);
                let take =
                    _mm256_and_si256(active, _mm256_or_si256(gt, _mm256_and_si256(eq, lex8)));
                let new_score = _mm256_blendv_epi8(best_score, cell, take);
                // Widen the take mask back to i16 halves (sign-extend keeps natural order).
                let take_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(take));
                let take_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(take, 1));
                let new_col = (
                    _mm256_blendv_epi8(best_col.0, colv, take_lo),
                    _mm256_blendv_epi8(best_col.1, colv, take_hi),
                );
                let new_row = (
                    _mm256_blendv_epi8(best_row.0, rowv, take_lo),
                    _mm256_blendv_epi8(best_row.1, rowv, take_hi),
                );
                (new_score, new_col, new_row)
            }
        }
    }

    #[target_feature(enable = "avx2")]
    #[allow(clippy::too_many_arguments)]
    unsafe fn scan_ff(
        packed: &PackedDb,
        sequences: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        search_type: SearchType,
        query: &[u8],
        sc: &mut SimdScratch,
        dp: &mut DpBuffers,
    ) -> BestHit {
        scan_batched::<Avx2>(packed, sequences, scoring, mode, search_type, query, sc, dp)
    }

    /// Safe entry point; only called for a resolved `Avx2` backend, which the resolver returns
    /// only when `avx2` is detected.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        packed: &PackedDb,
        sequences: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        search_type: SearchType,
        query: &[u8],
        sc: &mut SimdScratch,
        dp: &mut DpBuffers,
    ) -> BestHit {
        debug_assert!(std::is_x86_feature_detected!("avx2"));
        unsafe { scan_ff(packed, sequences, scoring, mode, search_type, query, sc, dp) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn scores_ff(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_scores_lanes::<Avx2>(packed, query, go, ge, flags, sc);
    }

    /// Per-target scores (see [`super::fill_scores`]). Same feature precondition as [`run`].
    pub(crate) fn run_scores(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        debug_assert!(std::is_x86_feature_detected!("avx2"));
        unsafe { scores_ff(packed, query, go, ge, flags, sc) }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn ends_ff(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_ends_lanes::<Avx2>(packed, query, go, ge, flags, sc);
    }

    /// Per-target scores and end positions (see [`super::fill_ends`]). Same precondition as [`run`].
    pub(crate) fn run_ends(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        debug_assert!(std::is_x86_feature_detected!("avx2"));
        unsafe { ends_ff(packed, query, go, ge, flags, sc) }
    }
}

/// NEON lane backend: 16 `i8` lanes per `int8x16_t`. NEON is baseline on aarch64, so there is no
/// runtime detection and no `#[target_feature]` shim — the intrinsics are always available.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    #![allow(unsafe_code)]

    use super::{
        Flags, Lanes, LanesEnds, PackedDb, SimdScratch, fill_ends_lanes, fill_scores_lanes,
        scan_batched,
    };
    use crate::hit::BestHit;
    use crate::kernel::DpBuffers;
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use crate::search::SearchType;
    use core::arch::aarch64::*;

    /// 16-lane NEON backend.
    #[derive(Clone, Copy)]
    pub(crate) struct Neon;

    impl Lanes for Neon {
        const LANES: usize = 16;
        type V = int8x16_t;

        #[inline(always)]
        fn splat(v: i8) -> int8x16_t {
            unsafe { vdupq_n_s8(v) }
        }
        #[inline(always)]
        fn add_sat(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vqaddq_s8(a, b) }
        }
        #[inline(always)]
        fn sub_sat(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vqsubq_s8(a, b) }
        }
        #[inline(always)]
        fn max(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vmaxq_s8(a, b) }
        }
        #[inline(always)]
        fn select(mask: int8x16_t, a: int8x16_t, b: int8x16_t) -> int8x16_t {
            // `vbslq_s8(m, a, b)`: per bit, `m` set selects `a` else `b`. Our masks are 0x00/0xFF.
            unsafe { vbslq_s8(vreinterpretq_u8_s8(mask), a, b) }
        }
        #[inline(always)]
        fn load(src: &[i8]) -> int8x16_t {
            debug_assert!(src.len() >= 16);
            unsafe { vld1q_s8(src.as_ptr()) }
        }
        #[inline(always)]
        fn store(v: int8x16_t, dst: &mut [i8]) {
            debug_assert!(dst.len() >= 16);
            unsafe { vst1q_s8(dst.as_mut_ptr(), v) }
        }
        #[inline(always)]
        fn shuffle_lookup(table: &[i8], indices: &[u8]) -> int8x16_t {
            debug_assert!(table.len() <= 16 && indices.len() >= 16);
            unsafe {
                // Table lookup: `vqtbl1q_s8` returns `table[idx[k]]`, or 0 where `idx[k] >= 16`.
                // Residues are `< alphabet_len <= 16`, so every lane is a plain lookup.
                let mut padded = [0i8; 16];
                padded[..table.len()].copy_from_slice(table);
                let t = vld1q_s8(padded.as_ptr());
                let idx = vld1q_u8(indices.as_ptr());
                vqtbl1q_s8(t, idx)
            }
        }
    }

    impl LanesEnds for Neon {
        // 16 `i16` positions across two registers: lanes 0..8 and 8..16.
        type PosV = (int16x8_t, int16x8_t);

        #[inline(always)]
        fn pos_splat(v: i16) -> (int16x8_t, int16x8_t) {
            unsafe {
                let s = vdupq_n_s16(v);
                (s, s)
            }
        }
        #[inline(always)]
        fn pos_store(v: (int16x8_t, int16x8_t), dst: &mut [i16]) {
            debug_assert!(dst.len() >= 16);
            unsafe {
                vst1q_s16(dst.as_mut_ptr(), v.0);
                vst1q_s16(dst[8..].as_mut_ptr(), v.1);
            }
        }
        #[inline(always)]
        fn update_answer(
            active: int8x16_t,
            best_score: int8x16_t,
            best_col: (int16x8_t, int16x8_t),
            best_row: (int16x8_t, int16x8_t),
            cell: int8x16_t,
            col: i16,
            row: i16,
        ) -> (int8x16_t, (int16x8_t, int16x8_t), (int16x8_t, int16x8_t)) {
            unsafe {
                let gt = vreinterpretq_s8_u8(vcgtq_s8(cell, best_score));
                let eq = vreinterpretq_s8_u8(vceqq_s8(cell, best_score));
                let colv = vdupq_n_s16(col);
                let rowv = vdupq_n_s16(row);
                let lex = |bc: int16x8_t, br: int16x8_t| {
                    let col_lt = vreinterpretq_s16_u16(vcgtq_s16(bc, colv)); // col < bc
                    let col_eq = vreinterpretq_s16_u16(vceqq_s16(colv, bc));
                    let row_lt = vreinterpretq_s16_u16(vcgtq_s16(br, rowv));
                    vorrq_s16(col_lt, vandq_s16(col_eq, row_lt))
                };
                let lex_lo = lex(best_col.0, best_row.0);
                let lex_hi = lex(best_col.1, best_row.1);
                // Narrow the two i16 lex masks to one i8 mask (saturating narrow keeps natural order).
                let lex8 = vcombine_s8(vqmovn_s16(lex_lo), vqmovn_s16(lex_hi));
                let take = vandq_s8(active, vorrq_s8(gt, vandq_s8(eq, lex8)));
                let take_u = vreinterpretq_u8_s8(take);
                let new_score = vbslq_s8(take_u, cell, best_score);
                // Widen the take mask back to i16 halves (sign-extend keeps natural order).
                let take_lo = vreinterpretq_u16_s16(vmovl_s8(vget_low_s8(take)));
                let take_hi = vreinterpretq_u16_s16(vmovl_s8(vget_high_s8(take)));
                let new_col = (
                    vbslq_s16(take_lo, colv, best_col.0),
                    vbslq_s16(take_hi, colv, best_col.1),
                );
                let new_row = (
                    vbslq_s16(take_lo, rowv, best_row.0),
                    vbslq_s16(take_hi, rowv, best_row.1),
                );
                (new_score, new_col, new_row)
            }
        }
    }

    /// Safe entry point. NEON is always available on aarch64, so no feature guard is needed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        packed: &PackedDb,
        sequences: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        search_type: SearchType,
        query: &[u8],
        sc: &mut SimdScratch,
        dp: &mut DpBuffers,
    ) -> BestHit {
        scan_batched::<Neon>(packed, sequences, scoring, mode, search_type, query, sc, dp)
    }

    /// Per-target scores (see [`super::fill_scores`]). NEON is baseline, so no feature guard.
    pub(crate) fn run_scores(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_scores_lanes::<Neon>(packed, query, go, ge, flags, sc);
    }

    /// Per-target scores and end positions (see [`super::fill_ends`]). NEON is baseline.
    pub(crate) fn run_ends(
        packed: &PackedDb,
        query: &[u8],
        go: i8,
        ge: i8,
        flags: &Flags,
        sc: &mut SimdScratch,
    ) {
        fill_ends_lanes::<Neon>(packed, query, go, ge, flags, sc);
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

    /// Run the inter-sequence kernel with an `N`-lane scalar reference backend in the given
    /// `layout`, driving the same packed-database + reusable-scratch path production uses.
    fn inter_scan<const N: usize>(
        seqs: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        st: SearchType,
        query: &[u8],
        layout: Layout,
    ) -> BestHit {
        let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let packed = PackedDb::build(seqs, N, layout, scoring);
        let mut sc = SimdScratch::new(seqs.len(), max_t, N);
        let mut dp = kernel::DpBuffers::new();
        scan_batched::<ScalarLanes<N>>(&packed, seqs, scoring, mode, st, query, &mut sc, &mut dp)
    }

    const MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];

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
        /// search types, several lane counts, and **both layouts** — the lane-count and layout
        /// independence the determinism contract demands, checked before any real SIMD exists.
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
                    for layout in [Layout::Gathered, Layout::Precomputed] {
                        let got1 = inter_scan::<1>(&seqs, &scoring, mode, st, &q, layout);
                        let got4 = inter_scan::<4>(&seqs, &scoring, mode, st, &q, layout);
                        let got8 = inter_scan::<8>(&seqs, &scoring, mode, st, &q, layout);
                        prop_assert_eq!(got1, want, "1 lane, {} {} {}", mode, st, layout);
                        prop_assert_eq!(got4, want, "4 lanes, {} {} {}", mode, st, layout);
                        prop_assert_eq!(got8, want, "8 lanes, {} {} {}", mode, st, layout);
                    }
                }
            }
        }
    }

    /// Fill per-target scores with the `N`-lane reference backend in `layout`.
    fn per_target_scores<const N: usize>(
        seqs: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        query: &[u8],
        layout: Layout,
    ) -> Vec<i8> {
        let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let packed = PackedDb::build(seqs, N, layout, scoring);
        let mut sc = SimdScratch::new(seqs.len(), max_t, N);
        fill_scores_lanes::<ScalarLanes<N>>(
            &packed,
            query,
            scoring.gap_open() as i8,
            scoring.gap_ext() as i8,
            &Flags::for_mode(mode),
            &mut sc,
        );
        sc.scores().to_vec()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        /// The per-target `fill_scores` primitive returns each database sequence's score, matching
        /// `align_pair` for every sequence, across lane counts and layouts.
        #[test]
        fn fill_scores_matches_per_sequence((scoring, seqs, q) in scenario()) {
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
                for layout in [Layout::Gathered, Layout::Precomputed] {
                    let s1 = per_target_scores::<1>(&seqs, &scoring, mode, &q, layout);
                    let s4 = per_target_scores::<4>(&seqs, &scoring, mode, &q, layout);
                    let s8 = per_target_scores::<8>(&seqs, &scoring, mode, &q, layout);
                    for (i, seq) in seqs.iter().enumerate() {
                        let want = crate::align_pair(&q, seq, &scoring, mode, SearchType::Score)
                            .unwrap()
                            .score as i8;
                        prop_assert_eq!(s1[i], want, "1 lane {} {} seq {}", mode, layout, i);
                        prop_assert_eq!(s4[i], want, "4 lanes {} {} seq {}", mode, layout, i);
                        prop_assert_eq!(s8[i], want, "8 lanes {} {} seq {}", mode, layout, i);
                    }
                }
            }
        }
    }

    /// Per-target `(score, query_end, target_end)` via the in-vector `ScoreEnd` kernel with the
    /// `N`-lane reference backend.
    fn per_target_ends<const N: usize>(
        seqs: &[Vec<u8>],
        scoring: &Scoring,
        mode: Mode,
        query: &[u8],
        layout: Layout,
    ) -> Vec<BestHit> {
        let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
        let packed = PackedDb::build(seqs, N, layout, scoring);
        let mut sc = SimdScratch::new(seqs.len(), max_t, N);
        fill_ends_lanes::<ScalarLanes<N>>(
            &packed,
            query,
            scoring.gap_open() as i8,
            scoring.gap_ext() as i8,
            &Flags::for_mode(mode),
            &mut sc,
        );
        (0..seqs.len())
            .map(|i| BestHit {
                score: sc.scores[i] as i32,
                db_index: i,
                query_end: (sc.rows[i] as usize).checked_sub(1),
                target_end: (sc.cols[i] as usize).checked_sub(1),
            })
            .collect()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(300))]

        /// In-vector `ScoreEnd`: per-target score AND end positions match `align_pair` for every
        /// sequence — including the exact tie-break — across lane counts and layouts. This validates
        /// the end-tracking algorithm on the safe reference before any SIMD impl.
        #[test]
        fn fill_ends_matches_per_sequence((scoring, seqs, q) in scenario()) {
            let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
            for mode in MODES {
                prop_assume!(
                    kernel_applies(
                        scoring.required_width(mode, q.len(), max_t).unwrap(),
                        scoring.alphabet_len()
                    )
                );
                // Positions must fit i16 for the in-vector tracker.
                prop_assume!(q.len() <= i16::MAX as usize && max_t <= i16::MAX as usize);
            }

            for mode in MODES {
                for layout in [Layout::Gathered, Layout::Precomputed] {
                    let e1 = per_target_ends::<1>(&seqs, &scoring, mode, &q, layout);
                    let e4 = per_target_ends::<4>(&seqs, &scoring, mode, &q, layout);
                    let e8 = per_target_ends::<8>(&seqs, &scoring, mode, &q, layout);
                    for (i, seq) in seqs.iter().enumerate() {
                        let want = BestHit {
                            db_index: i,
                            ..crate::align_pair(&q, seq, &scoring, mode, SearchType::ScoreEnd).unwrap()
                        };
                        prop_assert_eq!(e1[i], want, "1 lane {} {} seq {}", mode, layout, i);
                        prop_assert_eq!(e4[i], want, "4 lanes {} {} seq {}", mode, layout, i);
                        prop_assert_eq!(e8[i], want, "8 lanes {} {} seq {}", mode, layout, i);
                    }
                }
            }
        }
    }
}
