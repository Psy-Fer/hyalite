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
    /// the table is small enough to stay cache-resident (see [`Database::builder`]).
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
    /// One batch of per-lane scores.
    lane_scores: Vec<i8>,
}

impl SimdScratch {
    pub(crate) fn new(max_target_len: usize, lanes: usize) -> Self {
        let row = (max_target_len + 1) * lanes;
        SimdScratch {
            h: vec![0i8; 2 * row],
            f: vec![0i8; row],
            lane_scores: vec![0i8; lanes],
        }
    }

    /// Empty buffers for a scalar-backend database, which never runs the SIMD kernel.
    pub(crate) fn empty() -> Self {
        SimdScratch {
            h: Vec::new(),
            f: Vec::new(),
            lane_scores: Vec::new(),
        }
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

    // SW running max and OV best-last-column, accumulated over every row.
    let mut sw_ans = zero;
    let mut ov_lastcol = zero;

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

/// SSE4.1 lane backend: 16 `i8` lanes per `__m128i`.
#[cfg(target_arch = "x86_64")]
pub(crate) mod sse41 {
    // Intrinsics require `unsafe`; the crate is otherwise `deny(unsafe_code)`.
    #![allow(unsafe_code)]

    use super::{Lanes, PackedDb, SimdScratch, scan_batched};
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
}

/// AVX2 lane backend: 32 `i8` lanes per `__m256i`.
#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    #![allow(unsafe_code)]

    use super::{Lanes, PackedDb, SimdScratch, scan_batched};
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
}

/// NEON lane backend: 16 `i8` lanes per `int8x16_t`. NEON is baseline on aarch64, so there is no
/// runtime detection and no `#[target_feature]` shim — the intrinsics are always available.
#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    #![allow(unsafe_code)]

    use super::{Lanes, PackedDb, SimdScratch, scan_batched};
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
        let mut sc = SimdScratch::new(max_t, N);
        let mut dp = kernel::DpBuffers::new();
        scan_batched::<ScalarLanes<N>>(&packed, seqs, scoring, mode, st, query, &mut sc, &mut dp)
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
}
