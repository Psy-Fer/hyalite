//! Striped (Farrar) intra-sequence SIMD alignment for the single-pair path.
//!
//! Where the inter-sequence kernel ([`crate::inter`]) puts one database sequence per lane, the
//! striped kernel vectorises a **single** pairwise alignment: the query is laid out in `p` lanes
//! of `segLen = ceil(qlen / p)` stripes, and the DP marches over target columns with a
//! horizontally-vectorised inner loop plus Farrar's "lazy-F" correction for the cross-lane
//! vertical-gap dependency.
//!
//! The algorithm was first proven on a scalar stand-in (`ScalarStriped<N>`, kept under
//! `#[cfg(test)]`), exactly as the in-vector ScoreEnd work was proven on `ScalarLanes` before any
//! intrinsics. The real SSE4.1 (x86-64) and NEON (aarch64) backends implement the same
//! [`StripedLanes`] surface and run the same generic [`farrar_score`], so both are bit-identical to
//! the scalar oracle.
//!
//! Scope: **`Score`** for all four modes, `i8` width. `SW` clamps at `0`, so its lazy-F stops the
//! moment the carried F decays to `<= 0`; the non-clamped modes (`NW`/`HW`/`OV`) instead stop once
//! the carried F decays below the (fixed) minimum valid `H`, since `H` only rises during the loop.
//! `ScoreEnd`/`Alignment`/wider widths stay on the scalar path — striped end-position tracking is a
//! separate concern, and the single-pair path is not the throughput-critical one (that is the
//! batched database scan).

use crate::kernel::{Flags, gap_penalty};
use crate::mode::Mode;
use crate::scoring::Scoring;

/// The `-∞` sentinel at i8 width, matching the scalar kernel's `NEG` for unreachable `E`/`F`
/// cells. Saturating subtraction keeps it pinned, so it behaves as a true floor.
const NEG8: i8 = i8::MIN;

/// Saturating cast of an `i32` score into `i8` (never wraps; see the profile note below).
#[inline(always)]
fn sat_i8(v: i32) -> i8 {
    v.clamp(i8::MIN as i32, i8::MAX as i32) as i8
}

/// The lane operations the striped kernel needs. Implemented by SSE4.1 / NEON with intrinsics and,
/// under test, by a scalar array stand-in.
trait StripedLanes {
    /// Number of query stripes processed in parallel.
    const LANES: usize;
    /// The vector type: `LANES` packed `i8` lanes.
    type V: Copy;

    fn splat(v: i8) -> Self::V;
    /// Saturating signed `i8` addition, lane-wise.
    fn adds(a: Self::V, b: Self::V) -> Self::V;
    /// Saturating signed `i8` subtraction, lane-wise.
    fn subs(a: Self::V, b: Self::V) -> Self::V;
    fn max(a: Self::V, b: Self::V) -> Self::V;
    /// Shift lanes up by one (`out[l] = in[l-1]`), inserting `insert` at lane 0. This is the
    /// cross-stripe carry (`_mm_slli_si128(v, 1)` with a lane-0 insert on x86).
    fn shift_up(v: Self::V, insert: i8) -> Self::V;
    /// Horizontal maximum across all lanes.
    fn hmax(v: Self::V) -> i8;
    /// Horizontal minimum across all lanes.
    fn hmin(v: Self::V) -> i8;
    /// Whether any lane of `a` is strictly greater than the matching lane of `b`.
    fn any_gt(a: Self::V, b: Self::V) -> bool;
    fn load(src: &[i8]) -> Self::V;
    fn store(v: Self::V, dst: &mut [i8]);
}

/// Striped **score** for one query/target pair, in the saturating `i8` model, for any mode's
/// border/answer geometry ([`Flags`]).
///
/// Returns the same value as the scalar oracle when the score provably fits `i8` (which the
/// caller's width proof guarantees). The reported answer is reduced over the mode's answer region
/// by scalar reads over the *valid* query positions, so lane padding never contributes and the
/// result is independent of the lane count — the property the SIMD backends rely on for
/// determinism. Callers pass non-empty `query`/`target` (the degenerate cases go to the scalar
/// kernel).
fn farrar_score<L: StripedLanes>(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    flags: &Flags,
) -> i32 {
    let p = L::LANES;
    let qlen = query.len();
    let tlen = target.len();
    let seg = qlen.div_ceil(p);
    let (go_i, ge_i) = (scoring.gap_open(), scoring.gap_ext());
    let (go, ge) = (go_i as i8, ge_i as i8);
    let al = scoring.alphabet_len();

    // Query profile, striped: profile[t][v*p + l] = score(query[l*seg + v], t), or NEG8 for the
    // padding lanes. Entries are **saturated** into `i8`: `SW`'s width proof bounds the score
    // magnitude but not `|min_entry|`, so a mismatch below `-127` must clamp (not wrap) — it then
    // drives `H` to the `0` floor exactly as the i32 oracle does.
    let mut profile = vec![NEG8; al * seg * p];
    for (t, chunk) in profile.chunks_mut(seg * p).enumerate() {
        for v in 0..seg {
            for l in 0..p {
                let qpos = l * seg + v;
                if qpos < qlen {
                    chunk[v * p + l] = sat_i8(scoring.score(query[qpos] as usize, t));
                }
            }
        }
    }

    // Striped index of query position `pos`, and the position holding the last query row (`H[m]`).
    let idx = |pos: usize| (pos % seg) * p + pos / seg;
    let last_row = idx(qlen - 1);

    let mut h_store = vec![0i8; seg * p]; // H[*][j]
    let mut h_load = vec![0i8; seg * p]; //  H[*][j-1]
    let mut e = vec![NEG8; seg * p]; //      E[*][j]   (padding lanes stay at the -inf sentinel)
    // Column-0 borders per query position: the left-column `H[i][0]` (`0` when free, else a
    // penalised gap of length `i = pos + 1`) and the matching `E[i][1] = H[i][0] - gap_open`
    // (opening a horizontal gap from that border; the extend term is `-inf`).
    for pos in 0..qlen {
        let border = if flags.left_col_free {
            0
        } else {
            -gap_penalty(go_i, ge_i, pos + 1)
        };
        h_store[idx(pos)] = sat_i8(border);
        e[idx(pos)] = sat_i8(border - go_i);
    }

    let vgo = L::splat(go);
    let vge = L::splat(ge);
    let zero = L::splat(0);
    let mut vmax = zero; // SW: the answer is the max over all cells
    // HW/OV last row includes the left-border cell `H[m][0]` (`j = 0`), which is exactly the
    // initial last-row entry before any target column is processed.
    let mut row_best = h_store[last_row] as i32;

    // Non-local threshold mask: `MAX` at padding lanes, `MIN` at valid lanes, so `max(H, pad_hi)`
    // lifts padding out of the `hmin` reduction without touching `h_store` (padding is naturally
    // driven very negative by its `NEG8` profile, so it never inflates the F reduction either).
    let pad_hi: Vec<i8> = if flags.local {
        Vec::new()
    } else {
        let mut m = vec![i8::MIN; seg * p];
        for pos in qlen..seg * p {
            m[idx(pos)] = i8::MAX;
        }
        m
    };

    for (c, &tt) in target.iter().enumerate() {
        let prof = &profile[tt as usize * seg * p..];

        // Diagonal seed: `H[*][j-1]` of the last stripe, shifted up with the top-left border cell
        // `H[0][j-1]` — `0` when the top row is free, else the penalised leading-target gap.
        let seed = if flags.top_row_free {
            0
        } else {
            sat_i8(-gap_penalty(go_i, ge_i, c))
        };
        let mut vh = L::shift_up(L::load(&h_store[(seg - 1) * p..]), seed);
        core::mem::swap(&mut h_store, &mut h_load);
        // Initial F vector: `F[0][j] = NEG` everywhere except lane 0 (query position 0), whose
        // first real row has the top-border vertical-gap `F[1][j] = H[0][j] - gap_open`. Higher
        // lanes' first-stripe F is a cross-lane carry filled in by the lazy-F loop.
        let top_next = if flags.top_row_free {
            0
        } else {
            -gap_penalty(go_i, ge_i, c + 1)
        };
        let mut vf = L::shift_up(L::splat(NEG8), sat_i8(top_next - go_i));

        for v in 0..seg {
            vh = L::adds(vh, L::load(&prof[v * p..])); // H[i-1][j-1] + score
            vh = L::max(vh, L::load(&e[v * p..])); // vs E[i][j]
            vh = L::max(vh, vf); // vs F (partial; lazy-F fixes cross-lane)
            if flags.local {
                vh = L::max(vh, zero); // SW clamp
                vmax = L::max(vmax, vh);
            }
            L::store(vh, &mut h_store[v * p..]);
            vf = L::max(L::subs(vf, vge), L::subs(vh, vgo)); // F for the next row
            vh = L::load(&h_load[v * p..]); // next stripe's diagonal
        }

        // Lazy-F: within a lane's stripes the main loop already propagated F, but a vertical gap
        // crossing a lane boundary is invisible there. Shift the carried F up one lane and propagate
        // it by extension (`F - gap_ext`) into `H`. A gap crosses at most `p` lane boundaries.
        if flags.local {
            // Local cells are `>= 0`, so a shifted F can only raise some `H` while still positive;
            // stop once it has decayed to `<= 0` everywhere. (Farrar's tighter `F <= H - gap_open`
            // test is unsafe for linear gaps `gap_open == gap_ext`.) Shifting in `0` is inert here.
            for _ in 0..p {
                vf = L::shift_up(vf, 0);
                for v in 0..seg {
                    let vh = L::max(L::load(&h_store[v * p..]), vf);
                    L::store(vh, &mut h_store[v * p..]);
                    vmax = L::max(vmax, vh);
                    vf = L::subs(vf, vge);
                }
                if !L::any_gt(vf, zero) {
                    break;
                }
            }
        } else {
            // Non-local cells can be negative, so the `<= 0` shortcut is unavailable. A shifted F
            // can raise `H[pos]` only while it exceeds `H[pos]`; `H` only rises during the lazy-F,
            // so once the largest active F has decayed below the (fixed) minimum *valid* `H`, no
            // future shift can change anything. Inactive columns — almost all of them — break after
            // a single pass; the `NEG8` insert keeps a fresh F out of lane 0. `p` shifts is the cap.
            let mut hmin = i8::MAX;
            for v in 0..seg {
                let masked = L::max(L::load(&h_store[v * p..]), L::load(&pad_hi[v * p..]));
                hmin = hmin.min(L::hmin(masked));
            }
            for _ in 0..p {
                vf = L::shift_up(vf, NEG8);
                for v in 0..seg {
                    let vh = L::max(L::load(&h_store[v * p..]), vf);
                    L::store(vh, &mut h_store[v * p..]);
                    vf = L::subs(vf, vge);
                }
                if L::hmax(vf) <= hmin {
                    break;
                }
            }
        }

        // Recompute E from the *final* (post-lazy-F) H, so E[i][j+1] sees F's contribution to
        // H[i][j] exactly as the scalar oracle does.
        for v in 0..seg {
            let h = L::load(&h_store[v * p..]);
            let en = L::max(L::subs(h, vgo), L::subs(L::load(&e[v * p..]), vge));
            L::store(en, &mut e[v * p..]);
        }

        if flags.answer_last_row {
            row_best = row_best.max(h_store[last_row] as i32);
        }
    }

    if flags.local {
        return L::hmax(vmax) as i32;
    }
    // Corner `H[m][n]` is always in the answer region; add the last row (HW/OV) and, for OV, the
    // last column (`H[i][n]`, `i = 0..=m`). The `i = 0` cell of the last column is the top border
    // `H[0][n]`; all real rows are read over valid positions only (padding never contributes).
    let mut best = h_store[last_row] as i32; // corner
    if flags.answer_last_row {
        best = best.max(row_best);
    }
    if flags.answer_last_col {
        let top = if flags.top_row_free {
            0
        } else {
            sat_i8(-gap_penalty(go_i, ge_i, tlen)) as i32
        };
        best = best.max(top);
        for pos in 0..qlen {
            best = best.max(h_store[idx(pos)] as i32);
        }
    }
    best
}

/// Dispatch to [`farrar_score`] for the given mode. `_` on the mode keeps the call sites uniform.
fn farrar_mode<L: StripedLanes>(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> i32 {
    farrar_score::<L>(query, target, scoring, &Flags::for_mode(mode))
}

/// Striped score for `mode` on the fastest available SIMD backend for this build, or `None` when
/// none is available (so the caller falls back to the scalar kernel). Callers pass non-empty
/// sequences.
pub(crate) fn farrar_score_simd(
    query: &[u8],
    target: &[u8],
    scoring: &Scoring,
    mode: Mode,
) -> Option<i32> {
    #[cfg(target_arch = "x86_64")]
    {
        x86::score(query, target, scoring, mode)
    }
    #[cfg(target_arch = "aarch64")]
    {
        Some(arm::score(query, target, scoring, mode))
    }
}

/// SSE4.1 backend: 16 `i8` lanes per `__m128i`.
#[cfg(target_arch = "x86_64")]
mod x86 {
    // Intrinsics require `unsafe`; the crate is otherwise `deny(unsafe_code)`.
    #![allow(unsafe_code)]

    use super::{StripedLanes, farrar_mode};
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use core::arch::x86_64::*;

    #[derive(Clone, Copy)]
    pub(super) struct Striped128;

    impl StripedLanes for Striped128 {
        const LANES: usize = 16;
        type V = __m128i;

        #[inline(always)]
        fn splat(v: i8) -> __m128i {
            unsafe { _mm_set1_epi8(v) }
        }
        #[inline(always)]
        fn adds(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_adds_epi8(a, b) }
        }
        #[inline(always)]
        fn subs(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_subs_epi8(a, b) }
        }
        #[inline(always)]
        fn max(a: __m128i, b: __m128i) -> __m128i {
            unsafe { _mm_max_epi8(a, b) }
        }
        #[inline(always)]
        fn shift_up(v: __m128i, insert: i8) -> __m128i {
            unsafe { _mm_insert_epi8::<0>(_mm_slli_si128::<1>(v), insert as i32) }
        }
        #[inline(always)]
        fn hmax(v: __m128i) -> i8 {
            unsafe {
                let v = _mm_max_epi8(v, _mm_srli_si128::<8>(v));
                let v = _mm_max_epi8(v, _mm_srli_si128::<4>(v));
                let v = _mm_max_epi8(v, _mm_srli_si128::<2>(v));
                let v = _mm_max_epi8(v, _mm_srli_si128::<1>(v));
                _mm_extract_epi8::<0>(v) as i8
            }
        }
        #[inline(always)]
        fn hmin(v: __m128i) -> i8 {
            unsafe {
                let v = _mm_min_epi8(v, _mm_srli_si128::<8>(v));
                let v = _mm_min_epi8(v, _mm_srli_si128::<4>(v));
                let v = _mm_min_epi8(v, _mm_srli_si128::<2>(v));
                let v = _mm_min_epi8(v, _mm_srli_si128::<1>(v));
                _mm_extract_epi8::<0>(v) as i8
            }
        }
        #[inline(always)]
        fn any_gt(a: __m128i, b: __m128i) -> bool {
            unsafe { _mm_movemask_epi8(_mm_cmpgt_epi8(a, b)) != 0 }
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
    }

    #[target_feature(enable = "sse4.1")]
    unsafe fn run(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> i32 {
        farrar_mode::<Striped128>(query, target, scoring, mode)
    }

    pub(super) fn score(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> Option<i32> {
        if std::is_x86_feature_detected!("sse4.1") {
            Some(unsafe { run(query, target, scoring, mode) })
        } else {
            None
        }
    }
}

/// NEON backend: 16 `i8` lanes per `int8x16_t`. NEON is baseline on aarch64, so no feature guard.
#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(unsafe_code)]

    use super::{StripedLanes, farrar_mode};
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use core::arch::aarch64::*;

    #[derive(Clone, Copy)]
    pub(super) struct StripedNeon;

    impl StripedLanes for StripedNeon {
        const LANES: usize = 16;
        type V = int8x16_t;

        #[inline(always)]
        fn splat(v: i8) -> int8x16_t {
            unsafe { vdupq_n_s8(v) }
        }
        #[inline(always)]
        fn adds(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vqaddq_s8(a, b) }
        }
        #[inline(always)]
        fn subs(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vqsubq_s8(a, b) }
        }
        #[inline(always)]
        fn max(a: int8x16_t, b: int8x16_t) -> int8x16_t {
            unsafe { vmaxq_s8(a, b) }
        }
        #[inline(always)]
        fn shift_up(v: int8x16_t, insert: i8) -> int8x16_t {
            // out = [insert, v0, .., v14]
            unsafe { vextq_s8::<15>(vdupq_n_s8(insert), v) }
        }
        #[inline(always)]
        fn hmax(v: int8x16_t) -> i8 {
            unsafe { vmaxvq_s8(v) }
        }
        #[inline(always)]
        fn hmin(v: int8x16_t) -> i8 {
            unsafe { vminvq_s8(v) }
        }
        #[inline(always)]
        fn any_gt(a: int8x16_t, b: int8x16_t) -> bool {
            unsafe { vmaxvq_u8(vcgtq_s8(a, b)) != 0 }
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
    }

    pub(super) fn score(query: &[u8], target: &[u8], scoring: &Scoring, mode: Mode) -> i32 {
        farrar_mode::<StripedNeon>(query, target, scoring, mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{DpBuffers, align_core};
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use proptest::prelude::*;

    /// Scalar stand-in: a `[i8; N]` "vector". Validates the algorithm with no intrinsics, and lets
    /// the differential run at lane counts the hardware backends do not use (1, 2, 4, 8).
    struct ScalarStriped<const N: usize>;

    impl<const N: usize> StripedLanes for ScalarStriped<N> {
        const LANES: usize = N;
        type V = [i8; N];

        fn splat(v: i8) -> [i8; N] {
            [v; N]
        }
        fn adds(a: [i8; N], b: [i8; N]) -> [i8; N] {
            core::array::from_fn(|i| a[i].saturating_add(b[i]))
        }
        fn subs(a: [i8; N], b: [i8; N]) -> [i8; N] {
            core::array::from_fn(|i| a[i].saturating_sub(b[i]))
        }
        fn max(a: [i8; N], b: [i8; N]) -> [i8; N] {
            core::array::from_fn(|i| a[i].max(b[i]))
        }
        fn shift_up(v: [i8; N], insert: i8) -> [i8; N] {
            core::array::from_fn(|i| if i == 0 { insert } else { v[i - 1] })
        }
        fn hmax(v: [i8; N]) -> i8 {
            v.into_iter().max().unwrap_or(NEG8)
        }
        fn hmin(v: [i8; N]) -> i8 {
            v.into_iter().min().unwrap_or(i8::MAX)
        }
        fn any_gt(a: [i8; N], b: [i8; N]) -> bool {
            (0..N).any(|i| a[i] > b[i])
        }
        fn load(src: &[i8]) -> [i8; N] {
            core::array::from_fn(|i| src[i])
        }
        fn store(v: [i8; N], dst: &mut [i8]) {
            dst[..N].copy_from_slice(&v);
        }
    }

    fn id_matrix(al: usize, m: i32, x: i32) -> Vec<i32> {
        let mut v = vec![x; al * al];
        for i in 0..al {
            v[i * al + i] = m;
        }
        v
    }

    /// Scorings whose SW scores stay comfortably inside `i8` for short sequences. The last one has
    /// a mismatch below `-127` — legal for `SW` (its i8 width bound is `gap_open`), and the reason
    /// the profile must saturate rather than wrap.
    fn scorings() -> Vec<Scoring> {
        vec![
            Scoring::new(4, id_matrix(4, 2, -1), 2, 1).unwrap(),
            Scoring::new(4, id_matrix(4, 3, -2), 4, 0).unwrap(),
            Scoring::new(4, id_matrix(4, 1, -1), 1, 1).unwrap(),
            Scoring::new(4, id_matrix(4, 5, -4), 6, 3).unwrap(),
            Scoring::new(4, id_matrix(4, 2, -200), 2, 1).unwrap(),
        ]
    }

    const ALL_MODES: [Mode; 4] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov];

    /// The striped kernel realises the DP in `i8`; it matches the oracle exactly only where the
    /// width proof says `i8` suffices — which is precisely where `align_pair` uses it. Returns
    /// `true` when the case is in scope (non-empty and provably `i8`).
    fn in_scope(q: &[u8], t: &[u8], s: &Scoring, mode: Mode) -> bool {
        !q.is_empty()
            && !t.is_empty()
            && matches!(
                s.required_width(mode, q.len(), t.len()),
                Ok(crate::ScoreWidth::I8)
            )
    }

    /// Every lane count (scalar stand-in) and, where available, the hardware backend all agree with
    /// the oracle, for every mode the case is in scope for.
    fn assert_matches_oracle(q: &[u8], t: &[u8], s: &Scoring) {
        for mode in ALL_MODES {
            if !in_scope(q, t, s, mode) {
                continue;
            }
            let want = align_core(q, t, s, mode, &mut DpBuffers::new()).0;
            assert_eq!(
                farrar_mode::<ScalarStriped<1>>(q, t, s, mode),
                want,
                "N=1 {mode} q={q:?} t={t:?}"
            );
            assert_eq!(
                farrar_mode::<ScalarStriped<2>>(q, t, s, mode),
                want,
                "N=2 {mode} q={q:?} t={t:?}"
            );
            assert_eq!(
                farrar_mode::<ScalarStriped<4>>(q, t, s, mode),
                want,
                "N=4 {mode} q={q:?} t={t:?}"
            );
            assert_eq!(
                farrar_mode::<ScalarStriped<8>>(q, t, s, mode),
                want,
                "N=8 {mode} q={q:?} t={t:?}"
            );
            assert_eq!(
                farrar_mode::<ScalarStriped<16>>(q, t, s, mode),
                want,
                "N=16 {mode} q={q:?} t={t:?}"
            );
            if let Some(got) = farrar_score_simd(q, t, s, mode) {
                assert_eq!(got, want, "SIMD {mode} q={q:?} t={t:?}");
            }
        }
    }

    #[test]
    #[ignore = "timing; run with: cargo test --release striped::tests::timing -- --ignored --nocapture"]
    fn timing() {
        use std::time::Instant;
        let s = &scorings()[0];
        let reps = 100;
        // A multiple of the lane count (no padding) and one past it (padding present), so the
        // non-local threshold early-exit is exercised under padding too.
        for len in [2000usize, 2001] {
            let q: Vec<u8> = (0..len as u32)
                .map(|i| (i.wrapping_mul(7) % 4) as u8)
                .collect();
            let t: Vec<u8> = (0..len as u32)
                .map(|i| (i.wrapping_mul(5) % 4) as u8)
                .collect();
            for mode in ALL_MODES {
                let start = Instant::now();
                let mut acc = 0i32;
                for _ in 0..reps {
                    acc ^= align_core(&q, &t, s, mode, &mut DpBuffers::new()).0;
                }
                let scalar = start.elapsed().as_secs_f64();

                let start = Instant::now();
                for _ in 0..reps {
                    acc ^= farrar_score_simd(&q, &t, s, mode).unwrap();
                }
                let simd = start.elapsed().as_secs_f64();

                println!(
                    "{mode} {len}x{len} x{reps}: scalar {:.0}ms  striped {:.0}ms  speedup {:.1}x (acc {acc})",
                    scalar * 1e3,
                    simd * 1e3,
                    scalar / simd,
                );
            }
        }
    }

    #[test]
    fn hand_cases() {
        let s = &scorings()[0];
        assert_matches_oracle(&[2, 2, 0, 1, 2, 3, 2], &[0, 1, 2, 3], s);
        assert_matches_oracle(&[0, 0, 0], &[1, 1, 1], s);
        assert_matches_oracle(&[0, 1, 2, 3, 0, 1], &[0, 1, 3, 0, 1], s);
        assert_matches_oracle(&[2], &[2], s);
        assert_matches_oracle(&[2], &[0], s);
    }

    /// Empty inputs are handled by `align_pair`'s scalar path (the striped kernel is only called
    /// for non-empty sequences); confirm the public entry stays correct for every mode.
    #[test]
    fn empty_inputs_via_align_pair() {
        use crate::{SearchType, align_pair};
        let s = &scorings()[0];
        for mode in ALL_MODES {
            for (q, t) in [
                (&[][..], &[][..]),
                (&[0, 1, 2][..], &[][..]),
                (&[][..], &[0, 1, 2][..]),
            ] {
                let got = align_pair(q, t, s, mode, SearchType::Score).unwrap().score;
                let want = align_core(q, t, s, mode, &mut DpBuffers::new()).0;
                assert_eq!(got, want, "{mode} q={q:?} t={t:?}");
            }
        }
    }

    /// A mismatch below `-127` must clamp, not wrap (the case is in scope for `SW`).
    #[test]
    fn out_of_range_mismatch_saturates() {
        let s = &scorings()[4]; // mismatch = -200
        assert_matches_oracle(&[0, 1, 2, 3, 0, 1, 2, 3], &[1, 0, 3, 2, 1, 0, 3, 2], s);
        assert_matches_oracle(&[0, 0, 1, 1, 2, 2], &[0, 3, 3, 1, 3, 2], s);
    }

    /// Query lengths straddling the 16-lane hardware boundary (padding vs none), and `segLen`
    /// transitions, for a few scorings and both a matching and a shifted target.
    #[test]
    fn length_boundaries_around_lane_count() {
        for s in scorings() {
            for qlen in [1usize, 14, 15, 16, 17, 31, 32, 33, 47, 48] {
                let q: Vec<u8> = (0..qlen).map(|i| (i % 4) as u8).collect();
                for tlen in [1usize, 15, 16, 17, 33] {
                    // A target that matches a prefix of the query, and one offset by a base.
                    let t_match: Vec<u8> = (0..tlen).map(|i| (i % 4) as u8).collect();
                    let t_shift: Vec<u8> = (0..tlen).map(|i| ((i + 1) % 4) as u8).collect();
                    assert_matches_oracle(&q, &t_match, &s);
                    assert_matches_oracle(&q, &t_shift, &s);
                }
            }
        }
    }

    /// Homopolymer / low-complexity inputs force long gap runs, stressing the lazy-F cross-lane
    /// propagation and its early-exit — including gaps that span many lane boundaries.
    #[test]
    fn long_gaps_stress_lazy_f() {
        for s in scorings() {
            // A long run of one symbol vs a short run of another (a big vertical/horizontal gap).
            assert_matches_oracle(&[0u8; 40], &[1u8; 3], &s);
            assert_matches_oracle(&[0u8; 3], &[1u8; 40], &s);
            // A match block buried in long homopolymer padding on the query side.
            let mut q = vec![0u8; 20];
            q.extend_from_slice(&[1, 2, 3, 0, 1]);
            q.extend_from_slice(&[0u8; 20]);
            assert_matches_oracle(&q, &[1, 2, 3, 0, 1], &s);
            // Two match blocks separated by a long query-only insertion (crosses lane boundaries).
            let mut q2 = vec![1u8, 2, 3];
            q2.extend_from_slice(&[0u8; 33]);
            q2.extend_from_slice(&[1, 2, 3]);
            assert_matches_oracle(&q2, &[1, 2, 3, 1, 2, 3], &s);
        }
    }

    /// A large `gap_open` (with `gap_ext = 1`) pushes penalised borders and `E`/`F` cells toward
    /// the `i8` edge; the striped path must still match wherever the width proof says `i8`.
    #[test]
    fn steep_gaps_near_the_i8_edge() {
        let s = Scoring::new(4, id_matrix(4, 10, -3), 60, 1).unwrap();
        assert_matches_oracle(&[0, 1, 2, 3, 0, 1, 2, 3, 0], &[0, 1, 3, 0, 1, 2, 3], &s);
        assert_matches_oracle(&[2u8; 18], &[2, 2, 0, 2, 2], &s);
    }

    /// Exhaustive over all short pairs, scorings, modes and lane counts: pins the striping, the
    /// per-mode borders/answer, and the lazy-F loop.
    #[test]
    fn exhaustive_short_pairs_match_oracle() {
        fn all_seqs(al: u8, maxlen: usize) -> Vec<Vec<u8>> {
            let mut out = vec![vec![]];
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

        let seqs = all_seqs(3, 4);
        for s in scorings() {
            for q in &seqs {
                for t in &seqs {
                    assert_matches_oracle(q, t, &s);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3000))]

        #[test]
        fn random_pairs_match_oracle(
            q in prop::collection::vec(0u8..4, 1..30),
            t in prop::collection::vec(0u8..4, 1..30),
        ) {
            for s in scorings() {
                for mode in ALL_MODES {
                    if !in_scope(&q, &t, &s, mode) { continue; }
                    let want = align_core(&q, &t, &s, mode, &mut DpBuffers::new()).0;
                    prop_assert_eq!(farrar_mode::<ScalarStriped<1>>(&q, &t, &s, mode), want, "N=1 {}", mode);
                    prop_assert_eq!(farrar_mode::<ScalarStriped<8>>(&q, &t, &s, mode), want, "N=8 {}", mode);
                    prop_assert_eq!(farrar_mode::<ScalarStriped<16>>(&q, &t, &s, mode), want, "N=16 {}", mode);
                    if let Some(got) = farrar_score_simd(&q, &t, &s, mode) {
                        prop_assert_eq!(got, want, "SIMD {}", mode);
                    }
                }
            }
        }
    }
}
