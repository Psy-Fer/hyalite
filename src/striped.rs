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
//! [`StripedLanes`] surface and run the same generic [`farrar_sw_score`], so both are bit-identical
//! to the scalar oracle.
//!
//! Scope: `SW` **`Score`** only, `i8` width. `ScoreEnd`/`Alignment`/wider widths stay on the scalar
//! path — striped end-position tracking is a separate concern, and the single-pair path is not the
//! throughput-critical one (that is the batched database scan).

use crate::scoring::Scoring;

/// The `-∞` sentinel at i8 width, matching the scalar kernel's `NEG` for unreachable `E`/`F`
/// cells. Saturating subtraction keeps it pinned, so it behaves as a true floor.
const NEG8: i8 = i8::MIN;

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
    /// Whether any lane of `a` is strictly greater than the matching lane of `b`.
    fn any_gt(a: Self::V, b: Self::V) -> bool;
    fn load(src: &[i8]) -> Self::V;
    fn store(v: Self::V, dst: &mut [i8]);
}

/// Striped Smith-Waterman **score** for one query/target pair, in the saturating `i8` model.
///
/// Returns the same value as the scalar oracle in `SW` mode when the score provably fits `i8`
/// (which the caller's width proof guarantees). The result is a horizontal max reduction, so it is
/// independent of the lane count — the property the SIMD backends rely on for determinism.
fn farrar_sw_score<L: StripedLanes>(query: &[u8], target: &[u8], scoring: &Scoring) -> i32 {
    let p = L::LANES;
    let qlen = query.len();
    if qlen == 0 || target.is_empty() {
        return 0; // SW: the empty alignment scores 0
    }
    let seg = qlen.div_ceil(p);
    let (go, ge) = (scoring.gap_open() as i8, scoring.gap_ext() as i8);
    let al = scoring.alphabet_len();

    // Query profile, striped: profile[t][v*p + l] = score(query[l*seg + v], t), or NEG8 for the
    // padding lanes. Entries are **saturated** into `i8`: for `SW` the width proof bounds the score
    // magnitude but not `|min_entry|`, so a mismatch below `-127` must clamp (not wrap) — it then
    // drives `H` to the `0` floor exactly as the i32 oracle does.
    let mut profile = vec![NEG8; al * seg * p];
    for (t, chunk) in profile.chunks_mut(seg * p).enumerate() {
        for v in 0..seg {
            for l in 0..p {
                let qpos = l * seg + v;
                if qpos < qlen {
                    let s = scoring.score(query[qpos] as usize, t);
                    chunk[v * p + l] = s.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
                }
            }
        }
    }

    let mut h_store = vec![0i8; seg * p]; // H[*][j]   (column 0 border = 0, SW left free)
    let mut h_load = vec![0i8; seg * p]; //  H[*][j-1]
    let mut e = vec![NEG8; seg * p]; //      E[*][j]   (E[i][0] = NEG)
    let vgo = L::splat(go);
    let vge = L::splat(ge);
    let zero = L::splat(0);
    let mut vmax = zero;

    for &tt in target {
        let prof = &profile[tt as usize * seg * p..];

        // Diagonal seed: H[*][j-1] of the last stripe, shifted up with the top-left border (0).
        let mut vh = L::shift_up(L::load(&h_store[(seg - 1) * p..]), 0);
        core::mem::swap(&mut h_store, &mut h_load);
        let mut vf = L::splat(NEG8); // F[0][j] = NEG (no vertical gap ends at row 0)

        for v in 0..seg {
            vh = L::adds(vh, L::load(&prof[v * p..])); // H[i-1][j-1] + score
            vh = L::max(vh, L::load(&e[v * p..])); // vs E[i][j]
            vh = L::max(vh, vf); // vs F (partial; lazy-F fixes cross-lane)
            vh = L::max(vh, zero); // SW clamp
            vmax = L::max(vmax, vh);
            L::store(vh, &mut h_store[v * p..]);
            vf = L::max(L::subs(vf, vge), L::subs(vh, vgo)); // F for the next row
            vh = L::load(&h_load[v * p..]); // next stripe's diagonal
        }

        // Lazy-F: within a lane's stripes the main loop already propagated F, but a vertical gap
        // crossing a lane boundary is invisible there. Shift the carried F up one lane and propagate
        // it by *extension* (`F - gap_ext`) into `H`. Because every `SW` cell is `>= 0`, a shifted
        // F can only raise some `H` while it is still positive somewhere; once it has decayed to
        // `<= 0` across all lanes no further shift can change anything, so we stop. (Farrar's tighter
        // `F <= H - gap_open` test is unsafe for linear gaps `gap_open == gap_ext`, where it stops
        // early.) The hard cap is `p` shifts — the most lane boundaries a gap can cross — but the
        // exponential decay means the common case is one or two.
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

        // Recompute E from the *final* (post-lazy-F) H, so E[i][j+1] sees F's contribution to
        // H[i][j] exactly as the scalar oracle does.
        for v in 0..seg {
            let h = L::load(&h_store[v * p..]);
            let en = L::max(L::subs(h, vgo), L::subs(L::load(&e[v * p..]), vge));
            L::store(en, &mut e[v * p..]);
        }
    }

    L::hmax(vmax) as i32
}

/// Striped `SW` score on the fastest available SIMD backend for this build, or `None` when none is
/// available (so the caller falls back to the scalar kernel).
pub(crate) fn farrar_sw_score_simd(query: &[u8], target: &[u8], scoring: &Scoring) -> Option<i32> {
    #[cfg(target_arch = "x86_64")]
    {
        x86::score(query, target, scoring)
    }
    #[cfg(target_arch = "aarch64")]
    {
        Some(arm::score(query, target, scoring))
    }
}

/// SSE4.1 backend: 16 `i8` lanes per `__m128i`.
#[cfg(target_arch = "x86_64")]
mod x86 {
    // Intrinsics require `unsafe`; the crate is otherwise `deny(unsafe_code)`.
    #![allow(unsafe_code)]

    use super::{StripedLanes, farrar_sw_score};
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
    unsafe fn run(query: &[u8], target: &[u8], scoring: &Scoring) -> i32 {
        farrar_sw_score::<Striped128>(query, target, scoring)
    }

    pub(super) fn score(query: &[u8], target: &[u8], scoring: &Scoring) -> Option<i32> {
        if std::is_x86_feature_detected!("sse4.1") {
            Some(unsafe { run(query, target, scoring) })
        } else {
            None
        }
    }
}

/// NEON backend: 16 `i8` lanes per `int8x16_t`. NEON is baseline on aarch64, so no feature guard.
#[cfg(target_arch = "aarch64")]
mod arm {
    #![allow(unsafe_code)]

    use super::{StripedLanes, farrar_sw_score};
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

    pub(super) fn score(query: &[u8], target: &[u8], scoring: &Scoring) -> i32 {
        farrar_sw_score::<StripedNeon>(query, target, scoring)
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

    fn oracle(query: &[u8], target: &[u8], s: &Scoring) -> i32 {
        align_core(query, target, s, Mode::Sw, &mut DpBuffers::new()).0
    }

    /// Every supported lane count (scalar stand-in) and, where available, the hardware backend all
    /// agree with the oracle for one pair/scoring.
    fn assert_matches_oracle(q: &[u8], t: &[u8], s: &Scoring) {
        let want = oracle(q, t, s);
        assert_eq!(
            farrar_sw_score::<ScalarStriped<1>>(q, t, s),
            want,
            "N=1 q={q:?} t={t:?}"
        );
        assert_eq!(
            farrar_sw_score::<ScalarStriped<2>>(q, t, s),
            want,
            "N=2 q={q:?} t={t:?}"
        );
        assert_eq!(
            farrar_sw_score::<ScalarStriped<4>>(q, t, s),
            want,
            "N=4 q={q:?} t={t:?}"
        );
        assert_eq!(
            farrar_sw_score::<ScalarStriped<8>>(q, t, s),
            want,
            "N=8 q={q:?} t={t:?}"
        );
        assert_eq!(
            farrar_sw_score::<ScalarStriped<16>>(q, t, s),
            want,
            "N=16 q={q:?} t={t:?}"
        );
        if let Some(got) = farrar_sw_score_simd(q, t, s) {
            assert_eq!(got, want, "SIMD q={q:?} t={t:?}");
        }
    }

    #[test]
    #[ignore = "timing; run with: cargo test --release striped::tests::timing -- --ignored --nocapture"]
    fn timing() {
        use std::time::Instant;
        let s = &scorings()[0];
        let q: Vec<u8> = (0..2000u32)
            .map(|i| (i.wrapping_mul(7) % 4) as u8)
            .collect();
        let t: Vec<u8> = (0..2000u32)
            .map(|i| (i.wrapping_mul(5) % 4) as u8)
            .collect();
        let reps = 200;

        let start = Instant::now();
        let mut acc = 0i32;
        for _ in 0..reps {
            acc ^= align_core(&q, &t, s, Mode::Sw, &mut DpBuffers::new()).0;
        }
        let scalar = start.elapsed().as_secs_f64();

        let start = Instant::now();
        for _ in 0..reps {
            acc ^= farrar_sw_score_simd(&q, &t, s).unwrap();
        }
        let simd = start.elapsed().as_secs_f64();

        println!(
            "SW score, 2000x2000, {reps} reps: scalar {:.1}ms  striped {:.1}ms  speedup {:.1}x  (acc {acc})",
            scalar * 1e3,
            simd * 1e3,
            scalar / simd,
        );
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

    #[test]
    fn empty_inputs() {
        let s = &scorings()[0];
        assert_matches_oracle(&[], &[], s);
        assert_matches_oracle(&[0, 1, 2], &[], s);
        assert_matches_oracle(&[], &[0, 1, 2], s);
    }

    /// A mismatch below `-127` must clamp, not wrap. This forces a long mismatched region.
    #[test]
    fn out_of_range_mismatch_saturates() {
        let s = &scorings()[4]; // mismatch = -200
        assert_matches_oracle(&[0, 1, 2, 3, 0, 1, 2, 3], &[1, 0, 3, 2, 1, 0, 3, 2], s);
        assert_matches_oracle(&[0, 0, 1, 1, 2, 2], &[0, 3, 3, 1, 3, 2], s);
    }

    /// Exhaustive over all short pairs and several scorings: the striped score equals the oracle
    /// for every lane count. This is what pins the striping and lazy-F loop.
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
            q in prop::collection::vec(0u8..4, 0..30),
            t in prop::collection::vec(0u8..4, 0..30),
        ) {
            for s in scorings() {
                let want = oracle(&q, &t, &s);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<1>>(&q, &t, &s), want);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<8>>(&q, &t, &s), want);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<16>>(&q, &t, &s), want);
                if let Some(got) = farrar_sw_score_simd(&q, &t, &s) {
                    prop_assert_eq!(got, want);
                }
            }
        }
    }
}
