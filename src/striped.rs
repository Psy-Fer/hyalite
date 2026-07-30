//! Striped (Farrar) intra-sequence SIMD alignment — algorithm validation.
//!
//! Where the inter-sequence kernel ([`crate::inter`]) puts one database sequence per lane, the
//! striped kernel vectorises a **single** pairwise alignment: the query is laid out in `p` lanes
//! of `segLen = ceil(qlen / p)` stripes, and the DP marches over target columns with a
//! horizontally-vectorised inner loop plus Farrar's "lazy-F" correction for the cross-lane
//! vertical-gap dependency.
//!
//! This module currently holds the algorithm proven on a **scalar** stand-in (`ScalarStriped<N>`),
//! exactly as the in-vector ScoreEnd work was proven on `ScalarLanes` before any intrinsics. It is
//! `#[cfg(test)]`: the point is to pin the striping, the lazy-F loop, and the boundary sentinels
//! against the scalar oracle ([`crate::kernel::align_core`]) before the real SSE4.1/AVX2/NEON
//! backends are written against this same generic kernel.
//!
//! Scope (see the module history): `Score` only, bit-identical to the oracle. `ScoreEnd` and
//! `Alignment` stay on the scalar path; striped end-position tracking is a separate concern.

use crate::scoring::Scoring;

/// The `-∞` sentinel at i8 width, matching the scalar kernel's `NEG` for unreachable `E`/`F`
/// cells. Saturating subtraction keeps it pinned, so it behaves as a true floor.
const NEG8: i8 = i8::MIN;

/// The lane operations the striped kernel needs. Implemented here by a scalar array stand-in; the
/// SIMD backends will implement the same surface with intrinsics.
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
    fn load(src: &[i8]) -> Self::V;
    fn store(v: Self::V, dst: &mut [i8]);
}

/// Scalar stand-in: a `[i8; N]` "vector". Validates the algorithm with no intrinsics.
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
    fn load(src: &[i8]) -> [i8; N] {
        core::array::from_fn(|i| src[i])
    }
    fn store(v: [i8; N], dst: &mut [i8]) {
        dst[..N].copy_from_slice(&v);
    }
}

/// Striped Smith-Waterman **score** for one query/target pair, in the saturating `i8` model.
///
/// Returns the same value as the scalar oracle in `SW` mode when the score provably fits `i8`
/// (which the caller's width proof guarantees). The result is a horizontal max reduction, so it is
/// independent of the lane count — the property the SIMD backends will rely on for determinism.
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
    // padding lanes (query positions past `qlen`), so padding can never contribute a positive cell.
    let mut profile = vec![NEG8; al * seg * p];
    for (t, chunk) in profile.chunks_mut(seg * p).enumerate() {
        for v in 0..seg {
            for l in 0..p {
                let qpos = l * seg + v;
                if qpos < qlen {
                    chunk[v * p + l] = scoring.score(query[qpos] as usize, t) as i8;
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
        // that crosses a lane boundary is invisible there. Shift the carried F up one lane and
        // propagate it by *extension* (`F - gap_ext`) into `H`. A gap crosses at most `p` lane
        // boundaries, so `p` shifts always suffice. (An early exit once no lane can still improve
        // is a correctness-neutral optimisation deferred to the SIMD backends.)
        for _ in 0..p {
            vf = L::shift_up(vf, NEG8);
            for v in 0..seg {
                let vh = L::max(L::load(&h_store[v * p..]), vf);
                L::store(vh, &mut h_store[v * p..]);
                vmax = L::max(vmax, vh);
                vf = L::subs(vf, vge);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{DpBuffers, align_core};
    use crate::mode::Mode;
    use crate::scoring::Scoring;
    use proptest::prelude::*;

    fn id_matrix(al: usize, m: i32, x: i32) -> Vec<i32> {
        let mut v = vec![x; al * al];
        for i in 0..al {
            v[i * al + i] = m;
        }
        v
    }

    /// Scorings whose SW scores stay comfortably inside `i8` for short sequences.
    fn scorings() -> Vec<Scoring> {
        vec![
            Scoring::new(4, id_matrix(4, 2, -1), 2, 1).unwrap(),
            Scoring::new(4, id_matrix(4, 3, -2), 4, 0).unwrap(),
            Scoring::new(4, id_matrix(4, 1, -1), 1, 1).unwrap(),
            Scoring::new(4, id_matrix(4, 5, -4), 6, 3).unwrap(),
        ]
    }

    fn oracle(query: &[u8], target: &[u8], s: &Scoring) -> i32 {
        align_core(query, target, s, Mode::Sw, &mut DpBuffers::new()).0
    }

    /// Run every supported lane count against the oracle for one pair/scoring.
    fn assert_all_lane_counts(q: &[u8], t: &[u8], s: &Scoring) {
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
    }

    #[test]
    fn hand_cases() {
        let s = &scorings()[0];
        // Exact substring: score = 4 matches * 2.
        assert_all_lane_counts(&[2, 2, 0, 1, 2, 3, 2], &[0, 1, 2, 3], s);
        // All-mismatch: local score 0.
        assert_all_lane_counts(&[0, 0, 0], &[1, 1, 1], s);
        // A gap in the middle.
        assert_all_lane_counts(&[0, 1, 2, 3, 0, 1], &[0, 1, 3, 0, 1], s);
        // Single bases.
        assert_all_lane_counts(&[2], &[2], s);
        assert_all_lane_counts(&[2], &[0], s);
    }

    #[test]
    fn empty_inputs() {
        let s = &scorings()[0];
        assert_all_lane_counts(&[], &[], s);
        assert_all_lane_counts(&[0, 1, 2], &[], s);
        assert_all_lane_counts(&[], &[0, 1, 2], s);
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

        // All pairs over a 3-symbol alphabet up to length 4 (120 sequences), every scoring and
        // lane count. Quadratic in the sequence set, so length 4 keeps it quick; the random
        // proptest below reaches longer sequences.
        let seqs = all_seqs(3, 4);
        for s in scorings() {
            for q in &seqs {
                for t in &seqs {
                    assert_all_lane_counts(q, t, &s);
                }
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(3000))]

        #[test]
        fn random_pairs_match_oracle(
            q in prop::collection::vec(0u8..4, 0..20),
            t in prop::collection::vec(0u8..4, 0..20),
        ) {
            for s in scorings() {
                let want = oracle(&q, &t, &s);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<1>>(&q, &t, &s), want);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<4>>(&q, &t, &s), want);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<8>>(&q, &t, &s), want);
                prop_assert_eq!(farrar_sw_score::<ScalarStriped<16>>(&q, &t, &s), want);
            }
        }
    }
}
