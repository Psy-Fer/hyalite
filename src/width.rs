//! Static width proof.
//!
//! Instead of Opal's runtime overflow *detection* (which it only performs for `SW`, silently
//! producing wrong scores in the other modes), `hyalite` *proves* at construction that a chosen
//! integer width cannot overflow, then runs the kernel in that width infallibly.
//!
//! The bound computed here is **conservative**: it over-estimates the reachable score
//! magnitude, so the chosen width is always safe but may occasionally be wider than strictly
//! necessary. Over-provisioning costs a little performance; it never costs correctness.
//! Tightening the bound is future work and cannot change results, only the width selected.
//!
//! The most-negative representable value of each width is reserved as a saturation sentinel, so
//! the usable magnitude is `TYPE::MAX` (e.g. `[-127, 127]` for `i8`).

use crate::error::{Error, Result};
use crate::mode::Mode;
use crate::scoring::Gaps;

/// The signed integer width a kernel runs in. The most-negative value is reserved as a
/// saturation sentinel and is not a usable score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ScoreWidth {
    /// 8-bit lanes: usable score magnitude `<= 127`.
    I8,
    /// 16-bit lanes: usable score magnitude `<= 32767`.
    I16,
    /// 32-bit lanes: usable score magnitude `<= 2_147_483_647`.
    I32,
}

impl ScoreWidth {
    /// Widths from narrowest to widest — the escalation order.
    const ORDER: [ScoreWidth; 3] = [ScoreWidth::I8, ScoreWidth::I16, ScoreWidth::I32];

    /// The largest score magnitude this width can hold, with the sentinel reserved.
    #[must_use]
    pub const fn max_abs(self) -> i64 {
        match self {
            ScoreWidth::I8 => i8::MAX as i64,
            ScoreWidth::I16 => i16::MAX as i64,
            ScoreWidth::I32 => i32::MAX as i64,
        }
    }

    /// The width in bytes of one lane element.
    #[must_use]
    pub const fn bytes(self) -> usize {
        match self {
            ScoreWidth::I8 => 1,
            ScoreWidth::I16 => 2,
            ScoreWidth::I32 => 4,
        }
    }

    /// The narrowest width whose usable range covers `magnitude`, or `None` if none does.
    #[must_use]
    pub fn narrowest_for(magnitude: i64) -> Option<ScoreWidth> {
        Self::ORDER.into_iter().find(|w| magnitude <= w.max_abs())
    }
}

impl core::fmt::Display for ScoreWidth {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let s = match self {
            ScoreWidth::I8 => "i8",
            ScoreWidth::I16 => "i16",
            ScoreWidth::I32 => "i32",
        };
        f.write_str(s)
    }
}

/// A safe bound on the reachable score magnitude `|score|` for the given inputs.
///
/// This bounds not only the final score but **every** intermediate `H`/`E`/`F` cell — that is what
/// lets a saturating narrow-width backend stay bit-identical to the wide scalar oracle (no *real*
/// cell ever saturates). The `intermediate_cells_fit_the_proven_width` test checks this directly;
/// see `DETERMINISM.md` §2. Computed in `i128` so intermediate products cannot wrap.
///
/// **Positive reach** (all modes): an optimal path has at most `min(m, n)` aligned pairs, each
/// worth at most `max(0, max_entry)`; gaps only subtract. So `score <= min(m, n) * max(0, max_entry)`.
///
/// **Negative reach** is mode-specific, because a *free* end gap lets a path restart at a `0`
/// border and so caps how negative a cell can get:
///
/// - **`SW`** (local, cells clamped at `0`): mismatch negatives vanish, but a gap opening from a
///   `>= 0` cell drives `E`/`F` down to `-gap_open`. → `gap_open`.
/// - **`OV`** (both ends free): every cell is reachable by a pure diagonal from a `0` border in at
///   most `min(m, n)` steps, so `|H| <= min(m, n) * |min_entry|`; `E`/`F` add one gap opening.
///   → `min(m, n) * max(0, -min_entry) + gap_open`.
/// - **`NW` / `HW`** (a penalised border — the whole query/target overhang can be a charged gap):
///   bound a full worst-case mismatch run `(m + n) * max(0, -min_entry)` *and* a full-span gap
///   `gap_open + (m + n - 1) * gap_ext`. This over-counts (a path cannot be all substitutions and
///   all gaps at once), so it is a safe over-estimate.
///
/// Under an asymmetric scheme the two directions can be charged differently; each bound above
/// takes the **larger** open and the larger extend across both, which is exact when the scheme
/// is symmetric and a safe over-estimate otherwise (no gap is charged more than that per base).
fn magnitude_bound(
    mode: Mode,
    min_entry: i32,
    max_entry: i32,
    gaps: Gaps,
    max_query_len: usize,
    max_target_len: usize,
) -> i128 {
    let m = max_query_len as i128;
    let n = max_target_len as i128;
    let min_ij = m.min(n);
    let max_pos = (max_entry as i128).max(0); // max(0, max_entry)
    let max_neg = (-(min_entry as i128)).max(0); // max(0, -min_entry) = |min_entry| when negative
    let (gap_open, gap_ext) = gaps.worst();
    let go = gap_open as i128;
    let ge = gap_ext as i128;

    let positive = min_ij * max_pos;

    // Exhaustive match (allowed within the defining crate despite `#[non_exhaustive]`): a new mode
    // must consciously choose its negative-reach bound rather than silently inherit one.
    let negative = match mode {
        Mode::Sw => go,
        Mode::Ov => min_ij * max_neg + go,
        // `Shw` is the transpose of `Hw`: one penalised border (the target overhang can be a
        // charged gap), so it takes the same wide bound as `Nw`/`Hw`.
        Mode::Nw | Mode::Hw | Mode::Shw => (m + n) * max_neg + go + (m + n - 1).max(0) * ge,
    };

    positive.max(negative)
}

/// Prove the narrowest [`ScoreWidth`] that cannot overflow for these inputs, or return
/// [`Error::ScoreRangeTooWide`] if even `i32` is insufficient.
///
/// `min_entry` / `max_entry` are the extreme substitution-matrix entries; `gaps` holds the
/// non-negative penalty magnitudes for both gap directions; the lengths are the maxima over all
/// sequences the width must cover.
pub fn required_width(
    mode: Mode,
    min_entry: i32,
    max_entry: i32,
    gaps: Gaps,
    max_query_len: usize,
    max_target_len: usize,
) -> Result<ScoreWidth> {
    let bound = magnitude_bound(
        mode,
        min_entry,
        max_entry,
        gaps,
        max_query_len,
        max_target_len,
    );

    // Clamp the reported bound into i64 so the error stays informative even for pathological
    // (m * n) products that exceed i64 but are already far past i32.
    let bound_i64 = bound.min(i64::MAX as i128) as i64;

    ScoreWidth::narrowest_for(bound_i64).ok_or(Error::ScoreRangeTooWide { bound: bound_i64 })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];

    /// Symmetric-scheme shims with the pre-asymmetric signature. These shadow the glob-imported
    /// items of the same name, so the bound tests below read as they did when there was one
    /// penalty pair; the asymmetric cases are covered separately.
    fn required_width(
        mode: Mode,
        min_entry: i32,
        max_entry: i32,
        gap_open: i32,
        gap_ext: i32,
        max_query_len: usize,
        max_target_len: usize,
    ) -> Result<ScoreWidth> {
        super::required_width(
            mode,
            min_entry,
            max_entry,
            Gaps::symmetric(gap_open, gap_ext),
            max_query_len,
            max_target_len,
        )
    }

    fn magnitude_bound(
        mode: Mode,
        min_entry: i32,
        max_entry: i32,
        gap_open: i32,
        gap_ext: i32,
        max_query_len: usize,
        max_target_len: usize,
    ) -> i128 {
        super::magnitude_bound(
            mode,
            min_entry,
            max_entry,
            Gaps::symmetric(gap_open, gap_ext),
            max_query_len,
            max_target_len,
        )
    }

    #[test]
    fn asymmetric_gaps_bound_by_the_worse_direction() {
        // The proof takes max(open) and max(ext) over both directions, so an asymmetric scheme
        // picks exactly the width its worse half would.
        for (q, t) in [((200, 1), (2, 1)), ((2, 1), (200, 1))] {
            let asym = Gaps {
                query_open: q.0,
                query_ext: q.1,
                target_open: t.0,
                target_ext: t.1,
            };
            assert_eq!(
                super::required_width(Mode::Sw, -1, 1, asym, 30, 30).unwrap(),
                required_width(Mode::Sw, -1, 1, 200, 1, 30, 30).unwrap(),
                "SW must widen for the larger gap_open, whichever direction carries it"
            );
        }
        // And a scheme whose halves agree is indistinguishable from the symmetric one.
        assert_eq!(
            super::magnitude_bound(Mode::Nw, -2, 1, Gaps::symmetric(5, 3), 91, 33),
            magnitude_bound(Mode::Nw, -2, 1, 5, 3, 91, 33)
        );
    }

    #[test]
    fn width_constants_are_sane() {
        assert_eq!(ScoreWidth::I8.max_abs(), 127);
        assert_eq!(ScoreWidth::I16.max_abs(), 32_767);
        assert_eq!(ScoreWidth::I32.max_abs(), 2_147_483_647);
        assert!(ScoreWidth::I8 < ScoreWidth::I16 && ScoreWidth::I16 < ScoreWidth::I32);
        assert_eq!(
            (
                ScoreWidth::I8.bytes(),
                ScoreWidth::I16.bytes(),
                ScoreWidth::I32.bytes()
            ),
            (1, 2, 4)
        );
    }

    #[test]
    fn narrowest_for_picks_smallest_fit_across_boundaries() {
        // Exhaustively probe each boundary from both sides.
        for (mag, expect) in [
            (0, Some(ScoreWidth::I8)),
            (127, Some(ScoreWidth::I8)),
            (128, Some(ScoreWidth::I16)),
            (32_767, Some(ScoreWidth::I16)),
            (32_768, Some(ScoreWidth::I32)),
            (2_147_483_647, Some(ScoreWidth::I32)),
            (2_147_483_648, None),
            (i64::MAX, None),
        ] {
            assert_eq!(ScoreWidth::narrowest_for(mag), expect, "magnitude {mag}");
        }
    }

    #[test]
    fn small_local_alignment_fits_i8() {
        // 10x10, match +2, mismatch -1: positive reach = 10*2 = 20; SW ignores negatives.
        let w = required_width(Mode::Sw, -1, 2, 3, 1, 10, 10).unwrap();
        assert_eq!(w, ScoreWidth::I8);
    }

    #[test]
    fn local_mode_ignores_mismatch_negatives_but_not_gap_open() {
        // SW clamps cells at 0, so an enormous *mismatch* penalty is irrelevant: with a tiny gap
        // it stays i8 (positive reach min(m,n)*max_e = 8). The non-local modes escalate on the
        // same mismatch. But SW's E/F reach -gap_open, so a huge *gap_open* does force SW wider.
        let (m, n) = (8, 8);
        assert_eq!(
            required_width(Mode::Sw, -100, 1, 2, 1, m, n).unwrap(),
            ScoreWidth::I8,
            "SW ignores the -100 mismatch; positive reach is only 8, gap_open 2"
        );
        for mode in [Mode::Nw, Mode::Hw, Mode::Ov] {
            assert!(
                required_width(mode, -100, 1, 2, 1, m, n).unwrap() > ScoreWidth::I8,
                "{mode} must escalate on the -100 mismatch"
            );
        }
        // A gap_open beyond the i8 range drives SW's E/F below -127, so it must widen.
        assert_eq!(
            required_width(Mode::Sw, -1, 1, 200, 1, m, n).unwrap(),
            ScoreWidth::I16,
            "SW must widen for gap_open 200 (E/F reach -200)"
        );
    }

    #[test]
    fn overlap_bound_is_tight_enough_for_cr4_but_global_is_not() {
        // The CR4 workload: overlap mode, 91 nt reads vs ~33 nt adapters, mismatch -2, gap 2.
        // Overlap's free ends cap the negative reach at min(91,33)*2 + 2 = 68, so it stays i8 and
        // the SIMD kernel applies. The same inputs in global (NW) accumulate a full-span gap and
        // so need i16 — correctly, since NW scores of dissimilar long sequences are very negative.
        assert_eq!(
            required_width(Mode::Ov, -2, 1, 2, 2, 91, 33).unwrap(),
            ScoreWidth::I8,
            "CR4 overlap must stay i8"
        );
        assert_eq!(
            required_width(Mode::Nw, -2, 1, 2, 2, 91, 33).unwrap(),
            ScoreWidth::I16,
            "the same lengths in global mode need i16"
        );
    }

    #[test]
    fn escalates_i8_to_i16_to_i32_as_length_grows() {
        // Match score 100; positive reach = min(m,n)*100. Walk lengths across both boundaries
        // for every mode and assert monotonic, correct escalation.
        for mode in ALL_MODES {
            let mut prev = ScoreWidth::I8;
            for &len in &[1usize, 2, 100, 300, 327, 328, 1000, 22_000, 50_000] {
                let w = required_width(mode, 0, 100, 0, 0, len, len).unwrap();
                assert!(w >= prev, "{mode}: width went backwards at len {len}");
                prev = w;
            }
            // reach = len*100. Probe both sides of each width boundary:
            // len 1   -> 100      -> i8
            // len 327 -> 32_700   -> i16   (<= 32_767)
            // len 328 -> 32_800   -> i32   (just over i16)
            // len 50_000 -> 5e6   -> i32
            assert_eq!(
                required_width(mode, 0, 100, 0, 0, 1, 1).unwrap(),
                ScoreWidth::I8
            );
            assert_eq!(
                required_width(mode, 0, 100, 0, 0, 327, 327).unwrap(),
                ScoreWidth::I16
            );
            assert_eq!(
                required_width(mode, 0, 100, 0, 0, 328, 328).unwrap(),
                ScoreWidth::I32
            );
            assert_eq!(
                required_width(mode, 0, 100, 0, 0, 50_000, 50_000).unwrap(),
                ScoreWidth::I32
            );
        }
    }

    #[test]
    fn positive_reach_uses_min_of_the_two_lengths() {
        // A short query against a long target: positive reach is bounded by the shorter one.
        // reach = min(2, 1e6) * 100 = 200, which needs i16 — the long target does not force i32.
        let short_long = required_width(Mode::Sw, 0, 100, 0, 0, 2, 1_000_000).unwrap();
        assert_eq!(short_long, ScoreWidth::I16);
        // Drop the query to length 1: reach = 100, back within i8 despite the huge target.
        assert_eq!(
            required_width(Mode::Sw, 0, 100, 0, 0, 1, 1_000_000).unwrap(),
            ScoreWidth::I8
        );
    }

    #[test]
    fn zero_length_inputs_never_panic_and_fit_i8() {
        for mode in ALL_MODES {
            // Empty sequences: no aligned pairs, gap run length clamps at zero.
            let w = required_width(mode, -5, 5, 4, 2, 0, 0).unwrap();
            assert_eq!(w, ScoreWidth::I8, "{mode} on empty inputs");
        }
    }

    #[test]
    fn overflowing_i32_is_a_typed_error_not_a_panic() {
        // Global alignment of two ~2-billion-long sequences with unit mismatch penalty blows
        // past i32. Must be a clean ScoreRangeTooWide, not an overflow panic.
        let err = required_width(Mode::Nw, -1, 1, 0, 1, 2_000_000_000, 2_000_000_000).unwrap_err();
        match err {
            Error::ScoreRangeTooWide { bound } => assert!(bound > i32::MAX as i64),
            other => panic!("expected ScoreRangeTooWide, got {other:?}"),
        }
    }

    #[test]
    fn pathological_lengths_do_not_overflow_the_bound_computation() {
        // usize::MAX lengths would overflow i64 products; the i128 computation and the i64
        // clamp must keep this a graceful error.
        let err = required_width(Mode::Ov, -1, 1, 1, 1, usize::MAX, usize::MAX).unwrap_err();
        assert!(matches!(err, Error::ScoreRangeTooWide { .. }));
    }

    #[test]
    fn chosen_width_actually_contains_the_bound() {
        // The core contract: for a spread of inputs, whatever width is returned must have a
        // usable range that covers the conservative bound — and the next width down must not.
        let cases = [
            (Mode::Sw, -1, 2, 3, 1, 50usize, 50usize),
            (Mode::Nw, -4, 5, 8, 2, 200, 200),
            (Mode::Hw, -2, 3, 6, 1, 500, 30),
            (Mode::Ov, -7, 7, 10, 3, 4000, 4000),
            (Mode::Nw, -1, 1, 0, 1, 40_000, 40_000),
        ];
        for (mode, min_e, max_e, go, ge, m, n) in cases {
            let bound = magnitude_bound(mode, min_e, max_e, go, ge, m, n);
            let w = required_width(mode, min_e, max_e, go, ge, m, n).unwrap();
            assert!(
                bound <= w.max_abs() as i128,
                "{mode}: bound {bound} > {w} range"
            );
            // The selection is minimal: no narrower width would have fit.
            for narrower in ScoreWidth::ORDER.into_iter().take_while(|&x| x < w) {
                assert!(
                    bound > narrower.max_abs() as i128,
                    "{mode}: {narrower} would also have fit bound {bound}; selection not minimal"
                );
            }
        }
    }
}
