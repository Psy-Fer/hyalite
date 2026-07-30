//! Alignment modes.

use core::fmt;

/// The alignment mode: which sequence ends are free (unpenalised) and whether the score is
/// clamped to be non-negative (local).
///
/// Naming follows Opal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Mode {
    /// Smith-Waterman local alignment. Scores are clamped at zero; the best local score
    /// anywhere in the matrix wins.
    Sw,
    /// Needleman-Wunsch global alignment. Both sequences are aligned end to end; all end gaps
    /// are penalised.
    Nw,
    /// Semi-global ("half-Waterman"): both ends of the **target** are free, so the whole query
    /// is placed optimally within (as a substring of) a longer target.
    Hw,
    /// Overlap alignment: gaps at the end of either sequence are free, scoring the best
    /// suffix-prefix overlap.
    Ov,
    /// The transpose of [`Hw`](Mode::Hw): both ends of the **query** are free, so the whole
    /// target is placed optimally within (as a substring of) a longer query (Opal issue #29).
    Shw,
}

impl Mode {
    /// Whether this mode clamps cell scores at zero (i.e. is local). Only [`Mode::Sw`] does.
    #[must_use]
    pub const fn is_local(self) -> bool {
        matches!(self, Mode::Sw)
    }

    /// A short uppercase code for this mode (`"SW"`, `"NW"`, `"HW"`, `"OV"`, `"SHW"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Mode::Sw => "SW",
            Mode::Nw => "NW",
            Mode::Hw => "HW",
            Mode::Ov => "OV",
            Mode::Shw => "SHW",
        }
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_sw_is_local() {
        assert!(Mode::Sw.is_local());
        for m in [Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw] {
            assert!(!m.is_local(), "{m} should not be local");
        }
    }

    #[test]
    fn code_and_display_agree_for_every_mode() {
        for (m, code) in [
            (Mode::Sw, "SW"),
            (Mode::Nw, "NW"),
            (Mode::Hw, "HW"),
            (Mode::Ov, "OV"),
            (Mode::Shw, "SHW"),
        ] {
            assert_eq!(m.code(), code);
            assert_eq!(m.to_string(), code);
        }
    }

    #[test]
    fn modes_are_distinct() {
        let all = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                assert_eq!(i == j, a == b, "equality mismatch for {a} vs {b}");
            }
        }
    }
}
