//! Which compute backend a [`Database`](crate::Database) resolves to, and how to override it.
//!
//! Reporting *and* controlling the backend is load-bearing, not cosmetic: a benchmark number is
//! meaningless without knowing which kernel ran, the CI matrix must be able to force each backend
//! in turn to exercise the determinism contract, and a downstream tool (rustar) will log the
//! resolved backend so a reproducibility question can be answered from the log rather than by
//! guessing at the user's CPU. See `handover.md` §4 and §7.
//!
//! # Availability in M0
//!
//! All four backend *names* exist so the override API and env var do not churn when the SIMD
//! kernels land, but only [`Backend::Scalar`] is **implemented and available** in M0. Forcing any
//! other backend returns [`Error::BackendUnavailable`](crate::Error::BackendUnavailable) today;
//! it will simply start succeeding once that kernel exists and the CPU supports it.

use crate::error::{Error, Result};
use core::fmt;

/// The environment variable that overrides backend selection at [`build`](crate::DatabaseBuilder::build).
pub const BACKEND_ENV_VAR: &str = "HYALITE_BACKEND";

/// The alignment backend actually used, or one that could be requested.
///
/// `#[non_exhaustive]` — more tiers may be added without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Backend {
    /// The portable scalar reference kernel. Always available; the correctness oracle.
    Scalar,
    /// x86-64 SSE4.1 (16 lanes @ i8). Not yet implemented in M0.
    Sse41,
    /// x86-64 AVX2 (32 lanes @ i8). Not yet implemented in M0.
    Avx2,
    /// aarch64 NEON (16 lanes @ i8). Not yet implemented in M0.
    Neon,
}

impl Backend {
    /// The canonical lowercase name, matching what [`BackendChoice::parse`] accepts.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Backend::Scalar => "scalar",
            Backend::Sse41 => "sse4.1",
            Backend::Avx2 => "avx2",
            Backend::Neon => "neon",
        }
    }

    /// Whether this backend is implemented and usable on the current build/CPU. Gated on both the
    /// target architecture and runtime CPU-feature detection.
    ///
    /// Availability is *not* the whole story for a given database: even an available SIMD backend
    /// is only *used* when the database is SIMD-eligible (i8 width, small alphabet). That extra
    /// gate lives in `DatabaseBuilder::build`.
    #[must_use]
    pub fn is_available(self) -> bool {
        match self {
            Backend::Scalar => true,
            Backend::Sse41 => sse41_detected(),
            Backend::Avx2 => avx2_detected(),
            Backend::Neon => neon_available(),
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn sse41_detected() -> bool {
    std::is_x86_feature_detected!("sse4.1")
}

#[cfg(not(target_arch = "x86_64"))]
fn sse41_detected() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn avx2_detected() -> bool {
    std::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn avx2_detected() -> bool {
    false
}

// NEON is mandatory on aarch64 — always present, no runtime detection needed (handover §5).
#[cfg(target_arch = "aarch64")]
fn neon_available() -> bool {
    true
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_available() -> bool {
    false
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// How a [`Database`](crate::Database) should pick its backend: detect automatically, or force a
/// specific one (for the CI matrix and benchmarking).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum BackendChoice {
    /// Pick the fastest available backend at build time.
    #[default]
    Auto,
    /// Force a specific backend; [`build`](crate::DatabaseBuilder::build) fails with
    /// [`Error::BackendUnavailable`] if it is not available.
    Force(Backend),
}

impl BackendChoice {
    /// Parse a backend choice from a string (case-insensitive). Accepts `auto`, `scalar`,
    /// `sse4.1`/`sse41`, `avx2`, and `neon`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidBackendName`] if the string matches none of those.
    pub fn parse(s: &str) -> Result<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "auto" => Ok(BackendChoice::Auto),
            "scalar" => Ok(BackendChoice::Force(Backend::Scalar)),
            "sse4.1" | "sse41" => Ok(BackendChoice::Force(Backend::Sse41)),
            "avx2" => Ok(BackendChoice::Force(Backend::Avx2)),
            "neon" => Ok(BackendChoice::Force(Backend::Neon)),
            _ => Err(Error::InvalidBackendName {
                name: s.to_string(),
            }),
        }
    }
}

/// The fastest available backend, in descending preference: AVX2, SSE4.1 (x86-64), NEON
/// (aarch64), then scalar.
fn detect_best() -> Backend {
    if Backend::Avx2.is_available() {
        Backend::Avx2
    } else if Backend::Sse41.is_available() {
        Backend::Sse41
    } else if Backend::Neon.is_available() {
        Backend::Neon
    } else {
        Backend::Scalar
    }
}

/// Resolve a [`BackendChoice`] to a concrete, available [`Backend`].
///
/// # Errors
///
/// [`Error::BackendUnavailable`] if a forced backend is not available.
pub(crate) fn resolve(choice: BackendChoice) -> Result<Backend> {
    match choice {
        BackendChoice::Auto => Ok(detect_best()),
        BackendChoice::Force(backend) => {
            if backend.is_available() {
                Ok(backend)
            } else {
                Err(Error::BackendUnavailable { backend })
            }
        }
    }
}

/// The backend choice requested via the [`BACKEND_ENV_VAR`] environment variable, if any.
/// Reading is safe; the variable is consulted once at build time.
///
/// # Errors
///
/// [`Error::InvalidBackendName`] if the variable is set to an unrecognised or non-Unicode value.
pub(crate) fn choice_from_env() -> Result<Option<BackendChoice>> {
    match std::env::var(BACKEND_ENV_VAR) {
        Ok(s) if s.trim().is_empty() => Ok(None),
        Ok(s) => BackendChoice::parse(&s).map(Some),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(Error::InvalidBackendName {
            name: "<non-unicode>".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn availability_matches_arch_and_cpu() {
        // Scalar is always available; x86 SIMD tracks the CPU; NEON is baseline on aarch64.
        assert!(Backend::Scalar.is_available());
        assert_eq!(Backend::Sse41.is_available(), sse41_detected());
        assert_eq!(Backend::Avx2.is_available(), avx2_detected());
        assert_eq!(Backend::Neon.is_available(), neon_available());
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert!(!Backend::Sse41.is_available());
            assert!(!Backend::Avx2.is_available());
        }
        #[cfg(not(target_arch = "aarch64"))]
        assert!(!Backend::Neon.is_available());
    }

    #[test]
    fn name_round_trips_through_parse_for_every_backend() {
        for b in [
            Backend::Scalar,
            Backend::Sse41,
            Backend::Avx2,
            Backend::Neon,
        ] {
            assert_eq!(
                BackendChoice::parse(b.name()).unwrap(),
                BackendChoice::Force(b)
            );
            assert_eq!(b.to_string(), b.name());
        }
    }

    #[test]
    fn parse_accepts_aliases_and_is_case_insensitive() {
        assert_eq!(BackendChoice::parse("auto").unwrap(), BackendChoice::Auto);
        assert_eq!(BackendChoice::parse("AUTO").unwrap(), BackendChoice::Auto);
        assert_eq!(
            BackendChoice::parse("  SSE41 ").unwrap(),
            BackendChoice::Force(Backend::Sse41)
        );
        assert_eq!(
            BackendChoice::parse("sse4.1").unwrap(),
            BackendChoice::Force(Backend::Sse41)
        );
        assert_eq!(
            BackendChoice::parse("Avx2").unwrap(),
            BackendChoice::Force(Backend::Avx2)
        );
    }

    #[test]
    fn parse_rejects_unknown_names() {
        for bad in ["", "sse2", "ssse3", "avx512", "gpu", "x"] {
            let err = BackendChoice::parse(bad).unwrap_err();
            assert_eq!(
                err,
                Error::InvalidBackendName {
                    name: bad.to_string()
                }
            );
        }
    }

    #[test]
    fn resolve_auto_picks_an_available_backend() {
        let b = resolve(BackendChoice::Auto).unwrap();
        assert!(b.is_available());
        // Auto prefers SSE4.1 when the CPU supports it, else scalar.
        assert_eq!(b, detect_best());
    }

    #[test]
    fn resolve_forcing_scalar_always_succeeds() {
        assert_eq!(
            resolve(BackendChoice::Force(Backend::Scalar)).unwrap(),
            Backend::Scalar
        );
    }

    #[test]
    fn resolve_forcing_a_backend_tracks_its_availability() {
        for b in [Backend::Sse41, Backend::Avx2, Backend::Neon] {
            let got = resolve(BackendChoice::Force(b));
            if b.is_available() {
                assert_eq!(got.unwrap(), b);
            } else {
                assert_eq!(got.unwrap_err(), Error::BackendUnavailable { backend: b });
            }
        }
    }
}
