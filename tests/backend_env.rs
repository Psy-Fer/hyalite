//! Tests for the `HYALITE_BACKEND` environment-variable override.
//!
//! `HYALITE_BACKEND` is process-global, and in Rust 2024 `set_var` is `unsafe` because it is not
//! thread-safe. To avoid racing other tests that read the variable during `build()`, this lives in
//! its own integration-test binary (a separate process) and is the **only** test in the file, so
//! nothing else runs concurrently while the variable is mutated. Backend *logic* (parse/resolve)
//! is covered by pure unit tests in `src/backend.rs`; this only checks that the variable is wired
//! into `build()` with the right precedence.

use hyalite::{BACKEND_ENV_VAR, Backend, BackendChoice, Database, Error, Mode, Scoring};

fn scoring() -> Scoring {
    Scoring::new(2, vec![1, -1, -1, 1], 2, 1).unwrap()
}

/// Build a minimal database *without* an explicit backend choice, so `HYALITE_BACKEND` is what
/// decides.
fn build_using_env() -> hyalite::Result<Database> {
    Database::builder()
        .sequences(&[vec![0u8, 1]])
        .scoring(scoring())
        .mode(Mode::Sw)
        .max_query_len(4)
        .build()
}

/// Build with an explicit choice that must win over the environment.
fn build_forcing(choice: BackendChoice) -> hyalite::Result<Database> {
    Database::builder()
        .sequences(&[vec![0u8, 1]])
        .scoring(scoring())
        .mode(Mode::Sw)
        .max_query_len(4)
        .backend(choice)
        .build()
}

#[test]
fn hyalite_backend_env_var_is_honored_with_correct_precedence() {
    // SAFETY: single-threaded, sole test in this process; no other thread reads the environment
    // concurrently. We restore the variable's absence at each step.
    let set = |v: &str| unsafe { std::env::set_var(BACKEND_ENV_VAR, v) };
    let clear = || unsafe { std::env::remove_var(BACKEND_ENV_VAR) };

    clear();
    assert!(
        build_using_env().unwrap().backend().is_available(),
        "unset env should auto-resolve to an available backend"
    );

    set("scalar");
    assert_eq!(
        build_using_env().unwrap().backend(),
        Backend::Scalar,
        "HYALITE_BACKEND=scalar should build scalar"
    );

    set("auto");
    assert!(
        build_using_env().unwrap().backend().is_available(),
        "auto should resolve to an available backend"
    );

    // Forcing an unavailable backend via env is a build error. NEON is unavailable on every
    // target until M3, so it is a stable "forced-but-unavailable" case regardless of CPU.
    set("neon");
    assert_eq!(
        build_using_env().unwrap_err(),
        Error::BackendUnavailable {
            backend: Backend::Neon
        }
    );

    // An unparseable value is a distinct, clear error.
    set("definitely-not-a-backend");
    assert!(matches!(
        build_using_env().unwrap_err(),
        Error::InvalidBackendName { .. }
    ));

    // An explicit builder choice must take precedence over the env var, even an unavailable one.
    set("neon");
    assert_eq!(
        build_forcing(BackendChoice::Force(Backend::Scalar))
            .unwrap()
            .backend(),
        Backend::Scalar,
        "explicit .backend() must override HYALITE_BACKEND"
    );

    clear();
}
