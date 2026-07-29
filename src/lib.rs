//! `hyalite` — exact, SIMD-accelerated pairwise/database sequence alignment in pure Rust.
//!
//! A reimplementation and improvement of [Opal](https://github.com/Martinsos/opal)
//! (Rognes-style inter-sequence SIMD Smith-Waterman). `hyalite` is a Rust reimplementation
//! of Opal and is **not affiliated with or endorsed by** the Opal authors.
//!
//! # Determinism contract
//!
//! This is the crate's primary guarantee, not an implementation detail:
//!
//! > For identical inputs, every backend returns bit-identical results: the same score, the
//! > same database index, the same query end position, and the same target end position. The
//! > selected backend affects performance only, never results.
//!
//! Holding this is why scores use a single, once-defined signed-integer arithmetic model with
//! a documented saturation boundary (see [`ScoreWidth`]), and why tie-breaks are resolved by a
//! scalar argmax over database index rather than a lane-order-dependent horizontal max.
//!
//! The full specification every backend must implement against — the arithmetic model, the
//! score-width proof's coverage of intermediate cells, the tie-break rules, and what is and is
//! not promised — lives in `DETERMINISM.md` at the repository root.
//!
//! # Status
//!
//! Under construction (milestone M0): scalar backend, all four alignment [`Mode`]s, `Score`
//! and `ScoreEnd` [search types](SearchType). SIMD backends, traceback, and the striped
//! `align_pair` path land in later milestones — see `handover.md`.

// `deny` rather than `forbid` so the SIMD backend modules (M2b+) can locally `allow(unsafe_code)`
// for intrinsics; everything else stays unsafe-free.
#![deny(unsafe_code)]

mod backend;
mod database;
mod error;
mod hit;
mod inter;
mod kernel;
mod mode;
mod scoring;
mod search;
mod width;

pub use backend::{BACKEND_ENV_VAR, Backend, BackendChoice};
pub use database::{Database, DatabaseBuilder, Scratch};
pub use error::{Error, Result};
pub use hit::BestHit;
pub use inter::{Layout, LayoutChoice};
pub use kernel::align_pair;
pub use mode::Mode;
pub use scoring::Scoring;
pub use search::SearchType;
pub use width::ScoreWidth;
