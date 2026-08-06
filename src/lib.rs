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
//! Scalar and SIMD (SSE4.1/AVX2/NEON) backends across all five alignment [`Mode`]s (`SW`, `NW`,
//! `HW`, `OV`, `SHW`) at `i8`/`i16`/`i32` score width; `Score`, `ScoreEnd`, and full `Alignment`
//! [search types](SearchType); per-target [`Database::scan_all`] / [`Database::scan_scores`] and
//! [`Database::scan_aligned`]; per-sequence width escalation for mixed-length databases; and a
//! striped (Farrar) intra-sequence SIMD kernel for [`align_pair`] plus [`align_pair_span`] (the
//! aligned span without the operations) and traceback via [`align`] (full-matrix, with an automatic
//! linear-space path for large pairs). Substitution alphabets larger than 16 (e.g. proteins) run on
//! SIMD via the Precomputed layout.

// `deny` rather than `forbid` so the SIMD backend modules (M2b+) can locally `allow(unsafe_code)`
// for intrinsics; everything else stays unsafe-free.
#![deny(unsafe_code)]

mod align;
mod backend;
mod database;
mod error;
mod hit;
mod inter;
mod kernel;
mod mode;
mod scoring;
mod search;
// The striped (Farrar) single-pair kernel exists only on the SIMD architectures it targets; other
// architectures use the scalar `align_pair` path.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
mod striped;
mod width;

pub use align::{AlignOp, AlignedHit, Alignment, align};
pub use backend::{BACKEND_ENV_VAR, Backend, BackendChoice};
pub use database::{Database, DatabaseBuilder, Scratch};
pub use error::{Error, Result};
pub use hit::{BestHit, LocalSpan};
pub use inter::{Layout, LayoutChoice};
pub use kernel::{
    PairScratch, align_pair, align_pair_position_max, align_pair_position_max_with,
    align_pair_span, align_pair_with, align_pairs, score2,
};
pub use mode::Mode;
pub use scoring::Scoring;
pub use search::SearchType;
pub use width::ScoreWidth;
