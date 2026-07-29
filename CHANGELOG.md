# Changelog

All notable changes to `hyalite` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `Database::scan_all`: the per-target counterpart to `scan`, writing one `BestHit` per database
  sequence (in `db_index` order) into a caller-provided `Vec<BestHit>` (cleared and reused, so no
  per-call allocation). Per-target scores are SIMD-accelerated via a new `fill_scores` kernel path;
  for `SearchType::ScoreEnd`, per-sequence end positions are currently recovered by the scalar
  kernel (in-vector end tracking is a planned optimisation that will not change this API).

## [0.1.0] - 2026-07-29

The v0.1 foundation: exact SIMD-accelerated pairwise and database sequence alignment with a
bit-identical-across-backends guarantee.

### Added

- **Alignment modes**: Smith-Waterman (`SW`, local), Needleman-Wunsch (`NW`, global), semi-global
  (`HW`, query ends free), and overlap (`OV`), with affine gap penalties (Opal's
  `gap_open + (n-1)·gap_ext` convention).
- **Search types**: `Score` and `ScoreEnd` (score plus query/target end positions).
- **Scalar reference kernel** — the correctness oracle, computing in `i32`.
- **Database scan API**: an immutable, `Send + Sync` `Database` (built via a validating
  `DatabaseBuilder`) split from per-thread `Scratch`; `Database::scan` is infallible and
  allocation-free.
- **Single-pair entry point** `align_pair` (scalar-backed).
- **SIMD inter-sequence backends**: `sse4.1` (16 lanes) and `avx2` (32 lanes) on x86-64, and
  `neon` (16 lanes) on aarch64, generic over a `Lanes` trait, selected by runtime CPU detection.
- **Backend override**: the `HYALITE_BACKEND` environment variable and a builder method
  (`Backend`, `BackendChoice`), with `db.backend()` reporting the resolved backend.
- **Kernel layouts**: `Gathered` and `Precomputed` (a cache-resident score table for small fixed
  databases), auto-selected by size and overridable (`Layout`, `LayoutChoice`); reported via
  `db.layout()`.
- **Static width proof**: the score integer width (`i8`/`i16`/`i32`) is proven sufficient at
  construction — mode-aware, so free-end-gap modes are not over-provisioned — making the hot loop
  infallible. Reported via `db.score_width()`.
- **Typed construction errors** (`Error`), including validation of `gap_open ≥ gap_ext` (Opal
  issue #28) and out-of-range symbols.
- **Determinism contract** documented in `DETERMINISM.md`: bit-identical results across every
  backend and layout, with a specified arithmetic model, mode-aware width bounds, and
  lane-order-independent tie-breaks.
- **Test suite**: brute-force differential oracle, `proptest` property/differential tests across
  backends, layouts, and lane counts, intermediate-cell width validation, a stable adversarial
  robustness harness, real-sequence tests (phiX174, lambda, CellRanger4 adapters), and parameter
  sweeps.
- **Benchmark** (`benches/cr4_scan.rs`): a CellRanger4-style overlap scan across backends and
  layouts.
- **CI**: rustfmt, clippy (`-D warnings`), and the full test suite on x86-64 and aarch64.

### Notes

- SIMD currently accelerates `i8`-width databases with `alphabet_len ≤ 16`; wider inputs use the
  scalar path. A striped intra-sequence SIMD backend for `align_pair` and traceback
  (`SearchType::Alignment`) are planned for v0.2.
