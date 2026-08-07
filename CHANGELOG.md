# Changelog

All notable changes to `hyalite` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-08-07

### Added

- `align_pair_span`: the local (`SW`) alignment **span** — score plus the half-open aligned region
  in *both* sequences (`query_start`/`query_end`, `target_start`/`target_end`) — as a new
  [`LocalSpan`] result, without the alignment operations. It recovers both starts *and* both ends in
  a single forward Smith-Waterman pass with start tracking, using `O(target)` working memory: no
  traceback matrix, linear-space checkpointing, or CIGAR. This is the cheap way to get the aligned
  region's coordinates (e.g. bwa-style mate rescue) when the operations are not needed. The reported
  span is always self-consistent — a global (`NW`) alignment of the two aligned substrings scores
  exactly `score` — and matches the full [`align`] traceback's span (verified exhaustively for short
  pairs and by property tests).

- SIMD acceleration for **large alphabets** (`alphabet_len > 16`, e.g. proteins): the byte-shuffle
  Gathered gather can only index a 16-entry table, but the Precomputed layout looks scores up with a
  direct load and so serves any alphabet at any width. A database whose alphabet exceeds 16 (which
  previously fell back to the scalar path) now runs on the SIMD kernel via the Precomputed layout —
  the same layout `i16`/`i32` already used — as long as its score table fits the cache budget.
  Bit-identical to the scalar oracle across every backend, mode, and search type. (Databases whose
  Precomputed table would exceed the budget still fall back to scalar; a large-alphabet Gathered path
  for those is future work.)

- Per-sequence score-width escalation for database scans: a `Score`/`ScoreEnd` `Database` now proves
  each target's integer width from *its own* length and partitions the sequences into width groups,
  so a mixed-length database runs its short sequences at a narrow width (more SIMD lanes) instead of
  forcing the whole database to the single widest width. The partition is **static** (length-based),
  hence lane-count-independent, and every sufficient width yields the identical score — so a scan is
  bit-identical to the uniform-width and scalar paths (`DETERMINISM.md` §4). This is also an
  *alternative* SIMD-eligibility path: a mixed database whose single widest-width packing is too
  large for the kernel layout can still be SIMD-accelerated when each narrower group's packing fits.
  `scan`, `scan_all`, and `scan_scores` all use it; `Database::score_width()` reports the widest
  group; a uniform-length database is a single group (exactly the previous behaviour). Measured on a
  1000-short + 8-long (`i32`) `SW` `scan_scores` — a database only SIMD-eligible because of grouping —
  ~8× (SSE4.1) / ~14× (AVX2) over scalar.

- `i32`-width database scan on SIMD: the inter-sequence kernel now runs at `i32` (SSE4.1 → 4 lanes,
  AVX2 → 8 lanes, NEON → 4 lanes) via the Precomputed layout, so databases whose score proof
  exceeds `i16` — long reads, genome/contig-scale global or overlap alignment, and heavily scaled
  or log-odds matrices — are SIMD-accelerated instead of falling back to scalar. At `i32` the kernel
  is arithmetically identical to the scalar oracle (same `i32::MIN/4` `-∞` sentinel, same plain
  non-saturating add/sub, since the width proof already bounds every cell inside `i32`), so it is
  bit-identical by construction across every backend and mode. `Database`/`PackedDb` carry the packed
  database at whichever width the proof selected (`i8`, `i16`, or `i32`), with the lane count width-aware
  (register bytes ÷ element bytes). This covers the `Score` search type (single-best `scan`,
  per-target `scan_all`, and `scan_scores`). Measured over scalar for a 64-target × 2000-read `i32`
  `SW` `Score` scan: ~7× (SSE4.1) / ~12× (AVX2).
- In-vector `ScoreEnd` for `i32` `scan_all`: the SIMD backends track per-target end positions in-vector
  at `i32` width too, so `SearchType::ScoreEnd` scans over an `i32` database return scores *and*
  query/target ends at SIMD speed instead of recovering them per sequence with the scalar kernel.
  Positions fit `i16` (inside the `i32` score width), so they are carried in the same lanes as the
  scores and narrowed to `i16` on store — no separate position register split. Bit-identical to the
  scalar oracle, including the lane-order-independent end tie-break. Measured over scalar for a
  64-target × 2000-read `i32` `HW` `ScoreEnd` `scan_all`: ~5.5× (SSE4.1) / ~10× (AVX2).
- Striped (Farrar) SIMD `align_pair` at `i32` width: a `Score` alignment (and the SW per-position
  maxima behind `align_pair_position_max` / bwa `score2`) whose score proof selects `i32` now runs
  the intra-sequence striped kernel on SSE4.1 (4 lanes) / NEON (4 lanes), so long or large-magnitude
  pairs that exceed `i16` are SIMD too rather than scalar. `i32` uses plain (non-saturating)
  arithmetic with an `i32::MIN/4` `-∞` sentinel — the scalar oracle's own model, since x86 has no
  saturating 32-bit add/sub and the width proof already bounds every cell — so it is bit-identical to
  the scalar oracle at every mode and lane count. Measured over scalar for a 2000×2000 `i32` `SW`
  pair: ~4.5×. (`ScoreEnd` / `Alignment` still use the scalar path.)

### Fixed

- The `i32` score path is non-saturating (x86 has no saturating 32-bit add/subtract), so its usable
  score magnitude is bounded by the `-∞` sentinel it reserves (`i32::MIN / 4`) — `|i32::MIN / 4| - 1`,
  not `i32::MAX`. Inputs whose proven magnitude could reach the sentinel — only pathological gap
  penalties or matrix entries near `2³¹` — now fail construction with `ScoreRangeTooWide` instead of
  producing a silently-wrong score (a latent hole in the scalar oracle itself, present since v0.1),
  and two related `i32` wrap paths were closed: a pre-`max` intermediate (`cell ± penalty`) and the
  striped lazy-F gap carry. No realistic input is affected.

## [0.2.0] - 2026-08-04

### Added

- `Database::scan_scores`: write each sequence's best **score** (in `db_index` order) into a
  caller-provided `Vec<i32>` (cleared and reused) — the per-sequence array the single-best `scan`
  computes and discards, returned in full without a `BestHit` per sequence. For demultiplexers,
  tie/margin handling, and any all-scores use. Bit-identical across backends; allocation-free apart
  from growing the output.
- `align_pairs`: align a batch of independent `(query, target)` pairs, writing one `BestHit` per
  pair (in order, `db_index = pair index`) into a caller-provided `Vec` (cleared and reused). Reuses
  one internal `PairScratch` across the batch, so beyond growing the output it allocates nothing per
  pair; each result equals `align_pair` for that pair. Distinct from `Database::scan` (one query
  against many targets) and `align_pair` (a single pair).
- Reusable pairwise scratch: `PairScratch` plus `align_pair_with` / `align_pair_position_max_with`
  run the same alignments as `align_pair` / `align_pair_position_max` but reuse caller-owned working
  memory, so a hot loop of pair alignments allocates nothing per call (mirroring the `Database` +
  `Scratch` split for the DB-scan path).
- Per-position maxima and bwa-compatible `score2`: `align_pair_position_max(query, target, scoring,
  &mut Vec<i32>)` runs a local (SW) alignment and fills the caller's buffer with the best score
  **ending at each target position** (`out[t] = max_i H[i][t]`), returning the best score and its
  target end. `score2(colmax, best_score, best_target_end, matrix_max, min_score)` applies bwa-mem's
  `ksw.c` recipe — exclusion window `w = ceil(best_score / matrix_max)` around the best end, plus
  contiguous-peak collapse and a minimum-score threshold — to yield the competing second-best peak
  `(score, target_pos)`. This is the primitive a bwa-mem-style mate-rescue / mapping-quality
  consumer needs. The maxima array is a pure per-column maximum: it carries no tie-break and is
  bit-identical across every backend. It is filled by the striped (Farrar) SIMD kernel on SSE4.1
  (x86-64) or NEON (aarch64) at whichever of `i8`/`i16` the width proof selects, and by the scalar
  DP otherwise.

- `i16`-width database scan on SIMD: the inter-sequence kernel now runs at `i16` (SSE4.1 → 8 lanes,
  AVX2 → 16 lanes, NEON → 8 lanes) via the Precomputed layout, so databases whose score proof
  exceeds `i8` — long reads, large-magnitude matrices, and any alphabet larger than 16 (which the
  byte-shuffle Gathered gather cannot serve) — are SIMD-accelerated instead of falling back to
  scalar. `PackedDb`/`Database` carry the packed database at whichever width the proof selected
  (`i8` or `i16`), and the lane count is width-aware (register bytes ÷ element bytes). Both search
  types run in-vector, including per-target `ScoreEnd` end positions, with the same
  lane-order-independent tie-break. Bit-identical to the scalar oracle across every backend, mode,
  and search type. Measured over scalar: a 64-target × 2000-read `i16` `SW` `Score` scan runs ~15×
  (SSE4.1) / ~26× (AVX2), and a per-target `i16` `HW` `ScoreEnd` `scan_all` ~13× / ~22×.

- `Database::scan_all`: the per-target counterpart to `scan`, writing one `BestHit` per database
  sequence (in `db_index` order) into a caller-provided `Vec<BestHit>` (cleared and reused, so no
  per-call allocation). Per-target scores are SIMD-accelerated via a new `fill_scores` kernel path.
- In-vector `ScoreEnd` for `scan_all`: SIMD backends (SSE4.1, AVX2, NEON) now track per-target end
  positions in-vector, so `SearchType::ScoreEnd` scans return scores *and* query/target ends at
  SIMD speed. Positions are carried in a parallel `i16` domain alongside the `i8` scores; inputs
  whose lengths exceed the `i16` position range fall back to the scalar per-sequence recovery.
  Bit-identical to the scalar oracle on every backend.
- Traceback: `align()` returns a full `Alignment` (score, half-open query/target spans, and a
  `Vec<AlignOp>` of `Match`/`Mismatch`/`Ins`/`Del`), with `.cigar()` (M-collapsed) and
  `.cigar_extended()` (`=`/`X`) formatters. Scalar Gotoh affine DP with a documented canonical
  backward walk, yielding the optimal alignment in every mode.
- Linear-space traceback: when the full `H`/`E`/`F` matrices exceed the `max_bytes` budget,
  `align()` transparently switches to a **checkpoint** path that bounds memory to `O(n·√m)` by
  storing every `√m`-th DP row and recomputing row-strips on demand. It shares the walk logic and
  recomputes the full-matrix cells bit-for-bit, so the result is **byte-identical** regardless of
  budget. Measured cost: ~1.05–1.1× time for ~50× less memory at length 8000; a 100k×100k global
  alignment needs ~604 MiB via the checkpoint path where the full matrix would need ~112 GiB.
  Over-tight budgets that even the checkpoint path cannot meet return `TracebackBudgetExceeded`. The
  `max_bytes` budget is an exact upper bound on the peak allocation, and a `Database` built for
  `SearchType::Alignment` validates it once against every sub-problem it can pose (including empty
  queries and empty target sequences), so `scan_aligned` / `scan_all_aligned` are infallible.
- `Mode::Shw`: a fifth alignment mode (Opal issue #29, never implemented upstream) — the transpose
  of `HW`, aligning the whole **target** within a free window of the **query** (query end-gaps free,
  target aligned end to end; answer over the last column). Supported on every path (scalar, SIMD
  database scan, striped `align_pair`, and traceback), bit-identical across backends.
- Striped (Farrar) SIMD for `align_pair`: a `Score` alignment in any mode now runs an intra-sequence
  striped kernel on SSE4.1 (x86-64) or NEON (aarch64), at whichever of **`i8` (16 lanes) or `i16`
  (8 lanes)** the width proof selects — so 150 nt+ reads whose scores exceed `i8` are SIMD too, not
  scalar. Several× faster than the scalar path for a 2000x2000 pair (`i16`: SW 12x, OV 8x, HW 7x,
  SHW 6x, NW 5x). Bit-identical to the scalar oracle at every width, mode, and lane count.
  `ScoreEnd`/`Alignment` and `i32` width keep the scalar path.
- `SearchType::Alignment { max_bytes }` and the database traceback API: `Database::scan_aligned`
  returns the full `Alignment` of the single best hit (found by the fast score pass, then traced
  back once) as an `AlignedHit { db_index, alignment }`; `Database::scan_all_aligned` writes one
  `Alignment` per sequence into a caller `Vec`. The `max_bytes` budget is proven sufficient for the
  database's declared maximum problem size at construction, so both scans are infallible. Results
  are bit-identical to the per-target `align()` across every backend and mode.

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
