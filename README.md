# hyalite

**Exact, SIMD-accelerated pairwise and database sequence alignment in pure Rust.** Smith-Waterman
(local), Needleman-Wunsch (global), semi-global, and overlap modes with affine gaps, runtime CPU
dispatch, and a **bit-identical-across-backends** guarantee.

`hyalite` is a Rust reimplementation and improvement of
[Opal](https://github.com/Martinsos/opal) (Martin Šošić), which implements Rognes's inter-sequence
SIMD parallelisation of Smith-Waterman. It is not affiliated with or endorsed by the Opal authors.

## Why

The primary differentiator is **exactness combined with runtime dispatch and reproducibility**:

| Crate | Exact | SIMD | Affine gap | All 4 modes | Pure Rust |
|---|---|---|---|---|---|
| `block-aligner` | No (bounded error) | Yes | Yes | Partial | Yes |
| `rust-bio` | Yes | No | Yes | Partial | Yes |
| `parasailors` | Yes | Yes | Yes | Yes | No (FFI) |
| **`hyalite`** | **Yes** | **Yes** | **Yes** | **Yes** | **Yes** |

Notes: `block-aligner` is exact only at its maximum block size and is otherwise an adaptive,
bounded-error aligner (global, local, and X-drop). `rust-bio`'s pairwise alignment is exact but
scalar. `parasailors` provides FFI bindings to the C `parasail` library and is minimally maintained
(last release 2016).

Runtime dispatch means the same binary runs on any x86-64 or aarch64 CPU without recompilation.
Crucially:

> For identical inputs, every backend returns bit-identical results: the same score, database
> index, query end, and target end. The selected backend affects performance only, never results.

This is the crate's central promise, specified in [`DETERMINISM.md`](DETERMINISM.md).

## Example

```rust
use hyalite::{Database, Mode, Scoring, Scratch, SearchType};

// Pre-encoded alphabet indices (A,C,G,T = 0,1,2,3). Substitution matrix is row-major
// `matrix[q * alphabet_len + t]`; gap penalties are non-negative magnitudes.
let scoring = Scoring::new(
    4,
    vec![
        2, -1, -1, -1,
        -1, 2, -1, -1,
        -1, -1, 2, -1,
        -1, -1, -1, 2,
    ],
    /* gap_open */ 2,
    /* gap_ext  */ 1,
)?;

let db = Database::builder()
    .sequences(&[vec![0u8, 1, 2, 3], vec![2u8, 2, 2, 2]])
    .scoring(scoring)
    .mode(Mode::Sw)                 // local alignment
    .search_type(SearchType::ScoreEnd)
    .max_query_len(64)
    .build()?;                      // resolves backend + layout, proves the score width

// Immutable `Database` is `Send + Sync`; each worker thread keeps its own `Scratch`.
let mut scratch = Scratch::new(&db);
let hit = db.scan(&mut scratch, &[0u8, 1, 2, 3]);
assert_eq!(hit.db_index, 1);       // the perfect-match sequence wins
# Ok::<(), hyalite::Error>(())
```

## Design

- **Inter-sequence (Rognes/SWIPE) kernel.** One query is aligned against a whole database at once,
  one sequence per SIMD lane. The immutable, `Send + Sync` `Database` (shareable behind `Arc`) is
  split from per-thread `Scratch`; `Database::scan` is infallible and allocation-free.
- **Backends.** `scalar` (the reference oracle), `sse4.1` (16 lanes), and `avx2` (32 lanes) on
  x86-64, plus `neon` (16 lanes) on aarch64, selected at runtime. Override with the
  `HYALITE_BACKEND` environment variable or the builder.
- **Layouts.** `Gathered` (general) and `Precomputed` (a cache-resident score table for small
  fixed databases such as an adapter set), auto-selected by size.
- **Static width proof.** The score integer width (`i8`, `i16`, or `i32`) is proven sufficient at
  construction rather than detected at runtime, so the hot loop is infallible. SIMD currently
  accelerates `i8`-width databases with `alphabet_len` at most 16; wider inputs use the scalar path.
- **`align_pair`.** A single-pair entry point exists from day one. It is currently scalar-backed;
  a striped intra-sequence SIMD backend is planned for v0.2.
- **Modes.** `SW` (local), `NW` (global), `HW` (semi-global, query ends free), `OV` (overlap).
- **No `unsafe` outside the SIMD backend modules**, and the core is dependency-free.

## Testing

Determinism and correctness are enforced by a differential test strategy. The scalar kernel is the
oracle, checked against an independent brute-force alignment scorer. Every SIMD backend and both
layouts are checked bit-identical to scalar with `proptest` across all modes, search types, and
lane counts. The static width proof is validated against real intermediate DP cells. Real phage
genomes and the CellRanger4 adapter set exercise realistic sequence composition, and parameter
sweeps plus regression tests derived from known Opal bugs probe the edge cases. CI runs on x86-64
and aarch64.

Run the benchmark (a CellRanger4-style overlap scan) with `cargo bench`.

## Attribution

`hyalite` is an independent Rust reimplementation of the algorithm behind
[Opal](https://github.com/Martinsos/opal) (Martin Šošić, MIT-licensed), which itself implements
Rognes's inter-sequence SIMD Smith-Waterman. It does not copy Opal's source code, and it is not
affiliated with or endorsed by the Opal authors.

- Opal: <https://github.com/Martinsos/opal> (Martin Šošić, MIT).
- Rognes, T. (2011). Faster Smith-Waterman database searches with inter-sequence SIMD
  parallelisation. *BMC Bioinformatics*, 12, 221.
  [doi:10.1186/1471-2105-12-221](https://doi.org/10.1186/1471-2105-12-221)

## License

MIT.
