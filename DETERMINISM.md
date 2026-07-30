# The determinism contract

Determinism *is* the product. `hyalite` uses runtime CPU dispatch, so different machines run
different kernels — and the crate's central promise is that this never changes results, only
speed. This document is the single specification every backend must implement against. It is the
authority the code points to; if code and this document disagree, that is a bug in one of them.

Status: as of M1 only the scalar backend exists, so the cross-backend guarantee is not yet
exercisable. This document is written *before* the SIMD backends (M2) precisely so that they are
implemented against a fixed spec rather than retrofitted to one.

## The promise

> For identical inputs, every backend returns bit-identical results: the same score, the same
> database index, the same query end position, and the same target end position. The selected
> backend affects performance only, never results.

## Scope and preconditions

"Identical inputs" means all of: the encoded query and target sequences (pre-encoded alphabet
indices, not ASCII), `alphabet_len`, the substitution matrix, `gap_open`, `gap_ext`, the
[`Mode`], and the [`SearchType`]. The caller owns encoding; in particular **`N` is not
special-cased** by the library — the substitution matrix defines every symbol's scores, including
`N`, and two runs are only "identical inputs" if they use the same matrix.

The guarantee covers the observable fields of the **current** search types (`Score` and
`ScoreEnd`): `score`, `db_index`, `query_end`, `target_end`. It presumes the score-width proof
succeeded; if scores could overflow `i32` the builder fails loudly with `ScoreRangeTooWide` — a
build error, never a silent divergence.

Out of scope (v0.2): `SearchType::Alignment` will add a start position and a traceback path, which
have their own tie-break not yet specified here. This contract will be extended when that lands.

## 1. One arithmetic model, defined once

Every backend computes scores as **signed integers of a single width `W`**, where
`W ∈ {i8, i16, i32}` is chosen once per `Database`/`align_pair` by the width proof (§2). The model
is:

- **Saturating arithmetic at width `W`.** Adds and subtracts saturate at `W::MIN` / `W::MAX`
  rather than wrapping.
- **The sentinel `W::MIN` means −∞** (an unreachable cell — e.g. a gap that cannot have opened
  yet at a border). Real scores are provably confined to `[W::MIN + 1, W::MAX]` (§2), so a real
  score can never collide with the sentinel.
- **The sentinel never wins.** A max over cell candidates that includes −∞ (or any value that
  saturated toward `W::MIN`) selects a real value whenever one exists, because every real value is
  `> W::MIN`. Consequently the *result* depends only on real cell values — and those are exactly
  the values the proof guarantees no backend saturates.

The **scalar reference kernel realizes this same model** using `i32` with headroom instead of
saturation: unreachable cells use `NEG = i32::MIN / 4` as −∞ (chosen so repeated gap subtractions
cannot underflow), and the width proof guarantees no real cell reaches even `i32` saturation. A
narrow saturating backend and the wide scalar oracle therefore compute the *same maxima over the
same real values*, which is what makes them bit-identical.

The gap-penalty convention is fixed: a gap of length `n` costs `gap_open + (n - 1) * gap_ext`
(Opal's convention). `gap_open >= gap_ext >= 0` is enforced at construction (Opal issue #28), so
the kernel may assume it.

## 2. The score-width proof bounds *every cell*, not just the final score

`required_width` picks the narrowest `W` whose range provably holds all scores. The subtle,
load-bearing point for cross-backend determinism: the proof must bound **every intermediate
`H`/`E`/`F` cell**, not only the reported final score. A mid-path `E` gap-accumulator that
saturated at a narrow width while the scalar oracle kept the true value would diverge.

It does bound them, and here is why: each cell `H[i][j]`, `E[i][j]`, `F[i][j]` is itself the score
of an optimal *partial* alignment (ending, for `E`/`F`, in a gap), so it is subject to the same
`magnitude_bound` used for the final score:

- **Positive reach** (all modes): at most `min(m, n)` aligned pairs, each `≤ max(0, max_entry)`;
  gaps only subtract. So every cell `≤ min(m, n) · max(0, max_entry)`.
- **Negative reach is mode-specific.** A *free* end gap lets any path restart at a `0` border,
  which caps how negative a cell can get — so the bound must not assume a full-span charged gap for
  modes that don't have one:
  - `SW` (local, cells clamped at 0): mismatches vanish; a gap opening from a `≥ 0` cell reaches
    `-gap_open`. → `gap_open`.
  - `OV` (both ends free): every cell is reachable by a pure diagonal from a `0` border in `≤
    min(m, n)` steps, so `|H| ≤ min(m, n) · |min_entry|`; `E`/`F` add one `gap_open`.
    → `min(m, n) · max(0, −min_entry) + gap_open`.
  - `NW` / `HW` / `SHW` (a penalised border): a path can accumulate a full mismatch run *and* a
    full-span gap → `(m + n) · max(0, −min_entry) + gap_open + (m + n − 1) · gap_ext` (over-counts;
    safe). `SHW` is the transpose of `HW`, with the same bound.

Getting these bounds *tight* matters: too loose and a workload over-provisions to a wider integer
and loses SIMD (e.g. the CR4 overlap scan would fall back to scalar under a global-style bound);
too tight and a real cell saturates and diverges. The `intermediate_cells_fit_the_proven_width`
test enforces the second direction — **any future change to a bound must keep it green.**

Therefore no *real* cell reaches `W::MIN`; only sentinels sit at or below it. This is verified by
two tests: `width_proof_contains_actual_score` (the final score fits `W`) and
`intermediate_cells_fit_the_proven_width` (**every** real `H`/`E`/`F` cell fits `W`). The bound is
deliberately conservative — it may pick a wider `W` than strictly necessary. Over-provisioning
costs a little performance and never costs correctness, but **any future tightening of the bound
must keep the intermediate-cell test green**, or a narrow backend will silently diverge.

## 3. Tie-breaks are lane-order-independent — and this is a requirement, not just behavior

Ties are common on real data (near-identical adapter variants, homopolymers), and *which*
candidate wins is an observable the caller acts on. Both tie-breaks below are resolved by explicit
scalar comparison so they do not depend on lane order. **A SIMD backend must reproduce these exact
comparisons; a natural "first/last lane wins" reduction is a determinism bug.**

- **End position, within one alignment.** Among cells achieving the best score, report the one with
  the **smallest target end, then the smallest query end**. (For `Score` search the ends are
  `None`, so this does not apply; the score is tie-break-independent anyway.)
- **Database index, across sequences.** Among sequences achieving the best score, report the
  **smallest `db_index`**. The reduction must first materialize a per-sequence best score, then
  take a **scalar argmax** over the index — never a horizontal lane-wise max, which a 16-lane and a
  32-lane backend would resolve differently on the same input.

## 4. Uniform score width across a database — no bucketing

A `Database` resolves **one** width for the whole database (from the longest sequence and declared
max query length) and uses it for **every** sequence. This is a deliberate determinism decision,
not a simplification:

Per-sequence precision *escalation by buckets* (Opal's `OVERFLOW_BUCKETS`) makes which sequence is
computed at which precision a function of the **lane count**, so it makes performance ISA-dependent
and widens the surface on which a detection bug becomes an ISA-dependent wrong answer. `hyalite`
does not do it. If protein-scale escalation (`i8 → i16 → i32`) is added later, it must recompute
*individual overflowed sequences* at higher precision in a lane-count-independent way, preserving
this contract.

## 5. "Bit-identical across backends" is relative to the scalar oracle — not to Opal

The guarantee is that **every `hyalite` backend agrees with the `hyalite` scalar oracle**. It is
*not* a claim that `hyalite` matches Opal or STAR byte-for-byte. The `HW`/`OV` end-gap semantics
are standard textbook definitions chosen for a well-defined oracle; exact Opal/STAR parity is a
separate concern verified against reference vectors during the rustar integration (M5). Do not read
the determinism promise as a correctness proof against external tools.

## 6. Entry points may resolve different widths — results are still identical

`align_pair` (single pair) proves a width from the exact input lengths; `Database::scan`
(inter-sequence) proves one from the declared maximum lengths. The same pair can therefore be
computed at different widths through different entry points. This does not violate the contract:
any width the proof accepts is *sufficient* (no real cell saturates), so all of them yield the same
score and ends.

The **kernel [`Layout`]** (`Gathered` vs `Precomputed`) is the same kind of choice: `Precomputed`
bakes `score(q, target)` into a table while `Gathered` shuffles a per-letter profile, but both
produce the identical substitution value per cell, so the layout — auto-selected by database size
or forced via the builder — changes only speed. **Width, backend, and layout are performance
details, never result details.**

## How this is enforced

- **Differential vs an independent oracle:** proptest compares the kernel against an exhaustive
  brute-force scorer over all short inputs (`tests/properties.rs`, `tests/alignment.rs`).
- **Cross-backend agreement hook:** `assert_all_backends_agree` (`tests/properties.rs`) is a single
  call today; every SIMD backend drops into it at M2 and every property then covers them.
- **Width proof, final and intermediate:** `width_proof_contains_actual_score` and
  `intermediate_cells_fit_the_proven_width`.
- **Tie-breaks and the cross-sequence reduction:** covered by the differential `scan`-vs-reference
  tests and dedicated tie-break cases.
- **Backend override for CI:** `HYALITE_BACKEND` and the builder `.backend()` let CI force each
  backend in turn; the CI matrix expands to `sse4.1`/`avx2`/`neon` at M2.
- **Real-data composition:** phiX/lambda/CR4 tests (`tests/real_data.rs`) exercise the invariants on
  homopolymers and real adapter overlaps, not just random bytes.
