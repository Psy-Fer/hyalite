//! Parameter sweeps and edge-case stress: try to break the SIMD kernel against the scalar oracle.
//!
//! The existing property tests use tiny databases (≤ 6 sequences) that never cross the 16/32-lane
//! batch boundaries, and short sequences. These sweeps deliberately probe: databases sized around
//! every lane boundary, degenerate sequences (empty, single-symbol, homopolymer), empty queries,
//! the full alphabet the shuffle supports (16) and one past it (scalar fallback), score-width
//! boundaries, and a broad grid of scoring parameters. Every SIMD backend × layout must equal the
//! scalar oracle, which must in turn equal the best single-pair alignment.

mod common;

use common::reference_scan;
use hyalite::{
    Backend, BackendChoice, BestHit, Database, Layout, LayoutChoice, Mode, ScoreWidth, Scoring,
    Scratch, SearchType,
};
use proptest::prelude::*;

const MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];
const SIMD: [Backend; 3] = [Backend::Sse41, Backend::Avx2, Backend::Neon];

/// Deterministic PRNG (no external dep) so failures reproduce.
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn rand_seq(state: &mut u64, len: usize, al: usize) -> Vec<u8> {
    (0..len).map(|_| (next(state) % al as u64) as u8).collect()
}

fn build(
    seqs: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    st: SearchType,
    max_q: usize,
    backend: Backend,
    layout: LayoutChoice,
) -> Result<Database, hyalite::Error> {
    Database::builder()
        .sequences(seqs)
        .scoring(scoring.clone())
        .mode(mode)
        .search_type(st)
        .max_query_len(max_q.max(1))
        .backend(BackendChoice::Force(backend))
        .layout(layout)
        .build()
}

/// Core assertion: for every available SIMD backend × layout on an i8-eligible database, scanning
/// each query equals the scalar oracle; the scalar oracle equals best-of-`align_pair`.
fn check(seqs: &[Vec<u8>], scoring: &Scoring, mode: Mode, st: SearchType, queries: &[Vec<u8>]) {
    let max_q = queries.iter().map(Vec::len).max().unwrap_or(0);
    let max_t = seqs.iter().map(Vec::len).max().unwrap_or(0);
    let eligible = matches!(
        scoring.required_width(mode, max_q, max_t),
        Ok(ScoreWidth::I8)
    ) && scoring.alphabet_len() <= 16;

    let scalar_db = build(
        seqs,
        scoring,
        mode,
        st,
        max_q,
        Backend::Scalar,
        LayoutChoice::Auto,
    )
    .expect("scalar always builds");
    let mut scalar_scratch = Scratch::new(&scalar_db);

    // Build each available SIMD backend × layout once, with its own scratch — reused across every
    // query below, so a stale-buffer bug (differing query/target sizes into the same buffers) would
    // surface.
    let mut simd: Vec<(String, Database, Scratch)> = Vec::new();
    if eligible {
        for b in SIMD.into_iter().filter(|b| b.is_available()) {
            for layout in [Layout::Gathered, Layout::Precomputed] {
                let db = build(
                    seqs,
                    scoring,
                    mode,
                    st,
                    max_q,
                    b,
                    LayoutChoice::Force(layout),
                )
                .expect("eligible + available backend must build");
                let sc = Scratch::new(&db);
                simd.push((format!("{b}/{layout}"), db, sc));
            }
        }
    }

    for q in queries {
        let want = scalar_db.scan(&mut scalar_scratch, q);
        let reference: BestHit = reference_scan(seqs, scoring, mode, st, q);
        assert_eq!(
            want, reference,
            "scalar scan != best-of-align_pair; mode={mode} st={st} q={q:?}"
        );
        for (label, db, sc) in &mut simd {
            let got = db.scan(sc, q);
            assert_eq!(
                got,
                want,
                "{label} != scalar; mode={mode} st={st} q={q:?} nseqs={}",
                seqs.len()
            );
        }
    }
}

fn dna(m: i32, x: i32, go: i32, ge: i32) -> Scoring {
    Scoring::new(
        4,
        vec![m, x, x, x, x, m, x, x, x, x, m, x, x, x, x, m],
        go,
        ge,
    )
    .unwrap()
}

// ---------------------------------------------------------------------------

#[test]
fn database_sizes_around_every_lane_boundary() {
    // Short sequences (so scores stay i8), databases sized at and around 16/32/48/64 — the SSE
    // (16) and AVX2 (32) batch boundaries — plus partial final batches.
    let scoring = dna(2, -1, 2, 1);
    let mut state = 0x5EED_1234;
    let sizes = [
        1usize, 2, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 65, 96, 129,
    ];
    for &n in &sizes {
        let seqs: Vec<Vec<u8>> = (0..n)
            .map(|_| {
                let len = 3 + (next(&mut state) % 8) as usize;
                rand_seq(&mut state, len, 4)
            })
            .collect();
        let queries: Vec<Vec<u8>> = (0..6)
            .map(|_| {
                let len = 4 + (next(&mut state) % 6) as usize;
                rand_seq(&mut state, len, 4)
            })
            .collect();
        for mode in MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                check(&seqs, &scoring, mode, st, &queries);
            }
        }
    }
}

#[test]
fn degenerate_sequences_and_queries() {
    let scoring = dna(2, -1, 2, 1);
    // Databases that stress padding/length masks: all empty, mixed empty/non-empty, homopolymers,
    // one long among many short, exact single symbols. Sized to span lane boundaries too.
    let empt: Vec<u8> = vec![];
    let cases: Vec<Vec<Vec<u8>>> = vec![
        vec![empt.clone(); 20], // all empty
        (0..33)
            .map(|i| {
                if i % 2 == 0 {
                    empt.clone()
                } else {
                    vec![0, 1, 2]
                }
            })
            .collect(), // mixed
        vec![vec![0u8; 10]; 40], // homopolymer A, many
        (0..17).map(|i| vec![(i % 4) as u8]).collect(), // single-symbol seqs
        {
            let mut v = vec![vec![0u8, 1, 2, 3]; 32];
            v.push(vec![0u8; 20]); // one longer, forcing a wider padded batch
            v
        },
    ];
    // Queries including the empty query and single-symbol queries.
    let queries = vec![vec![], vec![0u8], vec![0u8, 1, 2, 3], vec![2u8, 2, 2, 2, 2]];
    for seqs in &cases {
        for mode in MODES {
            for st in [SearchType::Score, SearchType::ScoreEnd] {
                check(seqs, &scoring, mode, st, &queries);
            }
        }
    }
}

#[test]
fn scoring_parameter_grid() {
    // Sweep scoring parameters — including linear gaps (open == ext), free extension (ext = 0),
    // and asymmetric match/mismatch — over a fixed mixed database (short, includes empties).
    let mut state = 0xA11CE;
    let mut seqs: Vec<Vec<u8>> = (0..20)
        .map(|_| {
            let len = (next(&mut state) % 9) as usize;
            rand_seq(&mut state, len, 4)
        })
        .collect();
    seqs.push(vec![]); // ensure an empty is present
    let queries: Vec<Vec<u8>> = (0..8)
        .map(|_| {
            let len = (next(&mut state) % 9) as usize;
            rand_seq(&mut state, len, 4)
        })
        .collect();

    for (m, x) in [(1, -1), (2, -1), (3, -2), (1, 0)] {
        // Includes free gaps (0,0) and large gap_open where SW/OV E/F reach the i8 edge (-100,
        // -127) — the saturation/overflow class that Opal historically got wrong.
        for (go, ge) in [
            (1, 1),
            (2, 1),
            (3, 0),
            (5, 5),
            (4, 2),
            (0, 0),
            (100, 1),
            (127, 1),
        ] {
            let scoring = dna(m, x, go, ge);
            for mode in MODES {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    check(&seqs, &scoring, mode, st, &queries);
                }
            }
        }
    }
}

#[test]
fn alphabet_at_and_past_the_shuffle_limit() {
    // alphabet_len == 16 is the largest the byte-shuffle supports (SIMD); 17 must fall back to
    // scalar and still be correct.
    for al in [16usize, 17] {
        // Identity-ish matrix: match +1, mismatch -1.
        let mut matrix = vec![-1i32; al * al];
        for i in 0..al {
            matrix[i * al + i] = 1;
        }
        let scoring = Scoring::new(al, matrix, 2, 1).unwrap();
        let mut state = 0xF00D + al as u64;
        // Sequences using the full alphabet, sized to cross a lane boundary.
        let seqs: Vec<Vec<u8>> = (0..40)
            .map(|_| {
                let len = 4 + (next(&mut state) % 5) as usize;
                rand_seq(&mut state, len, al)
            })
            .collect();
        let queries: Vec<Vec<u8>> = (0..6).map(|_| rand_seq(&mut state, 6, al)).collect();
        for mode in MODES {
            check(&seqs, &scoring, mode, SearchType::ScoreEnd, &queries);
        }
    }
}

#[test]
fn score_width_boundaries() {
    // Databases whose best score lands exactly at the i8/i16 boundary. A query of L exact matches
    // scores L (match +1). L = 127 stays i8 (SIMD); L = 128 needs i16 (scalar). Both must be
    // correct, and the SIMD case must not saturate at the very edge of the i8 range.
    let scoring = Scoring::new(2, vec![1, -1, -1, 1], 2, 1).unwrap();
    for len in [126usize, 127, 128, 129] {
        let seq = vec![0u8; len];
        let query = vec![0u8; len];
        // SW: score is exactly `len` (a perfect self-overlap), width i8 iff len <= 127.
        let want_i8 = len <= 127;
        assert_eq!(
            matches!(
                scoring.required_width(Mode::Sw, len, len),
                Ok(ScoreWidth::I8)
            ),
            want_i8,
            "len {len} i8-eligibility"
        );
        check(&[seq], &scoring, Mode::Sw, SearchType::ScoreEnd, &[query]);
    }
}

#[test]
fn overlap_negative_reach_at_the_i8_edge_does_not_saturate() {
    // Directly stress the mode-aware i8 boundary at the deepest negative reach. In overlap mode an
    // all-mismatch alignment drives interior H cells to ~ -min(m,n)*|min_entry| and E/F one
    // gap_open lower — the exact quantity the OV bound accounts for. At the edge a real cell sits
    // at ~ -127 and must NOT collide with the i8 sentinel (-128); the scalar oracle (i32) never
    // saturates, so any SIMD divergence would be caught.
    let s = Scoring::new(2, vec![1, -1, -1, 1], 1, 1).unwrap();
    assert_eq!(
        s.required_width(Mode::Ov, 126, 126).unwrap(),
        ScoreWidth::I8
    );
    assert_eq!(
        s.required_width(Mode::Ov, 127, 127).unwrap(),
        ScoreWidth::I16
    );
    check(
        &[vec![1u8; 126]],
        &s,
        Mode::Ov,
        SearchType::ScoreEnd,
        &[vec![0u8; 126]],
    );

    // A long mostly-mismatched read with a short matching tail against a short adapter — the CR4
    // shape at the negative edge.
    let mut q = vec![0u8; 120];
    q.extend([1u8; 6]);
    check(&[vec![1u8; 6]], &s, Mode::Ov, SearchType::ScoreEnd, &[q]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Random i8-friendly databases of 1..=70 short sequences (crossing the 16/32/64-lane
    /// boundaries at random) with random short queries: every SIMD backend × layout equals scalar.
    #[test]
    fn random_lane_crossing_databases(
        seqs in prop::collection::vec(prop::collection::vec(0u8..4, 0..=6), 1..=70),
        queries in prop::collection::vec(prop::collection::vec(0u8..4, 0..=6), 1..=4),
        m in 1i32..=2,
        gap in prop::sample::select(&[(1i32, 1i32), (2, 1), (3, 0)][..]),
    ) {
        let scoring = dna(m, -1, gap.0, gap.1);
        for mode in MODES {
            check(&seqs, &scoring, mode, SearchType::ScoreEnd, &queries);
        }
    }
}
