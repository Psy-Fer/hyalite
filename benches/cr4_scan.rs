//! CellRanger4-style benchmark: short reads scanned against a small adapter database, across
//! backends (scalar / SSE4.1 / AVX2 / NEON) and layouts (Gathered / Precomputed). This is the
//! first consumer's workload shape (see `tests/data/PROVENANCE.md`).
//!
//! Run with `cargo bench`. Backends unavailable on the host CPU — or on a database the i8 SIMD
//! kernel does not apply to — are skipped, so the reported set depends on the machine. All
//! configurations compute identical results (see `DETERMINISM.md`); this only measures speed.
//!
//! Uses overlap (`OV`) mode, exactly as STAR's CellRanger4 clipper does. The mode-aware width
//! bound proves this workload to i8 (overlap's free end gaps cap the negative reach), so the SIMD
//! kernel applies.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hyalite::{
    Backend, BackendChoice, Database, Layout, LayoutChoice, Mode, Scoring, Scratch, SearchType,
};

/// Alphabet A, C, G, T, N.
const AL: usize = 5;
/// The 10x template-switch oligo (the real CR4 adapter).
const TSO: &str = "AAGCAGTGGTATCAACGCAGAGTACATGGG";
/// CR4 read length.
const READ_LEN: usize = 91;

/// STAR's CellRanger4 scoring: match +1, mismatch -2, any-vs-N -2, N-vs-N 0; gap_open/ext = 2.
fn cr4_scoring() -> Scoring {
    #[rustfmt::skip]
    let matrix = vec![
         1, -2, -2, -2, -2,
        -2,  1, -2, -2, -2,
        -2, -2,  1, -2, -2,
        -2, -2, -2,  1, -2,
        -2, -2, -2, -2,  0,
    ];
    Scoring::new(AL, matrix, 2, 2).unwrap()
}

fn encode(s: &str) -> Vec<u8> {
    s.bytes()
        .map(|b| match b {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            _ => 4,
        })
        .collect()
}

/// Deterministic PRNG so the benchmark corpus is reproducible run to run.
fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 33
}

fn random_dna(state: &mut u64, len: usize) -> Vec<u8> {
    (0..len).map(|_| (next(state) % 4) as u8).collect()
}

/// A `count`-sequence adapter database: the real TSO and homopolymers plus random ~30 nt fillers.
fn adapters(count: usize) -> Vec<Vec<u8>> {
    let mut v = vec![encode(TSO), vec![0; 30], vec![3; 30], vec![2; 30]];
    let mut state = 0x00C0_FFEE;
    while v.len() < count {
        let len = 28 + (next(&mut state) % 6) as usize;
        v.push(random_dna(&mut state, len));
    }
    v
}

/// Reads of `READ_LEN`, a third TSO-led, a third with a polyA tail, the rest plain — the mix the
/// CR4 clipper sees.
fn reads(count: usize) -> Vec<Vec<u8>> {
    let tso = encode(TSO);
    let mut state = 0x0000_BEEF;
    (0..count)
        .map(|i| {
            let mut r = random_dna(&mut state, READ_LEN);
            match i % 3 {
                0 => r[..tso.len()].copy_from_slice(&tso),
                1 => r[READ_LEN - 25..].fill(0),
                _ => {}
            }
            r
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let scoring = cr4_scoring();
    let db_seqs = adapters(64);
    let queries = reads(256);
    let max_query_len = queries.iter().map(Vec::len).max().unwrap();

    // (label, backend, layout). Scalar ignores layout; each available SIMD backend runs both.
    let mut configs = vec![("scalar".to_string(), Backend::Scalar, LayoutChoice::Auto)];
    for b in [Backend::Sse41, Backend::Avx2, Backend::Neon] {
        if b.is_available() {
            configs.push((
                format!("{b}/gathered"),
                b,
                LayoutChoice::Force(Layout::Gathered),
            ));
            configs.push((
                format!("{b}/precomputed"),
                b,
                LayoutChoice::Force(Layout::Precomputed),
            ));
        }
    }

    let mut group = c.benchmark_group("cr4_scan");
    group.throughput(Throughput::Elements(queries.len() as u64));
    for (label, backend, layout) in &configs {
        // Skip any config the database can't be built for (e.g. a SIMD backend on a non-i8-width
        // database). Scalar always builds.
        let Ok(db) = Database::builder()
            .sequences(&db_seqs)
            .scoring(scoring.clone())
            .mode(Mode::Ov)
            .search_type(SearchType::ScoreEnd)
            .max_query_len(max_query_len)
            .backend(BackendChoice::Force(*backend))
            .layout(*layout)
            .build()
        else {
            continue;
        };
        let mut scratch = Scratch::new(&db);
        group.bench_function(label.as_str(), |bch| {
            bch.iter(|| {
                let mut acc = 0i64;
                for r in &queries {
                    acc = acc.wrapping_add(db.scan(&mut scratch, r).score as i64);
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
