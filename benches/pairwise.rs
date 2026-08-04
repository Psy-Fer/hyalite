//! Pairwise-path benchmarks: the striped (Farrar) SIMD `align_pair`, `align_pair_position_max`, and
//! batched `align_pairs`, with a scalar baseline for context. These guard the paths `cr4_scan.rs`
//! does not — in particular the striped kernel's `#[target_feature]` inlining, whose regression
//! would quietly slow short-pair alignments several-fold. Run with `cargo bench`.
//!
//! `Score` uses the striped SIMD kernel (SSE4.1 / NEON); `ScoreEnd` uses the scalar DP (the striped
//! path is score-only), so the two `align_pair` rows are a SIMD-vs-scalar comparison of the same DP.
//! Sizes span a short pair (i8 width) and the mate-rescue shape (150 nt read vs ~1400 nt window,
//! i16 width). `Throughput::Elements` is set to the DP cell count, so criterion reports per-cell.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hyalite::{Mode, Scoring, SearchType, align_pair, align_pair_position_max, align_pairs};

const AL: usize = 4;

fn dna_scoring() -> Scoring {
    // match +2 / mismatch -1, gap_open 2, gap_ext 1. Short pairs prove i8; ~150 nt reaches i16.
    let mut m = vec![-1i32; AL * AL];
    for i in 0..AL {
        m[i * AL + i] = 2;
    }
    Scoring::new(AL, m, 2, 1).unwrap()
}

fn next(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn random_dna(state: &mut u64, len: usize) -> Vec<u8> {
    (0..len).map(|_| (next(state) % AL as u64) as u8).collect()
}

fn bench(c: &mut Criterion) {
    let s = dna_scoring();
    let mut st = 0x9e37_79b9_7f4a_7c15u64;

    // align_pair: SIMD (Score) vs scalar (ScoreEnd) across sizes.
    for (ql, tl) in [(30usize, 30usize), (150, 150), (150, 1400)] {
        let q = random_dna(&mut st, ql);
        let t = random_dna(&mut st, tl);
        let mut g = c.benchmark_group(format!("align_pair_{ql}x{tl}"));
        g.throughput(Throughput::Elements((ql * tl) as u64));
        g.bench_function("score_simd", |b| {
            b.iter(|| {
                black_box(
                    align_pair(&q, &t, &s, Mode::Sw, SearchType::Score)
                        .unwrap()
                        .score,
                )
            })
        });
        g.bench_function("scalar_scoreend", |b| {
            b.iter(|| {
                black_box(
                    align_pair(&q, &t, &s, Mode::Sw, SearchType::ScoreEnd)
                        .unwrap()
                        .score,
                )
            })
        });
        g.finish();
    }

    // Per-position maxima at the mate-rescue shape (striped SIMD, SW-only).
    {
        let (ql, tl) = (150usize, 1400usize);
        let q = random_dna(&mut st, ql);
        let t = random_dna(&mut st, tl);
        let mut out = Vec::new();
        let mut g = c.benchmark_group("align_pair_position_max_150x1400");
        g.throughput(Throughput::Elements((ql * tl) as u64));
        g.bench_function("simd", |b| {
            b.iter(|| {
                align_pair_position_max(&q, &t, &s, &mut out).unwrap();
                black_box(out.len())
            })
        });
        g.finish();
    }

    // Batched align_pairs: many short pairs (the regime a future inter-batched kernel would target;
    // for now, the reused-scratch striped loop).
    {
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = (0..256)
            .map(|_| (random_dna(&mut st, 30), random_dna(&mut st, 30)))
            .collect();
        let mut out = Vec::new();
        let mut g = c.benchmark_group("align_pairs_256x30x30");
        g.throughput(Throughput::Elements(pairs.len() as u64));
        g.bench_function("batch", |b| {
            b.iter(|| {
                align_pairs(&pairs, &s, Mode::Sw, SearchType::Score, &mut out).unwrap();
                black_box(out.len())
            })
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
