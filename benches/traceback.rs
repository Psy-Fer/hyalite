//! Traceback benchmarks: the full-matrix path vs the linear-space **checkpoint** path, at the same
//! problem so the ~2x checkpoint time (for ~O(√m) less memory) is visible and regression-guarded.
//! `cr4_scan.rs` covers the score scan but nothing exercises `align()`. Run with `cargo bench`.
//!
//! The checkpoint path is forced by setting `max_bytes` one byte below the full-matrix footprint
//! (`3 * (m+1) * (n+1) * 4`), which still comfortably fits the `O(n·√m)` checkpoint peak.

use criterion::{Criterion, Throughput, black_box, criterion_group, criterion_main};
use hyalite::{Mode, Scoring, align};

const AL: usize = 4;

fn dna_scoring() -> Scoring {
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
    let mut st = 0x1234_5678_9abc_def0u64;

    for len in [200usize, 1000] {
        let q = random_dna(&mut st, len);
        let t = random_dna(&mut st, len);
        // One byte below the full-matrix footprint forces the checkpoint path.
        let full_bytes = 3 * (len as u64 + 1).pow(2) * 4;
        let ckpt_budget = (full_bytes - 1) as usize;

        let mut g = c.benchmark_group(format!("traceback_{len}x{len}"));
        g.throughput(Throughput::Elements((len * len) as u64));
        g.bench_function("full", |b| {
            b.iter(|| black_box(align(&q, &t, &s, Mode::Nw, usize::MAX).unwrap().score))
        });
        g.bench_function("checkpoint", |b| {
            b.iter(|| black_box(align(&q, &t, &s, Mode::Nw, ckpt_budget).unwrap().score))
        });
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
