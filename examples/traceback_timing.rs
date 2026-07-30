//! Empirical cost of the checkpoint (linear-space) traceback vs the full-matrix path.
//!
//! Both produce a byte-identical `Alignment`; this measures what the memory saving costs in time.
//! Run with: `cargo run --release --example traceback_timing`.

use hyalite::{Mode, Scoring, align};
use std::time::Instant;

/// Deterministic pseudo-random sequence over a 4-symbol alphabet (no rand dependency).
fn seq(len: usize, mut state: u64) -> Vec<u8> {
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) % 4) as u8
        })
        .collect()
}

fn full_bytes(m: usize, n: usize) -> u64 {
    (m as u64 + 1) * (n as u64 + 1) * 3 * 4
}

fn checkpoint_bytes(m: usize, n: usize) -> u64 {
    let k = (m as u64).isqrt().max(1);
    let num_ckpt = m as u64 / k + 1;
    (2 * num_ckpt + 3 * (k + 1)) * (n as u64 + 1) * 4
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

fn main() {
    let scoring = Scoring::new(
        4,
        vec![2, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2],
        3,
        1,
    )
    .unwrap();

    println!(
        "{:>7} {:>12} {:>12} {:>10} {:>10} {:>7}",
        "size", "full (MiB)", "ckpt (MiB)", "full (ms)", "ckpt (ms)", "slow x"
    );
    for &len in &[1000usize, 2000, 4000, 8000] {
        let q = seq(len, 0x1234_5678);
        let t = seq(len, 0x9abc_def0);

        // Full-matrix path (unbounded budget).
        let start = Instant::now();
        let full = align(&q, &t, &scoring, Mode::Nw, usize::MAX).unwrap();
        let full_ms = start.elapsed().as_secs_f64() * 1e3;

        // Checkpoint path: a budget below the full footprint but above the checkpoint peak.
        let budget = (checkpoint_bytes(len, len) * 2).max(1 << 20) as usize;
        let start = Instant::now();
        let ckpt = align(&q, &t, &scoring, Mode::Nw, budget).unwrap();
        let ckpt_ms = start.elapsed().as_secs_f64() * 1e3;

        assert_eq!(full, ckpt, "checkpoint result must be byte-identical");

        println!(
            "{:>7} {:>12.1} {:>12.2} {:>10.1} {:>10.1} {:>6.2}x",
            len,
            mib(full_bytes(len, len)),
            mib(checkpoint_bytes(len, len)),
            full_ms,
            ckpt_ms,
            ckpt_ms / full_ms,
        );
    }
}
