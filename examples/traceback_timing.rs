//! Empirical cost of the checkpoint (linear-space) traceback vs the full-matrix path.
//!
//! Both produce a byte-identical `Alignment`; this measures what the memory saving costs in time.
//! Run with: `cargo run --release --example traceback_timing`. The largest size (100k) exercises
//! genome-scale linear-space traceback and takes ~1.5 min; the full matrix there would need >100
//! GiB and cannot be allocated, so only the checkpoint path runs.

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

/// Human-readable memory: MiB, or GiB once it passes 1024 MiB.
fn mem(bytes: u64) -> String {
    let mib = bytes as f64 / (1024.0 * 1024.0);
    if mib >= 1024.0 {
        format!("{:.1} GiB", mib / 1024.0)
    } else {
        format!("{mib:.1} MiB")
    }
}

/// Above this the full matrix is too large to allocate here, so only the checkpoint path runs.
const FULL_LIMIT: u64 = 2 * 1024 * 1024 * 1024;

fn main() {
    let scoring = Scoring::new(
        4,
        vec![2, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2, -1, -1, -1, -1, 2],
        3,
        1,
    )
    .unwrap();

    println!(
        "{:>8} {:>12} {:>12} {:>11} {:>11} {:>8}",
        "size", "full mem", "ckpt mem", "full (ms)", "ckpt (ms)", "cost"
    );
    println!(
        "  (large sizes run the checkpoint path only; the full matrix is too big to allocate)"
    );
    for &len in &[1000usize, 4000, 8000, 32000, 100_000] {
        let q = seq(len, 0x1234_5678);
        let t = seq(len, 0x9abc_def0);

        // Checkpoint path: a budget below the full footprint but above the checkpoint peak.
        let budget = (checkpoint_bytes(len, len) * 2).max(1 << 20) as usize;
        let start = Instant::now();
        let ckpt = align(&q, &t, &scoring, Mode::Nw, budget).unwrap();
        let ckpt_ms = start.elapsed().as_secs_f64() * 1e3;

        // Only run the full-matrix path when it is small enough to allocate; when it does, confirm
        // the checkpoint result is byte-identical to it.
        if full_bytes(len, len) < FULL_LIMIT {
            let start = Instant::now();
            let full = align(&q, &t, &scoring, Mode::Nw, usize::MAX).unwrap();
            let full_ms = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(full, ckpt, "checkpoint result must be byte-identical");
            println!(
                "{:>8} {:>12} {:>12} {:>11.1} {:>11.1} {:>7.2}x",
                len,
                mem(full_bytes(len, len)),
                mem(checkpoint_bytes(len, len)),
                full_ms,
                ckpt_ms,
                ckpt_ms / full_ms,
            );
        } else {
            println!(
                "{:>8} {:>12} {:>12} {:>11} {:>11.1} {:>8}",
                len,
                mem(full_bytes(len, len)),
                mem(checkpoint_bytes(len, len)),
                "infeasible",
                ckpt_ms,
                "-",
            );
        }
    }
}
