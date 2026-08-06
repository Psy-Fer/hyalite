//! Shared test helpers: an independent brute-force alignment oracle and small generators.
//!
//! This is compiled into each integration-test binary that declares `mod common;`. The
//! brute-force scorers here are deliberately structured differently from the library's Gotoh DP
//! (exhaustive path / substring enumeration) so agreement between them is meaningful.

#![allow(dead_code)] // Not every test binary uses every helper.

use hyalite::{BestHit, Mode, Scoring, SearchType, align_pair};

pub const ALL_MODES: [Mode; 5] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov, Mode::Shw];

/// A match/mismatch substitution matrix over an `al`-symbol alphabet.
pub fn identity_matrix(al: usize, m: i32, x: i32) -> Vec<i32> {
    let mut v = vec![x; al * al];
    for i in 0..al {
        v[i * al + i] = m;
    }
    v
}

/// The standard DNA test scoring: match +2, mismatch -1, gap_open 2, gap_ext 1 over {A,C,G,T}.
pub fn dna() -> Scoring {
    Scoring::new(4, identity_matrix(4, 2, -1), 2, 1).unwrap()
}

/// Every sequence over `alphabet` symbols of length `0..=max_len`.
pub fn all_sequences(alphabet: u8, max_len: usize) -> Vec<Vec<u8>> {
    let mut out = vec![vec![]];
    let mut frontier = vec![vec![]];
    for _ in 0..max_len {
        let mut next = Vec::new();
        for seq in &frontier {
            for sym in 0..alphabet {
                let mut s = seq.clone();
                s.push(sym);
                next.push(s);
            }
        }
        out.extend(next.iter().cloned());
        frontier = next;
    }
    out
}

#[derive(Clone, Copy, PartialEq)]
enum Move {
    Start,
    Right, // consume a target base as a gap in the query
    Down,  // consume a query base as a gap in the target
}

/// Fixed parameters threaded through the brute-force recursion.
pub struct Prob<'a> {
    pub q: &'a [u8],
    pub t: &'a [u8],
    pub mat: &'a [i32],
    pub al: usize,
    pub go: i32,
    pub ge: i32,
}

/// Exact global (NW) score by enumerating every alignment path, charging affine gap penalties
/// from maximal same-direction runs. Exponential; only for tiny slices.
pub fn brute_nw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    fn rec(p: &Prob, i: usize, j: usize, last: Move) -> i32 {
        let (m, n) = (p.q.len(), p.t.len());
        if i == m && j == n {
            return 0;
        }
        let mut best = i32::MIN;
        if i < m && j < n {
            let s = p.mat[p.q[i] as usize * p.al + p.t[j] as usize];
            best = best.max(s.saturating_add(rec(p, i + 1, j + 1, Move::Start)));
        }
        if j < n {
            let cost = if last == Move::Right { p.ge } else { p.go };
            best = best.max((-cost).saturating_add(rec(p, i, j + 1, Move::Right)));
        }
        if i < m {
            let cost = if last == Move::Down { p.ge } else { p.go };
            best = best.max((-cost).saturating_add(rec(p, i + 1, j, Move::Down)));
        }
        best
    }
    rec(
        &Prob {
            q,
            t,
            mat,
            al,
            go,
            ge,
        },
        0,
        0,
        Move::Start,
    )
}

/// Local (SW): best global score over every substring pair, floored at 0.
pub fn brute_sw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    best = best.max(brute_nw(&q[a..b], &t[c..d], mat, al, go, ge));
                }
            }
        }
    }
    best
}

/// Semi-global (HW): query fully aligned to the best target window.
pub fn brute_hw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let n = t.len();
    let mut best = i32::MIN;
    for c in 0..=n {
        for d in c..=n {
            best = best.max(brute_nw(q, &t[c..d], mat, al, go, ge));
        }
    }
    best
}

/// Semi-global transpose (SHW): the whole **target** is aligned to the best **query** window (the
/// mirror image of [`brute_hw`], which windows the target).
pub fn brute_shw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let m = q.len();
    let mut best = i32::MIN;
    for a in 0..=m {
        for b in a..=m {
            best = best.max(brute_nw(&q[a..b], t, mat, al, go, ge));
        }
    }
    best
}

/// Overlap (OV): best global score over substring pairs whose alignment touches a border at both
/// ends, floored at 0.
pub fn brute_ov(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    if (a == 0 || c == 0) && (b == m || d == n) {
                        best = best.max(brute_nw(&q[a..b], &t[c..d], mat, al, go, ge));
                    }
                }
            }
        }
    }
    best
}

/// Brute-force score for any mode.
pub fn brute(mode: Mode, q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    match mode {
        Mode::Nw => brute_nw(q, t, mat, al, go, ge),
        Mode::Sw => brute_sw(q, t, mat, al, go, ge),
        Mode::Hw => brute_hw(q, t, mat, al, go, ge),
        Mode::Ov => brute_ov(q, t, mat, al, go, ge),
        Mode::Shw => brute_shw(q, t, mat, al, go, ge),
        _ => unreachable!("ALL_MODES covers every mode"),
    }
}

/// Encode an ASCII DNA string to alphabet indices `A,C,G,T,N -> 0,1,2,3,4`, skipping whitespace.
/// Panics on any other character.
pub fn encode_dna(seq: &str) -> Vec<u8> {
    seq.bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .map(|b| match b.to_ascii_uppercase() {
            b'A' => 0,
            b'C' => 1,
            b'G' => 2,
            b'T' => 3,
            b'N' => 4,
            other => panic!("unexpected base {:?} in sequence", other as char),
        })
        .collect()
}

/// Minimal FASTA parser: returns `(id, encoded_sequence)` per record. `id` is the first
/// whitespace-delimited token of the header.
pub fn parse_fasta(text: &str) -> Vec<(String, Vec<u8>)> {
    let mut records = Vec::new();
    let mut id: Option<String> = None;
    let mut seq = String::new();
    for line in text.lines() {
        if let Some(header) = line.strip_prefix('>') {
            if let Some(prev) = id.take() {
                records.push((prev, encode_dna(&seq)));
                seq.clear();
            }
            id = Some(header.split_whitespace().next().unwrap_or("").to_string());
        } else {
            seq.push_str(line.trim());
        }
    }
    if let Some(prev) = id.take() {
        records.push((prev, encode_dna(&seq)));
    }
    records
}

/// STAR CellRanger4 scoring, transcribed from STAR source: alphabet `A,C,G,T,N`; match +1,
/// mismatch -2, any-vs-N -2, N-vs-N 0; `gap_open = 2`, `gap_ext = 2`. See `tests/data/PROVENANCE.md`.
pub fn cr4_scoring() -> Scoring {
    #[rustfmt::skip]
    let matrix = vec![
         1, -2, -2, -2, -2,
        -2,  1, -2, -2, -2,
        -2, -2,  1, -2, -2,
        -2, -2, -2,  1, -2,
        -2, -2, -2, -2,  0,
    ];
    Scoring::new(5, matrix, 2, 2).unwrap()
}

/// The expected database-scan result: best `align_pair` over `seqs`, smallest index on ties.
pub fn reference_scan(
    seqs: &[Vec<u8>],
    scoring: &Scoring,
    mode: Mode,
    st: SearchType,
    query: &[u8],
) -> BestHit {
    let mut best: Option<BestHit> = None;
    for (index, seq) in seqs.iter().enumerate() {
        let pair = align_pair(query, seq, scoring, mode, st).unwrap();
        let candidate = BestHit {
            db_index: index,
            ..pair
        };
        if best.is_none_or(|b| candidate.score > b.score) {
            best = Some(candidate);
        }
    }
    best.expect("database is non-empty")
}

// ---------------------------------------------------------------------------
// Protein / BLOSUM62 (large-alphabet, 20-symbol) helpers
// ---------------------------------------------------------------------------

/// Amino-acid order for the embedded [`BLOSUM62`] matrix and [`encode_protein`].
pub const AMINO_ACIDS: &[u8; 20] = b"ARNDCQEGHILKMFPSTWYV";

/// The standard NCBI BLOSUM62 substitution matrix, row-major in [`AMINO_ACIDS`] order. Public
/// domain (a published log-odds matrix); see `tests/data/PROVENANCE.md`. It is symmetric with a
/// diagonal of `+4..+11`, exercising the large-alphabet SIMD path at `i16` width. A test asserts
/// symmetry and a few landmark values so a transcription slip cannot pass silently.
#[rustfmt::skip]
pub const BLOSUM62: [i32; 400] = [
//   A   R   N   D   C   Q   E   G   H   I   L   K   M   F   P   S   T   W   Y   V
     4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, // A
    -1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, // R
    -2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3, // N
    -2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3, // D
     0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, // C
    -1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2, // Q
    -1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2, // E
     0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, // G
    -2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3, // H
    -1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, // I
    -1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, // L
    -1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2, // K
    -1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, // M
    -2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, // F
    -1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, // P
     1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2, // S
     0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, // T
    -3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, // W
    -2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -2, // Y
     0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -2,  4, // V
];

/// BLOSUM62 protein scoring over the 20-symbol amino-acid alphabet, with the standard BLASTP affine
/// gap penalty (`gap_open = 11`, `gap_ext = 1`; our convention charges `11 + (n-1)·1` for a length-`n`
/// gap). A large alphabet (20 > 16) forces the Precomputed SIMD layout.
pub fn blosum62() -> Scoring {
    Scoring::new(20, BLOSUM62.to_vec(), 11, 1).unwrap()
}

/// Encode an ASCII protein string to `0..20` indices in [`AMINO_ACIDS`] order, skipping whitespace.
/// Panics on any character outside the 20 standard amino acids.
pub fn encode_protein(seq: &str) -> Vec<u8> {
    seq.bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .map(|b| {
            let up = b.to_ascii_uppercase();
            AMINO_ACIDS
                .iter()
                .position(|&a| a == up)
                .unwrap_or_else(|| panic!("unexpected residue {:?}", up as char)) as u8
        })
        .collect()
}
