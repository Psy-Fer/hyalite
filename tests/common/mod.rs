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

/// The four gap penalties in `Scoring::new_asymmetric` order. A `Right` move (a gap in the
/// query) is charged the query pair, a `Down` move (a gap in the target) the target pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Gaps {
    pub query_open: i32,
    pub query_ext: i32,
    pub target_open: i32,
    pub target_ext: i32,
}

impl Gaps {
    /// Both directions charged alike, i.e. what `Scoring::new` builds.
    pub fn symmetric(go: i32, ge: i32) -> Self {
        Gaps {
            query_open: go,
            query_ext: ge,
            target_open: go,
            target_ext: ge,
        }
    }

    /// The scheme these penalties describe.
    pub fn scoring(self, al: usize, matrix: Vec<i32>) -> Scoring {
        Scoring::new_asymmetric(
            al,
            matrix,
            self.query_open,
            self.query_ext,
            self.target_open,
            self.target_ext,
        )
        .unwrap()
    }

    /// The same penalties with the two directions exchanged — the scheme that must score a
    /// transposed pair identically.
    pub fn transposed(self) -> Self {
        Gaps {
            query_open: self.target_open,
            query_ext: self.target_ext,
            target_open: self.query_open,
            target_ext: self.query_ext,
        }
    }
}

/// Fixed parameters threaded through the brute-force recursion.
pub struct Prob<'a> {
    pub q: &'a [u8],
    pub t: &'a [u8],
    pub mat: &'a [i32],
    pub al: usize,
    pub gaps: Gaps,
}

/// Exact global (NW) score by enumerating every alignment path, charging affine gap penalties
/// from maximal same-direction runs. Exponential; only for tiny slices.
pub fn brute_nw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_nw_asym(q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute_nw`] with the two gap directions charged independently.
pub fn brute_nw_asym(q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
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
            let cost = if last == Move::Right {
                p.gaps.query_ext
            } else {
                p.gaps.query_open
            };
            best = best.max((-cost).saturating_add(rec(p, i, j + 1, Move::Right)));
        }
        if i < m {
            let cost = if last == Move::Down {
                p.gaps.target_ext
            } else {
                p.gaps.target_open
            };
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
            gaps,
        },
        0,
        0,
        Move::Start,
    )
}

/// Local (SW): best global score over every substring pair, floored at 0.
pub fn brute_sw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_sw_asym(q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute_sw`] with the two gap directions charged independently.
pub fn brute_sw_asym(q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    best = best.max(brute_nw_asym(&q[a..b], &t[c..d], mat, al, gaps));
                }
            }
        }
    }
    best
}

/// Semi-global (HW): query fully aligned to the best target window.
pub fn brute_hw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_hw_asym(q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute_hw`] with the two gap directions charged independently.
pub fn brute_hw_asym(q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
    let n = t.len();
    let mut best = i32::MIN;
    for c in 0..=n {
        for d in c..=n {
            best = best.max(brute_nw_asym(q, &t[c..d], mat, al, gaps));
        }
    }
    best
}

/// Semi-global transpose (SHW): the whole **target** is aligned to the best **query** window (the
/// mirror image of [`brute_hw`], which windows the target).
pub fn brute_shw(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_shw_asym(q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute_shw`] with the two gap directions charged independently.
pub fn brute_shw_asym(q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
    let m = q.len();
    let mut best = i32::MIN;
    for a in 0..=m {
        for b in a..=m {
            best = best.max(brute_nw_asym(&q[a..b], t, mat, al, gaps));
        }
    }
    best
}

/// Overlap (OV): best global score over substring pairs whose alignment touches a border at both
/// ends, floored at 0.
pub fn brute_ov(q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_ov_asym(q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute_ov`] with the two gap directions charged independently.
pub fn brute_ov_asym(q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
    let (m, n) = (q.len(), t.len());
    let mut best = 0;
    for a in 0..=m {
        for b in a..=m {
            for c in 0..=n {
                for d in c..=n {
                    if (a == 0 || c == 0) && (b == m || d == n) {
                        best = best.max(brute_nw_asym(&q[a..b], &t[c..d], mat, al, gaps));
                    }
                }
            }
        }
    }
    best
}

/// Brute-force score for any mode.
pub fn brute(mode: Mode, q: &[u8], t: &[u8], mat: &[i32], al: usize, go: i32, ge: i32) -> i32 {
    brute_asym(mode, q, t, mat, al, Gaps::symmetric(go, ge))
}

/// [`brute`] with the two gap directions charged independently.
pub fn brute_asym(mode: Mode, q: &[u8], t: &[u8], mat: &[i32], al: usize, gaps: Gaps) -> i32 {
    match mode {
        Mode::Nw => brute_nw_asym(q, t, mat, al, gaps),
        Mode::Sw => brute_sw_asym(q, t, mat, al, gaps),
        Mode::Hw => brute_hw_asym(q, t, mat, al, gaps),
        Mode::Ov => brute_ov_asym(q, t, mat, al, gaps),
        Mode::Shw => brute_shw_asym(q, t, mat, al, gaps),
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
