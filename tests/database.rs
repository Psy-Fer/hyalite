//! Database-scan tests. `Database::scan` is validated *against* `align_pair` (itself pinned by
//! the brute-force oracle in `alignment.rs`): scanning a query must equal the best single-pair
//! alignment over the database, with the smallest-index tie-break. Checked exhaustively over all
//! short databases and queries, for every mode.

use hyalite::{BestHit, Database, Mode, Scoring, Scratch, SearchType, align_pair};

const ALL_MODES: [Mode; 4] = [Mode::Sw, Mode::Nw, Mode::Hw, Mode::Ov];

fn dna() -> Scoring {
    Scoring::new(
        4,
        vec![
            2, -1, -1, -1, //
            -1, 2, -1, -1, //
            -1, -1, 2, -1, //
            -1, -1, -1, 2,
        ],
        2,
        1,
    )
    .unwrap()
}

fn all_sequences(alphabet: u8, max_len: usize) -> Vec<Vec<u8>> {
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

/// The expected scan result, computed independently by taking the best `align_pair` over the
/// database with the smallest-index tie-break.
fn reference(
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
        // Strictly-greater keeps the smallest index on a tie.
        if best.is_none_or(|b| candidate.score > b.score) {
            best = Some(candidate);
        }
    }
    best.expect("database is non-empty")
}

#[test]
fn scan_matches_best_align_pair_over_all_short_databases() {
    let scoring = dna();
    let seqs = all_sequences(4, 2); // 21 sequences of length 0..=2
    let queries = all_sequences(4, 3);

    // Every 2-sequence database (ordered pairs, so tie-break ordering is exercised both ways).
    for a in &seqs {
        for b in &seqs {
            let database_seqs = [a.clone(), b.clone()];
            for mode in ALL_MODES {
                for st in [SearchType::Score, SearchType::ScoreEnd] {
                    let db = Database::builder()
                        .sequences(&database_seqs)
                        .scoring(scoring.clone())
                        .mode(mode)
                        .search_type(st)
                        .max_query_len(3)
                        .build()
                        .unwrap();
                    let mut scratch = Scratch::new(&db);
                    for q in &queries {
                        let got = db.scan(&mut scratch, q);
                        let want = reference(&database_seqs, &scoring, mode, st, q);
                        assert_eq!(
                            got, want,
                            "mode {mode}, {st}, db={database_seqs:?}, query={q:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn larger_database_scan_matches_reference() {
    // A single bigger database with a handful of varied queries, all modes.
    let scoring = dna();
    let seqs: Vec<Vec<u8>> = vec![
        vec![0, 1, 2, 3],
        vec![3, 2, 1, 0],
        vec![0, 0, 0, 0],
        vec![2, 2, 2],
        vec![0, 1, 2, 3, 0, 1, 2, 3],
        vec![1],
        vec![],
    ];
    let queries: [&[u8]; 5] = [&[0, 1, 2, 3], &[2, 2, 2], &[3], &[0, 1, 2, 3, 0, 1], &[]];

    for mode in ALL_MODES {
        let db = Database::builder()
            .sequences(&seqs)
            .scoring(scoring.clone())
            .mode(mode)
            .search_type(SearchType::ScoreEnd)
            .max_query_len(8)
            .build()
            .unwrap();
        let mut scratch = Scratch::new(&db);
        for q in queries {
            let got = db.scan(&mut scratch, q);
            let want = reference(&seqs, &scoring, mode, SearchType::ScoreEnd, q);
            assert_eq!(got, want, "mode {mode}, query={q:?}");
        }
    }
}

#[test]
fn database_can_be_shared_across_threads() {
    // The whole point of the immutable-Database / per-thread-Scratch split: share one Arc<Database>
    // across threads, each with its own Scratch, and get identical results.
    use std::sync::Arc;
    use std::thread;

    let db = Arc::new(
        Database::builder()
            .sequences(&[vec![0u8, 1, 2, 3], vec![2u8, 2, 2, 2]])
            .scoring(dna())
            .mode(Mode::Sw)
            .search_type(SearchType::ScoreEnd)
            .max_query_len(8)
            .build()
            .unwrap(),
    );

    let query = [0u8, 1, 2, 3];
    let expected = {
        let mut s = Scratch::new(&db);
        db.scan(&mut s, &query)
    };

    let handles: Vec<_> = (0..4)
        .map(|_| {
            let db = Arc::clone(&db);
            thread::spawn(move || {
                let mut scratch = Scratch::new(&db);
                // Many scans per thread to shake out any shared-state assumptions.
                (0..100)
                    .map(|_| db.scan(&mut scratch, &query))
                    .last()
                    .unwrap()
            })
        })
        .collect();

    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}
