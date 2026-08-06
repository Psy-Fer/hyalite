//! Tests over real biological sequences (see `tests/data/PROVENANCE.md`).
//!
//! Synthetic random data misses the sequence *composition* that trips fragile aligners —
//! homopolymer runs, low-complexity regions, real adapter/read overlaps. These tests slice real
//! phage genomes and use the exact STAR CellRanger4 adapter setup (the first consumer's actual
//! workload) to check the same invariants the property tests assert, on realistic input.
//!
//! Data is embedded with `include_str!`, so there is no network or filesystem access at test time.

mod common;

use common::{
    ALL_MODES, AMINO_ACIDS, BLOSUM62, blosum62, cr4_scoring, dna, encode_protein, parse_fasta,
    reference_scan,
};
use hyalite::{
    Backend, BackendChoice, Database, Layout, Mode, ScoreWidth, Scoring, Scratch, SearchType,
    align_pair,
};

const PHIX_FASTA: &str = include_str!("data/phix174_NC_001422.1.fasta");
const LAMBDA_FASTA: &str = include_str!("data/lambda_NC_001416.1.fasta");
const CR4_ADAPTERS_FASTA: &str = include_str!("data/cr4_adapters.fa");

fn phix() -> Vec<u8> {
    let records = parse_fasta(PHIX_FASTA);
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].1.len(), 5386, "phiX174 is 5386 bp");
    records[0].1.clone()
}

fn lambda() -> Vec<u8> {
    let records = parse_fasta(LAMBDA_FASTA);
    assert_eq!(records[0].1.len(), 48502, "lambda is 48502 bp");
    records[0].1.clone()
}

/// Count the (possibly overlapping) occurrences of `needle` in `hay`.
fn occurrences(hay: &[u8], needle: &[u8]) -> usize {
    if needle.is_empty() || needle.len() > hay.len() {
        return 0;
    }
    (0..=hay.len() - needle.len())
        .filter(|&i| &hay[i..i + needle.len()] == needle)
        .count()
}

#[test]
fn exact_substring_of_phix_is_recovered() {
    // A real 50 bp window taken from phiX must align to the full genome with a perfect score in
    // both local (SW) and semi-global (HW) modes.
    let genome = phix();
    let scoring = dna(); // match +2
    let (offset, len) = (1234, 50);
    let window = genome[offset..offset + len].to_vec();
    let perfect = (len as i32) * 2;

    for mode in [Mode::Sw, Mode::Hw] {
        let hit = align_pair(&window, &genome, &scoring, mode, SearchType::ScoreEnd).unwrap();
        assert_eq!(hit.score, perfect, "{mode}: window should match perfectly");
        assert_eq!(hit.query_end, Some(len - 1), "{mode}: whole query aligned");
        // The window is unique in phiX, so the target end is its exact position.
        assert_eq!(occurrences(&genome, &window), 1, "window assumed unique");
        assert_eq!(
            hit.target_end,
            Some(offset + len - 1),
            "{mode}: located at source"
        );
    }
}

#[test]
fn scan_over_phix_windows_matches_reference_on_real_data() {
    // Build a database of several real phiX windows and scan real query windows against it, in
    // every mode and search type. scan must equal best-of-align_pair, exactly as on synthetic data.
    let genome = phix();
    let db_windows: Vec<Vec<u8>> = [0usize, 500, 1500, 2500, 4000]
        .iter()
        .map(|&o| genome[o..o + 60].to_vec())
        .collect();
    let queries: Vec<Vec<u8>> = [
        genome[500..560].to_vec(),   // exact copy of db window 1
        genome[510..570].to_vec(),   // overlaps it, shifted
        genome[3000..3060].to_vec(), // matches nothing in the db well
        genome[1500..1530].to_vec(), // a prefix of db window 2
    ]
    .to_vec();

    let scoring = dna();
    for mode in ALL_MODES {
        for st in [SearchType::Score, SearchType::ScoreEnd] {
            let db = Database::builder()
                .sequences(&db_windows)
                .scoring(scoring.clone())
                .mode(mode)
                .search_type(st)
                .max_query_len(64)
                .build()
                .unwrap();
            let mut scratch = Scratch::new(&db);
            for q in &queries {
                let got = db.scan(&mut scratch, q);
                let want = reference_scan(&db_windows, &scoring, mode, st, q);
                assert_eq!(got, want, "{mode}/{st} on real phiX windows");
            }
        }
    }
}

#[test]
fn mode_ordering_and_score_end_consistency_on_phix() {
    // The universal invariants, exercised on real sequence composition rather than random bytes.
    let genome = phix();
    let scoring = dna();
    for &(qo, to, l) in &[
        (100usize, 100usize, 80usize),
        (2000, 2050, 120),
        (10, 4000, 40),
    ] {
        let q = genome[qo..qo + l].to_vec();
        let t = genome[to..(to + l).min(genome.len())].to_vec();
        let s = |mode| {
            align_pair(&q, &t, &scoring, mode, SearchType::Score)
                .unwrap()
                .score
        };
        let (sw, ov, hw, nw) = (s(Mode::Sw), s(Mode::Ov), s(Mode::Hw), s(Mode::Nw));
        assert!(
            sw >= 0 && sw >= ov && ov >= hw && hw >= nw,
            "ordering: {sw} {ov} {hw} {nw}"
        );

        for mode in ALL_MODES {
            let a = align_pair(&q, &t, &scoring, mode, SearchType::Score)
                .unwrap()
                .score;
            let b = align_pair(&q, &t, &scoring, mode, SearchType::ScoreEnd)
                .unwrap()
                .score;
            assert_eq!(a, b, "{mode}: Score vs ScoreEnd");
        }
    }
}

#[test]
fn cr4_adapter_scan_identifies_the_right_adapter() {
    // The primary consumer's workload: overlap-mode scan of reads against the CR4 adapter set,
    // using STAR's exact scoring. (Orientation is our adapter-as-database framing; see PROVENANCE.)
    let adapters = parse_fasta(CR4_ADAPTERS_FASTA);
    let adapter_seqs: Vec<Vec<u8>> = adapters.iter().map(|(_, s)| s.clone()).collect();
    assert_eq!(adapters[0].0, "TSO_10x");
    let tso = adapter_seqs[0].clone();
    let genome = phix();
    let scoring = cr4_scoring();

    let db = Database::builder()
        .sequences(&adapter_seqs)
        .scoring(scoring.clone())
        .mode(Mode::Ov)
        .search_type(SearchType::ScoreEnd)
        .max_query_len(128)
        .build()
        .unwrap();
    let mut scratch = Scratch::new(&db);

    // Read = TSO (30 nt) immediately followed by 40 nt of genomic sequence. Overlap mode aligns
    // the adapter to the read's prefix; the full 30 nt TSO matches → score 30, and TSO (index 0)
    // wins because no other 30 nt adapter can be beaten by this read.
    let mut tso_read = tso.clone();
    tso_read.extend_from_slice(&genome[800..840]);
    let hit = db.scan(&mut scratch, &tso_read);
    assert_eq!(hit.db_index, 0, "TSO-led read should match the TSO adapter");
    assert_eq!(hit.score, 30, "full 30 nt TSO overlap at +1 per base");

    // Read = 30 nt genomic followed by a 25 A polyA tail. polyA (index 1) should win with a
    // score of at least the 25-base tail overlap, and beat the TSO adapter for this read.
    let mut polya_read = genome[1500..1530].to_vec();
    polya_read.extend(std::iter::repeat_n(0u8, 25)); // 0 == 'A'
    let hit = db.scan(&mut scratch, &polya_read);
    assert_eq!(
        hit.db_index, 1,
        "polyA-tailed read should match the polyA adapter"
    );
    assert!(
        hit.score >= 25,
        "at least the 25 nt polyA overlap, got {}",
        hit.score
    );

    // scan must still equal best-of-align_pair on this real workload.
    for read in [&tso_read, &polya_read] {
        let want = reference_scan(
            &adapter_seqs,
            &scoring,
            Mode::Ov,
            SearchType::ScoreEnd,
            read,
        );
        assert_eq!(db.scan(&mut scratch, read), want);
    }
}

#[test]
fn real_data_exercises_i16_score_width_and_the_proof_holds() {
    // DNA/±2 scoring on short windows stays in i8; a few-hundred-bp real alignment pushes into
    // i16. This is the escalation path exercised by genuine sequence lengths.
    let genome = lambda();
    let scoring = Scoring::new(4, common::identity_matrix(4, 1, -1), 2, 1).unwrap(); // match +1

    let len = 250;
    let window = genome[10_000..10_000 + len].to_vec();
    let width = scoring.required_width(Mode::Nw, len, len).unwrap();
    assert_eq!(width, ScoreWidth::I16, "250 * 1 = 250 needs i16");

    let hit = align_pair(&window, &window, &scoring, Mode::Nw, SearchType::Score).unwrap();
    assert_eq!(hit.score, len as i32, "self-alignment is a perfect match");
    assert!(
        (hit.score as i64).abs() <= width.max_abs(),
        "score fits the proven width"
    );

    // The proof reaches i32 for genome-scale lengths without us running that DP (which would be
    // gigabytes of matrix). This checks the proof on realistic large lengths cheaply.
    assert_eq!(
        scoring.required_width(Mode::Nw, 40_000, 40_000).unwrap(),
        ScoreWidth::I32
    );
}

// ---------------------------------------------------------------------------
// Protein / BLOSUM62: the large-alphabet (20-symbol) SIMD path on real data
// ---------------------------------------------------------------------------

/// Human ubiquitin monomer (76 aa; UniProt P0CG48). A real large-alphabet sequence for the
/// BLOSUM62 tests; see `tests/data/PROVENANCE.md`.
const UBIQUITIN: &str =
    "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG";

/// Return `seq` with the given `(position, amino_acid_index)` point substitutions applied — a cheap
/// stand-in for protein homologs, so the database holds genuinely related-but-distinct sequences.
fn mutate(seq: &[u8], subs: &[(usize, u8)]) -> Vec<u8> {
    let mut out = seq.to_vec();
    for &(pos, aa) in subs {
        out[pos] = aa;
    }
    out
}

#[test]
fn blosum62_matrix_is_symmetric_with_expected_landmarks() {
    // A transcription slip in a 400-entry matrix must not pass silently. BLOSUM62 is symmetric, and
    // a handful of well-known values pin the orientation/scale.
    for i in 0..20 {
        for j in 0..20 {
            assert_eq!(
                BLOSUM62[i * 20 + j],
                BLOSUM62[j * 20 + i],
                "BLOSUM62 asymmetric at ({i}, {j})"
            );
        }
    }
    let idx = |c: u8| AMINO_ACIDS.iter().position(|&a| a == c).unwrap();
    let s = |a: u8, b: u8| BLOSUM62[idx(a) * 20 + idx(b)];
    assert_eq!(s(b'W', b'W'), 11, "Trp self-score");
    assert_eq!(s(b'C', b'C'), 9, "Cys self-score");
    assert_eq!(s(b'A', b'A'), 4, "Ala self-score");
    assert_eq!(s(b'L', b'I'), 2, "Leu/Ile (conservative)");
    assert_eq!(s(b'W', b'A'), -3, "Trp/Ala (dissimilar)");
    assert_eq!(s(b'P', b'W'), -4, "Pro/Trp");
}

#[test]
fn ubiquitin_self_alignment_scores_the_diagonal_sum() {
    // An independent end-to-end sanity check of protein scoring: a sequence aligned to itself under
    // any mode scores exactly the sum of its residues' BLOSUM62 self-scores (no gaps, all matches).
    let ub = encode_protein(UBIQUITIN);
    assert_eq!(ub.len(), 76);
    let expected: i32 = ub
        .iter()
        .map(|&a| BLOSUM62[a as usize * 20 + a as usize])
        .sum();
    let scoring = blosum62();
    for mode in ALL_MODES {
        let hit = align_pair(&ub, &ub, &scoring, mode, SearchType::Score).unwrap();
        assert_eq!(hit.score, expected, "ubiquitin self-alignment, {mode}");
    }
}

#[test]
fn protein_scan_matches_scalar_on_blosum62() {
    // A realistic large-alphabet workload: a small database of ubiquitin plus derived homologs and a
    // truncation, scanned with ubiquitin-variant queries under BLOSUM62. The alphabet (20 > 16)
    // forces the Precomputed SIMD layout; every SIMD backend must match the scalar oracle exactly.
    let ub = encode_protein(UBIQUITIN);
    let scoring = blosum62();

    let db_seqs = vec![
        ub.clone(),
        mutate(&ub, &[(3, 9), (17, 4), (40, 0), (60, 19)]), // a few point substitutions
        mutate(&ub, &[(0, 5), (10, 5), (20, 5), (30, 5), (70, 5)]), // more diverged
        ub[..50].to_vec(),                                  // N-terminal fragment
        ub[26..].to_vec(),                                  // C-terminal fragment
    ];
    let queries = vec![
        ub.clone(),
        mutate(&ub, &[(5, 0), (55, 11)]),
        ub[10..66].to_vec(),
    ];

    let simd: Vec<Backend> = [Backend::Sse41, Backend::Avx2]
        .into_iter()
        .filter(|b| b.is_available())
        .collect();
    if simd.is_empty() {
        return;
    }

    for mode in ALL_MODES {
        for st in [SearchType::Score, SearchType::ScoreEnd] {
            let build = |b: Backend| {
                Database::builder()
                    .sequences(&db_seqs)
                    .scoring(scoring.clone())
                    .mode(mode)
                    .search_type(st)
                    .max_query_len(76)
                    .backend(BackendChoice::Force(b))
                    .build()
                    .unwrap()
            };
            let oracle = build(Backend::Scalar);
            let mut os = Scratch::new(&oracle);
            for &b in &simd {
                let db = build(b);
                // 20-symbol alphabet ⇒ the byte-shuffle Gathered gather cannot apply; Precomputed does.
                assert_eq!(db.layout(), Some(Layout::Precomputed), "{b} {mode} {st}");
                let mut gs = Scratch::new(&db);
                for q in &queries {
                    assert_eq!(
                        db.scan(&mut gs, q),
                        oracle.scan(&mut os, q),
                        "{b} scan {mode} {st}"
                    );
                    let (mut ga, mut oa) = (Vec::new(), Vec::new());
                    db.scan_all(&mut gs, q, &mut ga);
                    oracle.scan_all(&mut os, q, &mut oa);
                    assert_eq!(ga, oa, "{b} scan_all {mode} {st}");
                    let (mut gsc, mut osc) = (Vec::new(), Vec::new());
                    db.scan_scores(&mut gs, q, &mut gsc);
                    oracle.scan_scores(&mut os, q, &mut osc);
                    assert_eq!(gsc, osc, "{b} scan_scores {mode} {st}");
                }
            }
        }
    }
}
