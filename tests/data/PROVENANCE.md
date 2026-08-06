# Test data provenance

Small, real biological sequences vendored for the `hyalite` test suite. Everything here is
committed directly (no network access at test time) and is public-domain or otherwise freely
redistributable under this repository's MIT license. Retrieved 2026-07-28.

| File | Source | Size | License / terms |
|------|--------|------|-----------------|
| `phix174_NC_001422.1.fasta` | NCBI RefSeq `NC_001422.1` — *Escherichia* phage phiX174, complete genome (5,386 bp), the classic Illumina sequencing control | 5,520 B | NCBI places no restrictions on use or redistribution of these data. phiX174 is a decades-old foundational reference genome with no known IP encumbrance. |
| `lambda_NC_001416.1.fasta` | NCBI RefSeq `NC_001416.1` — Enterobacteria phage lambda, complete genome (48,502 bp) | 49,254 B | Same as above. |
| `cr4_adapters.fa` | Hand-authored. The `TSO_10x` sequence is the 10x Genomics template-switch oligo used by STAR's `--clipAdapterType CellRanger4` path (`AAGCAGTGGTATCAACGCAGAGTACATGGG`); the `polyA`/`polyT`/`polyG` entries are synthetic homopolymer decoys. | <300 B | Short factual DNA sequences are not copyrightable. The CR4 scoring/params used alongside this file are transcribed from STAR (MIT, © 2019 Alexander Dobin). |
| BLOSUM62 matrix (`common::BLOSUM62`) | The standard BLOSUM62 amino-acid substitution matrix (Henikoff & Henikoff 1992), as distributed by NCBI BLAST. Embedded as a `const` in `tests/common/mod.rs`. | — | A published integer log-odds matrix; not copyrightable and freely redistributable (ships in NCBI BLAST and every alignment toolkit). |
| Ubiquitin sequence (`real_data::UBIQUITIN`) | Human ubiquitin monomer, 76 aa (UniProt `P0CG48`, the poly-ubiquitin `UBC` repeat unit) — used as a real large-alphabet protein for the BLOSUM62 tests, alongside programmatically derived homolog variants. | <100 B | Short factual protein sequence; not copyrightable. UniProt data is CC-BY 4.0. |

## Sources

- phiX174: <https://www.ncbi.nlm.nih.gov/nuccore/NC_001422.1>
  (efetch: `https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_001422.1&rettype=fasta&retmode=text`)
- lambda: <https://www.ncbi.nlm.nih.gov/nuccore/NC_001416.1>
- NCBI data-use policy: <https://www.ncbi.nlm.nih.gov/home/about/policies/>
- 10x TSO sequence: <https://kb.10xgenomics.com/hc/en-us/articles/27297395004429>
- STAR CellRanger4 params (MIT): <https://github.com/alexdobin/STAR> —
  `source/ParametersClip_initialize.cpp`, `source/ClipCR4.cpp`

## CellRanger4 scoring (transcribed from STAR source)

`hyalite`'s `cr4_scoring()` test helper reproduces STAR's Opal call parameters: alphabet
`A,C,G,T,N`; match `+1`, mismatch `-2`, any-vs-`N` `-2`, `N`-vs-`N` `0`; `gap_open = 2`,
`gap_ext = 2`; overlap mode (`OPAL_MODE_OV`); `SCORE_END`.

**Note on orientation:** in STAR the 64-member Opal "database" is a batch of reads and the
adapter (TSO / polyA) is the *query* — the transpose of the "adapter database, read query"
framing. This matters when reproducing STAR byte-for-byte during the future rustar integration,
not for the self-consistency tests here.
