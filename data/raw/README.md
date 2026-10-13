# Raw data (not redistributed)

The analyses use public third-party data that this repository does not redistribute. The processed outputs of these
sources are included in `data/processed/`, so the raw data are needed only to re-run the experiments (HAI), the analyses
that read MITRE ATLAS, and the parsers. Place the files at the paths below, which are the paths the scripts read.

| Path | Source | How to obtain and use |
|---|---|---|
| `data/raw/hai/` (`hai-21.03/`, `hai-22.04/`, `hai-23.05/`, `hai_dataset_technical_details.pdf`) | HAI dataset, https://github.com/icsdataset/hai | `python code/experiments/download_hai.py data/raw/hai` downloads the three versions and checks the SHA-256 identifiers of Git LFS. `MANIFEST.json` lists the digests of the files used in the article. The attack tables are parsed with `python code/aitcert/data_prep/parse_hai_attacks.py .` and `parse_hai2103_attacks.py .` |
| `data/raw/atlas-data/` | MITRE ATLAS data, release 2026.09, https://github.com/mitre-atlas/atlas-data | `git clone --depth 1 --branch v2026.09 https://github.com/mitre-atlas/atlas-data data/raw/atlas-data` (the scripts read `dist/v6/ATLAS-2026.09.yaml`) |
| `data/raw/frameworks/NIST.AI.100-1.txt` | NIST AI 100-1 (AI RMF 1.0), https://doi.org/10.6028/NIST.AI.100-1 | Text of the PDF (`pdftotext -enc UTF-8 -layout`); parsed with `python code/aitcert/data_prep/parse_nist.py .` |
| `data/raw/frameworks/csf2_cprt.json` | NIST CSF 2.0, https://doi.org/10.6028/NIST.CSWP.29 | JSON export of CSF 2.0 from the NIST Cybersecurity and Privacy Reference Tool (CPRT); parsed with `parse_nist.py` |
| `data/raw/frameworks/NIST.IR.8596.txt` | NIST IR 8596 (initial preliminary draft), https://doi.org/10.6028/NIST.IR.8596.iprd | Text of the PDF (`pdftotext -enc UTF-8 -layout`); parsed with `python code/aitcert/data_prep/parse_ir8596.py .` |
| `data/raw/frameworks/aiact/artN.html` (Articles 3, 6, 9–15, 17, 26, 43, 50, 72, 73 and 113) | EU AI Act, Regulation (EU) 2024/1689, as amended by Regulation (EU) 2026/1744; consolidated article pages at https://artificialintelligenceact.eu/article/N/, official text at https://eur-lex.europa.eu/eli/reg/2024/1689/oj | `python code/aitcert/data_prep/parse_aiact.py data/raw/frameworks/aiact data/processed/aiact_paragraphs.json` |
| UTF-8 text of ND TZI 2.5-005-99 | ND TZI 2.5-004-99 and 2.5-005-99, official texts in Ukrainian | `python code/aitcert/data_prep/parse_profiles.py <text> data/processed/nd_tzi_2.5-005-99_profiles.json` |
