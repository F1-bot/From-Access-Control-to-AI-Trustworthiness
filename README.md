# From Access Control to AI Trustworthiness: code and data

This repository holds the code and data of the article *From Access Control to AI Trustworthiness: Extending Graded
Security Evaluation Criteria with a Certifiable Service Class and Probabilistic Conformity Semantics*. The article has
been submitted to the *Journal of Cybersecurity and Privacy* (MDPI).

The article extends graded security evaluation criteria of the TCSEC, ITSEC and Common Criteria lineage to the
machine-learning components of AI systems. It uses the Ukrainian criteria ND TZI 2.5-004-99 and 2.5-005-99 as the
instantiation. The repository contains the material behind six results of the article:

- a machine-readable formalisation of the legacy criteria (22 services, 67 levels, the assurance matrix and 90 standard
profiles) with a profile algebra;
- the AI-trustworthiness service class I, with 9 services and 31 levels;
- 19 AI life-cycle assurance rows;
- probabilistic conformity semantics with a pre-test feasibility check;
- a coverage evaluation against MITRE ATLAS and a crosswalk of 276 requirement units;
- a case study that rates an attack detector for an industrial control system on HAI 22.04, with a confirmatory
replication on HAI 21.03.



## Contents


| Path                      | Contents                                                                                                                                                                                                                                                                                                                                        |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `code/aitcert/aitcert/`   | The `aitcert` package: formal model and profile algebra (`model.py`), probabilistic conformity semantics (`stats.py`), pre-test feasibility check (`feasibility.py`), ATLAS coverage (`atlas.py`) and crosswalk coverage (`crosswalk.py`).                                                                                                      |
| `code/aitcert/spec/`      | Machine-readable specifications: the legacy services (`legacy_services.yaml`), the assurance matrix (`legacy_guarantees.yaml`), class I (`ai_services.yaml`), the AI assurance rows (`ai_guarantees.yaml`), the mapping of the ATLAS mitigations (`atlas_mitigation_map.yaml`) and the crosswalk (`crosswalk_data.py`).                         |
| `code/aitcert/*.py`       | Analyses (`run_*.py`): ATLAS and crosswalk coverage, reference profiles, sensitivity, and the numbers of the statistical semantics (`run_stats_semantics.py`). `make_second_coder.py` builds the blinded coding package.                                                                                                                        |
| `code/aitcert/data_prep/` | Parsers that turn the source documents into `data/processed/`.                                                                                                                                                                                                                                                                                  |
| `code/aitcert/tests/`     | Unit tests.                                                                                                                                                                                                                                                                                                                                     |
| `code/experiments/`       | Case study on HAI 22.04: experiments E1–E5 and the signed-release test. It also holds the evaluation plan (`preregistration_original_2026-10-11.yaml`, plus `preregistration.yaml` with the log of deviations), the evidence classes (`case_evidence.yaml`), the results and decisions (`results/`) and the trained models (`results/models/`). |
| `code/replication/`       | Confirmatory replication on HAI 21.03. It holds the plan frozen before the test data were analysed (`replication_plan_2103.yaml`, digest in `results/plan_freeze.json`), the scripts `r1`–`r13`, the results, the models and the run log (`README.md`).                                                                                         |
| `data/processed/`         | Processed inputs: attack tables of HAI 21.03 and 22.04, paragraphs of the EU AI Act, subcategories of NIST AI RMF and CSF 2.0, priorities of NIST IR 8596 and the standard profiles of ND TZI 2.5-005-99.                                                                                                                                       |
| `data/results/`           | Results of the ATLAS and crosswalk coverage analyses, the sensitivity analyses, the reference profiles and the statistical semantics.                                                                                                                                                                                                           |
| `data/raw/`               | Raw third-party data, which are not redistributed (see `data/raw/README.md`). `data/raw/hai/MANIFEST.json` holds the SHA-256 digests of the HAI files used.                                                                                                                                                                                     |
| `supplementary/`          | Table S1, the crosswalk of 276 requirement units (CSV), and the blinded coding package for independent coders (File S6). The reference key of the package is not published; `code/aitcert/make_second_coder.py` rebuilds it and needs the ATLAS data to do so.                                                                                  |




## Installation

The study used Python 3.10.6:

```
pip install -r requirements.txt
```

`requirements.txt` installs the default build of PyTorch 2.6.0. For GPU training, install its CUDA build
([https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)). Training one detector took about 100 s on the NVIDIA GeForce RTX 4080
SUPER used in the study.

## Reproducing the results

Commands are run from the repository root unless stated otherwise. The result files, trained models and test scores are
included, so the following commands need no raw data:

```
python -m pytest code/aitcert/tests      # 24 tests; the one that needs the ATLAS data is skipped without it
python code/replication/r9_assess.py     # decisions and ratings of the replication (results/r_assessment.json)
python code/replication/r11_table.py     # Table 10 of the article
cd code/aitcert
python run_crosswalk.py                  # crosswalk coverage (data/results/crosswalk_coverage.json)
python run_reference_profiles.py         # reference profiles (data/results/reference_profiles.json)
python run_stats_semantics.py            # statistical semantics (data/results/stats_semantics.json)
```

The remaining steps need the raw data; `data/raw/README.md` describes how to obtain them.

1. **ATLAS analyses.** In `code/aitcert`, run `run_atlas_analysis.py` and `run_sensitivity.py`.
2. **Case study on HAI 22.04.** In `code/experiments`, run in this order:
  1. `run_all.sh` and `run_b.sh`.
  2. `assess_case.py`, the original assessment.
  3. The revision analyses:
    - `ir2_aggregation.py`
    - `e1c_false_alarm_dispersion.py`
    - `e1b_localisation.py`
    - `e5b_drift_retrained.py`
    - `cluster_bootstrap.py`
    - `e5b_rate_uncertainty.py`
    - `e5b_tag_mapping.py --sensitivity`
    - `im2_sign_v2.py`
    - `decision_sensitivity.py`
    - `pgd_sanity.py`
    - `stats_revision.py`
    - `e3c_poison_grouped.py`
    - `id3_units.py --retrain_poisoned`
    - `run_adv_gpu.sh`
  4. `assess_case_v2.py`. It gives the final decisions and ratings, and also re-checks the digests of the HAI files.
3. **Replication on HAI 21.03.** In `code/replication`, run the scripts in the order given in its `README.md`. Every
  script that reads test data first checks the digest of the frozen plan.

Seeds are fixed in the scripts. However, cuDNN ran in its default non-deterministic mode, so retrained models can differ
slightly from the published ones. The models and scores used in the article are included.

## Licence

The code and data of this repository are released under the MIT License (see `LICENSE`).

Raw third-party data are not redistributed. Some files contain extracts of public sources: the processed files in
`data/processed/` and the requirement texts in the crosswalk and the coding package. These extracts come from HAI, MITRE
ATLAS, the EU AI Act, NIST publications and ND TZI, and they remain subject to the terms of those sources.

## Citation

Please cite the article. Its full reference will be added here on publication.