# Confirmatory replication on HAI 21.03

The replication applies the final procedure of the HAI 22.04 case study (`code/experiments`) to detectors trained with the
same recipes on HAI 21.03, an earlier campaign on the same testbed that was not used to design the criteria, the plan or
the procedure. The plan `replication_plan_2103.yaml` was frozen on 2026-10-13 at 04:13 UTC (SHA-256
`da8704e640f427f885627b25fd175527b6c468ec12717675e8fceeae04b21783`, recorded in `results/plan_freeze.json`) after the
detectors had been trained and before any script read the sensor data of the test files. Every script that reads test data
verifies the digest (`r_lib.require_frozen_plan`).

## Scripts, in the order in which they were run

1. `r1_train.py` trains the detectors and the attacker-side surrogates on training data only. The detectors are baseline
   seeds 0–4 and the hardened variant.
2. `r2_plan.py` does all the pre-test work and then freezes the plan. It covers:
   - the feasibility check of the replication plan and, retrospectively, of the original 22.04 plan
     (`results/feasibility_original_plan.json`);
   - the alarm on-delay, chosen on the HAI 22.04 data;
   - the operating points;
   - the calibration of the operating-envelope drift monitor, with the HAI 23.05 drift pilot;
   - the planned sanitiser, trained on train1 only;
   - writing and freezing the plan.

   `r2_plan.py --dry` writes the plan without freezing it and was used to check the computations before the freeze.
3. `r3_e1.py` runs E1: validity, workload and explanations.
4. `r4_e2.py` runs E2: adaptive evasion and transfer attacks.
5. `r6_e3.py` runs E3: the poisoning drill.
6. `r5_e4.py` runs E4: extraction and the query-based chain.
7. `r7_e5.py` runs E5: the operating-envelope drift monitor.
8. `r8_im2.py` signs the release and runs the verification tests.
9. `r9_assess.py` computes decisions and ratings at alpha 0.10, 0.05 and 0.01 (`results/r_assessment.json`).
10. `r10_exploratory.py` holds the exploratory analyses, which are not part of the plan: the operating curve of the
    sanitiser and HAI 22.04 with the on-delay.
11. `r11_table.py` prints the replication table of the article (Table 10) from the result files.
12. `r12_e2_causal.py` and `r13_post_hoc.py` hold the post-hoc checks described in the verification notes below; they are
    not part of the plan.

## Run log

- **`r4_e2.py` crash.** The script first crashed at the first attack, before producing any result.
  - Cause: `hai_lib.residuals` switches the GRU to evaluation mode, and the cuDNN RNN backward pass needs training mode.
  - Fix: the script now restores training mode after scoring. Dropout was already disabled, so this does not change any
    result.
- **`r4_e2.py` restart in two processes.** The script was later restarted with one process per detector variant
  (`--variant=...`), to use the GPU in parallel.
  - The two outputs, `results/r_e2_<variant>.json`, are merged by `r9_assess.py`.
  - The procedure was not changed.
- **`r9_assess.py` merge step.** A merge step was added after the E1 and E5 results existed. It combines the two
  per-variant E2 files; the decision logic was not changed.
- **No deviations from the frozen plan.**

## Notes from the verification (post hoc)

- **Alarm onset.** The implemented alarm logic places the onset at the 20th second of an episode even if the score
  lies in a merged gap at that second. `r13_post_hoc.py` recomputes E1 with a strictly causal onset:
  - detections and false alarms are unchanged for all six detectors;
  - only the times to detection change (baseline_s0 median 32.5 s instead of 29 s).

  `r12_e2_causal.py` repeats E2 and judges every perturbation with both logics. No attack decision differs: the
  baseline has 5 of 44 under both logics and the hardened variant 2 of 45, and the repeated run reproduces
  `r_e2.json`.
- **Planning design effect of the drift test.**
  - The feasibility row treated the 60 in-distribution hours as independent.
  - A planning design effect estimated from the training hours is 5.4, with which the false-positive test is not
    decidable (`results/r_post_hoc.json`).
- **Monotonicity of `smallest_theta_with_at_most`.** The function assumes that the number of validation episodes does
  not increase with the threshold.
  - This holds above the 99.9th percentile for all saved detectors.
  - It was not needed for them, because every saved detector kept the 99.9th percentile.
  - The thresholds of the drill models (`r6_e3.py`) are reported in `r_e3.json`.
- **Frozen evidence file.** The text of IV-3.1 in `replication_evidence.yaml` still names the PSI monitor.
  - The test is the operating-envelope monitor of the plan; the status classes are unaffected.
  - The file is left unchanged because its digest is part of the frozen plan.
- **Pre-freeze reads.** Before the freeze, `r2_plan.py` also read:
  - the HAI 22.04 test files and the stored 22.04 test scores, for the on-delay pilot;
  - the HAI 23.05 training files, for the drift pilot.

  No sensor data of the HAI 21.03 test files were read; only their label column was.
- **Lag-1 autocorrelation in `r7_e5.py`.** The script uses a pooled mean. De-meaning within files gives 0.395 instead
  of 0.448 for the in-distribution hours; no decision changes.

