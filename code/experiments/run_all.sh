#!/usr/bin/env bash
# Full experiment pipeline for the HAI case study (sequential, logs to results/run_all.log)
set -e
cd "$(dirname "$0")"
LOG=results/run_all.log
echo "start $(date)" >> $LOG
for s in 0 1 2 3 4; do python e1_baseline.py --seed $s --epochs 12 --per_epoch 250000 --tag baseline >> $LOG 2>&1; done
python e1_baseline.py --seed 0 --epochs 12 --per_epoch 250000 --corrupt 0.3 --tag hardened >> $LOG 2>&1
python im_sign.py baseline_s0 >> $LOG 2>&1
python e5_drift.py --victim baseline_s0 >> $LOG 2>&1
python e2_evasion.py --tags baseline_s0,hardened_s0 >> $LOG 2>&1
python e3_poison.py --beta 0.01 --epochs 10 --per_epoch 200000 >> $LOG 2>&1
python e4_extract.py --victim baseline_s0 >> $LOG 2>&1
echo "done $(date)" >> $LOG
