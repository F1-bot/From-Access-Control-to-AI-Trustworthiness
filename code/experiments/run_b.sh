#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
LOG=results/run_b.log
echo "start $(date)" >> $LOG
python train_surrogate.py >> $LOG 2>&1
python e2b_greedy.py --victims baseline_s0,hardened_s0 --surrogate surrogate_data >> $LOG 2>&1
python e4b_extract.py --victim baseline_s0 >> $LOG 2>&1
python e3b_poison.py --betas 0.01,0.05,0.10 >> $LOG 2>&1
echo "done $(date)" >> $LOG
