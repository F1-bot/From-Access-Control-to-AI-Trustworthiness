#!/usr/bin/env bash
# Sequential GPU chain for the remaining adversarial re-analyses (run AFTER pgd_sanity finishes).
cd "$(dirname "$0")"
LOG=results/run_adv_gpu.log
echo "chain start $(date)" > $LOG
echo "=== ir1_query_chain ===" >> $LOG
PYTHONIOENCODING=utf-8 python ir1_query_chain.py >> $LOG 2>&1
echo "=== transfer_hardaware ===" >> $LOG
PYTHONIOENCODING=utf-8 python transfer_hardaware.py >> $LOG 2>&1
echo "=== ip2_reexamination ===" >> $LOG
PYTHONIOENCODING=utf-8 python ip2_reexamination.py >> $LOG 2>&1
echo "=== e4c_extract_deployed ===" >> $LOG
PYTHONIOENCODING=utf-8 python e4c_extract_deployed.py >> $LOG 2>&1
echo "chain done $(date)" >> $LOG
