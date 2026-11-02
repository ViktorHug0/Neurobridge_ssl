#!/bin/bash
# E4.3 streaming figure runs (user, 2026-10-21): 50 checkpoints, buffer / transductive-at-arrival (@new) /
# strict-causal (@prev) scores. Waits for the arm's prerequisites, then appends its 5 subject-pair parts to the CPU
# queue (run it as a cpu_queue item). 2 BLAS threads: the SAGE fit on tiny buffers depends on the thread count.
#   usage: stream50.sh <loso arm> <seed> <sage|score> [checkpoint name]
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
arm=$1 seed=$2 kind=$3 ckpt=${4:-checkpoint_test_best.pth}
src=results/things_eeg/cvpr/loso/$arm/seed$seed
params=results/things_eeg/cvpr/tta_screen_loso/${arm}_s$seed/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
# SCORE ignores the SAGE parameters; the stack without its own screen (yet) reuses ATM-128's file as a placeholder
[ $kind = score ] && params=results/things_eeg/cvpr/tta_screen_loso/atm128_s3300/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
until [ $(ls $src/*-sub-*/result.csv 2>/dev/null | wc -l) -eq 10 ] && [ -f $params ]; do sleep 300; done
o=results/things_eeg/cvpr/final/${arm}_s$seed/stream50_$kind L=logs/cvpr/final/${arm}_s$seed
mkdir -p $L results/things_eeg/cvpr/final/${arm}_s$seed
arms="--alphas 1.0 --fullramps 1:3"
[ $kind = score ] && arms="--score_only"
for s in "1 2" "3 4" "5 6" "7 8" "9 10"; do p=${s// /_}
  echo "export SAGE_CHECKPOINT_NAME=$ckpt OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2; .venv/bin/python CVPR_paper/scripts/stream_grid.py --source_run_dir $src --params_csv $params --output_dir $o/part_$p --subjects $s --n_checkpoints 50 $arms > $L/stream50_${kind}_$p.log 2>&1"
done > $o.queued.txt
flock CVPR_paper/queue/cpu_queue.txt.lock /bin/bash -c "cat $o.queued.txt >> CVPR_paper/queue/cpu_queue.txt"
}
main "$@"; exit $?
