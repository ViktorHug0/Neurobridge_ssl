#!/bin/bash
# E8.4 encoder study: R1 battery for one P-main LOSO arm (all 10 folds) — cross-fold SAGE screen, then every
# label-free method on R1 (CSLS / SCORE / SVTL / SAGE, SATTC, Hungarian / Sinkhorn / TransCLIP / EM) and the
# geometry diagnostics. No R2-R4 (regimes and streams are for the main methods only).
#   usage: encoder_battery.sh <arm> [seed]      (cpu_queue item; waits for the 10 folds)
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
arm=$1 seed=${2:-3300}
src=results/things_eeg/cvpr/loso/$arm/seed$seed
until [ $(ls $src/*-sub-*/result.csv 2>/dev/null | wc -l) -eq 10 ]; do sleep 300; done
FINAL_CKPT=checkpoint_test_best.pth bash CVPR_paper/scripts/tta_screen_loso.sh $arm $seed || return 1
params=results/things_eeg/cvpr/tta_screen_loso/${arm}_s$seed/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
t=${arm}_s$seed o=results/things_eeg/cvpr/final/$t L=logs/cvpr/final/$t
mkdir -p $o $L
env="export SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth;"
py=".venv/bin/python CVPR_paper/scripts"
common="--source_run_dir $src --params_csv $params"
{
  echo "$env $py/r1_eval.py $common --output_csv $o/r1_competitors.csv > $L/r1_eval.log 2>&1"
  echo "$env PYTHONPATH=/nasbrain/p20fores/SATTC-CVPR2026/.deps $py/sattc_on_embeddings.py ours --run_dir $src --output_csv $o/sattc.csv > $L/sattc.log 2>&1"
  echo "$env CUDA_VISIBLE_DEVICES= $py/r1_transductive.py $common --cache_dir results/things_eeg/cvpr/emb_cache/${t}_b --output_csv $o/r1_transductive.csv > $L/r1_transductive.log 2>&1"
  echo "$env $py/geometry.py $common --cache_dir results/things_eeg/cvpr/emb_cache/$t --output_csv $o/geometry.csv > $L/geometry.log 2>&1"
} > $o/queued.txt
flock CVPR_paper/queue/cpu_queue.txt.lock /bin/bash -c "cat $o/queued.txt >> CVPR_paper/queue/cpu_queue.txt"
}
main "$@"; exit $?
