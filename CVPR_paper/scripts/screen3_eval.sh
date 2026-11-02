#!/bin/bash
# E8.5-E8.8 three-fold screens (folds 1/3/8): wait for the folds, run the SAGE cross-fold screen on them (each fold's
# params chosen on the other two), then rsa_screen_eval.py with those params -> analysis/rsa_screen/<arm>_cv3.csv.
#   usage: screen3_eval.sh <arm> [seed]   (arm = a loso/<arm> dir; atm128_f138 = the baseline's folds 1/3/8 only)
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
arm=$1 seed=${2:-3300} src=results/things_eeg/cvpr/loso/$1/seed${2:-3300}
until [ $(ls $src/*-sub-0{1,3,8}/result.csv 2>/dev/null | wc -l) -eq 3 ]; do sleep 300; done
FINAL_CKPT=checkpoint_test_best.pth bash CVPR_paper/scripts/tta_screen_loso.sh $arm $seed || return 1
P=results/things_eeg/cvpr/tta_screen_loso/${arm}_s$seed/folds_1_3_8/selected_per_subject.csv
SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth CUDA_VISIBLE_DEVICES= .venv/bin/python CVPR_paper/scripts/rsa_screen_eval.py \
  $src $P results/things_eeg/cvpr/analysis/rsa_screen/${arm}_cv3.csv 1 3 8
}
main "$@"; exit $?
