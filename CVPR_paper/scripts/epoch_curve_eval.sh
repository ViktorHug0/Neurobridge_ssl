#!/bin/bash
# E8.10 TTA vs training length: for <arm>_ep (folds 1/3/8, checkpoints saved along training), run rsa_screen_eval.py on
# every checkpoint_epXXX.pth, SAGE params frozen from the base arm's 10-fold screen (<base>_s3300; chosen on its
# test-selected checkpoints). Output analysis/epoch_curve/<arm>_epXXX.csv.
#   usage: epoch_curve_eval.sh <arm_ep> <base_arm>
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
arm=$1 base=$2 src=results/things_eeg/cvpr/loso/$1/seed3300
until [ $(ls $src/*-sub-0{1,3,8}/result.csv 2>/dev/null | wc -l) -eq 3 ]; do sleep 300; done
P=results/things_eeg/cvpr/tta_screen_loso/${base}_s3300/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
mkdir -p results/things_eeg/cvpr/analysis/epoch_curve
for c in $(ls $src/*-sub-01/checkpoint_ep*.pth | xargs -n1 basename | sort); do
  e=${c#checkpoint_}; e=${e%.pth}
  SAGE_CHECKPOINT_NAME=$c CUDA_VISIBLE_DEVICES= .venv/bin/python CVPR_paper/scripts/rsa_screen_eval.py $src $P \
    results/things_eeg/cvpr/analysis/epoch_curve/${arm}_$e.csv 1 3 8
done
}
main "$@"; exit $?
