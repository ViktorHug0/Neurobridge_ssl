#!/bin/bash
# E3.2/E3.3 + E2.2: R1 table (plain, CSLS, SCORE, SVTL, SAGE) on the existing test-selected checkpoints.
cd /nasbrain/p20fores/Neurobridge_SSL
export SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
for arm in "$@"; do
  out=results/things_eeg/cvpr/r1_competitors/$arm.csv
  [ -f $out ] && continue
  params=results/things_eeg/cvpr/tta_screen_existing_wide/$arm/selected_per_subject.csv
  [ -f $params ] || params=results/things_eeg/cvpr/tta_screen_existing/$arm/selected_per_subject.csv
  .venv/bin/python CVPR_paper/scripts/r1_eval.py --source_run_dir results/things_eeg/honest_ensemble/$arm/seed3300 \
    --params_csv $params --output_csv $out
done
