#!/bin/bash
# E6.1 (first look): SAGE on the existing intra-subject one-cycle models (test-selected checkpoint_test_best.pth,
# 17 channels). Cross-fold selection = the other nine subjects' own models.
# usage: tta_screen_intra.sh arm [arm ...]   (CPU; srun step inside the interactive job)
cd /nasbrain/p20fores/Neurobridge_SSL
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth
for arm in "$@"; do
  out=results/things_eeg/cvpr/tta_screen_intra/$arm
  [ -f $out/protocol_summary.csv ] && { echo "skip $arm"; continue; }
  .venv/bin/python scripts/things_eeg/tta_rebuttal/run_honest_tta_selection.py \
    --source_run_dir results/things_eeg/intra-subjects/insights/$arm --output_dir $out \
    --device cpu --num_workers 2 \
    --grid_shrink 0.5 0.6 0.7 0.8 0.85 0.9 0.94 0.97 --grid_tau 0.03 0.05 0.07 0.10 0.14 0.20 \
    --grid_steps 16 --grid_power 1.0 1.2 1.5
done
