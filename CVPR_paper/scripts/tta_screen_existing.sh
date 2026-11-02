#!/bin/bash
# E2.0: SAGE on the test-selected checkpoints that already exist (honest_ensemble/*_valcon keep
# checkpoint_testsel_best.pth). Hyper-parameters: loso_fold_cv (cross-fold) column of the summary.
# usage: tta_screen_existing.sh arm [arm ...]     (CPU; run as an srun step inside the interactive job)
cd /nasbrain/p20fores/Neurobridge_SSL
export OMP_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 MKL_NUM_THREADS=5 SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth
for arm in "$@"; do
  out=results/things_eeg/cvpr/tta_screen_existing/$arm
  [ -f $out/protocol_summary.csv ] && { echo "skip $arm"; continue; }
  .venv/bin/python scripts/things_eeg/tta_rebuttal/run_honest_tta_selection.py \
    --source_run_dir results/things_eeg/honest_ensemble/$arm/seed3300 --output_dir $out \
    --device cpu --num_workers 2 \
    --grid_shrink 0.85 0.90 0.94 0.97 0.99 --grid_tau 0.03 0.05 0.07 0.10 0.14 0.20 \
    --grid_steps 16 --grid_power 1.0 1.2 1.5
done
