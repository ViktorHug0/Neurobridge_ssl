#!/bin/bash
# E6.1: SAGE wide-grid screen on the NEW intra-subject arms (results/things_eeg/cvpr/intra/<arm>/seed<S>, test-selected
# checkpoint_test_best.pth); cross-fold selection = the other subjects' own models. Same grid as tta_screen_intra.sh.
#   usage: tta_screen_intra_cvpr.sh <arm> [seed]
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 SAGE_CHECKPOINT_NAME=${FINAL_CKPT:-checkpoint_test_best.pth}
arm=$1 seed=${2:-2025}
out=results/things_eeg/cvpr/tta_screen_intra/${arm}_s$seed
[ -f $out/protocol_summary.csv ] && { echo "skip $out"; return 0; }
.venv/bin/python scripts/things_eeg/tta_rebuttal/run_honest_tta_selection.py \
  --source_run_dir results/things_eeg/cvpr/intra/$arm/seed$seed --output_dir $out --device cpu --num_workers 2 \
  --grid_shrink 0.5 0.6 0.7 0.8 0.85 0.9 0.94 0.97 --grid_tau 0.03 0.05 0.07 0.10 0.14 0.20 \
  --grid_steps 16 --grid_power 1.0 1.2 1.5
}
main "$@"; exit $?
