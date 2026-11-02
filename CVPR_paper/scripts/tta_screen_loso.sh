#!/bin/bash
# E2.0-new: SAGE wide-grid screen on the NEW P-main LOSO arms (results/things_eeg/cvpr/loso/<arm>/seed<S>,
# checkpoint_test_best.pth = test-selected epoch). Uses only the folds that have finished (result.csv present);
# loso_fold_cv then selects on the other finished folds. Re-run when more folds land (output dir is per fold set).
#   usage: tta_screen_loso.sh <arm> [seed]
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 SAGE_CHECKPOINT_NAME=${FINAL_CKPT:-checkpoint_test_best.pth}
arm=$1 seed=${2:-3300}
src=results/things_eeg/cvpr/loso/$arm/seed$seed
subs=$(for d in $src/*-sub-*/; do [ -f $d/result.csv ] && basename $d | sed 's/.*sub-0*//'; done | sort -n | tr '\n' ' ')
tag=$(echo $subs | tr ' ' '_')
out=results/things_eeg/cvpr/tta_screen_loso/${arm}_s$seed/folds_$tag
[ -f $out/protocol_summary.csv ] && { echo "skip $out"; return 0; }
.venv/bin/python scripts/things_eeg/tta_rebuttal/run_honest_tta_selection.py --source_run_dir $src --output_dir $out \
  --subjects $subs --device cpu --num_workers 2 \
  --grid_shrink 0.5 0.6 0.7 0.8 0.85 0.9 0.94 --grid_tau 0.03 0.05 0.07 0.10 0.14 0.20 \
  --grid_steps 16 --grid_power 1.0 1.2 1.5
}
main "$@"; exit $?
