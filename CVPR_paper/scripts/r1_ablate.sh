#!/bin/bash
# E2.4-E2.6/E3.6 driver: r1_ablate.py on ValCon test-selected arms. usage: r1_ablate.sh [--reps_sweep] arm...
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
export SAGE_CHECKPOINT_NAME=checkpoint_testsel_best.pth OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
flag=""; tag=ablate
[ "$1" = --reps_sweep ] && { flag=--reps_sweep; tag=reps; shift; }
for arm in "$@"; do
  out=results/things_eeg/cvpr/r1_ablate/${arm}_$tag.csv
  [ -f $out ] && continue
  params=results/things_eeg/cvpr/tta_screen_existing_wide/$arm/selected_per_subject.csv
  .venv/bin/python CVPR_paper/scripts/r1_ablate.py --source_run_dir results/things_eeg/honest_ensemble/$arm/seed3300 \
    --params_csv $params --output_csv $out $flag
done
}
main "$@"; exit $?
