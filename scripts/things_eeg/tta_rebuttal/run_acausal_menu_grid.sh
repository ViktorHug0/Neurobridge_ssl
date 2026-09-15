#!/usr/bin/env bash
# Acausal (transductive) menu-size grid, 25 refits per cell, alpha 0.5 and 1.0.
#
# menu 200 is already covered by results/.../streaming_geodesic_reps_acausal
# menu 100 is already covered by results/.../streaming_geodesic_reps_acausal_menu100
# so only menu 25 / 50 / 150 are run here. menu 25 uses reps 5 instead of reps 40,
# because 25 items x 2 blocks would leave a pool of only 50.
#
# Usage:  run_acausal_menu_grid.sh <cell> [<cell> ...]     cell = MENU:REPS
# e.g.    run_acausal_menu_grid.sh 25:5 25:10 25:20 50:10 50:20
set -u
cd /nasbrain/p20fores/Neurobridge_SSL/scripts/things_eeg/tta_rebuttal
source /nasbrain/p20fores/Neurobridge_SSL/.venv/bin/activate
R=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/tta_rebuttal/acausal_menu_grid
mkdir -p "$R"

JOBS=$(mktemp); trap 'rm -f "$JOBS"' EXIT
for CELL in "$@"; do
  MENU=${CELL%%:*}; REPS=${CELL##*:}
  POOL=$(( MENU * 80 / REPS ))
  EVERY=$(( POOL / 25 )); [ "$EVERY" -lt 1 ] && EVERY=1
  CK=$(python -c "print(' '.join(str($EVERY*(i+1)) for i in range(25)))")
  echo "menu $MENU reps $REPS -> pool $POOL, every $EVERY" >&2
  echo "python run_streaming_refit.py --block_size $REPS --menu $MENU --checkpoints $CK \
--acausal --csls --geodesic --alphas 0.5 1.0 --rho 1.0 --stream_seeds 3 \
--subjects 1 2 3 4 5 6 7 8 9 10 --device cpu \
--output_dir $R/menu${MENU}_reps${REPS} > $R/menu${MENU}_reps${REPS}.log 2>&1" >> "$JOBS"
done

echo "$(wc -l < "$JOBS") cells, 4 at a time"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  xargs -P 4 -I CMD bash -c CMD < "$JOBS"
echo GRID_DONE
