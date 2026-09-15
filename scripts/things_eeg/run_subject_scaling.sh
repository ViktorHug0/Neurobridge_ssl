#!/bin/bash
# Subject-count scaling curve: does held-out (LOSO) accuracy saturate at 9 training subjects?
#
# For each held-out test subject, train on N in {1,2,4,8} randomly drawn training subjects.
# Epochs are scaled 50*9/N so every point gets the SAME number of gradient steps -- otherwise
# low-N points are handicapped by compute, not by subject diversity, and the curve lies.
# The N=9 anchor is free: tsconv_dropout_sweep .../param_k30_pool51_do050_featdim512_seed3300.
#
# Config = current-best base model (featdim512, seed3300). Resumable: skips any run with result.csv.
# Usage: ./run_subject_scaling.sh          (env: TEST_SUBJECTS, N_VALUES, DRAWS, EPOCHS_BASE)
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

OUT="${OUT:-./results/things_eeg/inter-subjects/subject_scaling}"
# 4 test subjects spanning the N=9 accuracy range (25.0 / 29.5 / 37.0 / 48.0); their mean at
# N=9 is 34.9 vs the 10-subject average 35.9, so the subset is representative.
TEST_SUBJECTS="${TEST_SUBJECTS:-8 5 9 10}"
N_VALUES="${N_VALUES:-1 2 4 8}"
DRAWS="${DRAWS:-2}"
EPOCHS_BASE="${EPOCHS_BASE:-50}"
mkdir -p "$OUT"

BASE=(
  --eeg_encoder_type TSConv_parameterizable
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit
  --text_feature_dir ''
  --projector linear --feature_dim 512 --eeg_backbone_dim 1024
  --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_pool_stride 5
  --tsconv_dropout 0.5 --tsconv_activation elu
  --time_window 0 250 --data_average --img_l2norm --softplus
  --multi_positive_loss --grouped_batch_sampler --samples_per_image 9
  --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 --mixup_type pairwise
  --batch_size 1024 --learning_rate 3e-4 --num_workers 4
  --eval_mode plain_cosine --select_best_on test --seed 3300 --device "${DEVICE:-cuda:0}"
)

# Distinct subject subsets per (test subject, N). C(9,N) >= 9 > DRAWS for every N used here.
draw_subsets() {
  python - "$1" "$2" "$3" <<'PY'
import random, sys
s, n, d = map(int, sys.argv[1:4])
pool = [i for i in range(1, 11) if i != s]
r, seen = random.Random(s * 100 + n), []
while len(seen) < d:
    c = tuple(sorted(r.sample(pool, n)))
    if c not in seen:
        seen.append(c)
print('\n'.join(' '.join(map(str, c)) for c in seen))
PY
}

for S in $TEST_SUBJECTS; do
  for N in $N_VALUES; do
    EPOCHS=$(( EPOCHS_BASE * 9 / N )); [ "$EPOCHS" -lt 1 ] && EPOCHS=1
    D=0
    while read -r TRAIN_IDS; do
      NAME="$(printf 'sub-%02d' "$S")_n${N}_d${D}"
      D=$((D + 1))
      if compgen -G "${OUT}/*-${NAME}/result.csv" > /dev/null; then
        echo "[scaling] $NAME done, skipping"; continue
      fi
      echo "[scaling] === $NAME | train on [$TRAIN_IDS] | ${EPOCHS} epochs === $(date +%H:%M:%S)"
      python train.py "${BASE[@]}" --num_epochs "$EPOCHS" \
        --train_subject_ids $TRAIN_IDS --test_subject_ids "$S" \
        --output_dir "$OUT" --output_name "$NAME"
    done < <(draw_subsets "$S" "$N" "$DRAWS")
  done
done
