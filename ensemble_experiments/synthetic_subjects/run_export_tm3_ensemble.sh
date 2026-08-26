#!/bin/bash
# Export complete temporal-SubjectMix checkpoints for fixed zero-shot ensembles.
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
CHECKPOINT_ROOT=$PWD/results/things_eeg/hullaug/tm3/seed3300
DUMP_ROOT=$PWD/results/things_eeg/synthetic_subjects/ensemble_screen/dumps
EVAL_ROOT=$PWD/results/things_eeg/synthetic_subjects/ensemble_screen/eval_tm3
mkdir -p "$DUMP_ROOT" "$EVAL_ROOT"

for held in $(seq 1 10); do
  tag=$(printf '%02d' "$held")
  checkpoint_dir=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "*-sub-$tag" | sort | tail -1)
  if [ -z "$checkpoint_dir" ] || [ ! -f "$checkpoint_dir/checkpoint_test_best.pth" ]; then
    echo "missing checkpoint for subject $tag" >&2
    exit 1
  fi
  python evaluate.py \
    --checkpoint_dir "$checkpoint_dir" \
    --output_dir "$EVAL_ROOT" --output_name "tm3-sub$tag" \
    --eval_mode plain_cosine --test_subject_id "$held" \
    --device cuda:0 --num_workers 0 \
    --dump_npz "$DUMP_ROOT/tm3-sub$tag.npz"
done

python ensemble_experiments/synthetic_subjects/score_npz_ensemble.py \
  --dump-dir "$DUMP_ROOT" --members group pair tm3 \
  --subjects $(seq 1 10) \
  --output results/things_eeg/synthetic_subjects/ensemble_screen/group_pair_tm3_ensemble.csv
