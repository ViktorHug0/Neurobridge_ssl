#!/bin/bash
set -euo pipefail
REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

DUMP_ROOT=results/things_eeg/synthetic_subjects/ensemble_screen/dumps
EVAL_ROOT=results/things_eeg/synthetic_subjects/ensemble_screen/eval
GROUP_ROOT=results/things_eeg/subjectmix_rebuttal/beta_k9_paperbase/seed3300
PAIR_ROOT=results/things_eeg/subjectmix_rebuttal/method_subjectmix/seed3300
mkdir -p "$DUMP_ROOT" "$EVAL_ROOT"

for subject in $(seq 1 10); do
  tag=$(printf '%02d' "$subject")
  for model in group pair; do
    dump="$DUMP_ROOT/$model-sub$tag.npz"
    [ -f "$dump" ] && continue
    if [ "$model" = group ]; then root="$GROUP_ROOT"; else root="$PAIR_ROOT"; fi
    checkpoint_dir=$(find "$root" -maxdepth 1 -type d -name "*-sub-$tag" -print -quit)
    if [ -z "$checkpoint_dir" ]; then
      echo "Missing $model checkpoint for subject $tag" >&2
      exit 2
    fi
    python evaluate.py --checkpoint_dir "$checkpoint_dir" \
      --output_dir "$EVAL_ROOT" --output_name "$model-sub$tag" \
      --eval_mode plain_cosine --test_subject_id "$subject" --device cpu \
      --batch_size 32 --num_workers 0 --dump_npz "$dump"
  done
done

python ensemble_experiments/synthetic_subjects/score_npz_ensemble.py \
  --dump-dir "$DUMP_ROOT" --members group pair \
  --output results/things_eeg/synthetic_subjects/ensemble_screen/group_pair_allfolds.csv
