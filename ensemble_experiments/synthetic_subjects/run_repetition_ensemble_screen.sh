#!/bin/bash
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

OUTPUT_ROOT=results/things_eeg/synthetic_subjects/repetition_ensemble_screen
GROUP_ROOT=results/things_eeg/subjectmix_rebuttal/beta_k9_paperbase/seed3300
PAIR_ROOT=results/things_eeg/subjectmix_rebuttal/method_subjectmix/seed3300
mkdir -p "$OUTPUT_ROOT"

if [ "${PER_QUERY_BN:-0}" = 1 ]; then
  mode_tag=perquerybn
  mode_args=(--per-query-bn)
else
  mode_tag=evalbn
  mode_args=()
fi
mode_tag=${TAG:-$mode_tag}
read -r -a group_sizes <<< "${GROUP_SIZES:-4 80}"

# Fast gate: k=4 exactly matches the train-time repetition average, while k=80
# must reproduce ordinary evaluation.  Expand only if the gate beats k=80.
for model in group pair; do
  if [ "$model" = group ]; then root=$GROUP_ROOT; else root=$PAIR_ROOT; fi
  checkpoint_dir=$(find "$root" -maxdepth 1 -type d -name '*-sub-01' -print -quit)
  python ensemble_experiments/synthetic_subjects/evaluate_repetition_ensemble.py \
    --checkpoint-dir "$checkpoint_dir" \
    --test-subject-id 1 \
    --group-sizes "${group_sizes[@]}" \
    --num-partitions 4 \
    --partition-seed 20260813 \
    --batch-size 200 \
    "${mode_args[@]}" \
    --device cuda:0 \
    --output-csv "$OUTPUT_ROOT/${model}-sub01-${mode_tag}.csv" \
    --output-npz "$OUTPUT_ROOT/${model}-sub01-${mode_tag}.npz"
done

python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('results/things_eeg/synthetic_subjects/repetition_ensemble_screen')
environment = __import__('os').environ
mode_tag = environment.get('TAG') or ('perquerybn' if environment.get('PER_QUERY_BN') == '1' else 'evalbn')
group = np.load(root / f'group-sub01-{mode_tag}.npz')
pair = np.load(root / f'pair-sub01-{mode_tag}.npz')
rows = []
for key in sorted(set(group.files).intersection(pair.files)):
    if not key.startswith('k'):
        continue
    scores = (group[key] + pair[key]) / 2.0
    target = np.arange(len(scores))
    top1 = np.mean(scores.argmax(axis=1) == target) * 100.0
    top5_idx = np.argpartition(scores, -5, axis=1)[:, -5:]
    top5 = np.mean(np.any(top5_idx == target[:, None], axis=1)) * 100.0
    rows.append({'score_key': key, 'top1': top1, 'top5': top5})
frame = pd.DataFrame(rows)
frame.to_csv(root / f'group-pair-sub01-{mode_tag}.csv', index=False)
print(frame.to_string(index=False))
PY
