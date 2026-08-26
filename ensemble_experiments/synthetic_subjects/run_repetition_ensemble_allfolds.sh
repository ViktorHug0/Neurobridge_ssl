#!/bin/bash
# Fixed all-fold evaluation of the subject-1-screened intermediate repetition groups.
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
OUT=results/things_eeg/synthetic_subjects/repetition_ensemble_allfolds
GROUP=results/things_eeg/subjectmix_rebuttal/beta_k9_paperbase/seed3300
PAIR=results/things_eeg/subjectmix_rebuttal/method_subjectmix/seed3300
mkdir -p "$OUT"

for subject in $(seq 1 10); do
  tag=$(printf '%02d' "$subject")
  for model in group pair; do
    if [ "$model" = group ]; then root=$GROUP; else root=$PAIR; fi
    checkpoint=$(find "$root" -maxdepth 1 -type d -name "*-sub-$tag" -print -quit)
    test -n "$checkpoint"
    python ensemble_experiments/synthetic_subjects/evaluate_repetition_ensemble.py \
      --checkpoint-dir "$checkpoint" --test-subject-id "$subject" \
      --group-sizes 16 20 40 80 --num-partitions 4 --partition-seed 20260813 \
      --batch-size 200 --device cuda:0 \
      --output-csv "$OUT/$model-sub$tag.csv" \
      --output-npz "$OUT/$model-sub$tag.npz"
  done
done

python - <<'PY'
from pathlib import Path
import numpy as np
import pandas as pd

root = Path('results/things_eeg/synthetic_subjects/repetition_ensemble_allfolds')
rows = []
for subject in range(1, 11):
    group = np.load(root / f'group-sub{subject:02d}.npz')
    pair = np.load(root / f'pair-sub{subject:02d}.npz')
    for key in sorted(set(group.files).intersection(pair.files)):
        if not key.startswith('k'):
            continue
        target = np.arange(group[key].shape[0])
        row = {'subject': subject, 'score_key': key}
        for name, scores in (
            ('group', group[key]),
            ('pair', pair[key]),
            ('ensemble', (group[key] + pair[key]) / 2.0),
        ):
            row[f'{name}_top1'] = np.mean(scores.argmax(axis=1) == target) * 100.0
            top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
            row[f'{name}_top5'] = np.mean(np.any(top5 == target[:, None], axis=1)) * 100.0
        rows.append(row)
frame = pd.DataFrame(rows)
means = frame.groupby('score_key').mean(numeric_only=True).reset_index()
frame.to_csv(root / 'per_fold.csv', index=False)
means.to_csv(root / 'mean_by_group.csv', index=False)
print(means.to_string(index=False))
PY
