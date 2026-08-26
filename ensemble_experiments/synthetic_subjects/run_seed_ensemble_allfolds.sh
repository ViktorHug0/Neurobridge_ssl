#!/bin/bash
# Export five independently seeded pairwise SubjectMix families and combine them
# with the fixed stochastic-group and paper-base pairwise members.
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

DUMPS=results/things_eeg/synthetic_subjects/ensemble_screen/dumps
EVAL=results/things_eeg/synthetic_subjects/ensemble_screen/eval_seed_members
SESSION=results/things_eeg/inter-subjects/mixup_20260421-190931
mkdir -p "$DUMPS" "$EVAL"

for seed in 3298 3299 3300 3301 3302; do
  root="$SESSION/mix_raw_eeg_pairwise_linear_a0p5_seed$seed"
  for subject in $(seq 1 10); do
    tag=$(printf '%02d' "$subject")
    dump="$DUMPS/p$seed-sub$tag.npz"
    [ -f "$dump" ] && continue
    checkpoint=$(find "$root" -maxdepth 1 -type d -name "*-sub-$tag" -print -quit)
    test -n "$checkpoint"
    python evaluate.py --checkpoint_dir "$checkpoint" --output_dir "$EVAL" \
      --output_name "p$seed-sub$tag" --eval_mode plain_cosine \
      --test_subject_id "$subject" --device cpu --batch_size 32 --num_workers 0 \
      --dump_npz "$dump"
  done
done

python ensemble_experiments/synthetic_subjects/score_npz_ensemble.py \
  --dump-dir "$DUMPS" --members group pair p3298 p3299 p3300 p3301 p3302 \
  --output results/things_eeg/synthetic_subjects/ensemble_screen/group_pair_seed_ensemble.csv
