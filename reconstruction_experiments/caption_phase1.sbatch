#!/bin/bash
#SBATCH --job-name=caption_phase1
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/phase1_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/phase1_%j.err

# Cross-subject EEG->caption, retrieval-copy baseline + controls. ~10 min.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

CKPT=results/things_eeg/inter-subjects/tsconv_dropout_sweep_20260429-190741/param_k30_pool51_do050_featdim512_seed3300
DUMP=results/things_eeg/caption/feats
mkdir -p "$DUMP"

# 1) BLIP captions for the 200 test concepts (subject-independent; skip if present)
[ -f data/things_eeg/captions/test.json ] || python reconstruction_experiments/make_captions.py --split test

# 2) encode each held-out subject with its own LOSO checkpoint
for S in $(seq 1 10); do
  RUN=$(ls -d "$CKPT"/*sub-$(printf %02d "$S") | head -1)
  python evaluate.py --checkpoint_dir "$RUN" \
    --output_dir results/things_eeg/caption --output_name "eval-sub$S" \
    --eval_mode plain_cosine --test_subject_id "$S" \
    --dump_npz "$DUMP/sub$S.npz"
done

# 3) captions + controls, pooled over all 10 held-out subjects
python reconstruction_experiments/caption_eval.py --npz "$DUMP"/sub*.npz
