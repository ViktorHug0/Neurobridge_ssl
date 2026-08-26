#!/bin/bash
#SBATCH --job-name=e0_diag
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_diag_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_diag_%j.err

# E0 (embedding diagnostic) + E1 step 1 (does ENIGMA's train-split cosine match its test cosine?).
# If train cos ~= test cos = 0.36, in-sample optimism and the 4-rep vs 80-rep gap cancel, and E1
# can train the bridge on these predictions directly with no cross-fitting.
set -e
REPO=/nasbrain/p20fores/Neurobridge_SSL
PRED=$REPO/reconstruction_experiments/enigma_predict.py

# their code resolves configs/ relative to cwd and needs their pinned torch 2.4.1 venv
cd /nasbrain/p20fores/ENIGMA
echo "=== ENIGMA predictions: test (fast, validates the ordering assert) ==="
.venv-enigma/bin/python $PRED --split test
echo "=== ENIGMA predictions: training (16540 rows) ==="
.venv-enigma/bin/python $PRED --split training

cd $REPO
export PYTHONPATH=$REPO
echo "=== E0 diagnostic ==="
.venv/bin/python reconstruction_experiments/caption_diag.py --enigma
.venv/bin/python reconstruction_experiments/caption_diag.py \
    --npz data/things_eeg/captions/eeg_sub01.npz \
    --checkpoint_dir results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01
