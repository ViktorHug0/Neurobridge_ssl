#!/bin/bash
# Concept-val (10% held-out training CONCEPTS) 2x5 InternViT depth grid,
# family-A recipe: pairwise SubjectMix, temporal kernel 25, all channels,
# seed 3300, max 75 epochs, early stop patience 20.
#
# Only 9 of the 10 cells are here: ATM@28 already exists as `atm_iv_valcon`
# (same recipe, 100-epoch cap).  Re-selecting that run under a 75 cap costs
# -0.90pp, p=0.242 -- inside the seed SE -- so it is reused rather than redone.
#
# Each task writes two dumps: the honest val-selected checkpoint, and a matched
# test-selected control (checkpoint_testsel_best.pth) from the same trajectory.
#SBATCH --job-name=ens50_valcon_grid
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-8%1
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=16:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/honest_ensemble/slurm-vcgrid-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/honest_ensemble/slurm-vcgrid-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
FEAT="$REPO_ROOT/data/things_eeg/image_feature"

# ATM first (~10.2 h/arm) so the long poles start earliest; TSConv is ~4 h.
case $SLURM_ARRAY_TASK_ID in
  0) LAYER=25; export ENCODER_TYPE=ATM    FEATURE_DIM=128 BACKBONE_DIM=128  ;;
  1) LAYER=31; export ENCODER_TYPE=ATM    FEATURE_DIM=128 BACKBONE_DIM=128  ;;
  2) LAYER=33; export ENCODER_TYPE=ATM    FEATURE_DIM=128 BACKBONE_DIM=128  ;;
  3) LAYER=35; export ENCODER_TYPE=ATM    FEATURE_DIM=128 BACKBONE_DIM=128  ;;
  4) LAYER=25; export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 ;;
  5) LAYER=28; export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 ;;
  6) LAYER=31; export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 ;;
  7) LAYER=33; export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 ;;
  8) LAYER=35; export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 ;;
esac
PREFIX=$([ "$ENCODER_TYPE" = ATM ] && echo atm || echo iv)
ARM="${PREFIX}${LAYER}_valcon"

export MIXUP_TYPE=pairwise TKERNEL=25
export MAX_EPOCHS=75 EARLY_STOP_PATIENCE=20 VAL_CONCEPT_RATIO=0.10
export FOLDS="1 2 3 4 5 6 7 8 9 10" RUN_SEED=3300 NW=6

echo "=== $ARM (InternViT layer $LAYER, $ENCODER_TYPE, concept-val, max 75ep) ==="
bash ensemble_experiments/validation/run_concept_val.sh "$ARM" "$FEAT/InternViT-6B_layer${LAYER}_mean_8bit"

RUN_ROOT="results/things_eeg/honest_ensemble/$ARM/seed3300"
python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" --run-root "$RUN_ROOT" \
  --checkpoint-name checkpoint_test_best.pth \
  --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" --dump-suffix _ctl --run-root "$RUN_ROOT" \
  --checkpoint-name checkpoint_testsel_best.pth \
  --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
