#!/bin/bash
# Honest checkpoint selection for the k=3 winner trio, two regimes:
#   tasks 0-2  wave 1: LOSO-val   -- 8 train / 1 val subject / 1 test
#   tasks 3-5  wave 2: 10% concept-disjoint val, all 9 training subjects kept
# Both regimes: max 100 epochs, early-stop patience 20, seed 3300, and
# --save_testsel_checkpoint so each run also yields a matched test-selected
# control on the same trajectory.  Array indices run low-first, so wave 1
# completes before wave 2 starts.
#SBATCH --job-name=ens50_valsel
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-5%3
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/honest_ensemble/slurm-valsel-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/honest_ensemble/slurm-valsel-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
FEAT="$REPO_ROOT/data/things_eeg/image_feature"

# Per-arm recipe, identical to the test-selected family except for the
# selection rule and the raised epoch ceiling.
case $((SLURM_ARRAY_TASK_ID % 3)) in
  0) BASE=atm_iv
     FEATURES="$FEAT/InternViT-6B_layer28_mean_8bit"
     export ENCODER_TYPE=ATM FEATURE_DIM=128 BACKBONE_DIM=128 MIXUP_TYPE=pairwise TKERNEL=25 ;;
  1) BASE=iv33g
     FEATURES="$FEAT/InternViT-6B_layer33_mean_8bit"
     export ENCODER_TYPE=TSConv_parameterizable FEATURE_DIM=512 BACKBONE_DIM=1024 MIXUP_TYPE=group TKERNEL=30 ;;
  2) BASE=tsconv_bigg
     FEATURES="$FEAT/ViT-bigG-14_layer27_mean"
     export ENCODER_TYPE=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024 MIXUP_TYPE=pairwise TKERNEL=25 ;;
esac

if (( SLURM_ARRAY_TASK_ID < 3 )); then
  ARM="${BASE}_valsub"
  RUNNER=ensemble_experiments/validation/run_group_target_val.sh
else
  ARM="${BASE}_valcon"
  RUNNER=ensemble_experiments/validation/run_concept_val.sh
fi

export FOLDS="1 2 3 4 5 6 7 8 9 10" RUN_SEED=3300 NW=6
export MAX_EPOCHS=100 EARLY_STOP_PATIENCE=20
echo "=== $ARM via $RUNNER ($FEATURES) ==="
bash "$RUNNER" "$ARM" "$FEATURES"

# Two dumps per run: the honest val-selected checkpoint, and the matched
# test-selected control from the same trajectory.
RUN_ROOT="results/things_eeg/honest_ensemble/$ARM/seed3300"
python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" --run-root "$RUN_ROOT" \
  --checkpoint-name checkpoint_test_best.pth \
  --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" --dump-suffix _ctl --run-root "$RUN_ROOT" \
  --checkpoint-name checkpoint_testsel_best.pth \
  --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
