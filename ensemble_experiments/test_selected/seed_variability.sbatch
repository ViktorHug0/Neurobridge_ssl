#!/bin/bash
# Seed variability for the three members behind the k=2 and k=3 winners.
# Each task reruns one arm's own protocol unchanged at the requested seed, all ten folds.
#   task 0  atm_iv          ATM, bb/fd 128,  InternViT-28, pairwise mix, 50 ep, k25
#   task 1  iv33_group_e75  TSConv_param,    InternViT-33, group mix,    75 ep, k30
#   task 2  tsconv_bigg     TSConv,          ViT-bigG-27,  pairwise mix, 50 ep, k25
# SEED defaults to 3301 and can be overridden at submission time. Dumps use a
# seed-specific suffix so existing exports stay untouched.
#SBATCH --job-name=ens50_seedvar
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=16:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-seedvar-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-seedvar-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; print(torch.cuda.get_device_name(0))"

SEED=${SEED:-3301}
FEAT="$REPO_ROOT/data/things_eeg/image_feature"

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ARM=atm_iv
    FEATURES="$FEAT/InternViT-6B_layer28_mean_8bit"
    export ENCODER=ATM FEATURE_DIM=128 BACKBONE_DIM=128
    export MIXUP_TYPE=pairwise TKERNEL=25 NUM_EPOCHS=50 NW=4
    ;;
  1)
    ARM=iv33_group_e75
    FEATURES="$FEAT/InternViT-6B_layer33_mean_8bit"
    export ENCODER=TSConv_parameterizable FEATURE_DIM=512 BACKBONE_DIM=1024
    export MIXUP_TYPE=group TKERNEL=30 NUM_EPOCHS=75 NW=6
    ;;
  2)
    ARM=tsconv_bigg
    FEATURES="$FEAT/ViT-bigG-14_layer27_mean"
    export ENCODER=TSConv FEATURE_DIM=512 BACKBONE_DIM=1024
    export MIXUP_TYPE=pairwise TKERNEL=25 NUM_EPOCHS=50 NW=4
    ;;
  *) echo "unexpected task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

export FOLDS="1 2 3 4 5 6 7 8 9 10" RUN_SEED="$SEED"
bash ensemble_experiments/test_selected/run_group_target_testselected.sh "$ARM" "$FEATURES"

python ensemble_experiments/test_selected/dump_testselected_arm.py \
  --arm "$ARM" --dump-suffix "_seed$SEED" \
  --run-root "results/things_eeg/ensemble50_testselected/$ARM/seed$SEED" \
  --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
