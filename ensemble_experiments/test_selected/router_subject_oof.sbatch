#!/bin/bash
# Prepared only: strict R1 source-subject-OOF data generation.
# 150 tasks = five members x ten outer folds x three inner source groups.
# Do not submit unless the R0 acceptance threshold is met.
#SBATCH --job-name=ens50_router_oof
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=6
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --array=0-149%4
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-oof-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-oof-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate

members=(
  atm_iv
  ge100
  atm_vith_group_e75
  iv33_group_e75
  dino16_group_e75
)
member_index=$((SLURM_ARRAY_TASK_ID % 5))
outer_group=$((SLURM_ARRAY_TASK_ID / 5))
outer=$((outer_group / 3 + 1))
group=$((outer_group % 3))

python ensemble_experiments/test_selected/run_router_subject_oof.py \
  --member "${members[$member_index]}" --outer "$outer" --group "$group" \
  --device cuda:0 --num-workers 6
