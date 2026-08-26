#!/bin/bash
# Prepared only: do not submit until the router protocol is approved.
# Fifty tasks = five fixed members x ten outer folds.  Each task exports the
# nine source-subject held-out-stimulus dumps for one outer/member pair.
#SBATCH --job-name=ens50_router_dump
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-49%4
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-dump-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-router-dump-%A_%a.err
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
outer=$((SLURM_ARRAY_TASK_ID / 5 + 1))
member=${members[$member_index]}

python ensemble_experiments/test_selected/dump_router_source_scores.py \
  --arms "$member" --outers "$outer" --device cuda:0 --num-workers 0
