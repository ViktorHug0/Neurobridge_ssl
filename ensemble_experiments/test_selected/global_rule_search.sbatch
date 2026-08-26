#!/bin/bash
# CPU-only: search one globally fixed query-local fusion rule for k=2,3,4 over
# the frozen score-dump pool, plus a leave-one-fold-out honesty pass.
#SBATCH --job-name=ens50_rule_search
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-search-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-search-%j.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
cd "$REPO_ROOT"
source .venv/bin/activate
export OMP_NUM_THREADS=1

python -u ensemble_experiments/test_selected/global_rule_search.py \
  --k 2 3 4 \
  --width 600 \
  --keep 60 \
  --processes 16 \
  --nested \
  --output results/things_eeg/ensemble50_testselected/global_rule_search.json
