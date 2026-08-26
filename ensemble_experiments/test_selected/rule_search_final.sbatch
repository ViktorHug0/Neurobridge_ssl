#!/bin/bash
#SBATCH --job-name=ens50_rule_final
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-final-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-final-%A_%a.err
#SBATCH --array=0-1
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export OMP_NUM_THREADS=1

LEDGER16="atm_iv ge100 tsconv_eva tsconv_vith iv33 tm3 tsconv_bigg atm_vith \
vith10_group_e100 atm_vith_group_e75 atm_iv_group_e75 bigg27_group_e75 \
dino16_group_e75 iv33_group_e75 iv25_31_aux025_group_e75 iv_vith_dino_aux025_group_e75"

if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  python -u ensemble_experiments/test_selected/global_rule_search.py \
    --pool $LEDGER16 --k 2 3 4 5 --width 400 --keep 40 --processes 16 --nested \
    --output results/things_eeg/ensemble50_testselected/rule_final_ledger16.json
else
  python -u ensemble_experiments/test_selected/global_rule_search.py \
    --k 2 3 4 --width 600 --keep 60 --processes 16 --nested \
    --output results/things_eeg/ensemble50_testselected/rule_final_full66.json
fi
