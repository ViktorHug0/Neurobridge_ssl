#!/bin/bash
#SBATCH --job-name=ens50_rule_18
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-18-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble50_testselected/slurm-rule-18-%j.err
set -euo pipefail
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export OMP_NUM_THREADS=1

# 16-arm ledger pool + the two arms from array 49010
python -u ensemble_experiments/test_selected/global_rule_search.py \
  --pool atm_iv ge100 tsconv_eva tsconv_vith iv33 tm3 tsconv_bigg atm_vith \
         vith10_group_e100 atm_vith_group_e75 atm_iv_group_e75 bigg27_group_e75 \
         dino16_group_e75 iv33_group_e75 iv25_31_aux025_group_e75 iv_vith_dino_aux025_group_e75 \
         eva35_group_e75 iv_eva_aux025_group_e75 \
  --k 2 3 4 5 --width 400 --keep 40 --processes 16 --nested \
  --output results/things_eeg/ensemble50_testselected/rule_final_pool18.json
