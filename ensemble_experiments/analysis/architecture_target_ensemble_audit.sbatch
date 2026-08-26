#!/bin/bash
# Fixed row-z audits before/after the overnight architecture-target matrix.
#SBATCH --job-name=arch_target_audit
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=4
#SBATCH --mem=28G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-arch-target-audit-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-arch-target-audit-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

test_base=(
  atm23 atm25 atm_iv atm31 atm33 atm35
  iv23 iv25 iv28 iv31 iv33 iv35
  sqf28 inductive_v2_multiscale_ts_mixer
  temporal_conformer convnext1d ortho_mixer_group ortho_perceiver_group ortho_convgru_group
  ge100 tsconv_eva tsconv_vith tsconv_bigg atm_vith atm_iv_group_e75 iv33_group_e75
)
test_new=(sqf_bigg sqf_vith multi_eva multi_vith)

val_base=(
  atm25_valcon atm_iv_valcon atm31_valcon atm33_valcon atm35_valcon
  iv25_valcon iv28_valcon iv31_valcon iv33_valcon iv35_valcon
  iv33g_valcon tsconv_bigg_valcon
)
val_new=(
  sqf_bigg_valcon sqf_vith_valcon multi_eva_valcon multi_vith_valcon
  sqf_iv28_valcon multi_iv28_valcon
)

ctl_base=()
for arm in "${val_base[@]}"; do ctl_base+=("${arm}_ctl"); done
ctl_new=()
for arm in "${val_new[@]}"; do ctl_new+=("${arm}_ctl"); done

run_sweep() {
  local output=$1
  shift
  python ensemble_experiments/analysis/extended_roster_z_sweep.py \
    --device cuda --batch-size 128 --output "$output" --roster "$@"
}

run_sweep ensemble_experiments/analysis/target_matrix_test_base_z.json "${test_base[@]}"
run_sweep ensemble_experiments/analysis/target_matrix_test_extended_z.json "${test_base[@]}" "${test_new[@]}"
run_sweep ensemble_experiments/analysis/target_matrix_valcon_base_z.json "${val_base[@]}"
run_sweep ensemble_experiments/analysis/target_matrix_valcon_extended_z.json "${val_base[@]}" "${val_new[@]}"
run_sweep ensemble_experiments/analysis/target_matrix_ctl_base_z.json "${ctl_base[@]}"
run_sweep ensemble_experiments/analysis/target_matrix_ctl_extended_z.json "${ctl_base[@]}" "${ctl_new[@]}"
