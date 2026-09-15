#!/bin/bash
# Same-seed subject-1 TinyTSConv BF16 equivalence gate against the FP32 run.
#SBATCH --job-name=tiny40_bf16_eq
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:35:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-bf16-eq-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny40-bf16-eq-%j.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
export TARGET=1 MEMBER=0 DEVICE=cuda:0 NW=${SLURM_CPUS_PER_TASK:-6}
export AMP_DTYPE=bfloat16 RUN_TAG=tiny40_bf16
export RESULT_ROOT=$REPO_ROOT/results/things_eeg/tiny_reference_pair_40e_bf16/testselected_internvit28

cd "$REPO_ROOT"
start=$SECONDS
bash ensemble_experiments/run_tiny40_reference_fold.sh
elapsed=$((SECONDS - start))

source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from ensemble_experiments.analyze_tiny_compute_ensemble import load_dump, topk

fp32 = Path('results/things_eeg/tiny_reference_pair_40e/testselected_internvit28/tiny40_tsconv_seed3300/sub-01/embeddings.npz')
bf16 = Path('results/things_eeg/tiny_reference_pair_40e_bf16/testselected_internvit28/tiny40_bf16_tsconv_seed3300/sub-01/embeddings.npz')
fp32_top1 = topk(load_dump(fp32)[0])[0]
bf16_top1 = topk(load_dump(bf16)[0])[0]
delta = bf16_top1 - fp32_top1
print(f'BF16_EQUIVALENCE fp32={fp32_top1:.2f} bf16={bf16_top1:.2f} delta={delta:+.2f}')
if delta < -2.0:
    raise SystemExit('BF16 accuracy gate failed (>2 pp below FP32)')
PY

echo "BF16_EQUIVALENCE_TIME seconds=$elapsed"
