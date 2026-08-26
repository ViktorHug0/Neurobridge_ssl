#!/bin/bash
# Four full ten-fold candidates on two GPUs: exactly 20 folds per GPU.
# The slow multi-scale bodies are paired with the lighter ranked-stem bodies.
#SBATCH --job-name=encoder_div_v3
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=24:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-encoder-div-v3-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-encoder-div-v3-%A_%a.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

case "$SLURM_ARRAY_TASK_ID" in
  0)
    ARMS=(encv3_multiscale_sqf encv3_ranked_perceiver)
    ENCODERS=(OrthoMultiScaleSqueezeformer OrthoRankedPerceiver)
    ;;
  1)
    ARMS=(encv3_multiscale_bissm encv3_ranked_dilconv)
    ENCODERS=(OrthoMultiScaleBiSSM OrthoRankedDilatedConv)
    ;;
esac

for index in 0 1; do
  export ARM="${ARMS[$index]}" ENCODER="${ENCODERS[$index]}" NW=6
  echo "[candidate] $ARM / $ENCODER"
  python ensemble_experiments/architectures/smoke.py --cuda "$ENCODER"
  bash ensemble_experiments/architectures/run_encoder_diversity_v3.sh
done
