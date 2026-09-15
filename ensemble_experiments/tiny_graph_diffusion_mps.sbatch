#!/bin/bash
#SBATCH --job-name=tiny_graph
#SBATCH --partition=Brain_GPU
#SBATCH --array=0-2%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-graph-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-graph-%A_%a.err

set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly RUNNER=$REPO_ROOT/ensemble_experiments/run_tiny_graph_diffusion_fold.sh

case "$SLURM_ARRAY_TASK_ID" in
  0) readonly SUBJECTS=(1 2 3 4) ;;
  1) readonly SUBJECTS=(5 6 7) ;;
  2) readonly SUBJECTS=(8 9 10) ;;
  *) exit 2 ;;
esac

readonly LOG_ROOT=$REPO_ROOT/results/things_eeg/tiny_graph_diffusion/logs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$LOG_ROOT"

readonly MPS_ROOT=/tmp/mps-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}
export CUDA_MPS_PIPE_DIRECTORY=$MPS_ROOT/pipe
export CUDA_MPS_LOG_DIRECTORY=$MPS_ROOT/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
trap 'echo quit | nvidia-cuda-mps-control || true' EXIT

pids=()
for subject in "${SUBJECTS[@]}"; do
  TARGET=$subject DEVICE=cuda:0 NW=1 bash "$RUNNER" \
    >"$LOG_ROOT/sub-${subject}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
