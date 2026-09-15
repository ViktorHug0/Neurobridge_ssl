#!/bin/bash
# Ten TinyMultiScaleTSMixer folds in three concurrent MPS packs: 4/3/3.
#SBATCH --job-name=tiny_mixer10
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-2%3
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=00:50:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-mixer-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-mixer-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_multiscale_mixer/logs/$SLURM_ARRAY_JOB_ID"

case "$SLURM_ARRAY_TASK_ID" in
  0) subjects=(1 2 3 4) ;;
  1) subjects=(5 6 7) ;;
  2) subjects=(8 9 10) ;;
  *) echo "invalid task $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

export CUDA_MPS_PIPE_DIRECTORY="/tmp/neurobridge-mps-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/neurobridge-mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
cleanup_mps() {
  echo quit | nvidia-cuda-mps-control || true
}
trap cleanup_mps EXIT

run_subject() {
  local subject=$1
  (
    export TARGET="$subject" DEVICE=cuda:0 NW=1
    bash ensemble_experiments/run_tiny_multiscale_mixer_fold.sh
  ) >"$LOG_ROOT/sub-$(printf '%02d' "$subject").log" 2>&1
}

pids=()
for subject in "${subjects[@]}"; do
  run_subject "$subject" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
exit "$status"
