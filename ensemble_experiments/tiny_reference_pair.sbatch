#!/bin/bash
# Complete one fixed TinyTSConv/TinyATM reference pair over LOSO subjects 4--10.
# Subjects 1--3 already exist from the cancelled Tiny-10 experiment.
#SBATCH --job-name=tiny_ref_pair
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=3-9%3
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-ref-pair-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-ref-pair-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly TARGET=$((SLURM_ARRAY_TASK_ID + 1))
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_compute_ensemble/reference_pair_logs/$SLURM_ARRAY_JOB_ID"

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

run_member() {
  local member=$1
  local family=$2
  (
    export TARGET
    export MEMBER="$member"
    export DEVICE=cuda:0
    export NW=3
    bash ensemble_experiments/run_tiny_compute_ensemble_fold.sh
  ) >"$LOG_ROOT/sub-$(printf '%02d' "$TARGET")-${family}.log" 2>&1
}

# Member 0 is TinyTSConv seed 3300; member 5 is TinyATM seed 4300.
run_member 0 tiny_tsconv &
pid_ts=$!
run_member 5 tiny_atm &
pid_atm=$!

status=0
wait "$pid_ts" || status=$?
wait "$pid_atm" || status=$?
exit "$status"
