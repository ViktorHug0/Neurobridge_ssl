#!/bin/bash
# One MPS pack of independent TinySqueezeformer LOSO folds. TINYSQ_PACK maps
# to subjects: 0=(1,2,3), 1=(4,5,6), 2=(7,8), 3=(9,10).
#SBATCH --job-name=tiny_sqf_mps
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=00:50:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-sqf-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-sqf-%A_%a.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly PACK_ID=${TINYSQ_PACK:-${SLURM_ARRAY_TASK_ID:-0}}
readonly LOG_ROOT="$REPO_ROOT/results/things_eeg/tiny_squeezeformer/logs/$SLURM_JOB_ID"

case "$PACK_ID" in
  0) subjects=(1 2 3) ;;
  1) subjects=(4 5 6) ;;
  2) subjects=(7 8) ;;
  3) subjects=(9 10) ;;
  *) echo "invalid TINYSQ_PACK=$PACK_ID" >&2; exit 2 ;;
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
    export TARGET="$subject" DEVICE=cuda:0 NW=2
    bash ensemble_experiments/run_tiny_squeezeformer_fold.sh
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

