#!/bin/bash
# Four independent BF16 TinyTSConv processes on one 10 GB RTX 3080.
#SBATCH --job-name=tiny_bf16_x4
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH --time=00:20:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-bf16-x4-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tiny-bf16-x4-%j.err
set -euo pipefail

readonly REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
readonly ROOT=$REPO_ROOT/results/things_eeg/tiny_bf16_pack4_smoke/$SLURM_JOB_ID
mkdir -p "$ROOT"
cd "$REPO_ROOT"
source .venv/bin/activate

export CUDA_MPS_PIPE_DIRECTORY="/tmp/neurobridge-mps-$SLURM_JOB_ID"
export CUDA_MPS_LOG_DIRECTORY="/tmp/neurobridge-mps-log-$SLURM_JOB_ID"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d

nvidia-smi \
  --query-gpu=timestamp,memory.used,utilization.gpu,power.draw \
  --format=csv -l 1 >"$ROOT/nvidia-smi.csv" &
monitor_pid=$!

cleanup() {
  kill "$monitor_pid" 2>/dev/null || true
  echo quit | nvidia-cuda-mps-control || true
}
trap cleanup EXIT

run_member() {
  local index=$1
  local seed=$((7300 + index))
  python train.py \
    --batch_size 1024 \
    --num_workers 1 \
    --learning_rate 3e-4 \
    --output_name model \
    --train_subject_ids 2 3 4 5 6 7 8 9 10 \
    --test_subject_ids 1 \
    --select_best_on test \
    --softplus \
    --num_epochs 2 \
    --image_feature_dir "$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit" \
    --text_feature_dir "" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
    --device cuda:0 \
    --amp_dtype bfloat16 \
    --output_dir "$ROOT/member-$index" \
    --img_l2norm \
    --projector linear \
    --feature_dim 128 \
    --data_average \
    --seed "$seed" \
    --multi_positive_loss \
    --grouped_batch_sampler \
    --samples_per_image 9 \
    --subject_mixup_mode raw_eeg \
    --mixup_type pairwise \
    --subject_mixup_alpha 0.5 \
    --eeg_encoder_type TSConv_parameterizable \
    --eeg_backbone_dim 256 \
    --tsconv_temporal_filters 10 \
    --tsconv_temporal_kernel 25 \
    --tsconv_pool_kernel 51 \
    --tsconv_pool_stride 5 \
    --tsconv_spatial_filters 10 \
    --tsconv_projection_filters 10 \
    >"$ROOT/member-$index.log" 2>&1
}

pids=()
for index in 0 1 2 3; do
  run_member "$index" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [ "$status" -ne 0 ]; then
  exit "$status"
fi

python - "$ROOT" <<'PY'
from pathlib import Path
import pandas as pd
import sys

root = Path(sys.argv[1])
rows = []
for result in sorted(root.glob('member-*/*-model/result.csv')):
    row = pd.read_csv(result).iloc[0]
    rows.append((float(row['peak cuda allocated gb']), float(row['peak cuda reserved gb'])))
if len(rows) != 4:
    raise SystemExit(f'expected four result files, found {len(rows)}')
print(f'BF16_PACK4_OK members=4 max_process_allocated_gb={max(x[0] for x in rows):.3f} max_process_reserved_gb={max(x[1] for x in rows):.3f}')
PY
