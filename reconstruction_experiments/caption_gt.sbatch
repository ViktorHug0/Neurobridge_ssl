#!/bin/bash
#SBATCH --job-name=caption_gt
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
# sl-mee-br-502's driver is broken (NVML "Driver/library version mismatch" -> torch CUDA error 804)
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/gt_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/gt_%j.err

# Steps 1-2 of the generative captioning pipeline.
#   1. Qwen2-VL-7B attribute-rich reference captions (test first, then the 16540 training images)
#   2. BLIP-2 captions of the REAL test images = the ceiling any EEG pipeline can reach
# Both scripts are resumable, so a low-qos requeue costs only the concept in flight.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
# bare `python` picks up /nasbrain/p20fores/NeuroBridge/.venv on the compute nodes (different
# checkout, torch build that 804s on the 3090s) -- always call this venv's interpreter directly
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python

# fail in seconds, not after a 16GB download, if this node's driver can't run our torch
$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

# --- fast pass first: 200 test images each, so the design is validated before the 5h run ---
$PY reconstruction_experiments/make_captions.py --model qwen  --split test --images_per_concept 1
$PY reconstruction_experiments/make_captions.py --model blip2 --split test --images_per_concept 1

echo "=== step 2: ceiling (BLIP-2 on real images, scored against Qwen references) ==="
$PY reconstruction_experiments/caption_eval.py \
  --captions     data/things_eeg/captions/test.qwen.jsonl \
  --hyp_captions data/things_eeg/captions/test.blip2.jsonl

# --- the long one: references for the 16540 bridge-training images ---
echo "=== step 1b: training-split references ==="
$PY reconstruction_experiments/make_captions.py --model qwen --split training --images_per_concept 10
