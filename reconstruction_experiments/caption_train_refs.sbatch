#!/bin/bash
#SBATCH --job-name=cap_refs
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
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/refs_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/refs_%j.err

# Step 1b only: Qwen2-VL references for the 16540 training images (the bridge's CE targets).
# Split out of caption_gt.sbatch so it does not queue behind BLIP-2's download -- Qwen is already
# in HF_HOME, so this needs no network at all. Resumable: a requeue costs one concept.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
# bare `python` picks up /nasbrain/p20fores/NeuroBridge/.venv on the compute nodes
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"

$PY reconstruction_experiments/make_captions.py --model qwen --split training --images_per_concept 10
