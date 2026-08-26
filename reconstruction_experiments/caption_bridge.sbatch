#!/bin/bash
#SBATCH --job-name=cap_bridge
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/bridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/bridge_%j.err

# Steps 2-3. Depends on the training references (47318), and deliberately holds only one GPU:
# submitted with --dependency=afterok so it never runs alongside the reference job.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CKPT=results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== step 2: ceiling (BLIP-2 on the real test images vs the Qwen references) ==="
$PY reconstruction_experiments/make_captions.py --model blip2 --split test --images_per_concept 1
$PY reconstruction_experiments/caption_eval.py --captions data/things_eeg/captions/test.qwen.jsonl \
                    --hyp_captions data/things_eeg/captions/test.blip2.jsonl

echo "=== step 3a: Q-Former targets ==="
$PY reconstruction_experiments/caption_bridge.py --stage cache --split test     --per_concept 1  --input raw --batch_size 32
$PY reconstruction_experiments/caption_bridge.py --stage cache --split training --per_concept 10 --input raw --batch_size 32

echo "=== step 3b: bridge on images only (512-d shared space, subject 01) ==="
$PY reconstruction_experiments/caption_bridge.py --stage train --split training --input proj --checkpoint_dir $CKPT \
    --ce_weight 1.0 --epochs 30 --batch_size 32 --bridge data/things_eeg/captions/bridge_sub01.pth

echo "=== step 3c: does the image-side bridge reach the ceiling? ==="
$PY reconstruction_experiments/caption_bridge.py --stage generate --split test --input proj --checkpoint_dir $CKPT \
    --bridge data/things_eeg/captions/bridge_sub01.pth \
    --out data/things_eeg/captions/test.bridge_img.jsonl
$PY reconstruction_experiments/caption_eval.py --captions data/things_eeg/captions/test.qwen.jsonl \
                    --hyp_captions data/things_eeg/captions/test.bridge_img.jsonl
