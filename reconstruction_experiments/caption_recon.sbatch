#!/bin/bash
#SBATCH --job-name=cap_recon
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/recon_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/recon_%j.err

# EEG -> image -> caption, as the baseline for the generative route.
#
# Encoder: sub-01 nbaug (top1 30.5 / best 27.5 in raw CLIP ViT-H space) -- the best of the six recon
# checkpoints. It is a `learned` confidence model, so --rescale_norm stays 0 ("use raw predicted
# magnitude"); forcing a norm is only for fully-normalized models.
# Generator: SDXL-Turbo + IP-Adapter (ip-adapter_sdxl_vit-h), which consumes ViT-H image embeddings
# directly -- the reason the encoder targets ViT-H-14_final.
#
# First run downloads ~8GB (sdxl-turbo, IP-Adapter, sdxl-vae) into HF_HOME.
# NB: skipping --lowlevel_decoder on purpose. It trades semantics for pixel fidelity, and captions
# are scored on semantics.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CKPT=results/things_eeg/recon/20260706-134431-sub-01-nbaug
CAP=data/things_eeg/captions
# RESCALE=0 keeps the model's own magnitude; 22 forces the real ViT-H norm. Measured: the encoder
# predicts ||.||=28.61+-0.10 while real ViT-H is 22.27+-1.15 -- 28% too large AND effectively
# constant, so the `learned` confidence head is not modulating anything and IP-Adapter (which is
# scale-sensitive) sees off-distribution embeddings.
RESCALE=${RESCALE:-0}
GEN=results/things_eeg/recon/gen_sub01${TAG:-}

$PY -c "import torch, diffusers; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0), '| diffusers', diffusers.__version__)"

echo "=== 1: EEG -> image (SDXL-Turbo + IP-Adapter), 200 test concepts ==="
$PY reconstruction_experiments/reconstruct_eval.py --checkpoint_dir $CKPT --output_dir $GEN \
    --image_set_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set \
    --n_images 200 --rescale_norm $RESCALE --hf_cache $HF_HOME

echo "=== 2: caption the generated images with BLIP-2 ==="
rm -f $CAP/test.recon_blip2${TAG:-}.jsonl   # make_captions resumes by concept; force a clean pass
$PY reconstruction_experiments/make_captions.py --model blip2 --split test --flat_dir $GEN \
    --out_name test.recon_blip2${TAG:-}.jsonl --batch_size 4

echo "=== 3: score against the Qwen references and the controls ==="
$PY reconstruction_experiments/caption_eval.py --npz $CAP/recon_sub01.npz --captions $CAP/test.qwen.jsonl \
    --hyp_captions $CAP/test.recon_blip2${TAG:-}.jsonl

echo "=== 4: style control -- BLIP-2 on the REAL test images (already computed, for reference) ==="
$PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl --hyp_captions $CAP/test.blip2.jsonl
