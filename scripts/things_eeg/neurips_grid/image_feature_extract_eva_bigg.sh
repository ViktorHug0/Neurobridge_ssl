#!/bin/bash
set -e

# Redirect caches away from home directory to avoid disk quota issues
export HF_HOME="/nasbrain/p20fores/.cache/huggingface"
export TORCH_HOME="/nasbrain/p20fores/.cache/torch"
mkdir -p "$HF_HOME" "$TORCH_HOME"

# python3 extract_feature.py \
#   --model_type open_clip \
#   --backbone EVA02-E-14 \
#   --pretrained laion2b_s4b_b115k \
#   --feature_source intermediate \
#   --intermediate_layer 35 \
#   --intermediate_pool mean \
#   --quantization 8bit \
#   --output_dir ./data/things_eeg/image_feature/EVA02-E-14_layer35_mean \
#   --device cuda:0

python3 extract_feature.py \
  --model_type open_clip \
  --backbone ViT-bigG-14 \
  --pretrained laion2b_s39b_b160k \
  --feature_source intermediate \
  --intermediate_layer 27 \
  --intermediate_pool mean \
  --quantization 8bit \
  --output_dir ./data/things_eeg/image_feature/ViT-bigG-14_layer27_mean \
  --device cuda:0
