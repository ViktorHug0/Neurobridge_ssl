#!/bin/bash
#SBATCH --job-name=enigma_bridge
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_bridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_bridge_%j.err

# GENERATIVE captioning from ENIGMA's predicted CLIP embedding -- a text decoder in place of SDXL.
#
# The bridge maps a unit-norm ViT-H embedding to BLIP-2 Q-Former tokens, decoded by the frozen
# OPT-2.7b. Trained on REAL ViT-H image embeddings (targets already cached in qformer_training.npy),
# then fed ENIGMA's predictions at test time.
#
# The whole experiment turns on --noise. ENIGMA's predicted embedding has cos 0.36 with the true
# image embedding; a bridge fit on clean embeddings is evaluated far off its training distribution,
# which is how the earlier InternViT bridge failed. cos ~ 1/sqrt(1+sigma^2) gives sigma 2.59 as the
# matched corruption. sigma 0 is the control that isolates whether the augmentation is what matters.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
B=reconstruction_experiments/caption_bridge.py
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01

for SIG in 2.59 0; do
  TAG=$(echo $SIG | tr -d .)
  echo "=== training bridge, noise sigma=$SIG ==="
  $PY $B --stage train --split training --input vith --feat_dir $VITH \
      --noise $SIG --ce_weight 0.1 --bridge $CAP/bridge_vith_n$TAG.pth

  echo "=== generate: ENIGMA predicted embeddings (the EEG arm) ==="
  $PY $B --stage generate --split test --input vith --feat_dir $VITH \
      --enigma_dir $ENIG --bridge $CAP/bridge_vith_n$TAG.pth \
      --out $CAP/test.gencap_enigma_n$TAG.jsonl

  echo "=== generate: real test images through the same bridge (this arm's ceiling) ==="
  $PY $B --stage generate --split test --input vith --feat_dir $VITH \
      --bridge $CAP/bridge_vith_n$TAG.pth --out $CAP/test.gencap_img_n$TAG.jsonl

  for A in gencap_enigma_n$TAG gencap_img_n$TAG; do
    echo "--- $A"
    $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.$A.jsonl
    $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
        --hyp $CAP/test.$A.jsonl
  done
done
