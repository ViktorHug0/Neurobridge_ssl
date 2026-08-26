#!/bin/bash
#SBATCH --job-name=e4_rpred
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e4_rpred_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e4_rpred_%j.err

# Does E1's noise effect survive a solver that cannot overfit?
#
# Trained bridges: fit on images, sigma 0 -> 4.59, sigma 2.59 -> 7.09. Read as noise matching
# ENIGMA's error geometry. But closed-form ridge on the SAME images at lambda 0.1 (sigma ~0.32,
# i.e. no noise) scores 7.50, matching the sigma 2.59 trained bridge. So most of that gap may be
# SGD plus the CE term overfitting, which noise regularised, rather than a distribution match.
#
# Two arms, both closed-form so neither can overfit:
#   images  fit on real ViT-H image embeddings          (the E1 baseline condition)
#   pred    fit on ENIGMA's training-split predictions  (the E1 treatment)
# If pred still beats images, E1's finding is about the error geometry and survives. If they tie,
# the trained-bridge comparison was measuring optimisation, not distribution.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01
PRED=data/things_eeg/enigma/enigma_pred_training_sub-01.npy

$PY reconstruction_experiments/caption_ridge.py --enigma --pred_npy $PRED \
    --save $CAP/ridge_pred --save_lams 0.1 1 10

for L in 0.1 1 10; do
  echo "=== decode, bridge fit on PREDICTIONS, lambda=$L ==="
  $PY reconstruction_experiments/caption_bridge.py --stage generate --split test \
      --input vith --feat_dir $VITH --enigma_dir $ENIG \
      --bridge $CAP/ridge_pred_lam$L.pth --out $CAP/test.ridgepred_lam$L.jsonl
  $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
      --hyp_captions $CAP/test.ridgepred_lam$L.jsonl
done
