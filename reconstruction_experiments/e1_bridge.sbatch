#!/bin/bash
#SBATCH --job-name=e1_bridge
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
# 3080s have 9.76GiB; the CE term backprops through OPT-2.7b and needs ~11GB at batch 32 (job
# 49539 OOMed). Keep the batch size so the arm stays comparable to the 7.09 baseline.
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e1_bridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e1_bridge_%j.err

# E1: fit the bridge on ENIGMA's own predicted embeddings instead of on real image embeddings
# corrupted by Gaussian noise. sigma matches the error's MAGNITUDE (cos 0.36) but not its geometry;
# real predictions carry both.
#
# E0 settled step 1 and the answer was no: ENIGMA's cosine is 0.531 on the training split against
# 0.360 on test, so in-sample optimism does NOT cancel the 4-rep vs 80-rep gap. Two arms:
#   raw   train on the predictions as they are. Deliberately mismatched (0.531 vs 0.360); measures
#         how much the geometry alone is worth, and how much the residual gap costs.
#   n108  add sigma 1.08 on top, since 0.531/sqrt(1+1.08^2) = 0.360. Real geometry AND the test
#         cosine, without the five-fold cross-fit this would otherwise need.
# Baseline to beat: 7.09, real images with sigma 2.59.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
B=reconstruction_experiments/caption_bridge.py
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
PRED=data/things_eeg/enigma/enigma_pred_training_sub-01.npy
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01

for ARM in raw:0 n108:1.08; do
  TAG=${ARM%%:*}; SIG=${ARM##*:}
  echo "=== train bridge on ENIGMA predictions, extra sigma=$SIG ($TAG) ==="
  $PY $B --stage train --split training --input vith --feat_dir $VITH --pred_npy $PRED \
      --noise $SIG --ce_weight 0.1 --bridge $CAP/bridge_pred_$TAG.pth

  echo "=== generate: ENIGMA test predictions (the EEG arm) ==="
  $PY $B --stage generate --split test --input vith --feat_dir $VITH --enigma_dir $ENIG \
      --bridge $CAP/bridge_pred_$TAG.pth --out $CAP/test.gencap_pred_$TAG.jsonl

  echo "=== generate: real test images through the same bridge (its ceiling) ==="
  $PY $B --stage generate --split test --input vith --feat_dir $VITH \
      --bridge $CAP/bridge_pred_$TAG.pth --out $CAP/test.gencap_predimg_$TAG.jsonl

  for A in gencap_pred_$TAG gencap_predimg_$TAG; do
    echo "--- $A"
    $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.$A.jsonl
    $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
        --hyp $CAP/test.$A.jsonl
  done
done
