#!/bin/bash
#SBATCH --job-name=m2_rescore
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=06:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/m2_rescore_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/m2_rescore_%j.err

# Re-score every arm on the new metric set against the neutral 5-reference set.
# Needs CLIP ViT-H (~4GB) + RoBERTa-large (~1.4GB), so a 3080 is enough.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python

# ordered worst-to-best on the old cwBLEU, so any inversion under the new metrics is visible
ARMS="test.gencap_enigma_n0.jsonl test.enigma_blip2.jsonl test.gencap_enigma_nuc07.jsonl \
      test.gencap_enigma_n259.jsonl test.nn_eeg_enigma.jsonl test.gencap_pred_raw.jsonl \
      test.gencap_pred_n108.jsonl test.gencap_img_n259.jsonl test.gencap_img_n0.jsonl \
      test.blip2.jsonl"
for A in test.e2_base.jsonl test.e2_mse03.jsonl; do
  [ -f data/things_eeg/captions/$A ] && ARMS="$ARMS $A"
done

$PY reconstruction_experiments/caption_metrics.py --arms $ARMS --n 1000 --n_bert 50
