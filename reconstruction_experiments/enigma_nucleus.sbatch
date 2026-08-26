#!/bin/bash
#SBATCH --job-name=enigma_nuc
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_nuc_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_nuc_%j.err

# Nucleus sampling in place of beam search. 16% of the sigma=2.59 EEG decodes were repetition
# loops under beams (0% at sigma=0), because a fuzzy prefix makes the argmax continuation the
# previous token. Sampling cannot lock into that. Generation only -- the bridge .pth is reused.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
B=reconstruction_experiments/caption_bridge.py
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01

for T in 1.0 0.7; do
  TG=$(echo $T | tr -d .)
  echo "=== nucleus top_p 0.9, temperature $T ==="
  $PY $B --stage generate --split test --input vith --feat_dir $VITH --enigma_dir $ENIG \
      --bridge $CAP/bridge_vith_n259.pth --do_sample --top_p 0.9 --temperature $T \
      --out $CAP/test.gencap_enigma_nuc$TG.jsonl
  $PY $B --stage generate --split test --input vith --feat_dir $VITH \
      --bridge $CAP/bridge_vith_n259.pth --do_sample --top_p 0.9 --temperature $T \
      --out $CAP/test.gencap_img_nuc$TG.jsonl
  for A in gencap_enigma_nuc$TG gencap_img_nuc$TG; do
    echo "--- $A"
    $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.$A.jsonl
    $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
        --hyp $CAP/test.$A.jsonl
  done
done
