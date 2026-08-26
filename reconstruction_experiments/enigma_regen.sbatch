#!/bin/bash
#SBATCH --job-name=enigma_regen
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_regen_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_regen_%j.err

# Re-decode the sigma=2.59 bridge with repetition_penalty / no_repeat_ngram_size. The first pass
# collapsed into "the the the ..." on BOTH the EEG arm and the real-image ceiling, so those numbers
# measured a decoding artifact. Training is unchanged -- the bridge .pth is reused as-is.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
B=reconstruction_experiments/caption_bridge.py
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01

$PY $B --stage generate --split test --input vith --feat_dir $VITH --enigma_dir $ENIG \
    --bridge $CAP/bridge_vith_n259.pth --out $CAP/test.gencap_enigma_n259.jsonl
$PY $B --stage generate --split test --input vith --feat_dir $VITH \
    --bridge $CAP/bridge_vith_n259.pth --out $CAP/test.gencap_img_n259.jsonl

for A in gencap_enigma_n259 gencap_img_n259; do
  echo "--- $A"
  $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
      --hyp_captions $CAP/test.$A.jsonl
  $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
      --hyp $CAP/test.$A.jsonl
done
