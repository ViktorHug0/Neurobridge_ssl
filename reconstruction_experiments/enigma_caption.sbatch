#!/bin/bash
#SBATCH --job-name=enigma_cap
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_cap_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_cap_%j.err

# EEG -> image -> caption, but with the image step done by the faithful ENIGMA port (job 48496)
# instead of our own reconstruction_experiments/reconstruct_eval.py. This is the "SOTA generator" version of the arm that
# previously scored cwBLEU 4.13 (rescale 0) / 4.99 (rescale 22) against a prior floor of 5.38.
#
# Images: ENIGMA rep 0 for each of the 200 test concepts, symlinked flat in concept order.
# No --npz, so the arm's top1 column is nan -- retrieval is reported separately from
# retrieval_statistics_sub-01.csv (22.5%), not re-derived here.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CAP=data/things_eeg/captions
FLAT=data/things_eeg/enigma/flat_recon_sub01

echo "=== 1: caption ENIGMA reconstructions with BLIP-2 ==="
rm -f $CAP/test.enigma_blip2.jsonl   # make_captions resumes by concept; force a clean pass
$PY reconstruction_experiments/make_captions.py --model blip2 --split test --flat_dir $FLAT \
    --out_name test.enigma_blip2.jsonl --batch_size 4

echo "=== 2: score against Qwen references + controls ==="
$PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
    --hyp_captions $CAP/test.enigma_blip2.jsonl
