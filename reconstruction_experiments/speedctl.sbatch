#!/bin/bash
#SBATCH --job-name=speedctl
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/speedctl_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/speedctl_%j.err

# Bridge speed control only; step 1 (closed-form arms on the new metrics) completed in job 49847.
# batch 128 OOMed even on the 3090 (needed >24GiB), so this is batch 64: still 4x fewer steps
# than the 3h50 original (2x batch, 2x fewer epochs). A 3090 because step 2 needs the RAM: the CE term backprops
# through OPT-2.7b and OOMed on a 3080 at batch 32 (job 49539); this runs at batch 128.
# Step 1 would fit on a 3080 but rides along here to keep it to a single extra allocation.
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CAP=data/things_eeg/captions

echo "### step 2: bridge speed control, batch 64 / 15 epochs  ($(date))"
# The winning arm (n108: ENIGMA training-split predictions + sigma 1.08) rerun 8x cheaper per
# epoch and at half the epochs. If it lands near 10.06 the bridge stops costing 3h50 and the
# 10-subject replication becomes an overnight job instead of a two-day one.
# New filenames throughout: the 10.06 arm is the thing being compared against, do not clobber it.
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
B=reconstruction_experiments/caption_bridge.py
VITH=data/things_eeg/image_feature/ViT-H-14_final
PRED=data/things_eeg/enigma/enigma_pred_training_sub-01.npy
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01
set -e

$PY $B --stage train --split training --input vith --feat_dir $VITH --pred_npy $PRED \
    --noise 1.08 --ce_weight 0.1 --batch_size 64 --epochs 15 \
    --bridge $CAP/bridge_pred_n108_fast.pth

$PY $B --stage generate --split test --input vith --feat_dir $VITH --enigma_dir $ENIG \
    --bridge $CAP/bridge_pred_n108_fast.pth --out $CAP/test.gencap_pred_n108_fast.jsonl

echo "--- cwBLEU + permutation null"
$PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
    --hyp_captions $CAP/test.gencap_pred_n108_fast.jsonl
$PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
    --hyp $CAP/test.gencap_pred_n108_fast.jsonl

echo "--- paired against the 3h50 original (the actual pass/fail: expect no significant gap)"
$PY reconstruction_experiments/caption_bootstrap.py --n 10000 \
    --hyp $CAP/test.gencap_pred_n108_fast.jsonl --vs $CAP/test.gencap_pred_n108.jsonl
echo "### done  ($(date))"
