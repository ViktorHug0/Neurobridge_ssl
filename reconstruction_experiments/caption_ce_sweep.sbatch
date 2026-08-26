#!/bin/bash
#SBATCH --job-name=cap_ce
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/ce_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/ce_%j.err

# Option 1: did the CE term hijack training?
#
# The ce_weight=1.0 bridge scored R^2 = -0.38 on its own training data -- worse than predicting a
# constant -- while CE fell 3.39 -> 0.47. It learned to make OPT emit fluent Qwen-style text and
# abandoned the Q-Former target geometry, which is why bridge_img tops every arm on BLEU-1 (42.02,
# above the real ceiling) yet misses on cwBLEU.
#
# ce_weight=0 is the pure-regression control: can a linear map fit Q-Former at all? Its R^2 is the
# honest ceiling for a linear bridge and decides whether Option 2 (capacity) is even the right move.
#
# Nothing expensive is recomputed: qformer_{test,training}.npy and eeg_sub01.npz already exist.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CKPT=results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01
CAP=data/things_eeg/captions
NPZ=$CAP/eeg_sub01.npz

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"
for f in $CAP/qformer_test.npy $CAP/qformer_training.npy $NPZ; do
    [ -f "$f" ] || { echo "MISSING $f -- rerun the cache stage"; exit 1; }
done

for W in 0 0.1; do
    TAG=ce${W/./p}
    BR=$CAP/bridge_sub01_$TAG.pth

    echo "=== $TAG a: train (ce_weight=$W) ==="
    $PY reconstruction_experiments/caption_bridge.py --stage train --split training --input proj --checkpoint_dir $CKPT \
        --ce_weight $W --epochs 30 --batch_size 32 --bridge $BR

    echo "=== $TAG b: image-side bridge vs the ceiling ==="
    $PY reconstruction_experiments/caption_bridge.py --stage generate --split test --input proj --checkpoint_dir $CKPT \
        --bridge $BR --out $CAP/test.bridge_img_$TAG.jsonl
    $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.bridge_img_$TAG.jsonl

    echo "=== $TAG c: EEG through that same bridge, vs the controls ==="
    $PY reconstruction_experiments/caption_bridge.py --stage generate --split test --input proj --checkpoint_dir $CKPT \
        --npz $NPZ --bridge $BR --out $CAP/test.bridge_eeg_$TAG.jsonl
    $PY reconstruction_experiments/caption_eval.py --npz $NPZ --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.bridge_eeg_$TAG.jsonl
done
