#!/bin/bash
#SBATCH --job-name=cap_eeg
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/eeg_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/eeg_%j.err

# Step 4: swap the frozen encoder's EEG embedding into the slot the image embedding occupied.
# No retraining -- the bridge from step 3 is reused as-is, which is the whole point: if the shared
# space is really shared, an EEG vector should decode like the image vector it was aligned to.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CKPT=results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01
NPZ=data/things_eeg/captions/eeg_sub01.npz

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== step 4a: encode subject 01's test EEG into the 512-d shared space ==="
$PY evaluate.py --checkpoint_dir $CKPT --output_dir results/things_eeg/caption/eval_sub01 \
    --output_name sub01 --eval_mode plain_cosine --test_subject_id 1 --dump_npz $NPZ

echo "=== step 4b: EEG -> caption through the image-trained bridge ==="
$PY reconstruction_experiments/caption_bridge.py --stage generate --split test --input proj --checkpoint_dir $CKPT \
    --npz $NPZ --bridge data/things_eeg/captions/bridge_sub01.pth \
    --out data/things_eeg/captions/test.bridge_eeg.jsonl

echo "=== step 4c: against the controls (retrieval / prior / random / shuffle) ==="
# the 'ceiling' row here is our EEG bridge; 'prior' is the EEG-blind floor it must clear
$PY reconstruction_experiments/caption_eval.py --npz $NPZ --captions data/things_eeg/captions/test.qwen.jsonl \
    --hyp_captions data/things_eeg/captions/test.bridge_eeg.jsonl
