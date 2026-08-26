#!/bin/bash
#SBATCH --job-name=enigma_repro
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/enigma_%j.err

# Faithful ENIGMA reproduction on THINGS-EEG2 sub-01, using the authors' own code.
#
# Target = ENIGMA (Single-Subject) row of the paper's Table 1:
#   PixCorr .1718  SSIM .4233  Alex(2) 83.64  Alex(5) 89.49  Incep 77.65  CLIP 81.48  Top-1 27.60
#
# Two fixes were needed in their tree to make it run at all:
#   - source/dataset.py __init__ called get_eeg_data() before get_metadata(), but the former indexes
#     with self.stim_indices which the latter sets -> AttributeError on ANY dataset as shipped.
#   - the Alljoined stimulus dir was hardcoded in get_metadata(); now only applied if it exists.
# Inputs come from reconstruction_experiments/enigma_adapter.py (metadata parquet + ViT-H feature .pt + rep-averaged EEG).
#
# Defaults left untouched on purpose (num_epochs 150, batch 512); the only value passed is
# retrieval_txt_loss_scale 0.25, which is what their own reconstruction.sh uses.
set -e
cd /nasbrain/p20fores/ENIGMA
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
export WANDB_MODE=offline  # wandb is imported by source/utils.py but never init'd; keep it from ever phoning home
PY=/nasbrain/p20fores/ENIGMA/.venv-enigma/bin/python
NAME=ENIGMA_repro_sub01
CFG=things_eeg2_local

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

echo "=== 1/3 train ==="
# train.py calls dist.init_process_group() unconditionally, so it only runs under a torchrun
# launcher (their reconstruction.sh uses --nproc_per_node=4). We have one GPU, so 1 rank.
# --batch_size is per-GPU (default 512); their THINGS-EEG2 script runs one process with no
# override, so 1x512 is the faithful single-subject setting -- do not scale it.
$PY -m torch.distributed.run --nproc_per_node=1 train.py \
    --model_name $NAME --subj_ids 1 --config_name $CFG --retrieval_txt_loss_scale 0.25

echo "=== 2/3 reconstruct ==="
$PY recon_inference.py --model_name $NAME --subj_ids 1 --config_name $CFG

echo "=== 3/3 evaluate ==="
$PY evaluate_recons.py --model_name $NAME --subj_ids 1 --config_name $CFG
