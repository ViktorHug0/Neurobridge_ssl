#!/bin/bash
#SBATCH --job-name sattc
#SBATCH --account brain
#SBATCH --qos low
#SBATCH --partition Brain_GPU
#SBATCH --gres gpu:rtx3090:1
#SBATCH -c 8
#SBATCH --mem 60G
#SBATCH --time 2-00:00:00
#SBATCH --output /nasbrain/p20fores/Neurobridge_SSL/logs/cvpr/sattc_%j.out
# E3.1: SATTC (CVPR 2026) in its own conditions, official code at /nasbrain/p20fores/SATTC-CVPR2026 (commit 0cdfd1b).
# Deviations from their README command, all logistic: no --logger (wandb stubbed in .deps), --seed 3300 for
# reproducibility, our .venv instead of their conda env, a MEMORY-ONLY loader patch (eegdatasets_leaveone.py, values
# verified identical to .orig on sub-02; their 2x-peak concat OOMs at 60 GB), ATM-style ViT-H-14 features rebuilt by
# CVPR_paper/scripts/sattc_vith_features.py (their features are not distributed). rtx3090 node: loader needs ~38 GB.
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
mkdir -p "$HOME"
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
SATTC=/nasbrain/p20fores/SATTC-CVPR2026
OUT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/cvpr/competitors/sattc/own_conditions_seed3300
set -e
$PY /nasbrain/p20fores/Neurobridge_SSL/CVPR_paper/scripts/sattc_vith_features.py \
  /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set $SATTC/features
cd $SATTC/Retrieval
( while sleep 60; do echo "[mem] $(date +%T) $(sstat -n -j ${SLURM_JOB_ID}.batch -o MaxRSS 2>/dev/null)"; done ) &
PYTHONPATH=$SATTC/.deps $PY -u run_sattc_loso.py --gpu cuda:0 --batch_size 1024 --seed 3300 --output_dir $OUT \
  --data_path /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz  # README omits it; default None breaks
