#!/bin/bash
#SBATCH --job-name=atmts_b512
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts-b512-%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-atmts-b512-%j.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

python -m ensemble_experiments.decorrelated_models.train_twins \
  --arm atmts_batch512_smoke \
  --output-dir results/things_eeg/decorrelated_models/batch_size_smoke/b512_sub01_e3 \
  --dump-root results/things_eeg/decorrelated_models/batch_size_smoke/dumps \
  --held-subject 1 \
  --train-subject-ids 2 3 4 5 6 7 8 9 10 \
  --eeg-data-dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
  --image-feature-dir data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
  --device cuda:0 \
  --num-workers 6 \
  --batch-size 512 \
  --num-epochs 3 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --encoder-a ATM \
  --backbone-dim-a 128 \
  --feature-dim-a 128 \
  --encoder-b TSConv_parameterizable \
  --backbone-dim-b 1024 \
  --feature-dim-b 512 \
  --lambda-div 0.05 \
  --beta-ensemble 0 \
  --decorrelation-start-epoch 11 \
  --seed-a 3300 \
  --seed-b 3301 \
  --train-rng-seed 7330 \
  --mixup-alpha 0.5
