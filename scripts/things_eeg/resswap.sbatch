#!/bin/bash
#SBATCH --job-name=resswap
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-tp-br-513
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/hullaug/rs_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/hullaug/rs_%j.err
set -e
S=/nasbrain/p20fores/apr_code_aug/run_hullaug.sh
OUTROOT=/tmp/rssmoke EPOCHS=2 $S rs 1
echo "=== smoke ok, starting real arms ==="
$S rs   1 3 5   # replaces SubjectMix outright
$S rsk9 1 3 5   # stacked on beta_k9
$S rs15 1 3 5   # exaggerated borrowed residual
$S rs05 1 3 5
echo "RESSWAP DONE $(date)"
