#!/bin/bash
#SBATCH --job-name=e0_ridge
#SBATCH --partition=Mee_Commun_CPU
#SBATCH --account=mee
#SBATCH --qos=commun_short
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_ridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_ridge_%j.err

# Rank every encoder against the FIXED Q-Former targets, closed form. CPU-only, so it does not
# count against the two-GPU budget. Replaces the cosine ranking, which is confounded: train.py
# learns the image projector, so at low --alpha the target collapses onto the prediction (mse00:
# cos 0.998, top-1 2.00%).
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
R=reconstruction_experiments/caption_ridge.py
FEAT=data/things_eeg/image_feature

$PY $R --enigma
# the 81.00 checkpoint, unchanged, as the reference point the caption arms were measured on
$PY $R --no_header --tag base_orig --npz data/things_eeg/captions/eeg_sub01.npz \
    --checkpoint_dir results/things_eeg/intra-subjects/TTA/20260503-143144-sub-01

for D in e0_encoders e0_encoders2 e0_encoders3; do
  for T in $(ls results/things_eeg/caption/$D 2>/dev/null); do
    NPZ=results/things_eeg/caption/$D/$T/eeg.npz
    CK=$(ls -d results/things_eeg/caption/$D/$T/*sub-01 2>/dev/null | tail -1)
    [ -f "$NPZ" ] && [ -n "$CK" ] || continue
    case $T in cat3|cat3mse) F=$FEAT/InternViT-6B_concat_25-28-31 ;;
               cat7)         F=$FEAT/InternViT-6B_concat_all7 ;;
               *)            F=$FEAT/InternViT-6B_layer28_mean_8bit ;; esac
    $PY $R --no_header --tag $T --npz $NPZ --checkpoint_dir $CK --feature_dir $F
  done
done
