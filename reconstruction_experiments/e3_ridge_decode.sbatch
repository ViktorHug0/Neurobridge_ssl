#!/bin/bash
#SBATCH --job-name=e3_ridge
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e3_ridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e3_ridge_%j.err

# Is the CE term earning its cost? It is ~100% of a bridge's 2 h; without it the bridge is a linear
# MSE map with a closed-form solution. Solve it, decode it, score it -- no training run.
#
# This is also E3's sigma sweep for free: ridge lambda is the input-noise sigma (Bishop 1995), so
# three lambdas is three sigmas. Generation is forward-only (~5.4GB), hence a 3080, so this does
# not touch the 3090 that E2 is using.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/.cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
CAP=data/things_eeg/captions
VITH=data/things_eeg/image_feature/ViT-H-14_final
ENIG=/nasbrain/p20fores/ENIGMA/output/ENIGMA_repro_sub01/reconstructions_sub-01

echo "=== solve ridge on the ViT-H space, dump three lambdas ==="
$PY reconstruction_experiments/caption_ridge.py --enigma \
    --save $CAP/ridge_vith --save_lams 10 100 1000

for L in 10 100 1000; do
  echo "=== decode ENIGMA test predictions, lambda=$L ==="
  $PY reconstruction_experiments/caption_bridge.py --stage generate --split test \
      --input vith --feat_dir $VITH --enigma_dir $ENIG \
      --bridge $CAP/ridge_vith_lam$L.pth --out $CAP/test.ridge_lam$L.jsonl
  $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
      --hyp_captions $CAP/test.ridge_lam$L.jsonl
  $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
      --hyp $CAP/test.ridge_lam$L.jsonl
done
