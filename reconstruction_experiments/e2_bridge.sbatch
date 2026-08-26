#!/bin/bash
#SBATCH --job-name=e2_bridge
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e2_bridge_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e2_bridge_%j.err

# E2: turn the encoder screen's winner into a real cwBLEU, and calibrate the screen while doing it.
#
# The ridge screen ranks encoders by cos against the FIXED Q-Former targets: base 0.238, mse03
# 0.251 (--alpha 0.3, best of 12 variants), ENIGMA 0.271. But that proxy has exactly ONE clean
# anchor to cwBLEU -- ENIGMA's 7.09. base's old 6.51 predates the noise augmentation AND the
# decoding fix, so it cannot calibrate anything.
#
# Both arms below therefore run under identical modern settings, which buys two things: whether
# --alpha 0.3 beats plain InfoNCE end to end, and a second (base, mse03) anchor for the proxy.
# sigma per arm from its own measured cosine, cos ~ 1/sqrt(1+sigma^2):
#   base  cos 0.346 -> 2.71     mse03 cos 0.426 -> 2.12
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
B=reconstruction_experiments/caption_bridge.py
CAP=data/things_eeg/captions
E=results/things_eeg/caption

for ARM in "base:2.71:$E/e0_encoders/base" "mse03:2.12:$E/e0_encoders2/mse03"; do
  TAG=${ARM%%:*}; REST=${ARM#*:}; SIG=${REST%%:*}; DIR=${REST#*:}
  CK=$(ls -d $DIR/*sub-01 | tail -1)
  echo "=== $TAG: bridge on the 512-d projected space, sigma=$SIG, ckpt $CK ==="
  $PY $B --stage train --split training --input proj --checkpoint_dir $CK \
      --noise $SIG --ce_weight 0.1 --bridge $CAP/bridge_$TAG.pth

  echo "=== $TAG: EEG arm ==="
  $PY $B --stage generate --split test --input proj --checkpoint_dir $CK \
      --npz $DIR/eeg.npz --bridge $CAP/bridge_$TAG.pth --out $CAP/test.e2_$TAG.jsonl
  echo "=== $TAG: real images through the same bridge (its ceiling) ==="
  $PY $B --stage generate --split test --input proj --checkpoint_dir $CK \
      --bridge $CAP/bridge_$TAG.pth --out $CAP/test.e2_${TAG}_img.jsonl

  for A in e2_$TAG e2_${TAG}_img; do
    echo "--- $A"
    $PY reconstruction_experiments/caption_eval.py --captions $CAP/test.qwen.jsonl \
        --hyp_captions $CAP/test.$A.jsonl
    $PY reconstruction_experiments/caption_bootstrap.py --mode permute --n 1000 \
        --hyp $CAP/test.$A.jsonl
  done
done
