#!/bin/bash
#SBATCH --job-name=e0_enc
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=03:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_enc_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/e0_enc_%j.err

# E0 encoder screen. An intra-subject encoder trains in 2.5 min against a bridge's 2 h, so rank
# candidates on the diagnostic (cos / effective rank / norm CV) and pay for a bridge only once.
#
# Two hypotheses, both cheap:
#   --alpha <1 --mse_on_raw  adds a reconstruction term on the UNnormalized target. The baseline is
#                            alpha=1.0, which skips the MSE branch entirely (loss.py:174), so it is
#                            pure InfoNCE and discards magnitude via --img_l2norm. This is what
#                            ENIGMA does and what MindEye's second head is for.
#   concat feature dirs      multi-granularity image target, already built.
# --alpha 0.9 without --mse_on_raw is the control that isolates raw-vs-normalized MSE.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
FEAT=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature
OUT=results/things_eeg/caption/e0_encoders
CH="P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2"
mkdir -p $OUT

# baseline args copied from the 81.00 checkpoint's train.log; only alpha/mse_on_raw/features vary
train () {  # $1 = tag, $2 = feature dir, rest = extra flags
  local tag=$1 feat=$2; shift 2
  echo "=== train $tag ==="
  $PY train.py --output_dir $OUT/$tag --output_name sub-01 \
      --num_epochs 50 --batch_size 1024 --learning_rate 0.0001 \
      --train_subject_ids 1 --test_subject_ids 1 --data_average --softplus --img_l2norm \
      --init_temperature 0.07 --samples_per_image 4 --select_best_on test --save_weights \
      --eeg_encoder_type EEGProject --projector linear --feature_dim 512 --eeg_backbone_dim 1024 \
      --selected_channels $CH --time_window 0 250 --seed 2025 --text_feature_dir "" \
      --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
      --image_feature_dir $feat "$@"
}

train base      $FEAT/InternViT-6B_layer28_mean_8bit
train mse09     $FEAT/InternViT-6B_layer28_mean_8bit --alpha 0.9 --mse_on_raw
train mse07     $FEAT/InternViT-6B_layer28_mean_8bit --alpha 0.7 --mse_on_raw
train mse05     $FEAT/InternViT-6B_layer28_mean_8bit --alpha 0.5 --mse_on_raw
train mse09norm $FEAT/InternViT-6B_layer28_mean_8bit --alpha 0.9
train cat3      $FEAT/InternViT-6B_concat_25-28-31
train cat3mse   $FEAT/InternViT-6B_concat_25-28-31 --alpha 0.9 --mse_on_raw
train cat7      $FEAT/InternViT-6B_concat_all7

echo "=== diagnostics ==="
for tag in base mse09 mse07 mse05 mse09norm cat3 cat3mse cat7; do
  case $tag in cat3|cat3mse) F=$FEAT/InternViT-6B_concat_25-28-31 ;;
               cat7)         F=$FEAT/InternViT-6B_concat_all7 ;;
               *)            F=$FEAT/InternViT-6B_layer28_mean_8bit ;; esac
  # train.py timestamps its run dir, so resolve it rather than guessing the name
  CK=$(ls -d $OUT/$tag/*sub-01 2>/dev/null | tail -1)
  [ -z "$CK" ] && { echo "$tag: no checkpoint"; continue; }
  $PY evaluate.py --checkpoint_dir $CK --output_dir $OUT/$tag --output_name eval \
      --eval_mode plain_cosine --test_subject_id 1 --dump_npz $OUT/$tag/eeg.npz >/dev/null
  $PY reconstruction_experiments/caption_diag.py --npz $OUT/$tag/eeg.npz \
      --checkpoint_dir $CK --feature_dir $F --name $tag
done
