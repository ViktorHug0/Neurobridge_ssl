#!/bin/bash
# Within-subject EEG->image reconstruction pipeline for one subject.
#   ./run_recon_subject.sh <subject_id>
# Steps: (1) extract final ViT-H/14 features once, (2) train encoder->raw ViT-H, (3) reconstruct+score.
# Needs the GPU to itself (SDXL is ~9GB). Run when the card is free.
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
S="${1:-1}"

FEAT=data/things_eeg/image_feature/ViT-H-14_final
EEG=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
OUT=./results/things_eeg/recon
NAME="sub-$(printf %02d "$S")-vith"

# 1) ViT-H/14 final features (subject-independent; extract once)
if [ ! -f "$FEAT/image_train.npy" ]; then
  python extract_feature.py --model_type open_clip --backbone ViT-H-14 \
    --pretrained laion2b_s32b_b79k --feature_source final --output_dir "$FEAT"
fi

# 2) within-subject encoder: InfoNCE on normalized + MSE on the RAW ViT-H target (ENIGMA-faithful),
#    plus a learned EEG confidence head -- the best of the three confidence arms (top1 27% vs 4%
#    without it, and it wins 8/10 JMVR metrics). --projector direct + feature_dim 1024 keeps
#    img_projector an identity so the MSE target is the raw ViT-H direction.
#    NB: training WITHOUT --img_l2norm/--eeg_l2norm collapses the contrastive term (top1 ~1%,
#    mode-collapsed recons); and --eeg_aug_type smooth costs ~135s/epoch here, so it is off.
python train.py \
  --train_subject_ids "$S" --test_subject_ids "$S" \
  --eeg_data_dir "$EEG" --image_feature_dir "$FEAT" --text_feature_dir "" \
  --eeg_encoder_type TSConv --projector direct --feature_dim 1024 \
  --alpha 0.5 --img_l2norm --eeg_l2norm --mse_on_raw --softplus \
  --eeg_confidence_mode learned --data_average \
  --batch_size 1024 --learning_rate 1e-4 --num_epochs 50 --num_workers 4 \
  --output_dir "$OUT" --output_name "$NAME" --save_weights --seed 2025

# 3) reconstruct + score on the checkpoint just written (newest matching run dir)
#    --rescale_norm 0: keep the raw predicted magnitude (mse_on_raw learns the true CLIP scale).
CKPT=$(ls -dt "$OUT"/*"$NAME" | head -1)
echo "reconstructing from $CKPT"
python reconstruct_eval.py --checkpoint_dir "$CKPT" --output_dir "output/recon/$NAME" --rescale_norm 0
