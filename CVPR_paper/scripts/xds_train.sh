#!/bin/bash
# One LOSO fold of the paper's ATM recipe (atm128_vc: ATM, 128-d, InternViT L28 targets, SubjectMix, 10 % of the
# training concepts held out for checkpoint selection) on Alljoined-1.6M (20 people, 32 channels, 120 Hz) or THINGS-MEG
# (4 people, 271 channels, 200 samples). 50 epochs at most, early stopping after 5 epochs without a better validation
# loss (user, 2026-11-15). Extra train.py flags override the recipe.
#   usage: xds_train.sh <alljoined|meg> <test subject> <output tag> [train.py flags]
cd /nasbrain/p20fores/Neurobridge_SSL
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home
mkdir -p "$HOME"
DS=$1; S=$2; TAG=$3; shift 3
case "$DS" in
  alljoined) N=20; DATA=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench; TW="0 120"; SPI=9; OUT=results/alljoined/cvpr
             IMG=data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit ;;  # the stimuli are the THINGS-EEG2 images, same index
  meg)       N=4; DATA=data/things_meg/preprocessed_meg; TW="0 200"; SPI=3; OUT=results/things_meg/cvpr
             IMG=data/things_meg/image_feature/InternViT-6B_layer28_mean_8bit_meg ;;
  *) echo "unknown dataset $DS"; exit 1 ;;
esac
TRAIN=$(seq 1 $N | grep -vx "$S" | tr '\n' ' ')
.venv/bin/python train.py --train_subject_ids $TRAIN --test_subject_ids $S --output_name "$(printf 'sub-%02d' $S)" --output_dir $OUT/$TAG \
  --eeg_data_dir $DATA --time_window $TW --image_feature_dir $IMG --text_feature_dir "" \
  --eeg_encoder_type ATM --projector linear --feature_dim 128 --eeg_backbone_dim 128 \
  --num_epochs 50 --early_stop_patience 5 --learning_rate 3e-4 --num_workers 6 --data_average --softplus --img_l2norm \
  --multi_positive_loss --grouped_batch_sampler --samples_per_image $SPI --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 \
  --val_concept_ratio 0.1 --save_testsel_checkpoint --select_best_on val --save_weights --seed 3300 "$@"
