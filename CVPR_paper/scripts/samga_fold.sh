#!/bin/bash
# E3.2b SCORE in its ORIGINAL stack: one LOSO fold of SAMGA (official code, /nasbrain/p20fores/SAMGA, branch
# score_repro), with SAMGA's inter.sh configuration verbatim, except:
#   --early_stop_patience 0   SCORE trains 50 epochs and reports the FINAL epoch (checkpoint_last.pth)
#   --seed                    fixed (SAMGA defaults to a time-based seed)
#   variant score             + SCORE recovery-aware training: --score_mp --score_rec_weight 0.03 (tau = 0.07, k = 10, rho = 0.1)
#   variant samga             original SAMGA objective ("Standard" rows of SCORE's Table 2)
# Waits for the InternViT layer 20/24/32/36 features (extract_internvit_layers.py, queued before the folds).
#   variant intra             SAMGA's intra.sh verbatim (EEGProject, 17 channels, 60 epochs, default early stopping on
#                             test top-1 = its own protocol); SCORE's recovery loss needs >1 training subject, so the
#                             intra "SCORE (original)" column = this model + SCORE's deployment step.
#   variant scoreg / scoregm  E3.2c diagnostics (see the EXTRA lines)
#   usage: samga_fold.sh <samga|score|scoreg|scoregm|intra> <seed> <subject>
main() {
set -e
V=$1 SEED=$2 SUB=$3
F=/nasbrain/p20fores/SAMGA/data/things_eeg/image_feature/internvit_multilevel_20_24_28_32_36
for L in 20 24 32 36; do until [ -s $F/image_test_layer$L.npy ] && [ -s $F/image_train_layer$L.npy ]; do sleep 120; done; done
TRAIN=$(for s in 1 2 3 4 5 6 7 8 9 10; do [ $s -ne $SUB ] && echo -n "$s "; done; true)
EXTRA=""
[ "$V" = score ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1"
# E3.2c reproduction diagnostics: image-grouped batches (each image x all source subjects), and averaged L_MP
[ "$V" = scoreg ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --score_grouped"
[ "$V" = scoregm ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --score_grouped --score_mp_mean"
# user hypothesis: a lower lr peaks later, so a fixed 50-epoch budget lands nearer the peak (later flags win in argparse)
[ "$V" = scorelr ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --learning_rate 5e-5 --stage2_learning_rate 2.5e-5"
# SCORE's literal temperature: tau = tau_rec = 0.07 (scale 14.3) instead of SAMGA's fixed softplus scale 2.73
[ "$V" = scoret ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --score_tau 0.07"
[ "$V" = scoretg ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --score_tau 0.07 --score_grouped"
# component ablation (SCORE Table 4 rows): multi-positive only / SAMGA objective + recovery loss only
[ "$V" = scoremp ] && EXTRA="--score_mp"
[ "$V" = scorerec ] && EXTRA="--score_rec_weight 0.03 --score_k 10 --score_rho 0.1"
# user 2026-10-22: SCORE regime cut at 30 epochs (no LR scheduler, so = epoch 30 of the 50-epoch run); Standard anchor
[ "$V" = score30 ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --num_epochs 30"
[ "$V" = samga30 ] && EXTRA="--num_epochs 30"
# E3.2d faithful-recovery variants (30 epochs): tau_rec = 0.07 on L_rec only / L_rec only in stage 2 (epochs > 20)
[ "$V" = score30rt ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --num_epochs 30 --score_rec_tau 0.07"
[ "$V" = score30s2 ] && EXTRA="--score_mp --score_rec_weight 0.03 --score_k 10 --score_rho 0.1 --num_epochs 30 --score_rec_start_epoch 20"
case "$V" in score30rt|score30s2|samga30|score30|samga|score|scoreg|scoregm|scorelr|scoret|scoretg|scoremp|scorerec|intra) ;; *) echo "unknown variant $V"; return 1 ;; esac
cd /nasbrain/p20fores/SAMGA
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$V" = intra ]; then
/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python train.py \
  --batch_size 512 --stage2_learning_rate 5e-5 --learning_rate 1e-4 --output_name $(printf "sub-%02d" $SUB) \
  --eeg_encoder_type EEGProject --train_subject_ids $SUB --test_subject_ids $SUB --softplus --num_epochs 60 \
  --image_feature_dir $F --text_feature_dir "" \
  --eeg_data_dir /nasbrain/p20fores/SAMGA/data/things_eeg/preprocessed_eeg --device cuda:0 \
  --output_dir /nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/cvpr/competitors/score_original/intra/seed$SEED \
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 --eeg_aug --eeg_aug_type smooth \
  --frozen_eeg_prior --img_l2norm --projector linear --feature_dim 512 --data_average --save_weights \
  --stage1_mmd_start 0.9 --stage1_mmd_end 0.5 --use_multilayer_router --layer_ids 20 24 28 32 36 \
  --layer_prior_center 28 --layer_prior_strength 1.0 --router_eval_mode global --seed $SEED
return
fi
/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python train.py \
  --batch_size 1024 --learning_rate 1e-4 --output_name $(printf "sub-%02d" $SUB) --eeg_encoder_type TSConv \
  --train_subject_ids $TRAIN --test_subject_ids $SUB --softplus --num_epochs 50 \
  --image_feature_dir $F --text_feature_dir "" \
  --eeg_data_dir /nasbrain/p20fores/SAMGA/data/things_eeg/preprocessed_eeg --device cuda:0 \
  --output_dir /nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/cvpr/competitors/score_original/$V/seed$SEED \
  --selected_channels --eeg_aug --eeg_aug_type smooth --frozen_eeg_prior --img_l2norm --eeg_feature_dim 1024 \
  --projector linear --feature_dim 512 --data_average --save_weights --stage1_mmd_start 0.9 --stage1_mmd_end 0.5 \
  --use_multilayer_router --layer_ids 20 24 28 32 36 --layer_prior_center 28 --layer_prior_strength 1.0 \
  --router_eval_mode global --early_stop_patience 0 --seed $SEED $EXTRA
}
main "$@"; exit $?
