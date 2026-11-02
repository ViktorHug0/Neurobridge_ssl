#!/bin/bash
# Same as the original loso_fold.sh, wrapped in main() so bash parses the whole file before running: editing it
# while folds are training can no longer corrupt the running copies (bash reads scripts lazily).
main() {
# One LOSO fold of one CVPR arm (THINGS-EEG-2, averaged repetitions, InternViT-6B L28, SubjectMix pairwise a0.5,
# multi-positive grouped batches, checkpoint = lowest TEST loss as in the competitors' papers, weights saved).
#   usage: loso_fold.sh <arm> <seed> <test_subject>
# Output: results/things_eeg/cvpr/loso/<arm>/seed<seed>/<timestamp>-sub-XX (train.py skips finished folds).
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM=$1 SEED=$2 SUB=$3
TRAIN=$(for s in 1 2 3 4 5 6 7 8 9 10; do [ $s -ne $SUB ] && echo -n "$s "; done; true)  # `true`: else set -e kills fold 10
# Mainline recipe of the NeurIPS model (tsconv_dropout_sweep_20260429-190741/..._seed3300 train_config.json)
BASE="--num_epochs 50 --batch_size 1024 --learning_rate 3e-4 --data_average --init_temperature 0.07 --softplus
  --img_l2norm --multi_positive_loss --grouped_batch_sampler --samples_per_image 9
  --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5
  --projector linear --feature_dim 512 --eeg_backbone_dim 1024 --select_best_on test --save_weights
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
# ponytail: token models get the challenge's optimiser recipe (one-cycle 1e-3, wd 0.05, clip, EMA, bf16); later flags win
TOK="--learning_rate 1e-3 --lr_scheduler onecycle --weight_decay 0.05 --grad_clip 1.0
  --ema_decay 0.999 --ema_warmup --amp_dtype bfloat16"
# <arm>_vc = same recipe with ValCon selection (10% of training concepts held out, checkpoint chosen on them; atm_iv_valcon setup)
A=${ARM%_vc} VC=""
[ "$A" != "$ARM" ] && VC="--val_concept_ratio 0.1 --select_best_on val --early_stop_patience 20 --save_testsel_checkpoint"
case "$A" in
  tsconv) EXTRA="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51
            --tsconv_pool_stride 5 --tsconv_dropout 0.5" ;;
  atm)    EXTRA="--eeg_encoder_type ATM" ;;                                            # NeurIPS 77.2 arm (fd512/bb1024)
  atm128) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100" ;;  # atm_iv_valcon shape
  sqf)    EXTRA="--eeg_encoder_type OrthoFastTSConvSqueezeformer --num_epochs 75" ;;    # sqf_iv28_valcon recipe
  rmix)   EXTRA="--eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":50} $TOK" ;;
  rmixm)  EXTRA="--eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":50}" ;;  # mainline optimiser, fp32
  reve6)  EXTRA="--eeg_encoder_type REVE --fm_no_pretrained --fm_arch_kwargs {\"depth\":6}
            --batch_size 504 $TOK" ;;
  # E8.7 which image target space suits TTA (user goal 2026-10-22): ATM-128 recipe, target swapped
  atm128_cliph) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-H-14_final" ;;
  atm128_dino20) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L20_mean" ;;
  atm128_bigg27) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-bigG-14_layer27_mean" ;;
  atm128_eva35) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/EVA02-E-14_layer35_mean" ;;
  # E8.7b final vs intermediate layer of the SAME vision model (CLIP ViT-H final collapsed: final layer or CLIP?)
  atm128_iv44) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer44_mean_8bit" ;;
  atm128_iv20) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer20_mean_8bit" ;;  # E8.7c depth curve
  atm128_iv36) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer36_mean_8bit" ;;  # E8.7c depth curve
  atm128_iv39) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer39_mean_8bit" ;;  # E8.7c depth curve
  tsconv_iv44) EXTRA="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51
            --tsconv_pool_stride 5 --tsconv_dropout 0.5 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer44_mean_8bit" ;;  # E8.7d: final-layer effect on a 2nd architecture
  atm128_dino40) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L40_mean" ;;
  # E8.8 source diversity: ATM-128 on only the first N_TRAIN of the 9 source subjects
  atm128_n3|atm128_n6) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100"; N=${A#atm128_n}; TRAIN=$(echo $TRAIN | cut -d" " -f1-$N) ;;
  atm64)  EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 64 --feature_dim 64" ;;  # atm128 shape halved (backbone + alignment 64), 50 epochs (user test)
  eegnet)   EXTRA="--eeg_encoder_type EEGNet" ;;        # E1.6 grid encoders, mainline recipe (base-quality analysis)
  eegproject) EXTRA="--eeg_encoder_type EEGProject" ;;
  conformer) EXTRA="--eeg_encoder_type EEGConformer" ;;
  atm128_l31) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100
            --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer31_mean_8bit" ;;  # best layer in E2.0
  reve6m) EXTRA="--eeg_encoder_type REVE --fm_no_pretrained --fm_arch_kwargs {\"depth\":6}
            --batch_size 504 --amp_dtype bfloat16" ;;  # mainline optimiser (cf. rmixm)
  labram) export HF_HUB_OFFLINE=1  # scripts/things_eeg/inter-subject-foundation.sh LaBraM_mix recipe (AVDE contrastive + regression)
          EXTRA="--eeg_encoder_type LaBraM --alpha 0.8 --batch_size 256 --learning_rate 2e-3 --lr_scheduler cosine
            --warmup_epochs 5 --min_lr 1e-5 --weight_decay 0.05 --time_window 0 250" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/cvpr/loso/$ARM/seed$SEED
mkdir -p "$OUT"
.venv/bin/python train.py --train_subject_ids $TRAIN --test_subject_ids $SUB --output_name "$(printf sub-%02d $SUB)" \
  --output_dir "$OUT" --device cuda:0 --num_workers 4 --seed $SEED --text_feature_dir "" $BASE $EXTRA $VC
.venv/bin/python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
}
main "$@"; exit $?
