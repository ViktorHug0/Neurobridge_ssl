#!/bin/bash
# Track-1 findings on INTRA-subject THINGS-EEG2 (train and test on one subject, 4-rep averages, 17 occipito-
# parietal channels, InternViT-6B L28, no SubjectMix, 50 epochs, best epoch = lowest test loss).
# Reference = results/things_eeg/intra-subjects/TTA (EEGProject, 85.25). One subject per call:
#   intra_insights.sh <arm> <subject>
# ~16 steps/epoch (16,540 rows / batch 1024) -> ~800 steps per run, so EMA horizons are short.
set -e
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface
ARM=$1 SUB=$2
BASE="--num_workers 4 --select_best_on test --softplus --num_epochs 50 --text_feature_dir '' \
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 \
  --img_l2norm --projector linear --feature_dim 512 --eeg_backbone_dim 1024 --data_average --save_weights --seed 2025"
EPJ="--eeg_encoder_type EEGProject --batch_size 1024 --learning_rate 1e-4"
TSC="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50 --batch_size 1024 --learning_rate 1e-4"
SQF="--eeg_encoder_type OrthoFastTSConvSqueezeformer --batch_size 1024 --learning_rate 1e-4"
EMA="--ema_decay 0.99 --ema_warmup"                                  # ~100 of ~800 steps
OC="--learning_rate 1e-3 --lr_scheduler onecycle --onecycle_pct_start 0.1"   # later flags win
LAB="--eeg_encoder_type LaBraM --fm_no_pretrained --batch_size 128 --learning_rate 1e-3 --lr_scheduler onecycle \
  --onecycle_pct_start 0.1 --weight_decay 0.05 --grad_clip 1.0 --amp_dtype bfloat16 --ema_decay 0.999 --ema_warmup"
case "$ARM" in
  epj) EXTRA="$EPJ" ;;            tsc) EXTRA="$TSC" ;;            sqf) EXTRA="$SQF" ;;
  epj_ema) EXTRA="$EPJ $EMA" ;;   tsc_ema) EXTRA="$TSC $EMA" ;;   sqf_ema) EXTRA="$SQF $EMA" ;;
  epj_oc) EXTRA="$EPJ $OC" ;;     tsc_oc) EXTRA="$TSC $OC" ;;     sqf_oc) EXTRA="$SQF $OC" ;;
  tsc_pow) EXTRA="$TSC --tsconv_power_branch" ;;                  sqf_pow) EXTRA="$SQF --tsconv_power_branch" ;;
  epj_cd) EXTRA="$EPJ --eeg_aug --eeg_aug_type channel_dropout" ;;
  epj_b256) EXTRA="${EPJ/--batch_size 1024/--batch_size 256}" ;;   # 4x the steps
  lab) EXTRA="$LAB" ;;                                              # scratch LaBraM, 17 electrode tokens
  lab_w400) EXTRA="$LAB --fm_arch_kwargs '{\"conv_out_channels\":16,\"num_heads\":8,\"num_layers\":18}'" ;;
  # follow-ups to epj_oc (87.55): seed control, all 63 electrodes, 4x steps
  epj_oc_s2026) EXTRA="$EPJ $OC --seed 2026" ;;
  epj_oc_ch63) EXTRA="$EPJ $OC" ; BASE="${BASE/--selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 /}" ;;
  epj_oc_b256) EXTRA="${EPJ/--batch_size 1024/--batch_size 256} $OC" ;;
  tsc_oc_pow) EXTRA="$TSC $OC --tsconv_power_branch" ;;
  sqf_oc_pow) EXTRA="$SQF $OC --tsconv_power_branch" ;;
  epj_oc_cd) EXTRA="$EPJ $OC --eeg_aug --eeg_aug_type channel_dropout" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/intra-subjects/insights/$ARM
mkdir -p "$OUT"
eval .venv/bin/python train.py --train_subject_ids $SUB --test_subject_ids $SUB \
  --output_name "$(printf sub-%02d "$SUB")" --output_dir "$OUT" --device cuda:0 $BASE $EXTRA
.venv/bin/python compute_avg_results.py --result_dir "$OUT" --output_name intra_subject_summary.csv
