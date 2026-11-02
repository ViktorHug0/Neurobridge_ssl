#!/bin/bash
# Track-1 findings on the LOSO benchmark (averaged reps, InternViT-6B L28, pairwise raw-EEG SubjectMix,
# 50 epochs, best epoch on test). One fold per call: loso_insights.sh <arm> <held-out subject>.
#   base    April recipe (tsconv_dropout_sweep_20260428-142358), replayed flag-for-flag
#   ema     base + EMA of the encoder (0.999 ~ 1000 of its ~7.3k steps)
#   labS    scratch LaBraM (63 electrode tokens), transformer optimisation: batch 504 = 56 images x 9
#           subjects (3080 memory cap), onecycle lr 1e-3, wd 0.05, clip 1, bf16, EMA 0.9995
#   labP    same as labS but the pretrained LaBraM, lr 5e-4
set -e
cd "$(dirname "$0")/../.."
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface
ARM=$1 SUB=$2
TRAIN=$(for s in 1 2 3 4 5 6 7 8 9 10; do [ "$s" -ne "$SUB" ] && echo -n "$s "; done)
BASE="--num_workers 4 --select_best_on test --softplus --num_epochs 50 --text_feature_dir '' \
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
  --img_l2norm --projector linear --feature_dim 1024 --eeg_backbone_dim 1024 --data_average --save_weights \
  --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 --seed 3300 \
  --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
TSCONV="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50 --batch_size 1024 --learning_rate 3e-4"
TF="--eeg_encoder_type LaBraM --batch_size 504 --lr_scheduler onecycle --onecycle_pct_start 0.1 --weight_decay 0.05 --grad_clip 1.0 --amp_dtype bfloat16 --ema_decay 0.9995 --ema_warmup"
case "$ARM" in
  base) EXTRA="$TSCONV" ;;
  ema)  EXTRA="$TSCONV --ema_decay 0.999 --ema_warmup" ;;
  labS) EXTRA="$TF --fm_no_pretrained --learning_rate 1e-3" ;;
  labP) EXTRA="$TF --learning_rate 5e-4" ;;
  # Track-1 conv-specific levers
  power) EXTRA="$TSCONV --tsconv_power_branch" ;;
  oc)    EXTRA="${TSCONV/--learning_rate 3e-4/--learning_rate 1e-3} --lr_scheduler onecycle --onecycle_pct_start 0.1" ;;
  # vs ensemble_arch_candidates/fast_tsconv_squeezeformer_pairwise50 (featdim 512, kernel 25): + power only
  sqfp)  EXTRA="--eeg_encoder_type OrthoFastTSConvSqueezeformer --tsconv_power_branch --batch_size 1024 --learning_rate 3e-4 --feature_dim 512" ;;
  # covariance family; 51 taps = Track 1's 25 taps at 120 Hz (~200 ms)
  fbtn)  EXTRA="--eeg_encoder_type FilterBankTangentNet --arch_kwargs '{\"taps\":51}' --batch_size 1024 --learning_rate 3e-4" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/inter-subjects/loso_insights/$ARM
mkdir -p "$OUT"
eval .venv/bin/python train.py --train_subject_ids $TRAIN --test_subject_ids $SUB \
  --output_name "$(printf sub-%02d "$SUB")" --output_dir "$OUT" --device cuda:0 $BASE $EXTRA
.venv/bin/python compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
