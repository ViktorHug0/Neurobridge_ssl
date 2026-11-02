#!/bin/bash
# One LOSO fold of one CVPR arm (THINGS-EEG-2, averaged repetitions, InternViT-6B L28, SubjectMix pairwise a0.5,
# multi-positive grouped batches, checkpoint = lowest TEST loss as in the competitors' papers, weights saved).
#   usage: loso_fold.sh <arm> <seed> <test_subject>
# Output: results/things_eeg/cvpr/loso/<arm>/seed<seed>/<timestamp>-sub-XX (train.py skips finished folds).
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM=$1 SEED=$2 SUB=$3
TRAIN=$(for s in 1 2 3 4 5 6 7 8 9 10; do [ $s -ne $SUB ] && echo -n "$s "; done)
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
case "$ARM" in
  tsconv) EXTRA="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51
            --tsconv_pool_stride 5 --tsconv_dropout 0.5" ;;
  atm)    EXTRA="--eeg_encoder_type ATM" ;;                                            # NeurIPS 77.2 arm (fd512/bb1024)
  atm128) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --num_epochs 100" ;;  # atm_iv_valcon shape
  sqf)    EXTRA="--eeg_encoder_type OrthoFastTSConvSqueezeformer --num_epochs 75" ;;    # sqf_iv28_valcon recipe
  rmix)   EXTRA="--eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":50} $TOK" ;;
  rmixm)  EXTRA="--eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":50}" ;;  # mainline optimiser, fp32
  reve6)  EXTRA="--eeg_encoder_type REVE --fm_no_pretrained --fm_arch_kwargs {\"depth\":6}
            --batch_size 504 $TOK" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/cvpr/loso/$ARM/seed$SEED
mkdir -p "$OUT"
.venv/bin/python train.py --train_subject_ids $TRAIN --test_subject_ids $SUB --output_name "$(printf sub-%02d $SUB)" \
  --output_dir "$OUT" --device cuda:0 --num_workers 4 --seed $SEED --text_feature_dir "" $BASE $EXTRA
.venv/bin/python scripts/data/compute_avg_results.py --result_dir "$OUT" --output_name inter_subject_summary.csv
