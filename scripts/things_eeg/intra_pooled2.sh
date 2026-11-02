#!/bin/bash
# POOLED intra-subject protocol (report separately from per-subject intra): ONE model trained on all 10
# subjects' training data, tested on each subject's 200 held-out images. Each subject is scored every epoch
# exactly as its own single-subject run would be (per_subject_test.csv), and its best epoch is picked
# post hoc by ITS lowest test loss: scripts/things_eeg/pooled_select.py <run_dir>.
# Same data/targets/channels/eval as intra_insights.sh. usage: intra_pooled.sh <arm>
set -e
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface
ARM=$1
SUBS="1 2 3 4 5 6 7 8 9 10"
BASE="--num_workers 4 --select_best_on test --softplus --num_epochs 50 --text_feature_dir '' \
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2 \
  --img_l2norm --projector linear --feature_dim 512 --eeg_backbone_dim 1024 --data_average --save_weights --seed 2025 \
  --eval_unique_gallery --per_subject_test_log"
EPJ="--eeg_encoder_type EEGProject --batch_size 1024 --learning_rate 1e-4"
TSC="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.50 --batch_size 1024 --learning_rate 1e-4"
# Track-1 LaBraM recipe; pooled data gives ~1.3k steps/epoch (~65k per run), so Track-1's EMA 0.9998 applies
LAB="--eeg_encoder_type LaBraM --fm_no_pretrained --batch_size 128 --learning_rate 1e-3 --lr_scheduler onecycle \
  --onecycle_pct_start 0.1 --weight_decay 0.05 --grad_clip 1.0 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup"
case "$ARM" in
  epj) EXTRA="$EPJ" ;;
  tsc) EXTRA="$TSC" ;;
  lab) EXTRA="$LAB" ;;
  epj_oc) EXTRA="$EPJ --learning_rate 1e-3 --lr_scheduler onecycle --onecycle_pct_start 0.1" ;;
  tsc_oc) EXTRA="$TSC --learning_rate 1e-3 --lr_scheduler onecycle --onecycle_pct_start 0.1" ;;
  lab_w400) EXTRA="$LAB --fm_arch_kwargs '{\"conv_out_channels\":16,\"num_heads\":8,\"num_layers\":18}'" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/intra-subjects/pooled/$ARM
mkdir -p "$OUT"
eval .venv/bin/python train.py --train_subject_ids $SUBS --test_subject_ids $SUBS \
  --output_name pooled --output_dir "$OUT" --device cuda:0 $BASE $EXTRA
.venv/bin/python scripts/things_eeg/pooled_select.py "$(ls -d "$OUT"/*-pooled | tail -1)"
