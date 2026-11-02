#!/bin/bash
# Track-1 recipe on Alljoined-1.6M (neuralbench's 120 Hz grader preprocessing, exported by
# build_alljoined_cache.py): single trials, DINOv2-giant track1 targets (the stimuli are the THINGS-EEG2
# images, same index), direct 1536-D head, checkpoint picked on a 10% held-out-concept val split, per-epoch
# test numbers under the sealed rule (--sealed_test_metric), then the full score_track1.py pass.
#   SPLIT=pooled  : train and test on all 20 subjects (neuralbench's image-task protocol)
#   SPLIT=<any name> with TRAIN_SUBJECTS / TEST_SUBJECTS set: explicit lists. Cross-subject evaluation goes
#                   through neurips_challenge/cv4.sh (the 4x5-subject protocol, see CV4_PROTOCOL.md).
# The EEG memmaps are staged onto node-local disk first (stage_eeg_cache.py) to spare the NAS.
#   usage: [SPLIT=pooled] alljoined_run.sh <tsc|sqf|lab|atm|plab|reve> [extra train.py args]
#   lab = scratch LaBraM; plab / reve = pretrained LaBraM / REVE (warm-up full-data recipes + EMA; the 120 Hz
#   input is interpolated to their native 200 Hz inside the wrapper)
set -e
cd "$(dirname "$0")/.."
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME"
ARM=$1; shift
SPLIT=${SPLIT:-pooled}
DATA=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench
# all 32 Emotiv electrodes in cache order; naming them is what gives LaBraM its electrode tokens
CHANNELS=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('$DATA/info.json'))['ch_names']))")
if [ -n "$TRAIN_SUBJECTS" ]; then TRAIN=$TRAIN_SUBJECTS; TEST=$TEST_SUBJECTS
else case "$SPLIT" in
  pooled) TRAIN=$(seq -s ' ' 1 20); TEST=$TRAIN ;;
  *) echo "unknown SPLIT $SPLIT (or set TRAIN_SUBJECTS/TEST_SUBJECTS)"; exit 1 ;;
esac; fi
case "$ARM" in
  tsc) EXTRA="--eeg_encoder_type TSConv_parameterizable --tsconv_pool_kernel 25 --tsconv_pool_stride 2
         --batch_size 1024 --num_epochs 40" ;;
  sqf) EXTRA="--eeg_encoder_type OrthoFastTSConvSqueezeformer --tsconv_pool_kernel 25 --tsconv_pool_stride 2
         --tsconv_backbone_dim 512 --batch_size 1024 --num_epochs 50" ;;
  lab) EXTRA="--eeg_encoder_type LaBraM --fm_no_pretrained --fm_arch_kwargs {\"conv_out_channels\":16,\"num_heads\":8}
         --batch_size 128 --num_epochs 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup
         --eeg_aug --eeg_aug_type channel_dropout" ;;
  atm) EXTRA="--eeg_encoder_type ATM --atm_no_subject_token --atm_backbone_dim 1024 --batch_size 1024 --num_epochs 40" ;;
  plab) EXTRA="--eeg_encoder_type LaBraM --batch_size 128 --learning_rate 5e-4 --num_epochs 20 --stop_after_epoch 8
         --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
  reve) EXTRA="--eeg_encoder_type REVE --reve_input_normalization none --batch_size 128 --learning_rate 2e-4
         --num_epochs 10 --stop_after_epoch 5 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
  gen) EXTRA="" ;;  # generic: the caller passes the whole model recipe (cv4.sh recipes for new architectures)
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/alljoined/track1/$SPLIT/$ARM
# Several chains may walk the same run list: the first to mkdir the claim owns the run; a claim whose job
# has ended (failed run) is taken over.
ME=${SLURM_JOB_ID:-$$}
mkdir -p "$OUT"
if ! mkdir "$OUT/.claim" 2>/dev/null; then
  sleep 2; owner=$(cat "$OUT/.claim/job" 2>/dev/null)  # the claimer writes its id right after its mkdir
  # take over only when Slurm accounting positively says the owner finished: a failed query means "alive"
  state=$(sacct -n -X -j "$owner" -o State 2>/dev/null | head -1 | tr -d ' '); [ -n "$state" ] || state=UNKNOWN
  if [ -z "$owner" ] || { [ "$owner" != "$ME" ] && [[ "$state" =~ ^(RUNNING|PENDING|UNKNOWN|REQUEUED|SUSPENDED) ]]; }; then
    echo "skip $OUT: claimed by job $owner"; exit 0
  fi
fi
echo "$ME" > "$OUT/.claim/job"
LOCAL=/tmp/${USER}-aj-${SLURM_JOB_ID:-$$}
if [ -n "$SLURM_JOB_ID" ]; then
  # ponytail: a job keeps its staged copy for its later runs (a chain re-stages in seconds, not 1.5-15 min);
  # copies of finished jobs are swept here, so a node holds at most one dead job's copy until its next run
  for d in /tmp/${USER}-aj-*; do id=${d##*-}
    [ "$id" = "$SLURM_JOB_ID" ] || squeue -h -j "$id" 2>/dev/null | grep -q . || rm -rf "$d"; done
else trap 'rm -rf "$LOCAL"' EXIT; fi
EEG=$(.venv/bin/python neurips_challenge/stage_eeg_cache.py "$DATA" "$LOCAL" --subjects $TRAIN --test_subjects $TEST \
  --selected_channels $CHANNELS --time_window 0 120)
.venv/bin/python train.py --train_subject_ids $TRAIN --test_subject_ids $TEST --output_name "$ARM" --output_dir "$OUT" \
  --eeg_data_dir "$EEG" --selected_channels $CHANNELS --time_window 0 120 \
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1 --text_feature_dir "" \
  --projector direct --feature_dim 1536 --img_l2norm --init_temperature 1.0 --asymmetric_loss --grad_clip 1.0 \
  --lr_scheduler onecycle --onecycle_pct_start 0.1 --learning_rate 1e-3 --weight_decay 0.05 \
  --val_concept_ratio 0.1 --select_best_on val --select_best_metric top5 --early_stop_patience 0 --eval_unique_gallery \
  --sealed_test_metric --save_weights --seed 33 --device cuda:0 $EXTRA "$@"
RUN_DIR=$(ls -d "$OUT"/*-"$ARM" | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device cuda:0 --eeg_data_dir "$EEG"
echo "scored: $RUN_DIR/track1_score.json"
if [ -f "$RUN_DIR/checkpoint_sealed_best.pth" ]; then  # the run's best epoch by its per-epoch sealed score
  .venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device cuda:0 --eeg_data_dir "$EEG" \
    --checkpoint_name checkpoint_sealed_best.pth
  echo "scored: $RUN_DIR/track1_score_sealed.json"
fi
if [ -f "$RUN_DIR/checkpoint_testsel_best.pth" ]; then
  .venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device cuda:0 --eeg_data_dir "$EEG" \
    --checkpoint_name checkpoint_testsel_best.pth
  echo "scored: $RUN_DIR/track1_score_testsel.json"
fi
