#!/bin/bash
# Track-1 recipe on Alljoined-1.6M (neuralbench's 120 Hz grader preprocessing, exported by
# build_alljoined_cache.py): single trials, DINOv2-giant track1 targets (the stimuli are the THINGS-EEG2
# images, same index), direct 1536-D head, per-epoch test numbers under the sealed rule (--sealed_test_metric),
# then the full score_track1.py pass. Called by neurips_challenge/cv4.sh (the 4x5-subject protocol, CV4_PROTOCOL.md).
# The EEG memmaps are staged onto node-local disk first (stage_eeg_cache.py) to spare the NAS.
#   usage: SPLIT=<name> TRAIN_SUBJECTS=.. TEST_SUBJECTS=.. alljoined_run.sh <tsc|sqf|reve|gen> [extra train.py args]
#   reve = REVE architecture (cv4 recipes train it from scratch); gen = the caller passes the whole model recipe
set -e
cd "$(dirname "$0")/.."
export PATH=/usr/local/bin:/usr/bin:/bin HOME=/tmp/${USER}-home HF_HOME=/nasbrain/p20fores/.cache/huggingface HF_HUB_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HOME"
ARM=$1; shift
DATA=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench
# all 32 Emotiv electrodes in cache order
CHANNELS=$(.venv/bin/python -c "import json; print(' '.join(json.load(open('$DATA/info.json'))['ch_names']))")
[ -n "$SPLIT" ] && [ -n "$TRAIN_SUBJECTS" ] && [ -n "$TEST_SUBJECTS" ] || { echo "set SPLIT, TRAIN_SUBJECTS and TEST_SUBJECTS"; exit 1; }
TRAIN=$TRAIN_SUBJECTS; TEST=$TEST_SUBJECTS
case "$ARM" in
  tsc) EXTRA="--eeg_encoder_type TSConv_parameterizable --tsconv_pool_kernel 25 --tsconv_pool_stride 2" ;;
  sqf) EXTRA="--eeg_encoder_type OrthoFastTSConvSqueezeformer --tsconv_pool_kernel 25 --tsconv_pool_stride 2 --tsconv_backbone_dim 512" ;;
  reve) EXTRA="--eeg_encoder_type REVE --batch_size 128 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
  gen) EXTRA="" ;;
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
  --lr_scheduler onecycle --learning_rate 1e-3 --weight_decay 0.05 --select_best_metric top5 --eval_unique_gallery \
  --sealed_test_metric --save_weights --seed 33 $EXTRA "$@"
RUN_DIR=$(ls -d "$OUT"/*-"$ARM" | tail -1)
.venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device cuda:0 --eeg_data_dir "$EEG"
echo "scored: $RUN_DIR/track1_score.json"
if [ -f "$RUN_DIR/checkpoint_sealed_best.pth" ]; then  # the run's best epoch by its per-epoch sealed score
  .venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" --device cuda:0 --eeg_data_dir "$EEG" \
    --checkpoint_name checkpoint_sealed_best.pth
  echo "scored: $RUN_DIR/track1_score_sealed.json"
fi
