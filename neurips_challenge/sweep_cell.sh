#!/bin/bash
# Run one cell of the Track-1 sweep: three architectures, trained then scored, sequentially.
#
# A cell is one (preprocessing x repetition-averaging) combination. All three architectures run on
# the same GPU so the per-GPU load is balanced -- ATM is ~2.1x TSConv and ~3.5x EEGProject, so one
# of each per GPU equalises far better than grouping by architecture.
#
#   sweep_cell.sh <250hz|120hz> <avg|noavg> [--dummy]
#
# --dummy: 2 epochs and scoring on subjects 1-2 only. Exercises every code path end to end,
# including multi-subject aggregation, without paying for a real run.
set -euo pipefail
trap 'echo "[$(date +%H:%M:%S)] SWEEP CELL FAILED: $CELL" >&2' ERR

cd "$(dirname "$0")/.."

PREPROC="${1:?usage: sweep_cell.sh <250hz|120hz> <avg|noavg> [--dummy]}"
AVERAGING="${2:?usage: sweep_cell.sh <250hz|120hz> <avg|noavg> [--dummy]}"
DUMMY="${3:-}"
CELL="${PREPROC}_${AVERAGING}"

IMAGE_FEATURE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-33}"

case "$PREPROC" in
  # NICE-EEG's preprocessing.py wrote this one: MVNN spatial whitening, 0-1.0 s, no band-pass.
  250hz) EEG_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
         TIME_WINDOW=(0 250) ;;
  # NeuralBench's pipeline: 0.1-75 Hz + 50/60 notch, RobustScaler, clamp 20, -0.2 to +0.8 s.
  120hz) EEG_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_120Hz_neuralbench"
         TIME_WINDOW=(0 120) ;;
  *) echo "unknown preprocessing: $PREPROC" >&2; exit 1 ;;
esac

case "$AVERAGING" in
  avg)   AVERAGE_FLAG=(--data_average) ;;
  noavg) AVERAGE_FLAG=() ;;
  *) echo "unknown averaging: $AVERAGING" >&2; exit 1 ;;
esac

if [[ "$DUMMY" == "--dummy" ]]; then
  EPOCHS=2
  SCORE_SUBJECTS=(--test_subject_ids 1 2)
  ROOT="./results/things_eeg/neurips_track1/dummy/${CELL}"
else
  EPOCHS=50
  SCORE_SUBJECTS=()
  ROOT="./results/things_eeg/neurips_track1/sweep/${CELL}"
fi

echo "[$(date +%H:%M:%S)] cell=$CELL epochs=$EPOCHS eeg=$EEG_DIR window=${TIME_WINDOW[*]}"

for ARCH in TSConv EEGProject ATM; do
  OUT_DIR="${ROOT}/${ARCH}"
  echo "[$(date +%H:%M:%S)] === $CELL / $ARCH ==="

  .venv/bin/python train.py \
      --train_subject_ids 1 2 3 4 5 6 7 8 9 10 \
      --test_subject_ids  1 2 3 4 5 6 7 8 9 10 \
      --output_name "$ARCH" \
      --output_dir "$OUT_DIR" \
      --eeg_data_dir "$EEG_DIR" \
      --image_feature_dir "$IMAGE_FEATURE_DIR" \
      --text_feature_dir "" \
      --eeg_encoder_type "$ARCH" \
      --projector direct \
      --feature_dim 1536 \
      --time_window "${TIME_WINDOW[@]}" \
      "${AVERAGE_FLAG[@]}" \
      --multi_positive_loss \
      --grouped_batch_sampler \
      --samples_per_image 10 \
      --img_l2norm \
      --softplus \
      --batch_size 1024 \
      --learning_rate 3e-4 \
      --num_epochs "$EPOCHS" \
      --val_concept_ratio 0.1 \
      --select_best_on val \
      --save_weights \
      --seed "$SEED" \
      --device "$DEVICE"

  RUN_DIR=$(ls -d "${OUT_DIR}"/*-"${ARCH}" | tail -1)
  .venv/bin/python neurips_challenge/score_track1.py "$RUN_DIR" \
      --device "$DEVICE" "${SCORE_SUBJECTS[@]}"
  echo "[$(date +%H:%M:%S)] scored: $RUN_DIR/track1_score.json"
done

echo "[$(date +%H:%M:%S)] CELL COMPLETE: $CELL"
