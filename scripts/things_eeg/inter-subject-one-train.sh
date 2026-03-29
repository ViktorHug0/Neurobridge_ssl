#!/bin/bash
# Train on one subject per run; evaluate on a fixed TEST_SUBJECT_ID.
# Always runs once per *other* subject (default: 10 subjects → 9 runs, test held out).
# Fixed CL config: multi-positive loss, grouped batch sampler, 9 samples per image.
# Merged metrics: one_train_summary.csv (column "sub" = training subject id).
#
# Examples:
#   ./inter-subject-one-train.sh                        # test 10, train 1..9 each
#   TEST_SUBJECT_ID=1 ./inter-subject-one-train.sh   # test 1, train 2..10 each
#   NUM_SUBJECTS=12 TEST_SUBJECT_ID=3 ...            # if you ever change cohort size
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="InternViT-6B_layer28_mean_8bit"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="cuda:0"
EEG_ENCODER_TYPE="TSConv"
BATCH_SIZE=1024
LEARNING_RATE=1e-4
NUM_EPOCHS=25
NUM_WORKERS=4
SELECTED_CHANNELS=()
PROJECTOR="linear"
FEATURE_DIM=64
EEG_BACKBONE_DIM=64
OUTPUT_DIR_BASE=${OUTPUT_DIR:-"./results/things_eeg/inter-subject-one-train"}

# Held-out subject: train separately on every other ID in 1..NUM_SUBJECTS.
NUM_SUBJECTS="${NUM_SUBJECTS:-10}"
TEST_SUBJECT_ID="${TEST_SUBJECT_ID:-8}"

if ! [[ "$TEST_SUBJECT_ID" =~ ^[0-9]+$ ]] || [ "$TEST_SUBJECT_ID" -lt 1 ] || [ "$TEST_SUBJECT_ID" -gt "$NUM_SUBJECTS" ]; then
    echo "TEST_SUBJECT_ID must be an integer in 1..${NUM_SUBJECTS} (got: ${TEST_SUBJECT_ID})" >&2
    exit 1
fi

EXTRA_ARGS="--multi_positive_loss --grouped_batch_sampler --samples_per_image 9"

SEED=${SEED:-2009}
SSL_LAMBDA=0

SESSION_TIMESTAMP=$(date +'%Y%m%d-%H%M%S')
SESSION_DIR="${OUTPUT_DIR_BASE}/${SESSION_TIMESTAMP}_seed${SEED}_test${TEST_SUBJECT_ID}"
mkdir -p "$SESSION_DIR"

echo "Session directory: $SESSION_DIR"
echo "Test subject: $TEST_SUBJECT_ID | Training runs: $((NUM_SUBJECTS - 1)) (subjects 1..${NUM_SUBJECTS} except test)"

for SUB_ID in $(seq 1 "$NUM_SUBJECTS")
do
    if [ "$SUB_ID" -eq "$TEST_SUBJECT_ID" ]; then
        continue
    fi
    # output_name must end with a 6-char suffix "sub-XX" for compute_avg_results.py (run[-6:])
    OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
    echo "##########################################################"
    echo "Train subject ${SUB_ID} only -> test subject ${TEST_SUBJECT_ID}"
    echo "##########################################################"

    python3 train.py \
        --batch_size "$BATCH_SIZE" \
        --num_workers "$NUM_WORKERS" \
        --learning_rate "$LEARNING_RATE" \
        --output_name "$OUTPUT_NAME" \
        --eeg_encoder_type "$EEG_ENCODER_TYPE" \
        --train_subject_ids "$SUB_ID" \
        --test_subject_ids "$TEST_SUBJECT_ID" \
        --softplus \
        --num_epochs "$NUM_EPOCHS" \
        --image_feature_dir "$IMAGE_FEATURE_DIR" \
        --text_feature_dir "$TEXT_FEATURE_DIR" \
        --eeg_data_dir "$EEG_DATA_DIR" \
        --device "$DEVICE"  \
        --output_dir "$SESSION_DIR" \
        --selected_channels "${SELECTED_CHANNELS[@]}" \
        --img_l2norm \
        --projector "$PROJECTOR" \
        --feature_dim "$FEATURE_DIM" \
        --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
        --data_average \
        --save_weights \
        --ssl_lambda "$SSL_LAMBDA" \
        $EXTRA_ARGS \
        --seed "$SEED"

    python3 compute_avg_results.py --result_dir "$SESSION_DIR" --output_name "one_train_summary.csv"
done

echo "Done. $((NUM_SUBJECTS - 1)) runs → single table (test = subject ${TEST_SUBJECT_ID}):"
echo "  $SESSION_DIR/one_train_summary.csv"
echo "(Column \"sub\" is the training subject; use numeric columns to rank donors; Average row is mean over the $((NUM_SUBJECTS - 1)) runs.)"
