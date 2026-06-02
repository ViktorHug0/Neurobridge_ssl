#!/bin/bash
# Detached prenorm confidence x feature_dim x projector activation (LOSO on SUBJECT_IDS):
#   - relu (standard ReLU)
#   - relu_gelu (ReLU forward, GELU gradient; Sparse CLIP dead-neuron trick)
# Both use eeg_l2norm + detached ||z|| confidence.
# After each config, compute_avg_results.py averages metrics across completed subjects.
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "$REPO_ROOT"

if [ -f "${REPO_ROOT}/.venv/bin/activate" ]; then
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IMAGE_FEATURE_BASE_DIR="${IMAGE_FEATURE_BASE_DIR:-/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature}"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR="${TEXT_FEATURE_DIR:-}"
EEG_DATA_DIR="${EEG_DATA_DIR:-/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/}"

DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
SEED="${SEED:-3300}"
EEG_BACKBONE_DIM_FIXED="${EEG_BACKBONE_DIM_FIXED:-64}"
FEATURE_DIM_VALUES="${FEATURE_DIM_VALUES:-8192 16384 32768 65536}"
SUBJECT_IDS="${SUBJECT_IDS:-1 2 3 4 5 6 7 8 9 10}"

read -r -a FEATURE_DIM_ARR <<< "$FEATURE_DIM_VALUES"

read -r -a SUBJECT_ARR <<< "$SUBJECT_IDS"

OUTPUT_DIR_BASE="${OUTPUT_DIR:-${REPO_ROOT}/results/things_eeg/inter-subject-sparse}"
# Resume in-place: sparse_fd8192 relu done (10/10); relu_gelu interrupted on sub-01.
SESSION_DIR="${SESSION_DIR:-${OUTPUT_DIR_BASE}/sparse_clip_confidence_detached_relu_gelu_bb64_seed3300}"
SESSION_SUMMARY="${SESSION_DIR}/sweep_summary.csv"
mkdir -p "$SESSION_DIR"

VARIANT_ARGS="--eeg_l2norm --eeg_confidence_mode detached_prenorm"
CONF_MODE="detached_prenorm"
VARIANT_TAG="detached_confidence"

ACTIVATION_SPECS=(
    "relu|relu"
    "relu_gelu|relu_gelu"
)

echo "=============================================================="
echo "Sparse CLIP detached confidence sweep | bb=${EEG_BACKBONE_DIM_FIXED}"
echo "SESSION_DIR: $SESSION_DIR"
echo "FEATURE_DIM_VALUES: ${FEATURE_DIM_ARR[*]}"
echo "SUBJECT_IDS: ${SUBJECT_ARR[*]}"
echo "ACTIVATIONS: relu, relu_gelu"
echo "SEED: $SEED | NUM_EPOCHS: $NUM_EPOCHS | logit_scale: uncapped"
echo "=============================================================="

for DIM in "${FEATURE_DIM_ARR[@]}"
do
    for ACT_SPEC in "${ACTIVATION_SPECS[@]}"
    do
        IFS='|' read -r PROJECTOR_ACTIVATION ACT_TAG <<< "$ACT_SPEC"
        BASE_EXTRA_ARGS="--eeg_backbone_dim ${EEG_BACKBONE_DIM_FIXED} --t_learnable --projector_activation ${PROJECTOR_ACTIVATION}"

        CONFIG_NAME="sparse_fd${DIM}_${ACT_TAG}_${VARIANT_TAG}_seed${SEED}"
        RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
        mkdir -p "$RUN_DIR"

        echo "##########################################################"
        echo "Config: $CONFIG_NAME | fd=$DIM activation=$ACT_TAG"
        echo "##########################################################"

        COMPLETED_SUBJECTS=0
        for SUB_ID in "${SUBJECT_ARR[@]}"
        do
            OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
            if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
                COMPLETED_SUBJECTS=$((COMPLETED_SUBJECTS + 1))
            fi
        done

        if [ "$COMPLETED_SUBJECTS" -eq "${#SUBJECT_ARR[@]}" ]; then
            echo "Skipping fully completed config ${CONFIG_NAME} (${COMPLETED_SUBJECTS}/${#SUBJECT_ARR[@]} subjects)"
        else
            echo "Resuming config ${CONFIG_NAME} (${COMPLETED_SUBJECTS}/${#SUBJECT_ARR[@]} subjects already completed)"
            for SUB_ID in "${SUBJECT_ARR[@]}"
            do
                OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
                if compgen -G "${RUN_DIR}/*-${OUTPUT_NAME}/result.csv" > /dev/null; then
                    echo "Skipping completed subject ${OUTPUT_NAME} for ${CONFIG_NAME}"
                    continue
                fi
                echo "Training subject ${SUB_ID} for $CONFIG_NAME..."

                TRAIN_IDS=""
                for i in {1..10}
                do
                    if [ "$i" -ne "$SUB_ID" ]; then
                        TRAIN_IDS+="$i "
                    fi
                done

                PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
                python3 train.py \
                    --batch_size "$BATCH_SIZE" \
                    --num_workers "$NUM_WORKERS" \
                    --learning_rate "$LEARNING_RATE" \
                    --output_name "$OUTPUT_NAME" \
                    --eeg_encoder_type "$EEG_ENCODER_TYPE" \
                    --train_subject_ids $TRAIN_IDS \
                    --test_subject_ids "$SUB_ID" \
                    --select_best_on test \
                    --softplus \
                    --num_epochs "$NUM_EPOCHS" \
                    --image_feature_dir "$IMAGE_FEATURE_DIR" \
                    --text_feature_dir "$TEXT_FEATURE_DIR" \
                    --eeg_data_dir "$EEG_DATA_DIR" \
                    --device "$DEVICE" \
                    --output_dir "$RUN_DIR" \
                    --img_l2norm \
                    --projector "$PROJECTOR" \
                    --feature_dim "$DIM" \
                    --data_average \
                    --save_weights \
                    --seed "$SEED" \
                    $BASE_EXTRA_ARGS \
                    $VARIANT_ARGS
            done
        fi

        python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
        echo "Averaged inter_subject_summary.csv over completed subjects in ${RUN_DIR}"

        echo "Collecting average results for $CONFIG_NAME into run-local session row..."
        python3 - "$CONFIG_NAME" "$RUN_DIR" "$CONF_MODE" "$ACT_TAG" <<'PY'
import os
import sys
import pandas as pd

config_name, run_dir, conf_mode, act_tag = sys.argv[1:5]
avg_csv = os.path.join(run_dir, "inter_subject_summary.csv")
row_csv = os.path.join(run_dir, "session_summary_row.csv")

if os.path.exists(avg_csv):
    df = pd.read_csv(avg_csv)
    avg_row = df[df["sub"] == "Average"].copy()
    if not avg_row.empty:
        avg_row.insert(0, "config", config_name)
        avg_row["eeg_confidence_mode"] = conf_mode
        avg_row["projector_activation"] = act_tag
        avg_row.to_csv(row_csv, index=False)
PY
    done
done

python3 - "$SESSION_DIR" "$SESSION_SUMMARY" <<'PY'
import glob
import os
import sys
import pandas as pd

session_dir, summary_file = sys.argv[1:3]
row_csvs = sorted(glob.glob(os.path.join(session_dir, "sparse_fd*_seed*", "session_summary_row.csv")))
if row_csvs:
    frames = [pd.read_csv(path) for path in row_csvs]
    pd.concat(frames, ignore_index=True).to_csv(summary_file, index=False)
PY

echo "Sparse CLIP detached confidence sweep completed: $SESSION_SUMMARY"
