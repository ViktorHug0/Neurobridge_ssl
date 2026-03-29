#!/bin/bash
set -e
trap 'echo "Script Error"' ERR

IMAGE_FEATURE_BASE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature"
IMAGE_ENCODER_TYPE="${IMAGE_ENCODER_TYPE:-InternViT-6B_layer28_mean_8bit}"
IMAGE_FEATURE_DIR="${IMAGE_FEATURE_BASE_DIR}/${IMAGE_ENCODER_TYPE}"
TEXT_FEATURE_DIR=""
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
EEG_ENCODER_TYPE="${EEG_ENCODER_TYPE:-TSConv}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROJECTOR="${PROJECTOR:-linear}"
FEATURE_DIM="${FEATURE_DIM:-64}"
EEG_BACKBONE_DIM="${EEG_BACKBONE_DIM:-64}"
OUTPUT_ROOT="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"
RUN_TAG="${RUN_TAG:-clap_$(date +'%Y%m%d-%H%M%S')}"
RUN_ROOT="${OUTPUT_ROOT}/${RUN_TAG}"
UNIFIED_CSV="${RUN_ROOT}/clap_summary.csv"
SEED="${SEED:-2026}"

mkdir -p "$RUN_ROOT"

SUBSET_CHANNELS=()
HELD_OUT_SUBJECTS=(${HELD_OUT_SUBJECTS:-1 2 3 4 5 6 7 8 9 10})
ADAPTER_ALPHAS=(${ADAPTER_ALPHAS:-0.1 0.3 0.5 1.0})
CLAP_TAUS=(${CLAP_TAUS:-0.3 0.5})
ADAPTER_HIDDEN_DIMS=(${ADAPTER_HIDDEN_DIMS:-64})
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-40}"
CLAP_LOSS_LAMBDA="${CLAP_LOSS_LAMBDA:-1.0}"
SAMPLES_PER_IMAGE="${SAMPLES_PER_IMAGE:-8}"

# Fixed validation subject per test subject (must differ from test).
declare -A VAL_FOR_TEST
VAL_FOR_TEST[1]=2
VAL_FOR_TEST[2]=1
VAL_FOR_TEST[3]=4
VAL_FOR_TEST[4]=3
VAL_FOR_TEST[5]=6
VAL_FOR_TEST[6]=5
VAL_FOR_TEST[7]=8
VAL_FOR_TEST[8]=7
VAL_FOR_TEST[9]=10
VAL_FOR_TEST[10]=9

append_average_row() {
    local config_name="$1"
    local run_dir="$2"
    python3 - "$UNIFIED_CSV" "$config_name" "$run_dir" <<'PY'
import csv
import os
import sys

out_csv, config_name, run_dir = sys.argv[1:4]
summary_path = os.path.join(run_dir, "inter_subject_summary.csv")
row = {
    "config": config_name,
    "architecture": "",
    "top1 acc": "",
    "top5 acc": "",
    "best top1 acc": "",
    "best top5 acc": "",
    "best test loss": "",
    "best epoch": "",
}

if os.path.isfile(summary_path):
    with open(summary_path, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("sub", "").strip().lower() == "average":
                for key in row:
                    if key in r:
                        row[key] = r[key]
                break

write_header = not os.path.exists(out_csv)
with open(out_csv, "a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

for ADAPTER_ALPHA in "${ADAPTER_ALPHAS[@]}"
do
    for CLAP_TAU in "${CLAP_TAUS[@]}"
    do
        for ADAPTER_HIDDEN_DIM in "${ADAPTER_HIDDEN_DIMS[@]}"
        do
            CFG_NAME="clap_a${ADAPTER_ALPHA}_tau${CLAP_TAU}_h${ADAPTER_HIDDEN_DIM}"
            CFG_RUN_DIR="${RUN_ROOT}/${CFG_NAME}_seed${SEED}"
            mkdir -p "$CFG_RUN_DIR"

            echo "=========================================================="
            echo "Running CLAP config: ${CFG_NAME}"
            echo "=========================================================="

            for SUB_ID in "${HELD_OUT_SUBJECTS[@]}"
            do
                OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
                VAL_ID=${VAL_FOR_TEST[$SUB_ID]}
                TRAIN_IDS=""
                for i in {1..10}
                do
                    if [ "$i" -ne "$SUB_ID" ] && [ "$i" -ne "$VAL_ID" ]; then
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
                    --val_subject_id "$VAL_ID" \
                    --select_best_on val \
                    --softplus \
                    --num_epochs "$NUM_EPOCHS" \
                    --pretrain_epochs "$PRETRAIN_EPOCHS" \
                    --image_feature_dir "$IMAGE_FEATURE_DIR" \
                    --text_feature_dir "$TEXT_FEATURE_DIR" \
                    --eeg_data_dir "$EEG_DATA_DIR" \
                    --device "$DEVICE" \
                    --output_dir "$CFG_RUN_DIR" \
                    --selected_channels "${SUBSET_CHANNELS[@]}" \
                    --img_l2norm \
                    --projector "$PROJECTOR" \
                    --feature_dim "$FEATURE_DIM" \
                    --eeg_backbone_dim "$EEG_BACKBONE_DIM" \
                    --data_average \
                    --save_weights \
                    --multi_positive_loss \
                    --grouped_batch_sampler \
                    --samples_per_image "$SAMPLES_PER_IMAGE" \
                    --architecture clap_adapter \
                    --ssl_lambda 0.0 \
                    --adapter_alpha "$ADAPTER_ALPHA" \
                    --adapter_hidden_dim "$ADAPTER_HIDDEN_DIM" \
                    --clap_tau "$CLAP_TAU" \
                    --clap_loss_lambda "$CLAP_LOSS_LAMBDA" \
                    --seed "$SEED"
            done

            python3 compute_avg_results.py --result_dir "$CFG_RUN_DIR" --output_name "inter_subject_summary.csv"
            append_average_row "$CFG_NAME" "$CFG_RUN_DIR"
        done
    done
done

echo "CLAP sweep completed: ${UNIFIED_CSV}"
