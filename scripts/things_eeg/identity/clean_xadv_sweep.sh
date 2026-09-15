#!/bin/bash
# clean-xadv experiment A: adversarial-alone (GRL@backbone, mixup OFF) across a
# small annealed-lambda cap sweep, matched against SubjectMix-alone on the same
# 10 LOSO folds. Tests SAGE's "SubjectMix > adversarial" claim and maps the
# neutral->detrimental lambda curve. Same base recipe across all arms.
set -e
trap 'echo "Script Error"' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"

IMAGE_FEATURE_DIR="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-2099}"
OUTPUT_DIR_BASE="${OUTPUT_DIR:-./results/things_eeg/inter-subjects}"

# base recipe, identical across arms (only the subject-invariance intervention varies)
BASE_ARGS="--eeg_encoder_type TSConv --projector linear --feature_dim 128 --eeg_backbone_dim 128 \
  --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 \
  --data_average --img_l2norm --softplus --batch_size 1024 --learning_rate 3e-4 --num_epochs 50 \
  --num_workers 4 --save_weights"

# arms: SubjectMix-alone baseline vs adversarial-alone at three lambda caps
CONFIG_NAMES=(subjectmix_alone xadv_cap005 xadv_cap01 xadv_cap03)
CONFIG_ARGS=(
  "--subject_mixup_mode raw_eeg"
  "--subject_mixup_mode none --clean_xadv_lambda 0.05"
  "--subject_mixup_mode none --clean_xadv_lambda 0.1"
  "--subject_mixup_mode none --clean_xadv_lambda 0.3"
)

# resume: set RESUME_DIR=<existing session dir> to continue; default = fresh session
SESSION_DIR="${RESUME_DIR:-${OUTPUT_DIR_BASE}/$(date +'%Y%m%d-%H%M%S')_clean_xadv_seed${SEED}}"
SESSION_SUMMARY="${SESSION_DIR}/session_summary.csv"
mkdir -p "$SESSION_DIR"
echo "== session: $SESSION_DIR"

for c_idx in "${!CONFIG_NAMES[@]}"; do
    CONFIG_NAME="${CONFIG_NAMES[$c_idx]}"
    EXTRA_ARGS="${CONFIG_ARGS[$c_idx]}"
    RUN_DIR="${SESSION_DIR}/${CONFIG_NAME}"
    mkdir -p "$RUN_DIR"
    echo "### config: $CONFIG_NAME | $EXTRA_ARGS"

    for SUB_ID in {1..10}; do
        OUTPUT_NAME=$(printf "sub-%02d" "$SUB_ID")
        TRAIN_IDS=""; for i in {1..10}; do [ "$i" -ne "$SUB_ID" ] && TRAIN_IDS+="$i "; done

        python3 train.py \
            --train_subject_ids $TRAIN_IDS --test_subject_ids "$SUB_ID" \
            --output_name "$OUTPUT_NAME" --output_dir "$RUN_DIR" \
            --image_feature_dir "$IMAGE_FEATURE_DIR" --text_feature_dir "" \
            --eeg_data_dir "$EEG_DATA_DIR" --device "$DEVICE" \
            $BASE_ARGS $EXTRA_ARGS --seed "$SEED"

        python3 compute_avg_results.py --result_dir "$RUN_DIR" --output_name "inter_subject_summary.csv"
    done

    # append this config's Average row to the session summary
    python3 -c "
import os, sys, pandas as pd
run_dir, cfg, summ = sys.argv[1:4]
p = os.path.join(run_dir, 'inter_subject_summary.csv')
if os.path.exists(p):
    df = pd.read_csv(p); row = df[df['sub'] == 'Average'].copy()
    if not row.empty:
        row.insert(0, 'config', cfg)
        row.to_csv(summ, mode='a', index=False, header=not os.path.exists(summ))
        print('added', cfg, '->', summ)
" "$RUN_DIR" "$CONFIG_NAME" "$SESSION_SUMMARY"
done

echo "== done. compare arms in: $SESSION_SUMMARY"
