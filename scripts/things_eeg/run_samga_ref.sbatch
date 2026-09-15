#!/bin/bash
#SBATCH --job-name=samga_ref
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-mee-br-501
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25/samgaref_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25/samgaref_%j.err

# Run the AUTHORS' SAMGA code unmodified (github.com/LinJiang8/SAMGA), against this repo's data.
# Their inter.sh recipe verbatim except: 30 epochs (our fast-iteration protocol), and layers
# 23/25/28/31/33 because 20/24/32/36 were never extracted here. Same depth prior centre (28).
REF=/nasbrain/p20fores/SAMGA_ref
VENV=/nasbrain/p20fores/Neurobridge_SSL/.venv
IMG=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/internvit_multilayer_src
EEG=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/
OUT=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/goal25_samga/$1

cd "$REF"
mkdir -p "$OUT"
echo "host=$(hostname) gpus=$CUDA_VISIBLE_DEVICES start=$(date +%H:%M:%S) arm=$1"

run_subject () {   # $1=held-out subject  $2=cuda device  $3...=extra args
  local S=$1 DEV=$2; shift 2
  local TRAIN_IDS; TRAIN_IDS=$(seq 1 10 | grep -vw "$S" | tr '\n' ' ')
  local NAME; NAME=$(printf "sub-%02d" "$S")
  if compgen -G "${OUT}/*-${NAME}/result.csv" > /dev/null; then echo "[$NAME] done, skip"; return; fi
  "$VENV/bin/python" train.py \
    --batch_size 1024 --learning_rate 1e-4 --num_epochs 30 \
    --output_name "$NAME" --output_dir "$OUT" \
    --eeg_encoder_type TSConv \
    --train_subject_ids $TRAIN_IDS --test_subject_ids "$S" \
    --softplus --img_l2norm --data_average --save_weights \
    --image_feature_dir "$IMG" --text_feature_dir '' --eeg_data_dir "$EEG" \
    --eeg_aug --eeg_aug_type smooth --frozen_eeg_prior \
    --eeg_feature_dim 1024 --projector linear --feature_dim 512 \
    --stage1_mmd_start 0.9 --stage1_mmd_end 0.5 \
    --use_multilayer_router --layer_ids 23 25 28 31 33 \
    --layer_prior_center 28 --layer_prior_strength 1.0 --router_eval_mode global \
    --device "$DEV" --seed 3300 "$@"
}

( run_subject 1 cuda:0 "${@:2}"; run_subject 5 cuda:0 "${@:2}" )
( run_subject 3 cuda:0 "${@:2}" )
wait
"$VENV/bin/python" compute_avg_results.py --result_dir "$OUT" || true
echo "SAMGA-REF DONE $(date +%H:%M:%S)"
