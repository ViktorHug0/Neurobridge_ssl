#!/bin/bash
# Ten-epoch, same-GPU timing calibration: full/tiny TSConv and full/tiny ATM.
#SBATCH --job-name=tinycal
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --array=0-3%2
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tinycal-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-tinycal-%A_%a.err
set -euo pipefail

REPO_ROOT=/nasbrain/p20fores/Neurobridge_SSL
ROOT=$REPO_ROOT/results/things_eeg/tiny_compute_ensemble/calibration
cd "$REPO_ROOT"
source .venv/bin/activate

case "$SLURM_ARRAY_TASK_ID" in
  0)
    name=full_tsconv
    model_args=(--eeg_encoder_type TSConv --eeg_backbone_dim 1024 --feature_dim 512)
    ;;
  1)
    name=tiny_tsconv
    model_args=(--eeg_encoder_type TSConv_parameterizable --eeg_backbone_dim 256 --feature_dim 128 --tsconv_temporal_filters 10 --tsconv_temporal_kernel 25 --tsconv_pool_kernel 51 --tsconv_pool_stride 5 --tsconv_spatial_filters 10 --tsconv_projection_filters 10)
    ;;
  2)
    name=full_atm
    model_args=(--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128)
    ;;
  3)
    name=tiny_atm
    model_args=(--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128 --atm_d_model 112 --atm_n_heads 4 --atm_e_layers 1 --atm_d_ff 112 --atm_temporal_filters 12 --atm_temporal_kernel 25 --atm_pool_kernel 51 --atm_pool_stride 5 --atm_spatial_filters 12 --atm_projection_filters 12)
    ;;
esac

mkdir -p "$ROOT/$name"
start=$SECONDS
python train.py \
  --batch_size 1024 \
  --num_workers 6 \
  --learning_rate 3e-4 \
  --output_name model \
  --train_subject_ids 2 3 4 5 6 7 8 9 10 \
  --test_subject_ids 1 \
  --select_best_on test \
  --softplus \
  --num_epochs 10 \
  --image_feature_dir "$REPO_ROOT/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit" \
  --text_feature_dir "" \
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz \
  --device cuda:0 \
  --output_dir "$ROOT/$name" \
  --img_l2norm \
  --projector linear \
  --data_average \
  --save_weights \
  --seed 7391 \
  --multi_positive_loss \
  --grouped_batch_sampler \
  --samples_per_image 9 \
  --subject_mixup_mode raw_eeg \
  --mixup_type pairwise \
  --subject_mixup_alpha 0.5 \
  "${model_args[@]}"
elapsed=$((SECONDS - start))
echo "CALIBRATION_RESULT name=$name seconds=$elapsed epochs=10 seconds_per_epoch=$(awk -v t="$elapsed" 'BEGIN {printf "%.3f", t/10}')"
