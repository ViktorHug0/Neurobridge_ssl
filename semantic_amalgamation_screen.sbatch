#!/bin/bash
#SBATCH --job-name=semantic-amalgamation
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-tp-br-522,sl-tp-br-523,sl-tp-br-524
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/single_encoder/semantic_amalgamation_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/single_encoder/semantic_amalgamation_%j.err

set -e
cd /nasbrain/p20fores/Neurobridge_SSL

LAMBDA="${LAMBDA:-0.25}"
AUX_EPOCHS="${AUX_EPOCHS:-0}"
TARGETS="/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-H-14_layer10_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_layer16_mean"
OUTROOT="${OUTROOT:-./results/things_eeg/inter-subjects/single_encoder/semantic_amalgamation_l${LAMBDA}_s123}"

SUBJECTS="${SUBJECTS:-1 2 3}" EPOCHS="${EPOCHS:-30}" DEVICE=cuda:0 \
OUTROOT="$OUTROOT" \
EXTRA_ARGS="--image_feature_dir $TARGETS --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda $LAMBDA --auxiliary_alignment_epochs $AUX_EPOCHS --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.5" \
  ./scripts/things_eeg/run_geometry_encoder.sh TSConv_parameterizable
