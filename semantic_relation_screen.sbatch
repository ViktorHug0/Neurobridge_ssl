#!/bin/bash
#SBATCH --job-name=semantic-relation
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --nodelist=sl-tp-br-522,sl-tp-br-523,sl-tp-br-524
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/single_encoder/semantic_relation_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/inter-subjects/single_encoder/semantic_relation_%j.err

set -e
cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate

PRIMARY=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit
VITH=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/ViT-H-14_layer10_mean
DINO=/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_layer16_mean
LAMBDA="${LAMBDA:-0.25}"
REL_EPOCHS="${REL_EPOCHS:-10}"
OUTROOT="${OUTROOT:-./results/things_eeg/inter-subjects/single_encoder/semantic_relation_l${LAMBDA}_e${REL_EPOCHS}_s123}"

for SUBJECT in ${SUBJECTS:-1 2 3}; do
  TRAIN_IDS=$(seq 1 10 | grep -vw "$SUBJECT" | tr '\n' ' ')
  SUB=$(printf 'sub-%02d' "$SUBJECT")
  if compgen -G "${OUTROOT}/*-${SUB}/result.csv" > /dev/null; then
    echo "[semantic-relation] $SUB already complete; skipping"
    continue
  fi
  python train.py \
    --train_subject_ids $TRAIN_IDS --test_subject_ids "$SUBJECT" \
    --output_name "$SUB" --output_dir "$OUTROOT" \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ \
    --image_feature_dir "$PRIMARY,$VITH,$DINO" --auxiliary_image_targets \
    --text_feature_dir '' --eeg_encoder_type TSConv_parameterizable \
    --tsconv_temporal_kernel 30 --tsconv_pool_kernel 51 --tsconv_dropout 0.5 \
    --projector linear --feature_dim 1024 --eeg_backbone_dim 1024 \
    --semantic_relational_lambda "$LAMBDA" \
    --semantic_relational_epochs "$REL_EPOCHS" \
    --semantic_relational_temperature 0.1 \
    --time_window 0 250 --data_average --img_l2norm --softplus \
    --multi_positive_loss --grouped_batch_sampler --samples_per_image 9 \
    --subject_mixup_mode raw_eeg --subject_mixup_alpha 0.5 --mixup_type pairwise \
    --batch_size 1024 --learning_rate 3e-4 --num_epochs "${EPOCHS:-30}" \
    --num_workers 4 --eval_mode plain_cosine --select_best_on test \
    --seed 3300 --save_weights --device cuda:0
done

python compute_avg_results.py \
  --result_dir "$OUTROOT" --output_name inter_subject_summary.csv || true
