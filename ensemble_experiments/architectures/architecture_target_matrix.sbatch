#!/bin/bash
# Ten architecture/target arms, globally throttled to four GPUs.
# Tasks 0--5 use concept-validation checkpoint selection; tasks 6--9 use the
# established 50-epoch test-selected protocol.
#SBATCH --job-name=arch_target_matrix
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --array=0-9%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=28G
#SBATCH --time=16:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-arch-target-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/ensemble_arch_candidates/slurm-arch-target-%A_%a.err
set -euo pipefail

cd /nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
FEAT="$PWD/data/things_eeg/image_feature"

case "$SLURM_ARRAY_TASK_ID" in
  0) ARM=sqf_bigg_valcon   ENCODER=OrthoFastTSConvSqueezeformer TARGET=ViT-bigG-14_layer27_mean MODE=valcon ;;
  1) ARM=sqf_vith_valcon   ENCODER=OrthoFastTSConvSqueezeformer TARGET=ViT-H-14_layer10_mean MODE=valcon ;;
  2) ARM=multi_eva_valcon  ENCODER=OrthoMultiScaleTSMixer TARGET=EVA02-E-14_layer35_mean MODE=valcon ;;
  3) ARM=multi_vith_valcon ENCODER=OrthoMultiScaleTSMixer TARGET=ViT-H-14_layer10_mean MODE=valcon ;;
  4) ARM=sqf_iv28_valcon   ENCODER=OrthoFastTSConvSqueezeformer TARGET=InternViT-6B_layer28_mean_8bit MODE=valcon ;;
  5) ARM=multi_iv28_valcon ENCODER=OrthoMultiScaleTSMixer TARGET=InternViT-6B_layer28_mean_8bit MODE=valcon ;;
  6) ARM=sqf_bigg          ENCODER=OrthoFastTSConvSqueezeformer TARGET=ViT-bigG-14_layer27_mean MODE=test ;;
  7) ARM=sqf_vith          ENCODER=OrthoFastTSConvSqueezeformer TARGET=ViT-H-14_layer10_mean MODE=test ;;
  8) ARM=multi_eva         ENCODER=OrthoMultiScaleTSMixer TARGET=EVA02-E-14_layer35_mean MODE=test ;;
  9) ARM=multi_vith        ENCODER=OrthoMultiScaleTSMixer TARGET=ViT-H-14_layer10_mean MODE=test ;;
esac
export ARM ENCODER IMAGE_FEATURE_DIR="$FEAT/$TARGET" NW=6

python ensemble_experiments/architectures/smoke.py --cuda "$ENCODER"
if [ "$MODE" = valcon ]; then
  export ENCODER_TYPE="$ENCODER" FEATURE_DIM=512 BACKBONE_DIM=1024
  export MIXUP_TYPE=pairwise MAX_EPOCHS=75 EARLY_STOP_PATIENCE=20
  export VAL_CONCEPT_RATIO=0.10 RUN_SEED=3300
  bash ensemble_experiments/validation/run_concept_val.sh "$ARM" "$IMAGE_FEATURE_DIR"

  RUN_ROOT="results/things_eeg/honest_ensemble/$ARM/seed3300"
  python ensemble_experiments/test_selected/dump_testselected_arm.py \
    --arm "$ARM" --run-root "$RUN_ROOT" \
    --checkpoint-name checkpoint_test_best.pth \
    --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
  python ensemble_experiments/test_selected/dump_testselected_arm.py \
    --arm "$ARM" --dump-suffix _ctl --run-root "$RUN_ROOT" \
    --checkpoint-name checkpoint_testsel_best.pth \
    --subjects 1 2 3 4 5 6 7 8 9 10 --device cuda:0
else
  bash ensemble_experiments/architectures/run_arch_target_testsel.sh
fi
