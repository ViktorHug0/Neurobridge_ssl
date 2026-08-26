#!/bin/bash
# Seven-arm frozen-member fusion-aware pilot, subjects 1--6, four GPUs maximum.
# Each array task is one fold, making failures/requeues local and scheduling modular.
#SBATCH --job-name=frozen_fuse
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --array=0-41%4
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-frozen-fuse-%A_%a.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/slurm-frozen-fuse-%A_%a.err
set -euo pipefail

arm_index=$((SLURM_ARRAY_TASK_ID % 7))
subject=$((SLURM_ARRAY_TASK_ID / 7 + 1))

case "$arm_index" in
  0)
    export ARM=frozen_atm_to_tsconv_b000_valcon
    export BETA_ENSEMBLE=0
    export ENCODER_A=ATM BACKBONE_DIM_A=128 FEATURE_DIM_A=128
    export FROZEN_CHECKPOINT_MEMBER=a
    export ENCODER_B=TSConv_parameterizable BACKBONE_DIM_B=1024 FEATURE_DIM_B=512
    export SEED_A=3300 SEED_B=3301
    ;;
  1)
    export ARM=frozen_atm_to_tsconv_b030_valcon
    export BETA_ENSEMBLE=0.30
    export ENCODER_A=ATM BACKBONE_DIM_A=128 FEATURE_DIM_A=128
    export FROZEN_CHECKPOINT_MEMBER=a
    export ENCODER_B=TSConv_parameterizable BACKBONE_DIM_B=1024 FEATURE_DIM_B=512
    export SEED_A=3300 SEED_B=3301
    ;;
  2)
    export ARM=frozen_atm_to_tsconv_b100_valcon
    export BETA_ENSEMBLE=1.0
    export ENCODER_A=ATM BACKBONE_DIM_A=128 FEATURE_DIM_A=128
    export FROZEN_CHECKPOINT_MEMBER=a
    export ENCODER_B=TSConv_parameterizable BACKBONE_DIM_B=1024 FEATURE_DIM_B=512
    export SEED_A=3300 SEED_B=3301
    ;;
  3)
    export ARM=frozen_tsconv_to_atm_b000_valcon
    export BETA_ENSEMBLE=0
    export ENCODER_A=TSConv_parameterizable BACKBONE_DIM_A=1024 FEATURE_DIM_A=512
    export FROZEN_CHECKPOINT_MEMBER=b
    export ENCODER_B=ATM BACKBONE_DIM_B=128 FEATURE_DIM_B=128
    export SEED_A=3301 SEED_B=3300
    ;;
  4)
    export ARM=frozen_tsconv_to_atm_b030_valcon
    export BETA_ENSEMBLE=0.30
    export ENCODER_A=TSConv_parameterizable BACKBONE_DIM_A=1024 FEATURE_DIM_A=512
    export FROZEN_CHECKPOINT_MEMBER=b
    export ENCODER_B=ATM BACKBONE_DIM_B=128 FEATURE_DIM_B=128
    export SEED_A=3301 SEED_B=3300
    ;;
  5)
    export ARM=frozen_tsconv_to_atm_b100_valcon
    export BETA_ENSEMBLE=1.0
    export ENCODER_A=TSConv_parameterizable BACKBONE_DIM_A=1024 FEATURE_DIM_A=512
    export FROZEN_CHECKPOINT_MEMBER=b
    export ENCODER_B=ATM BACKBONE_DIM_B=128 FEATURE_DIM_B=128
    export SEED_A=3301 SEED_B=3300
    ;;
  6)
    export ARM=frozen_tsconv_to_tsconv_b030_valcon
    export BETA_ENSEMBLE=0.30
    export ENCODER_A=TSConv_parameterizable BACKBONE_DIM_A=1024 FEATURE_DIM_A=512
    export FROZEN_CHECKPOINT_MEMBER=b
    export ENCODER_B=TSConv_parameterizable BACKBONE_DIM_B=1024 FEATURE_DIM_B=512
    export SEED_A=3301 SEED_B=3302
    ;;
  *)
    echo "unexpected arm index $arm_index" >&2
    exit 2
    ;;
esac

cd /nasbrain/p20fores/Neurobridge_SSL
export SUBJECTS="$subject"
export DEVICE=cuda:0
export NW=${SLURM_CPUS_PER_TASK:-6}
export BATCH_SIZE=512
export NUM_EPOCHS=50

echo "[frozen-wave] task=$SLURM_ARRAY_TASK_ID arm=$ARM subject=$subject beta=$BETA_ENSEMBLE"
bash ensemble_experiments/decorrelated_models/run_frozen_fusion.sh
