#!/bin/bash
#SBATCH --job-name=recon_atm
#SBATCH --partition=Brain_GPU
#SBATCH --account=brain
#SBATCH --qos=low
#SBATCH --requeue
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --exclude=sl-mee-br-502
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/atm_%j.out
#SBATCH --error=/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/caption/atm_%j.err

# Close the encoder gap to the JMVR SOTA. The recon stack currently uses TSConv (27.5% top1
# 200-way), but reconstruction here is an INTRA-subject task and this repo's own finding is that
# encoder ranking inverts intra vs inter -- EEGProject wins intra, TSConv wins LOSO. ATM is the
# encoder the SOTA method itself uses and is already supported (train.py:128, ATMS).
#
# Everything except --eeg_encoder_type is copied from the nbaug config so the encoder is the only
# variable. Training is ~6 min/arm at batch 1024 with --data_average, so both arms are cheap; the
# reconstruct+caption stage (~25 min) runs once, on whichever arm wins on best top1.
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTHONPATH=/nasbrain/p20fores/Neurobridge_SSL
source .venv/bin/activate
export HF_HOME=/nasbrain/p20fores/hf-cache/huggingface
PY=/nasbrain/p20fores/Neurobridge_SSL/.venv/bin/python
EEG=/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz
FEAT=data/things_eeg/image_feature/ViT-H-14_final
OUT=results/things_eeg/recon
CAP=data/things_eeg/captions
RESCALE=${RESCALE:-0}

$PY -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('GPU:', torch.cuda.get_device_name(0))"

for ENC in ATM EEGProject; do
    echo "=== train $ENC -> raw ViT-H/14 (intra sub-01, nbaug recipe) ==="
    $PY train.py \
      --train_subject_ids 1 --test_subject_ids 1 \
      --eeg_data_dir "$EEG" --image_feature_dir "$FEAT" --text_feature_dir "" \
      --aug_image_feature_dirs "${FEAT}_augavg" --image_aug \
      --eeg_encoder_type "$ENC" --projector direct --feature_dim 1024 \
      --alpha 0.5 --img_l2norm --eeg_l2norm --mse_on_raw --softplus \
      --eeg_confidence_mode learned --data_average \
      --batch_size 1024 --learning_rate 1e-4 --num_epochs 100 --num_workers 4 \
      --output_dir "$OUT" --output_name "sub-01-vith-$ENC" --save_weights --seed 2025
done

echo "=== encoder comparison (best top1, 200-way) ==="
BEST=""; BESTACC=0
for ENC in ATM EEGProject; do
    D=$(ls -dt $OUT/*sub-01-vith-$ENC | head -1)
    ACC=$(tail -1 $D/result.csv | cut -d, -f6)
    echo "  $ENC  best_top1=$ACC  ($D)"
    if awk "BEGIN{exit !($ACC>$BESTACC)}"; then BESTACC=$ACC; BEST=$D; fi
done
echo "  TSConv (nbaug, incumbent) best_top1=27.500"
echo "  -> reconstructing from $BEST (best_top1=$BESTACC)"

# the incumbent predicted ||.||=28.61+-0.10 vs real ViT-H 22.27+-1.15 -- print it for the new
# encoder too, since IP-Adapter is scale-sensitive and that gap is what --rescale_norm exists for
echo "=== predicted embedding magnitude check ==="
$PY evaluate.py --checkpoint_dir "$BEST" --output_dir results/things_eeg/caption/eval_atm \
    --output_name atm --eval_mode plain_cosine --test_subject_id 1 --dump_npz $CAP/atm_sub01.npz
$PY -c "
import numpy as np
e=np.load('$CAP/atm_sub01.npz')['eeg']; i=np.load('$FEAT/image_test.npy').reshape(-1,1024)
n=np.linalg.norm(e,axis=1)
print(f'  predicted ||.||: mean {n.mean():.3f} std {n.std():.3f}   real ViT-H: mean {np.linalg.norm(i,axis=1).mean():.3f} std {np.linalg.norm(i,axis=1).std():.3f}')
"

echo "=== reconstruct + JMVR metrics ==="
GEN=$OUT/gen_sub01_atm
$PY reconstruction_experiments/reconstruct_eval.py --checkpoint_dir "$BEST" --output_dir $GEN \
    --image_set_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Images/Image_set \
    --n_images 200 --rescale_norm $RESCALE --hf_cache $HF_HOME

echo "=== caption the generated images and score ==="
rm -f $CAP/test.recon_blip2_atm.jsonl
$PY reconstruction_experiments/make_captions.py --model blip2 --split test --flat_dir $GEN \
    --out_name test.recon_blip2_atm.jsonl --batch_size 4
$PY reconstruction_experiments/caption_eval.py --npz $CAP/atm_sub01.npz --captions $CAP/test.qwen.jsonl \
    --hyp_captions $CAP/test.recon_blip2_atm.jsonl
