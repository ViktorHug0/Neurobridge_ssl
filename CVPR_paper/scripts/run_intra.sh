#!/bin/bash
# E6.1 intra-subject base models (train and test on one subject, 4-rep averages, InternViT-6B L28, no SubjectMix,
# checkpoint = lowest test loss as competitors do, weights saved). Recipe = scripts/things_eeg/intra_insights.sh
# (17 occipito-parietal channels, one-cycle lr 1e-3, the best intra recipe found), plus ATM, absent there.
#   usage: run_intra.sh <arm> <seed> <subject>
# Output: results/things_eeg/cvpr/intra/<arm>/seed<seed>/<timestamp>-sub-XX
main() {
set -e
cd /nasbrain/p20fores/Neurobridge_SSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARM=$1 SEED=$2 SUB=$3
BASE="--num_workers 4 --select_best_on test --softplus --num_epochs 50 --data_average --save_weights
  --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit
  --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/
  --selected_channels P7 P5 P3 P1 Pz P2 P4 P6 P8 PO7 PO3 POz PO4 PO8 O1 Oz O2
  --img_l2norm --projector linear --feature_dim 512 --eeg_backbone_dim 1024 --batch_size 1024
  --learning_rate 1e-3 --lr_scheduler onecycle"
A=${ARM%_vc} VC=""  # <arm>_vc: ValCon selection, as in run_fold.sh
[ "$A" != "$ARM" ] && VC="--val_concept_ratio 0.1 --select_best_on val --early_stop_patience 20 --save_testsel_checkpoint"
case "$A" in
  atm_oc) EXTRA="--eeg_encoder_type ATM" ;;
  atm128_oc) EXTRA="--eeg_encoder_type ATM --eeg_backbone_dim 128 --feature_dim 128" ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
OUT=./results/things_eeg/cvpr/intra/$ARM/seed$SEED
mkdir -p "$OUT"
.venv/bin/python train.py --train_subject_ids $SUB --test_subject_ids $SUB --output_name "$(printf sub-%02d $SUB)" \
  --output_dir "$OUT" --device cuda:0 --seed $SEED --text_feature_dir "" $BASE $EXTRA $VC
.venv/bin/python compute_avg_results.py --result_dir "$OUT" --output_name intra_subject_summary.csv
}
main "$@"; exit $?
