#!/bin/bash
# LOSO training arm on the SubjectMix paper recipe (TSConv k30 fd512, seed 3300) + extra flags.
# usage: train_arm.sh <arm> "<folds>" <extra train.py flags...>   -> results/things_eeg/mi/train/<arm>/
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
arm=$1; folds=$2; shift 2
out=results/things_eeg/mi/train/$arm; mkdir -p $out
for SUB in $folds; do
  name=$(printf 'sub-%02d' $SUB)
  ls $out/*-$name/result.csv >/dev/null 2>&1 && { echo "skip $arm $name"; continue; }
  train=""; for i in 1 2 3 4 5 6 7 8 9 10; do [ $i -ne $SUB ] && train+="$i "; done
  $R/.venv/bin/python train.py --batch_size 1024 --num_workers 4 --learning_rate 3e-4 \
    --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
    --image_feature_dir $R/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ --device cuda:0 \
    --feature_dim 512 --eeg_backbone_dim 1024 --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --multi_positive_loss --samples_per_image 9 \
    --num_epochs 50 --output_name $name --output_dir $out --train_subject_ids $train --test_subject_ids $SUB --seed 3300 "$@" \
    || echo "FAILED $arm $name"
done
$R/.venv/bin/python scripts/data/compute_avg_results.py --result_dir $out --output_name inter_subject_summary.csv
