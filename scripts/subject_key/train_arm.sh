#!/bin/bash
# one training arm of the subject-key experiments, validation-concept protocol (10% of the training concepts held out
# for checkpoint selection; the test-selected checkpoint and three fixed-epoch checkpoints are saved next to it), paper recipe otherwise.
#   train_arm.sh <arm> "<folds>" <encoder: atm|tsconv|ep|sqf|rmix> <extra train.py flags; FOLD in them = the fold number>  -> results/things_eeg/subject_key/train/<arm>/
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
arm=$1; folds=$2; enc=$3; shift 3
case $enc in
  atm)    ENC="--eeg_encoder_type ATM --feature_dim 128 --eeg_backbone_dim 128 --num_epochs 100 --save_epochs 60 80 100" ;;
  tsconv) ENC="--eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 --feature_dim 512 --eeg_backbone_dim 1024 --num_epochs 50 --save_epochs 30 40 50" ;;
  ep)     ENC="--eeg_encoder_type EEGProject --feature_dim 512 --eeg_backbone_dim 1024 --num_epochs 50 --save_epochs 30 40 50" ;;
  sqf)    ENC="--eeg_encoder_type OrthoFastTSConvSqueezeformer --feature_dim 512 --eeg_backbone_dim 1024 --num_epochs 75 --save_epochs 45 60 75" ;;
  rmix)   ENC="--eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":50} --feature_dim 512 --eeg_backbone_dim 1024 --num_epochs 50 --save_epochs 30 40 50 --amp_dtype bfloat16 --learning_rate 1e-3 --weight_decay 0.05 --lr_scheduler onecycle --ema_decay 0.999 --ema_warmup --grad_clip 1.0" ;;
esac
out=results/things_eeg/subject_key/train/$arm; mkdir -p $out
for SUB in $folds; do
  name=$(printf 'sub-%02d' $SUB)
  ls $out/*-$name/result.csv >/dev/null 2>&1 && { echo "skip $arm $name"; continue; }
  train=""; for i in 1 2 3 4 5 6 7 8 9 10; do [ $i -ne $SUB ] && train+="$i "; done
  $R/.venv/bin/python train.py --batch_size 1024 --num_workers 6 --learning_rate 3e-4 \
    --image_feature_dir $R/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit \
    --eeg_data_dir /nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/ --device cuda:0 \
    --softplus --img_l2norm --projector linear --save_weights --text_feature_dir '' \
    --data_average --grouped_batch_sampler --multi_positive_loss --samples_per_image 9 \
    --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5 \
    --val_concept_ratio 0.1 --save_testsel_checkpoint --select_best_on val --early_stop_patience 20 \
    --output_name $name --output_dir $out --train_subject_ids $train --test_subject_ids $SUB --seed 3300 $ENC "${@//FOLD/$SUB}" \
    || echo "FAILED $arm $name"
done
