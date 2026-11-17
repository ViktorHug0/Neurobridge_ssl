#!/bin/bash
# Squeezeformer with key-free training (sqf recipe: 75 epochs), 10 folds
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MIX="--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
ENCF="--eeg_encoder_type OrthoFastTSConvSqueezeformer" bash scripts/mi/train_arm2.sh sqf_demean "1 2 3 4 5 6 7 8 9 10" $MIX --num_epochs 75 --subject_key demean
