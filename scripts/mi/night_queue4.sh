#!/bin/bash
# TSConv with key-free training, the 9 remaining folds (fold 1 done at 02:41)
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MIX="--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
bash scripts/mi/train_arm2.sh demean "6 3 5 8 10 2 4 7 9" $MIX --subject_key demean
