#!/bin/bash
# EEGProject arms (0.7 GB each, ~12 min/fold alone): 3 folds at a time, then the TSConv arms
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MIX="--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
par () { arm=$1; shift; for grp in "1 2 3 4" "5 6 7" "8 9 10"; do ENCF="--eeg_encoder_type EEGProject" bash scripts/mi/train_arm2.sh $arm "$grp" $MIX "$@" > logs/mi/train_${arm}_${grp// /}.log 2>&1 & done; wait; }
par ep_demean --subject_key demean
par ep_swap --subject_key swap
bash scripts/mi/train_arm.sh demean "3 5 8 10 2 4 6 7 9" $MIX --subject_key demean
