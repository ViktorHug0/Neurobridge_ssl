#!/bin/bash
# after the EEGProject demean arm: stop queue2 (drops ep_swap / TSConv demean) and train ATM-128 with the split key
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while [ $(ls results/things_eeg/mi/train/ep_demean/*/result.csv 2>/dev/null | wc -l) -lt 10 ]; do sleep 20; done
pkill -f night_queue2.sh; sleep 2; pkill -f "mi/train/ep_swap"; sleep 10
MIX="--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
ENCF="--eeg_encoder_type ATM" bash scripts/mi/train_arm2.sh atm128_split "1 5 6 9 3 8" $MIX --eeg_backbone_dim 128 --feature_dim 128 --subject_key split
