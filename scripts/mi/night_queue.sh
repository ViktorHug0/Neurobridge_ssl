#!/bin/bash
# sequential GPU queue for the night (one training at a time: a run takes ~9.3 GB of the 10 GB)
R=/nasbrain/p20fores/Neurobridge_SSL; cd $R
MIX="--subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5"
while pgrep -f "output_dir results/things_eeg/mi/train/base_check" >/dev/null; do sleep 20; done
bash scripts/mi/train_arm.sh demean "1 3 5 8 10" $MIX --subject_key demean
bash scripts/mi/train_arm.sh swap "1 3 5 8 10" $MIX --subject_key swap
bash scripts/mi/train_arm.sh demean "2 4 6 7 9" $MIX --subject_key demean
bash scripts/mi/train_arm.sh swap "2 4 6 7 9" $MIX --subject_key swap
