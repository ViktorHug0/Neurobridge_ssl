#!/bin/bash
# THINGS-MEG with the ATM recipe of xds_train.sh at batch 128 (ATM's first convolution over 271 channels needs 9.3 GB
# at batch 1024): two 8-epoch tuning runs on fold 1 (learning rate 3e-4 and 1e-4), then the four LOSO folds at the
# learning rate with the lower best validation loss (held-out training concepts; no test subject involved).
cd /nasbrain/p20fores/Neurobridge_SSL
T=results/things_meg/cvpr/tune
for lr in 3e-4 1e-4; do
  bash CVPR_paper/scripts/xds_train.sh meg 1 tune/bs128_lr$lr --batch_size 128 --learning_rate $lr --num_epochs 8 > logs/cvpr/xds/tune_meg_bs128_lr$lr.log 2>&1
done
# result.csv, column 7: loss of the selected epoch on the validation concepts
LR=$(for lr in 3e-4 1e-4; do echo "$(tail -1 $T/bs128_lr$lr/*-sub-01/result.csv | cut -d, -f7) $lr"; done | sort -g | head -1 | cut -d' ' -f2)
echo "validation loss: $(for lr in 3e-4 1e-4; do echo -n "lr $lr $(tail -1 $T/bs128_lr$lr/*-sub-01/result.csv | cut -d, -f7)  "; done)-> lr ${LR:?no tuning result}" > logs/cvpr/xds/tune_meg_choice.log
for s in 1 2 3 4; do
  bash CVPR_paper/scripts/xds_train.sh meg $s loso/atm128_vc/seed3300 --batch_size 128 --learning_rate $LR > logs/cvpr/xds/loso_meg_0$s.log 2>&1
done
