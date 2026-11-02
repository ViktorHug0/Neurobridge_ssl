#!/bin/bash
# smoke test of the new alignment losses: 1 epoch TSConv, train subject 1 + THINGS 21, test subject 2
cd /nasbrain/p20fores/Neurobridge_SSL
for v in "gneg|--global_negatives_replace" "gnegc|--global_negatives_replace --global_concept_eps 0.2" "gsoft|--global_negatives_replace --global_soft_tau 0.02" "gsym|--global_negatives_replace --global_symmetric 0.5" "siglip|--siglip" "softclip|--soft_target_tau 0.02" "bn64|--eeg_backbone_dim 64 --projector bottleneck" "l16|--image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_layer16_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" "iaug|--image_aug --aug_image_feature_dirs /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Original /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianBlur"; do
  n=${v%%|*}; f=${v#*|}
  SPLIT=smoke/$n TRAIN_SUBJECTS="1 21" TEST_SUBJECTS="2" bash neurips_challenge/alljoined_run.sh tsc --select_best_on fixed --val_concept_ratio 0 --sealed_test_every 1 --sealed_test_fraction 0.5 --num_epochs 1 --tsconv_domain_bn_boundary 21 $f > logs/smoke_$n.out 2>&1
  echo "$n: $(grep -aE 'sealed-rule test top5' logs/smoke_$n.out | tail -1) $(grep -acE 'Traceback|Error' logs/smoke_$n.out) errors"
done
