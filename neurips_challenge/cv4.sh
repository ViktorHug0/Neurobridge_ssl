#!/bin/bash
# The cross-subject evaluation protocol for the sealed phase (new subjects, new images): 4 fixed folds of 5
# Alljoined test subjects, training on the other 15. Full description, rationale and decision rules:
# neurips_challenge/CV4_PROTOCOL.md. Do not edit FOLDS: every number in results/alljoined/track1/cv4/ is
# only comparable because the folds never change.
#
#   usage: cv4.sh <recipe> [fold ...]          one recipe on the given folds (default: 1 2 3 4)
#          cv4.sh --list                       print the recipes
#   e.g.   cv4.sh plab_gain 1 2                screen (10 test subjects); then cv4.sh plab_gain 3 4
# Results: results/alljoined/track1/cv4/<recipe>/fold<k>/<arm>/<run>/track1_score.json (per-subject sealed
# scores); compare with neurips_challenge/cv4_summary.py. Finished folds are skipped, and alljoined_run.sh
# claims each run, so several workers can be pointed at the same recipes and folds.
cd "$(dirname "$0")/.."
case "$1" in enscache_*|poe_sim|route_*|feat_*) exec bash neurips_challenge/overnight/ens_cache_job.sh "$@" ;; ftor_tsc_*) exec bash neurips_challenge/overnight/ft_oracle_parts.sh "$@" ;; ftor_*) exec bash neurips_challenge/overnight/ft_oracle.sh "$@" ;; esac  # SUBJECT_LOG D4 fine-tune oracle

FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")

# recipe -> "<alljoined_run.sh arm> <train.py flags>". Every recipe trains for a fixed budget and keeps its
# last epoch (--select_best_on fixed); the budgets come from the per-epoch sealed curves (CV4_PROTOCOL.md §5).
recipe() {
  case "$1" in
    tsc)       echo "tsc  --num_epochs 12" ;;
    sqf)       echo "sqf  --num_epochs 30" ;;
    atm)       echo "atm  --num_epochs 20" ;;
    lab)       echo "lab  --num_epochs 20 --stop_after_epoch 6" ;;
    plab_gain) echo "plab --num_epochs 20 --stop_after_epoch 6 --amp_gain_range 2" ;;
    reve_z)    echo "reve --num_epochs 10 --stop_after_epoch 4 --reve_input_normalization trial_zscore" ;;
    # 2026-10-01: lab / plab_gain were still rising at their stop epoch 6 on 15-subject folds (audit); same
    # schedule, later stop (epochs 1-6 are unchanged, the budget is picked inside 1-10)
    lab_s10)       echo "lab  --num_epochs 20 --stop_after_epoch 10" ;;
    plab_gain_s10) echo "plab --num_epochs 20 --stop_after_epoch 10 --amp_gain_range 2" ;;
    # *_things: the 10 THINGS-EEG2 subjects (sub-21..30, export_things_to_alljoined.py) join every fold's
    # training set; test subjects stay Alljoined. 25 training subjects = 1.67x the steps per epoch.
    plab_gain_things) echo "plab --num_epochs 20 --stop_after_epoch 6 --amp_gain_range 2" ;;
    tsc_things)       echo "tsc  --num_epochs 12" ;;
    reve_z_things)    echo "reve --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore" ;;  # LaBraM's peak moved 3 -> 5 with THINGS
    # 2026-10-02 overnight architecture screen (generic arm: the line is the whole model recipe)
    eegnet)       echo "gen --eeg_encoder_type EEGNet --batch_size 1024 --num_epochs 20" ;;
    mixer)        echo "gen --eeg_encoder_type MultiScaleTSMixer --batch_size 1024 --num_epochs 20" ;;
    gct)          echo "gen --eeg_encoder_type TinyGatedChannelTransformer --batch_size 1024 --num_epochs 20" ;;
    conformer)    echo "gen --eeg_encoder_type EEGConformer --batch_size 1024 --num_epochs 20" ;;
    cbramod_gain) echo "gen --eeg_encoder_type CBraMod --batch_size 128 --learning_rate 5e-4 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --amp_gain_range 2" ;;
    fbtn)         echo "gen --eeg_encoder_type FilterBankTangentNet --batch_size 1024 --num_epochs 20" ;;
    # new designs (ensemble_experiments/architectures/alljoined_encoders.py)
    dualstream)   echo "gen --eeg_encoder_type DualStreamNet --batch_size 1024 --num_epochs 20" ;;
    robust_ct)    echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 20" ;;
    robust_ct_e40) echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40" ;;  # robust_ct still rising at 20
    # 2026-10-05 levers (screened against tsc / robust_ct_e40): L1 latency + montage jitter, L2 repeat-mean
    # training, L3 target-subspace head, L4 RSVP-neighbour heads, L5 feature MixStyle
    tsc_lat)      echo "tsc  --num_epochs 12 --coherent_latency_max_shift 4 --coherent_latency_std 2" ;;
    tsc_stj)      echo "tsc  --num_epochs 12 --coherent_latency_max_shift 4 --coherent_latency_std 2 --spatial_jitter_deg 8" ;;
    tsc_rgrp)     echo "tsc  --num_epochs 12 --repeat_group_batches --multi_positive_loss" ;;
    tsc_rmean)    echo "tsc  --num_epochs 12 --repeat_group_batches --multi_positive_loss --repeat_mean_loss 0.5" ;;
    tsc_sub64)    echo "tsc  --num_epochs 12 --projector subspace --eeg_backbone_dim 64" ;;
    tsc_nbr)      echo "tsc  --num_epochs 12 --neighbor_loss 0.3" ;;
    tsc_mixstyle) echo "tsc  --num_epochs 12 --grouped_batch_sampler --subject_mixstyle_prob 0.5" ;;  # MixStyle needs matched-stimulus batches
    rct_stj)      echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40 --coherent_latency_max_shift 4 --coherent_latency_std 2 --spatial_jitter_deg 8" ;;
    rct_rmean)    echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40 --repeat_group_batches --multi_positive_loss --repeat_mean_loss 0.5" ;;
    rct_sub64)    echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40 --projector subspace --eeg_backbone_dim 64" ;;
    tsc_sub64c)   echo "tsc  --num_epochs 12 --projector subspace_c --eeg_backbone_dim 64" ;;  # centered: sub64's mean term swamped cosine
    rct_sub64c)   echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40 --projector subspace_c --eeg_backbone_dim 64" ;;
    rct_nbr)      echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 1024 --num_epochs 40 --neighbor_loss 0.3" ;;
    reve_z_nbr_things) echo "reve --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore --neighbor_loss 0.3" ;;  # L4 on the best single model (THINGS subjects get no neighbour loss)
    # 2026-10-06 REVE+THINGS recipe probes (vs reve_z_things, whose one-cycle lr is still ~90% of peak at its epoch-3 peak)
    reve_scratch_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    reve_lr1e4_things)   echo "reve --learning_rate 1e-4 --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore" ;;
    reve_llrd_things)    echo "reve --fm_layer_decay 0.85 --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore" ;;  # 22 blocks: embeddings at 0.85^23 = 2% lr
    reve_oc4_things)     echo "reve --num_epochs 4 --reve_input_normalization trial_zscore" ;;  # the whole one-cycle anneal ends at the old peak
    # scratch REVE+THINGS won its screen (+2.4): which half matters, and does scratch LaBraM gain the same?
    reve_scratch)        echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    lab_s10_things)      echo "lab  --num_epochs 20 --stop_after_epoch 10" ;;
    # what makes scratch REVE win (vs reve_scratch_things): capacity (depth 6 / 32 of 22) and recipe (the small
    # robust_ct with the FM recipe: batch 128 + EMA, which it never had)
    reve_d6_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    reve_d6)         echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # THINGS effect on scratch REVE (vs reve_d6_things, 4 folds)
    reve_d32_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":32} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    rct_w512_things) echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --arch_kwargs {\"d_model\":512,\"depth\":6,\"heads\":8,\"ff_mult\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # rct_fm_things at REVE-6 size (19.8M): does width close the gap?
    rct_fm_things)   echo "gen --eeg_encoder_type ScaleRobustChannelTransformer --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
    # source tokens (alljoined_encoders.py): scratch-REVE-sized electrode transformer + 64 latents that cross-attend to
    # the electrodes (per-trial unmixing), same FM recipe as reve_scratch_things
    src_things)      echo "gen --eeg_encoder_type SourceTokenTransformer --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
    # bridge rct_w512_things (= BridgeTransformer defaults, B0) -> REVE-6, one switch group per step (alljoined_encoders.py)
    br1_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # B1 = rct_w512 - amplitude input
    br2_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # B2 = B1 - dropout
    br3_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"reve\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # B3 = B2 + REVE backbone (RMSNorm, no biases, GEGLU, independent init, no final norm)
    br4_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"reve\",\"pos\":\"reve\",\"reve_input\":true} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # B4 = B3 + REVE positions + 200 Hz input = REVE clone (vs reve_d6_things)
    br4amp_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":true,\"dropout\":0,\"block\":\"reve\",\"pos\":\"reve\",\"reve_input\":true} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # reverse check: the REVE clone (B4) WITH the amplitude input
    # leave-one-out split of the bridge B2 -> B3 backbone swap (_SwitchBlock, verified == REVE backbone / torch layer)
    sw0_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 = B3 rebuilt from _SwitchBlock at REVE settings (replicate of B3)
    sw1_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"norm\":\"layer\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with LayerNorm instead of RMSNorm
    sw2_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"bias\":true} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with biases
    sw3_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"ff\":\"relu\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with the ReLU 4x feed-forward instead of GEGLU
    sw4_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"init\":\"cloned\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with every layer initialised as a copy of one (nn.TransformerEncoder)
    sw6_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"norm\":\"layer\",\"bias\":true,\"init\":\"cloned\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with the three neutral switches (LayerNorm, biases, cloned init) put back at once: only GEGLU + no final norm left of REVE
    sw5_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"amp_input\":false,\"dropout\":0,\"block\":\"switch\",\"final_norm\":true} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # S0 with a final LayerNorm before pooling
    # REVE authors' LP-FT (arXiv 2510.21585 §3.3, reve_eeg dt.py): linear probe on the frozen pretrained backbone, then a
    # short low-lr fine-tune from it. Run through neurips_challenge/overnight/lpft.sh, which passes the probe's checkpoint.
    reve_lp_things) echo "reve --num_epochs 20 --stop_after_epoch 20 --fm_head_warmup_epochs 20 --learning_rate 1e-3 --weight_decay 0.01 --sealed_early_stop_patience 5 --reve_input_normalization trial_zscore" ;;
    reve_ft_things) echo "reve --num_epochs 5 --stop_after_epoch 5 --lr_scheduler none --learning_rate 1e-4 --weight_decay 0.01 --reve_input_normalization trial_zscore" ;;  # needs CV4_EXTRA=--init_checkpoint
    # the same pair with the rest of the authors' recipe: their mixup, flatten head (query from the pretrained token),
    # head dropout 0.3 probe / 0.5 fine-tune (paper+config), StableAdamW, plateau lr (probe: 1-epoch exp warmup and
    # patience 1, scaled from their 10/200 epochs and 5-of-15 patience; fine-tune: their 5, inert in 5 epochs), clip 2, no EMA
    # probe lr 3e-5, not their 1e-3: the flatten Linear's fan-in is 33x the mean head's (16896 vs 512), and at 1e-3 the
    # unnormalised EEG embedding blows the contrastive loss up (1-subject smoke: 30.4 vs 5.1 mean head; 3e-4 7.1, 1e-4 4.9, 3e-5 4.9)
    # 2026-10-09 final-norm transfer (switch leave-one-out: the LayerNorm before pooling cost REVE's block -3.7, p 0.01)
    rct_nofn_things) echo "gen --eeg_encoder_type BridgeTransformer --arch_kwargs {\"final_norm\":false} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # B0 (= rct_w512_things) minus its final LayerNorm
    lab_s10_nofn)    echo "lab  --num_epochs 20 --stop_after_epoch 10 --fm_arch_kwargs {\"conv_out_channels\":16,\"num_heads\":8,\"final_norm\":false}" ;;  # vs lab_s10
    plab_gain_things_nofn) echo "plab --num_epochs 20 --stop_after_epoch 6 --amp_gain_range 2 --fm_arch_kwargs {\"final_norm\":false}" ;;  # vs plab_gain_things
    # fine-tune ladder from the core probe (reve_lp_things): authors' 5-epoch lr-1e-4 stage, + their mixup, or our own fine-tune
    reve_ftmix_things)  echo "reve --num_epochs 5 --stop_after_epoch 5 --lr_scheduler none --learning_rate 1e-4 --weight_decay 0.01 --reve_input_normalization trial_zscore --input_mixup" ;;  # needs CV4_EXTRA=--init_checkpoint
    reve_ftours_things) echo "reve --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore" ;;  # = reve_z_things from the probe; needs CV4_EXTRA=--init_checkpoint
    # 2026-10-09 REVE lessons -> TSConv: per-trial z-score input; a per-dataset spatial layer so TSConv can use THINGS
    tsc_inorm)        echo "tsc  --num_epochs 12 --eeg_instance_norm" ;;  # vs tsc
    tsc_e20_things)   echo "tsc  --num_epochs 20" ;;  # THINGS at a fair budget (tsc_things peaked at its last of 12)
    tsc_dbn_things)   echo "tsc  --num_epochs 20 --tsconv_domain_bn_boundary 21" ;;  # + BatchNorm per dataset (THINGS = IDs 21-30)
    tsc_dbn40_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;  # tsc_dbn_things peaked at its last of 20: one-cycle stretched over 40
    tsc_dbn40_bt_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --batch_transform" ;;  # trained AND evaluated under the batch transform (ROUTING_LOG E13); vs tsc_dbn40_things + post-hoc transform
    tsc_dbn40_btm_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --batch_transform --batch_transform_mixed 64" ;;  # trained under the SHUFFLED-batch transform (ROUTING_LOG E25); vs tsc_dbn40_things + post-hoc mixed transform
    sqf_dbn_things)   echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;  # Squeezeformer + THINGS, every BatchNorm per dataset (cf. tsc_dbn40_things); vs sqf
    atm_dbn_things)   echo "atm  --num_epochs 20 --tsconv_domain_bn_boundary 21" ;;  # ATM + THINGS, the conv stem's two BatchNorms per dataset (cf. sqf_dbn_things); vs atm
    tsc_dspat_things) echo "tsc  --num_epochs 20 --tsconv_domain_spatial_boundary 21 --tsconv_domain_bn_boundary 21" ;;  # + spatial layer per dataset
    plab_rand_things) echo "plab --num_epochs 20 --stop_after_epoch 6 --amp_gain_range 2 --fm_no_pretrained" ;;  # plab_gain_things' exact recipe and base architecture from a RANDOM init
    reve_d6_mix_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --input_mixup" ;;  # reve_d6_things + the REVE authors' mixup (it lifted their short fine-tune stage by 2.6)
    lab_d6_things) echo "lab  --num_epochs 20 --stop_after_epoch 10 --fm_arch_kwargs {\"conv_out_channels\":16,\"num_heads\":8,\"num_layers\":6}" ;;  # "small LaBraM" (6 of 12 blocks, 12.2M), the LaBraM analogue of REVE-6; vs lab_s10_things
    # 2026-10-12 ROUTING_LOG phase 2: input normalisations to draw more from THINGS (vs reve_d6_things / sqf_dbn_things)
    reve_d6_coral_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --things_coral" ;;  # THINGS trials recoloured to the Alljoined covariance
    reve_d6_car_things)   echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --eeg_car" ;;  # common average reference, train and test
    sqf_dbn_coral_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --things_coral" ;;
    # 2026-10-12 night: EEG-image alignment levers (ALIGN_LOG.md), vs reve_d6_things / tsc_dbn40_things / sqf_dbn_things
    reve_d6_gneg_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --global_negatives_replace" ;;  # softmax over ALL 16540 training images instead of the 127 in-batch negatives
    reve_d6_bn64_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --eeg_backbone_dim 64 --projector bottleneck" ;;  # 64-d code lifted linearly to DINOv2
    tsc_dbn40_gneg_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --global_negatives_replace" ;;
    tsc_dbn40_bn64_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    # 2026-10-14 A22 architecture screens (vs tsc_dbn40_bn64_things / sqf_dbn_things / reve_d6_things)
    tsc_dbn40_bn64gn_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --norm_type group" ;;     # every BatchNorm -> per-trial GroupNorm(1, C)
    tsc_dbn40_bn64in_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --norm_type instance" ;;  # every BatchNorm -> GroupNorm(C, C)
    tsc_dbn40_bn64l2_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --eeg_l2norm --t_learnable --init_temperature 0.05" ;;  # control: unit EEG + learnable global temperature (does the per-trial norm matter?)
    # A22 ideas 1-2 (user-approved 2026-10-14): learned per-trial latency aligner (+-6 samples = 50 ms) on each base;
    # factorised electrode x time attention (Squeezeformer with attention before the spatial collapse)
    tsc_dbn40_bn64lat_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --latency_align 6" ;;
    sqf_dbn_lat_things)  echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --latency_align 6" ;;
    reve_d6_lat_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --latency_align 6" ;;
    stf_dbn_things)      echo "gen --eeg_encoder_type FactorizedSTFormer --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 40 --amp_dtype bfloat16" ;;  # vs sqf_dbn_things
    # A22b (user, 2026-10-14): attention across electrodes on the TSConv stem; light REVE; wide SQF stem
    sta_dbn_things)  echo "gen --eeg_encoder_type FactorizedSTFormer --arch_kwargs {\"domain_bn_boundary\":21,\"time_attn\":false,\"readout\":\"flatten\",\"backbone_dim\":64} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 40 --amp_dtype bfloat16" ;;  # conv along time, attention along space at each time step, TSConv time-locked flatten; vs tsc_dbn40_bn64_things
    etok_dbn_things) echo "gen --eeg_encoder_type ElectrodeTokenTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --num_epochs 40" ;;  # TSConv step 4 -> REVE-style attention over 32 electrode tokens; vs tsc_dbn40_bn64_things
    reve_d6lite_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6,\"embed_dim\":160,\"heads\":4,\"head_dim\":40,\"mlp_dim_ratio\":2.0} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # 1.84M (vs 19.8M)
    sqf_dbn_sf160_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --arch_kwargs {\"stem_filters\":160}" ;;  # 160 stem filters (= token width), vs sqf_dbn_things
    # A24 literature sweep: SAMGA coarse stage = multi-kernel MMD between the (normalised) EEG and image clouds (never run on Alljoined)
    tsc_dbn40_bn64mmd_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --mmd_lambda 0.5" ;;
    reve_d6_mmd_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --mmd_lambda 0.5" ;;
    tsc_dbn40_bn64sfilm_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --subject_adapter_rank 8 --subject_adapter_dropout 0.3" ;;  # A24: per-subject FiLM at the temporal block + subject dropout (SAMGA); unseen subjects take the population path
    # A25 Alljoined-1 integration (user, 2026-10-14): _aj1 = + Alljoined-1 with its own BN path; _aj1a = + input adapter
    tsc_dbn40_bn64_aj1_things|tsc_dbn40_bn64_aj1a_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    sqf_dbn_aj1_things|sqf_dbn_aj1a_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_aj1_things|reve_d6_aj1a_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    # A26 REVE at a ~5M budget (user): shallow-wide vs deep-narrow vs same-depth (vs reve_d6_things 19.7M, reve_d6lite 1.8M)
    reve_d2w416_things)  echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":2,\"embed_dim\":416,\"heads\":8,\"head_dim\":52}" ;;   # 4.87M
    reve_d6w256_things)  echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":6,\"embed_dim\":256,\"heads\":8,\"head_dim\":32}" ;;   # 5.16M
    reve_d10w192_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":10,\"embed_dim\":192,\"heads\":6,\"head_dim\":32}" ;;  # 4.75M
    # 2026-10-15 ARCH_LOG.md: hybrid architecture search
    hfa_things) echo "gen --eeg_encoder_type FilterAttnTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H0: attention across the 40 filters after the spatial conv; vs tsc_dbn40_bn64_things
    hta_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H0: attention over time tokens + TSConv flatten readout; vs tsc_dbn40_bn64_things
    hvs_things) echo "gen --eeg_encoder_type VirtualSensorFormer --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H0: REVE patch tokens on 16 learned virtual sensors; vs tsc_dbn40_bn64_things
    hds_things) echo "gen --eeg_encoder_type DynSpatialTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H0: TSConv + trial-adaptive spatial filter residual; vs tsc_dbn40_bn64_things
    hdp_things) echo "gen --eeg_encoder_type DualPathTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H0: TSConv slow path + fast path (kernel 9, pool 5); vs tsc_dbn40_bn64_things
    hlp_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"learned_pool\":true} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H1: learned depthwise strided pooling (box init) instead of the fixed box; vs tsc_dbn40_bn64_things
    hsm2_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"spatial_mult\":2} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H1: 2 spatial filters per temporal filter (80 maps); vs tsc_dbn40_bn64_things
    hsep_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"post\":\"sepconv\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H1: residual depthwise-temporal (9) + pointwise conv after the spatial conv; vs tsc_dbn40_bn64_things
    hqr_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"readout\":\"query\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H1: 8 learned queries cross-attend to the 36 time steps instead of flatten; vs tsc_dbn40_bn64_things
    hf80_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"filters\":80} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H2: 80 temporal (and spatial) filters instead of 40; vs tsc_dbn40_bn64_things
    hf20_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"filters\":20} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H2: 20 filters; vs tsc_dbn40_bn64_things
    hp49_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"pool\":49,\"stride\":4} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H2: box pool 49 stride 4 (0.4 s smoothing, 18 steps); vs tsc_dbn40_bn64_things
    hk49_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"kernel\":49} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H2: temporal kernel 49 (0.4 s) instead of 25; vs tsc_dbn40_bn64_things
    hk13_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"kernel\":13} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H2: temporal kernel 13; vs tsc_dbn40_bn64_things
    hbase_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40 --amp_dtype bfloat16" ;;  # ARCH_LOG control: TSConvPlus defaults = TSConv-bn64 re-implemented, bf16 like every h* run
    hmoe4_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"readout\":\"moe\",\"experts\":4} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H7 (SUBJECT_LOG D4): 4 trial-gated flatten readouts; vs hbase32_things
    hmoe8_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"readout\":\"moe\",\"experts\":8} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H7 (SUBJECT_LOG D4): 8 trial-gated flatten readouts; vs hbase32_things
    hbase32_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # same in fp32 (= the tsc arm)
    hta32_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta in fp32 (bf16 cost ~1.2)
    hfa32_things) echo "gen --eeg_encoder_type FilterAttnTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hfa in fp32
    htsqf_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21,\"block\":\"sqf\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta with Squeezeformer blocks (attention + depthwise conv module + FFN); vs hta32_things
    htgat_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21,\"block\":\"gated\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta with sigmoid-gated attention + SwiGLU; vs hta32_things
    htdif_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21,\"block\":\"diff\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta with differential attention; vs hta32_things
    htd4_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21,\"depth\":4} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta with 4 blocks; vs hta32_things
    htd1_things) echo "gen --eeg_encoder_type TimeAttnFlatTSConv --arch_kwargs {\"domain_bn_boundary\":21,\"depth\":1,\"d_model\":64} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H3: hta with 1 block, d64; vs hta32_things
    hf120_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"filters\":120} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H4: 120 filters (hf80 trend); vs hbase32_things
    hf160_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"filters\":160} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H4: 160 filters; vs hbase32_things
    rvir_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"virtual\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series: REVE tokens on 64 learned virtual channels (5.15M); vs reve_d6w256_things
    rmix_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series: 32 electrode tokens + 5 all-electrode time-patch tokens (5.34M); vs reve_d6w256_things
    rtime_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"time\",\"patch\":6,\"readout\":\"flatten\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series: 20 time-patch tokens, time-locked flatten readout (5.20M); vs reve_d6w256_things
    rmix12_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":12} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rmix with 10 time-patch tokens of 0.1 s; vs rmix_things / relec_things
    rtime_mean_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"time\",\"patch\":6,\"readout\":\"mean\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rtime with the mean readout: is the time-locked flatten what helps?; vs rtime_things
    rtime12_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"time\",\"patch\":12,\"readout\":\"flatten\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rtime with 10 tokens of 0.1 s; vs rtime_things
    rtime3_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"time\",\"patch\":3,\"readout\":\"flatten\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rtime with 40 tokens of 25 ms; vs rtime_things
    rmix40_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch\":40} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rmix with 3 time tokens of 0.33 s; vs rmix_things
    rmixflat_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"readout\":\"flatten\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rmix with the time-locked flatten readout over all 37 tokens; vs rmix_things
    rmixd8_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"d_model\":224,\"depth\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rmix deeper/narrower (d224 x 8); vs rmix_things
    rmix_s2_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --seed 7" ;;  # rmix_things replicate (seed 7): seed noise of the ensemble swap, and a second member
    rmixd12_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"d_model\":192,\"depth\":11,\"heads\":6} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series iteration: rmix deeper still (d192 x 11, ~5.3M); vs rmixd8_things
    rmix_d12w256_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"d_model\":256,\"depth\":12,\"heads\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series scaling ladder (user: params may grow): d256 x 12, ~10M; vs rmixd8_things
    rmix_d8w320_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"d_model\":320,\"depth\":8,\"heads\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series scaling ladder (user: params may grow): d320 x 8, ~10.6M; vs rmixd8_things
    rmix_d16w256_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"d_model\":256,\"depth\":16,\"heads\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series scaling ladder (user: params may grow): d256 x 16, ~13M; vs rmixd8_things
    rmix_split_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"readout\":\"split\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 3 (post-scaling): mean of electrode tokens + time-locked flatten of the 5 time tokens; vs rmix_things
    rmix_ov_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"stride\":12} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 3 (post-scaling): 9 overlapping 0.2 s time tokens (stride 0.1 s); vs rmix_things
    rmix_ms_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"patch2\":12} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 3 (post-scaling): multi-scale time tokens: 5 x 0.2 s + 10 x 0.1 s; vs rmix_things
    rmix_tsc_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"time_source\":\"tsc\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 4 (REVE x TSConv hybrid): REVE electrode tokens + 36 TSConv-stem time tokens (40 filters each); vs rmix_things
    rmix_tsc9_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"time_source\":\"tsc\",\"tsc_pool\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 4 (REVE x TSConv hybrid): REVE electrode tokens + 9 TSConv-stem time tokens (stem output avg-pooled x4); vs rmix_things
    rmix_sqf_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"time_conv\":3} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 4: rmix + Squeezeformer conv module on the time tokens (REVE x SQF); vs rmix_things
    rmix_bip_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"attn\":\"bipartite\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 5 (attention mechanism): bipartite attention: each token attends only to itself + the other token type; vs rmix_things
    rmix_gate_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"attn\":\"gated\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 5 (attention mechanism): gated attention: sigmoid(W x) gate on every attention output; vs rmix_things
    rmix_reg_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 5 (attention mechanism): + 4 learned register tokens (shared workspace); vs rmix_things
    rmix_ep_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"elec_patch\":24} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 6: 160 electrode x 0.2 s patch tokens (factorised positions) + 5 global time tokens; vs rmix_things
    rmix_tffn_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"type_ffn\":true,\"d_model\":224,\"depth\":5} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 6: split paths: joint attention, separate FFN experts for electrode vs time tokens (d224 x 5); vs rmix_things
    rmix_apool_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"readout\":\"attnstat\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series round 7: rmix + Squeezeformer attention-stats readout; vs rmix_things
    sqfmix_things) echo "gen --eeg_encoder_type SqfMix --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG round 8: sqf_dbn + 32 REVE-style electrode tokens (SQF x rmix); vs sqf_dbn_things
    rmix_filt_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"time_source\":\"tsc\",\"tsc_axis\":\"filter\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG round 9: 32 electrode tokens + 40 TSConv-stem FILTER tokens (attention across filters); vs rmix_things
    rmix_cemb_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"elec_embed\":\"conv\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG round 10: electrode tokens embedded by TSConv temporal filters (16 x 25 taps, 0.2 s pool) instead of one Linear; vs rmix_things
    rmix_union_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"filt_tokens\":true} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG round 10: 32 electrode + 5 time-patch + 40 TSConv-stem filter tokens (union of the token sets); vs rmix_things
    relec_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"electrode\"} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # ARCH_LOG R-series: control = br3 at d256 (5.14M); vs reve_d6w256_things
    htfx_things) echo "gen --eeg_encoder_type TimeFreqXTSConv --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H5: time tokens cross-attend to the trial's own spectrum tokens; vs hbase32_things
    hmax_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"pool_type\":\"max\"} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H6: max pooling (25, stride 2) instead of the box average; vs hbase32_things
    hts2_things) echo "gen --eeg_encoder_type TSConvPlus --arch_kwargs {\"domain_bn_boundary\":21,\"temporal_depth\":2,\"kernel\":13} --tsconv_domain_bn_boundary 21 --batch_size 1024 --num_epochs 40" ;;  # ARCH_LOG H6: two stacked temporal convs (13 taps each, depthwise 2nd, ELU) before the box pool; vs hbase32_things
    tsc_dbn40_bn64_dupctl_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --extra_trials_dir /nasbrain/p20fores/alljoined/dupcontrol_120Hz --extra_subject_offset 0" ;;  # A25 control: 49,540 sub-01 trials (THINGS targets) through the extra-trial path; should equal tsc_dbn40_bn64_things
    tsc_dbn40_bn64gelu_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --tsconv_activation gelu" ;;
    sqf_dbn_gn_things)    echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --norm_type group" ;;
    sqf_dbn_noffn_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --arch_kwargs {\"ffn\":false}" ;;
    reve_d6_p40_things)   echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":6,\"patch_size\":40,\"patch_overlap\":0}" ;;   # 5 x 0.2 s patches per electrode: electrode x time tokens
    reve_d6_p50_things)   echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":6,\"patch_size\":50,\"patch_overlap\":25}" ;;  # 7 half-overlapping 0.25 s patches
    reve_d6_clora_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_arch_kwargs {\"depth\":6,\"chan_lora\":8}" ;;  # + per-electrode rank-8 patch-embedding correction
    sqf_dbn_gneg_things)  echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --global_negatives_replace" ;;
    reve_d6_l16_things)   echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_layer16_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" ;;  # + training-only head aligned to DINOv2 layer-16 (mid-level, "EEG-visible") features
    tsc_dbn40_l16_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_layer16_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" ;;
    tsc_dbn40_iaug_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --image_aug --aug_image_feature_dirs /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Original /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianBlur /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/LowResolution /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Mosaic /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianNoise" ;;  # targets = DINOv2 of a random view: original / blur / low-res / mosaic / noise (UBP-like)
    reve_d6_iaug_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --image_aug --aug_image_feature_dirs /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Original /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianBlur /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/LowResolution /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Mosaic /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/GaussianNoise" ;;
    tsc_dbn40_rkd_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --relational_lambda 0.5" ;;  # + EEG-EEG vs image-image similarity-structure loss (RKD / CKA-like)
    tsc_dbn40_gnegc_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --global_negatives_replace --global_concept_eps 0.2" ;;  # + 20% of the target on the concept's other 9 images
    tsc_dbn40_gsoft_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --global_negatives_replace --global_soft_tau 0.02" ;;  # target = image-image similarity over all training images
    tsc_dbn40_softclip_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --soft_target_tau 0.02" ;;  # in-batch SoftCLIP (MindEye2): softmax(image-image cos / .02) targets
    tsc_dbn40_siglip_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --siglip" ;;  # pairwise sigmoid loss
    tsc_dbn40_gsym_things)     echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --global_negatives_replace --global_symmetric 0.5" ;;  # global negatives + image->EEG term (anti-hub)
    tsc_dbn40_ml_things)  echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L8_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L13_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L20_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L40_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" ;;  # multi-scale: training-only heads on DINOv2 layers 8 / 13 / 20 / 40 (track1 = 26), same preprocessing as the grader
    reve_d6_ml_things)    echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L8_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L13_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L20_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L40_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" ;;
    tsc_dbn40_blur_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --image_aug --aug_image_feature_dirs /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Original /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur4 /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur8 /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur16" ;;  # UBP-like: target = track1 layer of the image blurred at sigma 0 / 4 / 8 / 16 px (random)
    reve_d6_blur_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --image_aug --aug_image_feature_dirs /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Original /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur4 /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur8 /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1_aug/Blur16" ;;
    tsc_dbn40_ravg_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --rep_avg_prob 0.3" ;;  # 30% of trials replaced by the mean of 2-4 own repeats (denoised view)
    reve_d6_ravg_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --rep_avg_prob 0.3" ;;
    tsc_dbn40_trunc_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --loss_trunc 0.2" ;;  # drop the 20% highest-loss trials of each batch
    reve_d6_trunc_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --loss_trunc 0.2" ;;
    tsc_dbn40_bn32_things)  echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 32 --projector bottleneck" ;;
    tsc_dbn40_bn128_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 128 --projector bottleneck" ;;
    sqf_dbn_bn64_things)    echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    tsc_dbn40_s2_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --seed 7" ;;  # tsc_dbn40_things replicate (seed 7 vs 33): run-to-run noise control for the alignment screens
    tsc_dbn40_bn64do_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck_do" ;;  # + dropout .3 on the 64-d code
    reve_d6_bn256_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --eeg_backbone_dim 256 --projector bottleneck" ;;
    tsc_dbn40_mcn_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --mask_concept_neg" ;;  # same-concept images are not negatives
    reve_d6_mcn_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --mask_concept_neg" ;;
    tsc_dbn40_bn64mcn_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --mask_concept_neg" ;;
    reve_d6_flat_things)     echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_head flatten" ;;  # REVE's own head (query pool + every token, flattened): keeps the channel x time layout
    reve_d6_flatbn64_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_head flatten --eeg_backbone_dim 64 --projector bottleneck" ;;  # same wide head squeezed through a 64-d code (cf. TSConv bn64)
    tsc_dbn40_smix_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --soft_target_tau 0.05 --soft_mix_beta 0.3" ;;  # NeuroCLIP soft targets: .7 one-hot + .3 softmax(img-img cos/.05)
    reve_d6_smix_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --soft_target_tau 0.05 --soft_mix_beta 0.3" ;;
    tsc_dbn60_bn64_things) echo "tsc  --num_epochs 60 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;  # tsc_dbn40_bn64_things peaked at its last epoch (40): one-cycle stretched to 60
    # head anatomy (ALIGN_LOG A13): what in the bottleneck helps -- capacity, or freeing the output norm?
    tsc_dbn40_noln_things)    echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_head_no_ln" ;;  # full 1536 head, no final LayerNorm (free output norm)
    tsc_dbn40_tb512_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_backbone_dim 512" ;;  # 512 trunk + LN, then a free 512->1536 lift
    tsc_dbn40_tb256_things)   echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_backbone_dim 256" ;;
    tsc_dbn40_bn64noln_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --tsconv_head_no_ln" ;;  # bottleneck without the LayerNorm on the code
    reve_d6_mlphead_things)   echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_head mlp" ;;  # REVE mean tokens -> TSConv/SQF-style head trunk (Linear, residual MLP, LN) -> free lift
    sqf_dbn_smix_things)  echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --soft_target_tau 0.05 --soft_mix_beta 0.3" ;;  # NeuroCLIP soft targets on Squeezeformer
    sqf_dbn_ml_things)   echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L8_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L13_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L20_mean,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_L40_mean --auxiliary_image_targets --auxiliary_alignment_dims 512 --auxiliary_alignment_lambda 0.5" ;;  # multi-scale aux heads on Squeezeformer
    reve_d6_siglip_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --siglip" ;;
    sqf_dbn_tb64_things)  echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_backbone_dim 64" ;;  # the TRUE TSConv-bn64 analogue: head trunk (incl. residual MLP) at 64, then 64->1536
    reve_d6_attnstat_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_head attnstat" ;;  # SQF-style attention mean+std pooling
    # regularisation where the capacity sits (ALIGN_LOG A14): EMA for the conv models, weight decay / channel dropout for REVE
    sqf_dbn_ema_things)       echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --ema_decay 0.9995 --ema_warmup" ;;
    tsc_dbn40_bn64ema_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --ema_decay 0.9995 --ema_warmup" ;;
    tsc_dbn40_bn64mpos_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --grouped_batch_sampler --multi_positive_loss" ;;  # SCORE 2608.19134: several subjects per image per batch, all same-image trials positives; vs tsc_dbn40_bn64_things
    sqf_dbn_mpos_things)      echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --grouped_batch_sampler --multi_positive_loss" ;;  # vs sqf_dbn_things
    reve_d6_mpos_things)      echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --grouped_batch_sampler --multi_positive_loss" ;;  # vs reve_d6_things
    reve_d6_dp1_things)       echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_drop_path 0.1" ;;  # stochastic depth (REVE has dropout 0); vs reve_d6_things
    reve_d6_dp2_things)       echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --fm_drop_path 0.2" ;;
    # 2026-10-13 A16: more training subjects (THINGS-EEG1, 48 subj x 1 rep of EEG2's training images) -- vs the same recipe without _e1
    tsc_dbn40_bn64_e1_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    sqf_dbn_e1_things)        echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_e1_things)        echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_d3_e1_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_domain_bn_boundary2 31 --eeg_backbone_dim 64 --projector bottleneck" ;;  # EEG1 gets its own BatchNorm path
    sqf_dbn_d3_e1_things)     echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_domain_bn_boundary2 31" ;;
    tsc_dbn40_bn64_e1m_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    sqf_dbn_e1m_things)       echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_e1m_things)       echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    pt_e1_tsc)  echo "tsc  --num_epochs 20 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;  # A16c pretraining, EEG1 sub-31..42 only
    pt_e1_sqf)  echo "sqf  --num_epochs 20 --tsconv_domain_bn_boundary 21" ;;
    pt_e1_reve) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_pte1_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;  # + --init_checkpoint pt_e1_tsc
    sqf_dbn_pte1_things)      echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_pte1_things)      echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_d3_e1w_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_domain_bn_boundary2 31 --eeg_backbone_dim 64 --projector bottleneck" ;;  # windowed EEG1 + its own BN
    sqf_dbn_d3_e1w_things)    echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --tsconv_domain_bn_boundary2 31" ;;
    reve_d6_e1w_things)       echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_e1s_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    sqf_dbn_e1s_things)     echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_e1s_things)     echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_e1r_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    sqf_dbn_e1r_things)     echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_e1r_things)     echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    tsc_dbn40_bn64_kd_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;  # + fold-ensemble KD (cv4.sh adds the flags)
    sqf_dbn_kd_things)        echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;
    reve_d6_kd_things)        echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;
    reve_d6_kdx_things)       echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # TTA_LOG T36: KD from the 10-member teacher
    tsc_dbn40_bn64_kd3_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;  # KD at lambda 0.3
    reve_d6_wd02_things)      echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --weight_decay 0.2" ;;
    reve_d6_chdrop_things)    echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --eeg_aug --eeg_aug_type channel_dropout" ;;
    reve_zrand_things) echo "reve --num_epochs 10 --stop_after_epoch 6 --reve_input_normalization trial_zscore --fm_no_pretrained" ;;  # reve_z_things' exact fine-tune from a RANDOM init: does pretraining add anything under our recipe?
    reve_lpa_things) echo "reve --num_epochs 20 --stop_after_epoch 20 --fm_head_warmup_epochs 20 --learning_rate 3e-5 --weight_decay 0.01 --sealed_early_stop_patience 5 --reve_input_normalization trial_zscore --fm_head flatten --fm_head_dropout 0.3 --input_mixup --optimizer stableadamw --lr_scheduler plateau --plateau_patience 1 --warmup_epochs 1 --ema_decay 0 --grad_clip 2" ;;
    reve_fta_things) echo "reve --num_epochs 5 --stop_after_epoch 5 --learning_rate 1e-4 --weight_decay 0.01 --reve_input_normalization trial_zscore --fm_head flatten --fm_head_dropout 0.5 --input_mixup --optimizer stableadamw --lr_scheduler plateau --plateau_patience 5 --ema_decay 0 --grad_clip 2" ;;  # needs CV4_EXTRA=--init_checkpoint
    # 2026-10-26 AUG_LOG: subject-invariance augmentations on the existing encoders (partners from the same dataset); vs reve_d6_things
    reve_d6_nswap_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --noise_swap_prob 0.5" ;;  # noise transplant: own evoked mean + another subject's single-trial residual
    reve_d6_optx_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5" ;;  # re-render the batch through another subject's channel covariance operator
    reve_d6_xmix_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --xsubj_mix_prob 0.5" ;;  # SubjectMix: Beta-mix with a same-image trial of another subject
    reve_d6_hyal_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --hyperalign_prob 0.5" ;;  # per trial: Procrustes rotation to another subject, fitted on repeat-averaged shared-image responses (hyperalign_maps.py)
    reve_d6_swa4_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --swa_from_epoch 4" ;;  # SWAD-style uniform weight average of every step from epoch 4 (replaces the 0.9998 EMA there)
    reve_d6_bswap_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --band_swap_prob 0.5" ;;  # 8-30 Hz (identity band) taken from another subject's random trial
    rmix_reg_xmix_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --xsubj_mix_prob 0.5" ;;  # SubjectMix on the rmix_reg_things member (AUG_LOG)
    sqf_dbn_xmix_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --xsubj_mix_prob 0.5" ;;  # SubjectMix on the sqf_dbn_things member (AUG_LOG)
    tsc_dbn40_bn64_xmix_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --xsubj_mix_prob 0.5" ;;  # SubjectMix on the tsc_dbn40_bn64_things member (AUG_LOG)
    reve_scratch_xmix_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --xsubj_mix_prob 0.5" ;;  # SubjectMix on the reve_scratch_things member (AUG_LOG)
    reve_d6_sam1_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --sam_rho 0.1" ;;  # sharpness-aware minimisation, ascent along the previous step's gradient (free)
    reve_d6_lp8_things)   echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --input_lowpass_hz 8" ;;  # input low-passed below 8 Hz (train and test): drops the alpha/beta identity band
    rmix_reg_optx_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5" ;;  # channel-operator transfer on the rmix_reg_things member (AUG_LOG)
    sqf_dbn_optx_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --op_transfer_prob 0.5" ;;  # channel-operator transfer on the sqf_dbn_things member (AUG_LOG)
    tsc_dbn40_bn64_optx_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --op_transfer_prob 0.5" ;;  # channel-operator transfer on the tsc_dbn40_bn64_things member (AUG_LOG)
    reve_scratch_optx_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5" ;;  # channel-operator transfer on the reve_scratch_things member (AUG_LOG)
    reve_d6_optxe2_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5 --op_transfer_extrapolate 2" ;;  # target operator at u~U(0,2) along the SPD geodesic toward the partner (past it: virtual subjects)
    reve_d6_optx14_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 14 --reve_input_normalization trial_zscore --op_transfer_prob 0.5" ;;  # reve_d6_optx was still rising at its stop epoch 10
    reve_d6_optxp1_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 1.0" ;;  # AUG_LOG op-transfer variant: op transfer on every batch
    reve_d6_optxfb_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5 --op_transfer_bands 0 4 8 13 30 61" ;;  # AUG_LOG op-transfer variant: per-band operator transfer (own covariance per band)
    reve_d6_optxxd_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5 --op_transfer_cross_dataset" ;;  # AUG_LOG op-transfer variant: Alljoined <-> THINGS-EEG2 partners allowed
    reve_d6_optxrot_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --op_transfer_prob 0.5 --op_transfer_rotate" ;;  # AUG_LOG op-transfer variant: control: partner spectrum, random spatial orientation
    reve_d6_nb5_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --neighbour_batch_prob 0.5" ;;  # AUG_LOG (2026-10-28): half the batches = one k-means cluster of DINOv2 targets (fine-grained negatives)
    reve_d6_optxnb5_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --neighbour_batch_prob 0.5 --op_transfer_prob 0.5" ;;  # AUG_LOG (2026-10-28): neighbour batches + op transfer; half the batches = one k-means cluster of DINOv2 targets (fine-grained negatives)
    reve_d6_oc10_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 10 --reve_input_normalization trial_zscore" ;;  # AUG_LOG training audit (2026-10-28): full one-cycle anneal over 10 epochs (base stops a 20-ep cycle at 10, lr still ~5e-4)
    reve_d6_oc20_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 20 --reve_input_normalization trial_zscore" ;;  # AUG_LOG training audit (2026-10-28): full 20-epoch one-cycle
    reve_d6_lr5_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 5e-4 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # AUG_LOG training audit (2026-10-28): peak lr 5e-4 (f2/f4 decline after e6)
    sqf_dbn60_things) echo "sqf  --num_epochs 60 --tsconv_domain_bn_boundary 21" ;;  # AUG_LOG training audit (2026-10-28): 60-epoch one-cycle (still rising at e40)
    reve_d6_wd01_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --weight_decay 0.01" ;;  # AUG_LOG training audit: weight decay 0.01 (0.2 hurt)
    reve_d6_bs256_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --batch_size 256" ;;  # AUG_LOG training audit: batch 256 (more in-batch negatives)
    rmix_reg_bs256_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 256 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG training audit: batch 256 (REVE-d6 bs256 singles +0.0/+0.8/+2.9)
    reve_scratch_bs256_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --batch_size 256" ;;  # AUG_LOG training audit: batch 256
    tsc_dbn40_bn64_bs256_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --batch_size 256" ;;  # AUG_LOG training audit: batch 256 instead of 1024 (4x the steps)
    sqf_dbn_bs256_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --batch_size 256" ;;  # AUG_LOG training audit: batch 256 instead of 1024 (4x the steps)
    reve_d6_optx20_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 20 --reve_input_normalization trial_zscore --op_transfer_prob 0.5" ;;  # AUG_LOG: optx on the full 20-epoch cosine (curves still rising at the e10 stop)
    rmix_reg_optx20_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 20 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5" ;;  # AUG_LOG: optx on the full 20-epoch cosine (curves still rising at the e10 stop)
    reve_scratch_optx20_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 20 --reve_input_normalization trial_zscore --op_transfer_prob 0.5" ;;  # AUG_LOG: optx on the full 20-epoch cosine (curves still rising at the e10 stop)
    reve_d6_nb1_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --neighbour_batch_prob 1.0" ;;  # AUG_LOG (2026-10-28): every batch from one image cluster
    reve_scratch_nb5_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --neighbour_batch_prob 0.5" ;;  # AUG_LOG (2026-10-28): neighbour batches on the REVE-scratch member
    rmix_reg_nb5_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --neighbour_batch_prob 0.5" ;;  # AUG_LOG (2026-10-28): neighbour batches on the rmix-reg member
    tsc_dbn40_bn64_lp8_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --input_lowpass_hz 8" ;;  # TSC member on the < 8 Hz view (AUG_LOG diversity)
    sqf_dbn_lp8_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --input_lowpass_hz 8" ;;  # SQF member on the < 8 Hz view
    rmix_reg_sam1_things) echo "gen --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --sam_rho 0.1" ;;  # SAM (stale ascent) on the rmix_reg_things member (AUG_LOG)
    sqf_dbn_sam1_things) echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21 --sam_rho 0.1" ;;  # SAM (stale ascent) on the sqf_dbn_things member (AUG_LOG)
    tsc_dbn40_bn64_sam1_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --sam_rho 0.1" ;;  # SAM (stale ascent) on the tsc_dbn40_bn64_things member (AUG_LOG)
    reve_scratch_sam1_things) echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --sam_rho 0.1" ;;  # SAM (stale ascent) on the reve_scratch_things member (AUG_LOG)
    reve_d6_s7_things)    echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --seed 7" ;;  # seed replicate of reve_d6_things: noise floor of an ensemble swap (AUG_LOG)
    reve_d6_vrex1_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --vrex_lambda 1" ;;  # V-REx: variance of the per-subject InfoNCE risk in the batch (invariance penalty)
    reve_d6_mse1_things)  echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --alpha 0.9" ;;  # loss = .9 InfoNCE + .1 MSE to the target (different loss geometry -> decorrelated errors? AUG_LOG)
    tsc_dbn40_bn64_siglip_things) echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --siglip" ;;  # TSC member with the pairwise sigmoid loss (loss-geometry diversity, AUG_LOG)
    reve_d6_nc_things)    echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # + negative correlation to the other 4 members (cv4.sh adds the flags; AUG_LOG)
    reve_d6_nc3_things)   echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore" ;;  # negative correlation, lambda 3
    # *_b<E>: a reference recipe stopped at its cv4 budget epoch E (same schedule), so its saved weights are the
    # budget epoch's (ensembles, submissions). lab = lab_s10 stopped at its budget 6.
    tsc_b11)       echo "tsc  --num_epochs 12 --stop_after_epoch 11" ;;
    plab_gain_b3)  echo "plab --num_epochs 20 --stop_after_epoch 3 --amp_gain_range 2" ;;
    reve_z_b3)     echo "reve --num_epochs 10 --stop_after_epoch 3 --reve_input_normalization trial_zscore" ;;
    sqf_b26)       echo "sqf  --num_epochs 30 --stop_after_epoch 26" ;;
    atm_b8)        echo "atm  --num_epochs 20 --stop_after_epoch 8" ;;
    # AUG_LOG 2026-10-29 decorrelated-encoder search: any braindecode model, bd_<Model>_things (Braindecode wrapper,
    # ensemble_experiments/architectures/braindecode_encoders.py; per-trial z-score input), 20-epoch one-cycle screen
    ff_things)    echo "gen --eeg_encoder_type FourierFormer --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG own design: transformer over frequency tokens (REVE-family recipe)
    ffr_things)   echo "gen --eeg_encoder_type FourierFormer --arch_kwargs {\"dropout\":0.2} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup --op_transfer_prob 0.5" ;;  # AUG_LOG: ff (plateau from e5) + dropout .2 + op transfer
    ffd_things)   echo "gen --eeg_encoder_type FourierFormer --arch_kwargs {\"d_model\":384,\"depth\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG: wider/deeper FourierFormer, no op transfer
    ff16_things)  echo "gen --eeg_encoder_type FourierFormer --arch_kwargs {\"max_hz\":16} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG: FourierFormer on 0-16 Hz
    ff40_things)  echo "gen --eeg_encoder_type FourierFormer --arch_kwargs {\"max_hz\":40} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG: FourierFormer on 0-40 Hz
    gruwd_things) echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384,\"dropout\":0.4} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 12" ;;  # AUG_LOG: gruwr WITHOUT op transfer (does optx make it REVE-like?)
    spf_things)   echo "gen --eeg_encoder_type SpecPatchFormer --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG own design: electrode x frequency-band tokens (REVE bias in the frequency domain)
    ddf_things)   echo "gen --eeg_encoder_type DualDomainFormer --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG own design: time-patch + frequency tokens in one transformer
    spfb_things)  echo "gen --eeg_encoder_type SpecPatchFormer --arch_kwargs {\"d_model\":384,\"depth\":8} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG: larger SpecPatchFormer
    spf3_things)  echo "gen --eeg_encoder_type SpecPatchFormer --arch_kwargs {\"patch\":3} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;  # AUG_LOG: SpecPatchFormer with 3-bin (3 Hz) bands
    dcn_bst5_things|dcn_bst10_things|dcn_ibst10_things|dcn_ibst20_things|dcn_ibst40_things) echo "gen --eeg_encoder_type DilatedConvNeXt1D --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 30" ;;  # AUG_LOG boosted dcn (gamma 2), boost flags added by the _bst branch
    dcn_things)   echo "gen --eeg_encoder_type DilatedConvNeXt1D --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 30" ;;  # AUG_LOG own design: deep dilated ConvNeXt-1D
    dcnph_things) echo "gen --eeg_encoder_type DilatedConvNeXt1D --arch_kwargs {\"domain_bn_boundary\":21,\"input_mode\":\"phase\"} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 30" ;;  # AUG_LOG own design: phase-only input (delta/theta/alpha analytic phase)
    gruwr_things) echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384,\"dropout\":0.4} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 12 --op_transfer_prob 0.5" ;;  # AUG_LOG: gruw (peaked e10 then overfit) + dropout .4, op transfer, 12-ep cycle
    lstmwr_things) echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384,\"dropout\":0.4,\"cell\":\"lstm\"} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 12 --op_transfer_prob 0.5" ;;  # AUG_LOG: gruwr with an LSTM cell
    gruwrc_things) echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384,\"dropout\":0.4,\"bidirectional\":false} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 12 --op_transfer_prob 0.5" ;;  # AUG_LOG: gruwr, causal (forward-only) recurrence
    grulate_things) echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 20 --time_window 30 120" ;;  # AUG_LOG: late-window specialist (0.25-1 s only), control = gruw_things
    dcnlate_things) echo "gen --eeg_encoder_type DilatedConvNeXt1D --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 30 --time_window 30 120" ;;  # AUG_LOG: late-window specialist on the dcn trunk, control = dcn_things
    gru_things)   echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 20" ;;  # AUG_LOG: recurrent member (BiGRU over learned spatial filters)
    gruw_things)  echo "gen --eeg_encoder_type GRUEncoder --arch_kwargs {\"domain_bn_boundary\":21,\"n_filters\":128,\"hidden\":384} --tsconv_domain_bn_boundary 21 --batch_size 512 --num_epochs 20" ;;  # wider GRU member
    reve_d6_xIV28w[0-9]*_things) w=${1#reve_d6_xIV28w}; echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --reve_input_normalization trial_zscore --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit --auxiliary_image_targets --auxiliary_alignment_dims 3200 --auxiliary_fixed_targets --auxiliary_alignment_lambda ${w%_things}" ;;  # AUG_LOG 2026-10-30 user idea (b): REVE-d6 on the grader layer + a head aligned to the TRUE InternViT L28 embedding, weight w; only the grader head is kept
    reve_d6_wsIV28_things) echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 2e-4 --num_epochs 4 --reve_input_normalization trial_zscore" ;;  # AUG_LOG 2026-10-30 (a) warm start: the InternViT-L28 member, head composed with its stacking bridge (warmstart_ckpt.py), fine-tuned on the grader layer; --init_checkpoint added per fold below
    *_sd[0-9]*_things) n=${1##*_sd}; s=$(recipe ${1%_sd*}_things) || return 1; echo "$s --seed ${n%_things}" ;;  # AUG_LOG 2026-11-02 seed replicate of any recipe (6th member = a second seed of a decorrelated member)
    *_xIV28w[0-9]*_things) w=${1##*_xIV28w}; s=$(recipe ${1%%_xIV28w*}_things) || return 1; echo "$s --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1,/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit --auxiliary_image_targets --auxiliary_alignment_dims 3200 --auxiliary_fixed_targets --auxiliary_alignment_lambda ${w%_things}" ;;  # AUG_LOG 2026-10-30 (b) on any member: + a head aligned to the TRUE InternViT L28 embedding, weight w
    *_t[A-Z]*_things) b=${1%_things}; t=${b##*_t}; s=$(recipe ${b%_t*}_things) || return 1  # AUG_LOG 2026-10-30 target diversity: <member>_t<L13|L16|L20|L33|L40|RN50|VITH|IV28|EVA>_things = that member trained on another image target; tta_cache.py bridges its output to the grader layer
      case $t in L16) d=dinov2-giant_layer16_mean ;; RN50) d="RN50 --feature_dim 1024" ;; VITH) d="ViT-H-14_final --feature_dim 1024" ;;
        IV[0-9]*) d="InternViT-6B_layer${t#IV}_mean_8bit --feature_dim 3200" ;; BIGG) d="ViT-bigG-14_layer27_mean --feature_dim 1664" ;; EVA) d="EVA02-E-14_layer35_mean --feature_dim 1792" ;; L[0-9]*) d=dinov2-giant_${t}_mean ;; *) return 1 ;; esac
      echo "$s --image_feature_dir /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/$d" ;;
    bd_*_things) m=${1#bd_}; m=${m%_things}; echo "gen --eeg_encoder_type Braindecode --arch_kwargs {\"model\":\"$m\"} --batch_size 512 --num_epochs 20" ;;
    *) return 1 ;;
  esac
}
RECIPES="tsc sqf rmix_cemb_things rmix_union_things rmix_filt_things sqfmix_things rmix_apool_things rmix_ep_things rmix_tffn_things rmix_bip_things rmix_gate_things rmix_reg_things rmix_sqf_things rmix_tsc_things rmix_tsc9_things rmix_split_things rmix_ov_things rmix_ms_things rmix_d12w256_things rmix_d8w320_things rmix_d16w256_things rmixd12_things rmix_s2_things rmixd8_things hmoe4_things hmoe8_things rmix40_things rmixflat_things rtime_mean_things rtime12_things rtime3_things rmix12_things rvir_things rmix_things rtime_things relec_things hmax_things hts2_things htfx_things hf120_things hf160_things htsqf_things htgat_things htdif_things htd4_things htd1_things hta32_things hfa32_things hbase_things hbase32_things hf80_things hf20_things hp49_things hk49_things hk13_things hlp_things hsm2_things hsep_things hqr_things hfa_things hta_things hvs_things hds_things hdp_things tsc_dbn40_bn64_dupctl_things reve_d2w416_things reve_d6w256_things reve_d10w192_things tsc_dbn40_bn64_aj1_things tsc_dbn40_bn64_aj1a_things sqf_dbn_aj1_things sqf_dbn_aj1a_things reve_d6_aj1_things reve_d6_aj1a_things tsc_dbn40_bn64sfilm_things tsc_dbn40_bn64mmd_things reve_d6_mmd_things sta_dbn_things etok_dbn_things reve_d6lite_things sqf_dbn_sf160_things tsc_dbn40_bn64lat_things sqf_dbn_lat_things reve_d6_lat_things stf_dbn_things tsc_dbn40_bn64l2_things tsc_dbn40_bn64gn_things tsc_dbn40_bn64in_things tsc_dbn40_bn64gelu_things sqf_dbn_gn_things sqf_dbn_noffn_things reve_d6_p40_things reve_d6_p50_things reve_d6_clora_things atm lab plab_gain reve_z lab_s10 plab_gain_s10 plab_gain_things tsc_things reve_z_things eegnet mixer gct conformer cbramod_gain fbtn dualstream robust_ct robust_ct_e40 tsc_lat tsc_stj tsc_rgrp tsc_rmean tsc_sub64 tsc_nbr tsc_mixstyle rct_stj rct_rmean rct_sub64 tsc_sub64c rct_sub64c rct_nbr reve_z_nbr_things reve_scratch_things reve_lr1e4_things reve_llrd_things reve_oc4_things reve_scratch lab_s10_things reve_d6 reve_d6_things reve_d32_things rct_fm_things rct_w512_things br1_things br2_things br3_things br4_things br4amp_things sw0_things sw1_things sw2_things sw3_things sw4_things sw5_things sw6_things reve_lp_things reve_ft_things reve_lpa_things reve_fta_things rct_nofn_things lab_s10_nofn plab_gain_things_nofn reve_ftmix_things reve_ftours_things reve_zrand_things lab_d6_things reve_d6_mix_things plab_rand_things tsc_inorm tsc_e20_things tsc_dbn_things tsc_dspat_things tsc_dbn40_things sqf_dbn_things atm_dbn_things tsc_dbn40_bt_things tsc_dbn40_btm_things reve_d6_coral_things reve_d6_car_things sqf_dbn_coral_things reve_d6_gneg_things reve_d6_bn64_things tsc_dbn40_gneg_things tsc_dbn40_bn64_things sqf_dbn_gneg_things reve_d6_l16_things tsc_dbn40_l16_things tsc_dbn40_iaug_things reve_d6_iaug_things tsc_dbn40_rkd_things tsc_dbn40_gnegc_things tsc_dbn40_gsoft_things tsc_dbn40_softclip_things tsc_dbn40_siglip_things tsc_dbn40_gsym_things tsc_dbn40_ml_things reve_d6_ml_things tsc_dbn40_blur_things reve_d6_blur_things tsc_dbn40_ravg_things reve_d6_ravg_things tsc_dbn40_trunc_things reve_d6_trunc_things tsc_dbn40_bn32_things tsc_dbn40_bn128_things sqf_dbn_bn64_things tsc_dbn40_s2_things tsc_dbn40_bn64do_things reve_d6_bn256_things tsc_dbn40_mcn_things reve_d6_mcn_things tsc_dbn40_bn64mcn_things reve_d6_flat_things reve_d6_flatbn64_things tsc_dbn40_smix_things reve_d6_smix_things tsc_dbn60_bn64_things tsc_dbn40_noln_things tsc_dbn40_tb512_things tsc_dbn40_tb256_things tsc_dbn40_bn64noln_things reve_d6_mlphead_things sqf_dbn_smix_things sqf_dbn_ml_things reve_d6_siglip_things sqf_dbn_tb64_things reve_d6_attnstat_things sqf_dbn_ema_things tsc_dbn40_bn64ema_things reve_d6_wd02_things tsc_dbn40_bn64_kd3_things tsc_dbn40_bn64_kd_things sqf_dbn_kd_things reve_d6_kd_things reve_d6_kdx_things tsc_dbn40_bn64_e1s_things sqf_dbn_e1s_things reve_d6_e1s_things tsc_dbn40_bn64_e1r_things sqf_dbn_e1r_things reve_d6_e1r_things tsc_dbn40_bn64_d3_e1w_things sqf_dbn_d3_e1w_things reve_d6_e1w_things pt_e1_tsc pt_e1_sqf pt_e1_reve tsc_dbn40_bn64_pte1_things sqf_dbn_pte1_things reve_d6_pte1_things tsc_dbn40_bn64_e1m_things sqf_dbn_e1m_things reve_d6_e1m_things tsc_dbn40_bn64_d3_e1_things sqf_dbn_d3_e1_things tsc_dbn40_bn64_e1_things sqf_dbn_e1_things reve_d6_e1_things reve_d6_dp1_things reve_d6_dp2_things tsc_dbn40_bn64mpos_things sqf_dbn_mpos_things reve_d6_mpos_things reve_d6_chdrop_things src_things tsc_b11 plab_gain_b3 reve_z_b3 sqf_b26 atm_b8"

if [ "$1" = --list ]; then for r in $RECIPES; do echo "$r: $(recipe $r)"; done; exit 0; fi
NAME=$1; shift
SPEC=$(recipe "$NAME") || { echo "unknown recipe '$NAME' (cv4.sh --list)"; exit 1; }
ARM=${SPEC%% *} FLAGS="${SPEC#* } $CV4_EXTRA"  # CV4_EXTRA: per-run flags a recipe cannot hold (lpft.sh)
[[ $NAME == reve_ft*_things && $CV4_EXTRA != *--init_checkpoint* ]] && { echo "reve_ft_things needs CV4_EXTRA=--init_checkpoint <probe>"; exit 1; }
for K in ${@:-1 2 3 4}; do
  TEST=${FOLDS[$K]}; [ -n "$TEST" ] || { echo "no fold $K"; exit 1; }
  OUT=results/alljoined/track1/cv4/$NAME/fold$K/$ARM
  if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip $NAME fold$K (scored)"; continue; fi
  TRAIN=""; for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || TRAIN+="$s "; done
  if [[ $NAME == *_things ]]; then
    grep -q things_eeg2_subjects /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json \
      || { echo "THINGS-EEG2 subjects not exported (export_things_to_alljoined.py)"; exit 1; }
    # their training caches must exist on the NAS first: built once there, staged like everyone else's,
    # instead of every job rebuilding ~10 GB into its node's /tmp (filled sl-tp-br-514's disk, 2026-10-02)
    [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/things_caches_ok ] \
      || { echo "THINGS-EEG2 training caches not built on the NAS yet"; exit 1; }
    TRAIN+="$(seq -s ' ' 21 30)"
  fi
  if [[ $NAME == *_e1_things ]]; then  # + THINGS-EEG1: 48 subjects packed 4 per pseudo-subject sub-31..42 (export_things_eeg1_to_alljoined.py)
    [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/eeg1_caches_ok ] \
      || { echo "THINGS-EEG1 pseudo-subjects or their caches not built on the NAS yet"; exit 1; }
    TRAIN+=" $(seq -s ' ' 31 42)"
  fi
  if [[ $NAME == *_kdx_things ]]; then  # TTA_LOG T36: distil the fold's 10-member ensemble (the TTA_LOG T22 members)
    T=results/alljoined/track1/cv4/teachers/fold${K}_10m.npy
    [ -f $T ] || .venv/bin/python neurips_challenge/export_teacher_cv4.py $K reve_d6_things sqf_dbn_things tsc_dbn40_bn64_things rmix_things reve_d6_kd_things sqf_dbn_kd_things reve_scratch_things lab_s10_things rmix_reg_things rmixd8_things
    [ -f $T ] || { echo "no teacher $T"; exit 1; }
    cat $T > /dev/null
    FLAGS="$FLAGS --distill_targets $T --distill_lambda 1 --distill_tau 0.005"
  fi
  if [[ $NAME == *_kd_things || $NAME =~ _kd[0-9]+_things$ ]]; then  # A18 (_kdN_: distill_lambda 0.N): distil the fold's 3-model ensemble (export_teacher_cv4.py; fold training subjects only)
    T=results/alljoined/track1/cv4/teachers/fold$K.npy
    mkdir -p "$OUT/.claim" && echo "${SLURM_JOB_ID:-$$}" > "$OUT/.claim/job"  # claim now: building/waiting takes ~30 min and other workers must skip this run
    if [ ! -f $T ] && mkdir results/alljoined/track1/cv4/teachers/.lock$K 2>/dev/null; then
      .venv/bin/python neurips_challenge/export_teacher_cv4.py $K; rmdir results/alljoined/track1/cv4/teachers/.lock$K
    fi
    for i in $(seq 120); do [ -f $T ] && break; sleep 60; done  # another job is building it
    [ -f $T ] || { echo "no teacher $T"; exit 1; }
    cat $T > /dev/null  # 5 GiB of random per-trial reads: warm it into the page cache like the EEG memmaps
    LAM=1; [[ $NAME =~ _kd([0-9]+)_things$ ]] && LAM=0.${BASH_REMATCH[1]}
    FLAGS="$FLAGS --distill_targets $T --distill_lambda $LAM --distill_tau 0.005"
  fi
  if [[ $NAME =~ _(i?)bst([0-9]+)_things$ ]]; then  # AUG_LOG boosting: weight trials by (1 - p_ensemble(true image))^gamma; 4 base members' training predictions
    G=$(awk "BEGIN{print ${BASH_REMATCH[2]}/10}")  # _bst5_ = gamma 0.5
    if [ -n "${BASH_REMATCH[1]}" ]; then  # _ibst: image-level hardness (mean over the image's trials)
      T=results/alljoined/track1/cv4/teachers/fold${K}_4m_pimg.npy; BW=--image
    else  # per-trial 1 - rank/n of the true image (boost_weights.py)
      T=results/alljoined/track1/cv4/teachers/fold${K}_4m_ptrue.npy; BW=
    fi
    [ -f $T ] || [ ! -f results/alljoined/track1/cv4/teachers/fold${K}_4m.npy ] || .venv/bin/python -W ignore neurips_challenge/boost_weights.py $K $BW
    mkdir -p "$OUT/.claim" && echo "${SLURM_JOB_ID:-$$}" > "$OUT/.claim/job"
    if [ ! -f $T ] && mkdir results/alljoined/track1/cv4/teachers/.lock4m$K 2>/dev/null; then
      .venv/bin/python neurips_challenge/export_teacher_cv4.py $K reve_d6_things sqf_dbn_things tsc_dbn40_bn64_things reve_scratch_things; rmdir results/alljoined/track1/cv4/teachers/.lock4m$K
      .venv/bin/python -W ignore neurips_challenge/boost_weights.py $K $BW  # the export writes the 4m rows; T is derived from them
    fi
    for i in $(seq 120); do [ -f $T ] && break; sleep 60; done
    [ -f $T ] || { echo "no members file $T"; exit 1; }
    FLAGS="$FLAGS --distill_targets $T --boost_gamma $G --distill_tau 0.02 --multi_positive_loss"
  fi
  if [[ $NAME == *_wsIV28_things ]]; then  # (a) warm start, per-fold initial checkpoint
    I=results/alljoined/track1/cv4/warmstart/reve_d6_tIV28_f${K}.pth; [ -f $I ] || { echo "no warm-start checkpoint $I"; exit 1; }
    FLAGS="$FLAGS --init_checkpoint $I"
  fi
  if [[ $NAME == *_nc_things || $NAME =~ _nc[0-9]+_things$ ]]; then  # AUG_LOG negative correlation (_ncN_: lambda N): the 4 OTHER members of the 5-set (rmix-reg slot) as the repulsion set
    T=results/alljoined/track1/cv4/teachers/fold${K}_4m.npy
    mkdir -p "$OUT/.claim" && echo "${SLURM_JOB_ID:-$$}" > "$OUT/.claim/job"
    if [ ! -f $T ] && mkdir results/alljoined/track1/cv4/teachers/.lock4m$K 2>/dev/null; then
      .venv/bin/python neurips_challenge/export_teacher_cv4.py $K reve_d6_things sqf_dbn_things tsc_dbn40_bn64_things reve_scratch_things; rmdir results/alljoined/track1/cv4/teachers/.lock4m$K
    fi
    for i in $(seq 120); do [ -f $T ] && break; sleep 60; done
    [ -f $T ] || { echo "no members file $T"; exit 1; }
    cat $T > /dev/null
    NCL=1; [[ $NAME =~ _nc([0-9]+)_things$ ]] && NCL=${BASH_REMATCH[1]}
    FLAGS="$FLAGS --distill_targets $T --anticorr_lambda $NCL --distill_tau 0.02"
  fi
  if [[ $NAME == pt_e1_* ]]; then TRAIN="$(seq -s ' ' 31 42)"; fi  # A16c: pretrain on THINGS-EEG1 only (never a test subject: one run serves every fold)
  if [[ $NAME == *_pte1_things ]]; then  # ... then fine-tune: wait for the pretraining checkpoint instead of silently training from scratch
    PT=results/alljoined/track1/cv4/pt_e1_$ARM/fold1/$ARM
    for i in $(seq 360); do ls $PT/*/track1_score.json >/dev/null 2>&1 && break; sleep 60; done
    CK=$(ls -t $PT/*/checkpoint_test_best.pth 2>/dev/null | head -1)
    [ -n "$CK" ] || { echo "no pretraining checkpoint in $PT"; exit 1; }
    FLAGS="$FLAGS --init_checkpoint $CK"
  fi
  if [[ $NAME == *_e1w_things ]]; then  # THINGS-EEG1 faded to 0 after 0.35-0.45 s (overnight/eeg1_window.py): sub-55..66
    [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/eeg1w_caches_ok ] \
      || { echo "windowed THINGS-EEG1 (sub-55..66) or its caches not built yet"; exit 1; }
    TRAIN+=" $(seq -s ' ' 55 66)"
  fi
  for V in s:67:eeg1s r:79:eeg1r; do  # A17 chimeras (overnight/eeg1_splice.py): *_e1s_things splice sub-67..78, *_e1r_things residual sub-79..90
    IFS=: read -r SFX FIRST MARK <<< "$V"
    if [[ $NAME == *_e1${SFX}_things ]]; then
      for i in $(seq 240); do [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/${MARK}_caches_ok ] && break; sleep 60; done  # still building: wait, never spin
      [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/${MARK}_caches_ok ] \
        || { echo "THINGS-EEG1 chimeras ${MARK} or their caches not built yet"; exit 1; }
      TRAIN+=" $(seq -s ' ' $FIRST $((FIRST + 11)))"
    fi
  done
  if [[ $NAME == *_e1m_things ]]; then  # THINGS-EEG1 minus each subject's grand-average epoch (overnight/eeg1_meansub.py): sub-43..54
    [ -f /nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/.cache/eeg1m_caches_ok ] \
      || { echo "mean-subtracted THINGS-EEG1 (sub-43..54) or its caches not built yet"; exit 1; }
    TRAIN+=" $(seq -s ' ' 43 54)"
  fi
  echo "=== $(date '+%F %T') $NAME fold$K  train: $TRAIN test: $TEST"
  # sealed-rule score of every test subject on a seeded half of its trials (~40 repeats/image), every epoch: cv4_summary.py picks the epoch budget from it
  if [[ $NAME == *_aj1*_things ]]; then  # + Alljoined-1 (NEMAR nm000133, export_alljoined1.py): 49,540 single trials of 960 COCO
    # images, subjects 41..48, appended after the grid (dataset.add_extra_trials); own BatchNorm path for ids >= 41
    [ -f /nasbrain/p20fores/alljoined/alljoined1_120Hz/dinov2_track1.npz ] || { echo "Alljoined-1 export missing"; exit 1; }
    FLAGS="$FLAGS --extra_trials_dir /nasbrain/p20fores/alljoined/alljoined1_120Hz --extra_subject_offset 40 --tsconv_domain_bn_boundary2 41"
    [[ $NAME == *_aj1a_things ]] && FLAGS="$FLAGS --extra_input_adapter"  # + identity-init per-dataset input translator
  fi
  SPLIT=cv4/$NAME/fold$K TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh "$ARM" \
    --select_best_on fixed --val_concept_ratio 0 --sealed_test_every 1 --sealed_test_fraction 0.5 $FLAGS \
    || echo "FAILED $NAME fold$K"
done
echo "done $NAME folds ${@:-1 2 3 4}"
