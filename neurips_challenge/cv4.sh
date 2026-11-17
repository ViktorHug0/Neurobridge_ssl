#!/bin/bash
# The cross-subject evaluation protocol for the sealed phase (new subjects, new images): 4 fixed folds of 5
# Alljoined test subjects, training on the other 15 + the 10 THINGS-EEG2 subjects (sub-21..30,
# export_things_to_alljoined.py). Protocol and decision rules: neurips_challenge/README.md.
# Do not edit FOLDS: every number in results/alljoined/track1/cv4/ is only comparable because the folds never change.
#
#   usage: cv4.sh <recipe> [fold ...]          one recipe on the given folds (default: 1 2 3 4)
#          cv4.sh --list                       print the recipes
# Results: results/alljoined/track1/cv4/<recipe>/fold<k>/<arm>/<run>/track1_score.json (per-subject sealed
# scores). Finished folds are skipped, and alljoined_run.sh claims each run, so several workers can be pointed
# at the same recipes and folds (align_dyn.sh).
cd "$(dirname "$0")/.."

FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")

# recipe -> "<alljoined_run.sh arm> <train.py flags>". Every recipe trains for a fixed budget and keeps its
# last epoch (--select_best_on fixed); the budgets come from the per-epoch sealed curves (README.md, protocol).
# The 5 *_optx_things recipes are the best ensemble (HISTORY.md: lean TTA 44.25 over 4 folds); the 5 plain
# *_things recipes are the same members without channel-operator transfer, kept as its baseline (44.02).
recipe() {
  case "$1" in
    reve_scratch_things)        echo "reve --fm_no_pretrained --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10" ;;
    reve_d6_things)             echo "reve --fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10" ;;
    sqf_dbn_things)             echo "sqf  --num_epochs 40 --tsconv_domain_bn_boundary 21" ;;  # every BatchNorm per dataset (ids >= 21 = THINGS)
    tsc_dbn40_bn64_things)      echo "tsc  --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck" ;;
    rmix_reg_things)            echo "gen  --eeg_encoder_type TokenREVE --arch_kwargs {\"tokens\":\"mixed\",\"registers\":4} --batch_size 128 --learning_rate 1e-3 --num_epochs 20 --stop_after_epoch 10 --amp_dtype bfloat16 --ema_decay 0.9998 --ema_warmup" ;;
    # + re-render each batch through another training subject's channel covariance operator (HISTORY.md)
    reve_scratch_optx_things)   echo "$(recipe reve_scratch_things) --op_transfer_prob 0.5" ;;
    reve_d6_optx_things)        echo "$(recipe reve_d6_things) --op_transfer_prob 0.5" ;;
    sqf_dbn_optx_things)        echo "$(recipe sqf_dbn_things) --op_transfer_prob 0.5" ;;
    tsc_dbn40_bn64_optx_things) echo "$(recipe tsc_dbn40_bn64_things) --op_transfer_prob 0.5" ;;
    rmix_reg_optx_things)       echo "$(recipe rmix_reg_things) --op_transfer_prob 0.5" ;;
    # students of the opt4 ensemble (export_teacher_cv4.py teacher, fold training subjects only; the loop adds the flags)
    reve_d6_optx_kd4_things)    echo "$(recipe reve_d6_optx_things)" ;;
    sqf_dbn_optx_kd4_things)    echo "$(recipe sqf_dbn_optx_things)" ;;
    rmix_reg_optx_kd4_things)   echo "$(recipe rmix_reg_optx_things)" ;;
    # KD variants (NIGHT_LOG.md N18): bigger student / person-centred teacher / full 20-epoch cycle / 2nd-generation teacher
    reve_scratch_optx_kd4_things) echo "$(recipe reve_scratch_optx_things)" ;;
    reve_d6_optx_kd4pc_things)  echo "$(recipe reve_d6_optx_things)" ;;
    reve_d6_optx20_kd4_things)  echo "$(recipe reve_d6_optx_things) --stop_after_epoch 20" ;;
    reve_d6_optx_kd5_things)    echo "$(recipe reve_d6_optx_things)" ;;
    # 5.8M-parameter TSC (384 temporal/spatial/projection filters; base 0.15M), 20 epochs, 4-chunk activation checkpointing (OOM at batch 1024 otherwise), KD from opt4
    tsc_dbn40_bn64_optx_kd4_things) echo "$(recipe tsc_dbn40_bn64_optx_things)" ;;  # control for tsc_big: same KD, base capacity
    # person-centred loss: each output minus its same-person batch-mates' mean (the test-time centring, true ids)
    reve_d6_optx_pct_things)    echo "$(recipe reve_d6_optx_things) --person_centre_train" ;;
    reve_d6_optx_dro_things)    echo "$(recipe reve_d6_optx_things) --group_dro_rho 1" ;;  # soft maximum over people of their batch loss
    reve_d6_optx_js_things)     echo "$(recipe reve_d6_optx_things) --js_groups 4" ;;  # James-Stein gradient shrinkage across 4 person groups
    tsc_big_optx_kd4_things)    echo "tsc  --num_epochs 20 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --op_transfer_prob 0.5 --tsconv_temporal_filters 384 --tsconv_spatial_filters 384 --tsconv_projection_filters 384 --backbone_checkpoint_chunks 4" ;;
    # MLP-only members (NIGHT_LOG N65), TSC recipe: dense time/electrode/feature mixer on the raw grid; flat MLP ablation
    mix3_dbn_bn64_optx_things)  echo "gen  --eeg_encoder_type Mixer3 --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --op_transfer_prob 0.5 --amp_dtype bfloat16" ;;  # bf16 + compiled trunk: 0.12 s/step
    mix3att_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"time_attn\":8}" ;;  # + 1 single-head time attention
    mix3att40_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"time_attn\":8,\"time_tokens\":40}" ;;  # + dense 120->40 time compression
    mix3attd5_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"time_attn\":8,\"drop\":0.5}" ;;  # + dropout .5 (mixers overfit)
    mix3att40d5_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"time_attn\":8,\"time_tokens\":40,\"drop\":0.5}" ;;  # both fixes
    mix3bn_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"norm\":\"bn\"}" ;;  # BatchNorm inside the blocks
    mix3attbn_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"time_attn\":8,\"norm\":\"bn\"}" ;;
    mix3nn_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"norm\":\"none\"}" ;;  # no normalisation inside the blocks
    mix3lp_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"band\":\"lp\"}" ;;  # TSConv-like 0.2 s smoothing of the input
    mix3hp_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"band\":\"hp\"}" ;;  # input minus its 0.2 s moving average
    reve_d6_hp_optx_things)     echo "$(recipe reve_d6_optx_things) --input_band hp" ;;  # REVE-d6 on the input minus its 0.2 s moving average
    reve_d6_lp_optx_things)     echo "$(recipe reve_d6_optx_things) --input_band lp" ;;  # REVE-d6 on the 0.2 s moving average (< ~3 Hz: the conv members' band)
    reve_d6_bp_optx_things)     echo "$(recipe reve_d6_optx_things) --input_band bp" ;;  # REVE-d6 on ~4-12 Hz (5-sample minus 25-sample moving average)
    reve_d6_vhp_optx_things)    echo "$(recipe reve_d6_optx_things) --input_band vhp" ;;  # REVE-d6 on > ~15 Hz (input minus its 5-sample moving average)
    reve_d6_bd_optx_things)     echo "$(recipe reve_d6_optx_things) --band_drop 0.5" ;;  # band-dropout augmentation (N65)
    sqf_dbn_bd_optx_things)     echo "$(recipe sqf_dbn_optx_things) --band_drop 0.5" ;;
    tsc_dbn40_bn64_bd_optx_things) echo "$(recipe tsc_dbn40_bn64_optx_things) --band_drop 0.5" ;;
    reve_d6_pr1_optx_things)    echo "$(recipe reve_d6_optx_things) --prior_lambda 1" ;;  # output-prior equipartition (N66)
    reve_d6_pr3_optx_things)    echo "$(recipe reve_d6_optx_things) --prior_lambda 3" ;;
    sqf_dbn_pr1_optx_things)    echo "$(recipe sqf_dbn_optx_things) --prior_lambda 1" ;;
    sqf_dbn_pr03_optx_things)   echo "$(recipe sqf_dbn_optx_things) --prior_lambda 0.3" ;;
    reve_d6_pr01_optx_things)   echo "$(recipe reve_d6_optx_things) --prior_lambda 0.1" ;;
    reve_d6_pr03_optx_things)   echo "$(recipe reve_d6_optx_things) --prior_lambda 0.3" ;;
    tsc_dbn40_bn64_pr03_optx_things) echo "$(recipe tsc_dbn40_bn64_optx_things) --prior_lambda 0.3" ;;
    sqf_dbn_pr01_optx_things)   echo "$(recipe sqf_dbn_optx_things) --prior_lambda 0.1" ;;
    # novel-image large-gallery proxy (NIGHT_LOG N71): 5,000 training images held out of training; bank loss / sharp temperature
    sqf_dbn_ho5k_optx_things)   echo "$(recipe sqf_dbn_optx_things) --holdout_images 5000" ;;
    sqf_dbn_ho5k_bank_optx_things)   echo "$(recipe sqf_dbn_optx_things) --holdout_images 5000 --bank_lambda 1" ;;
    sqf_dbn_ho5k_t01_optx_things)   echo "$(recipe sqf_dbn_optx_things) --holdout_images 5000 --init_temperature 0.1" ;;
    # epoch-budget check for the sealed regime: same trajectory as the base recipe, run further, checkpoints along the way
    reve_d6_long_optx_things)   echo "$(recipe reve_d6_optx_things) --stop_after_epoch 20 --save_epochs 6 8 10 12 14 16 18" ;;
    rmix_reg_long_optx_things)   echo "$(recipe rmix_reg_optx_things) --stop_after_epoch 20 --save_epochs 6 8 10 12 14 16 18" ;;
    reve_scratch_long_optx_things)   echo "$(recipe reve_scratch_optx_things) --stop_after_epoch 14 --save_epochs 6 8 10 12" ;;
    sqf_dbn_long_things)   echo "$(recipe sqf_dbn_things) --save_epochs 24 28 32 36" ;;
    # the same epoch scans on the clean novel-image proxy (5,000 training images held out; logs/holdout_eval.py)
    rmix_reg_ho5k_long_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --stop_after_epoch 20 --save_epochs 8 10 12 14 16 18" ;;
    reve_scratch_ho5k_long_optx_things)   echo "$(recipe reve_scratch_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 6 8 10 12" ;;
    reve_d6_ho5k_long_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 20 --save_epochs 4 6 8 10 12 14 16" ;;
    sqf_dbn_ho5k_long_optx_things)   echo "$(recipe sqf_dbn_optx_things) --holdout_images 5000 --save_epochs 16 20 24 28 32 36" ;;
    sqf_dbn_ho5k_long_things)   echo "$(recipe sqf_dbn_things) --holdout_images 5000 --save_epochs 16 20 24 28 32 36" ;;
    sqf_dbn_ho5k_cs05_long_optx_things)   echo "$(recipe sqf_dbn_optx_things) --holdout_images 5000 --concept_smooth 0.5 --save_epochs 16 20 24 28 32 36" ;;
    rmix_reg_ho5k_cs05_long_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --concept_smooth 0.5 --stop_after_epoch 16 --save_epochs 6 8 10 12 14" ;;
    reve_d6_ho5k_cs05_long_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --concept_smooth 0.5 --stop_after_epoch 16 --save_epochs 6 8 10 12 14" ;;
    reve_d6_ho5k_wd3_long_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --weight_decay 0.3 --stop_after_epoch 16 --save_epochs 6 8 10 12 14" ;;
    rmix_reg_ho5k_wd3_long_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --weight_decay 0.3 --stop_after_epoch 16 --save_epochs 6 8 10 12 14" ;;
    reve_d6_hoc500_long_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_concepts 500 --stop_after_epoch 20 --save_epochs 4 6 8 10 12 14 16" ;;
    reve_d6_ho5k_wd3_full_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --weight_decay 0.3 --stop_after_epoch 20 --save_epochs 12 14 16 18" ;;
    rmix_reg_ho5k_wd3_full_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --weight_decay 0.3 --stop_after_epoch 20 --save_epochs 12 14 16 18" ;;
    reve_d6_ho5k_mid_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 12 --save_epochs 8 10" ;;  # clean-proxy replication folds (budget 10 question)
    rmix_reg_ho5k_mid_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --stop_after_epoch 12 --save_epochs 8 10" ;;
    reve_scratch_ho5k_mid_optx_things)   echo "$(recipe reve_scratch_optx_things) --holdout_images 5000 --stop_after_epoch 10 --save_epochs 8" ;;
    reve_scratch_ho5k_m12_optx_things)   echo "$(recipe reve_scratch_optx_things) --holdout_images 5000 --stop_after_epoch 12 --save_epochs 10" ;;  # fold-3 scratch @12 (the budget-12 check, NIGHT_LOG 11-10 17:10)
    # image-count scaling probe: 9,000 held out (a superset of the 5k gallery) = 7,540 training images vs 11,540
    reve_d6_ho9k_mid_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 9000 --stop_after_epoch 14 --save_epochs 8 10 12" ;;
    rmix_reg_ho9k_mid_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 9000 --stop_after_epoch 14 --save_epochs 8 10 12" ;;
    # Alljoined-only training (no THINGS-EEG2 sub-21..30; the cv4.sh loop skips them for *_aj): 15 people, ~60% of the steps per epoch
    reve_d6_ho5k_mid_optx_aj)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 16 --save_epochs 8 10 12 14" ;;
    rmix_reg_ho5k_mid_optx_aj)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --stop_after_epoch 16 --save_epochs 8 10 12 14" ;;
    # large-gallery levers on the clean proxy (fold 3 vs reve_d6_ho5k_mid_optx_things ep8/10): cross-image mixup, operator transfer on every batch
    reve_d6_ho5k_imix_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12 --image_mixup_prob 0.5" ;;
    rmix_reg_ho5k_imix_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12 --image_mixup_prob 0.5" ;;
    reve_d6_ho5k_opt1_things)   echo "$(recipe reve_d6_things) --op_transfer_prob 1.0 --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12" ;;
    # NeuroBridge CPA image prior (paper default: mean DINOv2-g target over GaussianBlur 15 / GaussianNoise 25 / LowResolution 0.1 / Mosaic 5 views; logs/aug_features.py)
    reve_d6_ho5k_cpa4_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12 --train_image_targets /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_cpa4.npy" ;;
    rmix_reg_ho5k_cpa4_optx_things)   echo "$(recipe rmix_reg_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12 --train_image_targets /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_cpa4.npy" ;;
    reve_d6_ho5k_cpav_optx_things)   echo "$(recipe reve_d6_optx_things) --holdout_images 5000 --stop_after_epoch 14 --save_epochs 8 10 12 --train_image_targets /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_GaussianBlur.npy /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_GaussianNoise.npy /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_LowResolution.npy /nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/aug_Mosaic.npy" ;;  # random view per trial (clean + the 4 CPA views), the old dataset's per-sample pick
    sqf_dbn_optx_anti4_things)  echo "$(recipe sqf_dbn_optx_things) --anti_teacher_lambda 5" ;;  # shared-confusion penalty vs opt4 (N67)
    sqf_dbn_optx_l20_anti4_things) echo "$(recipe sqf_dbn_optx_things) --anti_teacher_lambda 20" ;;
    spec3_dbn_bn64_optx_things) echo "$(recipe mix3_dbn_bn64_optx_things) --arch_kwargs {\"spectral\":true}" ;;  # mixer over each electrode's rFFT
    mlp_dbn_bn64_optx_things)   echo "gen  --eeg_encoder_type FlatMLP --num_epochs 40 --tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --op_transfer_prob 0.5" ;;
    *) return 1 ;;
  esac
}
RECIPES="reve_d6_optx_things rmix_reg_optx_things sqf_dbn_optx_things tsc_dbn40_bn64_optx_things reve_scratch_optx_things
reve_d6_things rmix_reg_things sqf_dbn_things tsc_dbn40_bn64_things reve_scratch_things
reve_d6_optx_kd4_things sqf_dbn_optx_kd4_things rmix_reg_optx_kd4_things
reve_scratch_optx_kd4_things reve_d6_optx_kd4pc_things reve_d6_optx20_kd4_things reve_d6_optx_kd5_things tsc_big_optx_kd4_things tsc_dbn40_bn64_optx_kd4_things
reve_d6_optx_pct_things reve_d6_optx_dro_things reve_d6_optx_js_things
mix3_dbn_bn64_optx_things mlp_dbn_bn64_optx_things mix3att_dbn_bn64_optx_things spec3_dbn_bn64_optx_things mix3att40_dbn_bn64_optx_things mix3attd5_dbn_bn64_optx_things mix3att40d5_dbn_bn64_optx_things mix3bn_dbn_bn64_optx_things mix3attbn_dbn_bn64_optx_things mix3nn_dbn_bn64_optx_things mix3lp_dbn_bn64_optx_things mix3hp_dbn_bn64_optx_things reve_d6_hp_optx_things reve_d6_lp_optx_things reve_d6_bp_optx_things reve_d6_vhp_optx_things reve_d6_bd_optx_things sqf_dbn_bd_optx_things tsc_dbn40_bn64_bd_optx_things reve_d6_pr1_optx_things reve_d6_pr3_optx_things sqf_dbn_pr1_optx_things sqf_dbn_pr03_optx_things reve_d6_pr01_optx_things reve_d6_pr03_optx_things tsc_dbn40_bn64_pr03_optx_things sqf_dbn_pr01_optx_things sqf_dbn_optx_anti4_things sqf_dbn_optx_l20_anti4_things sqf_dbn_ho5k_optx_things sqf_dbn_ho5k_bank_optx_things sqf_dbn_ho5k_t01_optx_things reve_d6_long_optx_things rmix_reg_long_optx_things reve_scratch_long_optx_things sqf_dbn_long_things rmix_reg_ho5k_long_optx_things reve_scratch_ho5k_long_optx_things reve_d6_ho5k_long_optx_things sqf_dbn_ho5k_long_optx_things sqf_dbn_ho5k_long_things sqf_dbn_ho5k_cs05_long_optx_things rmix_reg_ho5k_cs05_long_optx_things reve_d6_ho5k_cs05_long_optx_things reve_d6_ho5k_wd3_long_optx_things rmix_reg_ho5k_wd3_long_optx_things reve_d6_hoc500_long_optx_things reve_d6_ho5k_wd3_full_optx_things rmix_reg_ho5k_wd3_full_optx_things"

if [ "$1" = --list ]; then for r in $RECIPES; do echo "$r: $(recipe $r)"; done; exit 0; fi
NAME=$1; shift
SPEC=$(recipe "$NAME") || { echo "unknown recipe '$NAME' (cv4.sh --list)"; exit 1; }
ARM=${SPEC%% *} FLAGS="${SPEC#* }"
DATA=/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench
grep -q things_eeg2_subjects $DATA/info.json || { echo "THINGS-EEG2 subjects not exported (export_things_to_alljoined.py)"; exit 1; }
# their training caches must exist on the NAS first: built once there, staged like everyone else's, instead of
# every job rebuilding ~10 GB into its node's /tmp. The marker is written by hand once they are built.
[ -f $DATA/.cache/things_caches_ok ] || { echo "THINGS-EEG2 training caches not built on the NAS yet"; exit 1; }
for K in ${@:-1 2 3 4}; do
  TEST=${FOLDS[$K]}; [ -n "$TEST" ] || { echo "no fold $K"; exit 1; }
  OUT=results/alljoined/track1/cv4/$NAME/fold$K/$ARM
  if ls "$OUT"/*/track1_score.json >/dev/null 2>&1; then echo "skip $NAME fold$K (scored)"; continue; fi
  TRAIN=""; for s in $(seq 1 20); do [[ " $TEST " == *" $s "* ]] || TRAIN+="$s "; done
  [[ $NAME =~ _aj$ ]] || TRAIN+="$(seq -s ' ' 21 30)"  # *_aj recipes: Alljoined people only, no THINGS-EEG2
  KD=""
  if [[ $NAME =~ _anti4_things$ ]]; then  # shared-confusion penalty against the fold's opt4 teacher (NIGHT_LOG N67)
    T=results/alljoined/track1/cv4/teachers/fold${K}_reve_d6_optx_things+rmix_reg_optx_things+sqf_dbn_optx_things+tsc_dbn40_bn64_optx_things.npy
    [ -f $T ] || { echo "no teacher $T"; continue; }
    cat $T > /dev/null  # 5 GiB of random per-trial reads: warm it like the EEG memmaps
    KD="--distill_targets $T"
  elif [[ $NAME =~ _kd(4|4pc|5)_things$ ]]; then  # distil a fold ensemble (NIGHT_LOG.md N18): 4 = opt4, 4pc = opt4 person-centred, 5 = kd-opt4 (2nd generation)
    case ${BASH_REMATCH[1]} in
      4) T=results/alljoined/track1/cv4/teachers/fold${K}_reve_d6_optx_things+rmix_reg_optx_things+sqf_dbn_optx_things+tsc_dbn40_bn64_optx_things.npy ;;
      4pc) T=results/alljoined/track1/cv4/teachers/fold${K}_reve_d6_optx_things+rmix_reg_optx_things+sqf_dbn_optx_things+tsc_dbn40_bn64_optx_things_pc.npy ;;
      5) T=results/alljoined/track1/cv4/teachers/fold${K}_reve_d6_optx_kd4_things+rmix_reg_optx_things+sqf_dbn_optx_things+tsc_dbn40_bn64_optx_things.npy ;;
    esac
    for i in $(seq 180); do [ -f $T ] && break; sleep 60; done
    [ -f $T ] || { echo "no teacher $T"; continue; }
    cat $T > /dev/null  # 5 GiB of random per-trial reads: warm it like the EEG memmaps
    KD="--distill_targets $T --distill_lambda 1"
  fi
  echo "=== $(date '+%F %T') $NAME fold$K  train: $TRAIN test: $TEST"
  # sealed-rule score of every test subject on a seeded half of its trials (~40 repeats/image), every epoch
  SPLIT=cv4/$NAME/fold$K TRAIN_SUBJECTS="$TRAIN" TEST_SUBJECTS="$TEST" bash neurips_challenge/alljoined_run.sh "$ARM" \
    --select_best_on fixed --sealed_test_every 1 --sealed_test_fraction 0.5 $FLAGS $KD \
    || echo "FAILED $NAME fold$K"
done; echo "done $NAME folds ${@:-1 2 3 4}"; exit  # one line: bash never re-reads this file after the loop (recipes get added while jobs run)
