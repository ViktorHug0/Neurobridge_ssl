#!/bin/bash
# SubjectMix rebuttal ablations — separate "convex interpolation" from "cross-subject averaging/smoothing".
# Reviewer claim: "the gain is just averaging more EEGs / smoothing, not the convex interpolation."
#
# Every arm is a DROP-IN swap of the same-stimulus cross-subject augmentation; the recipe is otherwise
# identical to the paper's SubjectMix config (TSConv_parameterizable k30, InternViT-L28, batch 1024,
# 50 epochs, grouped multi-positive, data_average, LOSO over all 10 subjects). Results -> subjectmix_rebuttal/.
#
# Trick that avoids new code: Beta(a,a)/Dirichlet(a) with a=1000 collapses the random mix weights to the
# EQUAL-WEIGHT MEAN, turning "stochastic convex mixup" into "deterministic averaging" with the same pairing.
# # ponytail: a=1000 ~= fixed lam=0.5 (std ~0.01); add an exact --mixup_fixed_lam flag only if a reviewer nitpicks.
#
# NOT auto-run. Usage:
#   DRY=1 bash scripts/things_eeg/subjectmix_rebuttal.sh          # print the full run matrix, execute nothing
#   bash scripts/things_eeg/subjectmix_rebuttal.sh                # run everything (see runtime note below)
#   SEEDS="3300 3301 3302" ARMS="method fixed_mean2" bash ...     # paper's 3 seeds, only chosen arms
#
# Runtime: ~18 min per (arm,seed,subject) on an RTX 3080; the xavg arm is ~9x that (step-matched 450 ep).
# Full matrix (10 arms x 10 subj x 1 seed) ~= 2 days; x3 seeds ~= a week. Scope with ARMS/SEEDS/SUBJECTS.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
[ -f "${REPO_ROOT}/.venv/bin/activate" ] && source "${REPO_ROOT}/.venv/bin/activate"
cd "$REPO_ROOT"

IMAGE_FEATURE_DIR="${REPO_ROOT}/data/things_eeg/image_feature/InternViT-6B_layer28_mean_8bit"
EEG_DATA_DIR="/nasbrain/p20fores/NICE-EEG/Data/Things-EEG2/Preprocessed_data_250Hz/"
DEVICE="${DEVICE:-cuda:0}"
SEEDS="${SEEDS:-3300}"                       # paper uses 3: SEEDS="3300 3301 3302"
SUBJECTS="${SUBJECTS:-1 2 3 4 5 6 7 8 9 10}"
ROOT="${OUTPUT_DIR:-./results/things_eeg/subjectmix_rebuttal}"

COMMON="--batch_size 1024 --num_workers 4 --learning_rate 3e-4 \
  --eeg_encoder_type TSConv_parameterizable --tsconv_temporal_kernel 30 \
  --image_feature_dir $IMAGE_FEATURE_DIR --eeg_data_dir $EEG_DATA_DIR --device $DEVICE \
  --feature_dim 512 --eeg_backbone_dim 1024 --softplus --img_l2norm --projector linear \
  --save_weights --text_feature_dir ''"
GROUPED="--data_average --grouped_batch_sampler --multi_positive_loss"   # cross-subject same-image batch

# run_arm <arm_name> <extra flags...>   (honors EPOCHS env, default 50; skips finished cells; continues on failure)
run_arm () {
    local arm="$1"; shift
    if [ -n "$ARMS" ] && [[ " $ARMS " != *" $arm "* ]]; then return; fi
    for SEED in $SEEDS; do
        local sess="$ROOT/$arm/seed${SEED}"; mkdir -p "$sess"
        for SUB in $SUBJECTS; do
            local name; name="$(printf 'sub-%02d' "$SUB")"
            if ls "$sess/$name"*/result.csv >/dev/null 2>&1; then echo "skip done: $arm seed$SEED $name"; continue; fi
            local train=""; for i in $SUBJECTS; do [ "$i" -ne "$SUB" ] && train+="$i "; done
            local cmd="python3 train.py $COMMON --num_epochs ${EPOCHS:-50} \
                --output_name $name --output_dir $sess \
                --train_subject_ids $train --test_subject_ids $SUB --seed $SEED $*"
            echo "### $arm seed$SEED $name ###"
            if [ -n "$DRY" ]; then echo "$cmd"; else eval "$cmd" || echo "FAILED: $arm seed$SEED $name (continuing)"; fi
        done
        [ -z "$DRY" ] && python3 compute_avg_results.py --result_dir "$sess" --output_name "inter_subject_summary.csv"
    done
}

# ============ Sub-claim 1: is the gain AVERAGING or INTERPOLATION? (the core 2x2) ============
# stochastic + convex interpolation (THE METHOD)
run_arm method_subjectmix  $GROUPED --samples_per_image 9 --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 0.5
# deterministic MEAN of the SAME 2 trials (removes stochasticity + endpoint mass; same "average 2 EEGs")
run_arm fixed_mean2        $GROUPED --samples_per_image 9 --subject_mixup_mode raw_eeg --mixup_type pairwise --subject_mixup_alpha 1000
# deterministic MEAN of k cross-subject trials — the literal "average more EEGs" claim; find its ceiling
for K in 2 3 5 9; do
run_arm "fixed_mean_k${K}" $GROUPED --samples_per_image "$K" --subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 1000
done
# STOCHASTIC averaging (random-size random-subset mean, redrawn per epoch) — no interpolation.
# One-sample-per-image epoch is ~16 batches, so 450 ep matches the 50-ep gradient-step budget.
EPOCHS=450 run_arm stoch_mean_xavg --cross_subject_average --xavg_beta_a 1 --xavg_beta_b 1 --xavg_kmin 4 --xavg_kmax 36

# ============ Sub-claim 2: is it "JUST SMOOTHING / generic augmentation"? (no cross-subject mixing) ============
run_arm noise_aug   $GROUPED --samples_per_image 9 --subject_mixup_mode none --eeg_aug --eeg_aug_type noise
run_arm smooth_aug  $GROUPED --samples_per_image 9 --subject_mixup_mode none --eeg_aug --eeg_aug_type smooth

# WITHIN-subject Beta mix of 2 of a subject's OWN reps (isolates the cross-subject ingredient). Uses
# un-averaged reps (no data_average), same grouped multi-positive batch structure as the method.
run_arm within_subject_mix --grouped_batch_sampler --multi_positive_loss --samples_per_image 9 --subject_mixup_within --within_mix_alpha 0.5

# ============ Sub-claim 3 (steelman): is there a BETTER averaging? ============
# Per-subject Euclidean Alignment (channel-cov whitening) THEN deterministic group-mean of 9.
run_arm aligned_average $GROUPED --samples_per_image 9 --subject_mixup_mode raw_eeg --mixup_type group --subject_mixup_alpha 1000 --subject_ea_align

# ============ Reference ============
run_arm base_no_aug $GROUPED --samples_per_image 9 --subject_mixup_mode none

echo "DONE. Summaries: $ROOT/<arm>/seed<seed>/inter_subject_summary.csv (Average row, 'best top1 acc')"
