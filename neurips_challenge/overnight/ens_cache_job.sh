#!/bin/bash
# align-queue one-off analysis jobs (claim -> run -> done marker align_dyn.sh reads). Dispatched by cv4.sh.
#   enscache_a|enscache_b: ens_cache.py on 5 ensemble candidates each;  poe_sim: product-of-experts combiner (A30)
#   route_emb_*: subject_embed.py (ROUTING_LOG E48)
cd "$(dirname "$0")/../.."
case $1 in
  enscache_a) CMD="neurips_challenge/ens_cache.py reve_d6_kd_things reve_scratch_things sw0_things lab_s10_things tsc_dbn40_bn64_things" ;;
  enscache_b) CMD="neurips_challenge/ens_cache.py sqf_dbn_kd_things sqf_dbn_things reve_d6_things reve_z_things plab_gain_things" ;;
  poe_sim)    CMD="neurips_challenge/poe_sim.py" ;;
  enscache_rmix) CMD="neurips_challenge/ens_cache.py rmix_things"; WAIT=rmix_things ;;
  enscache_rmix_reg) CMD="neurips_challenge/ens_cache.py rmix_reg_things"; WAIT=rmix_reg_things ;;
  enscache_rmixd8) CMD="neurips_challenge/ens_cache.py rmixd8_things"; WAIT=rmixd8_things ;;
  enscache_rmix_s2) CMD="neurips_challenge/ens_cache.py rmix_s2_things"; WAIT=rmix_s2_things ;;  # ARCH_LOG: rmix as an ensemble member
  route_emb_both) CMD="neurips_challenge/subject_embed.py --arch both" ;;  # ROUTING_LOG E48: learned participant embedding
  route_emb_fp)   CMD="neurips_challenge/subject_embed.py --arch fp" ;;
  route_emb_con)  CMD="neurips_challenge/subject_embed.py --arch both --loss supcon --tag con" ;;  # E48b: in-batch contrastive
  route_emb_big)  CMD="neurips_challenge/subject_embed.py --arch both --eeg1 --tag big" ;;  # E48c: + 50 THINGS-EEG1 identities
  route_whiten2)  CMD="neurips_challenge/route_study.py --cfgs whiten --emb both --device cuda --chunk 64" ;;  # E49 rerun: k x k eigenproblems
  route_emb_eval) CMD="neurips_challenge/route_study.py --cfgs emb --emb big both --device cuda --chunk 64" ;;  # E48d: route with the embeddings
  feat_coral)     CMD="neurips_challenge/feature_coral.py" ;;  # SUBJECT_LOG D7: recolour the 40 filter outputs to population stats
  feat_tf)        CMD="neurips_challenge/feature_transform.py" ;;  # SUBJECT_LOG D6: batch transform inside the head
  route_ldaE)     CMD="neurips_challenge/route_study.py --cfgs ldaE --device cuda --chunk 64" ;;  # E50: router LDA refit on 61 identities
  route_whiten)   CMD="neurips_challenge/route_study.py --cfgs whiten --emb con both --device cuda --chunk 64" ;;  # E49 (user): neighbour whitening
  *) echo "unknown $1"; exit 1 ;;
esac
# claim seen by align_dyn.sh (<first subdir>/.claim/job): other workers skip this entry while we run
mkdir -p results/alljoined/track1/cv4/$1/fold1/run/.claim && echo "${SLURM_JOB_ID:-$$}" > results/alljoined/track1/cv4/$1/fold1/run/.claim/job
[ -n "$WAIT" ] && for i in $(seq 240); do  # after the claim: wait for the recipe's 4 scored folds
  [ $(ls results/alljoined/track1/cv4/$WAIT/fold*/*/2026*/track1_score.json 2>/dev/null | wc -l) -ge 4 ] && break; sleep 30; done
echo "=== $(date '+%F %T') $1: $CMD"
.venv/bin/python $CMD | tee results/alljoined/track1/cv4/$1/fold1/run/output.txt && \
  mkdir -p results/alljoined/track1/cv4/$1/fold1/a_done/2026-done && touch results/alljoined/track1/cv4/$1/fold1/a_done/2026-done/track1_score.json
echo "done $1"
