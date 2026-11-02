#!/bin/bash
# SUBJECT_LOG D4, the intra-subject ceiling: for each held-out subject of cv4 fold K, start from the fold's model (which
# never saw the subject) and fine-tune it on the subject's OWN training trials (other images), scoring the subject's
# test trials every epoch. Dispatched by cv4.sh for recipes ftor_* so it runs from the align queue between screens.
#   ft_oracle_parts.sh <ftor_tsc_bn|ftor_tsc_temp|ftor_tsc_spat|ftor_tsc_head|ftor_tsc_lora<r>> <fold>  (fine-tune one part only)
#   ftor_s11_hconv <split>: the same from the s11 split models (11 unseen people per base model, one filter basis)
cd "$(dirname "$0")/../.."
R=$1; K=$2
FOLDS=("" "2 4 5 11 15" "7 14 16 17 20" "3 8 10 12 19" "1 6 9 13 18")
case $R in
  ftor_tsc)  BASE=tsc_dbn40_bn64_things; ARM=tsc
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs 6 --learning_rate 1e-4" ;;
  ftor_tsc_bn|ftor_tsc_temp|ftor_tsc_spat|ftor_tsc_head)  # which part carries the subject? (BN running stats adapt in all)
             BASE=tsc_dbn40_bn64_things; ARM=tsc
             RX=$(case $R in *_bn) echo 'tsconv\.(2|5)\.';; *_temp) echo 'tsconv\.0\.';; *_spat) echo 'tsconv\.4\.';; *_head) echo 'projection|proj_eeg|eproj';; esac)
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs 6 --learning_rate 1e-4 --trainable_regex $RX" ;;
  ftor_tsc_hbias|ftor_tsc_hconv)  # D5: head biases + LayerNorm only (per-person shifts) / the 40x40 1x1 filter-mixing conv only
             BASE=tsc_dbn40_bn64_things; ARM=tsc
             RX=$(case $R in *_hbias) echo 'proj_eeg\.0\.bias|proj_eeg\.1\.f\.1\.bias|proj_eeg\.2\.|eproj\.?bias|projection\.bias';; *_hconv) echo 'model\.projection\.';; esac)
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs 6 --learning_rate 1e-3 --trainable_regex $RX" ;;
  ftor_s11_hconv)  # SUBJECT_LOG D9: 1x1 filter-mixing conv per person, 11 people per base model (cv4 has 5 per fold model)
             BASE=tsc_dbn40_things; ARM=tsc; SET=s11; FOLDS=("" "2 3 4 5 7 11 14 15 16 17 20" "1 2 3 6 8 9 10 12 13 18 19")
             ARGS="--tsconv_domain_bn_boundary 21 --num_epochs 6 --learning_rate 1e-3 --trainable_regex model\.projection\." ;;
  ftor_tsc_fs*)  # SUBJECT_LOG D8: few-shot oracle, LoRA rank 16 on the head from K labelled trials, ~200 optimizer steps
             BASE=tsc_dbn40_bn64_things; ARM=tsc; NK=${R#ftor_tsc_fs}  # (K is the fold)
             case $NK in 6) BS=6; EP=200;; 64) BS=64; EP=200;; *) BS=256; EP=100;; esac
             STE=$((EP / 4))
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs $EP --learning_rate 1e-3 --lora_rank 16 --train_max_trials $NK --batch_size $BS --freeze_bn_stats" ;;
  ftor_tsc_lora*)  # SUBJECT_LOG D5: is the per-person head headroom low-rank? rank-r adapters on the head's Linear layers
             BASE=tsc_dbn40_bn64_things; ARM=tsc
             ARGS="--tsconv_domain_bn_boundary 21 --eeg_backbone_dim 64 --projector bottleneck --num_epochs 6 --learning_rate 1e-3 --lora_rank ${R#ftor_tsc_lora}" ;;
  ftor_reve) BASE=reve_d6_things; ARM=reve
             ARGS="--fm_no_pretrained --fm_arch_kwargs {\"depth\":6} --reve_input_normalization trial_zscore --num_epochs 6 --stop_after_epoch 6 --learning_rate 1e-4" ;;
  *) echo "unknown $R"; exit 1 ;;
esac
CKPT=$(ls results/alljoined/track1/${SET:-cv4}/$BASE/fold$K/*/2026*/checkpoint_sealed_best.pth | head -1)
OUT=results/alljoined/track1/cv4/$R/fold$K
# entry-level claim read by align_dyn.sh (<first subdir>/.claim/job): one worker per oracle entry
mkdir -p $OUT/run/.claim && echo "${SLURM_JOB_ID:-$$}" > $OUT/run/.claim/job
for s in ${FOLDS[$K]}; do
  ls $OUT/s$s/$ARM/2026*/track1_score.json >/dev/null 2>&1 && continue
  echo "=== $(date '+%F %T') $R fold$K subject $s from $CKPT"
  SPLIT=cv4/$R/fold$K/s$s TRAIN_SUBJECTS=$s TEST_SUBJECTS=$s bash neurips_challenge/alljoined_run.sh $ARM $ARGS \
    --init_checkpoint "$CKPT" --select_best_on fixed --val_concept_ratio 0 --sealed_test_every ${STE:-1} --sealed_test_fraction 0.5
done
# align_dyn.sh treats "<first subdir>/2026*/track1_score.json" as done: a marker that sorts first
n=$(ls $OUT/s*/$ARM/2026*/track1_score.json 2>/dev/null | wc -l)
[ "$n" -eq $(wc -w <<< "${FOLDS[$K]}") ] && mkdir -p $OUT/a_done/2026-done && touch $OUT/a_done/2026-done/track1_score.json
echo "done $R fold$K ($n/$(wc -w <<< "${FOLDS[$K]}") subjects)"
