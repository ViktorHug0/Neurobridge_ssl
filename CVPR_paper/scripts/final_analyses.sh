#!/bin/bash
# Full analysis battery on a finished P-main LOSO arm (all 10 folds of results/things_eeg/cvpr/loso/<arm>/seed<S>):
# 1) the cross-fold SAGE screen (here, blocking: everything else needs its selected_per_subject.csv), then
# 2) appends the per-subject analyses to the CPU queue and runs 4 CPU workers itself.
#   usage: final_analyses.sh <arm> [seed]      (run it as a cpu_queue item; FINAL_CKPT overrides the checkpoint,
#   NOT SAGE_CHECKPOINT_NAME: cpu_worker.sh exports a legacy default for that one)
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
arm=$1 seed=${2:-3300}
src=results/things_eeg/cvpr/loso/$arm/seed$seed
[ "${INTRA:-0}" = 1 ] && src=results/things_eeg/cvpr/intra/$arm/seed$seed  # INTRA=1: intra-subject arm
n=$(ls -d $src/*-sub-*/result.csv 2>/dev/null | wc -l)
[ "$n" -eq 10 ] || { echo "only $n/10 folds in $src"; return 1; }
if [ "${INTRA:-0}" = 1 ]; then
  bash CVPR_paper/scripts/tta_screen_intra_cvpr.sh $arm $seed || return 1
  params=results/things_eeg/cvpr/tta_screen_intra/${arm}_s$seed/selected_per_subject.csv
else
  bash CVPR_paper/scripts/tta_screen_loso.sh $arm $seed || return 1
  params=results/things_eeg/cvpr/tta_screen_loso/${arm}_s$seed/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv
fi
[ -f $params ] || { echo "missing $params"; return 1; }
t=${arm}_s$seed o=results/things_eeg/cvpr/final/$t L=logs/cvpr/final/$t
mkdir -p $o $L
env="export SAGE_CHECKPOINT_NAME=${FINAL_CKPT:-checkpoint_test_best.pth};"
py=".venv/bin/python CVPR_paper/scripts"
common="--source_run_dir $src --params_csv $params"
{
  echo "$env $py/r1_eval.py $common --output_csv $o/r1_competitors.csv > $L/r1_eval.log 2>&1"
  echo "$env PYTHONPATH=/nasbrain/p20fores/SATTC-CVPR2026/.deps $py/sattc_on_embeddings.py ours --run_dir $src --output_csv $o/sattc.csv > $L/sattc.log 2>&1"
  echo "$env $py/r1_ablate.py $common --output_csv $o/ablate.csv > $L/ablate.log 2>&1"
  echo "$env $py/r1_ablate.py $common --output_csv $o/reps.csv --reps_sweep > $L/reps.log 2>&1"
  echo "$env $py/session_calib.py $common --output_csv $o/session_calib.csv > $L/session_calib.log 2>&1"
  echo "$env $py/geometry.py $common --cache_dir results/things_eeg/cvpr/emb_cache/$t --output_csv $o/geometry.csv > $L/geometry.log 2>&1 && $py/sage_hpens.py --grid_csv ${params%/*}/grid_scores.csv --cache_dir results/things_eeg/cvpr/emb_cache/$t --output_csv $o/hpens.csv > $L/hpens.log 2>&1"
  echo "$env CUDA_VISIBLE_DEVICES= $py/r1_transductive.py $common --cache_dir results/things_eeg/cvpr/emb_cache/${t}_b --output_csv $o/r1_transductive.csv > $L/r1_transductive.log 2>&1"
  for s in "1 2 3 4 5" "6 7 8 9 10"; do p=${s// /_}
    echo "$env $py/bci_baselines.py $common --output_csv $o/bci_$p.csv --subjects $s > $L/bci_$p.log 2>&1"
    echo "$env $py/label_equiv2.py $common --output_csv $o/label_equiv2_$p.csv --subjects $s > $L/label_equiv2_$p.log 2>&1"
  done
  for s in "1 2" "3 4" "5 6" "7 8" "9 10"; do p=${s// /_}
    echo "$env $py/regimes.py $common --output_dir $o/regimes/part_$p --subjects $s > $L/regimes_$p.log 2>&1"
    echo "$env $py/stream_grid.py $common --output_dir $o/stream/part_$p --subjects $s --alphas 1.0 --ramps 1:3 --fullramps 1:1.5 1:3 1:6 1:12 > $L/stream_$p.log 2>&1"
    echo "$env $py/stream_grid.py $common --output_dir $o/stream_score/part_$p --subjects $s --score_only > $L/stream_score_$p.log 2>&1"
  done
} > $o/queued.txt
flock CVPR_paper/queue/cpu_queue.txt.lock /bin/bash -c "cat $o/queued.txt >> CVPR_paper/queue/cpu_queue.txt"
# env -u: these workers also pick up OTHER queue items, which must not inherit this battery's FINAL_CKPT (2026-10-22)
for i in 1 2 3 4; do env -u FINAL_CKPT -u INTRA THREADS=2 bash CVPR_paper/scripts/cpu_worker.sh & done; wait
}
main "$@"; exit $?
