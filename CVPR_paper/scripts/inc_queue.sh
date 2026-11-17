#!/bin/bash
# Run list of the incremental-setting results of the paper (2026-11-10), as one command per line. Tables and figures:
# make_inc_figures.py. Run the lines with e.g.
#   export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=/nasbrain/p20fores/SATTC-CVPR2026/.deps
#   bash CVPR_paper/scripts/inc_queue.sh | xargs -P 7 -d '\n' -I{} bash -c '{}'
# (the appendix grid and six model-level subjects ran in one CPU-only Slurm job, inc_slurm.sh).
# NOTE: this list gives today's method names. The results on disk were produced during 2026-11-10 while the names changed
# (inc/main: gala_noprior = gala_noshrink; v3/main: gala = gala_d; v3/main2d: gala_2d = gala; v4/abl, v4/enc: the
# ablation and encoder lines below rerun with the 2d prior; v5/full: appendix grid);
# make_inc_figures.py renames them when loading and documents which folder feeds which row.
# 2026-11-11: the main text moved to q = 20 and the partial stream became skew25. The lines below are the q = 10 run
# list of 2026-11-10 (now only context); the runs that feed the paper are the command files results/things_eeg/cvpr/inc/
# queue_v7.txt, queue_v7enc.txt, queue_v7model.txt (main text) and queue_v5.txt, queue_s25.txt, queue_v6.txt (appendix grid),
# each run with inc_slurm.sh.
cd "$(dirname "$0")/../.."
R=results/things_eeg/cvpr; O=$R/inc; L=logs/cvpr/inc; py=".venv/bin/python CVPR_paper/scripts"
P() { echo $R/tta_screen_loso/$1/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv; }
mkdir -p $L
# embedding caches
for s in 3300 3301 3302; do echo "SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth $py/inc_cache.py $R/loso/atm128_vc/seed$s $O/cache/atm128_vc_s$s"; done
for a in tsconv_vc eegnet_vc eegproject_vc conformer_vc labram_vc; do echo "SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth $py/inc_cache.py $R/loso/$a/seed3300 $O/cache/${a}_s3300"; done
echo "SAGE_CHECKPOINT_NAME=checkpoint_last.pth $py/inc_cache.py $R/loso/score30s2_orig/seed3300 $O/cache/score30s2_orig_s3300"
# main table and figure (ATM, 3 seeds, 3 orders, 30 refits): competitors, GALA, and the two shrinkage ablations
for s in 3300 3301 3302; do A="--cache_dir $O/cache/atm128_vc_s$s --params_csv $(P atm128_vc_s$s) --trans_csv $R/final/atm128_vc_s$s/r1_transductive.csv"
  for h in "1 2 3 4 5" "6 7 8 9 10"; do t=${h// /_}
    echo "$py/inc_stream.py $A --output_csv $O/main/atm128_vc_s${s}_$t.csv --subjects $h --n_refits 30 --methods plain csls hungarian sattc transclip em_dirichlet gala_noshrink > $L/main_s${s}_$t.log 2>&1"
    echo "$py/inc_stream.py $A --output_csv $O/v2/main/atm128_vc_s${s}_$t.csv --subjects $h --n_refits 30 --methods score gala_d > $L/v2_main_s${s}_$t.log 2>&1"
    echo "$py/inc_stream.py $A --output_csv $O/v3/main/atm128_vc_s${s}_$t.csv --subjects $h --n_refits 30 --methods gala > $L/v3_main_s${s}_$t.log 2>&1"
  done
  echo "$py/inc_stream.py $A --output_csv $O/v3/abl/atm128_vc_s$s.csv --n_refits 30 --methods gala_damp gala_nowhiten gala_nocsls gala_1step gala_norot gala_softmax > $L/v3_abl_s$s.log 2>&1"
done
# SCORE with its own encoder (SCORE with ATM = method score of the ATM lines)
W="--cache_dir $O/cache/score30s2_orig_s3300 --params_csv $(P score30s2_orig_s3300)"
echo "$py/inc_stream.py $W --output_csv $O/v3/ownfast/score30s2_orig_s3300.csv --n_refits 30 --methods plain score > $L/v3_ownfast.log 2>&1"
# other encoders
for a in tsconv_vc eegnet_vc eegproject_vc conformer_vc labram_vc; do
  echo "$py/inc_stream.py --cache_dir $O/cache/${a}_s3300 --params_csv $(P ${a}_s3300) --output_csv $O/v3/enc/${a}_s3300.csv --n_refits 20 --orders 1 --methods plain score gala > $L/v3_enc_$a.log 2>&1"
  echo "$py/inc_stream.py --cache_dir $O/cache/${a}_s3300 --params_csv $(P ${a}_s3300) --output_csv $O/v5/encsattc/${a}_s3300.csv --scenarios uniform --n_refits 20 --orders 1 --methods sattc > $L/v5_encsattc_$a.log 2>&1"
done
# model-level baselines (ATM seed 3300, one order, 12 refits)
for s in 1 2 3 4 5 6 7 8 9 10; do
  echo "SAGE_CHECKPOINT_NAME=checkpoint_test_best.pth $py/inc_model.py --source_run_dir $R/loso/atm128_vc/seed3300 --output_csv $O/model/atm128_vc_s3300_sub$s.csv --subjects $s --orders 1 --n_refits 12 > $L/model_sub$s.log 2>&1"
done
# appendix: menu size K x repetitions q x stream (3 seeds, 3 orders, 15 refits; ran as Slurm job 72922, 3 h 14 on 30 cores)
# (the job grouped the subjects more coarsely for K <= 100; the exact lines it ran are in $O/queue_v5.txt)
G="--K 25 50 100 150 200 --reps 5 10 20 40 --n_refits 15 --orders 3"
for h in "1 2 3 4 5" "6 7 8 9 10"; do t=${h// /_}
  echo "$py/inc_stream.py $W --output_csv $O/v5/full/score30s2_orig_s3300_$t.csv $G --subjects $h --methods plain score > $L/v5_full_own_$t.log 2>&1"
done
for s in 3300 3301 3302; do A="--cache_dir $O/cache/atm128_vc_s$s --params_csv $(P atm128_vc_s$s) --trans_csv $R/final/atm128_vc_s$s/r1_transductive.csv"
  for K in 200 150 100 50 25; do for h in "1 2" "3 4" "5 6" "7 8" "9 10"; do t=${h// /_}
    echo "$py/inc_stream.py $A --output_csv $O/v5/full/atm128_vc_s${s}_K${K}_a_$t.csv ${G/25 50 100 150 200/$K} --subjects $h --methods plain sattc score gala > $L/v5_full_s${s}_K${K}_a_$t.log 2>&1"
    echo "$py/inc_stream.py $A --output_csv $O/v5/full/atm128_vc_s${s}_K${K}_b_$t.csv ${G/25 50 100 150 200/$K} --subjects $h --methods csls hungarian transclip em_dirichlet > $L/v5_full_s${s}_K${K}_b_$t.log 2>&1"
  done; done
done
