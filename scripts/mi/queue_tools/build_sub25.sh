#!/bin/bash
# sub25 = sub24 with the two swaps confirmed on three folds (17 Nov 07:45): rmix -> plain ST-EEGFormer, d6 soft -> pretrained REVE soft.
# Waits for the two finals, then builds, replays fold 1 (delay 0 and +4), times, zips and runs the contract check. Two packages:
#   sub25  (ten members, as asked)   and   sub25b (the same without REVE-d6: equal forecast at 5k / 14k / 40k, 13 ms less per batch).
cd /nasbrain/p20fores/Neurobridge_SSL; F=results/alljoined/track1/final; S=neurips_challenge/submissions; J=${1:-76147}
ST=$(dirname $F/steegformer_optx_things_lag5jit/gen/*/train.log); RV=$(dirname $F/reve_pre_optx_things_lag5jitsoft/gen/*/train.log)
until [ -e $ST/track1_score_sealed.json ] && [ -e $RV/track1_score_sealed.json ]; do sleep 60; done; date
D6=$F/reve_d6_optx_things_lag5jit/reve/20261111-090213-reve
REST="$F/sqf_dbn_optx_things_lag5jit/sqf/20261111-090211-sqf $F/reve_scratch_optx_things_lag5jit/reve/20261111-090213-reve $F/sqf_dbn_things_lag5jit/sqf/20261111-090230-sqf $F/labram_optx_things_lag5jit/gen/20261112-043327-gen $F/rmix_reg_optx_things_lag5jitsoft/gen/20261115-021301-gen $F/labram_optx_things_lag5jitsoft/gen/20261115-094536-gen $F/steegformer_optx_things_lag5jitsoft/gen/20261116-084825-gen $ST $RV"
DET=$F/rmix_reg_optx_things_lag5/gen/20261110-222147-gen
R="srun --jobid=$J --overlap --export=ALL"; [ "$J" = local ] && R=""  # local: this node's own GPU
build() {  # build <name> <amp indices> <run dirs...>
  n=$1; amp=$2; shift 2; command rm -rf $S/$n $S/$n.zip
  $R .venv/bin/python neurips_challenge/package_submission.py $S/$n "$@" $DET --detector --bias 1.5 --amp $amp 2>&1 | grep "<-\|Error\|assert" | cut -c1-160
  for d in 0 4; do $R .venv/bin/python neurips_challenge/package_submission.py $S/$n --verify 1 --delay $d 2>&1 | grep "fold 1\|Error" | sed "s/^/$n delay $d: /"; done
  (cd $S/$n && command rm -rf __pycache__ && zip -q ../$n.zip *); bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-200; ls -la $S/$n.zip | awk '{print $5, $9}'; sha256sum $S/$n.zip
}
build sub25b_ens9_lag5jit_soft_st2_revepre_align "7 8" $REST; echo SUB25B READY; date
build sub25_ens10_lag5jit_soft_st2_revepre_align "8 9" $D6 $REST
PYTHONPATH=. $R .venv/bin/python logs/mi/queue/time_pkg.py $S/sub16_ens6_lag5jit_labram $S/sub24_ens10_lag5jit_soft3_stf_align $S/sub25_ens10_lag5jit_soft_st2_revepre_align $S/sub25b_ens9_lag5jit_soft_st2_revepre_align 2>&1 | grep "ms per"
PYTHONPATH=. .venv/bin/python scripts/mi/ny04_packages_out_of_things.py sub19_ens6_lag5jit_labram_align sub24_ens10_lag5jit_soft3_stf_align sub25b_ens9_lag5jit_soft_st2_revepre_align sub25_ens10_lag5jit_soft_st2_revepre_align 2>&1 | grep -v "Warning\|montage" > logs/mi/y04_sub25.out
date; echo BUILD DONE
