#!/bin/bash
# build_final.sh <name> <ST weight> [gate]: a final candidate from sub25b's exported members (no re-export: same member files), the
# hardened submission.py, the alignment gate (default 0.10) and weight w on the two ST-EEGFormer members (slots 6, 7). Replays fold 1 at
# delay 0 and -4, zips, runs the contract check, prints size and sha256. Short zip names: Codabench truncates long ones.
cd /nasbrain/p20fores/Neurobridge_SSL; S=neurips_challenge/submissions; n=$1; w=$2; g=${3:-0.10}; src=$S/sub25b_ens9_lag5jit_soft_st2_revepre_align
[ -n "$n" ] && [ -n "$w" ] || { echo "usage: build_final.sh <name> <ST weight> [gate]"; exit 1; }
command rm -rf "${S:?}/${n:?}" "${S:?}/${n:?}.zip"; mkdir $S/$n; command cp $src/member*.pt $S/$n/; command cp neurips_challenge/submission.py $S/$n/
.venv/bin/python -c "
import torch; a = torch.load('$src/members.pt', weights_only=True); a['weights'] = [1.] * 6 + [$w, $w] + [1.]; a['gate'] = $g
assert a['members'][6:8] == ['member6_gen.pt', 'member7_gen.pt'] and 'steegformer' in a['sources'][6] and 'steegformer' in a['sources'][7]
torch.save(a, '$S/$n/members.pt'); print('$n: weights', a['weights'], 'gate', a['gate'], 'amp', a['amp'])"
for d in 0 -4; do .venv/bin/python neurips_challenge/package_submission.py $S/$n --verify 1 --delay $d 2>&1 | grep "batch 64\|Error\|left out" | sed "s/^/$n delay $d: /"; done
(cd $S/$n && zip -q ../$n.zip * -x "__pycache__/*"); bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-200
ls -la $S/$n.zip | awk '{print $5, $9}'; sha256sum $S/$n.zip
