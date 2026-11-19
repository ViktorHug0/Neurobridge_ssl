#!/bin/bash
# build_swap.sh <name> <base package> <slot> <final run dir> [w1,..,wN]: <base> with member <slot> replaced by the TorchScript export of <run dir>
# (same file name, so members.pt keeps its list; its 'sources' entry is updated; optional new member weights). Then as build_last.sh: replay fold 1 at delay 0,
# statelessness check, zip, contract check on both versions of the organisers' code, size + sha256.
cd /nasbrain/p20fores/Neurobridge_SSL; S=neurips_challenge/submissions; n=$1 base=$2 slot=$3 run=$4  # $5 = optional weights
[ -n "$n" ] && [ -d "$S/$base" ] && [ -n "$slot" ] && [ -d "$run" ] || { echo "usage: build_swap.sh <name> <base> <slot> <run dir>"; exit 1; }
command rm -rf "${S:?}/${n:?}" "${S:?}/${n:?}.zip" "${S:?}/_tmp_${n:?}"; mkdir $S/$n; command cp $S/$base/*.pt $S/$base/submission.py $S/$n/
.venv/bin/python neurips_challenge/package_submission.py $S/_tmp_$n $run 2>&1 | grep "<-\|Error\|assert" | cut -c1-160
f=$(.venv/bin/python -c "import torch; print(torch.load('$S/$n/members.pt', weights_only=True)['members'][$slot])")
command mv $S/_tmp_$n/member0_*.pt $S/$n/$f && command rm -rf "${S:?}/_tmp_${n:?}"
.venv/bin/python -c "
import torch; a = torch.load('$S/$n/members.pt', weights_only=True); a['sources'][$slot] = '$run/checkpoint_test_best.pth'; a.update(weights=[float(v) for v in '$5'.split(',')]) if '$5' else None; torch.save(a, '$S/$n/members.pt')
print('$n: slot $slot =', a['members'][$slot], '<-', a['sources'][$slot], {k: a.get(k) for k in ('weights', 'amp', 'detector', 'dead', 'pow')})"
.venv/bin/python neurips_challenge/package_submission.py $S/$n --verify 1 --delay 0 2>&1 | grep "batch 64\|Error\|left out" | sed "s/^/$n delay 0: /"
PYTHONPATH=. .venv/bin/python neurips_challenge/check_stateless.py $S/$n 2>&1 | grep "first call\|Error" | sed "s/^/$n: /"
(cd $S/$n && zip -q ../$n.zip * -x "__pycache__/*")
echo "$n contract (Sep): $(bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-160)"
echo "$n contract (16 Nov): $(COMPET=/nasbrain/p20fores/mental_imagery/2026-competition-latest bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-200)"
ls -la $S/$n.zip | awk '{print $5, $9}'; sha256sum $S/$n.zip
