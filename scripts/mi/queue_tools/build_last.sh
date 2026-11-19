#!/bin/bash
# build_last.sh <name> <9 weights, comma-separated, member order of sub25b> <pow> <dead> [noalign]: the last-upload package from sub25b's exported
# members (same files), the current submission.py, gate 0.10, dead zone <dead>, per-trial power <pow>. Replay fold 1 at delay 0 and -4,
# statelessness check, zip, ('noalign': no detector file, detector=None: the batch is never moved, ~10 % faster), contract check on both versions of the organisers' code, size + sha256.
cd /nasbrain/p20fores/Neurobridge_SSL; S=neurips_challenge/submissions; n=$1; src=$S/sub25b_ens9_lag5jit_soft_st2_revepre_align
[ -n "$n" ] && [ -n "$2" ] && [ -n "$3" ] && [ -n "$4" ] || { echo "usage: build_last.sh <name> <w1,..,w9> <pow> <dead>"; exit 1; }
command rm -rf "${S:?}/${n:?}" "${S:?}/${n:?}.zip"; mkdir $S/$n; command cp $src/member*.pt $S/$n/; command cp neurips_challenge/submission.py $S/$n/
.venv/bin/python -c "
import torch; a = torch.load('$src/members.pt', weights_only=True); w = [float(v) for v in '$2'.split(',')]; assert len(w) == len(a['members']) == 9
a.update(weights=w, gate=0.10, dead=int($4), pow=float($3)); a.update(detector=None) if '$5' == 'noalign' else None; torch.save(a, '$S/$n/members.pt'); print('$n:', {k: a[k] for k in ('weights', 'gate', 'dead', 'pow', 'amp', 'detector')})"
[ "$5" = noalign ] && command rm $S/$n/member9_gen.pt
for d in 0 -4; do .venv/bin/python neurips_challenge/package_submission.py $S/$n --verify 1 --delay $d 2>&1 | grep "batch 64\|Error\|left out" | sed "s/^/$n delay $d: /"; done
PYTHONPATH=. .venv/bin/python neurips_challenge/check_stateless.py $S/$n 2>&1 | grep "first call\|Error" | sed "s/^/$n: /"
(cd $S/$n && zip -q ../$n.zip * -x "__pycache__/*")
echo "$n contract (Sep): $(bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-160)"
echo "$n contract (16 Nov): $(COMPET=/nasbrain/p20fores/mental_imagery/2026-competition-latest bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-200)"
ls -la $S/$n.zip | awk '{print $5, $9}'; sha256sum $S/$n.zip
