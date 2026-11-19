#!/bin/bash
# refresh_final.sh <names...>: put the current neurips_challenge/submission.py into built candidates (members unchanged), replay fold 1,
# re-zip, contract check, statelessness check, size + sha256. Log: logs/mi/build_final.out
cd /nasbrain/p20fores/Neurobridge_SSL; S=neurips_challenge/submissions
for n in "$@"; do
  command cp -f neurips_challenge/submission.py $S/$n/submission.py; command rm -rf "${S:?}/${n:?}/__pycache__"
  .venv/bin/python neurips_challenge/package_submission.py $S/$n --verify 1 --delay 0 2>&1 | grep "batch 64\|Error\|left out" | sed "s/^/$n (code $(md5sum neurips_challenge/submission.py | cut -c1-8)) delay 0: /"
  PYTHONPATH=. .venv/bin/python neurips_challenge/check_stateless.py $S/$n 2>&1 | grep "first call\|Error" | sed "s/^/$n: /"
  (cd $S/$n && zip -q ../$n.new.zip * -x "__pycache__/*") && command mv -f $S/$n.new.zip $S/$n.zip
  bash neurips_challenge/check_package.sh $S/$n.zip 2>&1 | tail -1 | cut -c1-200; ls -la $S/$n.zip | awk '{print $5, $9}'; sha256sum $S/$n.zip
done
