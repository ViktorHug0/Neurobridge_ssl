#!/bin/bash
# Screen under the cv4 protocol: every recipe in $ORDER on folds 1-2 (10 test subjects), then folds 3-4 only
# for recipes whose screen verdict against their reference is "promote" (CV4_PROTOCOL.md §6).
#   usage: ORDER="eegnet rct_sub64:robust_ct_e40 ..." bash neurips_challenge/overnight/cv4_arch_screen.sh
#   a token is <recipe>[:<reference recipe>], reference default tsc.
# Several workers can share recipes: cv4.sh skips scored folds and alljoined_run.sh claims runs.
cd "$(dirname "$0")/../.."
for t in $ORDER; do bash neurips_challenge/cv4.sh "${t%%:*}" 1 2; done
for t in $ORDER; do
  r=${t%%:*}; ref=tsc; [[ $t == *:* ]] && ref=${t#*:}
  v=$(.venv/bin/python neurips_challenge/cv4_summary.py --ref "$ref" --verdict "$r")
  echo "=== verdict $r vs $ref: $v"
  [[ $v == promote* ]] && bash neurips_challenge/cv4.sh "$r" 3 4
done
echo "screen done: $ORDER"
