#!/bin/bash
# Waits until the r1_competitors.csv of all 3 seeds of atm128 and tsconv exist (final_analyses.sh), then writes
# results/things_eeg/cvpr/final/main_table.md (seed_table.py) and inserts it at the top of EXPERIMENT_LOG.md.
main() {
cd /nasbrain/p20fores/Neurobridge_SSL
while [ $(ls results/things_eeg/cvpr/final/{atm128,tsconv}_s330{0,1,2}/r1_competitors.csv 2>/dev/null | wc -l) -lt 6 ]; do sleep 300; done
sleep 600  # let sattc.csv / ablate.csv of the last seed land too
.venv/bin/python CVPR_paper/scripts/seed_table.py --arms atm128 tsconv --seeds 3300 3301 3302 > /dev/null || return 1
.venv/bin/python - <<'PY'
p = 'CVPR_paper/EXPERIMENT_LOG.md'
s = open(p).read()
t = open('results/things_eeg/cvpr/final/main_table.md').read()
head = ('## AUTO (main_table_when_done.sh) — E2.2 main table over 3 seeds, P-main, cross-fold SAGE params\n\n'
        'Generated automatically by `seed_table.py` when the last seed battery finished; NOT yet reviewed.\n\n')
i = s.index('\n---\n') + 5
open(p, 'w').write(s[:i] + '\n' + head + t + '\n' + s[i:])
PY
}
main "$@"; exit $?
