"""ns09_export.py <s09_person_f*.npz> <out dir>: the per-subject alignment results of ns09_align_policy.py as two JSON files for the
'Alignment by subject' page: subject_shift.json (share of one-person batches per proposed correction, true delay 0 and -4) and
subject_gain.json (top-5 change against no alignment per subject, policy, batch type and delay; bootstrap sd over the subject's images)."""
import sys, json, re, numpy as np
from pathlib import Path
z = np.load(sys.argv[1]); out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
P, F = z['persons'].tolist(), z['fold_of_person'].tolist(); fold = dict(zip(P, F))
shift = []
for d in (0, -4):
    bp, bc = z['batch_person_d%d' % d], z['batch_corr_d%d' % d]
    for p in P:
        c = bc[bp == p]
        for v in np.unique(c):
            shift.append({'subject': 'S%02d' % p, 'fold': fold[p], 'delay': d, 'correction': int(v), 'share': round(float((c == v).mean()), 4), 'batches': int(len(c))})
gain = []
for k in z.files:
    m = re.match(r'(.+)\|d(-?\d+)\|(.+)', k)
    if not m:
        continue
    name, d, pol = m.group(1), int(m.group(2)), m.group(3)
    for p, (gn, sd, b) in zip(P, z[k]):
        gain.append({'subject': 'S%02d' % p, 'fold': fold[p], 'batches': name, 'delay': d, 'policy': pol, 'gain': round(float(gn), 2), 'sd': round(float(sd), 2), 'base_top5': round(float(b), 3)})
code = [{'label': f, 'value': Path(f).read_text(), 'type': 'code', 'lang': 'python'} for f in ('scripts/mi/ns09_align_policy.py', 'scripts/mi/ns09_export.py')]
meta = code + [{'label': 'Input', 'value': Path(sys.argv[1]).name}, {'label': 'Folds', 'value': ', '.join(map(str, sorted(set(F))))}]
for name, rows in (('subject_shift', shift), ('subject_gain', gain)):
    (out / f'{name}.json').write_text(json.dumps({'rows': rows, 'meta': meta}))
    print(name, len(rows), 'rows')
