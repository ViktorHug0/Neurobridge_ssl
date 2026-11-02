"""AUG_LOG: one line per queued (recipe, fold) that has epochs: its sealed curve vs the base member recipe on that fold."""
import glob, json
import numpy as np
R = '/nasbrain/p20fores/Neurobridge_SSL/results/alljoined/track1/cv4'
BASES = ['reve_d6', 'tsc_dbn40_bn64', 'sqf_dbn', 'reve_scratch', 'rmix_reg']


def curve(r, k):
    f = sorted(glob.glob(f'{R}/{r}/fold{k}/*/*/sealed_per_epoch.jsonl'))
    return [np.mean(list(json.loads(l)['by_subject'].values())) for l in open(f[-1])] if f else []


for line in open('/nasbrain/p20fores/Neurobridge_SSL/neurips_challenge/overnight/aug_queue.txt'):
    if line.startswith('#') or not line.strip():
        continue
    r, k = line.split()
    c = curve(r, k)
    if not c:
        continue
    base = {'tsc_dbn60_bn64_things': 'tsc_dbn40_bn64', 'sqf_dbn60_things': 'sqf_dbn'}.get(r)  # longer-schedule runs of a base member
    b = curve((base or next((x for x in sorted(BASES, key=len, reverse=True) if r.startswith(x + '_')), 'tsc_dbn40_bn64')) + '_things', k)
    d = [x - y for x, y in zip(c, b)]
    done = bool(glob.glob(f'{R}/{r}/fold{k}/*/*/track1_score_sealed.json'))
    print(f'{r:28s} f{k} ep{len(c):2d}{" done" if done else "     "} best {max(c):5.1f} vs {max(b):5.1f} ({max(c) - max(b):+.1f})  matched {np.mean(d):+.2f}  last {c[-1]:.1f}/{b[min(len(c), len(b)) - 1]:.1f}')
