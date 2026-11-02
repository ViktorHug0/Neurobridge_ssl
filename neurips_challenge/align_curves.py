"""Per-epoch sealed top-5 (half the trials, as logged during training) of a recipe vs its baseline, per fold, at
matched epochs (same one-cycle schedule up to that epoch), plus the best-epoch comparison over the folds both have.
    align_curves.py <baseline> <recipe> [<recipe> ...]
"""
import glob, json, sys
import numpy as np

ROOT = 'results/alljoined/track1/cv4'


def curves(recipe):
    """{fold: {epoch: mean over the fold's test subjects}}, newest run per fold"""
    out = {}
    for f in sorted(glob.glob(f'{ROOT}/{recipe}/fold*/*/2026*/sealed_per_epoch.jsonl')):
        k = int(f.split('/fold')[1].split('/')[0])
        out[k] = {r['epoch']: np.mean(list(r['by_subject'].values())) for r in map(json.loads, open(f))}
    return out


base = curves(sys.argv[1])
for r in sys.argv[2:]:
    c = curves(r)
    print(f'== {r} vs {sys.argv[1]}')
    best = []
    for k in sorted(c):
        if k not in base:
            continue
        ep = sorted(set(c[k]) & set(base[k]))
        row = ' '.join(f'e{e}:{c[k][e] - base[k][e]:+.1f}' for e in ep)
        print(f'  fold {k}: {row}')
        print(f'          best {max(c[k].values()):.2f} (e{max(c[k], key=c[k].get)}, {len(c[k])} ep) vs {max(base[k].values()):.2f} (e{max(base[k], key=base[k].get)})')
        best.append(max(c[k].values()) - max(base[k].values()))
    if best:
        print(f'  best-epoch delta, mean over {len(best)} folds: {np.mean(best):+.2f}')
