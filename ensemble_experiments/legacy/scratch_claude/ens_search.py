"""Exhaustive best-2 / best-3 score-ensemble search over dumped candidates.

Convention matches ensemble_experiments/synthetic_subjects/score_npz_ensemble.py: L2-normalise
both sides, cosine score matrix per fold, uniform mean over members, top-1 over the
200-way test set. Also reports a leave-one-fold-out nested estimate, because picking
the argmax over thousands of combos on the same 10 folds is selection on test.
"""
import itertools, json, sys
import numpy as np
from pathlib import Path

DUMP = Path('results/things_eeg/synthetic_subjects/ensemble_screen/dumps')
MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
SUBS = range(1, 11)


def load(name):
    mats, labs = [], []
    for s in SUBS:
        p = DUMP / f'{name}-sub{s:02d}.npz'
        if not p.exists():
            return None, None
        d = np.load(p)
        eeg = d['eeg'].astype(np.float64); img = d['image'].astype(np.float64)
        eeg /= np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-12)
        img /= np.maximum(np.linalg.norm(img, axis=1, keepdims=True), 1e-12)
        mats.append((eeg @ img.T).astype(np.float32))
        labs.append(np.stack([d['object'], d['image_idx']], axis=1))
    return mats, labs


def top1(m):
    return float(np.mean(m.argmax(axis=1) == np.arange(len(m))) * 100.0)


S, ref = {}, None
for name in sorted(MAN):
    mats, labs = load(name)
    if mats is None:
        print(f'skip {name} (incomplete dump)', file=sys.stderr); continue
    if ref is None:
        ref = labs
    elif not all(np.array_equal(a, b) for a, b in zip(ref, labs)):
        print(f'skip {name} (query order differs)', file=sys.stderr); continue
    S[name] = mats
names = sorted(S)
print(f'{len(names)} candidates with complete dumps\n')

perfold = {n: np.array([top1(m) for m in S[n]]) for n in names}


def combo(members):
    return np.array([top1(np.mean([S[n][i] for n in members], axis=0)) for i in range(10)])


results = {}
for k in (1, 2, 3):
    rows = []
    for c in itertools.combinations(names, k):
        f = perfold[c[0]] if k == 1 else combo(c)
        rows.append((f.mean(), c, f))
    rows.sort(key=lambda r: -r[0])
    results[k] = rows
    print(f'=== best {k}-model ensembles ===')
    for mean, c, f in rows[:12]:
        solo = ' + '.join(f'{n}({perfold[n].mean():.2f})' for n in c)
        print(f'  {mean:6.2f}  {solo}')
    # nested: choose on 9 folds, score the 10th
    nested = []
    for h in range(10):
        best = max(rows, key=lambda r: np.delete(r[2], h).mean())
        nested.append(best[2][h])
    print(f'  nested LOFO estimate (select on 9 folds, score held-out): {np.mean(nested):.2f}\n')

json.dump({str(k): [(m, list(c), list(f)) for m, c, f in v[:50]] for k, v in results.items()},
          open('ensemble_experiments/legacy/scratch_claude/ens_search_results.json', 'w'), indent=1)
best3 = results[3][0]
print('per-fold of best 3-model:', ' '.join(f'{s}:{v:.1f}' for s, v in zip(SUBS, best3[2])))
