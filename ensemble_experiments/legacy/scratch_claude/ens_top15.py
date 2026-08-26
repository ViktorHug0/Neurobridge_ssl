"""Ensemble selection restricted to the 15 strongest candidates.

Variant A (as asked): fix the top-15 by overall solo accuracy, then run the usual
  exhaustive search + nested leave-one-fold-out.
Variant B (leak-free): inside each LOFO iteration, re-pick the top 15 using ONLY the
  9 training folds, then select the combo on those same 9 and score the held-out fold.
  Nothing about the held-out subject touches either choice.
"""
import itertools, json
import numpy as np
from pathlib import Path

DUMP = Path('results/things_eeg/synthetic_subjects/ensemble_screen/dumps')
MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
TOPN = 15
names, mats, ref = [], [], None
for n in sorted(MAN):
    per, labs, ok = [], [], True
    for s in range(1, 11):
        p = DUMP / f'{n}-sub{s:02d}.npz'
        if not p.exists():
            ok = False; break
        d = np.load(p)
        e = d['eeg'].astype(np.float64); i = d['image'].astype(np.float64)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        i /= np.maximum(np.linalg.norm(i, axis=1, keepdims=True), 1e-12)
        per.append((e @ i.T).astype(np.float32))
        labs.append(np.stack([d['object'], d['image_idx']], axis=1))
    if not ok: continue
    if ref is None: ref = labs
    elif not all(np.array_equal(a, b) for a, b in zip(ref, labs)): continue
    names.append(n); mats.append(np.stack(per))
S = np.stack(mats)                                   # (N,10,200,200)
tgt = np.arange(200)
hit = np.stack([(S[k].argmax(2) == tgt[None, :]) for k in range(len(names))])   # (N,10,200)
solo = hit.mean((1, 2)) * 100
print(f'{len(names)} candidates loaded\n')

order = np.argsort(-solo)[:TOPN]
pool = [names[i] for i in order]
print(f'=== Variant A pool (top {TOPN} by overall solo) ===')
print('  ' + ', '.join(f'{n}({solo[i]:.2f})' for i, n in zip(order, pool)) + '\n')

def acc_of(idx, folds):
    """mean top1 over `folds` of the uniform ensemble of candidate indices idx"""
    f = S[list(idx)][:, folds].mean(0)
    return (f.argmax(2) == tgt[None, :]).mean(1) * 100

print(f'{"k":>2}  {"selected(biased)":>17}  {"nested LOFO":>11}   winner (variant A)')
for k in range(1, 6):
    combos = list(itertools.combinations(order, k))
    allf = np.arange(10)
    means = np.array([acc_of(c, allf).mean() for c in combos])
    best = int(means.argmax())
    nested = []
    for h in range(10):
        tr = np.array([x for x in range(10) if x != h])
        sc = np.array([acc_of(c, tr).mean() for c in combos])
        nested.append(acc_of(combos[int(sc.argmax())], np.array([h]))[0])
    print(f'{k:2d}  {means[best]:17.2f}  {np.mean(nested):11.2f}   ' +
          ' + '.join(names[i] for i in combos[best]))

print(f'\n=== Variant B: top {TOPN} re-picked inside each fold (no test leakage) ===')
print(f'{"k":>2}  {"nested LOFO":>11}')
for k in range(1, 6):
    nested = []
    for h in range(10):
        tr = np.array([x for x in range(10) if x != h])
        solo_tr = hit[:, tr, :].mean((1, 2)) * 100
        pool_h = np.argsort(-solo_tr)[:TOPN]
        combos = list(itertools.combinations(pool_h, k))
        sc = np.array([acc_of(c, tr).mean() for c in combos])
        nested.append(acc_of(combos[int(sc.argmax())], np.array([h]))[0])
    print(f'{k:2d}  {np.mean(nested):11.2f}')
