"""Diversity-aware pruning: keep a member only if it is not redundant with a
stronger one, instead of keeping the most accurate members.

Greedy filter, applied INSIDE each leave-one-fold-out iteration using only the 9
training folds (so neither the ranking, the correlations, nor the combo choice
sees the held-out subject):
   walk members strongest-first; keep one iff its correlation with every
   already-kept member is below tau.
Redundancy is measured on per-query-standardised score matrices, so two members
that rank the 200 candidates the same way count as redundant even if their
absolute accuracy differs.
"""
import itertools, json
import numpy as np
from pathlib import Path

DUMP = Path('results/things_eeg/synthetic_subjects/ensemble_screen/dumps')
MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
names, mats, ref = [], [], None
for n in sorted(MAN):
    per, labs, ok = [], [], True
    for s in range(1, 11):
        p = DUMP / f'{n}-sub{s:02d}.npz'
        if not p.exists(): ok = False; break
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
S = np.stack(mats); N = len(names); tgt = np.arange(200)
hit = np.stack([(S[k].argmax(2) == tgt[None, :]) for k in range(N)])
Z = (S - S.mean(3, keepdims=True)) / (S.std(3, keepdims=True) + 1e-12)   # per-query standardised
print(f'{N} candidates\n')

def acc(idx, folds):
    f = S[list(idx)][:, folds].mean(0)
    return float(((f.argmax(2) == tgt[None, :]).mean(1) * 100).mean())

def prune(tr, tau, cap, floor=6):
    solo = hit[:, tr, :].mean((1, 2))
    flat = Z[:, tr].reshape(N, -1)
    flat = (flat - flat.mean(1, keepdims=True)) / (flat.std(1, keepdims=True) + 1e-12)
    C = (flat @ flat.T) / flat.shape[1]
    order = np.argsort(-solo)
    kept = []
    for i in order:
        if all(C[i, j] < tau for j in kept):
            kept.append(int(i))
            if len(kept) >= cap: break
    # never let the filter starve the search below `floor` members
    for i in order:
        if len(kept) >= floor: break
        if int(i) not in kept: kept.append(int(i))
    return kept

print(f'{"tau":>5} {"pool":>5}  ' + '  '.join(f'k={k}' for k in range(1, 6)))
for tau in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01):
    row, sizes = [], []
    for k in range(1, 6):
        nested = []
        for h in range(10):
            tr = np.array([x for x in range(10) if x != h])
            pool = prune(tr, tau, cap=18)
            sizes.append(len(pool))
            combos = list(itertools.combinations(pool, k))
            sc = np.array([acc(c, tr) for c in combos])
            nested.append(acc(combos[int(sc.argmax())], np.array([h])))
        row.append(np.mean(nested))
    star = ' <-- best' if max(row) == max(row) else ''
    print(f'{tau:5.2f} {np.mean(sizes):5.1f}  ' + '  '.join(f'{v:5.2f}' for v in row))

print('\nexample pool at the best tau (trained on folds 2-10, i.e. held-out subject 1):')
for tau in (0.40, 0.50, 0.60):
    pool = prune(np.arange(1, 10), tau, cap=18)
    solo = hit[:, 1:, :].mean((1, 2)) * 100
    print(f'  tau={tau:.2f} ({len(pool)}): ' + ', '.join(f'{names[i]}({solo[i]:.1f})' for i in pool))
