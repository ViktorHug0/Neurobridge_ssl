"""Exhaustive best-4 / best-5 score-ensemble search, scored by nested leave-one-fold-out.

Same convention as ens_search.py (L2-normalised cosine matrices, uniform mean).
Reports only the nested estimate: for each held-out fold, pick the combo that is best
on the other 9 folds, then score it on the held-out fold; average over the 10 folds.
"""
import itertools, json, sys
import numpy as np, torch
from pathlib import Path

DUMP = Path('results/things_eeg/synthetic_subjects/ensemble_screen/dumps')
MAN = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'

names, mats, ref = [], [], None
for name in sorted(MAN):
    per, labs, ok = [], [], True
    for s in range(1, 11):
        p = DUMP / f'{name}-sub{s:02d}.npz'
        if not p.exists():
            ok = False; break
        d = np.load(p)
        e = d['eeg'].astype(np.float64); i = d['image'].astype(np.float64)
        e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
        i /= np.maximum(np.linalg.norm(i, axis=1, keepdims=True), 1e-12)
        per.append((e @ i.T).astype(np.float32))
        labs.append(np.stack([d['object'], d['image_idx']], axis=1))
    if not ok:
        continue
    if ref is None:
        ref = labs
    elif not all(np.array_equal(a, b) for a, b in zip(ref, labs)):
        print(f'skip {name} (query order differs)', file=sys.stderr); continue
    names.append(name); mats.append(np.stack(per))
S = torch.from_numpy(np.stack(mats)).to(dev)          # (N, 10, 200, 200)
N, F, Q, _ = S.shape
tgt = torch.arange(Q, device=dev)
print(f'{N} candidates, {F} folds, {Q}-way, device={dev}', flush=True)

solo = {names[n]: float(np.mean([(S[n, f].argmax(1) == tgt).float().mean().item() * 100
                                 for f in range(F)])) for n in range(N)}


def per_fold_acc(combos, batch=192):
    """combos: (C,k) int64 tensor -> (C,10) float32 accuracies."""
    out = torch.empty(len(combos), F, dtype=torch.float32, device=dev)
    for b in range(0, len(combos), batch):
        idx = combos[b:b + batch]
        acc = S[idx[:, 0]].clone()
        for j in range(1, idx.shape[1]):
            acc += S[idx[:, j]]
        out[b:b + batch] = (acc.argmax(-1) == tgt).float().mean(-1) * 100
    return out


report = {}
for k in (4, 5):
    combos = torch.tensor(list(itertools.combinations(range(N), k)), dtype=torch.long, device=dev)
    print(f'k={k}: {len(combos)} combos', flush=True)
    acc = per_fold_acc(combos)
    nested, picks = [], []
    for h in range(F):
        mask = torch.ones(F, dtype=torch.bool, device=dev); mask[h] = False
        best = int(acc[:, mask].mean(1).argmax())
        nested.append(float(acc[best, h])); picks.append(tuple(names[i] for i in combos[best].tolist()))
    sel = int(acc.mean(1).argmax())
    report[k] = dict(nested=float(np.mean(nested)), nested_perfold=nested, picks=picks,
                     selected=[names[i] for i in combos[sel].tolist()],
                     selected_mean=float(acc[sel].mean()), selected_perfold=acc[sel].tolist())
    print(f'  nested LOFO = {np.mean(nested):.2f}', flush=True)

json.dump(dict(solo=solo, report={str(k): v for k, v in report.items()}),
          open('ensemble_experiments/legacy/scratch_claude/ens45_results.json', 'w'), indent=1)
for k, r in report.items():
    print(f"\n=== k={k} ===")
    print(f"  nested LOFO estimate: {r['nested']:.2f}")
    print(f"  per held-out fold: " + ' '.join(f'{s}:{v:.1f}' for s, v in zip(range(1, 11), r['nested_perfold'])))
    uniq = {}
    for p in r['picks']:
        uniq[p] = uniq.get(p, 0) + 1
    print('  combo chosen by the 9-fold selector (count/10):')
    for p, c in sorted(uniq.items(), key=lambda kv: -kv[1]):
        print(f'    {c:2d}x  ' + ' + '.join(f'{n}({solo[n]:.2f})' for n in p))
