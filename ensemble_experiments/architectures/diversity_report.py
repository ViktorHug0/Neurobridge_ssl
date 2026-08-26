"""Do the ortho arms decorrelate the pool, and does the k=4 ceiling move?

Two questions:
 1. Correlation. The existing 48-member pool has a score-matrix correlation FLOOR of
    0.628 (median 0.820) -- no member is decorrelated from every other, which is why
    routing/stacking/diversity-pruning all failed. An ortho arm that lands below that
    floor is genuinely new information rather than another view of the same model.
 2. Does it pay? Nested LOFO at k=2..4 over the old pool vs old+ortho.

Same conventions as ensemble_experiments/legacy/scratch_claude/ens_search45.py (L2-normalised cosine, uniform
mean, nested leave-one-fold-out selection).

Usage: .venv/bin/python ensemble_experiments/architectures/diversity_report.py
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

S = torch.from_numpy(np.stack(mats)).to(dev)                 # N, 10, 200, 200
N, F, Q, _ = S.shape
tgt = torch.arange(Q, device=dev)
solo = np.array([float(np.mean([(S[n, f].argmax(1) == tgt).float().mean().item() * 100
                                for f in range(F)])) for n in range(N)])
new = [i for i, n in enumerate(names) if n.startswith('ortho_')]
old = [i for i, n in enumerate(names) if not n.startswith('ortho_')]
print(f'{N} candidates ({len(old)} existing, {len(new)} ortho), device={dev}\n', flush=True)

# --- 1. correlation, on per-query-standardised scores -----------------------------
Z = (S - S.mean(3, keepdim=True)) / (S.std(3, keepdim=True) + 1e-12)
flat = Z.reshape(N, -1).double()
flat = (flat - flat.mean(1, keepdim=True)) / (flat.std(1, keepdim=True) + 1e-12)
C = ((flat @ flat.T) / flat.shape[1]).cpu().numpy()
oldC = C[np.ix_(old, old)][np.triu_indices(len(old), 1)]
print(f'existing pool, {len(oldC)} pairs: score-corr min {oldC.min():.3f}  '
      f'median {np.median(oldC):.3f}  max {oldC.max():.3f}')
print(f'{"":16s} {"solo":>6s} {"min-corr":>9s} {"median":>8s}   (vs the existing pool)')
for i in new:
    c = C[i, old]
    flag = '  <-- below the old floor' if c.min() < oldC.min() else ''
    print(f'{names[i]:16s} {solo[i]:6.2f} {c.min():9.3f} {np.median(c):8.3f}{flag}')

# --- 2. nested LOFO, old pool vs old+ortho ----------------------------------------
def per_fold_acc(combos, batch=192):
    out = torch.empty(len(combos), F, dtype=torch.float32, device=dev)
    for b in range(0, len(combos), batch):
        idx = combos[b:b + batch]
        acc = S[idx[:, 0]].clone()
        for j in range(1, idx.shape[1]):
            acc += S[idx[:, j]]
        out[b:b + batch] = (acc.argmax(-1) == tgt).float().mean(-1) * 100
    return out


def nested(idx, k):
    combos = torch.tensor(list(itertools.combinations(sorted(idx), k)), dtype=torch.long, device=dev)
    acc = per_fold_acc(combos)
    vals, picks = [], []
    for h in range(F):
        mask = torch.ones(F, dtype=torch.bool, device=dev); mask[h] = False
        b = int(acc[:, mask].mean(1).argmax())
        vals.append(float(acc[b, h])); picks.append(tuple(names[i] for i in combos[b].tolist()))
    return float(np.mean(vals)), picks


print()
for k in (2, 3, 4):
    o, _ = nested(old, k)
    a, picks = nested(range(N), k)
    uniq = {}
    for p in picks:
        uniq[p] = uniq.get(p, 0) + 1
    top = max(uniq.items(), key=lambda kv: -kv[1] if False else kv[1])
    print(f'k={k}  old-pool {o:.2f}   old+ortho {a:.2f}   ({a - o:+.2f})')
    print(f'      modal pick {top[1]}/10: ' + ' + '.join(top[0]))
