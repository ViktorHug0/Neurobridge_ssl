"""Ensemble selection UNDER a test-time method (TTA_LOG T13). Every method in tta_lab acts per member with shared,
member-independent neighbours, and the scorer's (person, image) averaging is linear, so the ensemble's aggregate =
sum of the members' aggregates: cache each member's per-(person, image) means once, then score any subset exactly.
Honest estimate: pick the best subset on folds 1-2, score it on folds 3-4, and vice versa.
    tta_ens.py <method> <member> [...] [--sizes 2 3 4 5]
"""
import argparse, itertools, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.official_metrics import group_means, topk_accuracy  # noqa: E402
from neurips_challenge.tta_lab import OUT, load, method  # noqa: E402

FOLDS = {1: [2, 4, 5, 11, 15], 2: [7, 14, 16, 17, 20], 3: [3, 8, 10, 12, 19], 4: [1, 6, 9, 13, 18]}
n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def member_means(name, r):
    path = f'{OUT}/ens/{r}__{name}.npz'
    if not os.path.exists(path):
        keys, means, G = [], [], None
        for k in FOLDS:
            ctx = load(k, [r])
            y = method(name, ctx).double().cpu().numpy()
            kk, mm = group_means(y, np.stack([ctx['subj'], ctx['img']], 1))
            keys.append(kk); means.append(mm.astype(np.float32)); G = ctx['G']
            del ctx; torch.cuda.empty_cache()
        os.makedirs(f'{OUT}/ens', exist_ok=True)
        np.savez(path, keys=np.concatenate(keys), means=np.concatenate(means), G=G)
    return np.load(path)


def main(name, members, sizes):
    C = {r: member_means(name, r) for r in members}
    keys, G = C[members[0]]['keys'], n(C[members[0]]['G'])
    rows = {s: keys[:, 0] == s for s in np.unique(keys[:, 0])}

    def per_subject(sub):
        P = n(sum(C[r]['means'] for r in sub))
        return {s: topk_accuracy(P[m] @ G.T, keys[m, 1], k=5) for s, m in rows.items()}
    half = {'12': FOLDS[1] + FOLDS[2], '34': FOLDS[3] + FOLDS[4]}
    sc = lambda d, ss: 100 * np.mean([d[s] for s in ss])
    for size in sizes:
        res = {sub: per_subject(sub) for sub in itertools.combinations(members, size)}
        best = max(res, key=lambda sub: sc(res[sub], range(1, 21)))
        print(f'size {size}: best {sc(res[best], range(1, 21)):.2f} {"+".join(best)}', flush=True)
        nested = []
        for pick, test in (('12', '34'), ('34', '12')):
            b = max(res, key=lambda sub: sc(res[sub], half[pick]))
            nested.append(sc(res[b], half[test]))
            print(f'  pick on {pick}: {"+".join(b)} -> {test}: {nested[-1]:.2f}', flush=True)
        print(f'  NESTED size {size}: {np.mean(nested):.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('name'); ap.add_argument('members', nargs='+'); ap.add_argument('--sizes', nargs='+', type=int, default=[2, 3, 4])
    a = ap.parse_args()
    main(a.name, a.members, a.sizes)
