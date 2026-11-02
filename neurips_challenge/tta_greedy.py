"""TTA_LOG T49: greedy forward choice of <= 5 non-KD members under the lean transform (lwn K4 c1), NESTED: members are
chosen on folds A and scored on folds B (and the reverse), so the reported number is not selected on its own people.
Per fold, each member's person-centred output is computed once; a subset = sum, then one shrink, then the official score.
    tta_greedy.py
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import official  # noqa: E402
from neurips_challenge.tta_lab import BS, DEV, load, lw_shrink, mn, wccn_affinity  # noqa: E402

POOL = ['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_bn64_things', 'rmix_things', 'reve_scratch_things', 'lab_s10_things',
        'rmix_reg_things', 'rmixd8_things', 'atm_dbn_things', 'plab_gain_things', 'reve_z_things', 'dualstream']


def centred(k, groups=('base',)):
    """per member: person-centred (K4 c1) outputs + batch mean (fp16, CPU); + stream labels (groups: fingerprint)"""
    ctx = load(k, POOL); aff, feat = wccn_affinity(ctx, groups=groups)
    Ws = []
    for b in range(0, len(ctx['img']), BS):
        S = aff(feat[b:b + BS]).float(); S.fill_diagonal_(-9)
        W = torch.zeros_like(S).scatter_(1, S.topk(min(4, len(S) - 1), 1).indices, 1.); Ws.append(W / W.sum(1, keepdim=True))
    C = []
    for m in range(len(POOL)):
        out = []
        for i, b in enumerate(range(0, len(ctx['img']), BS)):
            q = mn([ctx['P'][m][b:b + BS]])[0]; out.append((q - (Ws[i] @ q - q.mean(0, keepdim=True))).half())
        C.append(torch.cat(out).cpu())  # CPU: 12 members x 4 folds do not fit on one GPU
    r = C, ctx['img'], ctx['subj'], ctx['G']; del ctx; torch.cuda.empty_cache()
    return r


def score(data, subset):
    res = {}
    for C, img, subj, G in data:
        y = sum(C[m].to(DEV).float() for m in subset)
        y = torch.cat([lw_shrink(y[b:b + BS], y.shape[1] / (y.shape[1] + len(y[b:b + BS])), False) for b in range(0, len(y), BS)])
        sc = official(y.double().cpu().numpy(), img, subj, G); res.update({int(s): sc[s] for s in np.unique(subj)})
    return 100 * np.mean(list(res.values()))


def greedy(data, n=5):
    chosen, best = [], -1
    for _ in range(n):
        cand = [(score(data, chosen + [m]), m) for m in range(len(POOL)) if m not in chosen]
        s, m = max(cand)
        if s <= best:
            break
        chosen.append(m); best = s
        print(f'  + {POOL[m]}: {s:.2f}', flush=True)
    return chosen, best


if __name__ == '__main__':
    data = {k: centred(k) for k in (1, 2, 3, 4)}
    for A, B in (((1, 2), (3, 4)), ((3, 4), (1, 2))):
        print(f'choose on folds {A}', flush=True)
        ch, s = greedy([data[k] for k in A])
        print(f'RESULT chosen on {A}: {[POOL[m] for m in ch]} -> {s:.2f} there, {score([data[k] for k in B], ch):.2f} on {B}', flush=True)
    ch, s = greedy([data[k] for k in (1, 2, 3, 4)])
    print(f'RESULT chosen on all 20 (optimistic): {[POOL[m] for m in ch]} -> {s:.2f}', flush=True)
