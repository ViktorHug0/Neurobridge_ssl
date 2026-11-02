"""Static SAGE-style map from the query side: score = q^T A g with A = Wq R Wg, Wq = (Sq + l)^(-1/2) (query
whitening after centring on mq), Wg = (Sg + l)^(-1/2) (gallery whitening, Sg from TRAINING images), R = orthogonal
Procrustes between whitened queries and whitened targets. Returned q' = Wg R^T Wq (q - mq).
Fitted on the OTHER folds' subjects' per-(subject, image) means (reduced Q files).
!! The fit uses the same 200 test images as the evaluation -> R (and Sq) are LEAKY: an upper bound on what a clean
fit (on training images) could give. Rank cut k keeps the whitening in the top-k query PCs.
    static_sage.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

OUT = 'results/alljoined/track1/routing'
recipes = sys.argv[1:]
Z = {r: np.load(f'{OUT}/Q_{r}.npz') for r in recipes}; G = Z[recipes[0]]['gallery'].astype(np.float64)
T = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy').reshape(-1, G.shape[1]).astype(np.float64)
T /= np.linalg.norm(T, axis=1, keepdims=True)
mg = T.mean(0); wg, Vg = np.linalg.eigh(np.cov(T, rowvar=False)); wg = np.maximum(wg, 0)
Q = {s: sum(Z[r][f'plain_s{s}'] for r in recipes).astype(np.float64) for f in FOLDS.values() for s in f}
fold_of = {s: k for k, v in FOLDS.items() for s in v}


def top5(P):
    Sc = P @ G.T
    return 100 * np.mean([(Sc[i] > Sc[i, i]).sum() < 5 for i in range(len(Sc))])


def inv_sqrt(w, V, lam):
    return (V * (w + lam) ** -0.5) @ V.T


res = {}
for s in sorted(Q):
    tr = [t for t in Q if fold_of[t] != fold_of[s]]
    X = np.concatenate([Q[t] for t in tr]); Y = np.concatenate([G] * len(tr))
    mq = X.mean(0); wq, Vq = np.linalg.eigh(np.cov(X, rowvar=False)); wq = np.maximum(wq, 0)
    res.setdefault('plain', []).append(top5(Q[s]))
    for lq in (1e-2, 1e-1, 1.0):
        Wq = inv_sqrt(wq, Vq, lq * wq.mean())
        for lg in (1e-1, 1.0):
            Wg = inv_sqrt(wg, Vg, lg * wg.mean())
            res.setdefault(f'whiten only lq={lq} lg={lg}', []).append(top5((Q[s] - mq) @ Wq @ Wg))
            A = (X - mq) @ Wq; B = (Y - mg) @ Wg
            U, _, Vt = np.linalg.svd(A.T @ B); R = U @ Vt
            res.setdefault(f'whiten + R(leaky) lq={lq} lg={lg}', []).append(top5((Q[s] - mq) @ Wq @ R @ Wg))
        res.setdefault(f'query whiten only lq={lq}', []).append(top5((Q[s] - mq) @ Wq))
base = np.array(res['plain'])
print('+'.join(recipes))
for k, v in res.items():
    v = np.array(v); print(f'  {k:40s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')
