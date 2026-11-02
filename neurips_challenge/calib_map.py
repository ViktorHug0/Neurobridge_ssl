"""Clean static calibration from held-out subjects' outputs on TRAINING images (train_split_cache.py), evaluated on
the 200 test images of a subject the fit never saw (leave-one-subject-out within each fold model; no image leak).
  static mean+dir  q - mu - 0.5 * proj_d(q - mu) + 0.5 * mu, mu/d from the other subjects' training-image outputs
  ridge map        q' = q W, W = argmin |Q W - Y|^2 + lam |W - I|^2 (shrunk toward identity), Q = per-image mean
                   outputs, Y = unit-norm DINOv2 targets of the training images
Applied to the reduced per-(subject, image) test means (linear maps commute with repeat averaging).
    calib_map.py <recipe>"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

OUT = 'results/alljoined/track1/routing'
r = sys.argv[1]
Z = np.load(f'{OUT}/Q_{r}.npz'); G = Z['gallery'].astype(np.float64)
Y = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy').reshape(-1, G.shape[1]).astype(np.float64)
Y /= np.linalg.norm(Y, axis=1, keepdims=True)
fold_of = {s: k for k, v in FOLDS.items() for s in v}
TQ = {s: np.load(f'{OUT}/trainQ_{r}_s{s}.npz')['means'].astype(np.float64) for s in fold_of}
mn = {s: np.linalg.norm(np.load(f'{OUT}/trainQ_{r}_s{s}.npz')['singles'].astype(np.float64), axis=1).mean() for s in TQ}  # single-trial norm, as in Q


def top5(P):
    Sc = P @ G.T
    return 100 * np.mean([(Sc[i] > Sc[i, i]).sum() < 5 for i in range(len(Sc))])


res = {}
rng = np.random.default_rng(0)
for s in sorted(fold_of):
    others = [t for t in FOLDS[fold_of[s]] if t != s]
    Q = Z[f'plain_s{s}'].astype(np.float64)                       # test means (already / member mean norm)
    scale = np.mean([mn[t] for t in others])
    X = np.concatenate([TQ[t] for t in others]) / scale            # training-image outputs, same scale convention
    res.setdefault('plain', []).append(top5(Q))
    mu = X.mean(0); d = np.linalg.svd(X[rng.choice(len(X), 6000, replace=False)] - mu, full_matrices=False)[2][0]
    c = Q - mu
    res.setdefault('clean static mean+dir', []).append(top5(c - 0.5 * np.outer(c @ d, d) + 0.5 * mu))
    res.setdefault('clean static mean (a=.5)', []).append(top5(Q - 0.5 * mu))
    Yt = np.concatenate([Y] * len(others))
    XtX, XtY = X.T @ X, X.T @ Yt
    for lam_rel in (1, 10, 100, 1000):
        lam = lam_rel * np.trace(XtX) / len(XtX)
        W = np.linalg.solve(XtX + lam * np.eye(len(XtX)), XtY + lam * np.eye(len(XtX)))
        res.setdefault(f'ridge map lam={lam_rel:g}', []).append(top5(Q @ W))
        # centred variant: no common offset toward the mean image (E5: constant mean-image shifts are ruinous)
        Xc, Yc = X - mu, Yt - Yt.mean(0)
        Wc = np.linalg.solve(Xc.T @ Xc + lam * np.eye(len(XtX)), Xc.T @ Yc + lam * np.eye(len(XtX)))
        res.setdefault(f'centred ridge map lam={lam_rel:g}', []).append(top5((Q - mu) @ Wc))
base = np.array(res['plain'])
print(r, '(clean: fit on other held-out subjects x TRAINING images; eval on test images)')
for k, v in res.items():
    v = np.array(v); print(f'  {k:28s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')
