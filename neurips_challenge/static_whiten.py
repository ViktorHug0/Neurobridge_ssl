"""Gallery-free whitening from the query side (SAGE-style whitening moved into predict()): the grader scores
cos(q, g) with unit-norm DINOv2 targets g; returning q' = (S + lam I)^(-alpha) q makes it score q^T (S+lam)^(-alpha) g,
i.e. a whitened inner product against the gallery, with S the covariance of unit-norm TRAINING-image DINOv2
features (carried by the submission). Static: no batch statistics, immune to shuffling. Linear, so it commutes
with the grader's repeat averaging; tested on the reduced per-(subject, image) means (ens_transform.py reduce).
    static_whiten.py <recipe> [<recipe> ...]"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

OUT = 'results/alljoined/track1/routing'
recipes = sys.argv[1:]
Z = {r: np.load(f'{OUT}/Q_{r}.npz') for r in recipes}; G = Z[recipes[0]]['gallery']
T = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy').reshape(-1, G.shape[1]).astype(np.float64)
T /= np.linalg.norm(T, axis=1, keepdims=True)
S = np.cov(T, rowvar=False); w, V = np.linalg.eigh(S); w = np.maximum(w, 0)
subs = sorted(s for f in FOLDS.values() for s in f)


def top5(Q):
    Sc = Q @ G.T
    return 100 * np.mean([(Sc[i] > Sc[i, i]).sum() < 5 for i in range(len(Sc))])


def run(tag):
    base = None
    print(f'== {"+".join(recipes)} [{tag}]')
    for alpha in (0, 0.25, 0.5, 0.75, 1.0):
        for lam_rel in ((0.0,) if alpha == 0 else (1e-3, 1e-2, 1e-1)):
            lam = lam_rel * w.mean()
            M = (V * (w + lam) ** (-alpha)) @ V.T
            acc = np.array([top5(sum(Z[r][f'{tag}_s{s}'] for r in recipes) @ M) for s in subs])
            base = acc if base is None else base
            print(f'  alpha {alpha:4.2f} lam {lam_rel:6.0e}: {acc.mean():6.2f}  {acc.mean() - base.mean():+.2f}  better {(acc > base).sum():2d} worse {(acc < base).sum():2d}')


run('plain'); run('tf')
