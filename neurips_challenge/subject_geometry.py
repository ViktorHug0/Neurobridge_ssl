"""How do unseen subjects differ, and how much of it is correctable? (SUBJECT_LOG.md, user question 2026-10-15)
Frozen cv4 fold models, the 20 held-out Alljoined subjects (each predicted by the fold model that held it out), per-trial
outputs from trial_cache.py. P_s = the subject's per-image mean output (the grader's aggregation), after the deployed
first-order fix (label-free centring P - 0.5 * subject mean). Corrections act on unit rows inside S = top-k principal
subspace of the DINOv2 TRAINING-image targets (a fixed basis shared by all subjects and models).
D0  relational structure: RSA between subjects' 200x200 output RDMs, and with the DINOv2 RDM.
D1  where does the gap live? Corrections fitted with labels on 100 images (split A) and scored on the other 100 (B,
    200-way gallery, subject-aggregated top-5), averaged over random splits:
      rotation  orthogonal Procrustes in S      linear  ridge x_S (I + D) toward the identity (lambda grid)
    (linear part only: both sides centred, targets scale-matched; the prediction keeps its own mean)
    each fitted on the subject itself (oracle, needs labels) and on the other 19 subjects' A images (population).
    subject-specific headroom = oracle - population.
D2  how many axes? PCA of the 20 per-subject affine corrections; leave-one-subject-out: the subject's correction
    projected on the top-j axes of the other 19 (around their mean), scored on B.
D3  inferable without labels? The subject's top-j coordinates regressed (LOO ridge over subjects) on label-free
    statistics: its mean output in S, from all trials or from 6 random trials (the sealed batch has ~6 per subject).
    subject_geometry.py <recipe> [...] [--k 16] [--splits 10]
"""
import argparse, glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_ensemble import DATA, run_dirs  # noqa: E402
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

TARGETS = '/nasbrain/p20fores/Neurobridge_SSL/data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy'
n = lambda a: a / np.linalg.norm(a, axis=-1, keepdims=True)


def top5(X, G, idx):
    s = n(X[idx]) @ n(G).T
    return float(np.mean([(s[r] > s[r, i]).sum() < 5 for r, i in enumerate(idx)]))


def load(recipe):
    """{subject: (P (200, 1536) per-image mean, m (1536) all-trial mean, T (n, 1536) valid trials, y (n,))}, gallery."""
    out, G = {}, None
    for k, subs in FOLDS.items():
        z = np.load(glob.glob(f'{run_dirs(recipe)[k]}/trial_dots.npz')[0])
        G = z['gallery']
        for s in subs:
            counts = np.minimum(np.load(f'{DATA}/sub-{s:02d}/test_counts.npy').ravel(), 80)
            valid = (np.arange(80)[None, :] < counts[:, None]).ravel()
            T, y = z[f's{s}_emb'][valid].astype(np.float32), z[f's{s}_label'][valid]
            P = np.stack([T[y == i].mean(0) for i in range(200)])
            keep = np.random.default_rng(s).choice(len(T), 600, replace=False)  # ponytail: a trial sample for D3, not all 16k
            out[s] = (P, T.mean(0), T[keep], y[keep])
    return out, G


def fit(kind, X, Y, lam=None):
    """linear correction of S-coordinates X (n, k) toward targets Y (n, k), both centred on their own means and Y
    scale-matched to X: only the linear part of the map is learned, the prediction keeps its own mean and scale
    (an intercept / target mean would collapse the ranking: E. 2026-10-15 first run). Returns f on (m, k) given the
    subject's own centre c."""
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Yc = Yc * np.linalg.norm(Xc) / np.linalg.norm(Yc)
    if kind == 'rotation':
        U, _, Vt = np.linalg.svd(Xc.T @ Yc); M = U @ Vt
    else:  # ridge toward the identity
        M = np.eye(X.shape[1]) + np.linalg.solve(Xc.T @ Xc + lam * np.trace(Xc.T @ Xc) / X.shape[1] * np.eye(X.shape[1]),
                                                 Xc.T @ (Yc - Xc))
    return M


def apply(M, Xn, V):
    """replace the S-component of unit rows Xn (one subject, all 200 images) by its centred part mapped by M."""
    Z = Xn @ V; c = Z.mean(0)
    return Xn + ((Z - c) @ M + c - Z) @ V.T


def main(recipes, k, n_splits):
    Gt = np.load(TARGETS).reshape(-1, 1536)
    V = np.linalg.svd(n(Gt) - n(Gt).mean(0), full_matrices=False)[2][:k].T  # (1536, k) fixed basis
    for recipe in recipes:
        data, G = load(recipe)
        subs = sorted(data)
        Gn = n(G)
        Xs = {s: n(data[s][0] - 0.5 * data[s][1]) for s in subs}  # deployed first-order fix
        # D0 relational structure
        rdm = lambda X: (n(X) @ n(X).T)[np.triu_indices(200, 1)]
        R = np.array([rdm(Xs[s]) for s in subs]); rg = rdm(G)
        cc = np.corrcoef(R); off = cc[np.triu_indices(len(subs), 1)]
        print(f'RESULT {recipe} D0: RSA subject-subject {off.mean():.3f} (min {off.min():.3f}), subject-DINOv2 '
              f'{np.mean([np.corrcoef(r, rg)[0, 1] for r in R]):.3f}', flush=True)
        kinds = [('rotation', None)] + [('linear', l) for l in (0.3, 1.0, 3.0)]
        res = {f'{kd}{"" if l is None else l}_{w}': [] for kd, l in kinds for w in ('oracle', 'pop')}
        res['none'] = []; res['raw'] = []
        d2 = {j: [] for j in (0, 1, 2, 4, 8)}; d3 = {f'{src}{j}': [] for src in ('all', 'six') for j in (1, 2, 4)}
        rng = np.random.default_rng(0)
        for sp in range(n_splits):
            perm = rng.permutation(200); A, B = perm[:100], perm[100:]
            for s in subs:
                res['raw'].append(top5(data[s][0], G, B)); res['none'].append(top5(Xs[s], G, B))
            for kd, l in kinds:
                for s in subs:
                    key = f'{kd}{"" if l is None else l}'
                    fo = fit(kd, Xs[s][A] @ V - (Xs[s] @ V).mean(0), Gn[A] @ V, l)
                    others = [o for o in subs if o != s]
                    fp = fit(kd, np.concatenate([Xs[o][A] @ V - (Xs[o] @ V).mean(0) for o in others]),
                             np.tile(Gn[A] @ V, (len(others), 1)), l)
                    res[f'{key}_oracle'].append(top5(apply(fo, Xs[s], V), G, B))
                    res[f'{key}_pop'].append(top5(apply(fp, Xs[s], V), G, B))
            # D2 / D3 on the linear correction at lambda 1 (vectorised M - I, k x k)
            Ds = {s: (fit('linear', Xs[s][A] @ V - (Xs[s] @ V).mean(0), Gn[A] @ V, 1.0) - np.eye(k)).ravel() for s in subs}
            stat_all = {s: n(data[s][1]) @ V for s in subs}
            stat_six = {s: n(data[s][2][rng.choice(len(data[s][2]), 6, replace=False)].mean(0)) @ V for s in subs}
            for s in subs:
                others = [o for o in subs if o != s]
                M = np.stack([Ds[o] for o in others]); mu = M.mean(0)
                W = np.linalg.svd(M - mu, full_matrices=False)[2]
                def score(dvec):
                    return top5(apply(np.eye(k) + dvec.reshape(k, k), Xs[s], V), G, B)
                for j in d2:
                    c = (Ds[s] - mu) @ W[:j].T if j else np.zeros(0)
                    d2[j].append(score(mu + (c @ W[:j] if j else 0)))
                for src, st in (('all', stat_all), ('six', stat_six)):
                    F = np.stack([st[o] for o in others]); f0 = F.mean(0)
                    for j in (1, 2, 4):
                        C = (M - mu) @ W[:j].T                                  # others' coordinates (19, j)
                        Fc = F - f0
                        B_ = np.linalg.solve(Fc.T @ Fc + 1e-2 * np.trace(Fc.T @ Fc) * np.eye(k), Fc.T @ C)
                        d3[f'{src}{j}'].append(score(mu + ((st[s] - f0) @ B_) @ W[:j]))
            if sp == 0:
                ev = np.linalg.svd(np.stack(list(Ds.values())) - np.mean(list(Ds.values()), 0), compute_uv=False) ** 2
                print(f'RESULT {recipe} D2: linear-correction variance explained by top 1/2/4/8 axes '
                      f'{np.round(np.cumsum(ev)[[0, 1, 3, 7]] / ev.sum(), 2).tolist()}', flush=True)
        m = lambda v: 100 * np.mean(v)
        print(f'RESULT {recipe} D1 (k={k}, top-5 on B, 20 subj x {n_splits} splits): raw {m(res["raw"]):.2f} | '
              f'centred {m(res["none"]):.2f}', flush=True)
        for kd, l in kinds:
            key = f'{kd}{"" if l is None else l}'
            print(f'RESULT {recipe} D1 {key:10s} oracle {m(res[key + "_oracle"]):6.2f}  population {m(res[key + "_pop"]):6.2f}  '
                  f'subject-specific {m(res[key + "_oracle"]) - m(res[key + "_pop"]):+.2f}', flush=True)
        print(f'RESULT {recipe} D2 LOO linear(l=1) on top-j axes of the others: ' +
              ' | '.join(f'j={j} {m(v):.2f}' for j, v in d2.items()), flush=True)
        print(f'RESULT {recipe} D3 label-free coords (all trials / 6 trials): ' +
              ' | '.join(f'{key} {m(v):.2f}' for key, v in d3.items()), flush=True)


if __name__ == '__main__':
    X = np.random.default_rng(1).normal(size=(50, 4)); Rq = np.linalg.qr(np.random.default_rng(2).normal(size=(4, 4)))[0]
    assert np.allclose(fit('rotation', X, 3 * X @ Rq + 5), Rq, atol=1e-8)  # Procrustes recovers a rotation (any scale/shift)
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='+'); ap.add_argument('--k', type=int, default=16)
    ap.add_argument('--splits', type=int, default=10)
    a = ap.parse_args(); main(a.recipes, a.k, a.splits)
