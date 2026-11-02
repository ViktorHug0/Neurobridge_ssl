"""In-batch person-grouping purity of label-free affinities (TTA_LOG T4): for each trial of the cv4 grader stream,
the share of its top-K most similar batch-mates (64-trial shuffled batches, N = 5 people) that are the same person.
Offline statistics come from the fold's TRAINING people (15 Alljoined + 10 THINGS-EEG2) fingerprint caches.
    group_purity.py [--K 4]
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import DATA, FOLDS  # noqa: E402
from neurips_challenge.tta_cache import OUT  # noqa: E402

THINGS = os.environ.get('GP_THINGS', '1') == '1'  # include the 10 THINGS-EEG2 people in the offline statistics
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
l2 = lambda x: x / x.norm(dim=-1, keepdim=True)


def base_cols(g):
    """'base' = all 656 columns; 'base@a-b' = columns a..b-1 (0-32 log var, 32-528 correlations, 528-656 band powers)"""
    if '@' not in g:
        return slice(None)
    a, b = g.split('@')[1].split('-'); return slice(int(a), int(b))



def train_fps(subs, groups=('base',)):
    """training-image identity features of `subs` (base = the deployed eeg_fingerprint; others: group_features.py)"""
    from neurips_challenge.group_features import path
    one = lambda g, s: (np.load(f'{DATA}/.cache/fp_train_s{s:02d}_n3000_seed0_1_4-4_8-8_13-13_30.npy')[:, base_cols(g)] if g.startswith('base')
                        else np.load(path(g, 'train', s)))
    Fs = [np.hstack([one(g, s) for g in groups]) for s in subs]
    return torch.as_tensor(np.concatenate(Fs), device=DEV).double(), torch.as_tensor(np.repeat(subs, [len(f) for f in Fs]), device=DEV)


def test_fps(c, groups=('base',)):
    """the same features for a cached fold stream (rows of test.npy = the dataset idx)"""
    from neurips_challenge.group_features import path
    cols = []
    for g in groups:
        if g.startswith('base'):
            cols.append(c['F'][:, base_cols(g)])
        else:
            col = None
            for s in np.unique(c['subj']):
                r = c['subj'] == s; v = np.load(path(g, 'test', s))[c['idx'][r]]
                col = np.empty((len(r), v.shape[1]), np.float32) if col is None else col
                col[r] = v
            cols.append(col)
    return torch.as_tensor(np.hstack(cols), device=DEV)


def within_between(X, y):
    """pooled within-person covariance and between-person covariance of the person means"""
    ids = torch.unique(y); M = torch.stack([X[y == i].mean(0) for i in ids])
    C = X - M[torch.searchsorted(ids, y)]
    return C.T @ C / len(C), (M - M.mean(0)).T @ (M - M.mean(0)) / len(M)


def lw_within(X, y):
    """Ledoit-Wolf (2004) optimal intensity for the pooled within-person covariance, toward (tr / p) I (TTA_LOG T39)"""
    ids = torch.unique(y); M = torch.stack([X[y == i].mean(0) for i in ids]); C = X - M[torch.searchsorted(ids, y)]
    n, p = C.shape; S = C.T @ C / n; mu = torch.trace(S) / p
    d2 = ((S - mu * torch.eye(p, dtype=S.dtype, device=S.device)) ** 2).sum()
    b2 = (((C ** 2).sum(1) ** 2).sum() / n - (S ** 2).sum()) / n
    return (torch.minimum(b2, d2) / d2).item()


def inv_sqrt(S, shrink):
    S = (1 - shrink) * S + shrink * torch.trace(S) / len(S) * torch.eye(len(S), dtype=S.dtype, device=S.device)
    w, V = torch.linalg.eigh(S)
    return (V * w.clamp_min(1e-12).rsqrt()) @ V.T


def fit(kind, X, y, **kw):
    """returns a function (B, 656) raw fingerprints of one batch -> (B, B) affinity"""
    m0, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - m0) / sd
    P = None
    if kw.get('r'):  # TTA_LOG T38: unsupervised PCA to r dims first (pooled training trials; no labels)
        P = torch.linalg.eigh(Xs.T @ Xs / len(Xs))[1][:, -int(kw['r']):]; Xs = Xs @ P
    Sw, Sb = within_between(Xs, y)
    if kind == 'raw':
        T = torch.eye(Xs.shape[1], dtype=X.dtype, device=DEV)
    elif kind == 'wccn':      # within-person whitening, all dimensions kept; shrink < 0: Ledoit-Wolf intensity
        sh = kw['shrink'] if kw['shrink'] >= 0 else lw_within(Xs, y)
        if kw['shrink'] < 0:
            print(f'  wccn LW intensity {sh:.3f}', flush=True)
        T = inv_sqrt(Sw, sh)
    elif kind == 'lda':       # generalised eigenvectors of (Sb, Sw), top d
        Wh = inv_sqrt(Sw, kw['shrink'])
        w, V = torch.linalg.eigh(Wh @ Sb @ Wh)
        T = Wh @ V[:, -kw['d']:]
    elif kind == 'plda':      # two-covariance PLDA log-likelihood ratio; between-person covariance shrunk toward isotropic
        eye = torch.eye(len(m0), dtype=X.dtype, device=DEV)
        Sw_ = (1 - kw['shrink']) * Sw + kw['shrink'] * torch.trace(Sw) / len(Sw) * eye
        Sb_ = (1 - kw['gb']) * Sb + kw['gb'] * torch.trace(Sb) / len(Sb) * eye
        St = Sb_ + Sw_; Sti = torch.linalg.inv(St); A = torch.linalg.inv(St - Sb_ @ Sti @ Sb_)
        Qm, Pm = Sti - A, Sti @ Sb_ @ A

        def aff(F):
            Z = ((F.double() - m0) / sd); Z = Z - Z.mean(0)
            q = ((Z @ Qm) * Z).sum(1)
            return 2 * Z @ Pm @ Z.T + q[None, :]
        return aff

    def proj(F):
        Z = ((F.double() - m0) / sd) @ T
        Z = Z - Z.mean(0)                                   # within-batch centring (the batch's common component)
        return Z / (Z.std(0) + 1e-6) if kw.get('zs') else Z

    def aff(F):
        Z = proj(F)
        return l2(Z) @ l2(Z).T if kw.get('metric', 'cos') == 'cos' else -torch.cdist(Z, Z) ** 2
    aff.proj = proj
    if P is not None:
        T = P @ T
    aff.m0, aff.sd, aff.T = m0, sd, T  # (tta_arm.py reuses the projection)
    return aff


CFGS = {
    'raw': ('raw', {}),
    'lda24 (deployed-like)': ('lda', dict(d=24, shrink=1e-3, zs=True)),
    'lda24 nozs': ('lda', dict(d=24, shrink=1e-3)),
    'lda12': ('lda', dict(d=12, shrink=1e-3, zs=True)),
    'wccn s.01': ('wccn', dict(shrink=.01)),
    'wccn s.1': ('wccn', dict(shrink=.1)),
    'wccn s.3': ('wccn', dict(shrink=.3)),
    'wccn s.5': ('wccn', dict(shrink=.5)),
    'wccn s.7': ('wccn', dict(shrink=.7)),
    'wccn s.9': ('wccn', dict(shrink=.9)),
    'plda s.3 gb.1': ('plda', dict(shrink=.3, gb=.1)),
    'plda s.3 gb.5': ('plda', dict(shrink=.3, gb=.5)),
    'plda s.3 gb.9': ('plda', dict(shrink=.3, gb=.9)),
    'wccn s.1 euclid': ('wccn', dict(shrink=.1, metric='euc')),
    'wccn s.1 zs': ('wccn', dict(shrink=.1, zs=True)),
}


def snn(S, K2):
    """shared-nearest-neighbour affinity: number of common top-K2 neighbours (+ tiny original score as tie-break)"""
    S = S.clone(); S.fill_diagonal_(-1e9)
    A = torch.zeros_like(S).scatter_(1, S.topk(min(K2, len(S) - 1), 1).indices, 1.)
    return A @ A.T + 1e-3 * (S - S.min()) / (S.max() - S.min() + 1e-12)


def purity(aff, F, subj, K, post=None):
    out = []
    for b in range(0, len(F), 64):
        S = aff(F[b:b + 64])
        if post:
            S = snn(S, post)
        S.fill_diagonal_(-1e9); s = subj[b:b + 64]
        nb = S.topk(min(K, len(S) - 1), 1).indices
        out.append((s[nb] == s[:, None]).float().mean(1))
    return torch.cat(out).mean().item()


def main(K, names, post=None, groups=('base',)):
    res = {n: [] for n in names}
    for k, subs in FOLDS.items():
        train = [s for s in range(1, 21) if s not in subs] + (list(range(21, 31)) if THINGS else [])
        X, y = train_fps(train, groups)
        c = np.load(f'{OUT}/fold{k}.npz')
        F, subj = test_fps(c, groups), torch.as_tensor(c['subj'], device=DEV)
        for n in names:
            kind, kw = CFGS[n]
            res[n].append(purity(fit(kind, X, y, **kw), F, subj, K, post))
    for n in names:
        print(f'PURITY@{K} {"+".join(groups)} snn{post} {n:24s} {np.mean(res[n]):.3f}  folds ' + ' '.join(f'{v:.3f}' for v in res[n]), flush=True)


def bb_main(recipe, layer, lams, K=4, shrink=.3, groups=('base',)):
    """T20: purity of model-internal features (tta_backbone.py caches), alone (lam = inf) and late-fused with the
    fingerprint affinity: S = S_fp + lam S_bb"""
    res = {l: [] for l in lams}
    for k, subs in FOLDS.items():
        c = np.load(f'{OUT}/fold{k}.npz'); subj = torch.as_tensor(c['subj'], device=DEV)
        X, y = train_fps([s for s in range(1, 21) if s not in subs], groups); aff_fp = fit('wccn', X, y, shrink=shrink)
        F = test_fps(c, groups)
        t = np.load(f'{OUT}/bb_{recipe}_{layer}_f{k}_train.npz')
        aff_bb = fit('wccn', torch.as_tensor(t['X'], device=DEV).double(), torch.as_tensor(t['y'], device=DEV), shrink=shrink)
        B = torch.as_tensor(np.load(f'{OUT}/bb_{recipe}_{layer}_f{k}.npy'), device=DEV).float()
        for l in lams:
            def aff(i, l=l):
                return aff_bb(B[i]) if l == float('inf') else aff_fp(F[i]) + l * aff_bb(B[i])
            out = []
            for b in range(0, len(F), 64):
                S = aff(slice(b, b + 64)); S.fill_diagonal_(-1e9); s = subj[b:b + 64]
                out.append((s[S.topk(min(K, len(S) - 1), 1).indices] == s[:, None]).float().mean(1))
            res[l].append(torch.cat(out).mean().item())
    for l in lams:
        print(f'PURITY@{K} bb {recipe}:{layer} {"+".join(groups)} lam {l}: {np.mean(res[l]):.3f}  folds ' + ' '.join(f'{v:.3f}' for v in res[l]), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--K', type=int, default=4); ap.add_argument('names', nargs='*'); ap.add_argument('--snn', type=int, default=None); ap.add_argument('--groups', nargs='+', default=['base']); ap.add_argument('--bb', default=None); ap.add_argument('--lams', nargs='+', type=float, default=[float('inf'), .5, 1, 2])
    a = ap.parse_args()
    if a.bb:
        bb_main(*a.bb.split(':'), a.lams, a.K, groups=tuple(a.groups))
    else:
        main(a.K, a.names or list(CFGS), a.snn, tuple(a.groups))
