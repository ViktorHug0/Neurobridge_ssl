"""Batch-transform search for the sealed cohort size (11 unseen participants, shuffled mixed 64-batches).
Composite testbed: every one of the 20 cv4 subjects is predicted by the fold model that held it out (trial caches,
DINOv2 output space, so all subjects share one space); each draw picks 11 subjects, shuffles all their trials
together and cuts 64-batches. Per batch: each member is divided by its batch mean norm, transformed, members summed.
Scored with the sealed rule (sum repeats per subject x image, cosine top-5 over the 200 test images).
Caveat: 4 different fold models in one batch add a model offset on top of the subject offsets.
    n11.py [--n 11] [--draws 4] [--cfgs grid|...] <recipe> [<recipe> ...]
"""
import argparse, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load  # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))
IMG = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy').reshape(-1, 1536)


def image_stats():
    """centred covariance of the 16540 training images' DINOv2 features (no test/sealed image involved)"""
    X = torch.tensor(IMG.reshape(1654, 10, 1536).mean(1))
    X = X / X.norm(dim=1, keepdim=True); X = X - X.mean(0)
    w, V = torch.linalg.eigh(X.T @ X / len(X))
    return w.flip(0).clamp_min(0), V.flip(1)


W_IMG, V_IMG = image_stats()


def shrink(C, k, f, M=None, Minv=None):
    """C (nb, n, D) centred batch; shrink its top-k principal directions by factors f(lambda) (nb, k) -> (nb, n, D).
    With a metric M (D, D): directions are found in Y = C M and the shrink is mapped back through Minv."""
    Y = C if M is None else C @ M
    Gm = Y @ Y.transpose(1, 2)
    lam, U = torch.linalg.eigh(Gm)                                    # ascending
    lam, U = lam.flip(-1)[..., :k].clamp_min(1e-12), U.flip(-1)[..., :k]
    V = (Y.transpose(1, 2) @ U) / lam.sqrt()[:, None, :]               # (nb, D, k) unit right singular vectors
    fac = f(lam, torch.linalg.eigvalsh(Gm).flip(-1))                  # (nb, k) kept fraction
    proj = (Y @ V) * (1 - fac)[:, None, :]
    if M is None:
        return C - proj @ V.transpose(1, 2)
    return C - proj @ (V.transpose(1, 2) @ Minv)


def T(a=.5, k=2, b=.5, spec=None, M=None, Minv=None, P=None, rownorm=None):
    """generic transform: (1-a) mean kept, top-k shrink (constant b, or spec(lam, allLam) kept fractions)."""
    def f(X, m=None, ctx=None):
        if P is not None:
            X = X @ P @ P.T
        mu = X.mean(1, keepdim=True); C = X - mu
        if k:
            C = shrink(C, k, spec or (lambda lam, al: torch.full_like(lam, 1 - b)), M, Minv)
        Y = C + (1 - a) * mu
        if rownorm is not None:
            Y = Y * Y.norm(dim=-1, keepdim=True).clamp_min(1e-8) ** (rownorm - 1)
        return Y
    return f


def static_fit(data, M, others, r=8):
    """offline, label-free-at-test statistics from subjects outside the draw (in deployment: the cv4 fold models'
    held-out subjects): per member, the population mean output and the top-r subspace of the per-subject mean
    offsets."""
    ctx = {'mu': [], 'U': []}
    for m in range(M):
        means = torch.tensor(np.stack([data[s]['emb'][m].mean(0) / np.linalg.norm(data[s]['emb'][m], axis=1).mean() for s in others]))
        mu = means.mean(0); ctx['mu'].append(mu)
        ctx['U'].append(torch.linalg.svd(means - mu, full_matrices=False)[2][:r].T)
    return ctx


def S(a=.5, k=2, b=.5, c=.5, r=8, wmu=0., spec=None):
    """E25 transform + static subject-offset subspace: also shrink the centred batch's component in the top-r
    offset subspace by c; wmu blends the static population mean into the batch mean."""
    base = T(a=a, k=k, b=b, spec=spec)
    def f(X, m, ctx):
        if wmu:
            mu = X.mean(1, keepdim=True); X = X - wmu * (mu - ctx['mu'][m] * X.norm(dim=-1).mean(1)[:, None, None])
        Y = base(X)
        if c:
            U = ctx['U'][m][:, :r]; mu = Y.mean(1, keepdim=True)
            Y = Y - c * ((Y - mu) @ U) @ U.T
        return Y
    return f


def FP(p=4, c=.5, k=3, control=False, lda=False, b=.5, key='lda', fkey='F'):
    """label-free regression routing: centre, regress the batch outputs on the top-p within-batch PCs of the trials'
    raw-EEG fingerprints (ctx['F'], z-scored with the batch's own statistics) and remove c x the fitted part, then
    the usual top-k shrink."""
    def f(X, m, ctx):
        F = ctx[fkey] if not control else torch.randn(ctx['F'].shape[:2] + (p,), generator=torch.Generator().manual_seed(0))
        if lda:  # subject-discriminative projection learned offline on OTHER subjects' raw EEG (model-independent)
            F = (F - ctx[key]['m0']) / ctx[key]['sd'] @ ctx[key]['W']
        F = (F - F.mean(1, keepdim=True)) / (F.std(1, keepdim=True) + 1e-6)
        Z = torch.linalg.svd(F, full_matrices=False)[0][..., :p]          # (nb, n, p) orthonormal trial-axis scores
        mu = X.mean(1, keepdim=True); C = X - mu
        C = C - c * Z @ (Z.transpose(1, 2) @ C)
        C = shrink(C, k, lambda lam, al: torch.full_like(lam, 1 - b))
        return C + .5 * mu
    return f


def LW(delta=None, gamma=1.0, a=.5, route=None):
    """batch-level whitening of the centred outputs with Ledoit-Wolf shrinkage Sigma = (1-d) S + d nu I.
    The centred trials lie in the span of S, so whitening = rescaling eigen-direction i by ((1-d) lam_i + d nu)^-1/2;
    normalised to 1 outside the span: kept fraction r_i = (d nu / ((1-d) lam_i + d nu))^(gamma/2) (gamma=1: full
    whitening, .5: half). delta None = the analytic Ledoit-Wolf (2004) intensity, per batch.
    route: optional FP router (dict p, key) applied before the whitening."""
    def f(X, m, ctx):
        mu = X.mean(1, keepdim=True); C = X - mu; B, D = C.shape[1], C.shape[2]
        if route:
            F = ctx['F']; F = (F - ctx[route['key']]['m0']) / ctx[route['key']]['sd'] @ ctx[route['key']]['W']
            F = (F - F.mean(1, keepdim=True)) / (F.std(1, keepdim=True) + 1e-6)
            Z = torch.linalg.svd(F, full_matrices=False)[0][..., :route['p']]
            C = C - Z @ (Z.transpose(1, 2) @ C)
        K = C @ C.transpose(1, 2)                                     # Gram (nb, B, B); S = C^T C / B
        lam, U = torch.linalg.eigh(K / B)                              # nonzero eigenvalues of S
        nu = (torch.diagonal(K, dim1=1, dim2=2).sum(-1) / B / D)[:, None]   # tr(S) / D
        if delta is None:  # LW: d = min(1, sum_i ||x_i x_i^T - S||_F^2 / B^2 / ||S - nu I||_F^2)
            S2 = (K ** 2).sum((1, 2)) / B ** 2                          # ||S||_F^2
            n2 = torch.diagonal(K, dim1=1, dim2=2)                      # ||x_i||^2
            xSx = (K ** 2).sum(-1) / B                                  # x_i^T S x_i
            num = (n2 ** 2 - 2 * xSx + S2[:, None]).sum(-1) / B ** 2
            den = S2 - nu[:, 0] ** 2 * D
            d = (num / den).clamp(0, 1)[:, None]
        else:
            d = torch.full_like(nu, delta)
        r = (d * nu / ((1 - d) * lam.clamp_min(0) + d * nu)).pow(gamma / 2)   # (nb, B)
        C = U @ (r[..., None] * (U.transpose(1, 2) @ C))
        if ctx is not None:
            ctx.setdefault('lw_delta', []).append(d.mean().item())
        return C + (1 - a) * mu
    return f


class Post:
    """ensemble-level transform: gets the list of member batches."""
    def __init__(self, fn): self.fn = fn
    def __call__(self, Xb, ctx): return self.fn(Xb, ctx)


def weighted(w, base=None):
    base = base or T()
    return Post(lambda Xb, ctx: sum(wi * base(x) for wi, x in zip(w, Xb)))


def smooth(lam, m_nb, tau, base=None):
    """in-batch query expansion: after the member transforms, add lam x the softmax(cos/tau)-weighted mean of
    each trial's m_nb nearest batch-mates (same-image trials of other subjects are its likely neighbours)."""
    base = base or T()
    def fn(Xb, ctx):
        Y = sum(base(x) for x in Xb)
        u = Y / Y.norm(dim=-1, keepdim=True)
        Sm = u @ u.transpose(1, 2); Sm.diagonal(dim1=1, dim2=2).fill_(-2)
        v, i = Sm.topk(m_nb, dim=-1)
        w = torch.softmax(v / tau, -1)
        nb = torch.gather(Y[:, None].expand(-1, Y.shape[1], -1, -1), 2, i[..., None].expand(-1, -1, -1, Y.shape[-1]))
        return Y + lam * (w[..., None] * nb).sum(2)
    return Post(fn)


def robust(kind, base_k=2):
    """centre on a robust batch location instead of the mean (trimmed by norm / coordinate-wise median)"""
    def f(X, m=None, ctx=None):
        if kind == 'median':
            mu = X.median(1, keepdim=True).values
        else:  # drop the 10% largest-norm trials from the mean
            nrm = X.norm(dim=-1); thr = nrm.quantile(.9, dim=1, keepdim=True)
            w = (nrm <= thr).float()[..., None]; mu = (w * X).sum(1, keepdim=True) / w.sum(1, keepdim=True)
        C = shrink(X - mu, base_k, lambda lam, al: torch.full_like(lam, .5))
        return C + .5 * mu
    return f


def cfgs(name):
    c = {'plain': None, 'E25 a.5 k2 b.5': T()}
    if name in ('grid', 'all'):
        for a in (.25, .5, .75, 1.):
            c[f'a{a} k2 b.5'] = T(a=a)
        for k in (1, 2, 3, 4, 6, 8, 12):
            for b in (.25, .5, .75, 1.):
                c[f'a.5 k{k} b{b}'] = T(k=k, b=b)
    if name in ('spec', 'all'):
        for g in (.25, .5, .75, 1.):  # kept fraction (lam / mean lam)^-g/2, capped at 1, for the top 16
            c[f'spec pow g{g}'] = T(k=16, spec=lambda lam, al, g=g: (lam / al.mean(-1, keepdim=True)).pow(-g / 2).clamp(max=1))
        for beta in (.5, 1., 2.):  # kept fraction 1 / (1 + beta lam/median)
            c[f'spec ridge b{beta}'] = T(k=16, spec=lambda lam, al, beta=beta: 1 / (1 + beta * lam / al[..., :63].median(-1, keepdim=True).values))
    if name in ('metric', 'all'):
        for r in (64, 256):
            P = V_IMG[:, :r]
            for k in (2, 4):
                c[f'imgproj r{r} k{k} b.5'] = T(k=k, P=P)
        for g in (.5, 1.):  # directions found in the image-covariance metric (what moves the ranking)
            M = V_IMG @ torch.diag((W_IMG + W_IMG[200]).pow(g / 2)) @ V_IMG.T
            Minv = V_IMG @ torch.diag((W_IMG + W_IMG[200]).pow(-g / 2)) @ V_IMG.T
            for k in (2, 4):
                c[f'imgmetric g{g} k{k} b.5'] = T(k=k, M=M, Minv=Minv)
    if name in ('rownorm', 'all'):
        for p in (0., .5, 1.5):
            c[f'E25 + rownorm p{p}'] = T(rownorm=p)
    if name == 'r2':
        for g in (.5, .75, 1.):
            for kk in (8, 32):
                c[f'spec pow g{g} top{kk}'] = T(k=kk, spec=lambda lam, al, g=g: (lam / al.mean(-1, keepdim=True)).pow(-g / 2).clamp(max=1))
        for r in (4, 8):
            for cc in (.5, 1.):
                c[f'static r{r} c{cc}'] = S(r=r, c=cc)
        for wmu in (.5, 1.):
            c[f'static mean wmu{wmu}'] = S(c=0, wmu=wmu)
        for w in ((1.5, 1, 1), (1, 1.5, 1), (1, 1, 1.5), (1, 1, .5)):
            c[f'weights {w}'] = weighted(w)
        for lam in (.25, .5):
            for m_nb in (1, 3):
                c[f'smooth lam{lam} m{m_nb}'] = smooth(lam, m_nb, .05)
        c['robust median'] = robust('median'); c['robust trim'] = robust('trim')
    if name == 'pair':  # same transforms on single members trained with / without THINGS
        c['a.5 k4 b.5'] = T(k=4); c['static k2 r4 c.5'] = S(r=4, c=.5)
    if name == 'best':
        c['a.5 k3 b.5'] = T(k=3); c['a.5 k4 b.5'] = T(k=4)
        c['static k3 r2 c.5'] = S(k=3, r=2, c=.5); c['static k2 r4 c.25'] = S(r=4, c=.25)
        c['static k3 r4 c.25'] = S(k=3, r=4, c=.25); c['static a.6 k4 r4 c.5'] = S(a=.6, k=4, r=4, c=.5)
    if name == 'fp':
        c['a.5 k3 b.5'] = T(k=3)
        for p in (1, 2, 4, 8):
            for cc in (.5, 1.):
                c[f'fpreg p{p} c{cc} k3'] = FP(p, cc)
    if name == 'fp2':
        c['a.5 k3 b.5'] = T(k=3)
        for p in (4, 8, 16):
            c[f'fpreg p{p} c1.0 k3'] = FP(p, 1.)
            c[f'fpreg CONTROL random p{p} c1.0 k3'] = FP(p, 1., control=True)
    if name == 'final':  # candidates for the s11 exact testbed
        c['a.5 k3 b.5'] = T(k=3); c['a.5 k4 b.5'] = T(k=4)
        c['static k3 r2 c.5'] = S(k=3, r=2, c=.5); c['static k2 r4 c.25'] = S(r=4, c=.25)
        c['fpreg p4 c1.0 k3'] = FP(4, 1.); c['fpreg p8 c1.0 k3'] = FP(8, 1.)
    if name == 'fp3':
        c['a.5 k3 b.5'] = T(k=3); c['fpreg p8 c1.0 k3'] = FP(8, 1.)
        for p in (6, 12):
            c[f'fpreg p{p} c1.0 k3'] = FP(p, 1.)
        for k in (2, 4):
            c[f'fpreg p8 c1.0 k{k}'] = FP(8, 1., k=k)
        c['fpreg p8 c.75 k3'] = FP(8, .75); c['fpreg p8 c1.0 k3 b.4'] = FP(8, 1., b=.4); c['fpreg p8 c1.0 k3 b.6'] = FP(8, 1., b=.6)
        for p in (4, 8):
            c[f'fplda p{p} c1.0 k3'] = FP(p, 1., lda=True)
    if name == 'fpbest':
        c['a.5 k3 b.5'] = T(k=3); c['fpreg p8 c.75 k3'] = FP(8, .75); c['fplda p4 c1.0 k3'] = FP(4, 1., lda=True)
    if name == 's11c':  # final candidates
        c['a.5 k3 b.5'] = T(k=3); c['a.5 k4 b.5'] = T(k=4)
        c['fpreg p8 c.75 k3'] = FP(8, .75); c['fpreg p8 c1.0 k3'] = FP(8, 1.)
        c['fplda p4 c1.0 k3'] = FP(4, 1., lda=True); c['fplda p8 c1.0 k3'] = FP(8, 1., lda=True)
    if name == 's11d':  # deployable LDA fit (training-image EEG), with / without THINGS subjects
        c['a.5 k3 b.5'] = T(k=3)
        for p in (4, 8, 12):
            c[f'fplda p{p} c1.0 k3'] = FP(p, 1., lda=True)
            c[f'fpldaT p{p} c1.0 k3'] = FP(p, 1., lda=True, key='ldaT')
    if name == 's11g':
        c['a.5 k3 b.5'] = T(k=3)
        for p in (4, 8, 12):
            c[f'fpldaT p{p} c1.0 k3'] = FP(p, 1., lda=True, key='ldaT')
            c[f'fpldaG p{p} c1.0 k3'] = FP(p, 1., lda=True, key='ldaG', fkey='FG')
    if name == 'fpT':
        c['a.5 k3 b.5'] = T(k=3)
        for p in (4, 8):
            c[f'fplda p{p} c1.0 k3'] = FP(p, 1., lda=True)
            c[f'fpldaT p{p} c1.0 k3'] = FP(p, 1., lda=True, key='ldaT')
    if name == 'fpTc':
        c['a.5 k3 b.5'] = T(k=3)
        for p, cc in ((8, 1.), (8, .75), (8, .5), (6, 1.), (10, .75)):
            c[f'fpldaT p{p} c{cc} k3'] = FP(p, cc, lda=True, key='ldaT')
    if name == 'deploy':  # plain / old / k3 / the deployed router
        c['a.5 k3 b.5'] = T(k=3); c['router fpldaT p6 c1 k3'] = FP(6, 1., lda=True, key='ldaT')
    if name == 'lw':
        c['a.5 k3 b.5'] = T(k=3); c['router fpldaT p6 c1 k3'] = FP(6, 1., lda=True, key='ldaT')
        c['LW analytic'] = LW(); c['LW analytic half'] = LW(gamma=.5)
        for dl in (.1, .5, .9):
            c[f'LW d{dl}'] = LW(dl); c[f'LW d{dl} half'] = LW(dl, gamma=.5)
        c['router + LW analytic'] = LW(route={'p': 6, 'key': 'ldaT'}); c['router + LW analytic half'] = LW(gamma=.5, route={'p': 6, 'key': 'ldaT'})
    if name == 'lwsweep':  # shrinkage intensity d x whitening strength gamma, + the router in front
        c['a.5 k3 b.5'] = T(k=3); c['router fpldaT p6 c1 k3'] = FP(6, 1., lda=True, key='ldaT')
        for g in (.25, .5, 1.):
            c[f'LW analytic g{g}'] = LW(gamma=g)
            for dl in (.01, .02, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99):
                c[f'LW d{dl} g{g}'] = LW(dl, gamma=g)
        for g in (.5, 1.):
            for dl in (None, .1, .3, .5, .7, .9):
                c[f'router + LW d{dl} g{g}'] = LW(dl, gamma=g, route={'p': 6, 'key': 'ldaT'})
    if name == 'r3':  # combinations of the round-2 winners
        g75 = lambda lam, al: (lam / al.mean(-1, keepdim=True)).pow(-.75 / 2).clamp(max=1)
        for k in (2, 3, 4):
            for r in (2, 4, 6):
                for cc in (.25, .5):
                    c[f'static k{k} r{r} c{cc}'] = S(k=k, r=r, c=cc)
        c['static spec.75 r4 c.5'] = S(k=8, spec=g75, r=4, c=.5)
        c['static spec.75 r4 c.25'] = S(k=8, spec=g75, r=4, c=.25)
        for a_ in (.4, .6):
            c[f'a{a_} k4 b.5'] = T(a=a_, k=4)
            c[f'static a{a_} k4 r4 c.5'] = S(a=a_, k=4, r=4, c=.5)
    return c


def run(data, M, subjects_draws, configs, rng, n_batch=64):
    """-> {cfg: {(subject, draw): top5}}"""
    res = {}
    G = torch.tensor(next(iter(data.values()))['G'], dtype=torch.float32)
    for d, subs in enumerate(subjects_draws):
        n = [len(data[s]['label']) for s in subs]
        X = [torch.tensor(np.concatenate([data[s]['emb'][m] for s in subs])) for m in range(M)]
        key = torch.tensor(np.concatenate([i * 200 + data[s]['label'] for i, s in enumerate(subs)]))
        o = torch.tensor(rng.permutation(len(key)))
        nb = len(o) // n_batch; o = o[:nb * n_batch]                  # ponytail: drops < 64 trials per draw
        Xb = [x[o].view(nb, n_batch, -1) for x in X]
        Xb = [x / x.norm(dim=-1).mean(1)[:, None, None] for x in Xb]
        ctx = static_fit(data, M, [s for s in data if s not in subs])
        if any(n.startswith(('fpreg', 'fplda', 'router')) for n in configs):  # (also feeds the random control's shape)
            from neurips_challenge.neighbour_lda import fingerprint
            ctx['F'] = torch.tensor(np.concatenate([fingerprint(s) for s in subs]), dtype=torch.float32)[o].view(nb, n_batch, -1)
            if any('lda' in n or 'router' in n for n in configs):  # the deployable fit: other subjects' TRAINING-image EEG (score_track1)
                from neurips_challenge.score_track1 import fit_fingerprint_lda
                L = fit_fingerprint_lda('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench', [s for s in data if s not in subs])
                ctx['lda'] = L
                if any('fpldaG' in n for n in configs):  # + a 30-60 Hz (EMG) band, LDA with THINGS; its own fingerprints
                    from neurips_challenge.score_track1 import FP_BANDS, eeg_fingerprint
                    bands = FP_BANDS + ((30, 60),)
                    ctx['ldaG'] = fit_fingerprint_lda('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench', [s for s in data if s not in subs] + list(range(21, 31)), bands=bands)
                    raw = [np.load(f'/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120) for s in subs]
                    ctx['FG'] = torch.cat([eeg_fingerprint(torch.tensor(np.asarray(r[i:i + 2000])), bands) for r in raw for i in range(0, len(r), 2000)])[o].view(nb, n_batch, -1)
                if any('fpldaT' in n or 'router' in n for n in configs):  # + the 10 THINGS-EEG2 subjects (same 32 electrodes)
                    ctx['ldaT'] = fit_fingerprint_lda('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench', [s for s in data if s not in subs] + list(range(21, 31)))
        for name, f in configs.items():
            t0 = time.time()
            Y = (sum(x if f is None else f(x, m, ctx) for m, x in enumerate(Xb)) if not isinstance(f, Post) else f(Xb, ctx)).reshape(nb * n_batch, -1)
            Q = torch.zeros(len(subs) * 200, Y.shape[1]).index_add_(0, key[o], Y).view(len(subs), 200, -1)
            S = Q @ G.T
            diag = S.diagonal(dim1=1, dim2=2)
            t5 = 100 * ((S > diag[..., None]).sum(-1) < 5).float().mean(1)
            for i, s in enumerate(subs):
                res.setdefault(name, {})[(s, d)] = t5[i].item()
            print(f'draw {d} {name:28s} {t5.mean():6.2f}  ({time.time() - t0:.0f}s)', flush=True)
    return res


def report(res, title):
    base = res['plain']; keys = sorted(base)
    half = {1: {s for f in (1, 2) for s in FOLDS[f]}, 2: {s for f in (3, 4) for s in FOLDS[f]}}
    print(title)
    for name, v in res.items():
        row = []
        for h in (1, 2):
            ks = [k for k in keys if k[0] in half[h]]
            x = np.array([v[k] for k in ks]); b = np.array([base[k] for k in ks])
            row.append(f'{x.mean():6.2f} {x.mean() - b.mean():+5.2f} ({(x > b).sum():2d}/{(x < b).sum():2d})')
        print(f'  {name:28s} ' + ' | '.join(row), flush=True)


def load_s11(recipes):
    """exact testbed (s11.sh): {split: {s: dict(label, G, emb[m])}}, one model per split, scaled by its mean norm"""
    import glob
    out = {}
    for k in (1, 2):
        for m, r in enumerate(recipes):
            z = np.load(glob.glob(f'results/alljoined/track1/s11/{r}/fold{k}/*/*/trial_dots.npz')[0])
            subs = sorted(int(x[1:].split('_')[0]) for x in z.files if x.endswith('_label'))
            mn = np.mean(np.concatenate([z[f's{s}_norm'] for s in subs]))
            for s in subs:
                e = out.setdefault(k, {}).setdefault(s, dict(emb=[None] * len(recipes)))
                e['label'], e['G'] = z[f's{s}_label'], z['gallery']; e['emb'][m] = z[f's{s}_emb'].astype(np.float32) / mn
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='+'); ap.add_argument('--n', type=int, default=11)
    ap.add_argument('--draws', type=int, default=4); ap.add_argument('--cfgs', default='grid'); ap.add_argument('--s11', action='store_true'); ap.add_argument('--batch', type=int, default=64); ap.add_argument('--cv', action='store_true')
    a = ap.parse_args()
    if a.cv:  # real cv4 batches: fold k's 5 subjects through fold k's model; static fits from the other 15
        # subjects as predicted by the OTHER fold models (the deployment situation: fit on cv models, apply to another)
        data = load(a.recipes); res = {}
        for k, subs in FOLDS.items():
            if not all(s in data for s in subs):  # recipes screened on folds 1-2 only
                continue
            for name, v in run(data, len(a.recipes), [subs], cfgs(a.cfgs), np.random.default_rng(k), a.batch).items():
                res.setdefault(name, {}).update(v)
        for name, v in res.items():  # per fold and per subject, vs the first transform after plain
            ref = res[list(res)[1]]
            fk = {k: subs for k, subs in FOLDS.items() if (subs[0], 0) in v}
            print(f'  {name:24s} ' + '  '.join(f'f{k} {np.mean([v[(s, 0)] - ref[(s, 0)] for s in subs]):+5.2f}' for k, subs in fk.items()))
            print('      per subject vs ref: ' + ' '.join(f'{s}:{v[(s, 0)] - ref[(s, 0)]:+.0f}' for subs in fk.values() for s in subs))
        report(res, f'{"+".join(a.recipes)}: cv4, N=5 per shuffled {a.batch}-batch, static fits from other folds\' models')
        sys.exit()
    if a.s11:  # each split: its 11 test subjects, one model; static fits from the OTHER split's model and subjects
        sp = load_s11(a.recipes); res = {}
        for k in (1, 2):
            other = {s: e for s, e in sp[3 - k].items() if s not in sp[k]}
            data = {**other, **sp[k]}
            r = run(data, len(a.recipes), [sorted(sp[k])], cfgs(a.cfgs), np.random.default_rng(11 + k), a.batch)
            for name, v in r.items():
                res.setdefault(name, {}).update({(s, k): x for (s, _), x in v.items()})
        base = res['plain']
        print(f'{"+".join(a.recipes)}: s11 exact testbed, 2 splits x 11 unseen subjects, one shuffled stream per split')
        for name, v in res.items():
            x = np.array([v[q] for q in base]); b = np.array([base[q] for q in base])
            print(f'  {name:28s} {x.mean():6.2f} {x.mean() - b.mean():+5.2f} ({(x > b).sum():2d}/{(x < b).sum():2d})', flush=True)
        sys.exit()
    data = load(a.recipes); rng = np.random.default_rng(11)
    draws = [sorted(rng.choice(sorted(data), a.n, replace=False).tolist()) for _ in range(a.draws)]
    res = run(data, len(a.recipes), draws, cfgs(a.cfgs), rng, a.batch)
    report(res, f'{"+".join(a.recipes)}: N={a.n} participants per shuffled {a.batch}-batch, {a.draws} draws '
                f'(subject halves: folds 1-2 | folds 3-4; mean, delta vs plain, better/worse over subject x draw)')
