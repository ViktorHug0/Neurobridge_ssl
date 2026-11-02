"""Person routing study (ROUTING_LOG E47+, user idea 2026-10-15): group each shuffled 64-batch's trials by participant
and centre each group on its own statistics. s11 exact testbed (n11.py --s11: one model per split, 11 unseen
participants mixed per batch = the sealed condition), members' trial caches, sealed rule (sum repeats, top-5 / 200).

Group transform with a row-stochastic in-batch affinity W (n x n, "who are my same-person batch-mates"):
  offset_i = (W X)_i - batch mean       C = X - batch mean - c * offset       Y = shrink_k3(C) + 0.5 * batch mean
(c = 1: each trial loses its group's full offset from the batch, the shared mean is halved as deployed).
W sources: oracle labels (headroom), oracle with purity q (q of the weight on true batch-mates, 1-q spread on the
others: the accuracy a router must reach), and label-free embeddings (subject_embed.py) as soft kNN.
    route_study.py [--cfgs oracle|emb] [--emb <embedder tag>] [recipes...]
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.n11 import shrink  # noqa: E402

torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 2)))
DATA = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
SPLITS = {1: [2, 3, 4, 5, 7, 11, 14, 15, 16, 17, 20], 2: [1, 2, 3, 6, 8, 9, 10, 12, 13, 18, 19]}  # = s11.sh


def load_split(recipes, k):
    """labels (n,), [member emb (n, D) float16, subjects concatenated in SPLITS order], gallery; each member scaled by
    its mean norm over the split (n11.load_s11). One member in memory at a time (the node step has ~4 GB)."""
    X, lab, G = [], None, None
    for r in recipes:
        z = np.load(glob.glob(f'results/alljoined/track1/s11/{r}/fold{k}/*/*/trial_dots.npz')[0])
        mn = np.float32(np.mean(np.concatenate([z[f's{s}_norm'] for s in SPLITS[k]])))
        X.append(np.concatenate([(z[f's{s}_emb'] / mn).astype(np.float16) for s in SPLITS[k]]))
        lab, G = [z[f's{s}_label'] for s in SPLITS[k]], z['gallery']
        del z
    return lab, X, torch.tensor(G, dtype=torch.float32)


def affinities(sid, cfg, E=None):
    """(nb, n) participant ids -> (nb, n, n) row-stochastic W (zero diagonal unless self-inclusive)."""
    same = (sid[:, :, None] == sid[:, None, :]).float()
    eye = torch.eye(sid.shape[1], device=sid.device)
    if cfg['src'] == 'oracle':
        S = same - eye if not cfg.get('self') else same
        O = 1 - same
        nS = S.sum(-1, keepdim=True)
        W = cfg['q'] * S / nS.clamp_min(1) + (1 - cfg['q']) * O / O.sum(-1, keepdim=True).clamp_min(1)
        W = torch.where(nS > 0, W, O / O.sum(-1, keepdim=True).clamp_min(1))  # no batch-mate: batch mean of the others
    else:  # embedding: soft kNN among the batch-mates (self excluded)
        D = torch.cdist(E, E) ** 2 + 1e9 * eye
        v, i = (-D).topk(cfg['K'], dim=-1)
        W = torch.zeros_like(D).scatter_(-1, i, torch.softmax(v / cfg['tau'], -1) if cfg.get('tau') else 1.0 / cfg['K'])
    if cfg.get('lam'):  # shrink small groups toward the batch mean: (n_g * group mean + lam * batch mean) / (n_g + lam)
        n_g = (same - eye).sum(-1, keepdim=True) if cfg['src'] == 'oracle' else cfg['K']
        W = (n_g * W + cfg['lam'] / sid.shape[1]) / (n_g + cfg['lam'])
    return W


def transform(X, W, c=1.0, k=3, b=.5):
    mu = X.mean(1, keepdim=True)
    C = X - mu - c * (W @ X - mu) if W is not None else X - mu
    return shrink(C, k, lambda lam, al: torch.full_like(lam, 1 - b)) + .5 * mu


def nbr_whiten(X, w, d, gamma=1.0, keep=.5):
    """user idea (E49): whiten each trial with the statistics of its neighbours. w (B, n, n): row i = uniform weights
    on trial i's neighbours (zero diagonal). Per trial: mu_i = neighbour mean, S_i = neighbour covariance (rank < k),
    Ledoit-Wolf shrunk (1-d) S_i + d nu_i I (nu_i = tr S_i / D); y_i = x_i - mu_i rescaled along S_i's eigendirections
    by r_t = (d nu / ((1-d) lam_t + d nu))^(gamma/2); output y_i + keep * mu_i. d = 1: neighbour centring (E20).
    Every output is a combination of the batch rows, so all of it is computed in the n x n Gram space."""
    B, n, D = X.shape
    I = torch.eye(n, device=X.device)
    kmax = min(int((w > 0).sum(-1).max()), 32)  # neighbour space: k x k eigenproblems (batched on GPU up to 32)
    v, idx = w.topk(kmax, dim=-1); vm = (v > 0).float()
    if kmax < int((w > 0).sum(-1).max()):  # ponytail: rows with > 32 neighbours (oracle fallback) keep their top 32
        w = torch.zeros_like(w).scatter_(-1, idx, v * vm); w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        v = w.gather(-1, idx)
    kk = vm.sum(-1).clamp_min(1)                                                # (B, n)
    onehot = torch.zeros(B, n, kmax, n, device=X.device).scatter_(-1, idx[..., None], 1.0)
    E = vm[..., None] * (onehot - w[:, :, None, :])                             # E[b,i,a] = coefs of x_idx(a) - mu_i
    EG = E @ (X @ X.transpose(1, 2))[:, None]                                   # (B, n, n, n)
    K = EG @ E.transpose(2, 3) / kk[..., None, None]                            # Gram of the centred neighbours / k
    H = I[None] - w                                                             # y_i = H_i X
    Cy = (EG @ H[..., None]).squeeze(-1)                                        # (B, n, k): (x_a - mu_i) . y_i
    lam, U = torch.linalg.eigh(K)
    nu = K.diagonal(dim1=2, dim2=3).sum(-1, keepdim=True) / D
    live = lam > 1e-6 * lam[..., -1:].clamp_min(1e-12)
    r = torch.where(live, (d * nu / ((1 - d) * lam.clamp_min(0) + d * nu)).pow(gamma / 2), torch.ones_like(lam))
    st = (1 - r) * (U.transpose(2, 3) @ Cy[..., None]).squeeze(-1) / (kk[..., None] * lam.clamp_min(1e-12)) * live
    z = (U @ st[..., None]).squeeze(-1)                                         # (B, n, n) over neighbours j
    coef = H - (E.transpose(2, 3) @ z[..., None]).squeeze(-1)
    return (coef + keep * w) @ X


def router(X, F, lda, p=6):  # = score_track1.batch_transform_fp (deployed), batched
    Fp = (F - lda['m0']) / lda['sd'] @ lda['W']
    Fp = (Fp - Fp.mean(1, keepdim=True)) / (Fp.std(1, keepdim=True) + 1e-6)
    Z = torch.linalg.svd(Fp, full_matrices=False)[0][..., :p]
    mu = X.mean(1, keepdim=True); C = X - mu
    C = C - Z @ (Z.transpose(1, 2) @ C)
    return shrink(C, 3, lambda lam, al: torch.full_like(lam, .5)) + .5 * mu


def fit_lda_eeg1(subjects):
    """= score_track1.fit_fingerprint_lda over `subjects` + the 42 extracted THINGS-EEG1 identities (E48c caches)."""
    from neurips_challenge.score_track1 import FP_BANDS, eeg_fingerprint, fit_fingerprint_lda
    from neurips_challenge.neighbour_lda import lda
    fit_fingerprint_lda(DATA, subjects)  # builds the per-subject fingerprint caches if missing
    tag = '-'.join(f'{lo}_{hi}' for lo, hi in FP_BANDS)
    Fs = [np.load(f'{DATA}/.cache/fp_train_s{s:02d}_n3000_seed0_{tag}.npy') for s in subjects]
    Fs += [eeg_fingerprint(torch.tensor(np.load(f)[:3000])).numpy() for f in sorted(glob.glob(f'{DATA}/.cache/eeg1id_s*.npy'))]
    Ftr = np.concatenate(Fs).astype(np.float64); m0, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    mu, W = lda((Ftr - m0) / sd, np.repeat(np.arange(len(Fs)), [len(f) for f in Fs]), dim=32)
    return {'m0': torch.tensor(m0 + mu * sd, dtype=torch.float32), 'sd': torch.tensor(sd, dtype=torch.float32),
            'W': torch.tensor(W, dtype=torch.float32)}


def configs(name, tags=()):
    c = {'plain': None, 'k3': {'W': None}, 'router fpldaT p6': {'router': True}}
    if name == 'oracle':
        for q in (1.0, .9, .7, .5, .3):
            c[f'oracle LOO q{q} c1'] = {'W': dict(src='oracle', q=q)}
        c['oracle LOO q1 c.5'] = {'W': dict(src='oracle', q=1.0), 'c': .5}
        c['oracle self q1 c1'] = {'W': dict(src='oracle', q=1.0, self=True)}
        for lam in (2, 6):
            c[f'oracle LOO q1 c1 lam{lam}'] = {'W': dict(src='oracle', q=1.0, lam=lam)}
    if name == 'whiten':  # E49: neighbour whitening, neighbours from the oracle / the LDA fingerprint / embeddings
        ds = (.95, .97, .98, .99, .995, .999, 1.0)
        for d in ds:
            c[f'oracle whiten d{d}'] = {'NW': dict(src='oracle', q=1.0), 'd': d}
            c[f'oracle whiten d{d} +k3'] = {'NW': dict(src='oracle', q=1.0), 'd': d, 'post': 3}
        for tag in ('lda',) + tuple(tags):
            for K in (4, 8, 16, 32):
                for d in ds:
                    c[f'{tag} whiten K{K} d{d}'] = {'NW': dict(src='emb', tag=tag, K=K), 'd': d}
            for d in ds:
                c[f'{tag} whiten K8 d{d} +k3'] = {'NW': dict(src='emb', tag=tag, K=8), 'd': d, 'post': 3}
    if name == 'ldaE':  # E50: the deployed router with its LDA refitted on 61 identities (+ THINGS-EEG1)
        for p in (4, 6, 8):
            c[f'router fpldaE p{p}'] = {'router_lda': 'ldaE', 'p': p}
        c['router fpldaT p8'] = {'router_lda': 'ldaT', 'p': 8}
    if name == 'emb':
        for tag in tags:
            for K in (4, 8):
                c[f'{tag} kNN K{K}'] = {'W': dict(src='emb', tag=tag, K=K)}
                c[f'{tag} kNN K{K} lam4'] = {'W': dict(src='emb', tag=tag, K=K, lam=4)}
            c[f'{tag} kNN K4 tau.05'] = {'W': dict(src='emb', tag=tag, K=4, tau=.05)}
            c[f'{tag} as router p6'] = {'router': tag}
    return c


def main(a):
    from neurips_challenge.neighbour_lda import fingerprint
    from neurips_challenge.score_track1 import fit_fingerprint_lda
    cfg = configs(a.cfgs, a.emb); res = {n: {} for n in cfg}; pur = {}
    for k in (1, 2):
        t0 = time.time()
        out = f'logs/route_{a.cfgs}{"_" + "_".join(a.emb) if a.emb else ""}_split{k}.json'
        if os.path.exists(out):  # finished split (a rerun after an OOM skips it)
            for nm, v in json.load(open(out)).items():
                res.setdefault(nm, {}).update({(int(s), k): x for s, x in v.items()})
            continue
        lab, Xall, G = load_split(a.recipes, k); subs = SPLITS[k]
        n = [len(x) for x in lab]; lab = np.concatenate(lab)
        sid_all = np.repeat(np.arange(len(subs)), n); key = torch.tensor(sid_all * 200 + lab)
        Fall = np.concatenate([fingerprint(s) for s in subs]).astype(np.float32)
        for t in a.emb:  # queued behind the embedder job: wait for its output (up to 3 h)
            for _ in range(360):
                if os.path.exists(f'results/alljoined/track1/routing/emb_{t}_split2.npy'):
                    break
                time.sleep(30)
        Eall = {t: np.load(f'results/alljoined/track1/routing/emb_{t}_split{k}.npy') for t in a.emb}
        assert all(len(e) == len(key) for e in Eall.values()), 'embedding / trial cache order mismatch'
        others = [s for s in SPLITS[3 - k] if s not in subs]
        lda = fit_fingerprint_lda(DATA, others + list(range(21, 31)))
        lda = {x: v.to(dev) for x, v in lda.items() if x != 'bands'}; G = G.to(dev)
        ldas = {'ldaT': lda}
        if any('router_lda' in c for c in cfg.values() if c):
            ldas['ldaE'] = {x: v.to(dev) for x, v in fit_lda_eeg1(others + list(range(21, 31))).items()}
        o = np.random.default_rng(11 + k).permutation(len(key)); nb = len(o) // 64; o = o[:nb * 64]
        Q = {nm: torch.zeros(len(subs) * 200, 1536, device=dev) for nm in cfg}; key = key.to(dev)
        for j in range(0, nb, a.chunk):
            idx = o[j * 64:(j + a.chunk) * 64]; B = len(idx) // 64
            sid = torch.tensor(sid_all[idx], device=dev).view(B, 64)
            F = torch.tensor(Fall[idx], device=dev).view(B, 64, -1)
            Xs = [torch.tensor(x[idx], dtype=torch.float32, device=dev).view(B, 64, -1) for x in Xall]
            Xs = [x / x.norm(dim=-1).mean(1)[:, None, None] for x in Xs]
            Es = {t: torch.tensor(e[idx], dtype=torch.float32, device=dev).view(B, 64, -1) for t, e in Eall.items()}
            Fl = (F - lda['m0']) / lda['sd'] @ lda['W']                     # the deployed LDA fingerprint space
            Es['lda'] = (Fl - Fl.mean(1, keepdim=True)) / (Fl.std(1, keepdim=True) + 1e-6)
            ident = lambda d: {'m0': torch.zeros(d, device=dev), 'sd': torch.ones(d, device=dev), 'W': torch.eye(d, device=dev)}
            Ws = {}
            for nm, c in cfg.items():
                if c is None:
                    Y = sum(Xs)
                else:
                    if 'router_lda' in c:
                        Y = sum(router(x, F, ldas[c['router_lda']], c['p']) for x in Xs)
                        Q[nm].index_add_(0, key[idx], Y.reshape(B * 64, -1))
                        continue
                    if 'NW' in c:
                        nw = c['NW']; wk = (nw['src'], nw.get('tag'), nw.get('K'))
                        if wk not in Ws:
                            Ws[wk] = affinities(sid, nw, Es.get(nw.get('tag')))
                        Y = sum(nbr_whiten(x, Ws[wk], c['d']) for x in Xs)
                        if c.get('post'):
                            mu = Y.mean(1, keepdim=True)
                            Y = shrink(Y - mu, c['post'], lambda lam, al: torch.full_like(lam, .5)) + mu
                        Q[nm].index_add_(0, key[idx], Y.reshape(B * 64, -1))
                        continue
                    if c.get('W') is not None:
                        Ws[nm] = affinities(sid, c['W'], Es.get(c['W'].get('tag')))
                    W = Ws.get(nm)
                    r = c.get('router'); Fr, Lr = (F, lda) if r is True else (Es.get(r), ident(Es[r].shape[-1]) if r else None)
                    Y = sum(router(x, Fr, Lr) if r else transform(x, W, c.get('c', 1.0)) for x in Xs)
                Q[nm].index_add_(0, key[idx], Y.reshape(B * 64, -1))
        for nm in cfg:
            S = Q[nm].view(len(subs), 200, -1) @ G.T; d = S.diagonal(dim1=1, dim2=2)
            t5 = 100 * ((S > d[..., None]).sum(-1) < 5).float().mean(1)
            res[nm].update({(s, k): t5[i].item() for i, s in enumerate(subs)})
        Q = {nm: None for nm in Q}
        json.dump({nm: {str(s): res[nm][(s, k)] for s in subs} for nm in cfg}, open(out, 'w'))
        del Xall, Fall, Eall, Q
        print(f'split {k} done ({time.time() - t0:.0f}s)', flush=True)
    base = res['plain']; ref = res['router fpldaT p6']
    print(f'{"+".join(a.recipes)}: s11 exact testbed, 11 unseen participants per shuffled 64-batch (22 subject x split)')
    for nm, v in res.items():
        x = np.array([v[q] for q in base]); b = np.array([base[q] for q in base]); r = np.array([ref[q] for q in base])
        print(f'RESULT {nm:30s} {x.mean():6.2f} {x.mean() - b.mean():+5.2f} ({(x > b).sum():2d}/{(x < b).sum():2d})'
              f'  vs router {x.mean() - r.mean():+5.2f} ({(x > r).sum():2d}/{(x < r).sum():2d})', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='*', default=['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_things'])
    ap.add_argument('--cfgs', default='oracle'); ap.add_argument('--emb', nargs='*', default=[])
    ap.add_argument('--device', default='cpu'); ap.add_argument('--chunk', type=int, default=128); a = ap.parse_args()
    dev = torch.device(a.device)
    sid = torch.tensor([[0, 0, 1, 1, 1, 2]])
    W = affinities(sid, dict(src='oracle', q=1.0))
    assert torch.allclose(W.sum(-1), torch.ones(1, 6)) and W[0, 0, 1] == 1 and W[0, 2, 3] == .5 and W[0, 5, :5].sum() == 1
    X = torch.randn(1, 6, 8) + torch.tensor([5., 5., -3., -3., -3., 0.])[None, :, None]  # per-person offsets
    C = X - X.mean(1, keepdim=True) - (W @ X - X.mean(1, keepdim=True))
    assert C[0, :5].mean().abs() < X[0, :5].mean().abs() / 2, 'group centring must remove the person offsets'
    # nbr_whiten == the direct per-trial computation (LW-shrunk neighbour covariance, eigendecomposition in D space)
    torch.manual_seed(0); X = torch.randn(2, 10, 30, dtype=torch.float64); wq = torch.zeros(2, 10, 10, dtype=torch.float64)
    for b in range(2):
        for i in range(10):
            nbr = torch.randperm(10)[:5]; nbr = nbr[nbr != i][:4]; wq[b, i, nbr] = 1 / len(nbr)
    Y = nbr_whiten(X, wq, .9)
    b, i = 1, 3; nbr = wq[b, i] > 0; mu = X[b, nbr].mean(0); Cn = X[b, nbr] - mu; Sn = Cn.T @ Cn / nbr.sum()
    lam, V = torch.linalg.eigh(.1 * Sn + .9 * Sn.trace() / 30 * torch.eye(30, dtype=torch.float64))
    want = V @ torch.diag((.9 * Sn.trace() / 30 / lam).sqrt()) @ V.T @ (X[b, i] - mu) + .5 * mu
    assert torch.allclose(Y[b, i], want, atol=1e-8), 'nbr_whiten != direct LW whitening'
    assert torch.allclose(nbr_whiten(X, wq, 1.0), X - .5 * wq @ X), 'd = 1 must be neighbour centring'
    main(a)
