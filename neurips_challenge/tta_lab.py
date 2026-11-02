"""Replay test-time methods on the per-trial cache (tta_cache.py), official subject-aggregated top-5 (TTA_LOG.md).
A method maps the fold's stream to the summed ensemble output (N, 1536); deployable methods only look at one 64-trial
batch at a time (`per_batch`); ORACLE methods may use identities/labels and exist only to bound headroom.
    tta_lab.py <method> [...] [--members reve_d6_things sqf_dbn_things tsc_dbn40_bn64_things]
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import official  # noqa: E402
from neurips_challenge.score_track1 import batch_transform, batch_transform_fp  # noqa: E402
from neurips_challenge.tta_cache import SETS  # noqa: E402
OUT = SETS['cv4'][0]

BS = int(os.environ.get('TTA_BS', 64))  # predict() batch size of the replayed stream
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
REF = ['reve_d6_things', 'sqf_dbn_things', 'tsc_dbn40_bn64_things']
l2 = lambda x: x / x.norm(dim=-1, keepdim=True)


def load(k, members):
    c = np.load(f'{OUT}/fold{k}.npz')
    ctx = {key: c[key] for key in ('img', 'subj', 'idx', 'G')}
    ctx['F'] = torch.as_tensor(c['F'], device=DEV)
    ctx['lda'] = {key[4:]: torch.as_tensor(c[key], device=DEV) for key in c.files if key.startswith('lda_')}
    def one(r):  # '<recipe>@tta' = mean over the time-shifted views (tta_cache.py --shifts)
        if r.endswith('@mir'):  # mean of the normal and the left-right mirrored scalp (T44)
            v = [np.load(f'{OUT}/{r[:-4]}_f{k}' + s_ + '.npy').astype(np.float32) for s_ in ('', '_sm')]
            return torch.as_tensor((v[0] + v[1]) / 2, device=DEV).half()
        if '@s' in r:  # '<recipe>@s<d>' = the input shifted by d samples (T43; d = m: mirrored)
            return torch.as_tensor(np.load(f'{OUT}/{r.split("@s")[0]}_f{k}_s{r.split("@s")[1]}.npy'), device=DEV)
        if r.endswith('@tta'):
            v = [np.load(f'{OUT}/{r[:-4]}_f{k}' + (f'_s{d}' if d else '') + '.npy').astype(np.float32) for d in (0, -2, -1, 1, 2)]
            return torch.as_tensor(sum(v) / len(v), device=DEV).half()
        return torch.as_tensor(np.load(f'{OUT}/{r}_f{k}.npy'), device=DEV)
    ctx['P'] = [one(r) for r in members]  # float16; float per batch
    return ctx


def session_keys(ctx, level):
    """per-trial grouping key: 'p' person, 's' person x session, 'r' person x session x run (from the event parquet,
    the same walk as grader_sim.chronological)"""
    if level == 'p':
        return ctx['subj'].astype(np.int64)
    import glob, pandas as pd
    from neurips_challenge.grader_sim import DATA, PARQUET_GLOB, N_TRAIN_IMAGES
    lut = {}
    for subject in np.unique(ctx['subj']):
        ev = pd.concat([pd.read_parquet(f, columns=['type', 'start', 'label', 'stim_file', 'session', 'run'])
                        for f in glob.glob(PARQUET_GLOB) if f'subject={subject}-' in os.path.basename(f)])
        ev = ev[(ev.type == 'Image') & (ev.label == 'stim_test')].sort_values(['session', 'run', 'start'])
        counts = np.load(f'{DATA}/sub-{subject:02d}/test_counts.npy').ravel(); seen = np.zeros(200, int)
        sess = {v: j for j, v in enumerate(sorted(ev.session.unique()))}
        for stim, se, ru in zip(ev.stim_file, ev.session, ev.run):
            i = int(stim.split('.')[0]) - N_TRAIN_IMAGES
            if seen[i] < min(counts[i], 80):
                lut[(int(subject), i * 80 + seen[i])] = sess[se] * 100 + int(ru) * (level == 'r')
            seen[i] += 1
    return np.array([s_ * 100000 + lut[(int(s_), int(i))] for s_, i in zip(ctx['subj'], ctx['idx'])], np.int64)


def per_batch(fn):
    """deployable: fn(list of member outputs (B, D), F (B, 656), lda) -> (B, D), one 64-trial batch at a time"""
    def run(ctx):
        out = []
        for b in range(0, len(ctx['img']), BS):
            out.append(fn([P[b:b + BS] for P in ctx['P']], ctx['F'][b:b + BS], ctx['lda']))
        return torch.cat(out)
    return run


def mn(P):  # every member's batch output divided by its mean norm (= grader_sim)
    return [p.float() / p.float().norm(dim=1).mean() for p in P]


METHODS = {
    'plain': per_batch(lambda P, F, lda: sum(mn(P))),
    'k2': per_batch(lambda P, F, lda: sum(batch_transform(p, k=2) for p in mn(P))),
    'k3': per_batch(lambda P, F, lda: sum(batch_transform(p, k=3) for p in mn(P))),
    'ref': per_batch(lambda P, F, lda: sum(batch_transform_fp(p, F, lda) for p in mn(P))),
}


def routed(ctx):
    """per-member router outputs (list of (N, D)), cached on ctx"""
    if 'R' not in ctx:
        ctx['R'] = [per_batch(lambda P, F, lda, m=m: batch_transform_fp(mn(P)[m], F, lda))(ctx) for m in range(len(ctx['P']))]
    return ctx['R']


def oracle_person_centring(ctx, k=3, level='p'):
    """ORACLE: per member, subtract the person's mean over ALL their trials (first-order person offset, exact), shrink the
    batch's top-k directions, keep half the batch mean (= router without estimation error in the offset)"""
    keys = session_keys(ctx, level); subj, out = torch.as_tensor(keys, device=DEV), 0
    for m in range(len(ctx['P'])):
        Q = per_batch(lambda P, F, lda, m=m: mn(P)[m])(ctx)
        ms = {int(s): Q[subj == s].mean(0) for s in np.unique(keys)}
        C = Q - torch.stack([ms[int(s)] for s in keys])
        out = out + torch.cat([batch_transform(C[b:b + BS], a=0, k=k) + .5 * Q[b:b + BS].mean(0) for b in range(0, len(Q), BS)])
    return out


def weighted(ctx, kind, g):
    """vote weights on the routed ensemble output y: the scorer averages raw outputs, so y_i * w_i is a weighted mean"""
    R = routed(ctx); y = sum(R)
    if kind == 'norm':      # amplify (g > 0) / flatten (g < 0) the existing norm weights
        w = y.norm(dim=1) ** g
    elif kind == 'agree':   # member consensus: mean pairwise cosine of the members' routed outputs
        U = [l2(r) for r in R]
        a = sum((U[i] * U[j]).sum(1) for i in range(len(U)) for j in range(i + 1, len(U))) / (len(U) * (len(U) - 1) / 2)
        w = a.clamp_min(1e-3) ** g
    elif kind == 'oracle':  # LEAKS the target (pulls each mean toward it): not a valid bound, kept as a warning
        G = torch.as_tensor(ctx['G'], device=DEV).float()[torch.as_tensor(ctx['img'], device=DEV)]
        w = (l2(y) * l2(G)).sum(1).clamp_min(1e-3) ** g
    elif kind == 'resid':   # ORACLE reliability without the target: distance to the trial's own (person, image) mean
        key = torch.as_tensor(ctx['subj'] * 1000 + ctx['img'], device=DEV)
        u, inv = torch.unique(key, return_inverse=True)
        mean = torch.zeros(len(u), y.shape[1], device=DEV).index_add_(0, inv, y) / torch.bincount(inv)[:, None]
        r = (y - mean[inv]).norm(dim=1)
        w = (r / r.mean()) ** -g
    return y * w[:, None]


def lw_shrink(Y, d, centre):
    """Ledoit-Wolf-style batch whitening in Gram form: each batch direction i (eigenvalue lam_i of the (un)centred second
    moment) is scaled by sqrt(d nu / ((1 - d) lam_i + d nu)), nu = mean eigenvalue over all D dims; directions outside
    the batch span keep scale 1. d -> 1 touches only the very top directions (TTA_LOG T23)."""
    mu = Y.mean(0, keepdim=True) if centre else torch.zeros_like(Y[:1]); C = Y - mu
    U, sv, Vt = torch.linalg.svd(C, full_matrices=False); lam = sv ** 2 / len(Y); nu = lam.sum() / Y.shape[1]
    r = torch.sqrt(d * nu / ((1 - d) * lam + d * nu))
    return C - ((C @ Vt.T) * (1 - r)) @ Vt + mu


def lw_intensity(Y):
    """Ledoit-Wolf (2004) optimal shrinkage intensity of the uncentred second moment S = Y'Y / n toward nu I, from the
    batch itself (Gram form): d = min(b2, d2) / d2, d2 = ||S - nu I||_F^2, b2 = (sum_k ||y_k||^4 / n - ||S||_F^2) / n"""
    n, p = Y.shape; G = Y @ Y.T; S2 = (G ** 2).sum() / n ** 2; tr = G.diagonal().sum() / n
    d2 = S2 - tr ** 2 / p; b2 = ((G.diagonal() ** 2).sum() / n - S2) / n
    return (torch.minimum(b2, d2) / d2).item()


LWA = []  # chosen intensities (printed)


def target_match(Y, M):
    """T36 prior-matched shrink (no knob): batch direction v_j (uncentred energy lam_j = s_j^2 / B) is scaled by
    min(1, sqrt(alpha v_j' M v_j / lam_j)), M = uncentred second moment of the TRAINING image targets, alpha = the
    energy ratio tr(Y'Y / B) / tr(M); directions louder than the image prior allows are nuisance"""
    _, sv, Vt = torch.linalg.svd(Y, full_matrices=False); lam = sv ** 2 / len(Y)
    tau = ((Vt @ M) * Vt).sum(1) * lam.sum() / torch.trace(M)
    r = torch.sqrt(tau / lam.clamp_min(1e-12)).clamp(max=1.)
    return Y - ((Y @ Vt.T) * (1 - r)) @ Vt


def batch_grouping(ctx, weights, c=1., k=3, tag='', reg=None, b_=.5, a_=.5, post=None):
    """centre each trial on a weighted mean of its batch-mates, W = weights(b) (B, B) with zero diagonal and rows
    summing to 1 (or 0: fall back to the batch mean); then shrink the batch's top-k directions, keep half the batch
    mean. reg = (proj, p, cr): then regress the top-p within-batch PCs of proj(F) out of the centred outputs (x cr).
    Prints the purity = mean weight on true same-person batch-mates."""
    subj, out, pur = torch.as_tensor(ctx['subj'], device=DEV), 0, []
    Ws = [weights(b) for b in range(0, len(ctx['img']), BS)]
    for b, W in zip(range(0, len(ctx['img']), BS), Ws):
        s = subj[b:b + BS]; same = (s[:, None] == s[None, :]).float(); Wm = W[:, :len(s)]
        pur.append(((Wm * same).sum(1) / Wm.sum(1).clamp_min(1e-9))[Wm.sum(1) > 0])
    print(f'  {tag} purity {torch.cat(pur).mean():.3f}', flush=True)
    for m in range(len(ctx['P'])):
        Q = per_batch(lambda P, F, lda, m=m: mn(P)[m])(ctx)
        res = []
        for b, W in zip(range(0, len(Q), BS), Ws):
            q = Q[b:b + BS]; mu = q.mean(0, keepdim=True)
            if W.shape[1] == len(q) + 1:  # rows already normalised, last column = weight on the batch mean
                loo = W[:, :-1] @ q + W[:, -1:] * mu
            else:
                n = W.sum(1, keepdim=True); loo = torch.where(n > 0, W @ q / n.clamp_min(1e-9), mu)
            cm = c[m] if isinstance(c, (list, tuple)) else c  # per-member strength
            C = q - mu - cm * (loo - mu)
            if reg:
                Z = reg[0](ctx['F'][b:b + BS]).float(); p_ = min(reg[1], (len(q) - 1) // 8)
                if p_ > 0:
                    U = torch.linalg.svd(Z, full_matrices=False)[0][:, :p_]; C = C - reg[2] * U @ (U.T @ C)
            if post and post[0] == 'lw':     # centred LW whitening of the person-centred batch, keep (1 - a) mu
                res.append(lw_shrink(C, post[1], True) + (1 - a_) * mu)
            elif post and post[0] == 'lwu':  # ONE op: person offsets removed, batch mean kept, uncentred LW shrink
                res.append(lw_shrink(C + mu, post[1], False))
            elif post and post[0] == 'lwun':  # T31: lwu, then each member re-scaled to unit mean norm (equal votes)
                y_ = lw_shrink(C + mu, post[1], False); res.append(y_ / y_.norm(dim=1).mean())
            elif post and post[0] in ('lwj', 'lws', 'tms', 'lwa', 'lwn'):  # T30 / T32: shrink JOINT over the members (concatenated / summed)
                res.append((C + mu).half())  # fp16: 10 members x the whole stream
            else:
                res.append(batch_transform(C, a=0, k=k, b=b_) + (1 - a_) * mu)
        if post and post[0] in ('lws', 'tms', 'lwa', 'lwn'):
            joint = res if m == 0 else [j + r for j, r in zip(joint, res)]
            continue
        if post and post[0] == 'lwj':
            joint = res if m == 0 else [torch.cat([j, r], 1) for j, r in zip(joint, res)]
            continue
        out = out + torch.cat(res)
    if post and post[0] == 'lws':
        out = torch.cat([lw_shrink(j.float(), post[1], False) for j in joint])
    if post and post[0] == 'lwn':  # T39: no knob: shrink toward the MEAN BATCH eigenvalue, d = p / (p + n)
        out = torch.cat([lw_shrink(j.float(), j.shape[1] / (j.shape[1] + len(j)), False) for j in joint])
    if post and post[0] == 'lwa':  # T39: lws with the batch's own Ledoit-Wolf intensity (no d)
        out = []
        for j in joint:
            LWA.append(lw_intensity(j.float())); out.append(lw_shrink(j.float(), LWA[-1], False))
        out = torch.cat(out); print(f'  lwa intensity mean {np.mean(LWA):.3f} sd {np.std(LWA):.3f}', flush=True); LWA.clear()
    if post and post[0] == 'tms':
        B_ = torch.as_tensor(np.load(BANK), device=DEV).float()
        B_ = B_ - B_.mean(0) if post[1] else B_  # tms:1 = centred target prior (the batch mean then counts as nuisance)
        M = B_.T @ B_ / len(B_)
        out = torch.cat([target_match(j.float(), M) for j in joint])
    if post and post[0] == 'lwj':
        D = joint[0].shape[1] // len(ctx['P'])
        out = torch.cat([lw_shrink(j.float(), post[1], False).reshape(len(j), -1, D).sum(1) for j in joint])
    return out


def oracle_batch_grouping(ctx, c=1., k=3, level='p', post=None):
    """ORACLE stateless grouping: same-person (or session) batch-mates, leave-one-out = the best any in-batch person
    router can do with first-order statistics"""
    key = torch.as_tensor(session_keys(ctx, level), device=DEV)

    def weights(b):
        s = key[b:b + BS]; W = (s[:, None] == s[None, :]).float(); W.fill_diagonal_(0)
        return W
    return batch_grouping(ctx, weights, c, k, tag=f'oracle {level}', post=post)


def fp_lda(F, lda):
    """the router's fingerprint space: LDA projection, z-scored within the batch"""
    Fp = ((F - lda['m0']) / lda['sd']) @ lda['W']
    return (Fp - Fp.mean(0)) / (Fp.std(0) + 1e-6)


def knn_grouping(ctx, K, c, tau=0., lam=0.):
    """deployable: neighbours = top-K cosine batch-mates in the router's fingerprint space (tau > 0: softmax weights);
    lam > 0 adds lam x the cosine of the batch-centred summed member outputs (they carry the person offset)"""
    def weights(b):
        Z = l2(fp_lda(ctx['F'][b:b + BS], ctx['lda'])); S = Z @ Z.T
        if lam:
            O = sum(mn([P[b:b + BS] for P in ctx['P']])); O = l2(O - O.mean(0)); S = S + lam * O @ O.T
        S.fill_diagonal_(-9)
        if tau:
            W = torch.softmax(S / tau, 1)
        else:
            W = torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.)
        W.fill_diagonal_(0)
        return W
    return batch_grouping(ctx, weights, c, tag=f'knn K{K} tau{tau} lam{lam}')


def wccn_affinity(ctx, shrink=.3, things=False, groups=('base',)):
    """fingerprint affinity after within-person whitening (WCCN, all 656 dims), fitted on the fold's TRAINING people;
    cached on ctx (group_purity.py: purity@4 0.74 vs 0.59 for the router's LDA space). Alljoined people only by default
    (T6: the THINGS-EEG2 people, other hardware, do not help the within-person covariance)"""
    key = ('wccn', shrink, groups)
    if key not in ctx:
        for k_ in [k_ for k_ in ctx if isinstance(k_, tuple)]:  # one feature set on the GPU at a time
            del ctx[k_]
        torch.cuda.empty_cache()
        from neurips_challenge.group_purity import fit, train_fps, test_fps
        subs = set(np.unique(ctx['subj']).tolist())
        r = [int(g[3:]) for g in groups if g.startswith('pca')]  # 'pca<r>' token: PCA to r dims before WCCN (T38)
        fg = tuple(g for g in groups if not g.startswith('pca'))
        X, y = train_fps([s for s in range(1, 21) if s not in subs] + (list(range(21, 31)) if things else []), fg)
        ctx[key] = fit('wccn', X, y, shrink=shrink, r=r[0] if r else None)
        ctx[('feat', groups)] = test_fps({'F': ctx['F'].cpu().numpy(), 'subj': ctx['subj'], 'idx': ctx['idx']}, fg)
    return ctx[key], ctx[('feat', groups)]


def wknn_grouping(ctx, K, c, tau=0., shrink=.3, p=0, cr=1., groups=('base',), k=3, b_=.5, a_=.5, post=None):
    aff, feat = wccn_affinity(ctx, shrink, groups=groups)

    def weights(b):
        S = aff(feat[b:b + BS]).float()
        if PROFILE:  # T47 second-order affinity: cosine between the trials' similarity profiles over the whole batch
            Pf = S.clone(); Pf.fill_diagonal_(0); Pf = l2(Pf - Pf.mean(1, keepdim=True)); S = Pf @ Pf.T
        S.fill_diagonal_(-9)
        W = torch.softmax(S / tau, 1) if tau else torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.)
        W.fill_diagonal_(0)
        return W
    return batch_grouping(ctx, weights, c, k=k, tag=f'wknn K{K} tau{tau} s{shrink} p{p}', reg=(aff.proj, int(p), cr) if p else None, b_=b_, a_=a_, post=post)


def spectral_labels(S, k=0, kmax=12):
    """T42 diarisation-style clustering of one batch: normalised-Laplacian spectral clustering of the affinity
    max(S, 0); the number of people = the largest eigengap (k = 0) unless given; k-means (farthest-point init)"""
    A = S.clamp_min(0); A.fill_diagonal_(0); dm = A.sum(1).clamp_min(1e-9).rsqrt()
    w, V = torch.linalg.eigh(torch.eye(len(A), device=A.device) - dm[:, None] * A * dm[None, :])
    if not k:
        k = int((w[2:kmax + 1] - w[1:kmax]).argmax()) + 2
    X = l2(V[:, :k]); C = X[:1]
    for _ in range(k - 1):
        C = torch.cat([C, X[(X @ C.T).max(1).values.argmin()][None]])
    for _ in range(20):
        lab = (X @ C.T).argmax(1)
        C = torch.stack([l2(X[lab == j].mean(0)) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def spectral_grouping(ctx, c, k=0, groups=('base',), post=('lwn', 0.)):
    aff, feat = wccn_affinity(ctx, .3, groups=groups); ks = []

    def weights(b):
        lab = spectral_labels(aff(feat[b:b + BS]).float(), k); ks.append(int(lab.max()) + 1)
        W = (lab[:, None] == lab[None, :]).float(); W.fill_diagonal_(0)
        return W
    out = batch_grouping(ctx, weights, c, tag=f'spectral k{k}', post=post)
    print(f'  spectral clusters per batch: mean {np.mean(ks):.2f}', flush=True)
    return out


def bayes_grouping(ctx, beta, theta, kappa, c=1., groups=('base',)):
    """N-agnostic person-mean estimate: batch-mate weights w_ij = exp(beta (S_ij - theta)) (S = WCCN cosine), and the
    trial's person mean is shrunk to the batch mean by its evidence: m = (sum_j w_ij q_j + kappa mu) / (sum_j w_ij + kappa).
    Few confident mates (large N, small batches) -> little centring; many -> full. Then as batch_grouping."""
    aff, feat = wccn_affinity(ctx, groups=groups)

    def weights(b):
        S = aff(feat[b:b + BS]).float()
        W = torch.exp(beta * (S - theta)).clamp_max(1e3); W.fill_diagonal_(0)
        n = W.sum(1, keepdim=True)
        return torch.cat([W / (n + kappa), kappa / (n + kappa)], 1)  # last column: weight of the batch mean
    return batch_grouping(ctx, weights, c, tag=f'bayes b{beta} t{theta} k{kappa}')


def thr_grouping(ctx, theta, kmin, kmax, c=.7, groups=('base',)):
    """adaptive K: even average over the batch-mates whose WCCN cosine exceeds theta, at least kmin and at most kmax of
    them (top-ranked first). T11: P(same | S) depends on S, so the number of trusted mates should follow the evidence."""
    aff, feat = wccn_affinity(ctx, groups=groups)

    def weights(b):
        S = aff(feat[b:b + BS]).float(); S.fill_diagonal_(-9)
        hi = min(int(kmax), len(S) - 1); v, i = S.topk(hi, 1)
        keep = (v > theta) | (torch.arange(hi, device=DEV)[None, :] < kmin)
        return torch.zeros_like(S).scatter_(1, i, keep.float())
    return batch_grouping(ctx, weights, c, tag=f'thr t{theta} k{kmin}-{kmax}')


def mutual_grouping(ctx, K, c, wnm, groups=('base',)):
    """top-K neighbours, weight 1 if the neighbourhood is mutual (j also has i in its top K), wnm otherwise"""
    aff, feat = wccn_affinity(ctx, groups=groups)

    def weights(b):
        S = aff(feat[b:b + BS]).float(); S.fill_diagonal_(-9)
        A = torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.)
        return A * (wnm + (1 - wnm) * A.T)
    return batch_grouping(ctx, weights, c, tag=f'mutual K{K} wnm{wnm}')


def z_grouping(ctx, z, c=.7, kmin=2, kmax=12, groups=('base',)):
    """self-calibrated adaptive K (T25): trial i trusts the batch-mates whose WCCN cosine exceeds ITS OWN row median by
    z robust deviations (1.4826 MAD); most pairs in a mixed batch are different people, so the row median calibrates
    the threshold to the batch (N people, feature scale) with no offline calibration; kmin..kmax mates, top first."""
    aff, feat = wccn_affinity(ctx, groups=groups)

    def weights(b):
        S = aff(feat[b:b + BS]).float(); S.fill_diagonal_(float('nan'))
        med = S.nanmedian(1, keepdim=True).values; mad = 1.4826 * (S - med).abs().nanmedian(1, keepdim=True).values
        S = torch.nan_to_num(S, nan=-9.)
        hi = min(int(kmax), len(S) - 1); v, i = S.topk(hi, 1)
        keep = (v > med + z * mad) | (torch.arange(hi, device=DEV)[None, :] < kmin)
        return torch.zeros_like(S).scatter_(1, i, keep.float())
    return batch_grouping(ctx, weights, c, tag=f'zknn z{z}')


def arm_grouping(ctx, K, c, rank=32, soft=False):
    """T26: neighbours in the affinity space learned end to end by tta_arm.py on the OTHER folds' streams"""
    from neurips_challenge.grader_sim import FOLDS
    aff, feat = wccn_affinity(ctx)
    k = next(k_ for k_, v in FOLDS.items() if set(v) == set(np.unique(ctx['subj']).tolist()))
    A = torch.load(f'{OUT}/arm/fold{k}_r{rank}.pt'); U, V = A['U'].to(DEV), A['V'].to(DEV)
    c = A['c'] if c < 0 else c

    def weights(b):
        z = (((feat[b:b + BS].double() - aff.m0) / aff.sd) @ aff.T).float(); z = z - z.mean(0); z = l2(z + (z @ V) @ U.T)
        S = z @ z.T; S.fill_diagonal_(-9)
        if soft:
            W = torch.softmax(S / A['tau'], 1); W.fill_diagonal_(0); return W
        return torch.zeros_like(S).scatter_(1, S.topk(min(K, len(S) - 1), 1).indices, 1.)
    return batch_grouping(ctx, weights, c, tag=f'arm K{K} soft{soft}')


def gp_grouping(ctx, beta, lam, c=.7, k=3, groups=('base',)):
    """in-batch kernel ridge (GP) regression of the batch-centred outputs on the fingerprint, kernel exp(beta (S - 1))
    on the WCCN cosine S, EXACT leave-one-out prediction m_i = q_i - [(A + lam I)^-1 Q]_i / [(A + lam I)^-1]_ii
    (never uses the trial's own output); centre by c x m, shrink top-k, keep half the batch mean."""
    aff, feat = wccn_affinity(ctx, groups=groups)
    out = 0
    for m in range(len(ctx['P'])):
        Q = per_batch(lambda P, F, lda, m=m: mn(P)[m])(ctx)
        res = []
        for b in range(0, len(Q), BS):
            q = Q[b:b + BS]; mu = q.mean(0, keepdim=True); X = q - mu
            A = torch.exp(beta * (aff(feat[b:b + BS]).float() - 1))
            Hi = torch.linalg.inv(A + lam * torch.eye(len(q), device=DEV))
            loo = X - (Hi @ X) / torch.diag(Hi)[:, None]
            res.append(batch_transform(X - c * loo, a=0, k=k) + .5 * mu)
        out = out + torch.cat(res)
    return out


def oracle_batch_person_pca(ctx, c=.7, kp=1, bp=.5, k=3):
    """ORACLE: perfect in-batch grouping (ocb), then each person's top-kp directions estimated from THEIR batch-mates
    only (centred outputs, >= 4 trials) shrunk by bp: is in-batch second order estimable at all?"""
    subj, out = torch.as_tensor(ctx['subj'], device=DEV), 0
    for m in range(len(ctx['P'])):
        Q = per_batch(lambda P, F, lda, m=m: mn(P)[m])(ctx)
        res = []
        for i in range(0, len(Q), BS):
            q, s = Q[i:i + BS], subj[i:i + BS]; mu = q.mean(0, keepdim=True)
            same = (s[:, None] == s[None, :]).float(); same.fill_diagonal_(0); n = same.sum(1, keepdim=True)
            C = q - mu - c * (torch.where(n > 0, same @ q / n.clamp_min(1), mu) - mu)
            for p_ in torch.unique(s):
                r = s == p_
                if r.sum() >= 4:
                    Cs = C[r] - C[r].mean(0); V = torch.linalg.svd(Cs, full_matrices=False)[2][:kp]
                    C[r] = C[r] - bp * (C[r] @ V.T) @ V
            res.append(batch_transform(C, a=0, k=k) + .5 * mu)
        out = out + torch.cat(res)
    return out


def oracle_person_pca(ctx, kp, b, k=3):
    """ORACLE second order: as oracle_person_centring, then shrink by b the person's OWN top-kp directions (from all
    their centred trials) before the batch top-k shrink"""
    subj, out = torch.as_tensor(ctx['subj'], device=DEV), 0
    for m in range(len(ctx['P'])):
        Q = per_batch(lambda P, F, lda, m=m: mn(P)[m])(ctx)
        C = torch.empty_like(Q)
        for s in np.unique(ctx['subj']):
            r = subj == int(s); Cs = Q[r] - Q[r].mean(0)
            V = torch.linalg.svd(Cs, full_matrices=False)[2][:kp]
            C[r] = Cs - b * (Cs @ V.T) @ V
        out = out + torch.cat([batch_transform(C[i:i + BS], a=0, k=k) + .5 * Q[i:i + BS].mean(0) for i in range(0, len(Q), BS)])
    return out


def method(name, ctx):
    if name in METHODS:
        return METHODS[name](ctx)
    if name.startswith('deploy'):  # deploy:K:c = score_track1.batch_transform_wknn, the submission code path
        from neurips_challenge.grader_sim import DATA
        from neurips_challenge.score_track1 import batch_transform_wknn, fit_fingerprint_wccn
        a = name.split(':'); subs = set(np.unique(ctx['subj']).tolist())
        w = {k_: v.to(DEV) for k_, v in fit_fingerprint_wccn(DATA, [s for s in range(1, 21) if s not in subs]).items()}
        return per_batch(lambda P, F, lda: batch_transform_wknn(P, F, w, K=int(a[1]), c=float(a[2])))(ctx)
    if name.startswith('mknn'):  # mknn:K:c:wnm[:groups]
        a = name.split(':')
        return mutual_grouping(ctx, int(a[1]), float(a[2]), float(a[3]), tuple(a[4].split('+')) if len(a) > 4 else ('base',))
    if name.startswith('zknn'):  # zknn:z:c[:groups]
        a = name.split(':')
        return z_grouping(ctx, float(a[1]), float(a[2]), groups=tuple(a[3].split('+')) if len(a) > 3 else ('base',))
    if name.startswith('arms'):  # arms:c (learned soft weights; c < 0 = learned c)
        return arm_grouping(ctx, 0, float(name.split(':')[1]), soft=True)
    if name.startswith('arm'):  # arm:K:c
        a = name.split(':')
        return arm_grouping(ctx, int(a[1]), float(a[2]))
    if name.startswith('thr'):  # thr:theta:kmin:kmax:c[:groups]
        a = name.split(':')
        return thr_grouping(ctx, float(a[1]), int(a[2]), int(a[3]), float(a[4]), tuple(a[5].split('+')) if len(a) > 5 else ('base',))
    if name.startswith('gp'):  # gp:beta:lam[:c]
        a = [float(x) for x in name.split(':')[1:]]
        return gp_grouping(ctx, *a)
    if name.startswith('bayes'):  # bayes:beta:theta:kappa[:c]
        a = [float(x) for x in name.split(':')[1:]]
        return bayes_grouping(ctx, *a)
    if name.startswith('spc'):  # spc:c[:k[:group+group]]  (T42: spectral clustering of the batch, lwn post-step)
        a = name.split(':')
        return spectral_grouping(ctx, float(a[1]), int(a[2]) if len(a) > 2 else 0, tuple(a[3].split('+')) if len(a) > 3 else ('base',))
    if name.startswith('wlw'):  # wlw:<lw|lwu>:d:K:c[:shrink[:group+group]]  (T23: whitening replaces the k/b/a shrink)
        a = name.split(':')
        return wknn_grouping(ctx, int(a[3]), float(a[4]), shrink=float(a[5]) if len(a) > 5 else .3, post=(a[1], float(a[2])),
                             groups=tuple(a[6].split('+')) if len(a) > 6 else ('base',))
    if name.startswith('wknnx'):  # wknnx:K:c:k:b:a  (global shrink k / b and kept batch mean 1 - a)
        a = name.split(':')
        return wknn_grouping(ctx, int(a[1]), float(a[2]), k=int(a[3]), b_=float(a[4]), a_=float(a[5]))
    if name.startswith('wknng'):  # wknng:K:c:<group+group> (identity feature groups, group_features.py)
        a = name.split(':')
        return wknn_grouping(ctx, int(a[1]), float(a[2]), groups=tuple(a[3].split('+')))
    if name.startswith('wknn'):  # wknn:K:c[:tau[:shrink[:p[:cr]]]], c may be per member: c1,c2,c3
        a = name.split(':')
        c = [float(x) for x in a[2].split(',')] if ',' in a[2] else float(a[2])
        return wknn_grouping(ctx, int(a[1]), c, *(float(x) for x in a[3:]))
    if name.startswith('knn'):  # knn:<K>:<c>[:<tau>]
        a = name.split(':')
        return knn_grouping(ctx, int(a[1]), float(a[2]), *(float(x) for x in a[3:]))  # knn:K:c[:tau[:lam]]
    if name.startswith('ocp'):  # ocp:<kp>:<b>[:<k>]
        a = name.split(':')[1:]
        return oracle_person_pca(ctx, int(a[0]), float(a[1]), *(int(x) for x in a[2:]))
    if name.startswith('ocbp'):  # ocbp:c:kp:bp
        a = name.split(':')
        return oracle_batch_person_pca(ctx, float(a[1]), int(a[2]), float(a[3]))
    if name.startswith('ocbn'):  # ocbn:<c>  perfect in-batch grouping + the lean lwn shrink of the sum (T47)
        return oracle_batch_grouping(ctx, c=float(name.split(':')[1]), post=('lwn', 0.))
    if name.startswith('ocbl'):  # ocbl:<c>:<d>  perfect in-batch grouping + the one-op lwu shrink (T32)
        a = name.split(':')
        return oracle_batch_grouping(ctx, c=float(a[1]), post=('lwu', float(a[2])))
    if name.startswith('ocb'):  # ocb:<c>[:<level p|s|r>]
        a = name.split(':')
        return oracle_batch_grouping(ctx, c=float(a[1]), level=a[2] if len(a) > 2 else 'p')
    if name.startswith('oc'):  # oc<k>[:<level>]
        a = name.split(':')
        return oracle_person_centring(ctx, k=int(a[0][2:] or 3), level=a[1] if len(a) > 1 else 'p')
    kind, g = name.split(':')
    return weighted(ctx, kind, float(g))


PROFILE = os.environ.get('TTA_PROFILE') == '1'  # T47: group on similarity profiles (second-order affinity)
BANK = os.environ.get('TTA_BANK')  # .npy of TRAINING-image CLIP targets (T28); the test gallery G is never used


def gallery_post(y, post, ctx):
    """T28 gallery-geometry post-steps on the final per-trial output (per-trial norm kept = vote weight kept):
    gm:a       remove a fraction a of the component along the mean training target (hubness direction)
    kr:tau:b   y + b |y| softmax(cos(y, B) / tau) B: soft projection onto the training-target manifold"""
    a = post.split(':')
    if a[0] == 'pow':  # T45 power-law normalisation per dimension (image-retrieval burstiness fix), per-trial norm kept
        z = y.sign() * y.abs() ** float(a[1])
        return z * (y.norm(dim=1, keepdim=True) / z.norm(dim=1, keepdim=True).clamp_min(1e-12))
    if a[0] in ('oim', 'sim'):  # T41: pool each trial with OTHER trials of the same image in its batch
        img = torch.as_tensor(ctx['img'], device=DEV); out = []
        for b in range(0, len(y), BS):
            q = y[b:b + BS]; u = l2(q - q.mean(0))
            if a[0] == 'oim':  # ORACLE: true same-image batch-mates (any person)
                W = (img[b:b + BS, None] == img[None, b:b + BS]).float(); W.fill_diagonal_(0)
            else:              # label-free: softmax over output cosine, temperature a[2]
                S = u @ u.T; S.fill_diagonal_(-1e4); W = torch.softmax(S / float(a[2]), 1)
            n = W.sum(1, keepdim=True)
            out.append(q + float(a[1]) * torch.where(n > 0, W @ q / n.clamp_min(1e-9), torch.zeros_like(q)))
        return torch.cat(out)
    B = torch.as_tensor(np.load(BANK), device=DEV).float(); Bn = l2(B)
    if a[0] == 'gm':
        g = l2(B.mean(0, keepdim=True))
        return y - float(a[1]) * (y @ g.T) * g
    if a[0] in ('kr', 'krc'):  # krc: bank centred first (kr adds the bank's mean direction = hubness, T28)
        if a[0] == 'krc':
            Bn = l2(B - B.mean(0, keepdim=True))
        out = torch.cat([torch.softmax(l2(y[i:i + 4096]) @ Bn.T / float(a[1]), 1) @ Bn for i in range(0, len(y), 4096)])
        return y + float(a[2]) * y.norm(dim=1, keepdim=True) * l2(out)


def main(names, members, folds=(1, 2, 3, 4)):
    res = {n: {} for n in names}
    for k in folds:
        ctx = load(k, members)
        for n in names:
            base, _, post = n.partition('|')
            y = method(base, ctx)
            if post:
                y = gallery_post(y, post, ctx)
            y = y.double().cpu().numpy()
            sc = official(y, ctx['img'], ctx['subj'], ctx['G'])
            res[n].update({int(s): sc[s] for s in np.unique(ctx['subj'])})
        del ctx; torch.cuda.empty_cache()
    ref = res.get('ref')
    for n in names:
        v = np.array([res[n][s] for s in sorted(res[n])])
        line = f'RESULT {n:28s} {100 * v.mean():.2f}'
        if ref is not None and n != 'ref':
            d = v - np.array([ref[s] for s in sorted(res[n])])
            line += f'  {100 * d.mean():+.2f} ({(d > 0).sum()}/{(d < 0).sum()})'
        print(line, flush=True)
    os.makedirs(f'{OUT}/results', exist_ok=True)
    for n in names:
        json.dump({str(s): x for s, x in res[n].items()}, open(f'{OUT}/results/{"+".join(members)}__{n}' + (f'__bs{BS}' if BS != 64 else '') + '.json', 'w'))  # T19 fix


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('names', nargs='+')
    ap.add_argument('--members', nargs='+', default=REF)
    ap.add_argument('--folds', nargs='+', type=int, default=None)
    ap.add_argument('--set', default='cv4')
    a = ap.parse_args()
    OUT = SETS[a.set][0]
    main(a.names, a.members, a.folds or list(SETS[a.set][1]))
