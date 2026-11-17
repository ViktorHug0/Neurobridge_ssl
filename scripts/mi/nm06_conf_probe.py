"""M06 is a trial's quality readable from WHERE its output lies, not only from its norm?
M03: a regressor on scalar label-free features (norms, member agreement, input statistics) predicts the quality
q = cosine(ensemble output, target) no better than the output norm (r .44), while the true q as a repeat weight is worth
+29..69 %. Here the probe reads the full vectors: an MLP on the direction of the ensemble output (1536) or of every
member's output (5 x 1536), plus the log norms. Deployable if it works: predict() is free to rescale each trial's output.
Fitted on the new people of two folds x half the never-trained images, applied to the third fold's people x the other
half (other people, other images, other models); the trial is re-weighted by exp(b * z(predicted q)). ens5 (cached),
5k gallery, r=3, exact ranks, folds pooled, paired against the plain scaled sum.
With `within` the probe is fitted on q minus its mean over the (person, image) group: only the part of the quality that
differs between the repeats of one image, which is all a repeat weight can use.
With `margin` q is replaced by the z-score of the target's cosine among the cosines to 9.4k other images.
    nm06_conf_probe.py [ens | members] [within] [margin]"""
import torch.nn as nn
from nc_lib import *

mode = sys.argv[1] if len(sys.argv) > 1 else 'ens'
gal = gallery(); dis = gallery(big=True)[5000:]; half = torch.as_tensor(np.random.default_rng(0).permutation(5000) < 2500)  # image in the fitting half
z = lambda v: (v - v.mean()) / v.std()
D = {}
for k in (1, 2, 3):
    F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    Y = [scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in ENS5[k]]; y = sum(Y)
    src = [y] if mode == 'ens' else Y
    x = torch.cat([nz(v) for v in src] + [v.norm(dim=1, keepdim=True).log() for v in src], 1).half()
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4]))
    q = (nz(y) * gal[img].cpu()).sum(1)
    if 'margin' in sys.argv:  # the target's cosine as a z-score among 9.4k other images: what is specific to the target
        c = torch.cat([nz(y[b:b + 4096]).to(DEV) @ dis.T for b in range(0, len(y), 4096)]).cpu(); q = (q - c.mean(1)) / c.std(1); del c
    gid = torch.as_tensor(np.unique(np.stack([subj, st['img']]), axis=1, return_inverse=True)[1].ravel())
    cen = lambda v: v - (torch.zeros(int(gid.max()) + 1).index_add_(0, gid, v) / torch.bincount(gid))[gid]  # minus the (person, image) mean
    D[k] = dict(x=x, q=q, qw=cen(q), cen=cen, t=cen(q) if 'within' in sys.argv else q, y=y, img=img, fit=half[img], g3=g3[~half[img[g3[:, 0]]]])
    print('fold', k, tuple(x.shape), flush=True); del Y
hits = lambda d, w: (rank((d['y'] * w[:, None])[d['g3']].mean(1), d['img'][d['g3'][:, 0]], gal) < 5).numpy()
base = {k: hits(D[k], torch.ones(len(D[k]['q']))) for k in D}
show = lambda n, h: print('  %-44s %.3f   %+.1f +- %.1f %%' % (n, 100 * np.concatenate(list(h.values())).mean(), *paired(np.concatenate(list(h.values())), np.concatenate(list(base.values())))), flush=True)
show('plain scaled sum', base)
show('oracle: weight exp(.25 z(true q))', {k: hits(D[k], (.25 * z(D[k]['q'])).exp()) for k in D})
pred = {}
for k in D:
    tr = [j for j in D if j != k]
    X = torch.cat([D[j]['x'][D[j]['fit']] for j in tr]); Q = torch.cat([D[j]['t'][D[j]['fit']] for j in tr]); mu, sd = Q.mean(), Q.std()
    torch.manual_seed(0)
    net = nn.Sequential(nn.Linear(X.shape[1], 512), nn.GELU(), nn.Dropout(.2), nn.Linear(512, 1)).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=.05)
    te = ~D[k]['fit']; xt = D[k]['x'][te].to(DEV).float(); qt = D[k]['q'][te]
    for ep in range(6):
        net.train(); perm = torch.randperm(len(X))
        for b in range(0, len(perm), 1024):
            i = perm[b:b + 1024]
            loss = ((net(X[i].to(DEV).float())[:, 0] - ((Q[i] - mu) / sd).to(DEV)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = torch.cat([net(xt[b:b + 8192])[:, 0] for b in range(0, len(xt), 8192)]).cpu()
        full = torch.zeros(len(te)); full[te] = p; ln = D[k]['y'].norm(dim=1).log(); c = D[k]['cen']
        print(f'left-out fold {k} epoch {ep + 1}: r(predicted, true q) {np.corrcoef(p, qt)[0, 1]:.3f} | norm alone {np.corrcoef(ln[te], qt)[0, 1]:.3f} || within (person, image): predicted {np.corrcoef(c(full)[te], D[k]["qw"][te])[0, 1]:.3f} | norm {np.corrcoef(c(ln)[te], D[k]["qw"][te])[0, 1]:.3f}', flush=True)
    pred[k] = torch.zeros(len(te)); pred[k][te] = z(p)
for b in (.1, .2, .4):
    show(f'weight exp({b} z(pred q))', {k: hits(D[k], (b * pred[k]).exp()) for k in D})
for b in (.1, .2, .4):  # the predicted quality replaces the norm
    show(f'unit norm, weight exp({b} z(pred q))', {k: hits(D[k], (b * pred[k]).exp() / D[k]['y'].norm(dim=1)) for k in D})
# what a predictor of a given within-group correlation rho WOULD be worth if its errors were independent noise
g = torch.Generator().manual_seed(0)
for rho in (.25, .5, .75, 1.):
    sim = {k: rho * z(D[k]['qw']) + (1 - rho ** 2) ** .5 * torch.randn(len(D[k]['qw']), generator=g) for k in D}
    show(f'simulated within-group r {rho}: weight exp(.2 z)', {k: hits(D[k], (.2 * sim[k]).exp()) for k in D})
    show(f'   same, unit norm', {k: hits(D[k], (.2 * sim[k]).exp() / D[k]['y'].norm(dim=1)) for k in D})
