"""P02 what makes a gallery image a hub or a hit, and what a NEW REGION of image space costs (no retraining).
Cached lag five, new people, never-trained images, 5k gallery, r=3, three folds.
Per gallery image: wins (top-1 count over the queries of other targets), hits (top-5 when it is the target),
typicality (cosine to the gallery mean), norm of the raw embedding, novelty = cosine to its nearest TRAINED image
(11,540), and to its nearest trained image of ANOTHER concept.
(a) hubs: are they the same images across folds (disjoint people, different models)? what predicts them?
(b) hit rate by novelty quintile, inside typicality terciles (typical images are both dense and close to training images).
    np02_hubs_novelty.py"""
from nc_lib import *
from nc06_names import NAMES

Y = lambda k, r, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')).float())
raw = torch.tensor(np.asarray(TARGETS)).float(); E = nz(raw).to(DEV); G = E[HELD]; tr = np.setdiff1d(np.arange(16540), HELD)
S = G @ E[tr].T; nov = S.max(1).values.cpu().numpy()
other = torch.as_tensor(HELD // 10)[:, None] != torch.as_tensor(tr // 10)[None]; nov_o = S.masked_fill(~other.to(DEV), -1).max(1).values.cpu().numpy()
typ = (G @ nz(G.mean(0, keepdim=True)).T)[:, 0].cpu().numpy(); norm = raw[HELD].norm(dim=1).numpy()
wins, hits, ranks = [], [], []
for k in (1, 2, 3):
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], torch.as_tensor(st['img'])
    y = sum([Y(k, 'd6', 12), Y(k, 'rmix', 10), Y(k, 'sqfx', 40)] + ([Y(k, 'scr', 12), Y(k, 'sqf', 32)] if k in (1, 2) else [scaled(torch.as_tensor(np.load(f'{R}/logs/holdout/f{k}/{m}.npy')).float()) for m in NAMES[k][3:]]))
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); t = img[g4[:, 0]]; q = nz(y[g4[:, :3]].mean(1))
    w, h, rk = torch.zeros(5000), torch.zeros(5000), []
    for b in range(0, len(q), 4096):
        Sc = q[b:b + 4096].to(DEV) @ G.T; tt = t[b:b + 4096].to(DEV); r_ = (Sc > Sc.gather(1, tt[:, None])).sum(1)
        top = Sc.topk(2).indices; win = torch.where(top[:, 0] == tt, top[:, 1], top[:, 0])
        w += torch.bincount(win, minlength=5000).float().cpu(); h += torch.bincount(tt[r_ < 5], minlength=5000).float().cpu(); rk.append(torch.stack([tt, r_]).cpu())
    wins.append(w.numpy()); hits.append(h.numpy()); ranks.append(torch.cat(rk, 1))
c = lambda a, b: np.corrcoef(a, b)[0, 1]; sp = lambda a, b: c(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))
W, H = sum(wins), sum(hits)
print('(a) hubs. wins per image across folds, rank correlation: f1-f2 %.2f, f1-f3 %.2f, f2-f3 %.2f | hits per image: %.2f %.2f %.2f' % (sp(wins[0], wins[1]), sp(wins[0], wins[2]), sp(wins[1], wins[2]), sp(hits[0], hits[1]), sp(hits[0], hits[2]), sp(hits[1], hits[2])))
print('    rank correlation of wins with: typicality %.2f, embedding norm %.2f, novelty %.2f, hits %.2f | of hits with: typicality %.2f, norm %.2f, novelty %.2f' % (sp(W, typ), sp(W, norm), sp(W, -nov), sp(W, H), sp(H, typ), sp(H, norm), sp(H, -nov)))
hub = np.argsort(-W)[:50]; print('    the 50 hubs (1 %%): %.1f %% of the wins, %.1f %% of the hits (they are the target in 1 %% of the queries); the 500 most-winning: %.1f %% of the wins, %.1f %% of the hits' % (100 * W[hub].sum() / W.sum(), 100 * H[hub].sum() / H.sum(), 100 * W[np.argsort(-W)[:500]].sum() / W.sum(), 100 * H[np.argsort(-W)[:500]].sum() / H.sum()))
print('    typicality %.3f for hubs vs %.3f overall (sd %.3f); novelty (cos to nearest trained image) %.3f vs %.3f (sd %.3f)' % (typ[hub].mean(), typ.mean(), typ.std(), nov[hub].mean(), nov.mean(), nov.std()))
R_ = torch.cat(ranks, 1); tt, hit = R_[0].numpy(), (R_[1] < 5).numpy(); auc = 1 - R_[1].numpy() / 4999
print('(b) novelty. cos to nearest trained image: quintile edges', np.quantile(nov, [.2, .4, .6, .8]).round(3), '| nearest trained image is of the same concept for %.0f %% of the held images; r(novelty, typicality) %.2f' % (100 * (nov > nov_o + 1e-6).mean(), c(nov, typ)))
for name, v in (('nearest trained image', nov), ('nearest trained image of another concept', nov_o)):
    qn = np.searchsorted(np.quantile(v, [.2, .4, .6, .8]), v)
    print('    top-5 %% by quintile of cosine to the %s (far -> near): %s | mean P(target > random): %s' % (name, ' '.join('%.2f' % (100 * hit[qn[tt] == i].mean()) for i in range(5)), ' '.join('%.3f' % auc[qn[tt] == i].mean() for i in range(5))))
    for j, tn in enumerate(('atypical third', 'middle third', 'typical third')):
        m = np.searchsorted(np.quantile(typ, [1 / 3, 2 / 3]), typ) == j; qj = np.searchsorted(np.quantile(v[m], [1 / 3, 2 / 3]), v)
        print('        %-14s far / mid / near thirds: %s | P(target > random) %s' % (tn, ' '.join('%.2f' % (100 * hit[m[tt] & (qj[tt] == i)].mean()) for i in range(3)), ' '.join('%.3f' % auc[m[tt] & (qj[tt] == i)].mean() for i in range(3))), flush=True)
