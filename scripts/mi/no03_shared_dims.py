"""O03 how many image dimensions does the population EEG carry, and how many does the image embedding / the ensemble reach?
(a) EEG side, no model: the fold's 15 Alljoined training people are split into two groups (8 / 7); each group's mean
    response per image (every repeat, each person scaled to unit trial variance) is a (16,540 x 3,840) matrix. The SVD of
    the cross-covariance between the two groups, fitted on the 11,540 trained images, gives paired directions; their
    correlation ACROSS THE 5,000 NEVER-TRAINED IMAGES counts the image dimensions that are shared and repeatable.
    For each: how well a ridge from the DINOv2-g embedding predicts it on the never-trained images (what a linear
    image -> EEG model reaches), against its own repeatability.
(b) decoder side: the lag ensemble's outputs for the fold's NEW people, averaged over the 5 people x 4 repeats of each
    never-trained image, projected on the principal axes of the gallery embeddings: correlation with the true
    coordinate per axis = which embedding dimensions are decoded at all (with 20 trials, far more than the 3 of the score).
    no03_shared_dims.py <fold>"""
from nc_lib import *

k = int(sys.argv[1]); NEW = FOLDS[k]; KNOWN = [s for s in range(1, 21) if s not in NEW]; GA, GB = KNOWN[::2], KNOWN[1::2]
tr = np.setdiff1d(np.arange(16540), HELD)


def erp(s):
    x = torch.tensor(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(16540, 4, 3840)[:]); ok = (x.abs().sum(2) > 0).float()
    e = x.sum(1) / ok.sum(1, keepdim=True).clamp(min=1); return e / x[::20].std()


M = []
for grp in (GA, GB):
    m = 0
    for s in grp:
        m = m + erp(s) / len(grp); print('person', s, flush=True)
    M.append(m - m[tr].mean(0))
A, B = M
C = (A[tr].T.to(DEV) @ B[tr].to(DEV)) / len(tr); U, S, Vh = torch.linalg.svd(C); U, V = U[:, :300].cpu(), Vh[:300].T.cpu()
za, zb = A @ U, B @ V  # (16540, 300) scores
cc = lambda a, b: ((a - a.mean(0)) * (b - b.mean(0))).sum(0) / ((a - a.mean(0)).norm(dim=0) * (b - b.mean(0)).norm(dim=0))
r_held = cc(za[HELD], zb[HELD]); perm = torch.randperm(5000, generator=torch.Generator().manual_seed(0)); r_null = cc(za[HELD], zb[HELD][perm])
thr = 3 / 5000 ** .5
print('(a) cross-group correlation on never-trained images, first 30 paired directions:', ' '.join('%.2f' % v for v in r_held[:30]))
print('    directions with r > %.3f (3 sigma of the null): %d of 300 (null: %d); r > 0.1: %d; r > 0.2: %d' % (thr, (r_held > thr).sum(), (r_null > thr).sum(), (r_held > .1).sum(), (r_held > .2).sum()))
G = nz(torch.tensor(np.asarray(TARGETS)).float()); mu = G[tr].mean(0); F = (G - mu).to(DEV)
z = ((za + zb) / 2).to(DEV)  # the population score of every image on each shared direction
for lam in (10., 100.):
    W = torch.linalg.solve(F[tr].T @ F[tr] + lam * torch.eye(1536, device=DEV), F[tr].T @ z[tr])
    rp = cc((F[HELD] @ W).cpu(), z[HELD].cpu()); rel = (2 * r_held / (1 + r_held)).clamp_min(1e-6).sqrt()  # r(mean of both groups, true score)
    print('    ridge %g from the embedding, r(prediction, population score) on never-trained images, first 30:' % lam, ' '.join('%.2f' % v for v in rp[:30]))
    good = r_held > thr
    print('    over the %d repeatable directions: mean r %.3f against a ceiling of %.3f; directions predicted with r > 0.1: %d, > 0.2: %d' % (good.sum(), rp[good].mean(), rel[good].mean(), (rp[good] > .1).sum(), (rp[good] > .2).sum()))
# (b) decoded embedding axes
F1 = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F1}/stream.npz'); img = torch.as_tensor(st['img'])
Y = lambda r, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')).float())
y = sum(Y(r, e) for r, e in (('d6', 12), ('rmix', 10), ('sqfx', 40), ('scr', 12), ('sqf', 32)))
n = torch.bincount(img, minlength=5000).float()
for name, yy in (('all 20 trials per image', torch.zeros(5000, 1536).index_add_(0, img, y) / n[:, None]), ):
    gh = G[HELD]; pc = torch.linalg.svd(gh - gh.mean(0), full_matrices=False)[2][:300].T  # principal axes of the gallery
    rd = cc((yy - yy.mean(0)) @ pc, (gh - gh.mean(0)) @ pc)
    print('(b) ensemble, %s: r(decoded, true) per gallery principal axis, first 30:' % name, ' '.join('%.2f' % v for v in rd[:30]))
    print('    axes decoded with r > %.3f: %d of 300; r > 0.1: %d; r > 0.2: %d; r > 0.4: %d' % (thr, (rd > thr).sum(), (rd > .1).sum(), (rd > .2).sum(), (rd > .4).sum()))
    ev = ((gh - gh.mean(0)) @ pc).var(0); print('    share of the gallery variance on the axes decoded with r > 0.1: %.2f (first 300 axes hold %.2f)' % (ev[rd > .1].sum() / (gh - gh.mean(0)).var(0).sum(), ev.sum() / (gh - gh.mean(0)).var(0).sum()))
