"""E01 (challenge-first round) what are the hits made of? ens5 at sub-6 budgets, new people x never-trained images, cached
outputs of logs/holdout/f<k>, folds 1-3 (15 people), exact ranks, 5k gallery unless stated, r=3.
  A. repeat weighting: the grader averages raw outputs within (person, image), so the trial norm is its weight.
     y -> y |y|^g for g in -1 (every trial equal) .. 2; and members unit-normalised per trial before the sum.
  B. which images are hit: concentration over images, reliability of an image's hit rate across disjoint people/models
     (folds), and target-side predictors of it (gallery density, similarity to the gallery mean).
  C. which target directions are decoded: per principal direction of the gallery (14k), correlation between the r=3
     prediction and the target; then a linear readout fitted on folds 1+2 x half the images and tested on fold 3 x the
     other half (gallery = that half + 9.4k other THINGS images), against the plain sum on the same gallery."""
import pandas as pd
from nc_lib import *

NAMES = {1: ('reve_d6_ho5k_long_optx_things__ep10', 'rmix_reg_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         2: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         3: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_things__ep32')}
gal, galb = gallery(), gallery(big=True)
D = {}  # fold -> dict(y3 (groups, 1536) r=3 mean of the scaled sum, m3 per-member r=3 means, t, subj, hit, rank)
A = {}
for k in (1, 2, 3):
    F = f'{R}/logs/holdout/f{k}'
    st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4]))
    t = torch.as_tensor(img)[g3[:, 0]]
    Y = [scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in NAMES[k]]
    y = sum(Y)
    base = rank(y[g3].mean(1), t, gal)
    for g in (-1, -.5, .5, 1, 2):
        A.setdefault(f'trial weight |y|^{1 + g:.1f}', []).append(((rank((y * y.norm(dim=1, keepdim=True) ** g)[g3].mean(1), t, gal) < 5).numpy(), (base < 5).numpy()))
    A.setdefault('members unit-norm per trial', []).append(((rank(sum(nz(m) for m in Y)[g3].mean(1), t, gal) < 5).numpy(), (base < 5).numpy()))
    yn = y.norm(dim=1)[g3]
    D[k] = dict(y3=y[g3].mean(1), m3=[m[g3].mean(1) for m in Y], t=t, subj=subj[g3[:, 0].numpy()], rank=base, cv=(yn.std(1) / yn.mean(1)).mean().item())
    print(f'fold {k}: {len(t)} groups, top-5 {100 * (base < 5).float().mean():.3f} %, top-1 {100 * (base < 1).float().mean():.3f} %, trial-norm CV within group {D[k]["cv"]:.3f}', flush=True)
    del Y, y

print('\n== A. repeat weighting (5k, r=3, folds pooled): gain over the plain scaled sum')
for n, v in A.items():
    a, b = np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])
    print('  %-30s %.3f vs %.3f   %+.1f +- %.1f %%' % (n, 100 * a.mean(), 100 * b.mean(), *paired(a, b)))

print('\n== B. which images are hit')
H = np.zeros((3, 5000)); N = np.zeros((3, 5000)); LR = np.zeros((3, 5000))
for k in (1, 2, 3):
    np.add.at(H[k - 1], D[k]['t'].numpy(), (D[k]['rank'] < 5).numpy()); np.add.at(N[k - 1], D[k]['t'].numpy(), 1)
    np.add.at(LR[k - 1], D[k]['t'].numpy(), -np.log1p(D[k]['rank'].numpy()))
h = H.sum(0); o = np.argsort(-h)
print('  hits per image (15 people): mean %.3f; images with 0 hits %.1f %%; share of hits from the top 5 / 10 / 20 %% of images: %.0f / %.0f / %.0f %%'
      % (h.mean(), 100 * (h == 0).mean(), *[100 * h[o[:int(5000 * q)]].sum() / h.sum() for q in (.05, .1, .2)]))
p = h.sum() / N.sum(); print('  if hits were spread at random (binomial, same rate): images with 0 hits %.1f %%' % (100 * np.mean((1 - p) ** N.sum(0))))
cc = lambda a, b: np.corrcoef(a, b)[0, 1]
print('  per-image hit rate, correlation across folds (disjoint people, different models): %.3f %.3f %.3f' % (cc(H[0], H[1]), cc(H[0], H[2]), cc(H[1], H[2])))
print('  per-image mean -log(1+rank), same: %.3f %.3f %.3f' % (cc(LR[0], LR[1]), cc(LR[0], LR[2]), cc(LR[1], LR[2])))
S = gal @ gal.T; S.fill_diagonal_(-1)
dens = S.topk(10, 1).values.mean(1).cpu().numpy(); tomean = (gal @ nz(gal.mean(0, keepdim=True)).T).squeeze(1).cpu().numpy()
nrm = np.linalg.norm(np.asarray(TARGETS[HELD]), axis=1)
ease = LR.sum(0) / N.sum(0)
for n, v in (('gallery density (mean cosine to its 10 nearest)', dens), ('cosine to the gallery mean', tomean), ('raw target norm', nrm)):
    q = np.argsort(v); fifth = [100 * h[q[i * 1000:(i + 1) * 1000]].sum() / N.sum(0)[q[i * 1000:(i + 1) * 1000]].sum() for i in range(5)]
    print('  %-48s r with ease %+.3f | top-5 %% by quintile (low -> high): %s' % (n, cc(v, ease), ' '.join('%.2f' % x for x in fifth)))
names = [l.strip() for l in open(f'{R}/data/things_eeg/image_set/train_images_names.txt')] if os.path.exists(f'{R}/data/things_eeg/image_set/train_images_names.txt') else None
np.save(f'{OUT}/e01_image_hits.npy', np.stack([H.sum(0), N.sum(0), ease]))

print('\n== C. decoded target directions (new people, r=3) and a linear readout for new people')
mu = galb.mean(0); U, Sv, Vt = torch.linalg.svd(galb - mu, full_matrices=False)  # principal directions of the 14k gallery
for k in (1, 2, 3):
    P = (D[k]['y3'].to(DEV) - D[k]['y3'].to(DEV).mean(0)) @ Vt.T; T = (gal[D[k]['t'].to(DEV)] - mu) @ Vt.T
    r = ((P - P.mean(0)) * (T - T.mean(0))).mean(0) / (P.std(0) * T.std(0))
    edges = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1536]
    print(f'  fold {k}: corr(prediction, target) by principal direction of the gallery: ' + ' '.join('%d-%d: %.3f' % (a + 1, b, r[a:b].mean()) for a, b in zip(edges[:-1], edges[1:])))
    D[k]['r'] = r
print('  gallery variance share of the same bins: ' + ' '.join('%.2f' % ((Sv[a:b] ** 2).sum() / (Sv ** 2).sum()) for a, b in zip(edges[:-1], edges[1:])))
print('  r across folds (per direction, first 256): %.3f %.3f' % (cc(D[1]['r'][:256].cpu().numpy(), D[3]['r'][:256].cpu().numpy()), cc(D[2]['r'][:256].cpu().numpy(), D[3]['r'][:256].cpu().numpy())))

half = torch.as_tensor(np.random.default_rng(0).permutation(5000) < 2500)  # images of the fit half
extra = galb[5000:]
tr = [(D[k]['y3'][half[D[k]['t']]], gal[D[k]['t'][half[D[k]['t']]].to(DEV)], [m[half[D[k]['t']]] for m in D[k]['m3']]) for k in (1, 2)]
del S; torch.cuda.empty_cache()
te = ~half[D[3]['t']]
idx = torch.full((5000,), -1); idx[~half] = torch.arange(int((~half).sum()))
g_te = torch.cat([gal[(~half).to(DEV)], extra]); t_te = idx[D[3]['t'][te]]
q0 = D[3]['y3'][te]; b0 = (rank(q0, t_te, g_te) < 5).numpy()
print(f'  test: fold 3 x {int((~half).sum())} images, gallery {len(g_te)}; plain sum top-5 {100 * b0.mean():.3f} %')
rep = lambda n, q: print('    %-52s %.3f   %+.1f +- %.1f %%' % (n, 100 * (rank(q, t_te, g_te) < 5).float().mean(), *paired((rank(q, t_te, g_te) < 5).numpy(), b0)), flush=True)
Xf, Tf = torch.cat([a for a, _, _ in tr]).to(DEV), torch.cat([b for _, b, _ in tr])
Xf = Xf / Xf.norm(dim=1).mean()
for lam in (.01, .1, 1, 10):
    G = Xf.T @ Xf; W = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(1536, device=DEV), Xf.T @ Tf)
    rep(f'ridge readout of the sum, lambda {lam}', (q0.to(DEV) @ W).cpu())
    rep(f'  ... blended 50/50 with the plain sum', (nz(q0.to(DEV) @ W) + nz(q0.to(DEV))).cpu())
del Xf, G, W; torch.cuda.empty_cache()
Xc = torch.cat([torch.cat(m, 1) for _, _, m in tr]).to(DEV); Xc = Xc / Xc.norm(dim=1).mean(); qc = torch.cat([m[te] for m in D[3]['m3']], 1).to(DEV)
for lam in (.1, 1, 10):
    G = Xc.T @ Xc; W = torch.linalg.solve(G + lam * G.diagonal().mean() * torch.eye(len(G), device=DEV), Xc.T @ Tf)
    rep(f'ridge readout of the 5 members (7680 -> 1536), lambda {lam}', (qc @ W).cpu())
del Xc, G, W; torch.cuda.empty_cache()
for kk in (16, 64, 256, 512):  # keep only the first principal directions of the gallery in the prediction
    rep(f'prediction projected on the first {kk} gallery directions', (q0.to(DEV) @ Vt[:kk].T @ Vt[:kk]).cpu())
w = torch.stack([D[1]['r'], D[2]['r']]).mean(0).clamp(min=0)
for pw in (.5, 1):
    rep(f'directions weighted by r^{pw} (r from folds 1-2)', ((q0.to(DEV) @ Vt.T) * w ** pw @ Vt).cpu())
m12 = torch.cat([D[1]['y3'], D[2]['y3']]).mean(0)
for a in (.25, .5, 1):
    rep(f'minus {a} x mean prediction of folds 1-2', q0 - a * m12 * (q0.norm(dim=1).mean() / torch.cat([D[1]["y3"], D[2]["y3"]]).norm(dim=1).mean()))
sd = Sv / Sv.mean()
for pw in (-.5, -.25, .25, .5, .75, 1):
    rep(f'directions weighted by (gallery sd)^-{pw} (whitening direction if > 0)', ((q0.to(DEV) @ Vt.T) * sd ** -pw @ Vt).cpu())
for pw in (.25, .5, 1):
    rep(f'directions weighted by r^-{pw}', ((q0.to(DEV) @ Vt.T) * (w.clamp(min=1e-3) / w.mean()) ** -pw @ Vt).cpu())
for kk in (1, 4, 16):
    rep(f'first {kk} gallery directions removed', (q0.to(DEV) - q0.to(DEV) @ Vt[:kk].T @ Vt[:kk]).cpu())
P3 = q0.to(DEV) @ Vt.T; print('  prediction energy share by direction bin: ' + ' '.join('%.3f' % ((P3[:, a:b] ** 2).sum() / (P3 ** 2).sum()) for a, b in zip(edges[:-1], edges[1:])), '| along the gallery mean: %.3f' % (((q0.to(DEV) @ nz(mu[None]).T) ** 2).sum() / (q0 ** 2).sum()))
