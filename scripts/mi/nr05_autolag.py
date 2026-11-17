"""R06 can a 64-trial test batch reveal a latency offset of its cohort, without labels? (G01/G07: a +-4-sample offset in
the sealed cohort would cost about 2x, the size of the unexplained sealed gap; jitter only softens it.)
Held-out people of a fold, never-trained images, batches of 64 mixing people. The input is shifted by t samples (content
later by t). Two label-free estimators of t per batch:
(a) model confidence: mean output norm of a frozen member over the batch, as a function of the candidate correction;
(b) template: lag of the best cross-correlation between the batch-mean ERP and the mean key of the TRAINING people.
Reported: the per-batch estimate of the total shift when the truth is 0 (bias, spread, share within +-1), and the
top-5 (5k, r=3) of the member at each shift for reference.
    nr05_autolag.py <recipe> <fold> <checkpoint>"""
from nc_lib import *

recipe, k, ck = sys.argv[1], int(sys.argv[2]), sys.argv[3]; f = load(recipe, k, ck); gal = gallery(); X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); tt = torch.arange(120)
groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
nb = len(X) // 64; S = list(range(-8, 9)); N = np.zeros((nb, len(S))); top5 = {}
for j, s in enumerate(S):
    y = f(X[:, :, (tt - s).clamp(0, 119)], bs=1000).cpu().float(); N[:, j] = y.norm(dim=1)[:nb * 64].view(nb, 64).mean(1).numpy()
    top5[s] = 100 * (rank(scaled(y)[g4[:, :3]].mean(1), tg, gal) < 5).float().mean().item()
print(recipe, ck, 'fold', k, '| top-5 % (5k, r=3) by input shift: ' + ' '.join('%+d: %.2f' % (s, top5[s]) for s in S))
print('mean output norm by shift (all batches): ' + ' '.join('%+d: %.3f' % (s, N[:, j].mean()) for j, s in enumerate(S)))
np.save(f'{OUT}/autolag_norm_f{k}_{recipe}.npy', N)
for name, win in (('candidates -6..+6', 6), ('candidates -4..+4', 4)):
    sel = [j for j, s in enumerate(S) if abs(s) <= win]; est = np.array(S)[sel][N[:, sel].argmax(1)]
    # parabolic refinement around the argmax over the smoothed curve
    print('(a) confidence, %s: estimate mean %+.2f, sd %.2f | exact 0: %.0f %% | within +-1: %.0f %% | within +-2: %.0f %%' % (name, est.mean(), est.std(), 100 * (est == 0).mean(), 100 * (np.abs(est) <= 1).mean(), 100 * (np.abs(est) <= 2).mean()))
# (b) template: mean key of the training people of this fold (Alljoined ids only), cross-correlated with the batch-mean ERP
train = [s for s in range(1, 21) if s not in FOLDS[k]]; tmpl = torch.stack([key(s) for s in train]).mean(0); tmpl = tmpl - tmpl.mean(1, keepdim=True)
bm = X[:nb * 64].view(nb, 64, 32, 120).mean(1); bm = bm - bm.mean(2, keepdim=True)
def xc(a, lag):  # correlation between the batch mean delayed by -lag and the template, on the overlap
    if lag >= 0:
        u, v = a[:, :, lag:], tmpl[:, :120 - lag]
    else:
        u, v = a[:, :, :lag], tmpl[:, -lag:]
    return (u * v).flatten(1).sum(1) / (u.flatten(1).norm(dim=1) * v.norm())
C = torch.stack([xc(bm, l) for l in range(-8, 9)], 1).numpy(); est = np.arange(-8, 9)[C.argmax(1)]
print('(b) template, truth 0: estimate mean %+.2f, sd %.2f | exact 0: %.0f %% | within +-1: %.0f %% | within +-2: %.0f %% | mean peak correlation %.3f' % (est.mean(), est.std(), 100 * (est == 0).mean(), 100 * (np.abs(est) <= 1).mean(), 100 * (np.abs(est) <= 2).mean(), C.max(1).mean()))
for d in (-4, 4):  # the same after delaying the data by d: the estimate should move to d
    bd = X[:nb * 64][:, :, (tt - d).clamp(0, 119)].view(nb, 64, 32, 120).mean(1); bd = bd - bd.mean(2, keepdim=True); e = np.arange(-8, 9)[torch.stack([xc(bd, l) for l in range(-8, 9)], 1).numpy().argmax(1)]
    print('    data delayed by %+d: estimate mean %+.2f, sd %.2f, within +-1 of the truth %.0f %%' % (d, e.mean(), e.std(), 100 * (np.abs(e - d) <= 1).mean()))
