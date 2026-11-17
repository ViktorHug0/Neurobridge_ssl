"""Q11 lower-variance read of the second proxy. Top-5 at 5k is a rare event (1.5 % of the queries: +-4-5 % relative per fold),
too coarse for effects of a few percent. Same cached outputs as nq09 (r=3 queries, 5k gallery), paired against the first
recipe: hit rates at top-5 / top-20 / top-100 and the mean P(target > random image) error (1 - AUC; lower is better).
    nq11_sensitive.py <folds, e.g. 1,3> <reference recipe> <candidate recipe> [epochs, default 8,10,12]"""
from nc_lib import *

T = np.load('/nasbrain/p20fores/ds007964/targets.npz'); pool = np.setdiff1d(np.arange(16540), (T['obj'] * 10 + T['image'])[T['obj'] < 1654])
HELD2 = np.sort(np.random.default_rng(0).choice(pool, 5000, replace=False)); gal = gallery(HELD2)
folds, ref, cand = [int(k) for k in sys.argv[1].split(',')], sys.argv[2], sys.argv[3]; eps = [int(e) for e in (sys.argv[4] if len(sys.argv) > 4 else '8,10,12').split(',')]
R_ = {ref: [], cand: []}
for k in folds:
    _, subj, img = cell(FOLDS[k], HELD2); groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); t = torch.as_tensor(img)[g4[:, 0]]
    for r in R_:  # the epochs of a run are averaged in rank space: one number per query
        R_[r].append(np.mean([rank(scaled(torch.as_tensor(np.load(f'{OUT}/hold2_f{k}_{r}_ep{e:02d}_raw.npy')).float())[g4[:, :3]].mean(1), t, gal).cpu().numpy() for e in eps], 0))
a, b = np.concatenate(R_[ref]), np.concatenate(R_[cand]); i = np.random.default_rng(0).integers(0, len(a), (2000, len(a)))
print('%s vs %s, folds %s, epochs %s averaged per query (%d queries)' % (cand, ref, folds, eps, len(a)))
for name, f in (('top-5', lambda r: (r < 5).mean(-1)), ('top-20', lambda r: (r < 20).mean(-1)), ('top-100', lambda r: (r < 100).mean(-1)), ('1 - AUC', lambda r: r.mean(-1) / 4999)):
    g = 100 * (f(b[i]) / f(a[i]) - 1); print('  %-8s reference %.4f candidate %.4f | %+.1f +- %.1f %%' % (name, f(a), f(b), 100 * (f(b) / f(a) - 1), g.std()), flush=True)
