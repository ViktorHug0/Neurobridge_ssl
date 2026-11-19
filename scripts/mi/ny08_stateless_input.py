"""Y08 two stateless per-batch corrections closed on thin evidence, re-read on sub25b's nine (retrospective of 18 Nov):
V = per-electrode variance of the 64-batch matched to the training population (S05: +0.2..+2.5 % in 9 of 9 cells, three members, fold 1);
B = BatchNorm of the two SQF members with a x batch statistics + (1 - a) x training statistics (S03: SQF alone +6.3 +- 3.6 %, fold 1).
Fold-k models; held-out people of the fold (5k and 14k galleries) and the two foreign cohorts (own gallery); r=3; paired bootstrap
against the unchanged nine.
    ny08_stateless_input.py <fold>"""
from nc_lib import *


def batch_stats(model, a):  # as ns03: every BatchNorm uses a x batch statistics + (1 - a) x training statistics
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            def fwd(x, m=m):
                d = [0] + list(range(2, x.dim()))
                return torch.nn.functional.batch_norm(x, a * x.mean(d) + (1 - a) * m.running_mean, a * x.var(d) + (1 - a) * m.running_var, m.weight, m.bias, False, 0., m.eps)
            m.forward = fwd


k = int(sys.argv[1])
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'),
        ('mi_plab_ho5k_lag5_long_soft3' if k != 2 else 'mi_plab_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth' if k != 2 else 'checkpoint_test_best.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
Dp = torch.as_tensor(cell(FOLDS[k % 3 + 1])[0][:40000]).pow(2).mean((0, 2)).sqrt()  # training people of the fold-k models
vmatch = lambda X: torch.cat([x * (Dp / x.pow(2).mean((0, 2)).sqrt())[:, None] for x in X.split(64)])

data = {}
X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; groups = {}
for i, kk in enumerate(zip(subj[:n].tolist(), img[:n].tolist())):
    groups.setdefault(kk, []).append(i)
g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])); data[f'fold {k}'] = (torch.as_tensor(X[:n]), img, g3, {'5k': gallery(), '14k': gallery(big=True)})
for c in ('ds007964', 'ds004018'):
    X, subj, img, gal, g3 = cohort(c); n = len(X) // 64 * 64; data[c] = (X[:n], img, g3[(g3 < n).all(1)], {'own': gal})

S = {(name, v): 0 for name in data for v in ('base', 'V', 'B.25', 'B.5', 'V+B.25')}
for r, ck in NINE:
    f = load(r, k, ck); bn = any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in f.model.modules()) and 'sqf' in r
    for name, (X, img, g3, gals) in data.items():
        Xv = vmatch(X); y0 = scaled(f(X, bs=64)); yv = scaled(f(Xv, bs=64)); S[name, 'base'] += y0; S[name, 'V'] += yv
        if bn:
            for a in (.25, .5):
                batch_stats(f.model, a); S[name, 'B' + f'{a:g}'[1:]] += scaled(f(X, bs=64))
                if a == .25: S[name, 'V+B.25'] += scaled(f(Xv, bs=64))
            batch_stats(f.model, 0.)
        else:
            S[name, 'B.25'] += y0; S[name, 'B.5'] += y0; S[name, 'V+B.25'] += yv
    del f; torch.cuda.empty_cache(); print('done', r, flush=True)
for name, (X, img, g3, gals) in data.items():
    tg = torch.as_tensor(img)[g3[:, 0]]; bi = np.random.default_rng(0).integers(0, len(g3), (500, len(g3)))
    for gn, gal in gals.items():
        b = rank(S[name, 'base'][g3].mean(1), tg, gal).numpy(); print('%-9s %-4s base top-5 %.3f %%' % (name, gn, 100 * (b < 5).mean()))
        for v in ('V', 'B.25', 'B.5', 'V+B.25'):
            a = rank(S[name, v][g3].mean(1), tg, gal).numpy()
            print('   %-7s ' % v + ' | '.join('%s %+5.1f +- %.1f' % (m, 100 * ((a < q).mean() / (b < q).mean() - 1), 100 * ((a[bi] < q).mean(1) / (b[bi] < q).mean(1) - 1).std()) for m, q in (('top-5', 5), ('top-20', 20), ('top-100', 100))), flush=True)
