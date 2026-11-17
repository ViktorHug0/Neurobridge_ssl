"""S03 BatchNorm statistics from the test batch (the two members with BatchNorm: SQF-X, SQF). E15: the statistics matter
(the THINGS path costs new Alljoined people 43 %), and a cohort recorded elsewhere has its own. Every BatchNorm uses
a x (statistics of the current 64-trial batch) + (1 - a) x (training statistics); a = 0 is the deployed model.
Held-out people of the fold (cost when the cohort is ours) and the two external cohorts (fold-1 models, known shift applied).
Single member, r=3: top-5 %, top-100 %, ranking error.
    ns03_test_time_bn.py <fold> [alphas, default 0,.1,.25,.5,1]"""
from nc_lib import *

k = int(sys.argv[1]); AL = [float(a) for a in (sys.argv[2] if len(sys.argv) > 2 else '0,.1,.25,.5,1').split(',')]


def batch_stats(model, a):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            def fwd(x, m=m):
                d = [0] + list(range(2, x.dim()))
                return torch.nn.functional.batch_norm(x, a * x.mean(d) + (1 - a) * m.running_mean, a * x.var(d) + (1 - a) * m.running_var, m.weight, m.bias, False, 0., m.eps)
            m.forward = fwd


data = {}
X, subj, img = cell(FOLDS[k]); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
data[f'fold {k}'] = (torch.as_tensor(X), img, gallery(), torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])))
for c in ('ds004018', 'ds007964'):
    X, subj, img, gal, g3 = cohort(c); data[c] = (X, img, gal, g3)
for recipe, ck in (('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth')):
    f = load(recipe, k if os.path.exists(f'{ROOT}/{recipe}/fold{k}') else 1, ck); assert sum(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in f.model.modules()) > 0
    for name, (X, img, gal, g3) in data.items():
        base = None
        for a in AL:
            batch_stats(f.model, a); y = f(X, bs=64); r = rank(y[g3].mean(1), torch.as_tensor(img)[g3[:, 0]], gal).numpy(); base = r if base is None else base
            bi = np.random.default_rng(0).integers(0, len(r), (500, len(r))); out = []
            for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / (len(gal) - 1.)):
                p, q = fn(r).astype(float), fn(base).astype(float); out.append('%.3f (%+5.1f +- %.1f %%)' % (p.mean() * (100 if p.max() <= 1 and fn(np.array([0]))[0] == 1 else 1), 100 * (p.mean() / q.mean() - 1), 100 * (p[bi].mean(1) / q[bi].mean(1) - 1).std()))
            print('%-24s %-9s a %.2f | top-5 %s | top-100 %s | ranking error %s' % (recipe, name, a, *out), flush=True)
