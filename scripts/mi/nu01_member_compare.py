"""U01 a candidate member against a reference member of the same family, on the first clean proxy (held-out people x
5,000 never-trained images, 5k gallery, r=3), with the resolution of Q11: top-5, top-100 and ranking error, paired.
(a) each alone; (b) swapped for rmix in the sub16 composition of the fold (d6, SQF-X jitter, REVE-scratch, SQF, LaBraM
+ the member), against the same six with the reference in that slot.
Outputs are cached in <OUT>/u01_f<k>_<recipe>_<checkpoint>.npy.
    nu01_member_compare.py <fold> <reference recipe:checkpoint> <candidate recipe:checkpoint> [...]"""
from nc_lib import *

k = int(sys.argv[1]); X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g3 = torch.as_tensor(np.array([v[:3] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g3[:, 0]]


def out(spec):
    r, c = spec.split(':'); p = f'{OUT}/u01_f{k}_{r}_{c}.npy'
    if not os.path.exists(p):
        f = load(r, k, c); np.save(p, f(X, bs=1000).numpy().astype(np.float16)); del f; torch.cuda.empty_cache()
    return scaled(torch.as_tensor(np.load(p).astype(np.float32)))


def versus(r, b):
    bi = np.random.default_rng(0).integers(0, len(b), (1000, len(b))); o = []
    for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
        a, c = fn(r).astype(float), fn(b).astype(float); o.append('%+5.1f +- %.1f' % (100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
    return 'top-5 %.3f %% (%s) | top-100 %s | ranking error %s' % (100 * (r < 5).mean(), *o)


rk = lambda y: rank(y[g3].mean(1), tg, gal).numpy()
ref = out(sys.argv[2])  # the five other members of the sub16 composition, any fold (cached like the rest)
six = sum(out(s_) for s_ in ('mi_d6_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_sqfx_ho5k_lag5_jit3:checkpoint_test_best.pth', 'mi_scr_ho5k_lag5:checkpoint_test_best.pth',
                             'mi_sqf_ho5k_lag5:checkpoint_ep32.pth', 'mi_plab_ho5k_lag5_long:checkpoint_ep10.pth'))
print('fold %d | reference %s alone: top-5 %.3f %%, top-100 %.2f %%, ranking error %.4f' % (k, sys.argv[2], 100 * (rk(ref) < 5).mean(), 100 * (rk(ref) < 100).mean(), (rk(ref) / 4999.).mean()))
for spec in sys.argv[3:]:
    y = out(spec); print('  %-52s alone: %s' % (spec, versus(rk(y), rk(ref))), flush=True)
    if six is not None:
        print('  %-52s in the six: %s' % ('', versus(rk(six + y), rk(six + ref))), flush=True)
