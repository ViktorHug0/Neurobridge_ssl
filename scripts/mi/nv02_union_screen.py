"""V02 which cached member variants ADD to a package when put next to it? Soft-label models next to the plain six give +3 % (5k) /
+7 % (14k) top-5 where five extra plain models give +1 % (log 15 Nov 11:25): the gain needs a different error profile. Screen of every
cached output `<OUT>/u01_f<k>_<recipe>_<ckpt>.npy` present on all the folds asked: BASE (sub16-like six + BASE_ADD) + that member,
held-out people x 5,000 never-trained images, r=3 with all four leave-one-out triplets, 5k and 14k galleries, change against BASE in %
(paired bootstrap over (person, image) groups), sorted by the mean of the two top-5 changes.
    [BASE_ADD=recipe_ckpt,...] [LOO=1 | SWAPIN=recipe_ckpt] nv02_union_screen.py <folds> [substring filter]"""
from nc_lib import *

SIX = ['mi_d6_ho5k_lag5_jit3_checkpoint_test_best.pth', 'mi_rmix_ho5k_lag5_jit3_checkpoint_ep10.pth', 'mi_sqfx_ho5k_lag5_jit3_checkpoint_test_best.pth',
       'mi_scr_ho5k_lag5_checkpoint_test_best.pth', 'mi_sqf_ho5k_lag5_checkpoint_ep32.pth', 'mi_plab_ho5k_lag5_jit3_checkpoint_test_best.pth']
BASE = SIX + [a for a in os.environ.get('BASE_ADD', '').split(',') if a]
folds = [int(a) for a in sys.argv[1].split(',')]; flt = sys.argv[2] if len(sys.argv) > 2 else ''
names = lambda k: {os.path.basename(p)[len(f'u01_f{k}_'):-4] for p in glob.glob(f'{OUT}/u01_f{k}_*.npy')}
SW = os.environ.get('SWAPIN')  # SWAPIN=<cached name>: that member REPLACES each BASE member in turn (rows = the member replaced)
cands = BASE if os.environ.get('LOO') or SW else sorted(c for c in set.intersection(*[names(k) for k in folds]) - set(BASE) if flt in c)  # LOO=1: each BASE member DROPPED instead
assert all(set(BASE) <= names(k) for k in folds), [set(BASE) - names(k) for k in folds]
gals = {'5k': gallery(), '14k': gallery(big=True), **({'40k': gallery(big=2)} if os.environ.get('OPEN') else {})}; H = {}  # (candidate or 'base', gallery) -> per-fold list of per-group [top-5, top-100] hit rates
for k in folds:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; groups = {}
    for i, kk in enumerate(zip(subj[:n].tolist(), img[:n].tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)]); tg = torch.as_tensor(img)[g3[:, 0]]
    Y = lambda c: scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{c}.npy').astype(np.float32)))
    Sb = sum(Y(c) for c in BASE)
    for c in ['base'] + cands:
        S = Sb if c == 'base' else Sb - Y(c) + Y(SW) if SW else Sb - Y(c) if os.environ.get('LOO') else Sb + Y(c)
        for gn, g in gals.items():
            r = rank(S[g3].mean(1), tg, g).numpy().reshape(4, -1); H.setdefault((c, gn), []).append(np.stack([(r < 5).mean(0), (r < 100).mean(0)], 1))
    print('fold', k, 'done', flush=True)
nb = len(np.concatenate(H[('base', '5k')])); bi = np.random.default_rng(0).integers(0, nb, (1000, nb)); rows = []
for c in cands:
    o, z = [], []
    for gn in gals:
        a, b = np.concatenate(H[(c, gn)]), np.concatenate(H[('base', gn)]); d = 100 * (a.mean(0) / b.mean(0) - 1); se = 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std(0)
        o.append('%s top-5 %+5.1f +- %.1f (%s) top-100 %+5.1f +- %.1f' % (gn, d[0], se[0], ' / '.join('%+.1f' % (100 * (u[:, 0].mean() / v[:, 0].mean() - 1)) for u, v in zip(H[(c, gn)], H[('base', gn)])), d[1], se[1])); z.append(d[0])
    rows.append((np.mean(z), '%-52s %s' % (c, ' | '.join(o))))
print('BASE =', len(BASE), 'members; folds', folds); print('\n'.join(r for _, r in sorted(rows, reverse=True)))
