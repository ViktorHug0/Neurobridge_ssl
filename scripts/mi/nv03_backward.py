"""V03 backward elimination on a composition: from BASE (sub16-like six + BASE_ADD, cached outputs), drop at each step the member whose
removal costs least (mean of the 5k and 14k top-5 over the folds, all four leave-one-out triplets, r=3), print the path with the change
against the full BASE. The choice is made on the scored people: the path is optimistic by its selection noise (about +-1 %).
    BASE_ADD=recipe_ckpt,... nv03_backward.py <folds>"""
from nc_lib import *
SIX = ['mi_d6_ho5k_lag5_jit3_checkpoint_test_best.pth', 'mi_rmix_ho5k_lag5_jit3_checkpoint_ep10.pth', 'mi_sqfx_ho5k_lag5_jit3_checkpoint_test_best.pth',
       'mi_scr_ho5k_lag5_checkpoint_test_best.pth', 'mi_sqf_ho5k_lag5_checkpoint_ep32.pth', 'mi_plab_ho5k_lag5_jit3_checkpoint_test_best.pth']
BASE = SIX + [a for a in os.environ.get('BASE_ADD', '').split(',') if a]; folds = [int(a) for a in sys.argv[1].split(',')]
gals = {'5k': gallery(), '14k': gallery(big=True)}; D = []
for k in folds:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; groups = {}
    for i, kk in enumerate(zip(subj[:n].tolist(), img[:n].tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)])
    D.append(({c: scaled(torch.as_tensor(np.load(f'{OUT}/u01_f{k}_{c}.npy').astype(np.float32))) for c in BASE}, g3, torch.as_tensor(img)[g3[:, 0]]))
def hits(members):  # per gallery: per-fold top-5 rates
    o = {gn: [] for gn in gals}
    for Y, g3, tg in D:
        S = sum(Y[c] for c in members)[g3].mean(1)
        for gn, g in gals.items():
            o[gn].append((rank(S, tg, g) < 5).float().mean().item())
    return o
full = hits(BASE); cur = list(BASE)
rel = lambda h: {gn: [100 * (a / b - 1) for a, b in zip(h[gn], full[gn])] for gn in gals}
while len(cur) > 3:
    best = max(((np.mean([np.mean(v) for v in rel(h).values()]), c, h) for c in cur for h in [hits([m for m in cur if m != c])]), key=lambda t: t[0])
    cur.remove(best[1]); r = rel(best[2])
    print('%2d members after dropping %-52s 5k %+5.1f (%s) | 14k %+5.1f (%s)' % (len(cur), best[1], np.mean(r['5k']), ' / '.join('%+.1f' % v for v in r['5k']), np.mean(r['14k']), ' / '.join('%+.1f' % v for v in r['14k'])), flush=True)
print('left:', cur)
