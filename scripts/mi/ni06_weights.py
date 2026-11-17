"""I20 does the ensemble want more weight on its strongest member, and do MSS-trained twins add as extra members?
Cached clean-proxy outputs, folds given, 5k and 14k, r=3 / r=4, exact ranks, paired against the six (lag five + LaBraM @10).
    ni06_weights.py <folds>"""
from nc_lib import *
from nc06_names import NAMES

folds = [int(a) for a in sys.argv[1].split(',')]
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float()); L = lambda k, r, e: Y(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
gals = {'5k': gallery(), '14k': gallery(big=True)}; H = {}
for k in folds:
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
    five = sum([L(k, 'd6', 12), L(k, 'rmix', 10), L(k, 'sqfx', 40)] + ([L(k, 'scr', 12), L(k, 'sqf', 32)] if k in (1, 2) else [Y(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k][3:]]))
    lab = Y(f'{OUT}/hold_f{k}_mi_plab_ho5k_lag5_long_ep10_raw.npy'); sets = {'six': five + lab}
    for w in (1.5, 2, 3):
        sets[f'six, LaBraM x{w}'] = five + w * lab
    jit = f'{OUT}/hold_f{k}_mi_plab_ho5k_lag5_jit3_ep10_raw.npy'
    if os.path.exists(jit):
        sets['seven (two LaBraMs)'] = five + lab + Y(jit); sets['seven, each LaBraM x1.5'] = five + 1.5 * (lab + Y(jit))
    mss = f'{OUT}/hold_f{k}_mi_rmix_ho5k_lag5_mssS_ep12_raw.npy'
    if os.path.exists(mss):
        sets['six + rmix-MSS @12'] = five + lab + Y(mss)
        if os.path.exists(jit):
            sets['seven + rmix-MSS @12'] = five + lab + Y(jit) + Y(mss)
    for gn, g in gals.items():
        for sn, y in sets.items():
            h = score(y, subj, img, g, hits=True)[1]
            for rr in ('r3', 'r4'):
                H.setdefault((sn, gn, rr), []).append(h[rr])
for (sn, gn, rr), v in H.items():
    if len(v) == len(folds):
        a, b = np.concatenate(v), np.concatenate(H[('six', gn, rr)])
        print('%-26s %3s %s  %.3f  vs six %+.1f +- %.1f %%  per fold: %s' % (sn, gn, rr, 100 * a.mean(), *paired(a, b), ' '.join('%+.0f' % paired(x, y)[0] for x, y in zip(v, H[('six', gn, rr)]))), flush=True)
