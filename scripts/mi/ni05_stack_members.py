"""I15 do new members stack? The lag five + LaBraM @10 as the base; other candidates added as a seventh member.
Clean proxy, 5k and 14k, r=3 / r=4, exact ranks, folds pooled, paired against the lag five and against the six.
    ni05_stack_members.py <folds> <candidate recipe:epoch> [...]"""
from nc_lib import *
from nc06_names import NAMES

folds = [int(a) for a in sys.argv[1].split(',')]; cands = [a.split(':') for a in sys.argv[2:]]
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float()); L = lambda k, r, e: Y(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
gals = {'5k': gallery(), '14k': gallery(big=True)}; H = {}
for k in folds:
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
    five = sum([L(k, 'd6', 12), L(k, 'rmix', 10), L(k, 'sqfx', 40)] + ([L(k, 'scr', 12), L(k, 'sqf', 32)] if k in (1, 2) else [Y(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k][3:]]))
    six = five + Y(f'{OUT}/hold_f{k}_mi_plab_ho5k_lag5_long_ep10_raw.npy')
    sets = {'lag five': five, 'six (+ LaBraM @10)': six}
    for r, e in cands:
        c = Y(f'{OUT}/hold_f{k}_{r}_ep{int(e):02d}_raw.npy'); sets[f'five + {r}@{e}'] = five + c; sets[f'six + {r}@{e}'] = six + c
    for gn, g in gals.items():
        for sn, y in sets.items():
            h = score(y, subj, img, g, hits=True)[1]
            for rr in ('r3', 'r4'):
                H.setdefault((sn, gn, rr), []).append(h[rr])
cat = lambda v: np.concatenate(v)
for (sn, gn, rr), v in H.items():
    a, b5, b6 = cat(v), cat(H[('lag five', gn, rr)]), cat(H[('six (+ LaBraM @10)', gn, rr)])
    print('%-44s %3s %s  %.3f  vs five %+.1f +- %.1f %%  vs six %+.1f +- %.1f %%  per fold vs six: %s' % (sn, gn, rr, 100 * a.mean(), *paired(a, b5), *paired(a, b6), ' '.join('%+.0f' % paired(x, y)[0] for x, y in zip(v, H[('six (+ LaBraM @10)', gn, rr)]))), flush=True)
