"""G05 jitter twins as extra members or swaps (fold 1, cached outputs, exact ranks, paired against the sub13 composition:
d6 lag @12, rmix lag @10, SQF-X lag, REVE-scratch lag @12, SQF lag @32)."""
from nc_lib import *
F = f'{R}/logs/holdout/f1'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
L = lambda r, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f1_mi_{r}_ep{e:02d}_raw.npy')).float())
d6, rmix, rest = L('d6_ho5k_lag5', 12), L('rmix_ho5k_lag5', 10), L('sqfx_ho5k_lag5', 40) + L('scr_ho5k_lag5', 12) + L('sqf_ho5k_lag5', 32)
d6j, rmixj = L('d6_ho5k_lag5_jit3', 12), {e: L('rmix_ho5k_lag5_jit3', e) for e in (10, 12)}
sets = {'sub13 composition': d6 + rmix + rest, 'swap d6 -> d6 jitter': d6j + rmix + rest, 'swap rmix -> rmix jitter @10': d6 + rmixj[10] + rest, 'swap rmix -> rmix jitter @12': d6 + rmixj[12] + rest,
        'swap both (rmix jitter @10)': d6j + rmixj[10] + rest, 'swap both (rmix jitter @12)': d6j + rmixj[12] + rest,
        'add d6 jitter': d6 + d6j + rmix + rest, 'add rmix jitter @12': d6 + rmix + rmixj[12] + rest, 'add both': d6 + d6j + rmix + rmixj[12] + rest,
        'add both at half weight each twin': .5 * (d6 + d6j) + .5 * (rmix + rmixj[12]) + rest}
for gn, gal in (('5k', gallery()), ('14k', gallery(big=True))):
    H = {n: score(y, subj, img, gal, hits=True)[1] for n, y in sets.items()}
    for n in sets:
        print('%s  %-36s r3 %.3f r4 %.3f | %+.1f +- %.1f / %+.1f +- %.1f %%' % (gn, n, 100 * H[n]['r3'].mean(), 100 * H[n]['r4'].mean(), *paired(H[n]['r3'], H['sub13 composition']['r3']), *paired(H[n]['r4'], H['sub13 composition']['r4'])))
