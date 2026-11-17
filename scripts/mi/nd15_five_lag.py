"""D19 all five members lag-trained (fold 1, cached outputs, exact ranks, paired): the sub13 composition
(d6 lag @12, rmix lag @10, SQF-X lag, REVE-scratch lag @12, SQF lag @32) against the same budgets without the lag
(= sub10-like) and against the REVE@10 ens5 used so far; and sub12's composition (scratch without the lag)."""
from nc_lib import *
F = f'{R}/logs/holdout/f1'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float()); B = lambda n: Y(f'{F}/{n}.npy'); L = lambda r, e: Y(f'{OUT}/hold_f1_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
base12 = B('reve_d6_ho5k_long_optx_things__ep12') + B('rmix_reg_ho5k_long_optx_things__ep10') + B('sqf_dbn_ho5k_long_optx_things') + B('reve_scratch_ho5k_long_optx_things__ep12') + B('sqf_dbn_ho5k_long_things__ep32')
base10 = B('reve_d6_ho5k_long_optx_things__ep10') + B('rmix_reg_ho5k_long_optx_things__ep10') + B('sqf_dbn_ho5k_long_optx_things') + B('reve_scratch_ho5k_long_optx_things__ep10') + B('sqf_dbn_ho5k_long_things__ep32')
three = L('d6', 12) + L('rmix', 10) + L('sqfx', 40)
sets = {'no lag, sub-10 budgets (d6 12, rmix 10, scr 12)': base12, 'no lag, REVE@10': base10,
        'sub12 composition (d6, rmix, SQF-X, SQF lag; scratch not)': three + B('reve_scratch_ho5k_long_optx_things__ep12') + L('sqf', 32),
        'sub13 composition (all five lag)': three + L('scr', 12) + L('sqf', 32)}
for gn, gal in (('5k', gallery()), ('14k', gallery(big=True))):
    H = {n: score(y, subj, img, gal, hits=True)[1] for n, y in sets.items()}
    for n in sets:
        print('%s  %-58s r3 %.3f r4 %.3f | vs no lag same budgets: %+.1f +- %.1f / %+.1f +- %.1f %% | vs REVE@10: %+.1f +- %.1f / %+.1f +- %.1f %%' % (gn, n, 100 * H[n]['r3'].mean(), 100 * H[n]['r4'].mean(),
              *paired(H[n]['r3'], H['no lag, sub-10 budgets (d6 12, rmix 10, scr 12)']['r3']), *paired(H[n]['r4'], H['no lag, sub-10 budgets (d6 12, rmix 10, scr 12)']['r4']),
              *paired(H[n]['r3'], H['no lag, REVE@10']['r3']), *paired(H[n]['r4'], H['no lag, REVE@10']['r4'])))
