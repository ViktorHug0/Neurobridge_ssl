"""E16 checkpoints of one run as extra members: each REVE-family member = the mean output of its epochs 8, 10, 12
(or 10, 12) instead of one epoch. Fold 1 (all three epochs cached), 5k, exact ranks, paired vs ens5 REVE@10 and @12."""
from nc_lib import *
F = f'{R}/logs/holdout/f1'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']; gal = gallery()
Y = lambda n: scaled(torch.as_tensor(np.load(f'{F}/{n}.npy')).float())
rv = {m: {e: Y(f'{r}__ep{e:02d}') for e in (8, 10, 12)} for m, r in (('d6', 'reve_d6_ho5k_long_optx_things'), ('rmix', 'rmix_reg_ho5k_long_optx_things'), ('scr', 'reve_scratch_ho5k_long_optx_things'))}
sq = Y('sqf_dbn_ho5k_long_optx_things') + Y('sqf_dbn_ho5k_long_things__ep32')
sets = {'ens5 REVE@10': sum(rv[m][10] for m in rv) + sq, 'ens5 REVE@12': sum(rv[m][12] for m in rv) + sq,
        'REVE = mean of ep 10, 12': sum((rv[m][10] + rv[m][12]) / 2 for m in rv) + sq, 'REVE = mean of ep 8, 10, 12': sum(sum(rv[m].values()) / 3 for m in rv) + sq}
H = {n: score(y, subj, img, gal, hits=True)[1] for n, y in sets.items()}
for n in sets:
    print('%-30s r3 %.3f r4 %.3f | vs @10: %+.1f +- %.1f / %+.1f +- %.1f %% | vs @12: %+.1f +- %.1f / %+.1f +- %.1f %%' % (n, 100 * H[n]['r3'].mean(), 100 * H[n]['r4'].mean(),
          *paired(H[n]['r3'], H['ens5 REVE@10']['r3']), *paired(H[n]['r4'], H['ens5 REVE@10']['r4']), *paired(H[n]['r3'], H['ens5 REVE@12']['r3']), *paired(H[n]['r4'], H['ens5 REVE@12']['r4'])))
