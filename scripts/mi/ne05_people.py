"""E05 how the hits distribute over new people: ens5 (sub-6 budgets), per held-out person, 5k, exact ranks."""
from nc_lib import *
NAMES = {1: ('reve_d6_ho5k_long_optx_things__ep10', 'rmix_reg_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         2: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_long_optx_things__ep10', 'sqf_dbn_ho5k_long_things__ep32'),
         3: ('reve_d6_ho5k_mid_optx_things__ep10', 'rmix_reg_ho5k_mid_optx_things__ep10', 'sqf_dbn_ho5k_long_optx_things', 'reve_scratch_ho5k_mid_optx_things', 'sqf_dbn_ho5k_long_things__ep32')}
gal = gallery(); res = {}
for k in (1, 2, 3):
    F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); subj, img = st['subj'], st['img']
    y = sum(scaled(torch.as_tensor(np.load(f'{F}/{m}.npy')).float()) for m in NAMES[k])
    X = np.load(f'{F}/X.npy', mmap_mode='r')
    sc, h = score(y, subj, img, gal, hits=True)
    for s in FOLDS[k]:
        x = np.asarray(X[np.nonzero(subj == s)[0][:4000]])
        res[s] = (100 * h['r3'][h['subj'] == s].mean(), 100 * h['r4'][h['subj'] == s].mean(), np.sqrt((x ** 2).mean()), np.mean(np.abs(x) >= 19.99), y[torch.as_tensor(subj == s)].norm(dim=1).mean().item())
print('person: top-5 r=3 / r=4 | trial RMS | share of clamped samples | mean output norm')
for s, v in sorted(res.items(), key=lambda kv: -kv[1][0]):
    print('  sub-%02d  %.2f / %.2f | %.2f | %.4f | %.2f' % (s, *v))
a = np.array([v[0] for v in res.values()]); print('mean %.2f, median %.2f, top 5 people carry %.0f %% of the hits; r(accuracy, output norm) %.2f' % (a.mean(), np.median(a), 100 * np.sort(a)[-5:].sum() / a.sum(), np.corrcoef(a, [v[4] for v in res.values()])[0, 1]))
