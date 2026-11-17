"""G04 the ensemble under a cohort latency offset (fold 1, held-out people x never-trained images, 5k, exact ranks,
top-5 % r=1 / r=3 / r=4). Every trial shifted by n samples; ens5 = d6 + rmix + SQF-X + REVE-scratch + SQF, with the
first three either lag-only or lag + jitter 3 (the last two are the non-jitter fold members in both, so the fold test
understates what an all-jitter ensemble keeps). Paired gain of the jitter ensemble at each shift."""
from nc_lib import *
from nd03_cross_dataset_lib import shift
k = 1; F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); X = np.load(f'{F}/X.npy', mmap_mode='r'); gal = gallery()
M = {'d6L': ('mi_d6_ho5k_lag5', 'checkpoint_test_best.pth'), 'rmixL': ('mi_rmix_ho5k_lag5', 'checkpoint_ep10.pth'), 'sqfxL': ('mi_sqfx_ho5k_lag5', 'checkpoint_test_best.pth'),
     'd6J': ('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), 'rmixJ': ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), 'sqfxJ': ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
     'scr': ('reve_scratch_ho5k_long_optx_things', 'checkpoint_ep12.pth'), 'sqf': ('sqf_dbn_ho5k_long_things', 'checkpoint_ep32.pth')}
SH = [int(a) for a in (sys.argv[1] if len(sys.argv) > 1 else '-4,-2,0,2,4').split(',')]
Y = {}
for n_, (rec, ck) in M.items():
    f = load(rec, k, ck)
    for n in SH:
        p = f'{OUT}/shift_f{k}_{rec}_{ck[11:-4]}_{n:+d}.npy'
        if not os.path.exists(p):
            np.save(p, f(X, None if n == 0 else (lambda a, b: shift(torch.as_tensor(np.asarray(X[a:b])).to(DEV), n) - torch.as_tensor(np.asarray(X[a:b])).to(DEV))).numpy().astype(np.float16))
        Y[(n_, n)] = p
    del f; torch.cuda.empty_cache(); print(n_, 'done', flush=True)
ld = lambda p: scaled(torch.as_tensor(np.load(p)).float())
print('shift | lag ens5 r1 / r3 / r4 | jitter ens5 | gain r3, r4 | d6 alone lag -> jitter (r3) | rmix | SQF-X')
for n in SH:
    y = {m: ld(Y[(m, n)]) for m in M}
    a = score(y['d6L'] + y['rmixL'] + y['sqfxL'] + y['scr'] + y['sqf'], st['subj'], st['img'], gal, hits=True)
    b = score(y['d6J'] + y['rmixJ'] + y['sqfxJ'] + y['scr'] + y['sqf'], st['subj'], st['img'], gal, hits=True)
    single = ['%.2f -> %.2f' % (score(y[m + 'L'], st['subj'], st['img'], gal)['r3'], score(y[m + 'J'], st['subj'], st['img'], gal)['r3']) for m in ('d6', 'rmix', 'sqfx')]
    print('%+d | %s | %s | %+.1f +- %.1f %%, %+.1f +- %.1f %% | %s' % (n, ' / '.join('%.3f' % v for v in a[0].values()), ' / '.join('%.3f' % v for v in b[0].values()),
          *paired(b[1]['r3'], a[1]['r3']), *paired(b[1]['r4'], a[1]['r4']), ' | '.join(single)), flush=True)
