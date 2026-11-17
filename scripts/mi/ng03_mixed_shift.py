"""G08 which members should carry the jitter? Cached shifted outputs of ng02 (fold 1, held-out people x never-trained
images, 5k, exact ranks, top-5 % r=3 / r=4); REVE-scratch and SQF = non-jitter fold members in every composition.
Paired gain against the all-lag ensemble at the same shift."""
from nc_lib import *
k = 1; F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); gal = gallery()
M = {'d6L': 'mi_d6_ho5k_lag5_test_best', 'rmixL': 'mi_rmix_ho5k_lag5_ep10', 'sqfxL': 'mi_sqfx_ho5k_lag5_test_best', 'd6J': 'mi_d6_ho5k_lag5_jit3_test_best', 'rmixJ': 'mi_rmix_ho5k_lag5_jit3_ep10',
     'sqfxJ': 'mi_sqfx_ho5k_lag5_jit3_test_best', 'scr': 'reve_scratch_ho5k_long_optx_things_ep12', 'sqf': 'sqf_dbn_ho5k_long_things_ep32'}
SETS = {'all lag': 'd6L rmixL sqfxL', 'jitter d6 + rmix + SQF-X': 'd6J rmixJ sqfxJ', 'jitter d6 + rmix (SQF-X lag)': 'd6J rmixJ sqfxL', 'jitter d6 only': 'd6J rmixL sqfxL', 'jitter rmix only': 'd6L rmixJ sqfxL',
        'jitter SQF-X only': 'd6L rmixL sqfxJ'}
for n in (-4, -2, 0, 2, 4):
    y = {m: scaled(torch.as_tensor(np.load(f'{OUT}/shift_f{k}_{p}_{n:+d}.npy')).float()) for m, p in M.items()}
    H = {s: score(sum(y[m] for m in ms.split()) + y['scr'] + y['sqf'], st['subj'], st['img'], gal, hits=True) for s, ms in SETS.items()}
    for s in SETS:
        print('%+d | %-30s | %.3f / %.3f | %+5.1f +- %.1f %%, %+5.1f +- %.1f %%' % (n, s, H[s][0]['r3'], H[s][0]['r4'], *paired(H[s][1]['r3'], H['all lag'][1]['r3']), *paired(H[s][1]['r4'], H['all lag'][1]['r4'])), flush=True)
