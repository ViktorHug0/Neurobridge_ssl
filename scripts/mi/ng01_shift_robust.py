"""G01 robustness to a cohort latency offset: held-out people of fold k x never-trained images with every trial shifted
by n samples (n = -4..4), decoded by each listed REVE-d6 arm alone (5k, exact ranks, top-5 % r=1 / r=3).
    ng01_shift_robust.py <fold> <epoch> <arm> [<arm> ...]"""
from nc_lib import *
from nd03_cross_dataset_lib import shift
k, e, arms = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); X = np.load(f'{F}/X.npy', mmap_mode='r'); gal = gallery()
for arm in arms:
    ck = f'checkpoint_ep{e:02d}.pth' if glob.glob(f'{ROOT}/{arm}/fold{k}/*/*/checkpoint_ep{e:02d}.pth') else 'checkpoint_test_best.pth'
    f = load(arm, k, ck); row = []
    for n in range(-4, 5):
        s = score(f(X, None if n == 0 else (lambda a, b: shift(torch.as_tensor(np.asarray(X[a:b])).to(DEV), n) - torch.as_tensor(np.asarray(X[a:b])).to(DEV))), st['subj'], st['img'], gal)
        row.append('%+d: %.2f/%.2f' % (n, s['r1'], s['r3']))
    print('%-28s ep%d  %s' % (arm, e, '  '.join(row)), flush=True)
    del f; torch.cuda.empty_cache()
