"""I01 a new member family in the lag ensemble: added as a sixth member, or swapped for each current member.
Base per fold = d6 lag @12 + rmix lag @10 + SQF-X lag + REVE-scratch + SQF (folds 1, 2: both lag-trained; fold 3: the cached
non-lag ones). Held-out people x never-trained images, 5k and 14k, exact ranks, r=3 / r=4, folds pooled, paired.
    ni01_new_member.py <arm recipe> <folds> <epochs>"""
import pandas as pd
from nc_lib import *
from nc06_names import NAMES

arm, folds, epochs = sys.argv[1], [int(a) for a in sys.argv[2].split(',')], [int(a) for a in sys.argv[3].split(',')]
gals = {'5k': gallery(), '14k': gallery(big=True)}
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float()); L = lambda k, r, e: Y(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
MEM = ('d6', 'rmix', 'SQF-X', 'REVE-scratch', 'SQF')


def arm_out(k, e):
    p = f'{OUT}/hold_f{k}_{arm}_ep{e:02d}_raw.npy'
    if not os.path.exists(p):
        ck = f'checkpoint_ep{e:02d}.pth' if glob.glob(f'{ROOT}/{arm}/fold{k}/*/*/checkpoint_ep{e:02d}.pth') else 'checkpoint_test_best.pth'
        np.save(p, load(arm, k, ck)(np.load(f'{R}/logs/holdout/f{k}/X.npy', mmap_mode='r')).numpy().astype(np.float16)); torch.cuda.empty_cache()
    return Y(p)


H = {}
for k in folds:
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
    mem = [L(k, 'd6', 12), L(k, 'rmix', 10), L(k, 'sqfx', 40)] + ([L(k, 'scr', 12), L(k, 'sqf', 32)] if k in (1, 2) else [Y(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k][3:]])
    base = sum(mem)
    for e in epochs:
        T = arm_out(k, e)
        sets = {'base (5 members)': base, 'new member alone': T, 'added as 6th member': base + T, **{f'swapped for {n}': base - m + T for n, m in zip(MEM, mem)}}
        for gn, g in gals.items():
            hb = score(base, subj, img, g, hits=True)[1]
            for sn, y in sets.items():
                h = score(y, subj, img, g, hits=True)[1]
                for rr in ('r3', 'r4'):
                    H.setdefault((e, sn, gn, rr), []).append((hb[rr], h[rr]))
        print('fold', k, 'epoch', e, flush=True)
out = []
for (e, sn, gn, rr), v in H.items():
    b, a = np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])
    out.append(dict(epoch=e, set=sn, cell=f'{gn} {rr}', score=100 * a.mean(), gain='%+.1f +- %.1f %%' % paired(a, b), per_fold=' '.join('%+.0f' % (100 * (y.mean() / x.mean() - 1)) for x, y in v)))
df = pd.DataFrame(out).sort_values(['set', 'cell', 'epoch']); df.to_csv(f'{OUT}/ni01_{arm}.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False))
