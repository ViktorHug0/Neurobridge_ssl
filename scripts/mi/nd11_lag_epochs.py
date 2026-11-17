"""D13 lag arm by epoch, folds pooled, exact ranks (logs/holdout/f<k> streams): the arm vs its baseline at the SAME
epoch, alone and swapped into ens5 (other members at sub-6 budgets). 5k and 14k, top-5 r=1 / r=3 / r=4, paired bootstrap.
    nd11_lag_epochs.py <role: rmix|d6|sqfx> <arm recipe> <folds> <epochs>   (an epoch without checkpoint_epNN = the final one)
    BASE=<other arm recipe>: compare against that arm instead of the cached non-lag baseline"""
import pandas as pd
from nc_lib import *
from nc06_names import NAMES

role, arm, folds, epochs = sys.argv[1], sys.argv[2], [int(a) for a in sys.argv[3].split(',')], [int(a) for a in sys.argv[4].split(',')]
POS = {'d6': 0, 'rmix': 1, 'sqfx': 2, 'scr': 3, 'sqf': 4}[role]
gals = {'5k': gallery(), '14k': gallery(big=True)}
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float())


def base_name(k, e):  # the baseline's cached output at epoch e (no suffix = that run's last epoch)
    stem = NAMES[k][POS].split('__')[0]
    return f'{R}/logs/holdout/f{k}/{stem}__ep{e:02d}.npy' if os.path.exists(f'{R}/logs/holdout/f{k}/{stem}__ep{e:02d}.npy') else f'{R}/logs/holdout/f{k}/{stem}.npy'


def arm_out(k, e, arm=arm):
    p = f'{OUT}/hold_f{k}_{arm}_ep{e:02d}_raw.npy'
    if not os.path.exists(p):
        ck = f'checkpoint_ep{e:02d}.pth' if glob.glob(f'{ROOT}/{arm}/fold{k}/*/*/checkpoint_ep{e:02d}.pth') else 'checkpoint_test_best.pth'
        np.save(p, load(arm, k, ck)(np.load(f'{R}/logs/holdout/f{k}/X.npy', mmap_mode='r')).numpy().astype(np.float16)); torch.cuda.empty_cache()
    return Y(p)


H = {}
for k in folds:
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz'); subj, img = st['subj'], st['img']
    mem = [Y(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k]]
    rest = sum(m for i, m in enumerate(mem) if i != POS)
    for e in epochs:
        B, A = arm_out(k, e, os.environ['BASE']) if os.environ.get('BASE') else Y(base_name(k, e)), arm_out(k, e)
        for lvl, (b, a) in {'alone': (B, A), 'in ens5': (rest + B, rest + A)}.items():
            for gn, g in gals.items():
                (sb, hb), (sa, ha) = score(b, subj, img, g, hits=True), score(a, subj, img, g, hits=True)
                for rr in ('r3', 'r4'):
                    H.setdefault((e, lvl, gn, rr), []).append((hb[rr], ha[rr]))
        print(k, e, os.path.basename(base_name(k, e)), flush=True)
out = []
for (e, lvl, gn, rr), v in H.items():
    b, a = np.concatenate([x[0] for x in v]), np.concatenate([x[1] for x in v])
    out.append(dict(epoch=e, level=lvl, cell=f'{gn} {rr}', base=100 * b.mean(), arm=100 * a.mean(), gain='%+.1f +- %.1f %%' % paired(a, b),
                    per_fold=' '.join('%+.0f' % (100 * (y.mean() / x.mean() - 1)) for x, y in v)))
df = pd.DataFrame(out).sort_values(['level', 'cell', 'epoch']); df.to_csv(f'{OUT}/nd11_{arm}_{role}.csv', index=False)
pd.set_option('display.width', 200); print(df.round(3).to_string(index=False))
