"""I02 composition of the lag ensemble on the clean proxy (cached outputs, folds 1 and 3 pooled, exact ranks, paired
against the five-member scaled sum). Members: d6 lag @12, rmix lag @10, SQF-X lag, REVE-scratch, SQF (fold 1: both
lag-trained; fold 3: the cached non-lag ones).
  A. every subset of the five (31), ranked; the best subset of one fold scored on the other (nested).
  B. one member's weight changed at a time (0, 0.5, 1.5, 2), and the REVE trio / the SQF pair re-weighted together."""
import itertools
import pandas as pd
from nc_lib import *
from nc06_names import NAMES

gals = {'5k': gallery(), '14k': gallery(big=True)}
Y = lambda p: scaled(torch.as_tensor(np.load(p)).float()); L = lambda k, r, e: Y(f'{OUT}/hold_f{k}_mi_{r}_ho5k_lag5_ep{e:02d}_raw.npy')
MEM = ('d6', 'rmix', 'sqfx', 'scr', 'sqf'); D = {}
for k in (1, 3):
    st = np.load(f'{R}/logs/holdout/f{k}/stream.npz')
    mem = [L(k, 'd6', 12), L(k, 'rmix', 10), L(k, 'sqfx', 40)] + ([L(k, 'scr', 12), L(k, 'sqf', 32)] if k == 1 else [Y(f'{R}/logs/holdout/f{k}/{m}.npy') for m in NAMES[k][3:]])
    D[k] = (mem, st['subj'], st['img'])
CELLS = [('5k', 'r3'), ('5k', 'r4'), ('14k', 'r3'), ('14k', 'r4')]


def hits(k, w):
    mem, subj, img = D[k]; y = sum(wi * m for wi, m in zip(w, mem) if wi)
    return {(g, r): score(y, subj, img, gals[g], hits=True)[1][r] for g in gals for r in ('r3', 'r4')}


base = {k: hits(k, [1] * 5) for k in D}
def row(name, w):
    h = {k: hits(k, w) for k in D}; r = dict(set=name)
    for c in CELLS:
        a, b = np.concatenate([h[k][c] for k in D]), np.concatenate([base[k][c] for k in D])
        r[' '.join(c)] = '%.3f %+.1f +- %.1f' % (100 * a.mean(), *paired(a, b)); r['_' + ' '.join(c)] = 100 * (a.mean() / b.mean() - 1)
        for k in D:
            r[f'_f{k} ' + ' '.join(c)] = 100 * h[k][c].mean()
    r['_mean'] = np.mean([r['_' + ' '.join(c)] for c in CELLS]); return r


pd.set_option('display.width', 250)
show = lambda rows: print(pd.DataFrame(rows)[['set'] + [' '.join(c) for c in CELLS] + ['_mean']].round(2).to_string(index=False), flush=True)
print('== A. subsets (gain % over the five members; _mean = mean of the four cells)')
subs = [row(' + '.join(MEM[i] for i in s), [float(i in s) for i in range(5)]) for n in range(1, 6) for s in itertools.combinations(range(5), n)]
show(sorted(subs, key=lambda r: -r['_mean'])[:12]); show([r for r in subs if r['set'].count('+') == 3])
for a, b in ((1, 3), (3, 1)):
    best = max(subs, key=lambda r: np.mean([r[f'_f{a} ' + ' '.join(c)] / base[a][c].mean() / 100 for c in CELLS]))
    print(f'  best subset on fold {a}: {best["set"]} -> on fold {b}:', ' '.join('%s %+.1f %%' % (' '.join(c), 100 * (best[f"_f{b} " + " ".join(c)] / (100 * base[b][c].mean()) - 1)) for c in CELLS))
print('== B. weights')
rows = []
for i, n in enumerate(MEM):
    for w in (.5, 1.5, 2.):
        rows.append(row(f'{n} x{w}', [w if j == i else 1. for j in range(5)]))
for w in (1.5, 2., 3.):
    rows.append(row(f'REVE trio (d6, rmix, scr) x{w}', [w, w, 1, w, 1])); rows.append(row(f'SQF pair x{w}', [1, 1, w, 1, w]))
show(rows)
