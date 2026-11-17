"""R02 does the ensemble grow with more copies of its architectures? (cached outputs, first proxy, fold 1)
The neutral-data runs of the night (MSS, donors, ds004018: same recipe, another seed and a data change worth nothing)
stand in for second seeds of rmix, REVE-d6 and SQF-X. Base = the lag five + LaBraM (sub16's six). Top-5 % at 5k r=3 / r=4,
14k r=3, and the denser reads at 5k r=3 (top-100, 1 - AUC), paired against the six.
    nr02_more_members.py"""
from nc_lib import *

Y = lambda n, e: scaled(torch.as_tensor(np.load(f'{OUT}/hold_f1_mi_{n}_ep{e:02d}_raw.npy')).float())
st = np.load(f'{R}/logs/holdout/f1/stream.npz'); subj, img = st['subj'], st['img']; gals = {'5k': gallery(), '14k': gallery(big=True)}
six = [Y('d6_ho5k_lag5', 12), Y('rmix_ho5k_lag5', 10), Y('sqfx_ho5k_lag5', 40), Y('scr_ho5k_lag5', 12), Y('sqf_ho5k_lag5', 32), Y('plab_ho5k_lag5_long', 10)]
X = {'rmix b': Y('rmix_ho5k_lag5_mssS', 12), 'rmix c': Y('rmix_ho5k_lag5_don', 10), 'rmix d': Y('rmix_ho5k_lag5_g18', 10), 'rmix e': Y('rmix_ho5k_lag5_mssB', 12),
     'd6 b': Y('d6_ho5k_lag5_mssS', 10), 'sqfx b': Y('sqfx_ho5k_lag5_mssS', 40)}
groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]; bi = np.random.default_rng(0).integers(0, len(g4), (2000, len(g4)))


def read(members):
    y = sum(members); r3 = rank(y[g4[:, :3]].mean(1), tg, gals['5k']).cpu().numpy(); r4 = rank(y[g4].mean(1), tg, gals['5k']).cpu().numpy(); b3 = rank(y[g4[:, :3]].mean(1), tg, gals['14k']).cpu().numpy()
    return {'5k r3': r3 < 5, '5k r4': r4 < 5, '14k r3': b3 < 5, 'top-100': r3 < 100, '1-AUC': r3 / 4999}


base = read(six); print('six: ' + ' | '.join('%s %.4f' % (n, v.mean() * (1 if n == '1-AUC' else 100)) for n, v in base.items()))
for name, add in (('+ rmix b', ['rmix b']), ('+ d6 b', ['d6 b']), ('+ sqfx b', ['sqfx b']), ('+ rmix b, d6 b, sqfx b (9)', ['rmix b', 'd6 b', 'sqfx b']),
                  ('+ rmix b, c (8)', ['rmix b', 'rmix c']), ('+ rmix b, c, d, e (10)', ['rmix b', 'rmix c', 'rmix d', 'rmix e']), ('+ all six extras (12)', list(X))):
    v = read(six + [X[a] for a in add])
    print('%-28s' % name + ' | '.join('%s %+.1f +- %.1f %%' % (n, 100 * (v[n].mean() / base[n].mean() - 1), 100 * (v[n][bi].mean(1) / base[n][bi].mean(1) - 1).std()) for n in base), flush=True)
