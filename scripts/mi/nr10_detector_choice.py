"""R10 which detector (or set of detectors) gives the alignment estimate with the least person-dependent calibration?
Inputs: the per-batch norm curves of nr05 (members x folds 1-3, shifts -8..+8). Estimate per batch = vertex of the parabola
through the three points around the maximum on the grid -8..+8 step 2 (the packaged estimator); for a set of members the
curves are summed after dividing each by its mean over the grid (members have different output scales).
Per detector set: the estimate's mean and per-batch spread in each fold, its distance to the ensemble's decoding optimum
(nr09: +0.90 / +0.68 / +0.63 in folds 1 / 3 / 2), and the range of that distance over the folds = what a single
calibration constant cannot remove.
    nr10_detector_choice.py"""
import itertools, os
import numpy as np

OUT = '/nasbrain/p20fores/Neurobridge_SSL/results/things_eeg/mi/nc'; OPT = {1: 0.90, 3: 0.68, 2: 0.63}; t = np.arange(-8, 9); g = t % 2 == 0
M = {'rmix': 'mi_rmix_ho5k_lag5', 'd6': 'mi_d6_ho5k_lag5', 'sqfx': 'mi_sqfx_ho5k_lag5', 'scr': 'mi_scr_ho5k_lag5', 'sqf': 'mi_sqf_ho5k_lag5', 'labram': 'mi_plab_ho5k_lag5_long', 'rmix jitter': 'mi_rmix_ho5k_lag5_jit3'}


def vertex(c, v):
    j = np.clip(v.argmax(1), 1, len(c) - 2); r = np.arange(len(v)); l, m, h = v[r, j - 1], v[r, j], v[r, j + 1]; cur = l - 2 * m + h
    return c[j] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1) * (c[1] - c[0]), 0.0)


N = {(n, k): np.load(p) for n, r in M.items() for k in OPT if os.path.exists(p := f'{OUT}/autolag_norm_f{k}_{r}.npy')}
rows = []
for size in (1, 2, 3):
    for names in itertools.combinations(M, size):
        folds = [k for k in OPT if all((n, k) in N for n in names)]
        if len(folds) < 3 and size > 1:
            continue
        d, sd = [], []
        for k in folds:
            nb = min(len(N[(n, k)]) for n in names); v = sum(N[(n, k)][:nb, g] / N[(n, k)][:nb, g].mean(1, keepdims=True) for n in names); e = vertex(t[g], v); d.append(e.mean() - OPT[k]); sd.append(e.std())
        rows.append((max(d) - min(d) if len(d) == 3 else np.nan, ' + '.join(names), folds, d, sd))
for rng, name, folds, d, sd in sorted(rows, key=lambda r: (np.isnan(r[0]), r[0])):
    print('%-28s folds %s | distance to the optimum %s | range %.2f | per-batch sd %s' % (name, folds, ' '.join('%+.2f' % x for x in d), rng, ' '.join('%.2f' % x for x in sd)))
