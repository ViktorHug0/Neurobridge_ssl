"""R09 which label-free statistic of a batch tracks the ENSEMBLE's best timing with the least person-dependent bias?
From the caches of nr07 (ensemble sum of scaled member outputs at shifts -5..+5) and nr05 (rmix norm at -8..+8), per fold:
- the ensemble's decoding optimum: vertex of a parabola through its top-5 (5k, r=3) at the three shifts around the best;
- (a) rmix norm (the packaged detector), (b) ensemble coherence = mean norm of the summed scaled outputs over the batch
  (large when the members agree): per-batch parabola vertex on the grid -4..+4 step 2, mean and spread, and its distance
  to the decoding optimum. A statistic whose distance to the optimum is the same in every fold needs no person-dependent
  calibration.
    nr09_detector_stats.py"""
from nc_lib import *

S = list(range(-5, 6)); gal = gallery()
def vertex(c, v):  # per row: parabola through the three points around the maximum of v (rows, len(c)), evenly spaced c
    j = np.clip(v.argmax(1), 1, len(c) - 2); r = np.arange(len(v)); l, m, h = v[r, j - 1], v[r, j], v[r, j + 1]; cur = l - 2 * m + h
    return c[j] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1) * (c[1] - c[0]), 0.0)
for k in (1, 3, 2):
    p = f'{OUT}/autolag_e2e_f{k}.npy'
    if not os.path.exists(p):
        continue
    E = np.load(p, mmap_mode='r'); n = E.shape[1]; nb = n // 64; X, subj, img = cell(FOLDS[k]); subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
    top5, coh = [], []
    for j, s in enumerate(S):
        y = torch.as_tensor(np.asarray(E[j], dtype=np.float32)); top5.append(100 * (rank(y[g4[:, :3]].mean(1), tg, gal) < 5).float().mean().item()); coh.append(y.norm(dim=1).view(nb, 64).mean(1).numpy())
    top5, coh = np.array(top5), np.stack(coh, 1); opt = vertex(np.array(S), top5[None])[0]
    N = np.load(f'{OUT}/autolag_norm_f{k}_mi_rmix_ho5k_lag5.npy')[:nb]; t = np.arange(-8, 9); g = np.abs(t) <= 4; g2 = g & (t % 2 == 0)
    a = vertex(t[g2], N[:, g2]); sel = [S.index(s) for s in (-4, -2, 0, 2, 4)]; b = vertex(np.array([-4, -2, 0, 2, 4]), coh[:, sel]); b1 = vertex(np.array(S), coh)
    print('fold %d | ensemble top-5 by shift %s | decoding optimum %+.2f' % (k, ' '.join('%+d:%.2f' % (s, v) for s, v in zip(S, top5)), opt))
    print('    (a) rmix norm:            vertex %+.2f sd %.2f -> %+.2f from the optimum' % (a.mean(), a.std(), a.mean() - opt))
    print('    (b) ensemble coherence:   vertex %+.2f sd %.2f (step 2) / %+.2f sd %.2f (step 1) -> %+.2f from the optimum | mean curve %s' % (b.mean(), b.std(), b1.mean(), b1.std(), b.mean() - opt, ' '.join('%.2f' % v for v in coh.mean(0))), flush=True)
