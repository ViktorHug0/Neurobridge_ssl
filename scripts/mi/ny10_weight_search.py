"""Y10 member weights chosen on some folds, confirmed on another (after the board, 18 Nov: single-trial decoding level with the best,
the gap is in the combination). sub25b's nine (fold models, cached u01 outputs), held-out people x 5,000 never-trained images, four
leave-one-out triplets per (person, image), 5k gallery (the proxy that matched the board). Coordinate ascent over each member's weight in
{0, .5, 1, 1.5, 2} (and the per-trial power), maximising pooled top-5 on the SELECT folds; then the chosen setting against uniform weights
on the CHECK folds, paired bootstrap.
    ny10_weight_search.py <select folds> <check folds>   e.g.  1,2,3 4"""
from nc_lib import *

SEL, CHK = ([int(a) for a in s.split(',')] if s else [] for s in sys.argv[1:3])
NINE = [('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'),
        ('mi_plab_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3_soft3', 'checkpoint_ep10.pth'), ('mi_plab_ho5k_lag5_long_soft3', 'checkpoint_ep10.pth'),
        ('mi_stf_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth'), ('mi_stf_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_prev_ho5k_lag5_jit3_soft3', 'checkpoint_test_best.pth')]
NAMES = ['SQF-X', 'REVE-scr', 'SQF', 'LaBraM', 'rmix soft', 'LaBraM soft', 'ST soft', 'ST', 'REVE-pre soft']; W0 = [1.0] * 9
for e in filter(None, os.environ.get('SWAP', '').split(',')):  # SWAP=slot:recipe:checkpoint,...: another fold model in that slot of the nine
    j, r, c = e.split(':'); NINE[int(j)] = (r, c)
for e in filter(None, os.environ.get('EXTRA', '').split(',')):  # EXTRA=recipe:checkpoint,...: more members, starting at weight 0
    r, c = e.split(':'); NINE.append((r, c)); NAMES.append(r.replace('mi_', '').replace('_ho5k_lag5', '')); W0.append(0.0)
GRID = [float(v) for v in os.environ.get('GRID', '0,.5,1,1.5,2').split(',')]  # candidate weights per member
W0 = [float(v) for v in os.environ['W0'].split(',')] + W0[9:] if os.environ.get('W0') else W0  # starting weights (default uniform)
M_ = len(NINE); gal = gallery(); D = {}
for k in SEL + CHK:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; groups = {}; del X
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)])
    rk = lambda r, c: f"{OUT}/u01_f{k}_{r if glob.glob(f'{ROOT}/{r}/fold{k}/*/*/{c}') else r.replace('_jit3', '')}_{c}.npy"
    D[k] = (torch.stack([scaled(torch.as_tensor(np.load(rk(r, c)).astype(np.float32))) for r, c in NINE]), g3, torch.as_tensor(img)[g3[:, 0]])


def hits(k, w, a=1.0):
    M, g3, tg = D[k]; S = torch.einsum('m,mnd->nd', torch.tensor(w, dtype=torch.float32), M); S = S * S.norm(dim=1, keepdim=True).clamp_min(1e-12) ** (a - 1)
    return (rank(S[g3].mean(1), tg, gal) < 5).numpy().astype(float).reshape(4, -1).mean(0)


score = lambda ks, w, a: np.mean(np.concatenate([hits(k, w, a) for k in ks]))
w, a = list(W0), 1.0; best = score(SEL, w, a); base = best; print('select folds %s: uniform top-5 %.4f %%' % (SEL, 100 * base), flush=True)
for it in range(3):
    moved = False
    for j in range(M_):
        for v in GRID:
            if v == w[j]:
                continue
            t = list(w); t[j] = v; s = score(SEL, t, a)
            if s > best + 1e-9:
                best, w, moved = s, t, True
    for v in ((1.,) if os.environ.get("NOPOW") else (1., 1.25, 1.5)):
        s = score(SEL, w, v)
        if s > best + 1e-9 and v != a:
            best, a, moved = s, v, True
    print('round %d: weights %s, power %.2f, select top-5 %.4f %% (%+.2f %% vs uniform)' % (it + 1, dict(zip(NAMES, w)), a, 100 * best, 100 * (best / base - 1)), flush=True)
    if not moved:
        break
for ks in ([k] for k in CHK):
    for lab, (ww, aa) in (('chosen', (w, a)), ('uniform + power', (list(W0), a))):
        x, b = hits(ks[0], ww, aa), hits(ks[0], list(W0), 1.0); bi = np.random.default_rng(0).integers(0, len(b), (2000, len(b)))
        print('check fold %d, %-16s vs uniform: top-5 %+.2f ± %.2f %%' % (ks[0], lab, 100 * (x.mean() / b.mean() - 1), 100 * (x[bi].mean(1) / b[bi].mean(1) - 1).std()), flush=True)
