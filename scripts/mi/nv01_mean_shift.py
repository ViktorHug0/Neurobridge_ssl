"""V01 is the soft-label gain a shift of the outputs toward the typical-image direction? Soft labels spread ~20 % of the target mass over
the batch, which pulls every output toward the mean target; the soft-label rmix has a larger common component (|mean unit output| .386
vs .317) and gains at the top of the ranking while its bulk gets worse (round 11, T2). Test on cached outputs, no training: the sub16 six
of a fold, summed, plus c x (mean output norm) x m, with m = unit mean of the TRAINED image targets (a constant that a package can ship).
Held-out people x 5,000 never-trained images, r=3; 5k and 14k galleries; change against c = 0 in %, paired bootstrap.
    nv01_mean_shift.py <folds>"""
from nc_lib import *

SIX = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
       ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')]
for al in filter(None, os.environ.get('REF', '').split(',')):  # REF=slot:recipe[:checkpoint]: another member in that slot of the REFERENCE six (and of the candidate, unless ALT replaces it)
    j, r, *ck = al.split(':'); SIX[int(j)] = (r, ck[0] if ck else SIX[int(j)][1])
T = torch.tensor(np.asarray(TARGETS)); m = nz(nz(T[np.setdiff1d(np.arange(16540), HELD)]).mean(0, keepdim=True))[0]  # typical-image direction from the trained images only
gals = {'5k': gallery(), '14k': gallery(big=True), **({'40k': gallery(big=2)} if os.environ.get('OPEN') else {})}; C = tuple(float(a) for a in os.environ.get("CS", "-0.05,-0.02,0.02,0.05,0.1").split(",")); R = {}
for k in [int(a) for a in sys.argv[1].split(',')]:
    X, subj, img = cell(FOLDS[k]); n = len(X) // 64 * 64; subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[:int(os.environ.get("REPS", 3))] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g3[:, 0]]
    if os.environ.get('TRIPLES'):  # all four leave-one-out triplets of the 4 repeats: 4 queries per (person, image), averaged per group before the bootstrap (less noise at the very top)
        g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); g3 = torch.cat([g4[:, [a for a in range(4) if a != o]] for o in range(4)]); tg = torch.as_tensor(img)[g3[:, 0]]
    Y = lambda r, c: scaled(torch.as_tensor(np.load(f"{OUT}/u01_f{k}_{r if glob.glob(f'{ROOT}/{r}/fold{k}/*/*/{c}') else r.replace('_jit3', '')}_{c}.npy").astype(np.float32)))
    S0 = sum(Y(r, c) for r, c in SIX); S = S0
    if os.environ.get('ALT'):  # ALT=slot:recipe: that member replaced in the shifted six; the reference (c = 0) stays the unchanged six
        for al in os.environ['ALT'].split(','):  # several slots: ALT=0:recipe,1:recipe[:checkpoint]
            j, r, *ck = al.split(':'); S = S - Y(*SIX[int(j)]) + Y(r, ck[0] if ck else SIX[int(j)][1])
    for al in filter(None, os.environ.get('ADD', '').split(',')):  # ADD=slot:recipe[:checkpoint][@weight]: an EXTRA member (the slot only names the default checkpoint)
        al, _, w = al.partition('@'); j, r, *ck = al.split(':'); S = S + float(w or 1) * Y(r, ck[0] if ck else SIX[int(j)][1])
    if os.environ.get('POW'):  # POW=a: each trial's output scaled by its own norm^(a-1) before the grader's repeat average (a=0: unit norm, 1: as is)
        S = S * S.norm(dim=1, keepdim=True) ** (float(os.environ['POW']) - 1)
    a = S.norm(dim=1).mean(); d = nz(S.mean(0, keepdim=True))[0] if os.environ.get('DIR') == 'b' else m  # DIR=b: the six's own image-independent mean output (from the same trials: a headroom read, not deployable as is)
    for c in (0,) + C:
        for gn, g in gals.items():
            if c == 0 and (os.environ.get('ALT') or os.environ.get('ADD')):
                R.setdefault((c, gn), []).append(rank(S0[g3].mean(1), tg, g).numpy()); R.setdefault((1e-9, gn), []).append(rank(S[g3].mean(1), tg, g).numpy()); continue
            if os.environ.get('HALF') and gn == '5k':  # HALF=lo|hi: gallery AND targets restricted to the less / more typical half of the 5k images (does the shift help inside a typicality band?)
                t = g @ m.to(g.device); keep = (t < t.median()) if os.environ['HALF'] == 'lo' else (t >= t.median()); pos = torch.full((len(g),), -1, device=g.device); pos[keep] = torch.arange(int(keep.sum()), device=g.device)
                q = pos[tg.to(g.device)].cpu(); R.setdefault((c, gn), []).append(rank((S + c * a * d)[g3][q >= 0].mean(1), q[q >= 0], g[keep]).numpy())
            else:
                R.setdefault((c, gn), []).append(rank((S + c * a * d)[g3].mean(1), tg, g).numpy())
    print('fold', k, 'six: cosine of the mean output with m %.3f, |mean output| / mean norm %.3f' % (torch.cosine_similarity(S.mean(0), m, 0), S.mean(0).norm() / a), flush=True)
grp = lambda v: np.concatenate([u.reshape(4, -1).mean(0) for u in np.split(v.astype(float), np.cumsum(SZ)[:-1])]) if os.environ.get('TRIPLES') else v.astype(float)
for gn in gals:
    b0 = 1e-9 if os.environ.get('VS') else 0  # VS=1: changes are read against the CANDIDATE without the shift (paired test of the shift on a composition), not against the reference six
    B = np.concatenate(R[(b0, gn)]); SZ = [len(u) for u in R[(b0, gn)]]; nb = len(grp(B)); bi = np.random.default_rng(0).integers(0, nb, (1000, nb)); G = (len(gals[gn]) // 2 if os.environ.get('HALF') and gn == '5k' else len(gals[gn])) - 1
    for c in ((1e-9,) if os.environ.get('ALT') or os.environ.get('ADD') else ()) + C:
        N = np.concatenate(R[(c, gn)]); o = []
        for lab, fn in (('top-1', lambda v: v < 1), ('top-2', lambda v: v < 2), ('top-3', lambda v: v < 3), ('top-5', lambda v: v < 5), ('top-10', lambda v: v < 10), ('top-20', lambda v: v < 20), ('top-100', lambda v: v < 100), ('ranking error', lambda v: v / G)):
            x, y = grp(fn(N)), grp(fn(B)); o.append('%s %+5.1f +- %.1f' % (lab, 100 * (x.mean() / y.mean() - 1), 100 * (x[bi].mean(1) / y[bi].mean(1) - 1).std()))
        print('%s c %+.2f: %s | top-5 per fold %s' % (gn, c, ' | '.join(o), ' / '.join('%+.1f' % (100 * ((u < 5).mean() / (v < 5).mean() - 1)) for u, v in zip(R[(c, gn)], R[(b0, gn)]))), flush=True)
if os.environ.get('GSIZE'):  # expected top-5 in a random sub-gallery of size G' (hypergeometric, from the exact 14k ranks): how the gain moves with the gallery size
    from scipy.stats import hypergeom
    B = np.concatenate(R[(0, '14k')]); G = len(gals['14k'])
    for c in ((1e-9,) if os.environ.get('ALT') or os.environ.get('ADD') else ()) + C:
        N = np.concatenate(R[(c, '14k')]); print('c %+.2f, top-5 change by gallery size:' % c, ' | '.join('%d: %+.1f %%' % (g, 100 * (hypergeom.cdf(4, G - 1, N, g - 1).mean() / hypergeom.cdf(4, G - 1, B, g - 1).mean() - 1)) for g in (500, 1000, 2500, 5000, 10000, G)), flush=True)

