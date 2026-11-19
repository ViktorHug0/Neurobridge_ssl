"""S11 does the alignment cost more on weak decoders? (19 Nov, user: be critical about the alignment. The sealed people decode about twice
worse than held-out Alljoined people; S09 priced the alignment on all 20 held-out people together.) Same caches and estimator as
ns09_align_policy.py (detector = per-trial norm of the fold's lag-only rmix at shifts -8..+8; decoding of the six at total shifts -5..+5),
but the sealed-like batches (64 trials from 11 people) are drawn inside a COHORT: all 20, the 11 weakest, the 6 weakest, the 9 strongest
(strength = the person's r=3 top-5 without alignment). Per cohort, delay and policy: per-trial and r=3 top-5 against no alignment on that
cohort's own groups, three batch seeds pooled; and the distribution of the proposed corrections at zero delay.
    ns11_align_weak.py <folds>"""
from nc_lib import *

KS = [int(a) for a in sys.argv[1].split(',')]; C = np.arange(-8, 9); D = tuple(int(a) for a in os.environ.get('D', '-2,-1,0,1,2').split(',')); POL = [('gate 0.10 (finalA)', .1, 0), ('gate + dead zone 2', .1, 2)]
fold, per, nrm, G4, E = [], [], [], {}, {}
for k in KS:
    X, subj, img = cell(FOLDS[k]); E[k] = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r'); n = E[k].shape[1]; del X; N = np.load(f'{OUT}/s06_trialnorm_f{k}.npy'); subj, img = subj[:n], img[:n]; groups = {}
    for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g4 = np.array([v[:4] for v in groups.values() if len(v) >= 4]); G4[k] = (g4, torch.as_tensor(img), subj[g4[:, 0]]); fold.append(np.full(n, k)); per.append(subj); nrm.append(N / N.mean())
fold, per, nrm = map(np.concatenate, (fold, per, nrm)); T = len(fold); gal = gallery(); off = {k: np.nonzero(fold == k)[0][0] for k in KS}


def batches(people, seed):  # 64 trials from up to 11 of the cohort's people (ns09's mix11, inside the cohort)
    r = np.random.default_rng(seed); pools = {p: list(r.permutation(np.nonzero(per == p)[0])) for p in people}; out = []
    while any(pools.values()):
        alive = [p for p in pools if pools[p]]; pick = list(r.choice(alive, min(11, len(alive)), replace=False)); b = []
        while len(b) < 64:
            cand = [p for p in pick if pools[p]]
            if not cand:
                break
            b.append(pools[cand[r.integers(len(cand))]].pop())
        if len(b) >= 32:
            out.append(np.array(b))
    return out


def estimate(b, d):  # the package's estimator (ns09): (curve prominence, total shift after correction)
    ok = (np.abs(C - d) <= 8) & ((C - d) % 2 == 0); c = C[ok]; v = nrm[b][:, ok].mean(0); j = min(max(int(v.argmax()), 1), len(c) - 2)
    l, m, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m + h
    return v.max() / v.mean() - 1, int(round(min(max(c[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.), c[0]), c[-1]) + 1.5))


def hits(s):  # per-trial total shift (T,) -> per-group hits: per-trial (mean of the 4 trials) and r=3 (first triplet), all folds
    h1, h3, pp = [], [], []
    for k in KS:
        g4, img, sp = G4[k]; sk = np.clip(s[fold == k], -5, 5).astype(int); y = torch.as_tensor(np.stack([E[k][sh + 5][np.nonzero(sk == sh)[0]] for sh in np.unique(sk)]).reshape(-1, 1536).astype(np.float32)) if False else None
        idx = np.arange(len(sk)); y = torch.empty(len(sk), 1536)
        for sh in np.unique(sk):
            m = np.nonzero(sk == sh)[0]; y[m] = torch.as_tensor(np.asarray(E[k][sh + 5][m]).astype(np.float32))
        g = torch.as_tensor(g4); h1.append((rank(y[g.ravel()], img[g.ravel()], gal) < 5).float().reshape(-1, 4).mean(1).numpy()); h3.append((rank(y[g[:, :3]].mean(1), img[g[:, 0]], gal) < 5).float().numpy()); pp.append(sp)
    return np.concatenate(h1), np.concatenate(h3), np.concatenate(pp)


B1, B3, PP = hits(np.zeros(T, np.int8)); P = np.unique(per); strength = np.array([B3[PP == p].mean() for p in P]); o = P[np.argsort(strength)]
print('people weakest -> strongest (r=3 top-5 %%): ' + ' '.join('%d:%.2f' % (p, 100 * B3[PP == p].mean()) for p in o), flush=True)
COH = [('all %d' % len(P), o), ('11 weakest', o[:11]), ('6 weakest', o[:6]), ('9 strongest', o[11:])]
COH = [c for c in COH if c[0].split()[0] in os.environ.get('COH', 'all,11,6,9').split(',')]
base = {d: hits(np.full(T, d, np.int8)) for d in D}
for name, people in COH:
    m = np.isin(PP, people); print('%s: r=3 top-5 without alignment %.3f %%, per trial %.3f %%' % (name, 100 * B3[m].mean(), 100 * B1[m].mean()), flush=True)
    print('  an uncorrected delay costs (r=3, against zero delay): ' + ' | '.join('d%+d %+.1f' % (d, 100 * (base[d][1][m].mean() / B3[m].mean() - 1)) for d in D), flush=True)
    for lab, gate, dead in POL:
        cells = []; dist = 'n/a'
        for d in D:
            a1, a3, cors = [], [], []
            for seed in range(int(os.environ.get('SEEDS', 3))):
                s = np.full(T, d, np.int8)
                for b in batches(people, seed):
                    pr, t = estimate(b, d); c = t - d
                    if pr >= gate and abs(c) >= dead:
                        s[b] = t
                    cors.append(c if pr >= gate and abs(c) >= dead else 0)
                h1, h3, _ = hits(s); a1.append(h1[m]); a3.append(h3[m])
            a1, a3 = np.concatenate(a1), np.concatenate(a3); b1, b3 = np.tile(base[d][0][m], int(os.environ.get('SEEDS', 3))), np.tile(base[d][1][m], int(os.environ.get('SEEDS', 3)))
            cells.append('d%+d: r=3 %+.1f ± %.1f, per trial %+.1f ± %.1f' % (d, *paired(a3, b3, 300), *paired(a1, b1, 300)))
            if d == 0:
                cors = np.array(cors); dist = ' '.join('%+d:%.0f%%' % (c, 100 * (cors == c).mean()) for c in range(-3, 5) if (cors == c).any())
        print('  %-20s %s\n      applied corrections at zero delay: %s' % (lab, ' | '.join(cells), dist), flush=True)
