"""S01 which components of the input, besides timing, could a test-time step adapt? Oracle headroom first.
Held-out people of a fold, never-trained images (5k gallery), the sub16 composition of that fold (frozen). Each one-parameter
family of input transforms is swept: time scale about the stimulus onset, spectral tilt / band gains / cut-offs, spatial
smoothing or sharpening over neighbouring electrodes, common-average re-reference, posterior gain, global gain, detrend, clip.
Per setting: ranks of the target (r=1 per trial, r=3 per (person, image)) and the per-trial output norm of the alignment
detector (lag-only rmix) -> <OUT>/s01_f<k>.npz. Read-out:
(1) pooled over the fold's people: does any setting beat the identity? (a cohort-level optimum away from the identity)
(2) per person, setting chosen with labels on half of the images and scored on the other half (what a perfect per-person
    estimator would give), and chosen WITHOUT labels as the setting with the largest mean detector norm of that person.
COHORT=ds004018|ds007964: the same sweep on a cohort recorded elsewhere (fold-1 models, the cohort's known time shift applied
first, gallery = the cohort's images, every 3 trials of a (person, image) averaged) = what else differs between cohorts.
    [COHORT=<name>] ns01_transform_headroom.py <fold> [members, default 6; 3 = d6 + rmix + SQF-X only]"""
import mne, glob
from nc_lib import *

k = int(sys.argv[1]); CO = os.environ.get('COHORT', ''); P = f'{OUT}/s01_f{k}{CO}.npz'; CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']
fr = torch.arange(61.); T = torch.arange(120.)
pos = {n.lower(): p for n, p in mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos'].items()}; xyz = torch.tensor(np.stack([pos[c.lower()] for c in CH])).float()
A = torch.exp(-torch.cdist(xyz, xyz) ** 2 / (2 * 0.04 ** 2)); A.fill_diagonal_(0); A = A / A.sum(1, keepdim=True)  # neighbour average, 4 cm kernel
post = torch.tensor([c[0] in 'OP' and not c.lower().startswith('fp') for c in CH])


def spec(X, g):  # gain g (61,) per 1-Hz bin
    return torch.fft.irfft(torch.fft.rfft(X, dim=-1) * g, n=120, dim=-1)


def dilate(X, s):  # content after the onset (sample 24) stretched in time by s
    p = (24 + (T - 24) / s).clamp(0, 119); lo = p.floor().long().clamp(max=118); w = p - lo
    return X[..., lo] * (1 - w) + X[..., lo + 1] * w


band = lambda a, b, g: 1 + (g - 1) * ((fr >= a) & (fr <= b)).float()
F = {'identity': lambda X: X}
F.update({f'time scale {s}': (lambda X, s=s: dilate(X, s)) for s in (.9, .95, 1.05, 1.1)})
F.update({f'tilt f^{a}': (lambda X, a=a: spec(X, (fr.clamp(min=1) / 10) ** a)) for a in (-.5, .5)})
F.update({'cut low-pass 30 Hz': lambda X: spec(X, (fr <= 30).float()), 'cut high-pass 2 Hz': lambda X: spec(X, (fr >= 2).float())})
F.update({f'band {n} x{g}': (lambda X, a=a, b=b, g=g: spec(X, band(a, b, g))) for n, a, b in (('1-7', 1, 7), ('8-13', 8, 13), ('14-30', 14, 30), ('31-60', 31, 60)) for g in (.5, 2)})
F.update({f'spatial {l:+}': (lambda X, l=l: X + l * (torch.einsum('cd,ndt->nct', A, X) - X)) for l in (-.3, -.15, .15, .3)})
F.update({f'reference {l:+}': (lambda X, l=l: X - l * X.mean(1, keepdim=True)) for l in (-.5, .5, 1)})
F.update({f'posterior x{g}': (lambda X, g=g: X * (1 + (g - 1) * post.float())[:, None]) for g in (.7, 1.4)})
F.update({f'gain x{g}': (lambda X, g=g: X * g) for g in (.5, 2)})
F.update({'detrend': lambda X: X - (X * (T - 59.5)).sum(-1, keepdim=True) / ((T - 59.5) ** 2).sum() * (T - 59.5), 'clip 5': lambda X: X.clamp(-5, 5), 'clip 10': lambda X: X.clamp(-10, 10)})
fam = lambda n: n.split()[0]

if CO:
    D = f'/nasbrain/p20fores/{CO}'; gal = nz(torch.tensor(np.load(f'{D}/targets.npz')['F'])).to(DEV); fl = sorted(glob.glob(f'{D}/export/sub-*.npz')); dd = [np.load(p) for p in fl]
    X = torch.tensor(np.concatenate([d['X'][:6000] for d in dd]).astype(np.float32))[:, :, (torch.arange(120) - {'ds004018': 5, 'ds007964': 4}[CO]).clamp(0, 119)]
    img = np.concatenate([d['img'][:6000] for d in dd]); subj = np.concatenate([[j] * len(d['img'][:6000]) for j, d in enumerate(dd)]); o = np.random.default_rng(0).permutation(len(X)); X, img, subj = X[o], img[o], subj[o]; people = sorted(set(subj.tolist()))
else:
    X, subj, img = cell(FOLDS[k]); X = torch.as_tensor(X); gal = gallery(); people = FOLDS[k]
groups = {}; G1 = len(gal) - 1
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[i:i + 3] for v in groups.values() for i in range(0, len(v) - 2, 3)] if CO else [v[:4] for v in groups.values() if len(v) >= 4])); t1 = torch.as_tensor(img); t3 = t1[g4[:, 0]]; s3 = subj[g4[:, 0].numpy()]
assert torch.allclose(dilate(X[:8], 1.0), X[:8]) and torch.allclose(spec(X[:8], torch.ones(61)), X[:8], atol=1e-4) and torch.allclose(F['detrend'](T[None, None].repeat(1, 32, 1)), torch.full((1, 32, 120), 59.5), atol=1e-3)
if not os.path.exists(P):
    M = [('mi_d6_ho5k_lag5_jit3', 'checkpoint_test_best.pth'), ('mi_rmix_ho5k_lag5_jit3', 'checkpoint_ep10.pth'), ('mi_sqfx_ho5k_lag5_jit3', 'checkpoint_test_best.pth'),
         ('mi_scr_ho5k_lag5', 'checkpoint_test_best.pth'), ('mi_sqf_ho5k_lag5', 'checkpoint_ep32.pth'), ('mi_plab_ho5k_lag5_long', 'checkpoint_ep10.pth')][:int(sys.argv[2]) if len(sys.argv) > 2 else 6]
    fs = [load(r if os.path.exists(f'{ROOT}/{r}/fold{k}') else r.replace('_jit3', ''), k, c) for r, c in M]; det = load('mi_rmix_ho5k_lag5', k, 'checkpoint_ep10.pth'); res = {}
    for name, tf in F.items():
        Xt = tf(X); y = sum(scaled(f(Xt, bs=1000)) for f in fs)
        res[name] = (rank(y, t1, gal).numpy().astype(np.int16), rank(y[g4[:, :3]].mean(1), t3, gal).numpy().astype(np.int16), det(Xt, bs=1000).norm(dim=1).numpy().astype(np.float32))
        print('done %-22s top-5 r=3 %.3f %%' % (name, 100 * (res[name][1] < 5).mean()), flush=True)
    np.savez(P, names=np.array(list(res)), r1=np.stack([v[0] for v in res.values()]), r3=np.stack([v[1] for v in res.values()]), norm=np.stack([v[2] for v in res.values()]))
d = np.load(P); names = d['names'].tolist(); r1, r3, norm = d['r1'].astype(np.int64), d['r3'].astype(np.int64), d['norm']
bi = np.random.default_rng(0).integers(0, r3.shape[1], (1000, r3.shape[1]))


def versus(r, base=None):  # r (groups,) ranks of a strategy against the identity: top-5, top-100, ranking error, in % +- bootstrap
    base = r3[0] if base is None else base
    out = []
    for fn in (lambda v: (v < 5), lambda v: (v < 100), lambda v: v / G1):
        a, b = fn(r).astype(float), fn(base).astype(float); out.append('%+5.1f +- %.1f' % (100 * (a.mean() / b.mean() - 1), 100 * (a[bi].mean(1) / b[bi].mean(1) - 1).std()))
    return ' | '.join(out)


print(CO, 'fold %d, %d groups, identity: top-5 %.3f %%, top-100 %.2f %%, ranking error %.4f' % (k, r3.shape[1], 100 * (r3[0] < 5).mean(), 100 * (r3[0] < 100).mean(), (r3[0] / G1).mean()))
print('(1) pooled, r=3 against the identity (%%): top-5 | top-100 | ranking error (negative = better) || detector norm')
for j, n in enumerate(names[1:], 1):
    print('  %-22s %s || %+.1f %%' % (n, versus(r3[j]), 100 * (norm[j].mean() / norm[0].mean() - 1)))
print('(2) one setting per person inside each family (identity included): chosen with labels on the other half of the images / by the detector norm, no labels')
half = np.arange(r3.shape[1]) % 2
for f_ in sorted(set(fam(n) for n in names[1:])):
    J = [0] + [j for j, n in enumerate(names) if fam(n) == f_]; lab, free, picks = r3[0].copy(), r3[0].copy(), []
    for p in people:
        m = s3 == p
        for h in (0, 1):
            best = J[int(np.argmin([np.log1p(r3[j][m & (half != h)]).mean() for j in J]))]; lab[m & (half == h)] = r3[best][m & (half == h)]
        jf = J[int(np.argmax([norm[j][subj == p].mean() for j in J]))]; free[m] = r3[jf][m]; picks.append(names[jf].replace(f_ + ' ', '') if jf else '-')
    print('  %-10s with labels %s || by norm %s (picks %s)' % (f_, versus(lab), versus(free), ' '.join(picks)))
