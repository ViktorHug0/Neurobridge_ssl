"""Y06 the alignment policies on the two foreign cohorts, through a real package. Their raw recordings arrive about 4-5 samples off
Alljoined's timing (ds004018, ds007964; neither trained the finals), so here the alignment has a real delay to find. Package = <package dir>
(its members, detector, bias and shifts); per 64-trial batch mixing the cohort's people, the detector's proposal (curve prominence,
correction) is computed once, and each policy decides what to apply; the members then decode the batch at that correction. r=3 groups
(3 trials of a person and image, averaged as the grader does); galleries: the cohort's own images, and the same + 14.4k THINGS images.
Top-5 % and change against no alignment, paired bootstrap over groups.
    ny06_align_cohorts.py <package dir> [trials per person, default 3000]"""
import glob, importlib.util, json, sys, types
from pathlib import Path
import numpy as np, torch

DEV = torch.device('cuda:0'); R = '/nasbrain/p20fores/Neurobridge_SSL'; nz = lambda x: x / x.norm(dim=1, keepdim=True)
stub = types.ModuleType('benchmark_utils.base_solver'); stub.CompetSolver = object; sys.modules.update({'benchmark_utils': types.ModuleType('benchmark_utils'), 'benchmark_utils.base_solver': stub})
CH = json.load(open('/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench/info.json'))['ch_names']; F = f'{R}/data/things_eeg/image_feature/dinov2-giant_track1'
T = np.load(f'{F}/image_train.npy').reshape(16540, 1536)
THINGS = nz(torch.cat([torch.tensor(T[np.sort(np.random.default_rng(0).choice(16540, 5000, replace=False))]).float(), torch.tensor(np.load(f'{F}/things_extra.npy')).float()]))
POL = [('No alignment', 9., 0, 99, False), ('Always', 0., 0, 99, False), ('Gate 0.10 (finalA)', .1, 0, 99, False), ('Dead zone 2', 0., 2, 99, False),
       ('Dead zone 3', 0., 3, 99, False), ('Gate + cap 5', .1, 0, 5, False), ('Gate + reject beyond 4', .1, 0, 4, True), ('Gate + reject beyond 3', .1, 0, 3, True),
       ('Gate + clip at 3', .1, 0, 3, False)]
out = Path(sys.argv[1]); n_per = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
spec = importlib.util.spec_from_file_location('submission', out / 'submission.py'); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
m = sub.Solver.load_model(None, {'submission_dir': out, 'ch_names': CH, 'device': DEV}); t = torch.arange(120, device=DEV)


@torch.inference_mode()
def propose(x):  # the package's align() up to the applied correction: (prominence of the detector curve, rounded correction)
    v = torch.cat([x[:, :, (t - c).clamp(0, 119)] for c in m.shifts]); n = m.detector(v).float().norm(dim=1).view(len(m.shifts), len(x)).mean(1)
    j = min(max(int(n.argmax()), 1), len(m.shifts) - 2); l, c0, h = n[j - 1].item(), n[j].item(), n[j + 1].item(); cur = l - 2 * c0 + h
    vx = m.shifts[j] + (0.5 * (l - h) / cur * (m.shifts[1] - m.shifts[0]) if cur < 0 else 0.0)
    return (n.max() / n.mean() - 1).item(), int(round(min(max(vx, m.shifts[0]), m.shifts[-1]) + m.bias))


@torch.inference_mode()
def decode(x, c):  # the package's member sum on the batch moved by c
    x = x[:, :, (t - c).clamp(0, 119)] if c else x; out = 0
    for i, (w, mem) in enumerate(zip(m.weights, m.members)):
        with torch.autocast('cuda', dtype=torch.float16, enabled=i in m.amp):
            P = mem(x).float()
        out = out + w * P / P.norm(dim=1).mean()
    return out.cpu()


for cohort in ('ds007964', 'ds004018'):
    D = f'/nasbrain/p20fores/{cohort}'; G0 = nz(torch.tensor(np.load(f'{D}/targets.npz')['F']).float()); X, img, who = [], [], []
    for p in sorted(glob.glob(f'{D}/export/sub-*.npz')):
        d = np.load(p); X.append(d['X'][:n_per].astype(np.float32)); img.append(d['img'][:n_per]); who += [p] * len(X[-1])
    X, img = torch.tensor(np.concatenate(X)), np.concatenate(img); o = np.random.default_rng(0).permutation(len(X)); X, img, who = X[o], img[o], np.array(who)[o]
    nb = len(X) // 64; X, img, who = X[:nb * 64], img[:nb * 64], who[:nb * 64]; groups = {}
    for i, kk in enumerate(zip(who.tolist(), img.tolist())):
        groups.setdefault(kk, []).append(i)
    g3 = torch.as_tensor(np.array([v[i:i + 3] for v in groups.values() for i in range(0, len(v) - 2, 3)])); tg = torch.as_tensor(img)[g3[:, 0]]
    Y = {lab: torch.empty(len(X), 1536) for lab, *_ in POL}; props = []
    for b in range(nb):
        x = X[b * 64:(b + 1) * 64].to(DEV)[:, m.order]; pr, c = propose(x); props.append(c); memo = {}
        for lab, g, z, cap, rej in POL:
            a = 0 if pr < g or abs(c) < z or (rej and abs(c) > cap) else max(-cap, min(cap, c))
            if a not in memo:
                memo[a] = decode(x, a)
            Y[lab][b * 64:(b + 1) * 64] = memo[a]
    props = np.array(props); print('%s: %d batches of 64, %d people, proposed correction: %s' % (cohort, nb, len(set(who)), ', '.join('%+d: %.0f %%' % (v, 100 * (props == v).mean()) for v in np.unique(props))), flush=True)
    for gname, G in (('own %d' % len(G0), G0), ('own + 14.4k THINGS', torch.cat([G0, THINGS]))):
        G = G.to(DEV); hits = {}
        for lab in Y:
            q = nz(Y[lab][g3].mean(1)).to(DEV); S = q @ G.T; hits[lab] = ((S > S.gather(1, tg.to(DEV)[:, None])).sum(1) < 5).float().cpu().numpy()
        b0 = hits['No alignment']; bi = np.random.default_rng(0).integers(0, len(b0), (1000, len(b0)))
        for lab, h in hits.items():
            print('  %-20s | %-24s top-5 %.3f %% | vs no alignment %+6.1f ± %.1f %%' % (gname, lab, 100 * h.mean(), 100 * (h.mean() / b0.mean() - 1), 100 * (h[bi].mean(1) / b0[bi].mean(1) - 1).std()), flush=True)
