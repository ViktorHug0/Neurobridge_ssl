"""S08 a supervised latency reader instead of the norm heuristic. S06 / S07: every label-free criterion read off the
members (norm, agreement) puts some atypical people 3-5 samples from their true timing. Here a small network is TRAINED
to read the delay: single trials of the fold's Alljoined training people, delayed by a known d in -8..+8, classified
into the 17 delays; a batch estimate = vertex of the mean log-probability curve over its trials. The network sees the
central 104 samples only (a crop moved by -d), so a delay never leaves an edge artefact to read.
Test on the held-out people (never seen): estimate per mixed 64-batch and per one-person 64-batch at true delays
0, +-2, +-4, against the norm criterion at delay 0 (`s06_trialnorm_f<k>.npy`), then decoding with the correction
(cached sub16-composition outputs, r=3, 5k): total shift = d - round(estimate - TARGET) (env TARGET, default 0.75).
Last: the raw exports of ds004018 / ds007964 (true offsets about +5 and +4 to +5): estimate per mixed batch and per person.
    ns08_latency_reader.py <fold> [epochs, default 12]
    ns08_latency_reader.py 0     # final reader on the 20 Alljoined people -> <OUT>/s08_reader_final.pt (TorchScript: (B, 32, 120) -> 17 logits)"""
from nc_lib import *
from torch import nn

k = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 12; TARGET = float(os.environ.get('TARGET', .75)); rng = np.random.default_rng(0); C = np.arange(-8, 9); W = 104; S = list(range(-5, 6)); torch.manual_seed(0)


def trials(s, n=8000):
    x = np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120); x = np.asarray(x[np.sort(rng.choice(len(x), 2 * n, replace=False))]); return x[np.abs(x).sum((1, 2)) > 0][:n]


def crop(x, d):  # x (n, 32, 120), d (n,) long: the trial delayed by d, seen through the central window
    return x.gather(2, (8 - d[:, None] + torch.arange(W, device=x.device)[None])[:, None, :].expand(-1, 32, -1))


P = f'{OUT}/s08_reader_f{k}.pt'
net = nn.Sequential(nn.Conv1d(32, 64, 9, padding=4), nn.BatchNorm1d(64), nn.ELU(), nn.Conv1d(64, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ELU(),
                    nn.Conv1d(64, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ELU(), nn.Dropout(.5), nn.Flatten(), nn.Linear(64 * 26, 17)).to(DEV)
if os.path.exists(P):
    net.load_state_dict(torch.load(P))
else:
    Xtr = torch.tensor(np.concatenate([trials(s) for s in range(1, 21) if not k or s not in FOLDS[k]]).astype(np.float32)); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EP * (len(Xtr) // 512))
    for ep in range(EP):
        net.train(); tot = acc = 0
        for b in torch.randperm(len(Xtr)).split(512)[:-1]:
            d = torch.randint(-8, 9, (len(b),), device=DEV); lg = net(crop(Xtr[b].to(DEV), d)); loss = nn.functional.cross_entropy(lg, d + 8)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item(); acc += ((lg.argmax(1) - 8 - d).abs() <= 1).float().mean().item()
        print('epoch %d loss %.3f, single trial within +-1 sample %.1f %%' % (ep + 1, tot / (len(Xtr) // 512), 100 * acc / (len(Xtr) // 512)), flush=True)
    torch.save(net.state_dict(), P)
net.eval()
if not k:
    class Reader(nn.Module):
        def __init__(self, net):
            super().__init__(); self.net = net

        def forward(self, x):
            return self.net(x[:, :, 8:8 + W])
    x = torch.tensor(np.load(f'{DATA}/sub-01/test.npy', mmap_mode='r')[:64, 0, 0]).to(DEV); tr = torch.jit.trace(Reader(net).eval(), x)
    for b in (64, 37, 1):
        assert (tr(x[:b]) - net(x[:b, :, 8:8 + W])).abs().max().item() < 1e-4
    tr.save(f'{OUT}/s08_reader_final.pt'); print('saved', f'{OUT}/s08_reader_final.pt', 'batch estimate on 64 test trials of sub-01: %+d' % (torch.log_softmax(tr(x), 1).mean(0).argmax().item() - 8)); sys.exit()
E = np.load(f'{OUT}/autolag_e2e_f{k}.npy', mmap_mode='r'); n = E.shape[1]; X, subj, img = cell(FOLDS[k]); X, subj, img = torch.as_tensor(X[:n]), subj[:n], img[:n]; gal = gallery(); groups = {}
for i, kk in enumerate(zip(subj.tolist(), img.tolist())):
    groups.setdefault(kk, []).append(i)
g4 = torch.as_tensor(np.array([v[:4] for v in groups.values() if len(v) >= 4])); tg = torch.as_tensor(img)[g4[:, 0]]
per = [np.nonzero(subj == q)[0] for q in FOLDS[k]]; order = np.arange(n); B = {'mixed': [order[i:i + 64] for i in range(0, n, 64)], 'one person': [q[i:i + 64] for q in per for i in range(0, len(q) - 31, 64)]}
who = np.array([subj[b[0]] for b in B['one person']])


def vertex(v):  # (batches, 17) -> fractional estimate
    j = np.clip(v.argmax(1), 1, len(C) - 2); r = np.arange(len(v)); l, m, h = v[r, j - 1], v[r, j], v[r, j + 1]; cur = l - 2 * m + h
    return np.clip(C[j] + np.where(cur < 0, 0.5 * (l - h) / np.where(cur < 0, cur, -1), 0.0), C[0], C[-1])


def ranks(tb):
    y = torch.empty(n, 1536)
    for s in np.unique(tb):
        rows = np.nonzero(tb == s)[0]; y[rows] = torch.as_tensor(np.asarray(E[S.index(s)], dtype=np.float32)[rows])
    return rank(y[g4[:, :3]].mean(1), tg, gal).numpy()


def report(tag, T, d):  # T (n, 17): per-trial curve over candidate delays (log-probability or norm)
    est = {bn: vertex(np.stack([T[b].mean(0) for b in bs])) for bn, bs in B.items()}; m, o = est['mixed'], est['one person']; pm = [o[who == q].mean() for q in FOLDS[k]]
    print('fold %d delay %+d %-7s mixed: estimate %+.2f sd %.2f | one person: per person %s (sd across people %.2f, per-batch sd within a person %.2f)' % (
        k, d, tag, m.mean(), m.std(), ' '.join('%+.1f' % v for v in pm), np.std(pm), np.mean([o[who == q].std() for q in FOLDS[k]])), flush=True)
    return est


N0 = np.load(f'{OUT}/s06_trialnorm_f{k}.npy'); report('norm', N0, 0)
for d in (0, -2, 2, -4, 4):
    with torch.no_grad():
        L = torch.cat([torch.log_softmax(net(crop(X[b:b + 2000].to(DEV), torch.full((len(X[b:b + 2000]),), d, device=DEV))), 1).cpu() for b in range(0, n, 2000)]).numpy()
    est = report('reader', L, d); base = ranks(np.full(n, d)); bi = np.random.default_rng(0).integers(0, len(base), (1000, len(base)))
    for bn, bs in B.items():
        tb = np.full(n, d)
        for b, e in zip(bs, est[bn]):
            tb[b] = d - int(round(e - TARGET))
        r = ranks(np.clip(tb, S[0], S[-1])); out = []
        for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
            a, c = fn(r).astype(float), fn(base).astype(float); out.append('%+5.1f +- %.1f' % (100 * (a.mean() / c.mean() - 1), 100 * (a[bi].mean(1) / c[bi].mean(1) - 1).std()))
        print('      target %.2f decoding with the reader, %-10s residual shift %+.2f sd %.2f | vs no alignment: top-5 %s | top-100 %s | ranking error %s' % (TARGET, bn, tb.mean(), tb.std(), *out), flush=True)
        # consensus: act only when the member-norm detector (packaged grid, +0.9 = its mean distance to the reader on new people) agrees with the reader within 2.5 samples
        ok = (np.abs(C - d) <= 8) & ((C - d) % 2 == 0); c2 = C[ok]; tb = np.full(n, d); acted = 0
        for b_, e in zip(bs, est[bn]):
            v = N0[b_][:, ok].mean(0); j = min(max(int(v.argmax()), 1), len(c2) - 2); l, m_, h = v[j - 1], v[j], v[j + 1]; cur = l - 2 * m_ + h; pk = c2[j] + (0.5 * (l - h) / cur * 2 if cur < 0 else 0.)
            if abs((d - pk - 0.9) - e) <= 2.5:
                tb[b_] = d - int(round(e - TARGET)); acted += len(b_)
        r = ranks(np.clip(tb, S[0], S[-1])); out = []
        for fn in (lambda v: v < 5, lambda v: v < 100, lambda v: v / 4999.):
            a_, c_ = fn(r).astype(float), fn(base).astype(float); out.append('%+5.1f +- %.1f' % (100 * (a_.mean() / c_.mean() - 1), 100 * (a_[bi].mean(1) / c_[bi].mean(1) - 1).std()))
        print('      target %.2f CONSENSUS (agreeing batches %.0f %%), %-10s residual shift %+.2f sd %.2f | vs no alignment: top-5 %s | top-100 %s | ranking error %s' % (TARGET, 100 * acted / n, bn, tb.mean(), tb.std(), *out), flush=True)
import glob
for co in ('ds004018', 'ds007964'):
    fl = sorted(glob.glob(f'/nasbrain/p20fores/{co}/export/sub-*.npz')); Xc = torch.tensor(np.concatenate([np.load(p)['X'][:4000] for p in fl]).astype(np.float32)); wc = np.repeat(np.arange(len(fl)), 4000)[:len(Xc)] if all(len(np.load(p)['X']) >= 4000 for p in fl) else None
    with torch.no_grad():
        L = torch.cat([torch.log_softmax(net(crop(Xc[b:b + 2000].to(DEV), torch.zeros(len(Xc[b:b + 2000]), dtype=torch.long, device=DEV))), 1).cpu() for b in range(0, len(Xc), 2000)]).numpy()
    o = np.random.default_rng(0).permutation(len(Xc)); mixed = vertex(np.stack([L[o[i:i + 64]].mean(0) for i in range(0, len(o) - 63, 64)])); pp = vertex(np.stack([L[i:i + 4000].mean(0) for i in range(0, len(Xc) - 3999, 4000)]))
    print('%s raw: mixed 64-batches estimate %+.2f sd %.2f | per person (4,000 trials each) %s' % (co, mixed.mean(), mixed.std(), ' '.join('%+.1f' % v for v in pp)), flush=True)
