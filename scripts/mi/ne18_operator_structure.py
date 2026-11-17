"""E18 structure of the per-person oracle operators of E17 (fold 1, REVE-d6): are they alike across people, and does
another person's operator (or the mean of the others') help a new person? temporal + spatial adapters; 5k, exact ranks."""
from nc_lib import *
from nd03_cross_dataset_lib import shift
k = 1; A = torch.load(f'{OUT}/e17_person_adapters_d6_f1.pt')['ts']; P = {s: v[0].to(DEV) for s, v in A.items()}; M = {s: v[1].to(DEV) for s, v in A.items()}
I = torch.eye(32, device=DEV); TAPS = list(range(-5, 6))
D = torch.stack([(M[s] - I).flatten() for s in FOLDS[k]]); Dn = D / D.norm(dim=1, keepdim=True)
print('|M - I| per person:', np.round(D.norm(dim=1).cpu().numpy(), 2).tolist(), '| diagonal mean', np.round([M[s].diagonal().mean().item() for s in FOLDS[k]], 2).tolist())
print('cosine between people of (M - I):\n', np.round((Dn @ Dn.T).cpu().numpy(), 2))
Mbar = torch.stack(list(M.values())).mean(0); sv = torch.linalg.svdvals(D); print('singular values of the 5 operators (M - I):', np.round(sv.cpu().numpy(), 2).tolist(), '| |mean (M - I)| %.2f' % (Mbar - I).norm())
f = load('reve_d6_ho5k_long_optx_things', k, 'checkpoint_ep10.pth'); gal = gallery()
F = f'{R}/logs/holdout/f{k}'; st = np.load(f'{F}/stream.npz'); Xh = np.load(f'{F}/X.npy', mmap_mode='r'); sid = torch.as_tensor(st['subj'])
def run(ops, filt=None):  # ops: person -> 32x32
    T = torch.stack([ops.get(s, I) for s in range(21)])
    def delta(a, b):
        x = torch.as_tensor(np.asarray(Xh[a:b])).to(DEV)
        y = x if filt is None else torch.stack([sum(filt[int(s)][i][:, None] * shift(x[j], n) for i, n in enumerate(TAPS)) for j, s in enumerate(sid[a:b])])
        return torch.einsum('ncd,ndt->nct', T[sid[a:b]], y) - x
    return score(f(Xh, delta), st['subj'], st['img'], gal, hits=True)
base = run({})
rep = lambda n, r: print('%-56s %s | r3 %+.1f +- %.1f %%, r4 %+.1f +- %.1f %%' % (n, ' / '.join('%.3f' % v for v in r[0].values()), *paired(r[1]['r3'], base[1]['r3']), *paired(r[1]['r4'], base[1]['r4'])), flush=True)
rep('none', base)
rep('own spatial operator only (no temporal filter)', run(M))
rep('own temporal + spatial (E17)', run(M, P))
rep('mean of the OTHER four people`s spatial operators', run({s: torch.stack([M[t] for t in FOLDS[k] if t != s]).mean(0) for s in FOLDS[k]}))
rep('half-way to the mean of the others', run({s: .5 * (I + torch.stack([M[t] for t in FOLDS[k] if t != s]).mean(0)) for s in FOLDS[k]}))
rep('another person`s operator (next in the list)', run({s: M[FOLDS[k][(i + 1) % 5]] for i, s in enumerate(FOLDS[k])}))
rep('own operator, half strength', run({s: .5 * (I + M[s]) for s in FOLDS[k]}))
rep('own operator, symmetric part only', run({s: .5 * (M[s] + M[s].T) for s in FOLDS[k]}))
