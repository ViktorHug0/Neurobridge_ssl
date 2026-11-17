"""E20 is the weakest person weak because of bad electrodes? Per person: for each electrode, the residual variance after
predicting it from the other 31 with population regression weights (fitted on the other Alljoined people), relative to
the population's residual; and what the fitted operator of E19 does to each electrode (column norm = how much the
operator reads from it)."""
from nc_lib import *
import json
ch = json.load(open(f'{DATA}/info.json'))['ch_names']
def trials(s, n=3000):
    x = np.asarray(np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)[::22][:n]); return torch.tensor(x[np.abs(x).sum((1, 2)) > 0]).double()
C = {s: (lambda x: torch.einsum('nct,ndt->cd', x, x) / (len(x) * 120))(trials(s)) for s in range(1, 21)}
def resid_ratio(s):
    R = torch.stack([C[t] for t in C if t != s]).mean(0); out = []
    for c in range(32):
        o = [i for i in range(32) if i != c]; w = torch.linalg.solve(R[o][:, o], R[o, c])
        res = lambda K: (K[c, c] - 2 * w @ K[o, c] + w @ K[o][:, o] @ w).item()
        out.append(res(C[s]) / res(R))
    return np.array(out)
np.set_printoptions(precision=1, suppress=True, linewidth=220)
print('electrodes:', ' '.join(ch))
RR = {s: resid_ratio(s) for s in range(1, 21)}
for s in range(1, 21):
    r = RR[s]; print('sub-%02d  max ratio %5.1f (%s)  n(>3) %d  variance of the worst electrode vs population %.2f' % (s, r.max(), ch[r.argmax()], (r > 3).sum(), (C[s].diagonal() / torch.stack(list(C.values())).mean(0).diagonal())[r.argmax()]))
for k in (2,):
    ops = torch.load(f'{OUT}/e19_ops_f{k}.pt')
    for s, M in ops.items():
        col = M.norm(dim=0).numpy(); print('sub-%02d operator: weakest-read electrodes %s | residual ratio of those %s' % (s, ', '.join('%s %.2f' % (ch[i], col[i]) for i in np.argsort(col)[:4]), np.round(RR[s][np.argsort(col)[:4]], 1)))
np.save(f'{OUT}/e20_resid_ratio.npy', np.stack([RR[s] for s in range(1, 21)]))
