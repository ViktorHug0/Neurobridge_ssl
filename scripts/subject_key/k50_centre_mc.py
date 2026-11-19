"""K50: the Monte-Carlo noise of the simulated centre (K49: accuracy still rises at J = 128 simulated trials per query).
Three distilled heads on the frozen ATM, comb key, 400 key-free training contents c (9-subject averages).
  fresh J     o = mean of e(k + c_j) over J contents drawn per query (K48: J = 64)
  all         J = 400: every content
  paired J    o = mean_400 e(K + c) + mean_J [e(k + c_j) - e(K + c_j)], the same J contents under the query's key and
              under the mean training key: the content cancels in the difference, the first term is computed once per model
Usage: k50_centre_mc.py <folds> [trunk]"""
import pandas as pd
from sk_lib import *

FOLDS = [int(f) for f in sys.argv[1].split(',')]
TRUNK = sys.argv[2] if len(sys.argv) > 2 else 'atm128_vc/seed3300'
TN = TRUNK.replace('/', '_')
IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))
rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte = torch.load(f'{SK}/cache/gte_t6_f{fold}.pt')['Gte']
    X = torch.load(f'{SK}/cache/trunk_{TN}_f{fold}.pt')['tr'].double(); mu = X.mean(0); X -= mu
    XtX = X.T @ X
    A = torch.linalg.inv(XtX + 0.1 * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
    W = [((A @ (X.T @ (t['eeg'][k].double() - t['eeg'][k].double().mean(0)))).float(), t['eeg'][k].float().mean(0)) for k in (0, 2, 3)]
    del X, XtX, A
    m = load(TRUNK, fold)
    pre = []
    m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.append(i[0].detach().float().cpu()))
    x, K = T(eeg(fold)), mean_key(m)
    G, anc = [Gte[k].float() for k in (0, 2, 3)], [t['img'][k].float().mean(0) for k in (0, 2, 3)]
    c = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs]).mean(0)

    def heads(v):
        pre.clear(); m.embed(v, fold); f = torch.cat(pre) - mu.float()
        return torch.stack([nz(f @ w + b, dim=1) for w, b in W])  # (3, n, d)

    score = lambda E, O: acc(sum(nz(e - o + a, dim=1) @ g.T for e, o, a, g in zip(E, O, anc, G))).item()
    per = (x - K).unflatten(-1, (5, 50)).mean(-2)
    k = K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)
    E, eK = heads(x), heads(K + c)                                     # (3, 200, d), (3, 400, d)
    sim = torch.stack([heads(k[i:i + 1] + c) for i in range(200)]).permute(1, 0, 2, 3)  # (3, 200, 400, d): content j under the key of query i
    r = dict(fold=fold, plain=acc(sum(e @ g.T for e, g in zip(E, G))).item(), cal=score(E, E.mean(1, keepdim=True)), all=score(E, sim.mean(2)))
    for J in (16, 64, 128):
        idx = torch.stack([torch.randperm(400, generator=torch.Generator().manual_seed(i))[:J] for i in range(200)])  # (200, J)
        pick = sim.gather(2, idx[None, :, :, None].expand(3, 200, J, sim.shape[-1]))
        r[f'fresh {J}'] = score(E, pick.mean(2))
        r[f'paired {J}'] = score(E, eK.mean(1)[:, None] + (pick - eK[:, idx]).mean(2))
    rows.append(r); print({k_: round(v, 2) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k50_centre_mc_{TN}_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
