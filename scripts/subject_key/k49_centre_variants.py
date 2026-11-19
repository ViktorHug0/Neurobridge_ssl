"""K49: variants of the in-trial simulated re-centring (K48), three distilled heads on the frozen ATM, comb key.
Per fold the head embeddings are computed once and saved (results/.../k49/<trunk>_f<fold>.pt), then scored:
  J        number of simulated trials per query: 8, 16, 32, 64, 128 (fresh contents per query)
  shared   the same 64 contents for every query
  beta     centre = o(K) + beta (o(k) - o(K)): 0 = a fixed shift (no subject information), 1 = K48, > 1 compensates a shrunk key
  +swap    mean of the centred score and the score with the key replaced in the readout (K37)
  r        queries of r = 4 / 16 repetitions (key and centre from that query; contents unchanged)
Usage: k49_centre_variants.py <folds> [trunk]"""
import pandas as pd
from sk_lib import *

FOLDS = [int(f) for f in sys.argv[1].split(',')]
TRUNK = sys.argv[2] if len(sys.argv) > 2 else 'atm128_vc/seed3300'
TN = TRUNK.replace('/', '_')
IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))
os.makedirs(f'{SK}/k49', exist_ok=True)
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
    K = mean_key(m)
    G = [Gte[k].float() for k in (0, 2, 3)]
    anc = [t['img'][k].float().mean(0) for k in (0, 2, 3)]
    c = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs]).mean(0)
    reps = np.asarray(eeg(fold, 'test', average=False))

    def heads(v, att=None):
        pre.clear(); atm_emb(m, v, fold, att); f = torch.cat(pre) - mu.float()
        return torch.stack([nz(f @ w + b, dim=1) for w, b in W])  # (3, n, d)

    def comb(d):
        per = d.unflatten(-1, (5, 50)).mean(-2)
        return (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)

    score = lambda E, O: sum(nz(e - o + a, dim=1) @ g.T for e, o, a, g in zip(E, O, anc, G))
    r = dict(fold=fold)
    for n in (80, 16, 4):
        x = T(reps[:, :n].mean(1))
        k = K + comb(x - K)
        E = heads(x)
        sim = torch.stack([heads(k[i:i + 1] + c[torch.randperm(400, generator=torch.Generator().manual_seed(i))[:128]]) for i in range(200)])  # (200, 3, 128, d)
        sim = sim.permute(1, 0, 2, 3)  # (3, 200, 128, d)
        plain = sum(e @ g.T for e, g in zip(E, G))
        r[f'r{n} plain'], r[f'r{n} centre'] = acc(plain).item(), acc(score(E, sim[:, :, :64].mean(2))).item()
        if n < 80:
            continue
        oK = heads(K + c[:128]).mean(1)[:, None]                       # (3, 1, d): the centre of the mean training key
        sh = torch.stack([heads(k[i:i + 1] + c[:64]).mean(1) for i in range(200)]).permute(1, 0, 2)
        swap = sum(e @ g.T for e, g in zip(heads(x - k + K, atm_emb(m, x, fold)[1]), G))
        O = sim[:, :, :64].mean(2)
        for j in (8, 16, 32, 128):
            r[f'J={j}'] = acc(score(E, sim[:, :, :j].mean(2))).item()
        r['shared 64'] = acc(score(E, sh)).item()
        for beta in (0, 0.5, 0.75, 1.25, 1.5, 2):
            r[f'beta={beta}'] = acc(score(E, oK + beta * (O - oK))).item()
        r['swap'], r['centre + swap'] = acc(swap).item(), acc(score(E, O) + swap).item()
        r['cal'] = acc(score(E, E.mean(1, keepdim=True))).item()
        torch.save(dict(E=E.half(), O=O.half(), oK=oK.half(), swap=swap.half()), f'{SK}/k49/{TN}_f{fold}.pt')
    rows.append(r); print({k_: round(v, 2) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k49_centre_variants_{TN}_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
