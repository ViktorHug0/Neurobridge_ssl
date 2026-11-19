"""K48: the simulated in-trial re-centring (K46 / K47) with the three-head readout. Pure inductive.
Frozen ATM trunk + the closed-form distilled heads of K21 (teacher t6, members 0, 2, 3). For each query x:
    k = K + est(x - K)                     in-trial key (sk_lib.key_estimators)
    o_h = mean_j head_h(k + c_j)           J key-free training contents (9-subject averages of training images)
    score = sum_h cos(head_h(x) - o_h + anchor_h, images_h)     anchor_h = mean training-image embedding of the space
Reported for the model's own head and the three heads: plain, cal (true mean of the 200 queries: test data required),
and the in-trial centre for each estimator.
Usage: k48_centre_heads.py <folds> [trunk] [J=64]"""
import pandas as pd
from sk_lib import *

FOLDS = [int(f) for f in sys.argv[1].split(',')]
TRUNK = sys.argv[2] if len(sys.argv) > 2 else 'atm128_vc/seed3300'
J = int(sys.argv[3]) if len(sys.argv) > 3 else 64
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
    g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
    G = [g] + [Gte[k].float() for k in (0, 2, 3)]                      # image embeddings: own head, then the three spaces
    anc = [img_emb(m, 'train').mean(0)] + [t['img'][k].float().mean(0) for k in (0, 2, 3)]
    c = torch.stack([T(eeg(s, 'train')[IM]) - train_key(s) for s in m.train_subs]).mean(0)

    def heads(v):
        """Normalised embeddings of trials v under the own head and the three distilled heads: list of 4 (n, d)."""
        pre.clear(); e = m.embed(v, fold); f = torch.cat(pre) - mu.float()
        return [nz(e, dim=1)] + [nz(f @ w + b, dim=1) for w, b in W]

    def score(E, O):  # O: per head, (d,) or (200, d) centres, or None
        S = [nz(e - o + a, dim=1) @ gg.T if o is not None else e @ gg.T for e, o, a, gg in zip(E, O, anc, G)]
        return {'one': acc(S[0]).item(), 'three': acc(sum(S[1:])).item()}

    E = heads(x)
    r = dict(fold=fold)
    r.update({f'plain {k}': v for k, v in score(E, [None] * 4).items()})
    r.update({f'cal {k}': v for k, v in score(E, [e.mean(0) for e in E]).items()})
    for name, est in key_estimators(m, K).items():
        k = K + est(x - K)
        O = [[] for _ in range(4)]
        for i in range(200):
            for o, h in zip(O, heads(k[i:i + 1] + c[torch.randperm(400, generator=torch.Generator().manual_seed(i))[:J]])):
                o.append(h.mean(0))
        r.update({f'centre {name} {k_}': v for k_, v in score(E, [torch.stack(o) for o in O]).items()})
    rows.append(r); print({k_: round(v, 2) for k_, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k48_centre_heads_{TN}_{sys.argv[1].replace(",", "")}.csv', index=False)
print(df.mean().drop('fold').round(2).to_string())
