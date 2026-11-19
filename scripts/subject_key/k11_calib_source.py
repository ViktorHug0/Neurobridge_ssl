"""K11: where may the calibration mean come from? Anchor = mean image embedding of the TRAINING images; the new
subject's embedding mean is estimated from
  self     its 200 test queries (K10)
  xhalf    the queries of the OTHER 100 test images (two random halves, each scored with the other's mean)
  n=5..100 n random queries, scored on the remaining ones (5 draws)
  causal   queries arrive in random order; query t uses the mean of queries 1..t (itself included), 5 orders;
           causal>20 = the same, accuracy over the queries after the 20th
Usage: k11_calib_source.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, a = img_emb(m), img_emb(m, 'train').mean(0)
        e = nz(m.embed(T(eeg(fold)), fold), dim=1)
        hit = lambda q, idx: (nz(q, dim=1) @ g.T).argmax(1) == idx
        allq = torch.arange(200)
        r = dict(run=RUN, fold=fold, plain=100 * hit(e, allq).float().mean().item(), self=100 * hit(e - e.mean(0) + a, allq).float().mean().item())
        gen = torch.Generator().manual_seed(fold)
        xh, caus, caus20 = [], [], []
        per_n = {n: [] for n in (5, 10, 20, 50, 100)}
        for _ in range(5):
            p = torch.randperm(200, generator=gen)
            A, B = p[:100], p[100:]
            xh.append((hit(e[A] - e[B].mean(0) + a, A).float().mean().item() + hit(e[B] - e[A].mean(0) + a, B).float().mean().item()) / 2)
            for n in per_n:
                per_n[n].append(hit(e[p[n:]] - e[p[:n]].mean(0) + a, p[n:]).float().mean().item())
            run_mean = e[p].cumsum(0) / torch.arange(1, 201)[:, None]
            h = hit(e[p] - run_mean + a, p).float()
            caus.append(h.mean().item()); caus20.append(h[20:].mean().item())
        r.update(xhalf=100 * np.mean(xh), causal=100 * np.mean(caus), **{'causal>20': 100 * np.mean(caus20)}, **{f'n={n}': 100 * np.mean(v) for n, v in per_n.items()})
        out.append(r)
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k11_calib_source.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
