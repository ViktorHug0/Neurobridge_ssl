"""K42: an in-trial key estimate that does not use the 200 ms period of the paradigm. K36's comb hard-codes it; here the
estimators are fitted on the TRAINING subjects only and know nothing of the stream:
  span        least-squares projection of x - K on the 9 training key deviations (the key as a mixture of known keys)
  time        k = K + (x - K) A_t, Wiener filter over time shared by the electrodes (K36's `wiener`: free to find the period)
  time_local  the same with the covariances tapered to lags below 100 ms (Bartlett): it cannot use the repetition of the
              key every 200 ms, as in a paradigm with one image per trial
  space       k = K + A_s (x - K), Wiener filter over the electrodes shared by the time points
  space+local both
Covariances: key deviations of the training subjects around K against the content of a training trial at 80 repetitions
(signal part from the agreement of two 2-repetition halves, noise part scaled 1 / 80).
Frozen ATM, attention of the raw trial, readout fed x - k + K; r2 = share of |m_s - K|^2 explained.
Usage: k42_key_no_paradigm.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))
psd = lambda S: (lambda w, V: V * w.clamp(min=0) @ V.T)(*torch.linalg.eigh(S))
gain = lambda Sd, Sc: (Sd @ torch.linalg.inv(Sd + Sc + 1e-6 * Sd.trace() / len(Sd) * torch.eye(len(Sd), dtype=torch.double))).float()
cov = lambda a, b, ax: torch.einsum('nct,ncu->tu' if ax == 't' else 'nct,ndt->cd', a, b) / len(a)


def filters(m, K, reps=80):
    S = {ax: dict(d=0, s=0, h=0) for ax in 'tc'}
    for s in m.train_subs:
        d = (train_key(s) - K[0]).double()[None]
        h = T(eeg(s, 'train', average=False)[IM]).double()
        h1, h2 = h[:, :2].mean(1) - train_key(s), h[:, 2:].mean(1) - train_key(s)
        for ax in 'tc':
            S[ax]['d'] += cov(d, d, ax) / len(m.train_subs)
            S[ax]['s'] += (cov(h1, h2, ax) + cov(h2, h1, ax)) / (2 * len(m.train_subs))
            S[ax]['h'] += (cov(h1, h1, ax) + cov(h2, h2, ax)) / (2 * len(m.train_subs))
    Sc = {ax: psd(S[ax]['s']) + 2 * psd(S[ax]['h'] - S[ax]['s']) / reps for ax in 'tc'}
    lag = (torch.arange(250)[:, None] - torch.arange(250)[None]).abs()
    taper = (1 - lag / 25).clamp(min=0).double()
    return gain(S['t']['d'], Sc['t']).T, gain(S['t']['d'] * taper, Sc['t'] * taper).T, gain(S['c']['d'], Sc['c'])


out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        mu, d = x.mean(0, keepdim=True), x - K
        At, Al, As = filters(m, K)
        D = torch.stack([train_key(s) - K[0] for s in m.train_subs]).flatten(1)  # (9, 15750)
        per = d.view(200, 63, 5, 50).mean(2)
        ks = {'plain': K.expand_as(x), 'comb': K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5),
              'span': K + (d.flatten(1) @ torch.linalg.pinv(D) @ D).view_as(x), 'time': K + d @ At, 'time_local': K + d @ Al,
              'space': K + As @ d, 'space+local': K + As @ d @ Al, 'oracle': mu.expand_as(x)}
        att = atm_emb(m, x, fold)[1]
        r = dict(run=RUN, fold=fold)
        for name, k in ks.items():
            r[name] = top1(atm_emb(m, x - k + K, fold, att)[0], g)
            r[f'r2 {name}'] = 1 - ((k - mu).pow(2).sum((1, 2)) / (mu - K).pow(2).sum()).mean().item()
        out.append(r)
        print({k: v if k == 'run' else round(v, 2) for k, v in r.items()}, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k42_key_no_paradigm.csv', index=False)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
