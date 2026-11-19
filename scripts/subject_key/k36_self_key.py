"""K36: can ONE trial give its own subject key? (pure inductive setting: nothing from the new subject but the trial decoded)
The key m_s is the subject's steady-state response to the 5 Hz image stream: 94 % of (m_s - K) repeats every 200 ms
(50 samples), the image-specific content c does not (lag-50 correlation .96 against .07). So a trial x = m_s + c carries
an estimate of m_s. Frozen ATM, held-out subject, 80-repetition test trials, per-trial key estimate k:
    attention map of the raw trial (own key, as the model computes it), readout fed  x - k + K   (the targeted substitution of E13 / K05)
  plain    the model as it is (k = K)
  oracle   k = the subject's mean ERP over its 200 test trials (test data required: the reference, 41.8 on 3 seeds)
  per      k = K + P (x - K), P = mean over the five 200 ms periods of the window (comb filter)
  per_ac   same without the temporal mean (the constant part of the key is taken from K)
  wiener   k = K + (x - K) A,  A = S_d (S_d + S_c)^-1 over time, shared by the electrodes; S_d = time covariance of the
           TRAINING keys around K, S_c = time covariance of the content of a training trial at 80 repetitions
           (signal part from the agreement of two 2-repetition halves, noise part scaled 1 / 80). Training subjects only.
  per_ac_.5  half strength (the midpoint key)
r2 = share of |m_s - K|^2 explained by the estimate (mean over trials).
Usage: k36_self_key.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 400, replace=False))


def wiener(m, K, reps=80):
    Sd, Ss, Sh = 0, 0, 0
    for s in m.train_subs:
        d = (train_key(s) - K[0]).double()
        Sd = Sd + d.T @ d / (len(m.train_subs) * len(d))
        h = T(eeg(s, 'train', average=False)[IM]).double()  # (400, 4, 63, 250)
        h1, h2 = (h[:, :2].mean(1) - train_key(s)).flatten(0, 1), (h[:, 2:].mean(1) - train_key(s)).flatten(0, 1)
        n = len(h1) * len(m.train_subs)
        Ss, Sh = Ss + (h1.T @ h2 + h2.T @ h1) / (2 * n), Sh + (h1.T @ h1 + h2.T @ h2) / (2 * n)
    psd = lambda S: (lambda w, V: V * w.clamp(min=0) @ V.T)(*torch.linalg.eigh(S))  # the half-agreement estimate is not positive by construction
    Sc = psd(Ss) + 2 * psd(Sh - Ss) / reps
    return (Sd @ torch.linalg.inv(Sd + Sc + 1e-6 * Sd.trace() / len(Sd) * torch.eye(len(Sd), dtype=torch.double))).float().T


out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        mu, d = x.mean(0, keepdim=True), x - K
        per = d.view(200, 63, 5, 50).mean(2).repeat(1, 1, 5)
        A = wiener(m, K)
        ac = per - per.mean(-1, keepdim=True)
        ks = {'plain': K.expand_as(x), 'oracle': mu.expand_as(x), 'per': K + per, 'per_ac': K + ac, 'per_ac_.5': K + 0.5 * ac, 'wiener': K + d @ A}
        att = atm_emb(m, x, fold)[1]
        r = dict(run=RUN, fold=fold)
        for name, k in ks.items():
            r[name] = top1(atm_emb(m, x - k + K, fold, att)[0], g)
            r[f'r2 {name}'] = 1 - ((k - mu).pow(2).sum((1, 2)) / (mu - K).pow(2).sum()).mean().item()
        out.append(r)
        print(r, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k36_self_key.csv', index=False)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
