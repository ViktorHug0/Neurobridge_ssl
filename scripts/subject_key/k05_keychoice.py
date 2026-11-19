"""K05: which key should the readout of ATM get for a new subject? (attention map always from the subject's own key)
  K        mean of the 9 training keys (the usual targeted substitution)
  half     midpoint between the own key and K
  nearest  the training key most correlated with the new key
  proj     least-squares combination of the training keys closest to the new key (the new subject as a mixture of
           training subjects; SubjectMix trained the model on such mixtures); projs: the same with weights on the simplex
  soft<t>  training keys weighted by softmax(correlation with the new key / t)
  ens9     mean of the 9 score matrices obtained with each training key in turn; ens9+K adds the K matrix
Usage: k05_keychoice.py <run>"""
import pandas as pd
from sk_lib import *

RUN = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
rows = []
for fold in range(1, 11):
    m = load(RUN, fold)
    g = img_emb(m)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); c = x - mu
    keys = torch.stack([train_key(t) for t in m.train_subs])  # (9, 63, 250)
    K = keys.mean(0, keepdim=True)
    A = atm_emb(m, mu, fold)[1]
    S = lambda key: nz(atm_emb(m, c + key, fold, A)[0], dim=1) @ g.T
    flat = keys.flatten(1); v = mu.flatten()
    corr = torch.corrcoef(torch.cat([flat, v[None]]))[-1, :-1]
    w_ls = torch.linalg.lstsq(flat.T.double(), v.double()[:, None]).solution[:, 0].float()
    w = torch.full((9,), 1 / 9, requires_grad=True)  # simplex weights by projected gradient (softmax parametrisation)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(300):
        opt.zero_grad(); ((torch.softmax(w, 0) @ flat - v) ** 2).mean().backward(); opt.step()
    w_s = torch.softmax(w, 0).detach()
    per = [S(k[None]) for k in keys]
    r = dict(fold=fold, own=acc(nz(m.embed(x, fold), dim=1) @ g.T).item(), K=acc(S(K)).item(), half=acc(S(mu + 0.5 * (K - mu))).item(),
             nearest=acc(per[int(corr.argmax())]).item(), proj=acc(S((w_ls @ flat).view_as(K))).item(), projs=acc(S((w_s @ flat).view_as(K))).item(),
             ens9=acc(sum(per)).item(), **{'ens9+K': acc(sum(per) / 9 + S(K)).item()},
             best1=max(acc(p).item() for p in per), corr_max=corr.max().item(), w_max=w_s.max().item())
    for t in (0.02, 0.05, 0.1):
        r[f'soft{t}'] = acc(S((torch.softmax(corr / t, 0) @ flat).view_as(K))).item()
    rows.append(r)
    print({k: round(v, 2) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k05_keychoice_{RUN.replace("/", "_")}.csv', index=False)
pd.set_option('display.width', 220)
print(df.mean().round(2).to_frame().T.to_string(index=False))
