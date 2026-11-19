"""Data for the swap-patching and stem-linearity figures. TSConv validation-selected, seed 3300, fold 1 (subject 1 held out).
Run from scripts/subject_key: MI_DEV=cpu ../../.venv/bin/python figures/stem_data.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sk_lib import *
from torch.func import jvp

m = load('tsconv_vc/seed3300', 1); e = m.enc; g = img_emb(m)
x = {s: T(eeg(s)) for s in range(1, 11)}; mu = {s: x[s].mean(0, keepdim=True) for s in x}
seen = m.train_subs
K = torch.stack([mu[s] for s in seen]).mean(0)
stages = [('temporal ELU', lambda z: e.tsconv[:4](z.unsqueeze(1))), ('spatial ELU', e.tsconv[4:7]), ('stem output', lambda z: e.projection(z).flatten(1)),
          ('head input h', e.proj_eeg[0]), ('residual GELU block', e.proj_eeg[1]), ('LayerNorm', e.proj_eeg[2:]), ('embedding', m.eeg_proj)]
stem = lambda z: e.projection(e.tsconv(z.unsqueeze(1))).flatten(1)
head = lambda f: m.eeg_proj(e.proj_eeg(f))


@torch.no_grad()
def acts(z):
    out = []
    for _, f in stages:
        z = f(z); out.append(z.flatten(1))
    return out


hit = lambda emb: (nz(emb, dim=1) @ g.T).argmax(1) == torch.arange(200)
out = {}
with torch.no_grad():
    # swap: per-image hits, own key vs another subject's key
    for s in range(1, 11):
        for t in [s] + seen:
            if t == s and s != 1 and t not in seen:
                continue
            out[f'hit_{s}_{t}'] = hit(acts(x[s] - mu[s] + mu[t])[-1]).numpy()
    # constant-shift share of the change caused by a key swap, per stage (seen pairs)
    num, den = np.zeros(len(stages)), np.zeros(len(stages))
    for a in seen:
        A = acts(x[a])
        for b in seen:
            if a == b:
                continue
            B = acts(x[a] - mu[a] + mu[b])
            for i, (u, v) in enumerate(zip(A, B)):
                D = v - u
                num[i] += (D - D.mean(0)).pow(2).sum().item(); den[i] += D.pow(2).sum().item()
    out['stage_names'] = np.array([n for n, _ in stages]); out['const_share'] = 1 - num / den
    print(dict(zip(out['stage_names'], out['const_share'].round(3))), flush=True)
    # ELU inputs (seen subjects, own key)
    pre1 = torch.cat([e.tsconv[:3](x[s].unsqueeze(1)).flatten() for s in seen])
    pre2 = torch.cat([e.tsconv[4:6](e.tsconv[:4](x[s].unsqueeze(1))).flatten() for s in seen])
    rng = np.random.default_rng(0)
    out['elu1_in'] = rng.choice(pre1.numpy(), 300000, replace=False); out['elu2_in'] = rng.choice(pre2.numpy(), 300000, replace=False)
    print('ELU inputs below -1:', (pre1 < -1).float().mean().item(), (pre2 < -1).float().mean().item(), 'below 0:', (pre1 < 0).float().mean().item(), (pre2 < 0).float().mean().item())
# stem linearised around a point p: f(p) + J(p)(x - p)
def lin_stem(z, p):
    pe = p.expand_as(z).clone()
    f0, df = jvp(stem, (pe,), (z - pe,))
    return f0 + df
res = {}
with torch.no_grad():
    for s in range(1, 11):
        F = stem(x[s])
        for tag, p in (('own', mu[s]), ('K', K)):
            L = lin_stem(x[s], p)
            res[(s, tag)] = (hit(head(F)).float().mean().item() * 100, hit(head(L)).float().mean().item() * 100,
                             1 - (F - L).pow(2).sum().item() / (F - F.mean(0)).pow(2).sum().item())
        if s == 3:
            out['stem_exact_3'], out['stem_lin_3'] = F.numpy(), lin_stem(x[s], mu[s]).numpy()
            out['stem_swap_3'] = stem(x[3] - mu[3] + mu[6]).numpy()
for k, v in res.items():
    print(k, np.round(v, 3))
r = np.array([[s, t == 'own', *v] for (s, t), v in res.items()], dtype=float); out['lin_rows'] = r
print('seen: exact / lin own / lin K', r[(r[:, 0] != 1) & (r[:, 1] == 1)][:, 2:].mean(0).round(2), r[(r[:, 0] != 1) & (r[:, 1] == 0)][:, 2:].mean(0).round(2))
print('held-out: ', r[(r[:, 0] == 1)][:, 2:].round(2))
out['seen'] = np.array(seen)
np.savez_compressed('figures/stem_data.npz', **out)
print({k: v.mean() for k, v in out.items() if k.startswith('hit_') and k.split('_')[1] in ('1', '3')})
