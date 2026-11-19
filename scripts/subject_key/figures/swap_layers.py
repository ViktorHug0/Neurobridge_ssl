"""(1) Accuracy after swapping the subject's mean activation for another subject's at one layer, then running on.
(2) MI_LOG E07 C on this model: wrong key at the input, own mean activation restored at one layer.
TSConv validation-selected, seed 3300, fold 1. Run from scripts/subject_key: MI_DEV=cpu ../../.venv/bin/python figures/swap_layers.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sk_lib import *
import pandas as pd

m = load('tsconv_vc/seed3300', 1); e = m.enc; g = img_emb(m)
x = {s: T(eeg(s)) for s in range(1, 11)}
seen = m.train_subs
fs = [lambda z: e.tsconv[:4](z.unsqueeze(1)), e.tsconv[4:7], lambda z: e.projection(z).flatten(1), e.proj_eeg[0], e.proj_eeg[1], e.proj_eeg[2:], m.eeg_proj]
names = ['input', 'after ELU 1', 'after ELU 2', 'stem output', 'head input h', 'after GELU block', 'after LayerNorm', 'embedding']


@torch.no_grad()
def acts(z):
    out = [z]
    for f in fs:
        z = f(z); out.append(z)
    return out


@torch.no_grad()
def run_from(i, z):
    for f in fs[i:]:
        z = f(z)
    return z


acc = lambda emb: 100 * ((nz(emb, dim=1) @ g.T).argmax(1) == torch.arange(200)).float().mean().item()
means = {s: [a.mean(0, keepdim=True) for a in acts(x[s])] for s in range(1, 11)}
rows = []
for a in seen + [1]:
    A = acts(x[a]); own = acc(A[-1])
    for b in seen:
        if b == a:
            continue
        r = dict(a=a, b=b, held=a == 1, own=own)
        for i, n in enumerate(names):
            r[n] = acc(run_from(i, A[i] - means[a][i] + means[b][i]))
        W = acts(x[a] - means[a][0] + means[b][0])          # wrong key at the input ...
        r['wrong'] = acc(W[-1])
        for i, n in enumerate(names):                         # ... own mean restored at one layer (MI_LOG E07 C)
            r['restore ' + n] = acc(run_from(i, W[i] - W[i].mean(0, keepdim=True) + A[i].mean(0, keepdim=True)))
        rows.append(r)
    print(a, {k: round(v, 1) for k, v in pd.DataFrame([q for q in rows if q['a'] == a]).mean(numeric_only=True).items() if k not in ('a', 'b')}, flush=True)
df = pd.DataFrame(rows); df.to_csv('figures/swap_layers.csv', index=False)
print(df.groupby('held').mean(numeric_only=True).drop(columns=['a', 'b']).round(1).T.to_string())
