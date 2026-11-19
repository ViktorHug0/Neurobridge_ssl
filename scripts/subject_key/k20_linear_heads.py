"""K20: one trunk, several heads ("hydra"), training-free probe. The trunk is a FROZEN ATM (atm128_vc seed 3300).
For every member k of the teacher file (t6) a linear head is fitted by ridge regression from a trunk feature to that
member's embeddings of the clean training trials; a new subject is then scored by each head against that member's own
image embeddings, every head calibrated (image anchor of its space), and the heads' cosine matrices are averaged.
  feature: 'bb' = the 128-d trunk output; 'pre' = the 1440-d input of the trunk's head (flattened conv readout).
If linear heads on a frozen trunk already move the score, a trained multi-head student has room to do better.
Usage: k20_linear_heads.py [folds]"""
import pandas as pd
from sk_lib import *
from module.eeg_encoder.model import run_eeg_backbone

FOLDS = [int(f) for f in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 5, 7]
rows = []
for fold in FOLDS:
    t = torch.load(f'{SK}/teachers/t6_f{fold}.pt', map_location='cpu')
    Gte, teacher = [], 0  # each member's test-image embeddings (its own image space); one member on the GPU at a time
    for run in t['runs']:
        mm = load(run, fold); Gte.append(img_emb(mm, 'test')); teacher = teacher + calibrated(mm, fold); del mm; torch.cuda.empty_cache()
    anchors = [g.float().mean(0) for g in t['img']]
    m = load('atm128_vc/seed3300', fold)
    pre = {}
    hook = m.enc.proj_eeg.register_forward_hook(lambda mod, i, o: pre.__setitem__('x', i[0].detach().float().cpu()))

    @torch.no_grad()
    def feats(x, s):
        bb, pr = [], []
        for i in range(0, len(x), 50):  # small batches: a training run shares the GPU
            xb = x[i:i + 50].to(DEV)
            bb.append(run_eeg_backbone(m.enc, m.args, xb, torch.full((len(xb),), s, device=DEV)).float().cpu()); pr.append(pre['x'])
        return torch.cat(bb), torch.cat(pr)

    tr = [feats(T(eeg(s, 'train')), s) for s in t['subjects']]
    te = feats(T(eeg(fold)), fold)
    hook.remove()
    r = dict(fold=fold)
    for j, name in enumerate(('bb', 'pre')):
        X = torch.cat([a[j] for a in tr]).double(); Xte = te[j].double()
        mu = X.mean(0); X = X - mu; Xte = Xte - mu
        XtX = X.T @ X
        for lam in (1e-3, 1e-1):
            A = torch.linalg.inv(XtX + lam * torch.trace(XtX) / len(XtX) * torch.eye(len(XtX), dtype=torch.float64))
            S = []
            for k in range(len(t['eeg'])):
                E = t['eeg'][k].double()
                W = A @ (X.T @ (E - E.mean(0)))
                e = nz((Xte @ W + E.mean(0)).float(), dim=1)
                S.append(nz(e - e.mean(0) + anchors[k], dim=1) @ Gte[k].T)
                r[f'{name}_l{lam}_head{k}'] = acc(S[-1]).item()
            r[f'{name}_l{lam}_all'] = acc(sum(S)).item()
            r[f'{name}_l{lam}_no_own'] = acc(sum(S[1:])).item()
    r['trunk_own_head'] = acc(calibrated(m, fold)).item()
    r['teacher'] = acc(teacher).item()
    rows.append(r)
    print({k: round(v, 1) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k20_linear_heads.csv', index=False)
print(df.mean().round(2).to_string())
