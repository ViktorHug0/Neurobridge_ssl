"""K07: headroom of the key lever in ATM. ORACLE: the readout key of the new subject is fitted with LABELS on its
training-image trials (contrastive cross-entropy through the frozen model, attention map from its own key), then
used on its test trials. Upper bound for any label-free choice of the readout key (K 42.4, mixture key 44.6).
Usage: k07_oracle_key.py <run> <folds>"""
import pandas as pd
from sk_lib import *
from e13_atm_attention import STATE
from module.eeg_encoder.model import run_eeg_backbone

RUN = sys.argv[1] if len(sys.argv) > 1 else 'atm128_vc/seed3300'
FOLDS = [int(f) for f in sys.argv[2].split(',')] if len(sys.argv) > 2 else [1, 3, 5, 7]
rows = []
for fold in FOLDS:
    m = load(RUN, fold)
    g, gtr = img_emb(m), img_emb(m, 'train').to(DEV)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); c = x - mu
    xtr = T(eeg(fold, 'train')); mutr = xtr.mean(0, keepdim=True); ctr = xtr - mutr
    K = mean_key(m)
    A_te, A_tr = atm_emb(m, mu, fold)[1], atm_emb(m, mutr, fold)[1]
    key = K.clone().to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([key], lr=0.02 * K.std().item())
    r = dict(fold=fold, own=top1(m.embed(x, fold), g), K=top1(atm_emb(m, c + K, fold, A_te)[0], g))
    gen = torch.Generator(device='cpu').manual_seed(0)
    for step in range(1, 601):
        idx = torch.randperm(len(ctr), generator=gen)[:128]
        STATE.update(rec=None, force=A_tr.to(DEV), pos=0)
        e = m.eeg_proj(run_eeg_backbone(m.enc, m.args, ctr[idx].to(DEV) + key, torch.full((128,), fold, device=DEV)))
        loss = torch.nn.functional.cross_entropy(nz(e.float(), dim=1) @ gtr[idx.to(DEV)].T / 0.05, torch.arange(128, device=DEV))
        opt.zero_grad(); loss.backward(); opt.step()
        STATE.update(force=None)
        if step in (100, 300, 600):
            r[f'oracle{step}'] = top1(atm_emb(m, c + key.detach().cpu(), fold, A_te)[0], g)
            r[f'loss{step}'] = loss.item()
    rows.append(r)
    print({k: round(v, 2) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k07_oracle_key.csv', index=False)
print(df.mean().round(2).to_frame().T.to_string(index=False))
