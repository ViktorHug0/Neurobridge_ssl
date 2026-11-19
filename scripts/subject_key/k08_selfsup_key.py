"""K08: can the readout key of a new subject be fitted WITHOUT labels, images or test trials? Self-supervised objective
on the subject's unlabelled calibration EEG (here its training-image trials, 4 repetitions per image, never its test
trials): two half-averages of the repetitions of one (unknown) image must land on the same embedding and away from the
other images of the batch (symmetric InfoNCE between halves, frozen model, attention map from the subject's own key).
Only the 63x250 additive readout key is trained, from K. K07 = the same with labels (oracle).
Usage: k08_selfsup_key.py <run> <folds> [tau] [lr_scale] [source: train|test]"""
import pandas as pd
from sk_lib import *
from e13_atm_attention import STATE
from module.eeg_encoder.model import run_eeg_backbone

RUN, FOLDS = sys.argv[1], [int(f) for f in sys.argv[2].split(',')]
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
LR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.02
SRC = sys.argv[5] if len(sys.argv) > 5 else 'train'
CE = torch.nn.functional.cross_entropy
rows = []
for fold in FOLDS:
    m = load(RUN, fold)
    g = img_emb(m)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); c = x - mu
    K = mean_key(m)
    A_te = atm_emb(m, mu, fold)[1]
    reps = np.asarray(eeg(fold, SRC, average=False))  # (images, reps, 63, 250)
    half = reps.shape[1] // 2
    h1, h2 = T(reps[:, :half].mean(1)), T(reps[:, half:].mean(1))
    mucal = (h1.mean(0, keepdim=True) + h2.mean(0, keepdim=True)) / 2
    A_cal = atm_emb(m, mucal, fold)[1].to(DEV)
    h1, h2 = h1 - mucal, h2 - mucal
    key = K.clone().to(DEV).requires_grad_(True)
    opt = torch.optim.Adam([key], lr=LR * K.std().item())
    r = dict(fold=fold, own=top1(m.embed(x, fold), g), K=top1(atm_emb(m, c + K, fold, A_te)[0], g))
    gen = torch.Generator(device='cpu').manual_seed(0)
    B = min(128, len(h1))
    for step in range(1, 601):
        idx = torch.randperm(len(h1), generator=gen)[:B]
        sid = torch.full((B,), fold, device=DEV)
        STATE.update(rec=None, force=A_cal, pos=0)
        e1 = nz(m.eeg_proj(run_eeg_backbone(m.enc, m.args, h1[idx].to(DEV) + key, sid)).float(), dim=1)
        STATE.update(pos=0)
        e2 = nz(m.eeg_proj(run_eeg_backbone(m.enc, m.args, h2[idx].to(DEV) + key, sid)).float(), dim=1)
        y = torch.arange(B, device=DEV)
        loss = (CE(e1 @ e2.T / TAU, y) + CE(e2 @ e1.T / TAU, y)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        STATE.update(force=None)
        if step in (25, 50, 100, 200, 400, 600):
            r[f's{step}'] = top1(atm_emb(m, c + key.detach().cpu(), fold, A_te)[0], g)
            r[f'l{step}'] = loss.item()
    rows.append(r)
    print({k: round(v, 2) for k, v in r.items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k08_selfsup_key_{SRC}_tau{TAU}_lr{LR}.csv', index=False)
print(df.mean().round(2).to_frame().T.to_string(index=False))
