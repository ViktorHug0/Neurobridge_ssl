"""K29: key fine-tuning for a new subject. Frozen ATM; the only trained parameter is the 63x250 additive key fed to the
readout (attention map from the subject's own key, as in K07). Calibration data = n of the subject's training-image
trials (4-repetition averages, never its test trials), init = the mean training key K or the subject's own key.
  labelled    contrastive cross-entropy with the true images (few-shot)
  unlabelled  the SET of the n calibration images is known, not which trial shows which: pseudo-labels by one-to-one
              assignment (Hungarian) of the re-centred trial embeddings to the n images, a few steps, repeat
Test: the 200 test queries under the tuned key, plain and re-centred (K10). References: K / own key, plain and re-centred.
Usage: k29_key_tuning.py [folds]     env STEPS (300), NS ("50,200,1000,5000")"""
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sk_lib import *
from e13_atm_attention import STATE
from module.eeg_encoder.model import run_eeg_backbone

RUN = 'atm128_vc/seed3300'
FOLDS = [int(f) for f in sys.argv[1].split(',')] if len(sys.argv) > 1 else [1, 3, 5, 7]
STEPS = int(os.environ.get('STEPS', 300))
NS = [int(n) for n in os.environ.get('NS', '50,200,1000,5000').split(',')]
CE = torch.nn.functional.cross_entropy
rows = []
for fold in FOLDS:
    m = load(RUN, fold)
    g, gtr = img_emb(m), img_emb(m, 'train')
    anchor = gtr.mean(0)
    x = T(eeg(fold)); mu = x.mean(0, keepdim=True); c = x - mu
    xtr = T(eeg(fold, 'train')); mutr = xtr.mean(0, keepdim=True); ctr = xtr - mutr
    K = mean_key(m)
    A_te, A_tr = atm_emb(m, mu, fold)[1], atm_emb(m, mutr, fold)[1].to(DEV)
    cal = lambda e: nz(e, dim=1) - nz(e, dim=1).mean(0) + anchor

    def test(key):
        e = atm_emb(m, c + key, fold, A_te)[0]
        return top1(e, g), top1(cal(e), g)

    def embed(xb, key):  # differentiable w.r.t. the key
        STATE.update(rec=None, force=A_tr, pos=0)
        e = m.eeg_proj(run_eeg_backbone(m.enc, m.args, xb.to(DEV) + key, torch.full((len(xb),), fold, device=DEV))).float()
        STATE.update(force=None)
        return nz(e, dim=1)

    for init, k0, k0tr in (('K', K, K), ('own', mu, mutr)):
        p, pc = test(k0)
        rows.append(dict(fold=fold, mode='reference', init=init, n=0, plain=p, cal=pc))
        for n in NS:
            idx = torch.randperm(16540, generator=torch.Generator().manual_seed(fold))[:n]
            xs, gs = ctr[idx], gtr[idx].to(DEV)
            for mode in ['labelled'] + (['unlabelled'] if n <= 1000 else []):
                key = k0tr.clone().to(DEV).requires_grad_(True)
                opt = torch.optim.Adam([key], lr=0.02 * K.std().item())
                gen = torch.Generator().manual_seed(0)
                target = torch.arange(n)
                for step in range(STEPS):
                    if mode == 'unlabelled' and step % 30 == 0:  # re-assign: re-centred embeddings against the n images, one-to-one
                        with torch.no_grad():
                            e = torch.cat([embed(xs[i:i + 100], key) for i in range(0, n, 100)]).cpu()
                        S = nz(e - e.mean(0) + gs.cpu().mean(0), dim=1) @ gs.cpu().T
                        target = torch.as_tensor(linear_sum_assignment(-S.numpy())[1])
                        pl_acc = 100 * (target == torch.arange(n)).float().mean().item()
                    b = torch.randperm(n, generator=gen)[:min(128, n)]
                    logits = embed(xs[b], key) @ gs.T / 0.05
                    loss = CE(logits, target[b].to(DEV))
                    opt.zero_grad(); loss.backward(); opt.step()
                p, pc = test(key.detach().cpu() - k0tr + k0)  # the tuned offset, carried to the test key
                rows.append(dict(fold=fold, mode=mode, init=init, n=n, plain=p, cal=pc, **({'pseudo_label_acc': pl_acc} if mode == 'unlabelled' else {})))
                print({k: (round(v, 1) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k29_key_tuning.csv', index=False)
pd.set_option('display.width', 200)
print(df.groupby(['mode', 'init', 'n'], sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
