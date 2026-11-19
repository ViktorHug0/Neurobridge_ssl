"""What the key does inside the projection head (fold 1 models, seed 3300, validation-selected; subject 1 held out).
h = input of the residual block h + W2 GELU(h). Run from scripts/subject_key: MI_DEV=cpu ../../.venv/bin/python figures/head_data.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sk_lib import *
from scipy.stats import norm as N
import pandas as pd

gelu_d = lambda h: N.cdf(h) + h * N.pdf(h)               # slope of the (exact) GELU
RUNS = [a.split('=') for a in sys.argv[1:]] or [('tsconv', 'tsconv_vc/seed3300'), ('atm', 'atm128_vc/seed3300')]   # tag=run; run relative to cvpr/loso or absolute
PATH = 'figures/head_data.npz'
out = dict(np.load(PATH, allow_pickle=True)) if os.path.exists(PATH) else {}
for tag, run in RUNS:
    m = load(run, 1); g = img_emb(m)
    blk, cap = m.enc.proj_eeg[1], {}
    hk = blk.register_forward_pre_hook(lambda mod, inp: cap.setdefault('h', []).append(inp[0].detach().float().cpu()))
    x = {s: T(eeg(s)) for s in range(1, 11)}; mu = {s: x[s].mean(0, keepdim=True) for s in x}
    H, E = {}, {}
    for s in x:
        cap.clear(); E[s] = m.embed(x[s], s); H[s] = torch.cat(cap['h'])
    seen = m.train_subs
    swapH, swapE = {}, {}
    for a in seen:
        for b in seen:
            if a != b and (a, b) not in swapH and len(swapH) < 72:
                cap.clear(); swapE[(a, b)] = m.embed(x[a] - mu[a] + mu[b], a); swapH[(a, b)] = torch.cat(cap['h'])
    hk.remove()
    inner = getattr(blk, "fn", None) or blk.f   # model.py names it f, atm.py fn
    W2, gelu = inner[1], inner[0]
    tail = lambda z: m.eeg_proj(torch.nn.Sequential(*list(m.enc.proj_eeg)[2:])(z))
    acc = lambda e: top1(e, g)
    rows = []
    with torch.no_grad():
        for (a, b), hs in swapH.items():
            ha, mua, mub = H[a], H[a].mean(0), H[b].mean(0)
            c = ha - mua
            ga, gb = (torch.tensor(gelu_d(v.numpy())).float() for v in (mua, mub))
            lin = lambda off_mu, gain: tail(off_mu + c + W2(gelu(off_mu) + gain * c))     # head linearised around an operating point
            rows.append(dict(a=a, b=b, own=acc(E[a]), input_swap=acc(swapE[(a, b)]), h_shift=acc(tail(blk(c + mub))),
                             lin_own=acc(lin(mua, ga)), lin_wrong=acc(lin(mub, gb)), own_offset_wrong_gain=acc(lin(mua, gb)), wrong_offset_own_gain=acc(lin(mub, ga)),
                             additive_r=torch.corrcoef(torch.stack([(hs - ha).flatten(), (mub - mua).expand_as(ha).flatten()]))[0, 1].item()))
    df = pd.DataFrame(rows)
    print(tag, df.drop(columns=['a', 'b']).mean().round(2).to_dict(), flush=True)
    Hs = torch.stack([H[s] for s in range(1, 11)])                       # (10, 200, D)
    out.update({f'{tag}_H': Hs.numpy(), f'{tag}_rows': df.to_numpy(), f'{tag}_cols': np.array(df.columns), f'{tag}_seen': np.array(seen)})
    Mu, Sd = Hs.mean(1), Hs.std(1)
    G = gelu_d(Mu.numpy())
    print(tag, 'D', Hs.shape[-1], 'between-subject sd of unit means / within-subject sd', (Mu.std(0) / Sd.mean(0)).median().item(),
          'units with gain < .2 for some subject and > .8 for another', ((G.min(0) < .2) & (G.max(0) > .8)).mean())
np.savez_compressed(PATH, **out)
