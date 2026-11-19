"""K02: how informative is the ensemble on the data it was trained on? Distillation runs on training batches
(~113 images x 9 training subjects, 4-repetition averages, pairwise SubjectMix), where the teachers are in-sample.
Per teacher and for the ensemble (mean cosine): 113-way top-1 inside the batch, cosine of the right / wrong images,
and the entropy / top probability of softmax(mean cosine / tau). 'val' = the 165 concepts the *_vc runs held out
(seen subjects, unseen images)."""
import pandas as pd
from sk_lib import *
from module.augment import cross_subject_stimulus_mix

FOLD = int(sys.argv[1]) if len(sys.argv) > 1 else 1
RUNS = ['atm128_vc/seed3300', 'atm128_vc/seed3302', 'tsconv_vc/seed3300', 'labram_vc/seed3300']
held = np.random.default_rng(20260822).permutation(1654)[:165]
is_val = np.isin(np.arange(16540) // 10, held)
ms = [load(r, FOLD) for r in RUNS]
G = [img_emb(m, 'train') for m in ms]
subs = ms[0].train_subs
rng = np.random.default_rng(0)
rows = []
for part, pool in (('train', np.flatnonzero(~is_val)), ('val', np.flatnonzero(is_val))):
    for b in range(4):
        im = np.sort(rng.choice(pool, 113, replace=False))
        x = torch.cat([T(eeg(s, 'train')[im]) for s in subs])
        sid = torch.tensor(np.repeat(subs, 113))
        idx = torch.tensor(np.tile(im, 9))
        for mix in (0, 1):
            xin = cross_subject_stimulus_mix(x, idx // 10, idx % 10, sid, alpha=0.5) if mix else x
            y = torch.arange(113).repeat(9)
            S = [nz(torch.cat([m.embed(xin[i:i + 113], int(sid[i])) for i in range(0, len(xin), 113)]), dim=1) @ g[im].T for m, g in zip(ms, G)]
            for name, s in list(zip(RUNS, S)) + [('ensemble', sum(S) / len(S))]:
                pos = s[torch.arange(len(s)), y]
                r = dict(part=part, mix=mix, member=name, top1=100 * (s.argmax(1) == y).float().mean().item(), cos_pos=pos.mean().item(),
                         cos_neg=((s.sum(1) - pos) / 112).mean().item(), neg_sd=s.std(1).mean().item())
                for tau in (0.01, 0.02, 0.05, 0.1):
                    p = (s / tau).softmax(1)
                    r[f'H@{tau}'] = -(p * p.clamp_min(1e-12).log()).sum(1).mean().item()
                    r[f'pmax@{tau}'] = p.max(1).values.mean().item()
                rows.append(r)
        print(part, b, flush=True)
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(df.groupby(['part', 'mix', 'member'], sort=False).mean().round(3).to_string())
df.to_csv(f'{SK}/k02_teacher_insample_f{FOLD}.csv', index=False)
