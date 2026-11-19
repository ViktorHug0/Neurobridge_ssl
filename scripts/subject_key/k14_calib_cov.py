"""K14: second-order anchored calibration. The new subject's normalised embeddings are centred, whitened with their own
(shrunk) covariance and re-coloured with the covariance of an anchor distribution, then moved to the anchor mean:
  e' = (e - mean_e) C_e^-1/2 C_a^1/2 + mean_a,   blended with the mean-only calibration by b (b = 0: K10).
Anchor distribution: the image embeddings of the training images ('img'), or the training subjects' embeddings of
1,500 training images ('eeg', 4-repetition averages). Shrinkage s of both covariances toward their scaled identity.
Usage: k14_calib_cov.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 1500, replace=False))


def cov(a, s):
    c = torch.cov(a.T.double())
    return (1 - s) * c + s * torch.trace(c) / len(c) * torch.eye(len(c), dtype=torch.float64)


def power(c, p):
    w, v = torch.linalg.eigh(c)
    return (v * w.clamp_min(1e-8) ** p) @ v.T


out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, gtr = img_emb(m), img_emb(m, 'train')
        e = nz(m.embed(T(eeg(fold)), fold), dim=1)
        ref = {'img': gtr, 'eeg': torch.cat([nz(m.embed(T(eeg(s, 'train')[IM]), s), dim=1) for s in m.train_subs])}
        r = dict(run=RUN, fold=fold, plain=top1(e, g))
        for name, a in ref.items():
            c0 = e - e.mean(0)
            r[f'{name}_mean'] = top1(c0 + a.mean(0), g)
            for s in (0.3, 0.6):
                W = (power(cov(e, s), -0.5) @ power(cov(a, s), 0.5)).float()
                for b in (0.5, 1.0):
                    r[f'{name}_s{s}_b{b}'] = top1((1 - b) * c0 + b * c0 @ W + a.mean(0), g)
        out.append(r)
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k14_calib_cov.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
