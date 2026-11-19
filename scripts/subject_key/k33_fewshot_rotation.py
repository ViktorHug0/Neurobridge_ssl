"""K33: few-shot output rotation. K32: a labelled orthogonal map of the centred output recovers most of what a new
subject loses (48.9 -> 71.9 with 4,000 labelled trials). How many labelled trials does it take?
New subject, frozen model, own key. n labelled training-image trials (4-repetition averages, never test trials):
  R = orthogonal Procrustes of  X^T Y + n0 c I   (X, Y = centred EEG / image embeddings of the n trials; the identity
  prior with pseudo-count n0 keeps R near "no rotation" when n is small; c = mean diagonal of X^T Y / n)
Test queries:  (e - centre) R + anchor, centre = mean of the test queries ("cal", unlabelled test data, K10)
               or the mean of the n labelled trials ("fs", nothing but the n trials).
Also: turn = 1 - tr(R) / d at n = 4000 (0 = identity) for the new subject and one seen subject (own key): how far the map is
from doing nothing.  3 draws per n.
Usage: k33_fewshot_rotation.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.arange(16540)
NS, N0S = (20, 50, 200, 1000, 4000, 16540), (0, 200, 1000, 5000)


def rotation(X, Y, n0):
    X, Y = X - X.mean(0), Y - Y.mean(0)
    C = X.T @ Y
    U, _, Vt = torch.linalg.svd(C + n0 * C.trace() / (len(X) * len(C)) * torch.eye(len(C)))
    return U @ Vt


out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, gtr = img_emb(m), img_emb(m, 'train')
        a, gtr = gtr.mean(0), gtr[IM]
        e, etr = nz(m.embed(T(eeg(fold)), fold), dim=1), nz(m.embed(T(eeg(fold, 'train')[IM]), fold), dim=1)
        s = m.train_subs[fold % 9]
        Rs = rotation(nz(m.embed(T(eeg(s, 'train')[IM[:4000]]), s), dim=1), gtr[:4000], 0)
        r = dict(run=RUN, fold=fold, plain=top1(e, g), cal=top1(e - e.mean(0) + a, g),
                 turn_new=1 - rotation(etr[:4000], gtr[:4000], 0).trace().item() / len(Rs), turn_seen=1 - Rs.trace().item() / len(Rs))
        for n in NS:
            for n0 in N0S:
                acc = []
                for draw in range(3):
                    i = torch.randperm(16540, generator=torch.Generator().manual_seed(draw))[:n]
                    R = rotation(etr[i], gtr[i], n0)
                    acc.append((top1((e - e.mean(0)) @ R + a, g), top1((e - etr[i].mean(0)) @ R + a, g)))
                r[f'n{n} n0={n0} cal'], r[f'n{n} n0={n0} fs'] = np.mean(acc, 0)
        out.append(r)
        print(RUN, fold, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k33_fewshot_rotation.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).T.to_string())
