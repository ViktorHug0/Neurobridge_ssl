"""Key-free trained models at low test SNR: query = mean of the first r repetitions (r = 1, 4, 80), plain cosine,
held-out subject; compared fold by fold with the baseline checkpoints of the same architecture."""
import pandas as pd
from mi_lib import *
from e09_arch import GM, LOSO

arm, base = sys.argv[1], sys.argv[2]
rows = []
for fold in range(1, 11):
    d = glob.glob(f'{OUT}/train/{arm}/*-sub-{fold:02d}')
    if not d or not os.path.exists(d[0] + '/result.csv'):
        continue
    mk, mb = GM(d[0]), GM(glob.glob(f'{LOSO}/{base}/seed3300/*-sub-{fold:02d}')[0])
    gk, gb = img_emb(mk), img_emb(mb)
    reps = eeg(fold, 'test', average=False)
    mu = T(eeg(fold)).mean(0, keepdim=True)
    for r in (1, 4, 16, 80):
        acc_b, acc_k = [], []
        for start in range(0, 80 - r + 1, max(r, 20)):  # a few disjoint groups of r repetitions
            x = T(reps[:, start:start + r].astype(np.float32).mean(1))
            acc_b.append(top1(mb.embed(x, fold), gb))
            acc_k.append(top1(mk.embed(x - mu, fold), gk))
        rows.append(dict(fold=fold, reps=r, base=np.mean(acc_b), keyfree=np.mean(acc_k)))
    print(fold, flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e19_demean_reps_{arm}.csv', index=False)
print(df.groupby('reps')[['base', 'keyfree']].mean().round(2))
