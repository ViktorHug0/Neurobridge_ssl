"""K17: stress test of the calibration. The mean is meant to describe the SUBJECT; estimated on a stream of queries it
also absorbs the mean CONTENT of that stream. What if the queries are not a representative sample of images?
The 200 test images are split by k-means on their image embeddings (k = 2, 4, 8 clusters = semantically narrow
streams). For the queries of one cluster, scored against the full 200-image gallery:
  plain | all = mean from all 200 queries | narrow = mean from that cluster's queries only
  narrow+img = the same, but the anchor is the mean image embedding of that cluster's TRAINING-set neighbours?  (not
  available without the gallery, so not tested) -> instead 'shrunk' = narrow mean shrunk halfway to the all-queries mean.
Usage: k17_calib_bias.py <run> [<run> ...]"""
import pandas as pd
from sklearn.cluster import KMeans
from sk_lib import *

out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, a = img_emb(m), img_emb(m, 'train').mean(0)
        e = nz(m.embed(T(eeg(fold)), fold), dim=1)
        hit = lambda q, idx: ((nz(q, dim=1) @ g.T).argmax(1) == idx).float()
        for k in (2, 4, 8):
            lab = torch.tensor(KMeans(k, n_init=5, random_state=0).fit_predict(g.numpy()))
            h = {n: torch.zeros(200) for n in ('plain', 'all', 'narrow')}
            for c in range(k):
                idx = torch.where(lab == c)[0]
                h['plain'][idx] = hit(e[idx], idx)
                h['all'][idx] = hit(e[idx] - e.mean(0) + a, idx)
                h['narrow'][idx] = hit(e[idx] - e[idx].mean(0) + a, idx)
            out.append(dict(run=RUN, fold=fold, k=k, **{n: 100 * v.mean().item() for n, v in h.items()}))
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k17_calib_bias.csv', index=False)
print(df.groupby(['run', 'k'], sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
