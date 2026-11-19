"""K39: can the in-trial key (K36) also give the output centre? Re-centring (K10) subtracts the subject's mean output
embedding, which needs its other trials. If the model were linear that mean would be the embedding of the mean ERP,
i.e. of the key, and the key can be estimated inside one trial. Frozen ATM, held-out subject, normalised embeddings e,
a = mean image embedding of the training images, k = in-trial key (K + the 200 ms-periodic part of x - K, no temporal mean):
  plain            e(x)                                          pure inductive
  centre e(k)      e(x) - e(k(x)) + a                            pure inductive
  comb             readout fed x - k + K (K36)                   pure inductive
  cal              e(x) - mean e + a  (K10)                      test data required
  centre e(m)      e(x) - e(m_s) + a, m_s = mean test ERP        test data required: is the mean embedding the embedding of the mean?
  comb + cal       K36, then K10                                 test data required
Usage: k39_self_centre.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

rows = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, x, K = img_emb(m), T(eeg(fold)), mean_key(m)
        a = img_emb(m, 'train').mean(0)
        per = (x - K).view(200, 63, 5, 50).mean(2)
        k = K + (per - per.mean(-1, keepdim=True)).repeat(1, 1, 5)
        e, att = atm_emb(m, x, fold)
        e, ec = nz(e, dim=1), nz(atm_emb(m, x - k + K, fold, att)[0], dim=1)
        rows.append({'run': RUN, 'fold': fold, 'plain': top1(e, g), 'centre e(k)': top1(e - nz(m.embed(k, fold), dim=1) + a, g), 'comb': top1(ec, g),
                     'cal': top1(e - e.mean(0) + a, g), 'centre e(m)': top1(e - nz(m.embed(x.mean(0, keepdim=True), fold), dim=1) + a, g),
                     'comb + cal': top1(ec - ec.mean(0) + a, g)})
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
df.to_csv(f'{SK}/k39_self_centre.csv', index=False)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
