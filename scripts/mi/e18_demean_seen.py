"""Models trained with the key removed (--subject_key demean): accuracy on SEEN subjects (new images), to see whether
subject-conditional decoding survives without the mean-ERP key. Input x_s - m_s for every subject."""
import pandas as pd
from mi_lib import *
from e09_arch import GM

rows = []
for d in sorted(glob.glob(f'{OUT}/train/{sys.argv[1]}/*-sub-*')):
    if not os.path.exists(d + '/result.csv'):
        continue
    m = GM(d)
    g = img_emb(m)
    for s in range(1, 11):
        x = T(eeg(s))
        rows.append(dict(fold=m.test_sub, sub=s, held=s == m.test_sub, acc=top1(m.embed(x - x.mean(0, keepdim=True), s), g)))
df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/e18_demean_seen_{sys.argv[1]}.csv', index=False)
print(df.groupby('held').acc.mean().round(1))
