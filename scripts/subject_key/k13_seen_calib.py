"""K13: is the output offset what a wrong key breaks? Seen subjects (3 per fold), 200 test images:
own key / another training subject's key / no key, each plain and after the anchored calibration (image anchor, K10).
If calibration restores a wrongly keyed seen subject, the key acts mostly through the centre of the output.
Usage: k13_seen_calib.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, a = img_emb(m), img_emb(m, 'train').mean(0)
        cal = lambda e: nz(e, dim=1) - nz(e, dim=1).mean(0) + a
        for j in range(3):
            s = m.train_subs[(fold * 3 + j * 3) % 9]
            others = [t for t in m.train_subs if t != s]
            t = others[(fold + j) % len(others)]
            x = T(eeg(s)); mu = x.mean(0, keepdim=True); mt = T(eeg(t)).mean(0, keepdim=True)
            r = dict(run=RUN, fold=fold)
            for name, inp in (('own', x), ('wrong', x - mu + mt), ('nokey', x - mu)):
                e = m.embed(inp, s)
                r[name], r[name + '+cal'] = top1(e, g), top1(cal(e), g)
            out.append(r)
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k13_seen_calib.csv', index=False)
pd.set_option('display.width', 200)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
