"""K15: what is the part of the offset that seen subjects share (K13)? Seen subjects (2 per fold), 200 test images,
queries = mean of the first r repetitions, r = 4 (the training condition), 16, 80. Plain vs calibrated with the image
anchor and with the EEG anchor (training subjects on training images, 4 repetitions).
If the gain vanishes at r = 4, the shared part is the repetition count; if not, it is the image set or the modality gap.
Usage: k15_condition_shift.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

IM = np.sort(np.random.default_rng(0).choice(16540, 1500, replace=False))
out = []
for RUN in sys.argv[1:]:
    for fold in range(1, 11):
        m = load(RUN, fold)
        g = img_emb(m)
        E = lambda v, s: nz(m.embed(v, s), dim=1)
        anchors = {'img': img_emb(m, 'train').mean(0), 'eeg': torch.cat([E(T(eeg(s, 'train')[IM]), s) for s in m.train_subs]).mean(0)}
        for who, subs in (('seen', [m.train_subs[(fold * 3 + j * 4) % 9] for j in range(2)]), ('held', [fold])):
            for s in subs:
                reps = np.asarray(eeg(s, 'test', average=False))
                r = dict(run=RUN, fold=fold, who=who)
                for n in (4, 16, 80):
                    e = E(T(reps[:, :n].mean(1)), s)
                    r[f'r{n}'] = top1(e, g)
                    for name, a in anchors.items():
                        r[f'r{n}+{name}'] = top1(e - e.mean(0) + a, g)
                out.append(r)
    print(RUN, flush=True)
df = pd.DataFrame(out)
df.to_csv(f'{SK}/k15_condition_shift.csv', index=False)
pd.set_option('display.width', 250)
print(df.groupby(['run', 'who'], sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
