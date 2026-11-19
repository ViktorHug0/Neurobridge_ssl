"""K10: anchored output calibration for a new subject (label-free, gallery-free): shift its embeddings so that their
mean equals the mean embedding of the TRAINING subjects on training images (K09: ATM 33.4 -> 48).
Per run, 10 folds, own key (no substitution) unless stated:
  plain            no calibration
  test             mean estimated on the subject's 200 test queries themselves
  cal              mean estimated on DISJOINT unlabelled trials of the subject (1,500 of its training-image trials,
                   4-repetition averages; the reference is at the same 4 repetitions), applied to the test queries
  cal+std          the same with the per-dimension std
  img              anchor = mean IMAGE embedding of the training images instead of the training subjects' EEG mean
  sub / sub+cal    key substitution, alone and calibrated
  r1 / r4 / r16    queries = mean of the first r repetitions: plain -> calibrated (mean from the same r-rep queries)
All moments in the L2-normalised embedding space.
Usage: k10_calib.py <run> [<run> ...]"""
import pandas as pd
from sk_lib import *

rng = np.random.default_rng(0)
IM = np.sort(rng.choice(16540, 1500, replace=False))
out = []
for RUN in sys.argv[1:]:
    rows = []
    for fold in range(1, 11):
        m = load(RUN, fold)
        g, gtr = img_emb(m), img_emb(m, 'train')
        x = T(eeg(fold)); mu = x.mean(0, keepdim=True); K = mean_key(m)
        E = lambda v, s=fold: nz(m.embed(v, s), dim=1)
        ref = torch.cat([E(T(eeg(s, 'train')[IM]), s) for s in m.train_subs])
        cal = E(T(eeg(fold, 'train')[IM]))
        e, es = E(x), nz(embed_sub(m, x, fold, mu, K), dim=1)
        r = dict(run=RUN, fold=fold, plain=top1(e, g), test=top1(e - e.mean(0) + ref.mean(0), g),
                 cal=top1(e - cal.mean(0) + ref.mean(0), g),
                 **{'cal+std': top1((e - cal.mean(0)) / cal.std(0) * ref.std(0) + ref.mean(0), g)},
                 img=top1(e - e.mean(0) + gtr.mean(0), g), sub=top1(es, g))
        r['sub+cal'] = top1(es - es.mean(0) + ref.mean(0), g)
        reps = np.asarray(eeg(fold, 'test', average=False))
        for n in (1, 4, 16):
            q = E(T(reps[:, :n].mean(1)))
            r[f'r{n}'], r[f'r{n}+test'] = top1(q, g), top1(q - q.mean(0) + ref.mean(0), g)
        rows.append(r)
        print({k: (round(v, 1) if not isinstance(v, str) else v) for k, v in r.items()}, flush=True)
    out += rows
    pd.DataFrame(out).to_csv(f'{SK}/k10_calib.csv', mode='w', index=False)
df = pd.DataFrame(out)
pd.set_option('display.width', 250)
print(df.groupby('run', sort=False).mean(numeric_only=True).drop(columns='fold').round(2).to_string())
