"""Standard batch test (200 queries of 80 repetitions, one per gallery image) for the final method and its variants,
ATM, 3 seeds x 10 subjects. In a single batch there is no previous update: 16 iterations from the identity.
  final          estimated column masses (pseudo-count 1), prior 2d, stream-selected settings (the method of the paper)
  final_noprior  the same without the prior
  uniform        uniform column masses, prior 2d, stream-selected settings
  uniform_batch  uniform column masses, no prior, settings selected on the batch test of the other folds (81.8 of the
                 earlier drafts)
usage: batch_final.py"""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import gala  # noqa: E402
from stream_grid import subject_params  # noqa: E402

B = 'results/things_eeg/cvpr'
V = {'final': ('s', dict(freq=1.0)), 'final_noprior': ('s', dict(freq=1.0, shrink=False)), 'uniform': ('s', {}), 'uniform_batch': ('b', dict(shrink=False))}
rows = []
for seed in (3300, 3301, 3302):
    P = {'s': f'{B}/inc/select/warm1_softmax/selected_per_subject_s{seed}.csv',
         'b': f'{B}/tta_screen_loso/atm128_vc_s{seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv'}
    for sid in range(1, 11):
        z = np.load(f'{B}/inc/cache/atm128_vc_s{seed}/sub-{sid:02d}.npz')
        x, g = z['q80'][0], z['g']
        for name, (sel, kw) in V.items():
            p = subject_params(P[sel], sid)
            top = np.argsort(-gala.fit(x, g, p.saw_shrink, p.sinkhorn_tau, p.soft_procrustes_power, **kw)(x), axis=1)[:, :5]
            hit = top == np.arange(200)[:, None]
            rows.append((seed, sid, name, 100 * hit[:, 0].mean(), 100 * hit.any(1).mean()))
d = pd.DataFrame(rows, columns=['seed', 'subject', 'variant', 'top1', 'top5'])
d.to_csv(f'{B}/inc/batch_final.csv', index=False)
t = d.groupby(['variant', 'subject']).top1.mean().unstack('subject')
t['top1'], t['top5'] = d.groupby('variant').top1.mean(), d.groupby('variant').top5.mean()
t['sd_seeds'] = d.groupby(['variant', 'seed']).top1.mean().groupby('variant').std()
print(t.round(1).to_string())
