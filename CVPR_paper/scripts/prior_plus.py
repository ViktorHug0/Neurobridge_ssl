"""What does each R1 method add beyond the one-to-one prior? (v2 paper, user 2026-10-24)

Every method's own score matrix is decoded twice: argmax (as reported) and with a Hungarian assignment (the bijection
of R1 imposed on top). `prior` = Hungarian on the zero-shot cosine scores. A method adds an inductive bias beyond the
one-to-one correspondence only if its Hungarian-decoded accuracy exceeds `prior`. Configs of EM-Dirichlet / TransCLIP
= the cross-fold ones of r1_transductive.csv; SAGE = cross-fold params. Embeddings: the r1_transductive cache.
usage: prior_plus.py <final dir, e.g. results/things_eeg/cvpr/final/atm128_s3300> <params_csv> <cache_dir> <out_csv>
"""
import os, sys

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_tta_calibration, fit_saw_transform, fit_tta_calibration, score_features  # noqa: E402
from module.transductive import infer  # noqa: E402
from stream_grid import subject_params  # noqa: E402

if not torch.cuda.is_available():  # as in r1_transductive.py: TransCLIP hard-codes .cuda()
    torch.Tensor.cuda = lambda self, *a, **k: self


def num(v):
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def main():
    final, params_csv, cache, out = sys.argv[1:5]
    sel = pd.read_csv(f'{final}/r1_transductive.csv')
    rows = []
    for sid in range(1, 11):
        z = np.load(f'{cache}/sub-{sid:02d}.npz')
        q, g = z['q'], z['g']
        p = subject_params(params_csv, sid)
        qw = fit_saw_transform(q, p)[0]
        S = {'zero-shot': score_features(q, g), 'csls': score_features(q, g, use_csls=True, csls_k=3),
             'sage': score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True, csls_k=p.csls_k)}
        for m in ('transclip', 'em_dirichlet'):
            r = sel[(sel.subject == sid) & (sel.method == m)].iloc[0]
            cfg = {k: num(v) for k, v in (kv.split('=') for kv in r.config.split())}
            if m == 'transclip':
                cfg['device'] = 'cpu'
            S[m] = infer(qw if r.front == 'saw' else q, g, m, cfg)
        for m, s in S.items():
            rows.append({'subject': sid, 'method': m, 'argmax': 100 * (s.argmax(1) == np.arange(200)).mean(),
                         'hungarian': 100 * (linear_sum_assignment(-s)[1] == np.arange(200)).mean()})
        print(sid, flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(out, index=False)
    print(d.groupby('method')[['argmax', 'hungarian']].mean().round(1))


if __name__ == '__main__':
    main()
