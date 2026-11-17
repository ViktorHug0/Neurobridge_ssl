"""Causal incremental calibration: every query is decoded BEFORE it enters the calibration, and only once.

Same streams as inc_stream.py. The calibration is refitted after every query: query t is decoded by the calibration
fitted on queries 1 .. t-1 (the first --warmup queries by the zero-shot decoder), and its first decision is the one
that counts.
  gala*  gala.fit on the past queries, its frozen scorer on the new one (gallery hubness term from the past queries)
  score  score_fit on the past queries, then CSLS whose gallery-side term comes from the past queries (csls_ref)
  csls   CSLS whose gallery-side term comes from the past queries
  sattc, hungarian, transclip, em_dirichlet   no fit / apply split: the method is run on [past queries; the new query]
         and the row of the new query is read, for at most --cap random queries of the stream (the same queries for
         all these methods)
Output: one row per stream, method and window of ceil(N / 100) consecutive queries (t = last query of the window):
top1 / top5 on the n scored queries of the window. (The 2026-11-13 runs in inc/causal/ came from an earlier version
that refitted at 30 geometrically spaced buffer sizes.)
usage: inc_causal.py --cache_dir C --params_csv P --output_csv O [--trans_csv T] [--methods ...] [--first_order 3] ...
"""
import argparse, os

import numpy as np
import pandas as pd

from inc_stream import GALA, SCENARIOS, cosine_scores, gala, make_stream, methods, sattc, subject_params, trans_configs
from label_free import csls_ref, score_deploy, score_fit


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--trans_csv', default='')
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--K', nargs='+', type=int, default=[200])
    ap.add_argument('--reps', nargs='+', type=int, default=[20])
    ap.add_argument('--scenarios', nargs='+', default=['uniform', 'skew', 'skew25'])
    ap.add_argument('--orders', type=int, default=10)
    ap.add_argument('--first_order', type=int, default=0)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--cap', type=int, default=600)
    ap.add_argument('--score', nargs=2, type=float, default=[5, 0.05], help='k and rho of SCORE (default: those of inc_stream.py)')
    ap.add_argument('--methods', nargs='+', default=['plain', 'csls', 'sattc', 'hungarian', 'score', 'gala'])
    a = ap.parse_args()
    sk, sr = int(a.score[0]), a.score[1]
    os.makedirs(os.path.dirname(os.path.abspath(a.output_csv)), exist_ok=True)
    try:  # several Slurm jobs may hold the same command in their queues: the first one to start it takes it
        os.close(os.open(a.output_csv + '.lock', os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        print('skipped: taken by another job', flush=True)
        return
    rows, checked = [], False
    for sid in a.subjects:
        z = np.load(f'{a.cache_dir}/sub-{sid:02d}.npz')
        p, trans = subject_params(a.params_csv, sid), trans_configs(a.trans_csv, sid)
        kw = dict(rho=p.saw_shrink, tau=p.sinkhorn_tau, gamma=p.soft_procrustes_power, steps=p.soft_procrustes_steps, k=p.csls_k)
        for K in a.K:
            keep = np.sort(np.random.default_rng(500 + sid).choice(200, K, replace=False))
            g = z['g'][keep]
            for q in a.reps:
                blocks = z[f'q{q}'][:, keep]
                for sc in a.scenarios:
                    for o in range(a.first_order, a.first_order + a.orders):
                        x, y = make_stream(blocks, sc, np.random.default_rng([sid, o, SCENARIOS.index(sc)]))  # as inc_stream.py
                        N = len(x)
                        sub = np.zeros(N, bool)  # the queries on which the one-call-per-query methods are scored
                        sub[np.random.default_rng([sid, o, SCENARIOS.index(sc), 1]).permutation(N)[:a.cap]] = True
                        zero = cosine_scores(x, g)
                        prev_R = {}  # warm-started variants: rotation of the previous update
                        rank = {m: np.full((N, 5), -1) for m in a.methods}  # top-5 of each query at its first decoding; -1 = not scored
                        for t in range(N):  # query t, calibration on x[:t]
                            past, new = x[:t], x[t:t + 1]
                            for m in a.methods:
                                if t < a.warmup or m == 'plain':
                                    s = zero[t]
                                elif m == 'csls':
                                    s = csls_ref(new, past, g, 3)[0]
                                elif m == 'score':
                                    f = score_fit(past, g, sk, sr)
                                    s = csls_ref(f(new), f(past), g, sk)[0]
                                    if not checked:  # fitted and applied to the same buffer, this is the transductive score
                                        assert np.allclose(csls_ref(f(past), f(past), g, sk), score_deploy(past, g, k=sk, rho=sr), atol=1e-5)  # float32 gallery on one side
                                        checked = True
                                elif m in GALA:
                                    k = {**kw, **GALA[m]}
                                    if k.pop('warm', False):
                                        k['R0'] = prev_R.get(m)
                                    f = gala.fit(past, g, **k)
                                    prev_R[m] = f.R
                                    s = f(new)[0]
                                elif not sub[t]:
                                    continue
                                elif m == 'sattc':
                                    s = sattc(x[:t + 1], y[:t + 1], g, logits=True)[-1]
                                else:
                                    s = methods([m], x[:t + 1], y[:t + 1], g, p, trans, K)[m][-1]
                                rank[m][t] = np.argsort(-s)[:5]
                        w = -(-N // 100)
                        for m, r in rank.items():
                            for e in range(w, N + w, w):
                                e = min(e, N)
                                ok = r[e - w if e % w == 0 else e - e % w:e]
                                lab = y[e - len(ok):e][ok[:, 0] >= 0]
                                ok = ok[ok[:, 0] >= 0]
                                if len(ok):
                                    rows.append((sid, K, q, sc, o, e, m, 100 * (ok[:, 0] == lab).mean(), 100 * (ok == lab[:, None]).any(1).mean(), len(ok)))
                        pd.DataFrame(rows, columns=['subject', 'K', 'q', 'scenario', 'order', 't', 'method', 'top1', 'top5', 'n']).to_csv(a.output_csv, index=False)
                    print(f'sub {sid} K {K} q {q} {sc} done', flush=True)


if __name__ == '__main__':
    main()
