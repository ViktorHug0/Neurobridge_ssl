"""K35: does the shrunk re-centring (K10 / K28) add to GALA in the causal stream protocol of the CVPR paper?
Same streams, settings and decoding rule as CVPR_paper/scripts/inc_causal.py (query t decoded by the calibration fitted
on queries 1 .. t-1, first 5 queries zero-shot; K = 200, q = 20; ATM seeds 3300-3302; GALA = gala_sw_freq1 with the
stream-selected settings; SCORE with ATM with its stream-selected k, rho).
Re-centring step, from the past queries only: x' = normalise(x) - t / (t + 5) * (mean of the normalised past queries - anchor),
anchor = mean image embedding of the TRAINING images (nothing from the gallery).
  plain / rc          zero-shot cosine / re-centred cosine
  gala / gala_nz      the paper's method / the same on normalised queries (control for the normalisation alone)
  rc_gala             re-centring, then GALA
  score / rc_score    SCORE with ATM / re-centring, then SCORE
Output: one row per stream, method and window of 8 queries (as inc_causal.py), results/things_eeg/subject_key/k35/.
Usage: k35_recentre_gala.py anchors            (once: the anchors, from the checkpoints)
       k35_recentre_gala.py [orders=5] [workers=30]"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INC, OUT = f'{R}/results/things_eeg/cvpr/inc', f'{R}/results/things_eeg/subject_key/k35'
METHODS = ('plain', 'rc', 'gala', 'gala_nz', 'rc_gala', 'score', 'rc_score')
WARMUP, N0 = 5, 5


def anchors():
    sys.argv = sys.argv[:1]
    import sk_lib as S
    out = {}
    for seed in (3300, 3301, 3302):
        for fold in range(1, 11):
            m = S.load(f'atm128_vc/seed{seed}', fold)
            g = np.load(f'{INC}/cache/atm128_vc_s{seed}/sub-{fold:02d}.npz')['g']
            assert np.allclose(S.nz(S.T(g), dim=1), S.img_emb(m), atol=1e-4), (seed, fold)  # the cache comes from this checkpoint
            out[f'{seed}_{fold}'] = S.img_emb(m, 'train').mean(0).numpy()
            print(seed, fold, flush=True)
    np.savez(f'{OUT}/anchors.npz', **out)


def run(job):
    seed, sid, orders = job
    sys.path[:0] = [f'{R}/CVPR_paper/scripts', f'{R}/scripts/things_eeg/tta_rebuttal', R]
    import gala
    from inc_stream import GALA, SCENARIOS, cosine_scores, make_stream
    from label_free import csls_ref, score_fit
    from stream_grid import subject_params
    z = np.load(f'{INC}/cache/atm128_vc_s{seed}/sub-{sid:02d}.npz')
    g, anchor = z['g'], np.load(f'{OUT}/anchors.npz')[f'{seed}_{sid}'].astype(np.float64)
    p = subject_params(f'{INC}/select/warm1_softmax/selected_per_subject_s{seed}.csv', sid)
    sc_ = pd.read_csv(f'{INC}/select/score_selected.csv').set_index(['seed', 'subject']).loc[(seed, sid)]
    sk, sr = int(sc_.k), float(sc_.rho)
    kw = {**dict(rho=p.saw_shrink, tau=p.sinkhorn_tau, gamma=p.soft_procrustes_power, k=p.csls_k), **GALA['gala_sw_freq1']}
    kw.pop('warm')
    rows = []
    for sc in ('uniform', 'skew', 'skew25'):
        for o in range(orders):
            x, y = make_stream(z['q20'], sc, np.random.default_rng([sid, o, SCENARIOS.index(sc)]))  # as inc_causal.py
            xn = gala._norm(x.astype(np.float64))
            N, zero = len(x), cosine_scores(x, g)
            hit, R0 = {m: np.zeros(N) for m in METHODS}, {}
            for t in range(N):
                shift = t / (t + N0) * (xn[:t].mean(0) - anchor) if t else 0
                feeds = {'raw': (x[:t], x[t:t + 1]), 'nz': (xn[:t], xn[t:t + 1]), 'rc': (xn[:t] - shift, xn[t:t + 1] - shift)}
                for m in METHODS:
                    past, new = feeds['rc' if m.startswith('rc') else 'nz' if m == 'gala_nz' else 'raw']
                    if t < WARMUP or m == 'plain':
                        s = zero[t]
                    elif m == 'rc':
                        s = cosine_scores(new, g)[0]
                    elif 'score' in m:
                        f = score_fit(past, g, sk, sr)
                        s = csls_ref(f(new), f(past), g, sk)[0]
                    else:
                        f = gala.fit(past, g, **kw, R0=R0.get(m))
                        R0[m] = f.R
                        s = f(new)[0]
                    hit[m][t] = s.argmax() == y[t]
            for m in METHODS:
                rows += [(seed, sid, sc, o, min(e + 8, N), m, 100 * hit[m][e:e + 8].mean(), len(hit[m][e:e + 8])) for e in range(0, N, 8)]
    pd.DataFrame(rows, columns=['seed', 'subject', 'scenario', 'order', 't', 'method', 'top1', 'n']).to_csv(f'{OUT}/s{seed}_{sid}.csv', index=False)
    return seed, sid


def table():
    x = pd.concat([pd.read_csv(f'{OUT}/s{s}_{i}.csv') for s in (3300, 3301, 3302) for i in range(1, 11)])
    out = {}
    for sc in ('uniform', 'skew', 'skew25'):
        for m in METHODS:
            d = x[(x.scenario == sc) & (x.method == m)]
            win = [d[(d.t >= 0.7 * 15 * k) & (d.t <= 1.3 * 15 * k)] for k in (1, 5, 20)] + [d[d.t > 0.9 * d.t.max()]]  # causal_tables.at
            out[(sc, m)] = [round((w.top1 * w.n).sum() / w.n.sum(), 1) for w in win]
    print(pd.DataFrame(out, index=['1 min', '5 min', '20 min', 'end']).T.to_string())


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if sys.argv[1:2] == ['anchors']:
        anchors()
    elif sys.argv[1:2] == ['table']:
        table()
    else:
        orders, workers = int(sys.argv[1]) if len(sys.argv) > 1 else 5, int(sys.argv[2]) if len(sys.argv) > 2 else 30
        with Pool(workers) as pool:
            for done in pool.imap_unordered(run, [(s, i, orders) for s in (3300, 3301, 3302) for i in range(1, 11)]):
                print(done, flush=True)
        table()
