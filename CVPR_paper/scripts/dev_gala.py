"""Development runs for the advisor's points 2 and 5 (2026-11-16): variants of GALA on the main setting (K = 200, q = 20),
causal protocol (query t decoded by the calibration fitted on queries 0 .. t-1), streams of inc_causal.py.
  cold variants   refitted from the identity, scored on every 4th query (the same queries for all of them)
  warm variants   rotation of the previous update as the start, a few iterations per update, every query processed
Writes one row per (stream, order, query): hit of every variant (NaN when not scored) and its fit time in seconds.
usage: dev_gala.py --seed 3300 --subject 1 --output_csv O [--orders 0 1]
"""
import argparse, os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import gala
from inc_stream import SCENARIOS, cosine_scores, make_stream, subject_params
from label_free import csls_ref, score_fit

B = 'results/things_eeg/cvpr'
COLD = {'gala': {}, 'softmax': dict(sinkhorn=False), 'cp25': dict(col_power=0.25), 'cp50': dict(col_power=0.5), 'cp75': dict(col_power=0.75),
        'em0.5': dict(freq=0.5), 'em1': dict(freq=1.0), 'em2': dict(freq=2.0), 'em4': dict(freq=4.0), 'em1cp50': dict(freq=1.0, col_power=0.5)}
WARM = {'warm1': dict(steps=1), 'warm2': dict(steps=2), 'warm4': dict(steps=4)}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--subject', type=int, required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--orders', nargs='+', type=int, default=[0, 1])
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.output_csv), exist_ok=True)
    sid = a.subject
    z = np.load(f'{B}/inc/cache/atm128_vc_s{a.seed}/sub-{sid:02d}.npz')
    g = z['g']
    p = subject_params(f'{B}/tta_screen_loso/atm128_vc_s{a.seed}/folds_1_2_3_4_5_6_7_8_9_10/selected_per_subject.csv', sid)
    kw = dict(rho=p.saw_shrink, tau=p.sinkhorn_tau, gamma=p.soft_procrustes_power, steps=p.soft_procrustes_steps, k=p.csls_k)
    rows = []
    for sc in ('uniform', 'skew', 'skew25'):
        for o in a.orders:
            x, y = make_stream(z['q20'], sc, np.random.default_rng([sid, o, SCENARIOS.index(sc)]))
            zero = cosine_scores(x, g).argmax(1) == y
            R = {v: None for v in WARM}
            for t in range(len(x)):
                r = {'seed': a.seed, 'subject': sid, 'scenario': sc, 'order': o, 't': t, 'plain': zero[t]}
                past, new = x[:t], x[t:t + 1]
                for v, k in WARM.items():
                    if t < 5:
                        r[v] = zero[t]
                        continue
                    t0 = time.perf_counter()
                    f = gala.fit(past, g, **{**kw, **k}, R0=R[v])
                    r[v + '_s'] = time.perf_counter() - t0
                    R[v] = f.R
                    r[v] = f(new)[0].argmax() == y[t]
                if t >= 5 and t % 4 == 0:
                    for v, k in COLD.items():
                        t0 = time.perf_counter()
                        f = gala.fit(past, g, **{**kw, **k})
                        r[v + '_s'] = time.perf_counter() - t0
                        r[v] = f(new)[0].argmax() == y[t]
                    t0 = time.perf_counter()
                    f = score_fit(past, g, 5, 0.05)
                    s = csls_ref(f(new), f(past), g, 5)[0]
                    r['score_s'] = time.perf_counter() - t0
                    r['score'] = s.argmax() == y[t]
                rows.append(r)
            pd.DataFrame(rows).to_csv(a.output_csv, index=False)
            print(sc, o, 'done', flush=True)
