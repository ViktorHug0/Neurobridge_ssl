"""Hyper-parameter selection on the incremental protocol (advisor's point 3, 2026-11-16).

run    for one (seed, subject): causal accuracy (query t decoded by the calibration fitted on queries 0 .. t-1) of every
       setting of the grid on the main setting (K = 200, q = 20), three streams, stream order 10 (not one of the
       evaluated orders 0-9). GALA: the grid of the batch selection (rho x tau x power); SCORE: k x rho.
       Cold fits are scored on every 8th query; --warm N (N iterations per update from the previous rotation) on all.
merge  per subject, the setting with the best mean accuracy (mean of the three streams) over the OTHER nine subjects of
       the same training seed -> selected_per_subject.csv in the format of tta_screen_loso (protocol loso_fold_cv).
usage: select_stream.py run --seed 3300 --subject 1 --chunk 0 3 --out_dir D [--fit_kwargs '{"freq": 1}'] [--warm 2]
       select_stream.py merge --out_dir D
"""
import argparse, glob, itertools, json, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

B = 'results/things_eeg/cvpr'
GRID = list(itertools.product((0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.94), (0.03, 0.05, 0.07, 0.1, 0.14, 0.2), (1.0, 1.2, 1.5)))
SCORE = list(itertools.product((3, 5, 10), (0.02, 0.05, 0.1, 0.2)))


def run(a):
    import gala
    from inc_stream import SCENARIOS, cosine_scores, make_stream
    from label_free import csls_ref, score_fit
    sid, extra = a.subject, json.loads(a.fit_kwargs)
    os.makedirs(a.out_dir, exist_ok=True)
    try:  # several jobs may read the same queue: the first one to start a command takes it
        os.close(os.open(f'{a.out_dir}/s{a.seed}_{sid}_c{a.chunk[0]}.lock', os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        return
    z = np.load(f"{a.cache_dir or f'{B}/inc/cache/{a.enc}_s{a.seed}'}/sub-{sid:02d}.npz")
    g = z['g']
    grid = (list(itertools.product(*json.loads(a.axes))) if a.axes else GRID)[a.chunk[0]::a.chunk[1]]
    rows = []
    for sc in ('uniform', 'skew', 'skew25'):
        x, y = make_stream(z[f'q{a.q}'], sc, np.random.default_rng([sid, 10, SCENARIOS.index(sc)]))
        ts = range(5, len(x)) if a.warm else range(8, len(x), 8)
        for rho, tau, gam in grid:
            R, hit, sub = None, [], []
            for t in ts:
                f = gala.fit(x[:t], g, rho, tau, gam, **{**extra, **({'steps': a.warm, 'R0': R} if a.warm else {})})
                R = f.R
                hit.append(f(x[t:t + 1])[0].argmax() == y[t])
                if t % 8 == 0 and t >= 8:  # the queries a cold run is scored on
                    sub.append(hit[-1])
            rows.append((a.seed, sid, sc, 'gala', rho, tau, gam, 100 * np.mean(hit), len(hit), 100 * np.mean(sub)))
        if a.chunk[0] == 0:
            for k, r in SCORE:
                hit = []
                for t in range(8, len(x), 8):
                    f = score_fit(x[:t], g, k, r)
                    hit.append(csls_ref(f(x[t:t + 1]), f(x[:t]), g, k)[0].argmax() == y[t])
                rows.append((a.seed, sid, sc, 'score', k, r, 0, 100 * np.mean(hit), len(hit), 100 * np.mean(hit)))
        print(sc, 'done', flush=True)
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows, columns=['seed', 'subject', 'scenario', 'method', 'p1', 'p2', 'p3', 'top1', 'n', 'top1_sub']).to_csv(
        f'{a.out_dir}/s{a.seed}_{sid}_c{a.chunk[0]}.csv', index=False)


def merge(a):
    d = pd.concat(pd.read_csv(f) for f in glob.glob(f'{a.out_dir}/s*_c*.csv'))
    d = d.groupby(['seed', 'subject', 'method', 'p1', 'p2', 'p3']).top1.mean().reset_index()  # mean of the three streams
    for seed, ds in d.groupby('seed'):
        for m, dm in ds.groupby('method'):
            t = dm.pivot_table(index=['p1', 'p2', 'p3'], columns='subject', values='top1')
            assert t.shape[1] >= 3 and not t.isna().any().any(), (seed, m, t.shape)
            pick = {s: t.drop(columns=s).mean(1).idxmax() for s in t.columns}
            own = np.mean([t.loc[pick[s], s] for s in t.columns])
            print(f'seed {seed} {m}: selected', pd.Series(list(pick.values())).value_counts().to_dict(), f'| accuracy with the selection {own:.1f}',
                  '| with the batch-selected ATM values %.1f' % t.mean(1).get((0.7, 0.1, 1.2) if m == 'gala' else (5, 0.05, 0), np.nan), '| best single setting %.1f' % t.mean(1).max())
            if m == 'gala':
                out = pd.DataFrame([(s, 'loso_fold_cv', *pick[s][:2], 16, pick[s][2]) for s in t.columns],
                                   columns=['subject_id', 'protocol', 'saw_shrink', 'sinkhorn_tau', 'soft_procrustes_steps', 'soft_procrustes_power'])
                out.to_csv(f'{a.out_dir}/selected_per_subject_s{seed}.csv', index=False)
            else:
                pd.DataFrame([(seed, s, int(pick[s][0]), pick[s][1]) for s in t.columns], columns=['seed', 'subject', 'k', 'rho']).to_csv(
                    f'{a.out_dir}/score_selected_s{seed}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=['run', 'merge'])
    ap.add_argument('--seed', type=int)
    ap.add_argument('--subject', type=int)
    ap.add_argument('--chunk', nargs=2, type=int, default=[0, 1], help='i n: settings i, i + n, i + 2n, ... of the grid')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--fit_kwargs', default='{}')
    ap.add_argument('--warm', type=int, default=0)
    ap.add_argument('--enc', default='atm128_vc', help='cache name without the seed')
    ap.add_argument('--cache_dir', default='', help='another dataset: its cache directory (Alljoined, THINGS-MEG)')
    ap.add_argument('--q', type=int, default=20, help='repetitions per query of the selection streams')
    ap.add_argument('--axes', default='', help='JSON [[rho ...], [tau ...], [power ...]] replacing the GALA grid')
    a = ap.parse_args()
    run(a) if a.mode == 'run' else merge(a)
