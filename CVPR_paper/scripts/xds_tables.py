"""Alljoined-1.6M and THINGS-MEG under the causal protocol (appendix table, 2026-11-16), ATM, one seed, LOSO.
Zero-shot, CSLS, SATTC, Hungarian: results/<dataset>/cvpr/causal; SCORE with ATM and GALA (final method, settings
selected on the streams of the other subjects): results/<dataset>/cvpr/causal2. Same streams (inc_causal.py seeds).
Accuracy of the incoming queries on the first 30 % and the last 30 % of each stream, pooled over subjects and orders.
usage: xds_tables.py
"""
import glob

import pandas as pd

M = [('Zero-shot', 'plain'), ('CSLS', 'csls'), ('SATTC', 'sattc'), ('Hungarian assignment', 'hungarian'), ('SCORE with ATM', 'score'), ('GALA', 'gala_sw_freq1')]
S = ('uniform', 'skew', 'skew25')


def load(D):
    a = pd.concat([pd.read_csv(f) for f in glob.glob(f'results/{D}/cvpr/causal/*.csv')])
    a = a[a.method != 'score']
    b = pd.concat([pd.read_csv(f) for f in glob.glob(f'results/{D}/cvpr/causal2/*.csv')])
    return pd.concat([a[a.method.isin([m for _, m in M])], b])


def part(x, lo, hi):
    T = x.groupby(['scenario', 'q']).t.transform('max')
    x = x[(x.t > lo * T) & (x.t <= hi * T)]
    g = x.assign(h=x.top1 * x.n).groupby(['scenario', 'q', 'method'])[['h', 'n']].sum()
    return (g.h / g.n).unstack('method')


out = {}
for D, qs in (('alljoined', (20, 40)), ('things_meg', (6, 4, 2))):
    x = load(D)
    print(D, 'streams per method:', x.drop_duplicates(['subject', 'q', 'scenario', 'order', 'method']).groupby('method').size().to_dict())
    for lab, lo, hi in (('first 30%', 0, 0.3), ('last 30%', 0.7, 1.0)):
        t = part(x, lo, hi)
        print(D, lab); print(t.round(1).to_string())
        if lab == 'last 30%':
            for q in qs:
                out[(D, q)] = t.xs(q, level='q')
    # per subject, last 30 % of the uniform stream at the largest q: GALA minus zero-shot / minus SCORE
    y = x[(x.scenario == 'uniform') & (x.q == qs[0])]
    y = y[y.t > 0.7 * y.t.max()]
    p = y.assign(h=y.top1 * y.n).groupby(['subject', 'method'])[['h', 'n']].sum()
    p = (p.h / p.n).unstack('method')
    print(D, 'uniform q =', qs[0], '| GALA > zero-shot for', int((p.gala_sw_freq1 > p.plain).sum()), 'of', len(p),
          '| GALA > SCORE for', int((p.gala_sw_freq1 > p.score).sum()), '| zero-shot range', p.plain.min().round(1), p.plain.max().round(1))

print('--- LaTeX rows: Alljoined q = 20 (U S P), q = 40 (U S P), THINGS-MEG q = 6 (U S P)')
cols = [('alljoined', 20), ('alljoined', 40), ('things_meg', 6)]
for name, m in M:
    v = [out[c].loc[sc, m] for c in cols for sc in S]
    best = [max(out[c].loc[sc, mm] for _, mm in M) for c in cols for sc in S]
    print(f'{name} & ' + ' & '.join((r'\textbf{%.1f}' if round(a, 1) == round(b, 1) else '%.1f') % a for a, b in zip(v, best)) + r' \\')
