"""Summary of gated_eval.py outputs over the 90 (encoder x setting x U x D) cells; ATM = mean over its seeds.
For each fusion variant: mean top-1 gap to the best single method of {GALA, SCORE, conservative}, worst gap, number
of cells more than 1 point below that best, and the worst loss against GALA at D = 0 (the R1 cell).
usage: summarize_agf.py [--detail VARIANT]
"""
import glob, sys

import pandas as pd

C = 'results/things_eeg/cvpr/next_paper/open_gallery'
d = pd.concat([pd.read_csv(f).assign(run=f.split('agf_')[1][:-4]) for f in glob.glob(f'{C}/agf_*.csv')])
d['arm'] = d.run.str.replace(r'_vc_s\d+', '', regex=True)
m = d.groupby(['arm', 'run', 'setting', 'U', 'D', 'method']).top1.mean().groupby(['arm', 'setting', 'U', 'D', 'method']).mean().unstack()
best = m[['gala', 'score', 'cons']].max(1)
r1 = m.xs(('train', 200, 0), level=('setting', 'U', 'D'))
rows = []
for v in [c for c in m.columns if c.startswith(('agf', 'tri', 'gate')) or c in ('robust_NM', 'score', 'cons', 'cons10', 'gala')]:
    gap = m[v] - best
    rows.append(dict(variant=v, mean_top1=m[v].mean(), mean_gap=gap.mean(), worst_gap=gap.min(), n_below1=(gap < -1).sum(),
                     worst_r1_vs_gala=(r1[v] - r1.gala).min()))
pd.set_option('display.width', 200, 'display.max_rows', 400)
print(pd.DataFrame(rows).set_index('variant').round(2).sort_values('mean_gap', ascending=False))
if len(sys.argv) > 2:
    v = sys.argv[2]
    t = m[['gala', 'score', 'cons', v]].round(1)
    t['gap'] = (m[v] - best).round(1)
    print(t[t.gap < -1])
if 'phat_G' in m.columns:  # label-free accuracy estimates vs true accuracies
    from scipy.stats import pearsonr
    for k, tru in (('G', 'gala'), ('C', 'cons'), ('S', 'score')):
        print(f'phat_{k} vs {tru}: r = {pearsonr(m[f"phat_{k}"], m[tru])[0]:.3f}, mean abs err = {(m[f"phat_{k}"] - m[tru]).abs().mean():.1f} pts')
