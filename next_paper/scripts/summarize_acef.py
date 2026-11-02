"""ACEF summary: development arms (6 encoders, ATM x 3 seeds) and held-out arms (5 ATM image targets), separately.
Per variant: mean gap to the per-cell best of {GALA, SCORE, C}, worst gap, cells < -1, worst R1 loss vs GALA, wins vs
SCORE / GALA. Plus the ATM training-distractor table with the correspondence bound and the number of true pairs.
usage: summarize_acef.py
"""
import glob
import pandas as pd

C = 'results/things_eeg/cvpr/next_paper/open_gallery'
d = pd.concat([pd.read_csv(f).assign(run=f.split('agf_')[1][:-4]) for f in glob.glob(f'{C}/agf_*.csv')])
d['arm'] = d.run.str.replace(r'_s\d+$', '', regex=True)
m = d.groupby(['arm', 'run', 'setting', 'U', 'D', 'method']).top1.mean().groupby(['arm', 'setting', 'U', 'D', 'method']).mean().unstack()
held = m.index.get_level_values('arm').str.contains('iv20|iv44|dino40|cliph|bigg27')
V = ['tri_p', 'acef_full4', 'acef_own3', 'acef_own4', 'acef_own3b', 'agf2_mix', 'robust_NM', 'score', 'gala']
pd.set_option('display.width', 200)
for name, mm in (('DEVELOPMENT (6 encoders)', m[~held]), ('HELD-OUT (5 image targets)', m[held])):
    best = mm[['gala', 'score', 'cons']].max(1)
    r1 = mm.xs(('train', 200, 0), level=('setting', 'U', 'D'))
    rows = [dict(variant=v, mean_top1=mm[v].mean(), mean_gap=(mm[v] - best).mean(), worst_gap=(mm[v] - best).min(),
                 n_below1=((mm[v] - best) < -1).sum(), worst_r1=(r1[v] - r1.gala).min(),
                 wins_score=(mm[v] > mm.score).sum(), wins_gala=(mm[v] > mm.gala).sum(), n=len(mm)) for v in V if v in mm]
    print(name); print(pd.DataFrame(rows).set_index('variant').round(2))
a = m.loc[('atm128_vc', 'train')]
print('\nATM training distractors'); print(a[[c for c in ['gala', 'score', 'tri_p', 'acef_own3', 'acef_own4', 'bound_true_pairs', 'n_true_pairs'] if c in a]].round(1).T)
