"""E2.2 main table over seeds: reads results/things_eeg/cvpr/final/<arm>_s<seed>/{r1_competitors,sattc,ablate}.csv
and prints, per arm, mean +- std OVER SEEDS of the 10-subject means, the harm rate (subject-seed pairs with
SAGE < plain), and SAGE vs each method (Wilcoxon over subjects on seed-averaged accuracies). Missing seeds are skipped.
usage: seed_table.py --arms atm128 tsconv --seeds 3300 3301 3302 [--out results/things_eeg/cvpr/final/main_table.md]
"""
import argparse, os

import pandas as pd
from scipy.stats import wilcoxon

F = 'results/things_eeg/cvpr/final'


def load(arm, seed):
    d = os.path.join(F, f'{arm}_s{seed}')
    if not os.path.exists(os.path.join(d, 'r1_competitors.csv')):
        return None
    t = pd.read_csv(os.path.join(d, 'r1_competitors.csv')).pivot(index='subject', columns='method', values='top1')
    if os.path.exists(os.path.join(d, 'sattc.csv')):
        s = pd.read_csv(os.path.join(d, 'sattc.csv'))
        t['sattc'] = s[s.method == 'sattc'].set_index('subject').top1
    if os.path.exists(os.path.join(d, 'ablate.csv')):
        a = pd.read_csv(os.path.join(d, 'ablate.csv'))
        t['prior_only'] = a[(a.reps == 80) & (a.method == 'plain+hung')].groupby('subject').top1.mean()
    return t


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--arms', nargs='+', required=True)
ap.add_argument('--seeds', nargs='+', type=int, required=True)
ap.add_argument('--out', default=os.path.join(F, 'main_table.md'))
a = ap.parse_args()
lines = []
for arm in a.arms:
    tabs = {s: t for s in a.seeds if (t := load(arm, s)) is not None}
    if not tabs:
        continue
    methods = [m for m in ['plain', 'csls3', 'sattc', 'svtl', 'score', 'prior_only', 'sage'] if all(m in t for t in tabs.values())]
    per_seed = pd.DataFrame({s: t[methods].mean() for s, t in tabs.items()}).T
    avg = sum(t[methods] for t in tabs.values()) / len(tabs)  # seed-averaged, per subject
    harm = sum(int((t.sage < t.plain).sum()) for t in tabs.values())
    lines += [f'### {arm} (seeds {sorted(tabs)}; {len(tabs)} x 10 subjects)', '',
              '| Method | mean over seeds | std over seeds | SAGE - method | subjects SAGE wins | Wilcoxon p |', '|---|---|---|---|---|---|']
    for m in methods:
        diff = avg.sage - avg[m]
        p = wilcoxon(diff).pvalue if m != 'sage' else float('nan')
        lines.append(f'| {m} | {per_seed[m].mean():.2f} | {per_seed[m].std(ddof=1) if len(tabs) > 1 else float("nan"):.2f} | '
                     f'{diff.mean():+.2f} | {int((diff > 0).sum())}/{len(diff)} | {p:.3g} |')
    lines += ['', f'Harm (SAGE < plain), subject-seed pairs: {harm}/{10 * len(tabs)}', '']
open(a.out, 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
