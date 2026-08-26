"""Progress/score report for the promising_allfolds reruns."""
import csv
import glob
import os

BASE = 'results/things_eeg/synthetic_subjects/promising_allfolds'
REFS = {
    'pairwise SubjectMix (baseline, seeds 3300/1/2)': (35.90, 10),
    'group Dirichlet k9 a0.5 (seed 3300)': (36.70, 10),
    'group + real/virtual p0.5 (seed 3300)': (36.05, 10),
}

BASE_FOLDS = {  # pairwise seed3300 baseline, per fold
    1: 50.5, 2: 43.0, 3: 27.0, 4: 29.5, 5: 29.5,
    6: 36.0, 7: 33.5, 8: 25.0, 9: 37.0, 10: 48.0,
}


def folds(arm_dir):
    out = {}
    for f in glob.glob(os.path.join(arm_dir, '*sub-*', 'result.csv')):
        sub = int(os.path.basename(os.path.dirname(f)).split('sub-')[-1])
        row = list(csv.DictReader(open(f)))[0]
        out[sub] = float(row['best top1 acc'])
    return out


print('reference all-fold means:')
for name, (v, n) in REFS.items():
    print(f'  {v:5.2f}  {name}')
print()
for arm_dir in sorted(glob.glob(os.path.join(BASE, '*', 'seed*'))):
    arm = arm_dir.split('/')[-2]
    d = folds(arm_dir)
    if not d:
        print(f'{arm:20s} (no folds yet)')
        continue
    mean = sum(d.values()) / len(d)
    paired = [d[s] - BASE_FOLDS[s] for s in d]
    delta = sum(paired) / len(paired)
    detail = ' '.join(f'{s}:{d[s]:.1f}' for s in sorted(d))
    print(f'{arm:20s} n={len(d):2d} mean={mean:6.2f}  vs-baseline(paired)={delta:+.2f}')
    print(f'{"":20s} {detail}')
