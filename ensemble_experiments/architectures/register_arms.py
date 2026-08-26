"""Add finished ortho arms to ensemble_experiments/legacy/scratch_claude/manifest.json as ortho_<arm>.

Scans every run under results/things_eeg/ortho_arch/, registers the ones with all
10 folds, and reads the encoder name from train_config.json rather than a hardcoded
map (arm dirs carry recipe suffixes like `_group`). Idempotent.

Usage: .venv/bin/python ensemble_experiments/architectures/register_arms.py
"""
import csv, glob, json, os

ROOT = 'results/things_eeg/ortho_arch'
man = json.load(open('ensemble_experiments/legacy/scratch_claude/manifest.json'))
added, partial = [], []

for run in sorted(glob.glob(f'{ROOT}/*/seed3300')):
    arm = os.path.basename(os.path.dirname(run))
    if arm.startswith('.'):
        continue
    folds, accs = {}, []
    for s in range(1, 11):
        hits = glob.glob(f'{run}/*-sub-{s:02d}/result.csv')
        if not hits:
            continue
        folds[str(s)] = os.path.dirname(hits[0])
        accs.append(float(list(csv.DictReader(open(hits[0])))[0]['best top1 acc']))
    if len(folds) != 10:
        partial.append((arm, len(folds), sum(accs) / len(accs) if accs else 0.0))
        continue
    cfg = json.load(open(os.path.join(next(iter(folds.values())), 'train_config.json')))
    name = f'ortho_{arm}'
    man[name] = {'run': run, 'mean': round(sum(accs) / 10, 2),
                 'enc': cfg['eeg_encoder_type'],
                 'img': os.path.basename(cfg['image_feature_dir'].rstrip('/')),
                 'folds': folds}
    added.append((name, man[name]['mean']))

json.dump(man, open('ensemble_experiments/legacy/scratch_claude/manifest.json', 'w'), indent=1)
for n, m in added:
    print(f'registered {n:24s} mean={m:.2f}')
for a, n, m in partial:
    print(f'skipped    {a:24s} {n}/10 folds (running mean {m:.2f})')
print(f'manifest now holds {len(man)} entries')
