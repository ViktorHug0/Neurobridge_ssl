"""Add every layer_sweep arm to the ensemble manifest, regardless of solo score."""
import csv, glob, json, os

MAN = 'ensemble_experiments/legacy/scratch_claude/manifest.json'
man = json.load(open(MAN))
ARMS = {'iv23':  ('TSConv', 'InternViT-6B_layer23_mean_8bit'),
        'iv33':  ('TSConv', 'InternViT-6B_layer33_mean_8bit'),
        'atm23': ('ATM',    'InternViT-6B_layer23_mean_8bit'),
        'atm33': ('ATM',    'InternViT-6B_layer33_mean_8bit'),
        'atm17': ('ATM',    'InternViT-6B_layer28_mean_8bit (17ch)'),
        'tsconv17': ('TSConv', 'InternViT-6B_layer28_mean_8bit (17ch)')}
added, skipped = [], []
for arm, (enc, img) in ARMS.items():
    root = f'results/things_eeg/synthetic_subjects/layer_sweep/{arm}/seed3300'
    folds, accs = {}, []
    for s in range(1, 11):
        d = glob.glob(f'{root}/*-sub-{s:02d}')
        if not d or not os.path.exists(f'{d[0]}/result.csv'):
            continue
        folds[str(s)] = d[0]
        accs.append(float(list(csv.DictReader(open(f'{d[0]}/result.csv')))[0]['best top1 acc']))
    if len(folds) != 10:
        skipped.append((arm, len(folds))); continue
    man[arm] = {'run': root, 'mean': sum(accs) / 10, 'enc': enc, 'img': img, 'folds': folds}
    added.append((arm, sum(accs) / 10))
json.dump(man, open(MAN, 'w'), indent=1)
print('added:', [(a, round(m, 2)) for a, m in added])
print('incomplete, skipped:', skipped)
print('manifest size:', len(man))
