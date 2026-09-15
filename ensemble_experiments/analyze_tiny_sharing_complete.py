"""Audit the completed sharing screen and export reproducible analysis artifacts."""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'results/things_eeg/tiny_sharing/screen_20260911_v2'
OUT = ROOT / 'ensemble_experiments/analysis/tiny_sharing_complete'
NAMES = ['Native independent', 'Compatible independent', 'Shared temporal',
         'Shared spatial', 'Shared readout', 'Shared temporal + spatial',
         'Shared conv + readout', 'Shared normalization too',
         'Attention after stem, independent', 'Compute stem once']


def read(path):
    return json.loads(path.read_text())


def rowz(x):
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True).clip(1e-8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, summaries, scores, audits = [], {}, {}, []
    alpha_grid = np.linspace(0, 1, 21)  # TS weight; ATM weight is 1-alpha
    fusion_sweep = {}
    for slot in (1, 2, 3):
        for i in range(10):
            config = f'C{i}'
            directory = RUNS / f'slot{slot}' / config
            s, m = read(directory / 'summary.json'), read(directory / 'metrics_best_mean.json')
            summaries[slot, config] = s
            assert s['source_hash'] == '5383ed884ef6806c'
            assert s['epochs'] == len(s['history']) == 40 and s['max_steps'] is None
            assert s['total_steps'] == 4640 and s['mixup']['type'] == 'pairwise'
            assert s['master_seed'] == 3300 and s['atm_branch_seed'] == 4300
            selected = min(s['history'], key=lambda x: x['val_loss_mean'])['epoch']
            assert selected == s['selected_epoch_common'] == m['selected_epoch']
            z = np.load(directory / 'scores_best_mean.npz')
            a, b, correct = z['ts'], z['atm'], z['correct']
            assert a.shape == b.shape == (1650, 200) and np.isfinite(a).all() and np.isfinite(b).all()
            az, bz = rowz(a), rowz(b)
            f = .5 * (az + bz)
            fusion_sweep[slot, config] = np.array([
                ((alpha*az + (1-alpha)*bz).argmax(1) == correct).mean()
                for alpha in alpha_grid])
            ha, hb, hf = [v.argmax(1) == correct for v in (a, b, f)]
            scores[slot, config] = dict(a=ha, b=hb, fused=hf)
            for key, value in [('top1_ts', ha.mean()), ('top1_atm', hb.mean()),
                               ('top1_fused', hf.mean()), ('oracle_top1', (ha | hb).mean())]:
                assert abs(value - m[key]) < 1e-6, (slot, config, key)
            for key, values in [('ts', a), ('atm', b), ('fused', f)]:
                hit5 = (np.argpartition(values, -5, axis=1)[:, -5:] == correct[:, None]).any(1)
                assert abs(hit5.mean() - m[f'top5_{key}']) < 1e-6
            both, onlya, onlyb, neither = ha & hb, ha & ~hb, ~ha & hb, ~ha & ~hb
            for key, val in zip(('both', 'ts_only', 'atm_only', 'neither'), (both, onlya, onlyb, neither)):
                assert val.sum() == m['overlap'][key]
            assert (~ha & ~hb & hf).sum() == m['fusion_rescues']
            assert ((ha | hb) & ~hf).sum() == m['fusion_losses']
            margins = []
            for key, values in [('ts', az), ('atm', bz), ('fused', f)]:
                masked = values.copy()
                gold = masked[np.arange(1650), correct].copy()
                masked[np.arange(1650), correct] = -np.inf
                margin = gold - masked.max(1)
                assert abs(margin.mean() - m[f'margin_{key}']) < 2e-6
                margins.append(margin)
            bonus = margins[2] - .5 * (margins[0] + margins[1])
            assert bonus.min() > -2e-6
            assert abs(bonus.mean() - m['distractor_complementarity_bonus']) < 2e-6
            early = read(directory / 'metrics_best_mean_20.json')
            ep20 = min(s['history'][:20], key=lambda x: x['val_loss_mean'])['epoch']
            assert ep20 == early['selected_epoch']
            row = dict(config=config, description=NAMES[i], slot=slot,
                       inner_subject=s['inner_subject'], selected_epoch_1based=selected+1,
                       eeg_params=s['parameters']['eeg_unique'],
                       eeg_unreachable=s['parameters']['eeg_unreachable'],
                       total_params=s['parameters']['unique_total'],
                       image_params=s['parameters']['image_side'],
                       epoch_seconds=s['mean_epoch_seconds'], step_ms=s['warm_step_seconds']*1000,
                       peak_gpu_mib=s['peak_gpu_bytes']/2**20,
                       train_seconds=s['total_train_seconds'],
                       early_fused=early['top1_fused']*100,
                       fusion_wins_vs_atm=int((hf & ~hb).sum()),
                       fusion_losses_vs_atm=int((~hf & hb).sum()))
            for key in ('top1_ts', 'top1_atm', 'top1_fused', 'top5_fused', 'oracle_top1',
                        'gain_over_best_branch', 'gain_over_mean_branch', 'prediction_agreement'):
                row[key] = m[key]*100
            for key in ('cka_readout', 'distractor_complementarity_bonus', 'margin_ts', 'margin_atm', 'margin_fused',
                        'fusion_rescues', 'fusion_losses'):
                row[key] = m[key]
            for key, value in m['overlap'].items():
                row['overlap_'+key] = value / 1650 * 100
            for block in ('T', 'S', 'R'):
                ds = [(h['epoch'], h['grad_diagnostics'][block]) for h in s['history']
                      if block in (h['grad_diagnostics'] or {})]
                cos = [(e, d['cosine']) for e, d in ds if d['cosine'] is not None]
                if cos:
                    row['grad_'+block+'_cos'] = float(np.mean([c for _, c in cos]))
                    row['grad_'+block+'_early_cos'] = float(np.mean([c for e, c in cos if e < 5]))
                    row['grad_'+block+'_late_cos'] = float(np.mean([c for e, c in cos if e >= 20]))
                    row['grad_'+block+'_negative_epoch_fraction'] = float(np.mean([c < 0 for _, c in cos]))
                if ds:
                    row['grad_'+block+'_norm_ratio_ts_atm'] = float(np.mean([
                        d['norm_ts']/max(d['norm_atm'], 1e-12) for _, d in ds]))
            if config in ('C0', 'C1'):
                aa, bb = [np.load(directory / f'scores_best_{key}.npz') for key in ('ts', 'atm')]
                assert np.array_equal(aa['correct'], bb['correct']) and np.array_equal(correct, aa['correct'])
                fi = rowz(aa['scores']) + rowz(bb['scores'])
                row['independent_fused'] = 100 * (fi.argmax(1) == correct).mean()
                row['independent_ts'] = 100 * (aa['scores'].argmax(1) == correct).mean()
                row['independent_atm'] = 100 * (bb['scores'].argmax(1) == correct).mean()
                row['independent_ts_epoch_1based'] = int(aa['epoch'])+1
                row['independent_atm_epoch_1based'] = int(bb['epoch'])+1
            rows.append(row)
            audits.append(f'slot{slot}/{config}: 40 epochs, selection, top1/top5, overlaps, margins and score identities passed')
    agg = {}
    for i in range(10):
        config = f'C{i}'
        subset = [r for r in rows if r['config'] == config]
        agg[config] = {key: float(np.mean([r[key] for r in subset])) for key in subset[0]
                       if isinstance(subset[0][key], (int, float))}
    contrasts = []
    for a, b in [('C0','C1'),('C1','C2'),('C1','C3'),('C1','C4'),('C1','C5'),('C5','C6'),
                 ('C1','C6'),('C6','C7'),('C1','C8'),('C8','C9'),('C0','C9')]:
        d = [100 * (scores[slot,b]['fused'].mean() - scores[slot,a]['fused'].mean()) for slot in (1,2,3)]
        contrasts.append(dict(a=a,b=b,mean_pp=float(np.mean(d)),per_fold_pp=d))
    interactions = {}
    for label, terms in [('TS', [('C5',1),('C2',-1),('C3',-1),('C1',1)]),
                         ('TS_R',[('C6',1),('C5',-1),('C4',-1),('C1',1)])]:
        interactions[label] = [sum(sign*100*scores[slot,c]['fused'].mean() for c,sign in terms)
                               for slot in (1,2,3)]
    cross_fold_fusion = {}
    for config in agg:
        chosen, held_scores = [], []
        for held in (1,2,3):
            selection_scores = np.mean([fusion_sweep[s,config] for s in (1,2,3) if s != held], axis=0)
            ties = np.flatnonzero(np.isclose(selection_scores, selection_scores.max(), rtol=0, atol=1e-12))
            best_idx = min(ties, key=lambda k: (abs(alpha_grid[k]-.5), k))
            chosen.append(float(alpha_grid[best_idx]))
            held_scores.append(float(100*fusion_sweep[held,config][best_idx]))
        cross_fold_fusion[config] = dict(ts_weight_by_held_fold=chosen, held_fold_top1=held_scores,
                                       mean_top1=float(np.mean(held_scores)))
    payload = dict(runs=rows, means=agg, contrasts=contrasts, interactions=interactions,
                   cross_fold_fusion=cross_fold_fusion,
                   audit=audits, total_training_gpu_hours=sum(r['train_seconds'] for r in rows)/3600)
    (OUT / 'analysis.json').write_text(json.dumps(payload, indent=2, allow_nan=False))
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / 'per_fold.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    with (OUT / 'configuration_means.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=['config'] + fields[3:], extrasaction='ignore')
        writer.writeheader(); writer.writerows(dict(config=k, **v) for k,v in agg.items())
    print(json.dumps(dict(means=agg, contrasts=contrasts, interactions=interactions,
                         cross_fold_fusion=cross_fold_fusion,
                         total_training_gpu_hours=payload['total_training_gpu_hours'], audits=len(audits)), indent=2))


if __name__ == '__main__':
    main()
