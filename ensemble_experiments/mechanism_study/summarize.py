"""Refresh complete-fold summaries and controlled geometry-transfer diagnostics."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import OUTPUT, margin_metrics, row_z, unit
from .score_analysis import geometry_rows


def load_pair(path):
    with np.load(path) as d:
        return (d['scores'].astype(np.float64),
            np.stack([unit(d[f'{m}_image']) @ unit(d[f'{m}_image']).T for m in ['a', 'b']]),
            np.stack([d['object'], d['image_idx']], axis=-1))


def rescue_examples(scores, kernels):
    """Predict rescue of member 0 from its margin and own-vs-partner geometry."""
    q = np.arange(scores.shape[1])
    prediction = scores[0].argmax(-1)
    wrong = prediction != q
    features = []
    for kernel in kernels:
        kernel = kernel.copy(); kernel[q, q] = -np.inf
        ranks = (rankdata(kernel, axis=1) - 2) / (len(q) - 2)
        # Label-informed explanation of the wrong winner's neighbourhood.
        features.append(ranks[q, prediction])
    ordered = np.sort(row_z(scores[0]), axis=1)
    gap = ordered[:, -1] - ordered[:, -2]
    x = np.column_stack([gap, features[0] - features[1]])
    y = row_z(scores).mean(0).argmax(-1) == q
    return x[wrong], y[wrong]


def main():
    rows, geometry, transfers, combinations = [], [], [], []
    for marker in (OUTPUT / 'controlled').glob('*/*/*/complete.json'):
        run = marker.parent
        config = json.loads((run / 'train_config.json').read_text())
        row = pd.read_csv(run / 'result.csv').iloc[0].to_dict()
        rows.append(row)
        loaded = {split: load_pair(run / f'{split}_scores.npz') for split in ['probe', 'test']}
        names = [f'{e}_{t}' for e, t in zip(config['spec']['encoders'], config['spec']['targets'])]
        for split, (scores, kernels, _) in loaded.items():
            geometry.extend(dict(arm=config['arm'], seed=config['seed'], split=split, **r)
                for r in geometry_rows(scores, kernels, config['subject'], names))
        x, y = rescue_examples(*loaded['probe'][:2])
        xt, yt = rescue_examples(*loaded['test'][:2])
        if len(np.unique(y)) == 2 and len(np.unique(yt)) == 2:
            for columns, name in [([0], 'margin_only'), ([0, 1], 'margin_plus_geometry')]:
                estimator = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))
                estimator.fit(x[:, columns], y)
                auc = roc_auc_score(yt, estimator.predict_proba(xt[:, columns])[:, 1])
                transfers.append(dict(arm=config['arm'], seed=config['seed'], subject=config['subject'],
                    predictor=name, auc=auc, probe_n=len(y), test_n=len(yt)))
    destination = OUTPUT / 'summaries'
    destination.mkdir(parents=True, exist_ok=True)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(destination / 'controlled_folds.csv', index=False)
        frame.groupby(['arm', 'seed']).agg(n_subjects=('subject', 'nunique'), top1=('top1', 'mean'),
            distractor_bonus=('distractor_bonus', 'mean'), parameters=('parameters', 'mean')).to_csv(destination / 'controlled_summary.csv')
        pd.DataFrame(geometry).to_csv(destination / 'controlled_geometry.csv', index=False)
        pd.DataFrame(transfers).to_csv(destination / 'geometry_probe_to_outer_test.csv', index=False)
    # Predeclared cross-arm fusions: factorial architecture/target pairs and
    # temporal same-window seed controls. Never search for a winning combination.
    for subject in range(1, 11):
        for seed in [3300]:
            p28 = OUTPUT / f'controlled/geometry28/seed{seed}/sub-{subject:02d}'
            p33 = OUTPUT / f'controlled/geometry33/seed{seed}/sub-{subject:02d}'
            if all((p / 'complete.json').exists() for p in [p28, p33]):
                a, b = load_pair(p28 / 'test_scores.npz'), load_pair(p33 / 'test_scores.npz')
                np.testing.assert_array_equal(a[2], b[2])
                all_scores = np.concatenate([a[0], b[0]])
                for i, j in itertools.combinations(range(4), 2):
                    names = ['ATM28', 'TS28', 'ATM33', 'TS33']
                    combinations.append(dict(subject=subject, seed=seed, members=f'{names[i]}+{names[j]}',
                        **margin_metrics(all_scores[[i, j]])))
        one_temporal = OUTPUT / f'controlled/temporal/seed3300/sub-{subject:02d}'
        one_full = OUTPUT / f'controlled/geometry28/seed3300/sub-{subject:02d}'
        if all((p / 'complete.json').exists() for p in [one_temporal, one_full]):
            t0, f0 = load_pair(one_temporal / 'test_scores.npz'), load_pair(one_full / 'test_scores.npz')
            np.testing.assert_array_equal(t0[2], f0[2])
            for name, scores in [('full_plus_early', np.stack([f0[0][1], t0[0][0]])),
                ('full_plus_late', np.stack([f0[0][1], t0[0][1]]))]:
                combinations.append(dict(subject=subject, seed=3300, members=name, **margin_metrics(scores)))
        # Optional future seed-control exports; no second seed is launched now.
        temporal = [OUTPUT / f'controlled/temporal/seed{s}/sub-{subject:02d}' for s in [3300, 3301]]
        full = [OUTPUT / f'controlled/geometry28/seed{s}/sub-{subject:02d}' for s in [3300, 3301]]
        if all((p / 'complete.json').exists() for p in temporal + full):
            t = [load_pair(p / 'test_scores.npz') for p in temporal]
            f = [load_pair(p / 'test_scores.npz') for p in full]
            for item in t[1:] + f:
                np.testing.assert_array_equal(t[0][2], item[2])
            for name, scores in [('early_seeds', np.stack([x[0][0] for x in t])),
                ('late_seeds', np.stack([x[0][1] for x in t])), ('full_seeds', np.stack([x[0][1] for x in f]))]:
                combinations.append(dict(subject=subject, seed=3300, members=name, **margin_metrics(scores)))
    if combinations:
        pd.DataFrame(combinations).to_csv(destination / 'predeclared_cross_arm_fusions.csv', index=False)
    files = sorted((OUTPUT / 'repetitions').glob('sub-*/curves.csv'))
    if files:
        frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        frame.to_csv(destination / 'repetition_draws.csv', index=False)
        folds = frame.groupby(['committee', 'repetitions', 'subject']).mean(numeric_only=True).reset_index()
        folds.to_csv(destination / 'repetition_folds.csv', index=False)
        folds.groupby(['committee', 'repetitions']).agg(n_subjects=('subject', 'nunique'), top1=('top1', 'mean'),
            distractor_bonus=('distractor_bonus', 'mean')).to_csv(destination / 'repetition_summary.csv')
    print('Summaries refreshed:', destination, flush=True)


if __name__ == '__main__':
    main()
