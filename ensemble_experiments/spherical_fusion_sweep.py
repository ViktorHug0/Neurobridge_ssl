"""User-requested descriptive test sweep; does not reselect checkpoints or weights."""
import csv
import numpy as np
from ensemble_experiments.spherical_fusion import OUTPUT, PREVIOUS, DUMPS, unit, slerp


def main():
    rows = []
    for subject in range(1, 11):
        with np.load(OUTPUT/f'sub-{subject:02d}/geometry.npz') as f:
            rotation = f['rotation']
        with np.load(PREVIOUS/f'runs/single/seed3300/sub-{subject:02d}/test_scores.npz') as f:
            q, v = [unit(f[k].astype(np.float64)) for k in ['eeg_0', 'image_0']]
            keys = np.stack([f['object'], f['image_idx']], axis=1)
        with np.load(DUMPS/f'atm_iv_valcon-sub{subject:02d}.npz') as f:
            b, y = [unit(f[k].astype(np.float64)) @ rotation for k in ['eeg', 'image']]
            np.testing.assert_array_equal(keys, np.stack([f['object'], f['image_idx']], axis=1))
        for i in range(1, 10):
            weight = i / 10
            scores = slerp(q, b, weight) @ slerp(v, y, weight).T
            assert scores.shape == (200, 200) and np.isfinite(scores).all()
            if i == 5:
                with np.load(OUTPUT/f'sub-{subject:02d}/test_scores.npz') as f:
                    np.testing.assert_allclose(scores, f['slerp_0.5'], atol=1e-12)
            rows.append(dict(subject=subject, atm_weight=weight,
                             top1=float((scores.argmax(1) == np.arange(200)).mean()*100)))
    summary = [dict(atm_weight=i/10, folds=10,
                    mean_top1=float(np.mean([r['top1'] for r in rows if r['atm_weight']==i/10])))
               for i in range(1, 10)]
    for name, data in [('posthoc_slerp_sweep_folds.csv', rows),
                       ('posthoc_slerp_sweep_summary.csv', summary)]:
        with (OUTPUT/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    print('ATM weight | mean test top1 (%) | validation-selected checkpoints')
    for r in summary: print(f"{r['atm_weight']:.1f} | {r['mean_top1']:.2f}")


if __name__ == '__main__': main()
