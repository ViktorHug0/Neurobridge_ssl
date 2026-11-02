"""SAGE hyper-parameter ensembling on R1: instead of the single cross-fold config, average the (row z-scored) final
CSLS score matrices of the top-m configs ranked on the OTHER subjects (m = 1 is the published cross-fold SAGE).
Label-free and selection-free beyond the cross-fold ranking. Uses cached 80-rep embeddings and the wide-grid table.
usage: sage_hpens.py --grid_csv G --cache_dir D --output_csv O
"""
import argparse, os, sys
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
from shared import TTAParams, apply_tta_calibration, fit_tta_calibration, score_features  # noqa: E402

CFG = ['saw_shrink', 'sinkhorn_tau', 'soft_procrustes_steps', 'soft_procrustes_power']


def zrows(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--grid_csv', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--output_csv', required=True)
    args = ap.parse_args()
    g = pd.read_csv(args.grid_csv)
    rows = []
    for sid in sorted(g.subject_id.unique()):
        z = np.load(os.path.join(args.cache_dir, f'sub-{sid:02d}.npz'))
        q, gal = z['q'], z['g']
        rank = g[g.subject_id != sid].groupby(CFG)['test_top1'].mean().sort_values(ascending=False).index
        mats = []
        for c in rank[:20]:
            p = replace(TTAParams(), **dict(zip(CFG, (float(c[0]), float(c[1]), int(c[2]), float(c[3])))))
            mats.append(zrows(score_features(apply_tta_calibration(q, fit_tta_calibration(q, gal, p)), gal, use_csls=True, csls_k=p.csls_k)))
        for m in (1, 3, 5, 10, 20):
            rows.append({'subject': sid, 'm': m, 'top1': 100 * np.mean(np.mean(mats[:m], 0).argmax(1) == np.arange(len(q)))})
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.groupby('m')['top1'].mean().round(2).to_string())


if __name__ == '__main__':
    main()
