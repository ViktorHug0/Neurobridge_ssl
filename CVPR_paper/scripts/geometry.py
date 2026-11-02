"""E8.3 geometry diagnostics of SAGE on R1 (200 x 200), per subject, cross-fold hyper-parameters.

  angle_*      principal rotation angles of the fitted orthogonal map R (eigenvalue phases, degrees): mean, median,
               fraction of the spectrum rotated by > 30 deg, and the largest angle
  pair_cos_*   mean cosine of each query with ITS OWN image (true pairs): raw, after SAW, after SAW + R
  gap_*        mean (true-pair cosine - mean cosine to the other 199 images): raw / SAW / SAGE
  hub_skew_*   skewness of the gallery k-occurrence N_5 (Radovanovic et al. 2010) under plain cosine / CSLS / SAGE
  hub_max_*    largest N_5 (how many queries rank one image in their top 5)
  top1_*       plain / SAGE top-1, for correlating the diagnostics with the gain
Embeddings from r1_transductive.embeddings (cached 80-repetition queries, seed 3300).
usage: geometry.py --source_run_dir R --params_csv C --cache_dir D --output_csv O [--subjects ...]
"""
import argparse, os, sys

import numpy as np
import pandas as pd
from scipy.stats import skew

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import apply_saw_transform, apply_tta_calibration, fit_tta_calibration, normalize_rows, score_features  # noqa: E402
from r1_transductive import embeddings  # noqa: E402
from stream_grid import subject_params  # noqa: E402


def k_occurrence(scores, k=5):
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    return np.bincount(top.ravel(), minlength=scores.shape[1])


def diagnostics(q, g, p):
    cal = fit_tta_calibration(q, g, p)
    phases = np.degrees(np.abs(np.angle(np.linalg.eigvals(cal['orthogonal_map'].astype(np.float64)))))
    gn = normalize_rows(g)
    views = {'raw': normalize_rows(q), 'saw': apply_saw_transform(q, cal['saw_stats']), 'sage': apply_tta_calibration(q, cal)}
    out = {'angle_mean': phases.mean(), 'angle_median': np.median(phases), 'angle_frac30': (phases > 30).mean(),
           'angle_max': phases.max()}
    for name, x in views.items():
        c = normalize_rows(x) @ gn.T
        d = np.diag(c)
        out[f'pair_cos_{name}'] = d.mean()
        out[f'gap_{name}'] = (d - (c.sum(1) - d) / (c.shape[1] - 1)).mean()
    scores = {'plain': score_features(q, g), 'csls': score_features(q, g, use_csls=True, csls_k=p.csls_k),
              'sage': score_features(views['sage'], g, use_csls=True, csls_k=p.csls_k)}
    for name, s in scores.items():
        n5 = k_occurrence(s)
        out[f'hub_skew_{name}'], out[f'hub_max_{name}'] = skew(n5), n5.max()
    y = np.arange(len(q))
    out['top1_plain'] = 100 * np.mean(scores['plain'].argmax(1) == y)
    out['top1_sage'] = 100 * np.mean(scores['sage'].argmax(1) == y)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        q, g = embeddings(args.source_run_dir, sid, args.cache_dir)
        rows.append({'subject': sid, **diagnostics(q, g, subject_params(args.params_csv, sid))})
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.mean(numeric_only=True).round(3).to_string())
    gain = df.top1_sage - df.top1_plain
    for c in ('angle_mean', 'angle_frac30', 'hub_skew_plain', 'gap_raw', 'top1_plain'):
        print(f'corr(gain, {c}) = {np.corrcoef(gain, df[c])[0, 1]:+.2f}')


if __name__ == '__main__':
    main()
