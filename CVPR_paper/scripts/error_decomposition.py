"""E8.4 encoder-gain diagnostics: why do equal zero-shot encoders gain differently from transductive calibration?
R1 (200 x 200), cached 80-repetition embeddings (emb_cache/<arm>_b from r1_transductive, else emb_cache/<arm>), per
subject. Label-free statistics of the cosine score matrix S (query x image), plus label-using nulls (analysis only):
  nn / rev_nn   row-wise (query -> image) and column-wise (image -> query) nearest-neighbour top-1
  mnn_frac      fraction of queries in a mutual nearest-neighbour pair; mnn_prec = precision of those pairs (= the
                quality of SCORE's landmarks)
  colcent       NN after removing each image's mean score (per-image additive offsets, i.e. hubness)
  hungarian     bijective assignment on the raw cosines (removes row + column additive effects; no geometry change)
  hung_rowshuf  Hungarian after permuting each row's off-diagonal entries (rows, hence NN, kept; columns broken)
  hung_colshuf  same per column (columns, hence rev_nn, kept; rows broken)
  reciprocity   corr(S_ij - S_ii, S_ji - S_jj) over pairs i < j (are confusions mutual?)
  rsa           corr of the off-diagonal query-query and image-image cosine matrices (relational structure)
  pr_q / pr_g   participation ratio (effective rank) of the query / image clouds
  hub_skew      skewness of the image k-occurrence N_5 under plain cosine
Per-arm method accuracies are joined from final/<arm>/{r1_competitors,sattc,r1_transductive}.csv.
usage: error_decomposition.py --arms atm128_s3300 tsconv_s3300 ... [--output_csv O]
"""
import argparse, os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew

C = 'results/things_eeg/cvpr/emb_cache'
F = 'results/things_eeg/cvpr/final'


def nrm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def acc(s):
    return 100 * (s.argmax(1) == np.arange(len(s))).mean()


def hung(s):
    r, c = linear_sum_assignment(-s)
    return 100 * (c[np.argsort(r)] == np.arange(len(s))).mean()


def shuffle_rows(s, rng):
    t, n = s.copy(), len(s)
    for i in range(n):
        idx = np.delete(np.arange(n), i)
        t[i, idx] = s[i, rng.permutation(idx)]
    return t


def pr(x):
    e = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    return e.sum() ** 2 / (e ** 2).sum()


def diagnostics(q, g, rng, n_null=5):
    s = nrm(q) @ nrm(g).T
    n, y = len(s), np.arange(len(s))
    fwd, bwd = s.argmax(1), s.argmax(0)
    mnn = bwd[fwd] == y
    iu = np.triu_indices(n, 1)
    m = s - np.diag(s)[:, None]
    qq, gg = nrm(q - q.mean(0)), nrm(g - g.mean(0))
    top5 = np.argpartition(-s, 5, axis=1)[:, :5]
    return dict(nn=acc(s), rev_nn=100 * (bwd == y).mean(), mnn_frac=100 * mnn.mean(),
                mnn_prec=100 * (fwd[mnn] == y[mnn]).mean() if mnn.any() else np.nan,
                colcent=acc(s - s.mean(0, keepdims=True)), hungarian=hung(s),
                hung_rowshuf=np.mean([hung(shuffle_rows(s, rng)) for _ in range(n_null)]),
                hung_colshuf=np.mean([hung(shuffle_rows(s.T, rng).T) for _ in range(n_null)]),
                reciprocity=np.corrcoef(m[iu], m.T[iu])[0, 1],
                rsa=np.corrcoef((qq @ qq.T)[iu], (gg @ gg.T)[iu])[0, 1], pr_q=pr(q), pr_g=pr(g),
                hub_skew=skew(np.bincount(top5.ravel(), minlength=n)))


def methods(arm):
    out = []
    for f, keep in (('r1_competitors.csv', None), ('sattc.csv', ['sattc']), ('r1_transductive.csv', None)):
        p = os.path.join(F, arm, f)
        if os.path.exists(p):
            d = pd.read_csv(p)
            out.append(d[d.method.isin(keep)] if keep else d)
    if not out:
        return pd.DataFrame()
    d = pd.concat(out).groupby(['subject', 'method']).top1.mean().unstack('method')
    return d.add_prefix('m_')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--output_csv', default='results/things_eeg/cvpr/analysis/encoder_gain_diagnostics.csv')
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        cache = next((c for c in (f'{C}/{arm}_b', f'{C}/{arm}') if os.path.exists(f'{c}/sub-01.npz')), None)
        if cache is None:
            print(arm, 'no embedding cache, skipped')
            continue
        meth = methods(arm)
        for sid in range(1, 11):
            f = f'{cache}/sub-{sid:02d}.npz'
            if not os.path.exists(f):
                continue
            z = np.load(f)
            r = dict(arm=arm, subject=sid, **diagnostics(z['q'].astype(np.float64), z['g'].astype(np.float64),
                                                         np.random.default_rng(sid)))
            if sid in meth.index:
                r.update(meth.loc[sid].to_dict())
            rows.append(r)
        print(arm, 'done', flush=True)
    d = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.output_csv), exist_ok=True)
    d.to_csv(a.output_csv, index=False)
    print(d.groupby('arm', sort=False).mean(numeric_only=True).drop(columns='subject').round(2).T.to_string())


if __name__ == '__main__':
    main()
