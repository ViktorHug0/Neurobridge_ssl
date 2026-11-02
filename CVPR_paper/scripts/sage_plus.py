"""SAGE improvement search on R1 (200 x 200, 80-rep queries, cached embeddings). Every knob changes one thing of the
published loop (SAW -> CSLS -> Sinkhorn -> weighted Procrustes, x steps), and every family's own knob is selected
CROSS-FOLD (best mean over the other subjects), on top of the subject's cross-fold SAGE params (--params_csv).
  white   saw (published) | saw2: SAW on BOTH clouds (gallery whitened with its own covariance, same shrink)
          | coral: queries whitened then re-coloured with the gallery's shrunk covariance and mean (CORAL)
  anneal  Sinkhorn temperature from anneal*tau down to tau, geometrically over the steps (1 = off; softassign-style)
  steps   16 (published) | 32 | 64
  k       CSLS k (published 3)
  ridge   after the loop: one non-orthogonal step W = argmin ||XW - PG||^2 + lam ||W - I||^2 on SAGE's own soft plan
          P, lam = ridge * ||X^T X||_2 (0 = off), then re-scored
usage: sage_plus.py --source_run_dir R --params_csv C --cache_dir D --output_csv O [--subjects ...]
"""
import argparse, itertools, os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import fit_saw_transform, normalize_rows, score_features  # noqa: E402
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize  # noqa: E402
from r1_transductive import embeddings  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402


def shrunk_cov(x, s):
    c = np.cov(x - x.mean(0), rowvar=False)
    return (1 - s) * c + s * np.trace(c) / len(c) * np.eye(len(c))


def powm(c, e):
    w, v = np.linalg.eigh(c)
    return (v * np.clip(w, 1e-8, None) ** e) @ v.T


def prepare(q, g, p, white):
    if white == 'saw':
        return fit_saw_transform(q, p)[0], g
    if white == 'saw2':
        return fit_saw_transform(q, p)[0], fit_saw_transform(g, p)[0]
    s = p.saw_shrink  # coral
    x = (q - q.mean(0)) @ powm(shrunk_cov(q, s), -0.5) @ powm(shrunk_cov(g, s), 0.5) + g.mean(0)
    return normalize_rows(x.astype(np.float32)), g


def sage_plus(q, g, p, white='saw', anneal=1.0, steps=16, k=None, ridge=0.0):
    k = p.csls_k if k is None else k
    x, gx = prepare(q, g, p, white)
    sc = lambda z: score_features(z, gx, use_csls=True, csls_k=k)  # noqa: E731
    s = sc(x)
    for t in range(steps):
        tau = p.sinkhorn_tau * anneal ** (1 - t / max(steps - 1, 1))
        m = fit_soft_assignment_procrustes(x, gx, sinkhorn_normalize(s, tau=tau, num_iters=p.sinkhorn_iters), power=p.soft_procrustes_power)
        if m is None:
            break
        x = apply_orthogonal_map(x, m)
        s = sc(x)
    if ridge > 0:
        P = sinkhorn_normalize(s, tau=p.sinkhorn_tau, num_iters=p.sinkhorn_iters)
        Y = (P / P.sum(1, keepdims=True)) @ normalize_rows(gx)
        xtx = x.T @ x
        lam = ridge * np.linalg.norm(xtx, 2)
        W = np.linalg.solve(xtx + lam * np.eye(len(xtx)), x.T @ Y + lam * np.eye(len(xtx)))
        s = sc(normalize_rows(x @ W))
    return s


FAMILIES = {  # family -> list of (variant name, kwargs); 'published' is in every family as its no-change option
    'white': [('saw', {}), ('saw2', dict(white='saw2')), ('coral', dict(white='coral'))],
    'anneal': [(f'anneal{a:g}', dict(anneal=a)) for a in (1, 2, 4, 8)],
    'steps': [(f'steps{s}', dict(steps=s)) for s in (16, 32, 64)],
    'k': [(f'k{k}', dict(k=k)) for k in (1, 3, 5, 10)],
    'ridge': [(f'ridge{r:g}', dict(ridge=r)) for r in (0, 0.03, 0.1, 0.3, 1.0)],
    'combo': [(f'combo_{w}_a{a:g}', dict(white=w, anneal=a)) for w, a in itertools.product(('saw', 'coral'), (1, 2, 4))],
}


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
        p = subject_params(args.params_csv, sid)
        y = np.arange(len(q))
        for fam, variants in FAMILIES.items():
            for name, kw in variants:
                rows.append({'subject': sid, 'family': fam, 'variant': name, 'top1': topk_acc(sage_plus(q, g, p, **kw), y)[1]})
        print(sid, 'done', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output_csv, index=False)
    print(df.groupby(['family', 'variant'])['top1'].mean().round(2).to_string())
    for fam, d in df.groupby('family'):  # cross-fold selection within the family
        t = d.pivot(index='subject', columns='variant', values='top1')
        cv = [t.loc[s, t.drop(index=s).mean().idxmax()] for s in t.index] if len(t) > 1 else [np.nan]
        print(f'{fam:7s} cross-fold {np.mean(cv):.2f}')


if __name__ == '__main__':
    main()
