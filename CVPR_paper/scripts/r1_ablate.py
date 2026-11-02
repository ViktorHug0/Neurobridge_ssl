"""E2.4 / E2.5 / E2.6 / E3.6 / E2.1-alpha on regime R1 (200 x 200), one model, per subject cross-fold SAGE params.

Every variant changes ONE thing of the SAGE loop (whiten -> CSLS -> plan -> weighted Procrustes, x steps):
  full            SAGE as published (= r1_eval.py 'sage')
  -whiten         no SAW (queries only L2-normalised)        -csls      cosine instead of CSLS (loop and final), k unchanged
  -procrustes     SAW + CSLS only (0 steps)                  1step      a single Sinkhorn-Procrustes step
  plan_<p>        the Sinkhorn plan replaced by: softmax (rows only, same tau), top5 (softmax over each row's top 5),
                  argmax (hard row NN), hungarian (hard bijection), mnn (hard mutual-NN landmarks, SCORE's estimator)
  mnn_1step       SCORE's estimator AND one step, inside SAGE's whitening (isolates iterated soft vs one-step hard)
  a<alpha>        the final rotation shrunk towards the identity (geodesic chord blend), alpha in {0.4, 0.6, 0.8}
  <x>+hung        decoding by a Hungarian bijection on the final scores (x in plain, csls3, full)
--reps_sweep instead: q-repetition queries (disjoint blocks, up to 4 per q), methods plain / csls3 / score / full.
usage: r1_ablate.py --source_run_dir R --params_csv C --output_csv O [--subjects ...] [--reps_sweep]
"""
import argparse, csv, os, sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import (apply_tta_calibration, blend_orthogonal_map, encode_repetition_blocks, fit_saw_transform,  # noqa: E402
                    fit_tta_calibration, load_subject_context, normalize_rows, score_features)
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes, sinkhorn_normalize  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402


def plan(scores, kind, p):
    if kind == 'sinkhorn':
        return sinkhorn_normalize(scores, tau=p.sinkhorn_tau, num_iters=p.sinkhorn_iters)
    if kind in ('softmax', 'top5'):
        s = scores / p.sinkhorn_tau
        if kind == 'top5':
            s = np.where(s >= np.partition(s, -5, axis=1)[:, -5:-4], s, -np.inf)
        e = np.exp(s - s.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    hard = np.zeros_like(scores)
    if kind == 'argmax':
        hard[np.arange(len(scores)), scores.argmax(1)] = 1
    elif kind == 'hungarian':
        r, c = linear_sum_assignment(-scores)
        hard[r, c] = 1
    elif kind == 'mnn':
        fwd = scores.argmax(1)
        mut = np.flatnonzero(scores.argmax(0)[fwd] == np.arange(len(scores)))
        hard[mut, fwd[mut]] = 1
    return hard


def sage(q, g, p, whiten=True, csls=True, kind='sinkhorn', steps=None, alpha=1.0):
    """Returns final scores. Same loop as shared.fit_tta_calibration with one switch flipped."""
    x = fit_saw_transform(q, p)[0] if whiten else normalize_rows(q)
    sc = lambda z: score_features(z, g, use_csls=csls, csls_k=p.csls_k)  # noqa: E731
    total = np.eye(x.shape[1], dtype=np.float32)
    s = sc(x)
    for _ in range(p.soft_procrustes_steps if steps is None else steps):
        m = fit_soft_assignment_procrustes(x, g, plan(s, kind, p), power=p.soft_procrustes_power)
        if m is None:
            break
        x = apply_orthogonal_map(x, m)
        total = total @ m
        s = sc(x)
    if alpha != 1.0:
        x0 = fit_saw_transform(q, p)[0] if whiten else normalize_rows(q)
        s = sc(apply_orthogonal_map(x0, blend_orthogonal_map(total, alpha)))
    return s


def hung_acc(scores):
    r, c = linear_sum_assignment(-scores)
    return 100.0 * np.mean(c[np.argsort(r)] == np.arange(len(scores)))


def ablations(q, g, p):
    out = {'plain': score_features(q, g), 'csls3': score_features(q, g, use_csls=True, csls_k=3)}
    out['full'] = score_features(apply_tta_calibration(q, fit_tta_calibration(q, g, p)), g, use_csls=True, csls_k=p.csls_k)
    out['-whiten'] = sage(q, g, p, whiten=False)
    out['-csls'] = sage(q, g, p, csls=False)
    out['-procrustes'] = sage(q, g, p, steps=0)
    out['1step'] = sage(q, g, p, steps=1)
    for k in ('softmax', 'top5', 'argmax', 'hungarian', 'mnn'):
        out[f'plan_{k}'] = sage(q, g, p, kind=k)
    out['mnn_1step'] = sage(q, g, p, kind='mnn', steps=1)
    for a in (0.4, 0.6, 0.8):
        out[f'a{a:g}'] = sage(q, g, p, alpha=a)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_csv', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--reps_sweep', action='store_true')
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    rows = []
    for sid in args.subjects:
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        y = np.arange(200)
        if args.reps_sweep:
            for q in (1, 5, 10, 20, 40, 60, 80):
                blocks, g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[q] * min(4, 80 // q), seed=3300)
                for b, x in enumerate(blocks):
                    sc = {'plain': score_features(x, g), 'csls3': score_features(x, g, use_csls=True, csls_k=3),
                          'score': score_deploy(x, g),
                          'full': score_features(apply_tta_calibration(x, fit_tta_calibration(x, g, p)), g, use_csls=True, csls_k=p.csls_k)}
                    for m, s in sc.items():
                        a = topk_acc(s, y)
                        rows.append({'subject': sid, 'reps': q, 'block': b, 'method': m, 'top1': a[1], 'top5': a[5]})
        else:
            (x,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
            sc = ablations(x, g, p)
            for m, s in sc.items():
                a = topk_acc(s, y)
                rows.append({'subject': sid, 'reps': 80, 'block': 0, 'method': m, 'top1': a[1], 'top5': a[5]})
            for m in ('plain', 'csls3', 'full'):
                rows.append({'subject': sid, 'reps': 80, 'block': 0, 'method': f'{m}+hung', 'top1': hung_acc(sc[m]), 'top5': np.nan})
        print(sid, 'done', flush=True)
    with open(args.output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    import pandas as pd
    print(pd.DataFrame(rows).groupby(['reps', 'method'])['top1'].agg(['mean', 'std']).round(2).to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selfcheck':  # Hungarian decoding and the plans on a toy problem
        rng = np.random.default_rng(0)
        g = rng.normal(size=(50, 16)).astype(np.float32)
        s = g @ g.T + rng.normal(scale=0.5, size=(50, 50))
        assert hung_acc(s) == 100.0
        p = SimpleNamespace(sinkhorn_tau=0.1, sinkhorn_iters=12)
        for k in ('sinkhorn', 'softmax', 'top5', 'argmax', 'hungarian', 'mnn'):
            P = plan(s.astype(np.float32), k, p)
            assert P.shape == s.shape and np.isfinite(P).all() and (P >= 0).all(), k
        print('selfcheck ok')
    else:
        main()
