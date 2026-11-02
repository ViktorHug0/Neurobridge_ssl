"""E4.3/E4.4 fix search: a robust SAGE for partial coverage (R3) and open galleries (R4) that keeps R1.

regimes.py showed SAGE (balanced Sinkhorn, 16 free Procrustes steps) collapses below plain when the bijection is false
(R3 U<=100, R4 D>=200) while SCORE (confident mutual-NN landmarks, identity-regularised, one step) stays positive.
Variant grid, same queries as regimes.py (R3: 10-rep blocks, R4/R1: 80-rep, D = 0 is R1):
  white  saw = SAGE's shrinkage whitening of the query batch (published); mm = SCORE's per-dimension moment matching
         to the gallery (Eq. 3); ctr = centring only. (A partial-coverage batch's covariance spans only the present
         concepts, so full whitening may be what breaks SAGE in R3.)
  plan   bal = balanced Sinkhorn (published); free = rows-only softmax; mnn = hard mutual-NN landmarks with SCORE's
         margin weights (SCORE's estimator, re-estimated at every step)
  reg    identity regulariser inside EVERY Procrustes step, lambda = reg * ||C||_2 (SCORE Eq. 6); 0 = published
  steps  1 / 4 / 16
(v1 of this grid also had a top-k CSLS gallery shortlist: no gain on sub-01, dropped.)
plus references: plain, csls3, score, and 'oracle_support' (published SAGE on the true present concepts, a ceiling).
SAGE's whitening/tau/power/csls_k: the subject's cross-fold R1 params (--params_csv). Writes fix_rows.csv; the
selection over variants is cross-fold (other subjects, mean over ALL regime cells) in the summary.
usage: regimes_fix.py --source_run_dir R --params_csv C --output_dir O [--subjects ...] [--draws 2]
"""
import argparse, csv, itertools, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import encode_repetition_blocks, fit_saw_transform, load_subject_context, normalize_rows, score_features  # noqa: E402
from module.util import apply_orthogonal_map, sinkhorn_normalize  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from regimes import r3_stream  # noqa: E402

GRID = list(itertools.product(['saw', 'mm', 'ctr'], ['bal', 'free', 'mnn'], [0.0, 0.1], [1, 4, 16]))


def procrustes(x, g, plan, power, reg):
    w = np.power(np.clip(plan, 0, None), power)
    c = x.T @ w @ g
    if reg > 0:
        c = c + reg * np.linalg.norm(c, 2) * np.eye(c.shape[0], dtype=c.dtype)
    u, _, vt = np.linalg.svd(c)
    return u @ vt


def whiten(q, g, p, kind):
    if kind == 'saw':
        return fit_saw_transform(q, p)[0]
    if kind == 'mm':
        return normalize_rows((q - q.mean(0)) / (q.std(0) + 1e-6) * (g.std(0) + 1e-6) + g.mean(0))
    return normalize_rows(q - q.mean(0))


def mnn_plan(s):
    fwd = s.argmax(1)
    mut = np.flatnonzero(s.argmax(0)[fwd] == np.arange(len(s)))
    P = np.zeros_like(s)
    if len(mut):
        top2 = np.sort(s[mut], axis=1)[:, -2:]
        P[mut, fwd[mut]] = np.maximum(top2[:, 1] - top2[:, 0], 1e-6)
    return P


def sage_variant(q, g, p, white='saw', plan='bal', reg=0.0, steps=16, keep=None):
    """Scores over the WHOLE gallery (columns outside the oracle support are -inf)."""
    x = whiten(q, g, p, white)
    cols = np.arange(len(g)) if keep is None else keep
    gs = g[cols]
    sc = lambda z: score_features(z, gs, use_csls=True, csls_k=p.csls_k)  # noqa: E731
    s = sc(x)
    for _ in range(steps):
        if plan == 'bal':
            P = sinkhorn_normalize(s, tau=p.sinkhorn_tau, num_iters=p.sinkhorn_iters)
        elif plan == 'mnn':
            P = mnn_plan(s)
            if not P.any():
                break
        else:
            e = np.exp((s - s.max(1, keepdims=True)) / p.sinkhorn_tau)
            P = e / e.sum(1, keepdims=True)
        x = apply_orthogonal_map(x, procrustes(x, gs, P, p.soft_procrustes_power, reg))
        s = sc(x)
    out = np.full((len(q), len(g)), -np.inf, dtype=np.float32)
    out[:, cols] = s
    return out


def evaluate(q, y, g, p):
    res = {'plain': score_features(q, g), 'csls3': score_features(q, g, use_csls=True, csls_k=3), 'score': score_deploy(q, g),
           'oracle_support': sage_variant(q, g, p, keep=np.unique(y))}
    for white, plan, reg, steps in GRID:
        res[f'{white}_{plan}_r{reg:g}_s{steps}'] = sage_variant(q, g, p, white, plan, reg, steps)
    acc = {m: topk_acc(s, y)[1] for m, s in res.items()}
    # label-free regime statistics for a gate (stored in the top1 column under 'stat_*' method names):
    init = score_features(fit_saw_transform(q, p)[0], g, use_csls=True, csls_k=p.csls_k)
    cnt = np.bincount(init.argmax(1), minlength=len(g)).astype(float)
    pr = cnt[cnt > 0] / cnt.sum()
    acc['stat_hnorm'] = float(-(pr * np.log(pr)).sum() / np.log(len(g)))  # argmax-histogram entropy / log K
    acc['stat_uniq'] = float((cnt > 0).mean())  # fraction of gallery items that win >= 1 query
    acc['stat_nk'] = len(q) / len(g)  # batch size over gallery size
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--draws', type=int, default=2)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'fix_rows.csv')
    done = {int(r['subject']) for r in csv.DictReader(open(out))} if os.path.exists(out) else set()
    f = open(out, 'a', newline='')
    w = csv.writer(f)
    if not done:
        w.writerow(['subject', 'regime', 'law', 'level', 'draw', 'method', 'top1'])
    for sid in args.subjects:
        if sid in done:
            continue
        p = subject_params(args.params_csv, sid)
        rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
        _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
        blocks, g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[10] * 8, seed=2000 + sid)
        (q80,), _, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
        rows = []
        for law in ('uniform', 'dirichlet'):
            for U in (25, 50, 100, 200):
                for d in range(args.draws):  # same seeds as regimes.py
                    q, y = r3_stream(blocks, U, law, np.random.default_rng(100_000 * sid + 1000 * U + d + (0 if law == 'uniform' else 7)))
                    rows += [(sid, 'R3', law, U, d, m, a) for m, a in evaluate(q, y, g, p).items()]
        raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy')).astype(np.float32)
        raw = raw.reshape(1654, -1, raw.shape[-1])[:, 0]
        distractors = project_raw_images(ea, mods, raw[np.random.default_rng(sid).permutation(1654)])
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, distractors[:D]], axis=0)
            rows += [(sid, 'R4', 'open', D, 0, m, a) for m, a in evaluate(q80, np.arange(200), gal, p).items()]
        w.writerows(rows)
        f.flush()
        print(f'subject {sid:02d} done', flush=True)
    f.close()


if __name__ == '__main__':
    main()
