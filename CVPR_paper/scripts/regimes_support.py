"""E4.5 SAGE with label-free SUPPORT ESTIMATION for the regimes that break its bijective prior (R3 partial coverage,
R4 open gallery). SAGE assumes every gallery item is present; here the gallery is first pruned to the items SCORE's
robust map (MNN landmarks, one Procrustes step) puts at the top of some query, then SAGE runs on that support only:
  sage_s1   support = union of SCORE top-1 items          sage_s2   support = union of SCORE top-2 items
  sage_s1f  sage_s1 with a free column marginal (rho = 0, multiplicity allowed)
  sage_it1  start from the top-1 support, then twice: fit SAGE on the support, re-score the FULL gallery under the
            fitted map, support = union of its top-1 items (wrongly pruned items can re-enter; top-2 re-entry let
            distractors back in: 24.2 vs 34.0 at D = 1654 on subjects 1, 3)
  sage_hung support = the N items a rectangular Hungarian on SCORE's scores gives the N queries (assumes distinct
            concepts per query: exact in R4, wrong under R3 multiplicity)
  sage_auto sage_s1 if |top-2 support| < 0.9 |gallery| else plain SAGE (label-free regime switch)
  sage_hung_csls  as sage_hung, Hungarian on SAGE's own first scores (whitening + CSLS, no rotation): no SCORE involved
            (v2 paper: our SCORE reimplementation is not reported until its authors validate it)
Same queries, distractors and draws as regimes.py (R3: 3 draws per level, both laws; R4: all D).
usage: regimes_support.py --source_run_dir R --params_csv C --output_dir O [--subjects ...]
"""
import argparse, csv, os, sys
from types import SimpleNamespace

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
sys.path.insert(0, os.path.dirname(__file__))
from shared import encode_repetition_blocks, load_subject_context  # noqa: E402
from run_partial_coverage import damped_assign, tta_scores, topk_acc  # noqa: E402
from shared import fit_saw_transform, score_features  # noqa: E402
from module.util import apply_orthogonal_map, fit_soft_assignment_procrustes  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from regimes import D_LIST, U_LIST, r3_stream  # noqa: E402


def on_support(q, g, params, support, rho):
    """SAGE on gallery[support]; scores scattered back to the full gallery (-inf elsewhere)."""
    s = np.full((len(q), len(g)), -np.inf, dtype=np.float32)
    s[:, support] = tta_scores(q, g[support], params, rho=rho)
    return s


def fit_queries(q, g, params, rho=1.0):
    """tta_scores' loop, returning the mapped queries (so any gallery can be scored under the fitted map)."""
    x, _ = fit_saw_transform(q, params)
    s = score_features(x, g, use_csls=params.use_csls, csls_k=params.csls_k)
    for _ in range(max(1, int(params.soft_procrustes_steps))):
        m = fit_soft_assignment_procrustes(x, g, damped_assign(s, params.sinkhorn_tau, params.sinkhorn_iters, rho),
                                           power=params.soft_procrustes_power,
                                           normalize_inputs=params.soft_procrustes_normalize_inputs)
        if m is None:
            break
        x = apply_orthogonal_map(x, m)
        s = score_features(x, g, use_csls=params.use_csls, csls_k=params.csls_k)
    return x


def sage_iterative(q, g, params, support, rounds=2, top=1):
    for _ in range(rounds):
        x = fit_queries(q, g[support], params)
        full = score_features(x, g, use_csls=params.use_csls, csls_k=params.csls_k)
        support = np.unique(np.argsort(-full, axis=1)[:, :top])
    return on_support(q, g, params, support, 1.0)


def sage_hung_iterative(q, g, params, sc, rounds=2):
    """Hungarian support, then re-run the rectangular Hungarian on the FULL gallery under the fitted map."""
    support = np.sort(linear_sum_assignment(-sc)[1])
    for _ in range(rounds):
        x = fit_queries(q, g[support], params)
        support = np.sort(linear_sum_assignment(-score_features(x, g, use_csls=params.use_csls, csls_k=params.csls_k))[1])
    return on_support(q, g, params, support, 1.0)


def methods(q, g, params, r4_only=False):
    x, _ = fit_saw_transform(q, params)
    cs = score_features(x, g, use_csls=params.use_csls, csls_k=params.csls_k)
    hung_csls = on_support(q, g, params, np.sort(linear_sum_assignment(-cs)[1]), 1.0)
    if r4_only:
        return {'sage': tta_scores(q, g, params, rho=1.0), 'sage_hung_csls': hung_csls}, np.arange(1), np.arange(1)
    sc = score_deploy(q, g)
    top2 = np.argsort(-sc, axis=1)[:, :2]
    s1, s2 = np.unique(top2[:, 0]), np.unique(top2)
    return {'score': sc, 'sage': tta_scores(q, g, params, rho=1.0),
            'sage_s1': on_support(q, g, params, s1, 1.0), 'sage_s2': on_support(q, g, params, s2, 1.0),
            'sage_s1f': on_support(q, g, params, s1, 0.0), 'sage_it1': sage_iterative(q, g, params, s1),
            # distinct-concept support: rectangular Hungarian on SCORE's scores (one gallery item per query)
            'sage_hung': on_support(q, g, params, np.sort(linear_sum_assignment(-sc)[1]), 1.0),
            'sage_hung_it': sage_hung_iterative(q, g, params, sc), 'sage_hung_csls': hung_csls,
            # label-free switch: prune only when SCORE's top-2 items leave >= 10% of the gallery unused
            'sage_auto': on_support(q, g, params, s1, 1.0) if len(s2) < 0.9 * len(g) else tta_scores(q, g, params, rho=1.0)}, s1, s2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_run_dir', required=True)
    p.add_argument('--params_csv', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--r4_only', action='store_true', help='open gallery only, SAGE and sage_hung_csls only')
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'support_rows.csv')
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['subject', 'regime', 'law', 'level', 'draw', 'n_queries', 'n_present', 'support1', 'support2', 'recall1', 'recall2',
                    'method', 'top1'])
        for sid in args.subjects:
            params = subject_params(args.params_csv, sid)
            rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
            _, ea, ds, mods = load_subject_context(os.path.abspath(args.source_run_dir), rt, sid, average=False)
            blocks, g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[10] * 8, seed=2000 + sid)
            (q80,), _, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
            for law in (() if args.r4_only else ('uniform', 'dirichlet')):
                for U in U_LIST:
                    for d in range(args.draws):
                        q, y = r3_stream(blocks, U, law, np.random.default_rng(100_000 * sid + 1000 * U + d + (0 if law == 'uniform' else 7)))
                        ms, s1, s2 = methods(q, g, params)
                        rc = [100 * np.isin(np.unique(y), x).mean() for x in (s1, s2)]
                        for m, s in ms.items():
                            w.writerow([sid, 'R3', law, U, d, len(q), U, len(s1), len(s2), *rc, m, topk_acc(s, y)[1]])
            raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy')).astype(np.float32)
            raw = raw[:, 0] if raw.ndim == 4 else raw.reshape(1654, -1, raw.shape[-1])[:, 0]
            distractors = project_raw_images(ea, mods, raw[np.random.default_rng(sid).permutation(1654)])
            for D in D_LIST:
                ms, s1, s2 = methods(q80, np.concatenate([g, distractors[:D]], axis=0), params, args.r4_only)
                rc = [100 * np.isin(np.arange(200), x).mean() for x in (s1, s2)]
                for m, s in ms.items():
                    w.writerow([sid, 'R4', 'open', D, 0, 200, 200, len(s1), len(s2), *rc, m, topk_acc(s, np.arange(200))[1]])
            f.flush()
            print(f'subject {sid:02d} done', flush=True)


if __name__ == '__main__':
    main()
