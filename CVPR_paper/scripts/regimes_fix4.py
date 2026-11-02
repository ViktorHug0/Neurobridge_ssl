"""E4.3/E4.4 fix search v4: capacity-constrained correspondence ("one-to-one onto an unknown subset").

Balanced Sinkhorn forces every gallery item to receive equal mass (wrong with distractors or absent concepts); the
free plan (rows only) drops the prior. Capacity-constrained Sinkhorn keeps rows fully assigned and only caps columns:
column mass <= cap, cap = kappa * max(1, N/K) with N/K observable. With kappa = 1 it equals balanced SAGE whenever
N = K * cap (R1; R3 at full coverage); in R4 (N < K) it says "each image takes at most one query".
Grid: whitening saw | mm  x  kappa 1 | 1.5 | 2 | 4  x  steps 4 | 16 (reg 0). Same problems/seeds as regimes_fix.py.
--family uot: the soft version, unbalanced Sinkhorn (Chizat et al. 2018) with a KL column penalty: rows exact, columns
scaled by (N/K / colsum)^rho; rho = 1 is balanced SAGE, rho = 0 the free (rows-only) plan. rho 0.25|0.5|0.75|0.9.
usage: regimes_fix4.py --source_run_dir R --params_csv C --output_dir O [--subjects ...] [--draws 2] [--family uot]
"""
import argparse, csv, itertools, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from regimes_fix import procrustes, whiten  # noqa: E402  (also puts tta_rebuttal on sys.path)
from shared import encode_repetition_blocks, load_subject_context, score_features  # noqa: E402
from module.util import apply_orthogonal_map  # noqa: E402
from run_partial_coverage import topk_acc  # noqa: E402
from run_openset_streaming import project_raw_images  # noqa: E402
from label_free import score_deploy  # noqa: E402
from stream_grid import subject_params  # noqa: E402
from regimes import r3_stream  # noqa: E402

GRID = list(itertools.product(['saw', 'mm'], [1.0, 1.5, 2.0, 4.0], [4, 16]))
GRID_UOT = list(itertools.product(['saw', 'mm'], [0.25, 0.5, 0.75, 0.9], [4, 16]))


def cap_sinkhorn(s, tau, cap, iters=50):
    """Rows sum to 1; columns are scaled down only where their mass exceeds cap (an upper-bound marginal)."""
    P = np.exp((s - s.max(1, keepdims=True)) / tau)
    for _ in range(iters):
        P /= P.sum(1, keepdims=True)
        P *= np.minimum(1.0, cap / np.clip(P.sum(0, keepdims=True), 1e-12, None))
    return P / P.sum(1, keepdims=True)


def uot_sinkhorn(s, tau, rho, iters=50):
    """Rows sum to 1; column masses are pulled towards N/K with strength rho (1 = balanced, 0 = free)."""
    K = np.exp((s.astype(np.float64) - s.max()) / tau)
    b, v = len(s) / s.shape[1], np.ones(s.shape[1])
    for _ in range(iters):  # Chizat scaling: v is recomputed from the kernel, not compounded (that converges to balanced)
        u = 1.0 / np.clip(K @ v, 1e-30, None)
        v = (b / np.clip(K.T @ u, 1e-30, None)) ** rho
    P = K * v
    return P / P.sum(1, keepdims=True)


def sage_cap(q, g, p, white, kappa, steps, family='cap'):
    x = whiten(q, g, p, white)
    cap = kappa * max(1.0, len(q) / len(g))
    plan = (lambda s: cap_sinkhorn(s, p.sinkhorn_tau, cap)) if family == 'cap' else (lambda s: uot_sinkhorn(s, p.sinkhorn_tau, kappa))
    s = score_features(x, g, use_csls=True, csls_k=p.csls_k)
    for _ in range(steps):
        x = apply_orthogonal_map(x, procrustes(x, g, plan(s), p.soft_procrustes_power, 0.0))
        s = score_features(x, g, use_csls=True, csls_k=p.csls_k)
    return s


def evaluate(q, y, g, p, family='cap'):
    acc = {'plain': topk_acc(score_features(q, g), y)[1], 'score': topk_acc(score_deploy(q, g), y)[1]}
    for white, kappa, steps in (GRID if family == 'cap' else GRID_UOT):
        acc[f'{family}_{white}_k{kappa:g}_s{steps}'] = topk_acc(sage_cap(q, g, p, white, kappa, steps, family), y)[1]
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source_run_dir', required=True)
    ap.add_argument('--params_csv', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--subjects', nargs='+', type=int, default=list(range(1, 11)))
    ap.add_argument('--draws', type=int, default=2)
    ap.add_argument('--family', choices=['cap', 'uot'], default='cap')
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, 'fix4_rows.csv' if args.family == 'cap' else 'fix4_uot_rows.csv')
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
                for d in range(args.draws):
                    q, y = r3_stream(blocks, U, law, np.random.default_rng(100_000 * sid + 1000 * U + d + (0 if law == 'uniform' else 7)))
                    rows += [(sid, 'R3', law, U, d, m, a) for m, a in evaluate(q, y, g, p, args.family).items()]
        raw = np.load(os.path.join(ea.image_feature_dir, 'image_train.npy')).astype(np.float32)
        raw = raw.reshape(1654, -1, raw.shape[-1])[:, 0]
        distractors = project_raw_images(ea, mods, raw[np.random.default_rng(sid).permutation(1654)])
        for D in (0, 200, 800, 1654):
            gal = np.concatenate([g, distractors[:D]], axis=0)
            rows += [(sid, 'R4', 'open', D, 0, m, a) for m, a in evaluate(q80, np.arange(200), gal, p, args.family).items()]
        w.writerows(rows)
        f.flush()
        print(f'subject {sid:02d} done', flush=True)
    f.close()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selfcheck':  # cap=1 with N=K must equal balanced Sinkhorn's argmax
        from module.util import sinkhorn_normalize
        s = np.random.default_rng(0).normal(size=(50, 50)).astype(np.float32)
        a, b = cap_sinkhorn(s, 0.1, 1.0, iters=200), sinkhorn_normalize(s, tau=0.1, num_iters=200)
        assert (a.argmax(1) == b.argmax(1)).all() and np.abs(a - b).max() < 2e-2, np.abs(a - b).max()  # slower convergence
        s2 = np.random.default_rng(1).normal(size=(20, 60)).astype(np.float32)
        P = cap_sinkhorn(s2, 0.1, 1.0, iters=200)
        assert np.allclose(P.sum(1), 1) and P.sum(0).max() <= 1 + 1e-2
        u = uot_sinkhorn(s, 0.1, 1.0, iters=200)  # rho = 1 is balanced Sinkhorn
        assert (u.argmax(1) == b.argmax(1)).all() and np.abs(u - b).max() < 2e-2
        f = uot_sinkhorn(s, 0.1, 0.0)  # rho = 0 is the row softmax
        assert np.allclose(f, np.exp(s / 0.1) / np.exp(s / 0.1).sum(1, keepdims=True), atol=1e-5)
        c = [uot_sinkhorn(s, 0.1, r, iters=200).sum(0).std() for r in (0.0, 0.5, 1.0)]  # column imbalance shrinks with rho
        assert c[0] > c[1] > c[2] + 1e-3, c
        print('selfcheck ok')
    else:
        main()
