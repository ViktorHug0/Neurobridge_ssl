"""E8.9 label-free encoder diagnostic: entropic Gromov-Wasserstein between the EEG query cloud and the image gallery
cloud (square loss on centred cosine distances, uniform marginals, Peyre et al. 2016). Needs NO pairing, so it is
computable at test time. Reports per subject: gw (discrepancy, lower = more isometric), gw_acc (top-1 of the GW plan's
row argmax = purely relational matching, uses labels only to score it), plus the arm's method accuracies.
usage: gw_label_free.py --arms ... [--eps 0.002 --iters 60]
"""
import argparse

import numpy as np
import pandas as pd
from scipy.special import logsumexp

C = 'results/things_eeg/cvpr/emb_cache'


def dist(x):
    x = x - x.mean(0)
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return 1 - x @ x.T


def sinkhorn_log(cost, eps, n_iter=200):
    n, m = cost.shape
    la, lb = -np.log(n), -np.log(m)
    f, g = np.zeros(n), np.zeros(m)
    for _ in range(n_iter):
        f = eps * (la - logsumexp((g[None, :] - cost) / eps, axis=1))
        g = eps * (lb - logsumexp((f[:, None] - cost) / eps, axis=0))
    return np.exp((f[:, None] + g[None, :] - cost) / eps)


def entropic_gw(C1, C2, eps, iters):
    n, m = len(C1), len(C2)
    p, q = np.full(n, 1 / n), np.full(m, 1 / m)
    const = (C1 ** 2 @ p)[:, None] + ((C2 ** 2) @ q)[None, :]
    T = np.outer(p, q)
    for _ in range(iters):
        tens = const - 2 * C1 @ T @ C2.T
        T = sinkhorn_log(tens - tens.min(), eps)
    tens = const - 2 * C1 @ T @ C2.T
    return float((tens * T).sum()), T


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--arms', nargs='+', required=True)
    ap.add_argument('--eps', type=float, default=0.002)
    ap.add_argument('--iters', type=int, default=60)
    ap.add_argument('--output_csv', default='results/things_eeg/cvpr/analysis/gw_label_free.csv')
    a = ap.parse_args()
    rows = []
    for arm in a.arms:
        for sid in range(1, 11):
            z = np.load(f'{C}/{arm}_b/sub-{sid:02d}.npz')
            gw, T = entropic_gw(dist(z['q'].astype(float)), dist(z['g'].astype(float)), a.eps, a.iters)
            rows.append(dict(arm=arm, subject=sid, gw=gw, gw_acc=100 * (T.argmax(1) == np.arange(len(T))).mean()))
        print(arm, np.mean([r['gw'] for r in rows[-10:]]).round(5), np.mean([r['gw_acc'] for r in rows[-10:]]).round(1), flush=True)
    pd.DataFrame(rows).to_csv(a.output_csv, index=False)


if __name__ == '__main__':
    # self-check: a rotated copy of a cloud has ~0 GW discrepancy and its plan recovers the pairing
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 16))
    R = np.linalg.qr(rng.normal(size=(16, 16)))[0]
    gw0, T0 = entropic_gw(dist(x), dist(x @ R), 0.002, 60)
    gw1, _ = entropic_gw(dist(x), dist(rng.normal(size=(60, 16))), 0.002, 60)
    assert gw0 < gw1 and (T0.argmax(1) == np.arange(60)).mean() > 0.9, (gw0, gw1)
    main()
