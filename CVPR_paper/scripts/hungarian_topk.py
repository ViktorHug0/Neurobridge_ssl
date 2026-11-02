"""Top-k for the Hungarian assignment of the zero-shot scores (Table 1). Candidate j is ranked for query i by the best
total score of an assignment that sends i to j. Rank 1 is the Hungarian assignment itself. Exact, through the
alternating cycles of the optimal assignment: forcing i -> j costs c(i,j) - c(i,s(i)) + d(j, s(i)), with d the
shortest path in the graph on columns where a -> b costs c(o(a),b) - c(o(a),a) (o(a) = the row assigned to a).
Optimality = no negative cycle, so Johnson's algorithm applies.
usage: hungarian_topk.py <emb_cache dir> [...]      (each dir holds sub-XX.npz with q, g)
"""
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import shortest_path


def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def forced_cost(s):
    c = -s
    r, sig = linear_sum_assignment(c)
    sig = sig[np.argsort(r)]
    own = np.argsort(sig)                                    # own[a] = row assigned to column a
    w = c[own] - c[own, np.arange(len(c))][:, None]           # w[a, b] = c(o(a), b) - c(o(a), a)
    np.fill_diagonal(w, 0)
    d = shortest_path(w, method='J', directed=True)
    return c - c[np.arange(len(c)), sig][:, None] + d[:, sig].T, sig  # [i, j]: c(i,j) - c(i,s(i)) + d(j, s(i))


def demo():  # brute force on a small random problem
    rng = np.random.default_rng(0)
    s = rng.normal(size=(6, 6))
    f, sig = forced_cost(s)
    for i in range(6):
        for j in range(6):
            keep_r, keep_c = np.delete(np.arange(6), i), np.delete(np.arange(6), j)
            sub = -s[np.ix_(keep_r, keep_c)]
            rr, cc = linear_sum_assignment(sub)
            best = -s[i, j] + sub[rr, cc].sum()
            opt = -s[np.arange(6), sig].sum()
            assert abs((best - opt) - f[i, j]) < 1e-9, (i, j)


if __name__ == '__main__':
    demo()
    for cache in sys.argv[1:]:
        t1, t5 = [], []
        for sid in range(1, 11):
            z = np.load(f'{cache}/sub-{sid:02d}.npz')
            f, sig = forced_cost(nrm(z['q'].astype(float)) @ nrm(z['g'].astype(float)).T)
            rank = (f < f[np.arange(200), np.arange(200)][:, None] - 1e-12).sum(1)  # candidates strictly better
            assert (np.argmin(f, 1) == sig).all() or np.allclose(f[np.arange(200), sig], 0)
            t1.append(100 * (sig == np.arange(200)).mean())
            t5.append(100 * (rank < 5).mean())
        print(cache, f'top1 {np.mean(t1):.2f} top5 {np.mean(t5):.2f}')
