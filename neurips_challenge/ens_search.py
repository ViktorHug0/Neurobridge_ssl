"""Ensemble subset search on ens_cache.py caches (exact grader_sim --shuffle scores, official metric, 20 subjects).
Score of a subset = mean over subjects of the official top-5 of the summed per-(subject, image) member outputs.
Reports greedy forward selection, the exhaustive best per size, and an honest estimate of the selection: the subset
picked on folds 1-2 scored on folds 3-4 and vice versa (picking the max of hundreds of subsets on the same 20 subjects
is optimistic).
    ens_search.py <recipe> [...] [--tag fp] [--sizes 3 4 5 6]
"""
import argparse, itertools, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.ens_cache import OUT  # noqa: E402
from neurips_challenge.official_metrics import topk_accuracy  # noqa: E402
from neurips_challenge.trial_routing import FOLDS  # noqa: E402

n = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)


def main(recipes, tag, sizes):
    caches = {r: np.load(f'{OUT}/{r}.npz') for r in recipes}
    keys, G = caches[recipes[0]]['keys'], caches[recipes[0]]['G']
    assert all(np.array_equal(c['keys'], keys) for c in caches.values()), 'members disagree on the (subject, image) keys'
    M = {r: c[tag] for r, c in caches.items()}
    subjects = np.unique(keys[:, 0]); Gn = n(G)
    rows = {s: keys[:, 0] == s for s in subjects}

    def per_subject(members):
        P = n(sum(M[r] for r in members))
        return {s: topk_accuracy(P[m] @ Gn.T, keys[m, 1], k=5) for s, m in rows.items()}

    score = lambda members, subs=subjects: 100 * np.mean([per_subject(members)[s] for s in subs])
    for r in sorted(recipes, key=lambda r: -score([r])):
        print(f'single {r:24s} {score([r]):.2f}')
    # greedy forward selection from the best single
    chosen = [max(recipes, key=lambda r: score([r]))]
    while len(chosen) < max(sizes):
        best = max((r for r in recipes if r not in chosen), key=lambda r: score(chosen + [r]))
        chosen.append(best)
        print(f'greedy {len(chosen)}: {" + ".join(chosen)} = {score(chosen):.2f}', flush=True)
    half = {1: [s for k in (1, 2) for s in FOLDS[k]], 2: [s for k in (3, 4) for s in FOLDS[k]]}
    cache_ps = {}
    for size in sizes:
        combos = list(itertools.combinations(recipes, size))
        for c in combos:
            cache_ps[c] = per_subject(c)
        mean = lambda c, subs: 100 * np.mean([cache_ps[c][s] for s in subs])
        best = max(combos, key=lambda c: mean(c, subjects))
        honest = np.mean([mean(max(combos, key=lambda c: mean(c, half[a])), half[3 - a]) for a in (1, 2)])
        print(f'exhaustive size {size} ({len(combos)} subsets): best {" + ".join(best)} = {mean(best, subjects):.2f} | '
              f'picked on one half, scored on the other: {honest:.2f}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('recipes', nargs='+'); ap.add_argument('--tag', default='fp')
    ap.add_argument('--sizes', type=int, nargs='+', default=[3, 4, 5, 6])
    a = ap.parse_args(); main(a.recipes, a.tag, a.sizes)
