"""Subject-affinity datamodel (affinity.sh). Per model, one epoch: the best mean over its 15 held-out targets.
For each target j, regress its score on training-set membership over the models that held j out:
    score[m, j] = b_j + sum_i a[i, j] * [i in S_m]         (least squares, small ridge)
a[i, j] = how much having source i in a 5-subject training set moves target j, data size held fixed.
Is it real? Split-half reliability: fit a on two disjoint halves of the models (5 permutations each) and correlate,
raw and double-centred (the interaction left after removing each source's and target's average effect --
the part an affinity router could exploit). Usage: affinity_summary.py [--self_test]"""
import argparse, glob, json, re
import numpy as np

ROOT = 'results/alljoined/track1/affinity'


def load():
    subsets = [list(map(int, l.split())) for l in open('neurips_challenge/affinity_subsets.txt') if not l.startswith('#')]
    scores = np.full((len(subsets), 21), np.nan)
    for f in glob.glob(f'{ROOT}/s*/reve/2026*/sealed_per_epoch.jsonl'):
        m = int(re.search(r'/s(\d+)/', f).group(1)) - 1
        ev = [json.loads(l)['by_subject'] for l in open(f)]
        best = ev[int(np.argmax([np.mean(list(e.values())) for e in ev]))]
        for s, v in best.items():
            scores[m, int(s)] = v
    return subsets, scores


def fit(subsets, scores, models, ridge=1.0):
    a = np.full((21, 21), np.nan)  # a[source, target]
    for j in range(1, 21):
        rows = [m for m in models if j not in subsets[m] and not np.isnan(scores[m, j])]
        if len(rows) < 8:
            continue
        src = [i for i in range(1, 21) if i != j]
        X = np.array([[i in subsets[m] for i in src] for m in rows], float)
        X = np.hstack([np.ones((len(rows), 1)), X - X.mean(0)])
        y = scores[rows, j]
        reg = ridge * np.eye(X.shape[1]); reg[0, 0] = 0
        a[src, j] = np.linalg.solve(X.T @ X + reg, X.T @ y)[1:]
    return a


def double_centre(a):
    c = a - np.nanmean(a, axis=1, keepdims=True)
    return c - np.nanmean(c, axis=0, keepdims=True)


def reliability(subsets, scores):
    h1, h2 = list(range(0, 20)), list(range(20, 40))  # permutations 1-5 / 6-10: each half is itself balanced
    a1, a2 = fit(subsets, scores, h1), fit(subsets, scores, h2)
    out = {}
    for name, f in (('raw', lambda a: a), ('interaction', double_centre)):
        x, y = f(a1)[1:, 1:].ravel(), f(a2)[1:, 1:].ravel()
        ok = ~np.isnan(x) & ~np.isnan(y)
        out[name] = np.corrcoef(x[ok], y[ok])[0, 1]
    return out


def main():
    subsets, scores = load()
    done = int((~np.isnan(scores).all(1)).sum())
    print(f'{done}/40 models scored')
    if done < 40:
        return
    a = fit(subsets, scores, range(40))
    rel = reliability(subsets, scores)
    src_effect = np.nanmean(a, axis=1)[1:]
    print('split-half reliability of a[i, j]: raw %.2f, interaction (double-centred) %.2f' % (rel['raw'], rel['interaction']))
    print('source main effects (mean effect of adding source i on held-out targets):')
    print('  ' + '  '.join(f'{i}:{e:+.1f}' for i, e in sorted(zip(range(1, 21), src_effect), key=lambda t: -t[1])))
    inter = double_centre(a)
    print('largest interactions (source -> target, points beyond both main effects):')
    flat = [(inter[i, j], i, j) for i in range(1, 21) for j in range(1, 21) if not np.isnan(inter[i, j])]
    for v, i, j in sorted(flat, key=lambda t: -abs(t[0]))[:10]:
        print(f'  {i:2d} -> {j:2d}: {v:+.1f}')
    np.save(f'{ROOT}/affinity_matrix.npy', a)


def self_test():
    rng = np.random.default_rng(0)
    subsets = [list(map(int, l.split())) for l in open('neurips_challenge/affinity_subsets.txt') if not l.startswith('#')]
    true = rng.normal(0, 3, (21, 21))  # planted per-pair affinity
    scores = np.full((40, 21), np.nan)
    for m, S in enumerate(subsets):
        for j in range(1, 21):
            if j not in S:
                scores[m, j] = 20 + sum(true[i, j] for i in S) + rng.normal(0, 1)
    a = fit(subsets, scores, range(40))
    ok = ~np.isnan(a)
    assert np.corrcoef(a[ok], true[ok])[0, 1] > 0.8, 'planted affinity not recovered'
    assert reliability(subsets, scores)['interaction'] > 0.5
    noise = np.where(np.isnan(scores), np.nan, 20 + rng.normal(0, 3, scores.shape))  # no affinity at all
    assert abs(reliability(subsets, noise)['interaction']) < 0.3, 'reliability must be ~0 without structure'
    print('affinity_summary self-test ok')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--self_test', action='store_true')
    self_test() if p.parse_args().self_test else main()
