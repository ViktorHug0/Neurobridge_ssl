"""Stress test of batch-mean centring on a grader-like stream: each fold's 5 test subjects back to back (trials
random within a subject, as in chronological RSVP), chunked into batches of B regardless of subject boundaries,
so boundary batches mix subjects. Also: batch sizes, image-major order (all repeats of an image together), and
per-dimension batch standardisation.
    centring_stream.py <recipe> [<recipe> ...]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load, score  # noqa: E402


def centre_stream(xs, order_per_subject, B, a, std=False):
    """xs: list over subjects of (n, D); returns centred copies. Batches are cut over the concatenated stream."""
    stream = np.concatenate([x[o] for x, o in zip(xs, order_per_subject)])
    out = np.empty_like(stream)
    for b in range(0, len(stream), B):
        blk = stream[b:b + B]; mu = blk.mean(0)
        out[b:b + B] = (blk - a * mu) / (blk.std(0) + 1e-6) if std else blk - a * mu
    res, i = [], 0
    for x, o in zip(xs, order_per_subject):
        y = np.empty_like(x); y[o] = out[i:i + len(x)]; res.append(y); i += len(x)
    return res


def main(recipes):
    data = load(recipes); M = len(recipes); rng = np.random.default_rng(1)
    res = {}
    for k, subs in FOLDS.items():
        n = len(data[subs[0]]['label'])
        rand = [rng.permutation(n) for _ in subs]
        imgmajor = [np.argsort(data[s]['label'], kind='stable') for s in subs]
        cfgs = {f'stream B={B} a=0.5': (rand, B, 0.5, False) for B in (16, 32, 64, 128, 256)}
        cfgs.update({'stream B=64 a=0.4': (rand, 64, 0.4, False), 'stream B=64 a=0.6': (rand, 64, 0.6, False),
                     'image-major B=64 a=0.5': (imgmajor, 64, 0.5, False), 'stream B=64 standardise a=0.5': (rand, 64, 0.5, True)})
        for name, (order, B, a, std) in cfgs.items():
            cen = [centre_stream([data[s]['emb'][m] for s in subs], order, B, a, std) for m in range(M)]
            for i, s in enumerate(subs):
                res.setdefault(name, {})[s] = score(dict(data[s], emb=[cen[m][i] for m in range(M)]), np.ones((M, n)))
        for s in subs:
            res.setdefault('uniform', {})[s] = score(data[s], np.ones((M, n)))
    subs = sorted(data); base = np.array([res['uniform'][s] for s in subs])
    print(f'{"+".join(recipes)} ({len(subs)} subjects)')
    for name, v in res.items():
        v = np.array([v[s] for s in subs])
        print(f'  {name:32s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')


if __name__ == '__main__':
    main(sys.argv[1:])
