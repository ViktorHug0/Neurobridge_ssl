"""Output centring (hubness correction) under the grader's rules. Returning q - a*m instead of q ranks the gallery by
q.g - a*m.g: images the average output already resembles are penalised. Where m comes from decides legality:
  subject   the subject's mean output over all its trials (needs every trial: an upper bound for stateful versions)
  loso      the same model's mean output on the OTHER test subjects of its fold (static: a constant in submission.py)
  refimg    the mean training-image DINOv2 feature direction, scaled like loso (static)
  running   online mean of all outputs seen so far, trials in random order (stateful; no revision of past batches)
  running_imgmajor  the same, trials in test.npy order (all repeats of image 0 first): the worst case for a running mean
  batch64   mean of each 64-trial batch, random order (stateless per call, needs single-subject batches)
    centring.py <recipe> [<recipe> ...]   (several recipes = the ensemble, each member centred on its own m)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, REF, load, score  # noqa: E402


def running_mean(x, order):
    """centre each trial on the mean of the trials returned before its batch (64-trial batches, given order)."""
    out = np.empty_like(x); tot = np.zeros(x.shape[1]); n = 0
    for b in range(0, len(order), 64):
        idx = order[b:b + 64]
        tot += x[idx].sum(0); n += len(idx)   # the batch's own trials are known when it is predicted
        out[idx] = tot / n
    return out


def main(recipes):
    data = load(recipes)
    M = len(recipes)
    subs = sorted(data)
    rng = np.random.default_rng(0)
    fold_of = {s: k for k, v in FOLDS.items() for s in v}
    ref_dir = REF.mean(0); ref_dir /= np.linalg.norm(ref_dir)
    smean = {s: [x.mean(0) for x in data[s]['emb']] for s in subs}
    res = {}
    for s in subs:
        e = data[s]; n = len(e['label']); ones = np.ones((M, n))
        loso = [np.mean([smean[t][m] for t in FOLDS[fold_of[s]] if t != s], 0) for m in range(M)]
        perm = rng.permutation(n)
        for a in (0.25, 0.5, 0.75, 1.0):
            variants = {
                'subject': [x - a * smean[s][m] for m, x in enumerate(e['emb'])],
                'loso': [x - a * loso[m] for m, x in enumerate(e['emb'])],
                'refimg': [x - a * np.linalg.norm(loso[m]) * ref_dir for m, x in enumerate(e['emb'])],
                'running': [x - a * running_mean(x, perm) for x in e['emb']],
                'running_imgmajor': [x - a * running_mean(x, np.arange(n)) for x in e['emb']],
                'batch64': [x - a * np.repeat(x[perm].reshape(-1, 64, x.shape[1]).mean(1), 64, 0)[np.argsort(perm)] for x in e['emb']],
            }
            for name, emb in variants.items():
                res.setdefault(f'{name} a={a}', {})[s] = score(dict(e, emb=emb), ones)
        res.setdefault('uniform', {})[s] = score(e, np.ones((M, n)))
        cos = [float(loso[m] @ smean[s][m] / np.linalg.norm(loso[m]) / np.linalg.norm(smean[s][m])) for m in range(M)]
        res.setdefault('_cos(loso, subject mean)', {})[s] = np.mean(cos)
    base = np.array([res['uniform'][s] for s in subs])
    print(f'{"+".join(recipes)} ({len(subs)} subjects)')
    for name, v in res.items():
        v = np.array([v[s] for s in subs])
        if name.startswith('_'):
            print(f'  {name:26s} {v.mean():.3f} (min {v.min():.3f})'); continue
        print(f'  {name:26s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')


if __name__ == '__main__':
    main(sys.argv[1:])
