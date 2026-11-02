"""Per-trial weighting / routing under the grader's rules (trial_cache.py + trial_features.py caches).
The grader sums our raw outputs per (subject, image), so returning w_t * p_t weights trial t; summing members'
scaled outputs with per-trial weights routes trials between members. Every feature here is computable inside
predict(): raw EEG statistics, output norms, member agreement, and confidence against a REFERENCE gallery of
training images (never the test gallery). Weight models are fit on the other folds' subjects (leave-fold-out).
    trial_routing.py <recipe> [<recipe> ...]
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.cv4_ensemble import run_dirs  # noqa: E402

FOLDS = {1: [2, 4, 5, 11, 15], 2: [7, 14, 16, 17, 20], 3: [3, 8, 10, 12, 19], 4: [1, 6, 9, 13, 18]}
FEAT = np.load('results/alljoined/track1/routing/trial_features.npz')
EEG_FEATS = ['logstd', 'logmax', 'clamp', 'erpcorr', 'pcadist', 'lowfrac', 'highfrac', 'baseratio', 'chanmaxz']
rng = np.random.default_rng(0)
REF = np.load('data/things_eeg/image_feature/dinov2-giant_track1/image_train.npy', mmap_mode='r')
REF = np.asarray(REF.reshape(-1, REF.shape[-1])[np.sort(rng.choice(int(np.prod(REF.shape[:-1])), 2000, replace=False))], np.float32)
REF /= np.linalg.norm(REF, axis=1, keepdims=True)


def load(recipes, fname='trial_dots.npz'):
    """{s: dict(label, G, emb[m] (n, D) scaled by the member's fold mean norm, norm[m])}"""
    data = {}
    for m, r in enumerate(recipes):
        for k, d in run_dirs(r).items():
            z = np.load(f'{d}/{fname}')
            mn = np.mean(np.concatenate([z[f's{s}_norm'] for s in FOLDS[k]]))
            for s in FOLDS[k]:
                e = data.setdefault(s, dict(emb=[None] * len(recipes), norm=[None] * len(recipes)))
                e['label'], e['G'] = z[f's{s}_label'], z['gallery']
                e['emb'][m] = z[f's{s}_emb'].astype(np.float32) / mn
                e['norm'][m] = z[f's{s}_norm'] / mn
    return data


def top5(Q, G):
    S = Q @ G.T
    return 100 * np.mean([(S[i] > S[i, i]).sum() < 5 for i in range(len(S))])


def score(e, W):
    """W: (members, n) per-trial weights -> subject top-5."""
    Q = np.zeros((200, e['G'].shape[1]))
    for m, emb in enumerate(e['emb']):
        np.add.at(Q, e['label'], emb * W[m][:, None])
    return top5(Q, e['G'])


def features(s, e):
    """(members, n, F) label-free per-trial features."""
    base = np.nan_to_num(np.stack([FEAT[f's{s}_{k}'] for k in EEG_FEATS], 1), posinf=0.0, neginf=0.0)  # 2 trials have a flat baseline
    M = len(e['emb'])
    unit = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in e['emb']]
    cons = sum(unit) / M
    out = []
    for m in range(M):
        R = unit[m] @ REF.T                                           # confidence vs training images only
        top = np.sort(R, 1)[:, -2:]
        f = [np.log(e['norm'][m]), top[:, 1] - top[:, 0], R.max(1), R.std(1),
             (unit[m] * cons).sum(1),                                  # agreement with the member consensus
             unit[m] @ (unit[m].mean(0) / np.linalg.norm(unit[m].mean(0)))]  # typicality within the subject (stateful)
        out.append(np.hstack([base, np.stack(f, 1)]))
    return np.stack(out)


def usefulness(e, m):
    """Label-using target for fitting only: z-scored margin of the true image among the 200 (cosine)."""
    u = e['emb'][m] / np.linalg.norm(e['emb'][m], axis=1, keepdims=True)
    S = u @ e['G'].T
    return (S[np.arange(len(S)), e['label']] - S.mean(1)) / S.std(1)


def main(recipes):
    data = load(recipes)
    M = len(recipes)
    X = {s: features(s, e) for s, e in data.items()}
    U = {s: np.stack([usefulness(e, m) for m in range(M)]) for s, e in data.items()}
    subs = sorted(data)
    res = {}
    res['uniform'] = {s: score(data[s], np.ones((M, len(data[s]['label'])))) for s in subs}
    res['oracle relu(u)'] = {s: score(data[s], np.maximum(U[s], 0)) for s in subs}           # cheats: upper bound
    for a in (0.5, 1.0):  # stateful: subtract a * the subject's mean output (all its trials, no labels)
        res[f'subject-centred a={a}'] = {s: score(dict(data[s], emb=[x - a * x.mean(0) for x in data[s]['emb']]),
                                                  np.ones((M, len(data[s]['label'])))) for s in subs}
    for k, d in FOLDS.items():
        tr = [s for s in subs if s not in d]
        Xtr = np.concatenate([X[s].reshape(-1, X[s].shape[-1]) for s in tr]); ytr = np.concatenate([U[s].ravel() for s in tr])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        A = np.hstack([np.ones((len(Xtr), 1)), (Xtr - mu) / sd])
        beta = np.linalg.lstsq(A, ytr, rcond=None)[0]
        r2 = 1 - ((A @ beta - ytr) ** 2).mean() / ytr.var()
        if k == 1:
            names = EEG_FEATS + ['lognorm', 'refmargin', 'refmax', 'refstd', 'agree', 'typical']
            print(f'fold-1 fit R2 {r2:.3f}; coefs', {n: round(float(b), 3) for n, b in zip(names, beta[1:])})
        for s in d:
            pred = np.hstack([np.ones((X[s].shape[0] * X[s].shape[1], 1)), (X[s].reshape(-1, X[s].shape[-1]) - mu) / sd]) @ beta
            pred = pred.reshape(M, -1)
            for g in (0.5, 1, 2):
                res.setdefault(f'learned exp({g}*pred)', {})[s] = score(data[s], np.exp(g * (pred - pred.mean())))
            for q in (0.1, 0.25):
                res.setdefault(f'learned drop bottom {q}', {})[s] = score(data[s], (pred > np.quantile(pred, q)).astype(float))
            if M > 1:
                for g in (1, 2):
                    w = np.exp(g * (pred - pred.max(0)))
                    res.setdefault(f'route softmax({g}*pred)', {})[s] = score(data[s], M * w / w.sum(0))
        for s in d:  # Hoke-style inverse-variance weights, no fitting
            for a in (1, 2):
                res.setdefault(f'hoke std^-{a}', {})[s] = score(data[s], np.repeat(np.exp(-a * FEAT[f's{s}_logstd'])[None], M, 0))
    base = np.array([res['uniform'][s] for s in subs])
    print(f'{"+".join(recipes)} ({len(subs)} subjects)')
    for name, v in res.items():
        v = np.array([v[s] for s in subs])
        print(f'  {name:28s} {v.mean():6.2f}  {v.mean() - base.mean():+.2f}  better {(v > base).sum():2d} worse {(v < base).sum():2d}')


if __name__ == '__main__':
    main(sys.argv[1:])
