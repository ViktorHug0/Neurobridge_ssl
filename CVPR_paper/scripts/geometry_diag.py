"""E8.11 why the RSA loss fails: per arm (a loso/<arm> dir) and fold 1/3/8, geometry of the R1 test embeddings (80 reps).
- pr_eeg / pr_img: participation ratio (sum l)^2 / sum l^2 of the centred, row-normalised clouds (effective dimension);
- knn10: mean overlap of each query's 10 nearest EEG neighbours with its image's 10 nearest image neighbours;
- cv_rot: split-half supervised rotation (fit on 100 true pairs, 100-way retrieval on the other 100; 5 splits x 2 ways),
  an honest version of the in-sample oracle of rsa_screen_eval.py.
usage: geometry_diag.py <out_csv> <arm> [<arm> ...]
"""
import csv, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
from shared import encode_repetition_blocks, load_subject_context  # noqa: E402


def nrm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def pr(x):
    lam = np.linalg.eigvalsh(np.cov(x.T))
    return lam.sum() ** 2 / (lam ** 2).sum()


def knn(x, k):
    s = x @ x.T
    np.fill_diagonal(s, -np.inf)
    return np.argsort(-s, axis=1)[:, :k]


def cv_rot(q, g, rng, splits=5):
    accs = []
    for _ in range(splits):
        p = rng.permutation(len(q))
        for fit, ev in ((p[:100], p[100:]), (p[100:], p[:100])):
            u, _, vt = np.linalg.svd(q[fit].T @ g[fit])
            accs.append(((q[ev] @ u @ vt @ g[ev].T).argmax(1) == np.arange(len(ev))).mean())
    return 100 * np.mean(accs)


def main():
    out, arms = sys.argv[1], sys.argv[2:]
    rows = []
    for arm in arms:
        run = os.path.abspath(f'results/things_eeg/cvpr/loso/{arm}/seed3300')
        for sid in (1, 3, 8):
            rt = SimpleNamespace(device='cpu', batch_size=1024, num_workers=0)
            _, ea, ds, mods = load_subject_context(run, rt, sid, average=False)
            (q,), g, _ = encode_repetition_blocks(ea, mods, ds, block_sizes=[80], seed=3300)
            qq, gg = nrm(nrm(np.asarray(q, float)) - nrm(np.asarray(q, float)).mean(0)), nrm(nrm(np.asarray(g, float)) - nrm(np.asarray(g, float)).mean(0))
            ke, ki = knn(qq, 10), knn(gg, 10)
            rec = dict(arm=arm, subject=sid, pr_eeg=pr(qq), pr_img=pr(gg),
                       knn10=np.mean([len(set(a) & set(b)) / 10 for a, b in zip(ke, ki)]),
                       cv_rot=cv_rot(qq, gg, np.random.default_rng(sid)))
            rows.append(rec)
            print(rec, flush=True)
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()
