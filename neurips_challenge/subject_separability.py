"""How identifiable is the participant from ONE label-free trial? (prerequisite for routing trials of a shuffled,
mixed-participant batch into per-participant groups). Within each fold's 5 test subjects (unseen by the model):
nearest-centroid / kNN subject identification on
  eeg   per-channel log-variance + channel-correlation (tangent-ish) fingerprint of the raw trial
  out   the model's output vector (cosine)
Also: purity of k-means clusters inside random mixed batches of 64.
    subject_separability.py <recipe>"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.trial_routing import FOLDS, load  # noqa: E402

D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'


def eeg_fp(s):
    x = np.load(f'{D}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
    out = []
    for i in range(0, len(x), 2000):
        b = np.asarray(x[i:i + 2000], np.float32); b = b - b.mean(-1, keepdims=True)
        v = (b ** 2).mean(-1) + 1e-6
        c = np.einsum('nct,ndt->ncd', b, b) / 120 / np.sqrt(v[:, :, None] * v[:, None, :])
        iu = np.triu_indices(32, 1)
        out.append(np.hstack([np.log(v), c[:, iu[0], iu[1]]]))
    return np.concatenate(out)


def kmeans(X, k, it=20, rng=None):
    C = X[rng.choice(len(X), k, replace=False)]
    for _ in range(it):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        C = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    return lab


def purity(lab, true):
    return sum(np.bincount(true[lab == j]).max() for j in np.unique(lab)) / len(lab)


def main():
    data = load([sys.argv[1]]); rng = np.random.default_rng(0)
    for kf, subs in FOLDS.items():
        feats = {'eeg': [eeg_fp(s) for s in subs], 'out': [data[s]['emb'][0] / np.linalg.norm(data[s]['emb'][0], axis=1, keepdims=True) for s in subs]}
        n = len(feats['out'][0])
        for name, F in feats.items():
            X = np.concatenate(F); y = np.repeat(np.arange(len(subs)), n)
            mu, sd = X.mean(0), X.std(0) + 1e-6; Xz = (X - mu) / sd
            tr = rng.random(len(X)) < 0.5  # centroids from half the trials, identify the other half
            C = np.array([Xz[tr & (y == j)].mean(0) for j in range(len(subs))])
            acc = (((Xz[~tr][:, None] - C[None]) ** 2).sum(-1).argmin(1) == y[~tr]).mean()
            pur = []
            for _ in range(30):  # unsupervised: k-means inside one shuffled batch of 64
                idx = rng.choice(len(X), 64, replace=False)
                pur.append(purity(kmeans(Xz[idx], len(subs), rng=rng), y[idx]))
            print(f'fold {kf} {name}: nearest-centroid subject id {100 * acc:.1f}% (chance 20%); k-means purity in a 64-batch {np.mean(pur):.2f}', flush=True)


if __name__ == '__main__':
    main()
