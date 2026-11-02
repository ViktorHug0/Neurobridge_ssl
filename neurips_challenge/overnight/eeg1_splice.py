"""THINGS-EEG1 chimeras: keep each EEG1 trial's paradigm-invariant early window, take the late window from a THINGS-EEG2
trial of the SAME image (5 Hz paradigm = the target's). Partners come only from EEG2 sub-21..30 (never test subjects, so
one build serves every fold). Same partner draws (seed 0) for both modes:
  splice   (sub-67..78): y = w*E1 + (1-w)*E2                           (segment recombination, Lotte 2015)
  residual (sub-79..90): y = w*E1 + (1-w)*[P(E1) + E2 - P(E2)]          (P: ridge late<-early fitted on EEG2 single trials:
                          E1's own ongoing activity continues smoothly, E2 contributes its unpredicted image-specific late part)
w = 1 up to 0.30 s after onset, cosine to 0 at 0.40 s. Epoch -0.2..0.8 s at 120 Hz, onset = sample 24.
    eeg1_splice.py splice|residual
"""
import os
import sys
import numpy as np

D = '/nasbrain/p20fores/alljoined/Preprocessed_data_120Hz_neuralbench'
MODE = sys.argv[1]
OFF = {'splice': 36, 'residual': 48}[MODE]                       # sub-31..42 -> sub-67..78 / sub-79..90
t = (np.arange(120) - 24) / 120.0
w = (0.5 - 0.5 * np.cos(np.pi * np.clip((0.40 - t) / 0.10, 0, 1))).astype(np.float32)
assert (w[:60] == 1).all() and w[-1] == 0                          # the early window (< 0.30 s) is untouched
E, L = slice(0, 72), slice(60, 120)                              # ridge: early -0.2..0.4 s -> late 0.3..0.8 s

rng = np.random.default_rng(0)
partner = rng.integers(0, 40, size=(12, 1654, 10, 4))           # per EEG1 slot: EEG2 subject (//4) and rep (%4)


def fit_ridge():
    """P: late window from early window, all channels jointly, on 100k EEG2 single trials (closed form, alpha by holdout)."""
    xs = []
    for s in range(21, 31):
        X = np.load(f'{D}/sub-{s:02d}/train.npy', mmap_mode='r')
        idx = np.sort(np.random.default_rng(s).choice(66160, 10000, replace=False))
        xs.append(np.asarray(X.reshape(-1, 32, 120)[idx], np.float32))
    X = np.concatenate(xs)
    A, B = X[:, :, E].reshape(len(X), -1).astype(np.float64), X[:, :, L].reshape(len(X), -1).astype(np.float64)
    n = int(0.9 * len(X)); G = A[:n].T @ A[:n]; H = A[:n].T @ B[:n]
    best = None
    for alpha in (1e2, 1e3, 1e4, 1e5):
        W = np.linalg.solve(G + alpha * np.eye(len(G)), H)
        r2 = 1 - ((B[n:] - A[n:] @ W) ** 2).sum() / ((B[n:] - B[n:].mean(0)) ** 2).sum()
        print(f'ridge alpha {alpha:g}: held-out R2 {r2:.4f}', flush=True)
        if best is None or r2 > best[0]:
            best = (r2, W)
    return best[1].astype(np.float32)


def predict(W, x):                                               # x (..., 32, 120) -> (..., 32, 60) late window
    sh = x.shape[:-2]
    return (x[..., E].reshape(-1, 32 * 72) @ W).reshape(*sh, 32, 60)


def main():
    W = fit_ridge() if MODE == 'residual' else None
    groups = [g for g in range(12) if not os.path.exists(f'{D}/sub-{31 + OFF + g:02d}/train.npy')]
    late = {g: np.zeros((1654, 10, 4, 32, 60), np.float32) for g in groups}
    for s in range(10):                                           # each EEG2 subject read once, for every group
        X2 = np.load(f'{D}/sub-{21 + s:02d}/train.npy')
        for g in groups:
            sel = partner[g] // 4 == s
            o, i, r = np.nonzero(sel)
            p2 = X2[o, i, partner[g][sel] % 4]                     # (n, 32, 120): same image, EEG2 subject s
            late[g][o, i, r] = p2[..., L] - (predict(W, p2) if W is not None else 0)
        del X2
        print(f'EEG2 sub-{21 + s} gathered', flush=True)
    for g in groups:
        out = f'{D}/sub-{31 + OFF + g:02d}/train.npy'
        X1 = np.load(f'{D}/sub-{31 + g:02d}/train.npy')          # (1654, 10, 4, 32, 120)
        lg = late.pop(g)
        if W is not None:
            lg += predict(W, X1)
        X1[..., L] = w[L] * X1[..., L] + (1 - w[L]) * lg
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.save(out + '.tmp.npy', np.clip(X1, -20, 20)); os.replace(out + '.tmp.npy', out)
        print(f'sub-{31 + OFF + g:02d} {MODE}: mean|x| {float(np.abs(X1[:50]).mean()):.3f}', flush=True)


if __name__ == '__main__':
    main()
