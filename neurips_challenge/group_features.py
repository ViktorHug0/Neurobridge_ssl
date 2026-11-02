"""Candidate person-identity features for in-batch grouping (TTA_LOG T6), per raw 1-s trial (32 x 120 at 120 Hz):
  fine   log power per channel in 2-Hz bins 1-45 Hz (22 x 32)       (individual alpha peak, spectral slope)
  calpha / cbeta  channel correlations of the 8-13 / 13-30 Hz band-passed trial (496 each)
  erp    the trial's waveform per channel, 10-sample means (12 x 32)  (person-specific evoked shape)
  cdelta / ctheta / cgamma  channel correlations at 1-4 / 4-8 / 30-58 Hz;  pgamma  log power per channel, 30-58 Hz in
         7-Hz bins (4 x 32)  (EMG: muscle tone is person- and placement-specific)
  tangent  log-Euclidean tangent vector of the trial's 32 x 32 channel covariance (+1e-3 trace shrink): upper
         triangle of logm(C), off-diagonals x sqrt 2 (528): the standard Riemannian EEG signature, one object (T24)
Cached per subject under DATA/.cache: train = the SAME 3000 training-image trials as the fingerprint caches
(fit_fingerprint_lda sampling), test = every row of test.npy (index by the dataset idx).
    group_features.py <subject> [...]
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from neurips_challenge.grader_sim import DATA  # noqa: E402

GROUPS = ('fine', 'calpha', 'cbeta', 'erp', 'cdelta', 'ctheta', 'cgamma', 'pgamma', 'tangent')


def bandpass(b, lo, hi):
    F = np.fft.rfft(b, axis=-1); f = np.fft.rfftfreq(b.shape[-1], 1 / 120)
    F[..., (f < lo) | (f >= hi)] = 0
    return np.fft.irfft(F, n=b.shape[-1], axis=-1)


def corr(b):
    b = b - b.mean(-1, keepdims=True); v = (b ** 2).mean(-1) + 1e-9
    c = np.einsum('nct,ndt->ncd', b, b) / b.shape[-1] / np.sqrt(v[:, :, None] * v[:, None, :])
    iu = np.triu_indices(b.shape[1], 1)
    return c[:, iu[0], iu[1]]


def tangent(b):
    C = np.einsum('nct,ndt->ncd', b, b) / b.shape[-1]
    C = C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1] * np.eye(C.shape[1])
    w, V = np.linalg.eigh(C); L = (V * np.log(w)[:, None, :]) @ V.transpose(0, 2, 1)
    iu = np.triu_indices(C.shape[1]); scale = np.where(iu[0] == iu[1], 1., np.sqrt(2))
    return L[:, iu[0], iu[1]] * scale


def features(x, groups=GROUPS):
    """x (n, 32, 120) float32 -> {group: (n, d) float32}"""
    out = {g: [] for g in groups}
    for i in range(0, len(x), 1000):
        b = np.asarray(x[i:i + 1000], np.float64); b = b - b.mean(-1, keepdims=True)
        P = np.abs(np.fft.rfft(b, axis=-1)) ** 2  # 1-Hz bins
        f = {'fine': lambda: np.log(np.stack([P[..., lo:lo + 2].mean(-1) for lo in range(1, 45, 2)], -1) + 1e-6).reshape(len(b), -1),
             'calpha': lambda: corr(bandpass(b, 8, 13)), 'cbeta': lambda: corr(bandpass(b, 13, 30)),
             'erp': lambda: b.reshape(len(b), 32, 12, 10).mean(-1).reshape(len(b), -1),
             'cdelta': lambda: corr(bandpass(b, 1, 4)), 'ctheta': lambda: corr(bandpass(b, 4, 8)), 'cgamma': lambda: corr(bandpass(b, 30, 58)),
             'tangent': lambda: tangent(b),
             'pgamma': lambda: np.log(np.stack([P[..., lo:lo + 7].mean(-1) for lo in range(30, 58, 7)], -1) + 1e-6).reshape(len(b), -1)}
        for g in groups:
            out[g].append(f[g]())
    return {g: np.concatenate(v).astype(np.float32) for g, v in out.items()}


def path(g, split, s):
    return f'{DATA}/.cache/gf_{g}_{split}_s{s:02d}.npy'


def build(s, n_per=3000, seed=0):
    todo = [g for g in GROUPS if not all(os.path.exists(path(g, sp, s)) for sp in ('train', 'test'))]
    if not todo:
        return
    x = np.load(f'{DATA}/sub-{s:02d}/train.npy', mmap_mode='r').reshape(-1, 32, 120)
    idx = np.sort(np.random.default_rng([seed, s]).choice(len(x), 4 * n_per, replace=False))  # = fit_fingerprint_lda
    xs = np.asarray(x[idx], np.float32); xs = xs[np.abs(xs).sum((1, 2)) > 0][:n_per]
    for split, data in (('train', xs), ('test', np.load(f'{DATA}/sub-{s:02d}/test.npy', mmap_mode='r').reshape(-1, 32, 120)
                                        if os.path.exists(f'{DATA}/sub-{s:02d}/test.npy') else None)):
        if data is None:
            continue
        for g, v in features(data, todo).items():
            np.save(path(g, split, s) + '.tmp.npy', v); os.replace(path(g, split, s) + '.tmp.npy', path(g, split, s))
    print(f'subject {s} done', flush=True)


if __name__ == '__main__':
    t = np.sin(2 * np.pi * 10 * np.arange(120) / 120)[None, None] * np.ones((2, 32, 1))
    assert np.allclose(bandpass(t, 8, 13), t, atol=1e-9) and np.abs(bandpass(t, 13, 30)).max() < 1e-9
    f = features(t.astype(np.float32))
    r = np.random.default_rng(0).standard_normal((3, 32, 120)); T = tangent(r - r.mean(-1, keepdims=True))
    assert T.shape == (3, 528) and np.isfinite(T).all()
    assert f['fine'].shape == (2, 22 * 32) and f['calpha'].shape == (2, 496) and f['erp'].shape == (2, 384)
    for s in map(int, sys.argv[1:]):
        build(s)
