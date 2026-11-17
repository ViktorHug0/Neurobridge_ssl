"""GALA for incremental transductive calibration: one fit on the unlabeled buffer x (t, d) and the gallery g (K, d).

The zero-shot decoder (no centring, isotropic covariance, identity rotation) counts as n0 = prior * d observations
(paper: prior = 2, user decision of 2026-11-10): with a = t / (t + n0),
  mean        a * mean(x)
  covariance  (1 - rho) a S + (1 - (1 - rho) a) tr(S)/d I     (the usual shrinkage rho, increased on small buffers)
  rotation    R = polar(C + (n0/t) mean_sv(C) I), C = E^T P^gamma V  (n0 pseudo-pairs aligned by the identity, each
                                                                     carrying the mean cross-covariance of a real pair)
shrink=False is the batch method (shared.fit_tta_calibration): the paper uses no prior in the batch evaluation (81.8;
76.8 with prior = 2, 82.2 with prior = 1). damp=True (with shrink=False) is the earlier post-hoc damping, the
`full_dim` arm of stream_grid.py, kept as an ablation: it interpolates the fitted calibration with the zero-shot
decoder (query blend, fraction a of every rotation angle, (1 - a) cos + a CSLS). Both are checked in __main__.
On the incremental streams (ATM, 3 seeds, q = 10), prior = 2 is 2 to 4 points above prior = 1 at 5 minutes and equal
at 20 minutes. Switches for the ablation: whiten_on, use_csls, steps (0 = no rotation), sinkhorn.
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../scripts/things_eeg/tta_rebuttal'))
from run_streaming_refit import geodesic_blend, schur_form  # noqa: E402


def _norm(x):
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def whiten(x, rho, a=1.0, eps=1e-6):
    """Returns the row-wise map e -> normalised shrinkage-whitened e, fitted on x (a < 1: shrunk towards no whitening)."""
    mu = x.mean(0, keepdims=True)
    cov = (x - mu).T @ (x - mu) / max(len(x) - 1, 1)
    d = cov.shape[0]
    w = (1 - rho) * a
    cov = w * cov + (1 - w) * np.trace(cov) / d * np.eye(d) + eps * np.eye(d)
    mu = a * mu
    ev, U = np.linalg.eigh(cov)
    W = (U * np.clip(ev, eps, None) ** -0.5) @ U.T
    return lambda z: _norm((z - mu) @ W)


def plan(s, tau, iters=12, col_power=1.0, b=None):
    """Sinkhorn plan with unit row mass and uniform column mass (iters = 0: row softmax).
    col_power < 1: KL-relaxed column constraint (exponent lambda / (lambda + tau) of unbalanced Sinkhorn; 0 = no column
    constraint). b (K,): target column masses (mean 1) in place of the uniform ones."""
    P = np.exp((s - s.max(1, keepdims=True)) / tau)
    P /= P.sum(1, keepdims=True)
    for _ in range(iters):
        P /= np.clip(P.sum(1, keepdims=True), 1e-8, None)
        P *= ((1.0 if b is None else b) / np.clip(P.sum(0, keepdims=True), 1e-8, None)) ** col_power
    return P


def fit(x, g, rho, tau, gamma, steps=16, k=3, shrink=True, damp=False, whiten_on=True, use_csls=True, sinkhorn=True, prior=2.0,
        col_power=1.0, freq=None, R0=None):
    """Returns scorer: scorer(z) = scores (len(z), K) of any queries z under the calibration fitted on x.
    Under test (2026-11-16, defaults = the method of the paper): col_power (see plan); freq = pseudo-count a: the column
    masses are the query frequencies estimated from the buffer, (soft count + a) normalised to mean 1, re-estimated at
    every iteration; R0 = rotation of the previous update (warm start; the fitted one is scorer.R)."""
    x, graw = np.asarray(x, np.float64), np.asarray(g, np.float64)
    g = _norm(graw)
    t, d = x.shape
    K = len(g)
    n0 = prior * d  # the zero-shot decoder counts as n0 observations
    a = t / (t + n0) if shrink or damp else 1.0
    wh = whiten(x, rho, a if shrink else 1.0) if whiten_on else _norm
    e = wh(x)
    R = np.eye(d) if R0 is None else R0
    kk = max(1, min(k, t, K))

    def hub(c, ry=None):  # the two CSLS neighbourhood terms (0 without CSLS)
        if not use_csls:
            return 0.0, 0.0
        rx = np.partition(c, K - kk, axis=1)[:, -kk:].mean(1, keepdims=True)
        ry = np.partition(c, len(c) - kk, axis=0)[-kk:].mean(0, keepdims=True) if ry is None else ry
        return rx, ry

    for _ in range(steps):
        c = (e @ R) @ g.T
        rx, ry = hub(c)
        sc = 2 * c - rx - ry
        b = None
        if freq is not None:
            b = plan(sc, tau, 0).sum(0, keepdims=True) + freq
            b = b / b.mean()
        P = plan(sc, tau, 12 if sinkhorn else 0, col_power, b)
        C = e.T @ (P ** gamma) @ graw  # raw gallery, as the batch code
        # without shrinkage and t < d the rotation is not unique outside the span of the queries: the 1e-6 identity term
        # selects the one closest to the identity (deterministic; no effect once C is full-rank)
        lam = (n0 / t) * np.linalg.svd(C, compute_uv=False).mean() if shrink else 1e-6 * np.linalg.norm(C)
        U, _, Vt = np.linalg.svd(C + lam * np.eye(d))
        R = U @ Vt
    if not damp:
        a = 1.0  # the shrinkage acts inside the fit; the decoder below is the plain calibrated one
    else:
        R = geodesic_blend(schur_form(R), a).astype(np.float64)

    def emb(z):
        z = np.asarray(z, np.float64)
        return _norm(((1 - a) * _norm(z) + a * wh(z)) @ R)
    ry = hub(emb(x) @ g.T)[1]  # gallery-side hubness term comes from the fitted buffer

    def scorer(z):
        c = emb(z) @ g.T
        rx, _ = hub(c, ry)
        return (1 - a) * c + a * (2 * c - rx - ry)
    scorer.R = R
    return scorer


if __name__ == '__main__':  # self-check against the implementations used for every earlier number
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
    from module.util import apply_orthogonal_map
    from shared import TTAParams, apply_saw_transform, apply_tta_calibration, fit_tta_calibration, normalize_rows, score_features
    from stream_grid import frozen_scorer
    rng = np.random.default_rng(0)
    g = rng.normal(size=(200, 32)).astype(np.float32)
    q = (g @ np.linalg.qr(np.eye(32) + 0.3 * rng.normal(size=(32, 32)))[0] * 2 + rng.normal(size=g.shape) + 1).astype(np.float32)
    p = TTAParams(saw_shrink=0.7, sinkhorn_tau=0.07, soft_procrustes_power=1.2)
    cal = fit_tta_calibration(q, g, p)
    old = score_features(apply_tta_calibration(q, cal), g, use_csls=True, csls_k=3)
    new = fit(q, g, 0.7, 0.07, 1.2, shrink=False)(q)
    agree = (old.argmax(1) == new.argmax(1)).mean()
    print('undamped: argmax agreement with shared.fit_tta_calibration', agree, 'max |diff|', np.abs(old - new).max())
    assert agree > 0.99 and np.abs(old - new).max() < 1e-2
    for t in (40, 200):  # damped: the full_dim arm of stream_grid.py, rebuilt from its own functions (t >= d: unique rotation)
        x = q[:t]
        cal = fit_tta_calibration(x, g, p)
        a = t / (t + 32)
        m = lambda z, R=geodesic_blend(schur_form(cal['orthogonal_map']), a): apply_orthogonal_map(
            (1 - a) * normalize_rows(z) + a * normalize_rows(apply_saw_transform(z, cal['saw_stats'])), R)
        old = frozen_scorer(m, x, g, 3, a)(x)
        new = fit(x, g, 0.7, 0.07, 1.2, shrink=False, damp=True, prior=1)(x)  # full_dim uses a = t / (t + d)
        agree = (old.argmax(1) == new.argmax(1)).mean()
        print(f'damped t={t}: argmax agreement with stream_grid full_dim', agree, 'max |diff|', np.abs(old - new).max())
        assert agree > 0.95 and np.abs(old - new).max() < 1e-2  # the 1e-6 identity term may flip a near-tie
    q3, g40 = np.tile(q[:3], (1, 40)), np.tile(g, (1, 40))  # 3 queries in dimension 1280: decoded like the zero-shot decoder
    zero = score_features(q3, g40)
    for kw in ({}, {'shrink': False, 'damp': True}):
        assert (fit(q3, g40, 0.7, 0.07, 1.2, use_csls=False, **kw)(q3).argmax(1) == zero.argmax(1)).all(), kw
    print('ok')
