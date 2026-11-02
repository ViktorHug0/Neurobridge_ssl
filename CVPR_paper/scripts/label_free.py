"""Competitor label-free test-time calibrations, reimplemented on frozen embeddings (no code released).

score_deploy: SCORE (Cui et al., arXiv 2608.19134), deployment step only (Sec. 3, Eqs. 3-9):
  moment matching of the target EEG to the gallery (per-dimension mean/std), CSLS (k=10) mutual-nearest-neighbour
  landmarks weighted by their top-1/top-2 CSLS margin, ONE weighted orthogonal Procrustes step regularised towards
  the identity (lambda = rho * ||X^T W Y||_2, rho = 0.1), applied to the whole batch, then CSLS retrieval.
  Unstated constants: eps (variance floor) and eta (margin floor) -> 1e-6. Their training-time simulated-episode
  loss is NOT reproduced here (it changes the encoder, not the deployment step).
svtl_refine: SVTL's training-free refinement (Yadav et al., arXiv 2609.36971; repo github.com/Shalini-Y1/SVTL_EEG2Image
  is an empty placeholder as of 2026-10-20). The paper (Sec. 2.4) only NAMES the components: per-dimension moment
  matching, CSLS mutual-NN landmarks, 64 landmarks, confidence weighting, "regularized orthogonal alignment" with
  rho = 0.1, CSLS scoring. Every formula is taken from SCORE. Unstated: CSLS k (-> 10, SCORE's and Conneau et al.'s),
  confidence function (-> SCORE margin, top-64 kept; fewer if < 64 mutual NNs), regulariser form/scale (-> SCORE's
  identity term, lambda = rho*||C||_2), number of refinement iterations (-> 1).
"""
import numpy as np


def _cos(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def csls(q, g, k):
    s = _cos(q, g)
    k = max(1, min(k, *s.shape))
    rx = np.partition(s, s.shape[1] - k, axis=1)[:, -k:].mean(1, keepdims=True)
    ry = np.partition(s, s.shape[0] - k, axis=0)[-k:].mean(0, keepdims=True)
    return 2 * s - rx - ry


def csls_ref(q, ref, g, k):
    """CSLS of queries q whose gallery-side hubness term comes from a reference set `ref` (e.g. the buffer a frozen map
    was fitted on), so q never influences its own score. csls_ref(q, q, g, k) == csls(q, g, k)."""
    s = _cos(q, g)
    k = max(1, min(k, len(ref), g.shape[0]))
    rx = np.partition(s, s.shape[1] - k, axis=1)[:, -k:].mean(1, keepdims=True)
    sr = _cos(ref, g)
    ry = np.partition(sr, sr.shape[0] - k, axis=0)[-k:].mean(0, keepdims=True)
    return 2 * s - rx - ry


def score_fit(query, gallery, k=10, rho=0.1, eps=1e-6, eta=1e-6, n_landmarks=None):
    """SCORE's recovery map (Eqs. 3-8) fitted on `query`; returns the frozen row-wise map x -> q_hat(x)."""
    q = np.asarray(query, np.float64)
    g = np.asarray(gallery, np.float64)
    mu, sd, gm, gs = q.mean(0), q.std(0), g.mean(0), g.std(0)

    def mm(x):  # Eq. 3
        return (np.asarray(x, np.float64) - mu) / (sd + eps) * (gs + eps) + gm
    qt = mm(q)
    s = csls(qt, g, k)  # Eq. 4
    fwd, bwd = s.argmax(1), s.argmax(0)
    mut = np.flatnonzero(bwd[fwd] == np.arange(len(q)))  # mutual nearest neighbours = landmarks
    if len(mut) < 2:
        return mm
    top2 = np.sort(s, axis=1)[:, -2:]
    delta = np.maximum(top2[mut, 1] - top2[mut, 0], eta)
    if n_landmarks is not None and len(mut) > n_landmarks:  # keep the most confident (SVTL: 64)
        keep = np.argsort(-delta)[:n_landmarks]
        mut, delta = mut[keep], delta[keep]
    w = len(mut) * delta / delta.sum()  # Eq. 5
    x, y = qt[mut], g[fwd[mut]]
    mx, my = (w[:, None] * x).sum(0) / w.sum(), (w[:, None] * y).sum(0) / w.sum()
    c = (x - mx).T @ (w[:, None] * (y - my))
    u, _, vt = np.linalg.svd(c + rho * np.linalg.norm(c, 2) * np.eye(c.shape[0]))  # Eq. 6-7
    r = u @ vt
    return lambda z: (mm(z) - mx) @ r + my  # Eq. 8


def score_deploy(query, gallery, k=10, rho=0.1, eps=1e-6, eta=1e-6, n_landmarks=None):
    """Returns the CSLS score matrix (queries x gallery) after SCORE's recovery step."""
    return csls(score_fit(query, gallery, k, rho, eps, eta, n_landmarks)(query), np.asarray(gallery, np.float64), k)  # Eq. 9


def svtl_refine(query, gallery):
    return score_deploy(query, gallery, k=10, rho=0.1, n_landmarks=64)


if __name__ == '__main__':  # self-check: a rotated, shifted copy of the gallery is recovered
    rng = np.random.default_rng(0)
    g = rng.normal(size=(200, 32))
    u, _, vt = np.linalg.svd(np.eye(32) + 0.25 * rng.normal(size=(32, 32)))
    rot = u @ vt  # a moderate rotation (polar factor: near the identity, unlike a sign-flipped QR)
    q = (g + 0.5 * rng.normal(size=g.shape)) @ rot * 3 + 2  # plus scale and offset
    base = (_cos(q, g).argmax(1) == np.arange(200)).mean()
    rec = (score_deploy(q, g).argmax(1) == np.arange(200)).mean()
    svtl = (svtl_refine(q, g).argmax(1) == np.arange(200)).mean()
    print(f'plain {base:.3f} -> SCORE {rec:.3f}, SVTL {svtl:.3f}')
    assert rec > base and svtl > base
    assert np.allclose(csls_ref(q, q, g, 10), csls(q, g, 10))
