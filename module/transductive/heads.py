"""Transductive inference heads producing ``(N, K)`` score matrices.

Every head has the signature ``head(query_features, image_features, params) -> (N, K)``
where a higher score means a better EEG->image match, so the existing
``module.util.topk(scores, k, target_indices)`` works unchanged.

- Cheap / OT heads (D-group) reuse ``module.util`` directly.
- EM / clustering / Dirichlet heads (A-group) reuse the official transductive-CLIP
  solvers via :mod:`module.transductive.reference`.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from module.util import (
    apply_orthogonal_map,
    csls_scores,
    fit_soft_assignment_procrustes,
    score_query_features,
    sinkhorn_normalize,
)
from module.transductive.reference import (
    HARD_METHODS,
    REFERENCE_SOLVERS,
    run_reference_solver,
)
from module.transductive.transclip import TRANSCLIP_VARIANTS, make_transclip_head


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cosine(query, image):
    return cosine_similarity(
        np.asarray(query, dtype=np.float32), np.asarray(image, dtype=np.float32)
    ).astype(np.float32, copy=False)


def _front_scores(query, image, csls_k=0):
    """Cosine similarity, optionally CSLS-adjusted (``csls_k > 0``).

    Shared front-end so every head can use the same CSLS pre-conditioning that
    ``full_sattc`` applies internally (SAW whitening is applied upstream).
    """
    sim = _cosine(query, image)
    k = int(csls_k or 0)
    if k > 0:
        sim = csls_scores(sim, k=k)
    return sim


def simplex_features(query, image, T=30.0, csls_k=0):
    """``z_n = softmax(T * cos(eeg_n, img_k))`` — the CLIP-probability analogue.

    With ``csls_k > 0`` the cosine matrix is CSLS-adjusted before the softmax.
    """
    scaled = T * _front_scores(query, image, csls_k=csls_k)
    scaled -= scaled.max(axis=1, keepdims=True)
    expd = np.exp(scaled, dtype=np.float32)
    return (expd / np.clip(expd.sum(axis=1, keepdims=True), 1e-12, None)).astype(np.float32)


# --------------------------------------------------------------------------- #
# D-group: cheap / optimal-transport baselines
# --------------------------------------------------------------------------- #
def _head_plain_cosine(query, image, params):
    # True raw baseline: plain cosine, never SAW/CSLS (driver also skips SAW for it).
    return _cosine(query, image)


def _head_csls(query, image, params):
    return csls_scores(_cosine(query, image), k=int(params.get("csls_k", 12)))


def _head_sinkhorn(query, image, params):
    scores = _cosine(query, image)
    if params.get("csls_k"):
        scores = csls_scores(scores, k=int(params["csls_k"]))
    return sinkhorn_normalize(
        scores,
        tau=float(params.get("sinkhorn_tau", 0.05)),
        num_iters=int(params.get("sinkhorn_iters", 20)),
    )


def _hungarian_boost(scores):
    """Solve the optimal one-to-one assignment on ``scores`` and boost each matched
    pair above all other candidates, so top-1 follows the assignment while ranks
    2..K keep the original score order."""
    rows, cols = linear_sum_assignment(-scores)
    boosted = scores.copy()
    boost = float(scores.max() - scores.min() + 1.0)
    boosted[rows, cols] += boost
    return boosted


def _head_hungarian(query, image, params):
    """Hard one-to-one assignment (exploits the bijective N=K structure)."""
    return _hungarian_boost(_front_scores(query, image, csls_k=params.get("csls_k", 0)))


def _head_full_sattc(query, image, params):
    """Incumbent: (optional CSLS) + iterated Sinkhorn-soft-Procrustes (+ final Sinkhorn).

    Mirrors ``evaluate._refine_scores`` / the progressive sweep's ``full_sattc`` so
    that, with SAW preprocessing applied upstream, it reproduces current numbers.
    """
    use_csls = bool(params.get("use_csls", True))
    csls_k = int(params.get("csls_k", 12))
    tau = float(params.get("sinkhorn_tau", 0.05))
    n_iters = int(params.get("sinkhorn_iters", 20))

    scores, _ = score_query_features(query, image, use_csls=use_csls, csls_k=csls_k)

    if params.get("soft_procrustes_enabled", True):
        for _ in range(max(1, int(params.get("soft_procrustes_steps", 1)))):
            assignment = sinkhorn_normalize(scores, tau=tau, num_iters=n_iters)
            ortho = fit_soft_assignment_procrustes(
                query,
                image,
                assignment,
                power=float(params.get("soft_procrustes_power", 1.0)),
                normalize_inputs=bool(params.get("soft_procrustes_normalize_inputs", False)),
            )
            if ortho is None:
                break
            query = apply_orthogonal_map(query, ortho)
            scores, _ = score_query_features(query, image, use_csls=use_csls, csls_k=csls_k)

    if params.get("sinkhorn_enabled", True):
        scores = sinkhorn_normalize(scores, tau=tau, num_iters=n_iters)
    return scores


def _head_sattc_hungarian(query, image, params):
    """Incumbent SATTC refinement, then a final exact Hungarian assignment.

    Runs the full CSLS + Sinkhorn-soft-Procrustes refinement (which improves the
    embedding geometry), then replaces the soft argmax with an optimal one-to-one
    assignment on the refined score matrix.
    """
    scores = _head_full_sattc(query, image, params)
    return _hungarian_boost(scores)


def _head_hungarian_sattc(query, image, params):
    """Exact Hungarian assignment first, then the incumbent SATTC refinement.

    Solves the optimal one-to-one assignment on the front scores, uses that hard
    permutation to fit a Procrustes rotation that aligns the query to the image
    embeddings, applies it, then runs the full SATTC refinement on the rotated
    query. The mirror image of ``sattc_hungarian`` (which assigns last).
    """
    use_csls = bool(params.get("use_csls", True))
    csls_k = int(params.get("csls_k", 12))

    front, _ = score_query_features(query, image, use_csls=use_csls, csls_k=csls_k)
    rows, cols = linear_sum_assignment(-front)
    assignment = np.zeros_like(front)
    assignment[rows, cols] = 1.0

    ortho = fit_soft_assignment_procrustes(
        query,
        image,
        assignment,
        power=float(params.get("soft_procrustes_power", 1.0)),
        normalize_inputs=bool(params.get("soft_procrustes_normalize_inputs", False)),
    )
    if ortho is not None:
        query = apply_orthogonal_map(query, ortho)
    return _head_full_sattc(query, image, params)


# --------------------------------------------------------------------------- #
# A-group: reference EM / clustering / Dirichlet solvers (simplex regime)
# --------------------------------------------------------------------------- #
def _make_reference_head(method):
    def _run(query, image, params):
        T = float(params.get("T", 30.0))
        z = simplex_features(query, image, T=T, csls_k=params.get("csls_k", 0))
        u = run_reference_solver(method, z, params=params)
        if method in HARD_METHODS:
            # tiny tiebreaker so top-5 is ordered for (near) one-hot assignments
            u = u.astype(np.float32, copy=False) + 1e-4 * z
        return u

    _run.__name__ = f"_head_{method}"
    return _run


# --------------------------------------------------------------------------- #
# registry / public API
# --------------------------------------------------------------------------- #
METHOD_REGISTRY = {
    "plain_cosine": _head_plain_cosine,
    "csls": _head_csls,
    "sinkhorn": _head_sinkhorn,
    "hungarian": _head_hungarian,
    "full_sattc": _head_full_sattc,
    "sattc_hungarian": _head_sattc_hungarian,
    "hungarian_sattc": _head_hungarian_sattc,
}
for _method in REFERENCE_SOLVERS:
    METHOD_REGISTRY[_method] = _make_reference_head(_method)
for _variant in TRANSCLIP_VARIANTS:
    METHOD_REGISTRY[_variant] = make_transclip_head(_variant)


def list_methods():
    return sorted(METHOD_REGISTRY)


def infer(query_features, image_features, method, params=None):
    """Run ``method`` and return an ``(N, K)`` score matrix (higher = better)."""
    if method not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{method}'. Available: {list_methods()}")
    return METHOD_REGISTRY[method](
        np.asarray(query_features, dtype=np.float32),
        np.asarray(image_features, dtype=np.float32),
        dict(params or {}),
    )
