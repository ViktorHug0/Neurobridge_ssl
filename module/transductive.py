"""Transductive inference heads for the 200-way THINGS-EEG benchmark.

This module is deliberately kept separate from the existing evaluation code. It
provides a single entry point, :func:`infer`, that maps a method name to an
``(N, K)`` score matrix consumable by ``module.util.topk``. Cheap heads reuse
``module.util``; the EM / clustering / Dirichlet heads reuse the official
``transductive-CLIP`` solvers directly (see the reference-solver section below).

Reference solvers (A-group)
---------------------------
Adapters around the official ``transductive-CLIP`` zero-shot solvers.

We do **not** reimplement the EM / clustering math. Instead we import the solver
classes from the sibling ``../transductive-CLIP`` repository, feed them simplex
(probability) features ``z_n = softmax(T * cos(eeg_n, img_k))``, run them in the
``use_softmax_feature=True`` regime (which never touches CLIP's text encoder), and
read back the soft-assignment matrix ``self.u`` of shape ``(N, K)``.

Because the test set has exactly one EEG query per image-class (N = K = 200, one
sample per class), the simplex columns correspond 1:1 to the image prototypes, so
``self.u`` is directly the score matrix we want — no cluster->class relabeling.

The reference repo location defaults to ``<neurobridge_root>/../transductive-CLIP``
and can be overridden with the ``TRANSDUCTIVE_CLIP_ROOT`` environment variable.
``clip`` is stubbed at import time because it is only needed by the (unused)
text-feature code paths.

TransCLIP (B-group)
-------------------
Adapter around the official TransCLIP zero-shot solver (Phase 2, group B).

We reuse the official building blocks from the sibling ``../transduction-for-vlms``
repo (``TransCLIP_solver/TransCLIP_utils.py``) — the GMM ``Gaussian`` likelihood,
the Laplacian affinity, and the closed-form ``update_z`` / ``update_mu`` /
``update_sigma`` steps — and re-implement only the *outer block-MM loop* so we can
toggle the ablations the paper studies (Table 6): no text-KL anchor, no Laplacian,
update-mu-only / update-sigma-only. With default params this reproduces the
official ``TransCLIP_solver`` zero-shot path exactly (verified in tests).

Mapping to our setting (N = K = 200, one EEG query per image-class, S = ∅):
- ``query_features`` = EEG queries (N, d)
- ``clip_prototypes`` = L2-normalized image prototypes transposed (d, K) — the
  image features play the role of CLIP text embeddings ``t_k``.
- zero-shot branch only (no support / no validation).

Returns the (N, K) soft-assignment ``z`` as the score matrix for ``topk``.

Repo location defaults to ``<neurobridge_root>/../transduction-for-vlms`` and can be
overridden with the ``TRANSDUCTION_VLMS_ROOT`` environment variable.

Heads
-----
Transductive inference heads producing ``(N, K)`` score matrices.

Every head has the signature ``head(query_features, image_features, params) -> (N, K)``
where a higher score means a better EEG->image match, so the existing
``module.util.topk(scores, k, target_indices)`` works unchanged.

- Cheap / OT heads (D-group) reuse ``module.util`` directly.
- EM / clustering / Dirichlet heads (A-group) reuse the official transductive-CLIP
  solvers via the reference-solver adapters.
"""

import importlib
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from module.util import (
    apply_orthogonal_map,
    csls_scores,
    fit_soft_assignment_procrustes,
    score_query_features,
    sinkhorn_normalize,
)


# =========================================================================== #
# reference transductive-CLIP solvers
# =========================================================================== #
_NEURO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
_TRANSDUCTIVE_CLIP_ROOT = os.environ.get(
    "TRANSDUCTIVE_CLIP_ROOT",
    os.path.abspath(os.path.join(_NEURO_ROOT, "..", "transductive-CLIP")),
)

# method name -> (module path within transductive-CLIP, class name)
REFERENCE_SOLVERS = {
    "em_dirichlet": ("src.methods.zero_shot.em_dirichlet", "EM_DIRICHLET"),
    "hard_em_dirichlet": ("src.methods.zero_shot.hard_em_dirichlet", "HARD_EM_DIRICHLET"),
    "kl_kmeans": ("src.methods.zero_shot.kl_kmeans", "KL_KMEANS"),
    "soft_kmeans": ("src.methods.zero_shot.soft_kmeans", "SOFT_KMEANS"),
    "hard_kmeans": ("src.methods.zero_shot.hard_kmeans", "HARD_KMEANS"),
    "em_gaussian": ("src.methods.zero_shot.em_gaussian", "EM_GAUSSIAN"),
    "em_gaussian_cov": ("src.methods.zero_shot.em_gaussian_cov", "EM_GAUSSIAN_COV"),
}

# Methods whose ``self.u`` is (near) one-hot; a tiny tiebreaker is added downstream
# so that top-5 ranking is meaningful.
HARD_METHODS = {"hard_em_dirichlet", "hard_kmeans", "kl_kmeans"}

# Single reusable log directory so we do not spawn one tmp dir per call.
_LOG_DIR = tempfile.mkdtemp(prefix="transductive_ref_")
_LOG_FILE = os.path.join(_LOG_DIR, "reference_solvers.log")


def _ensure_importable(ref_root):
    if not os.path.isdir(ref_root):
        raise FileNotFoundError(
            f"transductive-CLIP repo not found at '{ref_root}'. Set TRANSDUCTIVE_CLIP_ROOT."
        )
    if ref_root not in sys.path:
        sys.path.insert(0, ref_root)
    if "clip" not in sys.modules:
        stub = types.ModuleType("clip")

        def _unavailable(*_args, **_kwargs):
            raise RuntimeError(
                "clip is stubbed: the reference adapters only support the simplex "
                "(use_softmax_feature=True) regime, which does not need text features."
            )

        stub.tokenize = _unavailable
        sys.modules["clip"] = stub


def _load_solver_class(method, ref_root):
    if method not in REFERENCE_SOLVERS:
        raise KeyError(f"Unknown reference solver '{method}'. Known: {sorted(REFERENCE_SOLVERS)}")
    module_path, class_name = REFERENCE_SOLVERS[method]
    _ensure_importable(ref_root)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def run_reference_solver(method, simplex_features, params=None, ref_root=None):
    """Run a reference zero-shot solver on ``(N, K)`` simplex features.

    Parameters
    ----------
    method : str
        Key in :data:`REFERENCE_SOLVERS`.
    simplex_features : np.ndarray
        ``(N, K)`` probability features on the simplex (rows sum to 1).
    params : dict, optional
        ``iter``, ``iter_mm``, ``T``, ``lambd`` (class-balance override), ``device``.

    Returns
    -------
    np.ndarray
        ``(N, K)`` soft-assignment matrix ``self.u``.
    """
    params = dict(params or {})
    ref_root = ref_root or _TRANSDUCTIVE_CLIP_ROOT
    solver_cls = _load_solver_class(method, ref_root)

    z = np.asarray(simplex_features, dtype=np.float32)
    if z.ndim != 2:
        raise ValueError("simplex_features must be a 2-D (N, K) array.")
    n, k = z.shape
    device = torch.device(params.get("device", "cpu"))

    args = SimpleNamespace(
        iter=int(params.get("iter", 20)),
        iter_mm=int(params.get("iter_mm", 50)),
        num_classes_test=k,
        n_query=n,
        n_class=k,
        T=float(params.get("T", 30.0)),
        use_softmax_feature=True,
        graph_matching=False,
        classnames=[str(i) for i in range(k)],
        template="{}",
    )

    solver = solver_cls(model=None, device=device, log_file=_LOG_FILE, args=args)
    # Allow sweeping the class-balance hyper-parameter (BASE hardcodes a default).
    if "lambd" in params and hasattr(solver, "lambd"):
        solver.lambd = float(params["lambd"])

    x_q = torch.from_numpy(z).unsqueeze(0).to(device)          # [1, N, K]
    y_q = torch.arange(n, dtype=torch.long).view(1, n, 1)      # dummy labels (unused by us)
    try:
        solver.run_task({"x_q": x_q, "y_q": y_q})
        u = solver.u[0].detach().cpu().numpy().astype(np.float32, copy=False)
    finally:
        # Explicitly tear down the logger so handlers do not accumulate across calls.
        logger = getattr(solver, "logger", None)
        if logger is not None:
            logger.del_logger()
    return u


# =========================================================================== #
# TransCLIP
# =========================================================================== #
_TRANSDUCTION_VLMS_ROOT = os.environ.get(
    "TRANSDUCTION_VLMS_ROOT",
    os.path.abspath(os.path.join(_NEURO_ROOT, "..", "transduction-for-vlms")),
)


def _load_utils(ref_root=None):
    """Import the official TransCLIP utility functions (namespace package)."""
    root = ref_root or _TRANSDUCTION_VLMS_ROOT
    if not os.path.isdir(root):
        raise FileNotFoundError(
            f"TransCLIP reference repo not found at '{root}'. "
            "Set TRANSDUCTION_VLMS_ROOT to override."
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    from TransCLIP_solver import TransCLIP_utils as U  # noqa: E402
    return U


def _zero_affinity(n, device):
    """An all-zero sparse (N, N) affinity, for the no-Laplacian ablation."""
    idx = torch.zeros((2, 0), dtype=torch.long, device=device)
    val = torch.zeros((0,), dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, (n, n), device=device).coalesce()


def run_transclip(query_features, image_features, params=None):
    """Run TransCLIP-ZS (or an ablation) and return the (N, K) assignment matrix.

    params:
      lambda (float)      text-KL anchor weight (``y_hat ** lambda``); 0 disables it. default 1.0
      n_neighbors (int)   k for the k-NN Laplacian affinity. default 3
      max_iter (int)      outer block-MM iterations. default 10
      use_laplacian (bool) include the Laplacian smoothness term. default True
      update_mu (bool)    update GMM means each iter. default True
      update_sigma (bool) update GMM (shared diagonal) variance each iter. default True
      anchor_mu (bool)    init mu = image prototypes instead of top-8 confident. default False
      clip_scale (float)  logit scale for y_hat = scale * cos. default 100.0
      device (str)        default "cuda"
    """
    params = dict(params or {})
    U = _load_utils(params.get("ref_root"))

    device = params.get("device", "cuda")
    lam = float(params.get("lambda", 1.0))
    n_neighbors = int(params.get("n_neighbors", 3))
    max_iter = int(params.get("max_iter", 10))
    use_laplacian = bool(params.get("use_laplacian", True))
    do_update_mu = bool(params.get("update_mu", True))
    do_update_sigma = bool(params.get("update_sigma", True))
    anchor_mu = bool(params.get("anchor_mu", False))
    clip_scale = float(params.get("clip_scale", 100.0))

    q = torch.as_tensor(np.asarray(query_features, dtype=np.float32), device=device)
    proto = torch.as_tensor(np.asarray(image_features, dtype=np.float32), device=device)
    # TransCLIP assumes L2-normalized CLIP features (Gaussian std_init = 1/d scale).
    q = F.normalize(q, dim=1)
    proto = F.normalize(proto, dim=1)

    K, d = proto.shape
    N = q.shape[0]
    std_init = 1.0 / d
    clip_prototypes = proto.t()  # (d, K)

    with torch.no_grad():
        y_hat = clip_scale * (q @ clip_prototypes)            # (N, K) prior logits
        y_hat, z = U.init_z(y_hat, softmax=True)              # both = softmax(y_hat)

        if anchor_mu:                                         # bijective shortcut
            mu = proto.unsqueeze(1).clone()                  # (K, 1, d), unit-norm
        else:
            mu = U.init_mu(K, d, z, q, None, None)           # top-8 confident (paper)

        std = U.init_sigma(d, std_init).to(device)
        adapter = U.Gaussian(mu=mu, std=std).to(device)

        W = U.build_affinity_matrix(q, None, N, n_neighbors) if use_laplacian \
            else _zero_affinity(N, device)

        for it in range(max_iter + 1):
            gmm_likelihood = adapter(q, no_exp=True)
            z = U.update_z(gmm_likelihood, y_hat, z, W, lam, n_neighbors, None)[0:N]
            if it == max_iter:
                break
            if do_update_mu:
                adapter = U.update_mu(adapter, 0, q, z, None, None)
            if do_update_sigma:
                adapter = U.update_sigma(adapter, 0, q, z, None, None)

    return z.detach().cpu().numpy().astype(np.float32)


# name -> param overrides on top of the TransCLIP-ZS defaults
TRANSCLIP_VARIANTS = {
    "transclip": {},                                   # B1: full TransCLIP-ZS
    "transclip_no_kl": {"lambda": 0.0},                # B2: no text-KL anchor
    "transclip_no_lap": {"use_laplacian": False},      # B3: no Laplacian
    "transclip_mu_only": {"update_sigma": False},      # B4: update mu only
    "transclip_sigma_only": {"update_mu": False},      # B4: update sigma only
    "transclip_anchor": {"anchor_mu": True, "update_mu": False},  # mu = image prototypes
}


def make_transclip_head(variant):
    """Build a head(query, image, params) for a named TransCLIP variant."""
    fixed = TRANSCLIP_VARIANTS[variant]

    def _run(query, image, params):
        merged = dict(params or {})
        merged.update(fixed)  # variant overrides win
        return run_transclip(query, image, merged)

    _run.__name__ = f"_head_{variant}"
    return _run


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


__all__ = ["METHOD_REGISTRY", "infer", "list_methods", "simplex_features"]
