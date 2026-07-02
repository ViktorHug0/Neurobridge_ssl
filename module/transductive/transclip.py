"""Adapter around the official TransCLIP zero-shot solver (Phase 2, group B).

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
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NEURO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DEFAULT_REF_ROOT = os.environ.get(
    "TRANSDUCTION_VLMS_ROOT",
    os.path.abspath(os.path.join(_NEURO_ROOT, "..", "transduction-for-vlms")),
)


def _load_utils(ref_root=None):
    """Import the official TransCLIP utility functions (namespace package)."""
    root = ref_root or _DEFAULT_REF_ROOT
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
