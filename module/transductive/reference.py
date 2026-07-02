"""Adapters around the official ``transductive-CLIP`` zero-shot solvers.

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
"""

import importlib
import os
import sys
import tempfile
import types
from types import SimpleNamespace

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_NEURO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_DEFAULT_REF_ROOT = os.environ.get(
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
    import torch

    params = dict(params or {})
    ref_root = ref_root or _DEFAULT_REF_ROOT
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
