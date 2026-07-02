"""Transductive inference heads for the 200-way THINGS-EEG benchmark.

This package is deliberately kept separate from the existing evaluation code. It
provides a single entry point, :func:`infer`, that maps a method name to an
``(N, K)`` score matrix consumable by ``module.util.topk``. Cheap heads reuse
``module.util``; the EM / clustering / Dirichlet heads reuse the official
``transductive-CLIP`` solvers directly (see :mod:`module.transductive.reference`).
"""

from module.transductive.heads import (
    METHOD_REGISTRY,
    infer,
    list_methods,
    simplex_features,
)

__all__ = ["METHOD_REGISTRY", "infer", "list_methods", "simplex_features"]
