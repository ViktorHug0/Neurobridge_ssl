"""Fixed, query-local fusion rules for closed-set retrieval ensembles."""

from __future__ import annotations

import numpy as np


FUSION_METHODS = (
    "raw",
    "probability",
    "row_z",
    "row_z_probability",
    "rank",
    "vote",
)


def cosine_scores(eeg: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Build a query-by-candidate cosine matrix from one embedding dump."""
    eeg = np.asarray(eeg, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if eeg.ndim != 2 or image.ndim != 2 or eeg.shape[1] != image.shape[1]:
        raise ValueError(
            "eeg and image must be 2-D arrays with the same feature dimension"
        )
    eeg = eeg / np.maximum(np.linalg.norm(eeg, axis=1, keepdims=True), 1e-8)
    image = image / np.maximum(np.linalg.norm(image, axis=1, keepdims=True), 1e-8)
    return (eeg @ image.T).astype(np.float32)


def row_z(scores: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Standardize every member/query row across its candidate images."""
    scores = np.asarray(scores)
    return (scores - scores.mean(axis=-1, keepdims=True)) / np.maximum(
        scores.std(axis=-1, keepdims=True), eps
    )


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - scores.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def fuse_scores(member_scores: np.ndarray, method: str = "row_z") -> np.ndarray:
    """Fuse ``[member, query, candidate]`` scores with one fixed uniform rule.

    Every transform is query-local: a query never reads another test query.  In
    particular, these rules are inductive and can be composed with plain-cosine
    evaluation without turning it into SATTC-style transductive adaptation.
    """
    scores = np.asarray(member_scores)
    if scores.ndim != 3 or scores.shape[0] == 0:
        raise ValueError("member_scores must have shape [member, query, candidate]")
    if not np.isfinite(scores).all():
        raise ValueError("member_scores contains non-finite values")
    if method not in FUSION_METHODS:
        raise ValueError(f"unknown fusion method {method!r}; choose from {FUSION_METHODS}")

    if method == "raw":
        transformed = scores
    elif method == "probability":
        transformed = _softmax(scores)
    elif method == "row_z":
        transformed = row_z(scores)
    elif method == "row_z_probability":
        transformed = _softmax(row_z(scores))
    elif method == "rank":
        order = np.argsort(scores, axis=-1)
        transformed = np.empty_like(scores, dtype=np.float32)
        ranks = np.broadcast_to(
            np.arange(scores.shape[-1], dtype=np.float32), scores.shape
        )
        np.put_along_axis(transformed, order, ranks, axis=-1)
    else:
        winners = scores.argmax(axis=-1)
        transformed = np.eye(scores.shape[-1], dtype=np.float32)[winners]
    return transformed.mean(axis=0, dtype=np.float64).astype(np.float32)


def retrieval_accuracies(
    scores: np.ndarray,
    targets: np.ndarray | None = None,
    topk: int = 5,
) -> tuple[float, float]:
    """Return top-1 and top-k percentages for a retrieval score matrix."""
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError("scores must have shape [query, candidate]")
    if targets is None:
        if scores.shape[0] != scores.shape[1]:
            raise ValueError("implicit diagonal targets require a square score matrix")
        targets = np.arange(scores.shape[0])
    targets = np.asarray(targets)
    if targets.shape != (scores.shape[0],):
        raise ValueError("targets must contain one candidate index per query")
    if not 1 <= topk <= scores.shape[1]:
        raise ValueError("topk must be between 1 and the number of candidates")

    top1 = float(np.mean(scores.argmax(axis=1) == targets) * 100.0)
    topk_indices = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    topk_accuracy = float(
        np.mean(np.any(topk_indices == targets[:, None], axis=1)) * 100.0
    )
    return top1, topk_accuracy


def mean_pairwise_score_correlation(member_scores: np.ndarray) -> float:
    """Mean off-diagonal Pearson correlation between member score matrices."""
    scores = np.asarray(member_scores)
    if scores.ndim != 3 or scores.shape[0] < 2:
        raise ValueError("at least two member score matrices are required")
    correlation = np.corrcoef(row_z(scores).reshape(scores.shape[0], -1))
    return float(correlation[np.triu_indices(scores.shape[0], k=1)].mean())


def oracle_top1(member_scores: np.ndarray, targets: np.ndarray | None = None) -> float:
    """Fraction of queries for which at least one individual member is correct."""
    scores = np.asarray(member_scores)
    if scores.ndim != 3:
        raise ValueError("member_scores must have shape [member, query, candidate]")
    if targets is None:
        if scores.shape[1] != scores.shape[2]:
            raise ValueError("implicit diagonal targets require square score matrices")
        targets = np.arange(scores.shape[1])
    targets = np.asarray(targets)
    correct = scores.argmax(axis=-1) == targets[None, :]
    return float(correct.any(axis=0).mean() * 100.0)
