import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def dump_pretty(obj, fp, indent=4, ensure_ascii=False):
    """Pretty-print dicts with indentation while keeping lists on a single line."""
    def _serialize(o, level):
        if isinstance(o, dict):
            if not o:
                return "{}"
            items = []
            indent_str = " " * (indent * level)
            child_indent_str = " " * (indent * (level + 1))
            items.append("{\n")
            kvs = list(o.items())
            for i, (k, v) in enumerate(kvs):
                key = json.dumps(k, ensure_ascii=ensure_ascii)
                value = _serialize(v, level + 1)
                items.append(f"{child_indent_str}{key}: {value}")
                if i < len(kvs) - 1:
                    items.append(",")
                items.append("\n")
            items.append(indent_str + "}")
            return "".join(items)
        elif isinstance(o, list):
            inner = ", ".join(_serialize(v, level) for v in o)
            return "[" + inner + "]"
        else:
            return json.dumps(o, ensure_ascii=ensure_ascii)
    fp.write(_serialize(obj, 0))


def _normalize_rows(features, eps=1e-12):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.clip(norms, eps, None)


def _estimate_mu_cov(features, shrink=0.2, diag=False, eps=1e-6):
    """Ledoit-Wolf-style shrinkage covariance estimate."""
    features = np.asarray(features, dtype=np.float32)
    if features.shape[0] == 0:
        raise ValueError("Cannot estimate covariance from an empty feature matrix.")
    mu = features.mean(axis=0, keepdims=True)
    centered = features - mu
    cov = (centered.T @ centered / float(max(features.shape[0] - 1, 1))).astype(np.float32, copy=False)
    if diag:
        cov = np.diag(np.diag(cov))
    dim = cov.shape[0]
    trace_mean = float(np.trace(cov)) / max(dim, 1)
    cov = (1.0 - float(shrink)) * cov + float(shrink) * trace_mean * np.eye(dim, dtype=np.float32)
    cov = cov + eps * np.eye(dim, dtype=np.float32)
    return mu.astype(np.float32, copy=False), cov


def _inv_sqrt_cov(cov, eps=1e-6):
    """Symmetric inverse square root via eigendecomposition (ZCA whitening matrix)."""
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, eps, None)
    inv_sqrt = evecs @ np.diag(np.power(evals, -0.5, dtype=np.float32)) @ evecs.T
    return inv_sqrt.astype(np.float32, copy=False)


def subject_adaptive_whiten(features, subject_ids, shrink=0.2, diag=False, normalize=True):
    """Per-subject ZCA whitening to align EEG feature distributions across subjects."""
    features = np.asarray(features, dtype=np.float32)
    subject_ids = np.asarray(subject_ids)
    if features.shape[0] != subject_ids.shape[0]:
        raise ValueError("features and subject_ids must have the same first dimension.")
    processed = np.empty_like(features, dtype=np.float32)
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        sub_feats = features[mask]
        if sub_feats.shape[0] == 0:
            continue
        mu, cov = _estimate_mu_cov(sub_feats, shrink=shrink, diag=diag)
        whitened = (sub_feats - mu) @ _inv_sqrt_cov(cov)
        processed[mask] = _normalize_rows(whitened.astype(np.float32, copy=False)) if normalize else whitened
    return processed


def apply_orthogonal_map(features, weights):
    features = np.asarray(features, dtype=np.float32)
    if weights is None:
        return features
    return _normalize_rows(features @ np.asarray(weights, dtype=np.float32))


def csls_scores(similarities, k=10):
    """CSLS re-ranking: penalises hubness by subtracting neighbourhood mean similarities."""
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.ndim != 2:
        raise ValueError("CSLS expects a 2-D similarity matrix.")
    n_q, n_c = similarities.shape
    if n_q == 0 or n_c == 0:
        return similarities
    k_eff = max(1, min(int(k), n_q, n_c))
    rx = np.partition(similarities, kth=n_c - k_eff, axis=1)[:, -k_eff:].mean(axis=1, keepdims=True)
    ry = np.partition(similarities, kth=n_q - k_eff, axis=0)[-k_eff:, :].mean(axis=0, keepdims=True)
    return (2.0 * similarities - rx - ry).astype(np.float32, copy=False)


def topk(matrix, k, target_indices=None):
    sorted_indices = np.argsort(-matrix, axis=1)
    rankings = np.argsort(sorted_indices, axis=1)
    if target_indices is None:
        target_ranks = np.diag(rankings) + 1
    else:
        target_indices = np.asarray(target_indices, dtype=np.int64)
        target_ranks = rankings[np.arange(sorted_indices.shape[0]), target_indices] + 1
    count_k = int((target_ranks <= k).sum())
    count_1 = int((target_ranks == 1).sum())
    return count_k, count_1


def sinkhorn_normalize(similarities, tau=0.05, num_iters=20, eps=1e-8):
    """Convert a similarity matrix into a doubly-stochastic soft assignment via Sinkhorn-Knopp."""
    similarities = np.asarray(similarities, dtype=np.float32)
    if similarities.ndim != 2 or similarities.shape[0] == 0 or similarities.shape[1] == 0:
        return similarities
    tau = max(float(tau), eps)
    scaled = similarities / tau
    scaled -= scaled.max(axis=1, keepdims=True)  # numerical stability
    matrix = np.exp(scaled).astype(np.float32, copy=False)
    for _ in range(max(1, int(num_iters))):
        matrix /= np.clip(matrix.sum(axis=1, keepdims=True), eps, None)
        matrix /= np.clip(matrix.sum(axis=0, keepdims=True), eps, None)
    return matrix.astype(np.float32, copy=False)


def fit_soft_assignment_procrustes(query_features, candidate_features, assignment, power=1.0, eps=1e-8, normalize_inputs=False):
    """Find the orthogonal rotation that best aligns queries to their soft-assigned image candidates."""
    query_features = np.asarray(query_features, dtype=np.float32)
    candidate_features = np.asarray(candidate_features, dtype=np.float32)
    assignment = np.asarray(assignment, dtype=np.float32)
    if query_features.ndim != 2 or candidate_features.ndim != 2 or assignment.ndim != 2:
        return None
    if query_features.shape[0] == 0 or candidate_features.shape[0] == 0:
        return None
    if assignment.shape != (query_features.shape[0], candidate_features.shape[0]):
        return None
    if normalize_inputs:
        query_features = _normalize_rows(query_features)
        candidate_features = _normalize_rows(candidate_features)
    weights = np.clip(assignment, 0.0, None)
    if float(power) != 1.0:
        weights = np.power(weights, max(float(power), eps)).astype(np.float32, copy=False)
    if float(weights.sum()) <= eps:
        return None
    cross = query_features.T @ weights @ candidate_features
    try:
        u, _, vt = np.linalg.svd(cross, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return (u @ vt).astype(np.float32, copy=False)


def process_query_features(eeg_features, subject_ids, eval_mode='plain_cosine', sattc_params=None):
    """Apply subject-adaptive whitening (SAW) to EEG features when the eval mode requires it."""
    sattc_params = sattc_params or {}
    eeg_features = np.asarray(eeg_features, dtype=np.float32)
    if eval_mode == 'plain_cosine':
        return eeg_features
    if eval_mode not in {'saw', 'saw_csls'}:
        raise ValueError(f"Unsupported eval_mode: {eval_mode}")
    return subject_adaptive_whiten(
        eeg_features,
        np.asarray(subject_ids),
        shrink=sattc_params.get('saw_shrink', 0.2),
        diag=sattc_params.get('saw_diag', False),
        normalize=sattc_params.get('saw_renorm', True),
    )


def score_query_features(query_features, image_features, use_csls=False, csls_k=12):
    """Cosine similarity between EEG queries and image candidates, with optional CSLS re-ranking.

    Returns (scores [n_queries x n_images], target_indices [n_queries]) where
    target_indices[i] = i since each query corresponds to exactly one image.
    """
    scores = cosine_similarity(query_features, image_features).astype(np.float32, copy=False)
    target_indices = np.arange(len(query_features), dtype=np.int64)
    if use_csls:
        scores = csls_scores(scores, k=csls_k)
    return scores, target_indices


def compute_retrieval_scores(eeg_features, image_features, subject_ids=None, eval_mode='plain_cosine', sattc_params=None):
    """Whiten EEG queries (if the eval mode requires it), then compute similarity scores."""
    sattc_params = sattc_params or {}
    processed = process_query_features(eeg_features, subject_ids, eval_mode, sattc_params)
    return score_query_features(
        processed, image_features,
        use_csls=(eval_mode == 'saw_csls'),
        csls_k=sattc_params.get('csls_k', 12),
    )


def retrieve_all(eeg_features, image_features, subject_ids=None, eval_mode='plain_cosine', sattc_params=None):
    """Score all EEG-image pairs and return (top5_hits, top1_hits, total)."""
    scores, targets = compute_retrieval_scores(eeg_features, image_features, subject_ids, eval_mode, sattc_params)
    count_5, count_1 = topk(scores, 5, target_indices=targets)
    return count_5, count_1, len(eeg_features)
